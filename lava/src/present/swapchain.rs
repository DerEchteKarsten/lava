//! A `PresentSurface`'s swapchain: its creation (format, present mode and extent picked from what the surface supports), its images with their views, bindless slots, tracked states and present semaphores, and its retirement once no submission or present uses it
use std::sync::atomic::AtomicU64;

use crate::vk;
use glam::UVec2;
use smallvec::SmallVec;

use crate::{
    bindless::{self, BindlessHandle},
    command_buffer::image_state::ImageState,
    error::Result,
    features::{On, gate},
    lifetime,
    present::table,
    queue::timeline::forget_image,
    state::{Ctx, Functions},
};

#[derive(Debug)]
pub(crate) struct SwapchainImage {
    pub(crate) image: vk::Image,
    pub(crate) view: vk::ImageView,
    /// Bound in every set the usage allows.
    pub(crate) handle: BindlessHandle,
    pub(crate) state: AtomicU64,
    /// Signalled by the submission that presents it, waited for by its present. Acquiring the
    /// image again means the present before is done with it.
    pub(crate) present: vk::Semaphore,
}

#[derive(Debug)]
pub(crate) struct Swapchain {
    pub(crate) handle: vk::SwapchainKHR,
    pub(crate) extent: UVec2,
    /// The index into the surface's present modes it presents with.
    pub(crate) present_mode: usize,
    pub(crate) images: SmallVec<[SwapchainImage; 5]>,
    /// The timeline of the last submission that presented one of its images.
    pub(crate) last_presenter: Option<usize>,
}

/// What [`Swapchain::new`] makes a swapchain with.
pub(crate) struct Params {
    pub(crate) surface: vk::SurfaceKHR,
    pub(crate) capabilities: vk::SurfaceCapabilitiesKHR,
    pub(crate) format: vk::SurfaceFormatKHR,
    pub(crate) usage: vk::ImageUsageFlags,
    pub(crate) extent: vk::Extent2D,
    pub(crate) present_mode: (usize, vk::PresentModeKHR),
}

impl Swapchain {
    /// A swapchain replacing `old` (which stays valid until retired), its images bound in
    /// every bindless set `usage` allows.
    pub(crate) fn new(params: &Params, old: Option<vk::SwapchainKHR>) -> Result<Self> {
        let caps = &params.capabilities;
        let mut info = vk::SwapchainCreateInfoKHR::default()
            .surface(params.surface)
            .min_image_count(caps.min_image_count)
            .image_format(params.format.format)
            .image_color_space(params.format.color_space)
            .image_extent(params.extent)
            .image_array_layers(1)
            .image_usage(params.usage)
            // Images are exclusive to one family at a time; presenting goes through graphics.
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(params.present_mode.1)
            .clipped(true)
            .old_swapchain(old.unwrap_or_default());
        // With swapchain maintenance, the modes it may switch between: just its own.
        let modes = [params.present_mode.1];
        let mut modes_info = vk::SwapchainPresentModesCreateInfoEXT::default().present_modes(&modes);
        if gate::PresentFences::on() {
            info = info.push_next(&mut modes_info);
        }
        let handle = unsafe { Functions::swapchain().create_swapchain(&info, None)? };
        tracing::info!(
            "made a swapchain of {}x{} images, presenting {:?}",
            params.extent.width,
            params.extent.height,
            params.present_mode.1
        );
        // From here on an error destroys what was made: nothing used it yet.
        let mut swapchain = Swapchain {
            handle,
            extent: UVec2::new(params.extent.width, params.extent.height),
            present_mode: params.present_mode.0,
            images: SmallVec::new(),
            last_presenter: None,
        };
        match swapchain.make_images(params) {
            Ok(()) => Ok(swapchain),
            Err(err) => {
                swapchain.destroy();
                Err(err)
            }
        }
    }

    fn make_images(&mut self, params: &Params) -> Result<()> {
        let device = Ctx::device();
        let images = unsafe { Functions::swapchain().get_swapchain_images(self.handle)? };
        for (i, image) in images.into_iter().enumerate() {
            Functions::set_debug_name(&format!("Swapchain image {i}"), image);
            let info = vk::ImageViewCreateInfo::default()
                .format(params.format.format)
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            let view = unsafe { device.create_image_view(&info, None)? };
            let present = match unsafe { device.create_semaphore(&Default::default(), None) } {
                Ok(present) => present,
                Err(err) => {
                    unsafe { device.destroy_image_view(view, None) };
                    return Err(err.into());
                }
            };
            self.images.push(SwapchainImage {
                image,
                view,
                handle: BindlessHandle::none(),
                state: ImageState::UNDEFINED.atomic(),
                present,
            });
        }
        for (set, flag) in [
            (bindless::Set::Sampled, vk::ImageUsageFlags::SAMPLED),
            (bindless::Set::Storage, vk::ImageUsageFlags::STORAGE),
        ] {
            if !params.usage.contains(flag) {
                continue;
            }
            let mut indices = SmallVec::<[u32; 5]>::new();
            for image in &mut self.images {
                let index = bindless::allocate(set, 1)?;
                *set.index_mut(&mut image.handle) = index;
                indices.push(index);
            }
            bindless::bind_raw(
                self.images.iter().map(|image| (image.view, params.usage)),
                indices.into_iter().map(|index| set.handle(index)),
            );
        }
        Ok(())
    }

    /// Destroys it right away: its images' table entries, tracked accesses, views, slots and
    /// present semaphores, then the swapchain.
    pub(crate) fn destroy(self) {
        let device = Ctx::device();
        for image in &self.images {
            table::take(image.image);
            forget_image(image.image);
            bindless::free_handle(image.handle);
            unsafe {
                device.destroy_image_view(image.view, None);
                device.destroy_semaphore(image.present, None);
            }
        }
        unsafe { Functions::swapchain().destroy_swapchain(self.handle, None) };
    }

    /// Destroys it once no submission or present may use it (`lifetime::retire_presented`).
    pub(crate) fn retire(self) {
        let presenter = self.last_presenter;
        lifetime::retire_presented(presenter, move || self.destroy());
    }
}

/// The format features images of `usage` need.
pub(crate) fn format_features(usage: vk::ImageUsageFlags) -> vk::FormatFeatureFlags {
    use vk::{FormatFeatureFlags as F, ImageUsageFlags as U};
    [
        (U::COLOR_ATTACHMENT, F::COLOR_ATTACHMENT),
        (U::STORAGE, F::STORAGE_IMAGE),
        (U::SAMPLED, F::SAMPLED_IMAGE),
        (U::TRANSFER_SRC, F::TRANSFER_SRC),
        (U::TRANSFER_DST, F::TRANSFER_DST),
    ]
    .into_iter()
    .filter(|(u, _)| usage.contains(*u))
    .fold(F::empty(), |all, (_, f)| all | f)
}

/// The first of `preferred` among the surface's `formats` whose format is `usable`, else the
/// first usable one of `formats`. A surface without formats, or with a single `UNDEFINED`
/// one, takes any: the first usable of `preferred`.
pub(crate) fn select_surface_format(
    formats: &[vk::SurfaceFormatKHR],
    preferred: &[vk::SurfaceFormatKHR],
    usable: impl Fn(vk::Format) -> bool,
) -> Option<vk::SurfaceFormatKHR> {
    let usable = |f: &&vk::SurfaceFormatKHR| usable(f.format);
    match formats {
        [] => preferred.iter().find(usable).copied(),
        [only] if only.format == vk::Format::UNDEFINED => preferred.iter().find(usable).copied(),
        _ => (preferred.iter().filter(|p| formats.contains(p)))
            .chain(formats)
            .find(usable)
            .copied(),
    }
}

/// The present modes lava offers, FIFO (the default, always supported and the most frugal)
/// first, then by falling latency.
const PRESENT_MODES: [(vk::PresentModeKHR, &str); 4] = [
    (vk::PresentModeKHR::FIFO, "FIFO (vsync)"),
    (vk::PresentModeKHR::FIFO_RELAXED, "FIFO relaxed"),
    (vk::PresentModeKHR::MAILBOX, "Mailbox"),
    (vk::PresentModeKHR::IMMEDIATE, "Immediate"),
];

/// The supported entries of `PRESENT_MODES`; FIFO is always supported, even if not reported.
pub(crate) fn supported_present_modes(
    modes: &[vk::PresentModeKHR],
) -> SmallVec<[(vk::PresentModeKHR, &'static str); 4]> {
    PRESENT_MODES
        .into_iter()
        .filter(|(mode, _)| *mode == vk::PresentModeKHR::FIFO || modes.contains(mode))
        .collect()
}

/// The requested size clamped to the surface's extent range, else the surface's current
/// extent, else (when the surface leaves the size to the swapchain, signalled by `u32::MAX`)
/// its minimum extent.
pub(crate) fn select_extent(
    size: Option<UVec2>,
    capabilities: &vk::SurfaceCapabilitiesKHR,
) -> vk::Extent2D {
    let (min, max) = (capabilities.min_image_extent, capabilities.max_image_extent);
    if let Some(size) = size {
        vk::Extent2D {
            width: size.x.clamp(min.width, max.width.max(min.width)),
            height: size.y.clamp(min.height, max.height.max(min.height)),
        }
    } else if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        min
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn format(format: vk::Format, color_space: vk::ColorSpaceKHR) -> vk::SurfaceFormatKHR {
        vk::SurfaceFormatKHR {
            format,
            color_space,
        }
    }

    const BGRA: vk::SurfaceFormatKHR = vk::SurfaceFormatKHR {
        format: vk::Format::B8G8R8A8_UNORM,
        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
    };

    fn any(_: vk::Format) -> bool {
        true
    }

    #[test]
    fn the_first_preferred_format_wins_wherever_the_surface_lists_it() {
        let srgb = format(vk::Format::R8G8B8A8_SRGB, vk::ColorSpaceKHR::SRGB_NONLINEAR);
        let formats = [
            srgb,
            format(
                vk::Format::A2B10G10R10_UNORM_PACK32,
                vk::ColorSpaceKHR::HDR10_ST2084_EXT,
            ),
            BGRA,
        ];
        assert_eq!(select_surface_format(&formats, &[BGRA, srgb], any), Some(BGRA));
        assert_eq!(select_surface_format(&formats, &[srgb, BGRA], any), Some(srgb));
    }

    #[test]
    fn a_format_in_another_colour_space_is_not_the_preferred_one() {
        let formats = [
            format(vk::Format::R8G8B8A8_SRGB, vk::ColorSpaceKHR::SRGB_NONLINEAR),
            format(
                vk::Format::B8G8R8A8_UNORM,
                vk::ColorSpaceKHR::HDR10_ST2084_EXT,
            ),
        ];
        assert_eq!(select_surface_format(&formats, &[BGRA], any), Some(formats[0]));
    }

    /// sRGB formats can't be storage images on most devices.
    #[test]
    fn formats_that_cant_have_the_usage_are_skipped() {
        let srgb = format(vk::Format::B8G8R8A8_SRGB, vk::ColorSpaceKHR::SRGB_NONLINEAR);
        let storable = |f: vk::Format| f != vk::Format::B8G8R8A8_SRGB;
        assert_eq!(select_surface_format(&[srgb, BGRA], &[srgb], storable), Some(BGRA));
        assert_eq!(select_surface_format(&[srgb], &[], storable), None);
        assert_eq!(
            format_features(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE),
            vk::FormatFeatureFlags::COLOR_ATTACHMENT | vk::FormatFeatureFlags::STORAGE_IMAGE
        );
    }

    #[test]
    fn surfaces_without_a_preference_take_the_first_usable_preferred_format() {
        assert_eq!(select_surface_format(&[], &[BGRA], any), Some(BGRA));
        let undefined = [format(
            vk::Format::UNDEFINED,
            vk::ColorSpaceKHR::SRGB_NONLINEAR,
        )];
        assert_eq!(select_surface_format(&undefined, &[BGRA], any), Some(BGRA));
        assert_eq!(select_surface_format(&undefined, &[], any), None);
    }

    #[test]
    fn present_modes_start_with_fifo_and_fall_in_latency() {
        use vk::PresentModeKHR as P;
        let modes = |supported: &[P]| -> Vec<P> {
            supported_present_modes(supported)
                .iter()
                .map(|m| m.0)
                .collect()
        };
        assert_eq!(
            modes(&[P::FIFO, P::MAILBOX, P::IMMEDIATE]),
            [P::FIFO, P::MAILBOX, P::IMMEDIATE]
        );
        assert_eq!(
            modes(&[P::FIFO_RELAXED, P::FIFO]),
            [P::FIFO, P::FIFO_RELAXED]
        );
        assert_eq!(modes(&[]), [P::FIFO]);
    }

    #[test]
    fn extent_prefers_the_requested_size_within_the_surfaces_range() {
        let caps = vk::SurfaceCapabilitiesKHR {
            current_extent: vk::Extent2D {
                width: 800,
                height: 600,
            },
            min_image_extent: vk::Extent2D {
                width: 1,
                height: 1,
            },
            max_image_extent: vk::Extent2D {
                width: 4096,
                height: 4096,
            },
            ..Default::default()
        };
        let requested = select_extent(Some(UVec2::new(320, 240)), &caps);
        assert_eq!((requested.width, requested.height), (320, 240));

        let current = select_extent(None, &caps);
        assert_eq!((current.width, current.height), (800, 600));

        // A window size the surface can't take (an X11 window that just changed, or a
        // minimised one with a 0 maximum) is clamped.
        let clamped = select_extent(Some(UVec2::new(8192, 0)), &caps);
        assert_eq!((clamped.width, clamped.height), (4096, 1));
        let minimised = vk::SurfaceCapabilitiesKHR {
            max_image_extent: vk::Extent2D::default(),
            ..caps
        };
        let none = select_extent(Some(UVec2::new(800, 600)), &minimised);
        assert_eq!((none.width, none.height), (1, 1));
    }

    #[test]
    fn extent_falls_back_to_the_minimum_when_the_surface_has_no_size() {
        let caps = vk::SurfaceCapabilitiesKHR {
            current_extent: vk::Extent2D {
                width: u32::MAX,
                height: u32::MAX,
            },
            min_image_extent: vk::Extent2D {
                width: 64,
                height: 48,
            },
            ..Default::default()
        };
        let extent = select_extent(None, &caps);
        assert_eq!((extent.width, extent.height), (64, 48));
    }
}

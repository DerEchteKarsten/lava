//! Swapchain creation (usage chosen by type, format by preference), image acquisition, resize-driven recreation, and the command that readies an image for presenting
use std::{
    ffi::CStr,
    iter,
    sync::{Arc, Mutex, MutexGuard, atomic::AtomicU32},
};

use crate::{
    bindless::BindlessHandle,
    error::{Error, Result},
};
use crate::vk::{self};
use glam::UVec2;
use lava_macros::validation_trace;
use smallvec::SmallVec;

use crate::{
    bindless,
    command_buffer::CommandBuffer,
    epoch,
    image::{
        D2,
        format::{self, Format},
        slice::ImageView,
        usage::{ColorAttachmentTransferDst, ImageUsage, IsStorage, StaticUsage},
    },
    state::{Ctx, Functions},
    vkobjects::{
        queue::{Binary, Fence, Semaphore, forget_image},
        surface::{Surface, SurfaceHandle},
    },
};

#[derive(Debug)]
struct SwapchainImage {
    image: vk::Image,
    view: vk::ImageView,
    handle: BindlessHandle,
    layout: AtomicU32,
}

/// The images a surface presents, with usage `U` (any static image usage the surface allows;
/// color attachment and transfer destination by default).
#[derive(Debug)]
pub struct Swapchain<U: StaticUsage = ColorAttachmentTransferDst> {
    size: UVec2,
    /// Picked from what the surface supports, by `preferred`; a recreated swapchain may pick
    /// another.
    format: format::Dynamic,
    /// The formats and colour spaces `new` was asked for, in order of preference.
    preferred: SmallVec<[vk::SurfaceFormatKHR; 2]>,
    /// The surface's supported modes of `PRESENT_MODES`, FIFO first, with their names.
    present_modes: SmallVec<[(vk::PresentModeKHR, &'static str); 4]>,
    /// Index into `present_modes`. A recreated swapchain keeps it.
    present_mode: usize,
    handle: vk::SwapchainKHR,
    /// Held while an image is acquired or presented: both need the swapchain externally
    /// synchronised, and a `&Swapchain` can be on several threads.
    sync: Mutex<()>,
    images: SmallVec<[SwapchainImage; 5]>,
    usage: U,
    /// Keeps the surface until the swapchain is destroyed, whenever the `Surface` is dropped.
    surface: Arc<SurfaceHandle>,
}

impl<U: StaticUsage> Swapchain<U> {
    /// The extent of its images.
    pub fn size(&self) -> UVec2 {
        self.size
    }

    /// The format of its images, picked by `new`'s preference.
    pub fn format(&self) -> format::Dynamic {
        self.format
    }

    /// The present modes the surface supports, FIFO first, then by falling latency, with
    /// their names.
    pub fn present_modes(&self) -> &[(vk::PresentModeKHR, &'static str)] {
        &self.present_modes
    }

    /// The index into `present_modes` of the mode it presents with.
    pub fn present_mode(&self) -> usize {
        self.present_mode
    }

    /// Picks the present mode `index` of `present_modes` for the swapchains `recreate` makes;
    /// this one keeps its mode. Panics if there is no such mode.
    pub fn set_present_mode(&mut self, index: usize) {
        assert!(
            index < self.present_modes.len(),
            "present mode {index} of {} supported ones",
            self.present_modes.len()
        );
        self.present_mode = index;
    }

    /// The swapchain locked for a present (`Queue::present`), with its handle.
    pub(crate) fn lock(&self) -> (MutexGuard<'_, ()>, vk::SwapchainKHR) {
        let guard = self.sync.lock().unwrap_or_else(|poison| poison.into_inner());
        (guard, self.handle)
    }

    pub fn image(&self, index: u32) -> ImageView<'_, format::Dynamic, U> {
        let image = &self.images[index as usize];
        ImageView {
            image: image.image,
            view: image.view,
            shader_view: image.view,
            mip_range: (0..1).into(),
            extent: self.size,
            handle: image.handle,
            layouts: std::slice::from_ref(&image.layout),
            format: self.format,
            usage: self.usage,
            dim: D2,
        }
    }

    pub fn num_images(&self) -> usize {
        self.images.len()
    }

    /// The images are unbound; the caller binds them. The image count can change between
    /// swapchains of one surface, e.g. with the present mode (Mailbox often needs one more).
    /// `size` is clamped to what the surface allows right now (its capabilities are
    /// re-queried); a surface without any extent (`Surface::is_minimised`) is an error. The
    /// format is the first of `preferred` the surface has that supports `U`, else the first
    /// such the surface has; none is an error.
    #[validation_trace]
    pub fn new(
        surface: &mut Surface,
        old: Option<&Swapchain<U>>,
        size: Option<UVec2>,
        preferred: &[vk::SurfaceFormatKHR],
    ) -> Result<Self> {
        surface.refresh_capabilities()?;
        let usage = U::VK;
        let instance = Functions::instance();
        let device = Ctx::physical_device().handel;
        let usable = |format: vk::Format| {
            let features =
                unsafe { instance.get_physical_device_format_properties(device, format) };
            features
                .optimal_tiling_features
                .contains(format_features(usage))
        };
        let format = select_surface_format(surface.formats(), preferred, usable).ok_or_else(|| {
            Error::message(format!(
                "the surface has no format whose images support {usage:?}"
            ))
        })?;

        let present_modes = supported_present_modes(surface.present_modes());
        let present_mode_index = old.map_or(0, |old| old.present_mode.min(present_modes.len() - 1));
        let present_mode = present_modes[present_mode_index].0;
        let extent = select_extent(size, surface.capabilities());
        if extent.width == 0 || extent.height == 0 {
            return Err(Error::message(
                "the surface has no extent to create a swapchain for (a minimised window)",
            ));
        }
        let supported = surface.capabilities().supported_usage_flags;
        if !supported.contains(usage) {
            return Err(Error::message(format!(
                "the surface supports the image usages {supported:?}, swapchain images need {usage:?}"
            )));
        }

        let image_count = surface.capabilities().min_image_count;

        let families_indices = [Ctx::gfx_queue_index(), Ctx::present_queue_index()];

        let create_info = {
            let mut builder = vk::SwapchainCreateInfoKHR::default()
                .surface(surface.handle())
                .min_image_count(image_count)
                .image_format(format.format)
                .image_color_space(format.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(usage);

            builder = if Ctx::gfx_queue_index() != Ctx::present_queue_index() {
                builder
                    .image_sharing_mode(vk::SharingMode::CONCURRENT)
                    .queue_family_indices(&families_indices)
            } else {
                builder.image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            };

            if let Some(old) = old {
                builder = builder.old_swapchain(old.handle);
            }

            builder
                .pre_transform(surface.capabilities().current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
        };

        let handle = unsafe { Functions::swapchain().create_swapchain(&create_info, None)? };
        let images = unsafe { Functions::swapchain().get_swapchain_images(handle)? };

        let images = images
            .into_iter()
            .enumerate()
            .map(|(i, image)| -> Result<SwapchainImage> {
                if let Some(debug_utils) = Functions::debug_utils() {
                    let name = format!("Swapchain Image {}\0", i);
                    let name =
                        CStr::from_bytes_with_nul(name.as_bytes()).expect("name is nul terminated");
                    let name_info = vk::DebugUtilsObjectNameInfoEXT::default()
                        .object_handle(image)
                        .object_name(name);
                    unsafe { debug_utils.set_debug_utils_object_name(&name_info) }?;
                }
                let create_info = vk::ImageViewCreateInfo::default()
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::R,
                        g: vk::ComponentSwizzle::G,
                        b: vk::ComponentSwizzle::B,
                        a: vk::ComponentSwizzle::A,
                    })
                    .format(format.format)
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_array_layer: 0,
                        layer_count: 1,
                        base_mip_level: 0,
                        level_count: 1,
                    });
                let view = unsafe { Ctx::device().create_image_view(&create_info, None)? };

                Ok(SwapchainImage {
                    handle: BindlessHandle::none(),
                    image,
                    view,
                    layout: AtomicU32::new(vk::ImageLayout::UNDEFINED.as_raw() as u32),
                })
            })
            .collect::<Result<SmallVec<[SwapchainImage; 5]>>>()?;

        Ok(Self {
            handle,
            format: format::Dynamic(format.format),
            preferred: SmallVec::from_slice(preferred),
            usage: U::default(),
            images,
            size: UVec2::new(extent.width, extent.height),
            present_modes,
            present_mode: present_mode_index,
            sync: Mutex::new(()),
            surface: surface.shared_handle(),
        })
    }

    /// Binds image `i` at the `i`-th of `storage_indices` of the storage set, right away. The
    /// indices are handed to the bindless drop hook (`bindless::on_drop`) once the swapchain is
    /// destroyed. Panics if there are fewer indices than images.
    pub fn bind_storage(&mut self, storage_indices: impl IntoIterator<Item = u32>)
    where
        U: IsStorage,
    {
        let mut indices = storage_indices.into_iter();
        for image in &mut self.images {
            let index = indices.next().expect("an index per swapchain image");
            image.handle = BindlessHandle::storage(index);
        }
        let images = 0..self.images.len() as u32;
        bindless::bind(
            images.clone().map(|i| self.image(i)),
            images.map(|i| self.image(i).handle),
        );
    }

    /// The index of the next image, once `wait_on` is signalled. Holds the swapchain while it
    /// waits for an image, so a present of it from another thread waits too.
    #[validation_trace]
    pub fn aquire_image(&self, wait_on: &Semaphore<Binary>, fence: Option<&Fence>) -> Result<u32> {
        let _span = tracing::info_span!("wait for swapchain image").entered();
        let (_sync, handle) = self.lock();
        let (image_index, _suboptimal) = unsafe {
            Functions::swapchain().acquire_next_image(
                handle,
                u64::MAX,
                wait_on.handle,
                fence.map(|e| e.handle).unwrap_or(vk::Fence::null()),
            )
        }?;
        Ok(image_index)
    }

    /// Replaces the swapchain with one of `size` and returns the old one, which can be dropped
    /// right away: it is destroyed once the frames and presents that may use it are done
    /// (`epoch`). The new images are unbound (see `new`); bind them again. On an error (see
    /// `new`) the old swapchain stays.
    #[validation_trace]
    pub fn recreate(&mut self, surface: &mut Surface, size: UVec2) -> Result<Swapchain<U>> {
        let _span = tracing::info_span!("Swapchain Recreation").entered();
        let preferred = self.preferred.clone();
        let swapchain = Swapchain::new(surface, Some(self), Some(size), &preferred)?;
        Ok(std::mem::replace(self, swapchain))
    }
}

/// Destroyed once the frames and presents that may use it are done (`epoch`).
impl<U: StaticUsage> Drop for Swapchain<U> {
    fn drop(&mut self) {
        let handle = self.handle;
        let images: SmallVec<[_; 5]> = (self.images.iter())
            .map(|i| (i.image, i.view, i.handle))
            .collect();
        // The surface is destroyed after its last swapchain.
        let surface = self.surface.clone();
        epoch::retire_after_present(move || {
            unsafe {
                for (image, view, handle) in images {
                    forget_image(image);
                    bindless::release(handle);
                    Ctx::device().destroy_image_view(view, None);
                }
                Functions::swapchain().destroy_swapchain(handle, None);
            }
            drop(surface);
        });
    }
}

/// The format features images of `usage` need.
fn format_features(usage: vk::ImageUsageFlags) -> vk::FormatFeatureFlags {
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

/// The swapchain's present modes, FIFO (the default, always supported and the most frugal)
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

impl CommandBuffer {
    /// Moves `swapchain_image` into `PRESENT_SRC_KHR`, the layout a present needs, once the
    /// frame's earlier accesses of it are done.
    #[validation_trace]
    pub fn present<'a, F: Format, U: ImageUsage>(
        &'a mut self,
        swapchain_image: ImageView<'a, F, U>,
    ) {
        self.flush_pending(
            iter::once(swapchain_image.access(
                vk::PipelineStageFlags2::empty(),
                vk::AccessFlags2::empty(),
                vk::ImageLayout::PRESENT_SRC_KHR,
            )),
            iter::empty(),
        );
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

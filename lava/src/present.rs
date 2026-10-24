//! Presenting to a window: `PresentSurface` owns a surface (its format and colour space, present mode, composite alpha and HDR metadata), its swapchain and the semaphores and present fences between them, recreates the swapchain when needed and destroys replaced ones once their present fences signalled, and keeps one current image, which the submission that presents it waits for and signals; `present_all` presents several surfaces in one call
use std::{
    collections::VecDeque,
    fmt,
    marker::PhantomData,
    sync::{Arc, atomic::Ordering},
};

use crate::vk;
use glam::UVec2;
#[cfg(feature = "hdr-metadata")]
use glam::Vec2;
use lava_macros::validation_trace;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use smallvec::SmallVec;

#[cfg(feature = "hdr-metadata")]
use crate::features::Gated;
#[cfg(feature = "fifo-latest-ready")]
use crate::features::On;
#[cfg(any(
    feature = "present-fences",
    feature = "hdr-metadata",
    feature = "fifo-latest-ready"
))]
use crate::features::gate;
use crate::{
    bindless::{self, Allocatable, Combined, ImageHandle, Sampled, Storage},
    command_buffer::{CommandBuffer, image_state::ImageState},
    error::{Error, Result},
    global::Live,
    image::{
        D2, ImageInternal,
        format::{self, StaticFormat},
        slice::ImageView,
        usage::{StaticUsage, usage},
    },
    lifetime::{self, Stamp, latest_stamp},
    queue::{
        HasGraphics, lock,
        timeline::{completed, poll},
    },
    state::{Ctx, Functions},
};

/// A recording's present of an acquired image: its submission waits for `acquire` and signals
/// `present`.
pub(crate) struct Present {
    pub(crate) image: vk::Image,
    pub(crate) acquire: vk::Semaphore,
    pub(crate) present: vk::Semaphore,
}

impl<K: HasGraphics> CommandBuffer<'_, K> {
    /// Moves the surface's current image into `PRESENT_SRC_KHR`. The submission waits for its
    /// acquire and signals what `PresentSurface::present` waits for; it must be the one that
    /// first uses the image, and this its last command on the image (a later access panics).
    /// Panics if a submission presented it already.
    #[validation_trace]
    pub fn present<U: StaticUsage>(&mut self, surface: &mut PresentSurface<U>) {
        let current = surface
            .current
            .expect("no image is current: `acquire` one first");
        assert!(
            current.presented.is_none(),
            "a submission presented the current image already: `PresentSurface::present` it"
        );
        let view = surface.current_image();
        self.rec.flush_pending(
            std::iter::once(view.access_in(
                vk::PipelineStageFlags2::empty(),
                vk::AccessFlags2::empty(),
                vk::ImageLayout::PRESENT_SRC_KHR,
            )),
            std::iter::empty(),
        );
        let swapchain = surface.swapchain.as_ref().expect("an image is current");
        self.rec.presents.push(Present {
            image: view.image(),
            acquire: current.acquire,
            present: swapchain.images[current.index as usize].present,
        });
        // Last, so a panic before leaves it unpresented. A recording that doesn't reach the GPU
        // is replaced by one with the same waits and signals, so the image counts as presented
        // by this submission either way.
        let current = surface.current.as_mut().expect("an image is current");
        current.presented = Some((self.rec.timeline, self.rec.value));
    }
}

/// How the presentation engine shows images (`VkPresentModeKHR`). FIFO is always supported.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PresentMode {
    /// Each image at a vertical blank, in order (vsync).
    Fifo,
    /// FIFO, but an image that missed its vertical blank is shown right away (tearing).
    FifoRelaxed,
    /// The newest image at each vertical blank, replacing waiting ones (no tearing, no waits).
    Mailbox,
    /// Right away (tearing).
    Immediate,
    /// FIFO that shows the newest ready image at each vertical blank and drops older ones
    /// (`fifo-latest-ready`).
    #[cfg(feature = "fifo-latest-ready")]
    FifoLatestReady,
}

impl PresentMode {
    /// Every mode lava has, FIFO first.
    pub const ALL: &[Self] = &[
        Self::Fifo,
        Self::FifoRelaxed,
        Self::Mailbox,
        Self::Immediate,
        #[cfg(feature = "fifo-latest-ready")]
        Self::FifoLatestReady,
    ];

    fn vk(self) -> vk::PresentModeKHR {
        match self {
            Self::Fifo => vk::PresentModeKHR::FIFO,
            Self::FifoRelaxed => vk::PresentModeKHR::FIFO_RELAXED,
            Self::Mailbox => vk::PresentModeKHR::MAILBOX,
            Self::Immediate => vk::PresentModeKHR::IMMEDIATE,
            #[cfg(feature = "fifo-latest-ready")]
            Self::FifoLatestReady => vk::PRESENT_MODE_FIFO_LATEST_READY_EXT,
        }
    }

    /// Whether the device can present with it at all (its feature).
    fn on_device(self) -> bool {
        match self {
            #[cfg(feature = "fifo-latest-ready")]
            Self::FifoLatestReady => gate::FifoLatestReady::on(),
            _ => true,
        }
    }
}

/// How a swapchain's alpha blends with what is behind the window (`VkCompositeAlphaFlagBitsKHR`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompositeAlpha {
    /// Alpha is ignored.
    Opaque,
    /// Colours are multiplied by alpha already.
    PreMultiplied,
    /// The compositor multiplies colours by alpha.
    PostMultiplied,
    /// The windowing system decides (set through its own API).
    Inherit,
}

impl CompositeAlpha {
    /// In the order lava picks the default from: the first the surface supports.
    pub const ALL: [Self; 4] = [
        Self::Opaque,
        Self::Inherit,
        Self::PreMultiplied,
        Self::PostMultiplied,
    ];

    fn vk(self) -> vk::CompositeAlphaFlagsKHR {
        match self {
            Self::Opaque => vk::CompositeAlphaFlagsKHR::OPAQUE,
            Self::PreMultiplied => vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
            Self::PostMultiplied => vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
            Self::Inherit => vk::CompositeAlphaFlagsKHR::INHERIT,
        }
    }
}

macro_rules! color_spaces {
    ($($(#[$doc:meta])* $name:ident => $vk:ident,)*) => {
        /// How a surface interprets the values of its images (`VkColorSpaceKHR`). Every one but
        /// [`ColorSpace::SrgbNonlinear`] needs `VK_EXT_swapchain_colorspace`, which lava enables
        /// with a display where the instance has it; a surface lists the ones it can show.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[non_exhaustive]
        pub enum ColorSpace {
            $($(#[$doc])* $name,)*
        }

        impl ColorSpace {
            fn vk(self) -> vk::ColorSpaceKHR {
                match self {
                    $(Self::$name => vk::ColorSpaceKHR::$vk,)*
                }
            }

            fn from_vk(space: vk::ColorSpaceKHR) -> Option<Self> {
                match space {
                    $(vk::ColorSpaceKHR::$vk => Some(Self::$name),)*
                    _ => None,
                }
            }
        }
    };
}

color_spaces! {
    /// sRGB primaries and transfer function: what every surface shows.
    SrgbNonlinear => SRGB_NONLINEAR,
    DisplayP3Nonlinear => DISPLAY_P3_NONLINEAR_EXT,
    /// scRGB: sRGB primaries, linear, values beyond 0..1 (HDR on Windows).
    ExtendedSrgbLinear => EXTENDED_SRGB_LINEAR_EXT,
    ExtendedSrgbNonlinear => EXTENDED_SRGB_NONLINEAR_EXT,
    DisplayP3Linear => DISPLAY_P3_LINEAR_EXT,
    DciP3Nonlinear => DCI_P3_NONLINEAR_EXT,
    Bt709Linear => BT709_LINEAR_EXT,
    Bt709Nonlinear => BT709_NONLINEAR_EXT,
    Bt2020Linear => BT2020_LINEAR_EXT,
    /// HDR10: BT.2020 primaries with the PQ (SMPTE ST 2084) transfer function.
    Hdr10St2084 => HDR10_ST2084_EXT,
    /// BT.2020 primaries with the HLG transfer function.
    Hdr10Hlg => HDR10_HLG_EXT,
    AdobeRgbLinear => ADOBERGB_LINEAR_EXT,
    AdobeRgbNonlinear => ADOBERGB_NONLINEAR_EXT,
    /// Values go to the display as they are.
    PassThrough => PASS_THROUGH_EXT,
}

/// A format and colour space of a surface's images, as [`PresentSurface::new`] prefers them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfaceFormat {
    pub format: format::Dynamic,
    pub color_space: ColorSpace,
}

impl SurfaceFormat {
    /// The format `F` in `color_space`.
    pub fn of<F: StaticFormat>(color_space: ColorSpace) -> Self {
        Self {
            format: format::Dynamic(F::default().vk_format()),
            color_space,
        }
    }

    fn vk(self) -> vk::SurfaceFormatKHR {
        vk::SurfaceFormatKHR {
            format: self.format.0,
            color_space: self.color_space.vk(),
        }
    }
}

/// The display a swapchain's content was mastered for (`VkHdrMetadataEXT`): chromaticities as
/// CIE 1931 xy, luminances in nits.
#[cfg(feature = "hdr-metadata")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HdrMetadata {
    pub red: Vec2,
    pub green: Vec2,
    pub blue: Vec2,
    pub white_point: Vec2,
    pub max_luminance: f32,
    pub min_luminance: f32,
    pub max_content_light_level: f32,
    pub max_frame_average_light_level: f32,
}

#[cfg(feature = "hdr-metadata")]
impl HdrMetadata {
    fn vk(&self) -> vk::HdrMetadataEXT<'static> {
        let xy = |v: Vec2| vk::XYColorEXT { x: v.x, y: v.y };
        vk::HdrMetadataEXT::default()
            .display_primary_red(xy(self.red))
            .display_primary_green(xy(self.green))
            .display_primary_blue(xy(self.blue))
            .white_point(xy(self.white_point))
            .max_luminance(self.max_luminance)
            .min_luminance(self.min_luminance)
            .max_content_light_level(self.max_content_light_level)
            .max_frame_average_light_level(self.max_frame_average_light_level)
    }

    fn apply(&self, swapchain: vk::SwapchainKHR) {
        unsafe { Functions::hdr_metadata().set_hdr_metadata(&[swapchain], &[self.vk()]) };
    }
}

struct SwapchainImage {
    /// The image, its view, handle and tracked state.
    internal: Arc<ImageInternal>,
    present: vk::Semaphore,
}

struct Swapchain {
    handle: vk::SwapchainKHR,
    extent: UVec2,
    present_mode: PresentMode,
    composite_alpha: CompositeAlpha,
    images: SmallVec<[SwapchainImage; 5]>,
    /// The fences of its presents not known to be done.
    fences: Vec<vk::Fence>,
}

impl Swapchain {
    fn destroy(self) {
        let device = Ctx::device();
        for image in self.images {
            // The internal destroys its view and frees its handle right away; nothing else
            // holds it once no submission reaches the swapchain.
            drop(image.internal);
            unsafe { device.destroy_semaphore(image.present, None) };
        }
        for fence in self.fences {
            unsafe { device.destroy_fence(fence, None) };
        }
        unsafe { Functions::swapchain().destroy_swapchain(self.handle, None) };
    }
}

/// A window's surface and its swapchain of images of usage `U`, each bound in every bindless
/// set `U` allows. A frame acquires an image, records into [`Self::current_image`] in a
/// submission that ends with `cmd.present(&surface)`, and presents it. An image no submission
/// used stays current for the next frame.
pub struct PresentSurface<U: StaticUsage = usage![ColorAttachment, TransferDst]> {
    surface: vk::SurfaceKHR,
    format: vk::SurfaceFormatKHR,
    /// Whether `format` is one `new` was asked for.
    preferred: bool,
    present_modes: SmallVec<[PresentMode; 5]>,
    present_mode: PresentMode,
    composite_alphas: vk::CompositeAlphaFlagsKHR,
    composite_alpha: CompositeAlpha,
    #[cfg(feature = "hdr-metadata")]
    hdr: Option<HdrMetadata>,
    /// The surface's capabilities, queried for a window of this size; dropped on a suboptimal or
    /// out-of-date result and when the swapchain is made.
    capabilities: Option<(UVec2, vk::SurfaceCapabilitiesKHR)>,
    swapchain: Option<Swapchain>,
    /// Replaced swapchains, with what they wait for.
    old: Vec<(Swapchain, Stamp)>,
    recreate: bool,
    /// Acquire semaphores nothing uses.
    free: Vec<vk::Semaphore>,
    /// Acquire semaphores a submission waited for, free once it finished.
    waited: VecDeque<(vk::Semaphore, usize, u64)>,
    /// Present fences nothing uses (signalled, or reset by a present that didn't happen).
    fences: Vec<vk::Fence>,
    /// The acquired image, with the submission that presented it.
    current: Option<Current>,
    _usage: PhantomData<fn() -> U>,
    /// Last, so it counts the surface until its drop finished.
    _live: Live,
}

/// The image a surface acquired.
#[derive(Clone, Copy, Debug)]
struct Current {
    index: u32,
    acquire: vk::Semaphore,
    /// The timeline and value of the recording that presented it (`cmd.present`), set while
    /// it records; `PresentSurface::present` runs only after its submission.
    presented: Option<(usize, u64)>,
}

impl<U: StaticUsage> fmt::Debug for PresentSurface<U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PresentSurface")
            .field("format", &self.format)
            .field("present_mode", &self.present_mode)
            .field("size", &self.size())
            .finish()
    }
}

fn surface_fn() -> Result<&'static ash::khr::surface::Instance> {
    Functions::surface().ok_or_else(|| {
        Error::Unsupported(
            "lava was initialised without a display handle; surfaces are unavailable".into(),
        )
    })
}

/// Windows need present fences, which tell when a swapchain's presents are done with it.
fn present_fences() -> Result<()> {
    #[cfg(feature = "present-fences")]
    {
        use crate::features::Feature;
        gate::PresentFences::check()?;
        Ok(())
    }
    #[cfg(not(feature = "present-fences"))]
    Err(crate::features::MissingFeature {
        feature: "present-fences",
    }
    .into())
}

/// Panics on a thread that records a submission: a present or acquire there would use a
/// semaphore its own submission signals, and destroying a swapchain locks the recording's state.
#[track_caller]
fn not_recording(what: &str, when: &str) {
    assert!(
        !lifetime::RECORDING.get(),
        "`{what}` is called while a submission records: call it {when} `submit`"
    );
}

impl<U: StaticUsage> PresentSurface<U> {
    /// A surface for `window`, whose images have the first of `preferred` the surface has whose
    /// format supports `U`; else another such format lava has a type for (logged, and
    /// [`Self::is_preferred_format`] is false then). The first [`Self::acquire`] makes the
    /// swapchain. Needs `present-fences` (`MissingFeature` without it).
    ///
    /// # Safety
    /// `display` and `window` must be valid handles, and the window must outlive the surface.
    #[validation_trace]
    pub unsafe fn new(
        display: &RawDisplayHandle,
        window: &RawWindowHandle,
        preferred: &[SurfaceFormat],
    ) -> Result<Self> {
        let live = Live::new();
        let surface_fn = surface_fn()?;
        present_fences()?;
        let (entry, instance) = (Functions::entry(), Functions::instance());
        let surface =
            unsafe { ash_window::create_surface(entry, instance, *display, *window, None)? };
        let made = (|| {
            let device = Ctx::physical_device().handle;
            let family = Ctx::get().queues.graphics().index;
            let can_present =
                unsafe { surface_fn.get_physical_device_surface_support(device, family, surface)? };
            if !can_present {
                return Err(Error::Unsupported(
                    "the graphics queue family cannot present to this surface".into(),
                ));
            }
            let formats =
                unsafe { surface_fn.get_physical_device_surface_formats(device, surface)? };
            let modes =
                unsafe { surface_fn.get_physical_device_surface_present_modes(device, surface)? };
            let capabilities =
                unsafe { surface_fn.get_physical_device_surface_capabilities(device, surface)? };
            let usage = U::VK;
            if !capabilities.supported_usage_flags.contains(usage) {
                return Err(Error::Unsupported(format!(
                    "the surface supports the image usages {:?}, swapchain images need {usage:?}",
                    capabilities.supported_usage_flags
                )));
            }
            use vk::{FormatFeatureFlags as F, ImageUsageFlags as I};
            let needed = [
                (I::COLOR_ATTACHMENT, F::COLOR_ATTACHMENT),
                (I::STORAGE, F::STORAGE_IMAGE),
                (I::SAMPLED, F::SAMPLED_IMAGE),
                (I::TRANSFER_SRC, F::TRANSFER_SRC),
                (I::TRANSFER_DST, F::TRANSFER_DST),
            ];
            let needed = (needed.into_iter())
                .filter(|(u, _)| usage.contains(*u))
                .fold(F::empty(), |all, (_, f)| all | f);
            let usable = |f: &vk::SurfaceFormatKHR| {
                let properties =
                    unsafe { instance.get_physical_device_format_properties(device, f.format) };
                properties.optimal_tiling_features.contains(needed)
            };
            let wanted: SmallVec<[vk::SurfaceFormatKHR; 4]> =
                preferred.iter().map(|f| f.vk()).collect();
            // No preference of the surface (none, or one `UNDEFINED`): any of `preferred`.
            let any = matches!(
                &formats[..],
                [] | [vk::SurfaceFormatKHR {
                    format: vk::Format::UNDEFINED,
                    ..
                }]
            );
            let chosen = (wanted.iter().copied())
                .filter(|p| any || formats.contains(p))
                .find(usable);
            let (format, preferred_format) = match chosen {
                Some(format) => (format, true),
                None => {
                    // Lava reads the image's format as a `format::Dynamic`.
                    let known = |f: &vk::SurfaceFormatKHR| {
                        format::Dynamic(f.format).known()
                            && ColorSpace::from_vk(f.color_space).is_some()
                    };
                    let format = (formats.iter().copied()).filter(known).find(usable);
                    let format = format.ok_or_else(|| {
                        Error::Unsupported(format!(
                            "the surface has no format whose images support {usage:?}"
                        ))
                    })?;
                    tracing::warn!(
                        "the surface has none of the formats {preferred:?} with images of {usage:?}: its images are {format:?}"
                    );
                    (format, false)
                }
            };
            let present_modes = (PresentMode::ALL.iter().copied())
                .filter(|mode| mode.on_device())
                .filter(|mode| *mode == PresentMode::Fifo || modes.contains(&mode.vk()))
                .collect();
            let composite_alphas = capabilities.supported_composite_alpha;
            let composite_alpha = (CompositeAlpha::ALL.into_iter())
                .find(|alpha| composite_alphas.contains(alpha.vk()))
                .ok_or_else(|| {
                    Error::Unsupported("the surface supports no composite alpha".into())
                })?;
            Ok(Self {
                surface,
                format,
                preferred: preferred_format,
                present_modes,
                present_mode: PresentMode::Fifo,
                composite_alphas,
                composite_alpha,
                #[cfg(feature = "hdr-metadata")]
                hdr: None,
                capabilities: None,
                swapchain: None,
                old: Vec::new(),
                recreate: false,
                free: Vec::new(),
                waited: VecDeque::new(),
                fences: Vec::new(),
                current: None,
                _usage: PhantomData,
                _live: live,
            })
        })();
        if made.is_err() {
            unsafe { surface_fn.destroy_surface(surface, None) };
        }
        made
    }

    pub fn format(&self) -> format::Dynamic {
        format::Dynamic(self.format.format)
    }

    pub fn color_space(&self) -> ColorSpace {
        ColorSpace::from_vk(self.format.color_space).expect("lava picks colour spaces it has")
    }

    pub fn surface_format(&self) -> SurfaceFormat {
        SurfaceFormat {
            format: self.format(),
            color_space: self.color_space(),
        }
    }

    /// Whether the images have one of the formats `new` preferred; false when the surface had
    /// none of them and lava picked another.
    pub fn is_preferred_format(&self) -> bool {
        self.preferred
    }

    /// The extent of the swapchain's images, zero before the first acquire.
    pub fn size(&self) -> UVec2 {
        self.swapchain.as_ref().map_or(UVec2::ZERO, |s| s.extent)
    }

    pub fn num_images(&self) -> usize {
        self.swapchain.as_ref().map_or(0, |s| s.images.len())
    }

    /// The present modes the surface supports, in [`PresentMode::ALL`]'s order (FIFO first).
    pub fn present_modes(&self) -> &[PresentMode] {
        &self.present_modes
    }

    pub fn supports(&self, mode: PresentMode) -> bool {
        self.present_modes.contains(&mode)
    }

    pub fn present_mode(&self) -> PresentMode {
        self.present_mode
    }

    /// Presents with `mode` from the next acquire on, which recreates the swapchain; false (and
    /// nothing changes) if the surface doesn't support it.
    pub fn set_present_mode(&mut self, mode: PresentMode) -> bool {
        let supported = self.supports(mode);
        if supported {
            self.present_mode = mode;
        }
        supported
    }

    pub fn composite_alpha(&self) -> CompositeAlpha {
        self.composite_alpha
    }

    pub fn supports_composite_alpha(&self, alpha: CompositeAlpha) -> bool {
        self.composite_alphas.contains(alpha.vk())
    }

    /// Blends with `alpha` from the next acquire on, which recreates the swapchain; false (and
    /// nothing changes) if the surface doesn't support it. The default is the first of
    /// [`CompositeAlpha::ALL`] it supports.
    pub fn set_composite_alpha(&mut self, alpha: CompositeAlpha) -> bool {
        let supported = self.supports_composite_alpha(alpha);
        if supported {
            self.composite_alpha = alpha;
        }
        supported
    }

    /// Tells the display what the images were mastered for (`vkSetHdrMetadataEXT`), now and for
    /// every swapchain the surface makes later.
    #[cfg(feature = "hdr-metadata")]
    pub fn set_hdr_metadata(&mut self, metadata: &HdrMetadata) -> Gated<gate::HdrMetadata, ()> {
        crate::features::gated::<gate::HdrMetadata, _>(|| {
            if let Some(swapchain) = &self.swapchain {
                metadata.apply(swapchain.handle);
            }
            self.hdr = Some(*metadata);
        })
    }

    pub fn handle(&self) -> vk::SurfaceKHR {
        self.surface
    }

    /// The surface's capabilities for a window of `size`: the cached ones unless the size
    /// changed or the swapchain needs recreating.
    fn capabilities(&mut self, size: UVec2) -> Result<vk::SurfaceCapabilitiesKHR> {
        match self.capabilities {
            Some((asked, capabilities)) if asked == size && !self.recreate => Ok(capabilities),
            _ => {
                let physical_device = Ctx::physical_device().handle;
                let capabilities = unsafe {
                    surface_fn()?
                        .get_physical_device_surface_capabilities(physical_device, self.surface)?
                };
                self.capabilities = Some((size, capabilities));
                Ok(capabilities)
            }
        }
    }

    /// Makes an image of a window of `size` current, and returns whether one is (`false` while
    /// the window has no extent). An image a submission presented is presented first; one no
    /// submission used stays current, at its size, so a resize takes effect a frame later.
    /// Its first access transitions it from `UNDEFINED`: an acquired image's contents are
    /// discarded. Panics while a submission records on this thread.
    #[validation_trace]
    pub fn acquire(&mut self, size: UVec2) -> Result<bool> {
        not_recording("PresentSurface::acquire", "before or after");
        let device = Ctx::device();
        match self.current {
            Some(current) if current.presented.is_none() => {
                // Only a failed recording can have used it (one that uses it must present
                // it), and that one's barriers were dropped: its contents are still undefined.
                let swapchain = self.swapchain.as_ref().expect("an image is current");
                let state = ImageState {
                    acquired: true,
                    ..ImageState::UNDEFINED
                };
                swapchain.images[current.index as usize].internal.states[0]
                    .store(state.pack(), Ordering::Relaxed);
                return Ok(size.x > 0 && size.y > 0);
            }
            Some(_) => self.present()?,
            None => {}
        }
        let free = &mut self.free;
        self.waited.retain(|&(semaphore, t, v)| {
            let done = completed(t) >= v || poll(t).is_ok_and(|done| done >= v);
            if done {
                free.push(semaphore);
            }
            !done
        });
        let signalled =
            |fence: &vk::Fence| unsafe { device.get_fence_status(*fence) }.unwrap_or(false);
        if let Some(swapchain) = &mut self.swapchain {
            for fence in swapchain.fences.extract_if(.., |fence| signalled(fence)) {
                self.fences.push(fence);
            }
        }
        // A replaced swapchain goes once no frame reaches it and its presents are done.
        for (swapchain, _) in (self.old).extract_if(.., |(swapchain, stamp)| {
            swapchain.fences.iter().all(signalled) && stamp.is_done()
        }) {
            swapchain.destroy();
        }
        if size.x == 0 || size.y == 0 {
            return Ok(false);
        }
        for _ in 0..3 {
            let caps = self.capabilities(size)?;
            let (min, max) = (caps.min_image_extent, caps.max_image_extent);
            if max.width == 0 || max.height == 0 {
                return Ok(false);
            }
            let extent = vk::Extent2D {
                width: size.x.clamp(min.width, max.width.max(min.width)),
                height: size.y.clamp(min.height, max.height.max(min.height)),
            };
            let stale = self.swapchain.as_ref().is_none_or(|s| {
                s.extent != UVec2::new(extent.width, extent.height)
                    || s.present_mode != self.present_mode
                    || s.composite_alpha != self.composite_alpha
            });
            if stale || self.recreate {
                self.recreate_swapchain(&caps, extent)?;
            }
            let semaphore = match self.free.pop() {
                Some(semaphore) => semaphore,
                None => unsafe { device.create_semaphore(&Default::default(), None)? },
            };
            let swapchain = self.swapchain.as_ref().expect("made above");
            let acquired = unsafe {
                let _span = tracing::info_span!("wait for swapchain image").entered();
                let fence = vk::Fence::null();
                Functions::swapchain().acquire_next_image(
                    swapchain.handle,
                    u64::MAX,
                    semaphore,
                    fence,
                )
            };
            match acquired {
                Ok((index, suboptimal)) => {
                    self.recreate |= suboptimal;
                    // Its contents are discarded, so its first access transitions from
                    // `UNDEFINED` and waits only for the acquire.
                    let state = ImageState {
                        acquired: true,
                        ..ImageState::UNDEFINED
                    };
                    swapchain.images[index as usize].internal.states[0]
                        .store(state.pack(), Ordering::Relaxed);
                    let acquire = semaphore;
                    self.current = Some(Current {
                        index,
                        acquire,
                        presented: None,
                    });
                    return Ok(true);
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.free.push(semaphore);
                    self.recreate = true;
                }
                Err(err) => {
                    self.free.push(semaphore);
                    return Err(err.into());
                }
            }
        }
        Err(Error::SurfaceOutOfDate)
    }

    /// Replaces the swapchain by one of `extent` (none is current).
    fn recreate_swapchain(
        &mut self,
        caps: &vk::SurfaceCapabilitiesKHR,
        extent: vk::Extent2D,
    ) -> Result<()> {
        let _span = tracing::info_span!("recreate swapchain").entered();
        let device = Ctx::device();
        // Vulkan retires the old swapchain even when making the new one fails, so it is
        // replaced first: a failed creation leaves none, and the next one starts afresh.
        let old = self.swapchain.take();
        let old_handle = old
            .as_ref()
            .map_or_else(vk::SwapchainKHR::null, |s| s.handle);
        if let Some(old) = old {
            self.old.push((old, latest_stamp()));
        }
        self.capabilities = None;
        let mode = self.present_mode.vk();
        let usage = U::VK;
        let modes = [mode];
        let mut modes_info =
            vk::SwapchainPresentModesCreateInfoEXT::default().present_modes(&modes);
        let info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(caps.min_image_count)
            .image_format(self.format.format)
            .image_color_space(self.format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(usage)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(self.composite_alpha.vk())
            .present_mode(mode)
            .clipped(true)
            .old_swapchain(old_handle)
            .push_next(&mut modes_info);
        let handle = unsafe { Functions::swapchain().create_swapchain(&info, None)? };
        tracing::info!(
            "made a swapchain of {}x{} images, presenting {:?}",
            extent.width,
            extent.height,
            self.present_mode
        );
        let mut swapchain = Swapchain {
            handle,
            extent: UVec2::new(extent.width, extent.height),
            present_mode: self.present_mode,
            composite_alpha: self.composite_alpha,
            images: SmallVec::new(),
            fences: Vec::new(),
        };
        let made = (|| -> Result<()> {
            let images = unsafe { Functions::swapchain().get_swapchain_images(handle)? };
            for (i, image) in images.into_iter().enumerate() {
                Functions::set_debug_name(&format!("Swapchain image {i}"), image);
                let info = vk::ImageViewCreateInfo::default()
                    .format(self.format.format)
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
                let internal = ImageInternal::swapchain(
                    image,
                    self.format.format,
                    usage,
                    view,
                    ImageHandle::NULL,
                );
                swapchain.images.push(SwapchainImage {
                    internal: Arc::new(internal),
                    present,
                });
            }
            // One handle per image, with every descriptor its usage allows.
            let sampled = usage.contains(vk::ImageUsageFlags::SAMPLED);
            let storage = usage.contains(vk::ImageUsageFlags::STORAGE);
            let (width, new): (u32, fn(u32) -> ImageHandle) = match (sampled, storage) {
                (true, true) => (Combined::WIDTH, |slot| {
                    ImageHandle::<Combined>::new(slot).erase()
                }),
                (true, false) => (Sampled::WIDTH, |slot| {
                    ImageHandle::<Sampled>::new(slot).erase()
                }),
                (false, true) => (Storage::WIDTH, |slot| {
                    ImageHandle::<Storage>::new(slot).erase()
                }),
                (false, false) => return Ok(()),
            };
            for image in &swapchain.images {
                image
                    .internal
                    .whole
                    .set_handle(new(bindless::allocate(width)?));
            }
            let format = self.format.format;
            let images = swapchain.images.iter().map(|image| bindless::ShaderImage {
                image: image.internal.image,
                format,
                aspects: vk::ImageAspectFlags::COLOR,
                view_type: vk::ImageViewType::TYPE_2D,
                mips: (0..1).into(),
                layers: (0..1).into(),
                usage,
            });
            bindless::write_descriptors(
                images,
                swapchain
                    .images
                    .iter()
                    .map(|image| image.internal.whole.handle()),
            )?;
            Ok(())
        })();
        if let Err(err) = made {
            swapchain.destroy();
            return Err(err);
        }
        #[cfg(feature = "hdr-metadata")]
        if let Some(hdr) = &self.hdr {
            hdr.apply(handle);
        }
        self.swapchain = Some(swapchain);
        self.recreate = false;
        Ok(())
    }

    /// The view of the current image, for the submission that draws into it and presents it.
    /// Panics before the first [`Self::acquire`].
    pub fn current_image(&self) -> ImageView<'_, format::Dynamic, U> {
        let current = self
            .current
            .expect("no image is current: `acquire` one first");
        let swapchain = self.swapchain.as_ref().expect("an image is current");
        let i = &swapchain.images[current.index as usize];
        ImageView {
            internal: &i.internal,
            view: i.internal.whole.view,
            mip_range: (0..1).into(),
            layers: (0..1).into(),
            extent: swapchain.extent,
            handle: i.internal.whole.handle(),
            format: self.format(),
            usage: U::default(),
            dim: D2,
        }
    }

    /// Presents the current image once a submission recorded `cmd.present(&surface)` of it;
    /// until then it stays current. [`present_all`] presents several surfaces at once. Panics
    /// while a submission records on this thread: call it after `submit`.
    #[validation_trace]
    pub fn present(&mut self) -> Result<()> {
        present_all(&mut [self])
    }
}

mod private {
    use crate::{error::Result, vk};

    /// A present one surface adds to a `vkQueuePresentKHR` call.
    pub struct Prepared {
        pub(super) swapchain: vk::SwapchainKHR,
        pub(super) index: u32,
        pub(super) present: vk::Semaphore,
        pub(super) fence: vk::Fence,
        pub(super) acquire: vk::Semaphore,
        /// The timeline and value of the submission that presented the image.
        pub(super) presented: (usize, u64),
    }

    pub trait Surface {
        /// The present of the current image, if a submission presented it, with a fresh fence.
        fn prepare(&mut self) -> Result<Option<Prepared>>;
        /// Takes back what `prepare` gave out, for a present that wasn't called.
        fn unprepare(&mut self, prepared: Prepared);
        /// After the present call: the image isn't current any more.
        fn presented(&mut self, prepared: Prepared, result: vk::Result) -> Result<()>;
    }
}

/// A [`PresentSurface`] of any usage, for [`present_all`]. Only lava implements it.
pub trait AnySurface: private::Surface {}
impl<U: StaticUsage> AnySurface for PresentSurface<U> {}

impl<U: StaticUsage> private::Surface for PresentSurface<U> {
    fn prepare(&mut self) -> Result<Option<private::Prepared>> {
        let Some(Current {
            index,
            acquire,
            presented: Some(presented),
        }) = self.current
        else {
            return Ok(None);
        };
        let device = Ctx::device();
        let fence = match self.fences.pop() {
            Some(fence) => match unsafe { device.reset_fences(&[fence]) } {
                Ok(()) => fence,
                Err(err) => {
                    self.fences.push(fence);
                    return Err(err.into());
                }
            },
            None => unsafe { device.create_fence(&Default::default(), None)? },
        };
        let swapchain = self.swapchain.as_ref().expect("an image was acquired");
        Ok(Some(private::Prepared {
            swapchain: swapchain.handle,
            index,
            present: swapchain.images[index as usize].present,
            fence,
            acquire,
            presented,
        }))
    }

    fn unprepare(&mut self, prepared: private::Prepared) {
        self.fences.push(prepared.fence);
    }

    fn presented(&mut self, prepared: private::Prepared, result: vk::Result) -> Result<()> {
        self.current = None;
        let (t, v) = prepared.presented;
        self.waited.push_back((prepared.acquire, t, v));
        let swapchain = self.swapchain.as_mut().expect("an image was acquired");
        swapchain.fences.push(prepared.fence);
        match result {
            vk::Result::SUCCESS => Ok(()),
            vk::Result::SUBOPTIMAL_KHR | vk::Result::ERROR_OUT_OF_DATE_KHR => {
                self.recreate = true;
                Ok(())
            }
            err => Err(err.into()),
        }
    }
}

/// Presents the current image of every surface whose image a submission presented, in one
/// `vkQueuePresentKHR` on graphics queue 0, so windows show their frames together; the others
/// keep theirs. Returns the first error of any surface, after handling all of them. Panics while
/// a submission records on this thread: call it after `submit`.
#[validation_trace]
pub fn present_all(surfaces: &mut [&mut dyn AnySurface]) -> Result<()> {
    not_recording("present", "after");
    let mut prepared: SmallVec<[(usize, private::Prepared); 2]> = SmallVec::new();
    for (i, surface) in surfaces.iter_mut().enumerate() {
        match surface.prepare() {
            Ok(Some(present)) => prepared.push((i, present)),
            Ok(None) => {}
            Err(err) => {
                for (j, present) in prepared {
                    surfaces[j].unprepare(present);
                }
                return Err(err);
            }
        }
    }
    if prepared.is_empty() {
        return Ok(());
    }
    let swapchains: SmallVec<[_; 2]> = prepared.iter().map(|(_, p)| p.swapchain).collect();
    let indices: SmallVec<[_; 2]> = prepared.iter().map(|(_, p)| p.index).collect();
    let waits: SmallVec<[_; 2]> = prepared.iter().map(|(_, p)| p.present).collect();
    let fences: SmallVec<[_; 2]> = prepared.iter().map(|(_, p)| p.fence).collect();
    // Unwritten results take the call's.
    let unwritten = vk::Result::from_raw(i32::MIN);
    let mut results: SmallVec<[_; 2]> = prepared.iter().map(|_| unwritten).collect();
    let presented = {
        let mut fence_info = vk::SwapchainPresentFenceInfoEXT::default().fences(&fences);
        let info = vk::PresentInfoKHR::default()
            .swapchains(&swapchains)
            .image_indices(&indices)
            .wait_semaphores(&waits)
            .results(&mut results)
            .push_next(&mut fence_info);
        let _span = tracing::info_span!("present").entered();
        let queue = lock(&Ctx::get().queues.graphics().queues[0]);
        match unsafe { Functions::swapchain().queue_present(*queue, &info) } {
            Ok(false) => vk::Result::SUCCESS,
            Ok(true) => vk::Result::SUBOPTIMAL_KHR,
            Err(err) => err,
        }
    };
    let mut first_error = None;
    for ((i, present), result) in prepared.into_iter().zip(results) {
        let result = if result == unwritten {
            presented
        } else {
            result
        };
        if let Err(err) = surfaces[i].presented(present, result) {
            first_error.get_or_insert(err);
        }
    }
    first_error.map_or(Ok(()), Err)
}

impl<U: StaticUsage> Drop for PresentSurface<U> {
    fn drop(&mut self) {
        // A current image left a semaphore signalled (its acquire, or its present once a
        // submission presented it), which a submission consumes before it is destroyed.
        if let Some(current) = self.current
            && let Some(swapchain) = &self.swapchain
        {
            let signalled = match current.presented {
                Some(_) => swapchain.images[current.index as usize].present,
                None => current.acquire,
            };
            let device = Ctx::device();
            let waits = [vk::SemaphoreSubmitInfo::default()
                .semaphore(signalled)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
            let consumed = (|| -> Result<()> {
                let fence = unsafe { device.create_fence(&Default::default(), None)? };
                let submitted = {
                    let queue = lock(&Ctx::get().queues.graphics().queues[0]);
                    let info = vk::SubmitInfo2::default().wait_semaphore_infos(&waits);
                    unsafe { device.queue_submit2(*queue, &[info], fence) }
                };
                let waited = submitted
                    .and_then(|()| unsafe { device.wait_for_fences(&[fence], true, u64::MAX) });
                unsafe { device.destroy_fence(fence, None) };
                Ok(waited?)
            })();
            if let Err(err) = consumed {
                tracing::error!(%err, "failed to wait for the current swapchain image");
            }
            self.free.push(current.acquire);
        }
        let mut swapchains: Vec<Swapchain> = self.old.drain(..).map(|(s, _)| s).collect();
        swapchains.extend(self.swapchain.take());
        let fences: Vec<vk::Fence> = swapchains
            .iter()
            .flat_map(|s| s.fences.iter().copied())
            .collect();
        let device = Ctx::device();
        if !fences.is_empty()
            && unsafe { device.wait_for_fences(&fences, true, 1_000_000_000) }.is_err()
        {
            tracing::warn!("a present's fence didn't signal before its surface was dropped");
        }
        let mut semaphores = std::mem::take(&mut self.free);
        semaphores.extend(self.waited.drain(..).map(|(semaphore, ..)| semaphore));
        let fences = std::mem::take(&mut self.fences);
        let surface = self.surface;
        lifetime::retire(move || {
            for swapchain in swapchains {
                swapchain.destroy();
            }
            let device = Ctx::device();
            for semaphore in semaphores {
                unsafe { device.destroy_semaphore(semaphore, None) };
            }
            for fence in fences {
                unsafe { device.destroy_fence(fence, None) };
            }
            if let Some(surface_fn) = Functions::surface() {
                unsafe { surface_fn.destroy_surface(surface, None) };
            }
        });
    }
}

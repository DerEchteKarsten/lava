//! Presenting to a window: `PresentSurface` owns a surface, its swapchain and the semaphores between them, recreates the swapchain when needed and destroys replaced ones once their presents are done, and keeps one current image, which the submission that presents it waits for and signals
use std::{
    collections::VecDeque,
    fmt,
    marker::PhantomData,
    sync::atomic::{AtomicU64, Ordering},
};

use crate::vk;
use glam::UVec2;
use lava_macros::validation_trace;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use smallvec::SmallVec;

use crate::{
    bindless::{self, Allocatable, Combined, ImageHandle, Sampled, Storage},
    command_buffer::{
        CommandBuffer,
        image_state::ImageState,
    },
    error::{Error, Result},
    features::{On, gate},
    image::{
        D2, format,
        slice::ImageView,
        usage::{ColorAttachmentTransferDst, StaticUsage},
    },
    lifetime::{self, Stamp, latest_stamp},
    queue::{
        HasGraphics, lock,
        timeline::{completed, forget_image, poll},
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
    /// first uses the image. Panics if a submission presented it already.
    #[validation_trace]
    pub fn present<U: StaticUsage>(&mut self, surface: &mut PresentSurface<U>) {
        let current = surface.current.as_mut().expect("no image is current: `acquire` one first");
        assert!(
            current.presented.is_none(),
            "a submission presented the current image already: `PresentSurface::present` it"
        );
        // A recording that doesn't reach the GPU is replaced by one with the same waits and
        // signals, so the image counts as presented by this submission either way.
        current.presented = Some((self.rec.timeline, self.rec.value));
        let (index, acquire) = (current.index, current.acquire);
        let view = surface.current_image();
        self.rec.flush_pending(
            std::iter::once(view.access(
                vk::PipelineStageFlags2::empty(),
                vk::AccessFlags2::empty(),
                vk::ImageLayout::PRESENT_SRC_KHR,
            )),
            std::iter::empty(),
        );
        let swapchain = surface.swapchain.as_ref().expect("an image is current");
        self.rec.presents.push(Present {
            image: view.image,
            acquire,
            present: swapchain.images[index as usize].present,
        });
    }
}

struct SwapchainImage {
    image: vk::Image,
    view: vk::ImageView,
    handle: ImageHandle,
    state: AtomicU64,
    present: vk::Semaphore,
}

struct Swapchain {
    handle: vk::SwapchainKHR,
    extent: UVec2,
    present_mode: usize,
    images: SmallVec<[SwapchainImage; 5]>,
    /// The fences of its presents not known to be done (`present-fences`).
    fences: Vec<vk::Fence>,
    /// The timeline and value of the last submission that presented one of its images.
    presenter: Option<(usize, u64)>,
}

impl Swapchain {
    fn destroy(self) {
        let device = Ctx::device();
        for image in &self.images {
            forget_image(image.image);
            bindless::free_handle(image.handle);
            unsafe {
                device.destroy_image_view(image.view, None);
                device.destroy_semaphore(image.present, None);
            }
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
pub struct PresentSurface<U: StaticUsage = ColorAttachmentTransferDst> {
    surface: vk::SurfaceKHR,
    format: vk::SurfaceFormatKHR,
    present_modes: SmallVec<[(vk::PresentModeKHR, &'static str); 4]>,
    present_mode: usize,
    swapchain: Option<Swapchain>,
    /// Replaced swapchains, with what they wait for.
    old: Vec<(Swapchain, Stamp)>,
    recreate: bool,
    /// Acquire semaphores nothing uses.
    free: Vec<vk::Semaphore>,
    /// Acquire semaphores a submission waited for, free once it finished.
    waited: VecDeque<(vk::Semaphore, usize, u64)>,
    /// Signalled present fences.
    fences: Vec<vk::Fence>,
    /// The acquired image, with the submission that presented it.
    current: Option<Current>,
    _usage: PhantomData<fn() -> U>,
}

/// The image a surface acquired.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Current {
    index: u32,
    acquire: vk::Semaphore,
    /// The timeline and value of the submission that presented it.
    pub(crate) presented: Option<(usize, u64)>,
}

impl<U: StaticUsage> fmt::Debug for PresentSurface<U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PresentSurface")
            .field("format", &self.format)
            .field("present_mode", &self.present_modes[self.present_mode].1)
            .field("size", &self.size())
            .finish()
    }
}

fn surface_fn() -> Result<&'static ash::khr::surface::Instance> {
    Functions::surface().ok_or_else(|| {
        Error::message("lava was initialised without a display handle; surfaces are unavailable")
    })
}

impl<U: StaticUsage> PresentSurface<U> {
    /// A surface for `window`, whose images have the first of `preferred` the surface has whose
    /// format supports `U`, else any such. The first [`Self::acquire`] makes the swapchain.
    ///
    /// # Safety
    /// `display` and `window` must be valid handles, and the window must outlive the surface.
    #[validation_trace]
    pub unsafe fn new(
        display: &RawDisplayHandle,
        window: &RawWindowHandle,
        preferred: &[vk::SurfaceFormatKHR],
    ) -> Result<Self> {
        let surface_fn = surface_fn()?;
        let (entry, instance) = (Functions::entry(), Functions::instance());
        let surface = unsafe { ash_window::create_surface(entry, instance, *display, *window, None)? };
        let made = (|| {
            let device = Ctx::physical_device().handel;
            let family = Ctx::get().queues.graphics().index;
            let can_present =
                unsafe { surface_fn.get_physical_device_surface_support(device, family, surface)? };
            if !can_present {
                return Err(Error::message("the graphics queue family cannot present to this surface"));
            }
            let formats = unsafe { surface_fn.get_physical_device_surface_formats(device, surface)? };
            let modes = unsafe { surface_fn.get_physical_device_surface_present_modes(device, surface)? };
            let capabilities =
                unsafe { surface_fn.get_physical_device_surface_capabilities(device, surface)? };
            let usage = U::VK;
            if !capabilities.supported_usage_flags.contains(usage) {
                return Err(Error::message(format!(
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
            let usable = |f: &&vk::SurfaceFormatKHR| {
                let properties =
                    unsafe { instance.get_physical_device_format_properties(device, f.format) };
                properties.optimal_tiling_features.contains(needed)
            };
            // No preference of the surface (none, or one `UNDEFINED`): any of `preferred`.
            let format = match &formats[..] {
                [] => preferred.iter().find(usable),
                [only] if only.format == vk::Format::UNDEFINED => preferred.iter().find(usable),
                _ => (preferred.iter().filter(|p| formats.contains(p)))
                    .chain(&formats)
                    .find(usable),
            };
            let format = *format.ok_or_else(|| {
                Error::message(format!("the surface has no format whose images support {usage:?}"))
            })?;
            // FIFO is always supported.
            let present_modes = [
                (vk::PresentModeKHR::FIFO, "FIFO (vsync)"),
                (vk::PresentModeKHR::FIFO_RELAXED, "FIFO relaxed"),
                (vk::PresentModeKHR::MAILBOX, "Mailbox"),
                (vk::PresentModeKHR::IMMEDIATE, "Immediate"),
            ];
            let present_modes = (present_modes.into_iter())
                .filter(|(mode, _)| *mode == vk::PresentModeKHR::FIFO || modes.contains(mode))
                .collect();
            Ok(Self {
                surface,
                format,
                present_modes,
                present_mode: 0,
                swapchain: None,
                old: Vec::new(),
                recreate: false,
                free: Vec::new(),
                waited: VecDeque::new(),
                fences: Vec::new(),
                current: None,
                _usage: PhantomData,
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

    /// The extent of the swapchain's images, zero before the first acquire.
    pub fn size(&self) -> UVec2 {
        self.swapchain.as_ref().map_or(UVec2::ZERO, |s| s.extent)
    }

    pub fn num_images(&self) -> usize {
        self.swapchain.as_ref().map_or(0, |s| s.images.len())
    }

    /// The present modes the surface supports, FIFO first, with their names.
    pub fn present_modes(&self) -> &[(vk::PresentModeKHR, &'static str)] {
        &self.present_modes
    }

    pub fn present_mode(&self) -> usize {
        self.present_mode
    }

    /// Presents with mode `index` of `present_modes` from the next acquire on.
    pub fn set_present_mode(&mut self, index: usize) {
        assert!(index < self.present_modes.len(), "present mode {index} of {}", self.present_modes.len());
        self.present_mode = index;
    }

    pub fn handle(&self) -> vk::SurfaceKHR {
        self.surface
    }

    /// Makes an image of a window of `size` current, and returns whether one is (`false` while
    /// the window has no extent). An image a submission presented is presented first; one no
    /// submission used stays current, at its size, so a resize takes effect a frame later.
    #[validation_trace]
    pub fn acquire(&mut self, size: UVec2) -> Result<bool> {
        let device = Ctx::device();
        match self.current {
            Some(current) if current.presented.is_none() => {
                // Only a failed recording can have used it (one that uses it must present
                // it), and that one's barriers were dropped: its contents are still undefined.
                let swapchain = self.swapchain.as_ref().expect("an image is current");
                let state = ImageState { acquired: true, ..ImageState::UNDEFINED };
                swapchain.images[current.index as usize].state.store(state.pack(), Ordering::Relaxed);
                return Ok(size.x > 0 && size.y > 0);
            }
            Some(_) => self.present()?,
            None => {}
        }
        let free = &mut self.free;
        self.waited.retain(|&(semaphore, t, v)| {
            let done = completed(t) >= v || poll(t) >= v;
            if done {
                free.push(semaphore);
            }
            !done
        });
        let signalled = |fence: &vk::Fence| unsafe { device.get_fence_status(*fence) }.unwrap_or(false);
        if let Some(swapchain) = &mut self.swapchain {
            for fence in swapchain.fences.extract_if(.., |fence| signalled(fence)) {
                self.fences.push(fence);
            }
        }
        for (swapchain, _) in self.old.extract_if(.., |(swapchain, stamp)| {
            let presented = match gate::PresentFences::on() {
                true => swapchain.fences.iter().all(signalled),
                false => swapchain.presenter.is_none_or(|(t, v)| poll(t) > v),
            };
            presented && stamp.is_done()
        }) {
            swapchain.destroy();
        }
        if size.x == 0 || size.y == 0 {
            return Ok(false);
        }
        for _ in 0..3 {
            let physical_device = Ctx::physical_device().handel;
            let caps = unsafe {
                surface_fn()?.get_physical_device_surface_capabilities(physical_device, self.surface)?
            };
            let (min, max) = (caps.min_image_extent, caps.max_image_extent);
            if max.width == 0 || max.height == 0 {
                return Ok(false);
            }
            let extent = vk::Extent2D {
                width: size.x.clamp(min.width, max.width.max(min.width)),
                height: size.y.clamp(min.height, max.height.max(min.height)),
            };
            let stale = self.swapchain.as_ref().is_none_or(|s| {
                s.extent != UVec2::new(extent.width, extent.height) || s.present_mode != self.present_mode
            });
            if stale || self.recreate {
                let _span = tracing::info_span!("recreate swapchain").entered();
                let mode = self.present_modes[self.present_mode].0;
                let usage = U::VK;
                let mut info = vk::SwapchainCreateInfoKHR::default()
                    .surface(self.surface)
                    .min_image_count(caps.min_image_count)
                    .image_format(self.format.format)
                    .image_color_space(self.format.color_space)
                    .image_extent(extent)
                    .image_array_layers(1)
                    .image_usage(usage)
                    .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .pre_transform(caps.current_transform)
                    .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                    .present_mode(mode)
                    .clipped(true)
                    .old_swapchain(self.swapchain.as_ref().map_or_else(Default::default, |s| s.handle));
                let modes = [mode];
                let mut modes_info = vk::SwapchainPresentModesCreateInfoEXT::default().present_modes(&modes);
                if gate::PresentFences::on() {
                    info = info.push_next(&mut modes_info);
                }
                let handle = unsafe { Functions::swapchain().create_swapchain(&info, None)? };
                tracing::info!("made a swapchain of {}x{} images, presenting {mode:?}", extent.width, extent.height);
                let mut swapchain = Swapchain {
                    handle,
                    extent: UVec2::new(extent.width, extent.height),
                    present_mode: self.present_mode,
                    images: SmallVec::new(),
                    fences: Vec::new(),
                    presenter: None,
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
                        let state = ImageState::UNDEFINED.atomic();
                        let handle = ImageHandle::NULL;
                        swapchain.images.push(SwapchainImage { image, view, handle, state, present });
                    }
                    // One handle per image, with every descriptor its usage allows.
                    let sampled = usage.contains(vk::ImageUsageFlags::SAMPLED);
                    let storage = usage.contains(vk::ImageUsageFlags::STORAGE);
                    let (width, new): (u32, fn(u32) -> ImageHandle) = match (sampled, storage) {
                        (true, true) => (Combined::WIDTH, |slot| ImageHandle::<Combined>::new(slot).erase()),
                        (true, false) => (Sampled::WIDTH, |slot| ImageHandle::<Sampled>::new(slot).erase()),
                        (false, true) => (Storage::WIDTH, |slot| ImageHandle::<Storage>::new(slot).erase()),
                        (false, false) => return Ok(()),
                    };
                    for image in &mut swapchain.images {
                        image.handle = new(bindless::allocate(width)?);
                    }
                    let format = self.format.format;
                    let images = swapchain.images.iter().map(|image| bindless::ShaderImage {
                        image: image.image,
                        format,
                        aspects: vk::ImageAspectFlags::COLOR,
                        view_type: vk::ImageViewType::TYPE_2D,
                        mips: (0..1).into(),
                        usage,
                    });
                    bindless::bind_raw(images, swapchain.images.iter().map(|image| image.handle), false)?;
                    Ok(())
                })();
                if let Err(err) = made {
                    swapchain.destroy();
                    return Err(err);
                }
                if let Some(old) = self.swapchain.replace(swapchain) {
                    self.old.push((old, latest_stamp()));
                }
                self.recreate = false;
            }
            let semaphore = match self.free.pop() {
                Some(semaphore) => semaphore,
                None => unsafe { device.create_semaphore(&Default::default(), None)? },
            };
            let swapchain = self.swapchain.as_ref().expect("made above");
            let acquired = unsafe {
                let _span = tracing::info_span!("wait for swapchain image").entered();
                let fence = vk::Fence::null();
                Functions::swapchain().acquire_next_image(swapchain.handle, u64::MAX, semaphore, fence)
            };
            match acquired {
                Ok((index, suboptimal)) => {
                    self.recreate |= suboptimal;
                    // Its contents are discarded, so its first access may transition from anywhere.
                    let state = ImageState { acquired: true, ..ImageState::UNDEFINED };
                    swapchain.images[index as usize].state.store(state.pack(), Ordering::Relaxed);
                    let acquire = semaphore;
                    self.current = Some(Current { index, acquire, presented: None });
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
        Err(Error::message("the swapchain was still out of date after three recreations"))
    }

    /// The view of the current image, for the submission that draws into it and presents it.
    /// Panics before the first [`Self::acquire`].
    pub fn current_image(&self) -> ImageView<'_, format::Dynamic, U> {
        let current = self.current.expect("no image is current: `acquire` one first");
        let swapchain = self.swapchain.as_ref().expect("an image is current");
        let i = &swapchain.images[current.index as usize];
        ImageView {
            image: i.image,
            view: i.view,
            mip_range: (0..1).into(),
            extent: swapchain.extent,
            handle: i.handle,
            states: std::slice::from_ref(&i.state),
            format: self.format(),
            usage: U::default(),
            dim: D2,
        }
    }

    /// Presents the current image once a submission recorded `cmd.present(&surface)` of it;
    /// until then it stays current.
    #[validation_trace]
    pub fn present(&mut self) -> Result<()> {
        let Some(Current { index, acquire, presented: Some((t, v)) }) = self.current else {
            return Ok(());
        };
        self.current = None;
        self.waited.push_back((acquire, t, v));
        let swapchain = self.swapchain.as_ref().expect("an image was acquired");
        let (handle, present) = (swapchain.handle, swapchain.images[index as usize].present);

        let device = Ctx::device();
        let (swapchains, indices, waits) = ([handle], [index], [present]);
        let mut info = vk::PresentInfoKHR::default()
            .swapchains(&swapchains)
            .image_indices(&indices)
            .wait_semaphores(&waits);
        let fence = match gate::PresentFences::on() {
            false => None,
            true => Some([match self.fences.pop() {
                Some(fence) => {
                    unsafe { device.reset_fences(&[fence])? };
                    fence
                }
                None => unsafe { device.create_fence(&Default::default(), None)? },
            }]),
        };
        let swapchain = self.swapchain.as_mut().expect("an image was acquired");
        swapchain.presenter = Some((t, v));
        let mut fence_info = vk::SwapchainPresentFenceInfoEXT::default();
        if let Some(fence) = &fence {
            fence_info = fence_info.fences(fence);
            info = info.push_next(&mut fence_info);
            swapchain.fences.push(fence[0]);
        }
        let _span = tracing::info_span!("present").entered();
        let queue = lock(&Ctx::get().queues.graphics().queues[0]);
        let presented = unsafe { Functions::swapchain().queue_present(*queue, &info) };
        match presented {
            Ok(suboptimal) => self.recreate |= suboptimal,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.recreate = true,
            Err(err) => return Err(err.into()),
        }
        Ok(())
    }
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
                let waited =
                    submitted.and_then(|()| unsafe { device.wait_for_fences(&[fence], true, u64::MAX) });
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
        let fences: Vec<vk::Fence> = swapchains.iter().flat_map(|s| s.fences.iter().copied()).collect();
        let device = Ctx::device();
        if !fences.is_empty() && unsafe { device.wait_for_fences(&fences, true, 1_000_000_000) }.is_err() {
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

//! Presenting to a window: `PresentSurface` owns a surface, its swapchain and the semaphores between them, recreates the swapchain when the window's size, the present mode or the surface asks for it, and hands out one acquired image at a time, whose waits and signals follow from what submissions do with it
use std::{
    collections::VecDeque,
    fmt,
    marker::PhantomData,
    sync::{Arc, Mutex},
};

use crate::vk;
use glam::UVec2;
use lava_macros::validation_trace;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use smallvec::SmallVec;

use crate::{
    command_buffer::image_state::set_acquired,
    error::{Error, Result},
    features::{On, gate},
    image::{
        D2, format,
        slice::ImageView,
        usage::{ColorAttachmentTransferDst, StaticUsage},
    },
    lifetime,
    queue::{
        lock,
        timeline::{completed, poll},
    },
    state::{Ctx, Functions},
};

pub(crate) mod swapchain;
pub(crate) mod table;

use swapchain::{Params, Swapchain, select_extent, select_surface_format, supported_present_modes};

/// A window's surface and its swapchain of images of usage `U` (color attachment and transfer
/// destination by default), each bound in every bindless set `U` allows. A frame acquires an
/// image ([`Self::acquire`], which recreates the swapchain whenever it has to), records into
/// [`Self::image`] in a submission that ends with `cmd.present(view)`, and presents it
/// ([`Self::present`]). The image's first access in a submission waits for the acquire, and the
/// submission that presents it signals what the present waits for: there are no semaphores to
/// pass around. Dropping it destroys everything once no submission or present uses it.
pub struct PresentSurface<U: StaticUsage = ColorAttachmentTransferDst> {
    surface: vk::SurfaceKHR,
    format: vk::SurfaceFormatKHR,
    /// The surface's supported modes of lava's list, FIFO first, with their names.
    present_modes: SmallVec<[(vk::PresentModeKHR, &'static str); 4]>,
    /// Index into `present_modes`.
    present_mode: usize,
    capabilities: vk::SurfaceCapabilitiesKHR,
    swapchain: Option<Swapchain>,
    /// The surface or a present asked for a new swapchain.
    recreate: bool,
    /// Acquire semaphores nothing uses.
    free: Vec<vk::Semaphore>,
    /// Acquire semaphores a submission waited for (on its timeline, at its value): free once
    /// it finished.
    waited: VecDeque<(vk::Semaphore, usize, u64)>,
    outstanding: Arc<Mutex<Outstanding>>,
    _usage: PhantomData<fn() -> U>,
}

/// The image handed out last, if it wasn't presented.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
enum Outstanding {
    #[default]
    None,
    Acquired(u32),
    /// Dropped without a present.
    Abandoned(u32),
}

/// An image [`PresentSurface::acquire`] handed out: present it with [`PresentSurface::present`].
/// Dropping it instead makes the next acquire recreate the swapchain.
#[must_use]
pub struct AcquiredImage {
    index: u32,
    outstanding: Arc<Mutex<Outstanding>>,
    presented: bool,
}

impl AcquiredImage {
    /// Its index in the swapchain.
    pub fn index(&self) -> u32 {
        self.index
    }
}

impl fmt::Debug for AcquiredImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AcquiredImage").field("index", &self.index).finish()
    }
}

impl Drop for AcquiredImage {
    fn drop(&mut self) {
        if !self.presented {
            *lock(&self.outstanding) = Outstanding::Abandoned(self.index);
        }
    }
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
    /// A surface for `window` whose swapchains have the first of `preferred` the surface has
    /// whose format supports `U`, else any such. One per window; lava must have been
    /// initialised with a display. The swapchain is made by the first [`Self::acquire`].
    ///
    /// # Safety
    /// `display` and `window` must be valid handles, and the window must outlive the surface:
    /// it is destroyed once the `PresentSurface` is dropped and no submission or present uses
    /// it (`lava::wait_idle` finishes them).
    #[validation_trace]
    pub unsafe fn new(
        display: &RawDisplayHandle,
        window: &RawWindowHandle,
        preferred: &[vk::SurfaceFormatKHR],
    ) -> Result<Self> {
        let surface_fn = surface_fn()?;
        let surface = unsafe {
            ash_window::create_surface(
                Functions::entry(),
                Functions::instance(),
                *display,
                *window,
                None,
            )?
        };
        let made = (|| {
            let device = Ctx::physical_device().handel;
            let family = Ctx::get().queues.graphics().index;
            let can_present = unsafe {
                surface_fn.get_physical_device_surface_support(device, family, surface)?
            };
            if !can_present {
                return Err(Error::message(
                    "the graphics queue family cannot present to this surface",
                ));
            }
            let formats =
                unsafe { surface_fn.get_physical_device_surface_formats(device, surface)? };
            let modes =
                unsafe { surface_fn.get_physical_device_surface_present_modes(device, surface)? };
            let capabilities = unsafe {
                surface_fn.get_physical_device_surface_capabilities(device, surface)?
            };
            let usage = U::VK;
            if !capabilities.supported_usage_flags.contains(usage) {
                return Err(Error::message(format!(
                    "the surface supports the image usages {:?}, swapchain images need {usage:?}",
                    capabilities.supported_usage_flags
                )));
            }
            let instance = Functions::instance();
            let usable = |format: vk::Format| {
                let properties =
                    unsafe { instance.get_physical_device_format_properties(device, format) };
                (properties.optimal_tiling_features).contains(swapchain::format_features(usage))
            };
            let format = select_surface_format(&formats, preferred, usable).ok_or_else(|| {
                Error::message(format!(
                    "the surface has no format whose images support {usage:?}"
                ))
            })?;
            Ok(Self {
                surface,
                format,
                present_modes: supported_present_modes(&modes),
                present_mode: 0,
                capabilities,
                swapchain: None,
                recreate: false,
                free: Vec::new(),
                waited: VecDeque::new(),
                outstanding: Arc::default(),
                _usage: PhantomData,
            })
        })();
        if made.is_err() {
            unsafe { surface_fn.destroy_surface(surface, None) };
        }
        made
    }

    /// The format of its images.
    pub fn format(&self) -> format::Dynamic {
        format::Dynamic(self.format.format)
    }

    /// The extent of the swapchain's images, zero before the first acquire.
    pub fn size(&self) -> UVec2 {
        self.swapchain.as_ref().map_or(UVec2::ZERO, |s| s.extent)
    }

    /// How many images the swapchain has (0 before the first acquire); it can change when the
    /// swapchain is recreated, e.g. with the present mode.
    pub fn num_images(&self) -> usize {
        self.swapchain.as_ref().map_or(0, |s| s.images.len())
    }

    /// The present modes the surface supports, FIFO first, then by falling latency, with
    /// their names.
    pub fn present_modes(&self) -> &[(vk::PresentModeKHR, &'static str)] {
        &self.present_modes
    }

    /// The index into `present_modes` of the mode it presents with from the next acquire on.
    pub fn present_mode(&self) -> usize {
        self.present_mode
    }

    /// Presents with mode `index` of `present_modes` from the next acquire on, which recreates
    /// the swapchain. Panics if there is no such mode.
    pub fn set_present_mode(&mut self, index: usize) {
        assert!(
            index < self.present_modes.len(),
            "present mode {index} of {} supported ones",
            self.present_modes.len()
        );
        self.present_mode = index;
    }

    pub fn handle(&self) -> vk::SurfaceKHR {
        self.surface
    }

    /// The next image, for a window of `size`; `None` while the window has no extent (zero or
    /// minimised). Recreates the swapchain first when the size or the present mode changed, a
    /// present asked for it, or the swapchain is out of date (up to three times). Panics while
    /// an image acquired before is neither presented nor dropped.
    #[validation_trace]
    pub fn acquire(&mut self, size: UVec2) -> Result<Option<AcquiredImage>> {
        let outstanding = *lock(&self.outstanding);
        match outstanding {
            Outstanding::Acquired(_) => panic!(
                "an image acquired before is still outstanding: present or drop it first"
            ),
            Outstanding::Abandoned(index) => {
                self.recover(index)?;
                *lock(&self.outstanding) = Outstanding::None;
            }
            Outstanding::None => {}
        }
        self.free_waited();
        if size.x == 0 || size.y == 0 {
            return Ok(None);
        }
        for _ in 0..3 {
            self.capabilities = unsafe {
                surface_fn()?.get_physical_device_surface_capabilities(
                    Ctx::physical_device().handel,
                    self.surface,
                )?
            };
            let max = self.capabilities.max_image_extent;
            if max.width == 0 || max.height == 0 {
                return Ok(None);
            }
            let extent = select_extent(Some(size), &self.capabilities);
            let stale = self.swapchain.as_ref().is_none_or(|s| {
                s.extent != UVec2::new(extent.width, extent.height)
                    || s.present_mode != self.present_mode
            });
            if stale || self.recreate {
                self.recreate_swapchain(extent)?;
            }
            let semaphore = match self.free.pop() {
                Some(semaphore) => semaphore,
                None => unsafe { Ctx::device().create_semaphore(&Default::default(), None)? },
            };
            let swapchain = self.swapchain.as_ref().expect("made above");
            let acquired = unsafe {
                let _span = tracing::info_span!("wait for swapchain image").entered();
                (Functions::swapchain()).acquire_next_image(
                    swapchain.handle,
                    u64::MAX,
                    semaphore,
                    vk::Fence::null(),
                )
            };
            match acquired {
                Ok((index, suboptimal)) => {
                    self.recreate |= suboptimal;
                    let image = &swapchain.images[index as usize];
                    table::insert(image.image, semaphore, image.present);
                    set_acquired(&image.state, true);
                    *lock(&self.outstanding) = Outstanding::Acquired(index);
                    return Ok(Some(AcquiredImage {
                        index,
                        outstanding: self.outstanding.clone(),
                        presented: false,
                    }));
                }
                // A failed acquire leaves the semaphore alone.
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
        Err(Error::message(
            "the swapchain was still out of date after three recreations",
        ))
    }

    /// The view of `image`, for the submission that draws into it and presents it.
    pub fn image(&self, image: &AcquiredImage) -> ImageView<'_, format::Dynamic, U> {
        let swapchain = self.swapchain.as_ref().expect("an image was acquired");
        let i = &swapchain.images[image.index as usize];
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

    /// Presents `image` on the queue of the submission that presented it (`cmd.present`),
    /// after it. Panics if no submission did. A suboptimal or out-of-date swapchain is
    /// recreated by the next acquire.
    #[validation_trace]
    pub fn present(&mut self, mut image: AcquiredImage) -> Result<()> {
        let swapchain = self.swapchain.as_mut().expect("an image was acquired");
        let index = image.index;
        let vk_image = swapchain.images[index as usize].image;
        let entry = table::get(vk_image).expect("an acquired image is in the table");
        let Some(presenter) = entry.presented else {
            panic!(
                "no submission presented the image: record `cmd.present(surface.image(&image))` \
                 and submit it first"
            );
        };
        table::take(vk_image);
        image.presented = true;
        drop(image);
        *lock(&self.outstanding) = Outstanding::None;
        // The submission that presented it waited for the acquire, at the latest.
        let (t, v) = entry.waited.expect("a presenting submission waited for the acquire");
        self.waited.push_back((entry.acquire, t, v));
        swapchain.last_presenter = Some(presenter.timeline);

        let (swapchains, indices, waits) = ([swapchain.handle], [index], [entry.present]);
        let mut info = vk::PresentInfoKHR::default()
            .swapchains(&swapchains)
            .image_indices(&indices)
            .wait_semaphores(&waits);
        // Signalled once the present is done with what it uses, which may be destroyed then.
        let fence = match gate::PresentFences::on() {
            true => Some([lifetime::present_fence()?]),
            false => None,
        };
        let mut fence_info = vk::SwapchainPresentFenceInfoEXT::default();
        if let Some(fence) = &fence {
            fence_info = fence_info.fences(fence);
            info = info.push_next(&mut fence_info);
        }
        let _span = tracing::info_span!("present").entered();
        let presented = {
            let queue = lock(&presenter.queue.queue);
            let presented = unsafe { Functions::swapchain().queue_present(*queue, &info) };
            // While the queue is locked, so `wait_idle` sees every present made before it.
            if let Some([fence]) = fence {
                lifetime::presented(fence);
            }
            presented
        };
        match presented {
            Ok(suboptimal) => self.recreate |= suboptimal,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.recreate = true,
            Err(err) => return Err(err.into()),
        }
        Ok(())
    }

    /// Replaces the swapchain with one of `extent` in the current present mode; the old one is
    /// destroyed once no submission or present uses it.
    fn recreate_swapchain(&mut self, extent: vk::Extent2D) -> Result<()> {
        let _span = tracing::info_span!("recreate swapchain").entered();
        let params = Params {
            surface: self.surface,
            capabilities: self.capabilities,
            format: self.format,
            usage: U::VK,
            extent,
            present_mode: (self.present_mode, self.present_modes[self.present_mode].0),
        };
        let old = self.swapchain.as_ref().map(|s| s.handle);
        let swapchain = Swapchain::new(&params, old)?;
        if let Some(old) = self.swapchain.replace(swapchain) {
            old.retire();
        }
        self.recreate = false;
        Ok(())
    }

    /// Takes back image `index`, which was dropped instead of presented: its semaphores are
    /// made unsignalled again, and the next acquire recreates the swapchain.
    fn recover(&mut self, index: u32) -> Result<()> {
        self.recreate = true;
        let Some(swapchain) = &self.swapchain else {
            return Ok(());
        };
        let image = &swapchain.images[index as usize];
        set_acquired(&image.state, false);
        let Some(entry) = table::take(image.image) else {
            return Ok(());
        };
        let mut unwaited = SmallVec::<[vk::Semaphore; 2]>::new();
        match entry.waited {
            Some((t, v)) => self.waited.push_back((entry.acquire, t, v)),
            None => unwaited.push(entry.acquire),
        }
        // Signalled by its submission, and no present will wait for it.
        if entry.presented.is_some() {
            unwaited.push(entry.present);
        }
        if !unwaited.is_empty() {
            consume(&unwaited)?;
        }
        if entry.waited.is_none() {
            self.free.push(entry.acquire);
        }
        Ok(())
    }

    /// Frees the acquire semaphores whose submissions finished.
    fn free_waited(&mut self) {
        let free = &mut self.free;
        self.waited.retain(|&(semaphore, t, v)| {
            let done = completed(t) >= v || poll(t) >= v;
            if done {
                free.push(semaphore);
            }
            !done
        });
    }
}

/// Waits on the GPU for binary `semaphores` that are signalled or will be, with nothing waiting
/// for them, and on the host until that wait is done: they are unsignalled again.
fn consume(semaphores: &[vk::Semaphore]) -> Result<()> {
    let device = Ctx::device();
    let waits: SmallVec<[_; 2]> = (semaphores.iter())
        .map(|&semaphore| {
            vk::SemaphoreSubmitInfo::default()
                .semaphore(semaphore)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        })
        .collect();
    let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? };
    let submitted = {
        let queue = lock(&Ctx::get().queues.graphics().queues[0].queue);
        let info = vk::SubmitInfo2::default().wait_semaphore_infos(&waits);
        unsafe { device.queue_submit2(*queue, &[info], fence) }
    };
    let waited = submitted.and_then(|()| unsafe { device.wait_for_fences(&[fence], true, u64::MAX) });
    unsafe { device.destroy_fence(fence, None) };
    Ok(waited?)
}

impl<U: StaticUsage> Drop for PresentSurface<U> {
    /// Destroys the swapchain, then the surface, with their semaphores, once no submission or
    /// present uses them.
    fn drop(&mut self) {
        let outstanding = *lock(&self.outstanding);
        if let Outstanding::Abandoned(index) = outstanding
            && let Err(err) = self.recover(index)
        {
            tracing::error!(%err, "failed to take back a dropped swapchain image");
        }
        let swapchain = self.swapchain.take();
        let presenter = swapchain.as_ref().and_then(|s| s.last_presenter);
        let mut semaphores = std::mem::take(&mut self.free);
        semaphores.extend(self.waited.drain(..).map(|(semaphore, ..)| semaphore));
        let surface = self.surface;
        lifetime::retire_presented(presenter, move || {
            if let Some(swapchain) = swapchain {
                swapchain.destroy();
            }
            for semaphore in semaphores {
                unsafe { Ctx::device().destroy_semaphore(semaphore, None) };
            }
            if let Some(surface_fn) = Functions::surface() {
                unsafe { surface_fn.destroy_surface(surface, None) };
            }
        });
    }
}

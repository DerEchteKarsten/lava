//! Window surface: an owned `SurfaceKHR` with its formats, present modes, and capabilities, destroyed after the last swapchain made for it
use std::sync::Arc;

use crate::vk;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use crate::{
    error::{Error, Result},
    state::{Ctx, Functions},
};

#[derive(Debug)]
pub struct Surface {
    handle: Arc<SurfaceHandle>,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
    capabilities: vk::SurfaceCapabilitiesKHR,
}

/// The `SurfaceKHR`, shared by the `Surface` and every swapchain made for it, and destroyed
/// once the last of them is (`epoch`): a surface may not go before its swapchains.
#[derive(Debug)]
pub(crate) struct SurfaceHandle(vk::SurfaceKHR);

impl Surface {
    /// Creates a surface for `window`. One surface per window; lava must have been
    /// initialised with a display handle.
    ///
    /// # Safety
    /// `display` and `window` must be valid handles, and the window must outlive the surface:
    /// it is destroyed once the `Surface` and every swapchain made for it are dropped and the
    /// frames and presents that may use them are done (`lava::wait_idle` finishes them).
    pub unsafe fn new(display: &RawDisplayHandle, window: &RawWindowHandle) -> Result<Self> {
        let surface_fn = surface_fn()?;
        let handle = unsafe {
            ash_window::create_surface(
                Functions::entry(),
                Functions::instance(),
                *display,
                *window,
                None,
            )?
        };
        // From here on `Drop` destroys the handle on every error path.
        let mut surface = Self {
            handle: Arc::new(SurfaceHandle(handle)),
            formats: Vec::new(),
            present_modes: Vec::new(),
            capabilities: vk::SurfaceCapabilitiesKHR::default(),
        };

        let physical_device = Ctx::physical_device().handel;
        let can_present = unsafe {
            surface_fn.get_physical_device_surface_support(
                physical_device,
                Ctx::gfx_queue_index(),
                handle,
            )?
        };
        if !can_present {
            return Err(Error::message(
                "the graphics queue family cannot present to this surface",
            ));
        }

        unsafe {
            surface.formats =
                surface_fn.get_physical_device_surface_formats(physical_device, handle)?;
            surface.present_modes =
                surface_fn.get_physical_device_surface_present_modes(physical_device, handle)?;
        }
        surface.refresh_capabilities()?;
        Ok(surface)
    }

    pub fn handle(&self) -> vk::SurfaceKHR {
        self.handle.0
    }

    pub(crate) fn shared_handle(&self) -> Arc<SurfaceHandle> {
        self.handle.clone()
    }

    /// The formats and colour spaces the surface supports.
    pub fn formats(&self) -> &[vk::SurfaceFormatKHR] {
        &self.formats
    }

    /// The present modes the surface supports.
    pub fn present_modes(&self) -> &[vk::PresentModeKHR] {
        &self.present_modes
    }

    /// Its capabilities as of the last `refresh_capabilities`.
    pub fn capabilities(&self) -> &vk::SurfaceCapabilitiesKHR {
        &self.capabilities
    }

    /// Whether the surface has no extent to create a swapchain for right now (a minimised
    /// window), as of the last `refresh_capabilities`.
    pub fn is_minimised(&self) -> bool {
        let max = self.capabilities.max_image_extent;
        max.width == 0 || max.height == 0
    }

    /// Re-queries the capabilities, which change when the window is resized.
    pub fn refresh_capabilities(&mut self) -> Result<()> {
        self.capabilities = unsafe {
            surface_fn()?.get_physical_device_surface_capabilities(
                Ctx::physical_device().handel,
                self.handle(),
            )?
        };
        Ok(())
    }
}

fn surface_fn() -> Result<&'static ash::khr::surface::Instance> {
    Functions::surface().ok_or_else(|| {
        Error::message("lava was initialised without a display handle; surfaces are unavailable")
    })
}

/// Destroyed after the swapchains made for it, which hold the handle too, once the presents that
/// may use it are done (`epoch`).
impl Drop for SurfaceHandle {
    fn drop(&mut self) {
        let handle = self.0;
        crate::epoch::retire_after_present(move || {
            if let Some(surface_fn) = Functions::surface() {
                unsafe { surface_fn.destroy_surface(handle, None) };
            }
        });
    }
}

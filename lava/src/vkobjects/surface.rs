//! Window surface: an owned `SurfaceKHR` with its formats, present modes, and capabilities
use ash::vk;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use std::mem::ManuallyDrop;

use crate::{
    error::{Error, Result},
    keepalive::KeepAlive,
    state::{Ctx, Functions},
};

/// Dropping it destroys it once the swapchains made for it are destroyed: each counts it
/// (`keepalive`).
#[derive(Debug)]
pub struct Surface {
    pub handle: vk::SurfaceKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    owner: ManuallyDrop<KeepAlive>,
}

struct RawSurface(vk::SurfaceKHR);
impl Drop for RawSurface {
    fn drop(&mut self) {
        if let Some(surface_fn) = Functions::surface() {
            unsafe { surface_fn.destroy_surface(self.0, None) };
        }
    }
}

impl Surface {
    /// Creates a surface for `window`. One surface per window; lava must have been
    /// initialised with a display handle.
    pub fn new(display: &RawDisplayHandle, window: &RawWindowHandle) -> Result<Self> {
        let surface_fn = surface_fn()?;
        let handle = unsafe {
            ash_window::SurfaceFactory::new(Functions::entry(), Functions::instance(), *display)?
                .create_surface(*window, None)?
        };
        // From here on `Drop` destroys the handle on every error path.
        let mut surface = Self {
            handle,
            formats: Vec::new(),
            present_modes: Vec::new(),
            capabilities: vk::SurfaceCapabilitiesKHR::default(),
            owner: ManuallyDrop::new(KeepAlive::new()),
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
                self.handle,
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

impl Surface {
    /// A count of the surface, which keeps it alive.
    pub(crate) fn keep_alive(&self) -> KeepAlive {
        KeepAlive::clone(&self.owner)
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        let owner = unsafe { ManuallyDrop::take(&mut self.owner) };
        owner.release(RawSurface(self.handle));
    }
}

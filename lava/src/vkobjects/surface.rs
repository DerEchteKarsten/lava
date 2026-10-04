//! Window surface: an owned `SurfaceKHR` with its formats, present modes, and capabilities
use ash::vk;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use crate::{
    error::{Error, Result},
    state::{Ctx, Functions},
};

#[derive(Debug)]
pub struct Surface {
    pub handle: vk::SurfaceKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
    pub capabilities: vk::SurfaceCapabilitiesKHR,
}

impl Surface {
    /// Creates a surface for `window`. One surface per window; lava must have been
    /// initialised with a display handle.
    pub fn new(display: &RawDisplayHandle, window: &RawWindowHandle) -> Result<Self> {
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
            handle,
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

impl Drop for Surface {
    fn drop(&mut self) {
        if let Some(surface_fn) = Functions::surface() {
            unsafe { surface_fn.destroy_surface(self.handle, None) };
        }
    }
}

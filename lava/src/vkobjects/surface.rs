//! Window surface handle with its supported formats, present modes, and capabilities
use ash::vk;

use crate::{error::Result, vkobjects::physical_device::PhysicalDevice};

#[derive(Debug)]
pub struct Surface {
    pub handle: vk::SurfaceKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
    pub capabilities: vk::SurfaceCapabilitiesKHR,
}

impl Surface {
    pub(crate) fn new(
        surface: vk::SurfaceKHR,
        physical_device: &PhysicalDevice,
        surface_fn: &ash::khr::surface::Instance,
    ) -> Result<Self> {
        unsafe {
            Ok(Self {
                handle: surface,
                formats: surface_fn
                    .get_physical_device_surface_formats(physical_device.handel, surface)?,
                present_modes: surface_fn
                    .get_physical_device_surface_present_modes(physical_device.handel, surface)?,
                capabilities: surface_fn
                    .get_physical_device_surface_capabilities(physical_device.handel, surface)?,
            })
        }
    }
}

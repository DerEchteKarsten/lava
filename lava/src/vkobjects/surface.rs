use ash::vk;

use crate::{state::{Ctx, Functions}, vkobjects::physical_device::PhysicalDevice};

pub struct Surface {
    pub handle: vk::SurfaceKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
    pub capabilities: vk::SurfaceCapabilitiesKHR,
}

impl Surface {
    pub fn new(surface: vk::SurfaceKHR) -> Self {
        unsafe { Self {
                    handle: surface,
                    formats: Functions::surface().get_physical_device_surface_formats(
                        Ctx::physical_device().handel,
                        surface,
                    )?,
                    present_modes: Functions::surface().get_physical_device_surface_present_modes(
                        Ctx::physical_device().handel,
                        surface,
                    )?,
                    capabilities: Functions::surface().get_physical_device_surface_capabilities(
                        Ctx::physical_device().handel,
                        surface,
                    )?,
                } }
    }
}
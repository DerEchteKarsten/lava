use ash::vk;

use crate::vkobjects::physical_device::PhysicalDevice;

#[derive(Debug)]
pub struct Surface {
    pub handle: vk::SurfaceKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
    pub capabilities: vk::SurfaceCapabilitiesKHR,
}

impl Surface {
    pub fn new(
        surface: vk::SurfaceKHR,
        physical_device: &PhysicalDevice,
        surface_fn: &ash::khr::surface::Instance,
    ) -> Self {
        unsafe {
            Self {
                handle: surface,
                formats: surface_fn
                    .get_physical_device_surface_formats(physical_device.handel, surface)
                    .unwrap(),
                present_modes: surface_fn
                    .get_physical_device_surface_present_modes(physical_device.handel, surface)
                    .unwrap(),
                capabilities: surface_fn
                    .get_physical_device_surface_capabilities(physical_device.handel, surface)
                    .unwrap(),
            }
        }
    }
}

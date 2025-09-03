use std::ffi::CStr;

use anyhow::Result;
use ash::vk;

#[derive(Clone, Debug)]
pub struct QueueFamily {
    pub index: u32,
    pub handel: vk::QueueFamilyProperties,
    pub supports_present: bool,
}

impl QueueFamily {
    pub fn supports_compute(&self) -> bool {
        self.handel.queue_flags.contains(vk::QueueFlags::COMPUTE)
    }

    pub fn supports_graphics(&self) -> bool {
        self.handel.queue_flags.contains(vk::QueueFlags::GRAPHICS)
    }

    pub fn supports_transfer(&self) -> bool {
        self.handel.queue_flags.contains(vk::QueueFlags::TRANSFER)
    }

    pub fn supports_present(&self) -> bool {
        self.supports_present
    }

    pub fn has_queues(&self) -> bool {
        self.handel.queue_count > 0
    }

    pub fn supports_timestamp_queries(&self) -> bool {
        self.handel.timestamp_valid_bits > 0
    }
}

#[derive(Clone, Debug)]
pub struct PhysicalDevice {
    pub handel: vk::PhysicalDevice,
    pub name: String,
    pub device_type: vk::PhysicalDeviceType,
    pub limits: vk::PhysicalDeviceLimits,
    pub queue_families: Vec<QueueFamily>,
    pub supported_extensions: Vec<String>,
    pub supported_surface_formats: Vec<vk::SurfaceFormatKHR>,
    pub supported_present_modes: Vec<vk::PresentModeKHR>,
}

impl PhysicalDevice {
    pub fn new(
        instance: &ash::Instance,
        ash_surface: &ash::khr::surface::Instance,
        vk_surface: &vk::SurfaceKHR,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        let props = unsafe { instance.get_physical_device_properties(physical_device) };

        let name = unsafe {
            CStr::from_ptr(props.device_name.as_ptr())
                .to_str()?
                .to_owned()
        };

        let device_type = props.device_type;
        let limits = props.limits;

        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let queue_families = queue_family_properties
            .into_iter()
            .enumerate()
            .map(|(index, p)| {
                let present_support = unsafe {
                    ash_surface
                        .get_physical_device_surface_support(
                            physical_device,
                            index as _,
                            *vk_surface,
                        )
                        .unwrap()
                };

                QueueFamily {
                    index: index as _,
                    handel: p,
                    supports_present: present_support,
                }
            })
            .collect::<Vec<_>>();

        let extension_properties =
            unsafe { instance.enumerate_device_extension_properties(physical_device)? };
        let supported_extensions = extension_properties
            .into_iter()
            .map(|p| {
                let name = unsafe { CStr::from_ptr(p.extension_name.as_ptr()) };
                name.to_str().unwrap().to_owned()
            })
            .collect();

        let supported_surface_formats = unsafe {
            ash_surface.get_physical_device_surface_formats(physical_device, *vk_surface)?
        };

        let supported_present_modes = unsafe {
            ash_surface.get_physical_device_surface_present_modes(physical_device, *vk_surface)?
        };
        let mut ray_tracing_feature = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default();
        let mut acceleration_struct_feature =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
        let mut atomics = vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default()
            .shader_buffer_float32_atomics(true)
            .shader_buffer_float64_atomic_add(true)
            .shader_buffer_float32_atomic_add(true);
        let mut atomics2 = vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default()
            .shader_buffer_float32_atomics(true)
            .shader_buffer_float32_atomic_add(true);
        let features = vk::PhysicalDeviceFeatures::default().shader_int64(true);
        let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
            .runtime_descriptor_array(true)
            .buffer_device_address(true)
            .shader_buffer_int64_atomics(true);
        let mut mesh_shading = vk::PhysicalDeviceMeshShaderFeaturesEXT::default()
            .task_shader(true)
            .mesh_shader(true);

        let mut features13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .features(features)
            .push_next(&mut atomics)
            .push_next(&mut features12)
            .push_next(&mut ray_tracing_feature)
            .push_next(&mut acceleration_struct_feature)
            .push_next(&mut features13)
            .push_next(&mut atomics2)
            .push_next(&mut mesh_shading);
        unsafe { instance.get_physical_device_features2(physical_device, &mut features2) };

        Ok(Self {
            handel: physical_device,
            name,
            device_type,
            limits,
            queue_families,
            supported_extensions,
            supported_surface_formats,
            supported_present_modes,
        })
    }

    pub fn supports_extensions(&self, extensions: &[&str]) -> bool {
        let supported_extensions = self
            .supported_extensions
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        extensions.iter().all(|e| supported_extensions.contains(e))
    }
}

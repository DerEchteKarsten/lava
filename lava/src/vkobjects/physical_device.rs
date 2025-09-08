use std::ffi::CStr;

use anyhow::Result;
use ash::vk;

use crate::state::Features;
use crate::state::Functions;

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
    pub supported_features: Features,
    pub bindless_supported: bool,
}

impl PhysicalDevice {
    pub fn new(
        surface: &vk::SurfaceKHR,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        let props = unsafe { Functions::instance().get_physical_device_properties(physical_device) };

        let name = unsafe {
            CStr::from_ptr(props.device_name.as_ptr())
                .to_str()?
                .to_owned()
        };

        let device_type = props.device_type;
        let limits = props.limits;

        let queue_family_properties =
            unsafe { Functions::instance().get_physical_device_queue_family_properties(physical_device) };
        let queue_families = queue_family_properties
            .into_iter()
            .enumerate()
            .map(|(index, p)| {
                let present_support = unsafe {
                    Functions::surface()
                        .get_physical_device_surface_support(
                            physical_device,
                            index as _,
                            *surface,
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
            unsafe { Functions::instance().enumerate_device_extension_properties(physical_device)? };
        let supported_extensions = extension_properties
            .into_iter()
            .map(|p| {
                let name = unsafe { CStr::from_ptr(p.extension_name.as_ptr()) };
                name.to_str().unwrap().to_owned()
            })
            .collect();

        let supported_surface_formats = unsafe {
            Functions::surface().get_physical_device_surface_formats(physical_device, *surface)?
        };

        let supported_present_modes = unsafe {
            Functions::surface().get_physical_device_surface_present_modes(physical_device, *surface)?
        };
        let mut ray_tracing_feature = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default();
        let mut acceleration_struct_feature = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
        let mut features12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut mesh_shading = vk::PhysicalDeviceMeshShaderFeaturesEXT::default();

        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut features12)
            .push_next(&mut ray_tracing_feature)
            .push_next(&mut acceleration_struct_feature)
            .push_next(&mut mesh_shading);
        unsafe { Functions::instance().get_physical_device_features2(physical_device, &mut features2) };

        let features = Features {
            debug_utils: true,
            mesh: mesh_shading.mesh_shader == vk::TRUE,
            raytracing: ray_tracing_feature.ray_tracing_pipeline == vk::TRUE && acceleration_struct_feature.acceleration_structure == vk::TRUE,
        };

        Ok(Self {
            handel: physical_device,
            name,
            device_type,
            limits,
            queue_families,
            supported_extensions,
            supported_surface_formats,
            supported_present_modes,
            bindless_supported: features12.runtime_descriptor_array == vk::TRUE && features12.descriptor_binding_partially_bound == vk::TRUE && features12.descriptor_binding_variable_descriptor_count == vk::TRUE,
            supported_features: features,
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


    pub fn enumerate_physical_devices(surface: vk::SurfaceKHR) -> Result<Vec<PhysicalDevice>> {
        let physical_devices = unsafe { Functions::instance().enumerate_physical_devices()? };

        let mut physical_devices = physical_devices
            .into_iter()
            .map(|pd| PhysicalDevice::new(surface, pd))
            .collect::<Result<Vec<PhysicalDevice>>>()?;

        physical_devices.sort_by_key(|pd| match pd.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 0,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
            _ => 2,
        });
        Ok(physical_devices)
    }


    pub fn select_suitable_physical_device(devices: &[PhysicalDevice]) -> Result<(PhysicalDevice, QueueFamily, Option<QueueFamily>, Option<QueueFamily>)> {
        let mut graphics = None;
        let mut present = None;
        let mut transfer_queue = None;

        let device = devices
            .iter()
            .find(|device| {
                for family in device.queue_families.iter().filter(|f| f.has_queues()) {
                    if family.supports_graphics()
                        && family.supports_compute()
                        && family.supports_timestamp_queries()
                        && graphics.is_none()
                    {
                        graphics = Some(family.clone());
                    } else if family.supports_present() && present.is_none(){
                        present = Some(family.clone());
                    } else if family.supports_transfer() && transfer_queue.is_none() {
                        transfer_queue = Some(family.clone());
                    }

                    if graphics.is_some() && present.is_some() {
                        break;
                    }
                }

                graphics.is_some()
                    && present.is_some()
                    && !device.supported_surface_formats.is_empty()
                    && !device.supported_present_modes.is_empty()
                    && (device.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
                        || device.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU)
            })
            .ok_or_else(|| anyhow::anyhow!("Could not find a suitable device"))?;
        assert!(device.supports_extensions(&device.supported_features.extensions()), "device does not support required extensions");
        log::info!("Using device: {}", device.name);
        log::info!("Device type: {:?}", device.device_type);
        log::info!("Features are: {:?}", device.supported_features);
        Ok((
            device.clone(),
            graphics.unwrap(),
            present,
            transfer_queue,
        ))
    }
}

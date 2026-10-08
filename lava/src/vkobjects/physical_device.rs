//! Physical device selection and queue family capability discovery
use std::ffi::CStr;

use crate::error::{Error, Result};
use ash::vk;
use lava_macros::validation_trace;

use crate::state::Features;

impl PartialEq for QueueFamily {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

#[derive(Clone, Debug)]
pub struct QueueFamily {
    pub index: u32,
    pub num_queues: u32,
    pub handel: vk::QueueFamilyProperties,
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
    pub mem_properties: vk::PhysicalDeviceMemoryProperties,
    pub device_type: vk::PhysicalDeviceType,
    pub limits: vk::PhysicalDeviceLimits,
    pub queue_families: Vec<QueueFamily>,
    pub supported_extensions: Vec<String>,
    pub supported_features: Features,
    pub bindless_supported: bool,
    pub ray_tracing_pipeline_properties:
        Option<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'static>>,
    pub acceleration_structure_properties:
        Option<vk::PhysicalDeviceAccelerationStructurePropertiesKHR<'static>>,
}

impl PhysicalDevice {
    pub(crate) fn new(
        instance: &ash::Instance,
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
            .map(|(index, p)| QueueFamily {
                num_queues: p.queue_count,
                index: index as _,
                handel: p,
            })
            .collect::<Vec<_>>();

        let extension_properties =
            unsafe { instance.enumerate_device_extension_properties(physical_device)? };
        let supported_extensions = extension_properties
            .into_iter()
            .map(|p| {
                let name = unsafe { CStr::from_ptr(p.extension_name.as_ptr()) };
                Ok(name.to_str()?.to_owned())
            })
            .collect::<Result<Vec<String>>>()?;

        let mut ray_tracing_feature = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default();
        let mut acceleration_struct_feature =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
        let mut features12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut mesh_shading = vk::PhysicalDeviceMeshShaderFeaturesEXT::default();
        let mut rt_pipeline_properties =
            vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
        let mut acc_properties = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();

        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut features12)
            .push_next(&mut ray_tracing_feature)
            .push_next(&mut acceleration_struct_feature)
            .push_next(&mut mesh_shading);
        unsafe { instance.get_physical_device_features2(physical_device, &mut features2) };
        let pipeline_statistics = features2.features.pipeline_statistics_query == vk::TRUE;

        let mut properties2 = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut rt_pipeline_properties)
            .push_next(&mut acc_properties);
        unsafe { instance.get_physical_device_properties2(physical_device, &mut properties2) };

        let mem_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let rebar = has_rebar(&mem_properties);
        let has_extension = |name: &CStr| {
            supported_extensions
                .iter()
                .any(|e| e.as_bytes() == name.to_bytes())
        };
        let features = Features {
            rebar,
            present: true,
            debug_utils: true,
            device_debug_utils: supported_extensions.contains(
                &ash::ext::debug_utils::NAME
                    .to_str()
                    .expect("extension names are valid UTF-8")
                    .to_owned(),
            ),
            mesh: mesh_shading.mesh_shader == vk::TRUE,
            raytracing: ray_tracing_feature.ray_tracing_pipeline == vk::TRUE
                && acceleration_struct_feature.acceleration_structure == vk::TRUE,
            pipeline_statistics,
            mesh_queries: mesh_shading.mesh_shader_queries == vk::TRUE,
            memory_budget: has_extension(ash::ext::memory_budget::NAME),
        };

        Ok(Self {
            mem_properties,
            handel: physical_device,
            name,
            device_type,
            limits,
            queue_families,
            supported_extensions,

            acceleration_structure_properties: if features.raytracing {
                Some(acc_properties)
            } else {
                None
            },
            ray_tracing_pipeline_properties: if features.raytracing {
                Some(rt_pipeline_properties)
            } else {
                None
            },

            bindless_supported: features12.runtime_descriptor_array == vk::TRUE
                && features12.descriptor_binding_partially_bound == vk::TRUE
                && features12.descriptor_binding_variable_descriptor_count == vk::TRUE,
            supported_features: features,
        })
    }

    pub(crate) fn unsupports_extensions(&self, extensions: &[&CStr]) -> Vec<String> {
        extensions
            .iter()
            .map(|e| {
                e.to_str()
                    .expect("Vulkan extension names are valid UTF-8")
                    .to_owned()
            })
            .filter(|e| self.supported_extensions.iter().find(|i| *i == e).is_none())
            .collect::<Vec<String>>()
    }

    #[validation_trace]
    pub(crate) fn enumerate_physical_devices(
        instance: &ash::Instance,
    ) -> Result<Vec<PhysicalDevice>> {
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };

        let mut physical_devices = physical_devices
            .into_iter()
            .map(|pd| PhysicalDevice::new(instance, pd))
            .collect::<Result<Vec<PhysicalDevice>>>()?;

        physical_devices.sort_by_key(|pd| match pd.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 0,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
            _ => 2,
        });
        Ok(physical_devices)
    }

    /// Picks the graphics family (graphics + compute + timestamps) and, among the remaining
    /// families, a dedicated transfer family. `None` if the device has no usable graphics family.
    pub(crate) fn pick_queue_families(&self) -> Option<(QueueFamily, Option<QueueFamily>)> {
        let mut graphics = None;
        let mut transfer = None;
        for family in self.queue_families.iter().filter(|f| f.has_queues()) {
            if graphics.is_none()
                && family.supports_graphics()
                && family.supports_compute()
                && family.supports_timestamp_queries()
            {
                graphics = Some(family.clone());
            } else if transfer.is_none() && family.supports_transfer() {
                transfer = Some(family.clone());
            }
        }
        graphics.map(|graphics| (graphics, transfer))
    }

    /// Returns the first GPU with a usable graphics family, plus its graphics and transfer
    /// families. Presentation goes through the graphics family (checked per surface).
    #[validation_trace]
    pub(crate) fn select_suitable_physical_device(
        devices: &[PhysicalDevice],
        features: &mut Features,
    ) -> Result<(PhysicalDevice, QueueFamily, Option<QueueFamily>)> {
        let (device, (graphics, transfer_queue)) = devices
            .iter()
            .filter(|device| {
                device.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
                    || device.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
            })
            .find_map(|device| Some((device, device.pick_queue_families()?)))
            .ok_or_else(|| Error::message("Could not find a suitable device"))?;
        let mut required = device.supported_features.clone();
        required.present = features.present;
        let unsuported_ext = device.unsupports_extensions(&required.extensions());
        if !unsuported_ext.is_empty() {
            tracing::error!("Unsuported Extensions: {:#?}", unsuported_ext);
        }
        features.device_debug_utils =
            device.supported_features.device_debug_utils && features.debug_utils;
        features.mesh = device.supported_features.mesh;
        features.raytracing = device.supported_features.raytracing;
        features.rebar = device.supported_features.rebar;
        features.pipeline_statistics = device.supported_features.pipeline_statistics;
        features.mesh_queries = device.supported_features.mesh_queries;
        features.memory_budget = device.supported_features.memory_budget;
        Ok((device.clone(), graphics, transfer_queue))
    }
}

/// Whether the device exposes a large heap that is both device-local and host-visible, i.e.
/// GPU memory can be written directly from the CPU (resizable BAR or an integrated GPU).
pub(crate) fn has_rebar(mem_properties: &vk::PhysicalDeviceMemoryProperties) -> bool {
    mem_properties.memory_types[..mem_properties.memory_type_count as usize]
        .iter()
        .any(|e| {
            e.property_flags.contains(
                vk::MemoryPropertyFlags::DEVICE_LOCAL
                    | vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT,
            ) && mem_properties.memory_heaps[e.heap_index as usize].size > 500 * 1024 * 1024
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use vk::QueueFlags as Q;

    fn family(index: u32, flags: Q, num_queues: u32, timestamps: bool) -> QueueFamily {
        QueueFamily {
            index,
            num_queues,
            handel: vk::QueueFamilyProperties {
                queue_flags: flags,
                queue_count: num_queues,
                timestamp_valid_bits: if timestamps { 64 } else { 0 },
                ..Default::default()
            },
        }
    }

    fn gfx_family(index: u32, num_queues: u32) -> QueueFamily {
        family(
            index,
            Q::GRAPHICS | Q::COMPUTE | Q::TRANSFER,
            num_queues,
            true,
        )
    }

    fn device(
        device_type: vk::PhysicalDeviceType,
        queue_families: Vec<QueueFamily>,
    ) -> PhysicalDevice {
        PhysicalDevice {
            handel: vk::PhysicalDevice::null(),
            name: "test device".into(),
            mem_properties: Default::default(),
            device_type,
            limits: Default::default(),
            queue_families,
            supported_extensions: Vec::new(),
            supported_features: Features::default(),
            bindless_supported: true,
            ray_tracing_pipeline_properties: None,
            acceleration_structure_properties: None,
        }
    }

    const DISCRETE: vk::PhysicalDeviceType = vk::PhysicalDeviceType::DISCRETE_GPU;
    const INTEGRATED: vk::PhysicalDeviceType = vk::PhysicalDeviceType::INTEGRATED_GPU;
    const CPU: vk::PhysicalDeviceType = vk::PhysicalDeviceType::CPU;

    #[test]
    fn queue_family_predicates_read_the_flags() {
        let f = family(0, Q::GRAPHICS | Q::TRANSFER, 2, false);
        assert!(f.supports_graphics());
        assert!(f.supports_transfer());
        assert!(!f.supports_compute());
        assert!(f.has_queues());
        assert!(!f.supports_timestamp_queries());

        let empty = family(1, Q::COMPUTE, 0, true);
        assert!(empty.supports_compute());
        assert!(!empty.has_queues());
        assert!(empty.supports_timestamp_queries());
    }

    #[test]
    fn queue_families_compare_by_index_only() {
        assert_eq!(
            family(3, Q::GRAPHICS, 1, true),
            family(3, Q::TRANSFER, 8, false)
        );
        assert_ne!(
            family(3, Q::GRAPHICS, 1, true),
            family(4, Q::GRAPHICS, 1, true)
        );
    }

    #[test]
    fn unsupported_extensions_are_the_requested_ones_the_device_lacks() {
        let mut dev = device(DISCRETE, vec![gfx_family(0, 1)]);
        dev.supported_extensions = vec!["VK_KHR_swapchain".into(), "VK_EXT_mesh_shader".into()];
        let missing = dev.unsupports_extensions(&[
            ash::khr::swapchain::NAME,
            ash::khr::ray_tracing_pipeline::NAME,
            ash::ext::mesh_shader::NAME,
        ]);
        assert_eq!(missing, ["VK_KHR_ray_tracing_pipeline"]);
        assert!(dev.unsupports_extensions(&[]).is_empty());
    }

    #[test]
    fn graphics_family_needs_graphics_compute_timestamps_and_queues() {
        let dev = device(
            DISCRETE,
            vec![
                family(0, Q::GRAPHICS | Q::COMPUTE, 1, false),
                family(1, Q::GRAPHICS, 1, true),
                family(2, Q::GRAPHICS | Q::COMPUTE, 0, true),
                gfx_family(3, 4),
            ],
        );
        let (graphics, _) = dev.pick_queue_families().unwrap();
        assert_eq!(graphics.index, 3);
        assert_eq!(graphics.num_queues, 4);
    }

    #[test]
    fn transfer_family_is_a_different_family_than_graphics() {
        let dev = device(
            DISCRETE,
            vec![
                gfx_family(0, 16),
                family(1, Q::TRANSFER, 2, false),
                family(2, Q::COMPUTE | Q::TRANSFER, 8, true),
            ],
        );
        let (graphics, transfer) = dev.pick_queue_families().unwrap();
        assert_eq!(graphics.index, 0);
        assert_eq!(transfer.unwrap().index, 1);

        // Only one family: no dedicated transfer family, callers fall back to graphics.
        let single = device(INTEGRATED, vec![gfx_family(0, 1)]);
        let (graphics, transfer) = single.pick_queue_families().unwrap();
        assert_eq!(graphics.index, 0);
        assert!(transfer.is_none());
    }

    #[test]
    fn devices_without_a_graphics_family_have_no_pick() {
        let dev = device(DISCRETE, vec![family(0, Q::TRANSFER, 1, true)]);
        assert!(dev.pick_queue_families().is_none());
    }

    #[test]
    fn selection_takes_the_first_suitable_gpu() {
        let mut first = device(DISCRETE, vec![gfx_family(0, 16)]);
        first.name = "first".into();
        let mut second = device(INTEGRATED, vec![gfx_family(0, 1)]);
        second.name = "second".into();
        let (dev, graphics, _) = PhysicalDevice::select_suitable_physical_device(
            &[first, second],
            &mut Features::default(),
        )
        .unwrap();
        assert_eq!(dev.name, "first");
        assert_eq!(graphics.num_queues, 16);
    }

    #[test]
    fn cpu_devices_are_rejected() {
        let result = PhysicalDevice::select_suitable_physical_device(
            &[device(CPU, vec![gfx_family(0, 1)])],
            &mut Features::default(),
        );
        assert!(matches!(result, Err(Error::Message(_))));
        assert!(
            PhysicalDevice::select_suitable_physical_device(&[], &mut Features::default()).is_err()
        );
    }

    /// Regression: queue families found on a rejected device used to leak into the result
    /// for the device that was finally chosen.
    #[test]
    fn queue_families_come_from_the_selected_device() {
        let mut rejected = device(
            CPU,
            vec![gfx_family(0, 99), family(1, Q::TRANSFER, 99, true)],
        );
        rejected.name = "rejected".into();
        let mut chosen = device(
            DISCRETE,
            vec![family(0, Q::TRANSFER, 2, true), gfx_family(1, 4)],
        );
        chosen.name = "chosen".into();

        let (dev, graphics, transfer) = PhysicalDevice::select_suitable_physical_device(
            &[rejected, chosen],
            &mut Features::default(),
        )
        .unwrap();
        assert_eq!(dev.name, "chosen");
        assert_eq!((graphics.index, graphics.num_queues), (1, 4));
        let transfer = transfer.unwrap();
        assert_eq!((transfer.index, transfer.num_queues), (0, 2));
    }

    #[test]
    fn selection_copies_the_devices_optional_features() {
        let mut dev = device(DISCRETE, vec![gfx_family(0, 1)]);
        dev.supported_features = Features {
            rebar: true,
            present: true,
            device_debug_utils: true,
            debug_utils: true,
            mesh: true,
            raytracing: false,
            pipeline_statistics: true,
            mesh_queries: false,
            memory_budget: true,
        };

        // Without validation the device debug utils stay off, and `present` is the caller's.
        let mut features = Features::default();
        PhysicalDevice::select_suitable_physical_device(&[dev.clone()], &mut features).unwrap();
        assert!(features.mesh && features.rebar);
        assert!(features.pipeline_statistics && features.memory_budget);
        assert!(!features.raytracing && !features.mesh_queries);
        assert!(!features.device_debug_utils);
        assert!(!features.present);

        let mut features = Features {
            debug_utils: true,
            present: true,
            ..Default::default()
        };
        PhysicalDevice::select_suitable_physical_device(&[dev], &mut features).unwrap();
        assert!(features.device_debug_utils);
        assert!(features.present);
    }

    fn memory(
        types: &[(vk::MemoryPropertyFlags, u32)],
        heaps_mib: &[u64],
    ) -> vk::PhysicalDeviceMemoryProperties {
        let mut props = vk::PhysicalDeviceMemoryProperties::default();
        props.memory_type_count = types.len() as u32;
        for (slot, (property_flags, heap_index)) in props.memory_types.iter_mut().zip(types) {
            *slot = vk::MemoryType {
                property_flags: *property_flags,
                heap_index: *heap_index,
            };
        }
        props.memory_heap_count = heaps_mib.len() as u32;
        for (slot, mib) in props.memory_heaps.iter_mut().zip(heaps_mib) {
            slot.size = mib * 1024 * 1024;
        }
        props
    }

    #[test]
    fn rebar_needs_a_large_device_local_host_visible_heap() {
        use vk::MemoryPropertyFlags as M;
        let mappable = M::DEVICE_LOCAL | M::HOST_VISIBLE | M::HOST_COHERENT;

        assert!(has_rebar(&memory(
            &[(M::DEVICE_LOCAL, 0), (mappable, 0)],
            &[8192]
        )));
        // The classic 256 MiB BAR window is too small.
        assert!(!has_rebar(&memory(
            &[(M::DEVICE_LOCAL, 0), (mappable, 1)],
            &[8192, 256]
        )));
        // Host-visible system RAM is not device-local.
        assert!(!has_rebar(&memory(
            &[
                (M::DEVICE_LOCAL, 0),
                (M::HOST_VISIBLE | M::HOST_COHERENT, 1)
            ],
            &[8192, 16384]
        )));
        assert!(!has_rebar(&memory(&[], &[])));
    }
}

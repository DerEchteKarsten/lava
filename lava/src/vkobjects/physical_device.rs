//! Physical device selection and queue family capability discovery
use std::ffi::CStr;

use crate::error::{Error, Result};
use crate::vk;
use lava_macros::validation_trace;

use crate::features::{FeatureChain, Features};

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
    pub properties: vk::PhysicalDeviceProperties,
    pub mesh_shader_properties: vk::PhysicalDeviceMeshShaderPropertiesEXT<'static>,
    /// The update-after-bind descriptor limits, which the bindless sets are sized by.
    pub vk12_properties: vk::PhysicalDeviceVulkan12Properties<'static>,
    pub queue_families: Vec<QueueFamily>,
    pub supported_extensions: Vec<String>,
    /// The optional features it has, which lava enables if it picks it.
    pub supported_features: Features,
    /// What lava needs that it lacks, by name; empty on a device lava can use.
    pub lava_missing: Vec<String>,
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
        let properties = unsafe { instance.get_physical_device_properties(physical_device) };

        let name = unsafe {
            CStr::from_ptr(properties.device_name.as_ptr())
                .to_str()?
                .to_owned()
        };

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

        let has = |name: &CStr| {
            supported_extensions
                .iter()
                .any(|e| e.as_bytes() == name.to_bytes())
        };
        let features = FeatureChain::query(instance, physical_device, &has);
        let mut vk12_properties = vk::PhysicalDeviceVulkan12Properties::default();
        let mut rt_pipeline_properties =
            vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
        let mut acc_properties = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
        let mut mesh_shader_properties = vk::PhysicalDeviceMeshShaderPropertiesEXT::default();

        let mut properties2 = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut rt_pipeline_properties)
            .push_next(&mut acc_properties)
            .push_next(&mut mesh_shader_properties)
            .push_next(&mut vk12_properties);
        unsafe { instance.get_physical_device_properties2(physical_device, &mut properties2) };
        // Detach them from the chain on this stack.
        mesh_shader_properties.p_next = std::ptr::null_mut();
        vk12_properties.p_next = std::ptr::null_mut();
        rt_pipeline_properties.p_next = std::ptr::null_mut();
        acc_properties.p_next = std::ptr::null_mut();

        let mem_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let rebar = has_rebar(&mem_properties);
        let supported_features = features.optional(&has, rebar);
        let lava_missing = features.lava_missing(&has);

        Ok(Self {
            properties,
            mesh_shader_properties,
            vk12_properties,
            mem_properties,
            handel: physical_device,
            name,
            queue_families,
            supported_extensions,

            acceleration_structure_properties: supported_features
                .acceleration_structure
                .then_some(acc_properties),
            ray_tracing_pipeline_properties: supported_features
                .ray_tracing_pipeline
                .then_some(rt_pipeline_properties),
            supported_features,
            lava_missing,
        })
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

        physical_devices.sort_by_key(|pd| match pd.properties.device_type {
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

    /// Returns the first GPU (discrete before integrated) with everything lava needs, the
    /// `required` features and a usable graphics family, plus its graphics and transfer
    /// families. Presentation goes through the graphics family (checked per surface). The
    /// error names what each GPU lacks.
    #[validation_trace]
    pub(crate) fn select_suitable_physical_device(
        devices: &[PhysicalDevice],
        required: &Features,
    ) -> Result<(PhysicalDevice, QueueFamily, Option<QueueFamily>)> {
        let mut rejected = Vec::new();
        for device in devices.iter().filter(|device| {
            device.properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
                || device.properties.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
        }) {
            let mut missing = device.lava_missing.clone();
            missing.extend(
                required
                    .missing_from(&device.supported_features)
                    .into_iter()
                    .map(|name| format!("Features::{name}")),
            );
            let families = device.pick_queue_families();
            if families.is_none() {
                missing.push("a queue family with graphics, compute and timestamps".into());
            }
            match families {
                Some((graphics, transfer)) if missing.is_empty() => {
                    return Ok((device.clone(), graphics, transfer));
                }
                _ => rejected.push(format!("{} lacks {}", device.name, missing.join(", "))),
            }
        }
        Err(Error::message(&if rejected.is_empty() {
            "no GPU found".to_owned()
        } else {
            format!("no suitable GPU: {}", rejected.join("; "))
        }))
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
            properties: vk::PhysicalDeviceProperties {
                device_type,
                ..Default::default()
            },
            mesh_shader_properties: Default::default(),
            vk12_properties: Default::default(),
            handel: vk::PhysicalDevice::null(),
            name: "test device".into(),
            mem_properties: Default::default(),
            queue_families,
            supported_extensions: Vec::new(),
            supported_features: Features::default(),
            lava_missing: Vec::new(),
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
            &Features::default(),
        )
        .unwrap();
        assert_eq!(dev.name, "first");
        assert_eq!(graphics.num_queues, 16);
    }

    #[test]
    fn cpu_devices_are_rejected() {
        let result = PhysicalDevice::select_suitable_physical_device(
            &[device(CPU, vec![gfx_family(0, 1)])],
            &Features::default(),
        );
        assert!(matches!(result, Err(Error::Message(_))));
        assert!(
            PhysicalDevice::select_suitable_physical_device(&[], &Features::default()).is_err()
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
            &Features::default(),
        )
        .unwrap();
        assert_eq!(dev.name, "chosen");
        assert_eq!((graphics.index, graphics.num_queues), (1, 4));
        let transfer = transfer.unwrap();
        assert_eq!((transfer.index, transfer.num_queues), (0, 2));
    }

    #[test]
    fn selection_skips_gpus_lacking_what_lava_or_the_application_needs() {
        let mut old = device(DISCRETE, vec![gfx_family(0, 1)]);
        old.name = "old".into();
        old.lava_missing = vec!["VK_EXT_extended_dynamic_state3".into()];
        let mut plain = device(DISCRETE, vec![gfx_family(0, 1)]);
        plain.name = "plain".into();
        let mut meshy = device(INTEGRATED, vec![gfx_family(0, 1)]);
        meshy.name = "meshy".into();
        meshy.supported_features.mesh = true;

        let mesh = Features {
            mesh: true,
            ..Default::default()
        };
        let devices = [old.clone(), plain.clone(), meshy];
        let (dev, ..) = PhysicalDevice::select_suitable_physical_device(&devices, &mesh).unwrap();
        assert_eq!(dev.name, "meshy");
        let (dev, ..) =
            PhysicalDevice::select_suitable_physical_device(&devices, &Features::default())
                .unwrap();
        assert_eq!(dev.name, "plain");

        let Err(Error::Message(message)) =
            PhysicalDevice::select_suitable_physical_device(&[old, plain], &mesh)
        else {
            panic!("no GPU has mesh shaders");
        };
        assert!(message.contains("old lacks VK_EXT_extended_dynamic_state3, Features::mesh"));
        assert!(message.contains("plain lacks Features::mesh"));
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

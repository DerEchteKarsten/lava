//! The physical device lava picked, the subgroup sizes its compute shaders can require, and its queue families: the graphics family, and the dedicated compute and transfer families of `async-compute` and `async-transfer`
use crate::features::Features;
use crate::vk;

#[derive(Clone, Debug)]
pub(crate) struct QueueFamily {
    pub(crate) index: u32,
    pub(crate) properties: vk::QueueFamilyProperties,
}

impl QueueFamily {
    pub(crate) fn supports_compute(&self) -> bool {
        self.properties
            .queue_flags
            .contains(vk::QueueFlags::COMPUTE)
    }

    pub(crate) fn supports_graphics(&self) -> bool {
        self.properties
            .queue_flags
            .contains(vk::QueueFlags::GRAPHICS)
    }

    pub(crate) fn supports_transfer(&self) -> bool {
        self.properties
            .queue_flags
            .contains(vk::QueueFlags::TRANSFER)
    }

    pub(crate) fn has_queues(&self) -> bool {
        self.properties.queue_count > 0
    }

    pub(crate) fn supports_timestamp_queries(&self) -> bool {
        self.properties.timestamp_valid_bits > 0
    }
}

#[derive(Clone, Debug)]
pub(crate) struct QueueFamilies {
    pub(crate) graphics: QueueFamily,
    pub(crate) compute: Option<QueueFamily>,
    pub(crate) transfer: Option<QueueFamily>,
}

#[derive(Clone, Debug)]
pub(crate) struct PhysicalDevice {
    pub(crate) handle: vk::PhysicalDevice,
    pub(crate) name: String,
    pub(crate) mem_properties: vk::PhysicalDeviceMemoryProperties,
    pub(crate) properties: vk::PhysicalDeviceProperties,
    #[cfg_attr(not(feature = "mesh-shader"), allow(dead_code))]
    pub(crate) mesh_shader_properties: vk::PhysicalDeviceMeshShaderPropertiesEXT<'static>,
    pub(crate) vk11_properties: vk::PhysicalDeviceVulkan11Properties<'static>,
    /// `VK_EXT_conservative_rasterization`'s `primitiveUnderestimation`.
    pub(crate) primitive_underestimation: bool,
    /// `VK_EXT_extended_dynamic_state3`'s `dynamicPrimitiveTopologyUnrestricted`: a dynamic
    /// topology may leave the class the pipeline was made with.
    pub(crate) topology_unrestricted: bool,
    /// `VK_EXT_multi_draw`'s `maxMultiDrawCount` (0 without it).
    #[cfg_attr(not(feature = "multi-draw"), allow(dead_code))]
    pub(crate) max_multi_draw_count: u32,
    /// `multiviewMeshShader`: mesh passes in multiview renderings.
    pub(crate) multiview_mesh_shader: bool,
    /// The subgroup sizes a pipeline stage can require (`subgroupSizeControl`, which lava
    /// enables where the device has it); `None` without it.
    pub(crate) subgroup_sizes: Option<SubgroupSizes>,
    /// The driver is Mesa's ANV, whose traces need `MEMORY_READ` to see a rebuild of a
    /// structure traced earlier in the submission (`acceleration_structure::trace_read`).
    #[cfg_attr(not(feature = "acceleration-structure"), allow(dead_code))]
    pub(crate) stale_trace_reads: bool,
    /// Zeroed on a device without `VK_EXT_descriptor_heap`, which lava never picks.
    pub(crate) descriptor_heap_properties: vk::PhysicalDeviceDescriptorHeapPropertiesEXT<'static>,
    /// Its graphics family and dedicated ones (`pick_queue_families`); `None` without a
    /// graphics family, which lava never picks.
    pub(crate) families: Option<QueueFamilies>,
    pub(crate) supported_extensions: Vec<String>,
    /// The optional features it has, which lava enables if it picks it.
    pub(crate) supported_features: Features,
    /// What lava needs that it lacks, by name.
    pub(crate) lava_missing: Vec<String>,
    #[cfg(feature = "ray-tracing-pipeline")]
    pub(crate) ray_tracing_pipeline_properties:
        Option<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'static>>,
    #[cfg(feature = "acceleration-structure")]
    pub(crate) acceleration_structure_properties:
        Option<vk::PhysicalDeviceAccelerationStructurePropertiesKHR<'static>>,
}

/// What `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` may ask for
/// (`VkPhysicalDeviceVulkan13Properties`).
#[derive(Clone, Copy, Debug)]
pub struct SubgroupSizes {
    pub min: u32,
    pub max: u32,
    /// The stages that can require a size.
    pub stages: vk::ShaderStageFlags,
    pub max_compute_workgroup_subgroups: u32,
}

/// The first graphics (with compute and timestamps), dedicated compute and dedicated transfer
/// family; `None` without a graphics family.
pub(crate) fn pick_queue_families(families: &[QueueFamily]) -> Option<QueueFamilies> {
    let first = |pick: fn(&QueueFamily) -> bool| families.iter().find(|f| pick(f)).cloned();
    Some(QueueFamilies {
        graphics: first(|f| {
            f.has_queues()
                && f.supports_graphics()
                && f.supports_compute()
                && f.supports_timestamp_queries()
        })?,
        compute: first(|f| f.has_queues() && f.supports_compute() && !f.supports_graphics()),
        transfer: first(|f| {
            f.has_queues()
                && f.supports_transfer()
                && !f.supports_graphics()
                && !f.supports_compute()
        }),
    })
}

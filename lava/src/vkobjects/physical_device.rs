//! The physical device lava picked and its queue families: the graphics family, and the dedicated compute and transfer families of `async-compute` and `async-transfer`
use crate::features::Features;
use crate::vk;

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
pub(crate) struct QueueFamilies {
    pub(crate) graphics: QueueFamily,
    pub(crate) compute: Option<QueueFamily>,
    pub(crate) transfer: Option<QueueFamily>,
}

#[derive(Clone, Debug)]
pub struct PhysicalDevice {
    pub handel: vk::PhysicalDevice,
    pub name: String,
    pub mem_properties: vk::PhysicalDeviceMemoryProperties,
    pub properties: vk::PhysicalDeviceProperties,
    pub mesh_shader_properties: vk::PhysicalDeviceMeshShaderPropertiesEXT<'static>,
    pub vk11_properties: vk::PhysicalDeviceVulkan11Properties<'static>,
    pub vk12_properties: vk::PhysicalDeviceVulkan12Properties<'static>,
    /// `VK_EXT_conservative_rasterization`'s `primitiveUnderestimation`.
    pub primitive_underestimation: bool,
    /// `VK_EXT_extended_dynamic_state3`'s `dynamicPrimitiveTopologyUnrestricted`: a dynamic
    /// topology may leave the class the pipeline was made with.
    pub topology_unrestricted: bool,
    /// `VK_EXT_multi_draw`'s `maxMultiDrawCount` (0 without it).
    pub max_multi_draw_count: u32,
    /// `multiviewMeshShader`: mesh passes in multiview renderings.
    pub multiview_mesh_shader: bool,
    /// Zeroed on a device without `VK_EXT_descriptor_heap`, which lava never picks.
    pub descriptor_heap_properties: vk::PhysicalDeviceDescriptorHeapPropertiesEXT<'static>,
    pub queue_families: Vec<QueueFamily>,
    pub supported_extensions: Vec<String>,
    /// The optional features it has, which lava enables if it picks it.
    pub supported_features: Features,
    /// What lava needs that it lacks, by name.
    pub lava_missing: Vec<String>,
    #[cfg(feature = "ray-tracing-pipeline")]
    pub ray_tracing_pipeline_properties:
        Option<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'static>>,
    #[cfg(feature = "acceleration-structure")]
    pub acceleration_structure_properties:
        Option<vk::PhysicalDeviceAccelerationStructurePropertiesKHR<'static>>,
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

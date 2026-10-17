//! Wrappers for Vulkan objects without a module of their own: the physical device, and with their features acceleration structures and ray tracing pipelines
#[cfg(feature = "acceleration-structure")]
pub mod acceleration_structure;
pub mod physical_device;
#[cfg(feature = "ray-tracing-pipeline")]
pub mod rt_pipeline;

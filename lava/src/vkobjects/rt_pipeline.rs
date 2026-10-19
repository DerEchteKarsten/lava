//! Ray tracing pipelines: creation from a pass's shader groups, and their shader binding table (raygen, miss, hit and callable regions, handles only)
use crate::vk;

use crate::{
    buffer::Buffer,
    error::{Error, Result},
    features::{Feature, gate},
    state::{Ctx, Functions},
};

pub struct RaytracingPipeline {
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) sbt: ShaderBindingTable,
}

impl RaytracingPipeline {
    /// `groups` are in binding table order: one raygen, then `counts[1]` miss, `counts[2]` hit
    /// and `counts[3]` callable groups.
    pub(crate) fn new(
        stages: &[vk::PipelineShaderStageCreateInfo],
        groups: &[vk::RayTracingShaderGroupCreateInfoKHR],
        counts: [u32; 4],
        max_recursion: u32,
    ) -> Result<Self> {
        let properties = Ctx::physical_device()
            .ray_tracing_pipeline_properties
            .ok_or_else(|| Error::message("ray tracing pipeline properties are unavailable"))?;
        if max_recursion > properties.max_ray_recursion_depth {
            return Err(Error::message(format!(
                "a recursion depth of {max_recursion} exceeds the device's {}",
                properties.max_ray_recursion_depth
            )));
        }
        let mut flags = crate::pipeline::heap_flags();
        let info = vk::RayTracingPipelineCreateInfoKHR::default()
            .push_next(&mut flags)
            .stages(stages)
            .groups(groups)
            .max_pipeline_ray_recursion_depth(max_recursion);
        gate::RayTracingPipeline::check()?;
        let functions = Functions::raytracing_pipeline();
        let pipeline = unsafe {
            functions
                .create_ray_tracing_pipelines(
                    vk::DeferredOperationKHR::null(),
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&info),
                    None,
                )
                .map_err(|(partial, result)| {
                    // Vulkan may hand back partially created pipelines with the error.
                    for pipeline in partial {
                        Ctx::device().destroy_pipeline(pipeline, None);
                    }
                    Error::Vulkan(result)
                })?[0]
        };
        let sbt = (|| -> Result<ShaderBindingTable> {
            let handle_size = properties.shader_group_handle_size as usize;
            let layout = sbt_layout(
                counts,
                properties.shader_group_handle_size,
                properties.shader_group_handle_alignment,
                properties.shader_group_base_alignment,
            );
            let group_count: u32 = counts.iter().sum();
            let handles = unsafe {
                functions.get_ray_tracing_shader_group_handles(
                    pipeline,
                    0,
                    group_count,
                    group_count as usize * handle_size,
                )?
            };
            let mut buffer = Buffer::with_flags(
                vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR,
                true,
                layout.size.max(1),
                Some(properties.shader_group_base_alignment),
            )?;
            // The handles come in group order, which is region order.
            let mut handles = handles.chunks_exact(handle_size);
            let data = buffer.host_mut();
            for (&(offset, _), count) in layout.regions.iter().zip(counts) {
                for i in 0..count as u64 {
                    let at = (offset + i * layout.stride) as usize;
                    data[at..at + handle_size].copy_from_slice(handles.next().unwrap());
                }
            }
            let address = buffer.address();
            let regions = layout.regions.map(|(offset, size)| {
                if size == 0 {
                    return vk::StridedDeviceAddressRegionKHR::default();
                }
                vk::StridedDeviceAddressRegionKHR::default()
                    .device_address(address + offset)
                    .size(size)
                    .stride(layout.stride)
            });
            Ok(ShaderBindingTable {
                _buffer: buffer,
                regions,
            })
        })()
        .inspect_err(|_| unsafe { Ctx::device().destroy_pipeline(pipeline, None) })?;
        Ok(Self { pipeline, sbt })
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct SbtLayout {
    pub(crate) stride: u64,
    /// (offset, size) of the raygen, miss, hit and callable region.
    pub(crate) regions: [(u64, u64); 4],
    pub(crate) size: u64,
}

/// Each group takes a handle rounded up to `handle_alignment` and each region starts at a
/// multiple of `base_alignment`; empty regions have size 0.
pub(crate) fn sbt_layout(
    counts: [u32; 4],
    handle_size: u32,
    handle_alignment: u32,
    base_alignment: u32,
) -> SbtLayout {
    let stride = (handle_size as u64).next_multiple_of(handle_alignment as u64);
    let mut offset: u64 = 0;
    let regions = counts.map(|count| {
        let size = count as u64 * stride;
        if size == 0 {
            return (offset, 0);
        }
        let start = offset.next_multiple_of(base_alignment as u64);
        offset = start + size;
        (start, size)
    });
    SbtLayout {
        stride,
        regions,
        size: offset,
    }
}

/// The handle of every group of a pipeline; groups carry no shader record data.
pub struct ShaderBindingTable {
    _buffer: Buffer<u8>,
    /// The raygen, miss, hit and callable regions; an empty one is all zeros.
    pub(crate) regions: [vk::StridedDeviceAddressRegionKHR; 4],
}

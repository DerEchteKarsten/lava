//! Ray tracing pipelines: creation from a pass's shader groups, and their shader binding table (raygen, miss, hit and callable regions, handles only)
use crate::vk;

use crate::{
    buffer::{Buffer, usage},
    error::{Error, Result},
    state::{Ctx, Functions},
};

fn functions() -> Result<&'static ash::khr::ray_tracing_pipeline::Device> {
    Functions::raytracing_pipeline().ok_or_else(|| {
        Error::message(
            "ray tracing pipelines aren't supported by this device (`Ctx::features().ray_tracing_pipeline`)",
        )
    })
}

pub struct RaytracingPipeline {
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) sbt: ShaderBindingTable,
}

impl RaytracingPipeline {
    /// `groups` are in binding table order: one raygen, then `counts[1]` miss, `counts[2]` hit
    /// and `counts[3]` callable groups.
    pub(crate) fn new(
        layout: vk::PipelineLayout,
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
        let info = vk::RayTracingPipelineCreateInfoKHR::default()
            .layout(layout)
            .stages(stages)
            .groups(groups)
            .max_pipeline_ray_recursion_depth(max_recursion);
        let functions = functions()?;
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
        let sbt = ShaderBindingTable::new(pipeline, counts, &properties).inspect_err(|_| unsafe {
            Ctx::device().destroy_pipeline(pipeline, None)
        })?;
        Ok(Self { pipeline, sbt })
    }
}

/// Where the regions of a shader binding table lie in its buffer.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct SbtLayout {
    /// The bytes per group: a handle, aligned.
    pub(crate) stride: u64,
    /// (offset, size) of the raygen, miss, hit and callable region.
    pub(crate) regions: [(u64, u64); 4],
    pub(crate) size: u64,
}

/// The layout of a binding table of `counts` groups per region: each group takes a handle
/// rounded up to `handle_alignment`, each region starts at a multiple of `base_alignment`, and
/// the raygen region is one group (its size has to equal the stride). Empty regions have size 0.
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

/// The handle of every group of a pipeline, laid out by [`sbt_layout`]. Groups carry no
/// shader record data: everything a shader reads comes through push constants and addresses.
pub struct ShaderBindingTable {
    _buffer: Buffer<u8, usage::ShaderBindingTable>,
    /// The raygen, miss, hit and callable regions; an empty one is all zeros.
    pub(crate) regions: [vk::StridedDeviceAddressRegionKHR; 4],
}

impl ShaderBindingTable {
    fn new(
        pipeline: vk::Pipeline,
        counts: [u32; 4],
        properties: &vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
    ) -> Result<Self> {
        let handle_size = properties.shader_group_handle_size as usize;
        let layout = sbt_layout(
            counts,
            properties.shader_group_handle_size,
            properties.shader_group_handle_alignment,
            properties.shader_group_base_alignment,
        );
        let group_count: u32 = counts.iter().sum();
        let handles = unsafe {
            functions()?.get_ray_tracing_shader_group_handles(
                pipeline,
                0,
                group_count,
                group_count as usize * handle_size,
            )?
        };

        let mut buffer = Buffer::raw(
            usage::ShaderBindingTable,
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

        let address = buffer.address;
        let regions = layout.regions.map(|(offset, size)| {
            if size == 0 {
                return vk::StridedDeviceAddressRegionKHR::default();
            }
            vk::StridedDeviceAddressRegionKHR::default()
                .device_address(address + offset)
                .size(size)
                .stride(layout.stride)
        });
        Ok(Self {
            _buffer: buffer,
            regions,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regions_start_at_the_base_alignment() {
        // The B390's handle size and alignments.
        assert_eq!(
            sbt_layout([1, 2, 2, 1], 32, 16, 16),
            SbtLayout {
                stride: 32,
                regions: [(0, 32), (32, 64), (96, 64), (160, 32)],
                size: 192,
            }
        );
        assert_eq!(
            sbt_layout([1, 2, 2, 1], 32, 32, 64),
            SbtLayout {
                stride: 32,
                regions: [(0, 32), (64, 64), (128, 64), (192, 32)],
                size: 224,
            }
        );
    }

    #[test]
    fn handles_are_padded_to_their_alignment() {
        let layout = sbt_layout([1, 3, 0, 0], 20, 32, 64);
        assert_eq!(layout.stride, 32);
        assert_eq!(layout.regions[0], (0, 32));
        assert_eq!(layout.regions[1], (64, 96));
    }

    #[test]
    fn empty_regions_have_no_size() {
        let layout = sbt_layout([1, 0, 1, 0], 32, 16, 64);
        assert_eq!(layout.regions[1].1, 0);
        assert_eq!(layout.regions[2], (64, 32));
        assert_eq!(layout.regions[3].1, 0);
        assert_eq!(layout.size, 96);
    }
}

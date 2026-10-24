//! Ray tracing pipelines: creation of a batch from passes' shader groups (feature checked first), and their shader binding table (raygen, miss, hit and callable regions, handles only)
use crate::vk;

use crate::{
    buffer::{Buffer, Upload},
    error::{Error, Result},
    features::{Feature, gate},
    state::{Ctx, Functions},
};

pub(crate) struct RaytracingPipeline {
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) sbt: ShaderBindingTable,
}

/// What a recording binds and traces with: the pipeline and its binding table's regions.
#[derive(Clone, Copy)]
pub(crate) struct BoundRayTracing {
    pub(crate) pipeline: vk::Pipeline,
    /// The raygen, miss, hit and callable regions.
    pub(crate) regions: [vk::StridedDeviceAddressRegionKHR; 4],
}

/// The create info of one ray tracing pipeline: its stages, its groups in binding table order
/// (one raygen, then `counts[1]` miss, `counts[2]` hit and `counts[3]` callable groups) and its
/// recursion depth.
pub(crate) struct RayTracingDesc<'a> {
    pub(crate) stages: smallvec::SmallVec<[vk::PipelineShaderStageCreateInfo<'a>; 8]>,
    pub(crate) groups: smallvec::SmallVec<[vk::RayTracingShaderGroupCreateInfoKHR<'a>; 8]>,
    pub(crate) counts: [u32; 4],
    pub(crate) max_recursion: u32,
}

impl RaytracingPipeline {
    /// One pipeline per description, made with one call, each with its binding table; on an
    /// error none is left.
    pub(crate) fn create_all(
        cache: vk::PipelineCache,
        descs: &[RayTracingDesc],
    ) -> Result<Vec<Self>> {
        gate::RayTracingPipeline::check()?;
        let properties = Ctx::physical_device()
            .ray_tracing_pipeline_properties
            .expect("the device has ray tracing pipelines, so lava read their properties");
        if let Some(desc) =
            (descs.iter()).find(|desc| desc.max_recursion > properties.max_ray_recursion_depth)
        {
            return Err(Error::Unsupported(format!(
                "a recursion depth of {} exceeds the device's {}",
                desc.max_recursion, properties.max_ray_recursion_depth
            )));
        }
        let mut flags: Vec<_> = descs
            .iter()
            .map(|_| crate::pipeline::heap_flags())
            .collect();
        let infos: Vec<_> = (descs.iter().zip(&mut flags))
            .map(|(desc, flags)| {
                vk::RayTracingPipelineCreateInfoKHR::default()
                    .push_next(flags)
                    .stages(&desc.stages)
                    .groups(&desc.groups)
                    .max_pipeline_ray_recursion_depth(desc.max_recursion)
            })
            .collect();
        let functions = Functions::raytracing_pipeline();
        let pipelines = unsafe {
            functions
                .create_ray_tracing_pipelines(vk::DeferredOperationKHR::null(), cache, &infos, None)
                .map_err(|(partial, result)| {
                    // Vulkan may hand back partially created pipelines with the error.
                    for pipeline in partial {
                        Ctx::device().destroy_pipeline(pipeline, None);
                    }
                    Error::Vulkan(result)
                })?
        };
        let mut made = Vec::with_capacity(descs.len());
        let mut pipelines = pipelines.into_iter().zip(descs);
        for (pipeline, desc) in pipelines.by_ref() {
            match ShaderBindingTable::new(pipeline, desc.counts, &properties) {
                Ok(sbt) => made.push(Self { pipeline, sbt }),
                Err(err) => {
                    let device = Ctx::device();
                    for (pipeline, _) in pipelines {
                        unsafe { device.destroy_pipeline(pipeline, None) };
                    }
                    for made in made {
                        unsafe { device.destroy_pipeline(made.pipeline, None) };
                    }
                    unsafe { device.destroy_pipeline(pipeline, None) };
                    return Err(err);
                }
            }
        }
        Ok(made)
    }

    pub(crate) fn bound(&self) -> BoundRayTracing {
        BoundRayTracing {
            pipeline: self.pipeline,
            regions: self.sbt.regions,
        }
    }
}

impl ShaderBindingTable {
    /// The table of `pipeline`'s groups, `counts` of them per region.
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
            Functions::raytracing_pipeline().get_ray_tracing_shader_group_handles(
                pipeline,
                0,
                group_count,
                group_count as usize * handle_size,
            )?
        };
        let mut buffer = Buffer::with_flags(
            vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR,
            Upload,
            // Never empty: every pass has a raygen.
            layout.size,
            Some(properties.shader_group_base_alignment as u64),
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
pub(crate) struct ShaderBindingTable {
    _buffer: Buffer<u8, Upload>,
    /// The raygen, miss, hit and callable regions; an empty one is all zeros.
    pub(crate) regions: [vk::StridedDeviceAddressRegionKHR; 4],
}

use anyhow::Result;
use ash::vk;

use crate::{
    buffer::Buffer,
    state::{Ctx, Functions},
};

pub fn alinged_size(size: u32, alignment: u32) -> u32 {
    (size + (alignment - 1)) & !(alignment - 1)
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RayTracingShaderGroupInfo {
    pub group_count: u32,
    pub raygen_shader_count: u32,
    pub miss_shader_count: u32,
    pub hit_shader_count: u32,
}

#[derive(Debug, Clone)]
pub struct RayTracingShaderCreateInfo<'a> {
    pub stages: &'a [vk::PipelineShaderStageCreateInfo<'a>],
    pub group: RayTracingShaderGroup,
}

#[derive(Debug, Clone, Copy)]
pub enum RayTracingShaderGroup {
    RayGen,
    Miss,
    Hit,
}

pub struct RaytracingPipeline {
    pub pipeline: vk::Pipeline,
    pub sbt: ShaderBindingTable,
}

impl RaytracingPipeline {
    pub fn new(
        pipeline_layout: vk::PipelineLayout,
        shaders_create_info: &[RayTracingShaderCreateInfo],
    ) -> Result<Self> {
        let mut shader_group_info = RayTracingShaderGroupInfo {
            group_count: shaders_create_info.len() as u32,
            ..Default::default()
        };

        let mut stages = vec![];
        let mut groups = vec![];

        for shader in shaders_create_info.iter() {
            match shader.group {
                RayTracingShaderGroup::RayGen => shader_group_info.raygen_shader_count += 1,
                RayTracingShaderGroup::Miss => shader_group_info.miss_shader_count += 1,
                RayTracingShaderGroup::Hit => shader_group_info.hit_shader_count += 1,
            };

            let shader_index = stages.len();

            let mut group = vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR);
            group = match shader.group {
                RayTracingShaderGroup::RayGen | RayTracingShaderGroup::Miss => {
                    group.general_shader(shader_index as _)
                }
                RayTracingShaderGroup::Hit => {
                    group = group
                        .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                        .closest_hit_shader(shader_index as _);
                    if shader.stages.len() >= 2 {
                        group = group
                            .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                            .any_hit_shader((shader_index as u32) + 1);
                    }
                    if shader.stages.len() >= 3 {
                        group = group
                            .ty(vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP)
                            .any_hit_shader((shader_index as u32) + 1)
                            .intersection_shader((shader_index as u32) + 2);
                    }

                    group
                }
            };

            stages.extend(shader.stages);
            groups.push(group);
        }

        let pipe_info = vk::RayTracingPipelineCreateInfoKHR::default()
            .layout(pipeline_layout)
            .stages(&stages)
            .groups(&groups)
            .max_pipeline_ray_recursion_depth(1);

        let pipeline = unsafe {
            Functions::raytracing_pipeline()
                .unwrap()
                .create_ray_tracing_pipelines(
                    vk::DeferredOperationKHR::null(),
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&pipe_info),
                    None,
                )
        }
        .unwrap();
        let sbt = ShaderBindingTable::new(&pipeline[0], &shader_group_info)?;
        Ok(RaytracingPipeline {
            pipeline: pipeline[0],
            sbt,
        })
    }
}

pub struct ShaderBindingTable {
    pub _buffer: Buffer<u8>,
    pub raygen_region: vk::StridedDeviceAddressRegionKHR,
    pub miss_region: vk::StridedDeviceAddressRegionKHR,
    pub hit_region: vk::StridedDeviceAddressRegionKHR,
}

impl ShaderBindingTable {
    pub fn new(pipeline: &vk::Pipeline, shaders: &RayTracingShaderGroupInfo) -> Result<Self> {
        let desc = shaders;

        let handle_size = Ctx::physical_device()
            .ray_tracing_pipeline_properties
            .unwrap()
            .shader_group_handle_size;
        let handle_alignment = Ctx::physical_device()
            .ray_tracing_pipeline_properties
            .unwrap()
            .shader_group_handle_alignment;
        let aligned_handle_size = alinged_size(handle_size, handle_alignment);
        let handle_pad = aligned_handle_size - handle_size;

        let group_alignment = Ctx::physical_device()
            .ray_tracing_pipeline_properties
            .unwrap()
            .shader_group_base_alignment;

        let data_size = desc.group_count * handle_size;
        let handles = unsafe {
            Functions::raytracing_pipeline()
                .unwrap()
                .get_ray_tracing_shader_group_handles(
                    *pipeline,
                    0,
                    desc.group_count,
                    data_size as _,
                )?
        };

        let raygen_region_size = alinged_size(
            desc.raygen_shader_count * aligned_handle_size,
            group_alignment,
        );

        let miss_region_size = alinged_size(
            desc.miss_shader_count * aligned_handle_size,
            group_alignment,
        );
        let hit_region_size =
            alinged_size(desc.hit_shader_count * aligned_handle_size, group_alignment);

        let buffer_size = raygen_region_size + miss_region_size + hit_region_size;
        let mut stb_data = Vec::<u8>::with_capacity(buffer_size as _);
        let groups_shader_count = [
            desc.raygen_shader_count,
            desc.miss_shader_count,
            desc.hit_shader_count,
        ];

        let buffer_usage = vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR;

        let buffer: Buffer<u8> = Buffer::raw(
            buffer_usage,
            true,
            buffer_size as _,
            Some(
                Ctx::physical_device()
                    .ray_tracing_pipeline_properties
                    .unwrap()
                    .shader_group_base_alignment,
            ),
        )?;

        let mut offset = 0;
        for group_shader_count in groups_shader_count {
            let group_size = group_shader_count * aligned_handle_size;
            let aligned_group_size = alinged_size(group_size, group_alignment);
            let group_pad = aligned_group_size - group_size;

            for _ in 0..group_shader_count {
                for _ in 0..handle_size as usize {
                    stb_data.push(handles[offset]);
                    offset += 1;
                }

                for _ in 0..handle_pad {
                    stb_data.push(0x0);
                }
            }

            for _ in 0..group_pad {
                stb_data.push(0x0);
            }
        }

        buffer.range(..).copy_from(stb_data.as_slice());

        let raygen_region = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(buffer.address)
            .size(raygen_region_size as _)
            .stride(raygen_region_size as _);

        let miss_region = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(buffer.address + raygen_region.size)
            .size(miss_region_size as _)
            .stride(aligned_handle_size as _);

        let hit_region = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(buffer.address + raygen_region.size + miss_region.size)
            .size(hit_region_size as _)
            .stride(aligned_handle_size as _);

        Ok(Self {
            _buffer: buffer,
            raygen_region,
            miss_region,
            hit_region,
        })
    }
}

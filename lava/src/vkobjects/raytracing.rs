use std::{ffi::CString, mem::MaybeUninit, sync::OnceLock};

use anyhow::Result;
use ash::{
    khr::{acceleration_structure, ray_tracing_pipeline},
    vk, Instance,
};
use bevy_ecs::resource::Resource;
use gpu_allocator::{vulkan::Allocation, MemoryLocation};
use log::debug;

use crate::{renderer::bindless::BindlessDescriptorHeap, PipelineCache};

use super::{
    buffer::{Buffer, BufferHandle, BufferType, DynamicBuffer},
    Context,
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
    pub source: &'a [(&'a str, &'a str, vk::ShaderStageFlags)],
    pub group: RayTracingShaderGroup,
}

#[derive(Debug, Clone, Copy)]
pub enum RayTracingShaderGroup {
    RayGen,
    Miss,
    Hit,
}

pub struct AccelerationStructure {
    pub ty: vk::AccelerationStructureTypeKHR,
    pub accel: vk::AccelerationStructureKHR,
    pub size: u64,
}

pub struct RayTracingContext {
    pub pipeline_properties: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'static>,
    pub pipeline_fn: ray_tracing_pipeline::Device,
    pub acceleration_structure_properties:
        vk::PhysicalDeviceAccelerationStructurePropertiesKHR<'static>,
    pub acceleration_structure_fn: acceleration_structure::Device,
}

pub static RAYTRACING: OnceLock<Option<RayTracingContext>> = OnceLock::new();

impl RayTracingContext {
    pub fn get() -> &'static Option<Self> {
        RAYTRACING.get().unwrap()
    }

    pub fn new() -> Self {
        let ctx = Context::get();
        let (pipeline_properties, acceleration_structure_properties) = unsafe {
            let mut rt_pipeline_properties =
                vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
            let mut acc_properties =
                vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
            let mut subgroups = vk::PhysicalDeviceSubgroupProperties::default();

            let mut physical_device_properties2 = vk::PhysicalDeviceProperties2::default()
                .push_next(&mut rt_pipeline_properties)
                .push_next(&mut acc_properties)
                .push_next(&mut subgroups);
            ctx.instance.get_physical_device_properties2(
                ctx.physical_device.handel,
                &mut physical_device_properties2,
            );
            (rt_pipeline_properties, acc_properties)
        };
        let pipeline_fn = ash::khr::ray_tracing_pipeline::Device::new(&ctx.instance, &ctx.device);

        let acceleration_structure_fn =
            ash::khr::acceleration_structure::Device::new(&ctx.instance, &ctx.device);

        Self {
            pipeline_properties,
            pipeline_fn,
            acceleration_structure_properties,
            acceleration_structure_fn,
        }
    }

    pub fn create_acceleration_structure(
        &self,
        level: vk::AccelerationStructureTypeKHR,
        as_geometry: &[vk::AccelerationStructureGeometryKHR],
        as_ranges: &[vk::AccelerationStructureBuildRangeInfoKHR],
        max_primitive_counts: &[u32],
        buffer: &impl BufferType,
        offset: u64,
        scratch_buffer: &mut DynamicBuffer,
        cmd: &vk::CommandBuffer,
    ) -> Result<AccelerationStructure> {
        let build_geo_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(level)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(as_geometry);

        let build_size = unsafe {
            let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
            self.acceleration_structure_fn
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &build_geo_info,
                    max_primitive_counts,
                    &mut size_info,
                );
            size_info
        };

        let create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(buffer.to_vk())
            .offset(offset)
            .size(build_size.acceleration_structure_size)
            .ty(level);
        let handle = unsafe {
            self.acceleration_structure_fn
                .create_acceleration_structure(&create_info, None)?
        };
        scratch_buffer.cmd_grow_to_size(build_size.build_scratch_size, cmd);
        let build_geo_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(level)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(as_geometry)
            .dst_acceleration_structure(handle)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch_buffer.get_address(),
            });

        unsafe {
            self.acceleration_structure_fn
                .cmd_build_acceleration_structures(*cmd, &[build_geo_info], &[as_ranges])
        };

        Ok(AccelerationStructure {
            accel: handle,
            ty: level,
            size: build_size.acceleration_structure_size,
        })
    }

    pub fn create_raytracing_pipeline(
        &self,
        shader_cache: &mut PipelineCache,
        pipeline_layout: vk::PipelineLayout,
        shaders_create_info: &[RayTracingShaderCreateInfo],
    ) -> Result<(vk::Pipeline, ShaderBindingTable)> {
        let mut shader_group_info = RayTracingShaderGroupInfo {
            group_count: shaders_create_info.len() as u32,
            ..Default::default()
        };

        let mut modules = vec![];
        let mut stages = vec![];
        let mut groups = vec![];

        for shader in shaders_create_info.iter() {
            let mut this_modules = vec![];
            let mut this_stages = vec![];

            shader.source.into_iter().for_each(|s| {
                let module = shader_cache.create_shader_module(s.0).unwrap();
                let stage = vk::PipelineShaderStageCreateInfo::default()
                    .stage(s.2)
                    .module(module)
                    .name(std::ffi::CStr::from_bytes_until_nul(s.1.as_bytes()).unwrap());
                this_modules.push(module);
                this_stages.push(stage);
            });

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
                    if shader.source.len() >= 2 {
                        group = group
                            .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                            .any_hit_shader((shader_index as u32) + 1);
                    }
                    if shader.source.len() >= 3 {
                        group = group
                            .ty(vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP)
                            .any_hit_shader((shader_index as u32) + 1)
                            .intersection_shader((shader_index as u32) + 2);
                    }

                    group
                }
            };

            modules.append(&mut this_modules);
            stages.append(&mut this_stages);
            groups.push(group);
        }

        let pipe_info = vk::RayTracingPipelineCreateInfoKHR::default()
            .layout(pipeline_layout)
            .stages(&stages)
            .groups(&groups)
            .max_pipeline_ray_recursion_depth(1);

        let pipeline = unsafe {
            self.pipeline_fn.create_ray_tracing_pipelines(
                vk::DeferredOperationKHR::null(),
                vk::PipelineCache::null(),
                std::slice::from_ref(&pipe_info),
                None,
            )
        }
        .unwrap();
        let sbt = ShaderBindingTable::new(&pipeline[0], &shader_group_info)?;
        Ok((pipeline[0], sbt))
    }
}

pub struct ShaderBindingTable {
    pub _buffer: Buffer,
    pub raygen_region: vk::StridedDeviceAddressRegionKHR,
    pub miss_region: vk::StridedDeviceAddressRegionKHR,
    pub hit_region: vk::StridedDeviceAddressRegionKHR,
}

impl ShaderBindingTable {
    pub fn new(pipeline: &vk::Pipeline, shaders: &RayTracingShaderGroupInfo) -> Result<Self> {
        let ctx = Context::get();
        let ray_tracing = RayTracingContext::get().as_ref().unwrap();
        let desc = shaders;

        let handle_size = ray_tracing.pipeline_properties.shader_group_handle_size;
        let handle_alignment = ray_tracing
            .pipeline_properties
            .shader_group_handle_alignment;
        let aligned_handle_size = alinged_size(handle_size, handle_alignment);
        let handle_pad = aligned_handle_size - handle_size;

        let group_alignment = ray_tracing.pipeline_properties.shader_group_base_alignment;

        let data_size = desc.group_count * handle_size;
        let handles = unsafe {
            ray_tracing
                .pipeline_fn
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

        let buffer_usage = vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let memory_location = MemoryLocation::CpuToGpu;

        let buffer = Buffer::new_aligned(
            buffer_usage,
            memory_location,
            buffer_size as _,
            Some(ray_tracing.pipeline_properties.shader_group_base_alignment as u64),
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

        buffer.copy_data_to_buffer(&stb_data)?;

        let raygen_region = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(buffer.address)
            .size(raygen_region_size as _)
            .stride(raygen_region_size as _); //REMINDER

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

use std::{
    collections::VecDeque,
    mem::MaybeUninit,
    sync::{atomic::AtomicU32, Mutex, OnceLock},
};

use anyhow::{Ok, Result};
use ash::vk;

use crate::{state::{Ctx, Features, Functions}, vkobjects::{acceleration_structure::AccelerationStructure, buffer::{Buffer, DynamicBuffer}, image::ImageType, queue::CommandBuffer}};

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct ImmutableSampler(u32);

impl ImmutableSampler {
    pub fn new(mag_filter: vk::Filter, min_filter: vk::Filter) -> Result<Self> {
        if mag_filter == vk::Filter::CUBIC_EXT
            || mag_filter == vk::Filter::CUBIC_IMG
            || min_filter == vk::Filter::CUBIC_EXT
            || min_filter == vk::Filter::CUBIC_IMG
        {
            Err(anyhow::Error::msg("Unsuported Sampler"))
        } else {
            Ok(Self(
                (mag_filter == vk::Filter::NEAREST) as u32
                    | (min_filter == vk::Filter::NEAREST) as u32,
            ))
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
#[repr(C)]
pub struct DescriptorHandle(pub u32);

enum AccessType {
    ReadOnly,
    ReadWrite,
}

#[derive(Clone, Copy)]
#[repr(u32)]
pub enum RenderResourceTag {
    Buffer,
    Image,
    Texture,
    AccelerationStructure,
}

impl RenderResourceTag {
    fn table(&self) -> BindlessTableType {
        match self {
            Self::Buffer => BindlessTableType::Buffers,
            Self::Image => BindlessTableType::Images,
            Self::Texture => BindlessTableType::Textures,
            Self::AccelerationStructure => BindlessTableType::AccelerationStructures,
        }
    }
}

trait ToBufferHandle {
    fn to_vk(&self) -> vk::Buffer;
    fn get_size(&self) -> u64;
}

impl ToBufferHandle for Buffer {
    fn to_vk(&self) -> vk::Buffer {
        self.buffer
    }
    fn get_size(&self) -> u64 {
        self.size
    }
}


impl ToBufferHandle for DynamicBuffer {
    fn to_vk(&self) -> vk::Buffer {
        self.buffer.buffer
    }
    fn get_size(&self) -> u64 {
        self.capacity
    }
}

impl DescriptorHandle {
    pub fn new(tag: RenderResourceTag, index: u32) -> Self {
        Self(((tag as u32) << 30) | index)
    }
    fn bump_version_and_update_tag(self, tag: RenderResourceTag) -> Self {
        Self::new(tag, self.index())
    }
    pub fn index(&self) -> u32 {
        self.0 & 0xfffffff
    }
}

#[derive(Debug)]
pub struct BindlessDescriptorHeap {
    available_recycled_descriptors: Mutex<VecDeque<DescriptorHandle>>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_index: [std::sync::atomic::AtomicU32; 4],
    set_layouts: [vk::DescriptorSetLayout; 4],
    sets: [vk::DescriptorSet; 4],
    pub layout: vk::PipelineLayout,
}

#[derive(PartialEq)]
enum BindlessTableType {
    Buffers,
    Images,
    Textures,
    AccelerationStructures,
}
impl BindlessTableType {
    fn all_tables() -> [BindlessTableType; 4] {
        [
            BindlessTableType::Buffers,
            BindlessTableType::Images,
            BindlessTableType::Textures,
            BindlessTableType::AccelerationStructures,
        ]
    }

    fn to_vk(&self) -> vk::DescriptorType {
        match self {
            BindlessTableType::AccelerationStructures => {
                vk::DescriptorType::ACCELERATION_STRUCTURE_KHR
            }
            BindlessTableType::Buffers => vk::DescriptorType::STORAGE_BUFFER,
            BindlessTableType::Images => vk::DescriptorType::STORAGE_IMAGE,
            BindlessTableType::Textures => vk::DescriptorType::SAMPLED_IMAGE,
        }
    }

    fn set_index(&self) -> usize {
        match self {
            BindlessTableType::Buffers => 0,
            BindlessTableType::Images => 1,
            BindlessTableType::Textures => 2,
            BindlessTableType::AccelerationStructures => 3,
        }
    }

    fn table_size(&self) -> u32 {
        match self {
            BindlessTableType::Buffers => {
                Ctx::physical_device()
                    .limits
                    .max_descriptor_set_storage_buffers
            }
            BindlessTableType::Images => {
                Ctx::physical_device().limits.max_descriptor_set_storage_images
            }
            BindlessTableType::Textures => {
                Ctx::physical_device().limits.max_descriptor_set_sampled_images
            }
            BindlessTableType::AccelerationStructures => {
                if let Some(r) = Ctx::physical_device().acceleration_structure_properties {
                    r
                        .max_descriptor_set_acceleration_structures
                } else {
                    0
                }
            }
        }
        .min(100000)
    }

    pub fn descriptor_pool_sizes(immutable_sampler_count: u32) -> Vec<vk::DescriptorPoolSize> {
        Self::all_tables()
            .iter()
            .filter_map(|table| {
                if Functions::acceleration_structure().is_some()
                    || table.to_vk() != vk::DescriptorType::ACCELERATION_STRUCTURE_KHR
                {
                    Some(vk::DescriptorPoolSize {
                        ty: table.to_vk(),
                        descriptor_count: table.table_size(),
                    })
                } else {
                    None
                }
            })
            .chain(std::iter::once(vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: immutable_sampler_count,
            }))
            .collect::<Vec<_>>()
    }
}

pub static BINDLESS: OnceLock<BindlessDescriptorHeap> = OnceLock::new();

impl BindlessDescriptorHeap {
    pub fn get() -> &'static Self {
        BINDLESS.get().unwrap()
    }

    pub fn retire_handle(&self, handle: DescriptorHandle) {
        self.available_recycled_descriptors
            .lock()
            .unwrap()
            .push_back(handle);
    }

    fn increment_descriptor(&self, table: BindlessTableType) -> u32 {
        self.descriptor_index[table.set_index()].fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    fn fetch_available_descriptor(&self, tag: RenderResourceTag) -> DescriptorHandle {
        self.available_recycled_descriptors
            .lock()
            .unwrap()
            .pop_front()
            .map_or_else(
                || DescriptorHandle::new(tag, self.increment_descriptor(tag.table())),
                |recycled_handle| recycled_handle.bump_version_and_update_tag(tag),
            )
    }

    pub fn allocate_buffer_handle(&self, buffer: &impl ToBufferHandle) -> DescriptorHandle {
        let handle = Self::fetch_available_descriptor(self, RenderResourceTag::Buffer);

        let buffer_info = [vk::DescriptorBufferInfo {
            buffer: buffer.to_vk(),
            offset: 0,
            range: buffer.get_size(),
        }];

        let write = [vk::WriteDescriptorSet {
            dst_set: self.sets[BindlessTableType::Buffers.set_index()],
            dst_binding: 0,
            descriptor_count: 1,
            dst_array_element: handle.index(),
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            ..Default::default()
        }
        .buffer_info(&buffer_info)];
        unsafe {
            Ctx::device().update_descriptor_sets(&write, &[]);
        };

        handle
    }

    pub fn update_buffer_handle(&self, buffer: &impl ToBufferHandle, handle: DescriptorHandle) {
        let buffer_info = [vk::DescriptorBufferInfo {
            buffer: buffer.to_vk(),
            offset: 0,
            range: buffer.get_size(),
        }];

        let write = [vk::WriteDescriptorSet {
            dst_set: self.sets[BindlessTableType::Buffers.set_index()],
            dst_binding: 0,
            descriptor_count: 1,
            dst_array_element: handle.index(),
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            ..Default::default()
        }
        .buffer_info(&buffer_info)];
        unsafe {
            Ctx::device().update_descriptor_sets(&write, &[]);
        };
    }

    pub fn allocate_image_handle(&self, image: &impl ImageType) -> DescriptorHandle {
        let handle = Self::fetch_available_descriptor(self, RenderResourceTag::Image);

        let image_info = vk::DescriptorImageInfo {
            image_layout: vk::ImageLayout::GENERAL,
            sampler: vk::Sampler::null(),
            image_view: image.get_view(),
        };

        let write = [vk::WriteDescriptorSet {
            dst_set: self.sets[BindlessTableType::Images.set_index()],
            dst_binding: 0,
            descriptor_count: 1,
            dst_array_element: handle.index(),
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            p_image_info: &image_info,
            ..Default::default()
        }];
        unsafe {
            Ctx::device().update_descriptor_sets(&write, &[]);
        };

        handle
    }

    pub fn allocate_texture_handle(&self, image: &impl ImageType) -> DescriptorHandle {
        let handle = Self::fetch_available_descriptor(self, RenderResourceTag::Texture);

        let image_info = vk::DescriptorImageInfo {
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            sampler: vk::Sampler::null(),
            image_view: image.get_view(),
        };

        let write = [vk::WriteDescriptorSet {
            dst_set: self.sets[BindlessTableType::Textures.set_index()],
            dst_binding: 4,
            descriptor_count: 1,
            dst_array_element: handle.index(),
            descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
            p_image_info: &image_info,
            ..Default::default()
        }];
        unsafe {
            Ctx::device().update_descriptor_sets(&write, &[]);
        };

        handle
    }

    pub fn allocate_acceleration_structure_handle(
        &self,
        tlas: &AccelerationStructure,
    ) -> DescriptorHandle {
        let handle =
            Self::fetch_available_descriptor(self, RenderResourceTag::AccelerationStructure);

        let mut write_set_as = vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(std::slice::from_ref(&tlas.accel));

        let mut write = vk::WriteDescriptorSet::default()
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .dst_binding(0)
            .dst_array_element(handle.index())
            .dst_set(self.sets[BindlessTableType::AccelerationStructures.set_index()])
            .push_next(&mut write_set_as);
        write.descriptor_count = 1;
        unsafe {
            Ctx::device().update_descriptor_sets(&[write], &[]);
        };

        handle
    }

    pub fn init() -> Result<()> {
        let immutable_samplers: [vk::Sampler; 4] = (0..4)
            .into_iter()
            .map(|i: i32| {
                let create_info = vk::SamplerCreateInfo {
                    min_filter: if i % 2 == 0 {
                        vk::Filter::NEAREST
                    } else {
                        vk::Filter::LINEAR
                    },
                    mag_filter: if i.div_floor(2) == 0 {
                        vk::Filter::NEAREST
                    } else {
                        vk::Filter::LINEAR
                    },
                    ..Default::default()
                };
                unsafe {
                    Ctx::device()
                        .create_sampler(&create_info, None)
                        .unwrap()
                }
            })
            .collect::<Vec<vk::Sampler>>()
            .try_into()
            .unwrap();

        let descriptor_sizes =
            BindlessTableType::descriptor_pool_sizes(immutable_samplers.len() as u32);

        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&descriptor_sizes)
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND_EXT)
            .max_sets(4);

        let descriptor_pool = unsafe {
            Ctx::device()
                .create_descriptor_pool(&descriptor_pool_info, None)
                .unwrap()
        };
        Functions::set_debug_name("BindlessDescriptorPool", descriptor_pool);
        let (set_layouts, layout) = Self::create_bindless_layout(&immutable_samplers);
        Functions::set_debug_name("BindlessLayout", layout);
        for i in 0..set_layouts.len() {
            if Functions::acceleration_structure().is_none() && i == 3 {
                continue;
            }
            Functions::set_debug_name(
                &format!(
                    "BindlessDescriptorSetLayout_{}",
                    match i {
                        0 => "Buffers",
                        1 => "Images",
                        2 => "Tectures",
                        3 => "Tlas",
                        _ => unreachable!(),
                    }
                ),
                set_layouts[i],
            );
        }

        let descriptor_counts = BindlessTableType::all_tables()
            .iter()
            .filter_map(|table| {
                if Functions::acceleration_structure().is_some() || *table != BindlessTableType::AccelerationStructures {
                    Some(table.table_size())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let mut set_counts = vk::DescriptorSetVariableDescriptorCountAllocateInfo::default()
            .descriptor_counts(&descriptor_counts);

        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(if Functions::acceleration_structure().is_none() {
                &set_layouts[0..3]
            } else {
                &set_layouts
            })
            .push_next(&mut set_counts);

        let sets = unsafe {
            Ctx::device()
                .allocate_descriptor_sets(&allocate_info)
                .unwrap()
        };
        let sets: [vk::DescriptorSet; 4] = if Functions::acceleration_structure().is_none() {
            let mut res = [vk::DescriptorSet::default(); 4];
            res[0..3].copy_from_slice(sets.as_slice());
            res
        } else {
            sets.try_into().unwrap()
        };

        for i in 0..sets.len() {
            if Functions::acceleration_structure().is_none() && i == 3 {
                continue;
            }
            Functions::set_debug_name(
                &format!(
                    "BindlessDescriptorSet_{}",
                    match i {
                        0 => "Buffers",
                        1 => "Images",
                        2 => "Textures",
                        3 => "Tlas",
                        _ => unreachable!(),
                    }
                ),
                sets[i],
            );
        }

        BINDLESS.set(Self {
            descriptor_pool,
            available_recycled_descriptors: Mutex::new(VecDeque::new()),
            descriptor_index: [
                AtomicU32::new(0),
                AtomicU32::new(0),
                AtomicU32::new(0),
                AtomicU32::new(0),
            ],
            set_layouts,
            sets,
            layout,
        }).unwrap();
        Ok(())
    }

    pub(super) fn create_bindless_layout(
        immutable_samplers: &[vk::Sampler],
    ) -> ([vk::DescriptorSetLayout; 4], vk::PipelineLayout) {
        let mut descriptor_layouts = [vk::DescriptorSetLayout::default(); 4];
        BindlessTableType::all_tables()
            .iter()
            .enumerate()
            .for_each(|(set_idx, table)| unsafe {
                assert_eq!(table.set_index(), set_idx);

                let mut descriptor_binding_flags = vec![
                    vk::DescriptorBindingFlags::PARTIALLY_BOUND
                        | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT
                        | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                ];

                let mut set = vec![vk::DescriptorSetLayoutBinding {
                    binding: 0,
                    descriptor_type: table.to_vk(),
                    descriptor_count: table.table_size(),
                    stage_flags: vk::ShaderStageFlags::ALL,
                    p_immutable_samplers: std::ptr::null(),
                    ..Default::default()
                }];

                if *table == BindlessTableType::Textures {
                    descriptor_binding_flags.push(vk::DescriptorBindingFlags::empty());

                    // Set texture binding start at the end of the immutable samplers.
                    set[0].binding = immutable_samplers.len() as u32;
                    set.push(vk::DescriptorSetLayoutBinding {
                        binding: 0,
                        descriptor_type: vk::DescriptorType::SAMPLER,
                        descriptor_count: immutable_samplers.len() as u32,
                        stage_flags: vk::ShaderStageFlags::ALL,
                        p_immutable_samplers: immutable_samplers.as_ptr(),
                        ..Default::default()
                    });
                }

                let mut ext_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT::default()
                    .binding_flags(&descriptor_binding_flags);
                if Functions::acceleration_structure().is_some() || *table != BindlessTableType::AccelerationStructures {
                    descriptor_layouts[set_idx] = Ctx::device()
                        .create_descriptor_set_layout(
                            &vk::DescriptorSetLayoutCreateInfo::default()
                                .bindings(&set)
                                .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                                .push_next(&mut ext_flags),
                            None,
                        )
                        .unwrap();
                }
            });

        let num_push_constants = 4;
        let num_push_constants_sized = std::mem::size_of::<u32>() as u32 * num_push_constants;

        let push_constant_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::ALL,
            offset: 0,
            size: num_push_constants_sized,
        };

        let push_constant_ranges = [push_constant_range];

        let layout_create_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(if Functions::acceleration_structure().is_none() {
                &descriptor_layouts[0..3]
            } else {
                &descriptor_layouts
            })
            .push_constant_ranges(&push_constant_ranges);

        let pipeline_layout = unsafe {
            Ctx::device()
                .create_pipeline_layout(&layout_create_info, None)
        }
        .expect("Failed creating pipeline layout.");

        (descriptor_layouts, pipeline_layout)
    }

    pub fn bind(&self, raytracing: bool, cmd: &vk::CommandBuffer) -> Result<()> {
        let sets = if raytracing {
            &self.sets
        } else {
            &self.sets[0..3]
        };
        unsafe {
            Ctx::device().cmd_bind_descriptor_sets(
                *cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.layout,
                0,
                sets,
                &[],
            )
        };
        if raytracing {
            unsafe {
                Ctx::device().cmd_bind_descriptor_sets(
                    *cmd,
                    vk::PipelineBindPoint::RAY_TRACING_KHR,
                    self.layout,
                    0,
                    sets,
                    &[],
                )
            };
        }
        unsafe {
            Ctx::device().cmd_bind_descriptor_sets(
                *cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                sets,
                &[],
            )
        };

        Ok(())
    }
}

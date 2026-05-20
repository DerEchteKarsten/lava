use std::sync::{OnceLock, atomic::AtomicU32};

use anyhow::Result;
use ash::vk::{self, BorderColor, CompareOp, SamplerAddressMode, SamplerMipmapMode};
use bytemuck::{Pod, Zeroable};

use crate::{
    image::{
        format::Format,
        slice::ImageView,
        usage::{BindlessImageUsageSet, IsSampled, IsStorage, UsageSet},
    },
    state::Ctx,
};

#[derive(Debug)]
pub struct Bindless {
    num_images: AtomicU32,
    num_textures: AtomicU32,
    layout: vk::PipelineLayout,
    layouts: [vk::DescriptorSetLayout; 2],
    sets: [vk::DescriptorSet; 2],
    pool: vk::DescriptorPool,
}

static BINDLESS: OnceLock<Bindless> = OnceLock::new();

pub const NULL_HANDLE: u32 = !0;
#[derive(Pod, Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct BindlessHandle {
    pub descriptor_index_set0: u32,
    pub descriptor_index_set1: u32,
}

impl Bindless {
    fn get() -> &'static Self {
        BINDLESS.get().unwrap()
    }
    pub fn layout() -> vk::PipelineLayout {
        Self::get().layout
    }
    pub fn init() -> Result<()> {
        let mut layouts = [vk::DescriptorSetLayout::default(); 2];
        let sci2 = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .mipmap_mode(SamplerMipmapMode::NEAREST)
            .address_mode_u(SamplerAddressMode::CLAMP_TO_BORDER)
            .address_mode_v(SamplerAddressMode::CLAMP_TO_BORDER)
            .address_mode_w(SamplerAddressMode::CLAMP_TO_BORDER)
            .border_color(BorderColor::FLOAT_OPAQUE_WHITE)
            .anisotropy_enable(false)
            .compare_enable(false);

        let sci = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .border_color(BorderColor::FLOAT_OPAQUE_WHITE)
            .address_mode_u(SamplerAddressMode::CLAMP_TO_BORDER)
            .address_mode_v(SamplerAddressMode::CLAMP_TO_BORDER)
            .address_mode_w(SamplerAddressMode::CLAMP_TO_BORDER)
            .mipmap_mode(SamplerMipmapMode::NEAREST);

        let samplers = [
            unsafe { Ctx::device().create_sampler(&sci2, None) }.unwrap(),
            unsafe { Ctx::device().create_sampler(&sci, None) }.unwrap(),
        ];
        let descriptor_binding_flags = [
            vk::DescriptorBindingFlags::empty(),
            vk::DescriptorBindingFlags::PARTIALLY_BOUND_EXT
                | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT_EXT
                | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND_EXT,
        ];
        let bindings = [
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_count: samplers.len() as u32,
                descriptor_type: vk::DescriptorType::SAMPLER,
                stage_flags: vk::ShaderStageFlags::ALL,
                ..Default::default()
            }
            .immutable_samplers(&samplers),
            vk::DescriptorSetLayoutBinding {
                binding: samplers.len() as u32,
                descriptor_count: Ctx::physical_device()
                    .limits
                    .max_descriptor_set_sampled_images,
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                stage_flags: vk::ShaderStageFlags::ALL,
                ..Default::default()
            },
        ];
        let mut ext_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT::default()
            .binding_flags(&descriptor_binding_flags);
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings)
            .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL_EXT)
            .push_next(&mut ext_flags);
        layouts[0] = unsafe { Ctx::device().create_descriptor_set_layout(&layout_info, None) }?;

        let descriptor_binding_flags = [vk::DescriptorBindingFlags::PARTIALLY_BOUND_EXT
            | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT_EXT
            | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND_EXT];

        let bindings = [vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_count: Ctx::physical_device()
                .limits
                .max_descriptor_set_storage_images,
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            stage_flags: vk::ShaderStageFlags::ALL,
            ..Default::default()
        }];
        let mut ext_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT::default()
            .binding_flags(&descriptor_binding_flags);
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings)
            .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL_EXT)
            .push_next(&mut ext_flags);
        layouts[1] = unsafe { Ctx::device().create_descriptor_set_layout(&layout_info, None) }?;

        let ranges = [vk::PushConstantRange {
            offset: 0,
            size: Ctx::physical_device().limits.max_push_constants_size,
            stage_flags: vk::ShaderStageFlags::ALL,
        }];
        let pipline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(&ranges);
        let layout = unsafe { Ctx::device().create_pipeline_layout(&pipline_layout_info, None)? };

        let pool_sizes = [
            vk::DescriptorPoolSize {
                descriptor_count: Ctx::physical_device()
                    .limits
                    .max_descriptor_set_sampled_images
                    .min(1000),
                ty: vk::DescriptorType::SAMPLED_IMAGE,
            },
            vk::DescriptorPoolSize {
                descriptor_count: samplers.len() as u32,
                ty: vk::DescriptorType::SAMPLER,
            },
            vk::DescriptorPoolSize {
                descriptor_count: Ctx::physical_device()
                    .limits
                    .max_descriptor_set_storage_images
                    .min(1000),
                ty: vk::DescriptorType::STORAGE_IMAGE,
            },
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND_EXT)
            .max_sets(2)
            .pool_sizes(&pool_sizes);
        let pool = unsafe { Ctx::device().create_descriptor_pool(&pool_info, None) }.unwrap();

        let desc_counts = [
            Ctx::physical_device()
                .limits
                .max_descriptor_set_sampled_images
                .min(1000),
            Ctx::physical_device()
                .limits
                .max_descriptor_set_storage_images
                .min(1000),
        ];
        let mut alloc_info = vk::DescriptorSetVariableDescriptorCountAllocateInfo::default()
            .descriptor_counts(&desc_counts);
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts)
            .push_next(&mut alloc_info);
        let sets = unsafe { Ctx::device().allocate_descriptor_sets(&allocate_info) }?
            .try_into()
            .unwrap();

        BINDLESS
            .set(Self {
                num_images: AtomicU32::new(0),
                num_textures: AtomicU32::new(0),
                layout,
                layouts,
                sets,
                pool,
            })
            .unwrap();
        Ok(())
    }

    pub fn push<F: Format, U: UsageSet>(image: ImageView<F, U>) -> Option<BindlessHandle> {
        let handle = match U::SET {
            BindlessImageUsageSet::None => return None,
            BindlessImageUsageSet::Both => BindlessHandle {
                descriptor_index_set1: Self::get()
                    .num_images
                    .fetch_add(1, std::sync::atomic::Ordering::Acquire),
                descriptor_index_set0: Self::get()
                    .num_textures
                    .fetch_add(1, std::sync::atomic::Ordering::Acquire),
            },
            BindlessImageUsageSet::SampledImage => BindlessHandle {
                descriptor_index_set1: NULL_HANDLE,
                descriptor_index_set0: Self::get()
                    .num_textures
                    .fetch_add(1, std::sync::atomic::Ordering::Acquire),
            },
            BindlessImageUsageSet::StorageImage => BindlessHandle {
                descriptor_index_set1: Self::get()
                    .num_images
                    .fetch_add(1, std::sync::atomic::Ordering::Acquire),
                descriptor_index_set0: NULL_HANDLE,
            },
        };
        Self::write_image(image, handle);
        Some(handle)
    }

    pub fn write_image<F: Format, U: UsageSet>(image: ImageView<F, U>, handle: BindlessHandle) {
        let image_info = [vk::DescriptorImageInfo {
            image_layout: U::PREFERED_LAYOUT,
            image_view: image.view,
            ..Default::default()
        }];
        let write = vk::WriteDescriptorSet::default()
            .descriptor_count(1)
            .dst_binding(0)
            .image_info(&image_info);
        let mut writes = Vec::new();
        if handle.descriptor_index_set0 != NULL_HANDLE {
            writes.push(
                write
                    .dst_array_element(handle.descriptor_index_set0)
                    .dst_set(Self::get().sets[0])
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE),
            );
        }
        if handle.descriptor_index_set1 != NULL_HANDLE {
            writes.push(
                write
                    .dst_array_element(handle.descriptor_index_set1)
                    .dst_set(Self::get().sets[1])
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE),
            );
        }
        unsafe { Ctx::device().update_descriptor_sets(&writes, &[]) };
    }

    pub fn bind(cmd: &vk::CommandBuffer) {
        let s = Self::get();
        unsafe {
            Ctx::device().cmd_bind_descriptor_sets(
                *cmd,
                vk::PipelineBindPoint::COMPUTE,
                s.layout,
                0,
                &s.sets,
                &[],
            )
        };
        if Ctx::features().raytracing {
            unsafe {
                Ctx::device().cmd_bind_descriptor_sets(
                    *cmd,
                    vk::PipelineBindPoint::RAY_TRACING_KHR,
                    s.layout,
                    0,
                    &s.sets,
                    &[],
                )
            };
        }
        unsafe {
            Ctx::device().cmd_bind_descriptor_sets(
                *cmd,
                vk::PipelineBindPoint::GRAPHICS,
                s.layout,
                0,
                &s.sets,
                &[],
            )
        };
    }

    pub fn destroy() {
        unsafe {
            let s = Self::get();
            Ctx::device().destroy_pipeline_layout(s.layout, None);
            for i in s.layouts {
                Ctx::device().destroy_descriptor_set_layout(i, None);
            }
            Ctx::device().free_descriptor_sets(s.pool, &s.sets).unwrap();
            Ctx::device().destroy_descriptor_pool(s.pool, None);
        }
    }
}

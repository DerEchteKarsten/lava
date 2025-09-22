use std::sync::{atomic::AtomicU32, OnceLock};

use anyhow::Result;
use ash::vk;

use crate::{state::Ctx, vkobjects::image::ImageType};

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

pub type BindlessHandle = u32;

impl Bindless {
    fn get() -> &'static Self {
        BINDLESS.get().unwrap()
    }
    pub fn layout() -> vk::PipelineLayout {
        Self::get().layout
    }
    pub fn init() -> Result<()> {
        let mut layouts = [vk::DescriptorSetLayout::default(); 2];
        let sci = vk::SamplerCreateInfo::default();
        let samplers = [
            unsafe { Ctx::device().create_sampler(&sci, None) }.unwrap()
        ];
        let mut descriptor_binding_flags = [
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
            }.immutable_samplers(&samplers),
            vk::DescriptorSetLayoutBinding {
                binding: samplers.len() as u32,
                descriptor_count: Ctx::physical_device().limits.max_descriptor_set_sampled_images,
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                stage_flags: vk::ShaderStageFlags::ALL,
                ..Default::default()
            }
        ];
        let mut ext_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT::default()
                .binding_flags(&descriptor_binding_flags);
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings)
            .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL_EXT)  
            .push_next(&mut ext_flags);
        layouts[0] = unsafe { Ctx::device().create_descriptor_set_layout(&layout_info, None) }?;

        let mut descriptor_binding_flags = [
                    vk::DescriptorBindingFlags::PARTIALLY_BOUND_EXT
                        | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT_EXT
                        | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND_EXT,
                ];

        let bindings = [
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_count: Ctx::physical_device().limits.max_descriptor_set_storage_images,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                stage_flags: vk::ShaderStageFlags::ALL,
                ..Default::default()
            }
        ];
        let mut ext_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT::default()
                .binding_flags(&descriptor_binding_flags);
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings)
            .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL_EXT)
            .push_next(&mut ext_flags);
        layouts[1] = unsafe { Ctx::device().create_descriptor_set_layout(&layout_info, None) }?;
        

        let ranges = [
                vk::PushConstantRange {
                    offset: 0,
                    size: Ctx::physical_device().limits.max_push_constants_size,
                    stage_flags: vk::ShaderStageFlags::ALL,
                }
            ];
        let pipline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(&ranges);
        let layout = unsafe { Ctx::device().create_pipeline_layout(&pipline_layout_info, None)? };

        let pool_sizes = [
            vk::DescriptorPoolSize {
                descriptor_count: Ctx::physical_device().limits.max_descriptor_set_sampled_images.min(1000),
                ty: vk::DescriptorType::SAMPLED_IMAGE,
            },
            vk::DescriptorPoolSize {
                descriptor_count: samplers.len() as u32,
                ty: vk::DescriptorType::SAMPLER,
            },
            vk::DescriptorPoolSize {
                descriptor_count: Ctx::physical_device().limits.max_descriptor_set_storage_images.min(1000),
                ty: vk::DescriptorType::STORAGE_IMAGE,
            }
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND_EXT)
            .max_sets(2)
            .pool_sizes(&pool_sizes);
        let pool = unsafe { Ctx::device().create_descriptor_pool(&pool_info, None) }.unwrap();

        let desc_counts = [
            Ctx::physical_device().limits.max_descriptor_set_sampled_images.min(1000),
            Ctx::physical_device().limits.max_descriptor_set_storage_images.min(1000),
        ];
        let mut alloc_info = vk::DescriptorSetVariableDescriptorCountAllocateInfo::default()
            .descriptor_counts(&desc_counts);
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts)
            .push_next(&mut alloc_info);
        let sets = unsafe { Ctx::device().allocate_descriptor_sets(&allocate_info) }?.try_into().unwrap();

        BINDLESS.set(Self { num_images: AtomicU32::new(0), num_textures: AtomicU32::new(0), layout, layouts, sets, pool }).unwrap();
        Ok(())
    }

    pub fn push_image(image: &impl ImageType) -> BindlessHandle {
        let handle = Self::get().num_images.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Self::write_image(image, handle);
        handle
    }

    pub fn push_texture(texture: &impl ImageType) -> BindlessHandle {
        let handle = Self::get().num_textures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Self::write_texture(texture, handle);
        handle
    }

    pub fn write_image(image: &impl ImageType, handle: BindlessHandle) {
        let image_info = [
            vk::DescriptorImageInfo {
                image_layout: vk::ImageLayout::GENERAL,
                image_view: image.get_view(),
                ..Default::default()
            }
        ];
        let write = vk::WriteDescriptorSet::default()
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .dst_array_element(handle)
            .dst_binding(0)
            .dst_set(Self::get().sets[1])
            .image_info(&image_info);
        unsafe { Ctx::device().update_descriptor_sets(std::slice::from_ref(&write), &[]) };
    }

    pub fn write_texture(texture: &impl ImageType, handle: BindlessHandle) {
        let image_info = [
            vk::DescriptorImageInfo {
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                image_view: texture.get_view(),
                ..Default::default()
            }
        ];
        let write = vk::WriteDescriptorSet::default()
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .dst_array_element(handle)
            .dst_binding(1)
            .dst_set(Self::get().sets[0])
            .image_info(&image_info);
        unsafe { Ctx::device().update_descriptor_sets(std::slice::from_ref(&write), &[]) };
    }
    pub fn bind(cmd: &vk::CommandBuffer) {
        let s = Self::get();
        unsafe { Ctx::device().cmd_bind_descriptor_sets(*cmd, vk::PipelineBindPoint::COMPUTE, s.layout, 0, &s.sets, &[]) };
        if Ctx::features().raytracing {
            unsafe { Ctx::device().cmd_bind_descriptor_sets(*cmd, vk::PipelineBindPoint::RAY_TRACING_KHR, s.layout, 0, &s.sets, &[]) };
        }
        unsafe { Ctx::device().cmd_bind_descriptor_sets(*cmd, vk::PipelineBindPoint::GRAPHICS, s.layout, 0, &s.sets, &[]) };
    }
}
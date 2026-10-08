//! Global bindless descriptor sets: immutable samplers, caller-indexed image descriptors, handles
use std::sync::OnceLock;

use crate::error::{Error, Result};
use ash::vk::{self, BorderColor, SamplerAddressMode, SamplerMipmapMode};
use bytemuck::{Pod, Zeroable};
use lava_macros::validation_trace;

use crate::{
    image::{
        format::Format,
        slice::ImageView,
        usage::{IsSampled, IsStorage},
    },
    state::Ctx,
};

#[derive(Debug)]
pub struct Bindless {
    /// Descriptors allocated in the sampled (set 0) and storage (set 1) image arrays.
    counts: [u32; 2],
    num_samplers: u32,
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
impl BindlessHandle {
    pub fn none() -> Self {
        Self {
            descriptor_index_set0: NULL_HANDLE,
            descriptor_index_set1: NULL_HANDLE,
        }
    }
}

impl Default for BindlessHandle {
    fn default() -> Self {
        Self::none()
    }
}

/// Size of the sampled image array: valid sampled indices are `0..max_sampled_images()`.
pub fn max_sampled_images() -> u32 {
    Bindless::get().counts[0]
}

/// Size of the storage image array: valid storage indices are `0..max_storage_images()`.
pub fn max_storage_images() -> u32 {
    Bindless::get().counts[1]
}

/// Descriptor writes collected for one `vkUpdateDescriptorSets`. The caller picks every index;
/// a slot may only be rewritten while no pending command buffer uses it.
#[derive(Default)]
pub struct BindlessWrites {
    writes: Vec<(usize, u32, vk::DescriptorImageInfo)>,
}

impl BindlessWrites {
    pub fn sampled<F: Format, U: IsSampled>(&mut self, view: ImageView<F, U>, index: u32) {
        self.push(0, index, view.view);
    }

    pub fn storage<F: Format, U: IsStorage>(&mut self, view: ImageView<F, U>, index: u32) {
        self.push(1, index, view.view);
    }

    fn push(&mut self, set: usize, index: u32, view: vk::ImageView) {
        let info = vk::DescriptorImageInfo {
            image_layout: vk::ImageLayout::GENERAL,
            image_view: view,
            ..Default::default()
        };
        self.writes.push((set, index, info));
    }

    #[validation_trace]
    pub fn submit(self) {
        if self.writes.is_empty() {
            return;
        }
        let bindless = Bindless::get();
        let writes: Vec<_> = self
            .writes
            .iter()
            .map(|(set, index, info)| {
                assert!(
                    *index < bindless.counts[*set],
                    "bindless index {index} is outside set {set} of {} descriptors",
                    bindless.counts[*set]
                );
                let (binding, ty) = match set {
                    0 => (bindless.num_samplers, vk::DescriptorType::SAMPLED_IMAGE),
                    _ => (0, vk::DescriptorType::STORAGE_IMAGE),
                };
                vk::WriteDescriptorSet::default()
                    .dst_set(bindless.sets[*set])
                    .dst_binding(binding)
                    .dst_array_element(*index)
                    .descriptor_type(ty)
                    .image_info(std::slice::from_ref(info))
            })
            .collect();
        unsafe { Ctx::device().update_descriptor_sets(&writes, &[]) };
    }
}

impl Bindless {
    fn get() -> &'static Self {
        BINDLESS
            .get()
            .expect("bindless resources have not been initialized")
    }
    pub(crate) fn layout() -> vk::PipelineLayout {
        Self::get().layout
    }
    pub(crate) fn init() -> Result<()> {
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

        // Sampler 2: trilinear + repeat, what mipmapped material textures use.
        let sci3 = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(SamplerMipmapMode::LINEAR)
            .address_mode_u(SamplerAddressMode::REPEAT)
            .address_mode_v(SamplerAddressMode::REPEAT)
            .address_mode_w(SamplerAddressMode::REPEAT)
            .max_lod(vk::LOD_CLAMP_NONE);

        let samplers = [
            unsafe { Ctx::device().create_sampler(&sci2, None) }?,
            unsafe { Ctx::device().create_sampler(&sci, None) }?,
            unsafe { Ctx::device().create_sampler(&sci3, None) }?,
        ];
        let descriptor_binding_flags = [
            vk::DescriptorBindingFlags::empty(),
            vk::DescriptorBindingFlags::PARTIALLY_BOUND_EXT
                | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT_EXT
                | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND_EXT
                | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING,
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
            | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND_EXT
            | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING];

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
                    .min(10000),
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
                    .min(10000),
                ty: vk::DescriptorType::STORAGE_IMAGE,
            },
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND_EXT)
            .max_sets(2)
            .pool_sizes(&pool_sizes);
        let pool = unsafe { Ctx::device().create_descriptor_pool(&pool_info, None) }?;

        let desc_counts = [
            Ctx::physical_device()
                .limits
                .max_descriptor_set_sampled_images
                .min(10000),
            Ctx::physical_device()
                .limits
                .max_descriptor_set_storage_images
                .min(10000),
        ];
        let mut alloc_info = vk::DescriptorSetVariableDescriptorCountAllocateInfo::default()
            .descriptor_counts(&desc_counts);
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts)
            .push_next(&mut alloc_info);
        let sets = unsafe { Ctx::device().allocate_descriptor_sets(&allocate_info) }?
            .try_into()
            .expect("allocated descriptor set count matches the two layouts");

        BINDLESS
            .set(Self {
                counts: desc_counts,
                num_samplers: samplers.len() as u32,
                layout,
                layouts,
                sets,
                pool,
            })
            .map_err(|_| Error::message("bindless resources were already initialized"))?;
        Ok(())
    }

    pub(crate) fn bind(cmd: &vk::CommandBuffer) {
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

    pub(crate) fn destroy() {
        unsafe {
            let s = Self::get();
            Ctx::device().destroy_pipeline_layout(s.layout, None);
            for i in s.layouts {
                Ctx::device().destroy_descriptor_set_layout(i, None);
            }
            if let Err(err) = Ctx::device().free_descriptor_sets(s.pool, &s.sets) {
                tracing::error!(%err, "failed to free bindless descriptor sets");
            }
            Ctx::device().destroy_descriptor_pool(s.pool, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_handle_is_null_in_both_sets() {
        let none = BindlessHandle::none();
        assert_eq!(none.descriptor_index_set0, NULL_HANDLE);
        assert_eq!(none.descriptor_index_set1, NULL_HANDLE);
        assert_eq!(NULL_HANDLE, u32::MAX);
    }

    /// Shaders receive the handle as two consecutive `uint`s (`Img.SampledIndex`, `StorageIndex`).
    #[test]
    fn handle_is_two_u32s_sampled_first() {
        assert_eq!(size_of::<BindlessHandle>(), 8);
        let handle = BindlessHandle {
            descriptor_index_set0: 1,
            descriptor_index_set1: 2,
        };
        assert_eq!(bytemuck::bytes_of(&handle), &[1, 0, 0, 0, 2, 0, 0, 0]);
    }
}

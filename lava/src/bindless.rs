//! Global bindless descriptor sets: the application's immutable samplers (`SamplerDesc`), caller-indexed image descriptors, handles
use ash::vk::TaggedStructure;
use std::sync::OnceLock;

use crate::error::{Error, Result};
use ash::vk::{self, BorderColor, Filter, SamplerAddressMode, SamplerMipmapMode};
use bytemuck::{Pod, Zeroable};
use lava_macros::validation_trace;

use crate::{
    pipeline::MAX_PUSH_CONSTANTS_SIZE,
    image::{Dim, format::Format, slice::ImageView, usage::ImageUsage},
    state::Ctx,
};

#[derive(Debug)]
pub struct Bindless {
    /// Descriptors allocated in the sampled (set 0) and storage (set 1) image arrays.
    counts: [u32; 2],
    layout: vk::PipelineLayout,
    layouts: [vk::DescriptorSetLayout; 2],
    /// Sampled images and storage images.
    sets: [vk::DescriptorSet; 2],
    pool: vk::DescriptorPool,
}

static BINDLESS: OnceLock<Bindless> = OnceLock::new();

/// The binding of the sampled images in set 0, after the samplers' array (binding 0).
const SAMPLED_IMAGES_BINDING: u32 = 1;

/// An immutable sampler, which shaders pick by its index in `Config::samplers`
/// (`Sample(uv, sampler)` in `bindless.slang`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplerDesc {
    pub mag_filter: Filter,
    pub min_filter: Filter,
    pub mipmap_mode: SamplerMipmapMode,
    /// Used for every coordinate.
    pub address_mode: SamplerAddressMode,
    /// The color outside the image with `CLAMP_TO_BORDER`.
    pub border_color: BorderColor,
    /// Anisotropic filtering up to this ratio; needs `Features::sampler_anisotropy`.
    pub max_anisotropy: Option<f32>,
}

/// The samplers of `Config::new`: 0 takes the nearest texel of the nearest mip and 1 filters
/// bilinearly within the nearest mip, both clamped to a white border; 2 filters trilinearly
/// and repeats, for mipmapped textures.
pub const DEFAULT_SAMPLERS: [SamplerDesc; 3] = [
    SamplerDesc {
        mag_filter: Filter::NEAREST,
        min_filter: Filter::NEAREST,
        mipmap_mode: SamplerMipmapMode::NEAREST,
        address_mode: SamplerAddressMode::CLAMP_TO_BORDER,
        border_color: BorderColor::FLOAT_OPAQUE_WHITE,
        max_anisotropy: None,
    },
    SamplerDesc {
        mag_filter: Filter::LINEAR,
        min_filter: Filter::LINEAR,
        mipmap_mode: SamplerMipmapMode::NEAREST,
        address_mode: SamplerAddressMode::CLAMP_TO_BORDER,
        border_color: BorderColor::FLOAT_OPAQUE_WHITE,
        max_anisotropy: None,
    },
    SamplerDesc {
        mag_filter: Filter::LINEAR,
        min_filter: Filter::LINEAR,
        mipmap_mode: SamplerMipmapMode::LINEAR,
        address_mode: SamplerAddressMode::REPEAT,
        border_color: BorderColor::FLOAT_TRANSPARENT_BLACK,
        max_anisotropy: None,
    },
];

impl SamplerDesc {
    fn create(&self) -> Result<vk::Sampler> {
        let mut info = vk::SamplerCreateInfo::default()
            .mag_filter(self.mag_filter)
            .min_filter(self.min_filter)
            .mipmap_mode(self.mipmap_mode)
            .address_mode_u(self.address_mode)
            .address_mode_v(self.address_mode)
            .address_mode_w(self.address_mode)
            .border_color(self.border_color)
            .max_lod(vk::LOD_CLAMP_NONE);
        if let Some(ratio) = self.max_anisotropy {
            assert!(
                Ctx::features().sampler_anisotropy,
                "anisotropic samplers need `Features::sampler_anisotropy`"
            );
            info = info.anisotropy_enable(true).max_anisotropy(ratio);
        }
        Ok(unsafe { Ctx::device().create_sampler(&info, None) }?)
    }
}

static ON_DROP: OnceLock<fn(BindlessHandle)> = OnceLock::new();

/// Sets the function called with every bound handle whose view is destroyed: each handle of an
/// image (its whole view's, then each mip's) and of a swapchain's images, once no frame can
/// reach it any more (`epoch`), so whoever picked the indices can hand them out again. It can
/// be set once.
pub fn on_drop(hook: fn(BindlessHandle)) -> Result<()> {
    ON_DROP
        .set(hook)
        .map_err(|_| Error::message("the bindless drop hook is already set"))
}

/// Hands `handle` to the hook unless it is unbound, right before its view is destroyed.
pub(crate) fn release(handle: BindlessHandle) {
    if handle != BindlessHandle::none()
        && let Some(hook) = ON_DROP.get()
    {
        hook(handle);
    }
}

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

    /// Only `index` of the sampled set.
    pub fn sampled(index: u32) -> Self {
        Self {
            descriptor_index_set0: index,
            ..Self::none()
        }
    }

    /// Only `index` of the storage set.
    pub fn storage(index: u32) -> Self {
        Self {
            descriptor_index_set1: index,
            ..Self::none()
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

/// Writes each image into the slots of the handle at the same position: the sampled set at
/// `descriptor_index_set0` and the storage set at `descriptor_index_set1`, skipping
/// `NULL_HANDLE`s, all in one `vkUpdateDescriptorSets`. The caller picks every index; a slot may
/// only be rewritten while no pending command buffer uses it. Panics if an image lacks the
/// usage of a set it is written to, or an index is outside its set.
#[validation_trace]
pub fn bind<'a, F: Format, U: ImageUsage, D: Dim>(
    images: impl IntoIterator<Item = ImageView<'a, F, U, D>>,
    handles: impl IntoIterator<Item = BindlessHandle>,
) {
    let bindless = Bindless::get();
    let mut infos = Vec::new();
    for (image, handle) in images.into_iter().zip(handles) {
        let indices = [handle.descriptor_index_set0, handle.descriptor_index_set1];
        let usages = [vk::ImageUsageFlags::SAMPLED, vk::ImageUsageFlags::STORAGE];
        for (set, (index, usage)) in indices.into_iter().zip(usages).enumerate() {
            if index == NULL_HANDLE {
                continue;
            }
            let flags = image.usage().flags();
            assert!(
                flags.contains(usage),
                "an image with usage {flags:?} can't be bound in the {usage:?} set"
            );
            assert!(
                index < bindless.counts[set],
                "bindless index {index} is outside set {set} of {} descriptors",
                bindless.counts[set]
            );
            let info = vk::DescriptorImageInfo {
                image_layout: vk::ImageLayout::GENERAL,
                image_view: image.view,
                ..Default::default()
            };
            infos.push((set, index, info));
        }
    }
    if infos.is_empty() {
        return;
    }
    let writes: Vec<_> = infos
        .iter()
        .map(|(set, index, info)| {
            let (binding, ty) = match set {
                0 => (SAMPLED_IMAGES_BINDING, vk::DescriptorType::SAMPLED_IMAGE),
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

impl Bindless {
    fn get() -> &'static Self {
        BINDLESS
            .get()
            .expect("bindless resources have not been initialized")
    }
    pub(crate) fn layout() -> vk::PipelineLayout {
        Self::get().layout
    }
    pub(crate) fn init(samplers: &[SamplerDesc]) -> Result<()> {
        assert!(!samplers.is_empty(), "shaders sample with sampler 0 by default");
        let mut layouts = [vk::DescriptorSetLayout::default(); 2];
        let samplers = samplers
            .iter()
            .map(SamplerDesc::create)
            .collect::<Result<Vec<_>>>()?;
        let descriptor_binding_flags = [
            vk::DescriptorBindingFlags::empty(),
            vk::DescriptorBindingFlags::PARTIALLY_BOUND_EXT
                | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT_EXT
                | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND_EXT
                | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING,
        ];
        // The arrays are update-after-bind and used by every stage, so the smaller of those
        // limits bounds them (the plain limits may be smaller than the update-after-bind ones).
        let limits = &Ctx::physical_device().vk12_properties;
        let desc_counts = [
            limits
                .max_descriptor_set_update_after_bind_sampled_images
                .min(limits.max_per_stage_descriptor_update_after_bind_sampled_images)
                .min(10000),
            limits
                .max_descriptor_set_update_after_bind_storage_images
                .min(limits.max_per_stage_descriptor_update_after_bind_storage_images)
                .min(10000),
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
                binding: SAMPLED_IMAGES_BINDING,
                descriptor_count: desc_counts[0],
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
            .push(&mut ext_flags);
        layouts[0] = unsafe { Ctx::device().create_descriptor_set_layout(&layout_info, None) }?;

        let descriptor_binding_flags = [vk::DescriptorBindingFlags::PARTIALLY_BOUND_EXT
            | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT_EXT
            | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND_EXT
            | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING];

        let bindings = [vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_count: desc_counts[1],
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            stage_flags: vk::ShaderStageFlags::ALL,
            ..Default::default()
        }];
        let mut ext_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT::default()
            .binding_flags(&descriptor_binding_flags);
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings)
            .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL_EXT)
            .push(&mut ext_flags);
        layouts[1] = unsafe { Ctx::device().create_descriptor_set_layout(&layout_info, None) }?;

        // Every pass pushes this many bytes.
        let size = MAX_PUSH_CONSTANTS_SIZE as u32;
        assert!(Ctx::device_properties().limits.max_push_constants_size >= size);
        let ranges = [vk::PushConstantRange {
            offset: 0,
            size,
            stage_flags: vk::ShaderStageFlags::ALL,
        }];
        let pipline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(&ranges);
        let layout = unsafe { Ctx::device().create_pipeline_layout(&pipline_layout_info, None)? };

        let pool_sizes = [
            vk::DescriptorPoolSize {
                descriptor_count: Ctx::device_properties()
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
                descriptor_count: Ctx::device_properties()
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

        let variable_counts = desc_counts;
        let mut alloc_info = vk::DescriptorSetVariableDescriptorCountAllocateInfo::default()
            .descriptor_counts(&variable_counts);
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts)
            .push(&mut alloc_info);
        let sets = unsafe { Ctx::device().allocate_descriptor_sets(&allocate_info) }?
            .try_into()
            .expect("allocated descriptor set count matches the two layouts");

        BINDLESS
            .set(Self {
                counts: desc_counts,
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
        if Ctx::features().ray_tracing_pipeline {
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
            // The pool isn't created with the free flag; destroying it frees its sets.
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

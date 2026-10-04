//! Global bindless descriptor sets: immutable samplers, image registration, handles
use std::sync::{
    Mutex, OnceLock,
    atomic::{AtomicU32, Ordering},
};

use crate::error::{Error, Result};
use ash::vk::{self, BorderColor, SamplerAddressMode, SamplerMipmapMode};
use bytemuck::{Pod, Zeroable};

use crate::{
    image::{
        format::Format,
        slice::ImageView,
        usage::{BindlessImageUsageSet, ImageUsage},
    },
    state::Ctx,
};

/// Hands out descriptor indices for the two bindless image arrays. Released indices are
/// handed out again before new ones.
#[derive(Debug)]
pub(crate) struct BindlessCounters {
    /// Next never-used index in set 0 (sampled images).
    num_textures: AtomicU32,
    /// Next never-used index in set 1 (storage images).
    num_images: AtomicU32,
    /// Number of descriptors allocated for each of the two sets.
    capacity: [u32; 2],
    /// Released indices of each set.
    free: [Mutex<Vec<u32>>; 2],
}

impl BindlessCounters {
    pub(crate) fn new(capacity: [u32; 2]) -> Self {
        Self {
            num_textures: AtomicU32::new(0),
            num_images: AtomicU32::new(0),
            capacity,
            free: Default::default(),
        }
    }

    fn next(&self, set: usize, kind: &str) -> Result<u32> {
        if let Some(index) = self.free[set].lock().ok().and_then(|mut free| free.pop()) {
            return Ok(index);
        }
        let counter = [&self.num_textures, &self.num_images][set];
        let capacity = self.capacity[set];
        counter
            .try_update(Ordering::AcqRel, Ordering::Acquire, |next| {
                (next < capacity).then_some(next + 1)
            })
            .map_err(|_| Error::message(format!("all {capacity} bindless {kind} slots are in use")))
    }

    /// Reserves the indices an image with the given usage needs; unused sets get `NULL_HANDLE`.
    pub(crate) fn alloc(&self, set: BindlessImageUsageSet) -> Result<BindlessHandle> {
        let (sampled, storage) = match set {
            BindlessImageUsageSet::None => (false, false),
            BindlessImageUsageSet::SampledImage => (true, false),
            BindlessImageUsageSet::StorageImage => (false, true),
            BindlessImageUsageSet::Both => (true, true),
        };
        let mut handle = BindlessHandle::none();
        if storage {
            handle.descriptor_index_set1 = self.next(1, "storage image")?;
        }
        if sampled {
            handle.descriptor_index_set0 = match self.next(0, "sampled image") {
                Ok(index) => index,
                Err(err) => {
                    self.release(handle);
                    return Err(err);
                }
            };
        }
        Ok(handle)
    }

    /// Makes the indices of `handle` available again. The caller must be done using them.
    pub(crate) fn release(&self, handle: BindlessHandle) {
        let indices = [handle.descriptor_index_set0, handle.descriptor_index_set1];
        for (free, index) in self.free.iter().zip(indices) {
            if index != NULL_HANDLE
                && let Ok(mut free) = free.lock()
            {
                free.push(index);
            }
        }
    }
}

#[derive(Debug)]
pub struct Bindless {
    counters: BindlessCounters,
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
                counters: BindlessCounters::new(desc_counts),
                num_samplers: samplers.len() as u32,
                layout,
                layouts,
                sets,
                pool,
            })
            .map_err(|_| Error::message("bindless resources were already initialized"))?;
        Ok(())
    }

    pub(crate) fn push<F: Format, U: ImageUsage>(image: ImageView<F, U>) -> Result<BindlessHandle> {
        let handle = Self::get().counters.alloc(U::SET)?;
        Self::write_image(image, handle);
        Ok(handle)
    }

    /// Returns the slots of a destroyed image. Its descriptors stay stale until the slots
    /// are handed out again, which is fine as long as no shader indexes them.
    pub(crate) fn release(handle: BindlessHandle) {
        if let Some(bindless) = BINDLESS.get() {
            bindless.counters.release(handle);
        }
    }

    pub(crate) fn write_image<F: Format, U: ImageUsage>(
        image: ImageView<F, U>,
        handle: BindlessHandle,
    ) {
        let image_info = [vk::DescriptorImageInfo {
            image_layout: vk::ImageLayout::GENERAL,
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
                    .dst_binding(Self::get().num_samplers)
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
    use BindlessImageUsageSet as Set;

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

    #[test]
    fn each_set_counts_independently() {
        let counters = BindlessCounters::new([8, 8]);
        let sampled = counters.alloc(Set::SampledImage).unwrap();
        assert_eq!(
            (sampled.descriptor_index_set0, sampled.descriptor_index_set1),
            (0, NULL_HANDLE)
        );

        let storage = counters.alloc(Set::StorageImage).unwrap();
        assert_eq!(
            (storage.descriptor_index_set0, storage.descriptor_index_set1),
            (NULL_HANDLE, 0)
        );

        let sampled = counters.alloc(Set::SampledImage).unwrap();
        assert_eq!(sampled.descriptor_index_set0, 1);

        // `Both` takes the next slot of each set; the two indices need not be equal.
        let both = counters.alloc(Set::Both).unwrap();
        assert_eq!(
            (both.descriptor_index_set0, both.descriptor_index_set1),
            (2, 1)
        );
    }

    #[test]
    fn unregistered_images_get_the_null_handle_and_use_no_slot() {
        let counters = BindlessCounters::new([1, 1]);
        for _ in 0..10 {
            assert_eq!(counters.alloc(Set::None).unwrap(), BindlessHandle::none());
        }
        assert!(counters.alloc(Set::Both).is_ok());
    }

    #[test]
    fn allocation_fails_once_a_set_is_full() {
        let counters = BindlessCounters::new([2, 1]);
        assert!(counters.alloc(Set::StorageImage).is_ok());
        assert!(counters.alloc(Set::StorageImage).is_err());
        // The sampled set still has room, and stays usable after the failure.
        assert_eq!(
            counters
                .alloc(Set::SampledImage)
                .unwrap()
                .descriptor_index_set0,
            0
        );
        assert_eq!(
            counters
                .alloc(Set::SampledImage)
                .unwrap()
                .descriptor_index_set0,
            1
        );
        assert!(counters.alloc(Set::SampledImage).is_err());
        assert!(counters.alloc(Set::Both).is_err());
    }

    #[test]
    fn released_indices_are_reused_before_new_ones() {
        let counters = BindlessCounters::new([2, 2]);
        let first = counters.alloc(Set::Both).unwrap();
        let second = counters.alloc(Set::Both).unwrap();
        assert!(counters.alloc(Set::SampledImage).is_err());

        counters.release(first);
        // A null handle releases nothing.
        counters.release(BindlessHandle::none());
        assert_eq!(counters.alloc(Set::Both).unwrap(), first);
        assert!(counters.alloc(Set::StorageImage).is_err());

        counters.release(second);
        assert_eq!(
            counters.alloc(Set::StorageImage).unwrap().descriptor_index_set1,
            second.descriptor_index_set1
        );
    }

    /// A failed `Both` allocation must not leak the storage index it already took.
    #[test]
    fn failed_allocation_returns_its_partial_indices() {
        let counters = BindlessCounters::new([0, 1]);
        assert!(counters.alloc(Set::Both).is_err());
        assert!(counters.alloc(Set::StorageImage).is_ok());
    }

    #[test]
    fn concurrent_allocations_hand_out_unique_indices() {
        let counters = BindlessCounters::new([400, 400]);
        let mut indices: Vec<u32> = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..4)
                .map(|_| {
                    scope.spawn(|| {
                        (0..100)
                            .map(|_| {
                                counters
                                    .alloc(Set::SampledImage)
                                    .unwrap()
                                    .descriptor_index_set0
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            workers
                .into_iter()
                .flat_map(|w| w.join().unwrap())
                .collect()
        });
        indices.sort_unstable();
        assert_eq!(indices, (0..400).collect::<Vec<_>>());
    }
}

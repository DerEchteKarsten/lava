//! Global bindless descriptor sets: the application's immutable samplers (`SamplerDesc`, anisotropic with `sampler-anisotropy`), image descriptors at slots lava allocates (first fit, freed with their views) or in ranges the application reserves (`reserve_sampled`), handles
use std::{
    ops::Range,
    sync::{Mutex, OnceLock},
};

use crate::error::{Error, Result};
use crate::vk::{self, BorderColor, Filter, SamplerAddressMode, SamplerMipmapMode};
use bytemuck::{Pod, Zeroable};
use lava_macros::validation_trace;

use crate::{
    image::{Dim, format::Format, slice::ImageView, usage::ImageUsage},
    lifetime,
    pipeline::MAX_PUSH_CONSTANTS_SIZE,
    queue::{Kind, lock},
    state::Ctx,
};

#[derive(Debug)]
pub struct Bindless {
    /// Of the sampled (set 0) and storage (set 1) image arrays.
    counts: [u32; 2],
    free: [Mutex<Ranges>; 2],
    /// The live ranges of `reserve_sampled`/`reserve_storage`, which `bind` writes.
    reserved: [Mutex<Vec<Range<u32>>>; 2],
    /// Held for `vkUpdateDescriptorSets`, which needs its sets externally synchronised.
    writes: Mutex<()>,
    layout: vk::PipelineLayout,
    layouts: [vk::DescriptorSetLayout; 2],
    sets: [vk::DescriptorSet; 2],
    pool: vk::DescriptorPool,
}

static BINDLESS: OnceLock<Bindless> = OnceLock::new();

/// After the samplers' array (binding 0).
const SAMPLED_IMAGES_BINDING: u32 = 1;

/// The most descriptors of each set, below the device's limits.
const MAX_DESCRIPTORS: u32 = 1 << 15;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Set {
    Sampled = 0,
    Storage = 1,
}

impl Set {
    pub(crate) fn index_mut(self, handle: &mut BindlessHandle) -> &mut u32 {
        match self {
            Set::Sampled => &mut handle.descriptor_index_set0,
            Set::Storage => &mut handle.descriptor_index_set1,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Set::Sampled => "sampled",
            Set::Storage => "storage",
        }
    }

    /// The handle of `index` in this set alone.
    pub(crate) fn handle(self, index: u32) -> BindlessHandle {
        match self {
            Set::Sampled => BindlessHandle::sampled(index),
            Set::Storage => BindlessHandle::storage(index),
        }
    }
}

/// The free slots of a set: sorted, disjoint ranges, never two adjacent ones.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct Ranges {
    free: Vec<Range<u32>>,
}

impl Ranges {
    /// The first `n` contiguous free slots (first fit), which are taken.
    pub(crate) fn take(&mut self, n: u32) -> Option<u32> {
        assert!(n > 0, "a slot range holds at least one slot");
        let i = self.free.iter().position(|range| range.len() >= n as usize)?;
        let start = self.free[i].start;
        self.free[i].start += n;
        if self.free[i].is_empty() {
            self.free.remove(i);
        }
        Some(start)
    }

    /// Frees `range`, merging it with its neighbours. Panics if a slot of it is free already.
    pub(crate) fn give(&mut self, range: Range<u32>) {
        if range.is_empty() {
            return;
        }
        let i = self.free.partition_point(|free| free.start < range.start);
        let overlaps = |free: Option<&Range<u32>>| {
            free.is_some_and(|free| free.start < range.end && range.start < free.end)
        };
        assert!(
            !overlaps(i.checked_sub(1).and_then(|i| self.free.get(i))) && !overlaps(self.free.get(i)),
            "bindless slots {range:?} are freed twice"
        );
        let merges_left = i > 0 && self.free[i - 1].end == range.start;
        let merges_right = self.free.get(i).is_some_and(|next| next.start == range.end);
        match (merges_left, merges_right) {
            (true, true) => {
                self.free[i - 1].end = self.free[i].end;
                self.free.remove(i);
            }
            (true, false) => self.free[i - 1].end = range.end,
            (false, true) => self.free[i].start = range.start,
            (false, false) => self.free.insert(i, range),
        }
    }
}

pub(crate) fn allocate(set: Set, n: u32) -> Result<u32> {
    let bindless = Bindless::get();
    lock(&bindless.free[set as usize]).take(n).ok_or(Error::OutOfBindlessSlots {
        set: set.name(),
        count: n,
    })
}

/// Frees slots no frame reaches any more.
pub(crate) fn free(set: Set, range: Range<u32>) {
    lock(&Bindless::get().free[set as usize]).give(range);
}

/// Frees the slots of a destroyed view's handle.
pub(crate) fn free_handle(handle: BindlessHandle) {
    for (set, index) in [
        (Set::Sampled, handle.descriptor_index_set0),
        (Set::Storage, handle.descriptor_index_set1),
    ] {
        if index != NULL_HANDLE {
            free(set, index..index + 1);
        }
    }
}

/// Slots of one set the application writes with [`bind`], made by [`reserve_sampled`] and
/// [`reserve_storage`]. Dropping it frees them once no submission reaches them, and [`bind`]
/// refuses them right away.
#[derive(Debug)]
#[must_use]
pub struct SlotRange {
    set: Set,
    start: u32,
    len: u32,
}

impl SlotRange {
    pub fn start(&self) -> u32 {
        self.start
    }

    pub fn len(&self) -> u32 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn indices(&self) -> Range<u32> {
        self.start..self.start + self.len
    }

    /// The handle of its `i`-th slot, in its set alone. Panics past its end.
    pub fn handle(&self, i: u32) -> BindlessHandle {
        assert!(i < self.len, "slot {i} of a range of {}", self.len);
        self.set.handle(self.start + i)
    }
}

impl Drop for SlotRange {
    fn drop(&mut self) {
        let (set, range) = (self.set, self.indices());
        let mut reserved = lock(&Bindless::get().reserved[set as usize]);
        if let Some(i) = reserved.iter().position(|r| *r == range) {
            reserved.swap_remove(i);
        }
        lifetime::retire(move || free(set, range));
    }
}

fn reserve(set: Set, n: u32) -> Result<SlotRange> {
    let start = allocate(set, n)?;
    lock(&Bindless::get().reserved[set as usize]).push(start..start + n);
    Ok(SlotRange { set, start, len: n })
}

/// The first `n` contiguous free slots of the sampled set, for [`bind`].
pub fn reserve_sampled(n: u32) -> Result<SlotRange> {
    reserve(Set::Sampled, n)
}

/// Like [`reserve_sampled`], in the storage set.
pub fn reserve_storage(n: u32) -> Result<SlotRange> {
    reserve(Set::Storage, n)
}

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
    /// Anisotropic filtering up to this ratio.
    #[cfg(feature = "sampler-anisotropy")]
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
        #[cfg(feature = "sampler-anisotropy")]
        max_anisotropy: None,
    },
    SamplerDesc {
        mag_filter: Filter::LINEAR,
        min_filter: Filter::LINEAR,
        mipmap_mode: SamplerMipmapMode::NEAREST,
        address_mode: SamplerAddressMode::CLAMP_TO_BORDER,
        border_color: BorderColor::FLOAT_OPAQUE_WHITE,
        #[cfg(feature = "sampler-anisotropy")]
        max_anisotropy: None,
    },
    SamplerDesc {
        mag_filter: Filter::LINEAR,
        min_filter: Filter::LINEAR,
        mipmap_mode: SamplerMipmapMode::LINEAR,
        address_mode: SamplerAddressMode::REPEAT,
        border_color: BorderColor::FLOAT_TRANSPARENT_BLACK,
        #[cfg(feature = "sampler-anisotropy")]
        max_anisotropy: None,
    },
];

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

/// Valid sampled indices are `0..max_sampled_images()`.
pub fn max_sampled_images() -> u32 {
    Bindless::get().counts[0]
}

/// Valid storage indices are `0..max_storage_images()`.
pub fn max_storage_images() -> u32 {
    Bindless::get().counts[1]
}

/// Writes each image into the slots of the handle at the same position, skipping
/// `NULL_HANDLE`s, in one `vkUpdateDescriptorSets`; a slot may only be rewritten while no
/// pending command buffer uses it. Panics unless every index lies in a live [`SlotRange`] and
/// every view has one aspect and the usage of its set.
#[validation_trace]
pub fn bind<'a, F: Format, U: ImageUsage, D: Dim>(
    images: impl IntoIterator<Item = ImageView<'a, F, U, D>>,
    handles: impl IntoIterator<Item = BindlessHandle>,
) {
    let bindless = Bindless::get();
    let reserved = [&bindless.reserved[0], &bindless.reserved[1]].map(lock);
    let handles = handles.into_iter().inspect(|handle| {
        let indices = [handle.descriptor_index_set0, handle.descriptor_index_set1];
        for (set, index) in indices.into_iter().enumerate() {
            assert!(
                index == NULL_HANDLE || reserved[set].iter().any(|r| r.contains(&index)),
                "bindless index {index} of the {} set lies in no `SlotRange`: `bind` writes only \
                 reserved slots (`reserve_sampled`, `reserve_storage`)",
                [Set::Sampled, Set::Storage][set].name()
            );
        }
    });
    let views = images.into_iter().map(|image| {
        let aspects = image.format().aspects();
        assert!(
            crate::image::slice::single_aspect(aspects) == aspects,
            "a view of {:?} has aspects {aspects:?}, but shaders read one: bind a depth-stencil \
             image through `DepthSplit` views",
            image.format().vk_format()
        );
        (image.view, image.usage().flags())
    });
    bind_raw(views, handles);
}

/// [`bind`] for raw views, which the caller vouches show one aspect.
pub(crate) fn bind_raw(
    views: impl IntoIterator<Item = (vk::ImageView, vk::ImageUsageFlags)>,
    handles: impl IntoIterator<Item = BindlessHandle>,
) {
    let bindless = Bindless::get();
    let mut infos = Vec::new();
    for ((view, flags), handle) in views.into_iter().zip(handles) {
        let indices = [handle.descriptor_index_set0, handle.descriptor_index_set1];
        let usages = [vk::ImageUsageFlags::SAMPLED, vk::ImageUsageFlags::STORAGE];
        for (set, (index, usage)) in indices.into_iter().zip(usages).enumerate() {
            if index == NULL_HANDLE {
                continue;
            }
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
                image_view: view,
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
    let _writing = lock(&bindless.writes);
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
            .map(|desc| -> Result<vk::Sampler> {
                #[allow(unused_mut)]
                let mut info = vk::SamplerCreateInfo::default()
                    .mag_filter(desc.mag_filter)
                    .min_filter(desc.min_filter)
                    .mipmap_mode(desc.mipmap_mode)
                    .address_mode_u(desc.address_mode)
                    .address_mode_v(desc.address_mode)
                    .address_mode_w(desc.address_mode)
                    .border_color(desc.border_color)
                    .max_lod(vk::LOD_CLAMP_NONE);
                #[cfg(feature = "sampler-anisotropy")]
                if let Some(ratio) = desc.max_anisotropy {
                    use crate::features::Feature;
                    crate::features::gate::SamplerAnisotropy::check()?;
                    info = info.anisotropy_enable(true).max_anisotropy(ratio);
                }
                Ok(unsafe { Ctx::device().create_sampler(&info, None) }?)
            })
            .collect::<Result<Vec<_>>>()?;
        let descriptor_binding_flags = [
            vk::DescriptorBindingFlags::empty(),
            vk::DescriptorBindingFlags::PARTIALLY_BOUND_EXT
                | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT_EXT
                | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND_EXT
                | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING,
        ];
        // Update-after-bind arrays used by every stage, so both of those limits bound them.
        let limits = &Ctx::physical_device().vk12_properties;
        let desc_counts = [
            limits
                .max_descriptor_set_update_after_bind_sampled_images
                .min(limits.max_per_stage_descriptor_update_after_bind_sampled_images)
                .min(MAX_DESCRIPTORS),
            limits
                .max_descriptor_set_update_after_bind_storage_images
                .min(limits.max_per_stage_descriptor_update_after_bind_storage_images)
                .min(MAX_DESCRIPTORS),
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
            .push_next(&mut ext_flags);
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
            .push_next(&mut ext_flags);
        layouts[1] = unsafe { Ctx::device().create_descriptor_set_layout(&layout_info, None) }?;

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

        // The variable counts above; the plain limits don't apply to an update-after-bind pool.
        let pool_sizes = [
            vk::DescriptorPoolSize {
                descriptor_count: desc_counts[0],
                ty: vk::DescriptorType::SAMPLED_IMAGE,
            },
            vk::DescriptorPoolSize {
                descriptor_count: samplers.len() as u32,
                ty: vk::DescriptorType::SAMPLER,
            },
            vk::DescriptorPoolSize {
                descriptor_count: desc_counts[1],
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
            .push_next(&mut alloc_info);
        let sets = unsafe { Ctx::device().allocate_descriptor_sets(&allocate_info) }?
            .try_into()
            .expect("allocated descriptor set count matches the two layouts");

        BINDLESS
            .set(Self {
                counts: desc_counts,
                free: desc_counts.map(|count| {
                    let free = if count == 0 { Vec::new() } else { vec![0..count] };
                    Mutex::new(Ranges { free })
                }),
                reserved: Default::default(),
                writes: Mutex::new(()),
                layout,
                layouts,
                sets,
                pool,
            })
            .map_err(|_| Error::message("bindless resources were already initialized"))?;
        Ok(())
    }

    /// Binds the sets at the bind points a family of `kind` has, none on a transfer family.
    pub(crate) fn bind(cmd: &vk::CommandBuffer, kind: Kind) {
        let s = Self::get();
        if kind == Kind::Transfer {
            return;
        }
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
        if <crate::features::gate::RayTracingPipeline as crate::features::On>::on() {
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
        if kind != Kind::Graphics {
            return;
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

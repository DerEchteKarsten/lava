//! Global bindless descriptor heaps (`VK_EXT_descriptor_heap`): image descriptors at slots lava allocates (first fit, freed with their images) or in ranges the application reserves (`reserve_sampled`, rewritten in submission order by `bind`), runtime `Sampler`s in the sampler heap, and the mappings that point shaders' set and binding declarations at both heaps
use std::{
    collections::BTreeMap,
    ops::Range,
    ptr::NonNull,
    sync::{Mutex, OnceLock},
};

use crate::error::{Error, Result};
use crate::vk::{self, BorderColor, CompareOp, Filter, SamplerAddressMode, SamplerMipmapMode};
use bytemuck::{Pod, Zeroable};
use lava_macros::validation_trace;
use smallvec::SmallVec;

use crate::{
    buffer::Buffer,
    image::{Dim, format::Format, slice::{ImageView, single_aspect}, usage::ImageUsage},
    lifetime,
    queue::{Kind, lock},
    state::{Ctx, Functions},
};

#[derive(Debug)]
pub struct Bindless {
    /// The resource heap: sampled images, then storage images, then the reserved range.
    images: HeapLayout,
    samplers: HeapLayout,
    free: [Mutex<Ranges>; 2],
    /// The live ranges of `reserve_sampled`/`reserve_storage`, which `bind` writes.
    reserved: [Mutex<Vec<Range<u32>>>; 2],
    free_samplers: Mutex<Ranges>,
    resource_heap: Heap,
    sampler_heap: Heap,
    /// Descriptors `bind` wrote for reserved slots, by byte offset in the resource heap, which
    /// the next submission of a timeline with shaders copies there before its commands.
    queued: Mutex<Queued>,
    /// Boxed, so `mapping_info` can point at them; read only through it.
    #[allow(dead_code)]
    mappings: Box<[vk::DescriptorSetAndBindingMappingEXT<'static>; 3]>,
    mapping_info: vk::ShaderDescriptorSetAndBindingMappingInfoEXT<'static>,
}

static BINDLESS: OnceLock<Bindless> = OnceLock::new();

/// The most image descriptors of each region.
const MAX_DESCRIPTORS: u32 = 1 << 15;
/// The most samplers.
const MAX_SAMPLERS: u32 = 1 << 12;

/// Descriptors waiting for a submission to copy them into the resource heap, by byte offset.
pub(crate) type Queued = BTreeMap<u64, SmallVec<[u8; 64]>>;

/// The most bytes one `vkCmdUpdateMemoryKHR` writes.
const MAX_UPDATE: usize = 65536;

/// Where a heap keeps its descriptors: `regions` arrays of `count` descriptors `stride` bytes
/// apart from offset 0, then the implementation's reserved range from `reserved_offset` to
/// `size`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HeapLayout {
    pub(crate) stride: u64,
    pub(crate) count: u32,
    pub(crate) reserved_offset: u64,
    pub(crate) size: u64,
}

impl HeapLayout {
    /// `regions` arrays of at most `max_count` descriptors of `size` bytes each, aligned to
    /// `alignment`, then `reserved` bytes at a multiple of `reserved_alignment`, all within
    /// `max_size`.
    pub(crate) fn new(
        size: u64,
        alignment: u64,
        regions: u32,
        max_count: u32,
        reserved: u64,
        reserved_alignment: u64,
        max_size: u64,
    ) -> Self {
        let stride = size.next_multiple_of(alignment.max(1));
        let reserved_alignment = reserved_alignment.max(1);
        // Room for the reserved range and the padding before it.
        let room = max_size.saturating_sub(reserved + reserved_alignment - 1);
        let count = (room / (regions as u64 * stride)).min(max_count as u64) as u32;
        let reserved_offset =
            (regions as u64 * count as u64 * stride).next_multiple_of(reserved_alignment);
        Self {
            stride,
            count,
            reserved_offset,
            size: reserved_offset + reserved,
        }
    }

    /// The byte offset of descriptor `index` of region `region`.
    pub(crate) fn offset(&self, region: u32, index: u32) -> u64 {
        (region as u64 * self.count as u64 + index as u64) * self.stride
    }
}

/// A heap's mapped buffer and how command buffers bind it.
#[derive(Debug)]
struct Heap {
    /// Taken by `Bindless::destroy`.
    buffer: Mutex<Option<Buffer<u8>>>,
    mapped: NonNull<u8>,
    bind: vk::BindHeapInfoEXT<'static>,
}

// SAFETY: `mapped` is written only at slots one owner allocated, or through `vkCmdUpdateMemoryKHR`.
unsafe impl Send for Heap {}
unsafe impl Sync for Heap {}

impl Heap {
    fn new(layout: &HeapLayout, alignment: u64) -> Result<Self> {
        let usage = vk::BUFFER_USAGE_DESCRIPTOR_HEAP_EXT;
        let buffer = Buffer::with_flags(usage, true, layout.size.max(1), Some(alignment as u32))?;
        let mapped = NonNull::new(buffer.host_ptr()).expect("a mapping isn't null");
        // Slots never written read as zeroes rather than whatever the memory held.
        unsafe { mapped.write_bytes(0, layout.size as usize) };
        let bind = vk::BindHeapInfoEXT::default()
            .heap_range(vk::DeviceAddressRangeKHR {
                address: buffer.address(),
                size: layout.size,
            })
            .reserved_range_offset(layout.reserved_offset)
            .reserved_range_size(layout.size - layout.reserved_offset);
        Ok(Heap {
            buffer: Mutex::new(Some(buffer)),
            mapped,
            bind,
        })
    }

    fn host_range(&self, offset: u64, size: u64) -> vk::HostAddressRangeEXT {
        vk::HostAddressRangeEXT {
            address: unsafe { self.mapped.as_ptr().add(offset as usize) }.cast(),
            size: size as usize,
        }
    }
}

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
    fn all(count: u32) -> Self {
        let free = if count == 0 { Vec::new() } else { vec![0..count] };
        Ranges { free }
    }

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

/// Frees the slots of a destroyed image's handle.
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
        let bindless = Bindless::get();
        let mut reserved = lock(&bindless.reserved[set as usize]);
        if let Some(i) = reserved.iter().position(|r| *r == range) {
            reserved.swap_remove(i);
        }
        drop(reserved);
        // Nothing reads them any more, and a later copy could overwrite the slots' next owner.
        let layout = &bindless.images;
        let bytes = layout.offset(set as u32, range.start)..layout.offset(set as u32, range.end);
        lock(&bindless.queued).retain(|offset, _| !bytes.contains(offset));
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

/// How a [`Sampler`] filters and addresses an image. [`SamplerDesc::new`] (and `Default`)
/// filters linearly within and between mips, repeats, and neither compares nor reduces; each
/// setter changes one setting.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplerDesc {
    mag_filter: Filter,
    min_filter: Filter,
    mipmap_mode: SamplerMipmapMode,
    address_mode: SamplerAddressMode,
    border_color: BorderColor,
    compare: Option<CompareOp>,
    #[cfg(feature = "sampler-minmax")]
    reduction: vk::SamplerReductionMode,
    #[cfg(feature = "sampler-anisotropy")]
    max_anisotropy: Option<f32>,
}

impl SamplerDesc {
    pub const fn new() -> Self {
        Self {
            mag_filter: Filter::LINEAR,
            min_filter: Filter::LINEAR,
            mipmap_mode: SamplerMipmapMode::LINEAR,
            address_mode: SamplerAddressMode::REPEAT,
            border_color: BorderColor::FLOAT_TRANSPARENT_BLACK,
            compare: None,
            #[cfg(feature = "sampler-minmax")]
            reduction: vk::SamplerReductionMode::WEIGHTED_AVERAGE,
            #[cfg(feature = "sampler-anisotropy")]
            max_anisotropy: None,
        }
    }

    /// Both the magnification and the minification filter.
    pub const fn filter(self, filter: Filter) -> Self {
        self.mag_filter(filter).min_filter(filter)
    }

    pub const fn mag_filter(mut self, filter: Filter) -> Self {
        self.mag_filter = filter;
        self
    }

    pub const fn min_filter(mut self, filter: Filter) -> Self {
        self.min_filter = filter;
        self
    }

    /// How it picks between mips.
    pub const fn mipmap_mode(mut self, mode: SamplerMipmapMode) -> Self {
        self.mipmap_mode = mode;
        self
    }

    /// For every coordinate.
    pub const fn address_mode(mut self, mode: SamplerAddressMode) -> Self {
        self.address_mode = mode;
        self
    }

    /// The color outside the image with `CLAMP_TO_BORDER`.
    pub const fn border_color(mut self, color: BorderColor) -> Self {
        self.border_color = color;
        self
    }

    /// A comparison sampler, for `SampleCmp` in shaders: each texel becomes 1 where
    /// `reference op texel` holds and 0 elsewhere before filtering (percentage-closer filtering
    /// of shadow maps).
    pub const fn compare(mut self, op: CompareOp) -> Self {
        self.compare = Some(op);
        self
    }

    /// The minimum or maximum of the texels it filters instead of their weighted average
    /// (depth pyramids); the image's format needs `SAMPLED_IMAGE_FILTER_MINMAX`.
    #[cfg(feature = "sampler-minmax")]
    pub const fn reduction(mut self, mode: vk::SamplerReductionMode) -> Self {
        self.reduction = mode;
        self
    }

    /// Anisotropic filtering up to `ratio`, from 1 to the device's `maxSamplerAnisotropy`.
    #[cfg(feature = "sampler-anisotropy")]
    pub const fn max_anisotropy(mut self, ratio: f32) -> Self {
        self.max_anisotropy = Some(ratio);
        self
    }

    /// Calls `write` with its create info, after checking what it needs of the device.
    fn with_create_info<R>(&self, write: impl FnOnce(&vk::SamplerCreateInfo) -> R) -> Result<R> {
        #[allow(unused_mut)]
        let mut info = vk::SamplerCreateInfo::default()
            .mag_filter(self.mag_filter)
            .min_filter(self.min_filter)
            .mipmap_mode(self.mipmap_mode)
            .address_mode_u(self.address_mode)
            .address_mode_v(self.address_mode)
            .address_mode_w(self.address_mode)
            .border_color(self.border_color)
            .compare_enable(self.compare.is_some())
            .compare_op(self.compare.unwrap_or(CompareOp::NEVER))
            .max_lod(vk::LOD_CLAMP_NONE);
        #[cfg(feature = "sampler-anisotropy")]
        if let Some(ratio) = self.max_anisotropy {
            use crate::features::Feature;
            crate::features::gate::SamplerAnisotropy::check()?;
            let max = Ctx::device_properties().limits.max_sampler_anisotropy;
            if !(1.0..=max).contains(&ratio) {
                return Err(Error::message(format!(
                    "a sampler's anisotropy is 1 to the device's {max}, not {ratio}"
                )));
            }
            info = info.anisotropy_enable(true).max_anisotropy(ratio);
        }
        #[cfg(feature = "sampler-minmax")]
        let mut reduction =
            vk::SamplerReductionModeCreateInfo::default().reduction_mode(self.reduction);
        #[cfg(feature = "sampler-minmax")]
        if self.reduction != vk::SamplerReductionMode::WEIGHTED_AVERAGE {
            use crate::features::Feature;
            crate::features::gate::SamplerMinmax::check()?;
            info = info.push_next(&mut reduction);
        }
        Ok(write(&info))
    }
}

impl Default for SamplerDesc {
    fn default() -> Self {
        Self::new()
    }
}

/// A sampler in lava's sampler heap, which shaders take as a `Sampler` ([`SamplerHandle`]).
/// Dropping it frees its slot once no submission reaches it.
#[derive(Debug)]
pub struct Sampler {
    index: u32,
}

impl Sampler {
    /// Writes `desc` into a free slot of the sampler heap; equal descriptions get slots of
    /// their own.
    #[validation_trace]
    pub fn new(desc: &SamplerDesc) -> Result<Self> {
        let bindless = Bindless::get();
        let index = lock(&bindless.free_samplers).take(1).ok_or(Error::OutOfBindlessSlots {
            set: "sampler",
            count: 1,
        })?;
        // From here on `Drop` frees the slot.
        let sampler = Sampler { index };
        let layout = &bindless.samplers;
        let range = bindless.sampler_heap.host_range(layout.offset(0, index), layout.stride);
        desc.with_create_info(|info| unsafe {
            Functions::descriptor_heap().write_sampler_descriptors(std::slice::from_ref(info), &[range])
        })??;
        Ok(sampler)
    }

    pub fn handle(&self) -> SamplerHandle {
        SamplerHandle { index: self.index }
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        let index = self.index;
        lifetime::retire(move || lock(&Bindless::get().free_samplers).give(index..index + 1));
    }
}

/// A [`Sampler`] as shaders see it (`Sampler` in `bindless.slang`): its slot in the sampler
/// heap.
#[derive(Pod, Zeroable, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct SamplerHandle {
    pub index: u32,
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

/// Valid sampled indices are `0..max_sampled_images()`.
pub fn max_sampled_images() -> u32 {
    Bindless::get().images.count
}

/// Valid storage indices are `0..max_storage_images()`.
pub fn max_storage_images() -> u32 {
    Bindless::get().images.count
}

/// What an image descriptor shows: `mips` of one image through a view of `view_type`, of the
/// format's one aspect, or depth for a combined depth-stencil format.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ShaderImage {
    pub(crate) image: vk::Image,
    pub(crate) format: vk::Format,
    pub(crate) aspects: vk::ImageAspectFlags,
    pub(crate) view_type: vk::ImageViewType,
    pub(crate) mips: std::range::Range<u32>,
    pub(crate) usage: vk::ImageUsageFlags,
}

impl<F: Format, U: ImageUsage, D: Dim> ImageView<'_, F, U, D> {
    pub(crate) fn shader_image(&self) -> ShaderImage {
        ShaderImage {
            image: self.image,
            format: self.format.vk_format(),
            aspects: self.format.aspects(),
            view_type: D::VIEW_TYPE,
            mips: self.mip_range,
            usage: self.usage.flags(),
        }
    }
}

/// Writes each image's descriptor into the slots of the handle at the same position, skipping
/// `NULL_HANDLE`s. Panics unless every index lies in a live [`SlotRange`] and every image has
/// the usage of its set.
///
/// The descriptors reach the heap in submission order: the next submission of a timeline with
/// shaders (graphics or compute) copies them in before its commands, after waiting for every
/// earlier command of its queue, so submissions before it read the old descriptors and those
/// after it the new ones. Batch rewrites, since each such submission waits once. Another
/// timeline's submissions see a rewrite only once they [`wait_for`] the one that copied it, and
/// one of them in flight must not read the slot meanwhile.
///
/// [`wait_for`]: crate::queue::Submission::wait_for
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
    let images = images.into_iter().map(|image| image.shader_image());
    bind_raw(images, handles, true).expect("writing image descriptors failed");
}

/// [`bind`] for any slots: writes the descriptors into the heap right away, or with `queued`
/// leaves them for the next submission (see [`bind`]). Right away is for slots no submission
/// reaches, freshly allocated ones.
pub(crate) fn bind_raw(
    images: impl IntoIterator<Item = ShaderImage>,
    handles: impl IntoIterator<Item = BindlessHandle>,
    queued: bool,
) -> Result<()> {
    let bindless = Bindless::get();
    let layout = &bindless.images;
    let mut targets: SmallVec<[(ShaderImage, Set, u32); 8]> = SmallVec::new();
    for (image, handle) in images.into_iter().zip(handles) {
        let indices = [handle.descriptor_index_set0, handle.descriptor_index_set1];
        let usages = [vk::ImageUsageFlags::SAMPLED, vk::ImageUsageFlags::STORAGE];
        for (set, (index, usage)) in [Set::Sampled, Set::Storage].into_iter().zip(indices.into_iter().zip(usages)) {
            if index == NULL_HANDLE {
                continue;
            }
            assert!(
                image.usage.contains(usage),
                "an image with usage {:?} can't be bound in the {} set",
                image.usage,
                set.name()
            );
            assert!(
                index < layout.count,
                "bindless index {index} is outside the {} set of {} descriptors",
                set.name(),
                layout.count
            );
            targets.push((image, set, index));
        }
    }
    if targets.is_empty() {
        return Ok(());
    }
    let views: SmallVec<[_; 8]> = targets
        .iter()
        .map(|(image, ..)| {
            vk::ImageViewCreateInfo::default()
                .image(image.image)
                .view_type(image.view_type)
                .format(image.format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: single_aspect(image.aspects),
                    base_mip_level: image.mips.start,
                    level_count: image.mips.end - image.mips.start,
                    base_array_layer: 0,
                    layer_count: 1,
                })
        })
        .collect();
    let infos: SmallVec<[_; 8]> = views
        .iter()
        .map(|view| {
            vk::ImageDescriptorInfoEXT::default()
                .p_view(view)
                .layout(vk::ImageLayout::GENERAL)
        })
        .collect();
    let resources: SmallVec<[_; 8]> = targets
        .iter()
        .zip(&infos)
        .map(|((_, set, _), info)| {
            let ty = match set {
                Set::Sampled => vk::DescriptorType::SAMPLED_IMAGE,
                Set::Storage => vk::DescriptorType::STORAGE_IMAGE,
            };
            vk::ResourceDescriptorInfoEXT::default()
                .ty(ty)
                .data((info as *const vk::ImageDescriptorInfoEXT).cast())
        })
        .collect();
    let offset = |set: Set, index: u32| layout.offset(set as u32, index);
    let functions = Functions::descriptor_heap();
    if !queued {
        let ranges: SmallVec<[_; 8]> = (targets.iter())
            .map(|&(_, set, index)| bindless.resource_heap.host_range(offset(set, index), layout.stride))
            .collect();
        unsafe { functions.write_resource_descriptors(&resources, &ranges)? };
        return Ok(());
    }
    let stride = layout.stride as usize;
    let mut staging: SmallVec<[u8; 512]> = smallvec::smallvec![0; targets.len() * stride];
    let base = staging.as_mut_ptr();
    let ranges: SmallVec<[_; 8]> = (0..targets.len())
        .map(|i| vk::HostAddressRangeEXT {
            address: unsafe { base.add(i * stride) }.cast(),
            size: stride,
        })
        .collect();
    unsafe { functions.write_resource_descriptors(&resources, &ranges)? };
    let mut queue = lock(&bindless.queued);
    for (&(_, set, index), descriptor) in targets.iter().zip(staging.chunks(stride)) {
        queue.insert(offset(set, index), SmallVec::from_slice(descriptor));
    }
    Ok(())
}

impl Bindless {
    fn get() -> &'static Self {
        BINDLESS
            .get()
            .expect("bindless resources have not been initialized")
    }

    /// The mappings of `bindless.slang`'s set and binding declarations to the heaps, which every
    /// shader stage gets.
    pub(crate) fn mapping_info() -> &'static vk::ShaderDescriptorSetAndBindingMappingInfoEXT<'static> {
        &Self::get().mapping_info
    }

    pub(crate) fn init() -> Result<()> {
        let p = &Ctx::physical_device().descriptor_heap_properties;
        let images = HeapLayout::new(
            p.image_descriptor_size,
            p.image_descriptor_alignment,
            2,
            MAX_DESCRIPTORS,
            p.min_resource_heap_reserved_range,
            p.image_descriptor_alignment.max(p.buffer_descriptor_alignment),
            p.max_resource_heap_size,
        );
        let samplers = HeapLayout::new(
            p.sampler_descriptor_size,
            p.sampler_descriptor_alignment,
            1,
            MAX_SAMPLERS,
            p.min_sampler_heap_reserved_range,
            p.sampler_descriptor_alignment,
            p.max_sampler_heap_size,
        );
        let resource_heap = Heap::new(&images, p.resource_heap_alignment)?;
        let sampler_heap = Heap::new(&samplers, p.sampler_heap_alignment)?;
        // Each array of `bindless.slang` at a constant offset, indexed by its slot.
        let mapping = |set, binding, offset: u64, stride: u64| {
            let constant_offset = vk::DescriptorMappingSourceConstantOffsetEXT {
                heap_offset: offset.try_into().expect("heap offsets fit 32 bits"),
                heap_array_stride: stride as u32,
                p_embedded_sampler: std::ptr::null(),
                sampler_heap_offset: 0,
                sampler_heap_array_stride: 0,
            };
            vk::DescriptorSetAndBindingMappingEXT::default()
                .descriptor_set(set)
                .first_binding(binding)
                .binding_count(1)
                .resource_mask(vk::SpirvResourceTypeFlagsEXT::ALL)
                .source(vk::DescriptorMappingSourceEXT::HEAP_WITH_CONSTANT_OFFSET)
                .source_data(vk::DescriptorMappingSourceDataEXT { constant_offset })
        };
        let mappings = Box::new([
            mapping(0, 0, 0, samplers.stride),
            mapping(0, 1, images.offset(Set::Sampled as u32, 0), images.stride),
            mapping(1, 0, images.offset(Set::Storage as u32, 0), images.stride),
        ]);
        let mapping_info = vk::ShaderDescriptorSetAndBindingMappingInfoEXT::default()
            .mapping_count(mappings.len() as u32)
            .p_mappings(mappings.as_ptr());
        BINDLESS
            .set(Self {
                free: [Ranges::all(images.count), Ranges::all(images.count)].map(Mutex::new),
                reserved: Default::default(),
                free_samplers: Mutex::new(Ranges::all(samplers.count)),
                images,
                samplers,
                resource_heap,
                sampler_heap,
                queued: Mutex::new(BTreeMap::new()),
                mappings,
                mapping_info,
            })
            .map_err(|_| Error::message("bindless resources were already initialized"))?;
        Ok(())
    }

    /// Binds both heaps, unless `kind` is a transfer family, which runs no shaders.
    pub(crate) fn bind(cmd: vk::CommandBuffer, kind: Kind) {
        if kind == Kind::Transfer {
            return;
        }
        let s = Self::get();
        let functions = Functions::descriptor_heap();
        unsafe {
            functions.cmd_bind_sampler_heap(cmd, &s.sampler_heap.bind);
            functions.cmd_bind_resource_heap(cmd, &s.resource_heap.bind);
        }
    }

    /// Records the copies of the descriptors [`bind`] queued into the resource heap, after
    /// every earlier command of the queue and before every later read of the heap, and returns
    /// them, so a recording that never reaches the GPU can queue them again ([`Self::requeue`]).
    pub(crate) fn flush_queued(cmd: vk::CommandBuffer) -> Queued {
        let s = Self::get();
        let queued = std::mem::take(&mut *lock(&s.queued));
        if queued.is_empty() {
            return queued;
        }
        let device = Ctx::device();
        let barrier = |src_stage, src_access, dst_stage, dst_access| {
            let barrier = [vk::MemoryBarrier2::default()
                .src_stage_mask(src_stage)
                .src_access_mask(src_access)
                .dst_stage_mask(dst_stage)
                .dst_access_mask(dst_access)];
            let info = vk::DependencyInfo::default().memory_barriers(&barrier);
            unsafe { device.cmd_pipeline_barrier2(cmd, &info) };
        };
        use vk::{AccessFlags2 as A, PipelineStageFlags2 as S};
        barrier(S::ALL_COMMANDS, A::NONE, S::TRANSFER, A::TRANSFER_WRITE);
        // Neighbouring descriptors in one update, up to its limit.
        let mut runs: SmallVec<[(u64, Vec<u8>); 4]> = SmallVec::new();
        for (&offset, descriptor) in &queued {
            match runs.last_mut() {
                Some((start, bytes))
                    if *start + bytes.len() as u64 == offset
                        && bytes.len() + descriptor.len() <= MAX_UPDATE =>
                {
                    bytes.extend_from_slice(descriptor)
                }
                _ => runs.push((offset, descriptor.to_vec())),
            }
        }
        let heap = s.resource_heap.bind.heap_range.address;
        let flags = vk::AddressCommandFlagsKHR::UNKNOWN_STORAGE_BUFFER_USAGE
            | vk::AddressCommandFlagsKHR::FULLY_BOUND;
        for (offset, bytes) in &runs {
            let range = vk::DeviceAddressRangeKHR {
                address: heap + offset,
                size: bytes.len() as u64,
            };
            unsafe {
                Functions::device_address_commands().cmd_update_memory(cmd, &range, flags, bytes)
            };
        }
        barrier(S::TRANSFER, A::TRANSFER_WRITE, S::ALL_COMMANDS, vk::ACCESS_2_RESOURCE_HEAP_READ_EXT);
        queued
    }

    /// Queues descriptors again whose submission never reached the GPU, unless `bind` queued a
    /// newer one for their slot meanwhile or their reservation was dropped.
    pub(crate) fn requeue(queued: Queued) {
        let s = Self::get();
        let reserved = [&s.reserved[0], &s.reserved[1]].map(lock);
        let region = s.images.offset(1, 0);
        let live = |offset: u64| {
            let set = (offset / region) as usize;
            let index = ((offset % region) / s.images.stride) as u32;
            reserved[set].iter().any(|range| range.contains(&index))
        };
        let mut queue = lock(&s.queued);
        for (offset, descriptor) in queued.into_iter().filter(|&(offset, _)| live(offset)) {
            queue.entry(offset).or_insert(descriptor);
        }
    }

    /// Hands the heaps' buffers to [`lifetime`], which destroys them with everything else.
    pub(crate) fn destroy() {
        let s = Self::get();
        for heap in [&s.resource_heap, &s.sampler_heap] {
            drop(lock(&heap.buffer).take());
        }
    }
}

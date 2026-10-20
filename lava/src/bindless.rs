//! Global bindless descriptor heaps (`VK_EXT_descriptor_heap`), which shaders index directly: image descriptors at slots lava allocates (first fit, freed with their images) or in ranges the application reserves (`reserve`, rewritten in submission order by `bind`, as a fitted image's slots are by `rebind`), the one-`u32` `ImageHandle<K>` that names them, and runtime `Sampler`s in the sampler heap
use std::{
    collections::BTreeMap,
    marker::PhantomData,
    ops::Range,
    ptr::NonNull,
    sync::Mutex,
};

use crate::error::{Error, Result};
use crate::vk::{self, BorderColor, CompareOp, Filter, SamplerAddressMode, SamplerMipmapMode};
use bytemuck::{Pod, Zeroable};
use lava_macros::validation_trace;
use smallvec::SmallVec;

use crate::{
    buffer::{Buffer, Upload},
    image::{
        Dim,
        format::Format,
        slice::{ImageView, single_aspect},
        usage::ImageUsage,
    },
    global::{Global, Live},
    lifetime,
    queue::{Kind, lock},
    state::{Ctx, Functions},
};

#[derive(Debug)]
pub struct Bindless {
    /// The resource heap: image descriptors, then the reserved range.
    images: HeapLayout,
    samplers: HeapLayout,
    free: Mutex<Ranges>,
    /// The live slot ranges of `reserve`, which `bind` writes.
    reserved: Mutex<Vec<Range<u32>>>,
    /// The slots of images `Image::fit` recreated, whose rewrites are queued like `bind`'s
    /// until the image is dropped.
    rebound: Mutex<Vec<Range<u32>>>,
    free_samplers: Mutex<Ranges>,
    resource_heap: HeapBuffer,
    sampler_heap: HeapBuffer,
    /// Descriptors `bind` wrote for reserved slots, by byte offset in the resource heap, which
    /// the next submission of a timeline with shaders copies there before its commands.
    queued: Mutex<Queued>,
}

static BINDLESS: Global<Bindless> = Global::new();

/// lava's two descriptor heaps, as [`Error::OutOfBindlessSlots`] names them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Heap {
    /// The resource heap's image descriptors.
    Image,
    /// The sampler heap.
    Sampler,
}

impl std::fmt::Display for Heap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Heap::Image => "image",
            Heap::Sampler => "sampler",
        })
    }
}

/// The most image descriptors.
const MAX_IMAGE_SLOTS: u32 = 1 << 16;
/// The most samplers.
const MAX_SAMPLERS: u32 = 1 << 12;

/// Descriptors waiting for a submission to copy them into the resource heap, by byte offset.
pub(crate) type Queued = BTreeMap<u64, SmallVec<[u8; 64]>>;

/// The most bytes one `vkCmdUpdateMemoryKHR` writes.
const MAX_UPDATE: usize = 65536;

/// Where a heap keeps its descriptors: `count` descriptors `stride` bytes apart from offset 0,
/// then the implementation's reserved range from `reserved_offset` to `size`. The stride is what
/// shaders index the heap by: `OpConstantSizeOfEXT` of the descriptor type, its size aligned to
/// its alignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HeapLayout {
    pub(crate) stride: u64,
    pub(crate) count: u32,
    pub(crate) reserved_offset: u64,
    pub(crate) size: u64,
}

impl HeapLayout {
    /// At most `max_count` descriptors of `size` bytes each, aligned to `alignment`, then
    /// `reserved` bytes at a multiple of `reserved_alignment`, all within `max_size`.
    pub(crate) fn new(
        size: u64,
        alignment: u64,
        max_count: u32,
        reserved: u64,
        reserved_alignment: u64,
        max_size: u64,
    ) -> Self {
        let stride = size.next_multiple_of(alignment.max(1));
        let reserved_alignment = reserved_alignment.max(1);
        // Room for the reserved range and the padding before it.
        let room = max_size.saturating_sub(reserved + reserved_alignment - 1);
        let count = (room / stride).min(max_count as u64) as u32;
        let reserved_offset = (count as u64 * stride).next_multiple_of(reserved_alignment);
        Self {
            stride,
            count,
            reserved_offset,
            size: reserved_offset + reserved,
        }
    }

    /// The byte offset of descriptor `index`.
    pub(crate) fn offset(&self, index: u32) -> u64 {
        index as u64 * self.stride
    }
}

/// A heap's mapped buffer and how command buffers bind it.
#[derive(Debug)]
struct HeapBuffer {
    /// Owns the memory; dropped with the heap at `shutdown`.
    _buffer: Buffer<u8, Upload>,
    mapped: NonNull<u8>,
    bind: vk::BindHeapInfoEXT<'static>,
}

// SAFETY: `mapped` is written only at slots one owner allocated, or through `vkCmdUpdateMemoryKHR`.
unsafe impl Send for HeapBuffer {}
unsafe impl Sync for HeapBuffer {}

impl HeapBuffer {
    fn new(layout: &HeapLayout, alignment: u64) -> Result<Self> {
        let usage = vk::BUFFER_USAGE_DESCRIPTOR_HEAP_EXT;
        let buffer = Buffer::with_flags(usage, Upload, layout.size.max(1), Some(alignment as u32))?;
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
        Ok(HeapBuffer {
            _buffer: buffer,
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

/// The free slots of a heap: sorted, disjoint ranges, never two adjacent ones.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct Ranges {
    free: Vec<Range<u32>>,
}

impl Ranges {
    fn all(count: u32) -> Self {
        let free = if count == 0 {
            Vec::new()
        } else {
            vec![0..count]
        };
        Ranges { free }
    }

    /// The first `n` contiguous free slots (first fit), which are taken.
    pub(crate) fn take(&mut self, n: u32) -> Option<u32> {
        assert!(n > 0, "a slot range holds at least one slot");
        let i = self
            .free
            .iter()
            .position(|range| range.len() >= n as usize)?;
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
            !overlaps(i.checked_sub(1).and_then(|i| self.free.get(i)))
                && !overlaps(self.free.get(i)),
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

/// The first `n` contiguous free image slots.
pub(crate) fn allocate(n: u32) -> Result<u32> {
    lock(&Bindless::get().free)
        .take(n)
        .ok_or(Error::OutOfBindlessSlots {
            heap: Heap::Image,
            count: n,
        })
}

/// Frees image slots no frame reaches any more.
pub(crate) fn free(range: Range<u32>) {
    lock(&Bindless::get().free).give(range);
}

/// Frees the slots of a destroyed image's handle.
pub(crate) fn free_handle(handle: ImageHandle) {
    if let Some(slots) = handle.slots() {
        free(slots);
    }
}

/// Writes `image`'s descriptors into the slots of `handle`, which an image `Image::fit`
/// replaced had, in submission order like [`bind`]'s: frames in flight read the old image's.
pub(crate) fn rebind(image: ShaderImage, handle: ImageHandle) -> Result<()> {
    let Some(slots) = handle.slots() else {
        return Ok(());
    };
    {
        let mut rebound = lock(&Bindless::get().rebound);
        if !rebound.contains(&slots) {
            rebound.push(slots);
        }
    }
    bind_raw([image], [handle], true)
}

/// Discards the queued rewrites of a dropped image's slots, which nothing reads any more and
/// which would overwrite the slots' next owner.
pub(crate) fn forget_handle(handle: ImageHandle) {
    let Some(slots) = handle.slots() else { return };
    let bindless = Bindless::get();
    {
        let mut rebound = lock(&bindless.rebound);
        if let Some(i) = rebound.iter().position(|r| *r == slots) {
            rebound.swap_remove(i);
        } else {
            return;
        }
    }
    let layout = &bindless.images;
    let bytes = layout.offset(slots.start)..layout.offset(slots.end);
    lock(&bindless.queued).retain(|offset, _| !bytes.contains(offset));
}

/// Image slots the application writes with [`bind`], made by [`reserve`]: `len` handles of kind
/// `K`, each `K::WIDTH` slots (none for `reserve(0)`). Dropping it frees them once no
/// submission reaches them, and [`bind`] refuses them right away.
#[derive(Debug)]
#[must_use]
pub struct SlotRange<K: Allocatable> {
    start: u32,
    len: u32,
    kind: PhantomData<K>,
    _live: Live,
}

impl<K: Allocatable> SlotRange<K> {
    /// How many handles it holds.
    pub fn len(&self) -> u32 {
        self.len
    }

    /// Whether it holds no handle (`reserve(0)`).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Its slots in the resource heap, `K::WIDTH` per handle.
    pub fn slots(&self) -> Range<u32> {
        self.start..self.start + self.len * K::WIDTH
    }

    /// Its `i`-th handle. Panics past its end.
    pub fn handle(&self, i: u32) -> ImageHandle<K> {
        assert!(i < self.len, "handle {i} of a range of {}", self.len);
        ImageHandle::new(self.start + i * K::WIDTH)
    }
}

impl<K: Allocatable> Drop for SlotRange<K> {
    fn drop(&mut self) {
        if self.is_empty() {
            return;
        }
        let range = self.slots();
        let bindless = Bindless::get();
        let mut reserved = lock(&bindless.reserved);
        if let Some(i) = reserved.iter().position(|r| *r == range) {
            reserved.swap_remove(i);
        }
        drop(reserved);
        // Nothing reads them any more, and a later copy could overwrite the slots' next owner.
        let layout = &bindless.images;
        let bytes = layout.offset(range.start)..layout.offset(range.end);
        lock(&bindless.queued).retain(|offset, _| !bytes.contains(offset));
        lifetime::retire(move || free(range));
    }
}

/// The first `n` handles' worth of contiguous free image slots (`n * K::WIDTH`), for [`bind`];
/// an empty range for 0.
pub fn reserve<K: Allocatable>(n: u32) -> Result<SlotRange<K>> {
    let live = Live::new();
    let slots = n.checked_mul(K::WIDTH).expect("too many slots");
    let start = match n {
        0 => 0,
        _ => allocate(slots)?,
    };
    if n > 0 {
        lock(&Bindless::get().reserved).push(start..start + slots);
    }
    Ok(SlotRange {
        start,
        len: n,
        kind: PhantomData,
        _live: live,
    })
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
            assert!(
                (1.0..=max).contains(&ratio),
                "a sampler's anisotropy is 1 to the device's {max}, not {ratio}"
            );
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
    _live: Live,
}

impl Sampler {
    /// Writes `desc` into a free slot of the sampler heap; equal descriptions get slots of
    /// their own.
    #[validation_trace]
    pub fn new(desc: &SamplerDesc) -> Result<Self> {
        let live = Live::new();
        let bindless = Bindless::get();
        let index = lock(&bindless.free_samplers)
            .take(1)
            .ok_or(Error::OutOfBindlessSlots {
                heap: Heap::Sampler,
                count: 1,
            })?;
        // From here on `Drop` frees the slot.
        let sampler = Sampler { index, _live: live };
        let layout = &bindless.samplers;
        let range = bindless
            .sampler_heap
            .host_range(layout.offset(index), layout.stride);
        desc.with_create_info(|info| unsafe {
            Functions::descriptor_heap()
                .write_sampler_descriptors(std::slice::from_ref(info), &[range])
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

/// The kinds of descriptors an [`ImageHandle`] names: [`Sampled`], [`Storage`], [`Combined`]
/// (both), or [`Dynamic`], any of them, known only at runtime.
pub trait HandleKind: crate::sealed::Sealed + Send + Sync + 'static {}

/// The kinds images are bound as: `WIDTH` descriptors from the handle's slot.
pub trait Allocatable: HandleKind {
    const WIDTH: u32;
    /// The handle's bits 30 and 31.
    const BITS: u32;
    /// Whether it has a sampled descriptor.
    const SAMPLED: bool;
    /// Whether it has a storage descriptor.
    const STORAGE: bool;
    /// The call that binds an image as it.
    const BIND: &str;
}

/// Kinds with a sampled descriptor, which `Tex2D` and the like read.
pub trait HasSampled: HandleKind {}
/// Kinds with a storage descriptor, which `Image2D` and the like read and write.
pub trait HasStorage: HandleKind {}

/// A sampled image descriptor.
#[derive(Debug)]
pub enum Sampled {}
/// A storage image descriptor.
#[derive(Debug)]
pub enum Storage {}
/// A sampled descriptor and, in the next slot, a storage descriptor of the same image
/// (`MutTex2D`).
#[derive(Debug)]
pub enum Combined {}
/// Any kind, known only at runtime (`DynImg`).
#[derive(Debug)]
pub enum Dynamic {}

macro_rules! handle_kinds {
    ($($kind:ident: $width:literal, $bits:expr, $sampled:literal, $storage:literal, $bind:literal;)*) => {$(
        impl crate::sealed::Sealed for $kind {}
        impl HandleKind for $kind {}
        impl Allocatable for $kind {
            const WIDTH: u32 = $width;
            const BITS: u32 = $bits;
            const SAMPLED: bool = $sampled;
            const STORAGE: bool = $storage;
            const BIND: &str = $bind;
        }
    )*};
}
handle_kinds! {
    Sampled: 1, 0, true, false, "bind_sampled";
    Storage: 1, STORAGE_FIRST, false, true, "bind_storage";
    Combined: 2, PAIR, true, true, "bind_combined";
}
impl crate::sealed::Sealed for Dynamic {}
impl HandleKind for Dynamic {}
impl HasSampled for Sampled {}
impl HasSampled for Combined {}
impl HasStorage for Storage {}
impl HasStorage for Combined {}

/// The slot bits of an [`ImageHandle`].
const SLOT: u32 = (1 << 30) - 1;
/// Set if the first descriptor is a storage descriptor.
const STORAGE_FIRST: u32 = 1 << 30;
/// Set if the handle names two descriptors, sampled then storage.
const PAIR: u32 = 1 << 31;

/// An image as shaders see it (`Img` and `DynImg` in `bindless.slang`), one `u32`: bits 0 to 29
/// are the slot of its first descriptor in the resource heap, bit 30 that descriptor's kind (0
/// sampled, 1 storage), bit 31 how many there are (0 one, 1 two: sampled at the slot, storage
/// at the next). [`Self::NULL`], all ones (storage first, two of them), names nothing.
///
/// `K` is what it is known to name: lava's binds and [`reserve`] make the typed kinds, which
/// [`Self::erase`] turns into [`Dynamic`], and [`Self::as_sampled`]/[`Self::as_storage`] into
/// the part of a [`Combined`] handle a shader reads; every one of them keeps the bits.
#[repr(transparent)]
pub struct ImageHandle<K: HandleKind = Dynamic> {
    bits: u32,
    kind: PhantomData<K>,
}

// SAFETY: a `u32` and a zero-sized marker.
unsafe impl<K: HandleKind> Zeroable for ImageHandle<K> {}
unsafe impl<K: HandleKind> Pod for ImageHandle<K> {}

impl<K: HandleKind> Clone for ImageHandle<K> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<K: HandleKind> Copy for ImageHandle<K> {}
impl<K: HandleKind> PartialEq for ImageHandle<K> {
    fn eq(&self, other: &Self) -> bool {
        self.bits == other.bits
    }
}
impl<K: HandleKind> Eq for ImageHandle<K> {}
impl<K: HandleKind> std::hash::Hash for ImageHandle<K> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.bits.hash(state);
    }
}
impl<K: HandleKind> Default for ImageHandle<K> {
    fn default() -> Self {
        Self::NULL
    }
}
impl<K: HandleKind> std::fmt::Debug for ImageHandle<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_null() {
            return f.write_str("ImageHandle(null)");
        }
        f.debug_struct("ImageHandle")
            .field("sampled", &self.sampled_slot())
            .field("storage", &self.storage_slot())
            .finish()
    }
}

impl<K: HandleKind> ImageHandle<K> {
    /// No image; a `DynImg` reads zeroes from it.
    pub const NULL: Self = Self::from_bits(!0);

    pub(crate) const fn from_bits(bits: u32) -> Self {
        Self {
            bits,
            kind: PhantomData,
        }
    }

    /// The `u32` shaders read.
    pub const fn bits(self) -> u32 {
        self.bits
    }

    pub const fn is_null(self) -> bool {
        self.bits == Self::NULL.bits
    }

    /// How many descriptors it names.
    const fn width(self) -> u32 {
        1 + (self.bits >> 31)
    }

    /// Its slots in the resource heap, `None` for `NULL`.
    pub(crate) fn slots(self) -> Option<Range<u32>> {
        let first = self.bits & SLOT;
        (!self.is_null()).then(|| first..first + self.width())
    }

    /// The slot of its sampled descriptor, if it has one.
    pub fn sampled_slot(self) -> Option<u32> {
        (!self.is_null() && self.bits & STORAGE_FIRST == 0).then_some(self.bits & SLOT)
    }

    /// The slot of its storage descriptor, if it has one.
    pub fn storage_slot(self) -> Option<u32> {
        let has = !self.is_null() && self.bits & (STORAGE_FIRST | PAIR) != 0;
        has.then_some((self.bits & SLOT) + (self.bits >> 31))
    }

    /// The handle of mip `level` of a mip chain this is mip 0 of (`bind_mips_*`), which lava
    /// binds `width` slots apart (`Img.Mip(level)` in shaders). Panics for `NULL`.
    pub fn mip(self, level: u32) -> Self {
        assert!(!self.is_null(), "a null handle has no mips");
        let slot = (self.bits & SLOT) + level * self.width();
        assert!(slot <= SLOT, "mip {level} lies past the last slot");
        Self::from_bits(self.bits & !SLOT | slot)
    }

    /// The same handle as [`Dynamic`], for a `DynImg`.
    pub const fn erase(self) -> ImageHandle {
        ImageHandle::from_bits(self.bits)
    }

    /// The same handle, for a shader that reads its sampled descriptor.
    pub const fn as_sampled(self) -> ImageHandle<Sampled>
    where
        K: HasSampled,
    {
        ImageHandle::from_bits(self.bits)
    }

    /// The same handle, for a shader that reads or writes its storage descriptor.
    pub const fn as_storage(self) -> ImageHandle<Storage>
    where
        K: HasStorage,
    {
        ImageHandle::from_bits(self.bits)
    }
}

impl ImageHandle {
    /// The same handle as `K`, if it has `K`'s descriptors.
    pub(crate) fn narrow<K: Allocatable>(self) -> Option<ImageHandle<K>> {
        let has = (!K::SAMPLED || self.sampled_slot().is_some())
            && (!K::STORAGE || self.storage_slot().is_some());
        (has && !self.is_null()).then(|| ImageHandle::from_bits(self.bits))
    }
}

impl<K: Allocatable> ImageHandle<K> {
    /// The handle of `K`'s descriptors from `slot` on.
    pub(crate) fn new(slot: u32) -> Self {
        assert!(
            slot + K::WIDTH - 1 <= SLOT,
            "image slot {slot} doesn't fit a handle"
        );
        Self::from_bits(slot | K::BITS)
    }
}

/// Valid image slots are `0..max_image_slots()`; a [`Combined`] handle takes two.
pub fn max_image_slots() -> u32 {
    Bindless::get().images.count
}

/// What an image descriptor shows: `mips` and `layers` of one image through a view of
/// `view_type`, of the format's one aspect, or depth for a combined depth-stencil format.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ShaderImage {
    pub(crate) image: vk::Image,
    pub(crate) format: vk::Format,
    pub(crate) aspects: vk::ImageAspectFlags,
    pub(crate) view_type: vk::ImageViewType,
    pub(crate) mips: std::range::Range<u32>,
    pub(crate) layers: std::range::Range<u32>,
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
            layers: self.layers,
            usage: self.usage.flags(),
        }
    }
}

/// Writes each image's descriptors into the slots of the handle at the same position, skipping
/// `NULL` handles. Panics unless every slot lies in a live [`SlotRange`] and every image has
/// the usages of its handle's descriptors.
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
pub fn bind<'a, F: Format, U: ImageUsage, D: Dim, K: Allocatable>(
    images: impl IntoIterator<Item = ImageView<'a, F, U, D>>,
    handles: impl IntoIterator<Item = ImageHandle<K>>,
) {
    let _call = crate::state::call();
    let bindless = Bindless::get();
    let reserved = lock(&bindless.reserved);
    let handles = handles.into_iter().inspect(|handle| {
        if let Some(slots) = handle.slots() {
            assert!(
                reserved
                    .iter()
                    .any(|r| r.start <= slots.start && slots.end <= r.end),
                "bindless slots {slots:?} lie in no `SlotRange`: `bind` writes only reserved \
                 slots (`reserve`)"
            );
        }
    });
    let images = images.into_iter().map(|image| image.shader_image());
    bind_raw(images, handles.map(ImageHandle::erase), true)
        .expect("writing image descriptors failed");
}

/// [`bind`] for any slots: writes the descriptors into the heap right away, or with `queued`
/// leaves them for the next submission (see [`bind`]). Right away is for slots no submission
/// reaches, freshly allocated ones.
pub(crate) fn bind_raw(
    images: impl IntoIterator<Item = ShaderImage>,
    handles: impl IntoIterator<Item = ImageHandle>,
    queued: bool,
) -> Result<()> {
    let bindless = Bindless::get();
    let layout = &bindless.images;
    let mut targets: SmallVec<[(ShaderImage, vk::DescriptorType, u32); 8]> = SmallVec::new();
    for (image, handle) in images.into_iter().zip(handles) {
        let descriptors = [
            (
                handle.sampled_slot(),
                vk::ImageUsageFlags::SAMPLED,
                vk::DescriptorType::SAMPLED_IMAGE,
            ),
            (
                handle.storage_slot(),
                vk::ImageUsageFlags::STORAGE,
                vk::DescriptorType::STORAGE_IMAGE,
            ),
        ];
        for (slot, usage, ty) in descriptors {
            let Some(slot) = slot else { continue };
            assert!(
                image.usage.contains(usage),
                "an image with usage {:?} can't be bound as {usage:?}",
                image.usage,
            );
            assert!(
                slot < layout.count,
                "bindless slot {slot} is outside the {} image slots",
                layout.count
            );
            targets.push((image, ty, slot));
        }
    }
    if targets.is_empty() {
        return Ok(());
    }
    // Each descriptor's view shows only its own usage: a `MutableFormat` image may have usages
    // its view's format lacks (storage of an sRGB image through a UNORM view).
    let mut usages: SmallVec<[_; 8]> = targets
        .iter()
        .map(|&(_, ty, _)| {
            let usage = match ty {
                vk::DescriptorType::SAMPLED_IMAGE => vk::ImageUsageFlags::SAMPLED,
                _ => vk::ImageUsageFlags::STORAGE,
            };
            vk::ImageViewUsageCreateInfo::default().usage(usage)
        })
        .collect();
    let views: SmallVec<[_; 8]> = targets
        .iter()
        .zip(usages.iter_mut())
        .map(|((image, ..), usage)| {
            vk::ImageViewCreateInfo::default()
                .push_next(usage)
                .image(image.image)
                .view_type(image.view_type)
                .format(image.format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: single_aspect(image.aspects),
                    base_mip_level: image.mips.start,
                    level_count: image.mips.end - image.mips.start,
                    base_array_layer: image.layers.start,
                    layer_count: image.layers.end - image.layers.start,
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
        .map(|(&(_, ty, _), info)| {
            vk::ResourceDescriptorInfoEXT::default()
                .ty(ty)
                .data((info as *const vk::ImageDescriptorInfoEXT).cast())
        })
        .collect();
    let functions = Functions::descriptor_heap();
    if !queued {
        let ranges: SmallVec<[_; 8]> = (targets.iter())
            .map(|&(.., slot)| {
                bindless
                    .resource_heap
                    .host_range(layout.offset(slot), layout.stride)
            })
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
    for (&(.., slot), descriptor) in targets.iter().zip(staging.chunks(stride)) {
        queue.insert(layout.offset(slot), SmallVec::from_slice(descriptor));
    }
    Ok(())
}

impl Bindless {
    #[track_caller]
    fn get() -> &'static Self {
        BINDLESS.get()
    }

    pub(crate) fn init() -> Result<()> {
        let p = &Ctx::physical_device().descriptor_heap_properties;
        let images = HeapLayout::new(
            p.image_descriptor_size,
            p.image_descriptor_alignment,
            MAX_IMAGE_SLOTS,
            p.min_resource_heap_reserved_range,
            p.image_descriptor_alignment
                .max(p.buffer_descriptor_alignment),
            p.max_resource_heap_size,
        );
        let samplers = HeapLayout::new(
            p.sampler_descriptor_size,
            p.sampler_descriptor_alignment,
            MAX_SAMPLERS,
            p.min_sampler_heap_reserved_range,
            p.sampler_descriptor_alignment,
            p.max_sampler_heap_size,
        );
        let resource_heap = HeapBuffer::new(&images, p.resource_heap_alignment)?;
        let sampler_heap = HeapBuffer::new(&samplers, p.sampler_heap_alignment)?;
        BINDLESS.set(Self {
                free: Mutex::new(Ranges::all(images.count)),
                reserved: Default::default(),
                rebound: Default::default(),
                free_samplers: Mutex::new(Ranges::all(samplers.count)),
                images,
                samplers,
                resource_heap,
                sampler_heap,
                queued: Mutex::new(BTreeMap::new()),
            });
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
        barrier(
            S::TRANSFER,
            A::TRANSFER_WRITE,
            S::ALL_COMMANDS,
            vk::ACCESS_2_RESOURCE_HEAP_READ_EXT,
        );
        queued
    }

    /// Queues descriptors again whose submission never reached the GPU, unless `bind` queued a
    /// newer one for their slot meanwhile or their reservation (or fitted image) was dropped.
    pub(crate) fn requeue(queued: Queued) {
        let s = Self::get();
        let reserved = lock(&s.reserved);
        let rebound = lock(&s.rebound);
        let live = |offset: u64| {
            let slot = (offset / s.images.stride) as u32;
            (reserved.iter().chain(rebound.iter())).any(|range| range.contains(&slot))
        };
        let mut queue = lock(&s.queued);
        for (offset, descriptor) in queued.into_iter().filter(|&(offset, _)| live(offset)) {
            queue.entry(offset).or_insert(descriptor);
        }
    }

    /// Unsets the heaps and hands their buffers to [`lifetime`], at `shutdown`.
    ///
    /// # Safety
    /// Nothing records or binds (lava is closed), and every image, sampler and slot range was
    /// destroyed.
    pub(crate) unsafe fn destroy() {
        drop(unsafe { BINDLESS.take() });
    }
}

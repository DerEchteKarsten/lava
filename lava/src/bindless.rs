//! Global bindless descriptor heaps (`VK_EXT_descriptor_heap`), which shaders index directly: image descriptors at slots lava allocates (first fit, written right away, freed with their images; slot 0 never, the null handle), the one-`u32` `ImageHandle<K>` that names them, and runtime `Sampler`s in the sampler heap (slot 0 lava's default sampler, each other in an `Arc` that referencing buffers hold); handles are `AnyBitPattern` but not `NoUninit`, so only reference infos (`ImageReference` for bare handles) write them into buffers
use std::{
    marker::PhantomData,
    ops::Range,
    ptr::NonNull,
    sync::{Arc, Mutex},
};

use crate::error::{Error, Result};
use crate::vk::{self, BorderColor, CompareOp, Filter, SamplerAddressMode, SamplerMipmapMode};
use bytemuck::{Pod, Zeroable};
use lava_macros::validation_trace;
use smallvec::SmallVec;

use crate::{
    buffer::{Buffer, Upload},
    global::{Global, Live},
    image::{
        Dim,
        format::Format,
        slice::{ImageView, single_aspect},
        usage::ImageUsage,
    },
    object::{Header, Object},
    queue::{Kind, lock},
    state::{Ctx, Functions},
};

#[derive(Debug)]
pub(crate) struct Bindless {
    /// The resource heap: image descriptors, then the reserved range.
    images: HeapLayout,
    samplers: HeapLayout,
    free: Mutex<Ranges>,
    free_samplers: Mutex<Ranges>,
    resource_heap: HeapBuffer,
    sampler_heap: HeapBuffer,
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
        let buffer = Buffer::with_flags(usage, Upload, layout.size.max(1), Some(alignment))?;
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
    /// Every slot free: one range of all of them (not a list of the slots).
    #[allow(clippy::single_range_in_vec_init)]
    pub(crate) fn all(count: u32) -> Self {
        let free = if count == 0 {
            Vec::new()
        } else {
            vec![0..count]
        };
        Ranges { free }
    }

    /// The first `n` contiguous free slots (first fit), which are taken.
    pub(crate) fn take(&mut self, n: u32) -> Option<u32> {
        assert!(n > 0, "a run of slots holds at least one");
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
    pub(crate) internal: Arc<SamplerInternal>,
}

/// A sampler as lava owns it: its slot, freed with the last owner (the [`Sampler`], buffers that
/// reference it) once no submission reaches it.
#[derive(Debug)]
pub(crate) struct SamplerInternal {
    index: u32,
    header: Header,
}

impl Object for SamplerInternal {
    fn header(&self) -> &Header {
        &self.header
    }
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
        let internal = Arc::new(SamplerInternal {
            index,
            header: Header::new(live),
        });
        write_sampler(index, desc)?;
        Ok(Sampler { internal })
    }

    pub fn handle(&self) -> SamplerHandle {
        SamplerHandle {
            index: self.internal.index,
        }
    }
}

impl Drop for SamplerInternal {
    fn drop(&mut self) {
        let index = self.index;
        // A push constant may name it without keeping it: its slot waits for what is in flight.
        crate::queue::timeline::destroy_when_idle(move || {
            lock(&Bindless::get().free_samplers).give(index..index + 1)
        });
    }
}

/// Writes `desc` into slot `index` of the sampler heap.
fn write_sampler(index: u32, desc: &SamplerDesc) -> Result<()> {
    let bindless = Bindless::get();
    let layout = &bindless.samplers;
    let range = bindless
        .sampler_heap
        .host_range(layout.offset(index), layout.stride);
    desc.with_create_info(|info| unsafe {
        Functions::descriptor_heap().write_sampler_descriptors(std::slice::from_ref(info), &[range])
    })??;
    Ok(())
}

/// A [`Sampler`] as shaders see it (`Sampler` in `bindless.slang`): its slot in the sampler
/// heap. A zeroed one ([`Self::DEFAULT`]) is slot 0, which lava fills with
/// [`SamplerDesc::new`]'s sampler (trilinear, repeating) and never hands out, so zeroed structs
/// sample sensibly.
#[derive(Pod, Zeroable, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct SamplerHandle {
    pub index: u32,
}

impl SamplerHandle {
    /// Slot 0: lava's default sampler, what a zeroed handle names.
    pub const DEFAULT: Self = SamplerHandle { index: 0 };
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
/// at the next). [`Self::NULL`], zero (sampled slot 0, which lava never hands out), names
/// nothing, so a zeroed struct of handles holds null ones:
/// ```
/// let handle: lava::bindless::ImageHandle = bytemuck::Zeroable::zeroed();
/// assert!(handle.is_null());
/// ```
///
/// `K` is what it is known to name: lava's binds make the typed kinds, which
/// [`Self::erase`] turns into [`Dynamic`], and [`Self::as_sampled`]/[`Self::as_storage`] into
/// the part of a [`Combined`] handle a shader reads; every one of them keeps the bits.
#[repr(transparent)]
pub struct ImageHandle<K: HandleKind = Dynamic> {
    bits: u32,
    kind: PhantomData<K>,
}

// SAFETY: a `u32` and a zero-sized marker: every bit pattern a value, and zero is `NULL`. Not
// `NoUninit`: a handle names an image, which a buffer of handles keeps alive only when they are
// written with references (`ReferencingData`); generated push constants, which commands track,
// implement `NoUninit` themselves.
unsafe impl<K: HandleKind> bytemuck::Zeroable for ImageHandle<K> {}
// SAFETY: as above.
unsafe impl<K: HandleKind> bytemuck::AnyBitPattern for ImageHandle<K> {}

// SAFETY: one handle, one reference: what `ImageReference` writes.
unsafe impl<K: HandleKind> crate::reference::ReferencingData for ImageHandle<K> {
    const REFERENCES: usize = 1;
    const WRITES: u64 = 0;
    const UNORDERED: u64 = 0;
}

/// A bare handle as a buffer element (`Buf<Tex2D>`, a buffer of `ImageHandle<K>`): the view
/// it names, read through `K`'s descriptors (the buffer's element type decides `K`).
pub struct ImageReference<K, V>(V, PhantomData<K>);

impl<K, V> ImageReference<K, V> {
    pub fn new(view: V) -> Self {
        ImageReference(view, PhantomData)
    }
}

impl<'a, K: Allocatable, V: crate::image::ImageRef + 'a> crate::reference::ReferenceInfo<'a>
    for ImageReference<K, V>
{
    type Data = ImageHandle<K>;
    #[track_caller]
    fn write(self, data: &mut ImageHandle<K>, references: &mut [crate::reference::Reference<'a>]) {
        let view = self.0.view();
        *data = view.handle::<K>();
        references[0] = crate::reference::Reference::image(view);
    }
}

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
    /// No image (zero, sampled slot 0, which lava never hands out); a `DynImg` reads zeroes
    /// from it.
    pub const NULL: Self = Self::from_bits(0);

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
        Self::width_of(self.bits)
    }

    /// How many descriptors a handle of `kind` (its bits 30 and 31) names.
    pub(crate) const fn width_of(kind: u32) -> u32 {
        1 + (kind >> 31)
    }

    /// Its kind: bits 30 and 31.
    pub(crate) const fn kind(self) -> u32 {
        self.bits & !SLOT
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

impl ImageHandle {
    /// The handle of the descriptors of `kind` (a handle's bits 30 and 31) from `slot` on.
    pub(crate) fn with_kind(kind: u32, slot: u32) -> Self {
        assert!(
            slot + Self::width_of(kind) - 1 <= SLOT,
            "image slot {slot} doesn't fit a handle"
        );
        assert!(
            slot > 0,
            "image slot 0 is reserved: its sampled handle is `NULL`"
        );
        Self::from_bits(slot | kind)
    }
}

impl<K: Allocatable> ImageHandle<K> {
    /// The handle of `K`'s descriptors from `slot` on.
    pub(crate) fn new(slot: u32) -> Self {
        assert!(
            slot + K::WIDTH - 1 <= SLOT,
            "image slot {slot} doesn't fit a handle"
        );
        assert!(
            slot > 0,
            "image slot 0 is reserved: its sampled handle is `NULL`"
        );
        Self::from_bits(slot | K::BITS)
    }
}

/// Image slots are `1..max_image_slots()` (slot 0 is the null handle's); a [`Combined`] handle
/// takes two.
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
            image: self.image(),
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
/// `NULL` handles, right away: the slots are freshly allocated, so no submission reaches them.
/// Panics unless every image has the usages of its handle's descriptors.
pub(crate) fn write_descriptors(
    images: impl IntoIterator<Item = ShaderImage>,
    handles: impl IntoIterator<Item = ImageHandle>,
) -> Result<()> {
    write_descriptors_in(images, handles, vk::ImageLayout::GENERAL)
}

/// `write_descriptors` with the sampled descriptors in `sampled_layout` (storage ones are always
/// `GENERAL`): lava's own are `GENERAL`, `raw::write_descriptors` writes others.
pub(crate) fn write_descriptors_in(
    images: impl IntoIterator<Item = ShaderImage>,
    handles: impl IntoIterator<Item = ImageHandle>,
    sampled_layout: vk::ImageLayout,
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
    let infos: SmallVec<[_; 8]> = (views.iter().zip(&targets))
        .map(|(view, &(_, ty, _))| {
            let layout = match ty {
                vk::DescriptorType::SAMPLED_IMAGE => sampled_layout,
                _ => vk::ImageLayout::GENERAL,
            };
            vk::ImageDescriptorInfoEXT::default().p_view(view).layout(layout)
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
    let ranges: SmallVec<[_; 8]> = (targets.iter())
        .map(|&(.., slot)| {
            bindless
                .resource_heap
                .host_range(layout.offset(slot), layout.stride)
        })
        .collect();
    unsafe { Functions::descriptor_heap().write_resource_descriptors(&resources, &ranges)? };
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
        // Slot 0 of each heap is never handed out: a zeroed image handle (`ImageHandle::NULL`)
        // names no image, and a zeroed `SamplerHandle` lava's default sampler, written below.
        let (mut free, mut free_samplers) =
            (Ranges::all(images.count), Ranges::all(samplers.count));
        let reserved = (free.take(1), free_samplers.take(1));
        debug_assert_eq!(reserved, (Some(0), Some(0)));
        BINDLESS.set(Self {
            free: Mutex::new(free),
            free_samplers: Mutex::new(free_samplers),
            images,
            samplers,
            resource_heap,
            sampler_heap,
        });
        write_sampler(0, &SamplerDesc::new())
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

    /// Unsets the heaps and destroys their buffers, at `shutdown`.
    ///
    /// # Safety
    /// Nothing records or binds (lava is closed), and every image and sampler was destroyed.
    pub(crate) unsafe fn destroy() {
        drop(unsafe { BINDLESS.take() });
    }
}

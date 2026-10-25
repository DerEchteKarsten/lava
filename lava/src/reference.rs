//! Referencing buffers: element types whose handles and addresses reference other GPU objects (`ReferencingData`, implemented by generated code), written only with their references (`ReferenceInfo`, `DynImage` for a `DynImg`, `Buffer::host_write_referencing`, `Uploader::write_referencing`), which the buffer keeps alive (`KeepAlive`: its distinct children, four looked up branch-free and more through a hash index, and a child index per reference) and which commands that name the buffer reach for barriers
use std::{
    collections::HashMap,
    hash::{BuildHasherDefault, Hasher},
    range::Range,
    sync::{Arc, Weak},
};

use smallvec::SmallVec;

#[cfg(feature = "acceleration-structure")]
use crate::vkobjects::acceleration_structure::{StructureInternal, Tlas};
use crate::{
    bindless::{ImageHandle, Sampler, SamplerInternal},
    buffer::{BufferInternal, slice::BufferSlice},
    image::{Dim, ImageInternal, ImageRef, format::Format, slice::ImageView, usage::ImageUsage},
    object::{Object, Strong, mark_writes_below},
    queue::lock,
};

/// An element type whose handles and addresses reference other GPU objects, written into a
/// buffer only together with its references ([`ReferenceInfo`]), which the buffer keeps alive
/// and which commands that name the buffer reach for barriers. lava-build-slang implements it
/// for every Slang struct with `Buf`, `MutBuf`, image, `DynImg`, `Accel` or `Sampler` fields.
///
/// It is `AnyBitPattern` (buffers of it can be read back) but not `NoUninit`, so none of
/// lava's plain writes (`host_write`, `host_mut`, `Uploader::write`, `cast`, ...) take it:
/// ```compile_fail
/// # use lava::{Buffer, buffer::Upload, reference::ReferencingData};
/// #[derive(Clone, Copy, lava::bytemuck::AnyBitPattern)]
/// #[bytemuck(crate = "lava::bytemuck")]
/// #[repr(C)]
/// struct Handles([u32; 2]);
/// unsafe impl ReferencingData for Handles {
///     const REFERENCES: usize = 2;
///     const WRITES: u64 = 0;
///     const UNORDERED: u64 = 0;
/// }
/// fn forge(buffer: &mut Buffer<Handles, Upload>) {
///     buffer.host_mut()[0] = Handles([1, 2]);
/// }
/// ```
/// and commands that would write a buffer of it panic (`fill_buffer`, a pass's `MutBuf`).
///
/// # Safety
/// Every handle and address in `Self` comes with a [`Reference`] from its [`ReferenceInfo`],
/// exactly `REFERENCES` of them, always in the same order; bit `j` of `WRITES` is set exactly
/// where the shader may write through reference `j`, bit `j` of `UNORDERED` only where `WRITES`
/// has it and draws of one rendering may write through it in any order (commuting atomics, or
/// elements no other draw touches). Lava keeps objects alive and synchronises them by this.
pub unsafe trait ReferencingData: bytemuck::AnyBitPattern {
    const REFERENCES: usize;
    /// Bit `j`: reference `j` is a write reference (`MutBuf`, `MutImage*`, `MutTex*`).
    const WRITES: u64;
    /// Bit `j`: write reference `j` is unordered (the field's `[Unordered]`).
    const UNORDERED: u64;
}

/// One element of a [`ReferencingData`] type as typed slices and views: lava-build-slang
/// generates one (`{Name}ReferenceInfo`) per referencing struct, which writes its handles and
/// addresses from them.
pub trait ReferenceInfo<'a> {
    type Data: ReferencingData;
    /// Writes the element into `data` (zeroed) and its references into `references`, exactly
    /// `Data::REFERENCES` of them (`Reference::NONE` until written), in order; nested structs
    /// write theirs into the sub-slice their fields cover.
    fn write(self, data: &mut Self::Data, references: &mut [Reference<'a>]);
}

/// `item`'s element and references.
pub(crate) fn output<'a, T: ReferencingData>(
    item: impl ReferenceInfo<'a, Data = T>,
) -> (T, SmallVec<[Reference<'a>; 8]>) {
    let mut data = T::zeroed();
    let mut references = smallvec::smallvec![Reference::NONE; T::REFERENCES];
    item.write(&mut data, &mut references);
    (data, references)
}

/// A borrowed reference to an object an element's handle or address names, and the byte range
/// of it the element reaches (a slice's; an image's whole). Made by generated code.
#[derive(Clone, Copy, Debug)]
pub struct Reference<'a> {
    pub(crate) object: Referenced<'a>,
    pub(crate) range: Range<u64>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Referenced<'a> {
    /// Nothing: a reference not written yet.
    None,
    Buffer(&'a Arc<BufferInternal>),
    Image(&'a Arc<ImageInternal>),
    Sampler(&'a Arc<SamplerInternal>),
    #[cfg(feature = "acceleration-structure")]
    Structure(&'a Arc<StructureInternal>),
}

impl<'a> Reference<'a> {
    /// No reference, what a slice of them starts as.
    pub const NONE: Self = Reference {
        object: Referenced::None,
        range: Range { start: 0, end: 0 },
    };

    /// The sampler a `Sampler` field's handle names.
    #[doc(hidden)]
    pub fn sampler(sampler: &'a Sampler) -> Self {
        Reference {
            object: Referenced::Sampler(&sampler.internal),
            range: (0..0).into(),
        }
    }

    /// The TLAS an `Accel` field's address names.
    #[doc(hidden)]
    #[cfg(feature = "acceleration-structure")]
    pub fn tlas(tlas: &'a Tlas) -> Self {
        Self::structure(tlas)
    }

    /// The structure an address names (an instance's BLAS, an `Accel`'s TLAS).
    #[cfg(feature = "acceleration-structure")]
    pub(crate) fn structure<L: crate::vkobjects::acceleration_structure::Level>(
        structure: &'a crate::vkobjects::acceleration_structure::AccelerationStructure<L>,
    ) -> Self {
        Reference {
            object: Referenced::Structure(&structure.internal),
            range: structure.internal.range,
        }
    }

    /// The slice a `Buf`/`MutBuf` field's address points at.
    #[doc(hidden)]
    pub fn buffer<T: Copy + 'static>(slice: BufferSlice<'a, T>) -> Self {
        Reference {
            object: Referenced::Buffer(slice.buffer),
            range: slice.get_range(),
        }
    }

    /// The view an image or `DynImg` field's handle names.
    #[doc(hidden)]
    pub fn image<F: Format, U: ImageUsage, D: Dim>(view: ImageView<'a, F, U, D>) -> Self {
        Reference {
            object: Referenced::Image(view.internal),
            range: (0..1).into(),
        }
    }

    /// `None` for `NONE`.
    fn strong(&self) -> Option<Strong> {
        Some(match self.object {
            Referenced::None => return None,
            Referenced::Buffer(buffer) => Strong::Buffer(buffer.clone()),
            Referenced::Image(image) => Strong::Image(image.clone()),
            Referenced::Sampler(sampler) => Strong::Sampler(sampler.clone()),
            #[cfg(feature = "acceleration-structure")]
            Referenced::Structure(structure) => Strong::Structure(structure.clone()),
        })
    }

    fn key(&self) -> usize {
        self.strong().map_or(0, |strong| strong.key())
    }
}

/// What a `DynImg` field names: a view of any kind its image is bound with, or nothing
/// ([`Self::NULL`], the null handle, which shaders read as zero).
#[derive(Clone, Copy, Debug)]
pub struct DynImage<'a> {
    handle: ImageHandle,
    reference: Reference<'a>,
}

impl<'a> DynImage<'a> {
    pub const NULL: Self = DynImage {
        handle: ImageHandle::NULL,
        reference: Reference::NONE,
    };

    /// `view`, through its image's handle (`ImageView::dynamic_handle`; panics unless it is
    /// bound).
    #[track_caller]
    pub fn new(view: impl ImageRef + 'a) -> Self {
        let view = view.view();
        DynImage {
            handle: view.dynamic_handle(),
            reference: Reference::image(view),
        }
    }

    #[doc(hidden)]
    pub fn handle(&self) -> ImageHandle {
        self.handle
    }

    #[doc(hidden)]
    pub fn reference(&self) -> Reference<'a> {
        self.reference
    }
}

impl Default for DynImage<'_> {
    fn default() -> Self {
        Self::NULL
    }
}

/// A write reference's ordering among draws of one rendering (`ReferencingData::UNORDERED`).
#[derive(Clone, Copy, PartialEq)]
enum Order {
    Ordered,
    Unordered,
}

/// What a referencing buffer references: its distinct children, and a child index per
/// reference of each element. 10 000 instances of 3 meshes are 3 children, 3 `Arc` clones and
/// 2 bytes per reference.
#[derive(Debug)]
pub(crate) struct KeepAlive {
    pub(crate) children: Children,
    /// `len * stride` indices into `children`, element `i`'s from `i * stride`. `NONE`: no
    /// child (never written, or a reference to the buffer itself).
    indices: Vec<u16>,
    /// The element type's `REFERENCES`.
    stride: u32,
    /// The element type's `WRITES` and `UNORDERED`.
    write_positions: u64,
    unordered_positions: u64,
    /// Children with a write reference, and children that reference objects themselves: the
    /// ones a command walks to when nothing is dirty.
    pub(crate) writers: SmallVec<[u16; 2]>,
    pub(crate) containers: SmallVec<[u16; 2]>,
}

const NONE: u16 = u16::MAX;

#[derive(Debug)]
pub(crate) struct Child {
    pub(crate) object: Strong,
    /// How many references point at it; the entry goes at 0.
    count: u32,
    /// How many of them are write references, and how many of those ordered: its accesses
    /// through the buffer are writes while `writes > 0`, unordered ones while
    /// `ordered_writes == 0`.
    pub(crate) writes: u32,
    pub(crate) ordered_writes: u32,
    /// The union of the ranges referenced in it: what a walk's barrier covers. Grows only.
    pub(crate) range: Range<u64>,
}

impl KeepAlive {
    pub(crate) fn new<T: ReferencingData>() -> Self {
        assert!(
            T::REFERENCES <= 64,
            "an element holds at most 64 references, not {}",
            T::REFERENCES
        );
        KeepAlive {
            children: Children::default(),
            indices: Vec::new(),
            stride: T::REFERENCES as u32,
            write_positions: T::WRITES,
            unordered_positions: T::UNORDERED,
            writers: SmallVec::new(),
            containers: SmallVec::new(),
        }
    }

    /// Element `i` (of a buffer of `len`) gets `new`; children no element references any more
    /// go to `released`, for the caller to drop once nothing reads the old element.
    pub(crate) fn replace(
        &mut self,
        parent: &Arc<BufferInternal>,
        len: usize,
        i: usize,
        new: &[Reference],
        released: &mut Vec<Strong>,
    ) {
        assert_eq!(
            new.len(),
            self.stride as usize,
            "an element's references, as its type has"
        );
        let stride = self.stride as usize;
        if self.indices.len() < len * stride {
            self.indices.resize(len * stride, NONE);
        }
        for (j, reference) in new.iter().enumerate() {
            let write = match (
                self.write_positions >> j & 1,
                self.unordered_positions >> j & 1,
            ) {
                (0, _) => None,
                (_, 0) => Some(Order::Ordered),
                _ => Some(Order::Unordered),
            };
            let own = matches!(reference.object, Referenced::Buffer(b) if Arc::ptr_eq(b, parent));
            // None, or the buffer itself: it is alive whenever it is read, and an `Arc` would
            // leak it.
            let none = matches!(reference.object, Referenced::None);
            let index = match own || none {
                true => NONE,
                false => self.insert(parent, reference, write),
            };
            let old = std::mem::replace(&mut self.indices[i * stride + j], index);
            if old != NONE {
                released.extend(self.release(parent, old, write));
            }
        }
    }

    /// Takes over `other`'s children and elements: a buffer that grew (`Buffer::reserve`)
    /// references what the old one did, which keeps its own.
    /// A copy whose children `parent` references (a grown buffer, a TLAS built from
    /// instances), dirty where they are.
    pub(crate) fn clone_for<P: Object>(&self, parent: &Arc<P>) -> Self {
        let mut children = Children::default();
        for (i, child) in self.children.entries.iter().enumerate() {
            if let Some(child) = child {
                register(parent, &child.object);
                let dirty = child.object.object().header().dirty.load(RELAXED);
                parent.header().dirty.fetch_or(dirty, RELAXED);
                let copy = Child {
                    object: child.object.clone(),
                    ..*child
                };
                children.put(i as u16, copy);
            }
        }
        children.free = self.children.free.clone();
        KeepAlive {
            children,
            indices: self.indices.clone(),
            stride: self.stride,
            write_positions: self.write_positions,
            unordered_positions: self.unordered_positions,
            writers: self.writers.clone(),
            containers: self.containers.clone(),
        }
    }

    fn insert(
        &mut self,
        parent: &Arc<BufferInternal>,
        reference: &Reference,
        write: Option<Order>,
    ) -> u16 {
        let index = match self.children.find(reference.key()) {
            Some(i) => {
                let child = self.children.get_mut(i);
                child.count += 1;
                child.range = Range {
                    start: child.range.start.min(reference.range.start),
                    end: child.range.end.max(reference.range.end),
                };
                i
            }
            None => {
                let object = reference.strong().expect("a reference to something");
                debug_assert!(
                    !reaches(object.object(), parent),
                    "a reference cycle between two buffers: both would stay alive for good"
                );
                register(parent, &object);
                let child_header = object.object().header();
                // A child dirty right now makes the buffer dirty too.
                (parent.header.dirty).fetch_or(child_header.dirty.load(RELAXED), RELAXED);
                let container = object.object().keep_alive().is_some();
                let writes_below = child_header.writes_below.load(RELAXED);
                let i = self.children.push(Child {
                    object,
                    count: 1,
                    writes: 0,
                    ordered_writes: 0,
                    range: reference.range,
                });
                if container {
                    self.containers.push(i);
                    if writes_below {
                        mark_writes_below(&**parent);
                    }
                }
                i
            }
        };
        if let Some(order) = write {
            let child = self.children.get_mut(index);
            child.writes += 1;
            child.ordered_writes += (order == Order::Ordered) as u32;
            if child.writes == 1 {
                self.writers.push(index);
                mark_writes_below(&**parent);
            }
        }
        index
    }

    fn release(
        &mut self,
        parent: &Arc<BufferInternal>,
        index: u16,
        write: Option<Order>,
    ) -> Option<Strong> {
        let child = self.children.get_mut(index);
        child.count -= 1;
        if let Some(order) = write {
            child.writes -= 1;
            child.ordered_writes -= (order == Order::Ordered) as u32;
            if child.writes == 0 {
                self.writers.retain(|&mut w| w != index);
            }
        }
        if child.count > 0 {
            return None;
        }
        let child = self.children.take(index);
        self.containers.retain(|&mut c| c != index);
        unregister(parent, &child.object);
        Some(child.object)
    }

    /// Forgets `parent`, which these were the children of, in its children's parents.
    #[cfg(feature = "acceleration-structure")]
    pub(crate) fn unregister<P: Object>(&self, parent: &Arc<P>) {
        for child in self.children.iter() {
            unregister(parent, &child.object);
        }
    }

    /// Forgets the buffer in its children's parents as it is destroyed.
    pub(crate) fn unregister_all(&self) {
        for child in self.children.iter() {
            prune(child.object.object());
        }
    }
}

const RELAXED: std::sync::atomic::Ordering = std::sync::atomic::Ordering::Relaxed;

/// `parent` references `child` now.
fn register<P: Object>(parent: &Arc<P>, child: &Strong) {
    let header = child.object().header();
    lock(&header.parents).push(Arc::downgrade(parent) as Weak<dyn Object>);
    header.parent_count.fetch_add(1, RELAXED);
}

/// `parent` no longer references `child`.
fn unregister<P: Object>(parent: &Arc<P>, child: &Strong) {
    let header = child.object().header();
    let parent = Arc::downgrade(parent) as Weak<dyn Object>;
    let mut parents = lock(&header.parents);
    if let Some(i) = parents.iter().position(|p| p.ptr_eq(&parent)) {
        parents.swap_remove(i);
        header.parent_count.fetch_sub(1, RELAXED);
    }
}

/// Drops `object`'s parents that are gone.
fn prune(object: &dyn Object) {
    let header = object.header();
    let mut parents = lock(&header.parents);
    let before = parents.len();
    parents.retain(|parent| parent.strong_count() > 0);
    header
        .parent_count
        .fetch_sub((before - parents.len()) as u32, RELAXED);
}

/// Whether `target` is `object` or below it.
fn reaches(object: &dyn Object, target: &Arc<BufferInternal>) -> bool {
    if std::ptr::addr_eq(object, Arc::as_ptr(target)) {
        return true;
    }
    let Some(keep_alive) = object.keep_alive() else {
        return false;
    };
    let keep_alive = lock(keep_alive);
    (keep_alive.children.iter()).any(|child| reaches(child.object.object(), target))
}

/// A multiplicative hash of object addresses.
#[derive(Default)]
pub(crate) struct AddressHasher(u64);

impl Hasher for AddressHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ b as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    fn write_usize(&mut self, n: usize) {
        self.0 = (n as u64 >> 4).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

/// A buffer's distinct children. Lookups take a fixed four-way compare while there are at most
/// four entries, and a hash index from the fifth on (measured 2026-10-25: 2.2 ns for the
/// compare, 3.4 ns for the hash, 10 to 35 ns for an early-exit scan of 2 to 32 in random order);
/// entries keep their indices, which the elements' indices name.
#[derive(Debug, Default)]
pub(crate) struct Children {
    /// The addresses of `entries[..4]`, 0 where there is none.
    keys: [usize; 4],
    entries: Vec<Option<Child>>,
    free: Vec<u16>,
    index: Option<HashMap<usize, u16, BuildHasherDefault<AddressHasher>>>,
}

impl Children {
    fn find(&self, key: usize) -> Option<u16> {
        match &self.index {
            Some(index) => index.get(&key).copied(),
            None => {
                let k = &self.keys;
                let hits = (k[0] == key) as u32
                    | ((k[1] == key) as u32) << 1
                    | ((k[2] == key) as u32) << 2
                    | ((k[3] == key) as u32) << 3;
                (hits != 0).then(|| hits.trailing_zeros() as u16)
            }
        }
    }

    fn get_mut(&mut self, i: u16) -> &mut Child {
        self.entries[i as usize].as_mut().expect("a live child")
    }

    pub(crate) fn get(&self, i: u16) -> &Child {
        self.entries[i as usize].as_ref().expect("a live child")
    }

    fn push(&mut self, child: Child) -> u16 {
        let i = match self.free.pop() {
            Some(i) => i,
            None => {
                assert!(
                    self.entries.len() < NONE as usize,
                    "a buffer references at most {} objects",
                    NONE
                );
                self.entries.push(None);
                (self.entries.len() - 1) as u16
            }
        };
        self.put(i, child);
        i
    }

    fn put(&mut self, i: u16, child: Child) {
        let key = child.object.key();
        if self.entries.len() <= i as usize {
            self.entries.resize_with(i as usize + 1, || None);
        }
        self.entries[i as usize] = Some(child);
        if (i as usize) < self.keys.len() {
            self.keys[i as usize] = key;
        }
        if self.index.is_none() && self.entries.len() > self.keys.len() {
            let index = (self.entries.iter().enumerate())
                .filter_map(|(i, c)| c.as_ref().map(|c| (c.object.key(), i as u16)))
                .collect();
            self.index = Some(index);
        } else if let Some(index) = &mut self.index {
            index.insert(key, i);
        }
    }

    fn take(&mut self, i: u16) -> Child {
        let child = self.entries[i as usize].take().expect("a live child");
        if (i as usize) < self.keys.len() {
            self.keys[i as usize] = 0;
        }
        if let Some(index) = &mut self.index {
            index.remove(&child.object.key());
        }
        self.free.push(i);
        child
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &Child> {
        self.entries.iter().flatten()
    }
}

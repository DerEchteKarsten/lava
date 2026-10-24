//! Typed GPU buffers backed by gpu-allocator memory in a location their type names (`memory`; `Gpu` from `Shared`'s memory types on `rebar` devices, so uploaders write it in place), each with every usage lava's commands need; host access to host-visible ones (writes of `NoUninit` elements, reads of `AnyBitPattern` ones), checked against GPU writes still in flight; growth that keeps the contents (`reserve`); the staging rings and uploaders (`staging`); casts; the internal its owners share (`BufferInternal`, also of sparse and external buffers, whose memory it frees), destroyed once the last owner dropped it and no frame can reach it
use std::{fmt::Debug, marker::PhantomData, mem, range::Range, sync::Arc};

use crate::error::{Error, Result};
use crate::vk;
use bytemuck::{AnyBitPattern, NoUninit};
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use lava_macros::validation_trace;
use smallvec::SmallVec;

use crate::{
    command_buffer::{BufferAccess, pending::incomplete_write},
    global::Live,
    lifetime,
    object::Header,
    queue::{
        lock,
        timeline::{INDICES, completed, poll},
    },
    state::Ctx,
};

pub mod memory;
pub(crate) mod slice;
pub(crate) mod staging;

pub use memory::{
    Dynamic, Gpu, HostVisible, Location, Memory, Readback, Shared, StaticMemory, Upload,
};
pub use slice::BufferSlice;
pub use staging::{UploadStats, Uploader};

/// The usages of every buffer an application creates: everything lava's commands bind a buffer
/// as (lava binds no vertex or uniform buffers). Giving every buffer every usage made no
/// measurable difference against one usage per buffer (2026-10-19, Arc B390 with Mesa's ANV:
/// the same memory requirements for every buffer, the same benchmark view and recording times).
pub const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
        | vk::BufferUsageFlags::INDEX_BUFFER.as_raw()
        | vk::BufferUsageFlags::INDIRECT_BUFFER.as_raw(),
);

/// A buffer of `T`s (`u8` when untyped) in memory `M` ([`Gpu`] by default). Any `Copy` element
/// works for what the GPU does with it; host writes and uploads take `NoUninit` elements
/// (generated structs with image handles among them), host reads `AnyBitPattern` ones. Elements
/// have a size:
/// ```compile_fail
/// # fn f() -> lava::Result<()> {
/// let buffer: lava::buffer::Buffer<(), lava::buffer::Upload> = lava::buffer::Buffer::upload(1)?;
/// # Ok(()) }
/// # let _ = f();
/// ```
/// Only host-visible memory has host access:
/// ```compile_fail
/// # fn f() -> lava::Result<()> {
/// let mut buffer = lava::buffer::Buffer::<u32>::new(1)?;
/// buffer.host();
/// # Ok(()) }
/// ```
/// ```compile_fail
/// # fn f() -> lava::Result<()> {
/// let mut buffer = lava::buffer::Buffer::<u32>::new(1)?;
/// buffer.host_mut();
/// # Ok(()) }
/// ```
/// ```
/// # fn f() -> lava::Result<()> {
/// use lava::buffer::{Buffer, Upload};
/// // `Buffer::<u32>` names `Buffer<u32, Gpu>`, which has no `upload`.
/// let mut buffer: Buffer<u32, Upload> = Buffer::upload(1)?;
/// buffer.host_mut()[0] = 1;
/// # Ok(()) }
/// ```
pub struct Buffer<T: Copy + 'static, M: Memory = Gpu> {
    pub(crate) internal: Arc<BufferInternal>,
    memory: M,
    _type_marker: PhantomData<T>,
}

impl<T: Copy + 'static, M: Memory> Debug for Buffer<T, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("address", &self.internal.address)
            .field("size", &self.internal.num_bytes)
            .field("memory", &self.memory.location())
            .finish()
    }
}

/// A buffer as lava owns it, shared by its [`Buffer`] and whatever else keeps it: the handle,
/// its addresses and its memory, destroyed with the last owner.
#[derive(Debug)]
pub(crate) struct BufferInternal {
    pub(crate) handle: vk::Buffer,
    pub(crate) address: u64,
    pub(crate) num_bytes: u64,
    pub(crate) memory: BufferMemory,
    /// Counts it as a lava user until its drop finished.
    #[allow(dead_code, reason = "held for its count")]
    pub(crate) header: Header,
}

/// Where a buffer's memory comes from.
#[derive(Debug)]
pub(crate) enum BufferMemory {
    /// lava's allocator; a null allocation for none.
    Allocation(Allocation),
    /// A dedicated allocation exported or imported (`ExternalBuffer`).
    #[cfg(all(feature = "external-memory", unix))]
    External(#[allow(dead_code, reason = "freed by its drop")] crate::external::Memory),
    /// Pages bound one by one (`SparseBuffer`).
    #[cfg(feature = "sparse-residency-buffer")]
    Sparse(crate::sparse::Pages),
}

impl BufferMemory {
    pub(crate) fn allocation(&self) -> Option<&Allocation> {
        match self {
            BufferMemory::Allocation(allocation) => Some(allocation),
            #[allow(unreachable_patterns)]
            _ => None,
        }
    }

    /// Frees the memory, once nothing uses it.
    pub(crate) fn free(self) {
        match self {
            BufferMemory::Allocation(allocation) => {
                if let Err(err) = Ctx::allocator().free(allocation) {
                    tracing::error!(%err, "failed to free buffer allocation");
                }
            }
            // Their own drops free them.
            #[allow(unreachable_patterns)]
            _ => {}
        }
    }
}

impl BufferInternal {
    /// The addresses it covers.
    pub(crate) fn range(&self) -> Range<u64> {
        (self.address..self.address + self.num_bytes).into()
    }

    /// Whether its pages may be unbound, so commands don't claim it fully bound.
    pub(crate) fn is_sparse(&self) -> bool {
        #[cfg(feature = "sparse-residency-buffer")]
        if let BufferMemory::Sparse(_) = self.memory {
            return true;
        }
        false
    }
}

impl Drop for BufferInternal {
    /// Forgets its pending accesses and destroys the buffer and its memory, once no submission
    /// reaches it.
    fn drop(&mut self) {
        let handle = self.handle;
        let range = self.address..self.address + self.num_bytes;
        let memory = mem::replace(
            &mut self.memory,
            BufferMemory::Allocation(Allocation::default()),
        );
        lifetime::retire(move || {
            for index in INDICES.iter() {
                let pending = &mut lock(&index.state).pending;
                let outside = |access: &BufferAccess| !range.contains(&access.range.start);
                pending.buffer_reads.retain(|read| outside(&read.read));
                pending.buffer_writes.retain(|write| outside(&write.write));
            }
            unsafe { Ctx::device().destroy_buffer(handle, None) };
            memory.free();
        });
    }
}

impl<'a, T: Copy + 'static, M: Memory> From<&'a Buffer<T, M>> for BufferSlice<'a, T> {
    fn from(val: &'a Buffer<T, M>) -> Self {
        val.range(..)
    }
}

impl<T: Copy + 'static> Buffer<T, Gpu> {
    /// A device-local buffer of `len` elements the host can't access; fill it with commands, or
    /// make it filled with [`Uploader::init`]/[`Uploader::from_slice`]. On a `rebar` device it
    /// is allocated from [`Shared`]'s memory types where they have room (the BAR heap is all of
    /// VRAM there), so uploaders write it in place.
    #[validation_trace]
    pub fn new(len: usize) -> Result<Self> {
        Self::with_memory(Gpu, len)
    }
}

impl<T: Copy + 'static> Buffer<T, Gpu> {
    /// A buffer whose memory its owner bound and that it frees after it (external memory).
    #[cfg(all(feature = "external-memory", unix))]
    pub(crate) fn from_external(
        handle: vk::Buffer,
        address: u64,
        num_bytes: u64,
        memory: crate::external::Memory,
        live: Live,
    ) -> Self {
        Buffer::from_internal(
            BufferInternal {
                handle,
                address,
                num_bytes,
                memory: BufferMemory::External(memory),
                header: Header::new(live),
            },
            Gpu,
        )
    }
}

impl<T: Copy + 'static> Buffer<T, Upload> {
    /// A buffer of `len` elements the host writes and the GPU reads (`Upload`).
    #[validation_trace]
    pub fn upload(len: usize) -> Result<Self> {
        Self::with_memory(Upload, len)
    }
}

impl<T: Copy + 'static> Buffer<T, Readback> {
    /// A buffer of `len` elements the GPU writes and the host reads (`Readback`).
    #[validation_trace]
    pub fn readback(len: usize) -> Result<Self> {
        Self::with_memory(Readback, len)
    }
}

impl<T: Copy + 'static> Buffer<T, Shared> {
    /// A device-local buffer of `len` elements the host reads and writes (`Shared`);
    /// `MissingFeature` without `rebar`.
    #[validation_trace]
    pub fn shared(len: usize) -> Result<Self> {
        Self::with_memory(Shared, len)
    }
}

impl<T: Copy + 'static, M: Memory> Buffer<T, M> {
    pub(crate) fn from_internal(internal: BufferInternal, memory: M) -> Self {
        Buffer {
            internal: Arc::new(internal),
            memory,
            _type_marker: PhantomData,
        }
    }

    /// A buffer of `len` elements in `memory`, any location, a [`Dynamic`] one included.
    #[validation_trace]
    pub fn with_memory(memory: M, len: usize) -> Result<Self> {
        let num_bytes = len.checked_mul(size_of::<T>()).ok_or_else(|| {
            Error::other(format!(
                "{len} elements of {} bytes overflow",
                size_of::<T>()
            ))
        })?;
        Self::raw(memory, num_bytes as u64, Some(align_of::<T>() as u64))
    }

    /// A buffer of `num_bytes` (at least 1, a whole number of `T`s) in `memory`, its address
    /// aligned to `alignment` (a power of two) or what the device requires, whichever is more.
    #[validation_trace]
    pub fn raw(memory: M, num_bytes: u64, alignment: Option<u64>) -> Result<Self> {
        assert!(
            num_bytes.is_multiple_of(size_of::<T>() as u64),
            "{num_bytes} bytes aren't a whole number of {}-byte elements",
            size_of::<T>()
        );
        Self::create(USAGE, memory, num_bytes, alignment, Live::new())
    }

    /// A buffer of lava's own with `usage` instead of [`USAGE`] (a shader binding table, an
    /// acceleration structure's memory, a descriptor heap). Its slices must not reach commands
    /// that take address flags, which claim storage usage (`BufferSlice::address_flags`).
    pub(crate) fn with_flags(
        usage: vk::BufferUsageFlags,
        memory: M,
        num_bytes: u64,
        alignment: Option<u64>,
    ) -> Result<Self> {
        Self::create(usage, memory, num_bytes, alignment, Live::internal())
    }

    fn create(
        usage: vk::BufferUsageFlags,
        memory: M,
        num_bytes: u64,
        alignment: Option<u64>,
        live: Live,
    ) -> Result<Self> {
        const {
            assert!(
                size_of::<T>() > 0,
                "buffers of zero-sized elements have no length"
            )
        };
        if let Some(alignment) = alignment {
            assert!(
                alignment.is_power_of_two(),
                "a buffer alignment of {alignment} isn't a power of two"
            );
        }
        let location = memory.location();
        if location == Location::Gpu && Ctx::features_ref().rebar {
            // Mapped, so uploaders write it in place; plain device-local memory is the
            // fallback once the BAR heap is full.
            let shared = Self::with_memory_types(
                usage,
                memory,
                num_bytes,
                alignment,
                Location::Shared,
                &live,
            );
            if let Ok(buffer) = shared {
                return Ok(buffer);
            }
        }
        Self::with_memory_types(usage, memory, num_bytes, alignment, location, &live)
    }

    /// Like [`Self::with_flags`], allocated from the memory types of `location` instead of
    /// its own (`Gpu` buffers from `Shared` memory on `rebar` devices).
    pub(crate) fn with_memory_types(
        usage: vk::BufferUsageFlags,
        memory: M,
        num_bytes: u64,
        alignment: Option<u64>,
        location: Location,
        live: &Live,
    ) -> Result<Self> {
        if num_bytes == 0 {
            return Err(Error::other(
                "a buffer can't be empty: Vulkan needs at least one byte",
            ));
        }
        if location == Location::Shared && !Ctx::features_ref().rebar {
            return Err(crate::features::MissingFeature { feature: "rebar" }.into());
        }
        let mut flags = usage
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        if <crate::features::gate::AccelerationStructure as crate::features::On>::on() {
            flags |= vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;
        }
        let mut create_info = vk::BufferCreateInfo::default().size(num_bytes).usage(flags);
        // Concurrent, so buffers need no ownership transfers: they have no compression it
        // could cost, unlike images.
        let families: SmallVec<[u32; 3]> = Ctx::get().queues.iter().map(|f| f.index).collect();
        if families.len() > 1 {
            create_info = create_info
                .sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&families);
        }
        let buffer = unsafe { Ctx::device().create_buffer(&create_info, None)? };
        let mut dedicated = vk::MemoryDedicatedRequirements::default();
        let mut requirements = {
            let mut requirements = vk::MemoryRequirements2::default().push_next(&mut dedicated);
            unsafe {
                Ctx::device().get_buffer_memory_requirements2(
                    &vk::BufferMemoryRequirementsInfo2::default().buffer(buffer),
                    &mut requirements,
                )
            };
            requirements.memory_requirements
        };
        if let Some(a) = alignment {
            requirements.alignment = requirements.alignment.max(a);
        }
        requirements.memory_type_bits = location.memory_type_bits(
            requirements.memory_type_bits,
            &Ctx::physical_device().mem_properties,
        );
        let allocation_scheme = if dedicated.requires_dedicated_allocation == vk::TRUE
            || dedicated.prefers_dedicated_allocation == vk::TRUE
        {
            AllocationScheme::DedicatedBuffer(buffer)
        } else {
            AllocationScheme::GpuAllocatorManaged
        };

        let allocation = Ctx::allocator().allocate(&AllocationCreateDesc {
            name: "buffer",
            requirements,
            location: location.allocator_location(),
            linear: true,
            allocation_scheme,
        });
        let allocation = match allocation {
            Ok(allocation) => allocation,
            Err(err) => {
                unsafe { Ctx::device().destroy_buffer(buffer, None) };
                return Err(err.into());
            }
        };
        // Lava never flushes or invalidates mapped memory: gpu-allocator only maps coherent
        // memory types for host-visible locations.
        debug_assert!(
            !location.host_visible()
                || allocation.memory_properties().contains(
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
                ),
            "a {location:?} buffer got memory of {:?}",
            allocation.memory_properties()
        );

        if let Err(err) = unsafe {
            Ctx::device().bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
        } {
            unsafe { Ctx::device().destroy_buffer(buffer, None) };
            let _ = Ctx::allocator().free(allocation);
            return Err(err.into());
        }
        let addr_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
        let address = unsafe { Ctx::device().get_buffer_device_address(&addr_info) };
        Ok(Self::from_internal(
            BufferInternal {
                handle: buffer,
                address,
                num_bytes,
                memory: BufferMemory::Allocation(allocation),
                header: Header::new(live.clone()),
            },
            memory,
        ))
    }

    /// The same buffer named `name`, for debuggers, the validation layer (with debug utils,
    /// `Config::validation`) and allocator reports.
    pub fn named(mut self, name: &str) -> Self {
        self.set_name(name);
        self
    }

    pub(crate) fn set_name(&mut self, name: &str) {
        // Only an allocation nothing else holds yet can be renamed.
        if let Some(internal) = Arc::get_mut(&mut self.internal)
            && let BufferMemory::Allocation(allocation) = &mut internal.memory
            && let Err(err) = Ctx::allocator().rename_allocation(allocation, name)
        {
            tracing::warn!(%err, "failed to name a buffer's allocation");
        }
        crate::state::Functions::set_debug_name(name, self.internal.handle);
    }

    /// The device address of its first byte.
    pub fn address(&self) -> u64 {
        self.internal.address
    }

    pub fn size(&self) -> u64 {
        self.internal.num_bytes
    }

    /// How many `T`s it holds.
    pub fn len(&self) -> usize {
        (self.size() / size_of::<T>() as u64) as usize
    }

    /// Whether it holds no `T` (only a `raw` buffer of fewer bytes than one can).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Where its memory lives.
    pub fn location(&self) -> Location {
        self.memory.location()
    }

    pub fn memory(&self) -> M {
        self.memory
    }

    /// Its mapping, where the memory has one: host-visible locations, `Gpu` buffers of `rebar`
    /// devices (allocated from `Shared`'s memory types) and `Gpu` memory in a host-visible
    /// memory type otherwise (integrated GPUs).
    pub(crate) fn mapping(&self) -> Option<*mut T> {
        let ptr = (self.internal.memory.allocation()?.mapped_ptr()?.as_ptr()).cast::<T>();
        assert!(
            ptr.is_aligned(),
            "the mapping at {ptr:?} isn't aligned to {} bytes for its elements",
            align_of::<T>()
        );
        Some(ptr)
    }

    /// Its mapping where the memory is coherent too, which lava writes without flushing.
    pub(crate) fn coherent_mapping(&self) -> Option<*mut T> {
        let coherent = (self.internal.memory.allocation()?.memory_properties())
            .contains(vk::MemoryPropertyFlags::HOST_COHERENT);
        coherent.then(|| self.mapping()).flatten()
    }

    /// The same buffer of `B`s; panics unless its size is a multiple of `B`'s and its address
    /// aligned to `B`.
    pub fn cast<B: Copy + 'static>(self) -> Buffer<B, M> {
        slice::cast_check::<B>(self.address(), self.size());
        Buffer {
            internal: self.internal,
            memory: self.memory,
            _type_marker: PhantomData,
        }
    }

    /// Grows the buffer to at least `len` elements (at least twice its length, so repeated
    /// growth stays linear), keeping its contents: a new buffer in the same memory location,
    /// which `cmd` fills with a copy of the old one before its later commands. Returns whether
    /// it grew; a buffer of `len` or more elements stays as it is.
    ///
    /// Only the old buffer's addresses can outlive this call (the `&mut` borrow ends its
    /// slices): addresses stored in other buffers or push constants stay valid until the old
    /// buffer is destroyed, once no frame reaches it, but point to the old memory, so rewrite
    /// them. Host writes to the old mapping after the copy is recorded are lost.
    #[validation_trace]
    pub fn reserve<K: crate::queue::QueueKind>(
        &mut self,
        len: usize,
        cmd: &mut crate::command_buffer::CommandBuffer<'_, K>,
    ) -> Result<bool> {
        let old = self.len();
        if len <= old {
            return Ok(false);
        }
        let grown = Buffer::with_memory(self.memory, len.max(old.saturating_mul(2)))?;
        cmd.copy_buffer(
            self.byte_range(..),
            grown.byte_range(..self.size() as usize),
        );
        // Retired, so destroyed once this recording's submission (and every other in flight)
        // completed.
        drop(mem::replace(self, grown));
        Ok(true)
    }

    /// The same buffer with its location as a runtime value.
    pub fn erase(self) -> Buffer<T, Dynamic> {
        Buffer {
            memory: Dynamic(self.memory.location()),
            internal: self.internal,
            _type_marker: PhantomData,
        }
    }
}

impl<T: Copy + 'static, M: HostVisible> Buffer<T, M> {
    #[track_caller]
    fn mapped(&self) -> *mut T {
        self.memory.host_check();
        self.mapping().expect("host-visible memory is mapped")
    }

    /// The timeline and value of a submitted GPU write of the buffer that hasn't completed.
    /// Polls the timelines it finds one on. Writes not submitted yet aren't seen: the `&mut`
    /// borrow of `host`/`host_mut` keeps new recordings from reaching the buffer.
    fn incomplete_write(&self) -> Option<(usize, u64)> {
        let range = self.internal.range();
        (0..INDICES.len()).find_map(|i| {
            let value = incomplete_write(
                lock(&INDICES[i].writes).iter().copied(),
                range,
                completed(i),
            )?;
            (!poll(i).is_ok_and(|done| done >= value)).then_some((i, value))
        })
    }

    #[track_caller]
    fn checked(&self) -> *mut T {
        let ptr = self.mapped();
        if let Some((timeline, value)) = self.incomplete_write() {
            panic!(
                "the GPU may still write this buffer: wait for value {value} of timeline \
                 {timeline} (the `Wait` of the submission that writes it) before the host \
                 accesses it, or use `try_host`"
            );
        }
        ptr
    }

    /// Writes `data` from element `first` on, checked like [`Self::host_mut`]. Only writes, so
    /// it takes any `NoUninit` element (structs with image handles among them, which aren't
    /// `AnyBitPattern`), where `host_mut` needs elements of any bit pattern.
    #[track_caller]
    pub fn host_write(&mut self, first: usize, data: &[T])
    where
        T: NoUninit,
    {
        let end = first.checked_add(data.len());
        assert!(
            end.is_some_and(|end| end <= self.len()),
            "writing {} elements from {first} on into a buffer of {}",
            data.len(),
            self.len()
        );
        let ptr = self.checked();
        // SAFETY: inside the mapping, which no reference covers while `self` is borrowed.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.add(first), data.len()) };
    }

    /// The mapping as a pointer, for elements owned one at a time through a shared reference.
    /// A dereference must own its element, with no `host`/`host_mut` borrow live and no GPU
    /// write of it in flight, and a read of an element needs it to have been written as a `T`.
    pub fn host_ptr(&self) -> *mut T {
        self.mapped()
    }
}

/// Host reads see whatever bytes the GPU (or nothing) wrote, so they take elements of any bit
/// pattern.
impl<T: AnyBitPattern, M: HostVisible> Buffer<T, M> {
    /// The mapping as a slice, or `None` while a submitted GPU write of the buffer hasn't
    /// completed.
    pub fn try_host(&mut self) -> Option<&[T]> {
        let ptr = self.mapped();
        match self.incomplete_write() {
            Some(_) => None,
            None => Some(unsafe { std::slice::from_raw_parts(ptr, self.len()) }),
        }
    }

    /// The mapping as a slice; panics while a submitted GPU write of the buffer hasn't
    /// completed (`try_host` returns `None` then). Writes a pass makes through an address
    /// without declaring them in its `Accesses` aren't seen.
    #[track_caller]
    pub fn host(&mut self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.checked(), self.len()) }
    }

    /// The mapping as a mutable slice, checked like [`Self::host`]. A frame in flight may
    /// still read it: per-frame data goes into one range per frame slot, or one buffer per slot
    /// ([`crate::queue::PerFrame`]). Elements that aren't `AnyBitPattern` use [`Self::host_write`].
    #[track_caller]
    pub fn host_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.checked(), self.len()) }
    }

    /// Copies its first `out.len()` elements into `out`, without forming a reference to the
    /// mapping or checking for GPU writes in flight, whose data it may tear.
    #[track_caller]
    pub fn read_into(&self, out: &mut [T]) {
        assert!(
            out.len() <= self.len(),
            "reading {} elements from a buffer of {}",
            out.len(),
            self.len()
        );
        unsafe { std::ptr::copy_nonoverlapping(self.mapped(), out.as_mut_ptr(), out.len()) };
    }

    /// A copy of element `index`, read like [`Self::read_into`].
    #[track_caller]
    pub fn get(&self, index: usize) -> T {
        assert!(
            index < self.len(),
            "element {index} of a buffer of {}",
            self.len()
        );
        unsafe { self.mapped().add(index).read() }
    }
}

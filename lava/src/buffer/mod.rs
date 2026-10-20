//! Typed GPU buffers backed by gpu-allocator memory in a location their type names (`memory`; `Gpu` from `Shared`'s memory types on `rebar` devices, so uploaders write it in place), each with every usage lava's commands need; host access to host-visible ones, checked against GPU writes still in flight; the staging rings and uploaders (`staging`); casts, and their destruction once no frame can reach them
use std::{
    fmt::Debug,
    marker::PhantomData,
    mem::{self, ManuallyDrop},
};

use crate::error::{Error, Result};
use crate::vk::{self};
use bytemuck::Pod;
use smallvec::SmallVec;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use lava_macros::validation_trace;

use crate::{
    buffer::slice::BufferSlice,
    lifetime,
    command_buffer::{BufferAccess, pending::incomplete_write},
    queue::{lock, timeline::{INDICES, completed, poll}},
    state::Ctx,
};

pub mod memory;
pub mod slice;
pub(crate) mod staging;

pub use memory::{Dynamic, Gpu, HostVisible, Location, Memory, Readback, Shared, StaticMemory, Upload};
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

/// A buffer of `T`s (`u8` when untyped) in memory `M` ([`Gpu`] by default). Only
/// host-visible memory has host access:
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
#[derive(Debug)]
pub struct Buffer<T: Copy + Pod, M: Memory = Gpu> {
    pub(crate) handle: vk::Buffer,
    address: u64,
    num_bytes: u64,
    allocation: Allocation,
    memory: M,
    _type_marker: PhantomData<T>,
}

impl<T: Copy + Pod, M: Memory> Drop for Buffer<T, M> {
    fn drop(&mut self) {
        let (handle, allocation) = (self.handle, mem::take(&mut self.allocation));
        let range = self.address..self.address + self.num_bytes;
        staging::forget_buffer(range.clone());
        lifetime::retire(move || {
            for index in INDICES.iter() {
                let pending = &mut lock(&index.state).pending;
                let outside = |access: &BufferAccess| !range.contains(&access.range.start);
                pending.buffer_reads.retain(|read| outside(&read.read));
                pending.buffer_writes.retain(|write| outside(&write.write));
            }
            unsafe { Ctx::device().destroy_buffer(handle, None) };
            if let Err(err) = Ctx::allocator().free(allocation) {
                tracing::error!(%err, "failed to free buffer allocation");
            }
        });
    }
}

impl<'a, T: Copy + Pod, M: Memory> Into<BufferSlice<'a, T>> for &'a Buffer<T, M> {
    fn into(self) -> BufferSlice<'a, T> {
        self.range(..)
    }
}

impl<T: Copy + Pod> Buffer<T, Gpu> {
    /// A device-local buffer of `len` elements the host can't access; fill it with commands, or
    /// make it filled with [`Uploader::init`]/[`Uploader::from_slice`]. On a `rebar` device it
    /// is allocated from [`Shared`]'s memory types where they have room (the BAR heap is all of
    /// VRAM there), so uploaders write it in place.
    #[validation_trace]
    pub fn new(len: usize) -> Result<Self> {
        Self::with_memory(Gpu, len)
    }
}

impl<T: Copy + Pod> Buffer<T, Upload> {
    /// A buffer of `len` elements the host writes and the GPU reads (`Upload`).
    #[validation_trace]
    pub fn upload(len: usize) -> Result<Self> {
        Self::with_memory(Upload, len)
    }
}

impl<T: Copy + Pod> Buffer<T, Readback> {
    /// A buffer of `len` elements the GPU writes and the host reads (`Readback`).
    #[validation_trace]
    pub fn readback(len: usize) -> Result<Self> {
        Self::with_memory(Readback, len)
    }
}

impl<T: Copy + Pod> Buffer<T, Shared> {
    /// A device-local buffer of `len` elements the host reads and writes (`Shared`);
    /// `MissingFeature` without `rebar`.
    #[validation_trace]
    pub fn shared(len: usize) -> Result<Self> {
        Self::with_memory(Shared, len)
    }
}

impl<T: Copy + Pod, M: Memory> Buffer<T, M> {
    /// A buffer of `len` elements in `memory`, any location, a [`Dynamic`] one included.
    #[validation_trace]
    pub fn with_memory(memory: M, len: usize) -> Result<Self> {
        let num_bytes = len.checked_mul(size_of::<T>()).ok_or_else(|| {
            Error::message(format!("{len} elements of {} bytes overflow", size_of::<T>()))
        })?;
        Self::raw(memory, num_bytes as u64, Some(align_of::<T>() as u32))
    }

    /// A buffer of `num_bytes` (at least 1) in `memory`, aligned to `alignment` or what the
    /// device requires.
    #[validation_trace]
    pub fn raw(memory: M, num_bytes: u64, alignment: Option<u32>) -> Result<Self> {
        Self::with_flags(USAGE, memory, num_bytes, alignment)
    }

    /// A buffer of lava's own with `usage` instead of [`USAGE`] (a shader binding table, an
    /// acceleration structure's memory, a descriptor heap). Its slices must not reach commands
    /// that take address flags, which claim storage usage (`BufferSlice::address_flags`).
    pub(crate) fn with_flags(
        usage: vk::BufferUsageFlags,
        memory: M,
        num_bytes: u64,
        alignment: Option<u32>,
    ) -> Result<Self> {
        let location = memory.location();
        if location == Location::Gpu && Ctx::features().rebar {
            // Mapped, so uploaders write it in place; plain device-local memory is the
            // fallback once the BAR heap is full.
            let shared = Self::with_memory_types(usage, memory, num_bytes, alignment, Location::Shared);
            if let Ok(buffer) = shared {
                return Ok(buffer);
            }
        }
        Self::with_memory_types(usage, memory, num_bytes, alignment, location)
    }

    /// Like [`Self::with_flags`], allocated from the memory types of `location` instead of
    /// its own (`Gpu` buffers from `Shared` memory on `rebar` devices).
    pub(crate) fn with_memory_types(
        usage: vk::BufferUsageFlags,
        memory: M,
        num_bytes: u64,
        alignment: Option<u32>,
        location: Location,
    ) -> Result<Self> {
        if num_bytes == 0 {
            return Err(Error::message("a buffer can't be empty: Vulkan needs at least one byte"));
        }
        if location == Location::Shared && !Ctx::features().rebar {
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
            requirements.alignment = requirements.alignment.max(a as u64);
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
        Ok(Self {
            _type_marker: PhantomData,
            address,
            num_bytes,
            allocation,
            memory,
            handle: buffer,
        })
    }

    /// The device address of its first byte.
    pub fn address(&self) -> u64 {
        self.address
    }

    pub fn handle(&self) -> vk::Buffer {
        self.handle
    }

    pub fn size(&self) -> u64 {
        self.num_bytes
    }

    pub fn len(&self) -> usize {
        (self.size() / size_of::<T>() as u64) as usize
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
        let ptr = self.allocation.mapped_ptr()?.as_ptr().cast::<T>();
        assert!(
            ptr.is_aligned(),
            "the mapping at {ptr:?} isn't aligned to {} bytes for its elements",
            align_of::<T>()
        );
        Some(ptr)
    }

    /// Its mapping where the memory is coherent too, which lava writes without flushing.
    pub(crate) fn coherent_mapping(&self) -> Option<*mut T> {
        let coherent = (self.allocation.memory_properties())
            .contains(vk::MemoryPropertyFlags::HOST_COHERENT);
        coherent.then(|| self.mapping()).flatten()
    }

    /// The same buffer of `B`s; panics unless its size is a multiple of `B`'s and its address
    /// aligned to `B`.
    pub fn cast<B: Copy + Pod>(self) -> Buffer<B, M> {
        slice::cast_check::<B>(self.address, self.num_bytes);
        // `self` is left without memory and never dropped.
        let mut this = ManuallyDrop::new(self);
        Buffer {
            handle: this.handle,
            address: this.address,
            allocation: mem::take(&mut this.allocation),
            num_bytes: this.num_bytes,
            memory: this.memory,
            _type_marker: PhantomData,
        }
    }

    /// The same buffer with its location as a runtime value.
    pub fn erase(self) -> Buffer<T, Dynamic> {
        let mut this = ManuallyDrop::new(self);
        Buffer {
            handle: this.handle,
            address: this.address,
            allocation: mem::take(&mut this.allocation),
            num_bytes: this.num_bytes,
            memory: Dynamic(this.memory.location()),
            _type_marker: PhantomData,
        }
    }
}

impl<T: Copy + Pod, M: HostVisible> Buffer<T, M> {
    #[track_caller]
    fn mapped(&self) -> *mut T {
        self.memory.host_check();
        self.mapping().expect("host-visible memory is mapped")
    }

    /// The timeline and value of a submitted GPU write of the buffer that hasn't completed.
    /// Polls the timelines it finds one on. Writes not submitted yet aren't seen: the `&mut`
    /// borrow of `host`/`host_mut` keeps new recordings from reaching the buffer.
    fn incomplete_write(&self) -> Option<(usize, u64)> {
        let range = (self.address..self.address + self.num_bytes).into();
        (0..INDICES.len()).find_map(|i| {
            let value = incomplete_write(lock(&INDICES[i].writes).iter().copied(), range, completed(i))?;
            (poll(i) < value).then_some((i, value))
        })
    }

    /// The mapping as a slice, or `None` while a submitted GPU write of the buffer hasn't
    /// completed.
    pub fn try_host(&mut self) -> Option<&[T]> {
        let ptr = self.mapped();
        match self.incomplete_write() {
            Some(_) => None,
            None => Some(unsafe { std::slice::from_raw_parts(ptr, self.len()) }),
        }
    }

    #[track_caller]
    fn checked(&self) -> *mut T {
        let ptr = self.mapped();
        if let Some((timeline, value)) = self.incomplete_write() {
            panic!(
                "the GPU may still write this buffer: wait for value {value} of timeline                  {timeline} (the `Wait` of the submission that writes it) before the host                  accesses it, or use `try_host`"
            );
        }
        ptr
    }

    /// The mapping as a slice; panics while a submitted GPU write of the buffer hasn't
    /// completed (`try_host` returns `None` then). Writes a pass makes through an address
    /// without declaring them in its `Accesses` aren't seen.
    #[track_caller]
    pub fn host(&mut self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.checked(), self.len()) }
    }

    /// The mapping as a mutable slice, checked like [`Self::host`]. A frame in flight may
    /// still read it: per-frame data goes into one range per frame slot.
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
        assert!(index < self.len(), "element {index} of a buffer of {}", self.len());
        unsafe { self.mapped().add(index).read() }
    }

    /// The mapping as a pointer, for elements owned one at a time through a shared reference.
    /// A dereference must own its element, with no `host`/`host_mut` borrow live and no GPU
    /// write of it in flight.
    pub fn host_ptr(&self) -> *mut T {
        self.mapped()
    }
}

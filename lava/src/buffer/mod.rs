//! Typed GPU buffers backed by gpu-allocator memory, each with every usage lava's commands need, their host access (`host`, `host_mut`, `host_ptr`), casts, and their destruction once no frame can reach them
use std::{
    fmt::Debug,
    marker::PhantomData,
    mem::{self, ManuallyDrop},
    ops::{Index, IndexMut},
};

use crate::error::{Error, Result};
use crate::vk::{self};
use bytemuck::Pod;
use smallvec::SmallVec;
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme},
};
use lava_macros::validation_trace;

use crate::{
    buffer::slice::BufferSlice,
    lifetime,
    command_buffer::BufferAccess,
    queue::{lock, timeline::INDICES},
    state::Ctx,
};

pub mod slice;

/// The usages of every buffer an application creates: everything lava's commands bind a buffer
/// as (lava binds no vertex or uniform buffers). Giving every buffer every usage made no
/// measurable difference against one usage per buffer (2026-10-19, Arc B390 with Mesa's ANV:
/// the same memory requirements for every buffer, the same benchmark view and recording times).
pub const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
        | vk::BufferUsageFlags::INDEX_BUFFER.as_raw()
        | vk::BufferUsageFlags::INDIRECT_BUFFER.as_raw(),
);

/// A buffer of `T`s (`u8` when untyped).
#[derive(Debug)]
pub struct Buffer<T: Copy + Pod> {
    pub(crate) handle: vk::Buffer,
    address: u64,
    num_bytes: u64,
    allocation: Allocation,
    _type_marker: PhantomData<T>,
}

impl<T: Copy + Pod> Index<usize> for Buffer<T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        &self.host()[index]
    }
}

impl<T: Copy + Pod> IndexMut<usize> for Buffer<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.host_mut()[index]
    }
}

impl<T: Copy + Pod> Drop for Buffer<T> {
    fn drop(&mut self) {
        let (handle, allocation) = (self.handle, mem::take(&mut self.allocation));
        let range = self.address..self.address + self.num_bytes;
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

impl<'a, T: Copy + Pod> Into<BufferSlice<'a, T>> for &'a Buffer<T> {
    fn into(self) -> BufferSlice<'a, T> {
        self.range(..)
    }
}

impl<T: Copy + Pod> Buffer<T> {
    /// A buffer of `size` elements; `cpu_writable` ones are mapped on every device, the others
    /// only with `rebar`.
    #[validation_trace]
    pub fn new(size: usize, cpu_writable: bool) -> Result<Self> {
        let num_bytes = size.checked_mul(size_of::<T>()).ok_or_else(|| {
            Error::message(format!("{size} elements of {} bytes overflow", size_of::<T>()))
        })?;
        Self::raw(cpu_writable, num_bytes as u64, Some(align_of::<T>() as u32))
    }

    /// A buffer of `num_bytes` (at least 1), aligned to `alignment` or what the device requires.
    #[validation_trace]
    pub fn raw(cpu_writable: bool, num_bytes: u64, alignment: Option<u32>) -> Result<Self> {
        Self::with_flags(USAGE, cpu_writable, num_bytes, alignment)
    }

    /// A buffer of lava's own with `usage` instead of [`USAGE`] (a shader binding table, an
    /// acceleration structure's memory, a descriptor heap). Its slices must not reach commands
    /// that take address flags, which claim storage usage (`BufferSlice::address_flags`).
    pub(crate) fn with_flags(
        usage: vk::BufferUsageFlags,
        cpu_writable: bool,
        num_bytes: u64,
        alignment: Option<u32>,
    ) -> Result<Self> {
        if num_bytes == 0 {
            return Err(Error::message("a buffer can't be empty: Vulkan needs at least one byte"));
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
        let allocation_scheme = if dedicated.requires_dedicated_allocation == vk::TRUE
            || dedicated.prefers_dedicated_allocation == vk::TRUE
        {
            AllocationScheme::DedicatedBuffer(buffer)
        } else {
            AllocationScheme::GpuAllocatorManaged
        };

        let allocation = {
            let mut allocator = Ctx::allocator();
            (*allocator).allocate(&AllocationCreateDesc {
                name: "buffer",
                requirements,
                location: if cpu_writable || Ctx::features().rebar {
                    MemoryLocation::CpuToGpu
                } else {
                    MemoryLocation::GpuOnly
                },
                linear: true,
                allocation_scheme,
            })
        };
        let allocation = match allocation {
            Ok(allocation) => allocation,
            Err(err) => {
                unsafe { Ctx::device().destroy_buffer(buffer, None) };
                return Err(err.into());
            }
        };

        unsafe {
            Ctx::device().bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?
        };
        let addr_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
        let address = unsafe { Ctx::device().get_buffer_device_address(&addr_info) };
        Ok(Self {
            _type_marker: PhantomData,
            address,
            num_bytes,
            allocation: allocation,
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

    /// Whether the host can access it: it was created `cpu_writable`, or the device has
    /// `rebar`.
    pub fn is_mapped(&self) -> bool {
        self.allocation.mapped_ptr().is_some()
    }

    fn mapped(&self) -> *mut T {
        let ptr = self
            .allocation
            .mapped_ptr()
            .expect("the host can only access mapped buffers (`is_mapped`): create it `cpu_writable`")
            .as_ptr()
            .cast::<T>();
        assert!(
            ptr.is_aligned(),
            "the mapping at {ptr:?} isn't aligned to {} bytes for its elements",
            align_of::<T>()
        );
        ptr
    }

    /// The mapping as a slice; panics unless `is_mapped`. GPU writes are visible once their
    /// submission completed.
    pub fn host(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.mapped(), self.len()) }
    }

    /// The mapping as a mutable slice; panics unless `is_mapped`. The GPU may still read it
    /// for a frame in flight.
    pub fn host_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.mapped(), self.len()) }
    }

    /// The mapping as a pointer, for elements owned one at a time through a shared reference;
    /// panics unless `is_mapped`. A dereference must own its element, with no `host`/`host_mut`
    /// borrow live.
    pub fn host_ptr(&self) -> *mut T {
        self.mapped()
    }

    /// The same buffer of `B`s; panics unless its size is a multiple of `B`'s and its address
    /// aligned to `B`.
    pub fn cast<B: Copy + Pod>(self) -> Buffer<B> {
        slice::cast_check::<B>(self.address, self.num_bytes);
        // `self` is left without memory and never dropped.
        let mut this = ManuallyDrop::new(self);
        Buffer {
            handle: this.handle,
            address: this.address,
            allocation: mem::take(&mut this.allocation),
            num_bytes: this.num_bytes,
            _type_marker: PhantomData,
        }
    }
}

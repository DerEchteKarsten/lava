//! Typed, usage-tagged GPU buffers backed by gpu-allocator memory, their host access (`host`, `host_mut`, `host_ptr`), type erasure, and their destruction once no frame can reach them
use std::{
    fmt::Debug,
    marker::PhantomData,
    mem::{self, ManuallyDrop},
    ops::{Index, IndexMut},
};

use crate::error::Result;
use crate::vk::{self};
use bytemuck::Pod;
use smallvec::SmallVec;
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme},
};
use lava_macros::validation_trace;

use crate::{
    buffer::{
        slice::BufferSlice,
        usage::{BufferUsage, StaticUsage, Storage},
    },
    lifetime,
    command_buffer::BufferAccess,
    queue::{lock, timeline::INDICES},
    state::Ctx,
};

pub mod slice;
pub mod usage;

/// A buffer of `T`s (`u8` when untyped). A static usage type is checked at compile time, a
/// `usage::Dynamic` value at runtime.
#[derive(Debug)]
pub struct Buffer<T: Copy + Pod, U: BufferUsage = Storage> {
    pub(crate) handle: vk::Buffer,
    address: u64,
    num_bytes: u64,
    allocation: Allocation,
    usage: U,
    _type_marker: PhantomData<T>,
}

/// A buffer of bytes whose usage is a runtime value (`Buffer::erase`).
pub type DynamicBuffer = Buffer<u8, usage::Dynamic>;

impl<T: Copy + Pod, U: BufferUsage> Index<usize> for Buffer<T, U> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        &self.host()[index]
    }
}

impl<T: Copy + Pod, U: BufferUsage> IndexMut<usize> for Buffer<T, U> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.host_mut()[index]
    }
}

impl<T: Copy + Pod, U: BufferUsage> Drop for Buffer<T, U> {
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

impl<'a, T: Copy + Pod, U: BufferUsage> Into<BufferSlice<'a, T, U>> for &'a Buffer<T, U> {
    fn into(self) -> BufferSlice<'a, T, U> {
        self.range(..)
    }
}

impl<T: Copy + Pod, U: StaticUsage> Buffer<T, U> {
    /// A buffer of `size` elements; `cpu_writable` ones are mapped on every device, the others
    /// only with `rebar`.
    #[validation_trace]
    pub fn new(size: usize, cpu_writable: bool) -> Result<Self> {
        Self::with_usage(U::default(), size, cpu_writable)
    }
}

impl<T: Copy + Pod, U: BufferUsage> Buffer<T, U> {
    /// Like `new`, for any usage, `Dynamic` ones included.
    #[validation_trace]
    pub fn with_usage(usage: U, size: usize, cpu_writable: bool) -> Result<Self> {
        Self::raw(usage, cpu_writable, (size * size_of::<T>()) as u64, None)
    }

    /// A buffer of `num_bytes`, aligned to `alignment` or what the device requires.
    #[validation_trace]
    pub fn raw(
        usage: U,
        cpu_writable: bool,
        num_bytes: u64,
        alignment: Option<u32>,
    ) -> Result<Self> {
        let mut flags = usage.flags()
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
            usage,
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

    pub fn usage(&self) -> U {
        self.usage
    }

    /// Whether the host can access it: it was created `cpu_writable`, or the device has
    /// `rebar`.
    pub fn is_mapped(&self) -> bool {
        self.allocation.mapped_ptr().is_some()
    }

    /// Panics unless the usage has every flag of `required`; does nothing for a static usage.
    pub fn usage_check(&self, required: vk::BufferUsageFlags) {
        self.usage.check(required);
    }

    fn mapped(&self) -> *mut T {
        let ptr = self
            .allocation
            .mapped_ptr()
            .expect("the host can only access mapped buffers (`is_mapped`): create it `cpu_writable`")
            .as_ptr()
            .cast::<T>();
        debug_assert!(ptr.is_aligned());
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

    pub fn cast<B: Copy + Pod>(self) -> Buffer<B, U> {
        let usage = self.usage;
        ManuallyDrop::new(self).take(usage)
    }

    /// The same buffer as bytes with its usage as a runtime value.
    pub fn erase(self) -> DynamicBuffer {
        let usage = usage::Dynamic(self.usage.flags());
        ManuallyDrop::new(self).take(usage)
    }

    /// Leaves `self` without memory, so `self` must not be dropped.
    fn take<B: Copy + Pod, NU: BufferUsage>(&mut self, usage: NU) -> Buffer<B, NU> {
        Buffer {
            handle: self.handle,
            address: self.address,
            allocation: mem::take(&mut self.allocation),
            num_bytes: self.num_bytes,
            usage,
            _type_marker: PhantomData,
        }
    }
}

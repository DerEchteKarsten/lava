//! Typed, usage-tagged GPU buffers backed by gpu-allocator memory, their host access (`host`, `host_mut`, `host_ptr`), type erasure and drop hook
use ash::vk::TaggedStructure;
use std::{
    fmt::Debug,
    marker::PhantomData,
    mem::{self, ManuallyDrop},
    ops::{Index, IndexMut},
};

use crate::error::Result;
use ash::vk::{self};
use bytemuck::Pod;
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
    keepalive::KeepAlive,
    state::Ctx,
};

pub mod slice;
pub mod usage;

/// A buffer of `T`s (`u8` when untyped). Its usage is a type (`usage::Storage`), which the
/// command buffer's bounds check at compile time, or a `usage::Dynamic` value, which it checks
/// at runtime (`usage_check`).
///
/// Dropping it destroys it once nothing keeps it alive any more: its slices, and the frames in
/// flight that used it, each count (`keepalive`).
#[derive(Debug)]
pub struct Buffer<T: Copy + Pod, U: BufferUsage = Storage> {
    pub handle: vk::Buffer,
    pub address: u64,
    pub allocation: Allocation,
    num_bytes: u64,
    usage: U,
    owner: ManuallyDrop<KeepAlive>,
    _type_marker: PhantomData<T>,
}

/// What destroying a buffer takes, parked while something still counts it.
struct RawBuffer {
    handle: vk::Buffer,
    allocation: Allocation,
}

impl Drop for RawBuffer {
    fn drop(&mut self) {
        unsafe { Ctx::device().destroy_buffer(self.handle, None) };
        if let Err(err) = Ctx::allocator().free(mem::take(&mut self.allocation)) {
            tracing::error!(%err, "failed to free buffer allocation");
        }
    }
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
        let owner = unsafe { ManuallyDrop::take(&mut self.owner) };
        owner.release(RawBuffer {
            handle: self.handle,
            allocation: mem::take(&mut self.allocation),
        });
    }
}

impl<T: Copy + Pod, U: BufferUsage> From<&Buffer<T, U>> for BufferSlice<T, U> {
    fn from(buffer: &Buffer<T, U>) -> Self {
        buffer.range(..)
    }
}

impl<T: Copy + Pod, U: StaticUsage> Buffer<T, U> {
    /// A buffer of `size` elements.
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
        let flags = usage.flags()
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let create_info = vk::BufferCreateInfo::default().size(num_bytes).usage(flags);
        let buffer = unsafe { Ctx::device().create_buffer(&create_info, None)? };
        let mut dedicated = vk::MemoryDedicatedRequirements::default();
        let mut requirements = {
            let mut requirements = vk::MemoryRequirements2::default().push(&mut dedicated);
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
            allocation,
            handle: buffer,
            owner: ManuallyDrop::new(KeepAlive::new()),
        })
    }

    /// A count of the buffer that keeps it alive, for a frame that reaches it in a way its
    /// commands don't name (`Frame::uses`): through an address stored in another buffer.
    pub fn keep_alive(&self) -> KeepAlive {
        KeepAlive::clone(&self.owner)
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

    /// Panics unless the usage has every flag of `required`; does nothing for a static usage.
    pub fn usage_check(&self, required: vk::BufferUsageFlags) {
        self.usage.check(required);
    }

    /// Whether the memory is host-visible: every buffer with `cpu_writable`, and every buffer
    /// on a device with `rebar`.
    pub fn is_mapped(&self) -> bool {
        self.allocation.mapped_ptr().is_some()
    }

    fn mapped(&self) -> *mut T {
        let ptr = self
            .allocation
            .mapped_ptr()
            .expect("buffer is not host-mapped")
            .as_ptr()
            .cast::<T>();
        debug_assert!(ptr.is_aligned());
        ptr
    }

    /// The mapping as a slice. Panics unless the buffer is mapped.
    ///
    /// What the GPU wrote is there once its submission's fence passed: every submission ends
    /// with a barrier that makes its writes host-visible.
    pub fn host(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.mapped(), self.len()) }
    }

    /// The mapping as a mutable slice. Panics unless the buffer is mapped.
    ///
    /// The GPU may still read the buffer for a frame in flight: write only that frame slot's
    /// part of a ring, or a buffer no pending frame uses.
    pub fn host_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.mapped(), self.len()) }
    }

    /// The mapping as a pointer, for buffers whose elements are owned one at a time through a
    /// shared reference (slots handed out by an allocator). Panics unless the buffer is mapped.
    ///
    /// Dereferencing it is `unsafe`: the caller must own the element it accesses, and nothing
    /// may call `host`/`host_mut` on the buffer while such elements are live.
    pub fn host_ptr(&self) -> *mut T {
        self.mapped()
    }

    pub fn cast<B: Copy + Pod>(self) -> Buffer<B, U> {
        let usage = self.usage;
        ManuallyDrop::new(self).take(usage)
    }

    /// The same buffer as bytes with its usage as a runtime value, e.g. to keep buffers of
    /// different types together. Commands check the usage at runtime.
    pub fn erase(self) -> DynamicBuffer {
        let usage = usage::Dynamic(self.usage.flags());
        ManuallyDrop::new(self).take(usage)
    }

    /// Moves the buffer into one of another element type and usage; `self` must not be
    /// dropped after.
    fn take<B: Copy + Pod, NU: BufferUsage>(&mut self, usage: NU) -> Buffer<B, NU> {
        Buffer {
            handle: self.handle,
            address: self.address,
            allocation: mem::take(&mut self.allocation),
            num_bytes: self.num_bytes,
            usage,
            owner: ManuallyDrop::new(unsafe { ManuallyDrop::take(&mut self.owner) }),
            _type_marker: PhantomData,
        }
    }
}

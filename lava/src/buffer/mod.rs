//! Typed, usage-tagged GPU buffers backed by host-mapped gpu-allocator memory, their type erasure and drop hook
use std::{
    fmt::Debug,
    marker::PhantomData,
    mem::{self, ManuallyDrop},
    ops::{Index, IndexMut},
    sync::OnceLock,
};

use crate::error::{Error, Result};
use ash::vk::{self};
use bytemuck::Pod;
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc},
};
use lava_macros::validation_trace;

use crate::{
    buffer::{
        slice::BufferSlice,
        usage::{BufferUsage, StaticUsage, Storage},
    },
    state::Ctx,
};

pub mod slice;
pub mod usage;

/// A buffer of `T`s (`u8` when untyped). Its usage is a type (`usage::Storage`), which the
/// command buffer's bounds check at compile time, or a `usage::Dynamic` value, which it checks
/// at runtime (`usage_check`).
#[derive(Debug)]
pub struct Buffer<T: Copy + Pod, U: BufferUsage = Storage> {
    pub handle: vk::Buffer,
    pub address: u64,
    pub allocation: Allocation,
    num_bytes: u64,
    usage: U,
    _type_marker: PhantomData<T>,
}

/// A buffer of bytes whose usage is a runtime value (`Buffer::erase`).
pub type DynamicBuffer = Buffer<u8, usage::Dynamic>;

static ON_DROP: OnceLock<fn(&DynamicBuffer)> = OnceLock::new();

/// Sets the function every dropped buffer calls with itself, erased, right before it is
/// destroyed. It can be set once.
pub fn on_drop(hook: fn(&DynamicBuffer)) -> Result<()> {
    ON_DROP
        .set(hook)
        .map_err(|_| Error::message("the buffer drop hook is already set"))
}

impl<T: Copy + Pod, U: BufferUsage> Index<usize> for Buffer<T, U> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        unsafe { &*self.range(..).element_ptr(index) }
    }
}

impl<T: Copy + Pod, U: BufferUsage> IndexMut<usize> for Buffer<T, U> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { &mut *self.range(..).element_ptr(index) }
    }
}

impl<'a, T: Copy + Pod, U: BufferUsage> Index<usize> for BufferSlice<'a, T, U> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        unsafe { &*self.element_ptr(index) }
    }
}

impl<'a, T: Copy + Pod, U: BufferUsage> IndexMut<usize> for BufferSlice<'a, T, U> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { &mut *self.element_ptr(index) }
    }
}

impl<T: Copy + Pod, U: BufferUsage> Drop for Buffer<T, U> {
    fn drop(&mut self) {
        if let Some(hook) = ON_DROP.get() {
            // Lent to the hook and never dropped: the memory comes back to be freed below.
            let mut buffer = ManuallyDrop::new(self.take(usage::Dynamic(self.usage.flags())));
            hook(&buffer);
            self.allocation = mem::take(&mut buffer.allocation);
        }
        unsafe { Ctx::device().destroy_buffer(self.handle, None) };
        let alloc = mem::take(&mut self.allocation);
        if let Err(err) = Ctx::allocator().free(alloc) {
            tracing::error!(%err, "failed to free buffer allocation");
        }
    }
}

impl<'a, T: Copy + Pod, U: BufferUsage> Into<BufferSlice<'a, T, U>> for &'a Buffer<T, U> {
    fn into(self) -> BufferSlice<'a, T, U> {
        self.range(..)
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
        let mut requirements = unsafe { Ctx::device().get_buffer_memory_requirements(buffer) };
        if let Some(a) = alignment {
            requirements.alignment = a as u64;
        }

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
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })
        }?;

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

    /// Moves the buffer into one of another element type and usage, leaving `self` without
    /// memory; `self` must not destroy its handle unless it gets the memory back.
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

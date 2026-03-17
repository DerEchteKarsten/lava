use std::{
    any::TypeId,
    fmt::Debug,
    marker::PhantomData,
    ops::{Deref, DerefMut, Index, IndexMut},
    sync::{Arc, Mutex, RwLock},
};

use anyhow::Result;
use ash::vk::{self};
use bytemuck::Pod;
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc},
};

use crate::{buffer::slice::BufferSlice, state::Ctx};

pub mod slice;

#[derive(Debug)]
pub struct Buffer<T: Copy + Pod> {
    pub handle: vk::Buffer,
    pub address: u64,
    pub allocation: Allocation,
    _type_marker: PhantomData<T>,
}

impl<T: Copy + Pod> Index<usize> for Buffer<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        unsafe { self.range(..).ptr().add(index).as_ref() }.unwrap()
    }
}

impl<T: Copy + Pod> IndexMut<usize> for Buffer<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { self.range(..).ptr().add(index).as_mut() }.unwrap()
    }
}

impl<'a, T: Copy + Pod> Index<usize> for BufferSlice<'a, T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        unsafe { self.ptr().add(index).as_ref() }.unwrap()
    }
}

impl<'a, T: Copy + Pod> IndexMut<usize> for BufferSlice<'a, T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { self.ptr().add(index).as_mut() }.unwrap()
    }
}

impl<T: Copy + Pod> Drop for Buffer<T> {
    fn drop(&mut self) {
        unsafe { Ctx::device().destroy_buffer(self.handle, None) };
        let alloc = std::mem::take(&mut self.allocation);
        Ctx::allocator().free(alloc).unwrap();
    }
}

impl<'a, T: Copy + Pod> Into<BufferSlice<'a, T>> for &'a Buffer<T> {
    fn into(self) -> BufferSlice<'a, T> {
        self.range(..)
    }
}

impl<T: Copy + Pod> Buffer<T> {
    pub fn raw(
        usage: vk::BufferUsageFlags,
        cpu_writable: bool,
        num_bytes: u64,
        alignment: Option<u32>,
    ) -> Result<Self> {
        let create_info = vk::BufferCreateInfo::default().size(num_bytes).usage(usage);
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
            address,
            allocation: allocation,
            handle: buffer,
        })
    }

    pub fn size(&self) -> u64 {
        self.allocation.size()
    }

    pub fn new(size: usize, cpu_writable: bool) -> Result<Self> {
        Self::raw(
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::INDIRECT_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            cpu_writable,
            (size * size_of::<T>()) as u64,
            None,
        )
    }

    pub fn len(&self) -> usize {
        (self.size() / size_of::<T>() as u64) as usize
    }

    pub fn cast<B: Copy + Pod>(self) -> Buffer<B> {
        unsafe { std::mem::transmute(self) }
    }
}

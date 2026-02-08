use std::{
    any::TypeId,
    fmt::Debug,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex, RwLock},
};

use anyhow::Result;
use ash::vk::{self};
use bitflags::bitflags;
use bytemuck::Pod;
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc},
};

use crate::{buffer::slice::BufferSlice, state::Ctx};

pub mod allocator;
pub mod slice;

pub trait Location: Copy + Clone {}
#[derive(Debug, Clone, Copy)]
pub struct GpuBuffer;
#[derive(Debug, Clone, Copy)]
pub struct CpuBuffer;

impl Location for GpuBuffer {}
impl Location for CpuBuffer {}

bitflags! {
    #[derive(Clone, Copy)]
    pub struct BufferUsageFlags: u32 {
        const STORAGE = vk::BufferUsageFlags::STORAGE_BUFFER.as_raw();
        const INDIRECT_COMMAND = vk::BufferUsageFlags::INDIRECT_BUFFER.as_raw();
        const VERTEX = vk::BufferUsageFlags::VERTEX_BUFFER.as_raw();
        const SHADER_BINDING_TABLE  = vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR.as_raw();
        const ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR  = vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR.as_raw();
        const ACCELERATION_STRUCTURE_STORAGE  = vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR.as_raw();
    }
}

impl BufferUsageFlags {
    fn to_vk(&self) -> vk::BufferUsageFlags {
        vk::BufferUsageFlags::from_raw(self.0.0)
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS_KHR
    }
}

pub trait AsBuffer {
    type DataType: Pod + Copy;
    type Location: Location + 'static;
    fn get_ref(&self) -> &Buffer<Self::DataType, Self::Location>;
    fn get_mut(&mut self) -> &mut Buffer<Self::DataType, Self::Location>;

    fn whole(&self) -> BufferSlice<Self::DataType, Self::Location> {
        From::from(self.get_ref())
    }
    fn byte_offset(&self, offset: u64) -> BufferSlice<Self::DataType, Self::Location> {
        self.whole().byte_offset(offset)
    }
    fn element_offset(&self, elements: usize) -> BufferSlice<Self::DataType, Self::Location> {
        self.whole().element_offset(elements)
    }
    fn num_bytes(&self, bytes: u64) -> BufferSlice<Self::DataType, Self::Location> {
        self.whole().num_bytes(bytes)
    }
    fn num_elements(&self, elements: usize) -> BufferSlice<Self::DataType, Self::Location> {
        self.whole().num_elements(elements)
    }
}

#[derive(Debug)]
pub struct Buffer<T: Copy + Pod, L: Location = GpuBuffer> {
    pub handle: vk::Buffer,
    pub address: u64,
    pub size: u64,
    pub allocation: Allocation,
    _location_marker: PhantomData<L>,
    _type_marker: PhantomData<T>,
}

impl<T: Copy + Pod, L: Location> Drop for Buffer<T, L> {
    fn drop(&mut self) {
        unsafe { Ctx::device().destroy_buffer(self.handle, None) };
        let alloc = std::mem::take(&mut self.allocation);
        Ctx::allocator().free(alloc).unwrap();
    }
}

impl<T: Pod + Copy, L: Location + 'static> AsBuffer for Buffer<T, L> {
    type DataType = T;
    type Location = L;
    fn get_ref(&self) -> &Buffer<Self::DataType, Self::Location> {
        self
    }
    fn get_mut(&mut self) -> &mut Buffer<Self::DataType, Self::Location> {
        self
    }
}

impl<T: Copy + Pod, L: Location + 'static> Buffer<T, L> {
    pub fn with_alignment(
        usage: BufferUsageFlags,
        num_bytes: u64,
        alignment: Option<u32>,
    ) -> Result<Self> {
        let usage = usage.to_vk();
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
                location: if TypeId::of::<L>() == TypeId::of::<GpuBuffer>() {
                    MemoryLocation::GpuOnly
                } else {
                    MemoryLocation::CpuToGpu
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
            _location_marker: PhantomData,
            _type_marker: PhantomData,
            address,
            allocation: allocation,
            handle: buffer,
            size: num_bytes as u64,
        })
    }

    pub fn assert_size(
        &mut self,
        size: u64,
    ) {
        if self.size < size {
            Ctx::delay_deletion(std::mem::replace(
                self,
                Buffer::with_alignment(
                    BufferUsageFlags::STORAGE,
                    size.next_power_of_two(),
                    None,
                ).unwrap()
            ));
        }
    }

    pub fn new(usage: BufferUsageFlags, size: usize) -> Result<Self> {
        Self::with_alignment(usage, (size * size_of::<T>()) as u64, None)
    }

    pub fn len(&self) -> usize {
        (self.size / size_of::<T>() as u64) as usize
    }

    pub fn cast_owned<B: Copy + Pod, J: Location>(self) -> Buffer<B, J> {
        unsafe { std::mem::transmute(self) }
    }
    pub fn cast<B: Copy + Pod>(&self) -> &Buffer<B, L> {
        unsafe {
            (self as *const Self as *const Buffer<B, L>)
                .as_ref()
                .unwrap()
        }
    }
    pub fn cast_mut<B: Copy + Pod>(&mut self) -> &mut Buffer<B, L> {
        unsafe { (self as *mut Self as *mut Buffer<B, L>).as_mut().unwrap() }
    }
}

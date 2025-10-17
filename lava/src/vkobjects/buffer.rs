use std::{
    any::TypeId, ffi::c_void, fmt::Debug, marker::PhantomData, mem::MaybeUninit, ops::DerefMut, ptr::NonNull, sync::{Arc, Mutex}
};

use anyhow::{Error, Result};
use ash::vk;
use derivative::Derivative;
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc},
};

use crate::state::Ctx;

use super::image::{Image, ImageType, get_aspects};

#[derive(Debug)]
pub struct MAllocation(pub Allocation);

impl Drop for MAllocation {
    fn drop(&mut self) {
        let mut allocator = Ctx::allocator();
        (*allocator)
            .free(std::mem::replace(&mut self.0, unsafe {
                std::mem::zeroed()
            }))
            .unwrap();
    }
}

#[derive(Clone)]
pub struct RawDynamicBuffer {
    pub buffer: vk::Buffer,
    pub address: vk::DeviceAddress,
    allocation: Arc<Mutex<MAllocation>>,
}

pub trait Location {}
pub struct GpuBuffer;
pub struct CpuBuffer;
impl Location for GpuBuffer {}
impl Location for CpuBuffer {}

#[derive(Clone)]
pub struct Buffer<T: Copy, L: Location + 'static = GpuBuffer> {
    pub buffer: Option<RawDynamicBuffer>,
    pub capacity: u64,
    pub usage: vk::BufferUsageFlags,
    pub size: u64,
    pub override_alignment: Option<u64>,
    pub _marker: PhantomData<T>,
    pub _location: PhantomData<L>,
}

impl<T: Copy> Default for Buffer<T, GpuBuffer> {
    fn default() -> Self {
        Self::new(vk::BufferUsageFlags::STORAGE_BUFFER).unwrap()
    }
}

impl<T: Copy, L: Location  + 'static> Buffer<T, L> {
    pub fn new(
        usage: vk::BufferUsageFlags,
    ) -> Result<Self> {
        Self::with_capacity(usage, 0)
    }
    

    pub fn with_capacity(
        usage: vk::BufferUsageFlags,
        capacity: u64,
    ) -> Result<Self> {
        Self::with_alignment(usage, capacity,0, None)
    }

    pub fn with_size(
        usage: vk::BufferUsageFlags,
        size: u64,
    ) -> Result<Self> {
        Self::with_alignment(usage, size, size, None)
    }
    
    pub fn with_alignment(
        mut usage: vk::BufferUsageFlags,
        capacity: u64,
        size: u64,
        override_alignment: Option<u64>,
    ) -> Result<Self> {
        let capacity = capacity * size_of::<T>() as u64;
        let size = size * size_of::<T>() as u64;
        usage |= vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS_KHR;
        Ok(Self {
            buffer: if capacity == 0 {None} else {Some(Self::create_buffer(capacity, usage, override_alignment, TypeId::of::<L>() == TypeId::of::<GpuBuffer>())?)},
            capacity,
            size,
            usage,
            override_alignment,
            _marker: PhantomData,
            _location: PhantomData,
        })
    }

    pub fn len(&self) -> usize {
        self.size as usize / size_of::<T>()
    }

    fn create_buffer(
        capacity: u64,
        usage: vk::BufferUsageFlags,
        override_alignment: Option<u64>,
        gpu: bool,
    ) -> Result<RawDynamicBuffer> {
        let create_info = vk::BufferCreateInfo::default()
            .size(capacity)
            .usage(usage);
        let buffer = unsafe { Ctx::device().create_buffer(&create_info, None)? };
        let mut requirements = unsafe { Ctx::device().get_buffer_memory_requirements(buffer) };
        if let Some(a) = override_alignment {
            requirements.alignment = a;
        }

        let allocation = {
            let mut allocator = Ctx::allocator();
            (*allocator).allocate(&AllocationCreateDesc {
                name: "buffer",
                requirements,
                location: if gpu {MemoryLocation::GpuOnly} else {MemoryLocation::CpuToGpu},
                linear: true,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })
        }?;

        unsafe {
            Ctx::device().bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?
        };
        let addr_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
        let address = unsafe { Ctx::device().get_buffer_device_address(&addr_info) };
        Ok(RawDynamicBuffer {
            address,
            allocation: Arc::new(Mutex::new(MAllocation(allocation))),
            buffer,
        })
    }

    pub fn ptr(&self) -> u64 {
        if let Some(buffer) = &self.buffer {
            buffer.address
        }else {
            0
        }
    }
    pub fn vk(&self) -> vk::Buffer {
        if let Some(buffer) = &self.buffer {
            buffer.buffer
        }else {
            vk::Buffer::null()
        }
    }
}

impl<T: Copy> Buffer<T, GpuBuffer> {
    pub fn copy_from<B:Copy, L: Location>(&mut self, src_buffer: &Buffer<B, L>, offset: u64, size: u64) {
        if size == 0 {
            return;
        }
        if self.size < size + offset {
            self.grow_to_size(size + offset).unwrap();
        }
        // log::info!("copying {} bytes", size);
        Ctx::transfer_queue()
            .execute_command_wait(|cmd| {
                unsafe {
                    Ctx::device().cmd_copy_buffer(
                        *cmd,
                        src_buffer.buffer.as_ref().unwrap().buffer,
                        self.buffer.as_ref().unwrap().buffer,
                        &[vk::BufferCopy {
                            src_offset: 0,
                            size,
                            dst_offset: offset,
                        }],
                    )
                };
            })
            .unwrap();
    }
    pub fn grow_to_size(&mut self, size: u64) -> Result<()> {
        let old_size = self.size;
        self.size = size;
        if size > self.capacity {
            self.capacity = self.size.next_power_of_two();
    
            let buffer = Self::create_buffer(self.capacity, self.usage, self.override_alignment, true)?;
            if old_size != 0 {
                Ctx::transfer_queue().execute_command_wait(|cmd| {
                    unsafe {
                        Ctx::device().cmd_copy_buffer(
                            *cmd,
                            self.buffer.as_ref().unwrap().buffer,
                            buffer.buffer,
                            &[vk::BufferCopy {
                                size: old_size,
                                src_offset: 0,
                                dst_offset: 0,
                            }],
                        )
                    };
                })?;
                unsafe { Ctx::device().destroy_buffer(self.buffer.as_ref().unwrap().buffer, None) };
            }
            self.buffer = Some(buffer);
        }
        Ok(())
    }
    pub fn push(&mut self, staging_buffer: &mut Buffer<u8, CpuBuffer>, data: &[T]) {
        if data.len() == 0 {
            return;
        }
        let offset = self.size;
        let size = data.len() * size_of::<T>();

        let old_size = self.size;
        if self.size < size as u64 + offset {
            self.grow_to_size(size as u64 + offset).unwrap();
            log::debug!("Resized Buffer form {} to {}", old_size, self.size);
        }

        for i in 0..size.div_ceil(staging_buffer.size as usize) {
            staging_buffer
                .copy_from_slice(
                    &data[i * (staging_buffer.size as usize / size_of::<T>())
                        ..data
                            .len()
                            .min((i + 1) * (staging_buffer.size as usize / size_of::<T>()))],
                )
                .unwrap();
            self.copy_from(
                staging_buffer,
                offset + i as u64 * staging_buffer.size as u64,
                (staging_buffer.size as u64).min(size as u64),
            );
        }
    }

    pub fn read_back(&self, staging_buffer: &mut Buffer<u8, CpuBuffer>) -> Vec<T> {
        staging_buffer.copy_from(self, 0, self.size);
        staging_buffer.read_type::<T>(self.len())
    }
}

impl<T: Copy> Buffer<T, CpuBuffer> {
    pub fn copy_from_slice<B: Copy>(&mut self, slice: &[B]) -> Result<()> {
        if self.buffer.is_none() {
            let size = self.capacity.max((slice.len() * size_of::<T>())  as u64);
            self.buffer = Some(Self::create_buffer(size, self.usage, self.override_alignment, false)?);
            self.size = size;
        }
        let allocation = &self.buffer.as_ref().unwrap().allocation;
        let mut alloc = allocation.lock().unwrap();
        let alloc = &mut alloc.0;
        presser::copy_from_slice_to_offset(slice, alloc, 0).unwrap();
        Ok(())
    }

    pub fn read_type<B: Copy>(&self, num_elements: usize) -> Vec<B> {
        if self.buffer.is_none() {
            return vec![];
        }
        let allocation = &self.buffer.as_ref().unwrap().allocation;
        let mut alloc = allocation.lock().unwrap();
        let alloc = &mut alloc.0;
        unsafe {
            let ptr = alloc.mapped_ptr().unwrap().as_ptr();
            let t_ptr = ptr as *const B;
            let mut vec = Vec::with_capacity(num_elements);
            t_ptr.copy_to(vec.as_mut_ptr(), num_elements);
            vec.set_len(num_elements);
            vec
        }
    }

    pub fn read(&self, num_elements: usize) -> Vec<T> {
        self.read_type(num_elements)
    }

    pub fn copy_from<B:Copy, L: Location>(&mut self, src_buffer: &Buffer<B, L>, offset: u64, size: u64) {
        if size == 0 {
            return;
        }
        if self.size < offset + size {
            panic!("Buffer to small!");
        }
        Ctx::transfer_queue()
            .execute_command_wait(|cmd| {
                unsafe {
                    Ctx::device().cmd_copy_buffer(
                        *cmd,
                        src_buffer.buffer.as_ref().unwrap().buffer,
                        self.buffer.as_ref().unwrap().buffer,
                        &[vk::BufferCopy {
                            src_offset: 0,
                            size,
                            dst_offset: offset,
                        }],
                    )
                };
            })
            .unwrap();
    }
}
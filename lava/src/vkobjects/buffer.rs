use std::{
    ffi::c_void,
    fmt::Debug,
    mem::MaybeUninit,
    ops::DerefMut,
    ptr::NonNull,
    sync::{Arc, Mutex},
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

#[derive(Derivative)]
#[derivative(Eq, PartialEq, Debug, Clone)]
pub struct Buffer {
    pub buffer: vk::Buffer,
    #[derivative(PartialEq = "ignore")]
    pub allocation: Arc<Mutex<MAllocation>>,
    pub address: vk::DeviceAddress,
    pub size: vk::DeviceSize,
    pub usage: vk::BufferUsageFlags,
}

impl Buffer {
    pub fn new_aligned(
        usage: vk::BufferUsageFlags,
        memory_location: MemoryLocation,
        size: vk::DeviceSize,
        alignment: Option<u64>,
    ) -> Result<Self> {
        let create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);
        let buffer = unsafe { Ctx::device().create_buffer(&create_info, None)? };
        let mut requirements = unsafe { Ctx::device().get_buffer_memory_requirements(buffer) };
        if let Some(a) = alignment {
            requirements.alignment = a;
        }

        let allocation = {
            let mut allocator = Ctx::allocator();
            (*allocator).allocate(&AllocationCreateDesc {
                name: "buffer",
                requirements,
                location: memory_location,
                linear: true,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })
        }?;

        unsafe {
            Ctx::device().bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?
        };
        let addr_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);

        Ok(Self {
            buffer,
            allocation: Arc::new(Mutex::new(MAllocation(allocation))),
            address: unsafe { Ctx::device().get_buffer_device_address(&addr_info) },
            size,
            usage,
        })
    }

    pub fn new(
        usage: vk::BufferUsageFlags,
        memory_location: MemoryLocation,
        size: vk::DeviceSize,
    ) -> Result<Self> {
        Self::new_aligned(usage, memory_location, size, None)
    }

    pub fn copy_data_to_buffer<T: Copy>(&self, data: &[T]) -> Result<()> {
        let mut allocation = self.allocation.lock().unwrap();

        presser::copy_from_slice_to_offset(data, &mut (*allocation).0, 0)?;

        Ok(())
    }

    pub fn copy_data_to_buffer_offset<T: Copy>(&self, data: &[T], offset: u64) -> Result<()> {
        let mut allocation = self.allocation.lock().unwrap();
        presser::copy_from_slice_to_offset(data, &mut (*allocation).0, offset as usize)?;

        Ok(())
    }

    pub fn copy_value_to_buffer_offset<T: Copy>(&self, data: &T, offset: u64) -> Result<()> {
        let mut allocation = self.allocation.lock().unwrap();
        presser::copy_to_offset(data, &mut (*allocation).0, offset as usize)?;

        Ok(())
    }

    pub fn copy_data_to_aligned_buffer<T: Copy>(&self, data: &[T], alignment: u32) -> Result<()> {
        let mut allocation = self.allocation.lock().unwrap();
        presser::copy_from_slice_to_offset_with_align(
            data,
            &mut (*allocation).0,
            0,
            alignment as usize,
        )?;

        Ok(())
    }

    // pub fn destroy(&mut self) -> Result<()> {
    //     unsafe { Ctx::device().destroy_buffer(self.buffer, None) };
    //     let mut allocator = Ctx::allocator();

    //     (*allocator).free((*self.allocation).into_inner().unwrap())?;
    //     Ok(())
    // }

    pub fn read<T: Clone>(&self, num_elements: usize) -> Result<Vec<T>> {
        let allocation = self.allocation.lock().unwrap();
        let slice = (*allocation).0.mapped_slice().unwrap();
        let mut vec = Vec::new();
        vec.extend_from_slice(
            &unsafe { std::mem::transmute::<&[u8], &[T]>(slice) }[0..num_elements],
        );
        Ok(vec)
    }
}

#[derive(Clone)]
pub struct RawDynamicBuffer {
    pub buffer: vk::Buffer,
    pub address: vk::DeviceAddress,
    allocation: Arc<Mutex<MAllocation>>,
}

#[derive(Clone)]
pub struct DynamicBuffer {
    pub buffer: RawDynamicBuffer,
    pub capacity: u64,
    pub usage: vk::BufferUsageFlags,
    pub size: u64,
    pub memory_location: MemoryLocation,
    pub override_alignment: Option<u64>,
}

pub trait CopySrc {
    fn to_vk(&self) -> vk::Buffer;
    fn size(&self) -> u64;
}

impl CopySrc for Buffer {
    fn size(&self) -> u64 {
        self.size
    }
    fn to_vk(&self) -> vk::Buffer {
        self.buffer
    }
}

impl CopySrc for DynamicBuffer {
    fn size(&self) -> u64 {
        self.size
    }
    fn to_vk(&self) -> vk::Buffer {
        self.buffer.buffer
    }
}

impl DynamicBuffer {
    pub fn new(
        usage: vk::BufferUsageFlags,
        memory_location: MemoryLocation,
        capacity: u64,
        override_alignment: Option<u64>,
    ) -> Result<Self> {
        Ok(Self {
            memory_location,
            buffer: Self::create_buffer(capacity, usage, override_alignment)?,
            capacity,
            size: 0,
            usage: usage
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            override_alignment,
        })
    }

    fn create_buffer(
        capacity: u64,
        usage: vk::BufferUsageFlags,
        override_alignment: Option<u64>,
    ) -> Result<RawDynamicBuffer> {
        let create_info = vk::BufferCreateInfo::default()
            .size(capacity)
            .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);
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
                location: MemoryLocation::GpuOnly,
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

    pub fn grow_to_size(&mut self, size: u64) -> Result<()> {
        let old_size = self.size;
        self.size = size;
        if size > self.capacity {
            self.capacity = self.size.next_power_of_two();

            let buffer = Self::create_buffer(self.capacity, self.usage, self.override_alignment)?;
            if old_size != 0 {
                Ctx::transfer_queue().execute_command_wait(|cmd| {
                    unsafe {
                        Ctx::device().cmd_copy_buffer(
                            *cmd,
                            self.buffer.buffer,
                            buffer.buffer,
                            &[vk::BufferCopy {
                                size: old_size,
                                src_offset: 0,
                                dst_offset: 0,
                            }],
                        )
                    };
                })?;
            }
            unsafe { Ctx::device().destroy_buffer(self.buffer.buffer, None) };
            self.buffer = buffer;
        }
        Ok(())
    }

    pub fn copy_from(&mut self, src_buffer: &impl CopySrc, offset: u64, size: u64) {
        if self.size < size + offset {
            self.grow_to_size(size + offset);
        }
        Ctx::transfer_queue()
            .execute_command_wait(|cmd| {
                unsafe {
                    Ctx::device().cmd_copy_buffer(
                        *cmd,
                        src_buffer.to_vk(),
                        self.buffer.buffer,
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
    pub fn copy_from_cmd(
        &mut self,
        src_buffer: &impl CopySrc,
        cmd: &vk::CommandBuffer,
        offset: u64,
        size: u64,
    ) -> Result<()> {
        if self.size < size + offset {
            self.grow_to_size(size + offset)?;
        }
        unsafe {
            Ctx::device().cmd_copy_buffer(
                *cmd,
                src_buffer.to_vk(),
                self.buffer.buffer,
                &[vk::BufferCopy {
                    src_offset: 0,
                    size: size,
                    dst_offset: offset,
                }],
            )
        };
        Ok(())
    }

    pub fn push<T: Copy>(&mut self, staging_buffer: &Buffer, data: &[T]) {
        let offset = self.size;
        let size = data.len() * size_of::<T>();

        let old_size = self.size;
        if self.size < size as u64 + offset {
            self.grow_to_size(size as u64 + offset).unwrap();
            log::debug!("Resized Buffer form {} to {}", old_size, self.size);
        }

        for i in 0..size.div_ceil(staging_buffer.size as usize) {
            staging_buffer
                .copy_data_to_buffer(
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

    pub fn ptr(&self) -> u64 {
        self.buffer.address
    }
}

use std::{
    any::TypeId,
    fmt::Debug,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex},
};

use anyhow::Result;
use ash::vk::{self};
use bitflags::bitflags;
use bytemuck::Pod;
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc},
};

use crate::state::Ctx;

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

pub trait Location {}
#[derive(Debug, Clone)]
pub struct GpuBuffer;
#[derive(Debug, Clone)]
pub struct CpuBuffer;
impl Location for GpuBuffer {}
impl Location for CpuBuffer {}

bitflags! {
    #[derive(Clone, Copy)]
    pub struct BufferUsageFlags: u32 {
        const STORAGE = vk::BufferUsageFlags::STORAGE_BUFFER.as_raw();
        const INDIRECT_COMMAND = vk::BufferUsageFlags::INDIRECT_BUFFER.as_raw();
        const VERTEX = vk::BufferUsageFlags::VERTEX_BUFFER.as_raw();
        const INDEX  = vk::BufferUsageFlags::INDEX_BUFFER.as_raw();
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

#[derive(Debug, Clone)]
pub struct Buffer<T: Copy + Pod, L: Location = GpuBuffer> {
    pub handle: vk::Buffer,
    pub address: u64,
    pub size: u64,
    pub(crate) allocation: Arc<Mutex<MAllocation>>,
    _location_marker: PhantomData<L>,
    _type_marker: PhantomData<T>,
}

pub struct StorageBuffer<T: Copy + Pod, L: Location = GpuBuffer> {
    pub buffer: Buffer<T, L>,
    size: u64,
    usage: vk::BufferUsageFlags,
    alignment: Option<u32>,
}

impl<T: Copy + Pod, L: Location+ 'static> Default for StorageBuffer<T, L> {
    fn default() -> Self {
        StorageBuffer::new(BufferUsageFlags::STORAGE).unwrap()
    }
}

impl<T: Copy + Pod> Buffer<T, CpuBuffer> {
    pub fn from_data(usage: BufferUsageFlags, data: &[T]) -> Result<Self> {
        let mut buffer = Buffer::new(usage, data.len())?;
        buffer.copy_from_slice(data, 0)?;
        Ok(buffer)
    }
}
impl<T: Copy + Pod> Buffer<T, GpuBuffer> {
    pub fn from_data(
        usage: BufferUsageFlags,
        staging_buffer: &mut Buffer<u8, CpuBuffer>,
        data: &[T],
    ) -> Result<Self> {
        staging_buffer.copy_from_slice(data, 0)?;
        let mut buffer = Buffer::new(usage, data.len())?;
        buffer.copy_from(
            staging_buffer.cast_mut(),
            0,
            (data.len() * size_of::<T>()) as u64,
        );
        Ok(buffer)
    }
}
impl<T: Copy + Pod> StorageBuffer<T, GpuBuffer> {
    pub fn from_data(
        usage: BufferUsageFlags,
        staging_buffer: &mut Buffer<u8, CpuBuffer>,
        data: &[T],
    ) -> Result<Self> {
        let buffer = Buffer::<T, GpuBuffer>::from_data(usage, staging_buffer, data)?;
        Ok(Self {
            size: buffer.size,
            buffer: buffer,
            usage: usage.to_vk(),
            alignment: None,
        })
    }
}

impl<T: Copy + Pod, L: Location> Deref for StorageBuffer<T, L> {
    type Target = Buffer<T, L>;
    fn deref(&self) -> &Buffer<T, L> {
        &self.buffer
    }
}

impl<T: Copy + Pod, L: Location> AsRef<Buffer<T, L>> for StorageBuffer<T, L> {
    fn as_ref(&self) -> &Buffer<T, L> {
        &self.buffer
    }
}

impl<T: Copy + Pod, L: Location> AsMut<Buffer<T, L>> for StorageBuffer<T, L> {
    fn as_mut(&mut self) -> &mut Buffer<T, L> {
        &mut self.buffer
    }
}

impl<T: Copy + Pod> AsMut<Buffer<T>> for StorageBuffer<T, CpuBuffer> {
    fn as_mut(&mut self) -> &mut Buffer<T> {
        self.buffer.cast_mut_location()
    }
}

impl<T: Copy + Pod> AsRef<Buffer<T>> for StorageBuffer<T, CpuBuffer> {
    fn as_ref(&self) -> &Buffer<T> {
        self.buffer.cast_location()
    }
}





impl<T: Copy + Pod, L: Location> DerefMut for StorageBuffer<T, L> {
    fn deref_mut(&mut self) -> &mut Buffer<T, L> {
        &mut self.buffer
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
            allocation: Arc::new(Mutex::new(MAllocation(allocation))),
            handle: buffer,
            size: num_bytes as u64,
        })
    }
    pub fn new(usage: BufferUsageFlags, size: usize) -> Result<Self> {
        Self::with_alignment(usage, (size * size_of::<T>()) as u64, None)
    }
    pub fn copy_from<J: Location>(
        &mut self,
        src_buffer: &Buffer<T, J>,
        offset: u64,
        num_bytes: u64,
    ) {
        if num_bytes == 0 {
            return;
        }
        Ctx::transfer_queue()
            .execute_command_wait(|cmd| {
                cmd.copy_buffer(src_buffer, self, num_bytes as usize / size_of::<T>(), 0, offset as u32);
            })
            .unwrap();
    }
    pub fn capacity(&self) -> usize {
        (self.size / size_of::<T>() as u64) as usize
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

    pub fn cast_location<newL: Location>(&self) -> &Buffer<T, newL> {
        unsafe {
            (self as *const Self as *const Buffer<T, newL>)
                .as_ref()
                .unwrap()
        }
    }
    pub fn cast_mut_location<newL: Location>(&mut self) -> &mut Buffer<T, newL> {
        unsafe { (self as *mut Self as *mut Buffer<T, newL>).as_mut().unwrap() }
    }
}

impl<T: Copy + Pod, L: Location + 'static> StorageBuffer<T, L> {
    pub fn with_capacity(usage: BufferUsageFlags, capacity: usize) -> Result<Self> {
        Ok(Self {
            usage: usage.clone().to_vk(),
            buffer: Buffer::new(usage, capacity)?,
            size: 0,
            alignment: None,
        })
    }
    pub fn new(usage: BufferUsageFlags) -> Result<Self> {
        Ok(Self {
            buffer: Buffer::new(usage.clone(), 1024)?,
            size: 0,
            alignment: None,
            usage: usage.to_vk()
        })
    }
    pub fn len(&self) -> usize {
        (self.size / size_of::<T>() as u64) as usize
    }
    pub fn clear(&mut self) {
        self.size = 0;
    }
    pub fn assert_size(&mut self, size: u64) -> Result<()> {
        if self.buffer.size < size {
            let capacity = size.next_power_of_two();
            let buffer = Buffer::with_alignment(
                BufferUsageFlags::from_bits_retain(self.usage.as_raw()),
                capacity,
                self.alignment,
            )?;
            if self.size != 0 {
                Ctx::transfer_queue().execute_command_wait(|cmd| {
                    unsafe {
                        Ctx::device().cmd_copy_buffer(
                            cmd.handle,
                            self.buffer.handle,
                            buffer.handle,
                            &[vk::BufferCopy {
                                size: self.size,
                                src_offset: 0,
                                dst_offset: 0,
                            }],
                        )
                    };
                })?;
            }
            unsafe { Ctx::device().destroy_buffer(self.buffer.handle, None) };
            self.buffer = buffer;
        }
        Ok(())
    }
}

impl<T: Copy + Pod> Buffer<T, GpuBuffer> {
    pub fn read(&self, staging_buffer: &mut Buffer<u8, CpuBuffer>) -> Vec<T> {
        self.read_len(staging_buffer, self.capacity())
    }
    pub fn read_len(&self, staging_buffer: &mut Buffer<u8, CpuBuffer>, len: usize) -> Vec<T> {
        staging_buffer
            .cast_mut()
            .copy_from(self, 0, (len * size_of::<T>()) as u64);
        staging_buffer.cast().read_len(len)
    }
}

impl<T: Copy + Pod> Buffer<T, CpuBuffer> {
    pub fn read_len(&self, num_elements: usize) -> Vec<T> {
        let allocation = &self.allocation;
        let mut alloc = allocation.lock().unwrap();
        let alloc = &mut alloc.0;
        unsafe {
            let ptr = alloc.mapped_ptr().unwrap().as_ptr();
            let t_ptr = ptr as *const T;
            let mut vec = Vec::with_capacity(num_elements);
            t_ptr.copy_to(vec.as_mut_ptr(), num_elements);
            vec.set_len(num_elements);
            vec
        }
    }
    pub fn read(&self) -> Vec<T> {
        self.read_len(self.capacity())
    }
    pub fn copy_from_slice<B: Copy>(&mut self, slice: &[B], offset: usize) -> Result<()> {
        let allocation = &self.allocation;
        let mut alloc = allocation.lock().unwrap();
        let alloc = &mut alloc.0;
        presser::copy_from_slice_to_offset(slice, alloc, offset).unwrap();
        Ok(())
    }
}

impl<T: Copy + Pod> StorageBuffer<T, GpuBuffer> {
    pub fn push(&mut self, staging_buffer: &mut Buffer<u8, CpuBuffer>, data: &[T]) {
        if data.len() == 0 {
            return;
        }
        let offset = self.size;
        let size = data.len() * size_of::<T>();

        self.assert_size(size as u64 + offset).unwrap();
        self.size += size as u64;

        for i in 0..size.div_ceil(staging_buffer.size as usize) {
            staging_buffer
                .copy_from_slice(
                    &data[i * (staging_buffer.size as usize / size_of::<T>())
                        ..data
                            .len()
                            .min((i + 1) * (staging_buffer.size as usize / size_of::<T>()))],
                    0
                )
                .unwrap();
            let staging_size = staging_buffer.size;
            self.copy_from(
                staging_buffer.cast_mut(),
                offset + i as u64 * staging_size as u64,
                (staging_size as u64).min(size as u64),
            );
        }
    }
}

impl<T: Copy + Pod> StorageBuffer<T, CpuBuffer> {
    pub fn push(&mut self, data: &[T]) {
        if data.len() == 0 {
            return;
        }
        let offset = self.size;
        let size = data.len() * size_of::<T>();

        self.assert_size(size as u64 + offset).unwrap();
        self.size += size as u64;

        self.copy_from_slice(data, offset as usize).unwrap();
    }
}


use std::{
    any::TypeId,
    ffi::c_void,
    fmt::Debug,
    marker::PhantomData,
    mem::MaybeUninit,
    ops::DerefMut,
    ptr::NonNull,
    sync::{Arc, Mutex},
};

use anyhow::{Error, Result};
use ash::vk;
use bitflags::bitflags;
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


pub trait Location {}
#[derive(Debug)]
pub struct GpuBuffer;
#[derive(Debug)]
pub struct CpuBuffer;
impl Location for GpuBuffer {}
impl Location for CpuBuffer {}

pub trait Size {}

#[derive(Debug)]
pub struct Sized;
#[derive(Debug)]
pub struct Static<const N: usize>;
#[derive(Debug)]
pub struct Dynamic;
#[derive(Debug)]
pub struct FirstUse;

impl<const N: usize> Size for Static<N> {}
impl Size for Dynamic {}
impl Size for FirstUse {}
impl Size for Sized {}

bitflags! {
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
        vk::BufferUsageFlags::from_raw(self.0.0) | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS_KHR
    }
}


fn create_buffer(usage: vk::BufferUsageFlags, size: u64, alignment: Option<u32>, gpu: bool) -> Result<(vk::Buffer, u64, Arc<Mutex<MAllocation>>)> {
    let create_info = vk::BufferCreateInfo::default().size(size).usage(vk::BufferUsageFlags::from_raw(usage));
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
            location: if gpu {
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
    Ok((
        buffer,
        address,
        Arc::new(Mutex::new(MAllocation(allocation))),
    ))
}

fn new<T: Copy, S: Size + 'static, L: Location + 'static>(usage: BufferUsageFlags, size: u64, capacity: u64, alignment: Option<u32>) -> Result<Buffer<T, S, L>> {
    let usage = usage.to_vk();
    let (buffer, address, allocation) = create_buffer(usage, if capacity != 0 {capacity} else {size}, alignment, TypeId::of::<L>() == TypeId::of::<GpuBuffer>())?;
    Ok(Buffer {
        usage,
        buffer, 
        allocation: Some(allocation),
        ptr: address, 
        size: if capacity != 0 {0} else {size},
        alignment,
        capacity,
        _location_maker: PhantomData,
        _size_marker: PhantomData,
        _type_marker: PhantomData,
    })
}

#[derive(Clone, Debug)]
pub struct Buffer<T: Copy, S: Size + 'static = FirstUse, L: Location + 'static = GpuBuffer>{
    buffer: vk::Buffer,
    allocation: Option<Arc<Mutex<MAllocation>>>,
    pub ptr: u64,
    usage: vk::BufferUsageFlags,
    alignment: Option<u32>,
    size: u64,
    capacity: u64,
    _type_marker: PhantomData<T>,
    _location_maker: PhantomData<L>,
    _size_marker: PhantomData<S>
}

impl<T: Copy, L: Location + 'static> Buffer<T, FirstUse, L> {
    pub fn with_alignment(usage: BufferUsageFlags, alignment: Option<u32>) -> Result<Self> {
        let usage = usage.to_vk();
        Ok(Self {
            usage,
            alignment,
            allocation: None,
            buffer: vk::Buffer::null(),
            ptr: 0,
            size: 0,
            capacity: 0,
            _location_maker: PhantomData,
            _size_marker: PhantomData,
            _type_marker: PhantomData,
        })
    }
    pub fn new(usage: BufferUsageFlags) -> Result<Self> {
        Self::with_alignment(usage, None)
    }
}
impl<T: Copy, L: Location + 'static> Default for Buffer<T, FirstUse, L> {
    fn default() -> Self {
        Self::new(BufferUsageFlags::STORAGE).unwrap()
    }
}


impl<T: Copy, L: Location + 'static, const N: usize> Buffer<T, Static<N>, L> {
    pub fn with_alignment(usage: BufferUsageFlags, alignment: Option<u32>) -> Result<Self> {
        let size = (N * size_of::<T>()) as u64;
        new(usage, size, 0, alignment)
    }
    pub fn new(usage: BufferUsageFlags) -> Result<Self> {
        Self::with_alignment(usage, None)
    }
}
impl<T: Copy, L: Location + 'static, const N: usize> Default for Buffer<T, Static<N>, L> {
    fn default() -> Self {
        Self::new(BufferUsageFlags::STORAGE).unwrap()
    }
}


impl<T: Copy, L: Location + 'static> Buffer<T, Sized, L> {
    pub fn with_alignment(usage: BufferUsageFlags, size: usize, alignment: Option<u32>) -> Result<Self> {
        let size = (size * size_of::<T>()) as u64;
        new(usage, size, 0, alignment)
    }
    pub fn new(usage: BufferUsageFlags, size: usize) -> Result<Self> {
        Self::with_alignment(usage, size, None)
    }
}

impl<T: Copy, L: Location + 'static> Buffer<T, Dynamic, L> {
    pub fn with_alignment(usage: BufferUsageFlags, capacity: usize, alignment: Option<u32>) -> Result<Self> {
        let capacity = (capacity * size_of::<T>()) as u64;
        new(usage, 0, capacity, alignment)
    }
    pub fn new(usage: BufferUsageFlags, capacity: usize) -> Result<Self> {
        Self::with_alignment(usage, capacity, None)
    }
}

impl<T: Copy, S: Size + 'static> Buffer<T, S, GpuBuffer> {
    pub fn from_data(usage: BufferUsageFlags, data: &[T]) -> Result<Self> {
        
    }
}
impl<T: Copy, S: Size + 'static> Buffer<T, S, CpuBuffer> {
    pub fn from_data(usage: BufferUsageFlags, data: &[T]) -> Result<Self> {
        
    }
}


impl<T: Copy, S: Size + 'static, L: Location + 'static> Buffer<T, S, L> {
    pub fn len(&self) -> usize {
        self.size as usize / size_of::<T>()
    }
}

impl<T: Copy, S: Size + 'static> Buffer<T, S, GpuBuffer> {
    pub fn copy_from<B: Copy, L: Location>(
        &mut self,
        src_buffer: &Buffer<B, L>,
        offset: u64,
        size: u64,
    ) {
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

            let buffer =
                Self::create_buffer(self.capacity, self.usage, self.override_alignment, true)?;
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
        self.read_back_len(staging_buffer, self.len())
    }

    pub fn read_back_len(&self, staging_buffer: &mut Buffer<u8, CpuBuffer>, len: usize) -> Vec<T> {
        staging_buffer.copy_from(self, 0, (len * size_of::<T>()) as u64);
        staging_buffer.read_type::<T>(len)
    }
}

impl<T: Copy> Buffer<T, CpuBuffer> {
    pub fn copy_from_slice<B: Copy>(&mut self, slice: &[B]) -> Result<()> {
        if self.buffer.is_none() {
            let size = self.capacity.max((slice.len() * size_of::<T>()) as u64);
            self.buffer = Some(Self::create_buffer(
                size,
                self.usage,
                self.override_alignment,
                false,
            )?);
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

    pub fn copy_from<B: Copy, L: Location>(
        &mut self,
        src_buffer: &Buffer<B, L>,
        offset: u64,
        size: u64,
    ) {
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

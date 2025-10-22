
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
use ash::vk::{self, Handle};
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
pub struct DynamicUninit;

#[derive(Debug)]
pub struct FirstUse;

impl<const N: usize> Size for Static<N> {}
impl Size for Dynamic {}
impl Size for FirstUse {}
impl Size for Sized {}
impl Size for DynamicUninit {}

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
    let create_info = vk::BufferCreateInfo::default().size(size).usage(usage);
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
    let (buffer, address, allocation) = create_buffer(usage, capacity.max(size), alignment, TypeId::of::<L>() == TypeId::of::<GpuBuffer>())?;
    Ok(Buffer {
        usage,
        buffer, 
        allocation: Some(allocation),
        ptr: address, 
        size: size,
        alignment,
        capacity: capacity.max(size),
        _location_maker: PhantomData,
        _size_marker: PhantomData,
        _type_marker: PhantomData,
    })
}

#[derive(Clone, Debug)]
pub struct Buffer<T: Copy, S: Size + 'static = DynamicUninit, L: Location + 'static = GpuBuffer>{
    pub buffer: vk::Buffer,
    allocation: Option<Arc<Mutex<MAllocation>>>,
    pub ptr: u64,
    usage: vk::BufferUsageFlags,
    alignment: Option<u32>,
    pub size: u64,
    capacity: u64,
    _type_marker: PhantomData<T>,
    _location_maker: PhantomData<L>,
    _size_marker: PhantomData<S>
}

impl<T: Copy, L: Location + 'static> Buffer<T, FirstUse, L> {
    pub fn with_alignment(usage: BufferUsageFlags, alignment: Option<u32>) -> Self {
        let usage = usage.to_vk();
        Self {
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
        }
    }
    pub fn new(usage: BufferUsageFlags) -> Self {
        Self::with_alignment(usage, None)
    }
}
impl<T: Copy, L: Location + 'static> Default for Buffer<T, FirstUse, L> {
    fn default() -> Self {
        Self::new(BufferUsageFlags::STORAGE)
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


impl<T: Copy, L: Location + 'static> Buffer<T, DynamicUninit, L> {
    pub fn with_alignment(usage: BufferUsageFlags, alignment: Option<u32>) -> Self {
        Self {
            _location_maker: PhantomData,
            _size_marker: PhantomData,
            _type_marker: PhantomData,
            usage: usage.to_vk(),
            alignment,
            allocation: None,
            buffer: vk::Buffer::null(),
            ptr: 0,
            size: 0,
            capacity: 0,
        }
    }
    pub fn new(usage: BufferUsageFlags) -> Self {
        Self::with_alignment(usage, None)
    }
}
impl<T: Copy, L: Location + 'static> Default for Buffer<T, DynamicUninit, L> {
    fn default() -> Self {
        Self::new(BufferUsageFlags::STORAGE)
    }
}

impl<T: Copy, S: Size + 'static> Buffer<T, S, CpuBuffer> {
    pub fn from_data(usage: BufferUsageFlags, data: &[T]) -> Result<Self> {
        let size = (data.len() * size_of::<T>()) as u64;
        let buffer = new(usage, size, size, None)?;
        Ok(buffer)
    }
}
impl<T: Copy, S: Size + 'static> Buffer<T, S, GpuBuffer> {
    pub fn from_data<B: Size + 'static>(usage: BufferUsageFlags, staging_buffer: &mut Buffer<u8, B, CpuBuffer>, data: &[T]) -> Result<Self> {
        let size = (data.len() * size_of::<T>()) as u64;
        staging_buffer.copy_from_slice(data)?;
        let mut buffer = new(usage, size, size, None)?;
        buffer.copy_from(staging_buffer.cast_mut(), 0, size);
        Ok(buffer)
    }
}


impl<T: Copy, S: Size + 'static, L: Location + 'static> Buffer<T, S, L> {
    pub fn len(&self) -> usize {
        self.size as usize / size_of::<T>()
    }
    pub fn cast<B: Copy>(&self) -> &Buffer<B, S, L>{
        unsafe { (self as *const Self as *const Buffer<B, S, L>).as_ref().unwrap()}
    }
    pub fn cast_mut<B: Copy>(&mut self) -> &mut Buffer<B, S, L>{
        unsafe { (self as *mut Self as *mut Buffer<B, S, L>).as_mut().unwrap()}
    }
    pub fn copy_from<J: Location, H: Size>(
        &mut self,
        src_buffer: &mut Buffer<T, H, J>,
        offset: u64,
        num_bytes: u64,
    ) {
        if num_bytes == 0 {
            return;
        }
        self.grow_to_size(num_bytes + offset).unwrap();
        src_buffer.grow_to_size(num_bytes).unwrap();
        if src_buffer.size < num_bytes + offset || self.size < num_bytes {
            panic!("Buffer to small!");
        }

        Ctx::transfer_queue()
            .execute_command_wait(|cmd| {
                unsafe {
                    Ctx::device().cmd_copy_buffer(
                        *cmd,
                        src_buffer.buffer,
                        self.buffer,
                        &[vk::BufferCopy {
                            src_offset: 0,
                            size: num_bytes,
                            dst_offset: offset,
                        }],
                    )
                };
            })
            .unwrap();
    }
}

pub trait Growable{
    fn grow_to_size(&mut self, size: u64) -> Result<()>;
}

impl<T: Copy, L: Location + 'static, S: Size + 'static>Growable for Buffer<T, S, L> {
    default fn grow_to_size(&mut self, size: u64) -> Result<()> {
        Ok(())
    }
}

fn grow_to_size_generic(old_size: u64, size: u64, usage: vk::BufferUsageFlags, alignment: Option<u32>, old_buffer: vk::Buffer, gpu: bool) -> Result<(vk::Buffer, u64, Arc<Mutex<MAllocation>>, u64)> {
    let capacity = size.next_power_of_two();

    let (buffer, ptr, allocation) =
        create_buffer(usage, capacity, alignment, gpu)?;
    if old_size != 0 {
        Ctx::transfer_queue().execute_command_wait(|cmd| {
            unsafe {
                Ctx::device().cmd_copy_buffer(
                    *cmd,
                    old_buffer,
                    buffer,
                    &[vk::BufferCopy {
                        size: old_size,
                        src_offset: 0,
                        dst_offset: 0,
                    }],
                )
            };
        })?;
        unsafe { Ctx::device().destroy_buffer(old_buffer, None) };
    }
    Ok((buffer, ptr, allocation, capacity))
}

impl<T: Copy, L: Location + 'static>Growable for Buffer<T, Dynamic, L>{
    fn grow_to_size(&mut self, size: u64) -> Result<()> {
        if size <= self.size {
            return Ok(());
        }
        let (buffer, ptr, allocation, capacity) =
            grow_to_size_generic(self.size, size, self.usage, self.alignment, self.buffer, TypeId::of::<L>() == TypeId::of::<GpuBuffer>())?;
        self.buffer = buffer;
        self.ptr = ptr;
        self.allocation = Some(allocation);
        self.size = size;
        self.capacity = capacity;
        Ok(())
    }
}
impl<T: Copy, L: Location + 'static>Growable for Buffer<T, DynamicUninit, L>{
    fn grow_to_size(&mut self, size: u64) -> Result<()> {
        if size <= self.size {
            return Ok(());
        }
        if self.buffer == vk::Buffer::null() {
            let (buffer, ptr, allocation) =
                create_buffer(self.usage, size, self.alignment, TypeId::of::<L>() == TypeId::of::<GpuBuffer>())?;
            self.buffer = buffer;
            self.ptr = ptr;
            self.allocation = Some(allocation);
            self.size = size;
            self.capacity = size;
            return Ok(());
        }

        let (buffer, ptr, allocation, capacity) =
            grow_to_size_generic(self.size, size, self.usage, self.alignment, self.buffer, TypeId::of::<L>() == TypeId::of::<GpuBuffer>())?;
        self.buffer = buffer;
        self.ptr = ptr;
        self.allocation = Some(allocation);
        self.size = size;
        self.capacity = capacity;
        Ok(())
    }
}

impl<T: Copy, L: Location + 'static>Growable for Buffer<T, FirstUse, L>{
    fn grow_to_size(&mut self, size: u64) -> Result<()> {
        if size != 0 {
            return Ok(());
        }
        
        let (buffer, ptr, allocation) =
            create_buffer(self.usage, size, self.alignment, TypeId::of::<L>() == TypeId::of::<GpuBuffer>())?;
        self.buffer = buffer;
        self.ptr = ptr;
        self.allocation = Some(allocation);
        self.size = size;
        self.capacity = size;
        Ok(())
    }
}

impl<T: Copy, S: Size + 'static> Buffer<T, S, GpuBuffer> {
    pub fn read<B: Size + 'static>(&mut self, staging_buffer: &mut Buffer<u8, S, CpuBuffer>) -> Vec<T> {
        self.read_len(staging_buffer, self.len())
    }
    pub fn read_len<B: Size + 'static>(&mut self, staging_buffer: &mut Buffer<u8, B, CpuBuffer>, len: usize) -> Vec<T> {
        staging_buffer.cast_mut().copy_from(self, 0, (len * size_of::<T>()) as u64);
        staging_buffer.cast().read_len(len)
    }
}


impl<T: Copy, S: Size + 'static> Buffer<T, S, CpuBuffer> {
    pub fn read_len(&self, num_elements: usize) -> Vec<T> {
        if self.buffer.is_null() {
            return vec![];
        }
        let allocation = &self.allocation.as_ref().unwrap();
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
        self.read_len(self.len())
    }
}

impl<T: Copy, S: Size+ 'static> Buffer<T, S, CpuBuffer> {
    pub fn copy_from_slice<B: Copy>(&mut self, slice: &[B]) -> Result<()> {
        self.grow_to_size((slice.len() * size_of::<T>()) as u64)?;

        let allocation = &self.allocation.as_ref().unwrap();
        let mut alloc = allocation.lock().unwrap();
        let alloc = &mut alloc.0;
        presser::copy_from_slice_to_offset(slice, alloc, 0).unwrap();
        Ok(())
    }
}


trait Pushable {}
impl Pushable for Dynamic {}
impl Pushable for DynamicUninit {}

impl<T: Copy, L: Location, S: Size + Pushable> Buffer<T, S, L> {
    pub fn push<B: Size>(&mut self, staging_buffer: &mut Buffer<u8, B, CpuBuffer>, data: &[T]) {
        if data.len() == 0 {
            return;
        }
        let offset = self.size;
        let size = data.len() * size_of::<T>();

        self.grow_to_size(size as u64 + offset).unwrap();

        for i in 0..size.div_ceil(staging_buffer.size as usize) {
            staging_buffer
                .copy_from_slice(
                    &data[i * (staging_buffer.size as usize / size_of::<T>())
                        ..data
                            .len()
                            .min((i + 1) * (staging_buffer.size as usize / size_of::<T>()))],
                )
                .unwrap();
            staging_buffer.grow_to_size(size as u64);
            let staging_size = staging_buffer.size;
            self.copy_from(
                staging_buffer.cast_mut(),
                offset + i as u64 * staging_size as u64,
                (staging_size as u64).min(size as u64),
            );
        }
    }
}
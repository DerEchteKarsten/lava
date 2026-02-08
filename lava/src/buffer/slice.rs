use std::marker::PhantomData;

use ash::vk;
use bytemuck::Pod;

use crate::buffer::{Buffer, CpuBuffer, GpuBuffer, Location};

#[derive(Copy, Clone)]
pub struct BufferSlice<T: Copy + Pod, L: Location = GpuBuffer> {
    pub handle: vk::Buffer,
    pub size: u64,
    pub offset: u64,
    pub(crate) cpu_base_ptr: usize,
    pub(crate) gpu_base_ptr: u64,
    pub(crate) _marker: PhantomData<T>,
    pub(crate) _location: PhantomData<L>,
}

impl<T: Copy + Pod> From<&[T]> for BufferSlice<T, CpuBuffer> {
    fn from(value: &[T]) -> Self {
        BufferSlice {
            handle: vk::Buffer::null(),
            size: size_of_val(value) as u64,
            offset: 0,
            cpu_base_ptr: value.as_ptr() as usize,
            gpu_base_ptr: 0,
            _marker: PhantomData,
            _location: PhantomData,
        }
    }
}

impl<T: Copy + Pod> From<&Buffer<T>> for BufferSlice<T> {
    fn from(value: &Buffer<T>) -> Self {
        BufferSlice {
            handle: value.handle,
            size: value.size,
            offset: 0,
            cpu_base_ptr: 0,
            gpu_base_ptr: value.address,
            _marker: PhantomData,
            _location: PhantomData,
        }
    }
}

impl<T: Copy + Pod> From<&Buffer<T, CpuBuffer>> for BufferSlice<T, CpuBuffer> {
    fn from(value: &Buffer<T, CpuBuffer>) -> Self {
        BufferSlice {
            handle: value.handle,
            size: value.size,
            offset: 0,
            gpu_base_ptr: value.address,
            cpu_base_ptr: value.allocation.mapped_ptr().unwrap().as_ptr() as usize,
            _marker: PhantomData,
            _location: PhantomData,
        }
    }
}

impl<T: Copy + Pod, L: Location> From<&Buffer<T, L>> for BufferSlice<T, L> {
    default fn from(value: &Buffer<T, L>) -> Self {
        unreachable!()
    }
}

impl<T: Copy + Pod, L: Location> BufferSlice<T, L> {
    pub fn add_byte_offset(mut self, offset: u64) -> Self {
        self.offset += offset;
        self
    }
    pub fn byte_offset(mut self, offset: u64) -> Self {
        self.offset = offset;
        self
    }
    pub fn element_offset(mut self, offset: usize) -> Self {
        self.offset = (offset * size_of::<T>()) as u64;
        self
    }
    pub fn add_element_offset(mut self, offset: usize) -> Self {
        self.offset += (offset * size_of::<T>()) as u64;
        self
    }
    pub fn num_elements(mut self, size: usize) -> Self {
        self.size = (size * size_of::<T>()) as u64;
        self
    }
    pub fn num_bytes(mut self, size: u64) -> Self {
        self.size = size;
        self
    }
    pub fn len(&self) -> usize {
        self.size as usize / size_of::<T>()
    }
    pub fn cpu_address(&self) -> usize {
        self.cpu_base_ptr + self.offset as usize
    }
    pub fn gpu_address(&self) -> u64 {
        self.gpu_base_ptr + self.offset
    }
    pub fn region<B: Location>(&self, other: BufferSlice<T, B>) -> vk::BufferCopy {
        vk::BufferCopy {
            src_offset: self.offset,
            dst_offset: other.offset,
            size: self.size,
        }
    }
    pub fn cast_owned<B: Copy + Pod>(self) -> BufferSlice<B, L> {
        unsafe { std::mem::transmute(self) }
    }
}

impl<T: Copy + Pod> BufferSlice<T, CpuBuffer> {
    pub fn mem_copy_to(&self, other: BufferSlice<T, CpuBuffer>) {
        unsafe {
            let src_ptr = self.cpu_base_ptr as *const T;
            let dst_ptr = other.cpu_base_ptr as *mut T;
            src_ptr.byte_add(self.offset as usize).copy_to(
                dst_ptr.byte_add(other.offset as usize),
                self.size as usize / size_of::<T>(),
            );
        };
    }
    pub fn mem_copy_from(&mut self, other: BufferSlice<T, CpuBuffer>) {
        other.mem_copy_to(*self);
    }
}

//! Typed buffer slices: sub-ranges, byte views, casts, and host copies
use ash::vk;
use bytemuck::Pod;
use std::marker::PhantomData;
use std::ops::RangeBounds;
use std::range::Range;

use crate::{
    buffer::{
        Buffer,
        usage::{BufferUsage, Storage},
    },
    command_buffer::BufferAccess,
};

impl<'a, T: Pod + Copy, U: BufferUsage> IntoIterator for BufferSlice<'a, T, U> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

#[derive(Copy, Clone, Debug)]
pub struct BufferSlice<'a, T: Copy + Pod, U: BufferUsage = Storage> {
    pub handle: vk::Buffer,
    pub size: u64,
    pub cpu_ptr: usize,
    pub gpu_ptr: u64,
    pub base_address: u64,
    pub(crate) _marker: PhantomData<T>,
    pub(crate) _usage: PhantomData<U>,
    pub(crate) _lifetime: PhantomData<&'a ()>,
}

fn new_slice<'a, T: Pod + Copy, R: RangeBounds<usize>, U: BufferUsage>(
    handle: vk::Buffer,
    index: R,
    cpu_ptr: usize,
    size: u64,
    address: u64,
    base_address: u64,
) -> BufferSlice<'a, T, U> {
    let start_offset = match index.start_bound() {
        std::ops::Bound::Unbounded => 0,
        std::ops::Bound::Excluded(size) => ((size + 1) * size_of::<T>()) as u64,
        std::ops::Bound::Included(size) => (size * size_of::<T>()) as u64,
    };
    BufferSlice {
        handle: handle,
        size: match index.end_bound() {
            std::ops::Bound::Unbounded => size,
            std::ops::Bound::Excluded(size) => (size * size_of::<T>()) as u64,
            std::ops::Bound::Included(size) => ((size + 1) * size_of::<T>()) as u64,
        } - start_offset,
        cpu_ptr: if cpu_ptr == 0 {
            0
        } else {
            cpu_ptr + start_offset as usize
        },
        gpu_ptr: address + start_offset,
        base_address,
        _marker: PhantomData,
        _usage: PhantomData,
        _lifetime: PhantomData,
    }
}

impl<T: Copy + Pod, U: BufferUsage> Buffer<T, U> {
    pub fn range<'a, R: RangeBounds<usize>>(&'a self, index: R) -> BufferSlice<'a, T, U> {
        new_slice(
            self.handle,
            index,
            self.allocation
                .mapped_ptr()
                .map(|e| e.as_ptr() as usize)
                .unwrap_or(0),
            self.size(),
            self.address,
            self.address,
        )
    }
    pub fn byte_range<'a, R: RangeBounds<usize>>(&'a self, index: R) -> BufferSlice<'a, u8, U> {
        new_slice::<u8, R, U>(
            self.handle,
            index.into(),
            self.allocation
                .mapped_ptr()
                .map(|e| e.as_ptr() as usize)
                .unwrap_or(0),
            self.size(),
            self.address,
            self.address,
        )
        .cast()
    }
}

impl<'a, T: Copy + Pod, U: BufferUsage> BufferSlice<'a, T, U> {
    pub(crate) fn access(
        &self,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
    ) -> BufferAccess {
        BufferAccess {
            stage,
            access,
            range: self.get_range(),
        }
    }
    pub fn get_range(&self) -> Range<u64> {
        (self.gpu_ptr..(self.gpu_ptr + self.size)).into()
    }
    pub fn range<R: RangeBounds<usize>>(self, index: R) -> BufferSlice<'a, T, U> {
        new_slice(
            self.handle,
            index,
            self.cpu_ptr,
            self.size,
            self.gpu_ptr,
            self.base_address,
        )
    }
    pub fn byte_range<R: RangeBounds<usize>>(self, index: R) -> BufferSlice<'a, u8, U> {
        new_slice::<u8, R, U>(
            self.handle,
            index,
            self.cpu_ptr,
            self.size,
            self.gpu_ptr,
            self.base_address,
        )
        .cast()
    }
    pub fn len(&self) -> usize {
        self.size as usize / size_of::<T>()
    }
    pub fn ptr(&self) -> *mut T {
        self.cpu_ptr as *mut T
    }
    pub fn region<'b, U2: BufferUsage>(&self, other: BufferSlice<'b, T, U2>) -> vk::BufferCopy {
        vk::BufferCopy {
            src_offset: self.offset(),
            dst_offset: other.offset(),
            size: self.size,
        }
    }
    pub fn offset(&self) -> u64 {
        self.gpu_ptr - self.base_address
    }
    pub fn cast<B: Copy + Pod>(self) -> BufferSlice<'a, B, U> {
        unsafe { std::mem::transmute(self) }
    }
    pub fn copy_from(self, slice: &[T]) {
        unsafe { self.ptr().copy_from(slice.as_ptr(), slice.len()) };
    }
    pub fn as_slice(self) -> &'a [T] {
        unsafe { std::slice::from_raw_parts(self.ptr(), self.len()) }
    }
}

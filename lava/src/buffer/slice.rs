use anyhow::{Result, anyhow};
use ash::vk;
use bytemuck::Pod;
use std::ops::{Bound, Index, Range, RangeBounds};
use std::{marker::PhantomData, slice::SliceIndex};
use std::sync::Arc;

use crate::{
    buffer::{Buffer},
    state::Ctx,
};

impl<'a, T: Pod+Copy> IntoIterator for BufferSlice<'a, T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

#[derive(Copy, Clone)]
pub struct BufferSlice<'a, T: Copy + Pod> {
    pub handle: vk::Buffer,
    pub size: u64,
    pub cpu_ptr: usize,
    pub gpu_ptr: u64,
    pub base_address: u64,
    pub(crate) _marker: PhantomData<T>,
    pub(crate) _lifetime: PhantomData<&'a ()>
}

impl<T: Copy + Pod> Buffer<T> {
    pub fn range<'a, R: RangeBounds<usize>>(&'a self, index: R) -> BufferSlice<'a, T> {
        let start_offset = match index.start_bound() {
            std::ops::Bound::Unbounded => 0,
            std::ops::Bound::Excluded(size) => ((size + 1) * size_of::<T>()) as u64,
            std::ops::Bound::Included(size) => (size * size_of::<T>()) as u64
        };
        BufferSlice {
            handle: self.handle,
            size: match index.end_bound() {
                std::ops::Bound::Unbounded => self.size(),
                std::ops::Bound::Excluded(size) => (size * size_of::<T>()) as u64,
                std::ops::Bound::Included(size) => ((size + 1) * size_of::<T>()) as u64
            },
            cpu_ptr: self.allocation.mapped_ptr().map(|e| e.as_ptr() as usize).unwrap_or(0) + start_offset as usize,
            gpu_ptr: self.address + start_offset,
            base_address: self.address,
            _marker: PhantomData,
            _lifetime: PhantomData,
        }
    } 
    pub fn byte_range<'a, R: RangeBounds<u64>>(&'a self, index: R) -> BufferSlice<'a, T> {
        let start_offset = match index.start_bound() {
            std::ops::Bound::Unbounded => 0,
            std::ops::Bound::Excluded(size) => size + 1,
            std::ops::Bound::Included(size) => *size
        };
        BufferSlice {
            handle: self.handle,
            size: match index.end_bound() {
                std::ops::Bound::Unbounded => self.size(),
                std::ops::Bound::Excluded(size) => size-1,
                std::ops::Bound::Included(size) => *size
            },
            cpu_ptr: self.allocation.mapped_ptr().map(|e| e.as_ptr() as usize).unwrap_or(0) + start_offset as usize,
            gpu_ptr: self.address + start_offset,
            base_address: self.address,
            _marker: PhantomData,
            _lifetime: PhantomData,
        }
    } 
}

impl<'a, T: Copy + Pod> BufferSlice<'a, T> {
    pub fn range<R: RangeBounds<usize>>(self, index: R) -> BufferSlice<'a, T> {
        let start_offset = match index.start_bound() {
            std::ops::Bound::Unbounded => 0,
            std::ops::Bound::Excluded(size) => ((size + 1) * size_of::<T>()) as u64,
            std::ops::Bound::Included(size) => (size * size_of::<T>()) as u64
        };
        BufferSlice {
            handle: self.handle,
            size: match index.end_bound() {
                std::ops::Bound::Unbounded => self.size,
                std::ops::Bound::Excluded(size) => ((size-1) * size_of::<T>()) as u64,
                std::ops::Bound::Included(size) => (size * size_of::<T>()) as u64
            }  - start_offset,
            cpu_ptr: self.cpu_ptr + start_offset as usize,
            gpu_ptr: self.gpu_ptr + start_offset,
            base_address: self.base_address,
            _marker: PhantomData,
            _lifetime: PhantomData,
        }
    }
    pub fn byte_range<R: RangeBounds<u64>>(self, index: R) -> BufferSlice<'a, T> {
        let start_offset = match index.start_bound() {
            std::ops::Bound::Unbounded => 0,
            std::ops::Bound::Excluded(size) => size + 1,
            std::ops::Bound::Included(size) => *size
        };
        BufferSlice {
            handle: self.handle,
            size: match index.end_bound() {
                std::ops::Bound::Unbounded => self.size,
                std::ops::Bound::Excluded(size) => size-1,
                std::ops::Bound::Included(size) => *size
            },
            cpu_ptr: self.cpu_ptr + start_offset as usize,
            gpu_ptr: self.gpu_ptr + start_offset,
            base_address: self.base_address,
            _marker: PhantomData,
            _lifetime: PhantomData,
        }
    }
    pub fn len(&self) -> usize {
        self.size as usize / size_of::<T>()
    }
    pub fn ptr(&self) -> *mut T {
        self.cpu_ptr as *mut T
    }
    pub fn region<'b>(&self, other: BufferSlice<'b, T>) -> vk::BufferCopy {
        vk::BufferCopy {
            src_offset: self.offset(),
            dst_offset: other.offset(),
            size: self.size,
        }
    }
    pub fn offset(&self) -> u64 {
        self.gpu_ptr - self.base_address 
    }
    pub fn cast<B: Copy + Pod>(self) -> BufferSlice<'a, B> {
        unsafe { std::mem::transmute(self) }
    }
    pub fn copy_from(self, slice: &[T]) {
        unsafe { self.ptr().copy_from(slice.as_ptr(), slice.len()) };
    }
    pub fn as_slice(self) -> &'a [T] {
        unsafe { std::slice::from_raw_parts(self.ptr(), self.len()) }
    }
}

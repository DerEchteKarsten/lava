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
    let element = size_of::<T>() as u64;
    let start_offset = match index.start_bound() {
        std::ops::Bound::Unbounded => 0,
        std::ops::Bound::Excluded(start) => (*start as u64 + 1) * element,
        std::ops::Bound::Included(start) => *start as u64 * element,
    };
    let end_offset = match index.end_bound() {
        std::ops::Bound::Unbounded => size,
        std::ops::Bound::Excluded(end) => *end as u64 * element,
        std::ops::Bound::Included(end) => (*end as u64 + 1) * element,
    };
    assert!(
        start_offset <= end_offset,
        "buffer slice starts at byte {start_offset} but ends at byte {end_offset}"
    );
    assert!(
        end_offset <= size,
        "buffer slice ends at byte {end_offset} but the buffer is only {size} bytes long"
    );
    BufferSlice {
        handle: handle,
        size: end_offset - start_offset,
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
    pub fn is_mapped(&self) -> bool {
        self.cpu_ptr != 0
    }
    pub(crate) fn element_ptr(&self, index: usize) -> *mut T {
        assert!(self.is_mapped(), "buffer is not host-mapped");
        assert!(
            index < self.len(),
            "index {index} is out of bounds for a buffer slice of length {}",
            self.len()
        );
        unsafe { self.ptr().add(index) }
    }
    pub fn copy_from(self, slice: &[T]) {
        assert!(self.is_mapped(), "buffer is not host-mapped");
        assert!(
            slice.len() <= self.len(),
            "cannot copy {} elements into a buffer slice of length {}",
            slice.len(),
            self.len()
        );
        unsafe { self.ptr().copy_from(slice.as_ptr(), slice.len()) };
    }
    pub fn as_slice(self) -> &'a [T] {
        assert!(self.is_mapped(), "buffer is not host-mapped");
        unsafe { std::slice::from_raw_parts(self.ptr(), self.len()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::usage::Storage;

    /// GPU address the fake buffers pretend to live at.
    const BASE: u64 = 0x1000;

    /// A slice over host memory: everything except GPU commands works on it without a device.
    fn host_slice(data: &mut [u32]) -> BufferSlice<'_, u32, Storage> {
        new_slice(
            vk::Buffer::null(),
            ..,
            data.as_mut_ptr() as usize,
            size_of_val(data) as u64,
            BASE,
            BASE,
        )
    }

    /// A slice of `len` u32s without host mapping, as for GPU-only memory.
    fn unmapped_slice(len: u64) -> BufferSlice<'static, u32, Storage> {
        new_slice(vk::Buffer::null(), .., 0, len * 4, BASE, BASE)
    }

    #[test]
    fn whole_slice_covers_the_buffer() {
        let mut data = [0u32; 8];
        let slice = host_slice(&mut data);
        assert_eq!(slice.len(), 8);
        assert_eq!(slice.size, 32);
        assert_eq!(slice.offset(), 0);
        assert_eq!(slice.gpu_ptr, BASE);
        assert_eq!(slice.get_range(), (BASE..BASE + 32).into());
    }

    #[test]
    fn range_bounds_are_in_elements() {
        let mut data = [0u32; 8];
        let slice = host_slice(&mut data);

        let half_open = slice.range(2..5);
        assert_eq!((half_open.offset(), half_open.len()), (8, 3));

        let inclusive = slice.range(2..=5);
        assert_eq!((inclusive.offset(), inclusive.len()), (8, 4));

        let from = slice.range(6..);
        assert_eq!((from.offset(), from.len()), (24, 2));

        let to = slice.range(..3);
        assert_eq!((to.offset(), to.len()), (0, 3));

        let excluded_start =
            slice.range((std::ops::Bound::Excluded(1), std::ops::Bound::Unbounded));
        assert_eq!((excluded_start.offset(), excluded_start.len()), (8, 6));

        let empty = slice.range(4..4);
        assert_eq!((empty.offset(), empty.len()), (16, 0));
    }

    #[test]
    fn nested_ranges_are_relative_to_the_parent_slice() {
        let mut data = [0u32; 16];
        let slice = host_slice(&mut data);
        let inner = slice.range(4..12).range(2..4);
        assert_eq!((inner.offset(), inner.len()), (24, 2));
        assert_eq!(inner.gpu_ptr, BASE + 24);
        // The base address is kept, so offsets stay relative to the buffer start.
        assert_eq!(inner.base_address, BASE);
        assert_eq!(inner.range(..).len(), 2);
    }

    #[test]
    fn byte_range_addresses_bytes_and_casts_reinterpret() {
        let mut data = [0u32; 8];
        let slice = host_slice(&mut data);
        let bytes = slice.byte_range(4..20);
        assert_eq!((bytes.offset(), bytes.len(), bytes.size), (4, 16, 16));

        let words = bytes.cast::<u32>();
        assert_eq!((words.offset(), words.len()), (4, 4));
        let pairs = slice.cast::<[u32; 2]>();
        assert_eq!(pairs.len(), 4);
    }

    #[test]
    fn region_copies_this_slice_to_the_other_slices_offset() {
        let mut a = [0u32; 8];
        let mut b = [0u32; 8];
        let src = host_slice(&mut a).range(1..3);
        let dst = host_slice(&mut b).range(5..);
        let region = src.region(dst);
        assert_eq!(
            (region.src_offset, region.dst_offset, region.size),
            (4, 20, 8)
        );
    }

    #[test]
    fn access_covers_the_gpu_address_range() {
        let mut data = [0u32; 8];
        let access = host_slice(&mut data).range(2..4).access(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_STORAGE_READ,
        );
        assert_eq!(access.range, (BASE + 8..BASE + 16).into());
        assert_eq!(access.stage, vk::PipelineStageFlags2::COMPUTE_SHADER);
        assert_eq!(access.access, vk::AccessFlags2::SHADER_STORAGE_READ);
    }

    #[test]
    fn host_reads_and_writes_go_to_the_sliced_elements() {
        let mut data = [0u32, 1, 2, 3, 4, 5, 6, 7];
        let slice = host_slice(&mut data);

        assert_eq!(slice.range(2..5).as_slice(), &[2, 3, 4]);
        assert_eq!(
            slice.range(6..).into_iter().copied().collect::<Vec<_>>(),
            [6, 7]
        );
        assert_eq!(slice.range(3..)[1], 4);

        slice.range(1..4).copy_from(&[10, 11]);
        let mut tail = slice.range(7..);
        tail[0] = 70;
        assert_eq!(data, [0, 10, 11, 3, 4, 5, 6, 70]);
    }

    #[test]
    fn unmapped_slices_keep_a_null_host_pointer() {
        let slice = unmapped_slice(8).range(2..4);
        assert!(!slice.is_mapped());
        assert_eq!(slice.cpu_ptr, 0);
        // GPU-side math still works.
        assert_eq!((slice.offset(), slice.len()), (8, 2));
    }

    #[test]
    #[should_panic(expected = "ends at byte")]
    fn range_with_end_before_start_panics() {
        let mut data = [0u32; 8];
        #[allow(clippy::reversed_empty_ranges)]
        let _ = host_slice(&mut data).range(5..2);
    }

    #[test]
    #[should_panic(expected = "only 32 bytes long")]
    fn range_past_the_end_panics() {
        let mut data = [0u32; 8];
        let _ = host_slice(&mut data).range(4..9);
    }

    #[test]
    #[should_panic(expected = "only 8 bytes long")]
    fn nested_range_past_the_parent_slice_panics() {
        let mut data = [0u32; 8];
        let _ = host_slice(&mut data).range(0..2).range(0..3);
    }

    #[test]
    #[should_panic(expected = "cannot copy 3 elements")]
    fn copy_from_longer_data_panics() {
        let mut data = [0u32; 8];
        host_slice(&mut data).range(0..2).copy_from(&[1, 2, 3]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn indexing_past_the_end_panics() {
        let mut data = [0u32; 8];
        let _ = host_slice(&mut data).range(0..2)[2];
    }

    #[test]
    #[should_panic(expected = "not host-mapped")]
    fn indexing_unmapped_memory_panics() {
        let _ = unmapped_slice(8)[1];
    }

    #[test]
    #[should_panic(expected = "not host-mapped")]
    fn as_slice_on_unmapped_memory_panics() {
        let _ = unmapped_slice(8).as_slice();
    }

    #[test]
    #[should_panic(expected = "not host-mapped")]
    fn copy_from_into_unmapped_memory_panics() {
        unmapped_slice(8).copy_from(&[1]);
    }
}

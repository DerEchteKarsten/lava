//! Typed buffer slices: the address ranges commands take (no buffer handle or offset), sub-ranges, byte views and casts, and the buffer transfer commands (fill, update, copy)
use crate::vk;
use bytemuck::Pod;
use std::marker::PhantomData;
use std::ops::RangeBounds;
use std::range::Range;

use lava_macros::validation_trace;
use smallvec::SmallVec;
use std::iter;

use crate::{
    buffer::{
        Buffer,
        usage::{BufferUsage, Storage},
    },
    command_buffer::{BufferAccess, CommandBuffer},
    queue::QueueKind,
    state::Functions,
};

/// A range of a buffer as the GPU sees it: an address, a size and the usage. It borrows the
/// buffer but gives no host access, so copies of it alias nothing.
#[derive(Copy, Clone, Debug)]
pub struct BufferSlice<'a, T: Copy + Pod, U: BufferUsage = Storage> {
    pub(crate) address: u64,
    pub(crate) size: u64,
    pub(crate) usage: U,
    pub(crate) _marker: PhantomData<&'a [T]>,
}

fn new_slice<'a, T: Pod + Copy, R: RangeBounds<usize>, U: BufferUsage>(
    usage: U,
    index: R,
    address: u64,
    size: u64,
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
        address: address + start_offset,
        size: end_offset - start_offset,
        usage,
        _marker: PhantomData,
    }
}

impl<T: Copy + Pod, U: BufferUsage> Buffer<T, U> {
    pub fn range<'a, R: RangeBounds<usize>>(&'a self, index: R) -> BufferSlice<'a, T, U> {
        new_slice(self.usage(), index, self.address, self.size())
    }
    pub fn byte_range<'a, R: RangeBounds<usize>>(&'a self, index: R) -> BufferSlice<'a, u8, U> {
        new_slice::<u8, R, U>(self.usage(), index, self.address, self.size())
    }
}

impl<'a, T: Copy + Pod, U: BufferUsage> BufferSlice<'a, T, U> {
    /// A pass's access of the slice, for [`crate::command_buffer::Bindings::new`].
    pub fn access(
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
    /// The addresses the slice covers.
    pub fn get_range(&self) -> Range<u64> {
        (self.address..(self.address + self.size)).into()
    }
    pub fn range<R: RangeBounds<usize>>(self, index: R) -> BufferSlice<'a, T, U> {
        new_slice(self.usage, index, self.address, self.size)
    }
    pub fn byte_range<R: RangeBounds<usize>>(self, index: R) -> BufferSlice<'a, u8, U> {
        new_slice::<u8, R, U>(self.usage, index, self.address, self.size)
    }
    pub fn len(&self) -> usize {
        self.size as usize / size_of::<T>()
    }
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
    pub fn size(&self) -> u64 {
        self.size
    }
    pub fn address(&self) -> u64 {
        self.address
    }
    pub fn cast<B: Copy + Pod>(self) -> BufferSlice<'a, B, U> {
        BufferSlice {
            address: self.address,
            size: self.size,
            usage: self.usage,
            _marker: PhantomData,
        }
    }
    pub fn usage(&self) -> U {
        self.usage
    }
    /// Panics unless the usage has every flag of `required`; does nothing for a static usage.
    pub fn usage_check(&self, required: vk::BufferUsageFlags) {
        self.usage.check(required);
    }
    pub(crate) fn address_range(&self) -> vk::DeviceAddressRangeKHR {
        vk::DeviceAddressRangeKHR {
            address: self.address,
            size: self.size,
        }
    }
    pub(crate) fn strided_range(&self) -> vk::StridedDeviceAddressRangeKHR {
        vk::StridedDeviceAddressRangeKHR {
            address: self.address,
            size: self.size,
            stride: size_of::<T>() as u64,
        }
    }
    /// Whether the range's memory has storage usage (it is this buffer's alone), and fully
    /// bound (lava has no sparse memory).
    pub(crate) fn address_flags(&self) -> vk::AddressCommandFlagsKHR {
        let storage = if self
            .usage
            .flags()
            .contains(vk::BufferUsageFlags::STORAGE_BUFFER)
        {
            vk::AddressCommandFlagsKHR::STORAGE_BUFFER_USAGE
        } else {
            vk::AddressCommandFlagsKHR::UNKNOWN_STORAGE_BUFFER_USAGE
        };
        storage | vk::AddressCommandFlagsKHR::FULLY_BOUND
    }
}

impl<K: QueueKind> CommandBuffer<'_, K> {
    #[validation_trace]
    pub fn fill_buffer<'a, T: Copy + Pod, U: BufferUsage>(
        &mut self,
        buffer: BufferSlice<'a, T, U>,
        data: u32,
    ) {
        self.rec.flush_pending(
            std::iter::empty(),
            std::iter::once(BufferAccess {
                access: vk::AccessFlags2::TRANSFER_WRITE,
                stage: vk::PipelineStageFlags2::TRANSFER,
                range: buffer.get_range(),
            }),
        );
        unsafe {
            Functions::device_address_commands().cmd_fill_memory(
                self.rec.handle,
                &buffer.address_range(),
                buffer.address_flags(),
                data,
            )
        };
    }

    #[validation_trace]
    pub fn update_buffer<'a, T: Copy + Pod, U: BufferUsage>(
        &mut self,
        buffer: BufferSlice<'a, T, U>,
        data: &T,
    ) {
        self.rec.flush_pending(
            iter::empty(),
            iter::once(BufferAccess {
                stage: vk::PipelineStageFlags2::TRANSFER,
                access: vk::AccessFlags2::TRANSFER_WRITE,
                range: buffer.get_range(),
            }),
        );
        unsafe {
            Functions::device_address_commands().cmd_update_memory(
                self.rec.handle,
                &buffer.address_range(),
                buffer.address_flags(),
                bytemuck::bytes_of(data),
            )
        };
    }
    /// Copies `src` to the start of `dst`, which must be at least as long.
    #[validation_trace]
    pub fn copy_buffer<'a, T: Copy + Pod, U: BufferUsage, U2: BufferUsage>(
        &mut self,
        src: BufferSlice<'a, T, U>,
        dst: BufferSlice<'a, T, U2>,
    ) {
        self.copy_buffer_regions(&[(src, dst)]);
    }
    /// One `copy_buffer` per pair, in one command.
    #[validation_trace]
    pub fn copy_buffer_regions<'a, T: Copy + Pod, U: BufferUsage, U2: BufferUsage>(
        &mut self,
        regions: &[(BufferSlice<'a, T, U>, BufferSlice<'a, T, U2>)],
    ) {
        let accesses = regions
            .iter()
            .flat_map(|(src, dst)| {
                assert!(
                    src.size <= dst.size,
                    "cannot copy {} bytes into a buffer slice of {} bytes",
                    src.size,
                    dst.size
                );
                [
                    src.access(
                        vk::PipelineStageFlags2::TRANSFER,
                        vk::AccessFlags2::TRANSFER_READ,
                    ),
                    dst.range(..src.len()).access(
                        vk::PipelineStageFlags2::TRANSFER,
                        vk::AccessFlags2::TRANSFER_WRITE,
                    ),
                ]
            })
            .collect::<SmallVec<[_; 4]>>();
        self.rec.flush_pending(iter::empty(), accesses.iter().cloned());
        let copies = regions
            .iter()
            .map(|(src, dst)| {
                vk::DeviceMemoryCopyKHR::default()
                    .src_range(src.address_range())
                    .src_flags(src.address_flags())
                    .dst_range(dst.range(..src.len()).address_range())
                    .dst_flags(dst.address_flags())
            })
            .collect::<SmallVec<[_; 4]>>();
        unsafe {
            Functions::device_address_commands().cmd_copy_memory(
                self.rec.handle,
                &vk::CopyDeviceMemoryInfoKHR::default().regions(&copies),
            )
        };
    }
}

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
    buffer::Buffer,
    command_buffer::{BufferAccess, CommandBuffer},
    queue::QueueKind,
    state::Functions,
};

/// A range of a buffer as the GPU sees it: an address and a size. It borrows the buffer but
/// gives no host access, so copies of it alias nothing.
#[derive(Copy, Clone, Debug)]
pub struct BufferSlice<'a, T: Copy + Pod> {
    pub(crate) address: u64,
    pub(crate) size: u64,
    pub(crate) _marker: PhantomData<&'a [T]>,
}

fn new_slice<'a, T: Pod + Copy, R: RangeBounds<usize>>(
    index: R,
    address: u64,
    size: u64,
) -> BufferSlice<'a, T> {
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
        _marker: PhantomData,
    }
}

impl<T: Copy + Pod> Buffer<T> {
    pub fn range<'a, R: RangeBounds<usize>>(&'a self, index: R) -> BufferSlice<'a, T> {
        new_slice(index, self.address, self.size())
    }
    pub fn byte_range<'a, R: RangeBounds<usize>>(&'a self, index: R) -> BufferSlice<'a, u8> {
        new_slice::<u8, R>(index, self.address, self.size())
    }
}

impl<'a, T: Copy + Pod> BufferSlice<'a, T> {
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
    pub fn range<R: RangeBounds<usize>>(self, index: R) -> BufferSlice<'a, T> {
        new_slice(index, self.address, self.size)
    }
    pub fn byte_range<R: RangeBounds<usize>>(self, index: R) -> BufferSlice<'a, u8> {
        new_slice::<u8, R>(index, self.address, self.size)
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
    /// The same range of `B`s; panics unless its size is a multiple of `B`'s and its address
    /// aligned to `B`.
    pub fn cast<B: Copy + Pod>(self) -> BufferSlice<'a, B> {
        cast_check::<B>(self.address, self.size);
        BufferSlice {
            address: self.address,
            size: self.size,
            _marker: PhantomData,
        }
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
    /// The range's memory has storage usage, as every application buffer has (`USAGE`), and
    /// is fully bound (lava has no sparse memory).
    pub(crate) fn address_flags(&self) -> vk::AddressCommandFlagsKHR {
        vk::AddressCommandFlagsKHR::STORAGE_BUFFER_USAGE | vk::AddressCommandFlagsKHR::FULLY_BOUND
    }
}

#[track_caller]
pub(crate) fn cast_check<B>(address: u64, size: u64) {
    let (element, align) = (size_of::<B>() as u64, align_of::<B>() as u64);
    assert!(
        element != 0 && size % element == 0,
        "{size} bytes aren't a whole number of {element}-byte elements"
    );
    assert!(
        address % align == 0,
        "address {address:#x} isn't aligned to the {align} bytes of the elements"
    );
}

/// What fills and updates write: 4-byte aligned, whole words.
#[track_caller]
fn check_words(address: u64, size: u64, what: &str) {
    assert!(
        address % 4 == 0 && size % 4 == 0,
        "{what} writes whole 4-byte words at 4-byte aligned addresses, not {size} bytes at {address:#x}"
    );
}

/// `vkCmdUpdateBuffer`'s limit on the bytes of one update.
const MAX_UPDATE: usize = 65536;

impl<K: QueueKind> CommandBuffer<'_, K> {
    /// Writes `data`, repeated, into every 4-byte word of `buffer`, whose address and size have
    /// to be multiples of 4.
    #[validation_trace]
    pub fn fill_buffer<'a, T: Copy + Pod>(
        &mut self,
        buffer: BufferSlice<'a, T>,
        data: u32,
    ) {
        check_words(buffer.address, buffer.size, "a fill");
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

    /// Writes `data` into the first element of `buffer`, from the command buffer itself: for
    /// small writes (at most 64 KiB, whole 4-byte words at a 4-byte aligned address).
    #[validation_trace]
    pub fn update_buffer<'a, T: Copy + Pod>(
        &mut self,
        buffer: BufferSlice<'a, T>,
        data: &T,
    ) {
        assert!(!buffer.is_empty(), "an update writes the first element of an empty slice");
        assert!(
            size_of::<T>() <= MAX_UPDATE,
            "an update writes at most {MAX_UPDATE} bytes, not {}",
            size_of::<T>()
        );
        let buffer = buffer.range(..1);
        check_words(buffer.address, buffer.size, "an update");
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
    pub fn copy_buffer<'a, T: Copy + Pod>(
        &mut self,
        src: BufferSlice<'a, T>,
        dst: BufferSlice<'a, T>,
    ) {
        self.copy_buffer_regions(&[(src, dst)]);
    }
    /// One `copy_buffer` per pair, in one command.
    #[validation_trace]
    pub fn copy_buffer_regions<'a, T: Copy + Pod>(
        &mut self,
        regions: &[(BufferSlice<'a, T>, BufferSlice<'a, T>)],
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

//! Typed buffer slices: the address ranges commands take (no buffer handle or offset; a sparse buffer's not claimed fully bound), sub-ranges, byte views and casts, and the buffer transfer commands (fill, update, copy; empty ones record nothing, copy regions may not overlap)
use crate::vk;
use bytemuck::NoUninit;
use std::marker::PhantomData;
use std::ops::{Bound, RangeBounds};
use std::range::Range;

use lava_macros::validation_trace;
use smallvec::SmallVec;
use std::iter;

use crate::{
    Error, Result,
    buffer::{Buffer, Memory},
    command_buffer::{BufferAccess, CommandBuffer},
    queue::QueueKind,
    state::Functions,
};

/// A range of a buffer as the GPU sees it: an address and a size. It borrows the buffer but
/// gives no host access, so copies of it alias nothing.
#[derive(Copy, Clone, Debug)]
pub struct BufferSlice<'a, T: Copy + 'static> {
    pub(crate) address: u64,
    pub(crate) size: u64,
    /// Of a sparse buffer, whose pages may be unbound: commands don't claim it fully bound.
    pub(crate) sparse: bool,
    pub(crate) _marker: PhantomData<&'a [T]>,
}

#[track_caller]
pub(crate) fn new_slice<'a, T: Copy + 'static, R: RangeBounds<usize>>(
    index: R,
    address: u64,
    size: u64,
) -> BufferSlice<'a, T> {
    try_new_slice(index, address, size).unwrap_or_else(|error| panic!("{error}"))
}

/// The `index` range of `T`s within the `size` bytes at `address`, or
/// [`Error::SliceOutOfRange`] when it starts after it ends or ends past `size`.
pub(crate) fn try_new_slice<'a, T: Copy + 'static, R: RangeBounds<usize>>(
    index: R,
    address: u64,
    size: u64,
) -> Result<BufferSlice<'a, T>> {
    let element = size_of::<T>() as u64;
    // Saturating, so an offset past `u64::MAX` fails the checks below instead of overflowing.
    let offset = |i: usize, plus_one: bool| {
        (i as u64)
            .saturating_add(plus_one as u64)
            .saturating_mul(element)
    };
    let start = match index.start_bound() {
        Bound::Unbounded => 0,
        Bound::Excluded(start) => offset(*start, true),
        Bound::Included(start) => offset(*start, false),
    };
    let end = match index.end_bound() {
        Bound::Unbounded => size,
        Bound::Excluded(end) => offset(*end, false),
        Bound::Included(end) => offset(*end, true),
    };
    if start > end || end > size {
        return Err(Error::SliceOutOfRange { start, end, size });
    }
    Ok(BufferSlice {
        address: address + start,
        size: end - start,
        sparse: false,
        _marker: PhantomData,
    })
}

impl<T: Copy + 'static, M: Memory> Buffer<T, M> {
    /// The `index` range of the buffer's elements; panics where [`Self::try_range`] fails.
    #[track_caller]
    pub fn range<'a, R: RangeBounds<usize>>(&'a self, index: R) -> BufferSlice<'a, T> {
        new_slice(index, self.address, self.size())
    }
    /// The `index` range of the buffer's bytes; panics where [`Self::try_byte_range`] fails.
    #[track_caller]
    pub fn byte_range<'a, R: RangeBounds<usize>>(&'a self, index: R) -> BufferSlice<'a, u8> {
        new_slice::<u8, R>(index, self.address, self.size())
    }
    /// The `index` range of the buffer's elements, or [`Error::SliceOutOfRange`] when it starts
    /// after it ends or ends past the buffer.
    pub fn try_range<'a, R: RangeBounds<usize>>(&'a self, index: R) -> Result<BufferSlice<'a, T>> {
        try_new_slice(index, self.address, self.size())
    }
    /// The `index` range of the buffer's bytes, or [`Error::SliceOutOfRange`] when it starts
    /// after it ends or ends past the buffer.
    pub fn try_byte_range<'a, R: RangeBounds<usize>>(
        &'a self,
        index: R,
    ) -> Result<BufferSlice<'a, u8>> {
        try_new_slice::<u8, R>(index, self.address, self.size())
    }
}

impl<'a, T: Copy + 'static> BufferSlice<'a, T> {
    /// The `size` bytes at `address`, which the caller keeps alive for `'a`.
    pub(crate) fn new(address: u64, size: u64) -> Self {
        BufferSlice {
            address,
            size,
            sparse: false,
            _marker: PhantomData,
        }
    }
    /// The same range of bytes.
    pub fn bytes(self) -> BufferSlice<'a, u8> {
        BufferSlice {
            address: self.address,
            size: self.size,
            sparse: self.sparse,
            _marker: PhantomData,
        }
    }
    /// A pass's access of the slice, for [`crate::command_buffer::Bindings::new`].
    pub fn access(&self, stage: vk::PipelineStageFlags2, access: vk::AccessFlags2) -> BufferAccess {
        BufferAccess {
            unordered: false,
            stage,
            access,
            range: self.get_range(),
        }
    }
    /// The addresses the slice covers.
    pub fn get_range(&self) -> Range<u64> {
        (self.address..(self.address + self.size)).into()
    }
    /// The `index` range of the slice's elements; panics where [`Self::try_range`] fails.
    #[track_caller]
    pub fn range<R: RangeBounds<usize>>(self, index: R) -> BufferSlice<'a, T> {
        BufferSlice {
            sparse: self.sparse,
            ..new_slice(index, self.address, self.size)
        }
    }
    /// The `index` range of the slice's bytes; panics where [`Self::try_byte_range`] fails.
    #[track_caller]
    pub fn byte_range<R: RangeBounds<usize>>(self, index: R) -> BufferSlice<'a, u8> {
        BufferSlice {
            sparse: self.sparse,
            ..new_slice::<u8, R>(index, self.address, self.size)
        }
    }
    /// The `index` range of the slice's elements, or [`Error::SliceOutOfRange`] when it starts
    /// after it ends or ends past the slice.
    pub fn try_range<R: RangeBounds<usize>>(self, index: R) -> Result<BufferSlice<'a, T>> {
        let slice = try_new_slice(index, self.address, self.size)?;
        Ok(BufferSlice {
            sparse: self.sparse,
            ..slice
        })
    }
    /// The `index` range of the slice's bytes, or [`Error::SliceOutOfRange`] when it starts
    /// after it ends or ends past the slice.
    pub fn try_byte_range<R: RangeBounds<usize>>(self, index: R) -> Result<BufferSlice<'a, u8>> {
        let slice = try_new_slice::<u8, R>(index, self.address, self.size)?;
        Ok(BufferSlice {
            sparse: self.sparse,
            ..slice
        })
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
    pub fn cast<B: Copy + 'static>(self) -> BufferSlice<'a, B> {
        cast_check::<B>(self.address, self.size);
        BufferSlice {
            address: self.address,
            size: self.size,
            sparse: self.sparse,
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
    /// is fully bound unless it is a sparse buffer's.
    pub(crate) fn address_flags(&self) -> vk::AddressCommandFlagsKHR {
        match self.sparse {
            true => vk::AddressCommandFlagsKHR::STORAGE_BUFFER_USAGE,
            false => {
                vk::AddressCommandFlagsKHR::STORAGE_BUFFER_USAGE
                    | vk::AddressCommandFlagsKHR::FULLY_BOUND
            }
        }
    }
}

#[track_caller]
pub(crate) fn cast_check<B>(address: u64, size: u64) {
    const { assert!(size_of::<B>() > 0, "buffers of zero-sized elements have no length") };
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

/// `vkCmdUpdateMemoryKHR`'s limit on the bytes of one update.
pub(crate) const MAX_UPDATE: usize = 65536;

impl<K: QueueKind> CommandBuffer<'_, K> {
    /// Writes `data`, repeated, into every 4-byte word of `buffer`, whose address and size have
    /// to be multiples of 4. An empty slice records nothing.
    #[validation_trace]
    pub fn fill_buffer<'a, T: Copy + 'static>(&mut self, buffer: BufferSlice<'a, T>, data: u32) {
        check_words(buffer.address, buffer.size, "a fill");
        if buffer.is_empty() {
            return;
        }
        self.rec.flush_pending(
            std::iter::empty(),
            std::iter::once(BufferAccess {
                unordered: false,
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

    /// Writes `data` to the start of `buffer`, from the command buffer itself: for small
    /// writes (at most 64 KiB, whole 4-byte words at a 4-byte aligned address); larger ones go
    /// through an [`Uploader`](crate::buffer::Uploader). Empty `data` records nothing.
    #[validation_trace]
    pub fn update_buffer<'a, T: NoUninit>(&mut self, buffer: BufferSlice<'a, T>, data: &[T]) {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        assert!(
            data.len() <= buffer.len(),
            "an update of {} elements into a slice of {}",
            data.len(),
            buffer.len()
        );
        assert!(
            bytes.len() <= MAX_UPDATE,
            "an update writes at most {MAX_UPDATE} bytes, not {}",
            bytes.len()
        );
        let buffer = buffer.byte_range(..bytes.len());
        check_words(buffer.address, buffer.size, "an update");
        if bytes.is_empty() {
            return;
        }
        self.rec.flush_pending(
            iter::empty(),
            iter::once(BufferAccess {
                unordered: false,
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
                bytes,
            )
        };
    }
    /// Copies `src` to the start of `dst`, which must be at least as long and not overlap it.
    #[validation_trace]
    pub fn copy_buffer<'a, T: Copy + 'static>(&mut self, src: BufferSlice<'a, T>, dst: BufferSlice<'a, T>) {
        self.copy_buffer_regions(&[(src, dst)]);
    }
    /// One `copy_buffer` per pair, in one command. No region's destination may overlap
    /// another destination or any source; empty regions are skipped, and a list of only those
    /// records nothing.
    #[validation_trace]
    pub fn copy_buffer_regions<'a, T: Copy + 'static>(
        &mut self,
        regions: &[(BufferSlice<'a, T>, BufferSlice<'a, T>)],
    ) {
        // Every region as the bytes it reads and writes.
        let regions: SmallVec<[_; 4]> = regions
            .iter()
            .filter(|(src, _)| !src.is_empty())
            .map(|&(src, dst)| {
                assert!(
                    src.size <= dst.size,
                    "cannot copy {} bytes into a buffer slice of {} bytes",
                    src.size,
                    dst.size
                );
                (src.bytes(), dst.byte_range(..src.size as usize))
            })
            .collect();
        if regions.is_empty() {
            return;
        }
        check_copy_overlaps(regions.iter().map(|(src, dst)| (src.get_range(), dst.get_range())));
        let accesses = regions.iter().flat_map(|(src, dst)| {
            [
                src.access(vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_READ),
                dst.access(vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE),
            ]
        });
        self.rec.flush_pending(iter::empty(), accesses);
        let copies = regions
            .iter()
            .map(|(src, dst)| {
                vk::DeviceMemoryCopyKHR::default()
                    .src_range(src.address_range())
                    .src_flags(src.address_flags())
                    .dst_range(dst.address_range())
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

/// Panics if a copy's destination overlaps another destination or any source (`(src, dst)`
/// address ranges, none empty): sorted by their starts, a range overlaps an earlier one exactly
/// when it starts before that one's end.
#[track_caller]
pub(crate) fn check_copy_overlaps(regions: impl Iterator<Item = (Range<u64>, Range<u64>)>) {
    let mut ranges: SmallVec<[(Range<u64>, bool, usize); 8]> = SmallVec::new();
    for (i, (src, dst)) in regions.enumerate() {
        ranges.push((src, false, i));
        ranges.push((dst, true, i));
    }
    ranges.sort_unstable_by_key(|(range, ..)| range.start);
    // The furthest end of the destinations so far, and of every range, with their regions.
    let (mut dst_end, mut any_end): ((u64, usize), (u64, usize)) = ((0, 0), (0, 0));
    for &(range, is_dst, i) in &ranges {
        let earlier = if range.start < dst_end.0 {
            Some(dst_end.1)
        } else if is_dst && range.start < any_end.0 {
            Some(any_end.1)
        } else {
            None
        };
        if earlier == Some(i) {
            panic!(
                "copy region {i} reads and writes the addresses {:#x}..{:#x}: a copy's source \
                 and destination may not overlap",
                range.start, range.end
            );
        }
        if let Some(j) = earlier {
            panic!(
                "copy regions {j} and {i} overlap (the addresses {:#x}..{:#x}): no region may write \
                 what another reads or writes",
                range.start, range.end
            );
        }
        if is_dst && range.end > dst_end.0 {
            dst_end = (range.end, i);
        }
        if range.end > any_end.0 {
            any_end = (range.end, i);
        }
    }
}

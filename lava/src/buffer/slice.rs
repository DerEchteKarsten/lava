//! Typed buffer slices: the address ranges commands take (no buffer handle or offset), sub-ranges, byte views and casts
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

/// A range of a buffer as the GPU sees it: an address, a size and the usage. `Copy`, since a
/// frame hands the same range to several passes; it borrows the buffer, so the buffer outlives
/// the recording, but gives no host access (that is `Buffer::host`/`host_mut`), so copying it
/// can't alias anything. Every command takes it as a `VkDeviceAddressRangeKHR`
/// (`VK_KHR_device_address_commands`).
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
    /// What the address commands take.
    pub(crate) fn address_range(&self) -> vk::DeviceAddressRangeKHR {
        vk::DeviceAddressRangeKHR {
            address: self.address,
            size: self.size,
        }
    }
    /// The same range with the stride of its elements, for indirect commands.
    pub(crate) fn strided_range(&self) -> vk::StridedDeviceAddressRangeKHR {
        vk::StridedDeviceAddressRangeKHR {
            address: self.address,
            size: self.size,
            stride: size_of::<T>() as u64,
        }
    }
    /// The flags the address commands take with a range: whether every buffer under it has
    /// storage usage (a buffer's memory is its own, so that is this buffer's usage), and that
    /// it is fully bound (lava has no sparse memory).
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::usage::{Dynamic, Index, Storage};

    /// GPU address the fake buffers pretend to live at.
    const BASE: u64 = 0x1000;

    /// A slice of `len` u32s, as of a buffer without a device.
    fn slice(len: u64) -> BufferSlice<'static, u32, Storage> {
        new_slice(Storage, .., BASE, len * 4)
    }

    #[test]
    fn whole_slice_covers_the_buffer() {
        let slice = slice(8);
        assert_eq!(slice.len(), 8);
        assert_eq!(slice.size(), 32);
        assert_eq!(slice.address(), BASE);
        assert_eq!(slice.get_range(), (BASE..BASE + 32).into());
    }

    #[test]
    fn range_bounds_are_in_elements() {
        let slice = slice(8);

        let half_open = slice.range(2..5);
        assert_eq!((half_open.address(), half_open.len()), (BASE + 8, 3));

        let inclusive = slice.range(2..=5);
        assert_eq!((inclusive.address(), inclusive.len()), (BASE + 8, 4));

        let from = slice.range(6..);
        assert_eq!((from.address(), from.len()), (BASE + 24, 2));

        let to = slice.range(..3);
        assert_eq!((to.address(), to.len()), (BASE, 3));

        let excluded_start =
            slice.range((std::ops::Bound::Excluded(1), std::ops::Bound::Unbounded));
        assert_eq!((excluded_start.address(), excluded_start.len()), (BASE + 8, 6));

        let empty = slice.range(4..4);
        assert_eq!((empty.address(), empty.len()), (BASE + 16, 0));
    }

    #[test]
    fn nested_ranges_are_relative_to_the_parent_slice() {
        let inner = slice(16).range(4..12).range(2..4);
        assert_eq!((inner.address(), inner.len()), (BASE + 24, 2));
        assert_eq!(inner.range(..).len(), 2);
    }

    #[test]
    fn byte_range_addresses_bytes_and_casts_reinterpret() {
        let slice = slice(8);
        let bytes = slice.byte_range(4..20);
        assert_eq!((bytes.address(), bytes.len(), bytes.size()), (BASE + 4, 16, 16));

        let words = bytes.cast::<u32>();
        assert_eq!((words.address(), words.len()), (BASE + 4, 4));
        let pairs = slice.cast::<[u32; 2]>();
        assert_eq!(pairs.len(), 4);
    }

    #[test]
    fn access_covers_the_gpu_address_range() {
        let access = slice(8).range(2..4).access(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_STORAGE_READ,
        );
        assert_eq!(access.range, (BASE + 8..BASE + 16).into());
        assert_eq!(access.stage, vk::PipelineStageFlags2::COMPUTE_SHADER);
        assert_eq!(access.access, vk::AccessFlags2::SHADER_STORAGE_READ);
    }

    #[test]
    fn address_commands_get_the_range_its_stride_and_the_usage_flags() {
        let slice = slice(8).range(2..6);
        let range = slice.address_range();
        assert_eq!((range.address, range.size), (BASE + 8, 16));
        let strided = slice.cast::<[u32; 2]>().strided_range();
        assert_eq!(
            (strided.address, strided.size, strided.stride),
            (BASE + 8, 16, 8)
        );
        assert_eq!(
            slice.address_flags(),
            vk::AddressCommandFlagsKHR::STORAGE_BUFFER_USAGE
                | vk::AddressCommandFlagsKHR::FULLY_BOUND
        );

        let index: BufferSlice<u32, Index> = new_slice(Index, .., BASE, 16);
        assert_eq!(
            index.address_flags(),
            vk::AddressCommandFlagsKHR::UNKNOWN_STORAGE_BUFFER_USAGE
                | vk::AddressCommandFlagsKHR::FULLY_BOUND
        );
        let dynamic: BufferSlice<u32, Dynamic> =
            new_slice(Dynamic(vk::BufferUsageFlags::STORAGE_BUFFER), .., BASE, 16);
        assert!(
            dynamic
                .address_flags()
                .contains(vk::AddressCommandFlagsKHR::STORAGE_BUFFER_USAGE)
        );
    }

    #[test]
    #[should_panic(expected = "ends at byte")]
    fn range_with_end_before_start_panics() {
        #[allow(clippy::reversed_empty_ranges)]
        let _ = slice(8).range(5..2);
    }

    #[test]
    #[should_panic(expected = "only 32 bytes long")]
    fn range_past_the_end_panics() {
        let _ = slice(8).range(4..9);
    }

    #[test]
    #[should_panic(expected = "only 8 bytes long")]
    fn nested_range_past_the_parent_slice_panics() {
        let _ = slice(8).range(0..2).range(0..3);
    }
}

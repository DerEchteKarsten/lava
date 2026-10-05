//! Format- and usage-typed GPU images with mip levels and host uploads
use std::{marker::PhantomData, sync::atomic::AtomicU32};

use crate::error::{Error, Result};
use ash::vk::{self, ComponentSwizzle};
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc};
use lava_macros::validation_trace;
use smallvec::SmallVec;

use crate::{
    bindless::{Bindless, BindlessHandle},
    image::{
        format::{Format, Undefined, bc_block},
        slice::AsImage,
        usage::ImageUsage,
    },
    state::{Ctx, Functions},
    vkobjects::queue::{CommandBufferMemory, Fence, Gfx, Queue, SemaphoreInfo},
};
use glam::UVec2;

pub mod format;
pub mod slice;
pub mod usage;

#[derive(Debug)]
pub struct Image<F: Format, U: ImageUsage> {
    pub(crate) image: vk::Image,
    pub(crate) whole_view: vk::ImageView,
    pub(crate) allocation: Allocation,
    pub extent: UVec2,
    pub handle: BindlessHandle,
    pub mip_levels: u32,
    pub(crate) layout: AtomicU32,
    _format: PhantomData<F>,
    _usage: PhantomData<U>,
}

impl<F: Format, U: ImageUsage> Drop for Image<F, U> {
    fn drop(&mut self) {
        unsafe {
            Ctx::device().destroy_image_view(self.whole_view, None);
            Ctx::device().destroy_image(self.image, None);
        }
        Bindless::release(self.handle);
        let alloc = std::mem::take(&mut self.allocation);
        if let Err(err) = Ctx::allocator().free(alloc) {
            tracing::error!(%err, "failed to free image allocation");
        }
    }
}

impl<F: Format, U: ImageUsage> Image<F, U> {
    #[validation_trace]
    pub fn new(width: u32, height: u32) -> Result<Self> {
        Self::with_mip_levels(width, height, 1)
    }
    #[validation_trace]
    pub fn with_mip_levels(width: u32, height: u32, mip_levels: u32) -> Result<Self> {
        let extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };
        let create_info = vk::ImageCreateInfo {
            array_layers: 1,
            extent,
            format: F::format(),
            image_type: vk::ImageType::TYPE_2D,
            mip_levels,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: U::VK
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | if Ctx::features().rebar {
                    vk::ImageUsageFlags::HOST_TRANSFER_EXT
                } else {
                    vk::ImageUsageFlags::empty()
                },
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };

        let image = unsafe { Ctx::device().create_image(&create_info, None) }?;
        let requirements = unsafe { Ctx::device().get_image_memory_requirements(image) };

        let desc = AllocationCreateDesc {
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            linear: false,
            location: gpu_allocator::MemoryLocation::GpuOnly,
            name: "ImageMemory",
            requirements,
        };
        let allocation = Ctx::allocator().allocate(&desc)?;

        unsafe {
            Ctx::device().bind_image_memory(image, allocation.memory(), allocation.offset())?
        };

        let mut s = Self {
            _format: PhantomData,
            _usage: PhantomData,
            handle: BindlessHandle::none(),
            allocation,
            extent: UVec2::new(width, height),
            mip_levels,
            image,
            whole_view: vk::ImageView::null(),
            layout: AtomicU32::new(vk::ImageLayout::UNDEFINED.as_raw() as u32),
        };
        let (handle, view) = {
            let view = s.create_new_view(
                (0..mip_levels).into(),
                vk::ComponentMapping {
                    r: ComponentSwizzle::R,
                    g: ComponentSwizzle::G,
                    b: ComponentSwizzle::B,
                    a: ComponentSwizzle::A,
                },
            )?;
            let handle = Bindless::push(view)?;
            (handle, view.view)
        };
        s.handle = handle;
        s.whole_view = view;
        Ok(s)
    }

    #[validation_trace]
    pub fn copy_from(&mut self, data: &[u8], mip_level: u32) -> Result<()> {
        self.check_host_copy(data, mip_level)?;
        let extent = self.mip_extent(mip_level);
        let expected = expected_data_len(extent, F::TEXEL_SIZE, bc_block(F::FORMAT));
        if expected.is_some_and(|expected| data.len() != expected) {
            return Err(Error::message(format!(
                "mip level {mip_level} holds {} bytes but {} were given",
                expected.unwrap_or_default(),
                data.len()
            )));
        }
        self.host_copy(data, mip_level, UVec2::ZERO, extent)
    }

    /// Like `copy_from`, but only writes the `extent` texels at `offset` of the mip level.
    /// `data` holds that rectangle, tightly packed.
    #[validation_trace]
    pub fn copy_region_from(
        &mut self,
        data: &[u8],
        mip_level: u32,
        offset: UVec2,
        extent: UVec2,
    ) -> Result<()> {
        self.check_host_copy(data, mip_level)?;
        let mip_extent = self.mip_extent(mip_level);
        if !region_fits(mip_extent, offset, extent) {
            return Err(Error::message(format!(
                "region {extent} at {offset} does not fit mip level {mip_level} of size {mip_extent}"
            )));
        }
        let expected = expected_data_len(extent, F::TEXEL_SIZE, bc_block(F::FORMAT));
        if expected.is_some_and(|expected| data.len() != expected) {
            return Err(Error::message(format!(
                "the region holds {} bytes but {} were given",
                expected.unwrap_or_default(),
                data.len()
            )));
        }
        self.host_copy(data, mip_level, offset, extent)
    }

    fn check_host_copy(&self, data: &[u8], mip_level: u32) -> Result<()> {
        if !Ctx::features().rebar {
            return Err(Error::message(
                "host image copies need host-visible device memory, which this device lacks",
            ));
        }
        if mip_level >= self.mip_levels {
            return Err(Error::message(format!(
                "mip level {mip_level} is out of range for an image with {} levels",
                self.mip_levels
            )));
        }
        if data.is_empty() {
            return Err(Error::message("no texel data to copy"));
        }
        Ok(())
    }

    fn host_copy(
        &mut self,
        data: &[u8],
        mip_level: u32,
        offset: UVec2,
        extent: UVec2,
    ) -> Result<()> {
        let range = self.whole_view().subresource_range();
        let layout = self.layout.get_mut();
        let old_layout = vk::ImageLayout::from_raw(*layout as i32);
        if old_layout != vk::ImageLayout::GENERAL {
            *layout = vk::ImageLayout::GENERAL.as_raw() as u32;
            let transition = vk::HostImageLayoutTransitionInfoEXT::default()
                .image(self.image)
                .old_layout(old_layout)
                .new_layout(vk::ImageLayout::GENERAL)
                .subresource_range(range);
            unsafe { Functions::host_image_copy().transition_image_layout(&[transition])? };
        }
        let regions = [vk::MemoryToImageCopyEXT::default()
            .host_pointer(data.as_ptr().cast())
            .image_offset(vk::Offset3D {
                x: offset.x as i32,
                y: offset.y as i32,
                z: 0,
            })
            .image_extent(vk::Extent3D {
                width: extent.x,
                height: extent.y,
                depth: 1,
            })
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: F::ASPECTS,
                mip_level,
                base_array_layer: 0,
                layer_count: 1,
            })
            // 0: the rows of `data` are as long as the region is wide, which is also right
            // for block-compressed formats, where a small mip is narrower than its one block.
            .memory_image_height(0)
            .memory_row_length(0)];
        let info = vk::CopyMemoryToImageInfoEXT::default()
            .dst_image(self.image)
            .dst_image_layout(vk::ImageLayout::GENERAL)
            .regions(&regions);
        unsafe { Functions::host_image_copy().copy_memory_to_image(&info)? };
        Ok(())
    }

    pub fn mip_extent(&self, level: u32) -> UVec2 {
        mip_extent(self.extent, level)
    }

    pub fn cast<NF: Format, NU: ImageUsage>(self) -> Image<NF, NU> {
        unsafe { std::mem::transmute(self) }
    }
}

/// Extent of mip `level` of an image whose level 0 is `extent`: halved per level, at least 1.
pub fn mip_extent(extent: UVec2, level: u32) -> UVec2 {
    UVec2::new(
        extent.x.checked_shr(level).unwrap_or(0).max(1),
        extent.y.checked_shr(level).unwrap_or(0).max(1),
    )
}

/// Bytes of tightly packed texel data for an image of `extent`: texels of `texel_size` bytes,
/// or whole blocks of `(side length, bytes)` when there is no fixed texel size
/// (`texel_size == 0`). `None` when the format has neither, so the length can't be checked.
fn expected_data_len(
    extent: UVec2,
    texel_size: usize,
    block: Option<(u32, usize)>,
) -> Option<usize> {
    if texel_size != 0 {
        return Some(extent.x as usize * extent.y as usize * texel_size);
    }
    let (side, bytes) = block?;
    Some(extent.x.div_ceil(side) as usize * extent.y.div_ceil(side) as usize * bytes)
}

/// Whether the `extent` rectangle at `offset` is non-empty and lies inside `mip_extent`.
fn region_fits(mip_extent: UVec2, offset: UVec2, extent: UVec2) -> bool {
    extent.cmpgt(UVec2::ZERO).all()
        && offset.cmple(mip_extent).all()
        && extent.cmple(mip_extent - offset.min(mip_extent)).all()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regions_must_lie_inside_the_mip() {
        let mip = UVec2::new(8, 4);
        assert!(region_fits(mip, UVec2::ZERO, mip));
        assert!(region_fits(mip, UVec2::new(6, 3), UVec2::new(2, 1)));
        assert!(!region_fits(mip, UVec2::new(6, 3), UVec2::new(3, 1)));
        assert!(!region_fits(mip, UVec2::new(9, 0), UVec2::ONE));
        assert!(!region_fits(mip, UVec2::ZERO, UVec2::new(0, 4)));
        // Offsets near `u32::MAX` must not overflow.
        assert!(!region_fits(
            mip,
            UVec2::splat(u32::MAX),
            UVec2::splat(u32::MAX)
        ));
    }

    #[test]
    fn expected_data_len_is_texels_times_texel_size() {
        assert_eq!(expected_data_len(UVec2::new(8, 4), 4, None), Some(128));
        assert_eq!(expected_data_len(UVec2::new(3, 5), 1, None), Some(15));
        assert_eq!(expected_data_len(UVec2::new(1, 1), 16, None), Some(16));
        // Block-compressed and runtime-defined formats can't be checked.
        assert_eq!(expected_data_len(UVec2::new(8, 4), 0, None), None);
    }

    #[test]
    fn block_formats_hold_whole_blocks() {
        let bc7 = bc_block(vk::Format::BC7_SRGB_BLOCK);
        assert_eq!(bc7, Some((4, 16)));
        assert_eq!(bc_block(vk::Format::BC4_UNORM_BLOCK), Some((4, 8)));
        assert_eq!(bc_block(vk::Format::R8G8B8A8_UNORM), None);
        assert_eq!(expected_data_len(UVec2::new(8, 4), 0, bc7), Some(32));
        // Mips smaller than a block, or not a multiple of it, still take whole blocks.
        assert_eq!(expected_data_len(UVec2::new(1, 1), 0, bc7), Some(16));
        assert_eq!(expected_data_len(UVec2::new(6, 2), 0, bc7), Some(32));
    }

    #[test]
    fn mip_extent_halves_per_level() {
        let extent = UVec2::new(256, 64);
        assert_eq!(mip_extent(extent, 0), UVec2::new(256, 64));
        assert_eq!(mip_extent(extent, 1), UVec2::new(128, 32));
        assert_eq!(mip_extent(extent, 6), UVec2::new(4, 1));
    }

    #[test]
    fn mip_extent_rounds_down_and_never_reaches_zero() {
        assert_eq!(mip_extent(UVec2::new(5, 3), 1), UVec2::new(2, 1));
        assert_eq!(mip_extent(UVec2::new(5, 3), 2), UVec2::new(1, 1));
        assert_eq!(mip_extent(UVec2::new(5, 3), 31), UVec2::new(1, 1));
        // Shifting by the full bit width must not overflow.
        assert_eq!(mip_extent(UVec2::new(5, 3), 32), UVec2::new(1, 1));
        assert_eq!(mip_extent(UVec2::new(5, 3), 200), UVec2::new(1, 1));
    }
}

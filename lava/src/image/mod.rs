//! Format- and usage-typed GPU images with mip levels and host uploads, their type erasure and drop hook
use std::{
    mem::{self, ManuallyDrop},
    sync::{OnceLock, atomic::AtomicU32},
};

use crate::error::{Error, Result};
use ash::vk::{self, ComponentSwizzle};
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc};
use lava_macros::validation_trace;

use crate::{
    bindless::{self, BindlessHandle},
    image::{
        format::{Format, StaticFormat, bc_block},
        usage::{ImageUsage, IsHostCopy, IsSampled, IsStorage, StaticUsage},
    },
    state::{Ctx, Functions},
};
use glam::UVec2;

pub mod format;
pub mod slice;
pub mod usage;

/// A 2D image. Its format and usage are types (`format::R8Unorm`, `usage::Sampled`), which
/// the command buffer's bounds check at compile time, or `format::Dynamic`/`usage::Dynamic`
/// values, which it checks at runtime (`aspect_check`, `texel_check`, `usage_check`).
#[derive(Debug)]
pub struct Image<F: Format, U: ImageUsage> {
    pub(crate) image: vk::Image,
    pub(crate) whole_view: vk::ImageView,
    pub(crate) allocation: Allocation,
    pub extent: UVec2,
    pub handle: BindlessHandle,
    pub mip_levels: u32,
    /// The layout the image will be in once everything built so far is recorded. It changes
    /// when an access is built (see `ImageView::access`), not when it is recorded.
    pub(crate) layout: AtomicU32,
    pub(crate) format: F,
    pub(crate) usage: U,
}

/// An image whose format and usage are runtime values (`Image::erase`).
pub type DynamicImage = Image<format::Dynamic, usage::Dynamic>;

static ON_DROP: OnceLock<fn(&DynamicImage)> = OnceLock::new();

/// Sets the function every dropped image calls with itself, erased, right before it is
/// destroyed, so whoever picked its bindless indices can hand them out again. It can be set
/// once.
pub fn on_drop(hook: fn(&DynamicImage)) -> Result<()> {
    ON_DROP
        .set(hook)
        .map_err(|_| Error::message("the image drop hook is already set"))
}

impl<F: Format, U: ImageUsage> Drop for Image<F, U> {
    fn drop(&mut self) {
        if let Some(hook) = ON_DROP.get() {
            let format = format::Dynamic(self.format.vk_format());
            let usage = usage::Dynamic(self.usage.flags());
            // Lent to the hook and never dropped: the memory comes back to be freed below.
            let mut image = ManuallyDrop::new(self.take(format, usage));
            hook(&image);
            self.allocation = mem::take(&mut image.allocation);
        }
        unsafe {
            Ctx::device().destroy_image_view(self.whole_view, None);
            Ctx::device().destroy_image(self.image, None);
        }
        let alloc = mem::take(&mut self.allocation);
        if let Err(err) = Ctx::allocator().free(alloc) {
            tracing::error!(%err, "failed to free image allocation");
        }
    }
}

impl<F: StaticFormat, U: StaticUsage> Image<F, U> {
    /// Creates an image that is not bound in any bindless set: its `handle` is null.
    #[validation_trace]
    pub fn new(width: u32, height: u32) -> Result<Self> {
        Self::with_mip_levels(width, height, 1)
    }
    #[validation_trace]
    pub fn with_mip_levels(width: u32, height: u32, mip_levels: u32) -> Result<Self> {
        Self::with_params(F::default(), U::default(), width, height, mip_levels)
    }
}

impl<F: Format, U: ImageUsage> Image<F, U> {
    /// Like `with_mip_levels`, for any format and usage, `Dynamic` ones included.
    #[validation_trace]
    pub fn with_params(
        format: F,
        usage: U,
        width: u32,
        height: u32,
        mip_levels: u32,
    ) -> Result<Self> {
        let extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };
        let create_info = vk::ImageCreateInfo {
            array_layers: 1,
            extent,
            format: format.vk_format(),
            image_type: vk::ImageType::TYPE_2D,
            mip_levels,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            // Without rebar the host image copy extension is off; `copy_from` then fails.
            usage: if Ctx::features().rebar {
                usage.flags()
            } else {
                usage.flags() & !vk::ImageUsageFlags::HOST_TRANSFER_EXT
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
            format,
            usage,
            handle: BindlessHandle::none(),
            allocation,
            extent: UVec2::new(width, height),
            mip_levels,
            image,
            whole_view: vk::ImageView::null(),
            layout: AtomicU32::new(vk::ImageLayout::UNDEFINED.as_raw() as u32),
        };
        s.whole_view = s
            .create_new_view(
                (0..mip_levels).into(),
                vk::ComponentMapping {
                    r: ComponentSwizzle::R,
                    g: ComponentSwizzle::G,
                    b: ComponentSwizzle::B,
                    a: ComponentSwizzle::A,
                },
            )?
            .view;
        Ok(s)
    }

    pub fn mip_extent(&self, level: u32) -> UVec2 {
        mip_extent(self.extent, level)
    }

    pub fn format(&self) -> F {
        self.format
    }

    pub fn usage(&self) -> U {
        self.usage
    }

    /// Panics unless the format has the `required` aspects; does nothing for a static format.
    pub fn aspect_check(&self, required: vk::ImageAspectFlags) {
        self.format.aspect_check(required);
    }

    /// Panics unless the format's texels have the type `T`; does nothing for a static format.
    pub fn texel_check<T: 'static>(&self) {
        self.format.texel_check::<T>();
    }

    /// Panics unless the usage has every flag of `required`; does nothing for a static usage.
    pub fn usage_check(&self, required: vk::ImageUsageFlags) {
        self.usage.check(required);
    }

    /// The same image with another format and usage, unchecked: the image still has the
    /// format and usage it was created with.
    pub fn cast<NF: Format, NU: ImageUsage>(self, format: NF, usage: NU) -> Image<NF, NU> {
        ManuallyDrop::new(self).take(format, usage)
    }

    /// The same image with its format and usage as runtime values, e.g. to keep images of
    /// different types together. Commands check them at runtime.
    pub fn erase(self) -> DynamicImage {
        let format = format::Dynamic(self.format.vk_format());
        let usage = usage::Dynamic(self.usage.flags());
        self.cast(format, usage)
    }

    /// Moves the image into one of another format and usage, leaving `self` without memory;
    /// `self` must not destroy its handles unless it gets the memory back.
    fn take<NF: Format, NU: ImageUsage>(&mut self, format: NF, usage: NU) -> Image<NF, NU> {
        Image {
            image: self.image,
            whole_view: self.whole_view,
            allocation: mem::take(&mut self.allocation),
            extent: self.extent,
            handle: self.handle,
            mip_levels: self.mip_levels,
            layout: AtomicU32::new(*self.layout.get_mut()),
            format,
            usage,
        }
    }
}

impl<F: Format, U: IsHostCopy> Image<F, U> {
    /// Writes a whole mip level from the host, right away. Like anything a command buffer
    /// doesn't record, it isn't synchronised with the GPU: frames in flight that read the image
    /// may see the old or the new texels, or a mix.
    #[validation_trace]
    pub fn copy_from(&mut self, data: &[u8], mip_level: u32) -> Result<()> {
        self.check_host_copy(data, mip_level)?;
        let extent = self.mip_extent(mip_level);
        let expected = expected_data_len(
            extent,
            self.format.texel_size(),
            bc_block(self.format.vk_format()),
        );
        if expected.is_some_and(|expected| data.len() != expected) {
            return Err(Error::message(format!(
                "mip level {mip_level} holds {} bytes but {} were given",
                expected.unwrap_or_default(),
                data.len()
            )));
        }
        self.host_copy(data, mip_level, UVec2::ZERO, extent)
    }

    /// Like `copy_from` (also unsynchronised), but only writes the `extent` texels at `offset`
    /// of the mip level. `data` holds that rectangle, tightly packed.
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
        let expected = expected_data_len(
            extent,
            self.format.texel_size(),
            bc_block(self.format.vk_format()),
        );
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
        self.usage_check(vk::ImageUsageFlags::HOST_TRANSFER_EXT);
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
                aspect_mask: self.format.aspects(),
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
}

impl<F: Format, U: IsSampled> Image<F, U> {
    /// Binds the image at `index` of the sampled set, right away:
    /// `Image::new(w, h)?.bind_sampled(index)`.
    #[validation_trace]
    pub fn bind_sampled(mut self, index: u32) -> Self {
        self.handle.descriptor_index_set0 = index;
        bindless::bind([self.whole_view()], [BindlessHandle::sampled(index)]);
        self
    }
}

impl<F: Format, U: IsStorage> Image<F, U> {
    /// Binds the image at `index` of the storage set, right away. An image with both usages
    /// can be bound in both sets: `.bind_sampled(a).bind_storage(b)`.
    #[validation_trace]
    pub fn bind_storage(mut self, index: u32) -> Self {
        self.handle.descriptor_index_set1 = index;
        bindless::bind([self.whole_view()], [BindlessHandle::storage(index)]);
        self
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

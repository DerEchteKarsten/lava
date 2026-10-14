//! Borrowed image views of a mip range and their slices, the only image types commands take, with the layouts they track for barriers, and the image transfer commands (clear, blit, copies to and from buffers)
use std::{iter, range::Range, sync::atomic::AtomicU32};

use ash::vk;
use bytemuck::Pod;
use lava_macros::validation_trace;

use crate::{
    bindless::BindlessHandle,
    buffer::{slice::BufferSlice, usage::BufferUsage},
    command_buffer::{BufferAccess, CommandBuffer, ImageAccess, ImageUse},
    image::{
        Image, Views,
        dim::{D2, Dim},
        format::{ColorAspect, Format},
        usage::{ImageUsage, IsTransferDst, IsTransferSrc},
    },
    state::{Ctx, Functions},
};

/// A borrowed view of `mip_range` of an image. Commands, bindless binds and generated bindings
/// take these, whichever kind of image they come from (a swapchain's included).
#[derive(Clone, Copy, Debug)]
pub struct ImageView<'a, F: Format, U: ImageUsage, D: Dim = D2> {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub mip_range: Range<u32>,
    /// The extent of mip `mip_range.start`.
    pub extent: D::Extent,
    pub handle: BindlessHandle,
    /// The tracked layouts: one for all of `mip_range` (an image with one view, or a swapchain
    /// image), or one per mip.
    pub(crate) layouts: &'a [AtomicU32],
    pub(crate) format: F,
    pub(crate) usage: U,
    pub(crate) dim: D,
}

impl<'a, F: Format, U: ImageUsage, D: Dim> ImageView<'a, F, U, D> {
    pub fn subresource_range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: self.format.aspects(),
            base_mip_level: self.mip_range.start,
            level_count: self.mip_range.end - self.mip_range.start,
            base_array_layer: 0,
            layer_count: 1,
        }
    }
    pub fn subresource_layers(&self, mip_level: u32) -> vk::ImageSubresourceLayers {
        vk::ImageSubresourceLayers {
            aspect_mask: self.format.aspects(),
            mip_level,
            base_array_layer: 0,
            layer_count: 1,
        }
    }
    /// All of mip `mip_range.start`, which copies and blits of the slice touch.
    pub fn whole(self) -> ImageSlice<'a, F, U, D> {
        self.region(self.extent)
    }
    /// The `extend` texels at the origin of mip `mip_range.start`.
    pub fn region(self, extend: D::Extent) -> ImageSlice<'a, F, U, D> {
        ImageSlice {
            view: self,
            offset: D::Offset::default(),
            extend,
        }
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

    /// The same view typed as `format` and `usage`, which must be the view's format and no
    /// more than its usage (see `Image::cast`).
    pub fn cast<NF: Format, NU: ImageUsage>(
        self,
        format: NF,
        usage: NU,
    ) -> ImageView<'a, NF, NU, D> {
        cast_check(
            (self.format.vk_format(), self.usage.flags()),
            (format.vk_format(), usage.flags()),
        );
        ImageView {
            image: self.image,
            view: self.view,
            mip_range: self.mip_range,
            extent: self.extent,
            handle: self.handle,
            layouts: self.layouts,
            format,
            usage,
            dim: self.dim,
        }
    }

    /// The access of a command that wants the view's mips in `layout`. It borrows their
    /// tracked layouts, which move to `layout` only when the command is recorded.
    pub(crate) fn access(
        &self,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
        layout: vk::ImageLayout,
    ) -> ImageUse<'a> {
        ImageUse {
            access: ImageAccess {
                stage,
                access,
                image: self.image,
                layout,
                aspect: self.format.aspects(),
                mip_range: self.mip_range,
            },
            layouts: self.layouts,
        }
    }
}

/// A box of texels of one mip: `extend` texels at `offset` of the view's first mip.
#[derive(Clone, Copy)]
pub struct ImageSlice<'a, F: Format, U: ImageUsage, D: Dim = D2> {
    pub view: ImageView<'a, F, U, D>,
    pub offset: D::Offset,
    pub extend: D::Extent,
}

impl<'a, F: Format, U: ImageUsage, D: Dim> ImageSlice<'a, F, U, D> {
    pub fn offset(mut self, offset: D::Offset) -> Self {
        self.offset = self.offset + offset;
        self
    }

    pub fn grow(mut self, extend: D::Extent) -> Self {
        self.extend = self.extend + extend;
        self
    }

    /// The same slice typed as `format` and `usage` (see `ImageView::cast`).
    pub fn cast<NF: Format, NU: ImageUsage>(
        self,
        format: NF,
        usage: NU,
    ) -> ImageSlice<'a, NF, NU, D> {
        ImageSlice {
            view: self.view.cast(format, usage),
            offset: self.offset,
            extend: self.extend,
        }
    }
}

/// Panics unless `to` is the same format as `from` and no more than its usage: a cast can only
/// narrow what the type says about an image, never claim a usage it was created without.
pub(crate) fn cast_check(
    from: (vk::Format, vk::ImageUsageFlags),
    to: (vk::Format, vk::ImageUsageFlags),
) {
    assert!(
        from.0 == to.0 && from.1.contains(to.1),
        "can't cast an image of format {:?} and usage {:?} to format {:?} and usage {:?}",
        from.0,
        from.1,
        to.0,
        to.1
    );
}

impl<F: Format, U: ImageUsage, D: Dim, V: Views> Image<F, U, D, V> {
    /// All of mip 0: `whole_view().whole()`.
    pub fn whole(&self) -> ImageSlice<'_, F, U, D> {
        self.whole_view().whole()
    }
}

// ---- transfer commands -------------------------------------------------------------------

/// How a blit samples its source.
#[repr(i32)]
pub enum Filter {
    Nearest = 0,
    Liniear = 1,
}

impl CommandBuffer {
    #[validation_trace]
    pub fn clear_image<'a, F: Format, U: IsTransferDst, D: Dim>(
        &'a mut self,
        image: ImageView<'a, F, U, D>,
        clear_color: F::Texels,
    ) where
        F: ColorAspect,
    {
        image.aspect_check(vk::ImageAspectFlags::COLOR);
        image.usage_check(vk::ImageUsageFlags::TRANSFER_DST);
        self.flush_pending(
            std::iter::once(&image.access(
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::ImageLayout::GENERAL,
            )),
            std::iter::empty(),
        );
        unsafe {
            Ctx::device().cmd_clear_color_image(
                self.handle,
                image.image,
                vk::ImageLayout::GENERAL,
                &vk::ClearValue::from(image.format().clear_value(clear_color)).color,
                &[image.subresource_range()],
            )
        };
    }
    #[validation_trace]
    pub fn blit_image<'a, F: Format, U: IsTransferSrc, F2: Format, U2: IsTransferDst, D: Dim>(
        &mut self,
        src: ImageSlice<'a, F, U, D>,
        dst: ImageSlice<'a, F2, U2, D>,
        filter: Filter,
    ) where
        F: ColorAspect,
        F2: ColorAspect,
    {
        src.view.aspect_check(vk::ImageAspectFlags::COLOR);
        dst.view.aspect_check(vk::ImageAspectFlags::COLOR);
        src.view.usage_check(vk::ImageUsageFlags::TRANSFER_SRC);
        dst.view.usage_check(vk::ImageUsageFlags::TRANSFER_DST);
        self.flush_pending(
            [
                src.view.access(
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::TRANSFER_READ,
                    vk::ImageLayout::GENERAL,
                ),
                dst.view.access(
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::TRANSFER_WRITE,
                    vk::ImageLayout::GENERAL,
                ),
            ]
            .iter(),
            iter::empty(),
        );

        let regions = [blit_region(&src, &dst)];
        unsafe {
            Ctx::device().cmd_blit_image(
                self.handle,
                src.view.image,
                vk::ImageLayout::GENERAL,
                dst.view.image,
                vk::ImageLayout::GENERAL,
                &regions,
                vk::Filter::from_raw(filter as i32),
            );
        }
    }
    #[validation_trace]
    pub fn copy_buffer_to_image<
        'a,
        T: Copy + Pod,
        BU: BufferUsage,
        F: Format,
        U: IsTransferDst,
        D: Dim,
    >(
        &mut self,
        src: BufferSlice<'a, T, BU>,
        dst: ImageSlice<'a, F, U, D>,
    ) where
        F: ColorAspect,
    {
        dst.view.aspect_check(vk::ImageAspectFlags::COLOR);
        dst.view.usage_check(vk::ImageUsageFlags::TRANSFER_DST);
        self.flush_pending(
            iter::once(&dst.view.access(
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::ImageLayout::GENERAL,
            )),
            iter::once(&BufferAccess {
                access: vk::AccessFlags2::TRANSFER_READ,
                stage: vk::PipelineStageFlags2::TRANSFER,
                range: src.get_range(),
            }),
        );
        let regions = [vk::DeviceMemoryImageCopyKHR::default()
            .address_range(src.address_range())
            .address_flags(src.address_flags())
            .image_subresource(dst.view.subresource_layers(dst.view.mip_range.start))
            .image_layout(vk::ImageLayout::GENERAL)
            .image_offset(D::offset_3d(dst.offset))
            .image_extent(D::extent_3d(dst.extend))];
        unsafe {
            Functions::device_address_commands().cmd_copy_memory_to_image(
                self.handle,
                &vk::CopyDeviceMemoryImageInfoKHR::default()
                    .image(dst.view.image)
                    .regions(&regions),
            )
        };
    }

    /// Copies the texels of `src` (its view's first mip level) into `dst`, tightly packed row
    /// by row (and slice by slice).
    #[validation_trace]
    pub fn copy_image_to_buffer<
        'a,
        T: Copy + Pod,
        F: Format,
        U: IsTransferSrc,
        BU: BufferUsage,
        D: Dim,
    >(
        &mut self,
        src: ImageSlice<'a, F, U, D>,
        dst: BufferSlice<'a, T, BU>,
    ) where
        F: ColorAspect,
    {
        src.view.aspect_check(vk::ImageAspectFlags::COLOR);
        src.view.usage_check(vk::ImageUsageFlags::TRANSFER_SRC);
        self.flush_pending(
            iter::once(&src.view.access(
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_READ,
                vk::ImageLayout::GENERAL,
            )),
            iter::once(&BufferAccess {
                access: vk::AccessFlags2::TRANSFER_WRITE,
                stage: vk::PipelineStageFlags2::TRANSFER,
                range: dst.get_range(),
            }),
        );
        let regions = [vk::DeviceMemoryImageCopyKHR::default()
            .address_range(dst.address_range())
            .address_flags(dst.address_flags())
            .image_subresource(src.view.subresource_layers(src.view.mip_range.start))
            .image_layout(vk::ImageLayout::GENERAL)
            .image_offset(D::offset_3d(src.offset))
            .image_extent(D::extent_3d(src.extend))];
        unsafe {
            Functions::device_address_commands().cmd_copy_image_to_memory(
                self.handle,
                &vk::CopyDeviceMemoryImageInfoKHR::default()
                    .image(src.view.image)
                    .regions(&regions),
            )
        };
    }
}

/// Blit region covering the two slices: from each slice's offset to offset + extent.
fn blit_region<F: Format, U: ImageUsage, F2: Format, U2: ImageUsage, D: Dim>(
    src: &ImageSlice<'_, F, U, D>,
    dst: &ImageSlice<'_, F2, U2, D>,
) -> vk::ImageBlit {
    let corners = |offset: D::Offset, extend: D::Extent| {
        let start = D::offset_3d(offset);
        let size = D::extent_3d(extend);
        [
            start,
            vk::Offset3D {
                x: start.x + size.width as i32,
                y: start.y + size.height as i32,
                z: start.z + size.depth as i32,
            },
        ]
    };
    vk::ImageBlit {
        src_offsets: corners(src.offset, src.extend),
        dst_offsets: corners(dst.offset, dst.extend),
        src_subresource: src.view.subresource_layers(src.view.mip_range.start),
        dst_subresource: dst.view.subresource_layers(dst.view.mip_range.start),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{
        format::{D32SfloatS8Uint, R8G8B8A8Unorm, StaticFormat},
        usage::{DepthAttachment, Sampled, SampledStorage, StaticUsage, Storage},
    };
    use ash::vk::Handle;
    use glam::{IVec2, UVec2};
    use std::sync::atomic::Ordering;

    /// A view over a fake image: layout tracking and region math need no device.
    fn view<F: StaticFormat, U: StaticUsage>(
        layout: &AtomicU32,
        mips: std::ops::Range<u32>,
    ) -> ImageView<'_, F, U> {
        ImageView {
            image: vk::Image::from_raw(42),
            view: vk::ImageView::null(),
            mip_range: mips.into(),
            extent: UVec2::new(64, 32),
            handle: BindlessHandle::none(),
            layouts: std::slice::from_ref(layout),
            format: F::default(),
            usage: U::default(),
            dim: D2,
        }
    }

    fn undefined() -> AtomicU32 {
        AtomicU32::new(vk::ImageLayout::UNDEFINED.as_raw() as u32)
    }

    #[test]
    fn subresource_range_spans_the_views_mips() {
        let layout = undefined();
        let range = view::<R8G8B8A8Unorm, Sampled>(&layout, 2..5).subresource_range();
        assert_eq!(range.aspect_mask, vk::ImageAspectFlags::COLOR);
        assert_eq!((range.base_mip_level, range.level_count), (2, 3));
        assert_eq!((range.base_array_layer, range.layer_count), (0, 1));
    }

    #[test]
    fn subresource_layers_use_the_formats_aspects() {
        let layout = undefined();
        let layers = view::<D32SfloatS8Uint, DepthAttachment>(&layout, 0..1).subresource_layers(3);
        assert_eq!(
            layers.aspect_mask,
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        );
        assert_eq!(layers.mip_level, 3);
        assert_eq!((layers.base_array_layer, layers.layer_count), (0, 1));
    }

    #[test]
    fn region_starts_at_the_origin_and_can_be_moved_and_grown() {
        let layout = undefined();
        let slice = view::<R8G8B8A8Unorm, Sampled>(&layout, 0..1).region(UVec2::new(16, 8));
        assert_eq!(
            (slice.offset, slice.extend),
            (IVec2::ZERO, UVec2::new(16, 8))
        );

        let moved = slice.offset(IVec2::new(4, 2)).offset(IVec2::new(1, 1));
        assert_eq!(moved.offset, IVec2::new(5, 3));
        assert_eq!(moved.extend, UVec2::new(16, 8));

        let grown = moved.grow(UVec2::new(2, 3));
        assert_eq!(grown.extend, UVec2::new(18, 11));
        assert_eq!(grown.offset, IVec2::new(5, 3));
    }

    #[test]
    fn whole_covers_the_views_first_mip() {
        let layout = undefined();
        let slice = view::<R8G8B8A8Unorm, Sampled>(&layout, 1..2).whole();
        assert_eq!(
            (slice.offset, slice.extend),
            (IVec2::ZERO, UVec2::new(64, 32))
        );
    }

    #[test]
    fn taking_an_access_layout_reports_the_previous_one_and_stores_the_new_one() {
        let layout = undefined();
        let view = view::<R8G8B8A8Unorm, Sampled>(&layout, 0..1);
        let taken = |access: &ImageUse| {
            let mut old = None;
            access.take_layouts(|_, layout| old = Some(layout));
            old
        };

        let first = view.access(
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::ImageLayout::GENERAL,
        );
        assert_eq!(first.access.layout, vk::ImageLayout::GENERAL);
        assert_eq!(first.access.image, vk::Image::from_raw(42));
        assert_eq!(first.access.aspect, vk::ImageAspectFlags::COLOR);
        assert_eq!(first.access.stage, vk::PipelineStageFlags2::TRANSFER);
        assert_eq!(first.access.access, vk::AccessFlags2::TRANSFER_WRITE);
        let present = view.access(
            vk::PipelineStageFlags2::empty(),
            vk::AccessFlags2::empty(),
            vk::ImageLayout::PRESENT_SRC_KHR,
        );
        // Building accesses leaves the layout alone.
        assert_eq!(
            layout.load(Ordering::Relaxed),
            vk::ImageLayout::UNDEFINED.as_raw() as u32
        );

        assert_eq!(taken(&first), Some(vk::ImageLayout::UNDEFINED));
        // Same layout again: no transition.
        assert_eq!(taken(&first), None);
        assert_eq!(taken(&present), Some(vk::ImageLayout::GENERAL));
        assert_eq!(
            layout.load(Ordering::Relaxed),
            vk::ImageLayout::PRESENT_SRC_KHR.as_raw() as u32
        );
    }

    /// Views are `Copy`; all copies (and casts) of a view must track the same layouts.
    #[test]
    fn copies_and_casts_share_the_layout_state() {
        let layout = undefined();
        let a = view::<R8G8B8A8Unorm, SampledStorage>(&layout, 0..1);
        let b = a;
        let c = a.cast(R8G8B8A8Unorm, Storage);
        let taken = |access: ImageUse| {
            let mut old = None;
            access.take_layouts(|_, layout| old = Some(layout));
            old
        };

        let general = vk::ImageLayout::GENERAL;
        let (stage, read) = (
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_READ,
        );
        taken(a.access(stage, vk::AccessFlags2::TRANSFER_WRITE, general));
        let from_copy = b.access(stage, read, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        assert_eq!(taken(from_copy), Some(general));
        let from_cast = c.access(stage, read, general);
        assert_eq!(
            taken(from_cast),
            Some(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        );
    }

    /// A cast may drop usages but never add one or change the format.
    #[test]
    fn casts_only_narrow() {
        let layout = undefined();
        let a = view::<R8G8B8A8Unorm, SampledStorage>(&layout, 0..1);
        a.cast(R8G8B8A8Unorm, Sampled);
        a.cast(R8G8B8A8Unorm, SampledStorage);
        assert!(std::panic::catch_unwind(|| a.cast(R8G8B8A8Unorm, DepthAttachment)).is_err());
        assert!(std::panic::catch_unwind(|| a.cast(D32SfloatS8Uint, Sampled)).is_err());
    }

    #[test]
    fn blit_region_spans_offset_to_offset_plus_extent() {
        let (a, b) = (undefined(), undefined());
        let src = view::<R8G8B8A8Unorm, Sampled>(&a, 0..1)
            .region(UVec2::new(16, 8))
            .offset(IVec2::new(4, 2));
        let dst = view::<R8G8B8A8Unorm, Sampled>(&b, 0..1)
            .region(UVec2::new(32, 16))
            .offset(IVec2::new(10, 20));
        let region = blit_region(&src, &dst);

        let corner = |o: vk::Offset3D| (o.x, o.y, o.z);
        assert_eq!(corner(region.src_offsets[0]), (4, 2, 0));
        assert_eq!(corner(region.src_offsets[1]), (20, 10, 1));
        assert_eq!(corner(region.dst_offsets[0]), (10, 20, 0));
        assert_eq!(corner(region.dst_offsets[1]), (42, 36, 1));
        assert_eq!(region.src_subresource.mip_level, 0);
        assert_eq!(
            region.src_subresource.aspect_mask,
            vk::ImageAspectFlags::COLOR
        );
    }
}

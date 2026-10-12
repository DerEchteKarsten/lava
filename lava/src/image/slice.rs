//! Borrowed image views and 2D slices, with per-image layout tracking for barriers
use std::{range::Range, sync::atomic::AtomicU32};

use ash::vk::{self, Offset3D};
use bytemuck::Zeroable;
use glam::{IVec2, UVec2};
use lava_macros::validation_trace;

use crate::{
    bindless::BindlessHandle,
    command_buffer::{ImageAccess, ImageUse},
    error::Result,
    image::{Image, format::Format, usage::ImageUsage},
    state::Ctx,
};

#[derive(Clone, Copy, Debug)]
pub struct ImageView<'a, F: Format, U: ImageUsage> {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub mip_range: Range<u32>,
    pub handle: BindlessHandle,
    pub(crate) layout: &'a AtomicU32,
    pub(crate) format: F,
    pub(crate) usage: U,
}

impl<'a, F: Format, U: ImageUsage> ImageView<'a, F, U> {
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
    pub fn region(self, extend: UVec2) -> ImageSlice<'a, F, U> {
        ImageSlice {
            view: self,
            offset: IVec2::ZERO,
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

    /// The same view with another format and usage, unchecked (see `Image::cast`).
    pub fn cast<NF: Format, NU: ImageUsage>(self, format: NF, usage: NU) -> ImageView<'a, NF, NU> {
        ImageView {
            image: self.image,
            view: self.view,
            mip_range: self.mip_range,
            handle: self.handle,
            layout: self.layout,
            format,
            usage,
        }
    }

    /// The access of a command that wants the image in `layout`. It borrows the image's
    /// tracked layout, which moves to `layout` only when the command is recorded.
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
            },
            tracked_layout: self.layout,
        }
    }
}

#[derive(Clone, Copy)]
pub struct ImageSlice<'a, F: Format, U: ImageUsage> {
    pub view: ImageView<'a, F, U>,
    pub offset: IVec2,
    pub extend: UVec2,
}

impl<'a, F: Format, U: ImageUsage> ImageSlice<'a, F, U> {
    pub fn offset(mut self, offset: IVec2) -> ImageSlice<'a, F, U> {
        self.offset += offset;
        self
    }

    pub fn grow(mut self, extend: UVec2) -> ImageSlice<'a, F, U> {
        self.extend += extend;
        self
    }

    /// The same slice with another format and usage, unchecked (see `Image::cast`).
    pub fn cast<NF: Format, NU: ImageUsage>(self, format: NF, usage: NU) -> ImageSlice<'a, NF, NU> {
        ImageSlice {
            view: self.view.cast(format, usage),
            offset: self.offset,
            extend: self.extend,
        }
    }
}

impl<F: Format, U: ImageUsage> Image<F, U> {
    pub fn whole_view(&self) -> ImageView<'_, F, U> {
        ImageView {
            image: self.image,
            view: self.whole_view,
            mip_range: (0..self.mip_levels).into(),
            handle: self.handle,
            layout: &self.layout,
            format: self.format,
            usage: self.usage,
        }
    }

    #[validation_trace]
    pub fn create_new_view(
        &self,
        mip_range: Range<u32>,
        swizzel: vk::ComponentMapping,
    ) -> Result<ImageView<'_, F, U>> {
        let create_info = vk::ImageViewCreateInfo::default()
            .components(swizzel)
            .format(self.format.vk_format())
            .image(self.image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: self.format.aspects(),
                base_array_layer: 0,
                layer_count: 1,
                base_mip_level: mip_range.start,
                level_count: mip_range.end - mip_range.start,
            });
        let view = unsafe { Ctx::device().create_image_view(&create_info, None)? };
        Ok(ImageView {
            view,
            mip_range,
            ..self.whole_view()
        })
    }

    pub fn whole(&self) -> ImageSlice<'_, F, U> {
        ImageSlice {
            view: self.whole_view(),
            extend: self.extent,
            offset: IVec2::ZERO,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{
        format::{D32SfloatS8Uint, R8G8B8A8Unorm, StaticFormat},
        usage::{DepthAttachment, Sampled, StaticUsage, Storage},
    };
    use ash::vk::Handle;
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
            handle: BindlessHandle::none(),
            layout,
            format: F::default(),
            usage: U::default(),
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
    fn taking_an_access_layout_reports_the_previous_one_and_stores_the_new_one() {
        let layout = undefined();
        let view = view::<R8G8B8A8Unorm, Sampled>(&layout, 0..1);

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

        assert_eq!(first.take_layout(), vk::ImageLayout::UNDEFINED);
        // Same layout again: no transition.
        assert_eq!(first.take_layout(), vk::ImageLayout::GENERAL);
        assert_eq!(present.take_layout(), vk::ImageLayout::GENERAL);
        assert_eq!(
            layout.load(Ordering::Relaxed),
            vk::ImageLayout::PRESENT_SRC_KHR.as_raw() as u32
        );
    }

    /// Views are `Copy`; all copies (and casts) of a view must track one layout per image.
    #[test]
    fn copies_and_casts_share_the_layout_state() {
        let layout = undefined();
        let a = view::<R8G8B8A8Unorm, Sampled>(&layout, 0..1);
        let b = a;
        let c = a.cast(R8G8B8A8Unorm, Storage);

        a.access(
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::ImageLayout::GENERAL,
        )
        .take_layout();
        let from_copy = b.access(
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_READ,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );
        assert_eq!(from_copy.take_layout(), vk::ImageLayout::GENERAL);
        let from_cast = c.access(
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_READ,
            vk::ImageLayout::GENERAL,
        );
        assert_eq!(
            from_cast.take_layout(),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL
        );
    }
}

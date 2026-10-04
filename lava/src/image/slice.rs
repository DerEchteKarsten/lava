//! Borrowed image views and 2D slices, with per-image layout tracking for barriers
use std::{
    marker::PhantomData,
    range::Range,
    sync::atomic::{AtomicU32, Ordering},
};

use ash::vk::{self, Offset3D};
use bytemuck::Zeroable;
use glam::{IVec2, UVec2};
use lava_macros::validation_trace;

use crate::{
    bindless::BindlessHandle,
    command_buffer::ImageAccess,
    error::Result,
    image::{
        Image,
        format::{Format, Undefined},
        usage::{ImageUsage, IsSampled, IsStorage},
    },
    state::Ctx,
};

#[derive(Clone, Copy, Debug)]
pub struct ImageView<'a, F: Format, U: ImageUsage> {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub mip_range: Range<u32>,
    pub handle: BindlessHandle,
    pub(crate) layout: &'a AtomicU32,
    pub(crate) _marker: PhantomData<F>,
    pub(crate) _marker2: PhantomData<U>,
}

impl<'a, F: Format, U: ImageUsage> ImageView<'a, F, U> {
    pub fn subresource_range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: F::ASPECTS,
            base_mip_level: self.mip_range.start,
            level_count: self.mip_range.end - self.mip_range.start,
            base_array_layer: 0,
            layer_count: 1,
        }
    }
    pub fn subresource_layers(&self, mip_level: u32) -> vk::ImageSubresourceLayers {
        vk::ImageSubresourceLayers {
            aspect_mask: F::ASPECTS,
            mip_level: mip_level,
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
    pub fn cast<NF: Format, NU: ImageUsage>(self) -> ImageView<'a, NF, NU> {
        unsafe { std::mem::transmute(self) }
    }

    pub(crate) fn access(
        &self,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
        layout: vk::ImageLayout,
    ) -> ImageAccess {
        let old_layout = vk::ImageLayout::from_raw(
            self.layout.swap(layout.as_raw() as u32, Ordering::Relaxed) as i32,
        );
        ImageAccess {
            stage,
            access,
            image: self.image,
            layout,
            aspect: F::ASPECTS,
            old_layout,
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

    pub fn cast<NF: Format, NU: ImageUsage>(self) -> ImageSlice<'a, NF, NU> {
        unsafe { std::mem::transmute(self) }
    }
}

impl<F: Format, U: ImageUsage> AsImage for Image<F, U> {
    type Format = F;
    type Usage = U;

    fn mip_range(&self) -> Range<u32> {
        (0..self.mip_levels).into()
    }
    fn get_ref(&self) -> &Image<Self::Format, Self::Usage> {
        self
    }
    fn get_mut(&mut self) -> &mut Image<Self::Format, Self::Usage> {
        self
    }
}

pub trait AsImage {
    type Format: Format;
    type Usage: ImageUsage;

    fn mip_range(&self) -> Range<u32>;
    fn get_ref(&self) -> &Image<Self::Format, Self::Usage>;
    fn get_mut(&mut self) -> &mut Image<Self::Format, Self::Usage>;

    fn whole_view<'a>(&'a self) -> ImageView<'a, Self::Format, Self::Usage> {
        let mip_range = self.mip_range();
        let image = self.get_ref();
        ImageView {
            image: image.image,
            view: image.whole_view,
            mip_range: mip_range,
            handle: image.handle,
            layout: &image.layout,
            _marker: PhantomData,
            _marker2: PhantomData,
        }
    }
    fn create_new_view<'a>(
        &'a self,
        mip_range: Range<u32>,
        swizzel: vk::ComponentMapping,
    ) -> Result<ImageView<'a, Self::Format, Self::Usage>> {
        let image = self.get_ref();
        let create_info = vk::ImageViewCreateInfo::default()
            .components(swizzel)
            .format(Self::Format::format())
            .image(image.image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: Self::Format::ASPECTS,
                base_array_layer: 0,
                layer_count: 1,
                base_mip_level: mip_range.start,
                level_count: mip_range.end - mip_range.start,
            });
        let view = unsafe { Ctx::device().create_image_view(&create_info, None)? };
        Ok(ImageView {
            image: image.image,
            view,
            mip_range: mip_range,
            handle: image.handle,
            layout: &image.layout,
            _marker: PhantomData,
            _marker2: PhantomData,
        })
    }

    fn whole<'a>(&'a self) -> ImageSlice<'a, Self::Format, Self::Usage> {
        let image = self.get_ref();
        ImageSlice {
            view: self.whole_view(),
            extend: image.extent,
            offset: IVec2::ZERO,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{
        format::{D32SfloatS8Uint, R8G8B8A8Unorm},
        usage::{DepthAttachment, Sampled},
    };
    use ash::vk::Handle;

    /// A view over a fake image: layout tracking and region math need no device.
    fn view<F: Format, U: ImageUsage>(
        layout: &AtomicU32,
        mips: std::ops::Range<u32>,
    ) -> ImageView<'_, F, U> {
        ImageView {
            image: vk::Image::from_raw(42),
            view: vk::ImageView::null(),
            mip_range: mips.into(),
            handle: BindlessHandle::none(),
            layout,
            _marker: PhantomData,
            _marker2: PhantomData,
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
    fn access_reports_the_previous_layout_and_stores_the_new_one() {
        let layout = undefined();
        let view = view::<R8G8B8A8Unorm, Sampled>(&layout, 0..1);

        let first = view.access(
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::ImageLayout::GENERAL,
        );
        assert_eq!(first.old_layout, vk::ImageLayout::UNDEFINED);
        assert_eq!(first.layout, vk::ImageLayout::GENERAL);
        assert_eq!(first.image, vk::Image::from_raw(42));
        assert_eq!(first.aspect, vk::ImageAspectFlags::COLOR);
        assert_eq!(first.stage, vk::PipelineStageFlags2::TRANSFER);
        assert_eq!(first.access, vk::AccessFlags2::TRANSFER_WRITE);

        // Same layout again: no transition pending.
        let second = view.access(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_STORAGE_READ,
            vk::ImageLayout::GENERAL,
        );
        assert_eq!(second.old_layout, vk::ImageLayout::GENERAL);

        let present = view.access(
            vk::PipelineStageFlags2::empty(),
            vk::AccessFlags2::empty(),
            vk::ImageLayout::PRESENT_SRC_KHR,
        );
        assert_eq!(present.old_layout, vk::ImageLayout::GENERAL);
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
        let c = a.cast::<R8G8B8A8Unorm, crate::image::usage::Storage>();

        a.access(
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::ImageLayout::GENERAL,
        );
        let from_copy = b.access(
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_READ,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );
        assert_eq!(from_copy.old_layout, vk::ImageLayout::GENERAL);
        let from_cast = c.access(
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_READ,
            vk::ImageLayout::GENERAL,
        );
        assert_eq!(from_cast.old_layout, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
    }
}

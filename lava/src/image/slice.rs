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
            self.layout.swap(layout.as_raw() as u32, Ordering::Relaxedaccess) as i32,
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

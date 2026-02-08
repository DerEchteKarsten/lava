use std::{marker::PhantomData, ops::Range};

use ash::vk::{self, Offset3D};
use glam::{UVec2, Vec2};

use crate::{image::{Image, format::VkFormat, usage::UsageSet}, state::Ctx};

pub struct ImageView<F: VkFormat> {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub mips: Range<u32>,
    _marker: PhantomData<F>,
}

pub struct ImageSlice<F: VkFormat> {
    pub view: ImageView<F>,
    pub offset: vk::Offset3D,
    pub extend: vk::Extent3D,
}

impl<F: VkFormat> ImageSlice<F> {
    fn offset(mut self, offset: UVec2) -> ImageSlice<F> {
        self.offset.x += offset.x;
        self.offset.y += offset.y;
        self
    }

    fn extend(mut self, extend: UVec2) -> ImageSlice<F> {
        self.extend.width += extend.x;
        self.extend.height += extend.y;
        self
    }
}

trait AsImage {
    type Format: VkFormat;
    type Usage: UsageSet;
    fn get_ref(&self) -> &Image<Self::Format, Self::Usage>;
    fn get_mut(&self) -> &Image<Self::Format, Self::Usage>;

    fn view(&self) -> ImageView<Self::Format> {
        let image = self.get_ref();
        ImageView {
            image: image.handle,
            view: image.whole,
            mips: 0..image.mips,
            _marker: PhantomData
        }
    }
    fn create_new_view(&self, mip_range: Range<u32>, swizzel: vk::ComponentMapping) -> ImageView<Self::Format> {
        let image = self.get_ref();
        let create_info = vk::ImageViewCreateInfo::default()
            .components(swizzel)
            .format(Self::Format::FORMAT)
            .image(image.handle)
            .view_type(vk::ImageViewType::TYPE_2D)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: Self::Format::ASPECTS,
                base_array_layer: 0,
                layer_count: 1,
                base_mip_level: mip_range.start,
                level_count: mip_range.end,
            });
        let view = unsafe { Ctx::device().create_image_view(&create_info, None).unwrap() };
        ImageView {
            image: image.handle,
            view,
            mips: mip_range,
            _marker: PhantomData
        }
    }

    fn whole(&self) -> ImageSlice<Self::Format> {
        let image = self.get_ref();
        ImageSlice {
            view: self.view(),
            extend: image.extend,
            offset: Offset3D::default(),
        }
    }
    fn offset(&self, offset: UVec2) -> ImageSlice<Self::Format> {
        self.whole().offset(offset)
    }
    fn extend(&self, extend: UVec2) -> ImageSlice<Self::Format> {
        self.whole().extend(extend)
    }
}

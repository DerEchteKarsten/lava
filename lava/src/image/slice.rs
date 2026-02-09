use std::{marker::PhantomData, ops::Range};

use ash::vk::{self, Offset3D};
use glam::{UVec2, Vec2};

use crate::{image::{Image, format::VkFormat, usage::UsageSet}, state::Ctx};

#[derive(Clone, Copy)]
pub struct ImageView<F: VkFormat, U: UsageSet> {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub base_mip: u32,
    pub num_mips: u32,
    pub(crate) _marker: PhantomData<F>,
    pub(crate) _marker2: PhantomData<U>
}

impl<F: VkFormat, U: UsageSet> ImageView<F, U> {
    pub fn subresource_range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange { aspect_mask: F::ASPECTS, base_mip_level: self.base_mip, level_count: self.num_mips, base_array_layer: 0, layer_count: 1 }
    }
}

#[derive(Clone, Copy)]
pub struct ImageSlice<F: VkFormat, U: UsageSet> {
    pub view: ImageView<F, U>,
    pub offset: vk::Offset3D,
    pub extend: vk::Extent3D,
}

impl<F: VkFormat, U: UsageSet> ImageSlice<F, U> {
    fn offset(mut self, offset: UVec2) -> ImageSlice<F, U> {
        self.offset.x += offset.x;
        self.offset.y += offset.y;
        self
    }

    fn extent(mut self, extend: UVec2) -> ImageSlice<F, U> {
        self.extend.width += extend.x;
        self.extend.height += extend.y;
        self
    }
    fn subresource_range(&self) -> vk::ImageSubresourceRange {
        self.view.subresource_range()
    }
}

impl<F: VkFormat, U: UsageSet> AsImage for Image<F, U> {
    type Format = F;
    type Usage = U;
    fn get_ref(&self) -> &Image<Self::Format, Self::Usage> {
        self
    }
    fn get_mut(&mut self) -> &mut Image<Self::Format, Self::Usage> {
        self
    }
}

pub trait AsImage {
    type Format: VkFormat;
    type Usage: UsageSet;
    fn get_ref(&self) -> &Image<Self::Format, Self::Usage>;
    fn get_mut(&mut self) -> &mut Image<Self::Format, Self::Usage>;

    fn view(&self) -> ImageView<Self::Format, Self::Usage> {
        let image = self.get_ref();
        ImageView {
            image: image.handle,
            view: image.whole_view,
            base_mip: 0,
            num_mips: image.mips,
            _marker: PhantomData,
            _marker2: PhantomData
        }
    }
    fn create_new_view(&self, base_mip: u32, num_mips: u32, swizzel: vk::ComponentMapping) -> ImageView<Self::Format, Self::Usage> {
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
                base_mip_level: base_mip,
                level_count: num_mips,
            });
        let view = unsafe { Ctx::device().create_image_view(&create_info, None).unwrap() };
        ImageView {
            image: image.handle,
            view,
            num_mips,
            base_mip,
            _marker: PhantomData,
            _marker2: PhantomData,
        }
    }

    fn whole(&self) -> ImageSlice<Self::Format, Self::Usage> {
        let image = self.get_ref();
        ImageSlice {
            view: self.view(),
            extend: image.extent,
            offset: Offset3D::default(),
        }
    }
    fn offset(&self, offset: UVec2) -> ImageSlice<Self::Format, Self::Usage> {
        self.whole().offset(offset)
    }
    fn extend(&self, extend: UVec2) -> ImageSlice<Self::Format, Self::Usage> {
        self.whole().extend(extend)
    }
}

use std::{marker::PhantomData, ops::Range};

use ash::vk::{self, Offset3D};
use glam::{UVec2, Vec2};

use crate::{
    bindless::BindlessHandle,
    image::{
        Image,
        format::{Format, Undefined},
        usage::{IsSampled, IsStorage, Unknown, UsageSet},
    },
    state::Ctx,
};

struct Dynamic<> {

}

#[derive(Clone, Copy, Debug)]
pub struct ImageView<F: Format = Undefined, U: UsageSet = Unknown> {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub base_mip: u32,
    pub num_mips: u32,
    pub handle: Option<BindlessHandle>,
    pub(crate) _marker: PhantomData<F>,
    pub(crate) _marker2: PhantomData<U>,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct TypeLessImageView {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub aspect: vk::ImageAspectFlags,
    pub base_mip: u32,
    pub num_mips: u32,
}

#[derive(Clone, Copy)]
pub struct StorageImageViewBinding {
    pub aspect: vk::ImageAspectFlags,
    pub prefered_layout: vk::ImageLayout,
    pub view: vk::ImageView,
    pub image: vk::Image,
    pub handle: BindlessHandle,
    pub base_mip: u32,
    pub num_mips: u32,
}
#[derive(Clone, Copy)]
pub struct SampledImageViewBinding {
    pub aspect: vk::ImageAspectFlags,
    pub prefered_layout: vk::ImageLayout,
    pub view: vk::ImageView,
    pub image: vk::Image,
    pub handle: BindlessHandle,
    pub base_mip: u32,
    pub num_mips: u32,
}

impl Into<TypeLessImageView> for StorageImageViewBinding {
    fn into(self) -> TypeLessImageView {
        TypeLessImageView { image: self.image, view: self.view, aspect: self.aspect, base_mip: self.base_mip, num_mips: self.num_mips }
    }
}

impl Into<TypeLessImageView> for SampledImageViewBinding {
    fn into(self) -> TypeLessImageView {
        TypeLessImageView { image: self.image, view: self.view, aspect: self.aspect, base_mip: self.base_mip, num_mips: self.num_mips }
    }
}

impl TypeLessImageView {
    pub(crate) fn subresource_range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: self.aspect,
            base_mip_level: self.base_mip,
            level_count: self.num_mips,
            base_array_layer: 0,
            layer_count: 1,
        }
    }
    pub(crate) fn subresource_layers(&self) -> vk::ImageSubresourceLayers {
        vk::ImageSubresourceLayers {
            aspect_mask: self.aspect,
            mip_level: self.base_mip,
            base_array_layer: 0,
            layer_count: 1,
        }
    }
}

impl<F: Format, U: UsageSet> Into<TypeLessImageView> for ImageView<F, U> {
    fn into(self) -> TypeLessImageView {
        TypeLessImageView {
            image: self.image,
            view: self.view,
            aspect: F::ASPECTS,
            base_mip: self.base_mip,
            num_mips: self.num_mips,
        }
    }
}

impl<F: Format, U: IsStorage> ImageView<F, U> {
    pub fn as_storage(self) -> StorageImageViewBinding {
        StorageImageViewBinding {
            aspect: F::ASPECTS,
            prefered_layout: U::PREFERED_LAYOUT,
            view: self.view,
            image: self.image,
            handle: self.handle.unwrap(),
            base_mip: self.base_mip,
            num_mips: self.num_mips
        }
    }
}

impl<F: Format, U: IsSampled> ImageView<F, U> {
    pub fn as_sampled(self) -> SampledImageViewBinding {
        SampledImageViewBinding {
            aspect: F::ASPECTS,
            prefered_layout: U::PREFERED_LAYOUT,
            view: self.view,
            image: self.image,
            handle: self.handle.unwrap(),
            base_mip: self.base_mip,
            num_mips: self.num_mips
        }
    }
}

impl<F: Format, U: UsageSet> ImageView<F, U> {
    pub(crate) fn subresource_range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: F::ASPECTS,
            base_mip_level: self.base_mip,
            level_count: self.num_mips,
            base_array_layer: 0,
            layer_count: 1,
        }
    }
    pub(crate) fn subresource_layers(&self) -> vk::ImageSubresourceLayers {
        vk::ImageSubresourceLayers {
            aspect_mask: F::ASPECTS,
            mip_level: self.base_mip,
            base_array_layer: 0,
            layer_count: 1,
        }
    }
    pub fn cast<NF: Format, NU: UsageSet>(self) -> ImageView<NF, NU> {
        unsafe { std::mem::transmute(self) }
    }
}

impl<F: Format, U: IsStorage> Image<F, U> {
    pub fn as_storage(&self) -> StorageImageViewBinding {
        self.view().as_storage()
    }
}
impl<F: Format, U: IsSampled> Image<F, U> {
    pub fn as_sampled(&self) -> SampledImageViewBinding {
        self.view().as_sampled()
    }
}

#[derive(Clone, Copy)]
pub struct ImageSlice<F: Format = Undefined, U: UsageSet = Unknown> {
    pub view: ImageView<F, U>,
    pub offset: vk::Offset3D,
    pub extend: vk::Extent3D,
}

impl<F: Format, U: UsageSet> ImageSlice<F, U> {
    pub fn offset(mut self, offset: UVec2) -> ImageSlice<F, U> {
        self.offset.x += offset.x as i32;
        self.offset.y += offset.y as i32;
        self
    }

    pub fn extent(mut self, extend: UVec2) -> ImageSlice<F, U> {
        self.extend.width += extend.x;
        self.extend.height += extend.y;
        self
    }

    pub fn cast<NF: Format, NU: UsageSet>(self) -> ImageSlice<NF, NU> {
        unsafe { std::mem::transmute(self) }
    }
}

impl<F: Format, U: UsageSet> AsImage for Image<F, U> {
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
    type Format: Format;
    type Usage: UsageSet;
    fn get_ref(&self) -> &Image<Self::Format, Self::Usage>;
    fn get_mut(&mut self) -> &mut Image<Self::Format, Self::Usage>;

    fn view(&self) -> ImageView<Self::Format, Self::Usage> {
        let image = self.get_ref();
        ImageView {
            image: image.image,
            view: image.whole_view,
            base_mip: 0,
            num_mips: image.mips,
            handle: image.handle,
            _marker: PhantomData,
            _marker2: PhantomData,
        }
    }
    fn create_new_view(
        &self,
        base_mip: u32,
        num_mips: u32,
        swizzel: vk::ComponentMapping,
    ) -> ImageView<Self::Format, Self::Usage> {
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
                base_mip_level: base_mip,
                level_count: num_mips,
            });
        let view = unsafe { Ctx::device().create_image_view(&create_info, None).unwrap() };
        ImageView {
            image: image.image,
            view,
            num_mips,
            base_mip,
            handle: image.handle,
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
        self.whole().extent(extend)
    }
}

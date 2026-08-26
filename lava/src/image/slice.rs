use std::marker::PhantomData;

use ash::vk::{self, Offset3D};
use glam::UVec2;

use crate::{
    bindless::BindlessHandle,
    image::{
        Image,
        format::{Format, Undefined},
        usage::{IsSampled, IsStorage, Unknown, UsageSet},
    },
    state::Ctx,
};

struct Dynamic {}

#[derive(Clone, Copy, Debug)]
pub struct ImageView<'a, F: Format = Undefined, U: UsageSet = Unknown> {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub base_mip: u32,
    pub num_mips: u32,
    pub handle: Option<BindlessHandle>,
    pub(crate) _marker: PhantomData<F>,
    pub(crate) _marker2: PhantomData<U>,
    pub(crate) _marker3: PhantomData<&'a ()>,
}

#[derive(Clone, Copy)]
pub struct StorageImageViewBinding<'a> {
    pub aspect: vk::ImageAspectFlags,
    pub prefered_layout: vk::ImageLayout,
    pub view: vk::ImageView,
    pub image: vk::Image,
    pub handle: BindlessHandle,
    pub base_mip: u32,
    pub num_mips: u32,
    marker: PhantomData<&'a ()>,
}
#[derive(Clone, Copy)]
pub struct SampledImageViewBinding<'a> {
    pub aspect: vk::ImageAspectFlags,
    pub prefered_layout: vk::ImageLayout,
    pub view: vk::ImageView,
    pub image: vk::Image,
    pub handle: BindlessHandle,
    pub base_mip: u32,
    pub num_mips: u32,
    marker: PhantomData<&'a ()>,
}

impl<'a, F: Format, U: IsStorage> ImageView<'a, F, U> {
    pub fn as_storage(self) -> StorageImageViewBinding<'a> {
        StorageImageViewBinding {
            aspect: F::ASPECTS,
            prefered_layout: U::PREFERED_LAYOUT,
            view: self.view,
            image: self.image,
            handle: self.handle.unwrap(),
            base_mip: self.base_mip,
            num_mips: self.num_mips,
            marker: PhantomData,
        }
    }
}

impl<'a, F: Format, U: IsSampled> ImageView<'a, F, U> {
    pub fn as_sampled(self) -> SampledImageViewBinding<'a> {
        SampledImageViewBinding {
            aspect: F::ASPECTS,
            prefered_layout: U::PREFERED_LAYOUT,
            view: self.view,
            image: self.image,
            handle: self.handle.unwrap(),
            base_mip: self.base_mip,
            num_mips: self.num_mips,
            marker: PhantomData,
        }
    }
}

impl<'a, F: Format, U: UsageSet> ImageView<'a, F, U> {
    pub fn subresource_range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: F::ASPECTS,
            base_mip_level: self.base_mip,
            level_count: self.num_mips,
            base_array_layer: 0,
            layer_count: 1,
        }
    }
    pub fn subresource_layers(&self) -> vk::ImageSubresourceLayers {
        vk::ImageSubresourceLayers {
            aspect_mask: F::ASPECTS,
            mip_level: self.base_mip,
            base_array_layer: 0,
            layer_count: 1,
        }
    }
    pub fn region(self, extend: UVec2) -> ImageSlice<'a, F, U> {
        ImageSlice {
            view: self,
            offset: Offset3D { x: 0, y: 0, z: 0 },
            extend: vk::Extent3D {
                width: extend.x,
                height: extend.y,
                depth: 1,
            },
        }
    }
    pub fn cast<NF: Format, NU: UsageSet>(self) -> ImageView<'a, NF, NU> {
        unsafe { std::mem::transmute(self) }
    }
}

impl<F: Format, U: IsStorage> Image<F, U> {
    pub fn as_storage<'a>(&'a self) -> StorageImageViewBinding<'a> {
        self.view().as_storage()
    }
}
impl<F: Format, U: IsSampled> Image<F, U> {
    pub fn as_sampled<'a>(&'a self) -> SampledImageViewBinding<'a> {
        self.view().as_sampled()
    }
}

#[derive(Clone, Copy)]
pub struct ImageSlice<'a, F: Format = Undefined, U: UsageSet = Unknown> {
    pub view: ImageView<'a, F, U>,
    pub offset: vk::Offset3D,
    pub extend: vk::Extent3D,
}

impl<'a, F: Format, U: UsageSet> ImageSlice<'a, F, U> {
    pub fn offset(mut self, offset: UVec2) -> ImageSlice<'a, F, U> {
        self.offset.x += offset.x as i32;
        self.offset.y += offset.y as i32;
        self
    }

    pub fn extent(mut self, extend: UVec2) -> ImageSlice<'a, F, U> {
        self.extend.width += extend.x;
        self.extend.height += extend.y;
        self
    }

    pub fn cast<NF: Format, NU: UsageSet>(self) -> ImageSlice<'a, NF, NU> {
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

    fn view<'a>(&'a self) -> ImageView<'a, Self::Format, Self::Usage> {
        let image = self.get_ref();
        ImageView {
            image: image.image,
            view: image.whole_view,
            base_mip: 0,
            num_mips: image.mips,
            handle: image.handle,
            _marker: PhantomData,
            _marker2: PhantomData,
            _marker3: PhantomData,
        }
    }
    fn create_new_view<'a>(
        &'a self,
        base_mip: u32,
        num_mips: u32,
        swizzel: vk::ComponentMapping,
    ) -> ImageView<'a, Self::Format, Self::Usage> {
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
            _marker3: PhantomData,
        }
    }

    fn whole<'a>(&'a self) -> ImageSlice<'a, Self::Format, Self::Usage> {
        let image = self.get_ref();
        ImageSlice {
            view: self.view(),
            extend: image.extent,
            offset: Offset3D::default(),
        }
    }
    fn offset<'a>(&'a self, offset: UVec2) -> ImageSlice<'a, Self::Format, Self::Usage> {
        self.whole().offset(offset)
    }
    fn extend<'a>(&'a self, extend: UVec2) -> ImageSlice<'a, Self::Format, Self::Usage> {
        self.whole().extent(extend)
    }
}

use std::{marker::PhantomData, range::Range};

use ash::vk::{self, Offset3D};
use glam::UVec2;

use crate::{
    bindless::BindlessHandle,
    image::{
        Image,
        format::{Format, Undefined},
        usage::{IsSampled, IsStorage, Unknown, UsageSet},
    },
    state::{Ctx, Functions},
};

#[derive(Clone, Copy, Debug)]
pub struct ImageView<'a, F: Format = Undefined, U: UsageSet = Unknown> {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub mip_range: Range<u32>,
    pub handle: Option<BindlessHandle>,
    pub(crate) _marker: PhantomData<F>,
    pub(crate) _marker2: PhantomData<U>,
    pub(crate) _marker3: PhantomData<&'a ()>,
}

impl<'a, F: Format, U: UsageSet> ImageView<'a, F, U> {
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

    pub fn grow(mut self, extend: UVec2) -> ImageSlice<'a, F, U> {
        self.extend.width += extend.x;
        self.extend.height += extend.y;
        self
    }

    pub fn cast<NF: Format, NU: UsageSet>(self) -> ImageSlice<'a, NF, NU> {
        unsafe { std::mem::transmute(self) }
    }

    pub fn copy_from(&self, data: &[u8], mip_level: u32) {
        let regions = [vk::MemoryToImageCopyEXT::default()
            .host_pointer(data.as_ptr().cast())
            .image_extent(self.extend)
            .image_offset(self.offset)
            .image_subresource(self.view.subresource_layers(mip_level))
            .memory_image_height(self.extend.height)
            .memory_row_length(self.extend.width)];
        let copy_memory_to_image_info = vk::CopyMemoryToImageInfoEXT::default()
            .dst_image(self.view.image)
            .dst_image_layout(vk::ImageLayout::UNDEFINED)
            .regions(&regions);
        unsafe {
            Functions::host_image_copy()
                .copy_memory_to_image(&copy_memory_to_image_info)
                .unwrap()
        };
    }
}

impl<const M: u32, F: Format, U: UsageSet> AsImage<M> for Image<M, F, U> {
    type Format = F;
    type Usage = U;

    fn mip_range(&self) -> Range<u32> {
        (0..M).into()
    }
    fn get_ref(&self) -> &Image<M, Self::Format, Self::Usage> {
        self
    }
    fn get_mut(&mut self) -> &mut Image<M, Self::Format, Self::Usage> {
        self
    }
}

pub trait AsImage<const M: u32> {
    type Format: Format;
    type Usage: UsageSet;

    fn mip_range(&self) -> Range<u32>;
    fn get_ref(&self) -> &Image<M, Self::Format, Self::Usage>;
    fn get_mut(&mut self) -> &mut Image<M, Self::Format, Self::Usage>;

    fn whole_view<'a>(&'a self) -> ImageView<'a, Self::Format, Self::Usage> {
        let mip_range = self.mip_range();
        let image = self.get_ref();
        ImageView {
            image: image.image,
            view: image.whole_view,
            mip_range: mip_range,
            handle: image.handle,
            _marker: PhantomData,
            _marker2: PhantomData,
            _marker3: PhantomData,
        }
    }
    fn create_new_view<'a>(
        &'a self,
        mip_range: Range<u32>,
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
                base_mip_level: mip_range.start,
                level_count: mip_range.end - mip_range.start,
            });
        let view = unsafe { Ctx::device().create_image_view(&create_info, None).unwrap() };
        ImageView {
            image: image.image,
            view,
            mip_range: mip_range,
            handle: image.handle,
            _marker: PhantomData,
            _marker2: PhantomData,
            _marker3: PhantomData,
        }
    }

    fn whole<'a>(&'a self) -> ImageSlice<'a, Self::Format, Self::Usage> {
        let image = self.get_ref();
        ImageSlice {
            view: self.whole_view(),
            extend: image.extent,
            offset: Offset3D::default(),
        }
    }
    fn offset<'a>(&'a self, offset: UVec2) -> ImageSlice<'a, Self::Format, Self::Usage> {
        self.whole().offset(offset)
    }
    fn extend<'a>(&'a self, extend: UVec2) -> ImageSlice<'a, Self::Format, Self::Usage> {
        self.whole().grow(extend)
    }
}

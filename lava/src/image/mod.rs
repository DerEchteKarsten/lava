use std::{marker::PhantomData, ops::Range};

use anyhow::Result;
use ash::vk;
use gpu_allocator::vulkan::Allocation;

use crate::{
    image::{format::VkFormat, usage::UsageSet},
    state::Ctx,
};

pub mod format;
pub mod usage;
pub mod slice;

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug, Default)]
pub enum ImageSize {
    #[default]
    FullScreen,
    FractionalFullScreen(u32, u32),
    XY(u32, u32),
}

impl ImageSize {
    pub fn size(self) -> vk::Extent3D {
        match self {
            Self::FullScreen => vk::Extent3D {
                width: Ctx::window_width(),
                height: Ctx::window_height(),
                depth: 1,
            },
            Self::FractionalFullScreen(dx, dy) => vk::Extent3D {
                width: (Ctx::window_width()).div_ceil(dx),
                height: (Ctx::window_height()).div_ceil(dy),
                depth: 1,
            },
            Self::XY(width, height) => vk::Extent3D {
                width,
                height,
                depth: 1,
            },
        }
    }
}

#[derive(Debug)]
pub struct Image<F: VkFormat, U: UsageSet> {
    pub handle: vk::Image,
    pub whole_view: vk::ImageView,
    pub allocation: Allocation,
    pub mips: u32,
    pub extend: vk::Extent3D,
    _format: PhantomData<F>,
    _usage: PhantomData<U>,
}

impl<F: VkFormat, U: UsageSet> Image<F, U> {
    fn new_mipped(size: ImageSize, mips: u32) -> Result<Self> {
        let extent = size.size();
        let create_info = vk::ImageCreateInfo {
            array_layers: 1,
            extent,
            format: F::FORMAT,
            image_type: vk::ImageType::TYPE_2D,
            initial_layout: vk::ImageLayout::UNDEFINED,
            mip_levels: mips,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            samples: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };
            
        let image = unsafe { Ctx::device().create_image(create_info, None) }?;

    }
}
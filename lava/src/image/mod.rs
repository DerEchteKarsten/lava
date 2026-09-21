use std::marker::PhantomData;

use anyhow::Result;
use ash::vk::{self, ComponentSwizzle};
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc};
use lava_macros::validation_trace;

use crate::{
    bindless::{Bindless, BindlessHandle},
    image::{
        format::{Format, Undefined},
        slice::AsImage,
        usage::{Unknown, UsageSet},
    },
    state::Ctx,
};

pub mod format;
pub mod slice;
pub mod usage;

#[derive(Debug)]
pub struct Image<const M: u32 = 1, F: Format = Undefined, U: UsageSet = Unknown> {
    pub image: vk::Image,
    pub whole_view: vk::ImageView,
    pub allocation: Allocation,
    pub extent: vk::Extent3D,
    pub handle: Option<BindlessHandle>,
    _format: PhantomData<F>,
    _usage: PhantomData<U>,
}

impl<const M: u32, F: Format, U: UsageSet> Image<M, F, U> {
    #[validation_trace]
    pub fn new(width: u32, height: u32) -> Result<Self> {
        let extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };
        let create_info = vk::ImageCreateInfo {
            array_layers: 1,
            extent,
            format: F::format(),
            image_type: vk::ImageType::TYPE_2D,
            mip_levels: M,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: U::VK
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | if Ctx::features().rebar {
                    vk::ImageUsageFlags::HOST_TRANSFER_EXT
                } else {
                    vk::ImageUsageFlags::empty()
                },
            initial_layout: vk::ImageLayout::GENERAL,
            ..Default::default()
        };

        let image = unsafe { Ctx::device().create_image(&create_info, None) }?;
        let requirements = unsafe { Ctx::device().get_image_memory_requirements(image) };

        let desc = AllocationCreateDesc {
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            linear: false,
            location: gpu_allocator::MemoryLocation::GpuOnly,
            name: "ImageMemory",
            requirements,
        };
        let allocation = Ctx::allocator().allocate(&desc)?;

        unsafe {
            Ctx::device().bind_image_memory(image, allocation.memory(), allocation.offset())?
        };

        let mut s = Self {
            _format: PhantomData,
            _usage: PhantomData,
            handle: None,
            allocation,
            extent,
            image,
            whole_view: vk::ImageView::null(),
        };
        let (handle, view) = {
            let view = s.create_new_view(
                (0..M).into(),
                vk::ComponentMapping {
                    r: ComponentSwizzle::R,
                    g: ComponentSwizzle::G,
                    b: ComponentSwizzle::B,
                    a: ComponentSwizzle::A,
                },
            );
            let handle = Bindless::push(view);
            (handle, view.view)
        };
        s.handle = handle;
        s.whole_view = view;
        Ok(s)
    }

    pub fn cast<NF: Format, NU: UsageSet>(self) -> Image<M, NF, NU> {
        unsafe { std::mem::transmute(self) }
    }
}

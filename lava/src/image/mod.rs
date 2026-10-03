//! Format- and usage-typed GPU images with mip levels and host uploads
use std::{marker::PhantomData, sync::atomic::AtomicU32};

use crate::error::Result;
use ash::vk::{self, ComponentSwizzle};
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc};
use lava_macros::validation_trace;
use smallvec::SmallVec;

use crate::{
    bindless::{Bindless, BindlessHandle},
    image::{
        format::{Format, Undefined},
        slice::AsImage,
        usage::ImageUsage,
    },
    state::{Ctx, Functions},
    vkobjects::queue::{CommandBufferMemory, Fence, Gfx, Queue, SemaphoreInfo},
};
use glam::UVec2;

pub mod format;
pub mod slice;
pub mod usage;

#[derive(Debug)]
pub struct Image<F: Format, U: ImageUsage> {
    pub(crate) image: vk::Image,
    pub(crate) whole_view: vk::ImageView,
    pub(crate) allocation: Allocation,
    pub extent: UVec2,
    pub handle: BindlessHandle,
    pub mip_levels: u32,
    pub(crate) layout: AtomicU32,
    _format: PhantomData<F>,
    _usage: PhantomData<U>,
}

impl<F: Format, U: ImageUsage> Drop for Image<F, U> {
    fn drop(&mut self) {
        unsafe {
            Ctx::device().destroy_image_view(self.whole_view, None);
            Ctx::device().destroy_image(self.image, None);
        }
        let alloc = std::mem::take(&mut self.allocation);
        if let Err(err) = Ctx::allocator().free(alloc) {
            tracing::error!(%err, "failed to free image allocation");
        }
    }
}

impl<F: Format, U: ImageUsage> Image<F, U> {
    #[validation_trace]
    pub fn new(width: u32, height: u32) -> Result<Self> {
        Self::with_mip_levels(width, height, 1)
    }
    #[validation_trace]
    pub fn with_mip_levels(width: u32, height: u32, mip_levels: u32) -> Result<Self> {
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
            mip_levels,
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
            initial_layout: vk::ImageLayout::UNDEFINED,
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
            handle: BindlessHandle::none(),
            allocation,
            extent: UVec2::new(width, height),
            mip_levels,
            image,
            whole_view: vk::ImageView::null(),
            layout: AtomicU32::new(vk::ImageLayout::UNDEFINED.as_raw() as u32),
        };
        let (handle, view) = {
            let view = s.create_new_view(
                (0..mip_levels).into(),
                vk::ComponentMapping {
                    r: ComponentSwizzle::R,
                    g: ComponentSwizzle::G,
                    b: ComponentSwizzle::B,
                    a: ComponentSwizzle::A,
                },
            )?;
            let handle = Bindless::push(view);
            (handle, view.view)
        };
        s.handle = handle;
        s.whole_view = view;
        Ok(s)
    }

    #[validation_trace]
    pub fn copy_from(&mut self, data: &[u8], mip_level: u32) -> Result<()> {
        let range = self.whole_view().subresource_range();
        let layout = self.layout.get_mut();
        let old_layout = vk::ImageLayout::from_raw(*layout as i32);
        if old_layout != vk::ImageLayout::GENERAL {
            *layout = vk::ImageLayout::GENERAL.as_raw() as u32;
            let transition = vk::HostImageLayoutTransitionInfoEXT::default()
                .image(self.image)
                .old_layout(old_layout)
                .new_layout(vk::ImageLayout::GENERAL)
                .subresource_range(range);
            unsafe { Functions::host_image_copy().transition_image_layout(&[transition])? };
        }
        let extent = self.mip_extent(mip_level);
        let regions = [vk::MemoryToImageCopyEXT::default()
            .host_pointer(data.as_ptr().cast())
            .image_extent(vk::Extent3D {
                width: extent.x,
                height: extent.y,
                depth: 1,
            })
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: F::ASPECTS,
                mip_level,
                base_array_layer: 0,
                layer_count: 1,
            })
            .memory_image_height(extent.y)
            .memory_row_length(extent.x)];
        let info = vk::CopyMemoryToImageInfoEXT::default()
            .dst_image(self.image)
            .dst_image_layout(vk::ImageLayout::GENERAL)
            .regions(&regions);
        unsafe { Functions::host_image_copy().copy_memory_to_image(&info)? };
        Ok(())
    }

    pub fn mip_extent(&self, level: u32) -> UVec2 {
        UVec2::new(
            (self.extent.x >> level).max(1),
            (self.extent.y >> level).max(1),
        )
    }

    pub fn cast<NF: Format, NU: ImageUsage>(self) -> Image<NF, NU> {
        unsafe { std::mem::transmute(self) }
    }
}

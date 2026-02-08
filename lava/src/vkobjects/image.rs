use std::{
    any::TypeId,
    marker::PhantomData,
    ops::Range,
    sync::{Arc, Mutex},
};

use anyhow::Result;
use ash::vk;
use bytemuck::Pod;
use glam::UVec2;
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc},
};

use derivative::Derivative;

use crate::{
    bindless::{Bindless, BindlessHandle},
    state::Ctx,
};

impl<F: VkFormat, U: UsageSet> Image<F, U> {
    pub(super) fn view(
        device: &ash::Device,
        image: vk::Image,
        format: vk::Format,
    ) -> vk::ImageView {
        let aspect = get_aspects(format);
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(aspect)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);
        let image_view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            })
            .subresource_range(subresource_range);
        unsafe { device.create_image_view(&image_view_info, None) }.unwrap()
    }

    pub fn new_2d(usage: vk::ImageUsageFlags, format: vk::Format, size: ImageSize) -> Result<Self> {
        let extent = vk::Extent3D {
            width: size.size().x,
            height: size.size().y,
            depth: 1,
        };

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe { Ctx::device().create_image(&image_info, None)? };
        let requirements = unsafe { Ctx::device().get_image_memory_requirements(image) };

        let allocation = Ctx::allocator().allocate(&AllocationCreateDesc {
            name: "image",
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        })?;

        unsafe {
            Ctx::device().bind_image_memory(image, allocation.memory(), allocation.offset())?
        };

        let view = Self::view(&Ctx::device(), image, format);

        let mut s = Self {
            usage,
            handle: image,
            allocation: Some(allocation),
            format,
            size,
            view,
            bindless_handle: None,
        };

        let handle = if usage.contains(vk::ImageUsageFlags::STORAGE) {
            Some(Bindless::push_image(&s))
        } else if usage.contains(vk::ImageUsageFlags::SAMPLED) {
            Some(Bindless::push_texture(&s))
        } else {
            None
        };
        s.bindless_handle = handle;
        Ok(s)
    }

    pub fn destroy(&mut self) {
        unsafe {
            if let Some(allocation) = self.allocation.take() {
                Ctx::allocator().free(allocation);
            }
            Ctx::device().destroy_image_view(self.view, None);
            Ctx::device().destroy_image(self.handle, None);
        }
    }

    pub fn prefered_layout(&self) -> vk::ImageLayout {
        if self.usage.contains(vk::ImageUsageFlags::STORAGE) {
            vk::ImageLayout::GENERAL
        } else if self.usage.contains(vk::ImageUsageFlags::SAMPLED) {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        } else {
            panic!("Image does not have SAMPELD or STORAGE usage flag")
        }
    }

    pub fn mut_access(&self) -> vk::AccessFlags2 {
        if self.usage.contains(vk::ImageUsageFlags::STORAGE) {
            vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::SHADER_STORAGE_READ
        } else {
            panic!("Trying to write to Image that didnt have the STORAGE usage flag");
        }
    }

    pub fn const_access(&self) -> vk::AccessFlags2 {
        if self.usage.contains(vk::ImageUsageFlags::STORAGE) {
            vk::AccessFlags2::SHADER_STORAGE_READ
        } else if self.usage.contains(vk::ImageUsageFlags::SAMPLED) {
            vk::AccessFlags2::SHADER_SAMPLED_READ
        } else {
            panic!("Image does not have SAMPLED or STORAGE usage flag")
        }
    }
}

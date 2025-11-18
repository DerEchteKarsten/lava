use std::sync::{Arc, Mutex};

use anyhow::Result;
use ash::vk;
use glam::UVec2;
use gpu_allocator::{MemoryLocation, vulkan::AllocationCreateDesc};

use derivative::Derivative;

use crate::{
    bindless::{Bindless, BindlessHandle},
    state::Ctx,
    vkobjects::buffer::MAllocation,
};

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug, Default)]
pub enum ImageSize {
    #[default]
    FullScreen,
    FractionalFullScreen(u32, u32),
    XY(u32, u32),
}

impl ImageSize {
    pub fn size(self) -> UVec2 {
        match self {
            Self::FullScreen => {
                UVec2::new(Ctx::window_width().unwrap(), Ctx::window_height().unwrap())
            }
            Self::FractionalFullScreen(dx, dy) => UVec2::new(
                (Ctx::window_width().unwrap()).div_ceil(dx),
                (Ctx::window_height().unwrap()).div_ceil(dy),
            ),
            Self::XY(x, y) => UVec2::new(x, y),
        }
    }
}

#[derive(Derivative)]
#[derivative(Eq, PartialEq, Debug)]
pub struct Image {
    pub bindless_handle: Option<BindlessHandle>,
    pub handle: vk::Image,
    pub view: vk::ImageView,
    #[derivative(PartialEq = "ignore")]
    pub allocation: Option<Arc<Mutex<MAllocation>>>,
    pub size: ImageSize,
    pub format: vk::Format,
    pub usage: vk::ImageUsageFlags,
}

impl Clone for Image {
    fn clone(&self) -> Self {
        Self {
            size: self.size,
            format: self.format,
            handle: self.handle,
            usage: self.usage,
            view: self.view,
            allocation: self.allocation.clone(),
            bindless_handle: self.bindless_handle,
        }
    }
}

pub fn get_aspects(format: vk::Format) -> vk::ImageAspectFlags {
    if format == vk::Format::D16_UNORM
        || format == vk::Format::D32_SFLOAT
        || format == vk::Format::X8_D24_UNORM_PACK32
    {
        vk::ImageAspectFlags::DEPTH
    } else if format == vk::Format::D16_UNORM_S8_UINT
        || format == vk::Format::D24_UNORM_S8_UINT
        || format == vk::Format::D32_SFLOAT_S8_UINT
    {
        vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
    } else if format == vk::Format::S8_UINT {
        vk::ImageAspectFlags::STENCIL
    } else {
        vk::ImageAspectFlags::COLOR
    }
}
pub trait ImageType {
    fn get_extent(&self) -> vk::Extent2D;
    fn get_image(&self) -> vk::Image;
    fn get_usage(&self) -> vk::ImageUsageFlags;
    fn get_format(&self) -> vk::Format;
    fn get_view(&self) -> vk::ImageView;
    fn copy(
        &self,
        cmd: &vk::CommandBuffer,
        other: &impl ImageType,
        src_layout: vk::ImageLayout,
        dst_layout: vk::ImageLayout,
    ) {
        unsafe {
            Ctx::device().cmd_copy_image(
                *cmd,
                self.get_image(),
                src_layout,
                other.get_image(),
                dst_layout,
                &[vk::ImageCopy::default()
                    .extent(vk::Extent3D {
                        width: self.get_extent().width,
                        height: self.get_extent().height,
                        depth: 1,
                    })
                    .src_offset(vk::Offset3D::default())
                    .dst_offset(vk::Offset3D::default())
                    .src_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: get_aspects(self.get_format()),
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .dst_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: get_aspects(other.get_format()),
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })],
            )
        };
    }

    fn blit(
        &self,
        cmd: &vk::CommandBuffer,
        other: &impl ImageType,
        src_layout: vk::ImageLayout,
        dst_layout: vk::ImageLayout,
    ) {
        unsafe {
            Ctx::device().cmd_blit_image(
                *cmd,
                self.get_image(),
                src_layout,
                other.get_image(),
                dst_layout,
                &[vk::ImageBlit::default()
                    .src_offsets([
                        vk::Offset3D::default(),
                        vk::Offset3D {
                            x: self.get_extent().width as _,
                            y: self.get_extent().height as _,
                            z: 1,
                        },
                    ])
                    .dst_offsets([
                        vk::Offset3D::default(),
                        vk::Offset3D {
                            x: other.get_extent().width as _,
                            y: other.get_extent().height as _,
                            z: 1,
                        },
                    ])
                    .src_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: get_aspects(self.get_format()),
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .dst_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: get_aspects(other.get_format()),
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })],
                vk::Filter::NEAREST,
            )
        };
    }
    fn subresource_range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: get_aspects(self.get_format()),
            base_array_layer: 0,
            base_mip_level: 0,
            layer_count: 1,
            level_count: 1,
        }
    }
}

impl ImageType for Image {
    fn get_extent(&self) -> vk::Extent2D {
        vk::Extent2D {
            width: self.size.size().x,
            height: self.size.size().y,
        }
    }
    fn get_format(&self) -> vk::Format {
        self.format
    }
    fn get_image(&self) -> vk::Image {
        self.handle
    }
    fn get_usage(&self) -> vk::ImageUsageFlags {
        self.usage
    }
    fn get_view(&self) -> vk::ImageView {
        self.view
    }
}

impl Image {
    // pub fn new_from_data(
    //     ctx: &mut Context,
    //     image: DynamicImage,
    //     format: vk::Format,
    // ) -> Result<Self> {
    //     let (width, height) = image.dimensions();
    //     let image_buffer = if format != vk::Format::R8G8B8A8_SRGB {
    //         let image_data = image.to_rgba32f();
    //         let image_data_raw = image_data.as_raw();

    //         let image_buffer = Buffer::new(
    //             ctx,
    //             vk::BufferUsageFlags::TRANSFER_SRC,
    //             MemoryLocation::CpuToGpu,
    //             (size_of::<f32>() * image_data.len()) as u64,
    //         )?;
    //         image_buffer.copy_data_to_buffer(image_data_raw.as_slice())?;
    //         image_buffer
    //     } else {
    //         let image_data = image.to_rgba8();
    //         let image_data_raw = image_data.as_raw();

    //         let image_buffer = Buffer::new(
    //             ctx,
    //             vk::BufferUsageFlags::TRANSFER_SRC,
    //             MemoryLocation::CpuToGpu,
    //             (size_of::<u8>() * image_data.len()) as u64,
    //         )?;

    //         image_buffer.copy_data_to_buffer(image_data_raw.as_slice())?;
    //         image_buffer
    //     };

    //     let texture_image = Image::new_2d(
    //         ctx,
    //         vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
    //         MemoryLocation::GpuOnly,
    //         format,
    //         width,
    //         height,
    //     )?;
    //     ctx.execute_one_time_commands(|cmd| {
    //         let barrier = texture_image.memory_barrier(
    //             vk::ImageLayout::UNDEFINED,
    //             vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    //         );
    //         unsafe {
    //             ctx.device.cmd_pipeline_barrier2(
    //                 *cmd,
    //                 &vk::DependencyInfo::default().image_memory_barriers(&[barrier]),
    //             )
    //         };

    //         image_buffer.copy_to_image(
    //             &ctx,
    //             cmd,
    //             &texture_image,
    //             vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    //         );

    //         let barrier = texture_image.memory_barrier(
    //             vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    //             vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    //         );
    //         unsafe {
    //             ctx.device.cmd_pipeline_barrier2(
    //                 *cmd,
    //                 &vk::DependencyInfo::default().image_memory_barriers(&[barrier]),
    //             )
    //         };
    //     })?;
    //     let extend = texture_image.extent;
    //     Ok(texture_image)
    // }

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
            allocation: Some(Arc::new(Mutex::new(MAllocation(allocation)))),
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

    pub fn destroy(&self) {
        unsafe {
            Ctx::device().destroy_image_view(self.view, None);
            Ctx::device().destroy_image(self.handle, None);
        }
    }
}

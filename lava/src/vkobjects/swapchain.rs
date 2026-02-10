use std::{ffi::CStr, marker::PhantomData};

use anyhow::Result;
use ash::{
    Device,
    vk::{self},
};

use crate::{
    image::{
        Image, format,
        slice::ImageView,
        usage::{ColorAttachment, ColorAttachmentStorage},
    },
    vkobjects::surface::Surface,
};

#[derive(Debug)]
pub struct Swapchain {
    pub resized: bool,
    pub size: [u32; 2],
    pub handle: vk::SwapchainKHR,
    pub format: vk::Format,
    pub color_space: vk::ColorSpaceKHR,
    pub present_mode: vk::PresentModeKHR,
    pub images: Vec<ImageView<format::Swapchain, ColorAttachmentStorage>>,
}

impl Swapchain {
    pub fn new(
        surface: &Surface,
        device: &Device,
        graphics_queue: u32,
        present_queue: u32,
        swapchain_fn: &ash::khr::swapchain::Device,
        debug_utils: Option<&ash::ext::debug_utils::Device>,
        old: Option<vk::SwapchainKHR>,
        size: Option<[u32; 2]>,
    ) -> Result<Self> {
        let format = {
            let formats = &surface.formats;
            if formats.len() == 1 && formats[0].format == vk::Format::UNDEFINED {
                vk::SurfaceFormatKHR {
                    format: vk::Format::B8G8R8A8_UNORM,
                    color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
                }
            } else {
                *formats
                    .iter()
                    .find(|format| {
                        format.format == vk::Format::B8G8R8A8_UNORM
                            && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                    })
                    .unwrap_or(&formats[0])
            }
        };

        let present_mode = {
            if surface
                .present_modes
                .contains(&vk::PresentModeKHR::IMMEDIATE)
            {
                vk::PresentModeKHR::IMMEDIATE
            } else {
                vk::PresentModeKHR::MAILBOX
            }
        };

        let extent = {
            if let Some(size) = size {
                vk::Extent2D {
                    width: size[0],
                    height: size[1],
                }
            } else if surface.capabilities.current_extent.width != std::u32::MAX {
                surface.capabilities.current_extent
            } else {
                surface.capabilities.min_image_extent
            }
        };

        let image_count = surface.capabilities.min_image_count;

        let families_indices = [graphics_queue as u32, present_queue as u32];

        let create_info = {
            let mut builder = vk::SwapchainCreateInfoKHR::default()
                .surface(surface.handle)
                .min_image_count(image_count)
                .image_format(format.format)
                .image_color_space(format.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(
                    vk::ImageUsageFlags::STORAGE
                        | vk::ImageUsageFlags::TRANSFER_DST
                        | vk::ImageUsageFlags::COLOR_ATTACHMENT,
                );

            builder = if graphics_queue != present_queue {
                builder
                    .image_sharing_mode(vk::SharingMode::CONCURRENT)
                    .queue_family_indices(&families_indices)
            } else {
                builder.image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            };

            if let Some(old) = old {
                builder = builder.old_swapchain(old);
            }

            builder
                .pre_transform(surface.capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED)
                .present_mode(present_mode)
                .clipped(true)
        };

        let handle = unsafe { swapchain_fn.create_swapchain(&create_info, None).unwrap() };
        let images = unsafe { swapchain_fn.get_swapchain_images(handle).unwrap() };

        let images = images
            .into_iter()
            .enumerate()
            .map(|(i, image)| {
                if let Some(debug_utils) = debug_utils {
                    let name = format!("Swapchain Image {}\0", i);
                    let name = CStr::from_bytes_with_nul(name.as_bytes()).unwrap();
                    let name_info = vk::DebugUtilsObjectNameInfoEXT::default()
                        .object_handle(image)
                        .object_name(name);
                    unsafe { debug_utils.set_debug_utils_object_name(&name_info) }.unwrap();
                }
                let create_info = vk::ImageViewCreateInfo::default()
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::R,
                        g: vk::ComponentSwizzle::G,
                        b: vk::ComponentSwizzle::B,
                        a: vk::ComponentSwizzle::A,
                    })
                    .format(format.format)
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_array_layer: 0,
                        layer_count: 1,
                        base_mip_level: 0,
                        level_count: 1,
                    });
                let view = unsafe { device.create_image_view(&create_info, None).unwrap() };
                ImageView {
                    handle: None,
                    image,
                    view,
                    base_mip: 0,
                    num_mips: 1,
                    _marker: PhantomData,
                    _marker2: PhantomData,
                }
            })
            .collect::<Vec<_>>();

        Ok(Self {
            resized: true,
            handle,
            format: format.format,
            color_space: format.color_space,
            present_mode,
            images,
            size: [extent.width, extent.height],
        })
    }
}

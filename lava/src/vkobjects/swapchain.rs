use std::ffi::CStr;

use anyhow::Result;
use ash::{khr, vk::{self}, Device, Instance};

use crate::{state::{Ctx, Functions}, vkobjects::{queue::Queue, surface::Surface}, FRAMES_IN_FLIGHT};

use super::{image::Image};

#[derive(Debug)]
pub struct FrameResources {
    pub image_availible_semaphore: vk::Semaphore,
    pub render_finished_semaphore: vk::Semaphore,
}

#[derive(Debug)]
pub struct Swapchain {
    pub size: [u32; 2],
    pub handle: vk::SwapchainKHR,
    pub format: vk::Format,
    pub color_space: vk::ColorSpaceKHR,
    pub present_mode: vk::PresentModeKHR,
    pub images: Vec<Image>,
    pub frame_resources: [FrameResources; FRAMES_IN_FLIGHT],
}

impl Swapchain {
    pub fn new(surface: &Surface, device: &Device, graphics_queue: u32, present_queue: u32, swapchain_fn: &ash::khr::swapchain::Device, debug_utils: Option<&ash::ext::debug_utils::Device>) -> Result<Self> {
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
            if surface.present_modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
                vk::PresentModeKHR::IMMEDIATE
            } else {
                vk::PresentModeKHR::MAILBOX
            }
        };

        let extent = {
            if surface.capabilities.current_extent.width != std::u32::MAX {
                surface.capabilities.current_extent
            } else {
               surface.capabilities.min_image_extent
            }
        };

        let image_count = surface.capabilities.min_image_count + 1;

        let families_indices = [
            graphics_queue as u32,
            present_queue as u32,
        ];

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

            builder
                .pre_transform(surface.capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
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
                Image {
                    usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                    image,
                    format: format.format,
                    extent,
                    view: Image::view(&device, extent, image, format.format),
                    allocation: None,
                }
            })
            .collect::<Vec<_>>();

        let mut frame_resources: [FrameResources; FRAMES_IN_FLIGHT] =
            unsafe { std::mem::MaybeUninit::uninit().assume_init() };

        for i in 0..FRAMES_IN_FLIGHT {
            let create_info = vk::SemaphoreCreateInfo::default();
            let image_availible_semaphore =
                unsafe { device.create_semaphore(&create_info, None).unwrap() };
            let render_finished_semaphore =
                unsafe { device.create_semaphore(&create_info, None).unwrap() };
            frame_resources[i] = FrameResources {
                image_availible_semaphore,
                render_finished_semaphore,
            }
        }

        Ok(Self {
            handle,
            format: format.format,
            color_space: format.color_space,
            present_mode,
            images,
            frame_resources,
            size: [extent.width, extent.height],
        })
    }
}

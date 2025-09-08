use anyhow::Result;
use ash::{khr, vk, Device, Instance};

use crate::{state::{Ctx, Functions}, FRAMES_IN_FLIGHT};

use super::{image::Image};

pub struct FrameResources {
    pub image_availible_semaphore: vk::Semaphore,
    pub render_finished_semaphore: vk::Semaphore,
}

pub struct Swapchain {
    pub handle: vk::SwapchainKHR,
    pub format: vk::Format,
    pub color_space: vk::ColorSpaceKHR,
    pub present_mode: vk::PresentModeKHR,
    pub images: Vec<Image>,
    pub frame_resources: [FrameResources; FRAMES_IN_FLIGHT],
}

impl Swapchain {
    pub fn new() -> Result<Self> {
        let format = {
            let formats = Ctx::surface().formats;
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
            if Ctx::surface().present_modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
                vk::PresentModeKHR::IMMEDIATE
            } else {
                vk::PresentModeKHR::MAILBOX
            }
        };

        let extent = {
            if Ctx::surface().capabilities.current_extent.width != std::u32::MAX {
                Ctx::surface().capabilities.current_extent
            } else {
                vk::Extent2D { width: Ctx::surface().capabilities.min_image_extent, height: Ctx::surface().capabilities.min_image_extent }
            }
        };

        let image_count = Ctx::surface().capabilities.min_image_count + 1;

        let families_indices = [
            Ctx::queue().family_index,
            Ctx::present_queue().family_index,
        ];

        let create_info = {
            let mut builder = vk::SwapchainCreateInfoKHR::default()
                .surface(Ctx::surface().handle)
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

            builder = if Ctx::queue().family_index != Ctx::present_queue().family_index {
                builder
                    .image_sharing_mode(vk::SharingMode::CONCURRENT)
                    .queue_family_indices(&families_indices)
            } else {
                builder.image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            };

            builder
                .pre_transform(Ctx::surface().current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
        };

        let handle = unsafe { Functions::swapchain().create_swapchain(&create_info, None).unwrap() };
        let images = unsafe { Functions::swapchain().get_swapchain_images(handle).unwrap() };

        let images = images
            .into_iter()
            .enumerate()
            .map(|(i, image)| {
                Functions::set_debug_name(&format!("SwpachainImage{}", i), image);

                Image {
                    usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                    image,
                    format: format.format,
                    extent,
                    view: Image::view(&Ctx::device(), extent, image, format.format),
                    allocation: None,
                }
            })
            .collect::<Vec<_>>();

        let mut frame_resources: [FrameResources; FRAMES_IN_FLIGHT] =
            unsafe { std::mem::MaybeUninit::uninit().assume_init() };

        for i in 0..FRAMES_IN_FLIGHT {
            let create_info = vk::SemaphoreCreateInfo::default();
            let image_availible_semaphore =
                unsafe { Ctx::device().create_semaphore(&create_info, None).unwrap() };
            let render_finished_semaphore =
                unsafe { Ctx::device().create_semaphore(&create_info, None).unwrap() };
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
        })
    }
}

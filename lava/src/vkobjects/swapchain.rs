use std::{ffi::CStr, marker::PhantomData, sync::OnceLock};

use anyhow::Result;
use ash::{
    Device,
    vk::{self},
};

use crate::{
    bindless::{Bindless, BindlessHandle, NULL_HANDLE},
    image::{
        Image, format,
        slice::{AsImage, ImageView},
        usage::{ColorAttachment, ColorAttachmentStorage},
    },
    state::{Ctx, Functions},
    tracy_span,
    vkobjects::{
        queue::{Binary, Semaphore},
        surface::Surface,
    },
};

pub static FORMAT: OnceLock<vk::Format> = OnceLock::new();

#[derive(Debug)]
pub struct Swapchain {
    pub size: [u32; 2],
    pub(crate) handle: vk::SwapchainKHR,
    pub(crate) present_mode: vk::PresentModeKHR,
    pub images: Vec<ImageView<format::Swapchain, ColorAttachmentStorage>>,
}

impl Swapchain {
    pub fn new(old: Option<&Swapchain>, size: Option<[u32; 2]>) -> Result<Self> {
        let format = {
            let formats = &Ctx::surface().formats;
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

        let _ = FORMAT.set(format.format);

        let present_mode = {
            if Ctx::surface()
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
            } else if Ctx::surface().capabilities.current_extent.width != std::u32::MAX {
                Ctx::surface().capabilities.current_extent
            } else {
                Ctx::surface().capabilities.min_image_extent
            }
        };

        let image_count = Ctx::surface().capabilities.min_image_count;

        let families_indices = [Ctx::gfx_queue_index(), Ctx::present_queue_index()];

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

            builder = if Ctx::gfx_queue_index() != Ctx::present_queue_index() {
                builder
                    .image_sharing_mode(vk::SharingMode::CONCURRENT)
                    .queue_family_indices(&families_indices)
            } else {
                builder.image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            };

            if let Some(old) = old {
                builder = builder.old_swapchain(old.handle);
            }

            builder
                .pre_transform(Ctx::surface().capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED)
                .present_mode(present_mode)
                .clipped(true)
        };

        let handle = unsafe {
            Functions::swapchain()
                .create_swapchain(&create_info, None)
                .unwrap()
        };
        let images = unsafe { Functions::swapchain().get_swapchain_images(handle).unwrap() };

        let images = images
            .into_iter()
            .enumerate()
            .map(|(i, image)| {
                if let Some(debug_utils) = Functions::debug_utils() {
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
                let view = unsafe { Ctx::device().create_image_view(&create_info, None).unwrap() };

                let mut image = ImageView {
                    handle: None,
                    image,
                    view,
                    base_mip: 0,
                    num_mips: 1,
                    _marker: PhantomData,
                    _marker2: PhantomData,
                };

                if let Some(old) = old {
                    let handle = old.images[i].handle.unwrap();
                    Bindless::write_image(image, handle);
                } else {
                    let handle = Bindless::push(image);
                    image.handle = handle;
                }
                image
            })
            .collect::<Vec<_>>();

        Ok(Self {
            handle,
            present_mode,
            images,
            size: [extent.width, extent.height],
        })
    }

    pub fn aquire_image(&self, wait_on: &Semaphore<Binary>) -> u32 {
        let (image_index, _suboptimal) = unsafe {
            Functions::swapchain().acquire_next_image(
                self.handle,
                u64::MAX,
                wait_on.handle,
                vk::Fence::null(),
            )
        }
        .unwrap();
        image_index
    }

    pub fn recreate(&mut self, size: [u32; 2]) {
        tracy_span!("Swapchain Recreation");
        log::info!("resized swapchain");
        let swapchain = Swapchain::new(Some(self), Some(size)).unwrap();

        unsafe {
            Ctx::device().device_wait_idle();
            Functions::swapchain().destroy_swapchain(self.handle, None);
        };
        *self = swapchain;
    }
}

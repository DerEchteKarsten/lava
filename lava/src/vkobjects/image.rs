use std::{any::TypeId, marker::PhantomData, ops::Range, sync::{Arc, Mutex}};

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
            Self::FullScreen => UVec2::new(Ctx::window_width(), Ctx::window_height()),
            Self::FractionalFullScreen(dx, dy) => UVec2::new(
                (Ctx::window_width()).div_ceil(dx),
                (Ctx::window_height()).div_ceil(dy),
            ),
            Self::XY(x, y) => UVec2::new(x, y),
        }
    }
}

trait UsageSet: 'static {
    const VK: vk::ImageUsageFlags;
}

pub struct Sampled;
pub struct Storage;
pub struct ColorAttachment;
pub struct DepthAttachment;
pub struct ColorAttachmentSampled;
pub struct DepthAttachmentSampled;
pub struct SampledStorage;

impl UsageSet for Sampled {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::SAMPLED;
}
impl UsageSet for Storage {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::STORAGE;
}
impl UsageSet for ColorAttachment {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::COLOR_ATTACHMENT;
}
impl UsageSet for DepthAttachment {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
}
impl UsageSet for ColorAttachmentSampled {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(vk::ImageUsageFlags::COLOR_ATTACHMENT.as_raw() | vk::ImageUsageFlags::SAMPLED.as_raw());
}
impl UsageSet for DepthAttachmentSampled {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT.as_raw() | vk::ImageUsageFlags::SAMPLED.as_raw());
}
impl UsageSet for SampledStorage {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(vk::ImageUsageFlags::SAMPLED.as_raw() | vk::ImageUsageFlags::STORAGE.as_raw());
}

pub(crate) trait IsSampled: UsageSet {}
pub(crate) trait IsStorage: UsageSet {}
pub(crate) trait IsColorAttachment: UsageSet {}
pub(crate) trait IsDepthAttachment: UsageSet {}


impl IsSampled for Sampled {}
impl IsSampled for ColorAttachmentSampled {}
impl IsSampled for DepthAttachmentSampled {}
impl IsSampled for SampledStorage {}

impl IsStorage for Storage {}
impl IsStorage for SampledStorage {}

impl IsColorAttachment for ColorAttachment {}
impl IsColorAttachment for ColorAttachmentSampled {}

impl IsDepthAttachment for DepthAttachment {}
impl IsDepthAttachment for DepthAttachmentSampled {}

trait TexelType {
    const _CHECK: ();
    type Component: Pod;
    const NUM_COMPONENT: usize;
}

impl<const N: usize> TexelType for [u8; N] {
    const _CHECK: () = assert!(N <= 4);
    const NUM_COMPONENT: usize = N;
    type Component = u8;
} 

impl<const N: usize> TexelType for [u16; N] {
    const _CHECK: () = assert!(N <= 4);
    const NUM_COMPONENT: usize = N;
    type Component = u16;
} 

impl<const N: usize> TexelType for [u32; N] {
    const _CHECK: () = assert!(N <= 4);
    const NUM_COMPONENT: usize = N;
    type Component = u32;
} 

impl<const N: usize> TexelType for [f32; N] {
    const _CHECK: () = assert!(N <= 4);
    const NUM_COMPONENT: usize = N;
    type Component = f32;
} 

trait Encoding {}

struct UNorm;
impl Encoding for UNorm {}
struct SNorm;
impl Encoding for SNorm {}
struct UInt;
impl Encoding for UInt {}
struct SInt;
impl Encoding for SInt {}
struct UScaled;
impl Encoding for UScaled {}
struct SScaled;
impl Encoding for SScaled {}
struct Float;
impl Encoding for Float {}
struct SRgb;
impl Encoding for SRgb {}


struct Format<T: TexelType, E: Encoding> {
    _m1: PhantomData<T>,
    _m2: PhantomData<E>,
}

impl<T: TexelType, E: Encoding> VkFormat for Format<T, E> {
    const FORMAT: vk::Format = const {
        use vk::Format;
        match (TypeId::of::<T::Component>(), T::NUM_COMPONENT, TypeId::of::<E>()) {
                (TypeId::of::<u8>(), 1, TypeId::of::<UNorm>()) => Format::R8_UNORM,
                (TypeId::of::<u8>(), 1, TypeId::of::<SNorm>()) => Format::R8_SNORM,
                (TypeId::of::<u8>(), 1, TypeId::of::<UInt>()) => Format::R8_UINT,
                (TypeId::of::<u8>(), 1, TypeId::of::<SInt>()) => Format::R8_SINT,

                (TypeId::of::<u8>(), 2, TypeId::of::<UNorm>()) => Format::R8G8_UNORM,
                (TypeId::of::<u8>(), 2, TypeId::of::<SNorm>()) => Format::R8G8_SNORM,
                (TypeId::of::<u8>(), 2, TypeId::of::<UInt>()) => Format::R8G8_UINT,
                (TypeId::of::<u8>(), 2, TypeId::of::<SInt>()) => Format::R8G8_SINT,

                (TypeId::of::<u8>(), 3, TypeId::of::<UNorm>()) => Format::R8G8B8_UNORM,
                (TypeId::of::<u8>(), 3, TypeId::of::<SNorm>()) => Format::R8G8B8_SNORM,
                (TypeId::of::<u8>(), 3, TypeId::of::<UInt>()) => Format::R8G8B8_UINT,
                (TypeId::of::<u8>(), 3, TypeId::of::<SInt>()) => Format::R8G8B8_SINT,

                (TypeId::of::<u8>(), 4, TypeId::of::<UNorm>()) => Format::R8G8B8A8_UNORM,
                (TypeId::of::<u8>(), 4, TypeId::of::<SNorm>()) => Format::R8G8B8A8_SNORM,
                (TypeId::of::<u8>(), 4, TypeId::of::<UInt>()) => Format::R8G8B8A8_UINT,
                (TypeId::of::<u8>(), 4, TypeId::of::<SInt>()) => Format::R8G8B8A8_SINT,

                // 16-bit UNORM/SNORM/UINT/SINT/FLOAT
                (TypeId::of::<u16>(), 1, TypeId::of::<UNorm>()) => Format::R16_UNORM,
                (TypeId::of::<u16>(), 1, TypeId::of::<SNorm>()) => Format::R16_SNORM,
                (TypeId::of::<u16>(), 1, TypeId::of::<UInt>()) => Format::R16_UINT,
                (TypeId::of::<u16>(), 1, TypeId::of::<SInt>()) => Format::R16_SINT,
                (TypeId::of::<u16>(), 1, TypeId::of::<Float>()) => Format::R16_SFLOAT,

                (TypeId::of::<u16>(), 2, TypeId::of::<UNorm>()) => Format::R16G16_UNORM,
                (TypeId::of::<u16>(), 2, TypeId::of::<SNorm>()) => Format::R16G16_SNORM,
                (TypeId::of::<u16>(), 2, TypeId::of::<UInt>()) => Format::R16G16_UINT,
                (TypeId::of::<u16>(), 2, TypeId::of::<SInt>()) => Format::R16G16_SINT,
                (TypeId::of::<u16>(), 2, TypeId::of::<Float>()) => Format::R16G16_SFLOAT,

                (TypeId::of::<u16>(), 3, TypeId::of::<UNorm>()) => Format::R16G16B16_UNORM,
                (TypeId::of::<u16>(), 3, TypeId::of::<SNorm>()) => Format::R16G16B16_SNORM,
                (TypeId::of::<u16>(), 3, TypeId::of::<UInt>()) => Format::R16G16B16_UINT,
                (TypeId::of::<u16>(), 3, TypeId::of::<SInt>()) => Format::R16G16B16_SINT,
                (TypeId::of::<u16>(), 3, TypeId::of::<Float>()) => Format::R16G16B16_SFLOAT,

                (TypeId::of::<u16>(), 4, TypeId::of::<UNorm>()) => Format::R16G16B16A16_UNORM,
                (TypeId::of::<u16>(), 4, TypeId::of::<SNorm>()) => Format::R16G16B16A16_SNORM,
                (TypeId::of::<u16>(), 4, TypeId::of::<UInt>()) => Format::R16G16B16A16_UINT,
                (TypeId::of::<u16>(), 4, TypeId::of::<SInt>()) => Format::R16G16B16A16_SINT,
                (TypeId::of::<u16>(), 4, TypeId::of::<Float>()) => Format::R16G16B16A16_SFLOAT,

                // 32-bit floats / ints
                (TypeId::of::<f32>(), 1, TypeId::of::<Float>()) => Format::R32_SFLOAT,
                (TypeId::of::<f32>(), 2, TypeId::of::<Float>()) => Format::R32G32_SFLOAT,
                (TypeId::of::<f32>(), 3, TypeId::of::<Float>()) => Format::R32G32B32_SFLOAT,
                (TypeId::of::<f32>(), 4, TypeId::of::<Float>()) => Format::R32G32B32A32_SFLOAT,

                (TypeId::of::<u32>(), 1, TypeId::of::<UInt>()) => Format::R32_UINT,
                (TypeId::of::<u32>(), 2, TypeId::of::<UInt>()) => Format::R32G32_UINT,
                (TypeId::of::<u32>(), 3, TypeId::of::<UInt>()) => Format::R32G32B32_UINT,
                (TypeId::of::<u32>(), 4, TypeId::of::<UInt>()) => Format::R32G32B32A32_UINT,

                (TypeId::of::<i32>(), 1, TypeId::of::<SInt>()) => Format::R32_SINT,
                (TypeId::of::<i32>(), 2, TypeId::of::<SInt>()) => Format::R32G32_SINT,
                (TypeId::of::<i32>(), 3, TypeId::of::<SInt>()) => Format::R32G32B32_SINT,
                (TypeId::of::<i32>(), 4, TypeId::of::<SInt>()) => Format::R32G32B32A32_SINT,

                // 64-bit floats / ints
                (TypeId::of::<f64>(), 1, TypeId::of::<Float>()) => Format::R64_SFLOAT,
                (TypeId::of::<f64>(), 2, TypeId::of::<Float>()) => Format::R64G64_SFLOAT,
                (TypeId::of::<f64>(), 3, TypeId::of::<Float>()) => Format::R64G64B64_SFLOAT,
                (TypeId::of::<f64>(), 4, TypeId::of::<Float>()) => Format::R64G64B64A64_SFLOAT,

                (TypeId::of::<u64>(), 1, TypeId::of::<UInt>()) => Format::R64_UINT,
                (TypeId::of::<u64>(), 2, TypeId::of::<UInt>()) => Format::R64G64_UINT,
                (TypeId::of::<u64>(), 3, TypeId::of::<UInt>()) => Format::R64G64B64_UINT,
                (TypeId::of::<u64>(), 4, TypeId::of::<UInt>()) => Format::R64G64B64A64_UINT,

                (TypeId::of::<i64>(), 1, TypeId::of::<SInt>()) => Format::R64_SINT,
                (TypeId::of::<i64>(), 2, TypeId::of::<SInt>()) => Format::R64G64_SINT,
                (TypeId::of::<i64>(), 3, TypeId::of::<SInt>()) => Format::R64G64B64_SINT,
                (TypeId::of::<i64>(), 4, TypeId::of::<SInt>()) => Format::R64G64B64A64_SINT,

                (TypeId::of::<f32>(), 1, TypeId::of::<Float>()) => Format::D32_SFLOAT,
                (TypeId::of::<u16>(), 1, TypeId::of::<UNorm>()) => Format::D16_UNORM,
                (TypeId::of::<u32>(), 1, TypeId::of::<UInt>()) => Format::S8_UINT,
                (TypeId::of::<u16>(), 2, TypeId::of::<UNorm>()) => Format::D16_UNORM_S8_UINT,
                (TypeId::of::<u32>(), 2, TypeId::of::<UNorm>()) => Format::D24_UNORM_S8_UINT,
                _ => unreachable!(),
        }
    };
}

trait VkFormat {
    const FORMAT: vk::Format;
}

#[derive(Debug)]
pub struct Image<F: VkFormat, U: UsageSet> {
    pub handle: vk::Image,
    pub allocation: Allocation,
    pub size: ImageSize,
    pub mips: u32,
    pub extend: vk::Extent3D,
    _format: PhantomData<F>,
    _usage: PhantomData<U>
}

pub struct ImageView {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub mips: Range<u32>,
    pub aspect: vk::ImageAspectFlags,
    pub offset: vk::Offset3D,
    pub extend: vk::Extent3D,
}

impl Image {
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

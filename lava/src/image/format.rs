use ash::vk;
use bytemuck::Pod;
use std::{any::TypeId, marker::PhantomData};

pub struct Depth<T: Pod> {
    value: T
}

pub struct Stencil<T: Pod> {
    value: T
}


trait TexelType {
    type Component: Pod;
    const NUM_COMPONENT: usize;
    const ASPECT: vk::ImageAspectFlags;
}

impl<const N: usize, T: Pod> TexelType for [Depth<T>; N] {
    const NUM_COMPONENT: usize = N;
    type Component = T;
    const ASPECT: vk::ImageAspectFlags = vk::ImageAspectFlags::DEPTH;
}

impl<const N: usize, T: Pod> TexelType for [Stencil<T>; N] {
    const NUM_COMPONENT: usize = N;
    type Component = T;
    const ASPECT: vk::ImageAspectFlags = vk::ImageAspectFlags::STENCIL;
}

impl<const N: usize, T: Pod> TexelType for [T; N] {
    const NUM_COMPONENT: usize = N;
    type Component = T;
    const ASPECT: vk::ImageAspectFlags = vk::ImageAspectFlags::COLOR;
}

trait Encoding {}

pub struct UNorm;
impl Encoding for UNorm {}
pub struct SNorm;
impl Encoding for SNorm {}
pub struct UInt;
impl Encoding for UInt {}
pub struct SInt;
impl Encoding for SInt {}
pub struct UScaled;
impl Encoding for UScaled {}
pub struct SScaled;
impl Encoding for SScaled {}
pub struct Float;
impl Encoding for Float {}
pub struct SRgb;
impl Encoding for SRgb {}


pub struct Format<T: TexelType, E: Encoding> {
    _m1: PhantomData<T>,
    _m2: PhantomData<E>,
}

impl<T: TexelType, E: Encoding> VkFormat for Format<T, E> {
    const ASPECTS: vk::ImageAspectFlags = T::ASPECT;
    const FORMAT: vk::Format = const {
        use vk::Format;
        match (
            TypeId::of::<T::Component>(),
            T::NUM_COMPONENT,
            TypeId::of::<E>(),
        ) {
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

            (TypeId::of::<Depth<f32>>(), 1, TypeId::of::<Float>()) => Format::D32_SFLOAT,
            (TypeId::of::<Depth<u16>>(), 1, TypeId::of::<UNorm>()) => Format::D16_UNORM,
            (TypeId::of::<Stencil<u32>>(), 1, TypeId::of::<UInt>()) => Format::S8_UINT,
            // (TypeId::of::<Depth<u16>>(), 2, TypeId::of::<UNorm>()) => Format::D16_UNORM_S8_UINT,
            // (TypeId::of::<Depth<u32>>(), 2, TypeId::of::<UNorm>()) => Format::D24_UNORM_S8_UINT,
            _ => unreachable!(),
        }
    };
    const SWAPCHAIN: bool = false;
}

pub(crate) struct SwapchainFormat;

impl VkFormat for SwapchainFormat {
    const FORMAT: vk::Format = vk::Format::UNDEFINED;
    const ASPECTS: vk::ImageAspectFlags = vk::ImageAspectFlags::COLOR;
    const SWAPCHAIN: bool = true;
}

pub(crate) trait VkFormat {
    const FORMAT: vk::Format;
    const SWAPCHAIN: bool;
    const ASPECTS: vk::ImageAspectFlags;
}

//! Type-level image formats with their Vulkan format, aspects, and clear-value texels
use ash::vk;

use crate::vkobjects::swapchain::FORMAT;

pub trait Format: 'static + Copy + Clone {
    const FORMAT: vk::Format;
    const ASPECTS: vk::ImageAspectFlags;
    /// Bytes per texel in host memory. 0 when there is no fixed per-texel size: block-compressed
    /// formats, and `Undefined`/`Swapchain`, whose format is only known at runtime.
    const TEXEL_SIZE: usize;
    type Texels;

    fn clear_value(value: Self::Texels) -> vk::ClearValue;

    fn format() -> vk::Format {
        if Self::FORMAT == vk::Format::UNDEFINED {
            *FORMAT
                .get()
                .expect("swapchain format has not been initialized")
        } else {
            Self::FORMAT
        }
    }
}

pub trait ColorAspect {}
pub trait DepthAspect {}
pub trait StencilAspect {}

fn pad_float<const N: usize>(value: [f32; N]) -> [f32; 4] {
    let mut result = [0.0; 4];
    result[..N].copy_from_slice(&value);
    result
}

fn pad_uint<const N: usize>(value: [u32; N]) -> [u32; 4] {
    let mut result = [0; 4];
    result[..N].copy_from_slice(&value);
    result
}

fn pad_sint<const N: usize>(value: [i32; N]) -> [i32; 4] {
    let mut result = [0; 4];
    result[..N].copy_from_slice(&value);
    result
}

macro_rules! define_format {
    ($($struct_name:ident => ($vk_format:ident, $size:literal, $aspect:expr, $texel:ty, $n:literal, color_float)),* $(,)?) => {
        $(
            #[derive(Copy, Clone, Debug)]
            pub struct $struct_name;
            impl Format for $struct_name {
                const FORMAT: vk::Format = vk::Format::$vk_format;
                const ASPECTS: vk::ImageAspectFlags = $aspect;
                const TEXEL_SIZE: usize = $size;
                type Texels = [$texel; $n];
                fn clear_value(value: Self::Texels) -> vk::ClearValue {
                    vk::ClearValue { color: vk::ClearColorValue { float32: pad_float(value) } }
                }
            }
            impl ColorAspect for $struct_name {}
        )*
    };
    ($($struct_name:ident => ($vk_format:ident, $size:literal, $aspect:expr, $texel:ty, $n:literal, color_uint)),* $(,)?) => {
        $(
            #[derive(Copy, Clone, Debug)]
            pub struct $struct_name;
            impl Format for $struct_name {
                const FORMAT: vk::Format = vk::Format::$vk_format;
                const ASPECTS: vk::ImageAspectFlags = $aspect;
                const TEXEL_SIZE: usize = $size;
                type Texels = [$texel; $n];
                fn clear_value(value: Self::Texels) -> vk::ClearValue {
                    vk::ClearValue { color: vk::ClearColorValue { uint32: pad_uint(value) } }
                }
            }
            impl ColorAspect for $struct_name {}
        )*
    };
    ($($struct_name:ident => ($vk_format:ident, $size:literal, $aspect:expr, $texel:ty, $n:literal, color_sint)),* $(,)?) => {
        $(
            #[derive(Copy, Clone, Debug)]
            pub struct $struct_name;
            impl Format for $struct_name {
                const FORMAT: vk::Format = vk::Format::$vk_format;
                const ASPECTS: vk::ImageAspectFlags = $aspect;
                const TEXEL_SIZE: usize = $size;
                type Texels = [$texel; $n];
                fn clear_value(value: Self::Texels) -> vk::ClearValue {
                    vk::ClearValue { color: vk::ClearColorValue { int32: pad_sint(value) } }
                }
            }
            impl ColorAspect for $struct_name {}
        )*
    };
    ($($struct_name:ident => ($vk_format:ident, $size:literal, $aspect:expr, $texel:ty, depth)),* $(,)?) => {
        $(
            #[derive(Copy, Clone, Debug)]
            pub struct $struct_name;
            impl Format for $struct_name {
                const FORMAT: vk::Format = vk::Format::$vk_format;
                const ASPECTS: vk::ImageAspectFlags = $aspect;
                const TEXEL_SIZE: usize = $size;
                type Texels = [$texel; 1];
                fn clear_value(value: Self::Texels) -> vk::ClearValue {
                    vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: value[0], stencil: 0 } }
                }
            }
            impl DepthAspect for $struct_name {}
        )*
    };
    ($($struct_name:ident => ($vk_format:ident, $size:literal, $aspect:expr, $texel:ty, stencil)),* $(,)?) => {
        $(
            #[derive(Copy, Clone, Debug)]
            pub struct $struct_name;
            impl Format for $struct_name {
                const FORMAT: vk::Format = vk::Format::$vk_format;
                const ASPECTS: vk::ImageAspectFlags = $aspect;
                const TEXEL_SIZE: usize = $size;
                type Texels = [u8; 1];
                fn clear_value(value: Self::Texels) -> vk::ClearValue {
                    vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: value[0] as u32 } }
                }
            }
            impl StencilAspect for $struct_name {}
        )*
    };
    ($($struct_name:ident => ($vk_format:ident, $size:literal, $aspect:expr, $texel:ty, depth_stencil)),* $(,)?) => {
        $(
            #[derive(Copy, Clone, Debug)]
            pub struct $struct_name;
            impl Format for $struct_name {
                const FORMAT: vk::Format = vk::Format::$vk_format;
                const ASPECTS: vk::ImageAspectFlags = $aspect;
                const TEXEL_SIZE: usize = $size;
                type Texels = ($texel, u8);
                fn clear_value(value: Self::Texels) -> vk::ClearValue {
                    vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: value.0, stencil: value.1 as u32 } }
                }
            }
            impl DepthAspect for $struct_name {}
            impl StencilAspect for $struct_name {}
        )*
    };
}
define_format!(
Undefined => (UNDEFINED, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
Swapchain => (UNDEFINED, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
R4G4UnormPack8 => (R4G4_UNORM_PACK8, 1, vk::ImageAspectFlags::COLOR, f32, 2, color_float),
R4G4B4A4UnormPack16 => (R4G4B4A4_UNORM_PACK16, 2, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
B4G4R4A4UnormPack16 => (B4G4R4A4_UNORM_PACK16, 2, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
R5G6B5UnormPack16 => (R5G6B5_UNORM_PACK16, 2, vk::ImageAspectFlags::COLOR, f32, 3, color_float),
B5G6R5UnormPack16 => (B5G6R5_UNORM_PACK16, 2, vk::ImageAspectFlags::COLOR, f32, 3, color_float),
R5G5B5A1UnormPack16 => (R5G5B5A1_UNORM_PACK16, 2, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
B5G5R5A1UnormPack16 => (B5G5R5A1_UNORM_PACK16, 2, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
A1R5G5B5UnormPack16 => (A1R5G5B5_UNORM_PACK16, 2, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
R8Unorm => (R8_UNORM, 1, vk::ImageAspectFlags::COLOR, f32, 1, color_float),
R8Snorm => (R8_SNORM, 1, vk::ImageAspectFlags::COLOR, f32, 1, color_float),
R8Uscaled => (R8_USCALED, 1, vk::ImageAspectFlags::COLOR, f32, 1, color_float),
R8Sscaled => (R8_SSCALED, 1, vk::ImageAspectFlags::COLOR, f32, 1, color_float),
R8Srgb => (R8_SRGB, 1, vk::ImageAspectFlags::COLOR, f32, 1, color_float),
R8G8Unorm => (R8G8_UNORM, 2, vk::ImageAspectFlags::COLOR, f32, 2, color_float),
R8G8Snorm => (R8G8_SNORM, 2, vk::ImageAspectFlags::COLOR, f32, 2, color_float),
R8G8Uscaled => (R8G8_USCALED, 2, vk::ImageAspectFlags::COLOR, f32, 2, color_float),
R8G8Sscaled => (R8G8_SSCALED, 2, vk::ImageAspectFlags::COLOR, f32, 2, color_float),
R8G8Srgb => (R8G8_SRGB, 2, vk::ImageAspectFlags::COLOR, f32, 2, color_float),
R8G8B8Unorm => (R8G8B8_UNORM, 3, vk::ImageAspectFlags::COLOR, f32, 3, color_float),
R8G8B8Snorm => (R8G8B8_SNORM, 3, vk::ImageAspectFlags::COLOR, f32, 3, color_float),
R8G8B8Uscaled => (R8G8B8_USCALED, 3, vk::ImageAspectFlags::COLOR, f32, 3, color_float),
R8G8B8Sscaled => (R8G8B8_SSCALED, 3, vk::ImageAspectFlags::COLOR, f32, 3, color_float),
R8G8B8Srgb => (R8G8B8_SRGB, 3, vk::ImageAspectFlags::COLOR, f32, 3, color_float),
B8G8R8Unorm => (B8G8R8_UNORM, 3, vk::ImageAspectFlags::COLOR, f32, 3, color_float),
B8G8R8Snorm => (B8G8R8_SNORM, 3, vk::ImageAspectFlags::COLOR, f32, 3, color_float),
B8G8R8Uscaled => (B8G8R8_USCALED, 3, vk::ImageAspectFlags::COLOR, f32, 3, color_float),
B8G8R8Sscaled => (B8G8R8_SSCALED, 3, vk::ImageAspectFlags::COLOR, f32, 3, color_float),
B8G8R8Srgb => (B8G8R8_SRGB, 3, vk::ImageAspectFlags::COLOR, f32, 3, color_float),
R8G8B8A8Unorm => (R8G8B8A8_UNORM, 4, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
R8G8B8A8Snorm => (R8G8B8A8_SNORM, 4, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
R8G8B8A8Uscaled => (R8G8B8A8_USCALED, 4, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
R8G8B8A8Sscaled => (R8G8B8A8_SSCALED, 4, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
R8G8B8A8Srgb => (R8G8B8A8_SRGB, 4, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
B8G8R8A8Unorm => (B8G8R8A8_UNORM, 4, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
B8G8R8A8Snorm => (B8G8R8A8_SNORM, 4, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
B8G8R8A8Uscaled => (B8G8R8A8_USCALED, 4, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
B8G8R8A8Sscaled => (B8G8R8A8_SSCALED, 4, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
B8G8R8A8Srgb => (B8G8R8A8_SRGB, 4, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
A8B8G8R8UnormPack32 => (A8B8G8R8_UNORM_PACK32, 4, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
A8B8G8R8SnormPack32 => (A8B8G8R8_SNORM_PACK32, 4, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
A8B8G8R8UscaledPack32 => (A8B8G8R8_USCALED_PACK32, 4, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
A8B8G8R8SscaledPack32 => (A8B8G8R8_SSCALED_PACK32, 4, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
A8B8G8R8SrgbPack32 => (A8B8G8R8_SRGB_PACK32, 4, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
A2R10G10B10UnormPack32 => (A2R10G10B10_UNORM_PACK32, 4, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
A2R10G10B10SnormPack32 => (A2R10G10B10_SNORM_PACK32, 4, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
A2R10G10B10UscaledPack32 => (A2R10G10B10_USCALED_PACK32, 4, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
A2R10G10B10SscaledPack32 => (A2R10G10B10_SSCALED_PACK32, 4, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
A2B10G10R10UnormPack32 => (A2B10G10R10_UNORM_PACK32, 4, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
A2B10G10R10SnormPack32 => (A2B10G10R10_SNORM_PACK32, 4, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
A2B10G10R10UscaledPack32 => (A2B10G10R10_USCALED_PACK32, 4, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
A2B10G10R10SscaledPack32 => (A2B10G10R10_SSCALED_PACK32, 4, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
R16Unorm => (R16_UNORM, 2, vk::ImageAspectFlags::COLOR, f32, 1, color_float),
R16Snorm => (R16_SNORM, 2, vk::ImageAspectFlags::COLOR, f32, 1, color_float),
R16Uscaled => (R16_USCALED, 2, vk::ImageAspectFlags::COLOR, f32, 1, color_float),
R16Sscaled => (R16_SSCALED, 2, vk::ImageAspectFlags::COLOR, f32, 1, color_float),
R16Sfloat => (R16_SFLOAT, 2, vk::ImageAspectFlags::COLOR, f32, 1, color_float),
R16G16Unorm => (R16G16_UNORM, 4, vk::ImageAspectFlags::COLOR, f32, 2, color_float),
R16G16Snorm => (R16G16_SNORM, 4, vk::ImageAspectFlags::COLOR, f32, 2, color_float),
R16G16Uscaled => (R16G16_USCALED, 4, vk::ImageAspectFlags::COLOR, f32, 2, color_float),
R16G16Sscaled => (R16G16_SSCALED, 4, vk::ImageAspectFlags::COLOR, f32, 2, color_float),
R16G16Sfloat => (R16G16_SFLOAT, 4, vk::ImageAspectFlags::COLOR, f32, 2, color_float),
R16G16B16Unorm => (R16G16B16_UNORM, 6, vk::ImageAspectFlags::COLOR, f32, 3, color_float),
R16G16B16Snorm => (R16G16B16_SNORM, 6, vk::ImageAspectFlags::COLOR, f32, 3, color_float),
R16G16B16Uscaled => (R16G16B16_USCALED, 6, vk::ImageAspectFlags::COLOR, f32, 3, color_float),
R16G16B16Sscaled => (R16G16B16_SSCALED, 6, vk::ImageAspectFlags::COLOR, f32, 3, color_float),
R16G16B16Sfloat => (R16G16B16_SFLOAT, 6, vk::ImageAspectFlags::COLOR, f32, 3, color_float),
R16G16B16A16Unorm => (R16G16B16A16_UNORM, 8, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
R16G16B16A16Snorm => (R16G16B16A16_SNORM, 8, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
R16G16B16A16Uscaled => (R16G16B16A16_USCALED, 8, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
R16G16B16A16Sscaled => (R16G16B16A16_SSCALED, 8, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
R16G16B16A16Sfloat => (R16G16B16A16_SFLOAT, 8, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
R32Sfloat => (R32_SFLOAT, 4, vk::ImageAspectFlags::COLOR, f32, 1, color_float),
R32G32Sfloat => (R32G32_SFLOAT, 8, vk::ImageAspectFlags::COLOR, f32, 2, color_float),
R32G32B32Sfloat => (R32G32B32_SFLOAT, 12, vk::ImageAspectFlags::COLOR, f32, 3, color_float),
R32G32B32A32Sfloat => (R32G32B32A32_SFLOAT, 16, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
R64Sfloat => (R64_SFLOAT, 8, vk::ImageAspectFlags::COLOR, f32, 1, color_float),
R64G64Sfloat => (R64G64_SFLOAT, 16, vk::ImageAspectFlags::COLOR, f32, 2, color_float),
R64G64B64Sfloat => (R64G64B64_SFLOAT, 24, vk::ImageAspectFlags::COLOR, f32, 3, color_float),
R64G64B64A64Sfloat => (R64G64B64A64_SFLOAT, 32, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
B10G11R11UfloatPack32 => (B10G11R11_UFLOAT_PACK32, 4, vk::ImageAspectFlags::COLOR, f32, 3, color_float),
E5B9G9R9UfloatPack32 => (E5B9G9R9_UFLOAT_PACK32, 4, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
BC1RgbUnormBlock => (BC1_RGB_UNORM_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 3, color_float),
BC1RgbSrgbBlock => (BC1_RGB_SRGB_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 3, color_float),
BC1RgbaUnormBlock => (BC1_RGBA_UNORM_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
BC1RgbaSrgbBlock => (BC1_RGBA_SRGB_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
BC2UnormBlock => (BC2_UNORM_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
BC2SrgbBlock => (BC2_SRGB_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
BC3UnormBlock => (BC3_UNORM_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
BC3SrgbBlock => (BC3_SRGB_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
BC4UnormBlock => (BC4_UNORM_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 1, color_float),
BC4SnormBlock => (BC4_SNORM_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 1, color_float),
BC5UnormBlock => (BC5_UNORM_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 2, color_float),
BC5SnormBlock => (BC5_SNORM_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 2, color_float),
BC6HUfloatBlock => (BC6H_UFLOAT_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 3, color_float),
BC6HSfloatBlock => (BC6H_SFLOAT_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 3, color_float),
BC7UnormBlock => (BC7_UNORM_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
BC7SrgbBlock => (BC7_SRGB_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
ETC2R8G8B8UnormBlock => (ETC2_R8G8B8_UNORM_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 3, color_float),
ETC2R8G8B8SrgbBlock => (ETC2_R8G8B8_SRGB_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 3, color_float),
ETC2R8G8B8A1UnormBlock => (ETC2_R8G8B8A1_UNORM_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
ETC2R8G8B8A1SrgbBlock => (ETC2_R8G8B8A1_SRGB_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
ETC2R8G8B8A8UnormBlock => (ETC2_R8G8B8A8_UNORM_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
ETC2R8G8B8A8SrgbBlock => (ETC2_R8G8B8A8_SRGB_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
EACR11UnormBlock => (EAC_R11_UNORM_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 1, color_float),
EACR11SnormBlock => (EAC_R11_SNORM_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 1, color_float),
EACR11G11UnormBlock => (EAC_R11G11_UNORM_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 2, color_float),
EACR11G11SnormBlock => (EAC_R11G11_SNORM_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 2, color_float),
ASTC4X4UnormBlock => (ASTC_4X4_UNORM_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
ASTC4X4SrgbBlock => (ASTC_4X4_SRGB_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
ASTC5X4UnormBlock => (ASTC_5X4_UNORM_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
ASTC5X4SrgbBlock => (ASTC_5X4_SRGB_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
ASTC5X5UnormBlock => (ASTC_5X5_UNORM_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
ASTC5X5SrgbBlock => (ASTC_5X5_SRGB_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
ASTC6X5UnormBlock => (ASTC_6X5_UNORM_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
ASTC6X5SrgbBlock => (ASTC_6X5_SRGB_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
ASTC6X6UnormBlock => (ASTC_6X6_UNORM_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
ASTC6X6SrgbBlock => (ASTC_6X6_SRGB_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
ASTC8X5UnormBlock => (ASTC_8X5_UNORM_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
ASTC8X5SrgbBlock => (ASTC_8X5_SRGB_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
ASTC8X6UnormBlock => (ASTC_8X6_UNORM_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
ASTC8X6SrgbBlock => (ASTC_8X6_SRGB_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
ASTC8X8UnormBlock => (ASTC_8X8_UNORM_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
ASTC8X8SrgbBlock => (ASTC_8X8_SRGB_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
ASTC10X5UnormBlock => (ASTC_10X5_UNORM_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
ASTC10X5SrgbBlock => (ASTC_10X5_SRGB_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
ASTC10X6UnormBlock => (ASTC_10X6_UNORM_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
ASTC10X6SrgbBlock => (ASTC_10X6_SRGB_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
ASTC10X8UnormBlock => (ASTC_10X8_UNORM_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
ASTC10X8SrgbBlock => (ASTC_10X8_SRGB_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
ASTC10X10UnormBlock => (ASTC_10X10_UNORM_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
ASTC10X10SrgbBlock => (ASTC_10X10_SRGB_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
ASTC12X10UnormBlock => (ASTC_12X10_UNORM_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
ASTC12X10SrgbBlock => (ASTC_12X10_SRGB_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
ASTC12X12UnormBlock => (ASTC_12X12_UNORM_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
ASTC12X12SrgbBlock => (ASTC_12X12_SRGB_BLOCK, 0, vk::ImageAspectFlags::COLOR, f32, 4, color_float),
);

define_format!(
R8Uint => (R8_UINT, 1, vk::ImageAspectFlags::COLOR, u32, 1, color_uint),
R8G8Uint => (R8G8_UINT, 2, vk::ImageAspectFlags::COLOR, u32, 2, color_uint),
R8G8B8Uint => (R8G8B8_UINT, 3, vk::ImageAspectFlags::COLOR, u32, 3, color_uint),
B8G8R8Uint => (B8G8R8_UINT, 3, vk::ImageAspectFlags::COLOR, u32, 3, color_uint),
R8G8B8A8Uint => (R8G8B8A8_UINT, 4, vk::ImageAspectFlags::COLOR, u32, 4, color_uint),
B8G8R8A8Uint => (B8G8R8A8_UINT, 4, vk::ImageAspectFlags::COLOR, u32, 4, color_uint),
A8B8G8R8UintPack32 => (A8B8G8R8_UINT_PACK32, 4, vk::ImageAspectFlags::COLOR, u32, 4, color_uint),
A2R10G10B10UintPack32 => (A2R10G10B10_UINT_PACK32, 4, vk::ImageAspectFlags::COLOR, u32, 4, color_uint),
A2B10G10R10UintPack32 => (A2B10G10R10_UINT_PACK32, 4, vk::ImageAspectFlags::COLOR, u32, 4, color_uint),
R16Uint => (R16_UINT, 2, vk::ImageAspectFlags::COLOR, u32, 1, color_uint),
R16G16Uint => (R16G16_UINT, 4, vk::ImageAspectFlags::COLOR, u32, 2, color_uint),
R16G16B16Uint => (R16G16B16_UINT, 6, vk::ImageAspectFlags::COLOR, u32, 3, color_uint),
R16G16B16A16Uint => (R16G16B16A16_UINT, 8, vk::ImageAspectFlags::COLOR, u32, 4, color_uint),
R32Uint => (R32_UINT, 4, vk::ImageAspectFlags::COLOR, u32, 1, color_uint),
R32G32Uint => (R32G32_UINT, 8, vk::ImageAspectFlags::COLOR, u32, 2, color_uint),
R32G32B32Uint => (R32G32B32_UINT, 12, vk::ImageAspectFlags::COLOR, u32, 3, color_uint),
R32G32B32A32Uint => (R32G32B32A32_UINT, 16, vk::ImageAspectFlags::COLOR, u32, 4, color_uint),
R64Uint => (R64_UINT, 8, vk::ImageAspectFlags::COLOR, u32, 1, color_uint),
R64G64Uint => (R64G64_UINT, 16, vk::ImageAspectFlags::COLOR, u32, 2, color_uint),
R64G64B64Uint => (R64G64B64_UINT, 24, vk::ImageAspectFlags::COLOR, u32, 3, color_uint),
R64G64B64A64Uint => (R64G64B64A64_UINT, 32, vk::ImageAspectFlags::COLOR, u32, 4, color_uint),
);

define_format!(
R8Sint => (R8_SINT, 1, vk::ImageAspectFlags::COLOR, i32, 1, color_sint),
R8G8Sint => (R8G8_SINT, 2, vk::ImageAspectFlags::COLOR, i32, 2, color_sint),
R8G8B8Sint => (R8G8B8_SINT, 3, vk::ImageAspectFlags::COLOR, i32, 3, color_sint),
B8G8R8Sint => (B8G8R8_SINT, 3, vk::ImageAspectFlags::COLOR, i32, 3, color_sint),
R8G8B8A8Sint => (R8G8B8A8_SINT, 4, vk::ImageAspectFlags::COLOR, i32, 4, color_sint),
B8G8R8A8Sint => (B8G8R8A8_SINT, 4, vk::ImageAspectFlags::COLOR, i32, 4, color_sint),
A8B8G8R8SintPack32 => (A8B8G8R8_SINT_PACK32, 4, vk::ImageAspectFlags::COLOR, i32, 4, color_sint),
A2R10G10B10SintPack32 => (A2R10G10B10_SINT_PACK32, 4, vk::ImageAspectFlags::COLOR, i32, 4, color_sint),
A2B10G10R10SintPack32 => (A2B10G10R10_SINT_PACK32, 4, vk::ImageAspectFlags::COLOR, i32, 4, color_sint),
R16Sint => (R16_SINT, 2, vk::ImageAspectFlags::COLOR, i32, 1, color_sint),
R16G16Sint => (R16G16_SINT, 4, vk::ImageAspectFlags::COLOR, i32, 2, color_sint),
R16G16B16Sint => (R16G16B16_SINT, 6, vk::ImageAspectFlags::COLOR, i32, 3, color_sint),
R16G16B16A16Sint => (R16G16B16A16_SINT, 8, vk::ImageAspectFlags::COLOR, i32, 4, color_sint),
R32Sint => (R32_SINT, 4, vk::ImageAspectFlags::COLOR, i32, 1, color_sint),
R32G32Sint => (R32G32_SINT, 8, vk::ImageAspectFlags::COLOR, i32, 2, color_sint),
R32G32B32Sint => (R32G32B32_SINT, 12, vk::ImageAspectFlags::COLOR, i32, 3, color_sint),
R32G32B32A32Sint => (R32G32B32A32_SINT, 16, vk::ImageAspectFlags::COLOR, i32, 4, color_sint),
R64Sint => (R64_SINT, 8, vk::ImageAspectFlags::COLOR, i32, 1, color_sint),
R64G64Sint => (R64G64_SINT, 16, vk::ImageAspectFlags::COLOR, i32, 2, color_sint),
R64G64B64Sint => (R64G64B64_SINT, 24, vk::ImageAspectFlags::COLOR, i32, 3, color_sint),
R64G64B64A64Sint => (R64G64B64A64_SINT, 32, vk::ImageAspectFlags::COLOR, i32, 4, color_sint),
);

define_format!(
D16Unorm => (D16_UNORM, 2, vk::ImageAspectFlags::DEPTH, f32, depth),
X8D24UnormPack32 => (X8_D24_UNORM_PACK32, 4, vk::ImageAspectFlags::DEPTH, f32, depth),
D32Sfloat => (D32_SFLOAT, 4, vk::ImageAspectFlags::DEPTH, f32, depth),
);

define_format!(
S8Uint => (S8_UINT, 1, vk::ImageAspectFlags::STENCIL, u8, stencil),
);

define_format!(
    D16UnormS8Uint => (D16_UNORM_S8_UINT, 3,  vk::ImageAspectFlags::from_raw(vk::ImageAspectFlags::DEPTH.as_raw() | vk::ImageAspectFlags::STENCIL.as_raw()), f32, depth_stencil),
    D24UnormS8Uint => (D24_UNORM_S8_UINT, 4,  vk::ImageAspectFlags::from_raw(vk::ImageAspectFlags::DEPTH.as_raw() | vk::ImageAspectFlags::STENCIL.as_raw()), f32, depth_stencil),
    D32SfloatS8Uint => (D32_SFLOAT_S8_UINT, 5, vk::ImageAspectFlags::from_raw(vk::ImageAspectFlags::DEPTH.as_raw() | vk::ImageAspectFlags::STENCIL.as_raw()), f32, depth_stencil),
);

#[cfg(test)]
mod tests {
    use super::*;
    use vk::ImageAspectFlags as A;

    #[test]
    fn padding_fills_missing_components_with_zero() {
        assert_eq!(pad_float([1.0, 2.0]), [1.0, 2.0, 0.0, 0.0]);
        assert_eq!(pad_float([1.0, 2.0, 3.0, 4.0]), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(pad_uint([7]), [7, 0, 0, 0]);
        assert_eq!(pad_sint([-1, -2, -3]), [-1, -2, -3, 0]);
    }

    #[test]
    fn formats_carry_their_vulkan_format_and_aspects() {
        assert_eq!(R8Unorm::FORMAT, vk::Format::R8_UNORM);
        assert_eq!(R8G8B8A8Unorm::FORMAT, vk::Format::R8G8B8A8_UNORM);
        assert_eq!(R32Uint::FORMAT, vk::Format::R32_UINT);
        assert_eq!(R8G8B8A8Unorm::ASPECTS, A::COLOR);
        assert_eq!(D32Sfloat::FORMAT, vk::Format::D32_SFLOAT);
        assert_eq!(D32Sfloat::ASPECTS, A::DEPTH);
        assert_eq!(S8Uint::ASPECTS, A::STENCIL);
        assert_eq!(D24UnormS8Uint::FORMAT, vk::Format::D24_UNORM_S8_UINT);
        assert_eq!(D16UnormS8Uint::ASPECTS, A::DEPTH | A::STENCIL);
        assert_eq!(D32SfloatS8Uint::ASPECTS, A::DEPTH | A::STENCIL);
    }

    #[test]
    fn texel_size_is_the_host_size_of_one_texel() {
        assert_eq!(R8Unorm::TEXEL_SIZE, 1);
        assert_eq!(R8G8B8A8Unorm::TEXEL_SIZE, 4);
        assert_eq!(R16G16Sfloat::TEXEL_SIZE, 4);
        assert_eq!(R32Uint::TEXEL_SIZE, 4);
        assert_eq!(R16G16B16A16Sfloat::TEXEL_SIZE, 8);
        assert_eq!(R32G32B32A32Sfloat::TEXEL_SIZE, 16);
        assert_eq!(R64G64B64A64Sfloat::TEXEL_SIZE, 32);
        // Packed formats take the size of their pack word.
        assert_eq!(R4G4UnormPack8::TEXEL_SIZE, 1);
        assert_eq!(R5G6B5UnormPack16::TEXEL_SIZE, 2);
        assert_eq!(A2B10G10R10UnormPack32::TEXEL_SIZE, 4);
        assert_eq!(B10G11R11UfloatPack32::TEXEL_SIZE, 4);
        assert_eq!(D16Unorm::TEXEL_SIZE, 2);
        assert_eq!(X8D24UnormPack32::TEXEL_SIZE, 4);
        assert_eq!(D32Sfloat::TEXEL_SIZE, 4);
        assert_eq!(S8Uint::TEXEL_SIZE, 1);
        assert_eq!(D16UnormS8Uint::TEXEL_SIZE, 3);
        assert_eq!(D24UnormS8Uint::TEXEL_SIZE, 4);
        assert_eq!(D32SfloatS8Uint::TEXEL_SIZE, 5);
    }

    #[test]
    fn formats_without_a_fixed_texel_size_report_zero() {
        assert_eq!(Undefined::TEXEL_SIZE, 0);
        assert_eq!(Swapchain::TEXEL_SIZE, 0);
        assert_eq!(BC1RgbUnormBlock::TEXEL_SIZE, 0);
        assert_eq!(BC7SrgbBlock::TEXEL_SIZE, 0);
        assert_eq!(ASTC4X4UnormBlock::TEXEL_SIZE, 0);
    }

    #[test]
    fn format_returns_the_constant_for_defined_formats() {
        assert_eq!(R8G8B8A8Unorm::format(), vk::Format::R8G8B8A8_UNORM);
        assert_eq!(D32Sfloat::format(), vk::Format::D32_SFLOAT);
    }

    #[test]
    fn color_clear_values_use_the_union_field_of_their_numeric_type() {
        let float = R16G16Sfloat::clear_value([0.25, 0.5]);
        assert_eq!(unsafe { float.color.float32 }, [0.25, 0.5, 0.0, 0.0]);

        let uint = R8G8B8A8Uint::clear_value([1, 2, 3, 4]);
        assert_eq!(unsafe { uint.color.uint32 }, [1, 2, 3, 4]);

        let sint = R32Sint::clear_value([-5]);
        assert_eq!(unsafe { sint.color.int32 }, [-5, 0, 0, 0]);
    }

    #[test]
    fn depth_and_stencil_clear_values() {
        let depth = unsafe { D32Sfloat::clear_value([0.75]).depth_stencil };
        assert_eq!((depth.depth, depth.stencil), (0.75, 0));

        let stencil = unsafe { S8Uint::clear_value([9]).depth_stencil };
        assert_eq!((stencil.depth, stencil.stencil), (0.0, 9));

        let both = unsafe { D24UnormS8Uint::clear_value((0.5, 3)).depth_stencil };
        assert_eq!((both.depth, both.stencil), (0.5, 3));
    }
}

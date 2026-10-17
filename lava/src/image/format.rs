//! Image formats: one marker type per Vulkan format with its aspects and clear-value texels, and `Dynamic`, whose format is only known at runtime
use std::{
    any::{TypeId, type_name},
    mem::{Discriminant, discriminant},
};

use crate::vk;

pub trait Format: crate::sealed::Sealed + 'static + Copy + Send + Sync + std::fmt::Debug {
    /// The texels of a clear value: an array of the format's component type, `(depth, stencil)`
    /// for depth-stencil formats, a `ClearValue` for `Dynamic`.
    type Texels;

    fn vk_format(&self) -> vk::Format;
    fn aspects(&self) -> vk::ImageAspectFlags;
    /// Bytes per texel in host memory. 0 for block-compressed formats, which have no fixed
    /// per-texel size.
    fn texel_size(&self) -> usize;
    fn clear_value(&self, value: Self::Texels) -> ClearValue;

    // The command buffer calls the checks with what its trait bounds require. A static
    // format's type already guarantees that, so its checks do nothing and compile away;
    // `Dynamic` meets every bound and checks at runtime instead.

    /// Panics unless the format has the `required` aspects (`ColorAspect`, `DepthAspect`,
    /// `StencilAspect`).
    fn aspect_check(&self, required: vk::ImageAspectFlags);
    /// Panics unless the format's texels have the type `T` (`HasTexels<T>`).
    fn texel_check<T: 'static>(&self);
}

/// Every format but `Dynamic`: the type alone says what the format is.
pub trait StaticFormat: Format + Default {}

/// Formats whose texels a shader can read as `T` (`[f32; 4]` for a `Tex2D<float4>`).
pub trait HasTexels<T>: Format {}

pub trait ColorAspect: crate::sealed::Sealed {}
pub trait DepthAspect: crate::sealed::Sealed {}
pub trait StencilAspect: crate::sealed::Sealed {}
/// Formats of one aspect, whose images shaders read through the view of every aspect
/// (`kind::ShaderViews`).
pub trait SingleAspect: crate::sealed::Sealed {}
/// Formats with a depth and a stencil aspect, whose images shaders read through a depth-only
/// view (`image::DepthSplit`).
pub trait CombinedDepthStencil: DepthAspect + StencilAspect {}

/// The value an attachment or image is cleared to, padded to 4 components.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClearValue {
    Float([f32; 4]),
    Uint([u32; 4]),
    Sint([i32; 4]),
    DepthStencil { depth: f32, stencil: u32 },
}

impl From<ClearValue> for vk::ClearValue {
    fn from(value: ClearValue) -> Self {
        match value {
            ClearValue::Float(float32) => vk::ClearValue {
                color: vk::ClearColorValue { float32 },
            },
            ClearValue::Uint(uint32) => vk::ClearValue {
                color: vk::ClearColorValue { uint32 },
            },
            ClearValue::Sint(int32) => vk::ClearValue {
                color: vk::ClearColorValue { int32 },
            },
            ClearValue::DepthStencil { depth, stencil } => vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth, stencil },
            },
        }
    }
}

/// A format only known at runtime, e.g. one read from a file or picked by the surface. It
/// meets every format bound, and its checks panic where the bound wouldn't have compiled. It
/// behaves like the static format of the same `vk::Format`, which must exist.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dynamic(pub vk::Format);

impl Dynamic {
    fn info(self) -> Info {
        info(self.0).unwrap_or_else(|| panic!("lava has no format type for {:?}", self.0))
    }
}

impl crate::sealed::Sealed for Dynamic {}
impl Format for Dynamic {
    type Texels = ClearValue;

    fn vk_format(&self) -> vk::Format {
        self.0
    }
    fn aspects(&self) -> vk::ImageAspectFlags {
        self.info().aspects
    }
    fn texel_size(&self) -> usize {
        self.info().texel_size
    }
    /// Panics unless the value is of the kind the format's static type clears with: `Float`,
    /// `Uint` and `Sint` for colour formats of those components, `DepthStencil` for the others.
    fn clear_value(&self, value: ClearValue) -> ClearValue {
        assert!(
            discriminant(&value) == self.info().clear,
            "an image of format {:?} can't be cleared to {value:?}",
            self.0
        );
        value
    }
    fn aspect_check(&self, required: vk::ImageAspectFlags) {
        assert!(
            self.aspects().contains(required),
            "an image of format {:?} lacks the aspects {required:?}",
            self.0
        );
    }
    fn texel_check<T: 'static>(&self) {
        assert!(
            self.info().texels == TypeId::of::<T>(),
            "an image of format {:?} doesn't have texels of type {}",
            self.0,
            type_name::<T>()
        );
    }
}

impl<T> HasTexels<T> for Dynamic {}
impl ColorAspect for Dynamic {}
impl DepthAspect for Dynamic {}
impl StencilAspect for Dynamic {}
// Checked at runtime: `bindless::bind` refuses a view of two aspects, `DepthSplit::create` a
// format without them.
impl SingleAspect for Dynamic {}
impl CombinedDepthStencil for Dynamic {}

/// What `Dynamic` takes from the static format of its `vk::Format`.
struct Info {
    aspects: vk::ImageAspectFlags,
    texel_size: usize,
    /// The `TypeId` of the texels shaders read (`HasTexels`).
    texels: TypeId,
    /// The `ClearValue` variant it clears with.
    clear: Discriminant<ClearValue>,
}

/// Side length in texels and size in bytes of the blocks of a BC format, `None` for every
/// other format.
pub fn bc_block(format: vk::Format) -> Option<(u32, usize)> {
    use vk::Format as F;
    match format {
        F::BC1_RGB_UNORM_BLOCK
        | F::BC1_RGB_SRGB_BLOCK
        | F::BC1_RGBA_UNORM_BLOCK
        | F::BC1_RGBA_SRGB_BLOCK
        | F::BC4_UNORM_BLOCK
        | F::BC4_SNORM_BLOCK => Some((4, 8)),
        F::BC2_UNORM_BLOCK
        | F::BC2_SRGB_BLOCK
        | F::BC3_UNORM_BLOCK
        | F::BC3_SRGB_BLOCK
        | F::BC5_UNORM_BLOCK
        | F::BC5_SNORM_BLOCK
        | F::BC6H_UFLOAT_BLOCK
        | F::BC6H_SFLOAT_BLOCK
        | F::BC7_UNORM_BLOCK
        | F::BC7_SRGB_BLOCK => Some((4, 16)),
        _ => None,
    }
}

fn pad<T: Copy + Default, const N: usize>(value: [T; N]) -> [T; 4] {
    let mut result = [T::default(); 4];
    result[..N].copy_from_slice(&value);
    result
}

/// Defines one format per `Name => (VK_FORMAT, texel size, kind[, components])` line, and
/// `info`, which tells `Dynamic` about them.
macro_rules! formats {
    (@texels float $n:literal) => { [f32; $n] };
    (@texels uint $n:literal) => { [u32; $n] };
    (@texels sint $n:literal) => { [i32; $n] };
    (@texels float64 $n:literal) => { [f64; $n] };
    (@texels uint64 $n:literal) => { [u64; $n] };
    (@texels sint64 $n:literal) => { [i64; $n] };
    (@texels depth) => { [f32; 1] };
    (@texels stencil) => { [u32; 1] };
    (@texels depth_stencil) => { (f32, u32) };

    // What shaders read (`HasTexels`): the texels, except for a combined depth-stencil format,
    // whose views show shaders the depth aspect alone.
    (@shader_texels depth_stencil) => { [f32; 1] };
    (@shader_texels $kind:ident $($n:literal)?) => { formats!(@texels $kind $($n)?) };

    (@aspects depth) => { vk::ImageAspectFlags::DEPTH };
    (@aspects stencil) => { vk::ImageAspectFlags::STENCIL };
    (@aspects depth_stencil) => {
        vk::ImageAspectFlags::from_raw(
            vk::ImageAspectFlags::DEPTH.as_raw() | vk::ImageAspectFlags::STENCIL.as_raw(),
        )
    };
    (@aspects $color:ident) => { vk::ImageAspectFlags::COLOR };

    (@clear float $v:expr) => { ClearValue::Float(pad($v)) };
    (@clear uint $v:expr) => { ClearValue::Uint(pad($v)) };
    (@clear sint $v:expr) => { ClearValue::Sint(pad($v)) };
    // Clear values have 32 bits per component.
    (@clear float64 $v:expr) => { ClearValue::Float(pad($v.map(|c| c as f32))) };
    (@clear uint64 $v:expr) => {
        ClearValue::Uint(pad($v.map(|c| u32::try_from(c).expect("a clear value fits 32 bits"))))
    };
    (@clear sint64 $v:expr) => {
        ClearValue::Sint(pad($v.map(|c| i32::try_from(c).expect("a clear value fits 32 bits"))))
    };
    (@clear depth $v:expr) => { ClearValue::DepthStencil { depth: $v[0], stencil: 0 } };
    (@clear stencil $v:expr) => {
        ClearValue::DepthStencil { depth: 0.0, stencil: $v[0] as u32 }
    };
    (@clear depth_stencil $v:expr) => {
        ClearValue::DepthStencil { depth: $v.0, stencil: $v.1 as u32 }
    };

    (@traits $name:ident depth) => {
        impl DepthAspect for $name {}
        impl SingleAspect for $name {}
    };
    (@traits $name:ident stencil) => {
        impl StencilAspect for $name {}
        impl SingleAspect for $name {}
    };
    (@traits $name:ident depth_stencil) => {
        impl DepthAspect for $name {}
        impl StencilAspect for $name {}
        impl CombinedDepthStencil for $name {}
    };
    (@traits $name:ident $color:ident) => {
        impl ColorAspect for $name {}
        impl SingleAspect for $name {}
    };

    ($info:ident; $($name:ident => ($vk:ident, $size:literal, $kind:ident $(, $n:literal)?),)*) => {
        $(
            #[derive(Clone, Copy, Debug, Default)]
            pub struct $name;
            impl crate::sealed::Sealed for $name {}
            impl StaticFormat for $name {}
            impl Format for $name {
                type Texels = formats!(@texels $kind $($n)?);

                fn vk_format(&self) -> vk::Format {
                    vk::Format::$vk
                }
                fn aspects(&self) -> vk::ImageAspectFlags {
                    formats!(@aspects $kind)
                }
                fn texel_size(&self) -> usize {
                    $size
                }
                fn clear_value(&self, value: Self::Texels) -> ClearValue {
                    formats!(@clear $kind value)
                }
                fn aspect_check(&self, _: vk::ImageAspectFlags) {}
                fn texel_check<T: 'static>(&self) {}
            }
            impl HasTexels<formats!(@shader_texels $kind $($n)?)> for $name {}
            formats!(@traits $name $kind);
        )*

        fn $info(format: vk::Format) -> Option<Info> {
            type Texels<F> = <F as Format>::Texels;
            match format {
                $(vk::Format::$vk => Some(Info {
                    aspects: formats!(@aspects $kind),
                    texel_size: $size,
                    texels: TypeId::of::<formats!(@shader_texels $kind $($n)?)>(),
                    clear: discriminant(&formats!(@clear $kind Texels::<$name>::default())),
                }),)*
                _ => None,
            }
        }
    };
}

formats! {
    plain_info;
    R4G4UnormPack8 => (R4G4_UNORM_PACK8, 1, float, 2),
    R4G4B4A4UnormPack16 => (R4G4B4A4_UNORM_PACK16, 2, float, 4),
    B4G4R4A4UnormPack16 => (B4G4R4A4_UNORM_PACK16, 2, float, 4),
    R5G6B5UnormPack16 => (R5G6B5_UNORM_PACK16, 2, float, 3),
    B5G6R5UnormPack16 => (B5G6R5_UNORM_PACK16, 2, float, 3),
    R5G5B5A1UnormPack16 => (R5G5B5A1_UNORM_PACK16, 2, float, 4),
    B5G5R5A1UnormPack16 => (B5G5R5A1_UNORM_PACK16, 2, float, 4),
    A1R5G5B5UnormPack16 => (A1R5G5B5_UNORM_PACK16, 2, float, 4),
    R8Unorm => (R8_UNORM, 1, float, 1),
    R8Snorm => (R8_SNORM, 1, float, 1),
    R8Uscaled => (R8_USCALED, 1, float, 1),
    R8Sscaled => (R8_SSCALED, 1, float, 1),
    R8Srgb => (R8_SRGB, 1, float, 1),
    R8G8Unorm => (R8G8_UNORM, 2, float, 2),
    R8G8Snorm => (R8G8_SNORM, 2, float, 2),
    R8G8Uscaled => (R8G8_USCALED, 2, float, 2),
    R8G8Sscaled => (R8G8_SSCALED, 2, float, 2),
    R8G8Srgb => (R8G8_SRGB, 2, float, 2),
    R8G8B8Unorm => (R8G8B8_UNORM, 3, float, 3),
    R8G8B8Snorm => (R8G8B8_SNORM, 3, float, 3),
    R8G8B8Uscaled => (R8G8B8_USCALED, 3, float, 3),
    R8G8B8Sscaled => (R8G8B8_SSCALED, 3, float, 3),
    R8G8B8Srgb => (R8G8B8_SRGB, 3, float, 3),
    B8G8R8Unorm => (B8G8R8_UNORM, 3, float, 3),
    B8G8R8Snorm => (B8G8R8_SNORM, 3, float, 3),
    B8G8R8Uscaled => (B8G8R8_USCALED, 3, float, 3),
    B8G8R8Sscaled => (B8G8R8_SSCALED, 3, float, 3),
    B8G8R8Srgb => (B8G8R8_SRGB, 3, float, 3),
    R8G8B8A8Unorm => (R8G8B8A8_UNORM, 4, float, 4),
    R8G8B8A8Snorm => (R8G8B8A8_SNORM, 4, float, 4),
    R8G8B8A8Uscaled => (R8G8B8A8_USCALED, 4, float, 4),
    R8G8B8A8Sscaled => (R8G8B8A8_SSCALED, 4, float, 4),
    R8G8B8A8Srgb => (R8G8B8A8_SRGB, 4, float, 4),
    B8G8R8A8Unorm => (B8G8R8A8_UNORM, 4, float, 4),
    B8G8R8A8Snorm => (B8G8R8A8_SNORM, 4, float, 4),
    B8G8R8A8Uscaled => (B8G8R8A8_USCALED, 4, float, 4),
    B8G8R8A8Sscaled => (B8G8R8A8_SSCALED, 4, float, 4),
    B8G8R8A8Srgb => (B8G8R8A8_SRGB, 4, float, 4),
    A8B8G8R8UnormPack32 => (A8B8G8R8_UNORM_PACK32, 4, float, 4),
    A8B8G8R8SnormPack32 => (A8B8G8R8_SNORM_PACK32, 4, float, 4),
    A8B8G8R8UscaledPack32 => (A8B8G8R8_USCALED_PACK32, 4, float, 4),
    A8B8G8R8SscaledPack32 => (A8B8G8R8_SSCALED_PACK32, 4, float, 4),
    A8B8G8R8SrgbPack32 => (A8B8G8R8_SRGB_PACK32, 4, float, 4),
    A2R10G10B10UnormPack32 => (A2R10G10B10_UNORM_PACK32, 4, float, 4),
    A2R10G10B10SnormPack32 => (A2R10G10B10_SNORM_PACK32, 4, float, 4),
    A2R10G10B10UscaledPack32 => (A2R10G10B10_USCALED_PACK32, 4, float, 4),
    A2R10G10B10SscaledPack32 => (A2R10G10B10_SSCALED_PACK32, 4, float, 4),
    A2B10G10R10UnormPack32 => (A2B10G10R10_UNORM_PACK32, 4, float, 4),
    A2B10G10R10SnormPack32 => (A2B10G10R10_SNORM_PACK32, 4, float, 4),
    A2B10G10R10UscaledPack32 => (A2B10G10R10_USCALED_PACK32, 4, float, 4),
    A2B10G10R10SscaledPack32 => (A2B10G10R10_SSCALED_PACK32, 4, float, 4),
    R16Unorm => (R16_UNORM, 2, float, 1),
    R16Snorm => (R16_SNORM, 2, float, 1),
    R16Uscaled => (R16_USCALED, 2, float, 1),
    R16Sscaled => (R16_SSCALED, 2, float, 1),
    R16Sfloat => (R16_SFLOAT, 2, float, 1),
    R16G16Unorm => (R16G16_UNORM, 4, float, 2),
    R16G16Snorm => (R16G16_SNORM, 4, float, 2),
    R16G16Uscaled => (R16G16_USCALED, 4, float, 2),
    R16G16Sscaled => (R16G16_SSCALED, 4, float, 2),
    R16G16Sfloat => (R16G16_SFLOAT, 4, float, 2),
    R16G16B16Unorm => (R16G16B16_UNORM, 6, float, 3),
    R16G16B16Snorm => (R16G16B16_SNORM, 6, float, 3),
    R16G16B16Uscaled => (R16G16B16_USCALED, 6, float, 3),
    R16G16B16Sscaled => (R16G16B16_SSCALED, 6, float, 3),
    R16G16B16Sfloat => (R16G16B16_SFLOAT, 6, float, 3),
    R16G16B16A16Unorm => (R16G16B16A16_UNORM, 8, float, 4),
    R16G16B16A16Snorm => (R16G16B16A16_SNORM, 8, float, 4),
    R16G16B16A16Uscaled => (R16G16B16A16_USCALED, 8, float, 4),
    R16G16B16A16Sscaled => (R16G16B16A16_SSCALED, 8, float, 4),
    R16G16B16A16Sfloat => (R16G16B16A16_SFLOAT, 8, float, 4),
    R32Sfloat => (R32_SFLOAT, 4, float, 1),
    R32G32Sfloat => (R32G32_SFLOAT, 8, float, 2),
    R32G32B32Sfloat => (R32G32B32_SFLOAT, 12, float, 3),
    R32G32B32A32Sfloat => (R32G32B32A32_SFLOAT, 16, float, 4),
    R64Sfloat => (R64_SFLOAT, 8, float64, 1),
    R64G64Sfloat => (R64G64_SFLOAT, 16, float64, 2),
    R64G64B64Sfloat => (R64G64B64_SFLOAT, 24, float64, 3),
    R64G64B64A64Sfloat => (R64G64B64A64_SFLOAT, 32, float64, 4),
    B10G11R11UfloatPack32 => (B10G11R11_UFLOAT_PACK32, 4, float, 3),
    E5B9G9R9UfloatPack32 => (E5B9G9R9_UFLOAT_PACK32, 4, float, 4),
    ETC2R8G8B8UnormBlock => (ETC2_R8G8B8_UNORM_BLOCK, 0, float, 3),
    ETC2R8G8B8SrgbBlock => (ETC2_R8G8B8_SRGB_BLOCK, 0, float, 3),
    ETC2R8G8B8A1UnormBlock => (ETC2_R8G8B8A1_UNORM_BLOCK, 0, float, 4),
    ETC2R8G8B8A1SrgbBlock => (ETC2_R8G8B8A1_SRGB_BLOCK, 0, float, 4),
    ETC2R8G8B8A8UnormBlock => (ETC2_R8G8B8A8_UNORM_BLOCK, 0, float, 4),
    ETC2R8G8B8A8SrgbBlock => (ETC2_R8G8B8A8_SRGB_BLOCK, 0, float, 4),
    EACR11UnormBlock => (EAC_R11_UNORM_BLOCK, 0, float, 1),
    EACR11SnormBlock => (EAC_R11_SNORM_BLOCK, 0, float, 1),
    EACR11G11UnormBlock => (EAC_R11G11_UNORM_BLOCK, 0, float, 2),
    EACR11G11SnormBlock => (EAC_R11G11_SNORM_BLOCK, 0, float, 2),
    ASTC4X4UnormBlock => (ASTC_4X4_UNORM_BLOCK, 0, float, 4),
    ASTC4X4SrgbBlock => (ASTC_4X4_SRGB_BLOCK, 0, float, 4),
    ASTC5X4UnormBlock => (ASTC_5X4_UNORM_BLOCK, 0, float, 4),
    ASTC5X4SrgbBlock => (ASTC_5X4_SRGB_BLOCK, 0, float, 4),
    ASTC5X5UnormBlock => (ASTC_5X5_UNORM_BLOCK, 0, float, 4),
    ASTC5X5SrgbBlock => (ASTC_5X5_SRGB_BLOCK, 0, float, 4),
    ASTC6X5UnormBlock => (ASTC_6X5_UNORM_BLOCK, 0, float, 4),
    ASTC6X5SrgbBlock => (ASTC_6X5_SRGB_BLOCK, 0, float, 4),
    ASTC6X6UnormBlock => (ASTC_6X6_UNORM_BLOCK, 0, float, 4),
    ASTC6X6SrgbBlock => (ASTC_6X6_SRGB_BLOCK, 0, float, 4),
    ASTC8X5UnormBlock => (ASTC_8X5_UNORM_BLOCK, 0, float, 4),
    ASTC8X5SrgbBlock => (ASTC_8X5_SRGB_BLOCK, 0, float, 4),
    ASTC8X6UnormBlock => (ASTC_8X6_UNORM_BLOCK, 0, float, 4),
    ASTC8X6SrgbBlock => (ASTC_8X6_SRGB_BLOCK, 0, float, 4),
    ASTC8X8UnormBlock => (ASTC_8X8_UNORM_BLOCK, 0, float, 4),
    ASTC8X8SrgbBlock => (ASTC_8X8_SRGB_BLOCK, 0, float, 4),
    ASTC10X5UnormBlock => (ASTC_10X5_UNORM_BLOCK, 0, float, 4),
    ASTC10X5SrgbBlock => (ASTC_10X5_SRGB_BLOCK, 0, float, 4),
    ASTC10X6UnormBlock => (ASTC_10X6_UNORM_BLOCK, 0, float, 4),
    ASTC10X6SrgbBlock => (ASTC_10X6_SRGB_BLOCK, 0, float, 4),
    ASTC10X8UnormBlock => (ASTC_10X8_UNORM_BLOCK, 0, float, 4),
    ASTC10X8SrgbBlock => (ASTC_10X8_SRGB_BLOCK, 0, float, 4),
    ASTC10X10UnormBlock => (ASTC_10X10_UNORM_BLOCK, 0, float, 4),
    ASTC10X10SrgbBlock => (ASTC_10X10_SRGB_BLOCK, 0, float, 4),
    ASTC12X10UnormBlock => (ASTC_12X10_UNORM_BLOCK, 0, float, 4),
    ASTC12X10SrgbBlock => (ASTC_12X10_SRGB_BLOCK, 0, float, 4),
    ASTC12X12UnormBlock => (ASTC_12X12_UNORM_BLOCK, 0, float, 4),
    ASTC12X12SrgbBlock => (ASTC_12X12_SRGB_BLOCK, 0, float, 4),
    R8Uint => (R8_UINT, 1, uint, 1),
    R8G8Uint => (R8G8_UINT, 2, uint, 2),
    R8G8B8Uint => (R8G8B8_UINT, 3, uint, 3),
    B8G8R8Uint => (B8G8R8_UINT, 3, uint, 3),
    R8G8B8A8Uint => (R8G8B8A8_UINT, 4, uint, 4),
    B8G8R8A8Uint => (B8G8R8A8_UINT, 4, uint, 4),
    A8B8G8R8UintPack32 => (A8B8G8R8_UINT_PACK32, 4, uint, 4),
    A2R10G10B10UintPack32 => (A2R10G10B10_UINT_PACK32, 4, uint, 4),
    A2B10G10R10UintPack32 => (A2B10G10R10_UINT_PACK32, 4, uint, 4),
    R16Uint => (R16_UINT, 2, uint, 1),
    R16G16Uint => (R16G16_UINT, 4, uint, 2),
    R16G16B16Uint => (R16G16B16_UINT, 6, uint, 3),
    R16G16B16A16Uint => (R16G16B16A16_UINT, 8, uint, 4),
    R32Uint => (R32_UINT, 4, uint, 1),
    R32G32Uint => (R32G32_UINT, 8, uint, 2),
    R32G32B32Uint => (R32G32B32_UINT, 12, uint, 3),
    R32G32B32A32Uint => (R32G32B32A32_UINT, 16, uint, 4),
    R64Uint => (R64_UINT, 8, uint64, 1),
    R64G64Uint => (R64G64_UINT, 16, uint64, 2),
    R64G64B64Uint => (R64G64B64_UINT, 24, uint64, 3),
    R64G64B64A64Uint => (R64G64B64A64_UINT, 32, uint64, 4),
    R8Sint => (R8_SINT, 1, sint, 1),
    R8G8Sint => (R8G8_SINT, 2, sint, 2),
    R8G8B8Sint => (R8G8B8_SINT, 3, sint, 3),
    B8G8R8Sint => (B8G8R8_SINT, 3, sint, 3),
    R8G8B8A8Sint => (R8G8B8A8_SINT, 4, sint, 4),
    B8G8R8A8Sint => (B8G8R8A8_SINT, 4, sint, 4),
    A8B8G8R8SintPack32 => (A8B8G8R8_SINT_PACK32, 4, sint, 4),
    A2R10G10B10SintPack32 => (A2R10G10B10_SINT_PACK32, 4, sint, 4),
    A2B10G10R10SintPack32 => (A2B10G10R10_SINT_PACK32, 4, sint, 4),
    R16Sint => (R16_SINT, 2, sint, 1),
    R16G16Sint => (R16G16_SINT, 4, sint, 2),
    R16G16B16Sint => (R16G16B16_SINT, 6, sint, 3),
    R16G16B16A16Sint => (R16G16B16A16_SINT, 8, sint, 4),
    R32Sint => (R32_SINT, 4, sint, 1),
    R32G32Sint => (R32G32_SINT, 8, sint, 2),
    R32G32B32Sint => (R32G32B32_SINT, 12, sint, 3),
    R32G32B32A32Sint => (R32G32B32A32_SINT, 16, sint, 4),
    R64Sint => (R64_SINT, 8, sint64, 1),
    R64G64Sint => (R64G64_SINT, 16, sint64, 2),
    R64G64B64Sint => (R64G64B64_SINT, 24, sint64, 3),
    R64G64B64A64Sint => (R64G64B64A64_SINT, 32, sint64, 4),
    D16Unorm => (D16_UNORM, 2, depth),
    X8D24UnormPack32 => (X8_D24_UNORM_PACK32, 4, depth),
    D32Sfloat => (D32_SFLOAT, 4, depth),
    S8Uint => (S8_UINT, 1, stencil),
    D16UnormS8Uint => (D16_UNORM_S8_UINT, 3, depth_stencil),
    D24UnormS8Uint => (D24_UNORM_S8_UINT, 4, depth_stencil),
    D32SfloatS8Uint => (D32_SFLOAT_S8_UINT, 5, depth_stencil),
}

// The block-compressed formats (`bc-compression`).
#[cfg(feature = "bc-compression")]
formats! {
    bc_info;
    BC1RgbUnormBlock => (BC1_RGB_UNORM_BLOCK, 0, float, 3),
    BC1RgbSrgbBlock => (BC1_RGB_SRGB_BLOCK, 0, float, 3),
    BC1RgbaUnormBlock => (BC1_RGBA_UNORM_BLOCK, 0, float, 4),
    BC1RgbaSrgbBlock => (BC1_RGBA_SRGB_BLOCK, 0, float, 4),
    BC2UnormBlock => (BC2_UNORM_BLOCK, 0, float, 4),
    BC2SrgbBlock => (BC2_SRGB_BLOCK, 0, float, 4),
    BC3UnormBlock => (BC3_UNORM_BLOCK, 0, float, 4),
    BC3SrgbBlock => (BC3_SRGB_BLOCK, 0, float, 4),
    BC4UnormBlock => (BC4_UNORM_BLOCK, 0, float, 1),
    BC4SnormBlock => (BC4_SNORM_BLOCK, 0, float, 1),
    BC5UnormBlock => (BC5_UNORM_BLOCK, 0, float, 2),
    BC5SnormBlock => (BC5_SNORM_BLOCK, 0, float, 2),
    BC6HUfloatBlock => (BC6H_UFLOAT_BLOCK, 0, float, 3),
    BC6HSfloatBlock => (BC6H_SFLOAT_BLOCK, 0, float, 3),
    BC7UnormBlock => (BC7_UNORM_BLOCK, 0, float, 4),
    BC7SrgbBlock => (BC7_SRGB_BLOCK, 0, float, 4),
}

/// What lava knows of `format`, `None` for a format it has no marker type for.
fn info(format: vk::Format) -> Option<Info> {
    #[cfg(feature = "bc-compression")]
    if let Some(info) = bc_info(format) {
        return Some(info);
    }
    plain_info(format)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vk::ImageAspectFlags as A;

    #[test]
    fn padding_fills_missing_components_with_zero() {
        assert_eq!(pad([1.0, 2.0]), [1.0, 2.0, 0.0, 0.0]);
        assert_eq!(pad([1.0, 2.0, 3.0, 4.0]), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(pad([7u32]), [7, 0, 0, 0]);
        assert_eq!(pad([-1, -2, -3]), [-1, -2, -3, 0]);
    }

    #[test]
    fn formats_carry_their_vulkan_format_and_aspects() {
        assert_eq!(R8Unorm.vk_format(), vk::Format::R8_UNORM);
        assert_eq!(R8G8B8A8Unorm.vk_format(), vk::Format::R8G8B8A8_UNORM);
        assert_eq!(R32Uint.vk_format(), vk::Format::R32_UINT);
        assert_eq!(R8G8B8A8Unorm.aspects(), A::COLOR);
        assert_eq!(D32Sfloat.vk_format(), vk::Format::D32_SFLOAT);
        assert_eq!(D32Sfloat.aspects(), A::DEPTH);
        assert_eq!(S8Uint.aspects(), A::STENCIL);
        assert_eq!(D24UnormS8Uint.vk_format(), vk::Format::D24_UNORM_S8_UINT);
        assert_eq!(D16UnormS8Uint.aspects(), A::DEPTH | A::STENCIL);
        assert_eq!(D32SfloatS8Uint.aspects(), A::DEPTH | A::STENCIL);
    }

    #[test]
    fn texel_size_is_the_host_size_of_one_texel() {
        assert_eq!(R8Unorm.texel_size(), 1);
        assert_eq!(R8G8B8A8Unorm.texel_size(), 4);
        assert_eq!(R16G16Sfloat.texel_size(), 4);
        assert_eq!(R32Uint.texel_size(), 4);
        assert_eq!(R16G16B16A16Sfloat.texel_size(), 8);
        assert_eq!(R32G32B32A32Sfloat.texel_size(), 16);
        assert_eq!(R64G64B64A64Sfloat.texel_size(), 32);
        // Packed formats take the size of their pack word.
        assert_eq!(R4G4UnormPack8.texel_size(), 1);
        assert_eq!(R5G6B5UnormPack16.texel_size(), 2);
        assert_eq!(A2B10G10R10UnormPack32.texel_size(), 4);
        assert_eq!(B10G11R11UfloatPack32.texel_size(), 4);
        assert_eq!(D16Unorm.texel_size(), 2);
        assert_eq!(X8D24UnormPack32.texel_size(), 4);
        assert_eq!(D32Sfloat.texel_size(), 4);
        assert_eq!(S8Uint.texel_size(), 1);
        assert_eq!(D16UnormS8Uint.texel_size(), 3);
        assert_eq!(D24UnormS8Uint.texel_size(), 4);
        assert_eq!(D32SfloatS8Uint.texel_size(), 5);
    }

    #[test]
    fn block_compressed_formats_report_zero() {
        #[cfg(feature = "bc-compression")]
        assert_eq!(BC1RgbUnormBlock.texel_size(), 0);
        #[cfg(feature = "bc-compression")]
        assert_eq!(BC7SrgbBlock.texel_size(), 0);
        assert_eq!(ASTC4X4UnormBlock.texel_size(), 0);
    }

    #[test]
    fn color_clear_values_use_the_variant_of_their_numeric_type() {
        assert_eq!(
            R16G16Sfloat.clear_value([0.25, 0.5]),
            ClearValue::Float([0.25, 0.5, 0.0, 0.0])
        );
        assert_eq!(
            R8G8B8A8Uint.clear_value([1, 2, 3, 4]),
            ClearValue::Uint([1, 2, 3, 4])
        );
        assert_eq!(R32Sint.clear_value([-5]), ClearValue::Sint([-5, 0, 0, 0]));
        let vk = vk::ClearValue::from(ClearValue::Uint([1, 2, 3, 4]));
        assert_eq!(unsafe { vk.color.uint32 }, [1, 2, 3, 4]);
    }

    #[test]
    fn depth_and_stencil_clear_values() {
        let depth_stencil = |depth, stencil| ClearValue::DepthStencil { depth, stencil };
        assert_eq!(D32Sfloat.clear_value([0.75]), depth_stencil(0.75, 0));
        assert_eq!(S8Uint.clear_value([9]), depth_stencil(0.0, 9));
        assert_eq!(D24UnormS8Uint.clear_value((0.5, 3)), depth_stencil(0.5, 3));
        let vk = vk::ClearValue::from(depth_stencil(0.5, 3));
        let vk = unsafe { vk.depth_stencil };
        assert_eq!((vk.depth, vk.stencil), (0.5, 3));
    }

    /// A dynamic format must behave like the static one of the same `vk::Format`.
    fn same<F: StaticFormat>(format: F, texels: F::Texels) {
        let dynamic = Dynamic(format.vk_format());
        assert_eq!(dynamic.aspects(), format.aspects());
        assert_eq!(dynamic.texel_size(), format.texel_size());
        dynamic.aspect_check(format.aspects());
        let clear = format.clear_value(texels);
        assert_eq!(dynamic.clear_value(clear), clear);
    }

    #[test]
    fn dynamic_formats_match_their_static_format() {
        same(R8Unorm, [0.5]);
        same(R8G8B8A8Srgb, [0.0; 4]);
        same(R32G32B32A32Uint, [1; 4]);
        same(R16Sint, [-1]);
        #[cfg(feature = "bc-compression")]
        same(BC7SrgbBlock, [0.0; 4]);
        #[cfg(feature = "bc-compression")]
        same(BC5UnormBlock, [0.0; 2]);
        same(D32Sfloat, [0.0]);
        same(S8Uint, [1]);
        same(D24UnormS8Uint, (1.0, 1));
    }

    /// What shaders read: 64-bit components for 64-bit formats, `uint` stencil, and the depth of
    /// a combined depth-stencil format (its views show shaders that aspect).
    fn reads<T: 'static, F: HasTexels<T>>(format: F) {
        Dynamic(format.vk_format()).texel_check::<T>();
    }

    #[test]
    fn shaders_read_texels_of_the_formats_component_type() {
        reads::<[f32; 4], _>(R8G8B8A8Unorm);
        reads::<[u32; 1], _>(R32Uint);
        reads::<[f64; 1], _>(R64Sfloat);
        reads::<[u64; 1], _>(R64Uint);
        reads::<[i64; 2], _>(R64G64Sint);
        reads::<[f32; 1], _>(D32Sfloat);
        reads::<[u32; 1], _>(S8Uint);
        reads::<[f32; 1], _>(D24UnormS8Uint);
    }

    #[test]
    fn clear_values_of_64_bit_formats_have_32_bit_components() {
        assert_eq!(R64Uint.clear_value([7]), ClearValue::Uint([7, 0, 0, 0]));
        assert_eq!(R64G64Sint.clear_value([-2, 3]), ClearValue::Sint([-2, 3, 0, 0]));
        assert_eq!(R64Sfloat.clear_value([0.5]), ClearValue::Float([0.5, 0.0, 0.0, 0.0]));
    }

    #[test]
    #[should_panic(expected = "doesn't have texels of type [f32; 4]")]
    fn dynamic_format_checks_texels() {
        Dynamic(vk::Format::R32G32B32A32_UINT).texel_check::<[f32; 4]>();
    }

    #[test]
    #[should_panic(expected = "lacks the aspects")]
    fn dynamic_format_checks_aspects() {
        Dynamic(vk::Format::R8G8B8A8_UNORM).aspect_check(A::DEPTH);
    }

    #[test]
    #[should_panic(expected = "can't be cleared to Uint")]
    fn dynamic_format_checks_clear_values() {
        Dynamic(vk::Format::R8G8B8A8_UNORM).clear_value(ClearValue::Uint([0; 4]));
    }

    #[test]
    #[should_panic(expected = "no format type")]
    fn dynamic_format_needs_a_known_vulkan_format() {
        Dynamic(vk::Format::UNDEFINED).aspects();
    }
}

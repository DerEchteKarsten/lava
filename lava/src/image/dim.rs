//! Image dimensions: `D2`, `D2Array` (2D images with array layers) and `D3` marker types with their extent and offset vectors, the 2D ones attachments take (`Dim2D`), and whether a box of texels lies inside a mip
use std::{fmt::Debug, ops::Add};

use crate::vk;
use glam::{IVec2, IVec3, UVec2, UVec3};

/// The dimension of an image and its views. A 3D image's mips halve in depth too; a
/// [`D2Array`] image's layers are no dimension of its extent, which is a layer's.
pub trait Dim: crate::sealed::Sealed + Copy + Default + Debug + Send + Sync + 'static {
    /// `UVec2` or `UVec3`: sizes.
    type Extent: Copy + Debug + PartialEq + Add<Output = Self::Extent> + Send + Sync;
    /// `IVec2` or `IVec3`: the offsets of boxes (`ImageSlice`s, host copies, sparse binds),
    /// signed like Vulkan's; a box lies inside its mip, so they are never negative.
    type Offset: Copy + Debug + Default + PartialEq + Add<Output = Self::Offset> + Send + Sync;
    const ZERO: Self::Extent;
    const IMAGE_TYPE: vk::ImageType;
    const VIEW_TYPE: vk::ImageViewType;
    fn extent_3d(extent: Self::Extent) -> vk::Extent3D;
    fn offset_3d(offset: Self::Offset) -> vk::Offset3D;
    /// Halved per level in every dimension, at least 1.
    fn mip_extent(extent: Self::Extent, level: u32) -> Self::Extent;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct D2;
/// 2D images with array layers (`Image::with_layers`), viewed as one array (`Tex2DArray`,
/// layered renderings) or a layer at a time (`ImageView::layer`, a [`D2`] view).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct D2Array;
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct D3;

/// The dimensions renderings draw into: [`D2`], and [`D2Array`] for layered and multiview
/// renderings.
pub trait Dim2D: Dim<Extent = UVec2, Offset = IVec2> {}
impl Dim2D for D2 {}
impl Dim2D for D2Array {}

/// Whether the `extent` box at `offset` lies inside a mip of `mip` texels, its offset not
/// negative (an empty box too).
pub(crate) fn box_inside(offset: vk::Offset3D, extent: vk::Extent3D, mip: vk::Extent3D) -> bool {
    let axes = [
        (offset.x, extent.width, mip.width),
        (offset.y, extent.height, mip.height),
        (offset.z, extent.depth, mip.depth),
    ];
    (axes.iter()).all(|&(o, e, m)| o >= 0 && o as u64 + e as u64 <= m as u64)
}

/// Whether a box of `extent` texels has none.
pub(crate) fn box_is_empty(extent: vk::Extent3D) -> bool {
    extent.width == 0 || extent.height == 0 || extent.depth == 0
}

fn halve(size: u32, level: u32) -> u32 {
    size.checked_shr(level).unwrap_or(0).max(1)
}

impl crate::sealed::Sealed for D2 {}
impl Dim for D2 {
    type Extent = UVec2;
    type Offset = IVec2;
    const ZERO: UVec2 = UVec2::ZERO;
    const IMAGE_TYPE: vk::ImageType = vk::ImageType::TYPE_2D;
    const VIEW_TYPE: vk::ImageViewType = vk::ImageViewType::TYPE_2D;
    fn extent_3d(e: UVec2) -> vk::Extent3D {
        vk::Extent3D {
            width: e.x,
            height: e.y,
            depth: 1,
        }
    }
    fn offset_3d(o: IVec2) -> vk::Offset3D {
        vk::Offset3D {
            x: o.x,
            y: o.y,
            z: 0,
        }
    }
    fn mip_extent(e: UVec2, level: u32) -> UVec2 {
        UVec2::new(halve(e.x, level), halve(e.y, level))
    }
}

impl crate::sealed::Sealed for D2Array {}
impl Dim for D2Array {
    type Extent = UVec2;
    type Offset = IVec2;
    const ZERO: UVec2 = UVec2::ZERO;
    const IMAGE_TYPE: vk::ImageType = vk::ImageType::TYPE_2D;
    const VIEW_TYPE: vk::ImageViewType = vk::ImageViewType::TYPE_2D_ARRAY;
    fn extent_3d(e: UVec2) -> vk::Extent3D {
        D2::extent_3d(e)
    }
    fn offset_3d(o: IVec2) -> vk::Offset3D {
        D2::offset_3d(o)
    }
    fn mip_extent(e: UVec2, level: u32) -> UVec2 {
        D2::mip_extent(e, level)
    }
}

impl crate::sealed::Sealed for D3 {}
impl Dim for D3 {
    type Extent = UVec3;
    type Offset = IVec3;
    const ZERO: UVec3 = UVec3::ZERO;
    const IMAGE_TYPE: vk::ImageType = vk::ImageType::TYPE_3D;
    const VIEW_TYPE: vk::ImageViewType = vk::ImageViewType::TYPE_3D;
    fn extent_3d(e: UVec3) -> vk::Extent3D {
        vk::Extent3D {
            width: e.x,
            height: e.y,
            depth: e.z,
        }
    }
    fn offset_3d(o: IVec3) -> vk::Offset3D {
        vk::Offset3D {
            x: o.x,
            y: o.y,
            z: o.z,
        }
    }
    fn mip_extent(e: UVec3, level: u32) -> UVec3 {
        UVec3::new(halve(e.x, level), halve(e.y, level), halve(e.z, level))
    }
}

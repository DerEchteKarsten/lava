//! Image dimensions: `D2` and `D3` marker types with their extent and offset vectors
use std::{fmt::Debug, ops::Add};

use crate::vk;
use glam::{IVec2, IVec3, UVec2, UVec3};

/// The dimension of an image and its views. A 3D image's mips halve in depth too.
pub trait Dim: crate::sealed::Sealed + Copy + Default + Debug + Send + Sync + 'static {
    /// `UVec2` or `UVec3`: sizes, and the unsigned offsets of host copies.
    type Extent: Copy + Debug + PartialEq + Add<Output = Self::Extent> + Send + Sync;
    /// `IVec2` or `IVec3`: the offsets of `ImageSlice`s, which blits may move outside.
    type Offset: Copy + Debug + Default + PartialEq + Add<Output = Self::Offset> + Send + Sync;
    const ZERO: Self::Extent;
    const IMAGE_TYPE: vk::ImageType;
    const VIEW_TYPE: vk::ImageViewType;
    fn extent_3d(extent: Self::Extent) -> vk::Extent3D;
    fn offset_3d(offset: Self::Offset) -> vk::Offset3D;
    /// The offset of a host copy, which can't be negative.
    fn unsigned_offset_3d(offset: Self::Extent) -> vk::Offset3D;
    /// Halved per level in every dimension, at least 1.
    fn mip_extent(extent: Self::Extent, level: u32) -> Self::Extent;
    /// Whether the `extent` box at `offset` is non-empty and lies inside `mip_extent`.
    fn region_fits(mip_extent: Self::Extent, offset: Self::Extent, extent: Self::Extent) -> bool;
    /// The texels of an `extent` box.
    fn texels(extent: Self::Extent) -> usize {
        let e = Self::extent_3d(extent);
        e.width as usize * e.height as usize * e.depth as usize
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct D2;
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct D3;

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
    fn unsigned_offset_3d(o: UVec2) -> vk::Offset3D {
        Self::offset_3d(o.as_ivec2())
    }
    fn mip_extent(e: UVec2, level: u32) -> UVec2 {
        UVec2::new(halve(e.x, level), halve(e.y, level))
    }
    fn region_fits(mip_extent: UVec2, offset: UVec2, extent: UVec2) -> bool {
        extent.cmpgt(UVec2::ZERO).all()
            && offset.cmple(mip_extent).all()
            && extent.cmple(mip_extent - offset.min(mip_extent)).all()
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
    fn unsigned_offset_3d(o: UVec3) -> vk::Offset3D {
        Self::offset_3d(o.as_ivec3())
    }
    fn mip_extent(e: UVec3, level: u32) -> UVec3 {
        UVec3::new(halve(e.x, level), halve(e.y, level), halve(e.z, level))
    }
    fn region_fits(mip_extent: UVec3, offset: UVec3, extent: UVec3) -> bool {
        extent.cmpgt(UVec3::ZERO).all()
            && offset.cmple(mip_extent).all()
            && extent.cmple(mip_extent - offset.min(mip_extent)).all()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regions_must_lie_inside_the_mip() {
        let mip = UVec2::new(8, 4);
        assert!(D2::region_fits(mip, UVec2::ZERO, mip));
        assert!(D2::region_fits(mip, UVec2::new(6, 3), UVec2::new(2, 1)));
        assert!(!D2::region_fits(mip, UVec2::new(6, 3), UVec2::new(3, 1)));
        assert!(!D2::region_fits(mip, UVec2::new(9, 0), UVec2::ONE));
        assert!(!D2::region_fits(mip, UVec2::ZERO, UVec2::new(0, 4)));
        // Offsets near `u32::MAX` must not overflow.
        assert!(!D2::region_fits(
            mip,
            UVec2::splat(u32::MAX),
            UVec2::splat(u32::MAX)
        ));
        let volume = UVec3::new(8, 4, 2);
        assert!(D3::region_fits(
            volume,
            UVec3::new(0, 0, 1),
            UVec3::new(8, 4, 1)
        ));
        assert!(!D3::region_fits(
            volume,
            UVec3::new(0, 0, 1),
            UVec3::new(8, 4, 2)
        ));
    }

    #[test]
    fn mip_extent_halves_per_level() {
        let extent = UVec2::new(256, 64);
        assert_eq!(D2::mip_extent(extent, 0), UVec2::new(256, 64));
        assert_eq!(D2::mip_extent(extent, 1), UVec2::new(128, 32));
        assert_eq!(D2::mip_extent(extent, 6), UVec2::new(4, 1));
        assert_eq!(
            D3::mip_extent(UVec3::new(64, 32, 8), 4),
            UVec3::new(4, 2, 1)
        );
    }

    #[test]
    fn mip_extent_rounds_down_and_never_reaches_zero() {
        assert_eq!(D2::mip_extent(UVec2::new(5, 3), 1), UVec2::new(2, 1));
        assert_eq!(D2::mip_extent(UVec2::new(5, 3), 2), UVec2::new(1, 1));
        assert_eq!(D2::mip_extent(UVec2::new(5, 3), 31), UVec2::new(1, 1));
        // Shifting by the full bit width must not overflow.
        assert_eq!(D2::mip_extent(UVec2::new(5, 3), 32), UVec2::new(1, 1));
        assert_eq!(D2::mip_extent(UVec2::new(5, 3), 200), UVec2::new(1, 1));
    }

    #[test]
    fn copy_offsets_and_extents_keep_2d_flat() {
        let offset = D2::unsigned_offset_3d(UVec2::new(3, 4));
        assert_eq!((offset.x, offset.y, offset.z), (3, 4, 0));
        let offset = D3::unsigned_offset_3d(UVec3::new(3, 4, 5));
        assert_eq!((offset.x, offset.y, offset.z), (3, 4, 5));
        assert_eq!(D2::extent_3d(UVec2::new(3, 4)).depth, 1);
        assert_eq!(D3::texels(UVec3::new(2, 3, 4)), 24);
    }
}

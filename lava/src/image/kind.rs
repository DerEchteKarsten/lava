//! Sealed marker traits that commands, bindless binds and generated bindings bound by: what a view can be used as (`SampledView`, `ColorTarget`, `TransferSrcView`, ...; regions are an `ImageSlice` of one) and which images shaders can bind (`SampledImage`, `StorageImage`)
use crate::image::{
    Image, Views,
    dim::{D2, Dim},
    format::{ColorAspect, DepthAspect, Format},
    slice::ImageView,
    usage::{
        ImageUsage, IsColorAttachment, IsDepthAttachment, IsSampled, IsStorage, IsTransferDst,
        IsTransferSrc,
    },
};

/// An [`ImageView`], with its format, usage and dimension as associated types.
pub trait ImageRef: crate::sealed::Sealed + Copy {
    type Format: Format;
    type Usage: ImageUsage;
    type Dim: Dim;
    /// The view, borrowed for any `'a` it outlives. On a concrete `ImageView`, `.view()` is the
    /// inherent getter of its `vk::ImageView`, so call this one as `ImageRef::view(view)`.
    fn view<'a>(self) -> ImageView<'a, Self::Format, Self::Usage, Self::Dim>
    where
        Self: 'a;
}

impl<F: Format, U: ImageUsage, D: Dim> crate::sealed::Sealed for ImageView<'_, F, U, D> {}
impl<F: Format, U: ImageUsage, D: Dim> ImageRef for ImageView<'_, F, U, D> {
    type Format = F;
    type Usage = U;
    type Dim = D;
    fn view<'a>(self) -> ImageView<'a, F, U, D>
    where
        Self: 'a,
    {
        self
    }
}

/// A view a pass samples (`Tex2D`, `ImageUse::sampled_read`).
pub trait SampledView: ImageRef<Usage: IsSampled> {}
impl<F: Format, U: IsSampled, D: Dim> SampledView for ImageView<'_, F, U, D> {}

/// A view a pass loads from or stores to as a storage image (`Image2D`, `MutImage2D`).
pub trait StorageView: ImageRef<Usage: IsStorage> {}
impl<F: Format, U: IsStorage, D: Dim> StorageView for ImageView<'_, F, U, D> {}

/// A 2D colour view a rendering draws to (`Attachment::color`).
pub trait ColorTarget: ImageRef<Format: ColorAspect, Usage: IsColorAttachment, Dim = D2> {}
impl<F: Format + ColorAspect, U: IsColorAttachment> ColorTarget for ImageView<'_, F, U, D2> {}

/// A 2D depth view a rendering tests against (`RasterBuilder::depth_attachment`).
pub trait DepthTarget: ImageRef<Format: DepthAspect, Usage: IsDepthAttachment, Dim = D2> {}
impl<F: Format + DepthAspect, U: IsDepthAttachment> DepthTarget for ImageView<'_, F, U, D2> {}

/// A colour view a blit or a copy to a buffer reads, through an `ImageSlice` of it.
pub trait TransferSrcView: ImageRef<Format: ColorAspect, Usage: IsTransferSrc> {}
impl<F: Format + ColorAspect, U: IsTransferSrc, D: Dim> TransferSrcView
    for ImageView<'_, F, U, D>
{
}

/// A colour view a clear, or through an `ImageSlice` a blit or a copy from a buffer, writes.
pub trait TransferDstView: ImageRef<Format: ColorAspect, Usage: IsTransferDst> {}
impl<F: Format + ColorAspect, U: IsTransferDst, D: Dim> TransferDstView
    for ImageView<'_, F, U, D>
{
}

/// An image that can be bound in the sampled set. Shaders see one aspect: depth for a combined
/// depth-stencil format.
pub trait SampledImage: crate::sealed::Sealed {}
impl<F: Format, U: IsSampled, D: Dim, V: Views> SampledImage for Image<F, U, D, V> {}

/// An image that can be bound in the storage set.
pub trait StorageImage: crate::sealed::Sealed {}
impl<F: Format, U: IsStorage, D: Dim, V: Views> StorageImage for Image<F, U, D, V> {}

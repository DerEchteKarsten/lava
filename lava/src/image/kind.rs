//! Sealed marker traits that commands, bindless binds and generated bindings bound by: what a view can be used as (`SampledView`, `ColorTarget`, `DepthTarget`, `StencilTarget`, `TransferSrcView`, ...; regions are an `ImageSlice` of one) and which images shaders can bind (`SampledImage`, `StorageImage`)
use crate::image::{
    Image, Views,
    dim::{Dim, Dim2D},
    format::{ColorAspect, DepthAspect, Format, StencilAspect},
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
    /// The view, borrowed for any `'a` it outlives (its `vk::ImageView` is
    /// `ImageView::vk_view`).
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

/// A 2D colour view a rendering draws to (`Attachment::color`): one layer, or layers of an
/// array for a layered or multiview rendering.
pub trait ColorTarget: ImageRef<Format: ColorAspect, Usage: IsColorAttachment, Dim: Dim2D> {}
impl<F: Format + ColorAspect, U: IsColorAttachment, D: Dim2D> ColorTarget
    for ImageView<'_, F, U, D>
{
}

/// A 2D depth view a rendering tests against (`RasterBuilder::depth_attachment`); a combined
/// depth-stencil format is its stencil attachment too.
pub trait DepthTarget: ImageRef<Format: DepthAspect, Usage: IsDepthAttachment, Dim: Dim2D> {}
impl<F: Format + DepthAspect, U: IsDepthAttachment, D: Dim2D> DepthTarget
    for ImageView<'_, F, U, D>
{
}

/// A 2D stencil view a rendering tests against (`RasterBuilder::stencil_attachment`): a
/// stencil-only format like `S8Uint`, or a combined one.
pub trait StencilTarget:
    ImageRef<Format: StencilAspect, Usage: IsDepthAttachment, Dim: Dim2D>
{
}
impl<F: Format + StencilAspect, U: IsDepthAttachment, D: Dim2D> StencilTarget
    for ImageView<'_, F, U, D>
{
}

/// A view that copies to buffers and images read through an `ImageSlice` of it, and blits too
/// if its format is a colour one (`Format: ColorAspect`).
pub trait TransferSrcView: ImageRef<Usage: IsTransferSrc> {}
impl<F: Format, U: IsTransferSrc, D: Dim> TransferSrcView for ImageView<'_, F, U, D> {}

/// A view that clears write (`clear_image` a colour one, `clear_depth_stencil` a depth or
/// stencil one), and through an `ImageSlice` copies from buffers and images, and blits into a
/// colour one.
pub trait TransferDstView: ImageRef<Usage: IsTransferDst> {}
impl<F: Format, U: IsTransferDst, D: Dim> TransferDstView for ImageView<'_, F, U, D> {}

/// An image that can be bound with a sampled descriptor. Shaders see one aspect: depth for a combined
/// depth-stencil format.
pub trait SampledImage: crate::sealed::Sealed {}
impl<F: Format, U: IsSampled, D: Dim, V: Views> SampledImage for Image<F, U, D, V> {}

/// An image that can be bound with a storage descriptor.
pub trait StorageImage: crate::sealed::Sealed {}
impl<F: Format, U: IsStorage, D: Dim, V: Views> StorageImage for Image<F, U, D, V> {}

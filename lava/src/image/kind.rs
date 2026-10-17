//! Sealed marker traits that commands, bindless binds and generated bindings bound by: what a view can be used as (`SampledView`, `ColorTarget`, `TransferSrcView`, ...; regions are an `ImageSlice` of one) and which images shaders can bind (`SampledImage`, `StorageImage`, through `ShaderViews`)
use crate::image::{
    DepthSplit, DepthSplitMips, Image, PerMip, Single, Views,
    dim::{D2, Dim},
    format::{ColorAspect, CombinedDepthStencil, DepthAspect, Format, SingleAspect},
    slice::ImageView,
    usage::{
        ImageUsage, IsColorAttachment, IsDepthAttachment, IsSampled, IsStorage, IsTransferDst,
        IsTransferSrc,
    },
};

/// An [`ImageView`], with its format, usage and dimension as associated types. A command that
/// works on a box of texels takes an [`ImageSlice`](crate::image::slice::ImageSlice) of one
/// instead (`ImageSlice<impl TransferSrcView>`), so a view always stands for all its mips and a
/// slice for its box. The traits below state what they require of these types as bounds, which
/// hold wherever the trait does.
pub trait ImageRef: crate::sealed::Sealed + Copy {
    type Format: Format;
    type Usage: ImageUsage;
    type Dim: Dim;
    /// The view, borrowed for any `'a` it outlives. On a concrete `ImageView`, `.view()` is its
    /// inherent getter of the `vk::ImageView`; this one is `ImageRef::view(view)` there.
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

/// A colour view a clear writes (all its mips), or a blit or a copy from a buffer writes
/// through an `ImageSlice` of it.
pub trait TransferDstView: ImageRef<Format: ColorAspect, Usage: IsTransferDst> {}
impl<F: Format + ColorAspect, U: IsTransferDst, D: Dim> TransferDstView
    for ImageView<'_, F, U, D>
{
}

/// Views that give shaders one aspect of format `F`: [`Single`] and [`PerMip`] for a format of
/// one aspect, [`DepthSplit`] and [`DepthSplitMips`] (their depth-only views) for a combined
/// depth-stencil one. `format::Dynamic` has both kinds, and is checked at runtime.
pub trait ShaderViews<F: Format>: Views {}
impl<F: Format + SingleAspect> ShaderViews<F> for Single {}
impl<F: Format + SingleAspect> ShaderViews<F> for PerMip {}
impl<F: Format + CombinedDepthStencil> ShaderViews<F> for DepthSplit {}
impl<F: Format + CombinedDepthStencil> ShaderViews<F> for DepthSplitMips {}

/// An image that can be bound in the sampled set (`Image::bind_sampled`): a sampled usage and
/// views that show shaders one aspect. A combined depth-stencil image needs [`DepthSplit`]
/// views:
/// ```
/// # use lava::image::{D2, DepthSplit, Image, format::D32SfloatS8Uint, usage::DepthAttachmentSampled};
/// fn bind(image: Image<D32SfloatS8Uint, DepthAttachmentSampled, D2, DepthSplit>) {
///     image.bind_sampled(0);
/// }
/// ```
/// ```compile_fail
/// # use lava::image::{Image, format::D32SfloatS8Uint, usage::DepthAttachmentSampled};
/// fn bind(image: Image<D32SfloatS8Uint, DepthAttachmentSampled>) {
///     image.bind_sampled(0);
/// }
/// ```
pub trait SampledImage: crate::sealed::Sealed {}
impl<F: Format, U: IsSampled, D: Dim, V: ShaderViews<F>> SampledImage for Image<F, U, D, V> {}

/// An image that can be bound in the storage set (`Image::bind_storage`): a storage usage and
/// views that show shaders one aspect.
pub trait StorageImage: crate::sealed::Sealed {}
impl<F: Format, U: IsStorage, D: Dim, V: ShaderViews<F>> StorageImage for Image<F, U, D, V> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{
        D3,
        format::{D32Sfloat, D32SfloatS8Uint, Dynamic, R8G8B8A8Unorm},
        usage::{ColorAttachmentSampled, DepthAttachmentSampled, Sampled, TransferSrc},
    };

    // Compile-time checks: each helper only compiles for types with the bound.
    fn color_target<T: ColorTarget>() {}
    fn depth_target<T: DepthTarget>() {}
    fn transfer_src<T: TransferSrcView>() {}
    fn sampled_view<T: SampledView>() {}
    fn shader_views<F: Format, V: ShaderViews<F>>() {}
    fn sampled_image<I: SampledImage>() {}

    #[test]
    fn views_implement_what_their_types_allow() {
        color_target::<ImageView<R8G8B8A8Unorm, ColorAttachmentSampled>>();
        color_target::<ImageView<Dynamic, crate::image::usage::Dynamic>>();
        depth_target::<ImageView<D32SfloatS8Uint, DepthAttachmentSampled>>();
        sampled_view::<ImageView<R8G8B8A8Unorm, Sampled, D3>>();
        transfer_src::<ImageView<R8G8B8A8Unorm, TransferSrc, D3>>();
    }

    #[test]
    fn shaders_see_one_aspect() {
        shader_views::<R8G8B8A8Unorm, Single>();
        shader_views::<D32Sfloat, PerMip>();
        shader_views::<D32SfloatS8Uint, DepthSplit>();
        shader_views::<D32SfloatS8Uint, DepthSplitMips>();
        shader_views::<Dynamic, Single>();
        shader_views::<Dynamic, DepthSplit>();
        sampled_image::<Image<R8G8B8A8Unorm, Sampled>>();
        sampled_image::<Image<D32SfloatS8Uint, DepthAttachmentSampled, D2, DepthSplit>>();
    }
}

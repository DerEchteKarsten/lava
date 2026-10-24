//! Image views and the boxes of texels transfers take: `ImageView` (a borrowed range of an image's mips and layers, its handle, its `VkImageView` and reinterpreted formats), `ImageSlice` (a box of one of a view's mips), `TexelBuffer` (buffer rows for copies), and the image transfer commands (clears, blits, image and buffer copies) with their format, box and block checks
use std::{
    iter,
    range::Range,
    sync::{Arc, atomic::AtomicU64},
};

use crate::vk;
use lava_macros::validation_trace;

use crate::{
    bindless::{Allocatable, ImageHandle},
    buffer::slice::BufferSlice,
    command_buffer::{CommandBuffer, ImageAccess, ImageUse},
    image::{
        Image, ImageInternal, Views,
        dim::{D2, D2Array, Dim, box_inside, box_is_empty},
        format::{
            self, Block, ColorAspect, DepthStencilAspect, Format, Numeric, StaticFormat,
            StencilAspect,
        },
        kind::{ImageRef, TransferDstView, TransferSrcView},
        usage::{ImageUsage, IsMutableFormat},
    },
    queue::{FamilyInfo, HasCompute, HasGraphics, Kind, QueueKind},
    state::{Ctx, Functions},
};

/// A borrowed view of `mip_range` and `layer_range` of an image, over every aspect: the image
/// type commands, bindless binds and generated bindings take.
#[derive(Clone, Copy)]
pub struct ImageView<'a, F: Format, U: ImageUsage, D: Dim = D2> {
    pub(crate) internal: &'a Arc<ImageInternal>,
    /// The image's own `VkImageView` of these mips and layers, if it has one.
    pub(crate) view: vk::ImageView,
    pub(crate) mip_range: Range<u32>,
    pub(crate) layers: Range<u32>,
    /// The extent of mip `mip_range.start`.
    pub(crate) extent: D::Extent,
    pub(crate) handle: ImageHandle,
    pub(crate) format: F,
    pub(crate) usage: U,
    pub(crate) dim: D,
}

impl<F: Format, U: ImageUsage, D: Dim> std::fmt::Debug for ImageView<'_, F, U, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImageView")
            .field("image", &self.internal.image)
            .field("format", &self.format.vk_format())
            .field("mip_range", &self.mip_range)
            .field("layers", &self.layers)
            .field("extent", &self.extent)
            .field("handle", &self.handle)
            .finish()
    }
}

impl<'a, F: Format, U: ImageUsage, D: Dim> ImageView<'a, F, U, D> {
    pub fn image(&self) -> vk::Image {
        self.internal.image
    }

    /// The image's tracked states: one for all its mips, or one per mip of the image; each
    /// covers every layer.
    pub(crate) fn states(&self) -> &'a [AtomicU64] {
        &self.internal.states
    }

    /// Its `VkImageView`, which only images with an attachment usage have (null otherwise),
    /// for renderings: its usage is the image's attachment usage alone. Descriptors are
    /// written without one. A view of some layers of an array, or of another format
    /// (`reinterpret`), gets its own the first time it is asked for, kept until the image is
    /// destroyed.
    pub fn vk_view(&self) -> vk::ImageView {
        match self.internal.attachment() {
            true if self.view == vk::ImageView::null() => self.internal.cache.get(
                self.internal.image,
                self.format.vk_format(),
                self.format.aspects(),
                D::VIEW_TYPE,
                self.mip_range,
                self.layers,
                self.usage.flags(),
            ),
            _ => self.view,
        }
    }

    pub fn mip_range(&self) -> Range<u32> {
        self.mip_range
    }

    /// The array layers it shows: `0..1` unless its image is a `D2Array` image.
    pub fn layer_range(&self) -> Range<u32> {
        self.layers
    }

    /// How many layers it shows.
    pub fn layer_count(&self) -> u32 {
        self.layers.end - self.layers.start
    }

    /// The extent of its first mip.
    pub fn extent(&self) -> D::Extent {
        self.extent
    }

    /// The view's handle as `K`, for a shader that reads it through `K`'s descriptors. Panics
    /// unless it is bound with them: a `Sampled` handle needs `bind_sampled` or `bind_combined`
    /// (`bind_mips_*` for a mip's view), a `Storage` one `bind_storage` or `bind_combined`, a
    /// `Combined` one `bind_combined`.
    #[track_caller]
    pub fn handle<K: Allocatable>(&self) -> ImageHandle<K> {
        self.handle.narrow().unwrap_or_else(|| {
            panic!(
                "a shader reads the view of {:?} as {}, which it isn't bound as ({:?}): call `{}` \
                 (or `bind_combined` for both descriptors, `bind_mips_*` for a mip's view) on its image",
                self.internal.image,
                std::any::type_name::<K>().rsplit("::").next().unwrap(),
                self.handle,
                K::BIND,
            )
        })
    }

    pub fn subresource_range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: self.format.aspects(),
            base_mip_level: self.mip_range.start,
            level_count: self.mip_range.end - self.mip_range.start,
            base_array_layer: self.layers.start,
            layer_count: self.layer_count(),
        }
    }
    /// The layers of its first mip a copy or blit addresses: the view's, of one aspect, depth
    /// for a combined depth-stencil format.
    pub fn subresource_layers(&self) -> vk::ImageSubresourceLayers {
        let aspect = single_aspect(self.format.aspects());
        self.subresource_layers_of(self.mip_range.start, aspect)
    }

    pub(crate) fn subresource_layers_of(
        &self,
        mip_level: u32,
        aspect: vk::ImageAspectFlags,
    ) -> vk::ImageSubresourceLayers {
        vk::ImageSubresourceLayers {
            aspect_mask: aspect,
            mip_level,
            base_array_layer: self.layers.start,
            layer_count: self.layer_count(),
        }
    }
    /// All of mip `mip_range.start`.
    pub fn whole(self) -> ImageSlice<Self> {
        self.region(self.extent)
    }
    /// The `extent` texels at the origin of mip `mip_range.start`.
    pub fn region(self, extent: D::Extent) -> ImageSlice<Self> {
        ImageSlice {
            level: self.mip_range.start,
            view: self,
            offset: D::Offset::default(),
            extent,
            stencil: false,
        }
    }
    pub fn format(&self) -> F {
        self.format
    }

    pub fn usage(&self) -> U {
        self.usage
    }

    /// Panics unless the format has the `required` aspects; does nothing for a static format.
    #[track_caller]
    pub fn aspect_check(&self, required: vk::ImageAspectFlags) {
        self.format.aspect_check(required);
    }

    /// Panics unless the format's texels have the type `T`; does nothing for a static format.
    #[track_caller]
    pub fn texel_check<T: 'static>(&self) {
        self.format.texel_check::<T>();
    }

    /// Panics unless the usage has every flag of `required`; does nothing for a static usage.
    #[track_caller]
    pub fn usage_check(&self, required: vk::ImageUsageFlags) {
        self.usage.check(required);
    }

    /// The same mips and layers as `F2`, a format of the image's compatibility class (sRGB
    /// texels as UNORM ones for storage, `R32Uint` for atomics on `R32Sfloat`): for an image
    /// whose usage has `MutableFormat`. It has no handle (bind it into reserved slots with
    /// `bindless::bind`) and, for an attachment image, its own `VkImageView`, made on first
    /// use. Panics for a format of another class (see `reinterpret_as`).
    #[track_caller]
    pub fn reinterpret<F2: StaticFormat>(self) -> ImageView<'a, F2, U, D>
    where
        U: IsMutableFormat,
    {
        self.reinterpret_as(F2::default())
    }

    /// [`Self::reinterpret`] for any format, `format::Dynamic` included. Panics unless the
    /// image was created with `MutableFormat` (a `Dynamic` usage's image may not be) and
    /// `format` is of its format's compatibility class: uncompressed colour formats of one
    /// texel size share one, block-compressed formats one per block type, and depth or
    /// stencil formats are alone in theirs.
    #[track_caller]
    pub fn reinterpret_as<F2: Format>(self, format: F2) -> ImageView<'a, F2, U, D>
    where
        U: IsMutableFormat,
    {
        let (from, to) = (self.format.vk_format(), format.vk_format());
        assert!(
            (self.internal.create_flags).contains(vk::ImageCreateFlags::MUTABLE_FORMAT),
            "a view of {from:?} as {to:?} needs an image created with `MutableFormat` in its usage"
        );
        let class = |f: vk::Format, size: usize, aspects| format::class(f, size, aspects);
        assert!(
            class(from, self.format.texel_size(), self.format.aspects())
                == class(to, format.texel_size(), format.aspects()),
            "{from:?} and {to:?} are of different compatibility classes: a view reinterprets \
             texels of one size, or blocks of one type"
        );
        let same = from == to;
        ImageView {
            internal: self.internal,
            view: if same {
                self.view
            } else {
                vk::ImageView::null()
            },
            mip_range: self.mip_range,
            layers: self.layers,
            extent: self.extent,
            handle: if same { self.handle } else { ImageHandle::NULL },
            format,
            usage: self.usage,
            dim: self.dim,
        }
    }

    /// A shader's access of the view's mips at `stage`, in `GENERAL` like every descriptor
    /// (generated bindings make these); their tracked states move only when the command is
    /// recorded.
    pub fn access(&self, stage: vk::PipelineStageFlags2, access: vk::AccessFlags2) -> ImageUse<'a> {
        self.access_in(stage, access, vk::ImageLayout::GENERAL)
    }

    /// The access of a command that wants the view's mips in `layout` (attachments, presents).
    pub(crate) fn access_in(
        &self,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
        layout: vk::ImageLayout,
    ) -> ImageUse<'a> {
        ImageUse {
            access: ImageAccess {
                stage,
                access,
                image: self.internal.image,
                layout,
                aspect: self.format.aspects(),
                mip_range: self.mip_range,
            },
            internal: self.internal,
        }
    }

    /// The `GENERAL` access of a transfer that touches only mip `level`: just that mip where
    /// the image tracks its mips one by one, else all of the view's, which share one state.
    pub(crate) fn mip_access(
        &self,
        level: u32,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
    ) -> ImageUse<'a> {
        let mut first = self.access(stage, access);
        if self.internal.states.len() > 1 {
            first.access.mip_range = (level..level + 1).into();
        }
        first
    }
}

impl<'a, F: Format, U: ImageUsage> ImageView<'a, F, U, D2Array> {
    /// Layer `layer` of the view, as a 2D view of the same mips: to render into one layer, copy
    /// it or bind it alone (`bindless::bind`). It has no handle of its own (the array's
    /// `bind_*` bind the array). Panics for a layer the view doesn't show.
    #[track_caller]
    pub fn layer(self, layer: u32) -> ImageView<'a, F, U, D2> {
        let layers = self.layers(layer..layer + 1);
        ImageView {
            internal: layers.internal,
            view: vk::ImageView::null(),
            mip_range: layers.mip_range,
            layers: layers.layers,
            extent: layers.extent,
            handle: ImageHandle::NULL,
            format: layers.format,
            usage: layers.usage,
            dim: D2,
        }
    }

    /// Layers `layers` of the view (counted from the image's first), still an array: for a
    /// layered or multiview rendering into some layers. Panics for layers the view doesn't
    /// show; fewer than the view's have no handle.
    #[track_caller]
    pub fn layers(self, layers: std::ops::Range<u32>) -> Self {
        assert!(
            self.layers.start <= layers.start
                && layers.start < layers.end
                && layers.end <= self.layers.end,
            "layers {layers:?} aren't some of the view's {:?}",
            self.layers
        );
        let layers: Range<u32> = layers.into();
        if layers == self.layers {
            return self;
        }
        ImageView {
            view: vk::ImageView::null(),
            layers,
            handle: ImageHandle::NULL,
            ..self
        }
    }
}

/// A box of texels: `extent` texels at `offset` of one mip of `view` (its first, unless
/// [`Self::mip`] picked another), made by `view.whole()` or `view.region(extent)`.
pub struct ImageSlice<R: ImageRef> {
    pub view: R,
    pub offset: <R::Dim as Dim>::Offset,
    pub extent: <R::Dim as Dim>::Extent,
    /// The mip it lies in, one of the view's.
    pub(crate) level: u32,
    /// Copies to and from buffers move a combined depth-stencil format's stencil, not its depth.
    pub(crate) stencil: bool,
}

impl<R: ImageRef> Clone for ImageSlice<R> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<R: ImageRef> Copy for ImageSlice<R> {}

impl<R: ImageRef> std::fmt::Debug for ImageSlice<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImageSlice")
            .field("offset", &self.offset)
            .field("extent", &self.extent)
            .field("level", &self.level)
            .field("stencil", &self.stencil)
            .finish_non_exhaustive()
    }
}

impl<R: ImageRef> ImageSlice<R> {
    pub fn offset(mut self, offset: <R::Dim as Dim>::Offset) -> Self {
        self.offset = self.offset + offset;
        self
    }

    pub fn grow(mut self, extent: <R::Dim as Dim>::Extent) -> Self {
        self.extent = self.extent + extent;
        self
    }

    /// The same offset with a box of `extent` texels.
    pub fn with_extent(mut self, extent: <R::Dim as Dim>::Extent) -> Self {
        self.extent = extent;
        self
    }

    /// The mip it lies in.
    pub fn level(&self) -> u32 {
        self.level
    }

    /// All of mip `level` of its view's image, one of the view's mips: a transfer of one mip
    /// of an image with one view for all of them (`Single`) needs no `MipChain`. Panics for a
    /// mip the view doesn't show.
    #[track_caller]
    pub fn mip<'a>(self, level: u32) -> Self
    where
        R: 'a,
    {
        let view = ImageRef::view::<'a>(self.view);
        let mips = view.mip_range;
        assert!(
            mips.start <= level && level < mips.end,
            "mip {level} isn't one of the view's mips {mips:?}"
        );
        ImageSlice {
            view: self.view,
            offset: Default::default(),
            extent: R::Dim::mip_extent(view.extent, level - mips.start),
            level,
            stencil: self.stencil,
        }
    }

    /// The same box of a combined depth-stencil format's stencil aspect (or of a stencil
    /// format), for copies to and from buffers, which move one aspect: its depth by default.
    pub fn stencil(mut self) -> Self
    where
        R::Format: StencilAspect,
    {
        let view = ImageRef::view(self.view);
        view.aspect_check(vk::ImageAspectFlags::STENCIL);
        self.stencil = true;
        self
    }

    /// The same box, of an `ImageView`.
    pub fn resolve<'a>(self) -> ImageSlice<ImageView<'a, R::Format, R::Usage, R::Dim>>
    where
        R: 'a,
    {
        ImageSlice {
            view: self.view.view(),
            offset: self.offset,
            extent: self.extent,
            level: self.level,
            stencil: self.stencil,
        }
    }
}

impl<F: Format, U: ImageUsage, D: Dim> ImageSlice<ImageView<'_, F, U, D>> {
    /// The extent of the mip it lies in.
    fn mip_extent(&self) -> D::Extent {
        D::mip_extent(self.view.extent, self.level - self.view.mip_range.start)
    }

    /// The aspect copies to and from buffers move: the stencil if `stencil` picked it, else
    /// the format's one aspect (depth for a combined depth-stencil format).
    fn copy_aspect(&self) -> vk::ImageAspectFlags {
        match self.stencil {
            true => vk::ImageAspectFlags::STENCIL,
            false => single_aspect(self.view.format.aspects()),
        }
    }

    fn subresource(&self, aspect: vk::ImageAspectFlags) -> vk::ImageSubresourceLayers {
        self.view.subresource_layers_of(self.level, aspect)
    }
}

/// Depth for a combined depth-stencil format, the aspect shaders and copies see.
pub(crate) fn single_aspect(aspects: vk::ImageAspectFlags) -> vk::ImageAspectFlags {
    if aspects.contains(vk::ImageAspectFlags::DEPTH) {
        vk::ImageAspectFlags::DEPTH
    } else {
        aspects
    }
}

pub(crate) fn cast_check(
    from: (vk::Format, vk::ImageUsageFlags),
    to: (vk::Format, vk::ImageUsageFlags),
) {
    assert!(
        from.0 == to.0 && from.1.contains(to.1),
        "can't cast an image of format {:?} and usage {:?} to format {:?} and usage {:?}",
        from.0,
        from.1,
        to.0,
        to.1
    );
}

impl<F: Format, U: ImageUsage, D: Dim, V: Views> Image<F, U, D, V> {
    /// All of mip 0: `whole_view().whole()`.
    pub fn whole(&self) -> ImageSlice<ImageView<'_, F, U, D>> {
        self.whole_view().whole()
    }
}

/// How a blit samples its source.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
    Nearest = 0,
    /// Needs `SAMPLED_IMAGE_FILTER_LINEAR` of the source format, which integer formats never
    /// have.
    Linear = 1,
}

/// The buffer side of a copy between a buffer and an image: a slice with the layout of the
/// texels in it. A `BufferSlice` converts into tightly packed rows; [`BufferSlice::rows`] sets
/// a row length and image height (in texels) of its own.
#[derive(Clone, Copy, Debug)]
pub struct TexelBuffer<'a> {
    pub(crate) slice: BufferSlice<'a, u8>,
    pub(crate) row_length: u32,
    pub(crate) image_height: u32,
}

impl<'a, T: Copy + 'static> From<BufferSlice<'a, T>> for TexelBuffer<'a> {
    fn from(slice: BufferSlice<'a, T>) -> Self {
        TexelBuffer {
            slice: slice.bytes(),
            row_length: 0,
            image_height: 0,
        }
    }
}

impl<'a, T: Copy + 'static> BufferSlice<'a, T> {
    /// The slice as the buffer side of an image copy whose rows start `row_length` texels apart
    /// and whose slices (and layers) `image_height` rows apart: a box of a larger image in the
    /// buffer. 0 for either is the box's own width or height, as a plain slice has them;
    /// otherwise at least that, and whole blocks of a block-compressed format.
    pub fn rows(self, row_length: u32, image_height: u32) -> TexelBuffer<'a> {
        TexelBuffer {
            slice: self.bytes(),
            row_length,
            image_height,
        }
    }
}

impl<K: HasCompute> CommandBuffer<'_, K> {
    /// Clears every mip and layer of the colour view `image` to `clear_color`. Panics for a
    /// block-compressed format and one without `TRANSFER_DST` in its format features.
    #[validation_trace]
    pub fn clear_image<T: TransferDstView<Format: ColorAspect>>(
        &mut self,
        image: T,
        clear_color: <T::Format as Format>::Texels,
    ) {
        let image = image.view();
        image.aspect_check(vk::ImageAspectFlags::COLOR);
        image.usage_check(vk::ImageUsageFlags::TRANSFER_DST);
        let format = image.format().vk_format();
        assert!(
            format::block(format).is_none(),
            "{format:?} is block-compressed, and Vulkan clears no compressed images"
        );
        transfer_feature_check(format, vk::FormatFeatureFlags2::TRANSFER_DST, "cleared");
        let value = vk::ClearValue::from(image.format().clear_value(clear_color));
        self.rec.flush_pending(
            std::iter::once(image.access(
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
            )),
            std::iter::empty(),
        );
        unsafe {
            Ctx::device().cmd_clear_color_image(
                self.rec.handle,
                image.image(),
                vk::ImageLayout::GENERAL,
                &value.color,
                &[image.subresource_range()],
            )
        };
    }
}

impl<K: HasGraphics> CommandBuffer<'_, K> {
    /// Clears every mip and layer of the depth or stencil view `image` to `value` (a depth in
    /// 0 to 1, a stencil value, or both for a combined format), every aspect it has.
    #[validation_trace]
    pub fn clear_depth_stencil<T: TransferDstView<Format: DepthStencilAspect>>(
        &mut self,
        image: T,
        value: <T::Format as Format>::Texels,
    ) {
        let image = image.view();
        let aspects = image.format().aspects();
        assert!(
            aspects.intersects(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL),
            "an image of format {:?} has no depth or stencil aspect: clear it with `clear_image`",
            image.format().vk_format()
        );
        image.usage_check(vk::ImageUsageFlags::TRANSFER_DST);
        let format = image.format().vk_format();
        transfer_feature_check(format, vk::FormatFeatureFlags2::TRANSFER_DST, "cleared");
        let format::ClearValue::DepthStencil { depth, stencil } = image.format().clear_value(value)
        else {
            unreachable!("depth and stencil formats clear to depth and stencil values")
        };
        assert!(
            (0.0..=1.0).contains(&depth),
            "a depth clear value lies in 0 to 1, not {depth}"
        );
        self.rec.flush_pending(
            std::iter::once(image.access(
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
            )),
            std::iter::empty(),
        );
        unsafe {
            Ctx::device().cmd_clear_depth_stencil_image(
                self.rec.handle,
                image.image(),
                vk::ImageLayout::GENERAL,
                &vk::ClearDepthStencilValue { depth, stencil },
                &[image.subresource_range()],
            )
        };
    }

    /// Blits the box `src` into the box `dst` of colour formats, scaling it to fit. Panics
    /// where the formats lack `BLIT_SRC`/`BLIT_DST`, for a linear filter of a format that
    /// can't filter linearly (integer ones never do), formats of different numeric types
    /// (integers blit only to integers of the same signedness), and boxes of one mip that
    /// overlap.
    #[validation_trace]
    pub fn blit_image<
        S: TransferSrcView<Format: ColorAspect>,
        T: TransferDstView<Dim = S::Dim, Format: ColorAspect>,
    >(
        &mut self,
        src: ImageSlice<S>,
        dst: ImageSlice<T>,
        filter: Filter,
    ) {
        let (src, dst) = (src.resolve(), dst.resolve());
        src.view.aspect_check(vk::ImageAspectFlags::COLOR);
        dst.view.aspect_check(vk::ImageAspectFlags::COLOR);
        src.view.usage_check(vk::ImageUsageFlags::TRANSFER_SRC);
        dst.view.usage_check(vk::ImageUsageFlags::TRANSFER_DST);
        check_box(&src, false, "source");
        check_box(&dst, false, "destination");
        assert_eq!(
            src.view.layer_count(),
            dst.view.layer_count(),
            "a blit's source and destination have as many layers"
        );
        let (from, to) = (src.view.format().vk_format(), dst.view.format().vk_format());
        let src_features = Ctx::format_features(from);
        assert!(
            src_features.contains(vk::FormatFeatureFlags2::BLIT_SRC),
            "the device doesn't blit from {from:?} (BLIT_SRC)"
        );
        assert!(
            Ctx::format_features(to).contains(vk::FormatFeatureFlags2::BLIT_DST),
            "the device doesn't blit into {to:?} (BLIT_DST)"
        );
        let numeric = |f: vk::Format| format::info(f).map(|info| info.numeric);
        let (src_numeric, dst_numeric) = (numeric(from), numeric(to));
        if filter == Filter::Linear {
            assert!(
                !matches!(src_numeric, Some(Numeric::Uint | Numeric::Sint)),
                "{from:?} is an integer format, which blits filter only by the nearest texel"
            );
            assert!(
                src_features.contains(vk::FormatFeatureFlags2::SAMPLED_IMAGE_FILTER_LINEAR),
                "the device doesn't filter {from:?} linearly (SAMPLED_IMAGE_FILTER_LINEAR): blit it with `Filter::Nearest`"
            );
        }
        let integer = |n: Option<Numeric>| matches!(n, Some(Numeric::Uint | Numeric::Sint));
        assert!(
            src_numeric == dst_numeric || !integer(src_numeric) && !integer(dst_numeric),
            "a blit from {from:?} into {to:?} mixes numeric types: integers blit only into \
             integers of the same signedness"
        );
        check_overlap(&src, &dst, "blit");
        self.rec.flush_pending(
            [
                src.view.mip_access(
                    src.level,
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::TRANSFER_READ,
                ),
                dst.view.mip_access(
                    dst.level,
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::TRANSFER_WRITE,
                ),
            ]
            .into_iter(),
            iter::empty(),
        );

        let corners = |offset: <S::Dim as Dim>::Offset, extent: <S::Dim as Dim>::Extent| {
            let start = S::Dim::offset_3d(offset);
            let size = S::Dim::extent_3d(extent);
            [
                start,
                vk::Offset3D {
                    x: start.x + size.width as i32,
                    y: start.y + size.height as i32,
                    z: start.z + size.depth as i32,
                },
            ]
        };
        let regions = [vk::ImageBlit {
            src_offsets: corners(src.offset, src.extent),
            dst_offsets: corners(dst.offset, dst.extent),
            src_subresource: src.subresource(vk::ImageAspectFlags::COLOR),
            dst_subresource: dst.subresource(vk::ImageAspectFlags::COLOR),
        }];
        unsafe {
            Ctx::device().cmd_blit_image(
                self.rec.handle,
                src.view.image(),
                vk::ImageLayout::GENERAL,
                dst.view.image(),
                vk::ImageLayout::GENERAL,
                &regions,
                vk::Filter::from_raw(filter as i32),
            );
        }
    }
}

impl<K: QueueKind> CommandBuffer<'_, K> {
    /// Copies the box `src` into the box `dst` of the same extent, texel for texel
    /// (`vkCmdCopyImage2`), between images of any dimension whose formats have the same texel
    /// block (a depth or stencil format only into itself, every aspect), as many layers, and,
    /// within one mip of one image, boxes that don't overlap. On a transfer family the boxes
    /// keep its image granularity, and depth or stencil copies need graphics.
    #[validation_trace]
    pub fn copy_image<S: TransferSrcView, T: TransferDstView<Dim = S::Dim>>(
        &mut self,
        src: ImageSlice<S>,
        dst: ImageSlice<T>,
    ) {
        let (src, dst) = (src.resolve(), dst.resolve());
        src.view.usage_check(vk::ImageUsageFlags::TRANSFER_SRC);
        dst.view.usage_check(vk::ImageUsageFlags::TRANSFER_DST);
        check_box(&src, true, "source");
        check_box(&dst, true, "destination");
        let (from, to) = (src.view.format(), dst.view.format());
        let (from_vk, to_vk) = (from.vk_format(), to.vk_format());
        let colour = vk::ImageAspectFlags::COLOR;
        let aspects = from.aspects();
        if !aspects.contains(colour) || !to.aspects().contains(colour) {
            assert!(
                from_vk == to_vk,
                "a depth or stencil image copies only into its own format, not {from_vk:?} into {to_vk:?}"
            );
        }
        let (src_block, dst_block) = (
            copy_block_of(&from, aspects),
            copy_block_of(&to, to.aspects()),
        );
        assert!(
            src_block == dst_block,
            "{from_vk:?} ({src_block:?}) and {to_vk:?} ({dst_block:?}) have different texel blocks: \
             image copies keep them"
        );
        assert!(
            src.extent == dst.extent,
            "an image copy's source box {:?} and destination box {:?} are of one extent",
            src.extent,
            dst.extent
        );
        assert_eq!(
            src.view.layer_count(),
            dst.view.layer_count(),
            "an image copy's source and destination have as many layers"
        );
        transfer_feature_check(
            from_vk,
            vk::FormatFeatureFlags2::TRANSFER_SRC,
            "copied from",
        );
        transfer_feature_check(to_vk, vk::FormatFeatureFlags2::TRANSFER_DST, "copied into");
        check_overlap(&src, &dst, "copy");
        check_image_family(&self.rec.family, &src);
        check_image_family(&self.rec.family, &dst);
        self.rec.flush_pending(
            [
                src.view.mip_access(
                    src.level,
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::TRANSFER_READ,
                ),
                dst.view.mip_access(
                    dst.level,
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::TRANSFER_WRITE,
                ),
            ]
            .into_iter(),
            iter::empty(),
        );
        let regions = [vk::ImageCopy2::default()
            .src_subresource(src.subresource(aspects))
            .src_offset(S::Dim::offset_3d(src.offset))
            .dst_subresource(dst.subresource(aspects))
            .dst_offset(S::Dim::offset_3d(dst.offset))
            .extent(S::Dim::extent_3d(src.extent))];
        let info = vk::CopyImageInfo2::default()
            .src_image(src.view.image())
            .src_image_layout(vk::ImageLayout::GENERAL)
            .dst_image(dst.view.image())
            .dst_image_layout(vk::ImageLayout::GENERAL)
            .regions(&regions);
        unsafe { Ctx::device().cmd_copy_image2(self.rec.handle, &info) };
    }

    /// Copies the texels of `src` into the box `dst` of every layer of its view, one layer
    /// after another: tightly packed from a `BufferSlice`, or with the row length and image
    /// height of [`BufferSlice::rows`]. One aspect: a combined depth-stencil format's depth, or
    /// its stencil with [`ImageSlice::stencil`]. The buffer's address is a multiple of the
    /// texel block's size (4 for depth and stencil, and on a transfer family).
    #[validation_trace]
    pub fn copy_buffer_to_image<'b, I: TransferDstView>(
        &mut self,
        src: impl Into<TexelBuffer<'b>>,
        dst: ImageSlice<I>,
    ) {
        let src = src.into();
        let dst = dst.resolve();
        dst.view.usage_check(vk::ImageUsageFlags::TRANSFER_DST);
        let region = buffer_image_region(&self.rec.family, &dst, src, true);
        self.rec.flush_pending(
            iter::once(dst.view.mip_access(
                dst.level,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
            )),
            iter::once(src.slice.access(
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_READ,
            )),
        );
        let regions = [region];
        unsafe {
            Functions::device_address_commands().cmd_copy_memory_to_image(
                self.rec.handle,
                &vk::CopyDeviceMemoryImageInfoKHR::default()
                    .image(dst.view.image())
                    .regions(&regions),
            )
        };
    }

    /// Copies the texels of the box `src` of every layer of its view into `dst`, one layer
    /// after another, tightly packed or with the rows of [`BufferSlice::rows`] (see
    /// [`Self::copy_buffer_to_image`]). It takes a slice, never a view:
    /// ```compile_fail
    /// # use lava::{buffer::BufferSlice, command_buffer::CommandBuffer, queue::Graphics};
    /// # use lava::image::{ImageView, format::R8G8B8A8Unorm, usage::usage};
    /// fn copy(cmd: &mut CommandBuffer<Graphics>, view: ImageView<R8G8B8A8Unorm, usage![TransferSrc]>, dst: BufferSlice<u8>) {
    ///     cmd.copy_image_to_buffer(view, dst);
    /// }
    /// ```
    /// ```
    /// # use lava::{buffer::BufferSlice, command_buffer::CommandBuffer, queue::Graphics};
    /// # use lava::image::{ImageView, format::R8G8B8A8Unorm, usage::usage};
    /// fn copy(cmd: &mut CommandBuffer<Graphics>, view: ImageView<R8G8B8A8Unorm, usage![TransferSrc]>, dst: BufferSlice<u8>) {
    ///     cmd.copy_image_to_buffer(view.whole(), dst);
    /// }
    /// ```
    #[validation_trace]
    pub fn copy_image_to_buffer<'b, I: TransferSrcView>(
        &mut self,
        src: ImageSlice<I>,
        dst: impl Into<TexelBuffer<'b>>,
    ) {
        let dst = dst.into();
        let src = src.resolve();
        src.view.usage_check(vk::ImageUsageFlags::TRANSFER_SRC);
        let region = buffer_image_region(&self.rec.family, &src, dst, false);
        self.rec.flush_pending(
            iter::once(src.view.mip_access(
                src.level,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_READ,
            )),
            iter::once(dst.slice.access(
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
            )),
        );
        let regions = [region];
        unsafe {
            Functions::device_address_commands().cmd_copy_image_to_memory(
                self.rec.handle,
                &vk::CopyDeviceMemoryImageInfoKHR::default()
                    .image(src.view.image())
                    .regions(&regions),
            )
        };
    }
}

/// Checks a copy between `slice` and `buffer` (the box, its blocks, the buffer's size, address
/// and rows, the family's rules) and returns its region.
#[track_caller]
fn buffer_image_region<'s, F: Format, U: ImageUsage, D: Dim>(
    family: &FamilyInfo,
    slice: &ImageSlice<ImageView<'_, F, U, D>>,
    buffer: TexelBuffer<'s>,
    into_image: bool,
) -> vk::DeviceMemoryImageCopyKHR<'s> {
    let what = if into_image { "destination" } else { "source" };
    let other = if into_image { "source" } else { "destination" };
    check_box(slice, true, what);
    let format = slice.view.format;
    let aspect = slice.copy_aspect();
    let block = copy_block_of(&format, aspect);
    let vk_format = format.vk_format();
    let rows = (buffer.row_length, buffer.image_height);
    let extent = D::extent_3d(slice.extent);
    check_rows(block, vk_format, extent, rows);
    // With address commands neither the driver nor the validation layer checks the size.
    let needed = rows_len(block, extent, slice.view.layer_count(), rows) as u64;
    let bytes = buffer.slice.size();
    assert!(
        needed <= bytes,
        "the {other} buffer has {bytes} bytes, the copy of {:?} texels of {vk_format:?} takes {needed}",
        slice.extent
    );
    let address = buffer.slice.address();
    let depth_stencil = vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL;
    let align = match aspect.intersects(depth_stencil) {
        true => 4,
        false => block.bytes as u64,
    };
    assert!(
        address.is_multiple_of(align),
        "copies of {vk_format:?} take buffer addresses that are multiples of {align} bytes, not {address:#x}"
    );
    if family.kind == Kind::Transfer {
        assert!(
            address.is_multiple_of(4),
            "a transfer family copies from and to 4-byte aligned buffer addresses, not {address:#x}"
        );
    }
    check_image_family(family, slice);
    vk::DeviceMemoryImageCopyKHR::default()
        .address_range(buffer.slice.address_range())
        .address_flags(buffer.slice.address_flags())
        .address_row_length(buffer.row_length)
        .address_image_height(buffer.image_height)
        .image_subresource(slice.subresource(aspect))
        .image_layout(vk::ImageLayout::GENERAL)
        .image_offset(D::offset_3d(slice.offset))
        .image_extent(extent)
}

/// The texel block copies of `aspect` of `format` move. Panics for a format lava knows no block
/// of, whose copies it can't check.
#[track_caller]
pub(crate) fn copy_block_of<F: Format>(format: &F, aspect: vk::ImageAspectFlags) -> Block {
    let vk_format = format.vk_format();
    format::copy_block(vk_format, format.texel_size(), single_aspect(aspect)).unwrap_or_else(|| {
        panic!("lava knows no texel block of {vk_format:?}, so it doesn't copy it")
    })
}

/// Bytes a copy of `extent` texels of every one of `layers` takes in memory whose rows are
/// `row_length` texels apart and slices `image_height` rows apart (0: the extent's), up to the
/// last block's end.
pub(crate) fn rows_len(
    block: Block,
    extent: vk::Extent3D,
    layers: u32,
    (row_length, image_height): (u32, u32),
) -> usize {
    let row_length = if row_length == 0 {
        extent.width
    } else {
        row_length
    };
    let image_height = if image_height == 0 {
        extent.height
    } else {
        image_height
    };
    let (w, h) = (block.width as usize, block.height as usize);
    let row_blocks = (row_length as usize).div_ceil(w);
    let slice_rows = (image_height as usize).div_ceil(h);
    let (width_blocks, height_blocks) = (
        (extent.width as usize).div_ceil(w),
        (extent.height as usize).div_ceil(h),
    );
    let slices = extent.depth as usize * layers as usize;
    if slices == 0 || height_blocks == 0 || width_blocks == 0 {
        return 0;
    }
    let last_row = (slices - 1) * slice_rows + height_blocks - 1;
    (last_row * row_blocks + width_blocks) * block.bytes as usize
}

/// Panics unless a row length and image height are 0 or hold the extent, in whole blocks.
#[track_caller]
fn check_rows(
    block: Block,
    format: vk::Format,
    extent: vk::Extent3D,
    (row_length, image_height): (u32, u32),
) {
    assert!(
        row_length == 0 || row_length >= extent.width,
        "rows of {row_length} texels are narrower than the box's {}",
        extent.width
    );
    assert!(
        image_height == 0 || image_height >= extent.height,
        "slices of {image_height} rows are lower than the box's {}",
        extent.height
    );
    assert!(
        row_length % block.width == 0 && image_height % block.height == 0,
        "rows of {row_length} texels and slices of {image_height} rows of {format:?} are whole \
         {}x{} blocks",
        block.width,
        block.height
    );
}

/// Panics unless the box lies inside its mip and, for a copy, has texels and covers whole blocks
/// of a block-compressed format.
#[track_caller]
fn check_box<F: Format, U: ImageUsage, D: Dim>(
    slice: &ImageSlice<ImageView<'_, F, U, D>>,
    copy: bool,
    what: &str,
) {
    let (offset, extent) = (D::offset_3d(slice.offset), D::extent_3d(slice.extent));
    let mip_extent = slice.mip_extent();
    let mip = D::extent_3d(mip_extent);
    assert!(
        box_inside(offset, extent, mip),
        "the {what} box of {:?} texels at {:?} lies outside its mip of {:?}",
        slice.extent,
        slice.offset,
        mip_extent
    );
    if !copy {
        return;
    }
    assert!(
        !box_is_empty(extent),
        "the {what} box of {:?} texels is empty",
        slice.extent
    );
    let block = copy_block_of(&slice.view.format, slice.copy_aspect());
    check_whole_blocks(block, slice.view.format.vk_format(), offset, extent, mip);
}

/// Panics unless the box at `offset` starts on a block and covers whole blocks or reaches the
/// mip's edge.
#[track_caller]
pub(crate) fn check_whole_blocks(
    block: Block,
    format: vk::Format,
    offset: vk::Offset3D,
    extent: vk::Extent3D,
    mip: vk::Extent3D,
) {
    if block.is_texel() {
        return;
    }
    let whole = |o: i32, e: u32, m: u32, side: u32| {
        (o as u32).is_multiple_of(side) && (e.is_multiple_of(side) || o as u32 + e == m)
    };
    assert!(
        whole(offset.x, extent.width, mip.width, block.width)
            && whole(offset.y, extent.height, mip.height, block.height),
        "copies of {format:?} cover whole {}x{} blocks (or reach the mip's edge), not \
         {extent:?} texels at {offset:?} of a mip of {mip:?}",
        block.width,
        block.height
    );
}

/// Panics unless the device's format features of `format` have `feature`.
#[track_caller]
fn transfer_feature_check(format: vk::Format, feature: vk::FormatFeatureFlags2, done: &str) {
    assert!(
        Ctx::format_features(format).contains(feature),
        "images of {format:?} can't be {done} on this device ({feature:?})"
    );
}

/// Panics when `src` and `dst` are boxes of one mip of one image that share texels of a layer.
#[track_caller]
fn check_overlap<F: Format, U: ImageUsage, F2: Format, U2: ImageUsage, D: Dim>(
    src: &ImageSlice<ImageView<'_, F, U, D>>,
    dst: &ImageSlice<ImageView<'_, F2, U2, D>>,
    what: &str,
) {
    if src.view.image() != dst.view.image() || src.level != dst.level {
        return;
    }
    let (sl, dl) = (src.view.layers, dst.view.layers);
    if sl.end <= dl.start || dl.end <= sl.start {
        return;
    }
    let (so, se) = (D::offset_3d(src.offset), D::extent_3d(src.extent));
    let (dof, de) = (D::offset_3d(dst.offset), D::extent_3d(dst.extent));
    let apart = |a: i32, al: u32, b: i32, bl: u32| {
        a as i64 + al as i64 <= b as i64 || b as i64 + bl as i64 <= a as i64
    };
    let disjoint = apart(so.x, se.width, dof.x, de.width)
        || apart(so.y, se.height, dof.y, de.height)
        || apart(so.z, se.depth, dof.z, de.depth);
    assert!(
        disjoint,
        "a {what} within one mip of one image takes boxes that don't overlap, not {:?} at {:?} \
         and {:?} at {:?}",
        src.extent, src.offset, dst.extent, dst.offset
    );
}

/// Panics where the family can't copy the box: depth or stencil images (either way) only on
/// graphics, and on a transfer family only boxes of its image granularity (in blocks for a
/// block-compressed format).
#[track_caller]
fn check_image_family<F: Format, U: ImageUsage, D: Dim>(
    family: &FamilyInfo,
    slice: &ImageSlice<ImageView<'_, F, U, D>>,
) {
    let depth_stencil = vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL;
    // Both ways: without graphics, buffer copies of a depth or stencil aspect are invalid
    // (`VUID-vkCmdCopyImageToMemoryKHR-commandBuffer-13106` for copies out of one).
    assert!(
        family.kind == Kind::Graphics || !slice.view.format.aspects().intersects(depth_stencil),
        "copies of depth or stencil images are recorded on the graphics family"
    );
    if family.kind != Kind::Transfer {
        return;
    }
    let (offset, extent) = (D::offset_3d(slice.offset), D::extent_3d(slice.extent));
    let mip = D::extent_3d(slice.mip_extent());
    let block = format::block(slice.view.format.vk_format()).unwrap_or(Block {
        width: 1,
        height: 1,
        bytes: 0,
    });
    let g = family.image_granularity;
    let granularity = vk::Extent3D {
        width: g.width * block.width,
        height: g.height * block.height,
        depth: g.depth,
    };
    let axes = [
        (granularity.width, offset.x, extent.width, mip.width),
        (granularity.height, offset.y, extent.height, mip.height),
        (granularity.depth, offset.z, extent.depth, mip.depth),
    ];
    let fits = axes.into_iter().all(|(g, offset, extent, mip)| match g {
        0 => offset == 0 && extent == mip,
        g => {
            (offset as u32).is_multiple_of(g) && (extent % g == 0 || offset as u32 + extent == mip)
        }
    });
    assert!(
        fits,
        "a transfer family copies boxes of its image granularity {granularity:?} (in texels), \
         not {extent:?} at {offset:?} of a mip of {mip:?}"
    );
}

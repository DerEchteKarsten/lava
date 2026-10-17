//! Borrowed image views of a mip range (over every aspect) and their slices, the only image types commands take (bounded by the traits of `kind`), with the layouts they track for barriers, and the image transfer commands (clear, blit, copies to and from buffers)
use std::{iter, range::Range, sync::atomic::AtomicU64};

use crate::vk;
use bytemuck::Pod;
use lava_macros::validation_trace;

use crate::{
    bindless::BindlessHandle,
    buffer::{slice::BufferSlice, usage::BufferUsage},
    command_buffer::{BufferAccess, CommandBuffer, ImageAccess, ImageUse},
    queue::{HasCompute, HasGraphics, QueueKind},
    image::{
        Image, Views,
        dim::{D2, Dim},
        expected_data_len,
        format::{Format, bc_block},
        kind::{ImageRef, TransferDstView, TransferSrcView},
        usage::ImageUsage,
    },
    state::{Ctx, Functions},
};

/// A borrowed view of `mip_range` of an image, over every aspect. Commands, bindless binds and
/// generated bindings take these (bounded by the traits of [`crate::image::kind`]), whichever
/// kind of image they come from (a swapchain's included).
#[derive(Clone, Copy, Debug)]
pub struct ImageView<'a, F: Format, U: ImageUsage, D: Dim = D2> {
    pub(crate) image: vk::Image,
    /// Over every aspect.
    pub(crate) view: vk::ImageView,
    pub(crate) mip_range: Range<u32>,
    /// The extent of mip `mip_range.start`.
    pub(crate) extent: D::Extent,
    pub(crate) handle: BindlessHandle,
    /// The tracked states (`command_buffer::image_state`): one for all of `mip_range` (an image
    /// with one view, or a swapchain image), or one per mip.
    pub(crate) states: &'a [AtomicU64],
    pub(crate) format: F,
    pub(crate) usage: U,
    pub(crate) dim: D,
}

impl<'a, F: Format, U: ImageUsage, D: Dim> ImageView<'a, F, U, D> {
    /// The image the view is of.
    pub fn image(&self) -> vk::Image {
        self.image
    }

    /// The Vulkan view over every aspect of the view's mips.
    pub fn view(&self) -> vk::ImageView {
        self.view
    }

    /// The mips the view covers.
    pub fn mip_range(&self) -> Range<u32> {
        self.mip_range
    }

    /// The extent of its first mip.
    pub fn extent(&self) -> D::Extent {
        self.extent
    }

    /// Where the view is bound in the bindless sets (`NULL_HANDLE`s where it isn't).
    pub fn handle(&self) -> BindlessHandle {
        self.handle
    }

    pub fn subresource_range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: self.format.aspects(),
            base_mip_level: self.mip_range.start,
            level_count: self.mip_range.end - self.mip_range.start,
            base_array_layer: 0,
            layer_count: 1,
        }
    }
    /// The layers of `mip_level` a copy or blit addresses: one aspect, the depth aspect of a
    /// combined depth-stencil format.
    pub fn subresource_layers(&self, mip_level: u32) -> vk::ImageSubresourceLayers {
        vk::ImageSubresourceLayers {
            aspect_mask: single_aspect(self.format.aspects()),
            mip_level,
            base_array_layer: 0,
            layer_count: 1,
        }
    }
    /// All of mip `mip_range.start`, which copies and blits of the slice touch.
    pub fn whole(self) -> ImageSlice<Self> {
        self.region(self.extent)
    }
    /// The `extend` texels at the origin of mip `mip_range.start`.
    pub fn region(self, extend: D::Extent) -> ImageSlice<Self> {
        ImageSlice {
            view: self,
            offset: D::Offset::default(),
            extend,
        }
    }
    pub fn format(&self) -> F {
        self.format
    }

    pub fn usage(&self) -> U {
        self.usage
    }

    /// Panics unless the format has the `required` aspects; does nothing for a static format.
    pub fn aspect_check(&self, required: vk::ImageAspectFlags) {
        self.format.aspect_check(required);
    }

    /// Panics unless the format's texels have the type `T`; does nothing for a static format.
    pub fn texel_check<T: 'static>(&self) {
        self.format.texel_check::<T>();
    }

    /// Panics unless the usage has every flag of `required`; does nothing for a static usage.
    pub fn usage_check(&self, required: vk::ImageUsageFlags) {
        self.usage.check(required);
    }

    /// The same view typed as `format` and `usage`, which must be the view's format and no
    /// more than its usage (see `Image::cast`).
    pub fn cast<NF: Format, NU: ImageUsage>(
        self,
        format: NF,
        usage: NU,
    ) -> ImageView<'a, NF, NU, D> {
        cast_check(
            (self.format.vk_format(), self.usage.flags()),
            (format.vk_format(), usage.flags()),
        );
        ImageView {
            image: self.image,
            view: self.view,
            mip_range: self.mip_range,
            extent: self.extent,
            handle: self.handle,
            states: self.states,
            format,
            usage,
            dim: self.dim,
        }
    }

    /// The access of a command that wants the view's mips in `layout`. It borrows their
    /// tracked states, which move to `layout` only when the command is recorded. A pass's
    /// access, for [`crate::command_buffer::Bindings::new`].
    pub fn access(
        &self,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
        layout: vk::ImageLayout,
    ) -> ImageUse<'a> {
        ImageUse {
            access: ImageAccess {
                stage,
                access,
                image: self.image,
                layout,
                aspect: self.format.aspects(),
                mip_range: self.mip_range,
            },
            states: self.states,
        }
    }
}

/// A box of texels of one mip: `extend` texels at `offset` of the first mip of `view`. Commands
/// that work on a box take one (`ImageSlice<impl TransferSrcView>`), never a bare view, which
/// stands for all its mips: `view.whole()` or `view.region(extent)` make one.
pub struct ImageSlice<R: ImageRef> {
    pub view: R,
    pub offset: <R::Dim as Dim>::Offset,
    pub extend: <R::Dim as Dim>::Extent,
}

impl<R: ImageRef> Clone for ImageSlice<R> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<R: ImageRef> Copy for ImageSlice<R> {}

impl<R: ImageRef> ImageSlice<R> {
    pub fn offset(mut self, offset: <R::Dim as Dim>::Offset) -> Self {
        self.offset = self.offset + offset;
        self
    }

    pub fn grow(mut self, extend: <R::Dim as Dim>::Extent) -> Self {
        self.extend = self.extend + extend;
        self
    }

    /// The same box of the same view, as an `ImageView`.
    pub fn resolve<'a>(self) -> ImageSlice<ImageView<'a, R::Format, R::Usage, R::Dim>>
    where
        R: 'a,
    {
        ImageSlice {
            view: self.view.view(),
            offset: self.offset,
            extend: self.extend,
        }
    }

    /// The same slice typed as `format` and `usage` (see `ImageView::cast`).
    pub fn cast<'a, NF: Format, NU: ImageUsage>(
        self,
        format: NF,
        usage: NU,
    ) -> ImageSlice<ImageView<'a, NF, NU, R::Dim>>
    where
        R: 'a,
    {
        ImageSlice {
            view: self.view.view().cast(format, usage),
            offset: self.offset,
            extend: self.extend,
        }
    }
}

/// The aspect a view of a combined depth-stencil format shows shaders, and copies address:
/// depth. Every other format has one aspect.
pub(crate) fn single_aspect(aspects: vk::ImageAspectFlags) -> vk::ImageAspectFlags {
    if aspects.contains(vk::ImageAspectFlags::DEPTH) {
        vk::ImageAspectFlags::DEPTH
    } else {
        aspects
    }
}

/// Panics unless `to` is the same format as `from` and no more than its usage: a cast can only
/// narrow what the type says about an image, never claim a usage it was created without.
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

// ---- transfer commands -------------------------------------------------------------------

/// How a blit samples its source.
#[repr(i32)]
pub enum Filter {
    Nearest = 0,
    Liniear = 1,
}

impl<K: HasCompute> CommandBuffer<K> {
    /// Clears every mip of the view `image` to `clear_color`.
    #[validation_trace]
    pub fn clear_image<T: TransferDstView>(
        &mut self,
        image: T,
        clear_color: <T::Format as Format>::Texels,
    ) {
        let image = image.view();
        image.aspect_check(vk::ImageAspectFlags::COLOR);
        image.usage_check(vk::ImageUsageFlags::TRANSFER_DST);
        self.rec.flush_pending(
            std::iter::once(image.access(
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::ImageLayout::GENERAL,
            )),
            std::iter::empty(),
        );
        unsafe {
            Ctx::device().cmd_clear_color_image(
                self.rec.handle,
                image.image,
                vk::ImageLayout::GENERAL,
                &vk::ClearValue::from(image.format().clear_value(clear_color)).color,
                &[image.subresource_range()],
            )
        };
    }
}

impl<K: HasGraphics> CommandBuffer<K> {
    /// Blits the box `src` into the box `dst`, scaling it to fit.
    #[validation_trace]
    pub fn blit_image<S: TransferSrcView, T: TransferDstView<Dim = S::Dim>>(
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
        self.rec.flush_pending(
            [
                src.view.access(
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::TRANSFER_READ,
                    vk::ImageLayout::GENERAL,
                ),
                dst.view.access(
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::TRANSFER_WRITE,
                    vk::ImageLayout::GENERAL,
                ),
            ]
            .into_iter(),
            iter::empty(),
        );

        let regions = [blit_region(&src, &dst)];
        unsafe {
            Ctx::device().cmd_blit_image(
                self.rec.handle,
                src.view.image,
                vk::ImageLayout::GENERAL,
                dst.view.image,
                vk::ImageLayout::GENERAL,
                &regions,
                vk::Filter::from_raw(filter as i32),
            );
        }
    }
}

impl<K: QueueKind> CommandBuffer<K> {
    /// Copies the tightly packed texels of `src` into the box `dst`.
    #[validation_trace]
    pub fn copy_buffer_to_image<T: Copy + Pod, BU: BufferUsage, I: TransferDstView>(
        &mut self,
        src: BufferSlice<'_, T, BU>,
        dst: ImageSlice<I>,
    ) {
        let dst = dst.resolve();
        dst.view.aspect_check(vk::ImageAspectFlags::COLOR);
        dst.view.usage_check(vk::ImageUsageFlags::TRANSFER_DST);
        check_copy_size(&dst, src.size(), "source");
        self.rec.flush_pending(
            iter::once(dst.view.access(
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::ImageLayout::GENERAL,
            )),
            iter::once(BufferAccess {
                access: vk::AccessFlags2::TRANSFER_READ,
                stage: vk::PipelineStageFlags2::TRANSFER,
                range: src.get_range(),
            }),
        );
        let regions = [vk::DeviceMemoryImageCopyKHR::default()
            .address_range(src.address_range())
            .address_flags(src.address_flags())
            .image_subresource(dst.view.subresource_layers(dst.view.mip_range.start))
            .image_layout(vk::ImageLayout::GENERAL)
            .image_offset(I::Dim::offset_3d(dst.offset))
            .image_extent(I::Dim::extent_3d(dst.extend))];
        unsafe {
            Functions::device_address_commands().cmd_copy_memory_to_image(
                self.rec.handle,
                &vk::CopyDeviceMemoryImageInfoKHR::default()
                    .image(dst.view.image)
                    .regions(&regions),
            )
        };
    }

    /// Copies the texels of the box `src` into `dst`, tightly packed row by row (and slice by
    /// slice). It takes a slice, never a view:
    /// ```compile_fail
    /// # use lava::{buffer::slice::BufferSlice, command_buffer::CommandBuffer, queue::Graphics};
    /// # use lava::image::{format::R8G8B8A8Unorm, slice::ImageView, usage::TransferSrc};
    /// fn copy(cmd: &mut CommandBuffer<Graphics>, view: ImageView<R8G8B8A8Unorm, TransferSrc>, dst: BufferSlice<u8>) {
    ///     cmd.copy_image_to_buffer(view, dst);
    /// }
    /// ```
    /// ```
    /// # use lava::{buffer::slice::BufferSlice, command_buffer::CommandBuffer, queue::Graphics};
    /// # use lava::image::{format::R8G8B8A8Unorm, slice::ImageView, usage::TransferSrc};
    /// fn copy(cmd: &mut CommandBuffer<Graphics>, view: ImageView<R8G8B8A8Unorm, TransferSrc>, dst: BufferSlice<u8>) {
    ///     cmd.copy_image_to_buffer(view.whole(), dst);
    /// }
    /// ```
    #[validation_trace]
    pub fn copy_image_to_buffer<I: TransferSrcView, T: Copy + Pod, BU: BufferUsage>(
        &mut self,
        src: ImageSlice<I>,
        dst: BufferSlice<'_, T, BU>,
    ) {
        let src = src.resolve();
        src.view.aspect_check(vk::ImageAspectFlags::COLOR);
        src.view.usage_check(vk::ImageUsageFlags::TRANSFER_SRC);
        check_copy_size(&src, dst.size(), "destination");
        self.rec.flush_pending(
            iter::once(src.view.access(
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_READ,
                vk::ImageLayout::GENERAL,
            )),
            iter::once(BufferAccess {
                access: vk::AccessFlags2::TRANSFER_WRITE,
                stage: vk::PipelineStageFlags2::TRANSFER,
                range: dst.get_range(),
            }),
        );
        let regions = [vk::DeviceMemoryImageCopyKHR::default()
            .address_range(dst.address_range())
            .address_flags(dst.address_flags())
            .image_subresource(src.view.subresource_layers(src.view.mip_range.start))
            .image_layout(vk::ImageLayout::GENERAL)
            .image_offset(I::Dim::offset_3d(src.offset))
            .image_extent(I::Dim::extent_3d(src.extend))];
        unsafe {
            Functions::device_address_commands().cmd_copy_image_to_memory(
                self.rec.handle,
                &vk::CopyDeviceMemoryImageInfoKHR::default()
                    .image(src.view.image)
                    .regions(&regions),
            )
        };
    }
}

/// Panics unless `buffer` holds the texels of `slice` tightly packed: with address commands,
/// neither the driver nor the validation layer checks a copy's buffer size.
fn check_copy_size<F: Format, U: ImageUsage, D: Dim>(
    slice: &ImageSlice<ImageView<'_, F, U, D>>,
    buffer_bytes: u64,
    what: &str,
) {
    let format = slice.view.format;
    let needed = expected_data_len(
        D::extent_3d(slice.extend),
        format.texel_size(),
        bc_block(format.vk_format()),
    );
    if let Some(needed) = needed {
        assert!(
            needed as u64 <= buffer_bytes,
            "the {what} buffer has {buffer_bytes} bytes, the copy of {:?} texels of {:?} takes {needed}",
            slice.extend,
            format.vk_format()
        );
    }
}

/// Blit region covering the two slices: from each slice's offset to offset + extent.
fn blit_region<F: Format, U: ImageUsage, F2: Format, U2: ImageUsage, D: Dim>(
    src: &ImageSlice<ImageView<'_, F, U, D>>,
    dst: &ImageSlice<ImageView<'_, F2, U2, D>>,
) -> vk::ImageBlit {
    let corners = |offset: D::Offset, extend: D::Extent| {
        let start = D::offset_3d(offset);
        let size = D::extent_3d(extend);
        [
            start,
            vk::Offset3D {
                x: start.x + size.width as i32,
                y: start.y + size.height as i32,
                z: start.z + size.depth as i32,
            },
        ]
    };
    vk::ImageBlit {
        src_offsets: corners(src.offset, src.extend),
        dst_offsets: corners(dst.offset, dst.extend),
        src_subresource: src.view.subresource_layers(src.view.mip_range.start),
        dst_subresource: dst.view.subresource_layers(dst.view.mip_range.start),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{
        format::{D32SfloatS8Uint, R8G8B8A8Unorm, StaticFormat},
        usage::{DepthAttachment, Sampled, SampledStorage, StaticUsage, Storage},
    };
    use crate::command_buffer::image_state::ImageState;
    use crate::queue::Kind;
    use crate::vk::Handle;
    use glam::{IVec2, UVec2};

    /// A view over a fake image: layout tracking and region math need no device.
    fn view<F: StaticFormat, U: StaticUsage>(
        layout: &AtomicU64,
        mips: std::ops::Range<u32>,
    ) -> ImageView<'_, F, U> {
        ImageView {
            image: vk::Image::from_raw(42),
            view: vk::ImageView::null(),
            mip_range: mips.into(),
            extent: UVec2::new(64, 32),
            handle: BindlessHandle::none(),
            states: std::slice::from_ref(layout),
            format: F::default(),
            usage: U::default(),
            dim: D2,
        }
    }

    fn undefined() -> AtomicU64 {
        ImageState::UNDEFINED.atomic()
    }

    #[test]
    fn subresource_range_spans_the_views_mips() {
        let layout = undefined();
        let range = view::<R8G8B8A8Unorm, Sampled>(&layout, 2..5).subresource_range();
        assert_eq!(range.aspect_mask, vk::ImageAspectFlags::COLOR);
        assert_eq!((range.base_mip_level, range.level_count), (2, 3));
        assert_eq!((range.base_array_layer, range.layer_count), (0, 1));
    }

    #[test]
    fn subresource_layers_name_one_aspect() {
        let layout = undefined();
        let layers = view::<D32SfloatS8Uint, DepthAttachment>(&layout, 0..1).subresource_layers(3);
        assert_eq!(layers.aspect_mask, vk::ImageAspectFlags::DEPTH);
        let color = view::<R8G8B8A8Unorm, Sampled>(&layout, 0..1).subresource_layers(0);
        assert_eq!(color.aspect_mask, vk::ImageAspectFlags::COLOR);
        assert_eq!(layers.mip_level, 3);
        assert_eq!((layers.base_array_layer, layers.layer_count), (0, 1));
    }

    #[test]
    fn region_starts_at_the_origin_and_can_be_moved_and_grown() {
        let layout = undefined();
        let slice = view::<R8G8B8A8Unorm, Sampled>(&layout, 0..1).region(UVec2::new(16, 8));
        assert_eq!(
            (slice.offset, slice.extend),
            (IVec2::ZERO, UVec2::new(16, 8))
        );

        let moved = slice.offset(IVec2::new(4, 2)).offset(IVec2::new(1, 1));
        assert_eq!(moved.offset, IVec2::new(5, 3));
        assert_eq!(moved.extend, UVec2::new(16, 8));

        let grown = moved.grow(UVec2::new(2, 3));
        assert_eq!(grown.extend, UVec2::new(18, 11));
        assert_eq!(grown.offset, IVec2::new(5, 3));
    }

    #[test]
    fn whole_covers_the_views_first_mip() {
        let layout = undefined();
        let slice = view::<R8G8B8A8Unorm, Sampled>(&layout, 1..2).whole();
        assert_eq!(
            (slice.offset, slice.extend),
            (IVec2::ZERO, UVec2::new(64, 32))
        );
    }

    #[test]
    fn taking_an_access_layout_reports_the_previous_one_and_stores_the_new_one() {
        let layout = undefined();
        let view = view::<R8G8B8A8Unorm, Sampled>(&layout, 0..1);
        let taken = |access: &ImageUse| {
            let mut old = None;
            access.take_states(Kind::Graphics, |_, state| {
                if state.layout != access.access.layout {
                    old = Some(state.layout)
                }
            });
            old
        };

        let first = view.access(
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::ImageLayout::GENERAL,
        );
        assert_eq!(first.access.layout, vk::ImageLayout::GENERAL);
        assert_eq!(first.access.image, vk::Image::from_raw(42));
        assert_eq!(first.access.aspect, vk::ImageAspectFlags::COLOR);
        assert_eq!(first.access.stage, vk::PipelineStageFlags2::TRANSFER);
        assert_eq!(first.access.access, vk::AccessFlags2::TRANSFER_WRITE);
        let present = view.access(
            vk::PipelineStageFlags2::empty(),
            vk::AccessFlags2::empty(),
            vk::ImageLayout::PRESENT_SRC_KHR,
        );
        // Building accesses leaves the layout alone.
        assert_eq!(ImageState::load(&layout), ImageState::UNDEFINED);

        assert_eq!(taken(&first), Some(vk::ImageLayout::UNDEFINED));
        // Same layout again: no transition.
        assert_eq!(taken(&first), None);
        assert_eq!(taken(&present), Some(vk::ImageLayout::GENERAL));
        assert_eq!(
            ImageState::load(&layout),
            ImageState::taken(vk::ImageLayout::PRESENT_SRC_KHR, Kind::Graphics)
        );
    }

    /// Views are `Copy`; all copies (and casts) of a view must track the same layouts.
    #[test]
    fn copies_and_casts_share_the_layout_state() {
        let layout = undefined();
        let a = view::<R8G8B8A8Unorm, SampledStorage>(&layout, 0..1);
        let b = a;
        let c = a.cast(R8G8B8A8Unorm, Storage);
        let taken = |access: ImageUse| {
            let mut old = None;
            access.take_states(Kind::Graphics, |_, state| {
                if state.layout != access.access.layout {
                    old = Some(state.layout)
                }
            });
            old
        };

        let general = vk::ImageLayout::GENERAL;
        let (stage, read) = (
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_READ,
        );
        taken(a.access(stage, vk::AccessFlags2::TRANSFER_WRITE, general));
        let from_copy = b.access(stage, read, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        assert_eq!(taken(from_copy), Some(general));
        let from_cast = c.access(stage, read, general);
        assert_eq!(
            taken(from_cast),
            Some(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        );
    }

    /// A cast may drop usages but never add one or change the format.
    #[test]
    fn casts_only_narrow() {
        let layout = undefined();
        let a = view::<R8G8B8A8Unorm, SampledStorage>(&layout, 0..1);
        a.cast(R8G8B8A8Unorm, Sampled);
        a.cast(R8G8B8A8Unorm, SampledStorage);
        assert!(std::panic::catch_unwind(|| a.cast(R8G8B8A8Unorm, DepthAttachment)).is_err());
        assert!(std::panic::catch_unwind(|| a.cast(D32SfloatS8Uint, Sampled)).is_err());
    }

    #[test]
    fn copies_check_the_buffer_holds_the_texels() {
        let layout = undefined();
        let slice = view::<R8G8B8A8Unorm, Sampled>(&layout, 0..1).region(UVec2::new(4, 2));
        check_copy_size(&slice, 32, "destination");
        check_copy_size(&slice, 64, "destination");
        let short = std::panic::catch_unwind(|| check_copy_size(&slice, 31, "destination"));
        assert!(short.is_err());
    }

    #[test]
    fn blit_region_spans_offset_to_offset_plus_extent() {
        let (a, b) = (undefined(), undefined());
        let src = view::<R8G8B8A8Unorm, Sampled>(&a, 0..1)
            .region(UVec2::new(16, 8))
            .offset(IVec2::new(4, 2));
        let dst = view::<R8G8B8A8Unorm, Sampled>(&b, 0..1)
            .region(UVec2::new(32, 16))
            .offset(IVec2::new(10, 20));
        let region = blit_region(&src, &dst);

        let corner = |o: vk::Offset3D| (o.x, o.y, o.z);
        assert_eq!(corner(region.src_offsets[0]), (4, 2, 0));
        assert_eq!(corner(region.src_offsets[1]), (20, 10, 1));
        assert_eq!(corner(region.dst_offsets[0]), (10, 20, 0));
        assert_eq!(corner(region.dst_offsets[1]), (42, 36, 1));
        assert_eq!(region.src_subresource.mip_level, 0);
        assert_eq!(
            region.src_subresource.aspect_mask,
            vk::ImageAspectFlags::COLOR
        );
    }
}

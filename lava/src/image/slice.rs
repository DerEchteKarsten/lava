//! Borrowed image views of a mip range (over every aspect) and their slices, the only image types commands take (bounded by the traits of `kind`), their accesses (shaders' always in `GENERAL`), and the image transfer commands (clear, blit, copies to and from buffers; blits and copies track only the mip they touch)
use std::{iter, range::Range, sync::atomic::AtomicU64};

use crate::vk;
use bytemuck::Pod;
use lava_macros::validation_trace;

use crate::{
    bindless::{Allocatable, ImageHandle},
    buffer::slice::BufferSlice,
    command_buffer::{BufferAccess, CommandBuffer, ImageAccess, ImageUse},
    queue::{HasCompute, HasGraphics, Kind, QueueKind, FamilyInfo},
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

/// A borrowed view of `mip_range` of an image, over every aspect: the image type commands,
/// bindless binds and generated bindings take.
#[derive(Clone, Copy, Debug)]
pub struct ImageView<'a, F: Format, U: ImageUsage, D: Dim = D2> {
    pub(crate) image: vk::Image,
    pub(crate) view: vk::ImageView,
    pub(crate) mip_range: Range<u32>,
    /// The extent of mip `mip_range.start`.
    pub(crate) extent: D::Extent,
    pub(crate) handle: ImageHandle,
    /// The image's tracked states: one for all its mips, or one per mip of the image.
    pub(crate) states: &'a [AtomicU64],
    pub(crate) format: F,
    pub(crate) usage: U,
    pub(crate) dim: D,
}

impl<'a, F: Format, U: ImageUsage, D: Dim> ImageView<'a, F, U, D> {
    pub fn image(&self) -> vk::Image {
        self.image
    }

    /// Its `VkImageView`, which only images with an attachment usage have (null otherwise):
    /// descriptors are written without one.
    pub fn view(&self) -> vk::ImageView {
        self.view
    }

    pub fn mip_range(&self) -> Range<u32> {
        self.mip_range
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
                self.image,
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
            base_array_layer: 0,
            layer_count: 1,
        }
    }
    /// The layers of `mip_level` a copy or blit addresses: one aspect, depth for a combined
    /// depth-stencil format.
    pub fn subresource_layers(&self, mip_level: u32) -> vk::ImageSubresourceLayers {
        vk::ImageSubresourceLayers {
            aspect_mask: single_aspect(self.format.aspects()),
            mip_level,
            base_array_layer: 0,
            layer_count: 1,
        }
    }
    /// All of mip `mip_range.start`.
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

    /// The same view typed as `format` and `usage` (see `Image::cast`).
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
                image: self.image,
                layout,
                aspect: self.format.aspects(),
                mip_range: self.mip_range,
            },
            states: self.states,
        }
    }

    /// The `GENERAL` access of a transfer that touches only the first mip: just that mip where
    /// the image tracks its mips one by one, else all of the view's, which share one state.
    pub(crate) fn first_mip_access(
        &self,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
    ) -> ImageUse<'a> {
        let mut first = self.access(stage, access);
        if self.states.len() > 1 {
            first.access.mip_range.end = first.access.mip_range.start + 1;
        }
        first
    }
}

/// A box of texels: `extend` texels at `offset` of the first mip of `view`, made by
/// `view.whole()` or `view.region(extent)`.
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

    /// The same box, of an `ImageView`.
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
pub enum Filter {
    Nearest = 0,
    Liniear = 1,
}

impl<K: HasCompute> CommandBuffer<'_, K> {
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

impl<K: HasGraphics> CommandBuffer<'_, K> {
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
        check_box(&src, false, "source");
        check_box(&dst, false, "destination");
        self.rec.flush_pending(
            [
                src.view.first_mip_access(
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::TRANSFER_READ,
                ),
                dst.view.first_mip_access(
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::TRANSFER_WRITE,
                ),
            ]
            .into_iter(),
            iter::empty(),
        );

        let corners = |offset: <S::Dim as Dim>::Offset, extend: <S::Dim as Dim>::Extent| {
            let start = S::Dim::offset_3d(offset);
            let size = S::Dim::extent_3d(extend);
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
            src_offsets: corners(src.offset, src.extend),
            dst_offsets: corners(dst.offset, dst.extend),
            src_subresource: src.view.subresource_layers(src.view.mip_range.start),
            dst_subresource: dst.view.subresource_layers(dst.view.mip_range.start),
        }];
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

impl<K: QueueKind> CommandBuffer<'_, K> {
    /// Copies the tightly packed texels of `src` into the box `dst`.
    #[validation_trace]
    pub fn copy_buffer_to_image<T: Copy + Pod, I: TransferDstView>(
        &mut self,
        src: BufferSlice<'_, T>,
        dst: ImageSlice<I>,
    ) {
        let dst = dst.resolve();
        dst.view.aspect_check(vk::ImageAspectFlags::COLOR);
        dst.view.usage_check(vk::ImageUsageFlags::TRANSFER_DST);
        check_box(&dst, true, "destination");
        check_copy_size(&dst, src.size(), "source");
        check_family(&self.rec.family, &dst, src.address(), true);
        self.rec.flush_pending(
            iter::once(dst.view.first_mip_access(
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
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

    /// Copies the texels of the box `src` into `dst`, tightly packed. It takes a slice, never a
    /// view:
    /// ```compile_fail
    /// # use lava::{buffer::slice::BufferSlice, command_buffer::CommandBuffer, queue::Graphics};
    /// # use lava::image::{format::R8G8B8A8Unorm, slice::ImageView, usage::usage};
    /// fn copy(cmd: &mut CommandBuffer<Graphics>, view: ImageView<R8G8B8A8Unorm, usage![TransferSrc]>, dst: BufferSlice<u8>) {
    ///     cmd.copy_image_to_buffer(view, dst);
    /// }
    /// ```
    /// ```
    /// # use lava::{buffer::slice::BufferSlice, command_buffer::CommandBuffer, queue::Graphics};
    /// # use lava::image::{format::R8G8B8A8Unorm, slice::ImageView, usage::usage};
    /// fn copy(cmd: &mut CommandBuffer<Graphics>, view: ImageView<R8G8B8A8Unorm, usage![TransferSrc]>, dst: BufferSlice<u8>) {
    ///     cmd.copy_image_to_buffer(view.whole(), dst);
    /// }
    /// ```
    #[validation_trace]
    pub fn copy_image_to_buffer<I: TransferSrcView, T: Copy + Pod>(
        &mut self,
        src: ImageSlice<I>,
        dst: BufferSlice<'_, T>,
    ) {
        let src = src.resolve();
        src.view.aspect_check(vk::ImageAspectFlags::COLOR);
        src.view.usage_check(vk::ImageUsageFlags::TRANSFER_SRC);
        check_box(&src, true, "source");
        check_copy_size(&src, dst.size(), "destination");
        check_family(&self.rec.family, &src, dst.address(), false);
        self.rec.flush_pending(
            iter::once(src.view.first_mip_access(
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_READ,
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

/// Panics unless the box lies inside its mip and, for a copy, has texels and covers whole blocks
/// of a block-compressed format.
#[track_caller]
fn check_box<F: Format, U: ImageUsage, D: Dim>(
    slice: &ImageSlice<ImageView<'_, F, U, D>>,
    copy: bool,
    what: &str,
) {
    let (offset, extent) = (D::offset_3d(slice.offset), D::extent_3d(slice.extend));
    let mip = D::extent_3d(slice.view.extent);
    let axes = [
        (offset.x, extent.width, mip.width),
        (offset.y, extent.height, mip.height),
        (offset.z, extent.depth, mip.depth),
    ];
    let inside = (axes.iter()).all(|&(o, e, m)| o >= 0 && o as u64 + e as u64 <= m as u64);
    assert!(
        inside,
        "the {what} box of {:?} texels at {:?} lies outside its mip of {:?}",
        slice.extend, slice.offset, slice.view.extent
    );
    if !copy {
        return;
    }
    assert!(
        axes.iter().all(|&(_, e, _)| e > 0),
        "the {what} box of {:?} texels is empty",
        slice.extend
    );
    if let Some((side, _)) = bc_block(slice.view.format.vk_format()) {
        let whole = |&(o, e, m): &(i32, u32, u32)| {
            o as u32 % side == 0 && (e % side == 0 || o as u32 + e == m)
        };
        assert!(
            axes[..2].iter().all(whole),
            "copies of {:?} cover whole {side}x{side} blocks (or reach the mip's edge), not \
             {:?} texels at {:?} of a mip of {:?}",
            slice.view.format.vk_format(),
            slice.extend,
            slice.offset,
            slice.view.extent
        );
    }
}

/// With address commands neither the driver nor the validation layer checks a copy's buffer size.
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

#[track_caller]
fn check_family<F: Format, U: ImageUsage, D: Dim>(
    family: &FamilyInfo,
    slice: &ImageSlice<ImageView<'_, F, U, D>>,
    address: u64,
    into_image: bool,
) {
    let depth_stencil = vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL;
    assert!(
        family.kind == Kind::Graphics
            || !into_image
            || !slice.view.format.aspects().intersects(depth_stencil),
        "copies into depth or stencil images are recorded on the graphics family"
    );
    if family.kind != Kind::Transfer {
        return;
    }
    assert!(
        address % 4 == 0,
        "a transfer family copies from and to 4-byte aligned buffer addresses, not {address:#x}"
    );
    let (offset, extent) = (D::offset_3d(slice.offset), D::extent_3d(slice.extend));
    let mip = D::extent_3d(slice.view.extent);
    let granularity = family.image_granularity;
    let axes = [
        (granularity.width, offset.x, extent.width, mip.width),
        (granularity.height, offset.y, extent.height, mip.height),
        (granularity.depth, offset.z, extent.depth, mip.depth),
    ];
    let fits = axes.into_iter().all(|(g, offset, extent, mip)| match g {
        0 => offset == 0 && extent == mip,
        g => {
            offset as u32 % g == 0 && (extent % g == 0 || offset as u32 + extent == mip)
        }
    });
    assert!(
        fits,
        "a transfer family copies boxes of its image granularity {granularity:?}, not \
         {extent:?} at {offset:?} of a mip of {mip:?}"
    );
}

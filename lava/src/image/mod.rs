//! Format-, usage- and dimension-typed GPU images with one view over all mips or one per mip (plus depth-only views for depth-stencil formats, `DepthSplit`), host uploads (`host-image-copy`), type erasure, and their destruction once no frame can reach them
use std::{
    iter,
    mem::{self, ManuallyDrop},
    range::Range,
    sync::atomic::AtomicU32,
};

use crate::error::{Error, Result};
use crate::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use lava_macros::validation_trace;

use crate::{
    bindless::{self, BindlessHandle},
    epoch,
    image::{
        format::{Format, StaticFormat, bc_block},
        slice::ImageView,
        usage::{ImageUsage, StaticUsage},
    },
    state::Ctx,
    vkobjects::queue::forget_image,
};

pub mod dim;
pub mod format;
pub mod kind;
pub mod slice;
pub mod usage;

pub use dim::{D2, D3, Dim};
pub use kind::{
    ColorTarget, DepthTarget, ImageRef, SampledImage, SampledView, ShaderViews, StorageImage,
    StorageView, TransferDstView, TransferSrcView,
};

/// The `vk::Image` and its memory, shared by every kind of image. It has no public
/// constructor, so only lava implements [`Views`].
#[derive(Debug)]
pub struct RawImage<F: Format, U: ImageUsage, D: Dim> {
    image: vk::Image,
    allocation: Allocation,
    extent: D::Extent,
    mip_levels: u32,
    format: F,
    usage: U,
}

impl<F: Format, U: ImageUsage, D: Dim> RawImage<F, U, D> {
    fn new(format: F, usage: U, extent: D::Extent, mip_levels: u32) -> Result<Self> {
        check_features(format.vk_format(), usage.flags())?;
        let create_info = vk::ImageCreateInfo {
            array_layers: 1,
            extent: D::extent_3d(extent),
            format: format.vk_format(),
            image_type: D::IMAGE_TYPE,
            mip_levels,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: usage.flags(),
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };

        let image = unsafe { Ctx::device().create_image(&create_info, None) }?;
        let mut dedicated = vk::MemoryDedicatedRequirements::default();
        let requirements = {
            let mut requirements = vk::MemoryRequirements2::default().push_next(&mut dedicated);
            unsafe {
                Ctx::device().get_image_memory_requirements2(
                    &vk::ImageMemoryRequirementsInfo2::default().image(image),
                    &mut requirements,
                )
            };
            requirements.memory_requirements
        };
        let allocation_scheme = if dedicated.requires_dedicated_allocation == vk::TRUE
            || dedicated.prefers_dedicated_allocation == vk::TRUE
        {
            AllocationScheme::DedicatedImage(image)
        } else {
            AllocationScheme::GpuAllocatorManaged
        };

        let desc = AllocationCreateDesc {
            allocation_scheme,
            linear: false,
            location: gpu_allocator::MemoryLocation::GpuOnly,
            name: "ImageMemory",
            requirements,
        };
        let allocation = match Ctx::allocator().allocate(&desc) {
            Ok(allocation) => allocation,
            Err(err) => {
                unsafe { Ctx::device().destroy_image(image, None) };
                return Err(err.into());
            }
        };
        // From here on `Drop` frees the image and its memory.
        let raw = Self {
            image,
            allocation,
            extent,
            mip_levels,
            format,
            usage,
        };
        unsafe {
            Ctx::device().bind_image_memory(
                image,
                raw.allocation.memory(),
                raw.allocation.offset(),
            )?
        };
        Ok(raw)
    }

    fn create_view(
        &self,
        mips: Range<u32>,
        aspect_mask: vk::ImageAspectFlags,
    ) -> Result<vk::ImageView> {
        let info = vk::ImageViewCreateInfo::default()
            .image(self.image)
            .view_type(D::VIEW_TYPE)
            .format(self.format.vk_format())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: mips.start,
                level_count: mips.end - mips.start,
                base_array_layer: 0,
                layer_count: 1,
            });
        Ok(unsafe { Ctx::device().create_image_view(&info, None)? })
    }

    fn cast<NF: Format, NU: ImageUsage>(self, format: NF, usage: NU) -> RawImage<NF, NU, D> {
        let mut this = ManuallyDrop::new(self);
        RawImage {
            image: this.image,
            allocation: mem::take(&mut this.allocation),
            extent: this.extent,
            mip_levels: this.mip_levels,
            format,
            usage,
        }
    }
}

impl<F: Format, U: ImageUsage, D: Dim> Drop for RawImage<F, U, D> {
    /// Destroyed after the views: `Image` declares `views` before `raw`, and destructors run
    /// in drop order (`epoch`).
    fn drop(&mut self) {
        let (image, allocation) = (self.image, mem::take(&mut self.allocation));
        epoch::retire(move || {
            forget_image(image);
            unsafe { Ctx::device().destroy_image(image, None) };
            if let Err(err) = Ctx::allocator().free(allocation) {
                tracing::error!(%err, "failed to free image allocation");
            }
        });
    }
}

/// What an image keeps per view: [`Single`], [`PerMip`], or for a combined depth-stencil format
/// [`DepthSplit`] and [`DepthSplitMips`]. Only lava implements it.
pub trait Views: crate::sealed::Sealed + std::fmt::Debug + Send + Sync + 'static + Sized {
    #[doc(hidden)]
    fn create<F: Format, U: ImageUsage, D: Dim>(raw: &RawImage<F, U, D>) -> Result<Self>;
    /// The view over every mip and aspect, its handle and the layouts it covers.
    #[doc(hidden)]
    fn whole(&self) -> (vk::ImageView, BindlessHandle, &[AtomicU32]);
    #[doc(hidden)]
    fn whole_handle(&mut self) -> &mut BindlessHandle;
    /// The view of every mip that descriptors get: the whole view, or its depth-only view for
    /// the split views of a depth-stencil format.
    #[doc(hidden)]
    fn shader_view(&self) -> vk::ImageView {
        self.whole().0
    }
}

/// Views with one view per mip besides the whole one: [`PerMip`] and [`DepthSplitMips`].
pub trait MipViews: Views {
    /// Mip `level`'s view over every aspect, the one descriptors get, its handle and layout.
    #[doc(hidden)]
    fn mip(&self, level: u32) -> (vk::ImageView, vk::ImageView, BindlessHandle, &AtomicU32);
    #[doc(hidden)]
    fn mip_handle_mut(&mut self, level: u32) -> &mut BindlessHandle;
}

/// One view over every mip, with one handle and one layout for all of them: render targets,
/// and textures, whose mips are only sampled together and written by the host.
#[derive(Debug)]
pub struct Single {
    view: vk::ImageView,
    handle: BindlessHandle,
    layout: AtomicU32,
}

/// The view over every mip plus one view per mip, each with its handle, and a layout per mip:
/// for images whose mips the GPU accesses one at a time.
#[derive(Debug)]
pub struct PerMip {
    view: vk::ImageView,
    handle: BindlessHandle,
    mips: Box<[MipView]>,
    layouts: Box<[AtomicU32]>,
}

#[derive(Debug)]
struct MipView {
    view: vk::ImageView,
    handle: BindlessHandle,
}

/// [`Single`] for a combined depth-stencil format, plus a depth-only view of every mip for
/// descriptors: a view a shader reads can't have both aspects. Attachments get the view of
/// both.
#[derive(Debug)]
pub struct DepthSplit {
    views: Single,
    shader: vk::ImageView,
}

/// [`PerMip`] for a combined depth-stencil format, plus a depth-only view of the whole image
/// and of each mip for descriptors (see [`DepthSplit`]).
#[derive(Debug)]
pub struct DepthSplitMips {
    views: PerMip,
    shader: vk::ImageView,
    mip_shaders: Box<[vk::ImageView]>,
}

fn undefined() -> AtomicU32 {
    AtomicU32::new(vk::ImageLayout::UNDEFINED.as_raw() as u32)
}

impl crate::sealed::Sealed for Single {}
impl Views for Single {
    fn create<F: Format, U: ImageUsage, D: Dim>(raw: &RawImage<F, U, D>) -> Result<Self> {
        let aspects = raw.format.aspects();
        Ok(Single {
            view: raw.create_view((0..raw.mip_levels).into(), aspects)?,
            handle: BindlessHandle::none(),
            layout: undefined(),
        })
    }
    fn whole(&self) -> (vk::ImageView, BindlessHandle, &[AtomicU32]) {
        (self.view, self.handle, std::slice::from_ref(&self.layout))
    }
    fn whole_handle(&mut self) -> &mut BindlessHandle {
        &mut self.handle
    }
}

impl crate::sealed::Sealed for PerMip {}
impl Views for PerMip {
    fn create<F: Format, U: ImageUsage, D: Dim>(raw: &RawImage<F, U, D>) -> Result<Self> {
        let aspects = raw.format.aspects();
        // Built up front, so `Drop` destroys whatever was created before an error.
        let mut views = PerMip {
            view: vk::ImageView::null(),
            handle: BindlessHandle::none(),
            mips: Box::new([]),
            layouts: (0..raw.mip_levels).map(|_| undefined()).collect(),
        };
        views.view = raw.create_view((0..raw.mip_levels).into(), aspects)?;
        let mut mips = Vec::with_capacity(raw.mip_levels as usize);
        for mip in 0..raw.mip_levels {
            match raw.create_view((mip..mip + 1).into(), aspects) {
                Ok(view) => mips.push(MipView {
                    view,
                    handle: BindlessHandle::none(),
                }),
                Err(err) => {
                    views.mips = mips.into();
                    return Err(err);
                }
            }
        }
        views.mips = mips.into();
        Ok(views)
    }
    fn whole(&self) -> (vk::ImageView, BindlessHandle, &[AtomicU32]) {
        (self.view, self.handle, &self.layouts)
    }
    fn whole_handle(&mut self) -> &mut BindlessHandle {
        &mut self.handle
    }
}

impl MipViews for PerMip {
    fn mip(&self, level: u32) -> (vk::ImageView, vk::ImageView, BindlessHandle, &AtomicU32) {
        let mip = &self.mips[level as usize];
        (
            mip.view,
            mip.view,
            mip.handle,
            &self.layouts[level as usize],
        )
    }
    fn mip_handle_mut(&mut self, level: u32) -> &mut BindlessHandle {
        &mut self.mips[level as usize].handle
    }
}

/// An error unless `raw` has a combined depth-stencil format, which split views are for.
fn check_depth_stencil<F: Format, U: ImageUsage, D: Dim>(raw: &RawImage<F, U, D>) -> Result<()> {
    let aspects = raw.format.aspects();
    if aspects.contains(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL) {
        return Ok(());
    }
    Err(Error::message(format!(
        "split depth views are for combined depth-stencil formats, not {:?}; use `Single` or `PerMip`",
        raw.format.vk_format()
    )))
}

impl crate::sealed::Sealed for DepthSplit {}
impl Views for DepthSplit {
    fn create<F: Format, U: ImageUsage, D: Dim>(raw: &RawImage<F, U, D>) -> Result<Self> {
        check_depth_stencil(raw)?;
        let views = Single::create(raw)?;
        // On an error `views` is dropped, which destroys its view.
        let shader = raw.create_view((0..raw.mip_levels).into(), vk::ImageAspectFlags::DEPTH)?;
        Ok(DepthSplit { views, shader })
    }
    fn whole(&self) -> (vk::ImageView, BindlessHandle, &[AtomicU32]) {
        self.views.whole()
    }
    fn whole_handle(&mut self) -> &mut BindlessHandle {
        self.views.whole_handle()
    }
    fn shader_view(&self) -> vk::ImageView {
        self.shader
    }
}

impl crate::sealed::Sealed for DepthSplitMips {}
impl Views for DepthSplitMips {
    fn create<F: Format, U: ImageUsage, D: Dim>(raw: &RawImage<F, U, D>) -> Result<Self> {
        check_depth_stencil(raw)?;
        // Built up front, so `Drop` destroys whatever was created before an error.
        let mut views = DepthSplitMips {
            views: PerMip::create(raw)?,
            shader: vk::ImageView::null(),
            mip_shaders: Box::new([]),
        };
        let depth = vk::ImageAspectFlags::DEPTH;
        views.shader = raw.create_view((0..raw.mip_levels).into(), depth)?;
        let mut mips = Vec::with_capacity(raw.mip_levels as usize);
        for mip in 0..raw.mip_levels {
            match raw.create_view((mip..mip + 1).into(), depth) {
                Ok(view) => mips.push(view),
                Err(err) => {
                    views.mip_shaders = mips.into();
                    return Err(err);
                }
            }
        }
        views.mip_shaders = mips.into();
        Ok(views)
    }
    fn whole(&self) -> (vk::ImageView, BindlessHandle, &[AtomicU32]) {
        self.views.whole()
    }
    fn whole_handle(&mut self) -> &mut BindlessHandle {
        self.views.whole_handle()
    }
    fn shader_view(&self) -> vk::ImageView {
        self.shader
    }
}

impl MipViews for DepthSplitMips {
    fn mip(&self, level: u32) -> (vk::ImageView, vk::ImageView, BindlessHandle, &AtomicU32) {
        let (view, _, handle, layout) = self.views.mip(level);
        (view, self.mip_shaders[level as usize], handle, layout)
    }
    fn mip_handle_mut(&mut self, level: u32) -> &mut BindlessHandle {
        self.views.mip_handle_mut(level)
    }
}

fn release(view: vk::ImageView, handle: BindlessHandle) {
    bindless::release(handle);
    unsafe { Ctx::device().destroy_image_view(view, None) };
}

impl Drop for Single {
    fn drop(&mut self) {
        let (view, handle) = (self.view, self.handle);
        epoch::retire(move || release(view, handle));
    }
}

impl Drop for PerMip {
    fn drop(&mut self) {
        let (view, handle, mips) = (self.view, self.handle, mem::take(&mut self.mips));
        epoch::retire(move || {
            release(view, handle);
            for mip in &mips {
                release(mip.view, mip.handle);
            }
        });
    }
}

/// Destroys the depth-only views; the wrapped views release the handles and their own views.
fn destroy_views(views: impl IntoIterator<Item = vk::ImageView>) {
    for view in views {
        unsafe { Ctx::device().destroy_image_view(view, None) };
    }
}

impl Drop for DepthSplit {
    fn drop(&mut self) {
        let shader = self.shader;
        epoch::retire(move || destroy_views([shader]));
    }
}

impl Drop for DepthSplitMips {
    fn drop(&mut self) {
        let (shader, mips) = (self.shader, mem::take(&mut self.mip_shaders));
        epoch::retire(move || destroy_views(iter::once(shader).chain(mips)));
    }
}

/// An image of format `F`, usage `U` and dimension `D`. Its format and usage are types
/// (`format::R8Unorm`, `usage::Sampled`), which the command buffer's bounds check at compile
/// time, or `format::Dynamic`/`usage::Dynamic` values, which it checks at runtime
/// (`aspect_check`, `texel_check`, `usage_check`). A [`Single`] image (the default) has one view
/// over all its mips, a [`MipChain`] also one per mip; a combined depth-stencil image that
/// shaders read takes [`DepthSplit`] or [`DepthSplitMips`] instead (`kind::SampledImage`).
#[derive(Debug)]
pub struct Image<F: Format, U: ImageUsage, D: Dim = D2, V: Views = Single> {
    // Dropped in this order: the views and their handles first, then the image.
    views: V,
    raw: RawImage<F, U, D>,
}

impl<F: Format, U: ImageUsage, D: Dim, V: Views> crate::sealed::Sealed for Image<F, U, D, V> {}

/// An image whose mips the GPU accesses one at a time ([`Image::mip_view`]).
pub type MipChain<F, U, D = D2> = Image<F, U, D, PerMip>;

/// An image whose format and usage are runtime values (`Image::erase`). The dimension stays a
/// type: shaders declare it, so nothing needs it at runtime.
pub type DynamicImage<D = D2, V = Single> = Image<format::Dynamic, usage::Dynamic, D, V>;

impl<F: StaticFormat, U: StaticUsage, D: Dim, V: Views> Image<F, U, D, V> {
    /// Creates an image that is not bound in any bindless set: its handles are null.
    #[validation_trace]
    pub fn new(extent: D::Extent) -> Result<Self> {
        Self::with_mip_levels(extent, 1)
    }
    #[validation_trace]
    pub fn with_mip_levels(extent: D::Extent, mip_levels: u32) -> Result<Self> {
        Self::with_params(F::default(), U::default(), extent, mip_levels)
    }
}

impl<F: Format, U: ImageUsage, D: Dim, V: Views> Image<F, U, D, V> {
    /// Like `with_mip_levels`, for any format and usage, `Dynamic` ones included.
    #[validation_trace]
    pub fn with_params(format: F, usage: U, extent: D::Extent, mip_levels: u32) -> Result<Self> {
        let raw = RawImage::new(format, usage, extent, mip_levels)?;
        Ok(Image {
            views: V::create(&raw)?,
            raw,
        })
    }

    /// The extent of mip 0.
    pub fn extent(&self) -> D::Extent {
        self.raw.extent
    }

    pub fn mip_levels(&self) -> u32 {
        self.raw.mip_levels
    }

    pub fn mip_extent(&self, level: u32) -> D::Extent {
        D::mip_extent(self.raw.extent, level)
    }

    pub fn format(&self) -> F {
        self.raw.format
    }

    pub fn usage(&self) -> U {
        self.raw.usage
    }

    /// The whole view's handle.
    pub fn handle(&self) -> BindlessHandle {
        self.views.whole().1
    }

    /// Panics unless the format has the `required` aspects; does nothing for a static format.
    pub fn aspect_check(&self, required: vk::ImageAspectFlags) {
        self.raw.format.aspect_check(required);
    }

    /// Panics unless the format's texels have the type `T`; does nothing for a static format.
    pub fn texel_check<T: 'static>(&self) {
        self.raw.format.texel_check::<T>();
    }

    /// Panics unless the usage has every flag of `required`; does nothing for a static usage.
    pub fn usage_check(&self, required: vk::ImageUsageFlags) {
        self.raw.usage.check(required);
    }

    /// The view over every mip.
    pub fn whole_view(&self) -> ImageView<'_, F, U, D> {
        let (view, handle, layouts) = self.views.whole();
        self.view(view, (0..self.raw.mip_levels).into(), handle, layouts)
    }

    fn view<'a>(
        &'a self,
        view: vk::ImageView,
        mip_range: Range<u32>,
        handle: BindlessHandle,
        layouts: &'a [AtomicU32],
    ) -> ImageView<'a, F, U, D> {
        ImageView {
            image: self.raw.image,
            view,
            mip_range,
            extent: self.mip_extent(mip_range.start),
            handle,
            layouts,
            format: self.raw.format,
            usage: self.raw.usage,
            dim: D::default(),
        }
    }

    /// The same image typed as `format` and `usage`, which must be the format it was created
    /// with and no more than its usage (an image of several usages as one of them, a static
    /// type as `Dynamic`); a cast that claims more panics.
    pub fn cast<NF: Format, NU: ImageUsage>(self, format: NF, usage: NU) -> Image<NF, NU, D, V> {
        slice::cast_check(
            (self.raw.format.vk_format(), self.raw.usage.flags()),
            (format.vk_format(), usage.flags()),
        );
        Image {
            views: self.views,
            raw: self.raw.cast(format, usage),
        }
    }

    /// The same image with its format and usage as runtime values, e.g. to keep images of
    /// different types together. Commands check them at runtime.
    pub fn erase(self) -> DynamicImage<D, V> {
        let format = format::Dynamic(self.raw.format.vk_format());
        let usage = usage::Dynamic(self.raw.usage.flags());
        self.cast(format, usage)
    }
}

impl<F: Format, U: ImageUsage, D: Dim, V: Views> Image<F, U, D, V> {
    /// Binds the whole view at `index` of the sampled set, right away:
    /// `Image::new(extent)?.bind_sampled(index)`. A combined depth-stencil image needs
    /// [`DepthSplit`] views for it, whose depth-only view is bound.
    #[validation_trace]
    pub fn bind_sampled(mut self, index: u32) -> Self
    where
        Self: SampledImage,
    {
        self.views.whole_handle().descriptor_index_set0 = index;
        let view = (self.views.shader_view(), self.raw.usage.flags());
        bindless::bind_raw([view], [BindlessHandle::sampled(index)]);
        self
    }

    /// Binds the whole view at `index` of the storage set, right away. An image with both
    /// usages can be bound in both sets: `.bind_sampled(a).bind_storage(b)`.
    #[validation_trace]
    pub fn bind_storage(mut self, index: u32) -> Self
    where
        Self: StorageImage,
    {
        self.views.whole_handle().descriptor_index_set1 = index;
        let view = (self.views.shader_view(), self.raw.usage.flags());
        bindless::bind_raw([view], [BindlessHandle::storage(index)]);
        self
    }
}

impl<F: Format, U: ImageUsage, D: Dim, V: MipViews> Image<F, U, D, V> {
    /// The view of mip `level` alone, sized `mip_extent(level)`: for a storage write, an
    /// attachment or a copy of one mip. The whole view samples across mips.
    pub fn mip_view(&self, level: u32) -> ImageView<'_, F, U, D> {
        let (view, _, handle, layout) = self.views.mip(level);
        let layouts = std::slice::from_ref(layout);
        self.view(view, (level..level + 1).into(), handle, layouts)
    }

    /// Binds mip `i`'s view at `first + i` of the sampled (`sampled`) or the storage set.
    fn bind_mips(mut self, first: u32, sampled: bool) -> Self {
        let levels = 0..self.mip_levels();
        for level in levels.clone() {
            let handle = self.views.mip_handle_mut(level);
            if sampled {
                handle.descriptor_index_set0 = first + level;
            } else {
                handle.descriptor_index_set1 = first + level;
            }
        }
        let usage = self.raw.usage.flags();
        bindless::bind_raw(
            levels.clone().map(|level| (self.views.mip(level).1, usage)),
            levels.map(|level| match sampled {
                true => BindlessHandle::sampled(first + level),
                false => BindlessHandle::storage(first + level),
            }),
        );
        self
    }

    /// Binds mip `i`'s view at `first + i` of the sampled set, right away.
    #[validation_trace]
    pub fn bind_mips_sampled(self, first: u32) -> Self
    where
        Self: SampledImage,
    {
        self.bind_mips(first, true)
    }

    /// Binds mip `i`'s view at `first + i` of the storage set, right away.
    #[validation_trace]
    pub fn bind_mips_storage(self, first: u32) -> Self
    where
        Self: StorageImage,
    {
        self.bind_mips(first, false)
    }
}

/// `MissingFeature` for an image the device can't have: a BC format without `bc-compression`,
/// host copies without `host-image-copy` (an error message where lava was compiled without
/// them, which only a dynamic format or usage can ask for).
fn check_features(format: vk::Format, usage: vk::ImageUsageFlags) -> Result<()> {
    if bc_block(format).is_some() {
        #[cfg(feature = "bc-compression")]
        gate::BcCompression::check()?;
        #[cfg(not(feature = "bc-compression"))]
        return Err(Error::message(format!(
            "{format:?} needs lava's `bc-compression` feature"
        )));
    }
    if usage.contains(vk::ImageUsageFlags::HOST_TRANSFER_EXT) {
        #[cfg(feature = "host-image-copy")]
        gate::HostImageCopy::check()?;
        #[cfg(not(feature = "host-image-copy"))]
        return Err(Error::message(
            "host copy usage needs lava's `host-image-copy` feature",
        ));
    }
    Ok(())
}

#[cfg(all(feature = "bc-compression", not(feature = "host-image-copy")))]
use crate::features::Feature;
#[cfg(any(feature = "bc-compression", feature = "host-image-copy"))]
use crate::features::gate;
#[cfg(feature = "host-image-copy")]
use crate::{
    command_buffer::{ImageAccess, ImageUse},
    features::Feature,
    image::usage::IsHostCopy,
    state::Functions,
};

#[cfg(feature = "host-image-copy")]
impl<F: Format, U: IsHostCopy, D: Dim, V: Views> Image<F, U, D, V> {
    /// Writes a whole mip level from the host, right away. Like anything a command buffer
    /// doesn't record, it isn't synchronised with the GPU: frames in flight that read the image
    /// may see the old or the new texels, or a mix.
    #[validation_trace]
    pub fn copy_from(&mut self, data: &[u8], mip_level: u32) -> Result<()> {
        self.check_host_copy(data, mip_level)?;
        let extent = self.mip_extent(mip_level);
        self.check_data_len(data, extent, || format!("mip level {mip_level}"))?;
        self.host_copy(data, mip_level, D::ZERO, extent)
    }

    /// Like `copy_from` (also unsynchronised), but only writes the `extent` texels at `offset`
    /// of the mip level. `data` holds that box, tightly packed.
    #[validation_trace]
    pub fn copy_region_from(
        &mut self,
        data: &[u8],
        mip_level: u32,
        offset: D::Extent,
        extent: D::Extent,
    ) -> Result<()> {
        self.check_host_copy(data, mip_level)?;
        let mip_extent = self.mip_extent(mip_level);
        if !D::region_fits(mip_extent, offset, extent) {
            return Err(Error::message(format!(
                "region {extent:?} at {offset:?} does not fit mip level {mip_level} of size {mip_extent:?}"
            )));
        }
        self.check_data_len(data, extent, || "the region".to_string())?;
        self.host_copy(data, mip_level, offset, extent)
    }

    fn check_host_copy(&self, data: &[u8], mip_level: u32) -> Result<()> {
        // The image exists, so the device has host copies (`check_features`).
        self.usage_check(vk::ImageUsageFlags::HOST_TRANSFER_EXT);
        if mip_level >= self.raw.mip_levels {
            return Err(Error::message(format!(
                "mip level {mip_level} is out of range for an image with {} levels",
                self.raw.mip_levels
            )));
        }
        if data.is_empty() {
            return Err(Error::message("no texel data to copy"));
        }
        let aspects = self.raw.format.aspects();
        if slice::single_aspect(aspects) != aspects {
            return Err(Error::message(
                "host copies of combined depth-stencil formats aren't supported: their texel data has no single layout",
            ));
        }
        Ok(())
    }

    fn check_data_len(
        &self,
        data: &[u8],
        extent: D::Extent,
        what: impl FnOnce() -> String,
    ) -> Result<()> {
        let format = self.raw.format;
        let expected = expected_data_len(
            D::extent_3d(extent),
            format.texel_size(),
            bc_block(format.vk_format()),
        );
        match expected {
            Some(expected) if data.len() != expected => Err(Error::message(format!(
                "{} holds {expected} bytes but {} were given",
                what(),
                data.len()
            ))),
            _ => Ok(()),
        }
    }

    fn host_copy(
        &mut self,
        data: &[u8],
        mip_level: u32,
        offset: D::Extent,
        extent: D::Extent,
    ) -> Result<()> {
        let image = self.raw.image;
        let aspect = self.raw.format.aspects();
        let (_, _, layouts) = self.views.whole();
        // Every mip not in `GENERAL` yet moves there, as if a command accessed the whole view.
        let to_general = ImageUse {
            access: ImageAccess {
                stage: vk::PipelineStageFlags2::HOST,
                access: vk::AccessFlags2::HOST_WRITE,
                image,
                layout: vk::ImageLayout::GENERAL,
                aspect,
                mip_range: (0..self.raw.mip_levels).into(),
            },
            layouts,
        };
        let mut transitions: smallvec::SmallVec<[vk::HostImageLayoutTransitionInfoEXT; 1]> =
            smallvec::SmallVec::new();
        to_general.take_layouts(|mips, old_layout| {
            transitions.push(
                vk::HostImageLayoutTransitionInfoEXT::default()
                    .image(image)
                    .old_layout(old_layout)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: aspect,
                        base_mip_level: mips.start,
                        level_count: mips.end - mips.start,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
            )
        });
        if !transitions.is_empty() {
            unsafe { Functions::host_image_copy().transition_image_layout(&transitions)? };
        }
        let regions = [vk::MemoryToImageCopyEXT::default()
            .host_pointer(data.as_ptr().cast())
            .image_offset(D::unsigned_offset_3d(offset))
            .image_extent(D::extent_3d(extent))
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: aspect,
                mip_level,
                base_array_layer: 0,
                layer_count: 1,
            })
            // 0: the rows of `data` are as long as the region is wide, which is also right
            // for block-compressed formats, where a small mip is narrower than its one block.
            .memory_image_height(0)
            .memory_row_length(0)];
        let info = vk::CopyMemoryToImageInfoEXT::default()
            .dst_image(image)
            .dst_image_layout(vk::ImageLayout::GENERAL)
            .regions(&regions);
        unsafe { Functions::host_image_copy().copy_memory_to_image(&info)? };
        Ok(())
    }
}

/// Bytes of tightly packed texel data for a box of `extent`: texels of `texel_size` bytes,
/// or whole blocks of `(side length, bytes)` (one texel deep) when there is no fixed texel size
/// (`texel_size == 0`). `None` when the format has neither, so the length can't be checked.
pub(crate) fn expected_data_len(
    extent: vk::Extent3D,
    texel_size: usize,
    block: Option<(u32, usize)>,
) -> Option<usize> {
    let depth = extent.depth as usize;
    if texel_size != 0 {
        return Some(extent.width as usize * extent.height as usize * depth * texel_size);
    }
    let (side, bytes) = block?;
    Some(
        extent.width.div_ceil(side) as usize
            * extent.height.div_ceil(side) as usize
            * depth
            * bytes,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extent(width: u32, height: u32, depth: u32) -> vk::Extent3D {
        vk::Extent3D {
            width,
            height,
            depth,
        }
    }

    #[test]
    fn expected_data_len_is_texels_times_texel_size() {
        assert_eq!(expected_data_len(extent(8, 4, 1), 4, None), Some(128));
        assert_eq!(expected_data_len(extent(3, 5, 1), 1, None), Some(15));
        assert_eq!(expected_data_len(extent(1, 1, 1), 16, None), Some(16));
        assert_eq!(expected_data_len(extent(2, 2, 3), 2, None), Some(24));
        // Block-compressed and runtime-defined formats can't be checked.
        assert_eq!(expected_data_len(extent(8, 4, 1), 0, None), None);
    }

    #[test]
    fn block_formats_hold_whole_blocks() {
        let bc7 = bc_block(vk::Format::BC7_SRGB_BLOCK);
        assert_eq!(bc7, Some((4, 16)));
        assert_eq!(bc_block(vk::Format::BC4_UNORM_BLOCK), Some((4, 8)));
        assert_eq!(bc_block(vk::Format::R8G8B8A8_UNORM), None);
        assert_eq!(expected_data_len(extent(8, 4, 1), 0, bc7), Some(32));
        // Mips smaller than a block, or not a multiple of it, still take whole blocks.
        assert_eq!(expected_data_len(extent(1, 1, 1), 0, bc7), Some(16));
        assert_eq!(expected_data_len(extent(6, 2, 1), 0, bc7), Some(32));
        // A 3D image's blocks are one texel deep.
        assert_eq!(expected_data_len(extent(4, 4, 3), 0, bc7), Some(48));
    }
}

//! Format-, usage- and dimension-typed GPU images with one view over all mips or one per mip (`VkImageView`s only for attachments; bindless descriptors need none), host uploads (`host-image-copy`; staged ones are `Uploader::image` in `buffer/staging.rs`), type erasure, and their destruction once no frame can reach them
use std::{
    mem::{self, ManuallyDrop},
    range::Range,
    sync::atomic::AtomicU64,
};

use crate::error::{Error, Result};
use crate::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use lava_macros::validation_trace;

use crate::{
    bindless::{self, Allocatable, Combined, ImageHandle, Sampled, Storage},
    command_buffer::image_state::ImageState,
    image::{
        format::{Format, StaticFormat, bc_block},
        slice::ImageView,
        usage::{ImageUsage, StaticUsage},
    },
    global::Live,
    lifetime,
    queue::timeline::forget_image,
    state::{Ctx, Functions},
};

pub mod dim;
pub mod format;
pub mod kind;
pub mod slice;
pub mod usage;

pub use dim::{D2, D3, Dim};
pub use kind::{
    ColorTarget, DepthTarget, ImageRef, SampledImage, SampledView, StorageImage, StorageView,
    TransferDstView, TransferSrcView,
};

/// The `vk::Image` and its memory. It has no public constructor, so only lava implements
/// [`Views`].
#[derive(Debug)]
pub struct RawImage<F: Format, U: ImageUsage, D: Dim> {
    image: vk::Image,
    allocation: Allocation,
    extent: D::Extent,
    mip_levels: u32,
    format: F,
    usage: U,
    /// Last, so it counts the image until its drop finished.
    live: Live,
}

impl<F: Format, U: ImageUsage, D: Dim> RawImage<F, U, D> {
    /// A view of `mips` over every aspect where renderings need one (attachment usages), else
    /// null: descriptors are written from a view's description, without a view object.
    fn create_view(&self, mips: Range<u32>) -> Result<vk::ImageView> {
        let attachment =
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        if !self.usage.flags().intersects(attachment) {
            return Ok(vk::ImageView::null());
        }
        let info = vk::ImageViewCreateInfo::default()
            .image(self.image)
            .view_type(D::VIEW_TYPE)
            .format(self.format.vk_format())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: self.format.aspects(),
                base_mip_level: mips.start,
                level_count: mips.end - mips.start,
                base_array_layer: 0,
                layer_count: 1,
            });
        Ok(unsafe { Ctx::device().create_image_view(&info, None)? })
    }
}

impl<F: Format, U: ImageUsage, D: Dim> Drop for RawImage<F, U, D> {
    fn drop(&mut self) {
        let (image, allocation) = (self.image, mem::take(&mut self.allocation));
        lifetime::retire(move || {
            forget_image(image);
            unsafe { Ctx::device().destroy_image(image, None) };
            if let Err(err) = Ctx::allocator().free(allocation) {
                tracing::error!(%err, "failed to free image allocation");
            }
        });
    }
}

/// What an image keeps per view: [`Single`] or [`PerMip`].
pub trait Views: crate::sealed::Sealed + std::fmt::Debug + Send + Sync + 'static + Sized {
    #[doc(hidden)]
    fn create<F: Format, U: ImageUsage, D: Dim>(raw: &RawImage<F, U, D>) -> Result<Self>;
    /// The view over every mip and aspect (null without attachment usage), its handle and the
    /// states it covers.
    #[doc(hidden)]
    fn whole(&self) -> (vk::ImageView, ImageHandle, &[AtomicU64]);
    #[doc(hidden)]
    fn whole_handle(&mut self) -> &mut ImageHandle;
}

/// Views with one view per mip besides the whole one: [`PerMip`].
pub trait MipViews: Views {
    /// Mip `level`'s view over every aspect (null without attachment usage) and its handle.
    #[doc(hidden)]
    fn mip(&self, level: u32) -> (vk::ImageView, ImageHandle);
    #[doc(hidden)]
    fn mip_handle_mut(&mut self, level: u32) -> &mut ImageHandle;
}

/// One view, handle and layout for all mips: render targets and textures.
#[derive(Debug)]
pub struct Single {
    view: vk::ImageView,
    handle: ImageHandle,
    /// Boxed, so a staged upload reaches it while the image moves (`Uploader::image`).
    state: Box<AtomicU64>,
}

/// The whole view plus a view, handle and layout per mip, for images whose mips the GPU
/// accesses one at a time.
#[derive(Debug)]
pub struct PerMip {
    view: vk::ImageView,
    handle: ImageHandle,
    mips: Box<[MipView]>,
    layouts: Box<[AtomicU64]>,
}

#[derive(Debug)]
struct MipView {
    view: vk::ImageView,
    handle: ImageHandle,
}

fn undefined() -> AtomicU64 {
    ImageState::UNDEFINED.atomic()
}

impl crate::sealed::Sealed for Single {}
impl Views for Single {
    fn create<F: Format, U: ImageUsage, D: Dim>(raw: &RawImage<F, U, D>) -> Result<Self> {
        Ok(Single {
            view: raw.create_view((0..raw.mip_levels).into())?,
            handle: ImageHandle::NULL,
            state: Box::new(undefined()),
        })
    }
    fn whole(&self) -> (vk::ImageView, ImageHandle, &[AtomicU64]) {
        (self.view, self.handle, std::slice::from_ref(&*self.state))
    }
    fn whole_handle(&mut self) -> &mut ImageHandle {
        &mut self.handle
    }
}

impl crate::sealed::Sealed for PerMip {}
impl Views for PerMip {
    fn create<F: Format, U: ImageUsage, D: Dim>(raw: &RawImage<F, U, D>) -> Result<Self> {
        // Built up front, so `Drop` destroys whatever was created before an error.
        let mut views = PerMip {
            view: vk::ImageView::null(),
            handle: ImageHandle::NULL,
            mips: Box::new([]),
            layouts: (0..raw.mip_levels).map(|_| undefined()).collect(),
        };
        views.view = raw.create_view((0..raw.mip_levels).into())?;
        let mut mips = Vec::with_capacity(raw.mip_levels as usize);
        for mip in 0..raw.mip_levels {
            match raw.create_view((mip..mip + 1).into()) {
                Ok(view) => mips.push(MipView {
                    view,
                    handle: ImageHandle::NULL,
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
    fn whole(&self) -> (vk::ImageView, ImageHandle, &[AtomicU64]) {
        (self.view, self.handle, &self.layouts)
    }
    fn whole_handle(&mut self) -> &mut ImageHandle {
        &mut self.handle
    }
}

impl MipViews for PerMip {
    fn mip(&self, level: u32) -> (vk::ImageView, ImageHandle) {
        let mip = &self.mips[level as usize];
        (mip.view, mip.handle)
    }
    fn mip_handle_mut(&mut self, level: u32) -> &mut ImageHandle {
        &mut self.mips[level as usize].handle
    }
}

fn release(view: vk::ImageView, handle: ImageHandle) {
    bindless::free_handle(handle);
    if view != vk::ImageView::null() {
        unsafe { Ctx::device().destroy_image_view(view, None) };
    }
}

impl Drop for Single {
    fn drop(&mut self) {
        let (view, handle) = (self.view, self.handle);
        lifetime::retire(move || release(view, handle));
    }
}

impl Drop for PerMip {
    fn drop(&mut self) {
        let (view, handle, mips) = (self.view, self.handle, mem::take(&mut self.mips));
        lifetime::retire(move || {
            release(view, handle);
            for mip in &mips {
                release(mip.view, mip.handle);
            }
        });
    }
}

/// An image of format `F`, usage `U` and dimension `D`. Static format and usage types are
/// checked at compile time, `format::Dynamic`/`usage::Dynamic` values at runtime.
#[derive(Debug)]
pub struct Image<F: Format, U: ImageUsage, D: Dim = D2, V: Views = Single> {
    // Dropped in this order: the views and their handles first, then the image.
    views: V,
    raw: RawImage<F, U, D>,
}

impl<F: Format, U: ImageUsage, D: Dim, V: Views> Drop for Image<F, U, D, V> {
    fn drop(&mut self) {
        // Before its states go with its views.
        crate::buffer::staging::forget_image(self.raw.image);
    }
}

impl<F: Format, U: ImageUsage, D: Dim, V: Views> crate::sealed::Sealed for Image<F, U, D, V> {}

/// An image whose mips the GPU accesses one at a time ([`Image::mip_view`]).
pub type MipChain<F, U, D = D2> = Image<F, U, D, PerMip>;

/// An image whose format and usage are runtime values (`Image::erase`).
pub type DynamicImage<D = D2, V = Single> = Image<format::Dynamic, usage::Dynamic, D, V>;

impl<F: StaticFormat, U: StaticUsage, D: Dim, V: Views> Image<F, U, D, V> {
    /// Creates an image bound in no bindless set: its handles are null.
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
        let live = Live::new();
        check_extent::<D>(extent, mip_levels)?;
        if bc_block(format.vk_format()).is_some() {
            #[cfg(feature = "bc-compression")]
            gate::BcCompression::check()?;
            #[cfg(not(feature = "bc-compression"))]
            return Err(Error::Unsupported(format!(
                "{:?} needs lava's `bc-compression` feature",
                format.vk_format()
            )));
        }
        if usage.flags().contains(vk::ImageUsageFlags::HOST_TRANSFER_EXT) {
            #[cfg(feature = "host-image-copy")]
            gate::HostImageCopy::check()?;
            #[cfg(not(feature = "host-image-copy"))]
            return Err(Error::Unsupported(
                "host copy usage needs lava's `host-image-copy` feature".into(),
            ));
        }
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
        let raw = RawImage {
            image,
            allocation,
            extent,
            mip_levels,
            format,
            usage,
            live,
        };
        unsafe {
            Ctx::device().bind_image_memory(
                image,
                raw.allocation.memory(),
                raw.allocation.offset(),
            )?
        };
        Ok(Image {
            views: V::create(&raw)?,
            raw,
        })
    }

    /// The same image named `name`, for debuggers, the validation layer (with debug utils,
    /// `Config::validation`) and allocator reports.
    pub fn named(mut self, name: &str) -> Self {
        if let Err(err) = Ctx::allocator().rename_allocation(&mut self.raw.allocation, name) {
            tracing::warn!(%err, "failed to name an image's allocation");
        }
        Functions::set_debug_name(name, self.raw.image);
        self
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

    /// The whole view's handle as `K`; panics unless it is bound with `K`'s descriptors (see
    /// [`ImageView::handle`]).
    #[track_caller]
    pub fn handle<K: Allocatable>(&self) -> ImageHandle<K> {
        self.whole_view().handle()
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
        let (view, handle, states) = self.views.whole();
        self.view(view, (0..self.raw.mip_levels).into(), handle, states)
    }

    fn view<'a>(
        &'a self,
        view: vk::ImageView,
        mip_range: Range<u32>,
        handle: ImageHandle,
        states: &'a [AtomicU64],
    ) -> ImageView<'a, F, U, D> {
        ImageView {
            image: self.raw.image,
            view,
            mip_range,
            extent: self.mip_extent(mip_range.start),
            handle,
            states,
            format: self.raw.format,
            usage: self.raw.usage,
            dim: D::default(),
        }
    }

    /// The same image typed as `format` and `usage`; panics unless the format is its own and
    /// the usage no more than its own.
    pub fn cast<NF: Format, NU: ImageUsage>(self, format: NF, usage: NU) -> Image<NF, NU, D, V> {
        slice::cast_check(
            (self.raw.format.vk_format(), self.raw.usage.flags()),
            (format.vk_format(), usage.flags()),
        );
        let this = ManuallyDrop::new(self);
        // SAFETY: `this` is never dropped, so its views and image move out once.
        let (views, raw) = unsafe { (std::ptr::read(&this.views), std::ptr::read(&this.raw)) };
        let mut raw = ManuallyDrop::new(raw);
        Image {
            views,
            raw: RawImage {
                image: raw.image,
                allocation: mem::take(&mut raw.allocation),
                extent: raw.extent,
                mip_levels: raw.mip_levels,
                format,
                usage,
                live: mem::replace(&mut raw.live, Live::internal()),
            },
        }
    }

    /// The same image with its format and usage as runtime values.
    pub fn erase(self) -> DynamicImage<D, V> {
        let format = format::Dynamic(self.raw.format.vk_format());
        let usage = usage::Dynamic(self.raw.usage.flags());
        self.cast(format, usage)
    }
}

impl<F: Format, U: ImageUsage, D: Dim, V: Views> Image<F, U, D, V> {
    /// Binds the whole view's sampled descriptor at a free slot right away, freed once the
    /// image is destroyed. Panics if it is bound already: a view is bound once.
    #[validation_trace]
    pub fn bind_sampled(self) -> Result<Self>
    where
        Self: SampledImage,
    {
        self.bind_whole::<Sampled>()
    }

    /// Like [`Self::bind_sampled`], its storage descriptor.
    #[validation_trace]
    pub fn bind_storage(self) -> Result<Self>
    where
        Self: StorageImage,
    {
        self.bind_whole::<Storage>()
    }

    /// Like [`Self::bind_sampled`], its sampled descriptor and, in the next slot, its storage
    /// descriptor, which a shader that reads through the sampler and writes as storage needs
    /// (`MutTex2D`).
    #[validation_trace]
    pub fn bind_combined(self) -> Result<Self>
    where
        Self: SampledImage + StorageImage,
    {
        self.bind_whole::<Combined>()
    }

    fn bind_whole<K: Allocatable>(mut self) -> Result<Self> {
        let handle = self.views.whole_handle();
        assert!(handle.is_null(), "the image is bound already ({handle:?}): bind it once, with `bind_combined` for both descriptors");
        *handle = ImageHandle::<K>::new(bindless::allocate(K::WIDTH)?).erase();
        let bound = *handle;
        bindless::bind_raw([self.whole_view().shader_image()], [bound], false)?;
        Ok(self)
    }
}

impl<F: Format, U: ImageUsage, D: Dim, V: MipViews> Image<F, U, D, V> {
    /// The view of mip `level` alone, sized `mip_extent(level)`.
    pub fn mip_view(&self, level: u32) -> ImageView<'_, F, U, D> {
        let (view, handle) = self.views.mip(level);
        let (_, _, states) = self.views.whole();
        self.view(view, (level..level + 1).into(), handle, states)
    }

    fn bind_mips<K: Allocatable>(mut self) -> Result<Self> {
        let levels = 0..self.mip_levels();
        assert!(
            levels.clone().all(|level| self.views.mip(level).1.is_null()),
            "the mips are bound already: bind them once, with `bind_mips_combined` for both descriptors"
        );
        let first = bindless::allocate(levels.len() as u32 * K::WIDTH)?;
        for level in levels.clone() {
            *self.views.mip_handle_mut(level) = ImageHandle::<K>::new(first + level * K::WIDTH).erase();
        }
        bindless::bind_raw(
            levels.clone().map(|level| self.mip_view(level).shader_image()),
            levels.map(|level| self.views.mip(level).1),
            false,
        )?;
        Ok(self)
    }

    /// Binds each mip's view's sampled descriptor right away, all in one block of contiguous
    /// free slots: mip `i`'s handle is mip 0's [`ImageHandle::mip`]`(i)`.
    #[validation_trace]
    pub fn bind_mips_sampled(self) -> Result<Self>
    where
        Self: SampledImage,
    {
        self.bind_mips::<Sampled>()
    }

    /// Like [`Self::bind_mips_sampled`], their storage descriptors.
    #[validation_trace]
    pub fn bind_mips_storage(self) -> Result<Self>
    where
        Self: StorageImage,
    {
        self.bind_mips::<Storage>()
    }

    /// Like [`Self::bind_mips_sampled`], each mip's sampled and storage descriptor
    /// ([`Self::bind_combined`]), the pairs one after another.
    #[validation_trace]
    pub fn bind_mips_combined(self) -> Result<Self>
    where
        Self: SampledImage + StorageImage,
    {
        self.bind_mips::<Combined>()
    }
}

#[cfg(all(feature = "bc-compression", not(feature = "host-image-copy")))]
use crate::features::Feature;
#[cfg(any(feature = "bc-compression", feature = "host-image-copy"))]
use crate::features::gate;
#[cfg(feature = "host-image-copy")]
use crate::{
    command_buffer::image_state::take_host_layouts,
    features::Feature,
    image::usage::IsHostCopy,
};

#[cfg(feature = "host-image-copy")]
impl<F: Format, U: IsHostCopy, D: Dim, V: Views> Image<F, U, D, V> {
    /// Writes a whole mip level from the host, right away, unsynchronised with frames in flight
    /// that read the image.
    #[validation_trace]
    ///
    /// Panics for a mip it doesn't have, data of another length than the mip's, and combined
    /// depth-stencil formats.
    pub fn copy_from(&mut self, data: &[u8], mip_level: u32) -> Result<()> {
        self.check_host_copy(data, mip_level);
        let extent = self.mip_extent(mip_level);
        self.check_data_len(data, extent, || format!("mip level {mip_level}"));
        self.host_copy(data, mip_level, D::ZERO, extent)
    }

    /// Like `copy_from`, for the `extent` texels at `offset`; `data` holds that box, tightly
    /// packed.
    #[validation_trace]
    pub fn copy_region_from(
        &mut self,
        data: &[u8],
        mip_level: u32,
        offset: D::Extent,
        extent: D::Extent,
    ) -> Result<()> {
        self.check_host_copy(data, mip_level);
        let mip_extent = self.mip_extent(mip_level);
        assert!(
            D::region_fits(mip_extent, offset, extent),
            "region {extent:?} at {offset:?} does not fit mip level {mip_level} of size {mip_extent:?}"
        );
        self.check_data_len(data, extent, || "the region".to_string());
        self.host_copy(data, mip_level, offset, extent)
    }

    #[track_caller]
    fn check_host_copy(&self, data: &[u8], mip_level: u32) {
        // The image exists, so the device has host copies.
        self.usage_check(vk::ImageUsageFlags::HOST_TRANSFER_EXT);
        assert!(
            mip_level < self.raw.mip_levels,
            "mip level {mip_level} is out of range for an image with {} levels",
            self.raw.mip_levels
        );
        assert!(!data.is_empty(), "no texel data to copy");
        let aspects = self.raw.format.aspects();
        assert!(
            slice::single_aspect(aspects) == aspects,
            "host copies of combined depth-stencil formats aren't supported: their texel data has no single layout"
        );
    }

    #[track_caller]
    fn check_data_len(&self, data: &[u8], extent: D::Extent, what: impl FnOnce() -> String) {
        let format = self.raw.format;
        let expected = expected_data_len(
            D::extent_3d(extent),
            format.texel_size(),
            bc_block(format.vk_format()),
        );
        if let Some(expected) = expected {
            assert!(
                data.len() == expected,
                "{} holds {expected} bytes but {} were given",
                what(),
                data.len()
            );
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
        let (_, _, states) = self.views.whole();
        // A host copy is no queue's access, so the image keeps its owner.
        let mut transitions: smallvec::SmallVec<[vk::HostImageLayoutTransitionInfoEXT; 1]> =
            smallvec::SmallVec::new();
        let (mips, general) = ((0..self.raw.mip_levels).into(), vk::ImageLayout::GENERAL);
        take_host_layouts(states, mips, general, |mips, old_layout| {
            transitions.push(
                vk::HostImageLayoutTransitionInfoEXT::default()
                    .image(image)
                    .old_layout(old_layout)
                    .new_layout(general)
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
            // 0: rows as long as the region is wide, also right for a BC mip narrower than
            // its block.
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

/// An image needs texels, at least one mip and no more than its largest side halves into, and
/// no side beyond the device's limit.
fn check_extent<D: Dim>(extent: D::Extent, mip_levels: u32) -> Result<()> {
    let e = D::extent_3d(extent);
    let largest = e.width.max(e.height).max(e.depth);
    if e.width == 0 || e.height == 0 || e.depth == 0 {
        return Err(Error::other(format!("an image of extent {extent:?} has no texels")));
    }
    let limits = &Ctx::device_properties().limits;
    let max = match D::IMAGE_TYPE {
        vk::ImageType::TYPE_3D => limits.max_image_dimension3_d,
        _ => limits.max_image_dimension2_d,
    };
    if largest > max {
        return Err(Error::Unsupported(format!(
            "an image of extent {extent:?} is larger than the device's {max} texels per side"
        )));
    }
    let most = largest.ilog2() + 1;
    if mip_levels == 0 || mip_levels > most {
        return Err(Error::other(format!(
            "an image of extent {extent:?} has 1 to {most} mip levels, not {mip_levels}"
        )));
    }
    Ok(())
}

/// Bytes of tightly packed texels of `extent`, in blocks of `(side, bytes)` where `texel_size`
/// is 0; `None` where the format has neither.
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

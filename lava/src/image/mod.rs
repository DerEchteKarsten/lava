//! Format-, usage- and dimension-typed GPU images, with array layers for `D2Array`, with one view over all mips or one per mip (`VkImageView`s only for attachments, made on first use for layer and reinterpreting views; bindless descriptors need none), created only where the device supports their description, recreated at a new extent in their bindless slots (`fit`), host uploads (`host-image-copy`; staged ones are `Uploader::image` in `buffer/staging.rs`), type erasure, and their destruction once no frame can reach them
use std::{
    mem::{self, ManuallyDrop},
    range::Range,
    sync::{Mutex, atomic::AtomicU64},
};

use crate::error::{Error, Result};
use crate::vk;
use glam::UVec2;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use lava_macros::validation_trace;

use crate::{
    bindless::{self, Allocatable, Combined, ImageHandle, Sampled, Storage},
    command_buffer::image_state::ImageState,
    image::{
        format::{Format, StaticFormat},
        slice::ImageView,
        usage::{ImageUsage, StaticUsage},
    },
    global::Live,
    lifetime,
    queue::timeline::forget_image,
    state::{Ctx, Functions, ImageFormatKey},
};

pub mod dim;
pub mod format;
pub mod kind;
pub mod slice;
pub mod usage;

pub use dim::{D2, D2Array, D3, Dim, Dim2D};
pub use kind::{
    ColorTarget, DepthTarget, ImageRef, SampledImage, SampledView, StencilTarget, StorageImage,
    StorageView, TransferDstView, TransferSrcView,
};

/// The `vk::Image` and its memory. It has no public constructor, so only lava implements
/// [`Views`].
#[derive(Debug)]
pub struct RawImage<F: Format, U: ImageUsage, D: Dim> {
    image: vk::Image,
    allocation: Allocation,
    extent: D::Extent,
    mip_levels: u32,
    /// 1 unless `D` is `D2Array`.
    layers: u32,
    format: F,
    usage: U,
    /// What it was created with besides its usage (`MUTABLE_FORMAT` and `EXTENDED_USAGE` of a
    /// `MutableFormat` usage, the sparse flags of a `SparseImage`).
    pub(crate) create_flags: vk::ImageCreateFlags,
    /// The `VkImageView`s of views other than the image's own, made when a rendering first
    /// needs one (`ImageView::view`).
    cache: ViewCache,
    /// Last, so it counts the image until its drop finished.
    live: Live,
}

fn has_attachment_usage(usage: vk::ImageUsageFlags) -> bool {
    usage.intersects(
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
    )
}

/// A `VkImageView` of `mips` and `layers` of `image` over every aspect, of `view_type`, for
/// `usage` (the image's attachment usages: renderings use these views).
fn create_view(
    image: vk::Image,
    format: vk::Format,
    aspects: vk::ImageAspectFlags,
    view_type: vk::ImageViewType,
    mips: Range<u32>,
    layers: Range<u32>,
    usage: vk::ImageUsageFlags,
) -> Result<vk::ImageView> {
    let mut usage = vk::ImageViewUsageCreateInfo::default().usage(
        usage & (vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT),
    );
    let info = vk::ImageViewCreateInfo::default()
        .push_next(&mut usage)
        .image(image)
        .view_type(view_type)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: aspects,
            base_mip_level: mips.start,
            level_count: mips.end - mips.start,
            base_array_layer: layers.start,
            layer_count: layers.end - layers.start,
        });
    Ok(unsafe { Ctx::device().create_image_view(&info, None)? })
}

/// The `VkImageView`s an attachment image makes on first use for views without one of their
/// own (one layer of an array, a range of layers), by mips, layers and view type; destroyed
/// with the image.
#[derive(Debug, Default)]
pub(crate) struct ViewCache(Mutex<Vec<(ViewKey, vk::ImageView)>>);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ViewKey {
    format: vk::Format,
    mips: (u32, u32),
    layers: (u32, u32),
    view_type: vk::ImageViewType,
}

impl ViewCache {
    /// The view of `mips` and `layers` of `image` as `format`, made the first time it is asked
    /// for.
    /// Panics if the device can't make it (out of memory).
    pub(crate) fn get(
        &self,
        image: vk::Image,
        format: vk::Format,
        aspects: vk::ImageAspectFlags,
        view_type: vk::ImageViewType,
        mips: Range<u32>,
        layers: Range<u32>,
        usage: vk::ImageUsageFlags,
    ) -> vk::ImageView {
        let key = ViewKey {
            format,
            mips: (mips.start, mips.end),
            layers: (layers.start, layers.end),
            view_type,
        };
        let mut views = crate::queue::lock(&self.0);
        if let Some(&(_, view)) = views.iter().find(|(k, _)| *k == key) {
            return view;
        }
        let view = create_view(image, format, aspects, view_type, mips, layers, usage)
            .unwrap_or_else(|err| panic!("creating a view of layers {layers:?} failed: {err}"));
        views.push((key, view));
        view
    }

    fn take(&mut self) -> Vec<vk::ImageView> {
        let views = self.0.get_mut().unwrap_or_else(std::sync::PoisonError::into_inner);
        mem::take(views).into_iter().map(|(_, view)| view).collect()
    }
}

impl<F: Format, U: ImageUsage, D: Dim> RawImage<F, U, D> {
    /// The `vk::Image` of the description, checked against the device, without memory
    /// (`allocate` binds lava's, a sparse image binds pages later, an external image its
    /// own); `external` are the handle types its memory will be exported or imported as.
    pub(crate) fn create(
        format: F,
        usage: U,
        extent: D::Extent,
        mip_levels: u32,
        layers: u32,
        flags: vk::ImageCreateFlags,
        external: vk::ExternalMemoryHandleTypeFlags,
    ) -> Result<Self> {
        let live = Live::new();
        check_extent::<D>(extent, mip_levels)?;
        let max_layers = Ctx::device_properties().limits.max_image_array_layers;
        if layers == 0 || layers > max_layers {
            return Err(Error::Unsupported(format!(
                "an image has 1 to {max_layers} layers on this device, not {layers}"
            )));
        }
        let vk_format = format.vk_format();
        compression_check(vk_format)?;
        let usage_flags = usage.flags();
        let attachment = has_attachment_usage(usage_flags);
        let host_copy = usage_flags.contains(vk::ImageUsageFlags::HOST_TRANSFER_EXT);
        assert!(
            !(attachment && host_copy),
            "an image with usage {usage_flags:?} has attachment and host copy usages, which lava \
             doesn't allow together: host copies don't take the attachments' layouts"
        );
        if host_copy {
            #[cfg(feature = "host-image-copy")]
            gate::HostImageCopy::check()?;
            #[cfg(not(feature = "host-image-copy"))]
            return Err(Error::Unsupported(
                "host copy usage needs lava's `host-image-copy` feature".into(),
            ));
        }
        check_support::<D>(vk_format, usage_flags, flags, external, D::extent_3d(extent), mip_levels, layers)?;
        let mut external_info = vk::ExternalMemoryImageCreateInfo::default().handle_types(external);
        let mut create_info = vk::ImageCreateInfo {
            flags,
            array_layers: layers,
            extent: D::extent_3d(extent),
            format: vk_format,
            image_type: D::IMAGE_TYPE,
            mip_levels,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: usage_flags,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };
        if !external.is_empty() {
            create_info = create_info.push_next(&mut external_info);
        }
        let image = unsafe { Ctx::device().create_image(&create_info, None) }?;
        // From here on `Drop` destroys the image (and frees its memory once it has some).
        Ok(RawImage {
            image,
            allocation: Allocation::default(),
            extent,
            mip_levels,
            layers,
            format,
            usage,
            create_flags: flags,
            cache: ViewCache::default(),
            live,
        })
    }

    /// Allocates device-local memory from lava's allocator and binds it.
    fn allocate(&mut self) -> Result<()> {
        let image = self.image;
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
        self.allocation = Ctx::allocator().allocate(&desc)?;
        unsafe {
            Ctx::device().bind_image_memory(
                image,
                self.allocation.memory(),
                self.allocation.offset(),
            )?
        };
        Ok(())
    }

    #[allow(dead_code)] // Sparse and external images'.
    pub(crate) fn handle(&self) -> vk::Image {
        self.image
    }

    /// A view of `mips` and every layer over every aspect where renderings need one
    /// (attachment usages), else null: descriptors are written from a view's description,
    /// without a view object.
    fn create_view(&self, mips: Range<u32>) -> Result<vk::ImageView> {
        if !has_attachment_usage(self.usage.flags()) {
            return Ok(vk::ImageView::null());
        }
        let format = self.format.vk_format();
        let layers = (0..self.layers).into();
        let usage = self.usage.flags();
        create_view(self.image, format, self.format.aspects(), D::VIEW_TYPE, mips, layers, usage)
    }
}

impl<F: Format, U: ImageUsage, D: Dim> Drop for RawImage<F, U, D> {
    fn drop(&mut self) {
        let (image, allocation) = (self.image, mem::take(&mut self.allocation));
        let views = self.cache.take();
        lifetime::retire(move || {
            for view in views {
                unsafe { Ctx::device().destroy_image_view(view, None) };
            }
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
    /// Every handle it keeps, with its mip for a mip's (`None` for the whole view's).
    #[doc(hidden)]
    fn handles_mut(&mut self, f: &mut dyn FnMut(Option<u32>, &mut ImageHandle));
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
    fn handles_mut(&mut self, f: &mut dyn FnMut(Option<u32>, &mut ImageHandle)) {
        f(None, &mut self.handle);
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
    fn handles_mut(&mut self, f: &mut dyn FnMut(Option<u32>, &mut ImageHandle)) {
        f(None, &mut self.handle);
        for (mip, view) in self.mips.iter_mut().enumerate() {
            f(Some(mip as u32), &mut view.handle);
        }
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
        bindless::forget_handle(handle);
        lifetime::retire(move || release(view, handle));
    }
}

impl Drop for PerMip {
    fn drop(&mut self) {
        let (view, handle, mips) = (self.view, self.handle, mem::take(&mut self.mips));
        bindless::forget_handle(handle);
        for mip in &mips {
            bindless::forget_handle(mip.handle);
        }
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
    /// Creates an image bound in no bindless set: its handles are null. `Error::Unsupported`
    /// where the device can't make images of its format, usage, dimension and extent
    /// (`vkGetPhysicalDeviceImageFormatProperties2`), `MissingFeature` where its format needs
    /// a compression feature the device lacks.
    #[validation_trace]
    pub fn new(extent: D::Extent) -> Result<Self> {
        Self::with_mip_levels(extent, 1)
    }
    #[validation_trace]
    pub fn with_mip_levels(extent: D::Extent, mip_levels: u32) -> Result<Self> {
        Self::with_params(F::default(), U::default(), extent, mip_levels)
    }
}

impl<F: StaticFormat, U: StaticUsage, V: Views> Image<F, U, D2Array, V> {
    /// Creates an image of `layers` array layers, each of `extent` and `mip_levels` mips, bound
    /// in no bindless set. Its views show every layer; [`ImageView::layer`] and
    /// [`ImageView::layers`] show fewer.
    #[validation_trace]
    pub fn with_layers(extent: UVec2, mip_levels: u32, layers: u32) -> Result<Self> {
        Self::with_layer_params(F::default(), U::default(), extent, mip_levels, layers)
    }
}

impl<F: Format, U: ImageUsage, V: Views> Image<F, U, D2Array, V> {
    /// Like `with_layers`, for any format and usage, `Dynamic` ones included.
    #[validation_trace]
    pub fn with_layer_params(
        format: F,
        usage: U,
        extent: UVec2,
        mip_levels: u32,
        layers: u32,
    ) -> Result<Self> {
        Self::create(format, usage, extent, mip_levels, layers)
    }
}

impl<F: Format, U: ImageUsage, D: Dim, V: Views> Image<F, U, D, V> {
    /// Like `with_mip_levels`, for any format and usage, `Dynamic` ones included; one layer.
    /// Panics for a `Dynamic` usage with attachment and host copy flags, which a static one
    /// can't have.
    #[validation_trace]
    pub fn with_params(format: F, usage: U, extent: D::Extent, mip_levels: u32) -> Result<Self> {
        Self::create(format, usage, extent, mip_levels, 1)
    }

    /// Checks the description, creates the image, allocates and binds its memory, and makes
    /// its views. `Error::Unsupported` where the device can't make it
    /// (`vkGetPhysicalDeviceImageFormatProperties2`), `MissingFeature` where its format or
    /// usage needs a feature the device lacks.
    fn create(format: F, usage: U, extent: D::Extent, mip_levels: u32, layers: u32) -> Result<Self> {
        let flags = usage.create_flags();
        let mut raw = RawImage::<F, U, D>::create(format, usage, extent, mip_levels, layers, flags, Default::default())?;
        raw.allocate()?;
        Self::from_raw(raw)
    }

    /// The image of `raw`, whose memory is bound (or sparse), with its views.
    pub(crate) fn from_raw(raw: RawImage<F, U, D>) -> Result<Self> {
        Ok(Image {
            views: V::create(&raw)?,
            raw,
        })
    }
    /// Recreates the image at `extent` unless it has that extent already, with the same
    /// format, usage, mips and layers, and returns whether it did. The new image keeps the
    /// bindless slots of the old one's `bind_*` bindings: their descriptors are rewritten in
    /// submission order, as [`bindless::bind`] rewrites reserved slots, so frames in flight
    /// read the old image (which lives until they are done) and submissions after the next
    /// graphics or compute submission the new one. Slots it was bound into with
    /// `bindless::bind` keep the old image's descriptors: bind the new one there too. The new
    /// image's contents are undefined, its name is gone, and the old views' borrows have
    /// ended (`&mut self`). Errors where the old image's description doesn't fit `extent`
    /// (too few texels for its mips); panics on a thread that is recording.
    #[validation_trace]
    pub fn fit(&mut self, extent: D::Extent) -> Result<bool> {
        if self.raw.extent == extent {
            return Ok(false);
        }
        assert!(
            !lifetime::RECORDING.get(),
            "`Image::fit` is called inside a recording, whose commands would read the old \
             descriptors: fit images before `submit`"
        );
        let raw = &self.raw;
        let mut new = RawImage::<F, U, D>::create(
            raw.format,
            raw.usage,
            extent,
            raw.mip_levels,
            raw.layers,
            raw.create_flags,
            vk::ExternalMemoryHandleTypeFlags::empty(),
        )?;
        new.allocate()?;
        let mut new = Image::<F, U, D, V>::from_raw(new)?;
        // The new views take the old ones' slots, which the old views then don't free.
        let mut handles = Vec::new();
        self.views.handles_mut(&mut |mip, handle| {
            handles.push((mip, std::mem::replace(handle, ImageHandle::NULL)));
        });
        let mut moved = handles.iter();
        new.views.handles_mut(&mut |_, handle| {
            *handle = moved.next().expect("the same views").1;
        });
        for &(mip, handle) in &handles {
            if handle.is_null() {
                continue;
            }
            let image = match mip {
                None => new.whole_view().shader_image(),
                // A mip's descriptor shows that mip alone.
                Some(mip) => {
                    let mut view = new.whole_view();
                    view.mip_range = (mip..mip + 1).into();
                    view.shader_image()
                }
            };
            bindless::rebind(image, handle)?;
        }
        *self = new;
        Ok(true)
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

    /// Its array layers: 1 unless it is a `D2Array` image made `with_layers`.
    pub fn layers(&self) -> u32 {
        self.raw.layers
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

    /// The view over every mip and layer.
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
        let attachment = has_attachment_usage(self.raw.usage.flags());
        ImageView {
            image: self.raw.image,
            view,
            mip_range,
            layers: (0..self.raw.layers).into(),
            extent: self.mip_extent(mip_range.start),
            handle,
            states,
            cache: attachment.then_some(&self.raw.cache),
            mutable: self.raw.create_flags.contains(vk::ImageCreateFlags::MUTABLE_FORMAT),
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
                layers: raw.layers,
                format,
                usage,
                create_flags: raw.create_flags,
                cache: mem::take(&mut raw.cache),
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

#[cfg(feature = "host-image-copy")]
use crate::{
    command_buffer::image_state::{commit_host_layouts, host_layouts},
    features::Feature,
    features::gate,
    image::usage::IsHostCopy,
};

#[cfg(feature = "host-image-copy")]
impl<F: Format, U: IsHostCopy, D: Dim, V: Views> Image<F, U, D, V> {
    /// Writes a whole mip level of every layer from the host (the layers one after another),
    /// right away, unsynchronised with frames in flight that read the image.
    #[validation_trace]
    ///
    /// Panics for a mip it doesn't have, data of another length than the mip's, and combined
    /// depth-stencil formats.
    pub fn copy_from(&mut self, data: &[u8], mip_level: u32) -> Result<()> {
        self.check_host_copy(data, mip_level);
        let extent = self.mip_extent(mip_level);
        self.check_data_len(data, extent, (0, 0), || format!("mip level {mip_level}"));
        self.host_copy(data, mip_level, D::ZERO, extent, (0, 0))
    }

    /// Like `copy_from`, for the `extent` texels at `offset` of every layer; `data` holds that
    /// box per layer, tightly packed. A block-compressed format's box covers whole blocks (or
    /// reaches the mip's edge), as GPU copies' do.
    #[validation_trace]
    pub fn copy_region_from(
        &mut self,
        data: &[u8],
        mip_level: u32,
        offset: D::Extent,
        extent: D::Extent,
    ) -> Result<()> {
        self.copy_rows_from(data, mip_level, offset, extent, 0, 0)
    }

    /// Like `copy_region_from`, with the box's rows `row_length` texels apart in `data` and its
    /// slices (and layers) `image_height` rows apart; 0 for either is the box's own width or
    /// height (tightly packed), else at least that and, for a block-compressed format, whole
    /// blocks.
    #[validation_trace]
    pub fn copy_rows_from(
        &mut self,
        data: &[u8],
        mip_level: u32,
        offset: D::Extent,
        extent: D::Extent,
        row_length: u32,
        image_height: u32,
    ) -> Result<()> {
        self.check_host_copy(data, mip_level);
        let mip_extent = self.mip_extent(mip_level);
        assert!(
            D::region_fits(mip_extent, offset, extent),
            "region {extent:?} at {offset:?} does not fit mip level {mip_level} of size {mip_extent:?}"
        );
        let format = self.raw.format;
        let block = slice::copy_block_of(&format, format.aspects());
        slice::check_whole_blocks(
            block,
            format.vk_format(),
            D::unsigned_offset_3d(offset),
            D::extent_3d(extent),
            D::extent_3d(mip_extent),
        );
        self.check_data_len(data, extent, (row_length, image_height), || "the region".to_string());
        self.host_copy(data, mip_level, offset, extent, (row_length, image_height))
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
    fn check_data_len(
        &self,
        data: &[u8],
        extent: D::Extent,
        rows: (u32, u32),
        what: impl FnOnce() -> String,
    ) {
        let format = self.raw.format;
        let block = slice::copy_block_of(&format, format.aspects());
        let expected = slice::rows_len(block, D::extent_3d(extent), self.raw.layers, rows);
        if rows == (0, 0) {
            assert!(
                data.len() == expected,
                "{} holds {expected} bytes but {} were given",
                what(),
                data.len()
            );
        } else {
            assert!(
                data.len() >= expected,
                "{} takes {expected} bytes at rows of {} texels and slices of {} rows, but {} were given",
                what(),
                rows.0,
                rows.1,
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
        (row_length, image_height): (u32, u32),
    ) -> Result<()> {
        let image = self.raw.image;
        let aspect = self.raw.format.aspects();
        let (_, _, states) = self.views.whole();
        // A host copy is no queue's access, so the image keeps its owner. The states move only
        // once the transition succeeded.
        let mut transitions: smallvec::SmallVec<[vk::HostImageLayoutTransitionInfoEXT; 1]> =
            smallvec::SmallVec::new();
        let (mips, general) = ((0..self.raw.mip_levels).into(), vk::ImageLayout::GENERAL);
        host_layouts(states, mips, general, |mips, old_layout| {
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
                        layer_count: vk::REMAINING_ARRAY_LAYERS,
                    }),
            )
        });
        if !transitions.is_empty() {
            unsafe { Functions::host_image_copy().transition_image_layout(&transitions)? };
            commit_host_layouts(states, mips, general);
        }
        let regions = [vk::MemoryToImageCopyEXT::default()
            .host_pointer(data.as_ptr().cast())
            .image_offset(D::unsigned_offset_3d(offset))
            .image_extent(D::extent_3d(extent))
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: aspect,
                mip_level,
                base_array_layer: 0,
                layer_count: self.raw.layers,
            })
            // 0: rows as long as the region is wide, also right for a mip narrower than its
            // block.
            .memory_image_height(image_height)
            .memory_row_length(row_length)];
        let info = vk::CopyMemoryToImageInfoEXT::default()
            .dst_image(image)
            .dst_image_layout(vk::ImageLayout::GENERAL)
            .regions(&regions);
        unsafe { Functions::host_image_copy().copy_memory_to_image(&info)? };
        Ok(())
    }
}

/// `MissingFeature` (or `Unsupported`, compiled without it) for a block-compressed format whose
/// feature the device lacks.
fn compression_check(format: vk::Format) -> Result<()> {
    #[cfg(any(
        feature = "bc-compression",
        feature = "etc2-compression",
        feature = "astc-compression"
    ))]
    use crate::features::Feature;
    let unsupported = |feature: &str| {
        Err(Error::Unsupported(format!("{format:?} needs lava's `{feature}` feature")))
    };
    match format::compression(format) {
        None => Ok(()),
        #[cfg(feature = "bc-compression")]
        Some("bc-compression") => Ok(crate::features::gate::BcCompression::check()?),
        #[cfg(feature = "etc2-compression")]
        Some("etc2-compression") => Ok(crate::features::gate::Etc2Compression::check()?),
        #[cfg(feature = "astc-compression")]
        Some("astc-compression") => Ok(crate::features::gate::AstcCompression::check()?),
        Some(feature) => unsupported(feature),
    }
}

/// `Error::Unsupported` unless the device makes optimal-tiling images of the description (a
/// usage of a `MutableFormat` image may need a view format's features, which the device
/// answers for), within its limits for it, and, with host copies, copies the format's texels
/// from the host.
fn check_support<D: Dim>(
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    flags: vk::ImageCreateFlags,
    external: vk::ExternalMemoryHandleTypeFlags,
    extent: vk::Extent3D,
    mip_levels: u32,
    layers: u32,
) -> Result<()> {
    let key = ImageFormatKey {
        format,
        image_type: D::IMAGE_TYPE,
        usage,
        flags,
        external,
    };
    let description = || {
        let mut text = format!("{:?} images of {format:?} with usage {usage:?}", D::IMAGE_TYPE);
        if !flags.is_empty() {
            text += &format!(" and flags {flags:?}");
        }
        if !external.is_empty() {
            text += &format!(" and external memory {external:?}");
        }
        text
    };
    let Some(limits) = Ctx::image_format_limits(key) else {
        return Err(Error::Unsupported(format!("the device can't make {}", description())));
    };
    let max = limits.max_extent;
    if extent.width > max.width || extent.height > max.height || extent.depth > max.depth {
        return Err(Error::Unsupported(format!(
            "{} are at most {max:?}, not {extent:?}",
            description()
        )));
    }
    if mip_levels > limits.max_mip_levels || layers > limits.max_array_layers {
        return Err(Error::Unsupported(format!(
            "{} have at most {} mips and {} layers, not {mip_levels} and {layers}",
            description(),
            limits.max_mip_levels,
            limits.max_array_layers
        )));
    }
    if usage.contains(vk::ImageUsageFlags::HOST_TRANSFER_EXT)
        && !Ctx::format_features(format).contains(vk::FormatFeatureFlags2::HOST_IMAGE_TRANSFER_EXT)
    {
        return Err(Error::Unsupported(format!(
            "the device doesn't copy texels of {format:?} from the host (HOST_IMAGE_TRANSFER)"
        )));
    }
    Ok(())
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

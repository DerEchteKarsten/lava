//! Format-, usage- and dimension-typed GPU images, with array layers for `D2Array`, with one view over all mips or one per mip (`VkImageView`s only for attachments, made on first use for layer and reinterpreting views; bindless descriptors need none), created only where the device supports their description, recreated at a new extent in their bindless slots (`fit`), host uploads (`host-image-copy`; staged ones are `Uploader::image` in `buffer/staging.rs`), type erasure, and the internal their owners share (`ImageInternal`: the image, its memory, views, handles and tracked states, also of sparse, external and swapchain images), destroyed once the last owner dropped it and no frame can reach it
use std::{
    marker::PhantomData,
    mem,
    range::Range,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU32, AtomicU64, Ordering},
    },
};

use crate::error::{Error, Result};
use crate::vk;
use glam::UVec2;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use lava_macros::validation_trace;
use smallvec::SmallVec;

#[cfg(feature = "host-image-copy")]
use crate::{
    command_buffer::image_state::{commit_host_layouts, host_layouts},
    features::Feature,
    features::gate,
    image::usage::IsHostCopy,
};

use crate::{
    bindless::{self, Allocatable, Combined, Dynamic, ImageHandle, Sampled, Storage},
    command_buffer::{access_state::AccessState, image_state::ImageState},
    global::Live,
    image::{
        format::{Format, StaticFormat},
        usage::{ImageUsage, StaticUsage},
    },
    lifetime,
    object::Header,
    state::{Ctx, Functions, ImageFormatKey},
};

pub mod dim;
pub mod format;
pub mod kind;
pub(crate) mod slice;
pub mod usage;

pub use dim::{D2, D2Array, D3, Dim, Dim2D};
pub use kind::{
    ColorTarget, DepthTarget, ImageRef, SampledImage, SampledView, StencilTarget, StorageImage,
    StorageView, TransferDstView, TransferSrcView,
};
pub use slice::{Filter, ImageSlice, ImageView, TexelBuffer};

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
        usage
            & (vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT),
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
pub(crate) struct ViewCache {
    views: Mutex<Vec<(ViewKey, vk::ImageView)>>,
    /// The handles of views bound on their own (`ImageView::bind`): layers, mip ranges and
    /// reinterpreting views, freed with the image.
    handles: Mutex<Vec<(ViewKey, ImageHandle)>>,
}

/// What a view shows: its format, mips, layers and view type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ViewKey {
    format: vk::Format,
    mips: (u32, u32),
    layers: (u32, u32),
    view_type: vk::ImageViewType,
}

impl ViewKey {
    pub(crate) fn new(
        format: vk::Format,
        mips: Range<u32>,
        layers: Range<u32>,
        view_type: vk::ImageViewType,
    ) -> Self {
        ViewKey {
            format,
            mips: (mips.start, mips.end),
            layers: (layers.start, layers.end),
            view_type,
        }
    }
}

impl ViewCache {
    /// The view of `mips` and `layers` of `image` as `format`, made the first time it is asked
    /// for.
    /// Panics if the device can't make it (out of memory).
    #[allow(clippy::too_many_arguments)] // The view's description, field by field.
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
        let key = ViewKey::new(format, mips, layers, view_type);
        let mut views = crate::queue::lock(&self.views);
        if let Some(&(_, view)) = views.iter().find(|(k, _)| *k == key) {
            return view;
        }
        let view = create_view(image, format, aspects, view_type, mips, layers, usage)
            .unwrap_or_else(|err| panic!("creating a view of layers {layers:?} failed: {err}"));
        views.push((key, view));
        view
    }

    fn take(&mut self) -> Vec<vk::ImageView> {
        let views = self
            .views
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        mem::take(views).into_iter().map(|(_, view)| view).collect()
    }

    /// The handle the view `key` was bound with on its own that has `K`'s descriptors.
    pub(crate) fn handle<K: Allocatable>(&self, key: ViewKey) -> Option<ImageHandle<K>> {
        (crate::queue::lock(&self.handles).iter())
            .filter(|(k, _)| *k == key)
            .find_map(|(_, handle)| handle.narrow())
    }

    /// The handle of `key` with `K`'s descriptors, bound by `bind` (which writes them into the
    /// slots it is given) the first time it is asked for.
    pub(crate) fn bind<K: Allocatable>(
        &self,
        key: ViewKey,
        bind: impl FnOnce(ImageHandle) -> Result<()>,
    ) -> Result<ImageHandle<K>> {
        let mut handles = crate::queue::lock(&self.handles);
        let found = (handles.iter()).filter(|(k, _)| *k == key);
        if let Some(handle) = found.clone().find_map(|(_, handle)| handle.narrow()) {
            return Ok(handle);
        }
        let handle = ImageHandle::<K>::new(bindless::allocate(K::WIDTH)?);
        if let Err(err) = bind(handle.erase()) {
            bindless::free_handle(handle.erase());
            return Err(err);
        }
        handles.push((key, handle.erase()));
        Ok(handle)
    }

    fn take_handles(&mut self) -> Vec<ImageHandle> {
        let handles = self
            .handles
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        mem::take(handles)
            .into_iter()
            .map(|(_, handle)| handle)
            .collect()
    }
}

/// A view an image keeps, with the handle of its bindless descriptors (null until it is bound).
#[derive(Debug)]
pub(crate) struct ViewSlot {
    /// Null without attachment usage.
    pub(crate) view: vk::ImageView,
    handle: AtomicU32,
}

impl ViewSlot {
    pub(crate) fn new(view: vk::ImageView, handle: ImageHandle) -> Self {
        Self {
            view,
            handle: AtomicU32::new(handle.bits()),
        }
    }

    pub(crate) fn handle(&self) -> ImageHandle {
        ImageHandle::from_bits(self.handle.load(Ordering::Relaxed))
    }

    pub(crate) fn set_handle(&self, handle: ImageHandle) {
        self.handle.store(handle.bits(), Ordering::Relaxed);
    }

    /// Its handle, which it no longer frees.
    fn take_handle(&self) -> ImageHandle {
        ImageHandle::from_bits(
            self.handle
                .swap(ImageHandle::<Dynamic>::NULL.bits(), Ordering::Relaxed),
        )
    }
}

/// Where an image's memory comes from.
#[derive(Debug)]
pub(crate) enum ImageMemory {
    /// None bound yet (sparse images until their pages are, external ones until they are
    /// allocated), or a null allocation.
    Allocation(Allocation),
    /// A dedicated allocation exported or imported (`ExternalImage`).
    #[cfg(all(feature = "external-memory", unix))]
    External(#[allow(dead_code, reason = "freed by its drop")] crate::external::Memory),
    /// Pages bound one by one (`SparseImage`).
    #[cfg(any(
        feature = "sparse-residency-image-2d",
        feature = "sparse-residency-image-3d"
    ))]
    Sparse(crate::sparse::Pages),
    /// A swapchain's image, which the swapchain owns.
    Swapchain,
}

impl ImageMemory {
    fn free(self) {
        match self {
            ImageMemory::Allocation(allocation) => {
                if let Err(err) = Ctx::allocator().free(allocation) {
                    tracing::error!(%err, "failed to free image allocation");
                }
            }
            // Their own drops free them.
            #[allow(unreachable_patterns)]
            _ => {}
        }
    }
}

/// An image as lava owns it, shared by its [`Image`] (and the types made from it) and whatever
/// else keeps it: the `vk::Image`, its memory, its views and their bindless handles, and its
/// tracked states, destroyed with the last owner.
#[derive(Debug)]
pub(crate) struct ImageInternal {
    pub(crate) image: vk::Image,
    pub(crate) memory: ImageMemory,
    pub(crate) format: vk::Format,
    pub(crate) aspects: vk::ImageAspectFlags,
    pub(crate) usage: vk::ImageUsageFlags,
    pub(crate) mip_levels: u32,
    /// 1 unless the image is a `D2Array` one.
    pub(crate) layers: u32,
    /// What it was created with besides its usage (`MUTABLE_FORMAT` and `EXTENDED_USAGE` of a
    /// `MutableFormat` usage, the sparse flags of a `SparseImage`).
    pub(crate) create_flags: vk::ImageCreateFlags,
    /// The `VkImageView`s of views other than the image's own, made when a rendering first
    /// needs one (`ImageView::vk_view`).
    pub(crate) cache: ViewCache,
    /// The view over every mip and layer.
    pub(crate) whole: ViewSlot,
    /// One view per mip of a `MipChain`, none otherwise.
    pub(crate) mips: Vec<ViewSlot>,
    /// Its tracked states: one for all mips, or one per mip of a `MipChain` (mip `i` at `i`).
    pub(crate) states: Box<[AtomicU64]>,
    /// What the timelines' submissions read and wrote of it, which later commands wait for.
    pub(crate) access: Mutex<AccessState>,
    /// Its lava user count and what references to it need.
    pub(crate) header: Header,
}

impl ImageInternal {
    /// Whether it has an attachment usage, so renderings need `VkImageView`s of its views.
    pub(crate) fn attachment(&self) -> bool {
        has_attachment_usage(self.usage)
    }

    /// A view of `mips` and every layer over every aspect where renderings need one
    /// (attachment usages), else null: descriptors are written from a view's description,
    /// without a view object.
    fn create_view(&self, mips: Range<u32>, view_type: vk::ImageViewType) -> Result<vk::ImageView> {
        if !self.attachment() {
            return Ok(vk::ImageView::null());
        }
        let layers = (0..self.layers).into();
        create_view(
            self.image,
            self.format,
            self.aspects,
            view_type,
            mips,
            layers,
            self.usage,
        )
    }

    /// The pages of a sparse image.
    #[cfg(any(
        feature = "sparse-residency-image-2d",
        feature = "sparse-residency-image-3d"
    ))]
    pub(crate) fn sparse_pages(&self) -> &crate::sparse::Pages {
        match &self.memory {
            ImageMemory::Sparse(pages) => pages,
            _ => unreachable!("only sparse images have pages"),
        }
    }

    /// A swapchain's `image` with its one view (bound as `handle`), which the swapchain
    /// destroys after this.
    pub(crate) fn swapchain(
        image: vk::Image,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        view: vk::ImageView,
        handle: ImageHandle,
    ) -> Self {
        ImageInternal {
            image,
            memory: ImageMemory::Swapchain,
            format,
            aspects: vk::ImageAspectFlags::COLOR,
            usage,
            mip_levels: 1,
            layers: 1,
            create_flags: vk::ImageCreateFlags::empty(),
            cache: ViewCache::default(),
            whole: ViewSlot::new(view, handle),
            mips: Vec::new(),
            states: Box::new([ImageState::UNDEFINED.atomic()]),
            access: Mutex::default(),
            header: Header::new(Live::internal()),
        }
    }
}

impl crate::object::Object for ImageInternal {
    fn header(&self) -> &Header {
        &self.header
    }
}

impl Drop for ImageInternal {
    /// Destroys its views, frees their slots and destroys the image and its memory, once no
    /// submission reaches it; a swapchain image's right away, since its swapchain is only
    /// destroyed once none does. Its tracked accesses go with it.
    fn drop(&mut self) {
        let image = self.image;
        let memory = mem::replace(
            &mut self.memory,
            ImageMemory::Allocation(Allocation::default()),
        );
        let mut views = self.cache.take();
        let whole = self.whole.view;
        views.push(whole);
        // One mip's view is the whole view, destroyed once.
        views.extend((self.mips.iter()).map(|m| m.view).filter(|&v| v != whole));
        views.retain(|&view| view != vk::ImageView::null());
        let mut handles: SmallVec<[ImageHandle; 1]> = std::iter::once(&self.whole)
            .chain(&self.mips)
            .map(|slot| slot.take_handle())
            .collect();
        handles.extend(self.cache.take_handles());
        let swapchain = matches!(memory, ImageMemory::Swapchain);
        let destroy = move || {
            for view in views {
                unsafe { Ctx::device().destroy_image_view(view, None) };
            }
            for handle in handles {
                bindless::free_handle(handle);
            }
            if !swapchain {
                unsafe { Ctx::device().destroy_image(image, None) };
            }
            memory.free();
        };
        match swapchain {
            true => destroy(),
            false => lifetime::retire(destroy),
        }
    }
}

/// An image being made: the `vk::Image` with its description and maybe its memory, but no
/// views yet ([`Image::from_raw`] makes those).
pub(crate) struct RawImage<F: Format, U: ImageUsage, D: Dim> {
    pub(crate) internal: ImageInternal,
    extent: D::Extent,
    format: F,
    usage: U,
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
        check_support::<D>(
            vk_format,
            usage_flags,
            flags,
            external,
            D::extent_3d(extent),
            mip_levels,
            layers,
        )?;
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
        // From here on the internal's `Drop` destroys the image (and frees its memory once it
        // has some).
        Ok(RawImage {
            internal: ImageInternal {
                image,
                memory: ImageMemory::Allocation(Allocation::default()),
                format: vk_format,
                aspects: format.aspects(),
                usage: usage_flags,
                mip_levels,
                layers,
                create_flags: flags,
                cache: ViewCache::default(),
                whole: ViewSlot::new(vk::ImageView::null(), ImageHandle::NULL),
                mips: Vec::new(),
                states: Box::new([]),
                access: Mutex::default(),
                header: Header::new(live),
            },
            extent,
            format,
            usage,
        })
    }

    /// Allocates device-local memory from lava's allocator and binds it.
    fn allocate(&mut self) -> Result<()> {
        let image = self.internal.image;
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
        let allocation = Ctx::allocator().allocate(&desc)?;
        let bound = unsafe {
            Ctx::device().bind_image_memory(image, allocation.memory(), allocation.offset())
        };
        self.internal.memory = ImageMemory::Allocation(allocation);
        Ok(bound?)
    }

    #[allow(dead_code)] // Sparse and external images'.
    pub(crate) fn handle(&self) -> vk::Image {
        self.internal.image
    }
}

/// What an image keeps per view: [`Single`] or [`PerMip`].
pub trait Views: crate::sealed::Sealed + std::fmt::Debug + Send + Sync + 'static + Sized {
    /// Whether it keeps a view, handle and state per mip.
    #[doc(hidden)]
    const PER_MIP: bool;
}

/// Views with one view per mip besides the whole one: [`PerMip`].
pub trait MipViews: Views {}

/// One view, handle and layout for all mips: render targets and textures.
#[derive(Debug)]
pub enum Single {}

/// The whole view plus a view, handle and layout per mip, for images whose mips the GPU
/// accesses one at a time.
#[derive(Debug)]
pub enum PerMip {}

impl crate::sealed::Sealed for Single {}
impl Views for Single {
    const PER_MIP: bool = false;
}

impl crate::sealed::Sealed for PerMip {}
impl Views for PerMip {
    const PER_MIP: bool = true;
}
impl MipViews for PerMip {}

/// An image of format `F`, usage `U` and dimension `D`. Static format and usage types are
/// checked at compile time, `format::Dynamic`/`usage::Dynamic` values at runtime.
pub struct Image<F: Format, U: ImageUsage, D: Dim = D2, V: Views = Single> {
    pub(crate) internal: Arc<ImageInternal>,
    extent: D::Extent,
    format: F,
    usage: U,
    _views: PhantomData<fn() -> V>,
}

impl<F: Format, U: ImageUsage, D: Dim, V: Views> std::fmt::Debug for Image<F, U, D, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Image")
            .field("image", &self.internal.image)
            .field("format", &self.internal.format)
            .field("extent", &self.extent)
            .field("mip_levels", &self.internal.mip_levels)
            .field("layers", &self.internal.layers)
            .finish()
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
    fn create(
        format: F,
        usage: U,
        extent: D::Extent,
        mip_levels: u32,
        layers: u32,
    ) -> Result<Self> {
        let flags = usage.create_flags();
        let mut raw = RawImage::<F, U, D>::create(
            format,
            usage,
            extent,
            mip_levels,
            layers,
            flags,
            Default::default(),
        )?;
        raw.allocate()?;
        Self::from_raw(raw)
    }

    /// The image of `raw`, whose memory is bound (or sparse), with its views and states.
    pub(crate) fn from_raw(raw: RawImage<F, U, D>) -> Result<Self> {
        let RawImage {
            mut internal,
            extent,
            format,
            usage,
        } = raw;
        let levels = internal.mip_levels;
        let states = if V::PER_MIP { levels } else { 1 };
        internal.states = (0..states)
            .map(|_| ImageState::UNDEFINED.atomic())
            .collect();
        // Set one by one, so the internal's `Drop` destroys whatever was made before an error.
        internal.whole.view = internal.create_view((0..levels).into(), D::VIEW_TYPE)?;
        if V::PER_MIP {
            for mip in 0..levels {
                let view = match levels {
                    1 => internal.whole.view,
                    _ => internal.create_view((mip..mip + 1).into(), D::VIEW_TYPE)?,
                };
                internal.mips.push(ViewSlot::new(view, ImageHandle::NULL));
            }
        }
        Ok(Image {
            internal: Arc::new(internal),
            extent,
            format,
            usage,
            _views: PhantomData,
        })
    }

    /// Recreates the image at `extent` unless it has that extent already, with the same
    /// format, usage, mips and layers, and returns whether it did. The new image is bound like
    /// the old one (`bind_*`, `bind_mips_*`) at new slots, so its handles differ: write them
    /// wherever the old ones were stored. Views bound on their own (`ImageView::bind`) aren't
    /// bound again. The old image lives on until nothing reaches it any more (frames in flight
    /// read it); the new one's contents are undefined, its name is gone, and the old views'
    /// borrows have ended (`&mut self`). Errors where the old image's description doesn't fit
    /// `extent` (too few texels for its mips).
    #[validation_trace]
    pub fn fit(&mut self, extent: D::Extent) -> Result<bool> {
        if self.extent == extent {
            return Ok(false);
        }
        let old = &self.internal;
        let mut new = RawImage::<F, U, D>::create(
            self.format,
            self.usage,
            extent,
            old.mip_levels,
            old.layers,
            old.create_flags,
            vk::ExternalMemoryHandleTypeFlags::empty(),
        )?;
        new.allocate()?;
        let new = Image::<F, U, D, V>::from_raw(new)?;
        // Bound as the old one was, at new slots.
        let whole = old.whole.handle();
        if !whole.is_null() {
            new.bind_whole_as(whole.kind())?;
        }
        if let Some(first) = old.mips.first().map(ViewSlot::handle)
            && !first.is_null()
        {
            new.bind_mips_as(first.kind())?;
        }
        *self = new;
        Ok(true)
    }

    /// The same image named `name`, for debuggers, the validation layer (with debug utils,
    /// `Config::validation`) and allocator reports.
    pub fn named(mut self, name: &str) -> Self {
        // Only an allocation nothing else holds yet can be renamed.
        if let Some(internal) = Arc::get_mut(&mut self.internal)
            && let ImageMemory::Allocation(allocation) = &mut internal.memory
            && let Err(err) = Ctx::allocator().rename_allocation(allocation, name)
        {
            tracing::warn!(%err, "failed to name an image's allocation");
        }
        Functions::set_debug_name(name, self.internal.image);
        self
    }

    /// The extent of mip 0.
    pub fn extent(&self) -> D::Extent {
        self.extent
    }

    pub fn mip_levels(&self) -> u32 {
        self.internal.mip_levels
    }

    /// Its array layers: 1 unless it is a `D2Array` image made `with_layers`.
    pub fn layers(&self) -> u32 {
        self.internal.layers
    }

    pub fn mip_extent(&self, level: u32) -> D::Extent {
        D::mip_extent(self.extent, level)
    }

    pub fn format(&self) -> F {
        self.format
    }

    pub fn usage(&self) -> U {
        self.usage
    }

    /// The whole view's handle as `K`; panics unless it is bound with `K`'s descriptors (see
    /// [`ImageView::handle`]).
    #[track_caller]
    pub fn handle<K: Allocatable>(&self) -> ImageHandle<K> {
        self.whole_view().handle()
    }

    /// Panics unless the usage has every flag of `required`; does nothing for a static usage.
    pub fn usage_check(&self, required: vk::ImageUsageFlags) {
        self.usage.check(required);
    }

    /// The view over every mip and layer.
    pub fn whole_view(&self) -> ImageView<'_, F, U, D> {
        let whole = &self.internal.whole;
        self.view(
            whole.view,
            (0..self.internal.mip_levels).into(),
            whole.handle(),
        )
    }

    fn view(
        &self,
        view: vk::ImageView,
        mip_range: Range<u32>,
        handle: ImageHandle,
    ) -> ImageView<'_, F, U, D> {
        ImageView {
            internal: &self.internal,
            view,
            mip_range,
            layers: (0..self.internal.layers).into(),
            extent: self.mip_extent(mip_range.start),
            handle,
            format: self.format,
            usage: self.usage,
            dim: D::default(),
        }
    }

    /// The same image typed as `format` and `usage`; panics unless the format is its own and
    /// the usage no more than its own.
    pub fn cast<NF: Format, NU: ImageUsage>(self, format: NF, usage: NU) -> Image<NF, NU, D, V> {
        slice::cast_check(
            (self.format.vk_format(), self.usage.flags()),
            (format.vk_format(), usage.flags()),
        );
        Image {
            internal: self.internal,
            extent: self.extent,
            format,
            usage,
            _views: PhantomData,
        }
    }

    /// The same image with its format and usage as runtime values.
    pub fn erase(self) -> DynamicImage<D, V> {
        let format = format::Dynamic(self.format.vk_format());
        let usage = usage::Dynamic(self.usage.flags());
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

    fn bind_whole<K: Allocatable>(self) -> Result<Self> {
        let handle = self.internal.whole.handle();
        assert!(
            handle.is_null(),
            "the image is bound already ({handle:?}): bind it once, with `bind_combined` for both descriptors"
        );
        self.bind_whole_as(K::BITS)?;
        Ok(self)
    }

    /// Binds the whole view with the descriptors of `kind` (a handle's bits 30 and 31), at new
    /// slots.
    fn bind_whole_as(&self, kind: u32) -> Result<()> {
        let width = ImageHandle::<Dynamic>::width_of(kind);
        let bound = ImageHandle::with_kind(kind, bindless::allocate(width)?);
        self.internal.whole.set_handle(bound);
        bindless::write_descriptors([self.whole_view().shader_image()], [bound])
    }

    /// Binds each mip's view with the descriptors of `kind`, one block of contiguous new slots
    /// for all.
    fn bind_mips_as(&self, kind: u32) -> Result<()> {
        let mips = &self.internal.mips;
        let width = ImageHandle::<Dynamic>::width_of(kind);
        let start = bindless::allocate(mips.len() as u32 * width)?;
        for (level, mip) in mips.iter().enumerate() {
            mip.set_handle(ImageHandle::with_kind(kind, start + level as u32 * width));
        }
        let images = (mips.iter().enumerate()).map(|(level, mip)| {
            let level = level as u32;
            self.view(mip.view, (level..level + 1).into(), mip.handle())
                .shader_image()
        });
        bindless::write_descriptors(images, mips.iter().map(ViewSlot::handle))
    }
}

impl<F: Format, U: ImageUsage, D: Dim, V: MipViews> Image<F, U, D, V> {
    /// The view of mip `level` alone, sized `mip_extent(level)`.
    pub fn mip_view(&self, level: u32) -> ImageView<'_, F, U, D> {
        let mip = &self.internal.mips[level as usize];
        self.view(mip.view, (level..level + 1).into(), mip.handle())
    }

    fn bind_mips<K: Allocatable>(self) -> Result<Self> {
        assert!(
            self.internal.mips.iter().all(|mip| mip.handle().is_null()),
            "the mips are bound already: bind them once, with `bind_mips_combined` for both descriptors"
        );
        self.bind_mips_as(K::BITS)?;
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
impl<F: Format, U: IsHostCopy, D: Dim, V: Views> Image<F, U, D, V> {
    /// Writes a whole mip level of every layer from the host (the layers one after another),
    /// right away, unsynchronised with frames in flight that read the image.
    ///
    /// Panics for a mip it doesn't have, data of another length than the mip's, and combined
    /// depth-stencil formats.
    #[validation_trace]
    pub fn copy_from(&mut self, data: &[u8], mip_level: u32) -> Result<()> {
        self.check_host_copy(data, mip_level);
        let extent = self.mip_extent(mip_level);
        self.check_data_len(data, extent, (0, 0), || format!("mip level {mip_level}"));
        self.host_copy(data, mip_level, Default::default(), extent, (0, 0))
    }

    /// Like `copy_from`, for the `extent` texels at `offset` of every layer; `data` holds that
    /// box per layer, tightly packed. A block-compressed format's box covers whole blocks (or
    /// reaches the mip's edge), as GPU copies' do.
    #[validation_trace]
    pub fn copy_region_from(
        &mut self,
        data: &[u8],
        mip_level: u32,
        offset: D::Offset,
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
        offset: D::Offset,
        extent: D::Extent,
        row_length: u32,
        image_height: u32,
    ) -> Result<()> {
        self.check_host_copy(data, mip_level);
        let mip_extent = self.mip_extent(mip_level);
        let (o, e, m) = (
            D::offset_3d(offset),
            D::extent_3d(extent),
            D::extent_3d(mip_extent),
        );
        assert!(
            dim::box_inside(o, e, m) && !dim::box_is_empty(e),
            "region {extent:?} at {offset:?} does not fit mip level {mip_level} of size {mip_extent:?}"
        );
        let format = self.format;
        let block = slice::copy_block_of(&format, format.aspects());
        slice::check_whole_blocks(block, format.vk_format(), o, e, m);
        self.check_data_len(data, extent, (row_length, image_height), || {
            "the region".to_string()
        });
        self.host_copy(data, mip_level, offset, extent, (row_length, image_height))
    }

    #[track_caller]
    fn check_host_copy(&self, data: &[u8], mip_level: u32) {
        // The image exists, so the device has host copies.
        self.usage_check(vk::ImageUsageFlags::HOST_TRANSFER_EXT);
        assert!(
            mip_level < self.internal.mip_levels,
            "mip level {mip_level} is out of range for an image with {} levels",
            self.internal.mip_levels
        );
        assert!(!data.is_empty(), "no texel data to copy");
        let aspects = self.format.aspects();
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
        let format = self.format;
        let block = slice::copy_block_of(&format, format.aspects());
        let expected = slice::rows_len(block, D::extent_3d(extent), self.internal.layers, rows);
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
        offset: D::Offset,
        extent: D::Extent,
        (row_length, image_height): (u32, u32),
    ) -> Result<()> {
        let image = self.internal.image;
        let aspect = self.format.aspects();
        let states = &self.internal.states;
        // A host copy is no queue's access, so the image keeps its owner. The states move only
        // once the transition succeeded.
        let mut transitions: smallvec::SmallVec<[vk::HostImageLayoutTransitionInfoEXT; 1]> =
            smallvec::SmallVec::new();
        let (mips, general) = (
            (0..self.internal.mip_levels).into(),
            vk::ImageLayout::GENERAL,
        );
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
            .image_offset(D::offset_3d(offset))
            .image_extent(D::extent_3d(extent))
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: aspect,
                mip_level,
                base_array_layer: 0,
                layer_count: self.internal.layers,
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
        Err(Error::Unsupported(format!(
            "{format:?} needs lava's `{feature}` feature"
        )))
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
        let mut text = format!(
            "{:?} images of {format:?} with usage {usage:?}",
            D::IMAGE_TYPE
        );
        if !flags.is_empty() {
            text += &format!(" and flags {flags:?}");
        }
        if !external.is_empty() {
            text += &format!(" and external memory {external:?}");
        }
        text
    };
    let Some(limits) = Ctx::image_format_limits(key) else {
        return Err(Error::Unsupported(format!(
            "the device can't make {}",
            description()
        )));
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
        return Err(Error::other(format!(
            "an image of extent {extent:?} has no texels"
        )));
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

//! Format-, usage- and dimension-typed GPU images with one view over all mips or one per mip, host uploads, type erasure and the drop hook
use ash::vk::TaggedStructure;
use std::{
    mem::{self, ManuallyDrop},
    range::Range,
    sync::OnceLock,
};

use crate::error::{Error, Result};
use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use lava_macros::validation_trace;

use crate::{
    bindless::{self, BindlessHandle},
    command_buffer::ImageUse,
    image::{
        format::{Format, StaticFormat, bc_block},
        slice::ImageView,
        usage::{ImageUsage, IsHostCopy, IsSampled, IsStorage, StaticUsage},
    },
    keepalive::{KeepAlive, MAX_LAYOUTS},
    state::{Ctx, Functions},
};

pub mod dim;
pub mod format;
pub mod slice;
pub mod usage;

pub use dim::{D2, D3, Dim};

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
        let create_info = vk::ImageCreateInfo {
            array_layers: 1,
            extent: D::extent_3d(extent),
            format: format.vk_format(),
            image_type: D::IMAGE_TYPE,
            mip_levels,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            // Without the extension `copy_from` fails, and the flag isn't allowed.
            usage: if Ctx::features().host_image_copy {
                usage.flags()
            } else {
                usage.flags() & !vk::ImageUsageFlags::HOST_TRANSFER_EXT
            },
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };

        let image = unsafe { Ctx::device().create_image(&create_info, None) }?;
        let mut dedicated = vk::MemoryDedicatedRequirements::default();
        let requirements = {
            let mut requirements = vk::MemoryRequirements2::default().push(&mut dedicated);
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

    fn create_view(&self, mips: Range<u32>) -> Result<vk::ImageView> {
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
    /// Runs after the views are destroyed: `Image` declares `views` before `raw`.
    fn drop(&mut self) {
        unsafe { Ctx::device().destroy_image(self.image, None) };
        if let Err(err) = Ctx::allocator().free(mem::take(&mut self.allocation)) {
            tracing::error!(%err, "failed to free image allocation");
        }
    }
}

/// What an image keeps per view: [`Single`] or [`PerMip`]. Only lava implements it.
pub trait Views: std::fmt::Debug + Send + Sync + 'static + Sized {
    /// Whether every mip has its own tracked layout (2 bits of the image's slot in
    /// `keepalive`), or one layout covers them all.
    #[doc(hidden)]
    const PER_MIP: bool;
    #[doc(hidden)]
    fn create<F: Format, U: ImageUsage, D: Dim>(raw: &RawImage<F, U, D>) -> Result<Self>;
    /// The view over every mip and its handle.
    #[doc(hidden)]
    fn whole(&self) -> (vk::ImageView, BindlessHandle);
    #[doc(hidden)]
    fn whole_handle(&mut self) -> &mut BindlessHandle;
}

/// One view over every mip, with one handle and one layout for all of them: render targets,
/// and textures, whose mips are only sampled together and written by the host.
#[derive(Debug)]
pub struct Single {
    view: vk::ImageView,
    handle: BindlessHandle,
}

/// The view over every mip plus one view per mip, each with its handle, and a layout per mip:
/// for images whose mips the GPU accesses one at a time.
#[derive(Debug)]
pub struct PerMip {
    view: vk::ImageView,
    handle: BindlessHandle,
    mips: Box<[MipView]>,
}

#[derive(Debug)]
struct MipView {
    view: vk::ImageView,
    handle: BindlessHandle,
}

impl Views for Single {
    const PER_MIP: bool = false;
    fn create<F: Format, U: ImageUsage, D: Dim>(raw: &RawImage<F, U, D>) -> Result<Self> {
        Ok(Single {
            view: raw.create_view((0..raw.mip_levels).into())?,
            handle: BindlessHandle::none(),
        })
    }
    fn whole(&self) -> (vk::ImageView, BindlessHandle) {
        (self.view, self.handle)
    }
    fn whole_handle(&mut self) -> &mut BindlessHandle {
        &mut self.handle
    }
}

impl Views for PerMip {
    const PER_MIP: bool = true;
    fn create<F: Format, U: ImageUsage, D: Dim>(raw: &RawImage<F, U, D>) -> Result<Self> {
        // Built up front, so `Drop` destroys whatever was created before an error.
        let mut views = PerMip {
            view: vk::ImageView::null(),
            handle: BindlessHandle::none(),
            mips: Box::new([]),
        };
        views.view = raw.create_view((0..raw.mip_levels).into())?;
        let mut mips = Vec::with_capacity(raw.mip_levels as usize);
        for mip in 0..raw.mip_levels {
            match raw.create_view((mip..mip + 1).into()) {
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
    fn whole(&self) -> (vk::ImageView, BindlessHandle) {
        (self.view, self.handle)
    }
    fn whole_handle(&mut self) -> &mut BindlessHandle {
        &mut self.handle
    }
}

static ON_DROP: OnceLock<fn(BindlessHandle)> = OnceLock::new();

/// Sets the function a destroyed image calls with each handle it has that isn't null (its
/// whole view's, then each mip's), right before the view is destroyed, so whoever picked the
/// indices can hand them out again. That is when its owner drops it, or, if a view or a frame
/// in flight still keeps it alive (`keepalive`), when the last of them is gone. It can be set
/// once.
pub fn on_drop(hook: fn(BindlessHandle)) -> Result<()> {
    ON_DROP
        .set(hook)
        .map_err(|_| Error::message("the image drop hook is already set"))
}

fn release(view: vk::ImageView, handle: BindlessHandle) {
    if handle != BindlessHandle::none()
        && let Some(hook) = ON_DROP.get()
    {
        hook(handle);
    }
    unsafe { Ctx::device().destroy_image_view(view, None) };
}

impl Drop for Single {
    fn drop(&mut self) {
        release(self.view, self.handle);
    }
}

impl Drop for PerMip {
    fn drop(&mut self) {
        release(self.view, self.handle);
        for mip in &self.mips {
            release(mip.view, mip.handle);
        }
    }
}

/// An image of format `F`, usage `U` and dimension `D`. Its format and usage are types
/// (`format::R8Unorm`, `usage::Sampled`), which the command buffer's bounds check at compile
/// time, or `format::Dynamic`/`usage::Dynamic` values, which it checks at runtime
/// (`aspect_check`, `texel_check`, `usage_check`). A [`Single`] image (the default) has one view
/// over all its mips, a [`MipChain`] also one per mip.
///
/// Dropping it destroys it once nothing keeps it alive any more: its views, and the frames in
/// flight that used it, each count (`keepalive`).
#[derive(Debug)]
pub struct Image<F: Format, U: ImageUsage, D: Dim = D2, V: Views = Single> {
    // Destroyed in this order: the views and their handles first, then the image.
    views: ManuallyDrop<V>,
    raw: ManuallyDrop<RawImage<F, U, D>>,
    /// The image's count and tracked layouts (one, or one per mip with `V::PER_MIP`).
    owner: ManuallyDrop<KeepAlive>,
}

impl<F: Format, U: ImageUsage, D: Dim, V: Views> Drop for Image<F, U, D, V> {
    fn drop(&mut self) {
        let (views, raw, owner) = unsafe {
            (
                ManuallyDrop::take(&mut self.views),
                ManuallyDrop::take(&mut self.raw),
                ManuallyDrop::take(&mut self.owner),
            )
        };
        owner.release((views, raw));
    }
}

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
        // One tracked layout per mip, 2 bits of the image's slot each.
        assert!(!V::PER_MIP || mip_levels <= MAX_LAYOUTS, "a mip chain has at most {MAX_LAYOUTS} mips");
        let raw = RawImage::new(format, usage, extent, mip_levels)?;
        let views = V::create(&raw)?;
        Ok(Image {
            views: ManuallyDrop::new(views),
            raw: ManuallyDrop::new(raw),
            owner: ManuallyDrop::new(KeepAlive::new()),
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
    pub fn whole_view(&self) -> ImageView<F, U, D> {
        let (view, handle) = self.views.whole();
        let layouts = if V::PER_MIP { self.raw.mip_levels } else { 1 };
        self.view(view, (0..self.raw.mip_levels).into(), handle, (0..layouts).into())
    }

    /// `layouts`: the layouts of the image's slot the view tracks.
    fn view(
        &self,
        view: vk::ImageView,
        mip_range: Range<u32>,
        handle: BindlessHandle,
        layouts: Range<u32>,
    ) -> ImageView<F, U, D> {
        ImageView {
            image: self.raw.image,
            view,
            mip_range,
            extent: self.mip_extent(mip_range.start),
            handle,
            keep: KeepAlive::clone(&self.owner),
            layouts,
            format: self.raw.format,
            usage: self.raw.usage,
            dim: D::default(),
        }
    }

    /// The parts, without dropping (destroying) the image.
    fn into_parts(self) -> (V, RawImage<F, U, D>, KeepAlive) {
        let mut this = ManuallyDrop::new(self);
        unsafe {
            (
                ManuallyDrop::take(&mut this.views),
                ManuallyDrop::take(&mut this.raw),
                ManuallyDrop::take(&mut this.owner),
            )
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
        let (views, raw, owner) = self.into_parts();
        Image {
            views: ManuallyDrop::new(views),
            raw: ManuallyDrop::new(raw.cast(format, usage)),
            owner: ManuallyDrop::new(owner),
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

impl<F: Format, U: IsSampled, D: Dim, V: Views> Image<F, U, D, V> {
    /// Binds the whole view at `index` of the sampled set, right away:
    /// `Image::new(extent)?.bind_sampled(index)`.
    #[validation_trace]
    pub fn bind_sampled(mut self, index: u32) -> Self {
        self.views.whole_handle().descriptor_index_set0 = index;
        bindless::bind([self.whole_view()], [BindlessHandle::sampled(index)]);
        self
    }
}

impl<F: Format, U: IsStorage, D: Dim, V: Views> Image<F, U, D, V> {
    /// Binds the whole view at `index` of the storage set, right away. An image with both
    /// usages can be bound in both sets: `.bind_sampled(a).bind_storage(b)`.
    #[validation_trace]
    pub fn bind_storage(mut self, index: u32) -> Self {
        self.views.whole_handle().descriptor_index_set1 = index;
        bindless::bind([self.whole_view()], [BindlessHandle::storage(index)]);
        self
    }
}

impl<F: Format, U: ImageUsage, D: Dim> MipChain<F, U, D> {
    /// The view of mip `level` alone, sized `mip_extent(level)`: for a storage write, an
    /// attachment or a copy of one mip. The whole view samples across mips.
    pub fn mip_view(&self, level: u32) -> ImageView<F, U, D> {
        let mip = &self.views.mips[level as usize];
        let range = (level..level + 1).into();
        self.view(mip.view, range, mip.handle, range)
    }

    /// Binds mip `i`'s view at `first + i` of the sampled (`sampled`) or the storage set.
    fn bind_mips(mut self, first: u32, sampled: bool) -> Self {
        for (i, mip) in self.views.mips.iter_mut().enumerate() {
            let index = first + i as u32;
            if sampled {
                mip.handle.descriptor_index_set0 = index;
            } else {
                mip.handle.descriptor_index_set1 = index;
            }
        }
        let levels = 0..self.mip_levels();
        bindless::bind(
            levels.clone().map(|level| self.mip_view(level)),
            levels.map(|level| match sampled {
                true => BindlessHandle::sampled(first + level),
                false => BindlessHandle::storage(first + level),
            }),
        );
        self
    }
}

impl<F: Format, U: IsSampled, D: Dim> MipChain<F, U, D> {
    /// Binds mip `i`'s view at `first + i` of the sampled set, right away.
    #[validation_trace]
    pub fn bind_mips_sampled(self, first: u32) -> Self {
        self.bind_mips(first, true)
    }
}

impl<F: Format, U: IsStorage, D: Dim> MipChain<F, U, D> {
    /// Binds mip `i`'s view at `first + i` of the storage set, right away.
    #[validation_trace]
    pub fn bind_mips_storage(self, first: u32) -> Self {
        self.bind_mips(first, false)
    }
}

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
        self.usage_check(vk::ImageUsageFlags::HOST_TRANSFER_EXT);
        if !Ctx::features().host_image_copy {
            return Err(Error::message(
                "host image copies need VK_EXT_host_image_copy on host-visible device memory, which this device lacks",
            ));
        }
        if mip_level >= self.raw.mip_levels {
            return Err(Error::message(format!(
                "mip level {mip_level} is out of range for an image with {} levels",
                self.raw.mip_levels
            )));
        }
        if data.is_empty() {
            return Err(Error::message("no texel data to copy"));
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
        // Every mip not in `GENERAL` yet moves there, as if a command accessed the whole view.
        let to_general: ImageUse = self.whole_view().into_access(
            vk::PipelineStageFlags2::HOST,
            vk::AccessFlags2::HOST_WRITE,
            vk::ImageLayout::GENERAL,
        );
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
fn expected_data_len(
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

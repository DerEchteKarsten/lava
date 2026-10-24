//! Sparse resources (`sparse-binding` with its residency features): partially resident buffers and images created without memory, whose pages a graphics timeline binds and unbinds (`Timeline::bind_sparse`, ordered with its submissions by its semaphore), lava allocating each resident page from its allocator and freeing it once no submission reaches it
#![cfg_attr(
    not(all(
        feature = "sparse-residency-buffer",
        any(
            feature = "sparse-residency-image-2d",
            feature = "sparse-residency-image-3d"
        )
    )),
    allow(dead_code, unused_imports)
)]
use std::{
    collections::HashMap,
    marker::PhantomData,
    ops::Deref,
    sync::{Arc, Mutex},
};

use glam::UVec3;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use lava_macros::validation_trace;
use smallvec::SmallVec;

use crate::{
    buffer::{self, BufferInternal, BufferMemory, slice::BufferSlice},
    error::{Error, Result},
    features::{MissingFeature, On, gate},
    global::Live,
    image::{
        D2, Dim, Image, ImageMemory, RawImage, Single,
        dim::{box_inside, box_is_empty},
        format::{ColorAspect, Format, StaticFormat},
        usage::{ImageUsage, StaticUsage},
    },
    lifetime,
    object::Header,
    queue::{Graphics, Timeline, Wait, lock},
    state::Ctx,
    vk,
};

/// Where a page lies: a buffer's `index`-th page, a tile of an image's mip (in tiles of its
/// granularity), or its mip tail.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Page {
    Buffer(u64),
    Tile { mip: u32, x: u32, y: u32, z: u32 },
    MipTail,
}

/// The pages of a sparse resource and what their memory needs; the resource's internal frees
/// them after it.
#[derive(Debug)]
pub(crate) struct Pages {
    requirements: vk::MemoryRequirements,
    /// Buffers' pages are linear, optimal images' not (gpu-allocator keeps them apart).
    linear: bool,
    resident: Mutex<HashMap<Page, Allocation>>,
}

impl Pages {
    fn new(requirements: vk::MemoryRequirements, linear: bool) -> Self {
        Pages {
            requirements,
            linear,
            resident: Mutex::new(HashMap::new()),
        }
    }

    /// A fresh page of `size` bytes (a multiple of the page size) in the resource's memory types.
    fn allocate(&self, size: u64) -> Result<Allocation> {
        let requirements = vk::MemoryRequirements {
            size,
            ..self.requirements
        };
        let desc = AllocationCreateDesc {
            name: "SparsePage",
            requirements,
            location: gpu_allocator::MemoryLocation::GpuOnly,
            linear: self.linear,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };
        Ok(Ctx::allocator().allocate(&desc)?)
    }
}

impl Drop for Pages {
    fn drop(&mut self) {
        let resident = std::mem::take(self.resident.get_mut().unwrap_or_else(|e| e.into_inner()));
        for (_, allocation) in resident {
            if let Err(err) = Ctx::allocator().free(allocation) {
                tracing::error!(%err, "failed to free a sparse page");
            }
        }
    }
}

/// The error of a resource whose residency feature the device lacks.
fn residency_check<F: On>(feature: &'static str) -> Result<()> {
    match F::on() {
        true => Ok(()),
        false => Err(MissingFeature { feature }.into()),
    }
}

#[cfg(feature = "sparse-residency-buffer")]
/// A partially resident buffer of `T`s (`sparse-residency-buffer`): created without memory, its
/// pages ([`Self::page_size`] bytes each) become resident when [`Timeline::bind_sparse`] binds
/// them. Commands take its slices like any buffer's; reads of pages that aren't resident return
/// undefined values (zeros where the device has `residencyNonResidentStrict`) and writes to them
/// are dropped. The host can't access it.
#[derive(Debug)]
pub struct SparseBuffer<T: Copy + 'static> {
    /// Its pages are its memory, freed after the buffer.
    internal: Arc<BufferInternal>,
    _type: PhantomData<T>,
}

#[cfg(feature = "sparse-residency-buffer")]
impl<T: Copy + 'static> SparseBuffer<T> {
    /// A buffer of `len` elements with no page resident, with every usage of [`buffer::USAGE`].
    /// `MissingFeature` without `sparse-residency-buffer`.
    #[validation_trace]
    pub fn new(len: usize) -> Result<Self> {
        let live = Live::new();
        residency_check::<gate::SparseResidencyBuffer>("sparse-residency-buffer")?;
        let num_bytes = (len as u64)
            .checked_mul(size_of::<T>() as u64)
            .filter(|&bytes| bytes > 0)
            .ok_or_else(|| {
                Error::other(format!(
                    "a sparse buffer of {len} elements of {} bytes",
                    size_of::<T>()
                ))
            })?;
        let mut usage = buffer::USAGE
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        if <gate::AccelerationStructure as On>::on() {
            usage |= vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;
        }
        let families: SmallVec<[u32; 3]> = Ctx::get().queues.iter().map(|f| f.index).collect();
        let mut info = vk::BufferCreateInfo::default()
            .flags(vk::BufferCreateFlags::SPARSE_BINDING | vk::BufferCreateFlags::SPARSE_RESIDENCY)
            .size(num_bytes)
            .usage(usage);
        if families.len() > 1 {
            info = info
                .sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&families);
        }
        let device = Ctx::device();
        let handle = unsafe { device.create_buffer(&info, None)? };
        let requirements = unsafe { device.get_buffer_memory_requirements(handle) };
        let address = unsafe {
            device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(handle))
        };
        Ok(SparseBuffer {
            internal: Arc::new(BufferInternal {
                handle,
                address,
                num_bytes,
                memory: BufferMemory::Sparse(Pages::new(requirements, true)),
                header: Header::new(live),
            }),
            _type: PhantomData,
        })
    }

    fn pages(&self) -> &Pages {
        match &self.internal.memory {
            BufferMemory::Sparse(pages) => pages,
            _ => unreachable!("a sparse buffer's memory is its pages"),
        }
    }

    /// The bytes of a page: what one resident page holds, and where pages start.
    pub fn page_size(&self) -> u64 {
        self.pages().requirements.alignment
    }

    /// How many pages it has.
    pub fn page_count(&self) -> u64 {
        self.pages().requirements.size.div_ceil(self.page_size())
    }

    /// Whether page `index` is resident, as far as the binds submitted so far made it.
    pub fn is_resident(&self, index: u64) -> bool {
        lock(&self.pages().resident).contains_key(&Page::Buffer(index))
    }

    pub fn len(&self) -> usize {
        (self.internal.num_bytes / size_of::<T>() as u64) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn address(&self) -> u64 {
        self.internal.address
    }

    /// The `index` range of its elements; panics past its end. Commands don't take it as fully
    /// bound.
    #[track_caller]
    pub fn range<R: std::ops::RangeBounds<usize>>(&self, index: R) -> BufferSlice<'_, T> {
        let internal = &self.internal;
        buffer::slice::new_slice(index, internal, internal.address, internal.num_bytes)
    }

    /// The `index` range of its bytes; panics past its end.
    #[track_caller]
    pub fn byte_range<R: std::ops::RangeBounds<usize>>(&self, index: R) -> BufferSlice<'_, u8> {
        let internal = &self.internal;
        buffer::slice::new_slice(index, internal, internal.address, internal.num_bytes)
    }
}

/// The mip tail of a sparse image: the mips from `first_mip` on, whose memory is bound as one
/// block of `size` bytes ([`SparseBind::mip_tail`]), since they are smaller than a tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MipTail {
    pub first_mip: u32,
    pub size: u64,
    offset: u64,
}

#[cfg(any(
    feature = "sparse-residency-image-2d",
    feature = "sparse-residency-image-3d"
))]
/// A partially resident image (`sparse-residency-image-2d`, or `-3d` for [`crate::image::D3`]):
/// created without memory, its tiles ([`Self::granularity`] texels of a mip) and its mip tail
/// become resident when [`Timeline::bind_sparse`] binds them. It derefs to its [`Image`], whose
/// views, bindings and commands work as any image's; accesses of what isn't resident read
/// undefined values (zeros where the device has `residencyNonResidentStrict`) and drop writes.
/// Colour formats only, one layer.
#[derive(Debug)]
pub struct SparseImage<F: Format, U: ImageUsage, D: Dim = D2> {
    /// Its pages are its memory, freed after the image.
    image: Image<F, U, D, Single>,
    granularity: vk::Extent3D,
    mip_tail: Option<MipTail>,
}

#[cfg(any(
    feature = "sparse-residency-image-2d",
    feature = "sparse-residency-image-3d"
))]
impl<F: Format, U: ImageUsage, D: Dim> Deref for SparseImage<F, U, D> {
    type Target = Image<F, U, D, Single>;
    fn deref(&self) -> &Self::Target {
        &self.image
    }
}

#[cfg(any(
    feature = "sparse-residency-image-2d",
    feature = "sparse-residency-image-3d"
))]
impl<F: StaticFormat + ColorAspect, U: StaticUsage, D: Dim> SparseImage<F, U, D> {
    /// An image of `extent` and `mip_levels` mips with nothing resident.
    #[validation_trace]
    pub fn new(extent: D::Extent, mip_levels: u32) -> Result<Self> {
        Self::with_params(F::default(), U::default(), extent, mip_levels)
    }
}

#[cfg(any(
    feature = "sparse-residency-image-2d",
    feature = "sparse-residency-image-3d"
))]
impl<F: Format + ColorAspect, U: ImageUsage, D: Dim> SparseImage<F, U, D> {
    /// Like `new`, for any format and usage. `MissingFeature` without the residency feature of
    /// its dimension, `Unsupported` where the device has no sparse images of its description.
    #[validation_trace]
    pub fn with_params(format: F, usage: U, extent: D::Extent, mip_levels: u32) -> Result<Self> {
        match D::IMAGE_TYPE {
            vk::ImageType::TYPE_3D => {
                residency_check::<gate::SparseResidencyImage3D>("sparse-residency-image-3d")?
            }
            _ => residency_check::<gate::SparseResidencyImage2D>("sparse-residency-image-2d")?,
        }
        let colour = vk::ImageAspectFlags::COLOR;
        assert!(
            format.aspects() == colour,
            "sparse images of {:?} aren't supported: lava binds colour formats' pages",
            format.vk_format()
        );
        let vk_format = format.vk_format();
        let device = Ctx::device();
        let instance = crate::state::Functions::instance();
        let physical = Ctx::physical_device().handle;
        let info = vk::PhysicalDeviceSparseImageFormatInfo2::default()
            .format(vk_format)
            .ty(D::IMAGE_TYPE)
            .samples(vk::SampleCountFlags::TYPE_1)
            .usage(usage.flags())
            .tiling(vk::ImageTiling::OPTIMAL);
        let count = unsafe {
            instance.get_physical_device_sparse_image_format_properties2_len(physical, &info)
        };
        let mut properties = vec![vk::SparseImageFormatProperties2::default(); count];
        unsafe {
            instance.get_physical_device_sparse_image_format_properties2(
                physical,
                &info,
                &mut properties,
            )
        };
        if !properties
            .iter()
            .any(|p| p.properties.aspect_mask.contains(colour))
        {
            return Err(Error::Unsupported(format!(
                "the device has no sparse {:?} images of {vk_format:?} with usage {:?}",
                D::IMAGE_TYPE,
                usage.flags()
            )));
        }
        let flags = usage.create_flags()
            | vk::ImageCreateFlags::SPARSE_BINDING
            | vk::ImageCreateFlags::SPARSE_RESIDENCY;
        let mut raw = RawImage::<F, U, D>::create(
            format,
            usage,
            extent,
            mip_levels,
            1,
            flags,
            vk::ExternalMemoryHandleTypeFlags::empty(),
        )?;
        let handle = raw.handle();
        let requirements = unsafe { device.get_image_memory_requirements(handle) };
        let info = vk::ImageSparseMemoryRequirementsInfo2::default().image(handle);
        let count = unsafe { device.get_image_sparse_memory_requirements2_len(&info) };
        let mut sparse = vec![vk::SparseImageMemoryRequirements2::default(); count];
        unsafe { device.get_image_sparse_memory_requirements2(&info, &mut sparse) };
        let Some(colour_requirements) = (sparse.iter())
            .map(|r| r.memory_requirements)
            .find(|r| r.format_properties.aspect_mask.contains(colour))
        else {
            return Err(Error::Unsupported(format!(
                "the device reports no sparse memory requirements of {vk_format:?}'s colour aspect"
            )));
        };
        let mip_tail =
            (colour_requirements.image_mip_tail_first_lod < mip_levels).then_some(MipTail {
                first_mip: colour_requirements.image_mip_tail_first_lod,
                size: colour_requirements.image_mip_tail_size,
                offset: colour_requirements.image_mip_tail_offset,
            });
        raw.internal.memory = ImageMemory::Sparse(Pages::new(requirements, false));
        Ok(SparseImage {
            image: Image::from_raw(raw)?,
            granularity: colour_requirements.format_properties.image_granularity,
            mip_tail,
        })
    }

    fn pages(&self) -> &Pages {
        self.image.internal.sparse_pages()
    }

    /// The texels of a tile, which offsets and extents of tile binds are multiples of (or
    /// reach the mip's edge).
    pub fn granularity(&self) -> UVec3 {
        let g = self.granularity;
        UVec3::new(g.width, g.height, g.depth)
    }

    /// The bytes of a tile's memory.
    pub fn page_size(&self) -> u64 {
        self.pages().requirements.alignment
    }

    /// Its mip tail, `None` where every mip has whole tiles.
    pub fn mip_tail(&self) -> Option<MipTail> {
        self.mip_tail
    }

    /// Whether the tile at `tile` (in tiles) of mip `mip` is resident, as far as the binds
    /// submitted so far made it.
    pub fn is_resident(&self, mip: u32, tile: UVec3) -> bool {
        let page = Page::Tile {
            mip,
            x: tile.x,
            y: tile.y,
            z: tile.z,
        };
        lock(&self.pages().resident).contains_key(&page)
    }
}

/// One change of residency for [`Timeline::bind_sparse`]: pages of a [`SparseBuffer`], tiles of
/// a [`SparseImage`]'s mip, or its mip tail, made resident (with fresh memory lava allocates)
/// or not (their memory is freed once no submission reaches it). Pages that already are what
/// a bind asks are left alone.
#[derive(Clone, Copy, Debug)]
pub struct SparseBind<'a> {
    target: Target<'a>,
    resident: bool,
}

#[derive(Clone, Copy, Debug)]
enum Target<'a> {
    Buffer {
        buffer: vk::Buffer,
        pages: &'a Pages,
        range: (u64, u64),
    },
    Tiles {
        image: vk::Image,
        pages: &'a Pages,
        granularity: vk::Extent3D,
        mip: u32,
        mip_extent: vk::Extent3D,
        offset: vk::Offset3D,
        extent: vk::Extent3D,
    },
    MipTail {
        image: vk::Image,
        pages: &'a Pages,
        tail: MipTail,
    },
}

impl<'a> SparseBind<'a> {
    /// Pages `pages` of `buffer`. Panics for pages it doesn't have.
    #[cfg(feature = "sparse-residency-buffer")]
    #[track_caller]
    pub fn buffer<T: Copy + 'static>(
        buffer: &'a SparseBuffer<T>,
        pages: std::ops::Range<u64>,
        resident: bool,
    ) -> Self {
        assert!(
            pages.start <= pages.end && pages.end <= buffer.page_count(),
            "pages {pages:?} of a sparse buffer of {} pages",
            buffer.page_count()
        );
        SparseBind {
            target: Target::Buffer {
                buffer: buffer.internal.handle,
                pages: buffer.pages(),
                range: (pages.start, pages.end),
            },
            resident,
        }
    }

    #[cfg(any(
        feature = "sparse-residency-image-2d",
        feature = "sparse-residency-image-3d"
    ))]
    /// The tiles of the box of `extent` texels at `offset` of mip `mip` of `image`. Panics
    /// unless the box lies in the mip, starts on a tile and covers whole tiles or reaches the
    /// mip's edge, and the mip isn't in the mip tail.
    #[track_caller]
    pub fn image<F: Format + ColorAspect, U: ImageUsage, D: Dim>(
        image: &'a SparseImage<F, U, D>,
        mip: u32,
        offset: D::Offset,
        extent: D::Extent,
        resident: bool,
    ) -> Self {
        assert!(
            mip < image.mip_levels(),
            "mip {mip} of an image of {} mips",
            image.mip_levels()
        );
        if let Some(tail) = image.mip_tail {
            assert!(
                mip < tail.first_mip,
                "mip {mip} lies in the mip tail (from mip {}): bind it with `SparseBind::mip_tail`",
                tail.first_mip
            );
        }
        let mip_extent = image.mip_extent(mip);
        let (o, e, m) = (
            D::offset_3d(offset),
            D::extent_3d(extent),
            D::extent_3d(mip_extent),
        );
        assert!(
            box_inside(o, e, m) && !box_is_empty(e),
            "the box of {extent:?} at {offset:?} doesn't fit mip {mip} of {mip_extent:?}"
        );
        let g = image.granularity;
        let whole = |o: i32, e: u32, m: u32, g: u32| {
            (o as u32).is_multiple_of(g) && (e.is_multiple_of(g) || o as u32 + e == m)
        };
        assert!(
            whole(o.x, e.width, m.width, g.width)
                && whole(o.y, e.height, m.height, g.height)
                && whole(o.z, e.depth, m.depth, g.depth),
            "sparse binds cover whole tiles of {g:?} texels (or reach the mip's edge), not {e:?} at {o:?}"
        );
        SparseBind {
            target: Target::Tiles {
                image: image.whole_view().image(),
                pages: image.internal.sparse_pages(),
                granularity: g,
                mip,
                mip_extent: m,
                offset: o,
                extent: e,
            },
            resident,
        }
    }

    /// The mip tail of `image`. Panics where it has none.
    #[cfg(any(
        feature = "sparse-residency-image-2d",
        feature = "sparse-residency-image-3d"
    ))]
    #[track_caller]
    pub fn mip_tail<F: Format + ColorAspect, U: ImageUsage, D: Dim>(
        image: &'a SparseImage<F, U, D>,
        resident: bool,
    ) -> Self {
        let tail = image
            .mip_tail
            .expect("the image has no mip tail: every mip has whole tiles");
        SparseBind {
            target: Target::MipTail {
                image: image.whole_view().image(),
                pages: image.internal.sparse_pages(),
                tail,
            },
            resident,
        }
    }
}

/// What one `vkQueueBindSparse` binds, and what it allocated and frees.
#[derive(Default)]
struct Batch<'a> {
    buffers: Vec<(vk::Buffer, vk::SparseMemoryBind)>,
    opaque: Vec<(vk::Image, vk::SparseMemoryBind)>,
    tiles: Vec<(vk::Image, vk::SparseImageMemoryBind)>,
    /// Pages made resident, kept once the bind was submitted.
    allocated: Vec<(&'a Pages, Page, Allocation)>,
    /// Pages made non-resident, freed once the bind was submitted.
    unbound: Vec<(&'a Pages, Page)>,
}

impl<'a> Batch<'a> {
    /// Adds `page`'s bind (`size` bytes at `offset` of its resource, through `bind`).
    fn page(
        &mut self,
        pages: &'a Pages,
        page: Page,
        resident: bool,
        size: u64,
        bind: impl FnOnce(&mut Self, vk::DeviceMemory, u64),
    ) -> Result<()> {
        let is = lock(&pages.resident).contains_key(&page);
        let pending = |batch: &Self| {
            batch
                .allocated
                .iter()
                .any(|(p, k, _)| std::ptr::eq(*p, pages) && *k == page)
                || batch
                    .unbound
                    .iter()
                    .any(|(p, k)| std::ptr::eq(*p, pages) && *k == page)
        };
        if is == resident || pending(self) {
            return Ok(());
        }
        if resident {
            let allocation = pages.allocate(size)?;
            let (memory, offset) = (unsafe { allocation.memory() }, allocation.offset());
            self.allocated.push((pages, page, allocation));
            bind(self, memory, offset);
        } else {
            self.unbound.push((pages, page));
            bind(self, vk::DeviceMemory::null(), 0);
        }
        Ok(())
    }

    fn add(&mut self, bind: &SparseBind<'a>) -> Result<()> {
        let resident = bind.resident;
        match bind.target {
            Target::Buffer {
                buffer,
                pages,
                range,
            } => {
                let (page_size, total) = (pages.requirements.alignment, pages.requirements.size);
                for index in range.0..range.1 {
                    let resource_offset = index * page_size;
                    let size = page_size.min(total - resource_offset);
                    self.page(
                        pages,
                        Page::Buffer(index),
                        resident,
                        page_size,
                        |batch, memory, memory_offset| {
                            batch.buffers.push((
                                buffer,
                                vk::SparseMemoryBind {
                                    resource_offset,
                                    size,
                                    memory,
                                    memory_offset,
                                    flags: vk::SparseMemoryBindFlags::empty(),
                                },
                            ))
                        },
                    )?;
                }
            }
            Target::Tiles {
                image,
                pages,
                granularity: g,
                mip,
                mip_extent,
                offset,
                extent,
            } => {
                let tiles = |o: i32, e: u32, g: u32| (o as u32 / g)..(o as u32 + e).div_ceil(g);
                let page_size = pages.requirements.alignment;
                for z in tiles(offset.z, extent.depth, g.depth) {
                    for y in tiles(offset.y, extent.height, g.height) {
                        for x in tiles(offset.x, extent.width, g.width) {
                            let tile_offset = vk::Offset3D {
                                x: (x * g.width) as i32,
                                y: (y * g.height) as i32,
                                z: (z * g.depth) as i32,
                            };
                            let tile_extent = vk::Extent3D {
                                width: g.width.min(mip_extent.width - x * g.width),
                                height: g.height.min(mip_extent.height - y * g.height),
                                depth: g.depth.min(mip_extent.depth - z * g.depth),
                            };
                            let page = Page::Tile { mip, x, y, z };
                            self.page(
                                pages,
                                page,
                                resident,
                                page_size,
                                |batch, memory, memory_offset| {
                                    batch.tiles.push((
                                        image,
                                        vk::SparseImageMemoryBind {
                                            subresource: vk::ImageSubresource {
                                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                                mip_level: mip,
                                                array_layer: 0,
                                            },
                                            offset: tile_offset,
                                            extent: tile_extent,
                                            memory,
                                            memory_offset,
                                            flags: vk::SparseMemoryBindFlags::empty(),
                                        },
                                    ))
                                },
                            )?;
                        }
                    }
                }
            }
            Target::MipTail { image, pages, tail } => {
                let size = tail.size.next_multiple_of(pages.requirements.alignment);
                self.page(
                    pages,
                    Page::MipTail,
                    resident,
                    size,
                    |batch, memory, memory_offset| {
                        batch.opaque.push((
                            image,
                            vk::SparseMemoryBind {
                                resource_offset: tail.offset,
                                size: tail.size,
                                memory,
                                memory_offset,
                                flags: vk::SparseMemoryBindFlags::empty(),
                            },
                        ))
                    },
                )?;
            }
        }
        Ok(())
    }

    /// Frees what it allocated, for a bind that was never submitted.
    fn discard(self) {
        let mut allocator = Ctx::allocator();
        for (_, _, allocation) in self.allocated {
            if let Err(err) = allocator.free(allocation) {
                tracing::error!(%err, "failed to free a sparse page");
            }
        }
    }

    /// Keeps what it allocated and retires what it unbound, for a submitted bind.
    fn commit(self) {
        for (pages, page, allocation) in self.allocated {
            lock(&pages.resident).insert(page, allocation);
        }
        let freed: Vec<Allocation> = (self.unbound.into_iter())
            .filter_map(|(pages, page)| lock(&pages.resident).remove(&page))
            .collect();
        if !freed.is_empty() {
            lifetime::retire(move || {
                let mut allocator = Ctx::allocator();
                for allocation in freed {
                    if let Err(err) = allocator.free(allocation) {
                        tracing::error!(%err, "failed to free a sparse page");
                    }
                }
            });
        }
    }
}

/// Groups `binds` by their resource, for `VkBindSparseInfo`.
fn grouped<R: Copy + PartialEq, B: Copy>(binds: &[(R, B)]) -> Vec<(R, Vec<B>)> {
    let mut groups: Vec<(R, Vec<B>)> = Vec::new();
    for &(resource, bind) in binds {
        match groups.iter_mut().find(|(r, _)| *r == resource) {
            Some((_, group)) => group.push(bind),
            None => groups.push((resource, vec![bind])),
        }
    }
    groups
}

impl Timeline<Graphics> {
    /// Binds and unbinds the pages of `binds` in one `vkQueueBindSparse` on the timeline's
    /// queue: after every submission of the timeline before it, and before every one after it
    /// (it waits for and signals the timeline's semaphore). Another timeline that uses the
    /// resources waits for the returned `Wait`. lava allocates the memory of pages made
    /// resident and frees that of pages made non-resident once no submission reaches it.
    /// `MissingFeature` without `sparse-binding`; on an error nothing is bound.
    #[validation_trace]
    pub fn bind_sparse(&mut self, binds: &[SparseBind<'_>]) -> Result<Wait> {
        use crate::features::Feature;
        gate::SparseBinding::check()?;
        let mut batch = Batch::default();
        for bind in binds {
            if let Err(err) = batch.add(bind) {
                batch.discard();
                return Err(err);
            }
        }
        let buffers = grouped(&batch.buffers);
        let opaque = grouped(&batch.opaque);
        let tiles = grouped(&batch.tiles);
        let buffer_infos: Vec<_> = (buffers.iter())
            .map(|(buffer, binds)| {
                vk::SparseBufferMemoryBindInfo::default()
                    .buffer(*buffer)
                    .binds(binds)
            })
            .collect();
        let opaque_infos: Vec<_> = (opaque.iter())
            .map(|(image, binds)| {
                vk::SparseImageOpaqueMemoryBindInfo::default()
                    .image(*image)
                    .binds(binds)
            })
            .collect();
        let tile_infos: Vec<_> = (tiles.iter())
            .map(|(image, binds)| {
                vk::SparseImageMemoryBindInfo::default()
                    .image(*image)
                    .binds(binds)
            })
            .collect();
        let submitted = self.submit_sparse(|queue, wait, (signal, value)| {
            let wait_values: SmallVec<[u64; 1]> = wait.iter().map(|&(_, v)| v).collect();
            let wait_semaphores: SmallVec<[vk::Semaphore; 1]> =
                wait.iter().map(|&(s, _)| s).collect();
            let signal_values = [value];
            let signal_semaphores = [signal];
            let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
                .wait_semaphore_values(&wait_values)
                .signal_semaphore_values(&signal_values);
            let info = vk::BindSparseInfo::default()
                .wait_semaphores(&wait_semaphores)
                .buffer_binds(&buffer_infos)
                .image_opaque_binds(&opaque_infos)
                .image_binds(&tile_infos)
                .signal_semaphores(&signal_semaphores)
                .push_next(&mut timeline_info);
            unsafe { Ctx::device().queue_bind_sparse(queue, &[info], vk::Fence::null()) }
                .map_err(crate::state::vk_error)
        });
        match submitted {
            Ok(wait) => {
                batch.commit();
                Ok(wait)
            }
            Err(err) => {
                batch.discard();
                Err(err)
            }
        }
    }
}

//! External memory (`external-memory`, Unix): buffers and images whose memory is exported to a file descriptor (opaque, or a dma-buf for buffers where the device has `VK_EXT_external_memory_dma_buf`) or imported from one, in dedicated allocations lava frees after the resource
use std::{
    ops::Deref,
    os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd},
};

use lava_macros::validation_trace;
use smallvec::SmallVec;

use crate::{
    buffer::{self, Buffer, Gpu},
    error::{Error, Result},
    features::{Feature, gate},
    global::Live,
    image::{
        D2, Dim, Image, RawImage, Single,
        format::{Format, StaticFormat},
        usage::{ImageUsage, StaticUsage},
    },
    lifetime,
    state::{Ctx, Functions},
    vk,
};

/// How exported memory is handed out: an opaque file descriptor (another Vulkan device or API
/// of the same driver imports it with the same description), or a Linux dma-buf (buffers
/// only; needs `VK_EXT_external_memory_dma_buf`, see [`HandleType::supported`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HandleType {
    OpaqueFd,
    DmaBuf,
}

impl HandleType {
    fn flags(self) -> vk::ExternalMemoryHandleTypeFlags {
        match self {
            HandleType::OpaqueFd => vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD,
            HandleType::DmaBuf => vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
        }
    }

    /// Whether lava enabled what this handle type needs on the device (`external-memory`, and
    /// `VK_EXT_external_memory_dma_buf` for dma-bufs).
    pub fn supported(self) -> bool {
        self.check().is_ok()
    }

    fn check(self) -> Result<()> {
        gate::ExternalMemory::check()?;
        match self {
            HandleType::DmaBuf if !Ctx::get().dma_buf => Err(Error::Unsupported(
                "the device has no VK_EXT_external_memory_dma_buf".into(),
            )),
            _ => Ok(()),
        }
    }
}

/// A dedicated `VkDeviceMemory` an external resource owns, freed once no submission reaches
/// it (after the resource it is bound to, which is dropped first).
#[derive(Debug)]
struct Memory {
    memory: vk::DeviceMemory,
    handle_type: HandleType,
}

impl Drop for Memory {
    fn drop(&mut self) {
        let memory = self.memory;
        lifetime::retire(move || unsafe { Ctx::device().free_memory(memory, None) });
    }
}

impl Memory {
    /// Exports the memory as a new file descriptor of its handle type.
    fn export(&self) -> Result<OwnedFd> {
        let info = vk::MemoryGetFdInfoKHR::default()
            .memory(self.memory)
            .handle_type(self.handle_type.flags());
        let fd = unsafe { Functions::external_memory_fd().get_memory_fd(&info)? };
        // SAFETY: Vulkan made the descriptor for the caller, who owns it now.
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }
}

/// Where a dedicated allocation's memory comes from.
enum Source {
    Export,
    Import(OwnedFd),
}

impl Source {
    fn feature(&self) -> vk::ExternalMemoryFeatureFlags {
        match self {
            Source::Export => vk::ExternalMemoryFeatureFlags::EXPORTABLE,
            Source::Import(_) => vk::ExternalMemoryFeatureFlags::IMPORTABLE,
        }
    }
}

/// Allocates (or imports) the dedicated memory of `requirements` for `dedicated`; an imported
/// descriptor belongs to Vulkan once this succeeds and is closed otherwise.
fn allocate(
    requirements: vk::MemoryRequirements,
    handle_type: HandleType,
    source: Source,
    mut dedicated: vk::MemoryDedicatedAllocateInfo<'_>,
    device_address: bool,
) -> Result<Memory> {
    let mut type_bits = requirements.memory_type_bits;
    if let Source::Import(fd) = &source
        && handle_type == HandleType::DmaBuf
    {
        let mut properties = vk::MemoryFdPropertiesKHR::default();
        unsafe {
            Functions::external_memory_fd().get_memory_fd_properties(
                handle_type.flags(),
                fd.as_raw_fd(),
                &mut properties,
            )?
        };
        type_bits &= properties.memory_type_bits;
    }
    // The first device-local type, as every exporter and importer of one description picks it.
    let memory = &Ctx::physical_device().mem_properties;
    let types = &memory.memory_types[..memory.memory_type_count as usize];
    let pick = |flags: vk::MemoryPropertyFlags| {
        (0..types.len() as u32).find(|&i| {
            type_bits & (1 << i) != 0 && types[i as usize].property_flags.contains(flags)
        })
    };
    let memory_type = pick(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        .or_else(|| pick(vk::MemoryPropertyFlags::empty()))
        .ok_or_else(|| Error::Unsupported("no memory type takes the external memory".into()))?;
    let mut flags =
        vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    let mut export = vk::ExportMemoryAllocateInfo::default().handle_types(handle_type.flags());
    let raw_fd = match &source {
        Source::Import(fd) => fd.as_raw_fd(),
        Source::Export => -1,
    };
    let mut import = vk::ImportMemoryFdInfoKHR::default()
        .handle_type(handle_type.flags())
        .fd(raw_fd);
    let mut info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type)
        .push_next(&mut dedicated);
    if device_address {
        info = info.push_next(&mut flags);
    }
    info = match source {
        Source::Export => info.push_next(&mut export),
        Source::Import(_) => info.push_next(&mut import),
    };
    let memory = unsafe { Ctx::device().allocate_memory(&info, None)? };
    if let Source::Import(fd) = source {
        // Vulkan owns an imported descriptor once the import succeeded.
        let _ = fd.into_raw_fd();
    }
    Ok(Memory {
        memory,
        handle_type,
    })
}

/// A device-local buffer of `T`s whose memory is exported to or imported from a file
/// descriptor. It derefs to its [`Buffer`], whose slices commands take as any buffer's.
#[derive(Debug)]
pub struct ExternalBuffer<T: Copy + 'static> {
    buffer: Buffer<T, Gpu>,
    /// After the buffer, so it is freed once the buffer is destroyed.
    memory: Memory,
}

impl<T: Copy + 'static> Deref for ExternalBuffer<T> {
    type Target = Buffer<T, Gpu>;
    fn deref(&self) -> &Buffer<T, Gpu> {
        &self.buffer
    }
}

impl<T: Copy + 'static> ExternalBuffer<T> {
    /// A buffer of `len` elements whose memory [`Self::export`] hands out as `handle_type`.
    /// `Unsupported` where the device doesn't export buffers as it.
    #[validation_trace]
    pub fn new(len: usize, handle_type: HandleType) -> Result<Self> {
        Self::create(len, handle_type, Source::Export)
    }

    /// A buffer of `len` elements in the memory `fd` holds (exported as `handle_type` for a
    /// buffer of that size, by this or another device of the same driver), which Vulkan owns
    /// once this succeeds.
    #[validation_trace]
    pub fn import(fd: OwnedFd, len: usize, handle_type: HandleType) -> Result<Self> {
        Self::create(len, handle_type, Source::Import(fd))
    }

    /// A new file descriptor of its memory, for another device or API to import.
    pub fn export(&self) -> Result<OwnedFd> {
        self.memory.export()
    }

    pub fn handle_type(&self) -> HandleType {
        self.memory.handle_type
    }

    fn create(len: usize, handle_type: HandleType, source: Source) -> Result<Self> {
        let live = Live::new();
        handle_type.check()?;
        let num_bytes = (len as u64)
            .checked_mul(size_of::<T>() as u64)
            .filter(|&bytes| bytes > 0)
            .ok_or_else(|| {
                Error::other(format!(
                    "an external buffer of {len} elements of {} bytes",
                    size_of::<T>()
                ))
            })?;
        let usage = buffer::USAGE
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let mut properties = vk::ExternalBufferProperties::default();
        unsafe {
            Functions::instance().get_physical_device_external_buffer_properties(
                Ctx::physical_device().handle,
                &vk::PhysicalDeviceExternalBufferInfo::default()
                    .usage(usage)
                    .handle_type(handle_type.flags()),
                &mut properties,
            )
        };
        let features = properties
            .external_memory_properties
            .external_memory_features;
        if !features.contains(source.feature()) {
            return Err(Error::Unsupported(format!(
                "the device doesn't {:?} buffers as {handle_type:?}",
                source.feature()
            )));
        }
        let families: SmallVec<[u32; 3]> = Ctx::get().queues.iter().map(|f| f.index).collect();
        let mut external =
            vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle_type.flags());
        let mut info = vk::BufferCreateInfo::default()
            .size(num_bytes)
            .usage(usage)
            .push_next(&mut external);
        if families.len() > 1 {
            info = info
                .sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&families);
        }
        let device = Ctx::device();
        let handle = unsafe { device.create_buffer(&info, None)? };
        let requirements = unsafe { device.get_buffer_memory_requirements(handle) };
        let dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(handle);
        let memory = match allocate(requirements, handle_type, source, dedicated, true) {
            Ok(memory) => memory,
            Err(err) => {
                unsafe { device.destroy_buffer(handle, None) };
                return Err(err);
            }
        };
        if let Err(err) = unsafe { device.bind_buffer_memory(handle, memory.memory, 0) } {
            unsafe { device.destroy_buffer(handle, None) };
            return Err(err.into());
        }
        let address = unsafe {
            device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(handle))
        };
        Ok(ExternalBuffer {
            buffer: Buffer::from_external(handle, address, num_bytes, live),
            memory,
        })
    }
}

/// A device-local image whose memory is exported to or imported from an opaque file
/// descriptor; the importer creates it with the exporter's format, usage, extent and mips. It
/// derefs to its [`Image`].
#[derive(Debug)]
pub struct ExternalImage<F: Format, U: ImageUsage, D: Dim = D2> {
    image: Image<F, U, D, Single>,
    /// After the image, so it is freed once the image is destroyed.
    memory: Memory,
}

impl<F: Format, U: ImageUsage, D: Dim> Deref for ExternalImage<F, U, D> {
    type Target = Image<F, U, D, Single>;
    fn deref(&self) -> &Self::Target {
        &self.image
    }
}

impl<F: StaticFormat, U: StaticUsage, D: Dim> ExternalImage<F, U, D> {
    /// An image whose memory [`Self::export`] hands out as an opaque file descriptor.
    /// `Unsupported` where the device doesn't export images of its description.
    #[validation_trace]
    pub fn new(extent: D::Extent, mip_levels: u32) -> Result<Self> {
        Self::create(
            F::default(),
            U::default(),
            extent,
            mip_levels,
            Source::Export,
        )
    }

    /// An image in the memory `fd` holds, exported by an image of the same description.
    #[validation_trace]
    pub fn import(fd: OwnedFd, extent: D::Extent, mip_levels: u32) -> Result<Self> {
        Self::create(
            F::default(),
            U::default(),
            extent,
            mip_levels,
            Source::Import(fd),
        )
    }
}

impl<F: Format, U: ImageUsage, D: Dim> ExternalImage<F, U, D> {
    /// A new opaque file descriptor of its memory.
    pub fn export(&self) -> Result<OwnedFd> {
        self.memory.export()
    }

    fn create(
        format: F,
        usage: U,
        extent: D::Extent,
        mip_levels: u32,
        source: Source,
    ) -> Result<Self> {
        let handle_type = HandleType::OpaqueFd;
        handle_type.check()?;
        let raw = RawImage::<F, U, D>::create(
            format,
            usage,
            extent,
            mip_levels,
            1,
            usage.create_flags(),
            handle_type.flags(),
        )?;
        let device = Ctx::device();
        let image = raw.handle();
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
        // `raw` destroys the image when this fails.
        let memory = allocate(requirements, handle_type, source, dedicated, false)?;
        unsafe { device.bind_image_memory(image, memory.memory, 0)? };
        Ok(ExternalImage {
            image: Image::from_raw(raw)?,
            memory,
        })
    }
}

//! Buffer memory locations as sealed marker types (`Gpu`, `Upload`, `Readback`, `Shared`, and `Dynamic`, chosen at runtime), which of them the host may access, and the memory types each allocates from
use std::fmt::Debug;

use gpu_allocator::MemoryLocation;

use crate::vk;

/// Where a buffer's memory lives, as a value: what [`Memory::location`] returns and
/// [`Dynamic`] holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Location {
    /// Device-local memory the host can't access, never in the BAR heap.
    Gpu,
    /// Host-visible memory the GPU reads well, device-local where the device has some
    /// (write-combined: host reads are slow).
    Upload,
    /// Host-visible, cached memory the host reads well.
    Readback,
    /// Device-local, host-visible memory (resizable BAR or an integrated GPU); its allocation
    /// fails instead of falling back to system memory. Only with `rebar`.
    Shared,
}

impl Location {
    /// Whether the host can access a buffer of this location.
    pub const fn host_visible(self) -> bool {
        !matches!(self, Location::Gpu)
    }

    /// `Shared` where the device has `rebar`, else `Upload`: for data the host edits in place
    /// that should be device-local where it can be.
    pub fn shared_or_upload() -> Location {
        match crate::state::Ctx::features().rebar {
            true => Location::Shared,
            false => Location::Upload,
        }
    }

    /// gpu-allocator's location, which prefers the memory types each location wants.
    pub(crate) fn allocator_location(self) -> MemoryLocation {
        match self {
            Location::Gpu => MemoryLocation::GpuOnly,
            Location::Upload | Location::Shared => MemoryLocation::CpuToGpu,
            Location::Readback => MemoryLocation::GpuToCpu,
        }
    }

    /// The memory types of `type_bits` this location may allocate from: for `Shared` only
    /// those both device-local and host-visible, so it fails rather than spill to system
    /// memory.
    pub(crate) fn memory_type_bits(
        self,
        type_bits: u32,
        properties: &vk::PhysicalDeviceMemoryProperties,
    ) -> u32 {
        if self != Location::Shared {
            return type_bits;
        }
        let shared = vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE;
        let types = &properties.memory_types[..properties.memory_type_count as usize];
        (types.iter().enumerate())
            .filter(|(_, t)| t.property_flags.contains(shared))
            .fold(0, |bits, (i, _)| bits | (1 << i))
            & type_bits
    }
}

/// A buffer's memory location: [`Gpu`] (the default), [`Upload`], [`Readback`], [`Shared`],
/// or [`Dynamic`], a location known only at runtime.
pub trait Memory: crate::sealed::Sealed + Copy + Debug + Send + Sync + 'static {
    fn location(&self) -> Location;
    /// Panics unless the host can access the memory; a static location's does nothing.
    #[track_caller]
    fn host_check(&self);
}

/// Every location but [`Dynamic`].
pub trait StaticMemory: Memory + Default {
    const LOCATION: Location;
}

/// Locations the host can read and write: every one but [`Gpu`] (and a [`Dynamic`] one of
/// it, which panics).
pub trait HostVisible: Memory {}

macro_rules! locations {
    ($($(#[doc = $doc:literal])* $name:ident $(: $host:ident)?,)*) => {$(
        $(#[doc = $doc])*
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
        pub struct $name;
        impl crate::sealed::Sealed for $name {}
        impl Memory for $name {
            fn location(&self) -> Location {
                Location::$name
            }
            fn host_check(&self) {}
        }
        impl StaticMemory for $name {
            const LOCATION: Location = Location::$name;
        }
        $(impl $host for $name {})?
    )*};
}

locations! {
    /// Device-local memory the host can't access (meshes, cull buffers, scratch): its data
    /// comes from `Buffer::from_slice`, `write` or commands.
    Gpu,
    /// Host-visible memory the GPU reads well, for data the host writes every frame (rings,
    /// UI, staging); host reads are slow.
    Upload: HostVisible,
    /// Cached host-visible memory, for data the host reads back (screenshots, counters).
    Readback: HostVisible,
    /// Device-local memory the host reads and writes in place; only on devices with `rebar`.
    Shared: HostVisible,
}

/// A location known only at runtime; host access to a `Gpu` one panics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Dynamic(pub Location);

impl crate::sealed::Sealed for Dynamic {}
impl Memory for Dynamic {
    fn location(&self) -> Location {
        self.0
    }
    fn host_check(&self) {
        assert!(
            self.0.host_visible(),
            "the host can't access a buffer of {:?} memory",
            self.0
        );
    }
}
impl HostVisible for Dynamic {}

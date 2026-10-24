//! Crate-wide error type and Result alias: typed failures of the device and environment that callers match on (device loss, memory, surfaces, missing features and GPUs, exhausted slots), and the rule that programmer errors panic instead
//!
//! # Panic or `Err`
//! Lava panics for programmer errors: arguments a correct program never passes (a mip the image
//! doesn't have, data of the wrong length, a view bound twice, an access by a family that doesn't
//! own the image), and states a correct program never reaches. It returns `Err` for what a
//! correct program can still meet: the device, driver, surface or environment failing or lacking
//! something (device loss, memory, a surface out of date, a feature or GPU that isn't there, a
//! full bindless heap or staging ring). Recording is infallible: a pipeline the driver fails to
//! create while a command is recorded panics, and so does a pass whose bindings were made by
//! hand for a device that can't run it.
use std::{borrow::Cow, fmt};

use crate::{features::MissingFeature, vk};

/// A failure of the device or environment. `Display` shows only this error's own description;
/// what caused it, if anything ([`Error::Other`]), is its `source()`. The wrappers of other
/// crates' errors (`Loading`, `Allocation`, `Io`, `MissingFeature`) display the wrapped error.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// The device was lost (`VK_ERROR_DEVICE_LOST`): a hang, a driver reset or a removed GPU.
    /// Nothing reaches the GPU any more: drop every lava object and call `lava::shutdown`
    /// before initialising lava again.
    DeviceLost,
    /// Device or host memory ran out, in Vulkan or in lava's allocator.
    OutOfMemory,
    /// The surface changed so that its swapchain can't present (`VK_ERROR_OUT_OF_DATE_KHR`),
    /// and recreating it didn't help.
    SurfaceOutOfDate,
    /// The surface is gone (`VK_ERROR_SURFACE_LOST_KHR`): make a new one for the window.
    SurfaceLost,
    /// No GPU has what lava needs: per GPU, its name and what it lacks (empty without GPUs).
    NoSuitableGpu { rejected: Vec<String> },
    /// What was asked is something this device, surface, driver or build of lava can't do.
    Unsupported(String),
    /// The device lacks an optional feature lava checks at runtime.
    MissingFeature(MissingFeature),
    /// A bindless heap has no `count` contiguous free slots.
    OutOfBindlessSlots {
        heap: crate::bindless::Heap,
        count: u32,
    },
    /// `queue::MAX_TIMELINES` timelines exist already.
    OutOfTimelines,
    /// A buffer sub-range that starts after it ends (`start > end`) or ends past the `size`
    /// bytes of what it is taken from; all in bytes.
    SliceOutOfRange { start: u64, end: u64, size: u64 },
    /// A staging ring of `capacity` bytes had no room for `requested` bytes within the
    /// uploader's timeout, or the write is larger than the ring.
    StagingFull { requested: u64, capacity: u64 },
    /// `requested` is more than the device's `limit` allows (`max`), e.g. an acceleration
    /// structure of more geometries than `maxGeometryCount`.
    OverLimit {
        limit: &'static str,
        requested: u64,
        max: u64,
    },
    /// An acceleration structure arena of `capacity` bytes had no free range of `requested`
    /// bytes.
    ArenaFull { requested: u64, capacity: u64 },
    /// Serialized data this device can't read (`vkGetDeviceAccelerationStructureCompatibilityKHR`):
    /// it was written by another device or driver.
    Incompatible,
    /// `lava::shutdown` found `count` lava objects (or calls in progress on other threads)
    /// alive, so it destroyed nothing and lava stays initialised.
    ObjectsAlive { count: u64 },
    /// Any other Vulkan error.
    Vulkan(vk::Result),
    /// The Vulkan library couldn't be loaded.
    Loading(ash::LoadingError),
    /// lava's allocator failed other than by running out of memory.
    Allocation(gpu_allocator::AllocationError),
    /// A file lava reads or writes (the pipeline cache).
    Io(std::io::Error),
    /// Anything else, described by `message`, with the error that caused it if any.
    Other {
        message: Cow<'static, str>,
        source: Option<Box<Error>>,
    },
}

impl Error {
    /// An error described only by `message`.
    pub(crate) fn other(message: impl Into<Cow<'static, str>>) -> Self {
        Error::Other {
            message: message.into(),
            source: None,
        }
    }

    /// `self` with a description of what failed, unless it is a variant callers match on by
    /// itself (device loss, memory, surfaces, features, slots), which stays as it is.
    pub(crate) fn context(self, message: impl Into<Cow<'static, str>>) -> Self {
        match self {
            Error::Vulkan(_)
            | Error::Loading(_)
            | Error::Allocation(_)
            | Error::Io(_)
            | Error::Other { .. } => Error::Other {
                message: message.into(),
                source: Some(Box::new(self)),
            },
            typed => typed,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::DeviceLost => f.write_str("the device was lost"),
            Error::OutOfMemory => f.write_str("out of GPU or host memory"),
            Error::SurfaceOutOfDate => {
                f.write_str("the surface's swapchain was still out of date after three recreations")
            }
            Error::SurfaceLost => f.write_str("the surface was lost"),
            Error::NoSuitableGpu { rejected } if rejected.is_empty() => f.write_str("no GPU found"),
            Error::NoSuitableGpu { rejected } => {
                write!(f, "no suitable GPU: {}", rejected.join("; "))
            }
            Error::Unsupported(what) => f.write_str(what),
            Error::MissingFeature(missing) => missing.fmt(f),
            Error::OutOfBindlessSlots { heap, count } => {
                write!(
                    f,
                    "the bindless {heap} heap has no {count} contiguous free slots"
                )
            }
            Error::OutOfTimelines => write!(
                f,
                "{} timelines exist already, the most lava has",
                crate::queue::MAX_TIMELINES
            ),
            Error::SliceOutOfRange { start, end, .. } if start > end => {
                write!(
                    f,
                    "buffer slice starts at byte {start} but ends at byte {end}"
                )
            }
            Error::SliceOutOfRange { end, size, .. } => write!(
                f,
                "buffer slice ends at byte {end} but the buffer is only {size} bytes long"
            ),
            Error::StagingFull {
                requested,
                capacity,
            } if requested > capacity => write!(
                f,
                "an upload of {requested} bytes is larger than the staging ring of {capacity} bytes"
            ),
            Error::StagingFull {
                requested,
                capacity,
            } => write!(
                f,
                "the staging ring of {capacity} bytes had no room for {requested} bytes within the timeout: drain it or make it larger"
            ),
            Error::OverLimit {
                limit,
                requested,
                max,
            } => write!(f, "{requested} is more than the device's `{limit}` of {max}"),
            Error::ArenaFull {
                requested,
                capacity,
            } => write!(
                f,
                "the acceleration structure arena of {capacity} bytes has no free range of {requested} bytes"
            ),
            Error::Incompatible => {
                f.write_str("the serialized data was written by another device or driver")
            }
            Error::ObjectsAlive { count } => write!(
                f,
                "{count} lava objects or calls are still alive, so lava wasn't shut down: drop every buffer, image, timeline, uploader, surface, sampler, slot range, query pool, acceleration structure and structure arena first"
            ),
            Error::Vulkan(result) => write!(f, "Vulkan error {result:?} ({result})"),
            Error::Loading(err) => write!(f, "the Vulkan library couldn't be loaded: {err}"),
            Error::Allocation(err) => err.fmt(f),
            Error::Io(err) => err.fmt(f),
            Error::Other { message, .. } => f.write_str(message),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Other { source, .. } => source.as_deref().map(|e| e as _),
            _ => None,
        }
    }
}

impl From<vk::Result> for Error {
    fn from(result: vk::Result) -> Self {
        match result {
            vk::Result::ERROR_DEVICE_LOST => Error::DeviceLost,
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | vk::Result::ERROR_OUT_OF_HOST_MEMORY => {
                Error::OutOfMemory
            }
            vk::Result::ERROR_OUT_OF_DATE_KHR => Error::SurfaceOutOfDate,
            vk::Result::ERROR_SURFACE_LOST_KHR => Error::SurfaceLost,
            result => Error::Vulkan(result),
        }
    }
}

impl From<ash::LoadingError> for Error {
    fn from(err: ash::LoadingError) -> Self {
        Error::Loading(err)
    }
}

impl From<gpu_allocator::AllocationError> for Error {
    fn from(err: gpu_allocator::AllocationError) -> Self {
        match err {
            gpu_allocator::AllocationError::OutOfMemory => Error::OutOfMemory,
            err => Error::Allocation(err),
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Io(err)
    }
}

impl From<MissingFeature> for Error {
    fn from(missing: MissingFeature) -> Self {
        Error::MissingFeature(missing)
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

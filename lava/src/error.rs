//! Crate-wide error type and Result alias wrapping Vulkan, allocator and IO failures, and optional features the device lacks
use std::fmt;

use crate::{features::MissingFeature, vk};

#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    Vulkan(vk::Result),
    Loading(ash::LoadingError),
    Allocation(gpu_allocator::AllocationError),
    Utf8(std::str::Utf8Error),
    Io(std::io::Error),
    Message(String),
    Lock,
    /// The device lacks an optional feature lava checks at runtime.
    MissingFeature(MissingFeature),
    /// A bindless heap has no `count` contiguous free slots (`heap`: "image" or "sampler").
    OutOfBindlessSlots { heap: &'static str, count: u32 },
    /// `queue::MAX_TIMELINES` timelines exist already.
    OutOfTimelines,
}

impl Error {
    pub fn message(msg: impl Into<String>) -> Self {
        Error::Message(msg.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Vulkan(result) => write!(f, "Vulkan error: {result}"),
            Error::Loading(err) => write!(f, "failed to load Vulkan library: {err}"),
            Error::Allocation(err) => write!(f, "GPU allocation error: {err}"),
            Error::Utf8(err) => write!(f, "invalid UTF-8: {err}"),
            Error::Io(err) => write!(f, "I/O error: {err}"),
            Error::Message(msg) => f.write_str(msg),
            Error::Lock => write!(f, "lock error"),
            Error::MissingFeature(missing) => missing.fmt(f),
            Error::OutOfBindlessSlots { heap, count } => {
                write!(f, "the bindless {heap} heap has no {count} contiguous free slots")
            }
            Error::OutOfTimelines => write!(
                f,
                "{} timelines exist already, the most lava has",
                crate::queue::MAX_TIMELINES
            ),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Vulkan(_) => None,
            Error::Loading(err) => Some(err),
            Error::Allocation(err) => Some(err),
            Error::Utf8(err) => Some(err),
            Error::Io(err) => Some(err),
            Error::Message(_) => None,
            Error::Lock => None,
            Error::MissingFeature(missing) => Some(missing),
            Error::OutOfBindlessSlots { .. } | Error::OutOfTimelines => None,
        }
    }
}

impl From<vk::Result> for Error {
    fn from(err: vk::Result) -> Self {
        Error::Vulkan(err)
    }
}

impl From<ash::LoadingError> for Error {
    fn from(err: ash::LoadingError) -> Self {
        Error::Loading(err)
    }
}

impl From<gpu_allocator::AllocationError> for Error {
    fn from(err: gpu_allocator::AllocationError) -> Self {
        Error::Allocation(err)
    }
}

impl From<std::str::Utf8Error> for Error {
    fn from(err: std::str::Utf8Error) -> Self {
        Error::Utf8(err)
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

impl From<String> for Error {
    fn from(err: String) -> Self {
        Error::Message(err)
    }
}

impl From<&str> for Error {
    fn from(err: &str) -> Self {
        Error::Message(err.to_owned())
    }
}

impl<T> From<std::sync::PoisonError<T>> for Error {
    fn from(_err: std::sync::PoisonError<T>) -> Self {
        Error::Lock
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

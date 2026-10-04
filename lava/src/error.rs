//! Crate-wide error type and Result alias wrapping Vulkan, allocator, and IO failures
use std::fmt;

use ash::vk;

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as _;

    #[test]
    fn display_describes_each_variant() {
        assert_eq!(
            Error::Vulkan(vk::Result::ERROR_DEVICE_LOST).to_string(),
            format!("Vulkan error: {}", vk::Result::ERROR_DEVICE_LOST)
        );
        assert_eq!(Error::message("boom").to_string(), "boom");
        assert_eq!(Error::Lock.to_string(), "lock error");

        let io = Error::from(std::io::Error::other("disk on fire"));
        assert_eq!(io.to_string(), "I/O error: disk on fire");

        let utf8 = Error::from(std::str::from_utf8(&[0xff]).unwrap_err());
        assert!(utf8.to_string().starts_with("invalid UTF-8: "));

        let alloc = Error::from(gpu_allocator::AllocationError::OutOfMemory);
        assert!(alloc.to_string().starts_with("GPU allocation error: "));
    }

    #[test]
    fn source_is_the_wrapped_error() {
        assert!(Error::Vulkan(vk::Result::ERROR_UNKNOWN).source().is_none());
        assert!(Error::message("m").source().is_none());
        assert!(Error::Lock.source().is_none());

        assert!(Error::from(std::io::Error::other("x")).source().is_some());
        assert!(
            Error::from(std::str::from_utf8(&[0xff]).unwrap_err())
                .source()
                .is_some()
        );
        assert!(
            Error::from(gpu_allocator::AllocationError::OutOfMemory)
                .source()
                .is_some()
        );
    }

    #[test]
    fn conversions_pick_the_matching_variant() {
        assert!(matches!(
            Error::from(vk::Result::ERROR_OUT_OF_DATE_KHR),
            Error::Vulkan(vk::Result::ERROR_OUT_OF_DATE_KHR)
        ));
        assert!(matches!(Error::from("text"), Error::Message(m) if m == "text"));
        assert!(matches!(Error::from(String::from("owned")), Error::Message(m) if m == "owned"));
        assert!(matches!(
            Error::from(std::sync::PoisonError::new(())),
            Error::Lock
        ));
        assert!(matches!(
            Error::from(std::io::Error::other("x")),
            Error::Io(_)
        ));
    }

    #[test]
    fn loading_errors_convert_and_keep_their_source() {
        let err = unsafe { ash::Entry::load_from("/nonexistent/libvulkan.so") }
            .err()
            .expect("loading a missing library fails");
        let err = Error::from(err);
        assert!(matches!(err, Error::Loading(_)));
        assert!(
            err.to_string()
                .starts_with("failed to load Vulkan library: ")
        );
        assert!(err.source().is_some());
    }

    #[test]
    fn question_mark_converts_vulkan_results() {
        fn fails() -> Result<()> {
            Err(vk::Result::TIMEOUT)?
        }
        assert!(matches!(fails(), Err(Error::Vulkan(vk::Result::TIMEOUT))));
    }
}

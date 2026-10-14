//! Lava: typed, bindless Vulkan abstraction over ash with shader-generated passes
// `RasterBuilder` grows its arrays of draws and color attachments by one per call.
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(const_trait_impl)]
#![feature(range_into_bounds)]
#![feature(range_bounds_is_empty)]
use std::panic::Location;

use crate::state::CALLSITE;
use crate::{bindless::Bindless, state::Ctx};

pub use crate::error::{Error, Result};
pub use ash::vk::Pipeline as VkPipeline;
pub use ash::vk::ShaderModule as VkShaderModule;
pub use ash::vk::{AccessFlags2, ImageLayout, PipelineStageFlags2};
/// What `Error::Vulkan` holds, e.g. `ERROR_OUT_OF_DATE_KHR` from `Swapchain::aquire_image`.
pub use ash::vk::Result as VkResult;
/// What `format::Dynamic`, `image::usage::Dynamic` and `buffer::usage::Dynamic` hold.
pub use ash::vk::{BufferUsageFlags, Format as VkFormat, ImageUsageFlags};

pub mod bindless;
pub mod buffer;
pub mod command_buffer;
pub mod epoch;
pub mod error;
#[cfg(feature = "hot-reload")]
mod hot_reload;
pub mod image;
pub mod pipeline;
pub mod profiling;
pub mod scratchpad;
pub mod state;
pub mod vkobjects;
use raw_window_handle::RawDisplayHandle;

/// The supertrait of the traits only lava implements (formats, usages, dimensions, views,
/// semaphore and queue kinds, acceleration structure levels, index formats): public in a
/// private module, so other crates can name and bound by those traits but not implement them.
mod sealed {
    pub trait Sealed {}
}

/// The bindings lava-build generated in the application's build script: every pass's
/// `Pass::push_bindings`, the types its shaders share, and the `PASS_MAP` that goes to
/// [`init`]. Expands where it is called, so `OUT_DIR` is the calling crate's.
#[macro_export]
macro_rules! include_passes {
    () => {
        include!(concat!(env!("OUT_DIR"), "/lava/bindings.rs"));
    };
}

/// Initialises the global Vulkan context with the application's passes (the `PASS_MAP` of
/// [`include_passes!`]). Pass the display handle to be able to create
/// [`vkobjects::surface::Surface`]s and swapchains; `None` runs headless.
pub fn init(
    display: Option<&RawDisplayHandle>,
    enable_validation: bool,
    enable_gpu_assited_validation: bool,
    passes: &'static pipeline::PassMap,
) -> Result<()> {
    assert_eq!(
        passes.profiling,
        cfg!(feature = "profiling"),
        "the passes were compiled for lava {} its `profiling` feature: run lava-build from the \
         build script of a crate that depends on lava directly",
        if passes.profiling { "with" } else { "without" }
    );
    Ctx::init(display, enable_validation, enable_gpu_assited_validation)?;
    pipeline::set_pass_map(passes);
    Bindless::init()?;
    profiling::init()?;
    pipeline::init();
    Ok(())
}

/// Waits until the device finished everything submitted so far, e.g. before exiting. A stall;
/// a dropped resource is destroyed once the frames that may use it are done anyway (`epoch`).
/// Destroys what every submitted frame and present kept alive, so call it outside of
/// `Frame::execute`.
pub fn wait_idle() -> Result<()> {
    unsafe { Ctx::device().device_wait_idle()? };
    epoch::device_idle();
    Ok(())
}

#[track_caller]
pub fn destroy() {
    CALLSITE.set(Some(Location::caller().clone()));
    Bindless::destroy();
}

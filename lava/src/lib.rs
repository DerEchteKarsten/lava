//! Lava: typed, bindless Vulkan abstraction over ash with shader-generated passes
//!
//! Lava takes SPIR-V and knows no shader language: a frontend crate (`lava-build-slang`)
//! compiles an application's passes in its build script and generates their bindings and
//! `PASS_MAP`, and may hot reload them through [`pipeline::reload_pass`].
//!
//! # Shader contract
//! What every frontend's shaders and generated code must match:
//! - Descriptor set 0: binding 0 holds the application's immutable samplers
//!   ([`Config::samplers`], by index), binding 1 the bindless sampled images. Set 1, binding 0:
//!   the bindless storage images. Passes have no other descriptors.
//! - A pass whose SPIR-V declares a capability behind an optional feature ([`Features`]) runs
//!   only where the device has it.
//! - One push-constant range over all stages, [`pipeline::MAX_PUSH_CONSTANTS_SIZE`] bytes,
//!   zeroed when a command buffer begins; a pass pushes its own struct from offset 0.
//!   Specialization constants have no size limit.
//! - [`bindless::BindlessHandle`] (two `u32` indices, sampled then storage,
//!   [`bindless::NULL_HANDLE`] for none) and [`vkobjects::acceleration_structure::Instance`]
//!   (`VkAccelerationStructureInstanceKHR`) are the layouts shaders see; a TLAS is a `u64`
//!   device address.
use std::panic::Location;

use crate::state::CALLSITE;
use crate::{bindless::Bindless, state::Ctx};

pub use crate::error::{Error, Result};
pub use crate::features::Features;
pub use crate::vk::Pipeline as VkPipeline;
/// What `Error::Vulkan` holds, e.g. `ERROR_OUT_OF_DATE_KHR` from `Swapchain::aquire_image`.
pub use crate::vk::Result as VkResult;
pub use crate::vk::ShaderModule as VkShaderModule;
pub use crate::vk::{AccessFlags2, ImageLayout, PipelineStageFlags2};
/// What `format::Dynamic`, `image::usage::Dynamic` and `buffer::usage::Dynamic` hold.
pub use crate::vk::{BufferUsageFlags, Format as VkFormat, ImageUsageFlags};

pub mod bindless;
pub mod buffer;
pub mod command_buffer;
pub mod epoch;
pub mod error;
pub mod features;
pub mod image;
pub mod memory;
pub mod pipeline;
pub mod query;
pub mod state;
pub mod vk;
pub mod vkobjects;
use raw_window_handle::RawDisplayHandle;

/// The supertrait of the traits only lava implements (formats, usages, dimensions, views,
/// semaphore and queue kinds, acceleration structure levels, index formats): public in a
/// private module, so other crates can name and bound by those traits but not implement them.
mod sealed {
    pub trait Sealed {}
}

/// The bindings lava-build-slang generated in the application's build script: every pass's
/// `Pass::push_bindings`, the types its shaders share, and the `PASS_MAP` that goes to
/// [`init`]. Expands where it is called, so `OUT_DIR` is the calling crate's.
#[macro_export]
macro_rules! include_passes {
    () => {
        include!(concat!(env!("OUT_DIR"), "/lava/bindings.rs"));
    };
}

/// What [`init`] sets up: the application's passes and samplers, the window system, validation,
/// and the optional features the application can't run without.
#[derive(Clone, Copy)]
pub struct Config<'a> {
    /// The `PASS_MAP` of [`include_passes!`].
    pub passes: &'static pipeline::PassMap,
    /// The display to create [`vkobjects::surface::Surface`]s and swapchains for; `None` runs
    /// headless.
    pub display: Option<RawDisplayHandle>,
    /// The validation layer with synchronization validation and best practices, in debug
    /// builds.
    pub validation: bool,
    /// GPU-assisted validation on top (`validation` only).
    pub gpu_assisted_validation: bool,
    /// The immutable samplers shaders pick by index; at least one.
    pub samplers: &'a [bindless::SamplerDesc],
    /// Optional features without which `init` fails (lava enables every one the device has).
    pub required: Features,
}

impl Config<'static> {
    /// Headless, without validation, with [`bindless::DEFAULT_SAMPLERS`] and no required
    /// features.
    pub fn new(passes: &'static pipeline::PassMap) -> Self {
        Self {
            passes,
            display: None,
            validation: false,
            gpu_assisted_validation: false,
            samplers: &bindless::DEFAULT_SAMPLERS,
            required: Features::default(),
        }
    }
}

/// Initialises the global Vulkan context: picks a device with everything lava needs and
/// `config.required`, enables every optional feature it has, and makes the passes' pipelines
/// (a pass whose SPIR-V needs a missing feature gets none and panics when recorded).
pub fn init(config: &Config) -> Result<()> {
    Ctx::init(config)?;
    pipeline::set_pass_map(config.passes);
    Bindless::init(config.samplers)?;
    query::init()?;
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

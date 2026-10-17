//! Lava: typed, bindless Vulkan abstraction over ash with shader-generated passes
//!
//! Lava takes SPIR-V and knows no shader language: a frontend crate (`lava-build-slang`)
//! compiles an application's passes in its build script and generates their bindings and
//! `PASS_MAP`, and may hot reload them through [`pipeline::reload_pass`].
//!
//! # Submission and lifetimes
//! Work goes to a [`queue::Timeline`] (graphics, and with `async-compute`/`async-transfer` the
//! device's dedicated compute and transfer families): reserve a value before reading anything
//! its GPU work reaches, begin it once a frame slot is free, record and submit it, which
//! returns a [`queue::Wait`] that other timelines and the host can wait for. Barriers within a
//! timeline are automatic; images move between families with
//! [`command_buffer::CommandBuffer::release`]. Dropped objects are destroyed once every
//! timeline finished the values reserved before the drop ([`lifetime`]). Windows present
//! through a [`present::PresentSurface`], whose acquire and present follow from what the
//! submission does with the image.
//!
//! # Shader contract
//! What every frontend's shaders and generated code must match:
//! - Descriptor set 0: binding 0 holds the application's immutable samplers
//!   ([`Config::samplers`], by index), binding 1 the bindless sampled images. Set 1, binding 0:
//!   the bindless storage images. Passes have no other descriptors.
//! - One push-constant range over all stages, [`pipeline::MAX_PUSH_CONSTANTS_SIZE`] bytes,
//!   zeroed when a command buffer begins; a pass pushes its own struct from offset 0.
//!   Specialization constants have no size limit.
//! - [`bindless::BindlessHandle`] (two `u32` indices, sampled then storage,
//!   [`bindless::NULL_HANDLE`] for none) and [`vkobjects::acceleration_structure::Instance`]
//!   (`VkAccelerationStructureInstanceKHR`) are the layouts shaders see; a TLAS is a `u64`
//!   device address.
//! - A pass whose SPIR-V declares a capability behind an optional feature ([`Features`]) runs
//!   only where the device has it: its generated bindings are [`features::Gated`] by it.
//!
//! # Optional features
//! Every optional device feature is two cargo features (all of the first kind are default):
//! - `X` compiles it, and lava checks the device where it is needed: functions that need it
//!   return [`features::Gated`] values, a `Result` with [`features::MissingFeature`] (also
//!   [`Error::MissingFeature`] where a function fails otherwise too).
//! - `assert-X` makes [`init`] refuse devices without it, and the checks go: those functions
//!   return their value directly.
//! - Without either, its API isn't compiled.
//!
//! The `dynamic-*` features are dynamic state: where the device has it, draws that differ in it
//! share a pipeline; elsewhere it is part of the pipeline's key. Asserting one keeps it out of
//! the key.
use std::panic::Location;

use crate::state::CALLSITE;
use crate::{bindless::Bindless, state::Ctx};

pub use crate::error::{Error, Result};
pub use crate::features::Features;
pub use crate::lifetime::{Stamp, latest_stamp, reclaim};
pub use crate::vk::Pipeline as VkPipeline;
/// What `Error::Vulkan` holds, e.g. `ERROR_DEVICE_LOST`.
pub use crate::vk::Result as VkResult;
pub use crate::vk::ShaderModule as VkShaderModule;
pub use crate::vk::{AccessFlags2, ImageLayout, PipelineStageFlags2};
/// What `format::Dynamic`, `image::usage::Dynamic` and `buffer::usage::Dynamic` hold.
pub use crate::vk::{BufferUsageFlags, Format as VkFormat, ImageUsageFlags};

pub mod bindless;
pub mod buffer;
pub mod command_buffer;
pub mod error;
pub mod features;
pub mod image;
pub mod lifetime;
pub mod memory;
pub mod pipeline;
pub mod present;
pub mod query;
pub mod queue;
pub mod state;
mod unsized_struct;
pub mod vk;
pub mod vkobjects;
use raw_window_handle::RawDisplayHandle;

/// The supertrait of the traits only lava implements (formats, usages, dimensions, views,
/// queue and query kinds, acceleration structure levels, index formats): public in a
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
/// and the optional features lava leaves off.
#[derive(Clone, Copy)]
pub struct Config<'a> {
    /// The `PASS_MAP` of [`include_passes!`].
    pub passes: &'static pipeline::PassMap,
    /// The display to create [`present::PresentSurface`]s for; `None` runs headless.
    pub display: Option<RawDisplayHandle>,
    /// The validation layer with synchronization validation and best practices, in debug
    /// builds.
    pub validation: bool,
    /// GPU-assisted validation on top (`validation` only).
    pub gpu_assisted_validation: bool,
    /// The immutable samplers shaders pick by index; at least one.
    pub samplers: &'a [bindless::SamplerDesc],
    /// Optional features lava leaves off even where the device has them (with the ones that
    /// need them), e.g. to test the paths without them. Asserted features can't be disabled.
    pub disabled: Features,
}

impl Config<'static> {
    /// Headless, without validation, with [`bindless::DEFAULT_SAMPLERS`] and every optional
    /// feature the device has.
    pub fn new(passes: &'static pipeline::PassMap) -> Self {
        Self {
            passes,
            display: None,
            validation: false,
            gpu_assisted_validation: false,
            samplers: &bindless::DEFAULT_SAMPLERS,
            disabled: Features::DEFAULT,
        }
    }
}

/// Initialises the global Vulkan context: picks a device with everything lava needs and the
/// asserted features, enables every optional feature it has (but `config.disabled`), and makes
/// the passes' pipelines (a pass whose SPIR-V needs a missing feature gets none; its bindings
/// return [`features::MissingFeature`]). Fails for a pass that needs a cargo feature lava was
/// compiled without.
pub fn init(config: &Config) -> Result<()> {
    Ctx::init(config)?;
    lifetime::set_present_fences(<features::gate::PresentFences as features::On>::on());
    pipeline::set_pass_map(config.passes);
    Bindless::init(config.samplers)?;
    query::init()?;
    pipeline::init()?;
    Ok(())
}

/// Waits until the device finished everything submitted so far, e.g. before exiting. A stall;
/// a dropped resource is destroyed once the submissions that may use it are done anyway
/// (`lifetime`). Destroys what the submissions and presents before it kept alive. Every queue
/// is locked meanwhile (Vulkan wants them externally synchronised), so other threads'
/// submissions and presents wait; call it outside of `Submission::submit`.
pub fn wait_idle() -> Result<()> {
    {
        let _queues = queue::device::lock_all_queues();
        unsafe { Ctx::device().device_wait_idle()? };
        // The presentation engine may still hold what the last presents use.
        lifetime::wait_for_presents(1_000_000_000);
        queue::timeline::poll_all();
        // Before the queues are unlocked: what is presented after it isn't done.
        lifetime::device_idle();
    }
    // Destructors lock the timelines' pending accesses, which a recording holds while it waits
    // to submit.
    lifetime::reclaim();
    Ok(())
}

/// Waits for the device and destroys everything dropped so far, done or not (the timelines
/// must be dropped first), then the bindless sets.
#[track_caller]
pub fn destroy() {
    CALLSITE.set(Some(Location::caller().clone()));
    if let Err(err) = wait_idle() {
        tracing::error!(%err, "failed to wait for the device before destroying lava");
    }
    lifetime::flush();
    Bindless::destroy();
}

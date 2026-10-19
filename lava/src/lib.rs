//! Lava: typed, bindless Vulkan abstraction over ash with shader-generated passes
//!
//! Lava takes SPIR-V: a frontend crate (`lava-build-slang`) compiles an application's passes in
//! its build script, generates their bindings and `PASS_MAP`, and may hot reload them through
//! [`pipeline::reload_pass`].
//!
//! # Submission and lifetimes
//! Work goes to a [`queue::Timeline`] (graphics, and with `async-compute`/`async-transfer` the
//! dedicated compute and transfer families): `reserve()` counts its stamp up before reading
//! anything the next submission reaches, `begin()` waits for a free frame slot, and `submit`
//! records, submits and returns a [`queue::Wait`] for other timelines and the host. Barriers
//! within a timeline are automatic; images move between families with
//! [`command_buffer::CommandBuffer::release`]. Dropped objects are destroyed once every timeline
//! finished its stamp at the drop ([`lifetime`]). Windows present through a [`present::PresentSurface`] and its current
//! image, whose acquire and present follow from what the submission does with it.
//!
//! # Shader contract
//! What every frontend's shaders and generated code must match:
//! - Descriptors live in two heaps (`VK_EXT_descriptor_heap`), which shaders index directly
//!   (`ResourceDescriptorHeap`, `SamplerDescriptorHeap`; `SPV_EXT_descriptor_heap`): the
//!   resource heap holds image descriptors, the sampler heap the [`bindless::Sampler`]s, each
//!   `OpConstantSizeOfEXT` of its descriptor type apart. Passes have no other descriptors.
//! - Push data: a pass pushes its own struct from offset 0, at most
//!   [`pipeline::MAX_PUSH_CONSTANTS_SIZE`] bytes, read as `[[vk::push_constant]]`.
//!   Specialization constants have no size limit.
//! - Shaders see [`bindless::ImageHandle`] (a `u32`: bits 0 to 29 the slot of its first
//!   descriptor, bit 30 that descriptor's kind, 0 sampled and 1 storage, bit 31 set for two
//!   descriptors, sampled at the slot and storage at the next; all ones for none, and mip `i` of
//!   a mip chain `i` times its width further), [`bindless::SamplerHandle`] (a `u32` slot; there
//!   are no samplers but the ones the application creates) and
//!   [`vkobjects::acceleration_structure::Instance`] (`VkAccelerationStructureInstanceKHR`); a
//!   TLAS is a `u64` device address.
//! - A pass whose SPIR-V declares a capability behind an optional feature ([`Features`]) has
//!   bindings [`features::Gated`] by it.
//!
//! # Optional features
//! Every optional device feature is two cargo features (the first kind are default):
//! - `X` compiles it, and lava checks the device where it is needed: those functions return
//!   [`features::Gated`] values, a `Result` with [`features::MissingFeature`] (or
//!   [`Error::MissingFeature`] where they fail otherwise too).
//! - `assert-X` makes [`init`] refuse devices without it, and those functions return their value
//!   directly.
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
/// What `format::Dynamic` and `image::usage::Dynamic` hold.
pub use crate::vk::{Format as VkFormat, ImageUsageFlags};

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
#[cfg(test)]
mod tests;
pub mod vk;
pub mod vkobjects;
use raw_window_handle::RawDisplayHandle;

/// Supertrait of the traits only lava implements: other crates can bound by them but not
/// implement them.
mod sealed {
    pub trait Sealed {}
}

/// Includes the bindings lava-build-slang generated in the calling crate's build script, with the
/// `PASS_MAP` for [`init`].
#[macro_export]
macro_rules! include_passes {
    () => {
        include!(concat!(env!("OUT_DIR"), "/lava/bindings.rs"));
    };
}

#[derive(Clone, Copy)]
pub struct Config {
    /// The `PASS_MAP` of [`include_passes!`].
    pub passes: &'static pipeline::PassMap,
    /// The display to create [`present::PresentSurface`]s for; `None` runs headless.
    pub display: Option<RawDisplayHandle>,
    /// The validation layer with synchronization validation and best practices (debug builds).
    pub validation: bool,
    /// GPU-assisted validation on top of `validation`.
    pub gpu_assisted_validation: bool,
    /// Optional features lava leaves off even where the device has them (not asserted ones).
    pub disabled: Features,
}

impl Config {
    /// Headless, without validation and with no disabled features.
    pub fn new(passes: &'static pipeline::PassMap) -> Self {
        Self {
            passes,
            display: None,
            validation: false,
            gpu_assisted_validation: false,
            disabled: Features::DEFAULT,
        }
    }
}

/// Picks a device with everything lava needs and the asserted features, enables its optional
/// features but `config.disabled`, and makes the passes' pipelines. Fails for a pass that needs a
/// cargo feature lava was compiled without.
pub fn init(config: &Config) -> Result<()> {
    Ctx::init(config)?;
    let _ = pipeline::PASS_MAP.set(config.passes);
    Bindless::init()?;
    #[cfg(feature = "performance-query")]
    query::counters::init()?;
    pipeline::init()?;
    Ok(())
}

/// Waits until the device finished everything submitted so far. Every queue is locked meanwhile,
/// so call it outside of `Submission::submit`.
pub fn wait_idle() -> Result<()> {
    {
        let queues = Ctx::get().queues.iter().flat_map(|family| &family.queues);
        let _queues: Vec<_> = queues.map(queue::lock).collect();
        unsafe { Ctx::device().device_wait_idle()? };
        queue::timeline::poll_all();
    }
    lifetime::reclaim();
    Ok(())
}

/// Waits for the device, then destroys everything dropped so far and the descriptor heaps; drop
/// the timelines first.
#[track_caller]
pub fn destroy() {
    CALLSITE.set(Some(Location::caller().clone()));
    if let Err(err) = wait_idle() {
        tracing::error!(%err, "failed to wait for the device before destroying lava");
    }
    Bindless::destroy();
    loop {
        let destroy = queue::lock(&lifetime::RETIRED).pop_front();
        match destroy {
            Some((_, destroy)) => destroy(),
            None => break,
        }
    }
}

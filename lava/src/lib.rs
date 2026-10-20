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
//! image, whose acquire and present follow from what the submission does with it. A submission
//! waits for another timeline's [`queue::Wait`] at a stage of its own family's set
//! ([`queue::GraphicsStage`], [`queue::ComputeStage`], [`queue::TransferStage`]).
//!
//! # Lifecycle
//! [`init`] makes lava's context; [`shutdown`] waits for the device, destroys everything and
//! allows another `init` in the same process, once every lava object was dropped (else it
//! returns `Error::ObjectsAlive` and destroys nothing). Without a context every lava call
//! panics.
//!
//! # Errors
//! Programmer errors panic; failures of the device and environment are typed [`Error`]s (see
//! [`error`] for the rule). Recording never fails: what goes wrong while a command is recorded
//! (a pipeline the driver can't create) panics.
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
use crate::{bindless::Bindless, global::LIFECYCLE, queue::lock, state::Ctx};
use lava_macros::validation_trace;

pub use crate::error::{Error, Result};
pub use crate::features::Features;
pub use crate::lifetime::{Stamp, latest_stamp, reclaim};
pub use crate::vk::Pipeline as VkPipeline;
/// What `Error::Vulkan` holds, e.g. `ERROR_INITIALIZATION_FAILED`.
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
mod global;
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

/// How lava's allocator sizes the memory blocks it sub-allocates from, in bytes: blocks start at
/// `device` (device-local memory) and `host` (host-visible memory) and double with every block
/// that didn't fit an allocation, up to `max_device` and `max_host`. Larger allocations get
/// blocks of their own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryBlocks {
    pub device: u64,
    pub host: u64,
    pub max_device: u64,
    pub max_host: u64,
}

impl MemoryBlocks {
    /// 64 MiB of device-local and 16 MiB of host-visible memory, growing to 256 and 64 MiB.
    pub const DEFAULT: Self = Self {
        device: 64 << 20,
        host: 16 << 20,
        max_device: 256 << 20,
        max_host: 64 << 20,
    };
}

impl Default for MemoryBlocks {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[derive(Clone, Copy)]
pub struct Config {
    /// The `PASS_MAP` of [`include_passes!`].
    pub passes: &'static pipeline::PassMap,
    /// The display to create [`present::PresentSurface`]s for; `None` runs headless.
    pub display: Option<RawDisplayHandle>,
    /// The application's name and version, which drivers and tools see
    /// (`VkApplicationInfo`; the engine is "lava" at its crate version).
    pub app_name: Option<&'static std::ffi::CStr>,
    /// `[major, minor, patch]`.
    pub app_version: [u32; 3],
    /// The Khronos validation layer with synchronization validation, best practices and shader
    /// printf, in any build. Without the layer installed lava warns and runs without it.
    /// Messages are `tracing` events (see `state`'s debug callback for their fields).
    pub validation: bool,
    /// GPU-assisted validation on top of `validation`, together with shader printf. Validation
    /// layers before Vulkan SDK 1.4.309 document the two as unable to run at the same time
    /// (1.4.304's documentation still does, 1.4.309's no longer); with an older layer leave
    /// this off or update the SDK.
    pub gpu_assisted_validation: bool,
    /// Optional features lava leaves off even where the device has them (not asserted ones).
    pub disabled: Features,
    /// A file with the data of lava's `VkPipelineCache`, which every pipeline is made through:
    /// loaded at init where this device and driver wrote it (else ignored with a warning; a
    /// missing file is an empty cache) and written by [`pipeline::save_cache`] and
    /// [`shutdown`]. `None` keeps the cache in memory.
    pub pipeline_cache: Option<&'static std::path::Path>,
    /// The sizes of the allocator's memory blocks.
    pub memory_blocks: MemoryBlocks,
}

impl Config {
    /// Headless, without validation, a name or disabled features, with default memory blocks.
    pub fn new(passes: &'static pipeline::PassMap) -> Self {
        Self {
            passes,
            display: None,
            app_name: None,
            app_version: [0, 0, 0],
            validation: false,
            gpu_assisted_validation: false,
            disabled: Features::DEFAULT,
            pipeline_cache: None,
            memory_blocks: MemoryBlocks::DEFAULT,
        }
    }
}

/// Picks a device with everything lava needs and the asserted features (discrete GPUs first,
/// then integrated ones, then virtual and software ones), enables its optional features but
/// `config.disabled`, and makes the passes' pipelines. Fails for a pass that needs a cargo
/// feature lava was compiled without (before it makes anything), when no GPU fits
/// (`Error::NoSuitableGpu`), and when lava is initialised already; a failed init destroys what
/// it made, so it can be retried. Panics when `config.disabled` names asserted features.
#[validation_trace]
pub fn init(config: &Config) -> Result<()> {
    let _lifecycle = lock(&LIFECYCLE);
    if Ctx::exists() {
        return Err(Error::other("lava is already initialised"));
    }
    let asserted = Features::asserted();
    let asserted_and_disabled = asserted.missing_from(&asserted.without(config.disabled));
    assert!(
        asserted_and_disabled.is_empty(),
        "`Config::disabled` names asserted features: {}",
        asserted_and_disabled.join(", ")
    );
    pipeline::check_compiled(config.passes)?;
    Ctx::init(config)?;
    let rest = (|| {
        pipeline::set_pass_map(config.passes);
        Bindless::init()?;
        #[cfg(feature = "performance-query")]
        query::counters::init()?;
        pipeline::init(config.pipeline_cache)
    })();
    if let Err(err) = rest {
        // SAFETY: lava isn't open, so nothing but this thread has used what it made.
        unsafe { teardown() };
        return Err(err);
    }
    global::open();
    Ok(())
}

/// Waits until the device finished everything submitted so far, then destroys what that frees.
/// Every queue is locked meanwhile, so call it outside of `Submission::submit`.
/// `Error::DeviceLost` once the device is lost.
pub fn wait_idle() -> Result<()> {
    let _call = state::call();
    device_idle()?;
    lifetime::run_ready();
    Ok(())
}

/// Waits for every queue and polls every timeline.
fn device_idle() -> Result<()> {
    {
        let queues = Ctx::get().queues.iter().flat_map(|family| &family.queues);
        let _queues: Vec<_> = queues.map(queue::lock).collect();
        unsafe { Ctx::device().device_wait_idle() }.map_err(state::vk_error)?;
    }
    queue::timeline::poll_all()
}

/// Shuts lava down so that [`init`] can run again: waits for the device, saves the pipeline
/// cache ([`pipeline::save_cache`]), runs every destructor of what was dropped, destroys the
/// pipelines, descriptor heaps, timeline semaphores, device, debug messenger and instance, and
/// leaves lava uninitialised: every lava call panics until the next `init`. Does nothing when
/// lava isn't initialised.
///
/// Drop every lava object first (buffers, images, acceleration structures, timelines,
/// uploaders, surfaces, samplers, slot ranges, query pools). While one is alive, or another
/// thread is inside a lava call, it destroys nothing, lava stays initialised, and it returns
/// `Error::ObjectsAlive`. A lost device is no error here: what is left is destroyed anyway.
/// [`Wait`](queue::Wait)s and [`Stamp`]s of the old context count as done in the next one;
/// counters of [`query::counters`] stay valid memory (they are leaked) but can't be queried.
#[validation_trace]
pub fn shutdown() -> Result<()> {
    let _lifecycle = lock(&LIFECYCLE);
    if !global::is_open() {
        return Ok(());
    }
    global::close().map_err(|count| Error::ObjectsAlive { count })?;
    // SAFETY: closed: no object and no call reads lava's state any more.
    unsafe { teardown() };
    Ok(())
}

/// Destroys everything lava made, in dependency order; what [`shutdown`] and a failed [`init`]
/// run.
///
/// # Safety
/// Under [`LIFECYCLE`], with lava closed or never opened.
unsafe fn teardown() {
    if !Ctx::exists() {
        return;
    }
    match device_idle() {
        Ok(()) | Err(Error::DeviceLost) => {}
        Err(err) => tracing::error!(%err, "failed to wait for the device before shutting lava down"),
    }
    if let Err(err) = pipeline::save_cache_now() {
        tracing::error!(%err, "failed to save the pipeline cache");
    }
    // What the application dropped, then lava's own objects, whose destructors need nothing
    // the earlier ones free.
    lifetime::run_all();
    unsafe {
        pipeline::destroy();
        Bindless::destroy();
    }
    lifetime::run_all();
    unsafe {
        queue::timeline::destroy();
        #[cfg(feature = "performance-query")]
        query::counters::destroy();
        pipeline::unset_pass_map();
        Ctx::destroy();
    }
}

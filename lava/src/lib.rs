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
//! finished its stamp at the drop ([`Stamp`]). Windows present through a [`present::PresentSurface`] and its current
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
//! [`Error`] for the rule). Recording never fails: what goes wrong while a command is recorded
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
//!   descriptors, sampled at the slot and storage at the next; zero for none (slot 0 of the
//!   resource heap is never handed out), and mip `i` of a mip chain `i` times its width
//!   further), [`bindless::SamplerHandle`] (a `u32` slot; slot 0 holds lava's default sampler,
//!   every other one a sampler the application created) and
//!   `acceleration_structure::Instance` (`VkAccelerationStructureInstanceKHR`); a
//!   TLAS is a `u64` device address. Zeroed handles are null, so generated structs are `Pod`.
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
//!
//! # Paths and the prelude
//! The types most code names are re-exported at the crate root (`lava::Buffer`,
//! `lava::BufferSlice`, `lava::Image`, `lava::ImageView`, `lava::CommandBuffer`,
//! `lava::Timeline`, ...) and, with the traits their bounds use, in [`prelude`]. Their modules
//! hold the rest (formats in [`image::format`], usages in [`image::usage`], raster state in
//! [`command_buffer`]).
//!
//! # ash
//! `ash` is a public dependency: [`vk`] re-exports `ash::vk` (with the additions of newer
//! headers lava needs), and lava's API takes and returns its types (stage and access flags,
//! formats, raw handles). An ash upgrade is a breaking lava release; use `lava::vk` rather than
//! an `ash` of your own, so the versions can't differ.
use crate::{bindless::Bindless, global::LIFECYCLE, queue::lock};
use lava_macros::validation_trace;

pub use crate::bindless::{ImageHandle, Sampler, SamplerDesc, SamplerHandle};
pub use crate::buffer::{Buffer, BufferSlice, Uploader};
pub use crate::command_buffer::{
    Accesses, Attachment, BindingOutput, Bindings, BufferAccess, CommandBuffer, Draw, ImageUse,
    RasterState, Rect,
};
pub use crate::error::{Error, Result};
pub use crate::features::{Features, Gated, MissingFeature};
pub use crate::image::{DynamicImage, Image, ImageSlice, ImageView, MipChain};
pub use crate::lifetime::{Stamp, latest_stamp, reclaim};
pub use crate::present::PresentSurface;
pub use crate::query::QueryPool;
pub use crate::queue::{PerFrame, Submission, Timeline, Wait};
pub use crate::state::{Ctx, FormatProperties};
/// The stages timestamps and accesses name.
pub use crate::vk::PipelineStageFlags2;
/// What `Error::Vulkan` holds, e.g. `ERROR_INITIALIZATION_FAILED`.
pub use crate::vk::Result as VkResult;
/// What `format::Dynamic` and `image::usage::Dynamic` hold.
pub use crate::vk::{Format as VkFormat, ImageUsageFlags};
#[cfg(feature = "acceleration-structure")]
pub use crate::vkobjects::acceleration_structure;
#[cfg(feature = "acceleration-structure")]
pub use crate::vkobjects::acceleration_structure::{Blas, Tlas};
pub use crate::vkobjects::physical_device::SubgroupSizes;
/// The derives and traits of generated bindings (with `derive` and `min_const_generics`, for
/// arrays of any length), re-exported like `glam`.
pub use bytemuck;
/// The math types of generated bindings, re-exported so generated code and its users need only
/// lava.
pub use glam;

pub mod bindless;
pub mod buffer;
pub mod command_buffer;
mod error;
#[cfg(all(feature = "external-memory", unix))]
pub mod external;
pub mod features;
mod global;
pub mod half;
pub mod image;
mod lifetime;
pub mod memory;
mod object;
pub mod pipeline;
pub mod present;
pub mod query;
pub mod queue;
#[cfg(feature = "sparse-binding")]
pub mod sparse;
mod state;
#[cfg(test)]
mod tests;
mod unsized_struct;
pub mod vk;
mod vkobjects;
use raw_window_handle::RawDisplayHandle;

/// The common types and traits of lava: `use lava::prelude::*;` brings what recording passes
/// names (buffers, images and their views, the view traits their bounds use, command buffers,
/// draws and their state, timelines, surfaces, samplers, handles, the dimension markers and
/// `usage!`). It also brings the [`glam`] and [`bytemuck`] modules lava re-exports, as modules
/// (`glam::Vec3`, `bytemuck::Pod`), so their items keep their paths and an application's own
/// imports of them take precedence. It leaves out `Result` and `Error`, which would shadow
/// std's.
pub mod prelude {
    pub use crate::bindless::{ImageHandle, Sampler, SamplerDesc, SamplerHandle};
    pub use crate::buffer::{Buffer, BufferSlice, HostVisible, Memory, Uploader};
    pub use crate::command_buffer::{
        Accesses, Attachment, Blend, BufferAccess, CommandBuffer, Draw, ImageUse, RasterState, Rect,
    };
    pub use crate::image::format::{Format, StaticFormat};
    pub use crate::image::usage::{ImageUsage, StaticUsage, usage};
    pub use crate::image::{
        ColorTarget, D2, D2Array, D3, DepthTarget, Dim, Image, ImageRef, ImageSlice, ImageView,
        MipChain, SampledView, StencilTarget, StorageView, TransferDstView, TransferSrcView,
    };
    pub use crate::present::PresentSurface;
    pub use crate::queue::{Graphics, PerFrame, Timeline, Wait};
    #[cfg(feature = "acceleration-structure")]
    pub use crate::vkobjects::acceleration_structure::{Blas, Tlas};
    pub use crate::{bytemuck, glam};
}

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
    /// The first device-local block.
    pub device: u64,
    /// The first host-visible block.
    pub host: u64,
    /// The largest device-local block lava grows to.
    pub max_device: u64,
    /// The largest host-visible block lava grows to.
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

/// What [`init`] makes lava's context with: the application's passes and display, how it
/// names itself to drivers, validation, the optional features to leave off, the pipeline cache
/// file and the allocator's block sizes. Start from [`Config::new`] and set the fields that
/// differ: `Config { display, validation: cfg!(debug_assertions), ..Config::new(&PASS_MAP) }`.
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
    /// printf (`debugPrintfEXT`, logged at `info`), in any build. Without the layer installed
    /// lava warns and runs without it.
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

impl std::fmt::Debug for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let passes: Vec<_> = self.passes.passes.iter().map(|pass| pass.name).collect();
        f.debug_struct("Config")
            .field("passes", &passes)
            .field("display", &self.display)
            .field("app_name", &self.app_name)
            .field("app_version", &self.app_version)
            .field("validation", &self.validation)
            .field("gpu_assisted_validation", &self.gpu_assisted_validation)
            .field("disabled", &self.disabled)
            .field("pipeline_cache", &self.pipeline_cache)
            .field("memory_blocks", &self.memory_blocks)
            .finish()
    }
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
            disabled: Features::NONE,
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
/// [`Wait`]s and [`Stamp`]s of the old context count as done in the next one; counters of
/// `query::counters` (`performance-query`) stay valid memory (they are leaked) but can't be queried.
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
        Err(err) => {
            tracing::error!(%err, "failed to wait for the device before shutting lava down")
        }
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

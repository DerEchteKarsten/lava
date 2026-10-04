//! Lava: typed, bindless Vulkan abstraction over ash with shader-generated passes
// `BindingOutput` grows its access arrays by one per registered resource.
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

#[cfg(not(feature = "test-passes"))]
pub mod bindings;
/// With `test-passes`, the bindings also cover the passes in `lava/tests/shaders` and are
/// generated into OUT_DIR instead of `src/bindings.rs`.
#[cfg(feature = "test-passes")]
pub mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}
pub mod bindless;
pub mod buffer;
pub mod command_buffer;
pub mod error;
pub mod image;
pub mod state;
pub mod vkobjects;
use raw_window_handle::RawDisplayHandle;

/// Initialises the global Vulkan context. Pass the display handle to be able to create
/// [`vkobjects::surface::Surface`]s and swapchains; `None` runs headless.
pub fn init(
    display: Option<&RawDisplayHandle>,
    enable_validation: bool,
    enable_gpu_assited_validation: bool,
) -> Result<()> {
    Ctx::init(display, enable_validation, enable_gpu_assited_validation)?;
    Bindless::init()?;
    command_buffer::init();
    Ok(())
}

#[track_caller]
pub fn destroy() {
    CALLSITE.set(Some(Location::caller().clone()));
    Bindless::destroy();
}

//! Lava: typed, bindless Vulkan abstraction over ash with shader-generated passes
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

pub mod bindings;
pub mod bindless;
pub mod buffer;
pub mod command_buffer;
pub mod error;
pub mod image;
pub mod state;
pub mod vkobjects;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

pub fn init(
    display: &RawDisplayHandle,
    window: &RawWindowHandle,
    enable_validation: bool,
    enable_gpu_assited_validation: bool,
) -> Result<()> {
    Ctx::init(
        display,
        window,
        enable_validation,
        enable_gpu_assited_validation,
    )?;
    Bindless::init()?;
    command_buffer::init();
    Ok(())
}

#[track_caller]
pub fn destroy() {
    CALLSITE.set(Some(Location::caller().clone()));
    Bindless::destroy();
}

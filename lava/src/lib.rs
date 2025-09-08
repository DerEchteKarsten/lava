use anyhow::Result;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::state::Ctx;

mod bindless;
mod vulkan;
mod pipeline_cache;
mod shader_cache;
mod vkobjects;
mod state;

#[cfg(test)]
mod tests;

const FRAMES_IN_FLIGHT: usize = 2;

fn init(display_handle: &dyn HasDisplayHandle, window_handle: &dyn HasWindowHandle) -> Result<()> {
    Ctx::init(display_handle, window_handle)?;
    Ok(())
}
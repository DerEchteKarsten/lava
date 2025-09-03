mod bindless;
mod vulkan;
mod pipeline_cache;
mod extensions;
mod shader_cache;
mod properties;
mod vkobjects;

#[cfg(test)]
mod tests;

pub const FRAMES_IN_FLIGHT: usize = 3;

pub struct Lava {
    device: Device,
}
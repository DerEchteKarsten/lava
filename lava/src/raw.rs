//! Lava's Vulkan handles for benchmarks against hand-written Vulkan on the same device (`lava-bench`'s baseline): hidden, unchecked and without stability
//!
//! Everything here hands out what lava already holds: its instance, device and extension
//! loaders, its descriptor heaps and its queues. Lava tracks nothing that is recorded with them:
//! a command buffer recorded through these must not touch what lava's own recordings track
//! (keep the resources apart), and the caller synchronises and keeps alive what it uses. The
//! handles are valid between `lava::init` and `lava::shutdown`.

use std::sync::MutexGuard;

use crate::Result;
use crate::bindless::{self, Bindless};
use crate::image::{Dim, ImageView, format::Format, usage::ImageUsage};
use crate::queue::{Kind, lock};
use crate::state::{Ctx, Functions};
use crate::vk;

/// Lava's instance.
pub fn instance() -> &'static ash::Instance {
    Functions::instance()
}

/// Lava's device.
pub fn device() -> &'static ash::Device {
    Ctx::device()
}

/// The physical device lava picked.
pub fn physical_device() -> vk::PhysicalDevice {
    Ctx::physical_device().handle
}

/// `VK_KHR_device_address_commands`' functions, which take address ranges.
pub fn device_address_commands() -> &'static vk::device_address_commands::Device {
    Functions::device_address_commands()
}

/// `VK_EXT_descriptor_heap`'s functions: heap binds and push data.
pub fn descriptor_heap() -> &'static vk::descriptor_heap::Device {
    Functions::descriptor_heap()
}

/// Binds lava's two descriptor heaps in `cmd`, as every lava recording of a graphics or compute
/// family does first.
///
/// # Safety
/// `cmd` is recording, on a graphics or compute family.
pub unsafe fn bind_heaps(cmd: vk::CommandBuffer) {
    Bindless::bind(cmd, Kind::Graphics);
}

/// Queue `index` of `kind`'s family, locked like lava locks it for its submissions, or `None`
/// where lava created none.
pub fn queue(kind: Kind, index: usize) -> Option<MutexGuard<'static, vk::Queue>> {
    let family = Ctx::get().queues.family(kind)?;
    family.queues.get(index).map(lock)
}

/// Rewrites the descriptors of `view`'s handle (`view.dynamic_handle()`, bound by lava), its
/// sampled descriptor in `sampled_layout` instead of lava's `GENERAL` (a storage descriptor
/// stays `GENERAL`), for an image only hand-written recordings access in optimal layouts.
///
/// # Safety
/// No submission in flight reads the handle's slots, and every later access through them sees
/// the image in that layout.
pub unsafe fn write_descriptors<F: Format, U: ImageUsage, D: Dim>(
    view: ImageView<'_, F, U, D>,
    sampled_layout: vk::ImageLayout,
) -> Result<()> {
    let handle = view.dynamic_handle();
    assert!(!handle.is_null(), "the view has no bindless handle");
    bindless::write_descriptors_in([view.shader_image()], [handle], sampled_layout)
}

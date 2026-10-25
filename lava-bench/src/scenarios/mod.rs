//! The workloads, from an everyday game frame to specialised compute, and what both modes of them share: setup submissions and the image helpers of the baseline

use lava::queue::Graphics;

use crate::raw::{RawCmd, RawQueue};
use crate::scenario::{Scenario, Stamped};

mod async_compute;
mod deferred;
mod fluid;
mod forward;
mod gpu_driven;
mod image_processing;
mod ml;
mod particles;
mod ray_tracing;
mod streaming;

/// Every scenario, in the order they run.
pub const NAMES: &[&str] = &["forward", "deferred", "gpu_driven", "streaming", "particles", "ml", "ml_train", "image_processing", "fluid", "ray_tracing", "async_compute"];

/// The scenario `name`, or `None` where the device lacks what it needs.
pub fn make(
    name: &str,
    graphics: &mut Stamped<Graphics>,
    queue: &mut RawQueue,
) -> Option<Box<dyn Scenario>> {
    let mut setup = Setup { graphics, queue };
    Some(match name {
        "forward" => Box::new(forward::Forward::new(&mut setup)),
        "deferred" => Box::new(deferred::Deferred::new(&mut setup)),
        "gpu_driven" => Box::new(gpu_driven::GpuDriven::new(&mut setup)),
        "streaming" => Box::new(streaming::Streaming::new(&mut setup)),
        "particles" => Box::new(particles::Particles::new(&mut setup)),
        "ml" => Box::new(ml::Ml::new(&mut setup, false)),
        "ml_train" => Box::new(ml::Ml::new(&mut setup, true)),
        "image_processing" => Box::new(image_processing::ImageProcessing::new(&mut setup)),
        "fluid" => Box::new(fluid::Fluid::new(&mut setup)),
        "ray_tracing" => Box::new(ray_tracing::RayTracing::new(&mut setup)?),
        "async_compute" => Box::new(async_compute::AsyncCompute::new(&mut setup)?),
        _ => unreachable!("{name} is checked by `args`"),
    })
}

/// Why `make` skipped `name`.
pub fn skip_reason(name: &str) -> &'static str {
    match name {
        "ray_tracing" => "the device has no ray queries",
        "async_compute" => "the device has no compute family without graphics",
        _ => "the device lacks a feature it needs",
    }
}

/// Whether the device has `VK_KHR_unified_image_layouts`, where `GENERAL` costs nothing.
pub fn has_unified_layouts() -> bool {
    let instance = lava::raw::instance();
    let extensions = unsafe {
        instance
            .enumerate_device_extension_properties(lava::raw::physical_device())
            .unwrap()
    };
    extensions.iter().any(|e| {
        e.extension_name_as_c_str() == Ok(c"VK_KHR_unified_image_layouts")
    })
}

/// What a scenario's constructor records its setup with: one lava submission and one raw one,
/// each blocked on.
pub struct Setup<'a> {
    graphics: &'a mut Stamped<Graphics>,
    queue: &'a mut RawQueue,
}

impl Setup<'_> {
    /// Records `f` in a lava submission (after draining the uploads) and waits for it.
    pub fn lava(&mut self, f: impl FnOnce(&mut lava::CommandBuffer<Graphics>)) {
        let (wait, _) = self.graphics.submit(
            |_| {},
            |cmd| {
                cmd.drain_uploads();
                f(cmd);
            },
        );
        wait.block().unwrap();
    }

    /// Records `f` by hand and waits for it.
    pub fn raw(&mut self, f: impl FnOnce(&mut RawCmd)) {
        self.queue.submit(&[], f);
        self.queue.block();
    }

    pub fn uploader(&self) -> lava::Uploader {
        self.graphics.timeline.uploader()
    }
}

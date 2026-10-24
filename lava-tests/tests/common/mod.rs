//! Shared harness for lava's GPU tests: one headless context per process (with the features a test binary disables, its staging ring size and pipeline cache file) and its graphics timeline, serialised and validation-checked (the layer asserted active, init's errors included), a subscriber that shows lava's warnings in failing tests, and panic messages
#![allow(dead_code)]

pub mod golden;
pub mod render;

use std::sync::{
    LazyLock, Mutex, MutexGuard, Once, OnceLock,
    atomic::{AtomicBool, Ordering},
};

use bytemuck::Pod;
use lava::{
    bindless::{Sampler, SamplerDesc, SamplerHandle},
    buffer::{Buffer, HostVisible, Readback, Upload, Uploader},
    command_buffer::CommandBuffer,
    queue::{Graphics, Timeline, Wait},
};
use tracing::{
    Level,
    field::{Field, Visit},
};
use tracing_subscriber::{
    Layer,
    filter::{Targets, filter_fn},
    layer::SubscriberExt,
};

static VALIDATION_ERRORS: Mutex<Vec<String>> = Mutex::new(Vec::new());
/// Set when lava warned that `Config::validation` found no validation layer.
static LAYER_MISSING: AtomicBool = AtomicBool::new(false);

/// Collects validation errors (with the lava call they came from) and notices lava's warning
/// that the validation layer is missing.
struct ValidationCapture;

#[derive(Default)]
struct Fields {
    message: String,
    location: Option<String>,
}

impl Visit for Fields {
    fn record_str(&mut self, field: &Field, value: &str) {
        self.record_debug(field, &value as &dyn std::fmt::Debug)
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        match field.name() {
            "message" => self.message = format!("{value:?}"),
            "validation_location" => self.location = Some(format!("{value:?}")),
            _ => {}
        }
    }
}

impl<S: tracing::Subscriber> Layer<S> for ValidationCapture {
    fn on_event(&self, event: &tracing::Event<'_>, _: tracing_subscriber::layer::Context<'_, S>) {
        let meta = event.metadata();
        let mut fields = Fields::default();
        event.record(&mut fields);
        if meta.target() == "vulkan-validation" {
            let message = match fields.location {
                Some(location) => format!("{}\n  (in the lava call at {location})", fields.message),
                None => fields.message,
            };
            VALIDATION_ERRORS.lock().unwrap().push(message);
        } else if fields.message.contains("validation layer isn't installed") {
            LAYER_MISSING.store(true, Ordering::Relaxed);
        }
    }
}

pub type Cmd<'p> = CommandBuffer<'p, Graphics>;

pub struct Gpu {
    pub timeline: Timeline<Graphics>,
}

/// Installs the tests' subscriber (once per process; call before `lava::init`): validation
/// errors are collected for [`take_validation_errors`], and lava's warnings and errors and the
/// validation errors are printed through the test writer, so a failing test shows them.
pub fn capture_validation_errors() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        let capture = ValidationCapture.with_filter(filter_fn(|meta| {
            let target = meta.target();
            (target == "vulkan-validation" && *meta.level() == Level::ERROR)
                || (target.starts_with("lava") && *meta.level() <= Level::WARN)
        }));
        let log = tracing_subscriber::fmt::layer()
            .with_test_writer()
            .with_filter(
                Targets::new()
                    .with_target("lava", Level::WARN)
                    .with_target("lava_build_slang", Level::WARN)
                    .with_target("vulkan-validation", Level::ERROR),
            );
        tracing::subscriber::set_global_default(
            tracing_subscriber::registry().with(capture).with(log),
        )
        .expect("no other tracing subscriber is installed");
    });
}

/// The validation errors raised since the last call.
pub fn take_validation_errors() -> Vec<String> {
    std::mem::take(&mut *VALIDATION_ERRORS.lock().unwrap())
}

/// Fails unless the validation layer `lava::init` was asked for is active: the GPU tests count
/// on it, so a missing layer must not pass them silently.
pub fn assert_validation_active() {
    assert!(
        !LAYER_MISSING.load(Ordering::Relaxed),
        "the GPU tests run with `Config::validation`, but lava found no Khronos validation layer: install it (e.g. the vulkan-validation-layers package)"
    );
}

static DISABLED: OnceLock<lava::Features> = OnceLock::new();

/// Makes the process's context leave `features` off; call before the first `gpu()`.
pub fn disable(features: lava::Features) {
    assert_eq!(
        *DISABLED.get_or_init(|| features),
        features,
        "one set of disabled features per process"
    );
}

static STAGING: OnceLock<u64> = OnceLock::new();

/// The size of the harness timeline's staging ring (16 MiB by default; none on a `rebar`
/// device whatever the size); call before the first `gpu()`.
pub fn staging(bytes: u64) {
    assert_eq!(
        *STAGING.get_or_init(|| bytes),
        bytes,
        "one ring size per process"
    );
}

static PIPELINE_CACHE: OnceLock<&'static std::path::Path> = OnceLock::new();

/// Makes the process's context load and save its pipeline cache at `path`
/// (`Config::pipeline_cache`); call before the first `gpu()`.
pub fn pipeline_cache(path: &'static std::path::Path) {
    assert_eq!(
        *PIPELINE_CACHE.get_or_init(|| path),
        path,
        "one cache per process"
    );
}

static GPU_ASSISTED: OnceLock<bool> = OnceLock::new();

/// Turns on GPU-assisted validation (`Config::gpu_assisted_validation`) for the process's
/// context; call before the first `gpu()`.
pub fn gpu_assisted_validation() {
    GPU_ASSISTED.get_or_init(|| true);
}

/// The process's context, or why it couldn't be made: every test then fails with that reason.
static GPU: LazyLock<Result<Mutex<Gpu>, String>> = LazyLock::new(|| {
    capture_validation_errors();
    std::panic::catch_unwind(init_gpu)
        .unwrap_or_else(|payload| {
            Err(format!(
                "the harness panicked: {}",
                payload_message(&*payload)
            ))
        })
        .map(Mutex::new)
});

fn init_gpu() -> Result<Gpu, String> {
    let config = lava::Config {
        validation: true,
        gpu_assisted_validation: GPU_ASSISTED.get().copied().unwrap_or(false),
        disabled: DISABLED.get().copied().unwrap_or_default(),
        pipeline_cache: PIPELINE_CACHE.get().copied(),
        ..lava::Config::new(&lava_tests::bindings::PASS_MAP)
    };
    lava::init(&config).map_err(|error| {
        format!(
            "headless lava init failed: {error}; the GPU tests need a Vulkan 1.4 GPU with VK_EXT_descriptor_heap and the Khronos validation layer"
        )
    })?;
    assert_validation_active();
    let nearest = SamplerDesc::new()
        .filter(lava::vk::Filter::NEAREST)
        .mipmap_mode(lava::vk::SamplerMipmapMode::NEAREST)
        .address_mode(lava::vk::SamplerAddressMode::CLAMP_TO_BORDER)
        .border_color(lava::vk::BorderColor::FLOAT_OPAQUE_WHITE);
    let _ = NEAREST.set(Sampler::new(&nearest).unwrap());
    let staging = STAGING.get().copied().unwrap_or(16 << 20);
    let gpu = Gpu {
        timeline: Timeline::new(2, staging).unwrap(),
    };
    let errors = take_validation_errors();
    if !errors.is_empty() {
        return Err(format!(
            "Vulkan validation errors in lava::init:\n{}",
            errors.join("\n\n")
        ));
    }
    Ok(gpu)
}

static NEAREST: OnceLock<Sampler> = OnceLock::new();

/// A sampler of the nearest texel and mip, clamped to a white border; after the first `gpu()`.
pub fn nearest() -> SamplerHandle {
    NEAREST
        .get()
        .expect("`gpu()` creates the samplers")
        .handle()
}

/// Exclusive access to the GPU for one test; fails it on validation errors when dropped.
pub struct GpuGuard(MutexGuard<'static, Gpu>);

impl std::ops::Deref for GpuGuard {
    type Target = Gpu;
    fn deref(&self) -> &Gpu {
        &self.0
    }
}

impl std::ops::DerefMut for GpuGuard {
    fn deref_mut(&mut self) -> &mut Gpu {
        &mut self.0
    }
}

impl Drop for GpuGuard {
    fn drop(&mut self) {
        let errors = take_validation_errors();
        if errors.is_empty() {
            return;
        }
        let report = format!("Vulkan validation errors:\n{}", errors.join("\n\n"));
        if std::thread::panicking() {
            eprintln!("{report}");
        } else {
            panic!("{report}");
        }
    }
}

/// Tests run one at a time: lava's context is global and some devices have one graphics queue.
/// Validation errors raised outside any test's guard (by destructors that ran late) fail the
/// next test that takes it.
pub fn gpu() -> GpuGuard {
    let gpu = match &*GPU {
        Ok(gpu) => gpu,
        Err(error) => panic!("{error}"),
    };
    GpuGuard(gpu.lock().unwrap_or_else(|poisoned| poisoned.into_inner()))
}

impl Gpu {
    /// Records one command buffer, submits it, and waits until the GPU has executed it.
    pub fn submit(&mut self, record: impl FnOnce(&mut Cmd)) -> Wait {
        self.submit_after(&[], record)
    }

    /// Like [`Self::submit`], after `waits` of other timelines.
    pub fn submit_after(&mut self, waits: &[Wait], record: impl FnOnce(&mut Cmd)) -> Wait {
        let mut submission = self.timeline.begin().unwrap();
        for &wait in waits {
            submission.wait_for(wait);
        }
        let done = submission.submit(record).unwrap();
        done.block().unwrap();
        done
    }

    /// An uploader of the timeline's ring, for graphics.
    pub fn uploader(&self) -> Uploader {
        self.timeline.uploader()
    }

    /// A device-local buffer holding `data`, its copy drained and completed where it was
    /// staged.
    pub fn buffer<T: Pod>(&mut self, data: &[T]) -> Buffer<T> {
        let uploader = self.uploader();
        let buffer = uploader.from_slice(data).unwrap();
        if uploader.pending() > 0 {
            self.submit(|cmd| {
                cmd.drain_uploads();
            });
        }
        buffer
    }
}

/// The message `f` panics with; fails the test if it doesn't panic.
#[track_caller]
pub fn panic_message<R>(f: impl FnOnce() -> R) -> String {
    let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) else {
        panic!("it should panic");
    };
    payload_message(&*payload)
}

/// The text of a panic's payload.
pub fn payload_message(payload: &(dyn std::any::Any + Send)) -> String {
    (payload.downcast_ref::<String>().cloned())
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

/// A buffer in `memory` holding `data`.
pub fn buffer_in<T: Pod, M: HostVisible>(memory: M, data: &[T]) -> Buffer<T, M> {
    let mut buffer = Buffer::with_memory(memory, data.len()).unwrap();
    buffer.host_mut().copy_from_slice(data);
    buffer
}

/// An `Upload` buffer holding `data`, for what the GPU reads.
pub fn buffer_with<T: Pod>(data: &[T]) -> Buffer<T, Upload> {
    buffer_in(Upload, data)
}

/// A `Readback` buffer of `len` zeroed elements, for what the GPU writes.
pub fn zeroed_buffer<T: Pod>(len: usize) -> Buffer<T, Readback> {
    buffer_in(Readback, &vec![T::zeroed(); len])
}

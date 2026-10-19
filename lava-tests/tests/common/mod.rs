//! Shared harness for lava's GPU tests: one headless context per process (with the features a test binary disables) and its graphics timeline, serialised, validation-checked
#![allow(dead_code)]

pub mod golden;

use std::sync::{Arc, LazyLock, Mutex, MutexGuard, OnceLock};

use bytemuck::Pod;
use lava::{
    bindless::{Sampler, SamplerDesc, SamplerHandle},
    buffer::Buffer,
    command_buffer::CommandBuffer,
    queue::{Graphics, Timeline, Wait},
};
use tracing::field::{Field, Visit};
use tracing_subscriber::{Layer, layer::SubscriberExt};

static VALIDATION_ERRORS: LazyLock<Arc<Mutex<Vec<String>>>> = LazyLock::new(Default::default);

struct ValidationCapture;

struct Message(String);

impl Visit for Message {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.0 = format!("{value:?}");
        }
    }
}

impl<S: tracing::Subscriber> Layer<S> for ValidationCapture {
    fn on_event(&self, event: &tracing::Event<'_>, _: tracing_subscriber::layer::Context<'_, S>) {
        let meta = event.metadata();
        if meta.target() == "vulkan-validation" && *meta.level() == tracing::Level::ERROR {
            let mut message = Message(String::new());
            event.record(&mut message);
            VALIDATION_ERRORS.lock().unwrap().push(message.0);
        }
    }
}

pub type Cmd<'p> = CommandBuffer<'p, Graphics>;

pub struct Gpu {
    pub timeline: Timeline<Graphics>,
}

/// Starts collecting validation errors; call once, before `lava::init`.
pub fn capture_validation_errors() {
    tracing::subscriber::set_global_default(tracing_subscriber::registry().with(ValidationCapture))
        .expect("no other tracing subscriber is installed");
}

pub fn take_validation_errors() -> Vec<String> {
    std::mem::take(&mut *VALIDATION_ERRORS.lock().unwrap())
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

static GPU: LazyLock<Mutex<Gpu>> = LazyLock::new(|| {
    capture_validation_errors();
    let config = lava::Config {
        validation: true,
        disabled: DISABLED.get().copied().unwrap_or_default(),
        ..lava::Config::new(&lava_tests::bindings::PASS_MAP)
    };
    lava::init(&config).expect(
        "headless lava init failed; the GPU tests need a Vulkan 1.4 GPU with VK_EXT_descriptor_heap and the Khronos validation layer",
    );
    let nearest = SamplerDesc::new()
        .filter(lava::vk::Filter::NEAREST)
        .mipmap_mode(lava::vk::SamplerMipmapMode::NEAREST)
        .address_mode(lava::vk::SamplerAddressMode::CLAMP_TO_BORDER)
        .border_color(lava::vk::BorderColor::FLOAT_OPAQUE_WHITE);
    let _ = NEAREST.set(Sampler::new(&nearest).unwrap());
    Mutex::new(Gpu {
        timeline: Timeline::new(2).unwrap(),
    })
});

static NEAREST: OnceLock<Sampler> = OnceLock::new();

/// A sampler of the nearest texel and mip, clamped to a white border; after the first `gpu()`.
pub fn nearest() -> SamplerHandle {
    NEAREST.get().expect("`gpu()` creates the samplers").handle()
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
pub fn gpu() -> GpuGuard {
    let guard = GPU.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    take_validation_errors();
    GpuGuard(guard)
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
}

/// The message `f` panics with; fails the test if it doesn't panic.
#[track_caller]
pub fn panic_message<R>(f: impl FnOnce() -> R) -> String {
    let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) else {
        panic!("it should panic");
    };
    (payload.downcast_ref::<String>().cloned())
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

/// A host-readable buffer holding `data`.
pub fn buffer_with<T: Pod>(data: &[T]) -> Buffer<T> {
    let mut buffer = Buffer::new(data.len(), true).unwrap();
    buffer.host_mut().copy_from_slice(data);
    buffer
}

/// A host-readable buffer of `len` zeroed elements.
pub fn zeroed_buffer<T: Pod>(len: usize) -> Buffer<T> {
    buffer_with(&vec![T::zeroed(); len])
}

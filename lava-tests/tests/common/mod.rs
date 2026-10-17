//! Shared harness for lava's GPU tests: one headless context per process (with the features a test binary disables) and its graphics timeline, serialised, validation-checked
#![allow(dead_code)]

pub mod golden;

use std::sync::{Arc, LazyLock, Mutex, MutexGuard, OnceLock};

use bytemuck::Pod;
use lava::{
    buffer::{Buffer, usage::StaticUsage},
    command_buffer::CommandBuffer,
    queue::{Graphics, Timeline, Wait},
};
use tracing::field::{Field, Visit};
use tracing_subscriber::{Layer, layer::SubscriberExt};

/// Validation errors reported by the Vulkan validation layer since the current test started.
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

/// What the tests record into.
pub type Cmd = CommandBuffer<Graphics>;

/// The process-wide lava context plus the one graphics timeline all tests share.
pub struct Gpu {
    pub timeline: Timeline<Graphics>,
}

/// Starts collecting validation errors; call once, before `lava::init`.
pub fn capture_validation_errors() {
    tracing::subscriber::set_global_default(tracing_subscriber::registry().with(ValidationCapture))
        .expect("no other tracing subscriber is installed");
}

/// Returns and forgets the validation errors reported so far.
pub fn take_validation_errors() -> Vec<String> {
    std::mem::take(&mut *VALIDATION_ERRORS.lock().unwrap())
}

/// The features the context leaves off (`Config::disabled`), set by a test binary before its
/// first `gpu()`.
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
        "headless lava init failed; the GPU tests need a Vulkan 1.4 GPU and the Khronos validation layer",
    );
    Mutex::new(Gpu {
        timeline: Timeline::builder().build().unwrap(),
    })
});

/// Exclusive access to the GPU for one test. When the test ends without panicking, the guard
/// fails it if the validation layer reported any error in the meantime.
pub struct GpuGuard(MutexGuard<'static, Gpu>);

impl std::ops::Deref for GpuGuard {
    type Target = Gpu;
    fn deref(&self) -> &Gpu {
        &self.0
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
            // The test already failed; the validation errors usually explain why.
            eprintln!("{report}");
        } else {
            panic!("{report}");
        }
    }
}

/// Locks the GPU for the calling test. Tests run one at a time: lava's context is global and
/// some devices have a single graphics queue.
pub fn gpu() -> GpuGuard {
    let guard = GPU.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    take_validation_errors();
    GpuGuard(guard)
}

impl Gpu {
    /// Records one command buffer, submits it, and waits until the GPU has executed it.
    pub fn submit(&self, record: impl FnOnce(&mut Cmd)) -> Wait {
        self.submit_after(&[], record)
    }

    /// Like [`Self::submit`], after `waits` of other timelines.
    pub fn submit_after(&self, waits: &[Wait], record: impl FnOnce(&mut Cmd)) -> Wait {
        let mut submission = self.timeline.reserve().begin().unwrap();
        for &wait in waits {
            submission.wait_for(wait);
        }
        let done = submission.submit(record).unwrap();
        done.block().unwrap();
        done
    }
}

/// A host-readable buffer holding `data`.
pub fn buffer_with<T: Pod, U: StaticUsage>(data: &[T]) -> Buffer<T, U> {
    let mut buffer = Buffer::new(data.len(), true).unwrap();
    buffer.host_mut().copy_from_slice(data);
    buffer
}

/// A host-readable buffer of `len` zeroed elements.
pub fn zeroed_buffer<T: Pod, U: StaticUsage>(len: usize) -> Buffer<T, U> {
    buffer_with(&vec![T::zeroed(); len])
}

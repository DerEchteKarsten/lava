//! Shared harness for lava's GPU tests: one headless context, serialised, validation-checked
#![allow(dead_code)]

pub mod golden;

use std::sync::{Arc, LazyLock, Mutex, MutexGuard};

use bytemuck::Pod;
use lava::{
    buffer::{Buffer, usage::BufferUsage},
    command_buffer::CommandBuffer,
    vkobjects::queue::{FrameSlot, Gfx, PendingAccesses, Queue, SemaphoreInfo},
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

/// The process-wide lava context plus the one graphics queue all tests share.
pub struct Gpu {
    pub queue: Queue<Gfx>,
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

static GPU: LazyLock<Mutex<Gpu>> = LazyLock::new(|| {
    capture_validation_errors();
    lava::init(None, true, false).expect(
        "headless lava init failed; the GPU tests need a Vulkan 1.4 GPU and the Khronos validation layer",
    );
    Mutex::new(Gpu {
        queue: Queue::new().expect("a graphics queue is available"),
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
    pub fn submit(&self, record: impl FnOnce(&mut CommandBuffer)) {
        self.submit_with(&[], &[], record);
    }

    pub fn submit_with(
        &self,
        wait_on: &[SemaphoreInfo],
        signal: &[SemaphoreInfo],
        record: impl FnOnce(&mut CommandBuffer),
    ) {
        let mut slot = FrameSlot::new(&self.queue).unwrap();
        slot.begin()
            .unwrap()
            .execute(
                &self.queue,
                PendingAccesses::default(),
                wait_on,
                signal,
                record,
            )
            .unwrap();
        // Dropping a submitted slot waits for its fence.
    }
}

/// A host-readable buffer holding `data`.
pub fn buffer_with<T: Pod, U: BufferUsage>(data: &[T]) -> Buffer<T, U> {
    let buffer = Buffer::new(data.len(), true).unwrap();
    buffer.range(..).copy_from(data);
    buffer
}

/// A host-readable buffer of `len` zeroed elements.
pub fn zeroed_buffer<T: Pod, U: BufferUsage>(len: usize) -> Buffer<T, U> {
    buffer_with(&vec![T::zeroed(); len])
}

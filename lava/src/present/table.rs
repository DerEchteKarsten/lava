//! The swapchain images an application acquired, by `vk::Image`: their acquire semaphore, which the first submission that accesses one waits for, and their present semaphore, which the submission that presents one (`CommandBuffer::present`) signals
use std::sync::{LazyLock, Mutex};

use crate::vk;
use hashbrown::HashMap;
use lava_macros::validation_trace;

use crate::{
    command_buffer::CommandBuffer,
    image::{D2, kind::ImageRef},
    queue::{HasGraphics, device::QueueHandle, lock},
};

/// One acquired image.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Entry {
    /// Signalled once the presentation engine is done with the image; the image's first access
    /// waits for it (its state's `acquired` flag).
    pub(crate) acquire: vk::Semaphore,
    /// The image's own: signalled by the submission that presents it, waited for by the
    /// present.
    pub(crate) present: vk::Semaphore,
    /// The timeline and value of the submission that waited for `acquire`.
    pub(crate) waited: Option<(usize, u64)>,
    /// The submission that presented it.
    pub(crate) presented: Option<Presenter>,
}

/// The submission that presented an image: the present goes to its queue, after it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Presenter {
    pub(crate) queue: &'static QueueHandle,
    pub(crate) timeline: usize,
}

/// A recording's wait for an acquired image, at the stage of its first access.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AcquireWait {
    pub(crate) image: vk::Image,
    pub(crate) semaphore: vk::Semaphore,
    pub(crate) stage: vk::PipelineStageFlags2,
}

/// A recording's present of an image: its submission signals the image's present semaphore.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PresentSignal {
    pub(crate) image: vk::Image,
    pub(crate) semaphore: vk::Semaphore,
}

static PRESENTS: LazyLock<Mutex<HashMap<vk::Image, Entry>>> = LazyLock::new(Default::default);

/// `image` was just acquired, signalling `acquire`.
pub(crate) fn insert(image: vk::Image, acquire: vk::Semaphore, present: vk::Semaphore) {
    let entry = Entry {
        acquire,
        present,
        waited: None,
        presented: None,
    };
    lock(&PRESENTS).insert(image, entry);
}

/// Removes `image`'s entry (it is presented, abandoned or destroyed).
pub(crate) fn take(image: vk::Image) -> Option<Entry> {
    lock(&PRESENTS).remove(&image)
}

pub(crate) fn get(image: vk::Image) -> Option<Entry> {
    lock(&PRESENTS).get(&image).copied()
}

/// The acquire semaphore of `image`, whose state says it is acquired.
pub(crate) fn acquire_semaphore(image: vk::Image) -> vk::Semaphore {
    let entry = get(image).expect("an image flagged acquired is in the table");
    entry.acquire
}

/// A submission of value `value` on `timeline` (to `queue`) waited for `waits` and presented
/// `presents`.
pub(crate) fn submitted(
    waits: &[AcquireWait],
    presents: &[PresentSignal],
    queue: &'static QueueHandle,
    timeline: usize,
    value: u64,
) {
    if waits.is_empty() && presents.is_empty() {
        return;
    }
    let mut table = lock(&PRESENTS);
    for wait in waits {
        if let Some(entry) = table.get_mut(&wait.image) {
            entry.waited = Some((timeline, value));
        }
    }
    for present in presents {
        if let Some(entry) = table.get_mut(&present.image) {
            entry.presented = Some(Presenter { queue, timeline });
        }
    }
}

impl<K: HasGraphics> CommandBuffer<K> {
    /// Moves `swapchain_image`, an acquired image of a `PresentSurface` (`surface.image(&img)`),
    /// into `PRESENT_SRC_KHR` once the earlier accesses of it are done. The submission signals
    /// the image's present semaphore, which `PresentSurface::present` waits for. Panics for any
    /// other image.
    #[validation_trace]
    pub fn present(&mut self, swapchain_image: impl ImageRef<Dim = D2>) {
        let image = swapchain_image.view();
        let Some(entry) = get(image.image) else {
            panic!(
                "`present` takes an image `PresentSurface::acquire` handed out, not presented yet"
            );
        };
        self.rec.flush_pending(
            std::iter::once(image.access(
                vk::PipelineStageFlags2::empty(),
                vk::AccessFlags2::empty(),
                vk::ImageLayout::PRESENT_SRC_KHR,
            )),
            std::iter::empty(),
        );
        if !self.rec.presents.iter().any(|p| p.image == image.image) {
            self.rec.presents.push(PresentSignal {
                image: image.image,
                semaphore: entry.present,
            });
        }
    }
}

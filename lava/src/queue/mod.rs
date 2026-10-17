//! Queues: the kinds of queue family lava submits to (graphics, and with `async-compute` and `async-transfer` dedicated compute and transfer families) and the device's queues of each
use std::sync::{Mutex, MutexGuard};

pub(crate) mod device;

/// The most timelines that can exist at once, and the most queues lava creates per family.
pub const MAX_TIMELINES: usize = 8;

/// A kind of queue family. The values are what an image's tracked owner holds (0: none).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Kind {
    /// Graphics, compute and transfer: every device has it, and it presents.
    Graphics = 1,
    /// Compute and transfer without graphics (`async-compute`).
    Compute = 2,
    /// Transfer alone, a copy engine (`async-transfer`).
    Transfer = 3,
}

impl Kind {
    pub(crate) const ALL: [Kind; 3] = [Kind::Graphics, Kind::Compute, Kind::Transfer];

    /// Its slot in per-kind arrays.
    pub(crate) fn slot(self) -> usize {
        self as usize - 1
    }
}

/// Locks `mutex`, also after a panic while it was held: what lava keeps in these mutexes stays
/// usable (pending accesses can only be more than what runs, which costs barriers at most).
pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poison| poison.into_inner())
}

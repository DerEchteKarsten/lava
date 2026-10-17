//! Queues and timelines: the kinds of queue family lava submits to (graphics, and with `async-compute`/`async-transfer` dedicated compute and transfer families), the timelines whose submissions run on them in order (`Timeline`, `Reservation`, `Submission`), and the `Wait`s that order work across timelines
use std::sync::{Mutex, MutexGuard};

use crate::{
    error::Result,
    features::MissingFeature,
    sealed::Sealed,
    vk,
};

pub(crate) mod device;
mod submit;
pub(crate) mod timeline;

pub use timeline::{Reservation, Submission, Timeline, TimelineBuilder};

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
    /// Its slot in per-kind arrays.
    pub(crate) fn slot(self) -> usize {
        self as usize - 1
    }
}

/// A kind of queue family as a type: what a [`Timeline`] submits to and a
/// `CommandBuffer` records for. Only lava implements it.
pub trait QueueKind: Sealed + Send + Sync + 'static {
    const KIND: Kind;
    /// `Ok` where lava has a family of this kind (`async-compute`, `async-transfer`).
    fn check() -> Result<(), MissingFeature>;
}

/// The kinds that dispatch compute, trace rays, build acceleration structures and clear images:
/// [`Graphics`] and [`Compute`].
pub trait HasCompute: QueueKind {}

/// The kind that renders, blits and presents: [`Graphics`].
pub trait HasGraphics: HasCompute {}

/// The graphics family, which every device has.
#[derive(Debug)]
pub enum Graphics {}
impl Sealed for Graphics {}
impl QueueKind for Graphics {
    const KIND: Kind = Kind::Graphics;
    fn check() -> Result<(), MissingFeature> {
        Ok(())
    }
}
impl HasCompute for Graphics {}
impl HasGraphics for Graphics {}

/// A dedicated compute family (`async-compute`): work that runs beside the graphics queue's.
#[cfg(feature = "async-compute")]
#[derive(Debug)]
pub enum Compute {}
#[cfg(feature = "async-compute")]
impl Sealed for Compute {}
#[cfg(feature = "async-compute")]
impl QueueKind for Compute {
    const KIND: Kind = Kind::Compute;
    fn check() -> Result<(), MissingFeature> {
        use crate::features::Feature;
        crate::features::gate::AsyncCompute::check()
    }
}
#[cfg(feature = "async-compute")]
impl HasCompute for Compute {}

/// A dedicated transfer family (`async-transfer`), a copy engine: copies, fills and updates.
#[cfg(feature = "async-transfer")]
#[derive(Debug)]
pub enum Transfer {}
#[cfg(feature = "async-transfer")]
impl Sealed for Transfer {}
#[cfg(feature = "async-transfer")]
impl QueueKind for Transfer {
    const KIND: Kind = Kind::Transfer;
    fn check() -> Result<(), MissingFeature> {
        use crate::features::Feature;
        crate::features::gate::AsyncTransfer::check()
    }
}

/// Where a submission waits for another timeline's work ([`Wait::at`]): at every command (the
/// default), or only at one kind of lava command, so that the others run before the wait. The
/// stages before it in the submission then run before the awaited work is done, so it must name
/// the kind of the first command that touches what the wait guards.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Stage {
    #[default]
    All,
    /// `compute` dispatches, direct and indirect.
    Compute,
    /// `raytrace` launches, direct and indirect.
    RayTracing,
    /// `raster` renderings: their draws, attachments, indirect arguments and index buffers.
    Raster,
    /// Copies, blits and clears.
    Transfer,
}

impl Stage {
    pub(crate) fn flags(self) -> vk::PipelineStageFlags2 {
        use vk::PipelineStageFlags2 as F;
        match self {
            Stage::All => F::ALL_COMMANDS,
            Stage::Compute => F::COMPUTE_SHADER | F::DRAW_INDIRECT,
            Stage::RayTracing => F::RAY_TRACING_SHADER_KHR | F::DRAW_INDIRECT,
            Stage::Raster => F::ALL_GRAPHICS,
            Stage::Transfer => F::ALL_TRANSFER,
        }
    }
}

/// The completion of one submission (`Submission::submit` returns it): another timeline's
/// submission waits for it with `Submission::wait_for`, at [`Stage::All`] unless
/// [`Self::at`] names a later one, and the host with [`Self::block`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Wait {
    pub(crate) timeline: u8,
    pub(crate) value: u64,
    stage: Stage,
}

impl Wait {
    pub(crate) fn new(timeline: usize, value: u64) -> Self {
        Self {
            timeline: timeline as u8,
            value,
            stage: Stage::All,
        }
    }

    /// Waited for only at `stage` (see [`Stage`]).
    pub fn at(self, stage: Stage) -> Self {
        Self { stage, ..self }
    }

    pub fn stage(&self) -> Stage {
        self.stage
    }

    /// Whether the GPU finished the submission. Polls the timeline, never waits.
    pub fn is_done(&self) -> bool {
        timeline::poll(self.timeline as usize) >= self.value
    }

    /// Waits until the GPU finished the submission, then destroys what that frees
    /// (`lifetime::reclaim`).
    pub fn block(&self) -> Result<()> {
        timeline::wait_value(self.timeline as usize, self.value)?;
        crate::lifetime::reclaim();
        Ok(())
    }
}

/// Locks `mutex`, also after a panic while it was held: what lava keeps in these mutexes stays
/// usable (pending accesses can only be more than what runs, which costs barriers at most).
pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poison| poison.into_inner())
}

//! Queues: the kinds of queue family lava submits to (graphics, and with `async-compute`/`async-transfer` dedicated compute and transfer families), the device's queues of each, and the `Wait`s that order work across timelines
use std::sync::{Mutex, MutexGuard};

use crate::{error::Result, features::MissingFeature, sealed::Sealed, vk};

pub(crate) mod timeline;

pub use timeline::{Submission, Timeline};

/// The most timelines that can exist at once, and the most queues lava creates per family.
pub const MAX_TIMELINES: usize = 8;

/// A kind of queue family. The values are what an image's tracked owner holds (0: none).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Kind {
    Graphics = 1,
    Compute = 2,
    Transfer = 3,
}

impl Kind {
    pub(crate) fn slot(self) -> usize {
        self as usize - 1
    }
}

/// A kind of queue family as a type, for `Timeline` and `CommandBuffer`. Only lava implements it.
pub trait QueueKind: Sealed + Send + Sync + 'static {
    const KIND: Kind;
    /// `Ok` where lava has a family of this kind.
    fn check() -> Result<(), MissingFeature>;
}

/// [`Graphics`] and [`Compute`]: dispatches, traces, acceleration structures, image clears.
pub trait HasCompute: QueueKind {}

/// [`Graphics`]: renderings, blits, presents.
pub trait HasGraphics: HasCompute {}

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

/// A dedicated compute family (`async-compute`).
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

/// A dedicated transfer family (`async-transfer`): copies, fills and updates.
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

/// The kind of command a [`Wait`] holds back: everything (the default), or only that kind, so
/// the commands before it run before the awaited work is done.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Stage {
    #[default]
    All,
    Compute,
    RayTracing,
    Raster,
    Transfer,
}

/// The completion of a submission (`Submission::submit`): another timeline's submission waits for
/// it with `Submission::wait_for`, the host with [`Self::block`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Wait {
    pub(crate) timeline: u8,
    pub(crate) value: u64,
    pub(crate) stage: Stage,
}

impl Wait {
    pub fn at(self, stage: Stage) -> Self {
        Self { stage, ..self }
    }

    pub fn is_done(&self) -> bool {
        timeline::poll(self.timeline as usize) >= self.value
    }

    /// Waits until the GPU finished the submission, then destroys what that frees.
    pub fn block(&self) -> Result<()> {
        timeline::wait_value(self.timeline as usize, self.value)?;
        crate::lifetime::reclaim();
        Ok(())
    }
}

/// Locks `mutex`, also after a panic while it was held.
pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poison| poison.into_inner())
}

#[derive(Debug)]
pub(crate) struct FamilyQueues {
    pub(crate) kind: Kind,
    pub(crate) index: u32,
    pub(crate) timestamp_valid_bits: u32,
    pub(crate) image_granularity: vk::Extent3D,
    pub(crate) queues: Box<[Mutex<vk::Queue>]>,
    /// How many timelines are pinned to each queue.
    pub(crate) timelines: Mutex<Box<[u32]>>,
}

#[derive(Debug)]
pub(crate) struct DeviceQueues(pub(crate) [Option<FamilyQueues>; 3]);

impl DeviceQueues {
    pub(crate) fn family(&self, kind: Kind) -> Option<&FamilyQueues> {
        self.0[kind.slot()].as_ref()
    }

    pub(crate) fn graphics(&self) -> &FamilyQueues {
        self.family(Kind::Graphics).expect("every device has one")
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &FamilyQueues> {
        self.0.iter().flatten()
    }
}

/// What a recording knows of its family, and every kind's family index
/// (`QUEUE_FAMILY_IGNORED` where lava has none).
#[derive(Clone, Copy, Debug)]
pub(crate) struct FamilyInfo {
    pub(crate) kind: Kind,
    pub(crate) timestamp_valid_bits: u32,
    pub(crate) image_granularity: vk::Extent3D,
    pub(crate) indices: [u32; 3],
}

impl FamilyInfo {
    #[track_caller]
    pub(crate) fn index(&self, kind: Kind) -> u32 {
        let index = self.indices[kind.slot()];
        assert_ne!(index, vk::QUEUE_FAMILY_IGNORED, "lava has no {kind:?} family");
        index
    }
}

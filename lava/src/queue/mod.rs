//! Queues: the kinds of queue family lava submits to (graphics, and with `async-compute`/`async-transfer` dedicated compute and transfer families), the device's queues of each, the stages each family's submissions can wait at, and the `Wait`s that order work across timelines and block the host
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
    /// The stages its submissions can wait at.
    type Stage: StageSet + StageOf<Self>;
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
    type Stage = GraphicsStage;
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
    type Stage = ComputeStage;
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
    type Stage = TransferStage;
    fn check() -> Result<(), MissingFeature> {
        use crate::features::Feature;
        crate::features::gate::AsyncTransfer::check()
    }
}

/// The stages a submission of one family can wait at ([`Submission::wait_for`]): each kind of
/// queue family has its set ([`QueueKind::Stage`]), so a stage the waiting family doesn't have
/// doesn't compile. Only lava implements it.
pub trait StageSet: Sealed + Copy + std::fmt::Debug + Send + Sync + 'static {
    /// Its stage mask on a device with `features` (ray tracing stages only with ray tracing
    /// pipelines).
    #[doc(hidden)]
    fn mask(self, features: &crate::Features) -> vk::PipelineStageFlags2;
}

/// Every stage: what [`Submission::submit`] returns, which a submission of any family can wait
/// for (holding back all its commands).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct AllStages;

impl Sealed for AllStages {}
impl StageSet for AllStages {
    fn mask(self, _: &crate::Features) -> vk::PipelineStageFlags2 {
        vk::PipelineStageFlags2::ALL_COMMANDS
    }
}

/// The ray tracing shader stage where the device has ray tracing pipelines, with the indirect
/// reads before it.
fn ray_tracing_mask(features: &crate::Features) -> vk::PipelineStageFlags2 {
    use vk::PipelineStageFlags2 as F;
    let _ = features;
    #[cfg(feature = "ray-tracing-pipeline")]
    if features.ray_tracing_pipeline {
        return F::RAY_TRACING_SHADER_KHR | F::DRAW_INDIRECT;
    }
    F::DRAW_INDIRECT
}

/// What a graphics submission holds back until the awaited work is done: everything, or only
/// one kind of command, so the commands before it run meanwhile.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum GraphicsStage {
    #[default]
    All,
    Compute,
    /// Ray tracing shaders (only where the device has ray tracing pipelines) and indirect reads.
    RayTracing,
    Raster,
    Transfer,
}

impl Sealed for GraphicsStage {}
impl StageSet for GraphicsStage {
    fn mask(self, features: &crate::Features) -> vk::PipelineStageFlags2 {
        use vk::PipelineStageFlags2 as F;
        match self {
            GraphicsStage::All => F::ALL_COMMANDS,
            GraphicsStage::Compute => F::COMPUTE_SHADER | F::DRAW_INDIRECT,
            GraphicsStage::RayTracing => ray_tracing_mask(features),
            GraphicsStage::Raster => F::ALL_GRAPHICS,
            GraphicsStage::Transfer => F::ALL_TRANSFER,
        }
    }
}

/// What a compute submission holds back ([`GraphicsStage`] without rendering).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ComputeStage {
    #[default]
    All,
    Compute,
    /// Ray tracing shaders (only where the device has ray tracing pipelines) and indirect reads.
    RayTracing,
    Transfer,
}

impl Sealed for ComputeStage {}
impl StageSet for ComputeStage {
    fn mask(self, features: &crate::Features) -> vk::PipelineStageFlags2 {
        use vk::PipelineStageFlags2 as F;
        match self {
            ComputeStage::All => F::ALL_COMMANDS,
            ComputeStage::Compute => F::COMPUTE_SHADER | F::DRAW_INDIRECT,
            ComputeStage::RayTracing => ray_tracing_mask(features),
            ComputeStage::Transfer => F::ALL_TRANSFER,
        }
    }
}

/// What a transfer submission holds back: it has only transfers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TransferStage {
    #[default]
    All,
    Transfer,
}

impl Sealed for TransferStage {}
impl StageSet for TransferStage {
    fn mask(self, _: &crate::Features) -> vk::PipelineStageFlags2 {
        use vk::PipelineStageFlags2 as F;
        match self {
            TransferStage::All => F::ALL_COMMANDS,
            TransferStage::Transfer => F::ALL_TRANSFER,
        }
    }
}

/// The completion of a submission (`Submission::submit`): another timeline's submission waits for
/// it with `Submission::wait_for`, the host with [`Self::block`]. `S` is the stage it holds the
/// waiting submission back at: [`AllStages`] as `submit` returns it, any submission's, or one
/// family's stage set after [`Self::at`]:
/// ```
/// use lava::queue::{Graphics, GraphicsStage, Submission, TransferStage, Wait};
/// fn wait(submission: &mut Submission<'_, Graphics>, uploaded: Wait) {
///     submission.wait_for(uploaded.at(GraphicsStage::Raster));
///     submission.wait_for(uploaded.at(TransferStage::Transfer)).wait_for(uploaded);
/// }
/// ```
/// A stage the waiting family doesn't have doesn't compile:
/// ```compile_fail
/// use lava::queue::{Compute, GraphicsStage, Submission, Wait};
/// fn wait(submission: &mut Submission<'_, Compute>, drawn: Wait) {
///     submission.wait_for(drawn.at(GraphicsStage::Raster));
/// }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Wait<S: StageSet = AllStages> {
    pub(crate) timeline: u8,
    pub(crate) value: u64,
    pub(crate) stage: S,
}

/// The stage sets a submission of family `K` can wait at: its own and [`AllStages`]. Transfer
/// stages are in every family's set and compute stages in the graphics family's.
pub trait StageOf<K: QueueKind + ?Sized>: StageSet {}
impl<K: QueueKind + ?Sized> StageOf<K> for AllStages {}
impl StageOf<Graphics> for GraphicsStage {}
impl<K: QueueKind + ?Sized> StageOf<K> for TransferStage {}
impl StageOf<Graphics> for ComputeStage {}
#[cfg(feature = "async-compute")]
impl StageOf<Compute> for ComputeStage {}

impl<S: StageSet> Wait<S> {
    /// The same completion, holding the waiting submission back only at `stage`.
    pub fn at<T: StageSet>(self, stage: T) -> Wait<T> {
        Wait {
            timeline: self.timeline,
            value: self.value,
            stage,
        }
    }

    /// Whether the GPU finished the submission; false once the device is lost.
    pub fn is_done(&self) -> bool {
        let _call = crate::state::call();
        timeline::poll(self.timeline as usize).is_ok_and(|done| done >= self.value)
    }

    /// Waits until the GPU finished the submission, then destroys what that frees.
    /// `Error::DeviceLost` once the device is lost.
    pub fn block(&self) -> Result<()> {
        self.block_timeout(std::time::Duration::MAX).map(|_| ())
    }

    /// Waits up to `timeout` for the GPU to finish the submission and returns whether it did,
    /// then destroys what that frees.
    pub fn block_timeout(&self, timeout: std::time::Duration) -> Result<bool> {
        Self::all_timeout(std::slice::from_ref(self), timeout)
    }

    /// Waits until the GPU finished every one of `waits` (one `vkWaitSemaphores`), then
    /// destroys what that frees.
    pub fn all(waits: &[Self]) -> Result<()> {
        Self::all_timeout(waits, std::time::Duration::MAX).map(|_| ())
    }

    /// Waits up to `timeout` for the GPU to finish every one of `waits` and returns whether it
    /// did, then destroys what that frees.
    pub fn all_timeout(waits: &[Self], timeout: std::time::Duration) -> Result<bool> {
        let _call = crate::state::call();
        let mut values = [0u64; MAX_TIMELINES];
        for wait in waits {
            let value = &mut values[wait.timeline as usize];
            *value = (*value).max(wait.value);
        }
        let done = timeline::wait_values(&values, timeout)?;
        crate::lifetime::run_ready();
        Ok(done)
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

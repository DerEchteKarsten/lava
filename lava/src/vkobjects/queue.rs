//! Queue submission and sync: queue leases, semaphores, fences, frame slots, presenting
use ash::vk::TaggedStructure;
use std::{
    fmt::Debug,
    marker::PhantomData,
    mem::ManuallyDrop,
    sync::{Arc, Mutex},
    time::Instant,
};

use crate::error::{Error, Result};
use ash::vk;
use lava_macros::validation_trace;
use smallvec::SmallVec;

#[cfg(feature = "hot-reload")]
use crate::pipeline::GenericPipeline;
use crate::{
    command_buffer::{BufferAccess, CommandBuffer, ImageAccess},
    keepalive::KeepAlive,
    profiling::{Counter, GpuScope, PassCounters, PassQueries, TimestampQueries},
    state::{CALLSITE, Ctx, Functions},
    vkobjects::swapchain::Swapchain,
};

/// Dropping it destroys it once no frame in flight that waited on or signalled it counts it
/// any more (`keepalive`).
#[derive(Debug)]
pub struct Semaphore<T: SemaphoreType + ?Sized> {
    pub(crate) handle: vk::Semaphore,
    owner: ManuallyDrop<KeepAlive>,
    marker: PhantomData<T>,
}

struct RawSemaphore(vk::Semaphore);
impl Drop for RawSemaphore {
    fn drop(&mut self) {
        unsafe { Ctx::device().destroy_semaphore(self.0, None) };
    }
}

impl<T: SemaphoreType + ?Sized> Semaphore<T> {
    fn from_handle(handle: vk::Semaphore) -> Self {
        Semaphore {
            handle,
            owner: ManuallyDrop::new(KeepAlive::new()),
            marker: PhantomData,
        }
    }
}

impl<T: SemaphoreType> Default for Semaphore<T> {
    fn default() -> Self {
        Self::new().expect("failed to create default semaphore")
    }
}

impl<T: SemaphoreType + ?Sized> Drop for Semaphore<T> {
    fn drop(&mut self) {
        let owner = unsafe { ManuallyDrop::take(&mut self.owner) };
        owner.release(RawSemaphore(self.handle));
    }
}

/// Where a submission waits for a semaphore: at every command (the default), or only at
/// one kind of lava command, so that the others run before the wait. A hint the caller sets
/// with `SemaphoreInfo::at`; the stages before it in the submission then run before the
/// semaphore is signalled, so it must name the kind of the first command that touches what
/// the semaphore guards (a transition out of `PRESENT_SRC_KHR` waits at `ALL_COMMANDS`,
/// which chains with any of them).
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

/// A semaphore to wait on or signal in a submission (`Semaphore::info`). A wait holds back
/// every command unless `at` names the kind of command it should hold back instead.
/// Signals always happen after everything. The frame counts the semaphore until the GPU
/// finished it.
#[derive(Clone, Debug)]
pub struct SemaphoreInfo {
    handle: vk::Semaphore,
    /// `None` for a binary semaphore.
    value: Option<u64>,
    stage: Stage,
    keep: KeepAlive,
}
impl SemaphoreInfo {
    /// Waited on only at `stage` (see `Stage`); the default is `Stage::All`.
    pub fn at(self, stage: Stage) -> Self {
        Self { stage, ..self }
    }

    pub(crate) fn to_vk(&self, stage: vk::PipelineStageFlags2) -> vk::SemaphoreSubmitInfo<'_> {
        vk::SemaphoreSubmitInfo::default()
            .stage_mask(stage)
            .semaphore(self.handle)
            .value(self.value.unwrap_or(0))
    }
}

pub trait SemaphoreType {
    fn create() -> Result<Semaphore<Self>>;
}

#[derive(Debug)]
pub struct Timeline;
impl SemaphoreType for Timeline {
    fn create() -> Result<Semaphore<Self>> {
        let mut timeline = vk::SemaphoreTypeCreateInfo {
            initial_value: 0,
            semaphore_type: vk::SemaphoreType::TIMELINE,
            ..Default::default()
        };
        let create_info = vk::SemaphoreCreateInfo::default().push(&mut timeline);
        let handle = unsafe { Ctx::device().create_semaphore(&create_info, None)? };
        Ok(Semaphore::from_handle(handle))
    }
}

#[derive(Debug)]
pub struct Binary;
impl SemaphoreType for Binary {
    fn create() -> Result<Semaphore<Self>> {
        let create_info = vk::SemaphoreCreateInfo::default();
        let handle = unsafe { Ctx::device().create_semaphore(&create_info, None)? };
        Ok(Semaphore::from_handle(handle))
    }
}

impl<T: SemaphoreType> Semaphore<T> {
    #[validation_trace]
    pub fn new() -> Result<Self> {
        T::create()
    }
}

impl Semaphore<Timeline> {
    #[validation_trace]
    pub fn block_until_value(&self, value: u64) -> Result<()> {
        let binding = [self.handle];
        let values = [value];
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&binding)
            .values(&values);
        unsafe { Ctx::device().wait_semaphores(&wait_info, u64::MAX)? };
        Ok(())
    }
    #[validation_trace]
    pub fn info(&self, value: u64) -> SemaphoreInfo {
        SemaphoreInfo {
            handle: self.handle,
            value: Some(value),
            stage: Stage::All,
            keep: KeepAlive::clone(&self.owner),
        }
    }
}

impl Semaphore<Binary> {
    #[validation_trace]
    pub fn info(&self) -> SemaphoreInfo {
        SemaphoreInfo {
            handle: self.handle,
            value: None,
            stage: Stage::All,
            keep: KeepAlive::clone(&self.owner),
        }
    }
}

pub struct Fence {
    pub(crate) handle: vk::Fence,
}
impl Drop for Fence {
    fn drop(&mut self) {
        unsafe { Ctx::device().destroy_fence(self.handle, None) };
    }
}

impl Default for Fence {
    fn default() -> Self {
        Self::new().expect("failed to create default fence")
    }
}

impl Fence {
    #[validation_trace]
    pub fn new() -> Result<Self> {
        let create_info = vk::FenceCreateInfo::default();
        let handle = unsafe { Ctx::device().create_fence(&create_info, None)? };
        Ok(Self { handle })
    }
    #[validation_trace]
    pub fn reset(&self) -> Result<()> {
        unsafe { Ctx::device().reset_fences(&[self.handle])? };
        Ok(())
    }
    #[validation_trace]
    pub fn wait(&self) -> Result<()> {
        unsafe { Ctx::device().wait_for_fences(&[self.handle], true, u64::MAX)? };
        Ok(())
    }
}

/// A queue family lava submits to. Present always goes through [`Gfx`].
pub trait QueueFamilie: Debug {
    fn queues() -> &'static FamilyQueues;
}

#[derive(Debug)]
pub struct Transfer;
#[derive(Debug)]
pub struct Gfx;

impl QueueFamilie for Transfer {
    fn queues() -> &'static FamilyQueues {
        let ctx = Ctx::get();
        ctx.transfer_queues.as_ref().unwrap_or(&ctx.gfx_queues)
    }
}
impl QueueFamilie for Gfx {
    fn queues() -> &'static FamilyQueues {
        &Ctx::get().gfx_queues
    }
}

/// Every queue of one family, handed out as [`Queue`] leases.
#[derive(Debug)]
pub struct FamilyQueues {
    pub(crate) family: u32,
    /// Vulkan wants submissions to one queue externally synchronised.
    handles: Box<[Mutex<vk::Queue>]>,
    /// The leases of each queue.
    leases: Mutex<Box<[u32]>>,
    /// Per queue: the swapchains and semaphores of the presents queued since its last
    /// submission. A frame's fence doesn't cover a present queued after it, so the next
    /// submission to the queue keeps them alive (`keepalive`) instead.
    presented: Box<[Mutex<Vec<KeepAlive>>]>,
}

impl FamilyQueues {
    pub(crate) fn new(device: &ash::Device, family: u32, count: u32) -> Self {
        Self {
            family,
            handles: (0..count)
                .map(|i| Mutex::new(unsafe { device.get_device_queue(family, i) }))
                .collect(),
            leases: Mutex::new(vec![0; count as usize].into()),
            presented: (0..count).map(|_| Mutex::new(Vec::new())).collect(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.handles.len()
    }

    /// Releases what the presents of every queue kept: only once the device is idle.
    pub(crate) fn release_presented(&self) {
        for presented in &self.presented {
            presented.lock().unwrap().clear();
        }
    }
}

/// A lease on one queue of family `Q`: one no other lease holds while there is one, then the
/// one the fewest leases share, which they take turns submitting to. Whoever submits a frame
/// loop's [`FrameSlot`]s through one lease gets them executed in order.
#[derive(Debug)]
pub struct Queue<Q: QueueFamilie> {
    family: &'static FamilyQueues,
    index: usize,
    _family: PhantomData<Q>,
}

impl<Q: QueueFamilie> Queue<Q> {
    pub fn new() -> Self {
        let family = Q::queues();
        let index = lease(&mut family.leases.lock().unwrap());
        Self {
            family,
            index,
            _family: PhantomData,
        }
    }

    /// Whether the queue is shared while a less used one of the family exists. Leases taken
    /// while every queue was taken keep sharing after the others were dropped; this finds out,
    /// [`Self::reselect`] fixes it.
    pub fn is_optimal(&self) -> bool {
        is_optimal(&self.family.leases.lock().unwrap(), self.index)
    }

    /// Moves the lease to the least used queue of its family. It waits until the old queue is
    /// idle first, so the submissions through the lease stay in order and its `FrameSlot`s and
    /// `PendingAccesses` stay valid. Meant for the rare case [`Self::is_optimal`] reports.
    #[validation_trace]
    pub fn reselect(&mut self) -> Result<()> {
        if self.is_optimal() {
            return Ok(());
        }
        self.with(|queue| unsafe { Ctx::device().queue_wait_idle(queue) })?;
        // Their presents are done too.
        self.family.presented[self.index].lock().unwrap().clear();
        let mut leases = self.family.leases.lock().unwrap();
        leases[self.index] -= 1;
        self.index = lease(&mut leases);
        Ok(())
    }

    /// Calls `f` with the queue, locked.
    fn with<R>(&self, f: impl FnOnce(vk::Queue) -> R) -> R {
        f(*self.family.handles[self.index].lock().unwrap())
    }
}

impl<Q: QueueFamilie> Default for Queue<Q> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Q: QueueFamilie> Drop for Queue<Q> {
    fn drop(&mut self) {
        self.family.leases.lock().unwrap()[self.index] -= 1;
    }
}

/// Leases the queue the fewest leases share.
fn lease(leases: &mut [u32]) -> usize {
    let (index, _) = leases
        .iter()
        .enumerate()
        .min_by_key(|&(_, count)| count)
        .expect("a queue family has at least one queue");
    leases[index] += 1;
    index
}

/// Whether moving a lease off queue `index` would leave fewer leases sharing one.
fn is_optimal(leases: &[u32], index: usize) -> bool {
    let least = leases.iter().min().copied().unwrap_or(0);
    leases[index] <= least + 1
}

#[derive(Debug)]
pub(crate) struct CommandPool {
    handle: vk::CommandPool,
}
impl CommandPool {
    #[validation_trace]
    pub(crate) fn new(family: u32) -> Result<Self> {
        let info = vk::CommandPoolCreateInfo::default().queue_family_index(family);
        Ok(Self {
            handle: unsafe { Ctx::device().create_command_pool(&info, None)? },
        })
    }
    #[validation_trace]
    pub(crate) fn reset(&self) -> Result<()> {
        unsafe {
            Ctx::device().reset_command_pool(self.handle, vk::CommandPoolResetFlags::empty())?
        };
        Ok(())
    }
    #[validation_trace]
    pub(crate) fn create_command_buffer(&self) -> Result<CommandBufferMemory> {
        let allocate_info = vk::CommandBufferAllocateInfo {
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            command_pool: self.handle,
            ..Default::default()
        };
        let handle = unsafe { Ctx::device().allocate_command_buffers(&allocate_info)? }[0];
        Ok(CommandBufferMemory {
            pool: self.handle,
            handle,
        })
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        unsafe { Ctx::device().destroy_command_pool(self.handle, None) };
    }
}

#[derive(Debug)]
pub(crate) struct CommandBufferMemory {
    pub(crate) pool: vk::CommandPool,
    pub(crate) handle: vk::CommandBuffer,
}

impl Drop for CommandBufferMemory {
    fn drop(&mut self) {
        unsafe {
            Ctx::device().free_command_buffers(self.pool, &[self.handle]);
        }
    }
}

/// A command buffer with its pool and fence. Submit its frames through one [`Queue`] lease,
/// so they reach the GPU in order and the barriers between them hold.
pub struct FrameSlot {
    /// The counts of everything the last submission used (`keepalive`), dropped once its
    /// fence signalled; and what `Frame::uses` added for the next.
    in_flight: Vec<KeepAlive>,
    /// The pipelines the last frame recorded, which a shader reload may have replaced since.
    #[cfg(feature = "hot-reload")]
    pipelines: Vec<Arc<GenericPipeline>>,
    buffer: CommandBufferMemory,
    pool: CommandPool,
    fence: Fence,
    submitted: bool,
    family: u32,
    timestamps_on: bool,
    timestamps: Option<TimestampQueries>,
    counter_selection: Option<Arc<[&'static Counter]>>,
    counters: Option<PassQueries>,
}

impl FrameSlot {
    #[validation_trace]
    pub fn new<Q: QueueFamilie>(queue: &Queue<Q>) -> Result<Self> {
        let pool = CommandPool::new(queue.family.family)?;
        let buffer = pool.create_command_buffer()?;
        Ok(Self {
            in_flight: Vec::new(),
            #[cfg(feature = "hot-reload")]
            pipelines: Vec::new(),
            buffer,
            pool,
            fence: Fence::new()?,
            submitted: false,
            family: queue.family.family,
            timestamps_on: false,
            timestamps: None,
            counter_selection: None,
            counters: None,
        })
    }

    /// Times the scopes of the frames begun from now on; see [`Frame::last_timestamps`]. Does
    /// nothing without the `profiling` feature.
    pub fn set_timestamps(&mut self, on: bool) {
        self.timestamps_on = on && cfg!(feature = "profiling");
    }

    /// Counts every pass and rendering of the frames begun from now on, with pipeline
    /// statistics and the hardware counters of `selection` (`None`: no pass queries); see
    /// [`Frame::last_counters`]. A selection of more than `MAX_COUNTERS` or more than one pass
    /// fails the next `begin`. Does nothing without the `profiling` feature.
    pub fn set_counters(&mut self, selection: Option<Arc<[&'static Counter]>>) {
        self.counter_selection = selection.filter(|_| cfg!(feature = "profiling"));
    }

    #[validation_trace]
    pub fn begin(&mut self) -> Result<Frame<'_>> {
        if self.submitted {
            tracing::info_span!("wait for frame slot").in_scope(|| self.fence.wait())?;
            if let Some(timestamps) = &mut self.timestamps {
                timestamps.read()?;
            }
            if let Some(counters) = &mut self.counters {
                counters.read()?;
            }
            self.fence.reset()?;
            self.pool.reset()?;
            self.submitted = false;
        }
        // Also after a frame that failed before it was submitted.
        self.in_flight.clear();
        #[cfg(feature = "hot-reload")]
        self.pipelines.clear();
        // The pools are idle now. A new counter selection needs new ones.
        if !self.timestamps_on {
            self.timestamps = None;
        } else if self.timestamps.is_none() {
            self.timestamps = Some(TimestampQueries::new(self.family)?);
        }
        let selection = &self.counter_selection;
        let current = |counters: &PassQueries| {
            selection
                .as_ref()
                .is_some_and(|s| Arc::ptr_eq(s, counters.selection()))
        };
        if !self.counters.as_ref().is_some_and(current) {
            self.counters = None;
            if let Some(selection) = selection {
                self.counters = Some(PassQueries::new(self.family, selection.clone())?);
            }
        }
        Ok(Frame { slot: self })
    }
}

pub struct Frame<'a> {
    slot: &'a mut FrameSlot,
}

impl Frame<'_> {
    /// Keeps what the frame reaches without its commands naming it alive until the GPU
    /// finished the frame, e.g. a buffer reached through an address stored in another buffer
    /// (`Buffer::keep_alive`). Whatever the commands take (views, slices, bindings) is kept
    /// already.
    pub fn uses(&mut self, keep: impl IntoIterator<Item = KeepAlive>) {
        self.slot.in_flight.extend(keep);
    }

    /// The GPU scopes of the frame this slot submitted before, if it was timed.
    pub fn last_timestamps(&self) -> Option<&[GpuScope]> {
        self.slot.timestamps.as_ref()?.results()
    }

    /// When the root scope of `last_timestamps` started, on the host's clock: from
    /// `VK_KHR_calibrated_timestamps` (`Features::calibrated_timestamps`), `None` without it.
    pub fn last_timestamps_start(&self) -> Option<Instant> {
        self.slot.timestamps.as_ref()?.root_start()
    }

    /// The pass queries of the frame this slot submitted before, if it was counted.
    pub fn last_counters(&self) -> Option<&PassCounters> {
        self.slot.counters.as_ref()?.results()
    }

    #[validation_trace]
    pub fn execute<Q: QueueFamilie, F: FnOnce(&mut CommandBuffer)>(
        self,
        queue: &Queue<Q>,
        pending: PendingAccesses,
        wait_on: &[SemaphoreInfo],
        signal: &[SemaphoreInfo],
        executor: F,
    ) -> Result<PendingAccesses> {
        let slot = self.slot;
        let result = execute_command(queue, pending, slot, wait_on, signal, executor);
        match result {
            Ok(_) => slot.submitted = true,
            Err(_) => slot.pool.reset()?,
        }
        result
    }
}

impl Drop for FrameSlot {
    fn drop(&mut self) {
        if self.submitted
            && let Err(err) = self.fence.wait()
        {
            tracing::error!(%err, "failed to wait for frame slot fence");
        }
    }
}

#[derive(Debug, Default)]
pub struct PendingAccesses {
    pub(crate) buffer_reads: SmallVec<[BufferAccess; 8]>,
    pub(crate) image_reads: SmallVec<[ImageAccess; 8]>,
    pub(crate) buffer_writes: SmallVec<[PendingWrite<BufferAccess>; 8]>,
    pub(crate) image_writes: SmallVec<[PendingWrite<ImageAccess>; 8]>,
}

/// A write later commands may have to wait for, and the stages and accesses a barrier recorded
/// since already made it visible to: those need no barrier for it anymore.
#[derive(Debug, Clone)]
pub(crate) struct PendingWrite<A> {
    pub(crate) write: A,
    pub(crate) visible_stages: vk::PipelineStageFlags2,
    pub(crate) visible_access: vk::AccessFlags2,
}

impl<A> PendingWrite<A> {
    pub(crate) fn new(write: A) -> Self {
        Self {
            write,
            visible_stages: vk::PipelineStageFlags2::NONE,
            visible_access: vk::AccessFlags2::NONE,
        }
    }

    pub(crate) fn visible_to(
        &self,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
    ) -> bool {
        self.visible_stages.contains(stage) && self.visible_access.contains(access)
    }
}

/// Records `executor` into `buffer` and submits it. With `timestamps`, the recording is
/// timed: the root scope "frame" around it, plus every pass and `begin_scope`. With
/// `counters`, every pass is counted.
#[validation_trace]
fn execute_command<Q: QueueFamilie, F: FnOnce(&mut CommandBuffer)>(
    queue: &Queue<Q>,
    pending: PendingAccesses,
    slot: &mut FrameSlot,
    wait_on: &[SemaphoreInfo],
    signal: &[SemaphoreInfo],
    executor: F,
) -> Result<PendingAccesses> {
    let buffer = &slot.buffer;
    let mut pending = pending;
    pending.prune();
    unsafe {
        let mut cmd_buffer = CommandBuffer {
            handle: buffer.handle,
            pending_accesses: pending,
            used: std::mem::take(&mut slot.in_flight),
            timestamps: slot.timestamps.take(),
            counters: slot.counters.take(),
            #[cfg(feature = "hot-reload")]
            pipelines: std::mem::take(&mut slot.pipelines),
        };

        cmd_buffer.begin()?;
        if let Some(timestamps) = &mut cmd_buffer.timestamps {
            timestamps.reset(buffer.handle);
        }
        if let Some(counters) = &mut cmd_buffer.counters {
            counters.reset(buffer.handle);
        }
        cmd_buffer.begin_scope("frame");
        let prev = CALLSITE.replace(None);
        executor(&mut cmd_buffer);
        CALLSITE.set(prev);
        cmd_buffer.end_scope();
        host_barrier(buffer.handle);
        slot.timestamps = cmd_buffer.timestamps.take();
        slot.counters = cmd_buffer.counters.take();
        #[cfg(feature = "hot-reload")]
        {
            slot.pipelines = std::mem::take(&mut cmd_buffer.pipelines);
        }
        slot.in_flight = std::mem::take(&mut cmd_buffer.used);
        let semaphores = wait_on.iter().chain(signal).map(|sem| sem.keep.clone());
        slot.in_flight.extend(semaphores);
        let presented = &queue.family.presented[queue.index];
        slot.in_flight.append(&mut presented.lock().unwrap());
        cmd_buffer.end()?;

        let cmd_buffer_submit_info =
            vk::CommandBufferSubmitInfo::default().command_buffer(buffer.handle);
        let wait_infos: SmallVec<[_; 1]> = wait_on
            .iter()
            .map(|sem| sem.to_vk(sem.stage.flags()))
            .collect();
        // Untracked work (queries, `flush_all`, host barriers) has to finish too.
        let signal_infos: SmallVec<[_; 1]> = signal
            .iter()
            .map(|sem| sem.to_vk(vk::PipelineStageFlags2::ALL_COMMANDS))
            .collect();

        let submit_info = vk::SubmitInfo2::default()
            .command_buffer_infos(std::slice::from_ref(&cmd_buffer_submit_info))
            .wait_semaphore_infos(&wait_infos)
            .signal_semaphore_infos(&signal_infos);

        queue.with(|handle| {
            Ctx::device().queue_submit2(
                handle,
                std::slice::from_ref(&submit_info),
                slot.fence.handle,
            )
        })?;

        Ok(cmd_buffer.pending_accesses)
    }
}

impl Queue<Gfx> {
    /// `Ok(true)` means the swapchain is suboptimal or out of date and should be recreated.
    #[validation_trace]
    pub fn present(
        &self,
        swapchain: &Swapchain,
        image_index: u32,
        wait_on: &[&Semaphore<Binary>],
    ) -> Result<bool> {
        let sc = [swapchain.handle];
        let ii = [image_index];
        let waits: SmallVec<[_; 2]> = wait_on.iter().map(|sem| sem.handle).collect();
        let _span = tracing::info_span!("present").entered();
        let present_info = vk::PresentInfoKHR::default()
            .swapchains(&sc)
            .image_indices(&ii)
            .wait_semaphores(waits.as_slice());
        let presented = self
            .with(|queue| unsafe { Functions::swapchain().queue_present(queue, &present_info) });
        let keep = wait_on.iter().map(|sem| KeepAlive::clone(&sem.owner));
        let mut kept = self.family.presented[self.index].lock().unwrap();
        kept.extend(keep.chain([swapchain.keep_alive()]));
        drop(kept);
        match presented {
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(true),
            Ok(false) => Ok(false),
            Err(e) => Err(Error::Vulkan(e)),
        }
    }
}

/// Ends every submission: makes its writes readable by the host once its fence or semaphores
/// signalled (readbacks, shader clock counters), which they alone don't.
fn host_barrier(cmd: vk::CommandBuffer) {
    let barrier = [vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::HOST)
        .dst_access_mask(vk::AccessFlags2::HOST_READ)];
    let info = vk::DependencyInfo::default().memory_barriers(&barrier);
    unsafe { Ctx::device().cmd_pipeline_barrier2(cmd, &info) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;

    #[test]
    fn leases_take_unused_queues_before_sharing_the_least_used() {
        let mut leases = [1, 0, 1];
        assert_eq!(lease(&mut leases), 1);
        assert_eq!(lease(&mut leases), 0);
        assert_eq!(lease(&mut leases), 1);
        assert_eq!(leases, [2, 2, 1]);
    }

    #[test]
    fn a_shared_queue_is_only_suboptimal_while_a_less_used_one_exists() {
        assert!(!is_optimal(&[2, 0], 0));
        assert!(is_optimal(&[2, 0], 1));
        assert!(is_optimal(&[2, 1], 0));
        assert!(is_optimal(&[2], 0), "a single queue can't do better");
        assert!(!is_optimal(&[3, 1], 0));
    }

    #[test]
    fn binary_semaphore_info_has_no_value() {
        let info = SemaphoreInfo {
            handle: vk::Semaphore::from_raw(7),
            value: None,
            stage: Stage::All,
            keep: KeepAlive::new(),
        };
        let submit = info.to_vk(vk::PipelineStageFlags2::COMPUTE_SHADER);
        assert_eq!(submit.semaphore, vk::Semaphore::from_raw(7));
        assert_eq!(submit.value, 0);
        assert_eq!(submit.stage_mask, vk::PipelineStageFlags2::COMPUTE_SHADER);
    }

    #[test]
    fn timeline_semaphore_info_carries_its_value() {
        let info = SemaphoreInfo {
            handle: vk::Semaphore::from_raw(9),
            value: Some(12),
            stage: Stage::All,
            keep: KeepAlive::new(),
        };
        let submit = info.to_vk(vk::PipelineStageFlags2::ALL_COMMANDS);
        assert_eq!(submit.semaphore, vk::Semaphore::from_raw(9));
        assert_eq!(submit.value, 12);
        assert_eq!(submit.stage_mask, vk::PipelineStageFlags2::ALL_COMMANDS);
    }
}

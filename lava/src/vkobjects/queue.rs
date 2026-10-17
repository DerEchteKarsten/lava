//! Queue submission and sync: queue leases with their pending accesses, semaphores, fences, frame slots with their epochs (a failed recording is replaced by its barriers), presenting with present fences
use std::{
    fmt::Debug,
    marker::PhantomData,
    ops::Range,
    panic::AssertUnwindSafe,
    sync::{
        Mutex,
        atomic::{AtomicU32, Ordering},
    },
};

use crate::error::{Error, Result};
use crate::vk;
use lava_macros::validation_trace;
use smallvec::SmallVec;

use crate::{
    command_buffer::{Barriers, CommandBuffer, QueueCaps, pending::PendingAccesses},
    epoch::{self, Epoch},
    queue::{
        Kind,
        device::FamilyQueues,
        lock,
    },
    state::{CALLSITE, Ctx, Functions},
    vkobjects::swapchain::Swapchain,
};

#[derive(Debug)]
pub struct Semaphore<T: SemaphoreType + ?Sized> {
    pub(crate) handle: vk::Semaphore,
    marker: PhantomData<T>,
}

impl<T: SemaphoreType> Default for Semaphore<T> {
    fn default() -> Self {
        Self::new().expect("failed to create default semaphore")
    }
}

/// Destroyed once the frames and presents that may wait on or signal it are done (`epoch`).
impl<T: SemaphoreType + ?Sized> Drop for Semaphore<T> {
    fn drop(&mut self) {
        let handle = self.handle;
        epoch::retire_after_present(move || unsafe {
            Ctx::device().destroy_semaphore(handle, None)
        });
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
/// Signals always happen after everything. It borrows the semaphore.
#[derive(Clone, Copy, Debug)]
pub struct SemaphoreInfo<'a> {
    handle: vk::Semaphore,
    /// `None` for a binary semaphore.
    value: Option<u64>,
    stage: Stage,
    _semaphore: PhantomData<&'a ()>,
}
impl SemaphoreInfo<'_> {
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

pub trait SemaphoreType: crate::sealed::Sealed {
    fn create() -> Result<Semaphore<Self>>;
}

#[derive(Debug)]
pub struct Timeline;
impl crate::sealed::Sealed for Timeline {}
impl SemaphoreType for Timeline {
    fn create() -> Result<Semaphore<Self>> {
        let mut timeline = vk::SemaphoreTypeCreateInfo {
            initial_value: 0,
            semaphore_type: vk::SemaphoreType::TIMELINE,
            ..Default::default()
        };
        let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut timeline);
        let handle = unsafe { Ctx::device().create_semaphore(&create_info, None)? };
        Ok(Semaphore {
            handle,
            marker: PhantomData,
        })
    }
}

#[derive(Debug)]
pub struct Binary;
impl crate::sealed::Sealed for Binary {}
impl SemaphoreType for Binary {
    fn create() -> Result<Semaphore<Self>> {
        let create_info = vk::SemaphoreCreateInfo::default();
        let handle = unsafe { Ctx::device().create_semaphore(&create_info, None)? };
        Ok(Semaphore {
            handle,
            marker: PhantomData,
        })
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
    pub fn info(&self, value: u64) -> SemaphoreInfo<'_> {
        SemaphoreInfo {
            handle: self.handle,
            value: Some(value),
            stage: Stage::All,
            _semaphore: PhantomData,
        }
    }
}

impl Semaphore<Binary> {
    #[validation_trace]
    pub fn info(&self) -> SemaphoreInfo<'_> {
        SemaphoreInfo {
            handle: self.handle,
            value: None,
            stage: Stage::All,
            _semaphore: PhantomData,
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
pub trait QueueFamilie: crate::sealed::Sealed + Debug {
    fn queues() -> &'static FamilyQueues;
}

#[derive(Debug)]
pub struct Transfer;
#[derive(Debug)]
pub struct Gfx;

impl crate::sealed::Sealed for Transfer {}
impl QueueFamilie for Transfer {
    fn queues() -> &'static FamilyQueues {
        let queues = &Ctx::get().queues;
        queues.family(Kind::Transfer).unwrap_or(queues.graphics())
    }
}
impl crate::sealed::Sealed for Gfx {}
impl QueueFamilie for Gfx {
    fn queues() -> &'static FamilyQueues {
        Ctx::get().queues.graphics()
    }
}

/// What a lease of `family` can record.
fn caps(family: &FamilyQueues) -> QueueCaps {
    QueueCaps {
        flags: family.flags,
        images: family.index == Ctx::get().queues.graphics().index,
    }
}

/// Calls `f` with the pending accesses of every queue.
fn for_each_pending(mut f: impl FnMut(&mut PendingAccesses)) {
    for family in Ctx::get().queues.iter() {
        for queue in &family.queues {
            f(&mut lock(&queue.pending));
        }
    }
}

/// Forgets the pending accesses of an image that is being destroyed, on every queue, so an
/// image that gets its handle next starts with nothing pending. Without it, every image ever
/// accessed would stay pending and be compared with every later access.
pub(crate) fn forget_image(image: vk::Image) {
    for_each_pending(|pending| pending.forget_image(image));
}

/// Like [`forget_image`], for the addresses of a buffer that is being destroyed.
pub(crate) fn forget_buffer(range: Range<u64>) {
    for_each_pending(|pending| pending.forget_buffer(range.clone()));
}

/// Held while leases are counted.
static LEASES: Mutex<()> = Mutex::new(());

fn leases(family: &FamilyQueues) -> Vec<u32> {
    (family.queues.iter())
        .map(|queue| queue.timelines.load(Ordering::Relaxed))
        .collect()
}

fn lease_count(family: &FamilyQueues, index: usize) -> &AtomicU32 {
    &family.queues[index].timelines
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
        let index = {
            let _leases = lock(&LEASES);
            let index = lease(&mut leases(family));
            lease_count(family, index).fetch_add(1, Ordering::Relaxed);
            index
        };
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
        let _leases = lock(&LEASES);
        is_optimal(&leases(self.family), self.index)
    }

    /// Moves the lease to the least used queue of its family. It waits until the old queue is
    /// idle first, so the submissions through the lease stay in order and its `FrameSlot`s stay
    /// valid, and the new queue takes over the old one's pending accesses too. Meant for the
    /// rare case [`Self::is_optimal`] reports.
    #[validation_trace]
    pub fn reselect(&mut self) -> Result<()> {
        if self.is_optimal() {
            return Ok(());
        }
        self.with(|queue| unsafe { Ctx::device().queue_wait_idle(queue) })?;
        let old = self.index;
        {
            let _leases = lock(&LEASES);
            lease_count(self.family, old).fetch_sub(1, Ordering::Relaxed);
            self.index = lease(&mut leases(self.family));
            lease_count(self.family, self.index).fetch_add(1, Ordering::Relaxed);
        }
        if self.index != old {
            let pending = lock(&self.family.queues[old].pending).clone();
            lock(&self.family.queues[self.index].pending).append(pending);
        }
        Ok(())
    }

    /// Calls `f` with the queue, locked.
    fn with<R>(&self, f: impl FnOnce(vk::Queue) -> R) -> R {
        f(*lock(&self.family.queues[self.index].queue))
    }
}

impl<Q: QueueFamilie> Default for Queue<Q> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Q: QueueFamilie> Drop for Queue<Q> {
    fn drop(&mut self) {
        let _leases = lock(&LEASES);
        lease_count(self.family, self.index).fetch_sub(1, Ordering::Relaxed);
    }
}

/// The queue the fewest leases share, whose count goes up.
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
    /// The epoch of the frame begun last, completed (dropped) once its fence signalled, which
    /// it may poll until then (`epoch`). Dropped first, before the fence is destroyed.
    epoch: Option<Epoch>,
    buffer: CommandBufferMemory,
    pool: CommandPool,
    fence: Fence,
    submitted: bool,
    family: u32,
    /// Lent to each recording (`CommandBuffer::barrier_log`), so it allocates once.
    barrier_log: Vec<Barriers>,
}

impl FrameSlot {
    #[validation_trace]
    pub fn new<Q: QueueFamilie>(queue: &Queue<Q>) -> Result<Self> {
        let pool = CommandPool::new(queue.family.index)?;
        let buffer = pool.create_command_buffer()?;
        Ok(Self {
            epoch: None,
            buffer,
            pool,
            fence: Fence::new()?,
            submitted: false,
            family: queue.family.index,
            barrier_log: Vec::new(),
        })
    }

    /// Waits for the frame this slot submitted last, completes its epoch, and begins one of
    /// `epoch`, which the caller reserved (`epoch::reserve`) before reading anything the frame
    /// reaches. The slot keeps it until the GPU finished the frame.
    #[validation_trace]
    pub fn begin(&mut self, epoch: Epoch) -> Result<Frame<'_>> {
        if self.submitted {
            tracing::info_span!("wait for frame slot").in_scope(|| self.fence.wait())?;
            // Completed before the fence is reset: nothing polls it after that.
            self.epoch = None;
            self.fence.reset()?;
            self.pool.reset()?;
            self.submitted = false;
        }
        // Also after a frame that failed before it was submitted.
        self.epoch = Some(epoch);
        Ok(Frame { slot: self })
    }
}

pub struct Frame<'a> {
    slot: &'a mut FrameSlot,
}

impl Frame<'_> {
    /// Records `executor` and submits it to `queue`, after the earlier submissions to it.
    ///
    /// A recording that doesn't reach the GPU, because `executor` panicked or recording or
    /// submitting failed, still moved the tracked layouts and accesses: it is replaced by a
    /// submission of just its barriers, waiting on `wait_on` and signalling the timeline
    /// semaphores of `signal` (not the binary ones, which nothing will wait on), so the tracking
    /// stays true. Then the panic continues, or the error is returned. A frame that fails before
    /// it records anything completes its epoch right away.
    #[validation_trace]
    pub fn execute<Q: QueueFamilie, F: FnOnce(&mut CommandBuffer)>(
        self,
        queue: &Queue<Q>,
        wait_on: &[SemaphoreInfo],
        signal: &[SemaphoreInfo],
        executor: F,
    ) -> Result<()> {
        execute_command(queue, self.slot, wait_on, signal, executor)
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

/// Why a recording didn't reach the GPU.
enum Failed {
    Panic(Box<dyn std::any::Any + Send>),
    Error(Error),
}

/// Records `executor` into the slot's command buffer and submits it (see `Frame::execute`).
#[validation_trace]
fn execute_command<Q: QueueFamilie, F: FnOnce(&mut CommandBuffer)>(
    queue: &Queue<Q>,
    slot: &mut FrameSlot,
    wait_on: &[SemaphoreInfo],
    signal: &[SemaphoreInfo],
    executor: F,
) -> Result<()> {
    // Held until the submission, so the queue gets command buffers in the order their
    // barriers were computed in.
    let mut pending = lock(&queue.family.queues[queue.index].pending);
    let epoch = slot.epoch.as_ref().map_or(0, Epoch::value);
    let mut taken = std::mem::take(&mut *pending);
    taken.retire(epoch::completed());
    // Lava records no ownership transfers yet, so only the graphics family accesses images.
    let kind = match caps(queue.family).images {
        true => Kind::Graphics,
        false => Kind::Transfer,
    };
    let mut cmd = CommandBuffer::new(
        slot.buffer.handle,
        kind,
        taken,
        epoch,
        slot.family,
        caps(queue.family),
        std::mem::take(&mut slot.barrier_log),
    );

    let result = match cmd.begin() {
        // Nothing was recorded: the accesses describe what earlier submissions do.
        Err(err) => {
            *pending = std::mem::take(&mut cmd.pending_accesses);
            slot.barrier_log = std::mem::take(&mut cmd.barrier_log);
            slot.epoch = None;
            if let Err(reset) = slot.pool.reset() {
                tracing::error!(%reset, "failed to reset a frame slot's command pool");
            }
            return Err(err);
        }
        Ok(()) => {
            let prev = CALLSITE.replace(None);
            let recorded = epoch::recording(|| {
                std::panic::catch_unwind(AssertUnwindSafe(|| executor(&mut cmd)))
            });
            CALLSITE.set(prev);
            match recorded {
                Err(panic) => Err(Failed::Panic(panic)),
                Ok(()) => {
                    host_barrier(cmd.handle);
                    (cmd.end())
                        .and_then(|()| submit(queue, slot, cmd.handle, wait_on, signal, true))
                        .map_err(Failed::Error)
                }
            }
        }
    };
    // Whatever happened: the recording's accesses and layouts are tracked already.
    *pending = std::mem::take(&mut cmd.pending_accesses);
    let failed = match result {
        Ok(()) => None,
        Err(failed) => {
            replay(queue, slot, &cmd, wait_on, signal);
            Some(failed)
        }
    };
    let mut log = std::mem::take(&mut cmd.barrier_log);
    log.clear();
    slot.barrier_log = log;
    drop(pending);
    match failed {
        None => Ok(()),
        Some(Failed::Error(err)) => Err(err),
        Some(Failed::Panic(panic)) => std::panic::resume_unwind(panic),
    }
}

/// Submits the slot's command buffer `handle`; on success the slot waits for its fence and
/// polls it for its epoch. `binary_signals`: also signal the binary semaphores of `signal`.
fn submit<Q: QueueFamilie>(
    queue: &Queue<Q>,
    slot: &mut FrameSlot,
    handle: vk::CommandBuffer,
    wait_on: &[SemaphoreInfo],
    signal: &[SemaphoreInfo],
    binary_signals: bool,
) -> Result<()> {
    let cmd_buffer_submit_info = vk::CommandBufferSubmitInfo::default().command_buffer(handle);
    let wait_infos: SmallVec<[_; 1]> = wait_on
        .iter()
        .map(|sem| sem.to_vk(sem.stage.flags()))
        .collect();
    // Untracked work (queries, `flush_all`, host barriers) has to finish too.
    let signal_infos: SmallVec<[_; 1]> = signal
        .iter()
        .filter(|sem| binary_signals || sem.value.is_some())
        .map(|sem| sem.to_vk(vk::PipelineStageFlags2::ALL_COMMANDS))
        .collect();
    let submit_info = vk::SubmitInfo2::default()
        .command_buffer_infos(std::slice::from_ref(&cmd_buffer_submit_info))
        .wait_semaphore_infos(&wait_infos)
        .signal_semaphore_infos(&signal_infos);
    queue.with(|queue| unsafe {
        Ctx::device().queue_submit2(queue, std::slice::from_ref(&submit_info), slot.fence.handle)
    })?;
    slot.submitted = true;
    if let Some(epoch) = &slot.epoch {
        epoch.submitted(slot.fence.handle);
    }
    Ok(())
}

/// Replaces a recording of `cmd` that didn't reach the GPU with a submission of just its
/// barriers, in the order they were computed: the tracking already counts their layout
/// transitions and accesses. If even that fails, the tracked layouts of the images it
/// transitioned are wrong from now on, which is logged.
fn replay<Q: QueueFamilie>(
    queue: &Queue<Q>,
    slot: &mut FrameSlot,
    cmd: &CommandBuffer,
    wait_on: &[SemaphoreInfo],
    signal: &[SemaphoreInfo],
) {
    let replayed = (|| -> Result<()> {
        slot.pool.reset()?;
        let device = Ctx::device();
        unsafe { device.begin_command_buffer(cmd.handle, &vk::CommandBufferBeginInfo::default())? };
        cmd.replay_barriers();
        unsafe { device.end_command_buffer(cmd.handle)? };
        submit(queue, slot, cmd.handle, wait_on, signal, false)
    })();
    if let Err(err) = replayed {
        tracing::error!(
            %err,
            "a frame that failed couldn't submit its barriers either: the layouts lava tracks for the images it used may be wrong"
        );
        slot.epoch = None;
        if let Err(reset) = slot.pool.reset() {
            tracing::error!(%reset, "failed to reset a frame slot's command pool");
        }
    }
}

impl Queue<Gfx> {
    /// `Ok(true)` means the swapchain is suboptimal or out of date and should be recreated.
    #[validation_trace]
    pub fn present<U: crate::image::usage::StaticUsage>(
        &self,
        swapchain: &Swapchain<U>,
        image_index: u32,
        wait_on: &[&Semaphore<Binary>],
    ) -> Result<bool> {
        let (_sync, handle) = swapchain.lock();
        let sc = [handle];
        let ii = [image_index];
        let waits: SmallVec<[_; 2]> = wait_on.iter().map(|sem| sem.handle).collect();
        let _span = tracing::info_span!("present").entered();
        let mut present_info = vk::PresentInfoKHR::default()
            .swapchains(&sc)
            .image_indices(&ii)
            .wait_semaphores(waits.as_slice());
        // Signalled once the present is done with the image and the semaphores, which may be
        // destroyed then (`epoch`).
        let fence = match <crate::features::gate::PresentFences as crate::features::On>::on() {
            true => Some([epoch::present_fence()?]),
            false => None,
        };
        let mut fence_info = vk::SwapchainPresentFenceInfoEXT::default();
        if let Some(fence) = &fence {
            fence_info = fence_info.fences(fence);
            present_info = present_info.push_next(&mut fence_info);
        }
        let presented = self.with(|queue| {
            let presented = unsafe { Functions::swapchain().queue_present(queue, &present_info) };
            // While the queue is locked, so `wait_idle` sees every present made before it.
            if let Some([fence]) = fence {
                epoch::presented(fence);
            }
            presented
        });
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
    use crate::vk::Handle;

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
            _semaphore: PhantomData,
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
            _semaphore: PhantomData,
        };
        let submit = info.to_vk(vk::PipelineStageFlags2::ALL_COMMANDS);
        assert_eq!(submit.semaphore, vk::Semaphore::from_raw(9));
        assert_eq!(submit.value, 12);
        assert_eq!(submit.stage_mask, vk::PipelineStageFlags2::ALL_COMMANDS);
    }
}

//! Submitting a timeline's recording: its waits (other timelines, acquired swapchain images) and signals (its value, presented images), the barriers-only replay of a recording that didn't reach the GPU, the empty submissions that signal dropped values, and the barrier that makes every submission's writes host-visible
use std::panic::AssertUnwindSafe;

use crate::vk;
use smallvec::SmallVec;

use crate::{
    command_buffer::{CommandBuffer, Recording},
    error::{Error, Result},
    lifetime, present,
    queue::{
        QueueKind, Wait, lock,
        timeline::{INDICES, Inner, SlotRes, Submission, completed, semaphore, wait_value},
    },
    state::{CALLSITE, Ctx},
};

/// Why a recording didn't reach the GPU.
enum Failed {
    Panic(Box<dyn std::any::Any + Send>),
    Error(Error),
}

/// What a submission waits for and signals besides its timeline's value.
struct Sync<'a> {
    /// Other timelines' values with the stages that wait for them.
    waits: &'a [(usize, u64, vk::PipelineStageFlags2)],
    /// Binary semaphores: acquired swapchain images' acquire semaphores, with their stages.
    binary_waits: SmallVec<[(vk::Semaphore, vk::PipelineStageFlags2); 1]>,
    /// Binary semaphores signalled at the end: presented images' present semaphores.
    binary_signals: SmallVec<[vk::Semaphore; 1]>,
}

/// Records `record` and submits it (see `Submission::submit`).
pub(crate) fn submit<K: QueueKind>(
    submission: Submission<K>,
    record: impl FnOnce(&mut CommandBuffer<K>),
) -> Result<Wait> {
    let Submission {
        mut ticket,
        slot,
        waits,
        ..
    } = submission;
    let inner = ticket.inner.clone();
    let (i, n) = (inner.index, ticket.value);
    let mut slot = lock(&inner.slots[slot]);
    // Held until the submission, so the queue gets command buffers in the order their barriers
    // were computed in.
    let mut state = lock(&INDICES[i].state);
    if n != state.order.submitted + 1 {
        let next = state.order.submitted + 1;
        drop((state, slot));
        panic!(
            "value {n} of a timeline is submitted before value {next}: submit (or drop) a \
             timeline's reservations in the order they were made"
        );
    }
    // From here on this function signals `n`, whatever happens.
    ticket.done = true;
    let mut pending = std::mem::take(&mut state.pending);
    pending.retire(completed(i));
    let log = std::mem::take(&mut slot.barrier_log);
    let valid_bits = inner.family.timestamp_valid_bits;
    let mut cmd =
        CommandBuffer::<K>::new(Recording::new(slot.cmd, K::KIND, pending, i, n, valid_bits, log));

    if let Err(err) = cmd.rec.begin() {
        // Nothing was recorded: the accesses describe what earlier submissions do.
        state.pending = std::mem::take(&mut cmd.rec.pending_accesses);
        slot.barrier_log = std::mem::take(&mut cmd.rec.barrier_log);
        if let Err(reset) = slot.reset() {
            tracing::error!(%reset, "failed to reset a timeline's command pool");
        }
        let signal = state.order.submit(n);
        signal_empty(&inner, n, signal);
        return Err(err);
    }
    let prev = CALLSITE.replace(None);
    let recorded = lifetime::recording(|| {
        std::panic::catch_unwind(AssertUnwindSafe(|| record(&mut cmd)))
    });
    CALLSITE.set(prev);

    let signal = state.order.submit(n);
    let rec = &mut cmd.rec;
    let mut sync = Sync {
        waits: &waits,
        binary_waits: rec.acquire_waits.iter().map(|w| (w.semaphore, w.stage)).collect(),
        binary_signals: rec.presents.iter().map(|p| p.semaphore).collect(),
    };
    let result = match recorded {
        Err(panic) => Err(Failed::Panic(panic)),
        Ok(()) => {
            host_barrier(rec.handle);
            (rec.end())
                .and_then(|()| queue_submit(&inner, Some(rec.handle), &sync, signal))
                .map_err(Failed::Error)
        }
    };
    let failed = match result {
        Ok(()) => None,
        Err(failed) => {
            // Nothing will wait for the present semaphores.
            sync.binary_signals.clear();
            replay(&inner, &slot, rec, &sync, n, signal);
            Some(failed)
        }
    };
    // Whatever happened, the acquire semaphores were waited for.
    let presents = if failed.is_none() { &rec.presents[..] } else { &[] };
    present::table::submitted(&rec.acquire_waits, presents, inner.queue(), i, n);
    // The recording's accesses and image states are tracked already.
    state.pending = std::mem::take(&mut rec.pending_accesses);
    let mut log = std::mem::take(&mut rec.barrier_log);
    log.clear();
    slot.barrier_log = log;
    drop((state, slot));
    match failed {
        None => Ok(Wait::new(i, n)),
        Some(Failed::Error(err)) => Err(err),
        Some(Failed::Panic(panic)) => std::panic::resume_unwind(panic),
    }
}

/// Submits `cmd` (or nothing) to `inner`'s queue with `sync`, signalling `signal` on its
/// timeline once everything before it in the queue finished too.
fn queue_submit(
    inner: &Inner,
    cmd: Option<vk::CommandBuffer>,
    sync: &Sync,
    signal: u64,
) -> Result<()> {
    let buffers: SmallVec<[_; 1]> = cmd
        .map(|cmd| vk::CommandBufferSubmitInfo::default().command_buffer(cmd))
        .into_iter()
        .collect();
    let waits: SmallVec<[_; 4]> = (sync.waits.iter())
        .map(|&(t, value, stages)| {
            vk::SemaphoreSubmitInfo::default()
                .semaphore(semaphore(t))
                .value(value)
                .stage_mask(stages)
        })
        .chain(sync.binary_waits.iter().map(|&(semaphore, stages)| {
            vk::SemaphoreSubmitInfo::default()
                .semaphore(semaphore)
                .stage_mask(stages)
        }))
        .collect();
    // Untracked work (queries, `flush_all`, host barriers) has to finish too.
    let all = vk::PipelineStageFlags2::ALL_COMMANDS;
    let signals: SmallVec<[_; 2]> = std::iter::once(
        vk::SemaphoreSubmitInfo::default()
            .semaphore(semaphore(inner.index))
            .value(signal)
            .stage_mask(all),
    )
    .chain((sync.binary_signals.iter()).map(|&semaphore| {
        vk::SemaphoreSubmitInfo::default()
            .semaphore(semaphore)
            .stage_mask(all)
    }))
    .collect();
    let info = vk::SubmitInfo2::default()
        .command_buffer_infos(&buffers)
        .wait_semaphore_infos(&waits)
        .signal_semaphore_infos(&signals);
    let queue = lock(&inner.queue().queue);
    unsafe { Ctx::device().queue_submit2(*queue, &[info], vk::Fence::null())? };
    Ok(())
}

/// Replaces a recording that didn't reach the GPU with a submission of just its barriers, in
/// the order they were computed, with its waits, signalling `signal`: the tracking already counts
/// their transitions and accesses. If even that fails, the tracked states of the images it
/// touched are wrong from now on, which is logged, and the value is signalled from the host.
fn replay(inner: &Inner, slot: &SlotRes, rec: &Recording, sync: &Sync, n: u64, signal: u64) {
    let replayed = (|| -> Result<()> {
        slot.reset()?;
        let device = Ctx::device();
        unsafe { device.begin_command_buffer(rec.handle, &vk::CommandBufferBeginInfo::default())? };
        rec.replay_barriers();
        unsafe { device.end_command_buffer(rec.handle)? };
        queue_submit(inner, Some(rec.handle), sync, signal)
    })();
    if let Err(err) = replayed {
        tracing::error!(
            %err,
            "a recording that failed couldn't submit its barriers either: the image states lava \
             tracks for it may be wrong"
        );
        if let Err(reset) = slot.reset() {
            tracing::error!(%reset, "failed to reset a timeline's command pool");
        }
        host_signal(inner, n, signal);
    }
}

/// Signals values `n..=signal` of `inner`'s timeline with an empty submission, after the work
/// before them on its queue: they were dropped, or their recording never began.
pub(crate) fn signal_empty(inner: &Inner, n: u64, signal: u64) {
    let sync = Sync {
        waits: &[],
        binary_waits: SmallVec::new(),
        binary_signals: SmallVec::new(),
    };
    if let Err(err) = queue_submit(inner, None, &sync, signal) {
        tracing::error!(%err, "failed to signal a timeline's dropped value");
        host_signal(inner, n, signal);
    }
}

/// The fallback when the queue takes nothing any more (a lost device): signals `signal` from the
/// host once the value before `n` was signalled, so no GPU signal is ever passed.
fn host_signal(inner: &Inner, n: u64, signal: u64) {
    let i = inner.index;
    let signalled = wait_value(i, n - 1).and_then(|()| {
        let info = vk::SemaphoreSignalInfo::default()
            .semaphore(semaphore(i))
            .value(signal);
        Ok(unsafe { Ctx::device().signal_semaphore(&info)? })
    });
    if let Err(err) = signalled {
        tracing::error!(%err, "failed to signal a timeline value from the host");
    }
}

/// Ends every submission: makes its writes readable by the host once its value is signalled
/// (readbacks, clock counters), which the signal alone doesn't.
fn host_barrier(cmd: vk::CommandBuffer) {
    let barrier = [vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::HOST)
        .dst_access_mask(vk::AccessFlags2::HOST_READ)];
    let info = vk::DependencyInfo::default().memory_barriers(&barrier);
    unsafe { Ctx::device().cmd_pipeline_barrier2(cmd, &info) };
}

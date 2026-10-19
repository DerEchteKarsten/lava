//! Timelines: up to `MAX_TIMELINES` timeline semaphores, each owned by one `Timeline` at a time and pinned to one queue with a command pool per frame in flight, and the submission of their recordings (queued uploads first), with the snapshot of their submitted buffer writes that host access checks
use std::{
    fmt,
    marker::PhantomData,
    panic::AssertUnwindSafe,
    range::Range,
    sync::{
        LazyLock, Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

use crate::vk;
use lava_macros::validation_trace;
use smallvec::SmallVec;

use crate::{
    bindless::Bindless,
    buffer::upload::{self, StagingRing},
    command_buffer::{Barriers, CommandBuffer, Recording, pending::PendingAccesses},
    error::{Error, Result},
    lifetime,
    queue::{FamilyInfo, FamilyQueues, Kind, MAX_TIMELINES, QueueKind, Stage, Wait, lock},
    state::{CALLSITE, Ctx},
};

/// One of the `MAX_TIMELINES` timeline semaphores. Its values go on across the timelines that
/// own it one after the other, so a `Wait` on it stays valid.
pub(crate) struct Index {
    semaphore: OnceLock<vk::Semaphore>,
    /// What drops are stamped with: `Timeline::reserve` counts it up, a submission signals it.
    pub(crate) stamp: AtomicU64,
    completed: AtomicU64,
    pub(crate) state: Mutex<IndexState>,
    /// The buffer writes of its submissions that hadn't completed when the last one was
    /// submitted, with their values: what host access checks (`Buffer::host`). A copy of
    /// `state`'s, so the check never waits for a recording.
    pub(crate) writes: Mutex<Vec<(Range<u64>, u64)>>,
}

#[derive(Default)]
pub(crate) struct IndexState {
    pub(crate) pending: PendingAccesses,
    submitted: u64,
}

pub(crate) static INDICES: LazyLock<[Index; MAX_TIMELINES]> = LazyLock::new(|| {
    std::array::from_fn(|_| Index {
        semaphore: OnceLock::new(),
        stamp: AtomicU64::new(0),
        completed: AtomicU64::new(0),
        state: Mutex::default(),
        writes: Mutex::default(),
    })
});

static LIVE: Mutex<[bool; MAX_TIMELINES]> = Mutex::new([false; MAX_TIMELINES]);

fn semaphore(i: usize) -> vk::Semaphore {
    *INDICES[i].semaphore.get().expect("a timeline existed")
}

pub(crate) fn completed(i: usize) -> u64 {
    INDICES[i].completed.load(Ordering::Acquire)
}

pub(crate) fn poll(i: usize) -> u64 {
    let index = &INDICES[i];
    if let Some(&semaphore) = index.semaphore.get() {
        match unsafe { Ctx::device().get_semaphore_counter_value(semaphore) } {
            Ok(value) => {
                index.completed.fetch_max(value, Ordering::AcqRel);
            }
            Err(err) => tracing::error!(%err, "failed to read a timeline semaphore"),
        }
    }
    completed(i)
}

pub(crate) fn poll_all() {
    for i in 0..MAX_TIMELINES {
        poll(i);
    }
}

pub(crate) fn wait_value(i: usize, value: u64) -> Result<()> {
    if poll(i) >= value {
        return Ok(());
    }
    let (semaphores, values) = ([semaphore(i)], [value]);
    let info = vk::SemaphoreWaitInfo::default()
        .semaphores(&semaphores)
        .values(&values);
    unsafe { Ctx::device().wait_semaphores(&info, u64::MAX)? };
    INDICES[i].completed.fetch_max(value, Ordering::AcqRel);
    Ok(())
}

pub(crate) fn forget_image(image: vk::Image) {
    for index in INDICES.iter() {
        lock(&index.state).pending.forget_image(image);
    }
}

struct Slot {
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    barrier_log: Vec<Barriers>,
    staging: StagingRing,
    signalled: u64,
}

struct Inner {
    index: usize,
    family: &'static FamilyQueues,
    /// What its recordings know of the family.
    info: FamilyInfo,
    queue: usize,
    slots: Box<[Slot]>,
    /// Submissions begun so far, which take the frame slots in turn.
    begun: u64,
}

impl fmt::Debug for Inner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Timeline")
            .field("index", &self.index)
            .field("kind", &self.family.kind)
            .field("queue", &self.queue)
            .field("in_flight", &self.slots.len())
            .finish()
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        let index = &INDICES[self.index];
        let submitted = lock(&index.state).submitted;
        if let Err(err) = wait_value(self.index, submitted) {
            tracing::error!(%err, "failed to wait for a dropped timeline's submissions");
        }
        for slot in &self.slots {
            unsafe { Ctx::device().destroy_command_pool(slot.pool, None) };
        }
        lock(&index.state).pending = PendingAccesses::default();
        lock(&index.writes).clear();
        lock(&self.family.timelines)[self.queue] -= 1;
        lock(&LIVE)[self.index] = false;
        lifetime::reclaim();
    }
}

/// Submissions to the queue of family `K` the fewest timelines share, with `in_flight` frame
/// slots. Dropping it waits for its submissions.
pub struct Timeline<K: QueueKind> {
    inner: Inner,
    _kind: PhantomData<fn() -> K>,
}

impl<K: QueueKind> fmt::Debug for Timeline<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}

impl<K: QueueKind> Timeline<K> {
    /// `MissingFeature` without a family of `K`, `OutOfTimelines` when `MAX_TIMELINES` exist.
    #[validation_trace]
    pub fn new(in_flight: u32) -> Result<Self> {
        assert!(in_flight > 0, "a timeline has at least one frame slot");
        K::check()?;
        let family = (Ctx::get().queues.family(K::KIND)).expect("lava has queues of every family whose feature is on");
        let device = Ctx::device();
        let index = {
            let mut live = lock(&LIVE);
            let index = live.iter().position(|live| !live).ok_or(Error::OutOfTimelines)?;
            if INDICES[index].semaphore.get().is_none() {
                let mut timeline = vk::SemaphoreTypeCreateInfo::default()
                    .semaphore_type(vk::SemaphoreType::TIMELINE);
                let info = vk::SemaphoreCreateInfo::default().push_next(&mut timeline);
                let semaphore = unsafe { device.create_semaphore(&info, None)? };
                let _ = INDICES[index].semaphore.set(semaphore);
            }
            live[index] = true;
            index
        };
        let slots = (0..in_flight).map(|_| -> Result<_> {
            // Its command buffer is rerecorded every submission.
            let info = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                .queue_family_index(family.index);
            let pool = unsafe { device.create_command_pool(&info, None)? };
            let allocate = vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmd = unsafe { device.allocate_command_buffers(&allocate)? }[0];
            Ok(Slot {
                pool,
                cmd,
                barrier_log: Vec::new(),
                staging: StagingRing::default(),
                signalled: 0,
            })
        });
        let slots = match slots.collect::<Result<Box<[_]>>>() {
            Ok(slots) => slots,
            Err(err) => {
                lock(&LIVE)[index] = false;
                return Err(err);
            }
        };
        let queue = {
            let mut timelines = lock(&family.timelines);
            let queue = (0..timelines.len()).min_by_key(|&q| timelines[q]).unwrap();
            timelines[queue] += 1;
            queue
        };
        let queues = &Ctx::get().queues;
        let info = FamilyInfo {
            kind: K::KIND,
            timestamp_valid_bits: family.timestamp_valid_bits,
            image_granularity: family.image_granularity,
            indices: [Kind::Graphics, Kind::Compute, Kind::Transfer]
                .map(|kind| queues.family(kind).map_or(vk::QUEUE_FAMILY_IGNORED, |f| f.index)),
        };
        Ok(Timeline {
            inner: Inner {
                index,
                family,
                info,
                queue,
                slots,
                begun: 0,
            },
            _kind: PhantomData,
        })
    }

    /// Counts the stamp up: call it before reading anything the next submission will reach, so
    /// what is dropped after it lives until that submission finished.
    pub fn reserve(&self) {
        INDICES[self.inner.index].stamp.fetch_add(1, Ordering::SeqCst);
    }

    /// Waits until the next frame slot's last submission finished and destroys what that frees.
    /// The submission borrows the timeline, so it has one at a time.
    #[validation_trace]
    pub fn begin(&mut self) -> Result<Submission<'_, K>> {
        let inner = &mut self.inner;
        let index = &INDICES[inner.index];
        let slot = (inner.begun % inner.slots.len() as u64) as usize;
        inner.begun += 1;
        index.stamp.fetch_max(lock(&index.state).submitted + 1, Ordering::SeqCst);
        let signalled = inner.slots[slot].signalled;
        if poll(inner.index) < signalled {
            let _span = tracing::info_span!("wait for frame slot").entered();
            wait_value(inner.index, signalled)?;
        }
        poll_all();
        lifetime::reclaim();
        inner.slots[slot].staging.reset();
        let pool = inner.slots[slot].pool;
        unsafe { Ctx::device().reset_command_pool(pool, vk::CommandPoolResetFlags::empty())? };
        Ok(Submission {
            inner,
            slot,
            waits: SmallVec::new(),
            _kind: PhantomData,
        })
    }
}

/// A begun submission of a timeline, which it borrows: its frame slot is free.
#[must_use]
pub struct Submission<'t, K: QueueKind> {
    inner: &'t mut Inner,
    slot: usize,
    waits: SmallVec<[(usize, u64, vk::PipelineStageFlags2); 2]>,
    _kind: PhantomData<fn() -> K>,
}

impl<K: QueueKind> fmt::Debug for Submission<'_, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Submission")
            .field("timeline", &self.inner)
            .field("slot", &self.slot)
            .finish()
    }
}

impl<K: QueueKind> Submission<'_, K> {
    /// Its frame slot, which per-frame resources (rings, query pools) index by.
    pub fn slot(&self) -> usize {
        self.slot
    }

    /// Waits for `wait` at its stage. A wait on its own timeline is dropped; of two on one
    /// timeline the later value is kept, at the stages of both.
    pub fn wait_for(&mut self, wait: Wait) -> &mut Self {
        use vk::PipelineStageFlags2 as F;
        let stage = match wait.stage {
            Stage::All => F::ALL_COMMANDS,
            Stage::Compute => F::COMPUTE_SHADER | F::DRAW_INDIRECT,
            Stage::RayTracing => F::RAY_TRACING_SHADER_KHR | F::DRAW_INDIRECT,
            Stage::Raster => F::ALL_GRAPHICS,
            Stage::Transfer => F::ALL_TRANSFER,
        };
        let timeline = wait.timeline as usize;
        if timeline == self.inner.index {
            return self;
        }
        match self.waits.iter_mut().find(|(t, ..)| *t == timeline) {
            Some((_, value, stages)) => {
                *value = (*value).max(wait.value);
                *stages |= stage;
            }
            None => self.waits.push((timeline, wait.value, stage)),
        }
        self
    }

    /// Records `record` and submits it after its waits, signalling the timeline's stamp. A
    /// recording that doesn't reach the GPU is replaced by a submission of just its barriers,
    /// then its panic continues or its error is returned.
    #[validation_trace]
    pub fn submit(self, record: impl FnOnce(&mut CommandBuffer<K>)) -> Result<Wait> {
        let inner = self.inner;
        let i = inner.index;
        let index = &INDICES[i];
        let device = Ctx::device();
        let slot = &mut inner.slots[self.slot];
        let mut state = lock(&index.state);
        let state = &mut *state;
        state.pending.retire(completed(i));
        // `begin` raised the stamp past the last submission's value.
        let value = index.stamp.load(Ordering::SeqCst);
        debug_assert!(value > state.submitted);
        let mut cmd = CommandBuffer::<K>::new(Recording::new(
            slot.cmd,
            inner.info,
            &mut state.pending,
            i,
            value,
            &mut slot.barrier_log,
            &mut slot.staging,
        ));
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { device.begin_command_buffer(slot.cmd, &begin_info)? };
        Bindless::bind(slot.cmd, K::KIND);
        // Rewrites of reserved slots, ordered after the queue's earlier commands.
        let queued = match K::KIND {
            Kind::Transfer => Default::default(),
            _ => Bindless::flush_queued(slot.cmd),
        };
        // Queued writes of buffers and images, ordered after the timeline's earlier commands.
        upload::flush(&mut cmd);
        let prev = CALLSITE.replace(None);
        let was_recording = lifetime::RECORDING.replace(true);
        let recorded = std::panic::catch_unwind(AssertUnwindSafe(|| record(&mut cmd)));
        lifetime::RECORDING.set(was_recording);
        CALLSITE.set(prev);

        let rec = &mut cmd.rec;
        let unpresented = (rec.acquired.iter()).any(|(image, _)| !rec.presents.iter().any(|p| p.image == *image));
        let recorded = recorded.and_then(|()| match unpresented {
            true => Err(Box::new("a submission that uses an acquired swapchain image must present it") as _),
            false => Ok(()),
        });
        let binary_waits: SmallVec<[_; 1]> = (rec.presents.iter())
            .map(|p| {
                let first = rec.acquired.iter().find(|(image, _)| *image == p.image);
                (p.acquire, first.map_or(vk::PipelineStageFlags2::ALL_COMMANDS, |(_, stage)| *stage))
            })
            .collect();
        let presents: SmallVec<[_; 1]> = rec.presents.iter().map(|p| p.present).collect();
        let queue = &inner.family.queues[inner.queue];
        let submit = |cmd: vk::CommandBuffer| -> Result<()> {
            let buffers = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
            let waits: SmallVec<[_; 4]> = (self.waits.iter())
                .map(|&(t, value, stages)| (semaphore(t), value, stages))
                .chain(binary_waits.iter().map(|&(s, stages)| (s, 0, stages)))
                .map(|(semaphore, value, stages)| {
                    vk::SemaphoreSubmitInfo::default()
                        .semaphore(semaphore)
                        .value(value)
                        .stage_mask(stages)
                })
                .collect();
            let all = vk::PipelineStageFlags2::ALL_COMMANDS;
            let signals: SmallVec<[_; 2]> = std::iter::once((semaphore(i), value))
                .chain(presents.iter().map(|&s| (s, 0)))
                .map(|(semaphore, value)| {
                    vk::SemaphoreSubmitInfo::default()
                        .semaphore(semaphore)
                        .value(value)
                        .stage_mask(all)
                })
                .collect();
            let info = vk::SubmitInfo2::default()
                .command_buffer_infos(&buffers)
                .wait_semaphore_infos(&waits)
                .signal_semaphore_infos(&signals);
            unsafe { device.queue_submit2(*lock(queue), &[info], vk::Fence::null())? };
            Ok(())
        };
        let result = match recorded {
            Err(panic) => Err(Ok(panic)),
            Ok(()) => {
                let barrier = [vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                    .dst_access_mask(vk::AccessFlags2::HOST_READ)];
                let info = vk::DependencyInfo::default().memory_barriers(&barrier);
                unsafe { device.cmd_pipeline_barrier2(rec.handle, &info) };
                unsafe { device.end_command_buffer(rec.handle) }
                    .map_err(Error::from)
                    .and_then(|()| submit(rec.handle))
                    .map_err(Err)
            }
        };
        upload::finish(i, value, result.is_ok());
        if result.is_err() {
            Bindless::requeue(queued);
            let replayed = (|| -> Result<()> {
                unsafe {
                    device.reset_command_pool(slot.pool, vk::CommandPoolResetFlags::empty())?;
                    device.begin_command_buffer(rec.handle, &Default::default())?;
                    // An acquired image without its acquire wait keeps its layout; the surface
                    // resets its state when it keeps it current or acquires it again.
                    let presented = |image: vk::Image| rec.presents.iter().any(|p| p.image == image);
                    for barriers in rec.barrier_log.iter_mut() {
                        barriers.images.retain(|b| !rec.acquired.iter().any(|&(i, _)| i == b.image && !presented(i)));
                    }
                    for barriers in rec.barrier_log.iter() {
                        rec.pipeline_barrier(barriers);
                    }
                    device.end_command_buffer(rec.handle)?;
                }
                submit(rec.handle)
            })();
            if let Err(err) = replayed {
                tracing::error!(
                    %err,
                    "a recording that failed couldn't submit its barriers either: the image states lava tracks for it may be wrong"
                );
            }
        }
        drop(cmd);
        slot.barrier_log.clear();
        slot.signalled = value;
        state.submitted = value;
        {
            let done = completed(i);
            let mut writes = lock(&index.writes);
            writes.clear();
            writes.extend(
                (state.pending.buffer_writes.iter())
                    .filter(|w| w.value > done)
                    .map(|w| (w.write.range, w.value)),
            );
        }
        match result {
            Ok(()) => Ok(Wait {
                timeline: i as u8,
                value,
                stage: Default::default(),
            }),
            Err(Err(err)) => Err(err),
            Err(Ok(panic)) => std::panic::resume_unwind(panic),
        }
    }
}

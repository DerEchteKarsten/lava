//! Timelines: up to `MAX_TIMELINES` timeline semaphores, each owned by one `Timeline` at a time and pinned to one queue with a command pool per frame in flight and an optional staging ring (`uploader()`), and the submission of their recordings (queued bindless rewrites first), with the snapshot of their submitted buffer writes that host access checks; values no submission signals (a dropped timeline's stamp, a failed replay) are signalled from the host, device loss comes out of polls and waits, and `shutdown` destroys the semaphores
use std::{
    fmt,
    marker::PhantomData,
    panic::AssertUnwindSafe,
    range::Range,
    sync::{
        Arc, LazyLock, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use crate::vk;
use lava_macros::validation_trace;
use smallvec::SmallVec;

use crate::{
    bindless::Bindless,
    buffer::staging::{Staging, Uploader},
    command_buffer::{Barriers, CommandBuffer, Recording, pending::PendingAccesses},
    error::{Error, Result},
    global::Live,
    lifetime,
    queue::{FamilyInfo, FamilyQueues, Kind, MAX_TIMELINES, QueueKind, StageOf, Wait, lock},
    state::{CALLSITE, Ctx, vk_error},
};

/// One of the `MAX_TIMELINES` timeline semaphores. Its values go on across the timelines that
/// own it one after the other, so a `Wait` on it stays valid.
pub(crate) struct Index {
    /// Its timeline semaphore's raw handle, 0 before the first timeline of the index (and
    /// after `shutdown`).
    semaphore: AtomicU64,
    /// What drops are stamped with: `Timeline::reserve` counts it up, a submission signals it.
    pub(crate) stamp: AtomicU64,
    completed: AtomicU64,
    pub(crate) state: Mutex<IndexState>,
    /// The buffer writes of its submissions that hadn't completed when the last one was
    /// submitted, with their values: what host access checks (`Buffer::host`). A copy of
    /// `state`'s, so the check never waits for a recording.
    pub(crate) writes: Mutex<Vec<(Range<u64>, u64)>>,
    /// The live timeline's staging ring, which dropped buffers and images forget their copies in.
    pub(crate) staging: Mutex<Option<Arc<Staging>>>,
}

#[derive(Default)]
pub(crate) struct IndexState {
    pub(crate) pending: PendingAccesses,
    submitted: u64,
}

pub(crate) static INDICES: LazyLock<[Index; MAX_TIMELINES]> = LazyLock::new(|| {
    std::array::from_fn(|_| Index {
        semaphore: AtomicU64::new(0),
        stamp: AtomicU64::new(0),
        completed: AtomicU64::new(0),
        state: Mutex::default(),
        writes: Mutex::default(),
        staging: Mutex::default(),
    })
});

static LIVE: Mutex<[bool; MAX_TIMELINES]> = Mutex::new([false; MAX_TIMELINES]);

/// Timeline `i`'s semaphore, if one was made.
fn semaphore_of(i: usize) -> Option<vk::Semaphore> {
    use vk::Handle;
    match INDICES[i].semaphore.load(Ordering::Acquire) {
        0 => None,
        raw => Some(vk::Semaphore::from_raw(raw)),
    }
}

fn semaphore(i: usize) -> vk::Semaphore {
    semaphore_of(i).expect("a timeline existed")
}

pub(crate) fn completed(i: usize) -> u64 {
    INDICES[i].completed.load(Ordering::Acquire)
}

/// Reads timeline `i`'s semaphore and returns its completed value; `Error::DeviceLost` once the
/// device is lost (logged once).
pub(crate) fn poll(i: usize) -> Result<u64> {
    let index = &INDICES[i];
    if let Some(semaphore) = semaphore_of(i) {
        let value = unsafe { Ctx::device().get_semaphore_counter_value(semaphore) }.map_err(vk_error)?;
        index.completed.fetch_max(value, Ordering::AcqRel);
    }
    Ok(completed(i))
}

/// Polls every timeline; the first error, after polling all.
pub(crate) fn poll_all() -> Result<()> {
    let mut result = Ok(());
    for i in 0..MAX_TIMELINES {
        if let Err(err) = poll(i)
            && result.is_ok()
        {
            result = Err(err);
        }
    }
    result
}

pub(crate) fn wait_value(i: usize, value: u64) -> Result<()> {
    wait_value_for(i, value, Duration::MAX).map(|_| ())
}

/// Waits up to `timeout` for `value` of timeline `i`; `false` when it ran out.
pub(crate) fn wait_value_for(i: usize, value: u64, timeout: Duration) -> Result<bool> {
    let mut values = [0; MAX_TIMELINES];
    values[i] = value;
    wait_values(&values, timeout)
}

/// Waits up to `timeout` until every timeline `i` reached `values[i]` (0 for none), with one
/// `vkWaitSemaphores`; `false` when it ran out.
pub(crate) fn wait_values(values: &[u64; MAX_TIMELINES], timeout: Duration) -> Result<bool> {
    let mut semaphores: SmallVec<[vk::Semaphore; MAX_TIMELINES]> = SmallVec::new();
    let mut waited: SmallVec<[u64; MAX_TIMELINES]> = SmallVec::new();
    for (i, &value) in values.iter().enumerate() {
        if value > completed(i) && value > poll(i)? {
            semaphores.push(semaphore(i));
            waited.push(value);
        }
    }
    if semaphores.is_empty() {
        return Ok(true);
    }
    let info = vk::SemaphoreWaitInfo::default()
        .semaphores(&semaphores)
        .values(&waited);
    let timeout = timeout.as_nanos().min(u64::MAX as u128) as u64;
    match unsafe { Ctx::device().wait_semaphores(&info, timeout) } {
        Ok(()) => {}
        Err(vk::Result::TIMEOUT) => return Ok(false),
        Err(err) => return Err(vk_error(err)),
    }
    for (i, &value) in values.iter().enumerate() {
        INDICES[i].completed.fetch_max(value, Ordering::AcqRel);
    }
    Ok(true)
}

/// Signals `value` of timeline `i` from the host, for a value no submission will signal (a
/// timeline dropped with its stamp ahead, a recording that never reached the GPU), so what
/// waits for it goes on. Waits for its earlier submissions (up to `submitted`) first, which a
/// host signal must not overtake. Leaves the values alone once the device is lost.
fn host_signal(i: usize, submitted: u64, value: u64) {
    let signalled = (|| -> Result<()> {
        wait_value(i, submitted.min(value.saturating_sub(1)))?;
        if poll(i)? >= value {
            return Ok(());
        }
        let info = vk::SemaphoreSignalInfo::default()
            .semaphore(semaphore(i))
            .value(value);
        unsafe { Ctx::device().signal_semaphore(&info) }.map_err(vk_error)?;
        INDICES[i].completed.fetch_max(value, Ordering::AcqRel);
        Ok(())
    })();
    match signalled {
        Ok(()) | Err(Error::DeviceLost) => {}
        Err(err) => tracing::error!(%err, "failed to signal a timeline value no submission signals"),
    }
}

/// Destroys every timeline semaphore at `shutdown`, once no timeline exists. Their values go
/// on in the next context, which starts each semaphore at its index's stamp, so `Wait`s and
/// `Stamp`s of this one count as done there.
///
/// # Safety
/// No timeline exists and nothing polls (lava is closed).
pub(crate) unsafe fn destroy() {
    for (i, index) in INDICES.iter().enumerate() {
        if let Some(semaphore) = semaphore_of(i) {
            unsafe { Ctx::device().destroy_semaphore(semaphore, None) };
            index.semaphore.store(0, Ordering::Release);
        }
        let stamp = index.stamp.load(Ordering::SeqCst);
        index.completed.fetch_max(stamp, Ordering::AcqRel);
        let mut state = lock(&index.state);
        state.pending = PendingAccesses::default();
        state.submitted = state.submitted.max(stamp);
        lock(&index.writes).clear();
        *lock(&index.staging) = None;
    }
    *lock(&LIVE) = [false; MAX_TIMELINES];
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
    /// Its staging ring; none on `rebar` devices or with a size of 0.
    staging: Option<Arc<Staging>>,
    /// Last, so it counts the timeline until its drop finished.
    _live: Live,
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
        match wait_value(self.index, submitted) {
            Ok(()) | Err(Error::DeviceLost) => {}
            Err(err) => tracing::error!(%err, "failed to wait for a dropped timeline's submissions"),
        }
        // A stamp `reserve` or an unsubmitted `begin` raised past the last submission: no
        // submission will signal it, and what was dropped meanwhile waits for it.
        let stamp = index.stamp.load(Ordering::SeqCst);
        if stamp > submitted {
            host_signal(self.index, submitted, stamp);
            // Signalled, so the next timeline of the index submits past it.
            lock(&index.state).submitted = stamp;
        }
        for slot in &self.slots {
            unsafe { Ctx::device().destroy_command_pool(slot.pool, None) };
        }
        if let Some(staging) = self.staging.take() {
            staging.close();
        }
        lock(&index.state).pending = PendingAccesses::default();
        lock(&index.writes).clear();
        lock(&self.family.timelines)[self.queue] -= 1;
        lock(&LIVE)[self.index] = false;
        lifetime::run_ready();
    }
}

/// Submissions to the queue of family `K` the fewest timelines share, with `in_flight` frame
/// slots and a staging ring its [`Uploader`]s fill (`uploader()`). Dropping it waits for its
/// submissions.
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
    /// A timeline with `in_flight` frame slots and a staging ring of `staging` bytes for its
    /// uploaders (0: none; never one on a `rebar` device, where every `Gpu` buffer is mapped
    /// and uploaders write in place). `MissingFeature` without a family of `K`,
    /// `OutOfTimelines` when `MAX_TIMELINES` exist.
    #[validation_trace]
    pub fn new(in_flight: u32, staging: u64) -> Result<Self> {
        let live = Live::new();
        assert!(in_flight > 0, "a timeline has at least one frame slot");
        K::check()?;
        let family = (Ctx::get().queues.family(K::KIND)).expect("lava has queues of every family whose feature is on");
        let device = Ctx::device();
        let index = {
            let mut live = lock(&LIVE);
            let index = live.iter().position(|live| !live).ok_or(Error::OutOfTimelines)?;
            if semaphore_of(index).is_none() {
                // Values go on from what the index's earlier semaphores reached.
                let start = INDICES[index].stamp.load(Ordering::SeqCst).max(completed(index));
                let mut timeline = vk::SemaphoreTypeCreateInfo::default()
                    .semaphore_type(vk::SemaphoreType::TIMELINE)
                    .initial_value(start);
                let info = vk::SemaphoreCreateInfo::default().push_next(&mut timeline);
                let semaphore = unsafe { device.create_semaphore(&info, None) }.map_err(vk_error)?;
                use vk::Handle;
                INDICES[index].semaphore.store(semaphore.as_raw(), Ordering::Release);
            }
            live[index] = true;
            index
        };
        let destroy_pools = |slots: &[Slot]| {
            for slot in slots {
                unsafe { device.destroy_command_pool(slot.pool, None) };
            }
        };
        let mut slots = Vec::with_capacity(in_flight as usize);
        for _ in 0..in_flight {
            let made = (|| -> Result<Slot> {
                // Its command buffer is rerecorded every submission.
                let info = vk::CommandPoolCreateInfo::default()
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                    .queue_family_index(family.index);
                let pool = unsafe { device.create_command_pool(&info, None)? };
                let allocate = vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);
                let cmd = match unsafe { device.allocate_command_buffers(&allocate) } {
                    Ok(cmd) => cmd[0],
                    Err(err) => {
                        unsafe { device.destroy_command_pool(pool, None) };
                        return Err(err.into());
                    }
                };
                Ok(Slot {
                    pool,
                    cmd,
                    barrier_log: Vec::new(),
                    signalled: 0,
                })
            })();
            match made {
                Ok(slot) => slots.push(slot),
                Err(err) => {
                    destroy_pools(&slots);
                    lock(&LIVE)[index] = false;
                    return Err(err);
                }
            }
        }
        let slots = slots.into_boxed_slice();
        let ring = (staging > 0 && !Ctx::features_ref().rebar).then(|| Staging::new(index, staging));
        let staging = match ring.transpose() {
            Ok(staging) => staging,
            Err(err) => {
                destroy_pools(&slots);
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
                staging,
                _live: live,
            },
            _kind: PhantomData,
        })
    }

    /// An uploader of its ring for data this timeline's family uses; clone it for every thread
    /// that uploads.
    pub fn uploader(&self) -> Uploader {
        Uploader::new(self.inner.staging.clone(), K::KIND)
    }

    /// An uploader of its ring for data the family of kind `D` uses: a drain on this timeline
    /// releases the images it copied into to `D`. `MissingFeature` where lava has no `D`.
    pub fn uploader_for<D: QueueKind>(&self) -> Result<Uploader> {
        D::check()?;
        Ok(Uploader::new(self.inner.staging.clone(), D::KIND))
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
        if poll(inner.index)? < signalled {
            let _span = tracing::info_span!("wait for frame slot").entered();
            wait_value(inner.index, signalled)?;
        }
        poll_all()?;
        lifetime::run_ready();
        let pool = inner.slots[slot].pool;
        unsafe { Ctx::device().reset_command_pool(pool, vk::CommandPoolResetFlags::empty()) }
            .map_err(vk_error)?;
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

    /// Waits for `wait` at its stage, one of this family's ([`StageOf`]; masked by the device's
    /// features). A wait on its own timeline is dropped; of two on one timeline the later value
    /// is kept, at the stages of both.
    pub fn wait_for<S: StageOf<K>>(&mut self, wait: Wait<S>) -> &mut Self {
        let stage = wait.stage.mask(Ctx::features_ref());
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
        let staging = inner.staging.as_deref();
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
            staging,
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
        let prev = CALLSITE.replace(None);
        let was_recording = lifetime::RECORDING.replace(true);
        let recorded = std::panic::catch_unwind(AssertUnwindSafe(|| record(&mut cmd)));
        lifetime::RECORDING.set(was_recording);
        CALLSITE.set(prev);

        let rec = &mut cmd.rec;
        let unpresented = (rec.acquired.iter()).any(|(image, _)| !rec.presents.iter().any(|p| p.image == *image));
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
            unsafe { device.queue_submit2(*lock(queue), &[info], vk::Fence::null()) }
                .map_err(vk_error)
        };
        /// Why a recording didn't reach the GPU.
        enum Failed {
            Panicked(Box<dyn std::any::Any + Send>),
            Unpresented,
            Error(Error),
        }
        let result = match recorded {
            Err(panic) => Err(Failed::Panicked(panic)),
            Ok(()) if unpresented => Err(Failed::Unpresented),
            Ok(()) => {
                let barrier = [vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                    .dst_access_mask(vk::AccessFlags2::HOST_READ)];
                let info = vk::DependencyInfo::default().memory_barriers(&barrier);
                unsafe { device.cmd_pipeline_barrier2(rec.handle, &info) };
                unsafe { device.end_command_buffer(rec.handle) }
                    .map_err(vk_error)
                    .and_then(|()| submit(rec.handle))
                    .map_err(Failed::Error)
            }
        };
        if result.is_err() {
            if let Some(staging) = staging {
                staging.requeue(value);
            }
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
                if !matches!(err, Error::DeviceLost) {
                    tracing::error!(
                        %err,
                        "a recording that failed couldn't submit its barriers either: the image states lava tracks for it may be wrong"
                    );
                }
                // Nothing signals its value, which the next `begin` of its slot and the
                // timeline's drop wait for.
                host_signal(i, state.submitted, value);
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
            Err(Failed::Error(err)) => Err(err),
            Err(Failed::Panicked(panic)) => std::panic::resume_unwind(panic),
            Err(Failed::Unpresented) => {
                panic!("a submission that uses an acquired swapchain image must present it")
            }
        }
    }
}

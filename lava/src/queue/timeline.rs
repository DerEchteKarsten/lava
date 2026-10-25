//! Timelines: up to `MAX_TIMELINES` timeline semaphores, each owned by one `Timeline` at a time and pinned to one queue with a command pool per frame in flight and an optional staging ring (`uploader()`), and the submission of their recordings (after the last sparse binding; with `submit_profiled` inside a hardware counter query, and `capture` once per counter pass) and sparse bindings (`submit_sparse`, ordered with them by the semaphore), the base value below which a new timeline of an index ignores its former owners' tracked accesses, and the reads of every acceleration structure; what each submission keeps alive until it completed (`keep_until`, and `keep_until_idle` for what no recording tracks), released by `begin`, waits, `reclaim` and `wait_idle`; values no submission signals (a failed replay) are signalled from the host, device loss comes out of polls and waits, and `shutdown` destroys the semaphores
use std::{
    fmt,
    marker::PhantomData,
    panic::AssertUnwindSafe,
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
    command_buffer::{
        Barriers, CommandBuffer, Recording,
        access_state::{StructureAccesses, Tracker},
    },
    error::{Error, Result},
    global::Live,
    object::Object,
    queue::{FamilyInfo, FamilyQueues, Kind, MAX_TIMELINES, QueueKind, StageOf, Wait, lock},
    state::{CALLSITE, Ctx, vk_error},
};
#[cfg(feature = "performance-query")]
use crate::{
    command_buffer::Profile,
    query::{Performance, QueryPool},
    queue::Graphics,
};

/// One of the `MAX_TIMELINES` timeline semaphores. Its values go on across the timelines that
/// own it one after the other, so a `Wait` on it stays valid.
pub(crate) struct Index {
    /// Its timeline semaphore's raw handle, 0 before the first timeline of the index (and
    /// after `shutdown`).
    semaphore: AtomicU64,
    /// The value of the recording in progress, 0 for none: what lava keeps objects no
    /// recording tracks alive for (`keep_until_idle`) besides the submitted values.
    recording: AtomicU64,
    completed: AtomicU64,
    /// The last value of the timelines that owned the index before its current one: accesses
    /// objects keep of it up to there are theirs, on another queue maybe, so its current
    /// timeline ignores them (a semaphore wait orders them).
    base: AtomicU64,
    /// `state`'s `submitted`, readable while a recording holds `state`: the value of its last
    /// submission, which host access checks writes against.
    submitted: AtomicU64,
    pub(crate) state: Mutex<IndexState>,
    /// What its submissions keep alive, by the value that completes them (`keep_until`).
    kept: Mutex<Vec<(u64, Vec<Kept>)>>,
}

/// What a submission keeps alive until it completed: the objects its recording named, and
/// what lava destroys only once nothing in flight can use it.
#[allow(dead_code, reason = "held until the submission completed")]
pub(crate) enum Kept {
    Object(Arc<dyn Object>),
    Other(Arc<dyn std::any::Any + Send + Sync>),
}

/// Keeps `kept` until timeline `i` completed `value`.
pub(crate) fn keep_until(i: usize, value: u64, kept: Vec<Kept>) {
    if kept.is_empty() {
        return;
    }
    lock(&INDICES[i].kept).push((value, kept));
}

/// Keeps `object` until every recording in progress and every submission so far completed:
/// what no recording tracks (a reloaded pass's pipelines, a freed sampler slot, unbound sparse
/// pages). Dropped right away when nothing is in flight.
pub(crate) fn keep_until_idle(object: Arc<dyn std::any::Any + Send + Sync>) {
    for (i, index) in INDICES.iter().enumerate() {
        let latest = (index.recording.load(Ordering::SeqCst)).max(index.submitted());
        if latest > completed(i) {
            keep_until(i, latest, vec![Kept::Other(object.clone())]);
        }
    }
}

/// Runs `destroy` once nothing in flight can use what it destroys (`keep_until_idle`).
pub(crate) fn destroy_when_idle(destroy: impl FnOnce() + Send + 'static) {
    struct OnDrop(Mutex<Option<Box<dyn FnOnce() + Send>>>);
    impl Drop for OnDrop {
        fn drop(&mut self) {
            if let Some(destroy) = lock(&self.0).take() {
                destroy();
            }
        }
    }
    keep_until_idle(Arc::new(OnDrop(Mutex::new(Some(Box::new(destroy))))));
}

/// Drops what timeline `i`'s completed submissions kept, as far as the last poll knows; outside
/// the lock, since destructors keep things too.
pub(crate) fn release(i: usize) {
    let done = completed(i);
    let released: Vec<(u64, Vec<Kept>)> = lock(&INDICES[i].kept)
        .extract_if(.., |(value, _)| *value <= done)
        .collect();
    drop(released);
}

/// `release` on every timeline.
pub(crate) fn release_completed() {
    for i in 0..MAX_TIMELINES {
        release(i);
    }
}

/// Drops everything every timeline keeps: `shutdown`, once the device is idle (or lost).
pub(crate) fn release_all() {
    for index in INDICES.iter() {
        let released = std::mem::take(&mut *lock(&index.kept));
        drop(released);
    }
}

/// Polls every timeline and drops what their completed submissions kept alive: objects the
/// application dropped while submissions used them are destroyed then. `begin`, `Wait::block`
/// and [`wait_idle`](crate::wait_idle) do it too.
pub fn reclaim() {
    let _call = crate::state::call();
    let _ = poll_all();
    release_completed();
}

thread_local! {
    /// The thread records a submission: presents and acquires, whose semaphores its own
    /// submission would signal, panic there.
    pub(crate) static RECORDING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

impl Index {
    pub(crate) fn base(&self) -> u64 {
        self.base.load(Ordering::Acquire)
    }

    pub(crate) fn submitted(&self) -> u64 {
        self.submitted.load(Ordering::Acquire)
    }
}

impl IndexState {
    fn set_submitted(&mut self, index: &Index, value: u64) {
        self.submitted = value;
        index.submitted.store(value, Ordering::Release);
    }
}

#[derive(Default)]
pub(crate) struct IndexState {
    /// Its reads of every acceleration structure and the structure writes they wait for.
    pub(crate) structures: StructureAccesses,
    submitted: u64,
}

pub(crate) static INDICES: LazyLock<[Index; MAX_TIMELINES]> = LazyLock::new(|| {
    std::array::from_fn(|_| Index {
        semaphore: AtomicU64::new(0),
        recording: AtomicU64::new(0),
        completed: AtomicU64::new(0),
        base: AtomicU64::new(0),
        submitted: AtomicU64::new(0),
        state: Mutex::default(),
        kept: Mutex::new(Vec::new()),
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
        let value =
            unsafe { Ctx::device().get_semaphore_counter_value(semaphore) }.map_err(vk_error)?;
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
/// recording that never reached the GPU, whose barrier-only replay failed too), so what waits
/// for it goes on. Waits for its earlier submissions (up to `submitted`) first, which a
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
        Err(err) => {
            tracing::error!(%err, "failed to signal a timeline value no submission signals")
        }
    }
}

/// Destroys every timeline semaphore at `shutdown`, once no timeline exists. Their values go
/// on in the next context, which starts each semaphore at its index's last submitted value, so
/// `Wait`s of this one count as done there.
///
/// # Safety
/// No timeline exists and nothing polls (lava is closed).
pub(crate) unsafe fn destroy() {
    for (i, index) in INDICES.iter().enumerate() {
        if let Some(semaphore) = semaphore_of(i) {
            unsafe { Ctx::device().destroy_semaphore(semaphore, None) };
            index.semaphore.store(0, Ordering::Release);
        }
        let mut state = lock(&index.state);
        state.structures = StructureAccesses::default();
        let submitted = state.submitted;
        index.completed.fetch_max(submitted, Ordering::AcqRel);
        index.base.fetch_max(submitted, Ordering::AcqRel);
    }
    *lock(&LIVE) = [false; MAX_TIMELINES];
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
    /// The value of its last sparse binding, which the next submission waits for: binds aren't
    /// ordered with command buffers even on one queue, and a later value mustn't be signalled
    /// first.
    sparse_wait: u64,
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
            Err(err) => {
                tracing::error!(%err, "failed to wait for a dropped timeline's submissions")
            }
        }
        for slot in &self.slots {
            unsafe { Ctx::device().destroy_command_pool(slot.pool, None) };
        }
        if let Some(staging) = self.staging.take() {
            staging.close();
        }
        {
            // The next timeline of the index ignores what objects keep of this one's accesses.
            let mut state = lock(&index.state);
            state.structures = StructureAccesses::default();
            index.base.fetch_max(state.submitted, Ordering::AcqRel);
        }
        lock(&self.family.timelines)[self.queue] -= 1;
        lock(&LIVE)[self.index] = false;
        // Its submissions completed: what they kept goes (on a lost device too).
        let released = std::mem::take(&mut *lock(&index.kept));
        drop(released);
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
        let family = (Ctx::get().queues.family(K::KIND))
            .expect("lava has queues of every family whose feature is on");
        let device = Ctx::device();
        let index = {
            let mut live = lock(&LIVE);
            let index = live
                .iter()
                .position(|live| !live)
                .ok_or(Error::OutOfTimelines)?;
            if semaphore_of(index).is_none() {
                // Values go on from what the index's earlier semaphores reached.
                let start = INDICES[index].submitted().max(completed(index));
                let mut timeline = vk::SemaphoreTypeCreateInfo::default()
                    .semaphore_type(vk::SemaphoreType::TIMELINE)
                    .initial_value(start);
                let info = vk::SemaphoreCreateInfo::default().push_next(&mut timeline);
                let semaphore =
                    unsafe { device.create_semaphore(&info, None) }.map_err(vk_error)?;
                use vk::Handle;
                INDICES[index]
                    .semaphore
                    .store(semaphore.as_raw(), Ordering::Release);
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
        let ring =
            (staging > 0 && !Ctx::features_ref().rebar).then(|| Staging::new(index, staging));
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
            indices: [Kind::Graphics, Kind::Compute, Kind::Transfer].map(|kind| {
                queues
                    .family(kind)
                    .map_or(vk::QUEUE_FAMILY_IGNORED, |f| f.index)
            }),
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
                sparse_wait: 0,
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

    /// Submits a sparse binding through `bind` on the timeline's queue, after every earlier
    /// submission of the timeline (its wait, the timeline's last value) and signalling its
    /// next value, which the next submission waits for. `bind` gets the queue, the wait (none
    /// before the first submission) and the signal.
    #[cfg(feature = "sparse-binding")]
    pub(crate) fn submit_sparse(
        &mut self,
        bind: impl FnOnce(vk::Queue, Option<(vk::Semaphore, u64)>, (vk::Semaphore, u64)) -> Result<()>,
    ) -> Result<Wait> {
        let inner = &mut self.inner;
        let i = inner.index;
        let index = &INDICES[i];
        let mut state = lock(&index.state);
        let value = state.submitted + 1;
        let wait = (state.submitted > 0).then_some((semaphore(i), state.submitted));
        let queue = &inner.family.queues[inner.queue];
        bind(*lock(queue), wait, (semaphore(i), value))?;
        state.set_submitted(index, value);
        inner.sparse_wait = value;
        Ok(Wait {
            timeline: i as u8,
            value,
            stage: Default::default(),
        })
    }

    /// Its frame slots: how many submissions can be in flight.
    pub fn in_flight(&self) -> u32 {
        self.inner.slots.len() as u32
    }

    /// Its index among the timeline semaphores, which a later timeline may reuse.
    pub(crate) fn index(&self) -> usize {
        self.inner.index
    }

    /// Waits until the next frame slot's last submission finished and destroys what the
    /// completed submissions kept alive and nothing else holds. The submission borrows the
    /// timeline, so it has one at a time.
    #[validation_trace]
    pub fn begin(&mut self) -> Result<Submission<'_, K>> {
        let inner = &mut self.inner;
        let slot = (inner.begun % inner.slots.len() as u64) as usize;
        inner.begun += 1;
        // One poll of every timeline, this one's slot included, then a wait only if it is behind.
        poll_all()?;
        let signalled = inner.slots[slot].signalled;
        if completed(inner.index) < signalled {
            let _span = tracing::info_span!("wait for frame slot").entered();
            wait_value(inner.index, signalled)?;
        }
        release_completed();
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
    /// Its frame slot, which per-frame resources (rings, query pools, a [`PerFrame`](super::PerFrame))
    /// index by: `begin` waited for its last submission, so nothing on the GPU reads them any more.
    pub fn slot(&self) -> usize {
        self.slot
    }

    /// The timeline's index and frame slot count, which a `PerFrame` checks.
    pub(crate) fn timeline(&self) -> (usize, usize) {
        (self.inner.index, self.inner.slots.len())
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

    /// Records `record` and submits it after its waits, signalling the timeline's next value;
    /// the objects its commands name live until it completed. A recording that doesn't reach
    /// the GPU is replaced by a submission of just its barriers, then its panic continues or
    /// its error is returned.
    #[validation_trace]
    pub fn submit(self, record: impl FnOnce(&mut CommandBuffer<K>)) -> Result<Wait> {
        self.submit_inner(None, record)
    }

    /// `submit`, inside query `index` of `profile`'s pool in counter pass `pass` when it is
    /// given, begun as the recording's first command and ended as its last.
    fn submit_inner(
        self,
        profile: ProfileArg<'_>,
        record: impl FnOnce(&mut CommandBuffer<K>),
    ) -> Result<Wait> {
        let inner = self.inner;
        let i = inner.index;
        let index = &INDICES[i];
        let device = Ctx::device();
        let staging = inner.staging.as_deref();
        let slot = &mut inner.slots[self.slot];
        let mut state = lock(&index.state);
        let state = &mut *state;
        let value = state.submitted + 1;
        // Before any lookup the recording makes: what lava retires meanwhile waits for it.
        index.recording.store(value, Ordering::SeqCst);
        let tracker = Tracker {
            timeline: i as u8,
            value,
            base: index.base(),
            completed: completed(i),
        };
        let mut cmd = CommandBuffer::<K>::new(Recording::new(
            slot.cmd,
            inner.info,
            &mut state.structures,
            tracker,
            &mut slot.barrier_log,
            staging,
        ));
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { device.begin_command_buffer(slot.cmd, &begin_info)? };
        let prev = CALLSITE.replace(None);
        let was_recording = RECORDING.replace(true);
        // Counters of command-buffer scope need their query's begin as the first command.
        let begun = std::panic::catch_unwind(AssertUnwindSafe(|| {
            #[cfg(feature = "performance-query")]
            if let Some((pool, index, pass)) = profile {
                cmd.rec.profile = Some(Profile {
                    pool: pool.handle,
                    index,
                    pass,
                });
                cmd.rec.profile_query(pool, index, true);
            }
            let _ = &profile;
        }));
        Bindless::bind(slot.cmd, K::KIND);
        let recorded =
            begun.and_then(|()| std::panic::catch_unwind(AssertUnwindSafe(|| record(&mut cmd))));
        RECORDING.set(was_recording);
        CALLSITE.set(prev);

        let rec = &mut cmd.rec;
        let unpresented =
            (rec.acquired.iter()).any(|(image, _)| !rec.presents.iter().any(|p| p.image == *image));
        let binary_waits: SmallVec<[_; 1]> = (rec.presents.iter())
            .map(|p| {
                let first = rec.acquired.iter().find(|(image, _)| *image == p.image);
                (
                    p.acquire,
                    first.map_or(vk::PipelineStageFlags2::ALL_COMMANDS, |(_, stage)| *stage),
                )
            })
            .collect();
        let presents: SmallVec<[_; 1]> = rec.presents.iter().map(|p| p.present).collect();
        let queue = &inner.family.queues[inner.queue];
        // After the timeline's last sparse binding, which nothing else orders it after.
        let sparse_wait = std::mem::take(&mut inner.sparse_wait);
        let own_wait = (sparse_wait > completed(i)).then_some((
            semaphore(i),
            sparse_wait,
            vk::PipelineStageFlags2::ALL_COMMANDS,
        ));
        // The counter pass of a profiled recording; a replay records no queries.
        let submit = |cmd: vk::CommandBuffer, pass: Option<u32>| -> Result<()> {
            let buffers = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
            let waits: SmallVec<[_; 4]> = (self.waits.iter())
                .map(|&(t, value, stages)| (semaphore(t), value, stages))
                .chain(own_wait)
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
            let mut info = vk::SubmitInfo2::default()
                .command_buffer_infos(&buffers)
                .wait_semaphore_infos(&waits)
                .signal_semaphore_infos(&signals);
            let mut performance = vk::PerformanceQuerySubmitInfoKHR::default();
            if let Some(pass) = pass {
                performance = performance.counter_pass_index(pass);
                info = info.push_next(&mut performance);
            }
            unsafe { device.queue_submit2(*lock(queue), &[info], vk::Fence::null()) }
                .map_err(vk_error)
        };
        /// Why a recording didn't reach the GPU.
        enum Failed {
            Panicked(Box<dyn std::any::Any + Send>),
            Unpresented,
            OpenQuery(u32),
            ProfileEnded,
            Error(Error),
        }
        let profiled = rec.profile;
        let result = match recorded {
            Err(panic) => Err(Failed::Panicked(panic)),
            Ok(()) if unpresented => Err(Failed::Unpresented),
            Ok(()) if let Some(index) = rec.left_open() => Err(Failed::OpenQuery(index)),
            Ok(())
                if profiled.is_some_and(|p| {
                    !rec.open_queries
                        .iter()
                        .flatten()
                        .any(|&q| q == (p.pool, p.index))
                }) =>
            {
                Err(Failed::ProfileEnded)
            }
            Ok(()) => {
                let barrier = [vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                    .dst_access_mask(vk::AccessFlags2::HOST_READ)];
                let info = vk::DependencyInfo::default().memory_barriers(&barrier);
                unsafe { device.cmd_pipeline_barrier2(rec.handle, &info) };
                // The profile's query ends as the last command.
                #[cfg(feature = "performance-query")]
                if let Some((pool, index, _)) = profile {
                    rec.profile_query(pool, index, false);
                }
                unsafe { device.end_command_buffer(rec.handle) }
                    .map_err(vk_error)
                    .and_then(|()| submit(rec.handle, profiled.map(|p| p.pass)))
                    .map_err(Failed::Error)
            }
        };
        if result.is_err() {
            if let Some(staging) = staging {
                staging.requeue(value);
            }
            let replayed = (|| -> Result<()> {
                unsafe {
                    device.reset_command_pool(slot.pool, vk::CommandPoolResetFlags::empty())?;
                    let info = vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                    device.begin_command_buffer(rec.handle, &info)?;
                    // An acquired image without its acquire wait keeps its layout; the surface
                    // resets its state when it keeps it current or acquires it again.
                    let presented =
                        |image: vk::Image| rec.presents.iter().any(|p| p.image == image);
                    for barriers in rec.barrier_log.iter_mut() {
                        barriers.images.retain(|b| {
                            !rec.acquired
                                .iter()
                                .any(|&(i, _)| i == b.image && !presented(i))
                        });
                    }
                    for barriers in rec.barrier_log.iter() {
                        rec.pipeline_barrier(barriers);
                    }
                    device.end_command_buffer(rec.handle)?;
                }
                submit(rec.handle, None)
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
        #[cfg(feature = "acceleration-structure")]
        cmd.rec.finish_structure_builds(result.is_ok());
        // Its tracked accesses go into their objects before its value counts as submitted,
        // which host access checks against.
        cmd.rec.publish_accesses();
        // Submitted, or replayed or signalled from the host: its value completes it.
        let kept = cmd.rec.take_kept();
        drop(cmd);
        slot.barrier_log.clear();
        slot.signalled = value;
        state.set_submitted(index, value);
        index.recording.store(0, Ordering::SeqCst);
        keep_until(i, value, kept);
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
            Err(Failed::OpenQuery(index)) => panic!(
                "a recording ended with query {index} still active: end every query it begins"
            ),
            Err(Failed::ProfileEnded) => {
                panic!("the counter query around a profiled recording was ended inside it")
            }
        }
    }
}

/// The counter pool, query and pass of `Submission::submit_profiled` and `Timeline::capture`.
#[cfg(feature = "performance-query")]
type ProfileArg<'a> = Option<(&'a QueryPool<Performance>, u32, u32)>;
#[cfg(not(feature = "performance-query"))]
type ProfileArg<'a> = Option<&'a std::convert::Infallible>;

#[cfg(feature = "performance-query")]
impl Submission<'_, Graphics> {
    /// [`Self::submit`], inside query `index` of the hardware counter pool `pool`, begun as the
    /// recording's first command and ended as its last: what counters of
    /// [`CounterScope::CommandBuffer`](crate::query::CounterScope::CommandBuffer) need, and
    /// any others count the whole recording this way too. Observers see the query as active.
    /// Panics if `pool`'s counters take more than one pass ([`Timeline::capture`] records those).
    #[validation_trace]
    pub fn submit_profiled(
        self,
        pool: &QueryPool<Performance>,
        index: u32,
        record: impl FnOnce(&mut CommandBuffer<Graphics>),
    ) -> Result<Wait> {
        assert!(
            index < pool.count(),
            "query {index} of a pool of {}",
            pool.count()
        );
        assert!(
            pool.passes() == 1,
            "the counters of this pool take {} submissions: record them with `Timeline::capture`",
            pool.passes()
        );
        self.submit_inner(Some((pool, index, 0)), record)
    }
}

#[cfg(feature = "performance-query")]
impl Timeline<Graphics> {
    /// A capture of hardware counters for tools: records `record` once per counter pass that
    /// `pool`'s selection needs ([`QueryPool::passes`], also more than [`counters::fits`]
    /// allows), each time in a submission of its own inside query `index` (as
    /// [`Submission::submit_profiled`] records it) counting that pass's counters, and blocks
    /// until each finished before recording the next; then `pool.read(index, ..)` has every
    /// counter. The recordings should be alike (a frame's commands, recorded again), and present
    /// nothing, since a frame presents once. Returns the last pass's `Wait`.
    ///
    /// [`counters::fits`]: crate::query::counters::fits
    #[validation_trace]
    pub fn capture(
        &mut self,
        pool: &QueryPool<Performance>,
        index: u32,
        mut record: impl FnMut(&mut CommandBuffer<Graphics>),
    ) -> Result<Wait> {
        assert!(
            index < pool.count(),
            "query {index} of a pool of {}",
            pool.count()
        );
        let mut last = None;
        for pass in 0..pool.passes() {
            let wait = self
                .begin()?
                .submit_inner(Some((pool, index, pass)), &mut record)?;
            wait.block()?;
            last = Some(wait);
        }
        Ok(last.expect("a selection takes at least one pass"))
    }
}

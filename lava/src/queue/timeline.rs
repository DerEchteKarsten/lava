//! Timelines: up to `MAX_TIMELINES` timeline semaphores, each owned by one `Timeline` at a time and pinned to one queue with a command pool per frame in flight; the pending accesses of its submissions, reserving values, beginning a submission once its frame slot is free, and the order values are signalled in, also when reservations are dropped unsubmitted
use std::{
    collections::BTreeSet,
    fmt,
    marker::PhantomData,
    range::Range,
    sync::{
        Arc, LazyLock, Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

use crate::vk;
use lava_macros::validation_trace;
use smallvec::SmallVec;

use crate::{
    command_buffer::{Barriers, CommandBuffer, pending::PendingAccesses},
    error::{Error, Result},
    lifetime,
    queue::{
        MAX_TIMELINES, QueueKind, Wait,
        device::{FamilyQueues, QueueHandle},
        lock, submit,
    },
    state::Ctx,
};

/// One of the `MAX_TIMELINES` timeline semaphores. Its values go on across the timelines that
/// own it one after the other, so a `Wait` on it stays valid.
pub(crate) struct Index {
    /// Created by the first timeline that owns it, kept for lava's lifetime.
    semaphore: OnceLock<vk::Semaphore>,
    /// The latest value reserved: what lifetime stamps hold.
    pub(crate) reserved: AtomicU64,
    /// Every value up to it signalled, as of the last poll.
    completed: AtomicU64,
    pub(crate) state: Mutex<IndexState>,
}

/// Locked while a recording of the timeline records and submits, so its submissions reach the
/// queue in the order their barriers were computed in.
#[derive(Default)]
pub(crate) struct IndexState {
    /// What its submissions accessed that later ones may have to wait for, tagged with their
    /// values.
    pub(crate) pending: PendingAccesses,
    pub(crate) order: Order,
}

pub(crate) static INDICES: LazyLock<[Index; MAX_TIMELINES]> = LazyLock::new(|| {
    std::array::from_fn(|_| Index {
        semaphore: OnceLock::new(),
        reserved: AtomicU64::new(0),
        completed: AtomicU64::new(0),
        state: Mutex::default(),
    })
});

/// Which indices a timeline owns.
static LIVE: Mutex<[bool; MAX_TIMELINES]> = Mutex::new([false; MAX_TIMELINES]);

/// The semaphore of index `i`, which a timeline owned.
pub(crate) fn semaphore(i: usize) -> vk::Semaphore {
    *INDICES[i]
        .semaphore
        .get()
        .expect("a value of a timeline that existed")
}

/// Every value of index `i` up to this one completed, as of the last poll.
pub(crate) fn completed(i: usize) -> u64 {
    INDICES[i].completed.load(Ordering::Acquire)
}

/// Reads index `i`'s semaphore and returns what completed.
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

/// Polls every timeline and the presents' fences.
pub(crate) fn poll_all() {
    for i in 0..MAX_TIMELINES {
        poll(i);
    }
    lifetime::poll_presents();
}

/// Waits on the host until `value` of index `i` is signalled.
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

/// Forgets the pending accesses of an image that is being destroyed, on every timeline, so an
/// image that gets its handle next starts with nothing pending. Locks one timeline at a time.
pub(crate) fn forget_image(image: vk::Image) {
    for index in INDICES.iter() {
        lock(&index.state).pending.forget_image(image);
    }
}

/// Like [`forget_image`], for the addresses of a buffer that is being destroyed.
pub(crate) fn forget_buffer(range: Range<u64>) {
    for index in INDICES.iter() {
        lock(&index.state).pending.forget_buffer(range.start..range.end);
    }
}

/// The order a timeline's values are signalled in: each by the submission of its reservation,
/// in reservation order, or, for one dropped unsubmitted, together with the value before it.
#[derive(Debug, Default)]
pub(crate) struct Order {
    /// Every value up to it is signalled by a submission made already.
    pub(crate) submitted: u64,
    /// Values dropped unsubmitted while one before them wasn't submitted yet.
    abandoned: BTreeSet<u64>,
}

impl Order {
    /// Value `n`, the next one, is submitted now: returns the value its submission signals,
    /// `n` or the last of the dropped values right after it.
    pub(crate) fn submit(&mut self, n: u64) -> u64 {
        debug_assert_eq!(n, self.submitted + 1, "values are submitted in order");
        let mut signal = n;
        while self.abandoned.remove(&(signal + 1)) {
            signal += 1;
        }
        self.submitted = signal;
        signal
    }

    /// Value `n` was dropped unsubmitted: the value to signal right away if it is the next one,
    /// else `None`, and it is signalled with the submission before it.
    pub(crate) fn abandon(&mut self, n: u64) -> Option<u64> {
        if n == self.submitted + 1 {
            Some(self.submit(n))
        } else {
            self.abandoned.insert(n);
            None
        }
    }
}

/// A timeline's command pool and buffer for one frame in flight, and the barrier log it lends
/// each recording.
pub(crate) struct SlotRes {
    pool: vk::CommandPool,
    pub(crate) cmd: vk::CommandBuffer,
    pub(crate) barrier_log: Vec<Barriers>,
}

impl SlotRes {
    fn new(family: u32) -> Result<Self> {
        let info = vk::CommandPoolCreateInfo::default().queue_family_index(family);
        let pool = unsafe { Ctx::device().create_command_pool(&info, None)? };
        let allocate = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        match unsafe { Ctx::device().allocate_command_buffers(&allocate) } {
            Ok(buffers) => Ok(Self {
                pool,
                cmd: buffers[0],
                barrier_log: Vec::new(),
            }),
            Err(err) => {
                unsafe { Ctx::device().destroy_command_pool(pool, None) };
                Err(err.into())
            }
        }
    }

    pub(crate) fn reset(&self) -> Result<()> {
        let flags = vk::CommandPoolResetFlags::empty();
        unsafe { Ctx::device().reset_command_pool(self.pool, flags)? };
        Ok(())
    }
}

/// What the owners of a timeline share: its index, queue and frame slots. The last one waits
/// for its submissions and frees the index.
pub(crate) struct Inner {
    pub(crate) index: usize,
    pub(crate) family: &'static FamilyQueues,
    queue: usize,
    in_flight: u32,
    pub(crate) slots: Box<[Mutex<SlotRes>]>,
}

impl Inner {
    /// The queue it is pinned to.
    pub(crate) fn queue(&self) -> &'static QueueHandle {
        &self.family.queues[self.queue]
    }
}

impl fmt::Debug for Inner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Timeline")
            .field("index", &self.index)
            .field("kind", &self.family.kind)
            .field("queue", &self.queue)
            .field("in_flight", &self.in_flight)
            .finish()
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        let index = &INDICES[self.index];
        // Every value was submitted or dropped, which signals it.
        let submitted = lock(&index.state).order.submitted;
        if let Err(err) = wait_value(self.index, submitted) {
            tracing::error!(%err, "failed to wait for a dropped timeline's submissions");
        }
        for slot in &self.slots {
            unsafe { Ctx::device().destroy_command_pool(lock(slot).pool, None) };
        }
        // Done: nothing waits for its accesses any more.
        lock(&index.state).pending = PendingAccesses::default();
        self.family.unpin(self.queue);
        lock(&LIVE)[self.index] = false;
        lifetime::reclaim();
    }
}

/// A value a reservation holds until it is submitted, or dropped, which signals it in order.
pub(crate) struct Ticket {
    pub(crate) inner: Arc<Inner>,
    pub(crate) value: u64,
    /// A submission took over signalling it.
    pub(crate) done: bool,
}

impl Drop for Ticket {
    fn drop(&mut self) {
        if !self.done {
            abandon(&self.inner, self.value);
        }
    }
}

/// Value `n` of `inner`'s timeline won't be submitted: it is signalled right away if it is the
/// next one, else with the submission before it.
fn abandon(inner: &Inner, n: u64) {
    let mut state = lock(&INDICES[inner.index].state);
    if let Some(signal) = state.order.abandon(n) {
        submit::signal_empty(inner, n, signal);
    }
}

/// A timeline: submissions to one queue of family `K`, signalled in the order their values
/// were reserved, with `in_flight` frame slots (command pools). A value is reserved
/// ([`Self::reserve`], before reading anything its work will reach), begun ([`Reservation::begin`],
/// which waits until the submission `in_flight` values before it finished) and submitted
/// ([`Submission::submit`]). At most `MAX_TIMELINES` exist at once; each takes the queue of its
/// family the fewest timelines share. Dropping the last handle waits for its submissions.
pub struct Timeline<K: QueueKind> {
    inner: Arc<Inner>,
    _kind: PhantomData<fn() -> K>,
}

impl<K: QueueKind> fmt::Debug for Timeline<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}

/// How [`Timeline::builder`] makes a timeline.
#[derive(Debug)]
#[must_use]
pub struct TimelineBuilder<K: QueueKind> {
    in_flight: u32,
    _kind: PhantomData<fn() -> K>,
}

impl<K: QueueKind> TimelineBuilder<K> {
    /// How many submissions may run at once (2 by default): value `n` begins once value
    /// `n - in_flight` finished, and `Submission::slot` cycles through them.
    pub fn in_flight(mut self, in_flight: u32) -> Self {
        assert!(in_flight > 0, "a timeline has at least one frame slot");
        self.in_flight = in_flight;
        self
    }

    /// `MissingFeature` without a family of `K`, `OutOfTimelines` when `MAX_TIMELINES` exist.
    #[validation_trace]
    pub fn build(self) -> Result<Timeline<K>> {
        K::check()?;
        let family = Ctx::get()
            .queues
            .family(K::KIND)
            .expect("lava creates queues of every family whose feature is on");
        let index = {
            let mut live = lock(&LIVE);
            let index = live.iter().position(|live| !live).ok_or(Error::OutOfTimelines)?;
            if INDICES[index].semaphore.get().is_none() {
                let mut timeline = vk::SemaphoreTypeCreateInfo::default()
                    .semaphore_type(vk::SemaphoreType::TIMELINE)
                    .initial_value(0);
                let info = vk::SemaphoreCreateInfo::default().push_next(&mut timeline);
                let semaphore = unsafe { Ctx::device().create_semaphore(&info, None)? };
                let _ = INDICES[index].semaphore.set(semaphore);
            }
            live[index] = true;
            index
        };
        let slots = (0..self.in_flight)
            .map(|_| SlotRes::new(family.index).map(Mutex::new))
            .collect::<Result<Box<[_]>>>();
        let slots = match slots {
            Ok(slots) => slots,
            Err(err) => {
                // Any pool made before the error is leaked; this only fails when out of memory.
                lock(&LIVE)[index] = false;
                return Err(err);
            }
        };
        Ok(Timeline {
            inner: Arc::new(Inner {
                index,
                family,
                queue: family.pin(),
                in_flight: self.in_flight,
                slots,
            }),
            _kind: PhantomData,
        })
    }
}

impl<K: QueueKind> Timeline<K> {
    pub fn builder() -> TimelineBuilder<K> {
        TimelineBuilder {
            in_flight: 2,
            _kind: PhantomData,
        }
    }

    pub fn in_flight(&self) -> u32 {
        self.inner.in_flight
    }

    /// Reserves the next value, without waiting: do it before reading anything the submission
    /// will reach (through its commands, addresses or bindless slots), since what is dropped
    /// after it lives until it finished (`lifetime`). Destroys what finished submissions free.
    pub fn reserve(&self) -> Reservation<K> {
        poll_all();
        lifetime::reclaim();
        let value = INDICES[self.inner.index]
            .reserved
            .fetch_add(1, Ordering::SeqCst)
            + 1;
        Reservation {
            ticket: Ticket {
                inner: self.inner.clone(),
                value,
                done: false,
            },
            _kind: PhantomData,
        }
    }
}

/// A reserved value of a timeline: what was dropped before it lives until its submission
/// finished. Dropping it unsubmitted signals it in order, with nothing to run. Hold it only
/// from the extract to the submission: every deletion waits for it.
#[must_use]
pub struct Reservation<K: QueueKind> {
    ticket: Ticket,
    _kind: PhantomData<fn() -> K>,
}

impl<K: QueueKind> fmt::Debug for Reservation<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Reservation")
            .field("timeline", &self.ticket.inner)
            .field("value", &self.ticket.value)
            .finish()
    }
}

impl<K: QueueKind> Reservation<K> {
    pub fn value(&self) -> u64 {
        self.ticket.value
    }

    /// Waits until the submission `in_flight` values before this one finished, so its frame
    /// slot is free, and destroys what that frees. Panics if that value isn't submitted yet:
    /// the wait could then never end.
    #[validation_trace]
    pub fn begin(self) -> Result<Submission<K>> {
        let Self { ticket, .. } = self;
        let inner = &ticket.inner;
        let (n, in_flight) = (ticket.value, inner.in_flight as u64);
        if n > in_flight {
            let earlier = n - in_flight;
            let submitted = lock(&INDICES[inner.index].state).order.submitted;
            if earlier > submitted {
                drop(ticket);
                panic!(
                    "value {n} of a timeline with {in_flight} in flight begins once value \
                     {earlier} finished, which isn't submitted: submit or drop the reservations \
                     before it first"
                );
            }
            if poll(inner.index) < earlier {
                let _span = tracing::info_span!("wait for frame slot").entered();
                wait_value(inner.index, earlier)?;
            }
        }
        poll_all();
        lifetime::reclaim();
        let slot = (n % in_flight) as usize;
        lock(&inner.slots[slot]).reset()?;
        Ok(Submission {
            ticket,
            slot,
            waits: SmallVec::new(),
            _kind: PhantomData,
        })
    }
}

/// A begun reservation: its frame slot is free. [`Self::submit`] records and submits its work.
#[must_use]
pub struct Submission<K: QueueKind> {
    pub(crate) ticket: Ticket,
    pub(crate) slot: usize,
    /// Other timelines' values it waits for, one per timeline, with the stages that wait.
    pub(crate) waits: SmallVec<[(usize, u64, vk::PipelineStageFlags2); 2]>,
    _kind: PhantomData<fn() -> K>,
}

impl<K: QueueKind> fmt::Debug for Submission<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Submission")
            .field("timeline", &self.ticket.inner)
            .field("value", &self.ticket.value)
            .field("slot", &self.slot)
            .finish()
    }
}

impl<K: QueueKind> Submission<K> {
    pub fn value(&self) -> u64 {
        self.ticket.value
    }

    /// Its frame slot, `value % in_flight`: per-frame resources (rings, query pools) index by it.
    pub fn slot(&self) -> usize {
        self.slot
    }

    /// Makes the submission wait for `wait`, another timeline's, at its stage. A wait on its
    /// own timeline is dropped (its submissions run in order anyway), and of two on one
    /// timeline the later value is kept, at the stages of both.
    pub fn wait_for(&mut self, wait: Wait) -> &mut Self {
        let (timeline, stage) = (wait.timeline as usize, wait.stage().flags());
        if timeline == self.ticket.inner.index {
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

    /// Records `record` into the frame slot's command buffer and submits it to the timeline's
    /// queue, after its waits; it signals the value. Values are submitted in the order they
    /// were reserved (dropped ones count as submitted): an earlier one still unsubmitted
    /// panics.
    ///
    /// A recording that doesn't reach the GPU, because `record` panicked or recording or
    /// submitting failed, still moved the tracked image states and accesses: it is replaced by
    /// a submission of just its barriers with the same waits, which signals the value, so the
    /// tracking stays true. Then the panic continues, or the error is returned.
    #[validation_trace]
    pub fn submit(self, record: impl FnOnce(&mut CommandBuffer<K>)) -> Result<Wait> {
        submit::submit(self, record)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_are_signalled_in_order_with_the_dropped_ones_after_them() {
        let mut order = Order::default();
        assert_eq!(order.submit(1), 1);
        // 3 and 4 are dropped while 2 records: its submission signals them too.
        assert_eq!(order.abandon(3), None);
        assert_eq!(order.abandon(4), None);
        assert_eq!(order.submit(2), 4);
        assert_eq!(order.submitted, 4);
        // A dropped value that is next is signalled right away, with the dropped ones after it.
        assert_eq!(order.abandon(7), None);
        assert_eq!(order.abandon(5), Some(5));
        assert_eq!(order.submit(6), 7);
        assert!(order.abandoned.is_empty());
    }
}

//! Approximate lifetimes: a dropped GPU object is stamped with every timeline's latest reserved value and destroyed, in drop order, once all of them completed (what a present may use also once the presents before it are done); stamps for slots the application frees by hand (`SlotPool`)
//!
//! Lava counts nothing per use. A submission reserves its value (`Timeline::reserve`) before it
//! reads what its commands, addresses or bindless slots will reach. An image, buffer, view,
//! acceleration structure, query pool, pipeline or present object that drops queues its
//! destructor, stamped with the latest value every timeline reserved so far ([`latest_stamp`]):
//! every submission that could still reach it, through a command or an index, reserved one of
//! them or an earlier value. The destructors run in drop order (views before their image, a
//! swapchain before its surface) once every timeline completed its stamp.
//!
//! A held reservation on any timeline holds back every deletion: reserve right before reading.
//! Presents have a fence only with `present-fences`, so without it what a present may still use
//! waits for a later value of the timeline that presented to complete. [`crate::wait_idle`]
//! completes everything submitted before it and runs what that frees.
//!
//! Timelines are polled (`vkGetSemaphoreCounterValue`) when a value is reserved or begun, a
//! `Wait` or `Stamp` is asked, and in [`reclaim`]. Destructors run then, never on a thread that
//! is recording a command buffer.
use std::{cell::Cell, collections::VecDeque, sync::Mutex};

use crate::vk;

use crate::{
    queue::{
        MAX_TIMELINES, lock,
        timeline::{INDICES, completed, poll_all},
    },
    state::Ctx,
};

type Destroy = Box<dyn FnOnce() + Send>;

/// The latest value of every timeline at one moment (`latest_stamp`), done once each of them
/// completed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stamp(pub(crate) [u64; MAX_TIMELINES]);

impl Stamp {
    /// Whether every timeline completed its value: nothing reserved before the stamp runs.
    /// Polls the timelines.
    pub fn is_done(&self) -> bool {
        poll_all();
        self.done_at(&completed_values())
    }

    fn done_at(&self, completed: &[u64; MAX_TIMELINES]) -> bool {
        self.0.iter().zip(completed).all(|(stamp, done)| stamp <= done)
    }
}

/// The latest value every timeline reserved so far: what a slot freed now has to wait for.
pub fn latest_stamp() -> Stamp {
    Stamp(std::array::from_fn(|i| {
        INDICES[i].reserved.load(std::sync::atomic::Ordering::SeqCst)
    }))
}

fn completed_values() -> [u64; MAX_TIMELINES] {
    std::array::from_fn(completed)
}

/// What a destructor waits for besides its stamp.
#[derive(Clone, Copy, Debug, PartialEq)]
enum After {
    /// Its stamp alone.
    Frames,
    /// A present may use the object: the presents made before its drop (with present fences),
    /// else a later value of `timeline`, the last that presented it, or an idle device.
    Presented { timeline: Option<usize> },
}

struct Retired {
    stamp: Stamp,
    after: After,
    /// Counts the destructors in drop order.
    seq: u64,
    destroy: Destroy,
}

struct State {
    /// Destructors in drop order.
    retired: VecDeque<Retired>,
    next_seq: u64,
    /// The destructors before this `seq` were retired before the device was last idle: the
    /// presents they waited for are over.
    idle_seq: u64,
    /// Presents signal fences (`present-fences`).
    fenced: bool,
    /// The fences of the presents that may still be running, oldest first, each with the
    /// `next_seq` of its present: the destructors from that one on were retired after it.
    presents: VecDeque<(u64, vk::Fence)>,
    /// Signalled present fences, reset when they are taken again.
    free_fences: Vec<vk::Fence>,
}

impl State {
    const fn new() -> Self {
        State {
            retired: VecDeque::new(),
            next_seq: 0,
            idle_seq: 0,
            fenced: false,
            presents: VecDeque::new(),
            free_fences: Vec::new(),
        }
    }

    fn retire(&mut self, stamp: Stamp, after: After, destroy: Destroy) {
        let seq = self.next_seq;
        self.next_seq += 1;
        self.retired.push_back(Retired {
            stamp,
            after,
            seq,
            destroy,
        });
    }

    /// The next destructor that may run, given what `completed`: the oldest, once its stamp is
    /// done and, for what a present may use, its presents. A later one never goes first.
    fn pop_ready(&mut self, completed: &[u64; MAX_TIMELINES]) -> Option<Destroy> {
        let front = self.retired.front()?;
        let ready = front.stamp.done_at(completed)
            && match front.after {
                After::Frames => true,
                After::Presented { .. } if self.fenced => {
                    // Every present made before the drop signalled its fence.
                    !(self.presents.front()).is_some_and(|&(seq, _)| seq <= front.seq)
                }
                After::Presented { timeline } => {
                    front.seq < self.idle_seq
                        || timeline.is_none_or(|t| completed[t] > front.stamp.0[t])
                }
            };
        if ready {
            self.retired.pop_front().map(|r| r.destroy)
        } else {
            None
        }
    }

    /// Drops the fences of the oldest presents that `signalled`.
    fn poll_presents(&mut self, signalled: impl Fn(vk::Fence) -> bool) {
        while let Some(&(_, fence)) = self.presents.front() {
            if !signalled(fence) {
                break;
            }
            self.presents.pop_front();
            self.free_fences.push(fence);
        }
    }
}

static STATE: Mutex<State> = Mutex::new(State::new());

thread_local! {
    /// The thread records a command buffer: it holds its timeline's pending accesses, which
    /// destroying an image or buffer locks, so destructors wait for another thread or call.
    static RECORDING: Cell<bool> = const { Cell::new(false) };
}

/// Runs `f` as a recording: destructors this thread would run meanwhile wait.
pub(crate) fn recording<R>(f: impl FnOnce() -> R) -> R {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            RECORDING.set(self.0);
        }
    }
    let _reset = Reset(RECORDING.replace(true));
    f()
}

/// Runs `destroy` once every submission that could still reach the object finished. It never
/// runs right away, also not when nothing is in flight, so it may lock what the caller holds.
pub(crate) fn retire(destroy: impl FnOnce() + Send + 'static) {
    let mut state = lock(&STATE);
    // Stamped under the lock: stamps grow in queue order.
    state.retire(latest_stamp(), After::Frames, Box::new(destroy));
}

/// Like [`retire`], for what a present may use (semaphores, swapchains, surfaces): also waits
/// for the presents made before it, or without present fences for a later value of `timeline`,
/// the last timeline whose submission presented it.
pub(crate) fn retire_presented(timeline: Option<usize>, destroy: impl FnOnce() + Send + 'static) {
    let mut state = lock(&STATE);
    state.retire(latest_stamp(), After::Presented { timeline }, Box::new(destroy));
}

/// Whether presents signal fences (`present-fences`), set by `init`.
pub(crate) fn set_present_fences(on: bool) {
    lock(&STATE).fenced = on;
}

/// A reset fence for a present, signalled ones first.
pub(crate) fn present_fence() -> crate::Result<vk::Fence> {
    let device = Ctx::device();
    if let Some(fence) = lock(&STATE).free_fences.pop() {
        unsafe { device.reset_fences(&[fence])? };
        return Ok(fence);
    }
    Ok(unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? })
}

/// A present that signals `fence` (from [`present_fence`]) was just made: what was dropped
/// before it waits for it. Called while its queue is locked, so `wait_idle` sees it.
pub(crate) fn presented(fence: vk::Fence) {
    let mut state = lock(&STATE);
    let seq = state.next_seq;
    state.presents.push_back((seq, fence));
}

/// Polls the presents' fences.
pub(crate) fn poll_presents() {
    let device = Ctx::device();
    let signalled = |fence| unsafe { device.get_fence_status(fence) }.unwrap_or(false);
    lock(&STATE).poll_presents(signalled);
}

/// Waits until the presents made so far signalled their fences, at most `timeout_ns`; one that
/// doesn't by then (its present failed before it was queued) is given up on, and its fence
/// leaked. Call it while no present can be made (`wait_idle` holds every queue).
pub(crate) fn wait_for_presents(timeout_ns: u64) {
    let device = Ctx::device();
    let fences: Vec<vk::Fence> = lock(&STATE).presents.iter().map(|&(_, f)| f).collect();
    if !fences.is_empty() && unsafe { device.wait_for_fences(&fences, true, timeout_ns) }.is_err() {
        tracing::warn!("a present's fence didn't signal after the device went idle; giving up on it");
        let signalled = |fence| unsafe { device.get_fence_status(fence) } == Ok(true);
        lock(&STATE).presents.retain(|&(_, fence)| signalled(fence));
    }
    poll_presents();
}

/// After the device went idle, while nothing could be submitted or presented: the presents of
/// the objects dropped so far are over. Runs no destructor (call [`reclaim`] after).
pub(crate) fn device_idle() {
    let mut state = lock(&STATE);
    state.idle_seq = state.next_seq;
}

/// Runs every destructor that may run, without holding the lock, since destroying an image or
/// buffer locks the timelines' pending accesses; not while this thread records, which holds
/// one of them.
pub fn reclaim() {
    if RECORDING.get() {
        return;
    }
    let completed = completed_values();
    loop {
        let Some(destroy) = lock(&STATE).pop_ready(&completed) else {
            return;
        };
        destroy();
    }
}

/// Runs every destructor, ready or not: `lava::destroy`, after the device went idle.
pub(crate) fn flush() {
    loop {
        let Some(retired) = lock(&STATE).retired.pop_front() else {
            break;
        };
        (retired.destroy)();
    }
    let device = Ctx::device();
    let mut state = lock(&STATE);
    let mut fences = std::mem::take(&mut state.free_fences);
    fences.extend(state.presents.drain(..).map(|(_, fence)| fence));
    for fence in fences {
        unsafe { device.destroy_fence(fence, None) };
    }
}

/// Indices into something the GPU reads by index (core's material slots), each handed out
/// again only once no submission that may read it runs: a freed index is stamped
/// ([`latest_stamp`]) and reused once its stamp is done, oldest first.
#[derive(Debug)]
pub struct SlotPool {
    capacity: u32,
    /// How many indices were handed out so far.
    next: u32,
    freed: VecDeque<(u32, Stamp)>,
}

impl SlotPool {
    /// Indices `0..capacity`.
    pub fn new(capacity: u32) -> Self {
        Self {
            capacity,
            next: 0,
            freed: VecDeque::new(),
        }
    }

    /// A free index: the oldest freed one no submission may read any more, else a new one;
    /// `None` when every index is in use or may still be read.
    pub fn take(&mut self) -> Option<u32> {
        if !self.freed.is_empty() {
            poll_all();
        }
        self.take_at(&completed_values())
    }

    fn take_at(&mut self, completed: &[u64; MAX_TIMELINES]) -> Option<u32> {
        if let Some(&(index, stamp)) = self.freed.front()
            && stamp.done_at(completed)
        {
            self.freed.pop_front();
            return Some(index);
        }
        (self.next < self.capacity).then(|| {
            self.next += 1;
            self.next - 1
        })
    }

    /// Frees `index`, which submissions reserved so far may still read.
    pub fn free(&mut self, index: u32) {
        self.freed.push_back((index, latest_stamp()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex as StdMutex};

    /// A destructor that logs `name` when it runs.
    fn logged(log: &Arc<StdMutex<Vec<&'static str>>>, name: &'static str) -> Destroy {
        let log = log.clone();
        Box::new(move || log.lock().unwrap().push(name))
    }

    fn run(state: &mut State, completed: [u64; MAX_TIMELINES]) {
        while let Some(destroy) = state.pop_ready(&completed) {
            destroy();
        }
    }

    /// A stamp of values on the first two timelines.
    fn stamp(a: u64, b: u64) -> Stamp {
        let mut stamp = [0; MAX_TIMELINES];
        (stamp[0], stamp[1]) = (a, b);
        Stamp(stamp)
    }

    fn done(a: u64, b: u64) -> [u64; MAX_TIMELINES] {
        stamp(a, b).0
    }

    fn fence(raw: u64) -> vk::Fence {
        use crate::vk::Handle;
        vk::Fence::from_raw(raw)
    }

    #[test]
    fn a_destructor_waits_for_every_timeline_of_its_stamp() {
        let log = Arc::default();
        let mut state = State::new();
        state.retire(stamp(3, 7), After::Frames, logged(&log, "object"));
        run(&mut state, done(3, 6));
        assert!(log.lock().unwrap().is_empty(), "timeline 1 hasn't completed 7");
        run(&mut state, done(2, 9));
        assert!(log.lock().unwrap().is_empty(), "timeline 0 hasn't completed 3");
        run(&mut state, done(3, 7));
        assert_eq!(*log.lock().unwrap(), ["object"]);
    }

    #[test]
    fn destructors_run_in_drop_order() {
        let log = Arc::default();
        let mut state = State::new();
        // A view dropped before its image, which a later stamp holds no shorter.
        state.retire(stamp(5, 0), After::Frames, logged(&log, "view"));
        state.retire(stamp(1, 0), After::Frames, logged(&log, "image"));
        run(&mut state, done(4, 0));
        assert!(log.lock().unwrap().is_empty(), "the image may not go first");
        run(&mut state, done(5, 0));
        assert_eq!(*log.lock().unwrap(), ["view", "image"]);
    }

    #[test]
    fn nothing_reserved_means_nothing_to_wait_for_but_destructors_never_run_on_their_own() {
        let log = Arc::default();
        let mut state = State::new();
        state.retire(Stamp::default(), After::Frames, logged(&log, "unused"));
        assert!(log.lock().unwrap().is_empty(), "`retire` doesn't run it");
        run(&mut state, done(0, 0));
        assert_eq!(*log.lock().unwrap(), ["unused"]);
    }

    /// Without present fences, what a present may use waits for a later value of the timeline
    /// that presented it, or an idle device.
    #[test]
    fn unfenced_presents_wait_for_a_later_value_of_their_timeline() {
        let log = Arc::default();
        let mut state = State::new();
        let presented = After::Presented { timeline: Some(1) };
        state.retire(stamp(2, 4), presented, logged(&log, "swapchain"));
        state.retire(stamp(2, 4), After::Frames, logged(&log, "surface"));
        run(&mut state, done(9, 4));
        assert!(log.lock().unwrap().is_empty());
        run(&mut state, done(9, 5));
        assert_eq!(*log.lock().unwrap(), ["swapchain", "surface"]);
        // Never presented: its frames are enough.
        let never = After::Presented { timeline: None };
        state.retire(stamp(2, 4), never, logged(&log, "unpresented"));
        run(&mut state, done(9, 5));
        assert_eq!(log.lock().unwrap().len(), 3);
        // An idle device finished the presents of everything retired before it.
        state.retire(stamp(2, 6), presented, logged(&log, "idle"));
        state.idle_seq = state.next_seq;
        run(&mut state, done(9, 6));
        assert_eq!(log.lock().unwrap().last(), Some(&"idle"));
    }

    /// With present fences, what a present may use waits for the presents made before its drop
    /// instead.
    #[test]
    fn fenced_presents_hold_back_what_was_dropped_after_them() {
        let log = Arc::default();
        let mut state = State::new();
        state.fenced = true;
        let presented = After::Presented { timeline: Some(0) };
        state.presents.push_back((state.next_seq, fence(1)));
        state.retire(stamp(1, 0), presented, logged(&log, "semaphore"));
        state.presents.push_back((state.next_seq, fence(2)));
        run(&mut state, done(1, 0));
        assert!(log.lock().unwrap().is_empty(), "the first present may still use it");
        state.poll_presents(|f| f == fence(1));
        run(&mut state, done(1, 0));
        assert_eq!(*log.lock().unwrap(), ["semaphore"], "the second present can't use it");
        assert_eq!(state.free_fences, [fence(1)]);
    }

    #[test]
    fn freed_slots_come_back_once_their_stamp_is_done() {
        let mut pool = SlotPool::new(4);
        assert_eq!(pool.take_at(&done(0, 0)), Some(0));
        assert_eq!(pool.take_at(&done(0, 0)), Some(1));
        pool.freed.push_back((0, stamp(5, 2)));
        // Value 5 of timeline 0 may still read slot 0.
        assert_eq!(pool.take_at(&done(4, 2)), Some(2));
        assert_eq!(pool.take_at(&done(5, 2)), Some(0));
        assert_eq!(pool.take_at(&done(5, 2)), Some(3));
        assert_eq!(pool.take_at(&done(5, 2)), None, "all in use");
        pool.freed.push_back((3, stamp(6, 0)));
        assert_eq!(pool.take_at(&done(5, 9)), None, "slot 3 isn't free yet");
        assert_eq!(pool.take_at(&done(6, 0)), Some(3));
    }

    #[test]
    fn a_recording_thread_runs_no_destructors() {
        assert!(!RECORDING.get());
        recording(|| {
            assert!(RECORDING.get());
            recording(|| assert!(RECORDING.get()));
            assert!(RECORDING.get());
        });
        assert!(!RECORDING.get());
        let caught = std::panic::catch_unwind(|| recording(|| panic!("recording failed")));
        assert!(caught.is_err());
        assert!(!RECORDING.get(), "reset also when the recording panics");
    }
}

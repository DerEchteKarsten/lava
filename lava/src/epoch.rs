//! Frame epochs and the deletion queue: a frame reserves an epoch before it reads anything the GPU will use, and a dropped GPU object is destroyed once every epoch up to the latest reserved at its drop has completed
//!
//! Lava counts nothing per use. A frame takes an [`Epoch`] from [`reserve`] before it reads
//! what its commands, buffers or bindless slots will reach (core reserves it at extract), hands
//! it to `FrameSlot::begin`, and the slot completes it once the GPU finished the frame (or
//! right away, if it was never submitted). An image, buffer, semaphore, swapchain or surface
//! that drops queues its destructor, stamped with the latest epoch reserved so far: every frame
//! that could still reach it, through a command or an index, reserved that epoch or an earlier
//! one. The destructors run in drop order once their epoch and every earlier one completed.
//!
//! Presents have no fence, so what a present may still use (semaphores, swapchains and their
//! surfaces) waits for one more epoch: a frame reserved after the drop has to finish too.
//! [`crate::wait_idle`] completes every epoch submitted before it and runs what that frees.
//!
//! A submitted epoch also completes once its fence signalled, which [`reserve`] and
//! [`completed`] poll, so a frame slot that isn't begun again holds nothing back. Destructors
//! run when an epoch is reserved or completed, never on a thread that is recording a command
//! buffer.
//!
//! Slots in shared buffers that the GPU reads by index, like core's materials, follow the same
//! rule by hand: remember [`latest`] when the slot is freed, and reuse it once [`completed`]
//! reached that.
use std::{cell::Cell, collections::VecDeque, sync::Mutex};

use crate::vk;

type Destroy = Box<dyn FnOnce() + Send>;

/// Where a reserved epoch is.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Open {
    /// Its frame is being recorded, or kept by a slot that hasn't submitted it.
    Reserved,
    /// Submitted; it completed once the fence signalled.
    Submitted(vk::Fence),
}

/// A destructor waiting for the frames that may still reach its object.
struct Retired {
    /// The latest epoch reserved when the object dropped.
    stamp: u64,
    /// A present may use the object, so a frame reserved after the drop has to finish too.
    after_present: bool,
    /// Counts the destructors in drop order; [`State::device_idle`] remembers the next one.
    seq: u64,
    destroy: Destroy,
}

struct State {
    /// The epoch `reserve` hands out next; 0 is never handed out.
    next: u64,
    /// The reserved epochs that haven't completed, oldest first.
    open: Vec<(u64, Open)>,
    /// Destructors in drop order.
    retired: VecDeque<Retired>,
    /// The `seq` of the next destructor.
    next_seq: u64,
    /// The destructors before this `seq` were retired before the device was last idle: the
    /// presents they waited for are over.
    idle_seq: u64,
    /// Presents signal fences (`Features::present_fences`): what a present may use waits for
    /// the presents made before its drop, not for another frame.
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
            next: 1,
            open: Vec::new(),
            retired: VecDeque::new(),
            next_seq: 0,
            idle_seq: 0,
            fenced: false,
            presents: VecDeque::new(),
            free_fences: Vec::new(),
        }
    }

    fn reserve(&mut self) -> u64 {
        let epoch = self.next;
        self.next += 1;
        self.open.push((epoch, Open::Reserved));
        epoch
    }

    fn latest(&self) -> u64 {
        self.next - 1
    }

    /// Every epoch up to this one completed.
    fn completed(&self) -> u64 {
        self.open
            .first()
            .map_or(self.latest(), |&(epoch, _)| epoch - 1)
    }

    fn submitted(&mut self, epoch: u64, fence: vk::Fence) {
        if let Some(open) = self.open.iter_mut().find(|(e, _)| *e == epoch) {
            open.1 = Open::Submitted(fence);
        }
    }

    fn complete(&mut self, epoch: u64) {
        self.open.retain(|&(e, _)| e != epoch);
    }

    /// Completes the oldest open epochs whose fences `signalled`, up to the first that isn't
    /// done: only those move `completed`. So a slot that isn't begun again doesn't hold back
    /// every later destruction.
    fn poll(&mut self, signalled: impl Fn(vk::Fence) -> bool) {
        while let Some(&(_, Open::Submitted(fence))) = self.open.first() {
            if !signalled(fence) {
                break;
            }
            self.open.remove(0);
        }
        while let Some(&(_, fence)) = self.presents.front() {
            if !signalled(fence) {
                break;
            }
            self.presents.pop_front();
            self.free_fences.push(fence);
        }
    }

    /// A present that signals `fence` was just made.
    fn presented(&mut self, fence: vk::Fence) {
        self.presents.push_back((self.next_seq, fence));
    }

    /// The device went idle after every submission so far: those epochs completed, and the
    /// presents of the objects dropped so far are over.
    fn device_idle(&mut self) {
        self.open.retain(|&(_, open)| open == Open::Reserved);
        self.idle_seq = self.next_seq;
    }

    fn retire(&mut self, after_present: bool, destroy: Destroy) {
        let seq = self.next_seq;
        self.next_seq += 1;
        self.retired.push_back(Retired {
            stamp: self.latest(),
            after_present,
            seq,
            destroy,
        });
    }

    /// The next destructor that may run: the oldest, once every epoch up to its stamp
    /// completed (one more for what a present may use, unless the device was idle since its
    /// drop). Destructors run in drop order, so a later one never goes first.
    fn pop_ready(&mut self) -> Option<Destroy> {
        let done = self.completed();
        let front = self.retired.front()?;
        let ready = if !front.after_present {
            front.stamp <= done
        } else if self.fenced {
            // Every present made before the drop signalled its fence.
            let presented_before = (self.presents.front()).is_some_and(|&(seq, _)| seq <= front.seq);
            front.stamp <= done && !presented_before
        } else {
            let presented = front.seq >= self.idle_seq;
            front.stamp + presented as u64 <= done
        };
        if ready {
            self.retired.pop_front().map(|r| r.destroy)
        } else {
            None
        }
    }
}

static STATE: Mutex<State> = Mutex::new(State::new());

fn state() -> std::sync::MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|poison| poison.into_inner())
}

thread_local! {
    /// The thread records a command buffer: it holds its queue's pending accesses, which
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

/// The epoch of one frame. Reserve it before reading anything the frame's GPU work will reach,
/// and hand it to `FrameSlot::begin`. Dropping it completes it, so the slot keeps it until the
/// GPU finished the frame.
#[derive(Debug)]
#[must_use]
pub struct Epoch(u64);

impl Epoch {
    pub fn value(&self) -> u64 {
        self.0
    }

    /// Its frame was submitted with `fence`, which may be polled until the epoch is dropped:
    /// its slot drops it before resetting or destroying the fence.
    pub(crate) fn submitted(&self, fence: vk::Fence) {
        state().submitted(self.0, fence);
    }
}

impl Drop for Epoch {
    fn drop(&mut self) {
        state().complete(self.0);
        reclaim();
    }
}

/// A new epoch, later than every one reserved before. Completes the submitted epochs whose
/// frames finished and runs what that frees.
pub fn reserve() -> Epoch {
    let epoch = {
        let mut state = state();
        poll(&mut state);
        Epoch(state.reserve())
    };
    reclaim();
    epoch
}

/// The latest epoch reserved: what a slot freed now has to wait for.
pub fn latest() -> u64 {
    state().latest()
}

/// Every epoch up to this one completed: the GPU is done with every frame that reserved one.
/// Runs no destructor, so it may be called while holding locks a destructor takes.
pub fn completed() -> u64 {
    let mut state = state();
    poll(&mut state);
    state.completed()
}

fn poll(state: &mut State) {
    use crate::state::Ctx;
    state.poll(|fence| unsafe { Ctx::device().get_fence_status(fence) }.unwrap_or(false));
}

/// Runs `destroy` once every frame that could still reach the object has finished. It never
/// runs right away, also not when no frame is open, so it may lock what the caller holds.
pub(crate) fn retire(destroy: impl FnOnce() + Send + 'static) {
    state().retire(false, Box::new(destroy));
}

/// Like [`retire`], for what a present may use: one more epoch has to complete.
pub(crate) fn retire_after_present(destroy: impl FnOnce() + Send + 'static) {
    state().retire(true, Box::new(destroy));
}

/// Whether presents signal fences (`Features::present_fences`), set by `init`.
pub(crate) fn set_present_fences(on: bool) {
    state().fenced = on;
}

/// A reset fence for a present, signalled ones first.
pub(crate) fn present_fence() -> crate::Result<vk::Fence> {
    use crate::state::Ctx;
    let device = Ctx::device();
    if let Some(fence) = state().free_fences.pop() {
        unsafe { device.reset_fences(&[fence])? };
        return Ok(fence);
    }
    Ok(unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? })
}

/// A present that signals `fence` (from [`present_fence`]) was just made: what was dropped
/// before it waits for it.
pub(crate) fn presented(fence: vk::Fence) {
    state().presented(fence);
}

/// Waits until the presents made so far signalled their fences, at most `timeout_ns`; one that
/// doesn't by then (its present failed before it was queued) is given up on, and its fence
/// leaked. Call it while no present can be made (`wait_idle` holds every queue).
pub(crate) fn wait_for_presents(timeout_ns: u64) {
    use crate::state::Ctx;
    let device = Ctx::device();
    let fences: Vec<vk::Fence> = state().presents.iter().map(|&(_, fence)| fence).collect();
    if !fences.is_empty() && unsafe { device.wait_for_fences(&fences, true, timeout_ns) }.is_err() {
        tracing::warn!("a present's fence didn't signal after the device went idle; giving up on it");
        let signalled = |fence| unsafe { device.get_fence_status(fence) } == Ok(true);
        state().presents.retain(|&(_, fence)| signalled(fence));
    }
    poll(&mut state());
}

/// After the device went idle, while no queue could be submitted to: completes every submitted
/// epoch. Runs no destructor (call [`reclaim`] once the queues are unlocked).
pub(crate) fn device_idle() {
    state().device_idle();
}

/// Runs every destructor that may run, without holding the lock, since destroying an image
/// or buffer locks the queues' pending accesses; not while this thread records, which holds
/// one of them.
pub(crate) fn reclaim() {
    if RECORDING.get() {
        return;
    }
    loop {
        let Some(destroy) = state().pop_ready() else {
            return;
        };
        destroy();
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

    fn run(state: &mut State) {
        while let Some(destroy) = state.pop_ready() {
            destroy();
        }
    }

    fn fence(raw: u64) -> vk::Fence {
        use crate::vk::Handle;
        vk::Fence::from_raw(raw)
    }

    #[test]
    fn epochs_complete_in_any_order_but_count_from_the_oldest_open_one() {
        let mut state = State::new();
        assert_eq!((state.latest(), state.completed()), (0, 0));
        let (a, b, c) = (state.reserve(), state.reserve(), state.reserve());
        assert_eq!((a, b, c), (1, 2, 3));
        assert_eq!(state.completed(), 0);
        state.complete(b);
        assert_eq!(state.completed(), 0);
        state.complete(a);
        assert_eq!(state.completed(), 2);
        state.complete(c);
        assert_eq!(state.completed(), 3);
    }

    #[test]
    fn a_destructor_waits_for_the_latest_epoch_reserved_at_its_drop() {
        let log = Arc::default();
        let mut state = State::new();
        let first = state.reserve();
        // Dropped while the first frame records: it may use the object.
        state.retire(false, logged(&log, "during first"));
        let second = state.reserve();
        state.retire(false, logged(&log, "during second"));

        state.complete(first);
        run(&mut state);
        assert_eq!(*log.lock().unwrap(), ["during first"]);
        state.complete(second);
        run(&mut state);
        assert_eq!(*log.lock().unwrap(), ["during first", "during second"]);
    }

    #[test]
    fn destructors_never_run_on_their_own_and_wait_for_an_open_epoch() {
        let log = Arc::default();
        let mut state = State::new();
        // No frame was ever reserved: nothing can use it, but `retire` doesn't run it.
        state.retire(false, logged(&log, "unused"));
        assert!(log.lock().unwrap().is_empty());
        let open = state.reserve();
        run(&mut state);
        assert_eq!(*log.lock().unwrap(), ["unused"]);
        state.retire(false, logged(&log, "used"));
        run(&mut state);
        assert_eq!(log.lock().unwrap().len(), 1);
        state.complete(open);
        run(&mut state);
        assert_eq!(*log.lock().unwrap(), ["unused", "used"]);
    }

    #[test]
    fn what_a_present_uses_waits_for_a_frame_reserved_after_its_drop() {
        let log = Arc::default();
        let mut state = State::new();
        let presented = state.reserve();
        state.complete(presented);
        state.retire(true, logged(&log, "swapchain"));
        run(&mut state);
        assert!(log.lock().unwrap().is_empty());
        let next = state.reserve();
        run(&mut state);
        assert!(log.lock().unwrap().is_empty());
        state.complete(next);
        run(&mut state);
        assert_eq!(*log.lock().unwrap(), ["swapchain"]);
    }

    #[test]
    fn destructors_run_in_drop_order() {
        let log = Arc::default();
        let mut state = State::new();
        let epoch = state.reserve();
        // A surface dropped after its swapchain: the swapchain waits one epoch longer, and
        // the surface may not go first.
        state.retire(true, logged(&log, "swapchain"));
        state.retire(false, logged(&log, "surface"));
        state.complete(epoch);
        run(&mut state);
        assert!(log.lock().unwrap().is_empty());
        let next = state.reserve();
        state.complete(next);
        run(&mut state);
        assert_eq!(*log.lock().unwrap(), ["swapchain", "surface"]);
    }

    #[test]
    fn an_idle_device_frees_all_but_what_unsubmitted_frames_may_use() {
        let log = Arc::default();
        let mut state = State::new();
        let submitted = state.reserve();
        state.submitted(submitted, fence(1));
        state.retire(false, logged(&log, "submitted"));
        let recording = state.reserve();
        state.retire(false, logged(&log, "recording"));
        state.device_idle();
        run(&mut state);
        assert_eq!(*log.lock().unwrap(), ["submitted"]);
        // The slot that held the submitted epoch drops it later: nothing changes.
        state.complete(submitted);
        state.complete(recording);
        run(&mut state);
        assert_eq!(*log.lock().unwrap(), ["submitted", "recording"]);
        // Dropped after the device was idle: it waits for a frame reserved after the drop.
        state.retire(true, logged(&log, "presented"));
        run(&mut state);
        assert_eq!(log.lock().unwrap().len(), 2);
        // Idle again: the presents before it are over too.
        state.device_idle();
        run(&mut state);
        assert_eq!(*log.lock().unwrap(), ["submitted", "recording", "presented"]);
    }

    /// An epoch submitted before the device went idle completed; one submitted after it, by
    /// another thread, did not.
    #[test]
    fn idleness_only_completes_what_was_submitted_before_it() {
        let mut state = State::new();
        let before = state.reserve();
        state.submitted(before, fence(1));
        let after = state.reserve();
        state.device_idle();
        state.submitted(after, fence(2));
        assert_eq!(state.completed(), before);
    }

    /// A slot that isn't begun again keeps its epoch, but its fence completes it.
    #[test]
    fn signalled_fences_complete_the_oldest_epochs() {
        let log = Arc::default();
        let mut state = State::new();
        let (a, b, c) = (state.reserve(), state.reserve(), state.reserve());
        state.submitted(a, fence(1));
        state.submitted(c, fence(3));
        state.retire(false, logged(&log, "used by a"));
        let signalled = |f: vk::Fence| f != fence(2);
        state.poll(signalled);
        assert_eq!(state.completed(), a, "b is still recording");
        state.submitted(b, fence(2));
        state.poll(signalled);
        assert_eq!(state.completed(), a, "b's fence hasn't signalled");
        state.poll(|_| true);
        assert_eq!(state.completed(), c);
        run(&mut state);
        assert_eq!(*log.lock().unwrap(), ["used by a"]);
        // The slot drops its epochs later: nothing changes.
        state.complete(a);
        state.complete(b);
        assert_eq!(state.completed(), c);
    }

    /// With present fences, what a present may use waits for the presents made before its drop
    /// instead of another frame.
    #[test]
    fn fenced_presents_hold_back_what_was_dropped_after_them() {
        let log = Arc::default();
        let mut state = State::new();
        state.fenced = true;
        let frame = state.reserve();
        state.complete(frame);
        state.presented(fence(1));
        state.retire(true, logged(&log, "semaphore"));
        state.presented(fence(2));
        run(&mut state);
        assert!(log.lock().unwrap().is_empty(), "the first present may still use it");
        state.poll(|f| f == fence(1));
        run(&mut state);
        assert_eq!(*log.lock().unwrap(), ["semaphore"], "the second present can't use it");
        assert_eq!(state.free_fences, [fence(1)]);
        // Without presents, only the frames count.
        state.poll(|_| true);
        let frame = state.reserve();
        state.retire(true, logged(&log, "swapchain"));
        run(&mut state);
        assert_eq!(log.lock().unwrap().len(), 1, "its frame may still use it");
        state.complete(frame);
        run(&mut state);
        assert_eq!(*log.lock().unwrap(), ["semaphore", "swapchain"]);
    }

    #[test]
    fn a_recording_thread_runs_no_destructors() {
        // Nothing is retired here; this checks the flag nests and resets.
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

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
//! [`crate::wait_idle`] completes every submitted epoch and runs what that frees.
//!
//! Slots in shared buffers that the GPU reads by index, like core's materials, follow the same
//! rule by hand: remember [`latest`] when the slot is freed, and reuse it once [`completed`]
//! reached that.
use std::{collections::VecDeque, sync::Mutex};

type Destroy = Box<dyn FnOnce() + Send>;

struct State {
    /// The epoch `reserve` hands out next; 0 is never handed out.
    next: u64,
    /// The reserved epochs that haven't completed, oldest first, and whether each was
    /// submitted.
    open: Vec<(u64, bool)>,
    /// Destructors in drop order, each with the epoch that has to complete before it runs.
    retired: VecDeque<(u64, Destroy)>,
}

impl State {
    const fn new() -> Self {
        State {
            next: 1,
            open: Vec::new(),
            retired: VecDeque::new(),
        }
    }

    fn reserve(&mut self) -> u64 {
        let epoch = self.next;
        self.next += 1;
        self.open.push((epoch, false));
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

    fn submitted(&mut self, epoch: u64) {
        if let Some(open) = self.open.iter_mut().find(|(e, _)| *e == epoch) {
            open.1 = true;
        }
    }

    fn complete(&mut self, epoch: u64) {
        self.open.retain(|&(e, _)| e != epoch);
    }

    /// Every submitted epoch is done once the device is idle.
    fn device_idle(&mut self) {
        self.open.retain(|&(_, submitted)| !submitted);
    }

    fn retire(&mut self, after_present: bool, destroy: Destroy) {
        let stamp = self.latest() + after_present as u64;
        // Destructors run in drop order, so none waits for less than the one before it.
        let stamp = self.retired.back().map_or(stamp, |&(last, _)| last.max(stamp));
        self.retired.push_back((stamp, destroy));
    }

    /// The next destructor that may run. `idle`: the device is idle, so only epochs that are
    /// reserved and not submitted yet hold destructors back, and presents are over too.
    fn pop_ready(&mut self, idle: bool) -> Option<Destroy> {
        let done = match self.open.first() {
            Some(&(epoch, _)) => epoch - 1,
            None if idle => u64::MAX,
            None => self.latest(),
        };
        match self.retired.front() {
            Some(&(stamp, _)) if stamp <= done => self.retired.pop_front().map(|(_, d)| d),
            _ => None,
        }
    }
}

static STATE: Mutex<State> = Mutex::new(State::new());

fn state() -> std::sync::MutexGuard<'static, State> {
    STATE.lock().unwrap()
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

    pub(crate) fn submitted(&self) {
        state().submitted(self.0);
    }
}

impl Drop for Epoch {
    fn drop(&mut self) {
        state().complete(self.0);
        reclaim(false);
    }
}

/// A new epoch, later than every one reserved before.
pub fn reserve() -> Epoch {
    Epoch(state().reserve())
}

/// The latest epoch reserved: what a slot freed now has to wait for.
pub fn latest() -> u64 {
    state().latest()
}

/// Every epoch up to this one completed: the GPU is done with every frame that reserved one.
pub fn completed() -> u64 {
    state().completed()
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

/// After the device went idle: completes every submitted epoch and runs what that frees.
pub(crate) fn device_idle() {
    state().device_idle();
    reclaim(true);
}

/// Runs every destructor that may run, without holding the lock, since destroying an image
/// or buffer locks the queues' pending accesses.
fn reclaim(idle: bool) {
    loop {
        let Some(destroy) = state().pop_ready(idle) else {
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

    fn run(state: &mut State, idle: bool) {
        while let Some(destroy) = state.pop_ready(idle) {
            destroy();
        }
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
        run(&mut state, false);
        assert_eq!(*log.lock().unwrap(), ["during first"]);
        state.complete(second);
        run(&mut state, false);
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
        run(&mut state, false);
        assert_eq!(*log.lock().unwrap(), ["unused"]);
        state.retire(false, logged(&log, "used"));
        run(&mut state, false);
        assert_eq!(log.lock().unwrap().len(), 1);
        state.complete(open);
        run(&mut state, false);
        assert_eq!(*log.lock().unwrap(), ["unused", "used"]);
    }

    #[test]
    fn what_a_present_uses_waits_for_a_frame_reserved_after_its_drop() {
        let log = Arc::default();
        let mut state = State::new();
        let presented = state.reserve();
        state.complete(presented);
        state.retire(true, logged(&log, "swapchain"));
        run(&mut state, false);
        assert!(log.lock().unwrap().is_empty());
        let next = state.reserve();
        run(&mut state, false);
        assert!(log.lock().unwrap().is_empty());
        state.complete(next);
        run(&mut state, false);
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
        run(&mut state, false);
        assert!(log.lock().unwrap().is_empty());
        let next = state.reserve();
        state.complete(next);
        run(&mut state, false);
        assert_eq!(*log.lock().unwrap(), ["swapchain", "surface"]);
    }

    #[test]
    fn an_idle_device_frees_all_but_what_unsubmitted_frames_may_use() {
        let log = Arc::default();
        let mut state = State::new();
        let submitted = state.reserve();
        state.submitted(submitted);
        state.retire(false, logged(&log, "submitted"));
        let recording = state.reserve();
        state.retire(false, logged(&log, "recording"));
        state.device_idle();
        run(&mut state, true);
        assert_eq!(*log.lock().unwrap(), ["submitted"]);
        // The slot that held the submitted epoch drops it later: nothing changes.
        state.complete(submitted);
        state.complete(recording);
        run(&mut state, true);
        assert_eq!(*log.lock().unwrap(), ["submitted", "recording"]);
        // The presents are over too, so their objects don't wait for another frame.
        state.retire(true, logged(&log, "presented"));
        state.device_idle();
        run(&mut state, true);
        assert_eq!(*log.lock().unwrap(), ["submitted", "recording", "presented"]);
    }
}

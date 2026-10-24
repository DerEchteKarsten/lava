//! Approximate lifetimes: a dropped GPU object is stamped with every timeline's stamp value and destroyed, in drop order and by one thread at a time, once all of them completed
use std::{
    cell::Cell,
    collections::VecDeque,
    sync::{Mutex, atomic::Ordering},
};

use crate::queue::{
    MAX_TIMELINES, lock,
    timeline::{INDICES, completed, poll},
};

/// Every timeline's stamp value at one moment ([`latest_stamp`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stamp(pub(crate) [u64; MAX_TIMELINES]);

impl Stamp {
    /// Whether every timeline completed its value. Polls only the timelines whose last known
    /// value is behind; false once the device is lost.
    pub fn is_done(&self) -> bool {
        if self.done() {
            return true;
        }
        let _call = crate::state::call();
        (0..MAX_TIMELINES)
            .all(|i| self.0[i] <= completed(i) || poll(i).is_ok_and(|v| self.0[i] <= v))
    }

    /// Whether every timeline completed its value, as far as the last polls know.
    pub(crate) fn done(&self) -> bool {
        (0..MAX_TIMELINES).all(|i| self.0[i] <= completed(i))
    }
}

/// What an object dropped now has to wait for.
pub fn latest_stamp() -> Stamp {
    Stamp(std::array::from_fn(|i| {
        INDICES[i].stamp.load(Ordering::SeqCst)
    }))
}

type Destroy = Box<dyn FnOnce() + Send>;

pub(crate) static RETIRED: Mutex<VecDeque<(Stamp, Destroy)>> = Mutex::new(VecDeque::new());

/// Held while destructors run, so they run in drop order, one thread at a time. Not `RETIRED`'s
/// lock: destructors retire things.
static RECLAIMING: Mutex<()> = Mutex::new(());

thread_local! {
    /// The thread records a submission, holding its timeline's index state: destructors, which
    /// wait anyway, run later on a thread that doesn't.
    pub(crate) static RECORDING: Cell<bool> = const { Cell::new(false) };
    /// The thread holds `RECLAIMING`: a destructor that reclaims returns right away.
    static RECLAIMS: Cell<bool> = const { Cell::new(false) };
}

/// Runs `destroy` once every submission that could still reach the object finished; never right
/// away, so it may lock what the caller holds.
pub(crate) fn retire(destroy: impl FnOnce() + Send + 'static) {
    let stamp = latest_stamp();
    lock(&RETIRED).push_back((stamp, Box::new(destroy)));
}

/// Polls every timeline, then runs every destructor that may run, in drop order (after another
/// thread that runs them finished); not on a thread that records.
pub fn reclaim() {
    let _call = crate::state::call();
    let _ = crate::queue::timeline::poll_all();
    run_ready();
}

/// Clears `RECLAIMS` also when a destructor panics.
struct Reclaiming;

impl Drop for Reclaiming {
    fn drop(&mut self) {
        RECLAIMS.set(false);
    }
}

/// Runs the retired destructors in drop order while `ready` says the first may run.
fn run(ready: impl Fn(&Stamp) -> bool) {
    if RECORDING.get() || RECLAIMS.get() {
        return;
    }
    let _lock = lock(&RECLAIMING);
    RECLAIMS.set(true);
    let _reclaiming = Reclaiming;
    loop {
        let destroy = {
            let mut retired = lock(&RETIRED);
            match retired.front() {
                Some((stamp, _)) if ready(stamp) => retired.pop_front().map(|(_, d)| d),
                _ => None,
            }
        };
        match destroy {
            Some(destroy) => destroy(),
            None => return,
        }
    }
}

/// Runs every destructor the last polls showed may run, without polling.
pub(crate) fn run_ready() {
    run(Stamp::done);
}

/// Runs every destructor, whatever its stamp: `shutdown`, once the device is idle (or lost).
pub(crate) fn run_all() {
    run(|_| true);
}

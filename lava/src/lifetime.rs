//! Approximate lifetimes: a dropped GPU object is stamped with every timeline's stamp value and destroyed, in drop order, once all of them completed
use std::{cell::Cell, collections::VecDeque, sync::Mutex};

use crate::queue::{
    MAX_TIMELINES, lock,
    timeline::{INDICES, completed, poll_all},
};

/// Every timeline's stamp value at one moment ([`latest_stamp`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stamp(pub(crate) [u64; MAX_TIMELINES]);

impl Stamp {
    /// Whether every timeline completed its value. Polls the timelines.
    pub fn is_done(&self) -> bool {
        poll_all();
        self.done()
    }

    pub(crate) fn done(&self) -> bool {
        (0..MAX_TIMELINES).all(|i| self.0[i] <= completed(i))
    }
}

/// What an object dropped now has to wait for.
pub fn latest_stamp() -> Stamp {
    Stamp(std::array::from_fn(|i| {
        INDICES[i].stamp.load(std::sync::atomic::Ordering::SeqCst)
    }))
}

type Destroy = Box<dyn FnOnce() + Send>;

pub(crate) static RETIRED: Mutex<VecDeque<(Stamp, Destroy)>> = Mutex::new(VecDeque::new());

thread_local! {
    /// The thread records and holds its timeline's pending accesses, which destroying an image
    /// or buffer locks.
    pub(crate) static RECORDING: Cell<bool> = const { Cell::new(false) };
}

/// Runs `destroy` once every submission that could still reach the object finished; never right
/// away, so it may lock what the caller holds.
pub(crate) fn retire(destroy: impl FnOnce() + Send + 'static) {
    let stamp = latest_stamp();
    lock(&RETIRED).push_back((stamp, Box::new(destroy)));
}

/// Runs every destructor that may run, in drop order; not on a thread that records.
pub fn reclaim() {
    if RECORDING.get() {
        return;
    }
    loop {
        let destroy = {
            let mut retired = lock(&RETIRED);
            match retired.front() {
                Some((stamp, _)) if stamp.done() => retired.pop_front().map(|(_, d)| d),
                _ => None,
            }
        };
        match destroy {
            Some(destroy) => destroy(),
            None => return,
        }
    }
}

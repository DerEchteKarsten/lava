//! Lava's resettable global state: globals set by `init` and taken by `shutdown`, and the count of live lava objects and calls in progress that keeps `shutdown` from destroying what they use
//!
//! # Safety argument
//! Lava's globals ([`Global`]) are pointers that `init` sets and `shutdown` (or a failed `init`)
//! takes, so reads stay cheap: one atomic load, no lock. What they point to is destroyed only
//! while nothing can read it:
//! - Every lava object (buffer, image, acceleration structure and arena, query pool, sampler,
//!   present surface, timeline, uploader) holds a [`Live`] count from before its constructor
//!   touches lava's state until after its `Drop` finished (it is the last field, so it drops
//!   last), and every public function that touches lava's state without such an object takes a
//!   count for its duration. Internal code runs only inside one of those or inside `init` and
//!   `shutdown`.
//! - `shutdown` closes the count only when it is zero ([`close`]), atomically: from then on
//!   [`Live::new`] fails, so nothing reads the globals while `shutdown` destroys them.
//! - `init` and `shutdown` hold [`LIFECYCLE`], so they never overlap.
//! - The context (`Ctx`) is never freed, only emptied, so a getter that reads it without a count
//!   (`Ctx::features()`, the feature checks) reads valid memory even while `shutdown` runs; after
//!   it, every such read panics with [`NOT_INITIALISED`].
use std::sync::{
    Mutex,
    atomic::{AtomicPtr, AtomicU64, Ordering},
};

/// What every lava call says when lava isn't initialised.
pub(crate) const NOT_INITIALISED: &str =
    "lava is not initialised: call `lava::init` first (and check that it succeeded)";

/// Held by `init` and `shutdown` for their whole run.
pub(crate) static LIFECYCLE: Mutex<()> = Mutex::new(());

/// A pointer to a `T` that `init` sets and `shutdown` takes.
pub(crate) struct Global<T>(AtomicPtr<T>);

impl<T> Global<T> {
    pub(crate) const fn new() -> Self {
        Self(AtomicPtr::new(std::ptr::null_mut()))
    }

    pub(crate) fn try_get(&self) -> Option<&'static T> {
        // SAFETY: a non-null pointer came from `Box::into_raw` in `set`, and the box is only
        // freed by `take`, which runs while nothing reads it (module docs).
        unsafe { self.0.load(Ordering::Acquire).as_ref() }
    }

    #[track_caller]
    pub(crate) fn get(&self) -> &'static T {
        self.try_get().expect(NOT_INITIALISED)
    }

    /// Sets it; panics if it was set.
    pub(crate) fn set(&self, value: T) {
        let new = Box::into_raw(Box::new(value));
        let old = self.0.compare_exchange(
            std::ptr::null_mut(),
            new,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        if old.is_err() {
            // SAFETY: `new` was never shared.
            drop(unsafe { Box::from_raw(new) });
            panic!("a lava global was set twice");
        }
    }

    /// Unsets it and returns what it held.
    ///
    /// # Safety
    /// Nothing may read it any more: lava's count is closed or was never opened, under
    /// [`LIFECYCLE`].
    pub(crate) unsafe fn take(&self) -> Option<Box<T>> {
        let old = self.0.swap(std::ptr::null_mut(), Ordering::AcqRel);
        // SAFETY: from `set`; the caller guarantees nobody reads it.
        (!old.is_null()).then(|| unsafe { Box::from_raw(old) })
    }
}

/// Set while lava is initialised and open for new objects and calls; the bits below count the
/// live objects and calls in progress.
const OPEN: u64 = 1 << 63;

static USERS: AtomicU64 = AtomicU64::new(0);

/// A count of lava's users: a live object, or a public call in progress (module docs). Lava's
/// own buffers and the like hold an uncounted one ([`Live::internal`]), since `shutdown`
/// destroys them itself.
#[derive(Debug)]
pub(crate) struct Live {
    counted: bool,
}

impl Live {
    /// Counts a user in; panics unless lava is initialised.
    #[track_caller]
    pub(crate) fn new() -> Self {
        let before = USERS.fetch_add(1, Ordering::Acquire);
        if before & OPEN == 0 {
            USERS.fetch_sub(1, Ordering::Release);
            panic!("{NOT_INITIALISED}");
        }
        Self { counted: true }
    }

    /// For lava's own objects, which `shutdown` destroys while they live.
    pub(crate) fn internal() -> Self {
        Self { counted: false }
    }
}

impl Clone for Live {
    fn clone(&self) -> Self {
        if self.counted {
            // This one keeps the count above zero, so lava stays open.
            USERS.fetch_add(1, Ordering::Relaxed);
        }
        Self {
            counted: self.counted,
        }
    }
}

impl Drop for Live {
    fn drop(&mut self) {
        if self.counted {
            USERS.fetch_sub(1, Ordering::Release);
        }
    }
}

/// Opens lava for users, at the end of a successful `init`.
pub(crate) fn open() {
    USERS.fetch_or(OPEN, Ordering::AcqRel);
}

/// Whether lava is open.
pub(crate) fn is_open() -> bool {
    USERS.load(Ordering::Acquire) & OPEN != 0
}

/// Closes lava if no user is counted in; else returns how many are.
pub(crate) fn close() -> Result<(), u64> {
    match USERS.compare_exchange(OPEN, 0, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => Ok(()),
        Err(users) => Err(users & !OPEN),
    }
}

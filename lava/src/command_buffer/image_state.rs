//! An image's tracked state, one `AtomicU64` per view or mip: its layout, the kind of queue family that owns it, the kind it was released to, and whether a swapchain image waits for its acquire; how recording an access takes it
use std::{
    range::Range,
    sync::atomic::{AtomicU64, Ordering},
};

use crate::{queue::Kind, vk};

/// What lava tracks of an image's mips, packed into one `u64` (`pack`): the layout in bits
/// 0..32, the owner in 32..34, the kind it was released to in 34..36, and bit 36 for an
/// acquired swapchain image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ImageState {
    pub(crate) layout: vk::ImageLayout,
    /// The kind of family whose recording accessed it last; `None` before any did. Images are
    /// `EXCLUSIVE`, so another family's recording needs a release first.
    pub(crate) owner: Option<Kind>,
    /// The owner recorded a release to this kind's family, whose first access acquires it.
    pub(crate) released_to: Option<Kind>,
    /// A swapchain image the presentation engine handed out: its first access waits for the
    /// acquire's semaphore.
    pub(crate) acquired: bool,
}

const OWNER_SHIFT: u32 = 32;
const RELEASED_SHIFT: u32 = 34;
const ACQUIRED: u64 = 1 << 36;

fn kind_bits(kind: Option<Kind>) -> u64 {
    kind.map_or(0, |kind| kind as u64)
}

fn kind_of(bits: u64) -> Option<Kind> {
    match bits & 3 {
        0 => None,
        1 => Some(Kind::Graphics),
        2 => Some(Kind::Compute),
        _ => Some(Kind::Transfer),
    }
}

impl ImageState {
    /// A new image's: undefined, owned by no family.
    pub(crate) const UNDEFINED: ImageState = ImageState {
        layout: vk::ImageLayout::UNDEFINED,
        owner: None,
        released_to: None,
        acquired: false,
    };

    /// What an access in `layout` recorded on a family of `kind` leaves.
    pub(crate) fn taken(layout: vk::ImageLayout, kind: Kind) -> Self {
        ImageState {
            layout,
            owner: Some(kind),
            released_to: None,
            acquired: false,
        }
    }

    pub(crate) fn pack(self) -> u64 {
        self.layout.as_raw() as u32 as u64
            | kind_bits(self.owner) << OWNER_SHIFT
            | kind_bits(self.released_to) << RELEASED_SHIFT
            | if self.acquired { ACQUIRED } else { 0 }
    }

    pub(crate) fn unpack(bits: u64) -> Self {
        ImageState {
            layout: vk::ImageLayout::from_raw(bits as u32 as i32),
            owner: kind_of(bits >> OWNER_SHIFT),
            released_to: kind_of(bits >> RELEASED_SHIFT),
            acquired: bits & ACQUIRED != 0,
        }
    }

    /// The state of a new image or swapchain image.
    pub(crate) fn atomic(self) -> AtomicU64 {
        AtomicU64::new(self.pack())
    }

    pub(crate) fn load(state: &AtomicU64) -> Self {
        Self::unpack(state.load(Ordering::Relaxed))
    }

    /// Whether a recording on a family of `kind` may access it: nobody owns it, `kind` does, or
    /// its owner released it to `kind`.
    pub(crate) fn accessible_by(self, kind: Kind) -> bool {
        self.owner.is_none_or(|owner| owner == kind) || self.released_to == Some(kind)
    }
}

/// Flags a swapchain image the presentation engine just handed out, or no longer does.
pub(crate) fn set_acquired(state: &AtomicU64, acquired: bool) {
    match acquired {
        true => state.fetch_or(ACQUIRED, Ordering::Relaxed),
        false => state.fetch_and(!ACQUIRED, Ordering::Relaxed),
    };
}

/// Panics unless a recording of `kind` may access an image in state `state`.
#[track_caller]
pub(crate) fn check_access(state: ImageState, kind: Kind) {
    if state.accessible_by(kind) {
        return;
    }
    let owner = state.owner.expect("an unowned image is accessible");
    panic!(
        "an image the {owner:?} family owns is accessed on the {kind:?} family: record \
         `cmd.release::<{kind:?}>(view)` on a {owner:?} timeline first, and wait for it"
    );
}

/// Calls `f` with the state of each mip of `mips`, which `states` holds one of for all of them
/// or one per mip, and the mips it covers.
fn for_each(states: &[AtomicU64], mips: Range<u32>, mut f: impl FnMut(&AtomicU64, Range<u32>)) {
    if let [state] = states {
        return f(state, mips);
    }
    for (mip, state) in (mips.start..).zip(states) {
        f(state, (mip..mip + 1).into());
    }
}

/// Calls `changed(mips, old)` once per run of neighbouring mips with the same old state that
/// isn't `new`, with what `swap` puts in each state and returns of the old one.
fn runs(
    states: &[AtomicU64],
    mips: Range<u32>,
    new: ImageState,
    mut swap: impl FnMut(&AtomicU64) -> ImageState,
    mut changed: impl FnMut(Range<u32>, ImageState),
) {
    let mut run: Option<(Range<u32>, ImageState)> = None;
    for_each(states, mips, |state, mips| {
        let old = swap(state);
        match &mut run {
            Some((run_mips, run_old)) if *run_old == old && run_mips.end == mips.start => {
                run_mips.end = mips.end;
            }
            _ => {
                if let Some((run_mips, run_old)) = run.replace((mips, old))
                    && run_old != new
                {
                    changed(run_mips, run_old);
                }
            }
        }
    });
    if let Some((run_mips, run_old)) = run
        && run_old != new
    {
        changed(run_mips, run_old);
    }
}

/// Panics unless a recording of `kind` may access every mip of `mips` (`check_access`).
#[track_caller]
pub(crate) fn check_states(states: &[AtomicU64], mips: Range<u32>, kind: Kind) {
    for_each(states, mips, |state, _| check_access(ImageState::load(state), kind));
}

/// Takes the states of `mips` for an access in `layout` recorded on a family of `kind`. First
/// checks them all (`check_access`, which panics before any moved, so the barriers logged so far
/// stay true), then moves each to `layout`, owned by `kind`, and calls `changed(mips, old)` once
/// per run of neighbouring mips with the same old state that differs from the new one.
#[cfg(test)]
#[track_caller]
pub(crate) fn take_states(
    states: &[AtomicU64],
    mips: Range<u32>,
    layout: vk::ImageLayout,
    kind: Kind,
    changed: impl FnMut(Range<u32>, ImageState),
) {
    check_states(states, mips, kind);
    take_checked_states(states, mips, layout, kind, changed);
}

/// [`take_states`] for states `check_states` checked already.
pub(crate) fn take_checked_states(
    states: &[AtomicU64],
    mips: Range<u32>,
    layout: vk::ImageLayout,
    kind: Kind,
    changed: impl FnMut(Range<u32>, ImageState),
) {
    let new = ImageState::taken(layout, kind);
    let swap = |state: &AtomicU64| ImageState::unpack(state.swap(new.pack(), Ordering::Relaxed));
    runs(states, mips, new, swap, changed);
}

/// Moves the layouts of `mips` to `layout` without a recording (host image copies), leaving
/// their owners alone, and calls `changed(mips, old)` per run of mips whose old state had
/// another layout.
pub(crate) fn take_host_layouts(
    states: &[AtomicU64],
    mips: Range<u32>,
    layout: vk::ImageLayout,
    mut changed: impl FnMut(Range<u32>, vk::ImageLayout),
) {
    let mut old: Option<(Range<u32>, vk::ImageLayout)> = None;
    for_each(states, mips, |state, mips| {
        let bits = state.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |bits| {
            Some(ImageState { layout, ..ImageState::unpack(bits) }.pack())
        });
        let previous = ImageState::unpack(bits.unwrap_or_else(|bits| bits)).layout;
        match &mut old {
            Some((run, run_layout)) if *run_layout == previous => *run = (run.start..mips.end).into(),
            _ => {
                if let Some((run, run_layout)) = old.replace((mips, previous))
                    && run_layout != layout
                {
                    changed(run, run_layout);
                }
            }
        }
    });
    if let Some((run, run_layout)) = old
        && run_layout != layout
    {
        changed(run, run_layout);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vk::ImageLayout as L;

    fn state(layout: L, owner: Option<Kind>) -> ImageState {
        ImageState {
            layout,
            owner,
            released_to: None,
            acquired: false,
        }
    }

    #[test]
    fn states_survive_packing() {
        for layout in [L::UNDEFINED, L::GENERAL, L::ATTACHMENT_OPTIMAL, L::PRESENT_SRC_KHR] {
            for owner in [None, Some(Kind::Graphics), Some(Kind::Compute), Some(Kind::Transfer)] {
                for released_to in [None, Some(Kind::Graphics), Some(Kind::Transfer)] {
                    for acquired in [false, true] {
                        let s = ImageState {
                            layout,
                            owner,
                            released_to,
                            acquired,
                        };
                        assert_eq!(ImageState::unpack(s.pack()), s);
                    }
                }
            }
        }
        // A new image's packs to its layout alone.
        assert_eq!(ImageState::UNDEFINED.pack(), 0);
        let present = state(L::PRESENT_SRC_KHR, None).pack();
        assert_eq!(present, L::PRESENT_SRC_KHR.as_raw() as u64);
    }

    #[test]
    fn an_image_is_accessible_to_its_owner_and_who_it_was_released_to() {
        let graphics = state(L::GENERAL, Some(Kind::Graphics));
        assert!(ImageState::UNDEFINED.accessible_by(Kind::Transfer));
        assert!(graphics.accessible_by(Kind::Graphics));
        assert!(!graphics.accessible_by(Kind::Compute));
        let released = ImageState {
            released_to: Some(Kind::Compute),
            ..graphics
        };
        assert!(released.accessible_by(Kind::Compute));
        assert!(!released.accessible_by(Kind::Transfer));
    }

    fn atomics(states: &[ImageState]) -> Vec<AtomicU64> {
        states.iter().map(|s| s.atomic()).collect()
    }

    fn taken(states: &[AtomicU64], mips: std::ops::Range<u32>, layout: L, kind: Kind) -> Vec<(u32, u32, ImageState)> {
        let mut runs = Vec::new();
        take_states(states, mips.into(), layout, kind, |mips, old| {
            runs.push((mips.start, mips.end, old))
        });
        runs
    }

    #[test]
    fn taking_reports_runs_of_neighbouring_mips_in_the_same_state() {
        let g = |layout| state(layout, Some(Kind::Graphics));
        let states = atomics(&[
            ImageState::UNDEFINED,
            ImageState::UNDEFINED,
            g(L::GENERAL),
            g(L::ATTACHMENT_OPTIMAL),
            g(L::GENERAL),
        ]);
        assert_eq!(
            taken(&states, 0..5, L::GENERAL, Kind::Graphics),
            [
                (0, 2, ImageState::UNDEFINED),
                (3, 4, g(L::ATTACHMENT_OPTIMAL)),
            ]
        );
        assert!(taken(&states, 0..5, L::GENERAL, Kind::Graphics).is_empty());
        // One state for all mips reports them all at once.
        let single = atomics(&[g(L::GENERAL)]);
        assert_eq!(
            taken(&single, 2..6, L::ATTACHMENT_OPTIMAL, Kind::Graphics),
            [(2, 6, g(L::GENERAL))]
        );
        // A view of some mips of a per-mip image holds their states only.
        assert_eq!(
            taken(&states[3..4], 3..4, L::PRESENT_SRC_KHR, Kind::Graphics),
            [(3, 4, g(L::GENERAL))]
        );
    }

    /// The first access on a family takes ownership, also in the same layout.
    #[test]
    fn taking_an_unowned_image_reports_its_state() {
        let states = atomics(&[state(L::GENERAL, None)]);
        let runs = taken(&states, 0..1, L::GENERAL, Kind::Graphics);
        assert_eq!(runs, [(0, 1, state(L::GENERAL, None))]);
        assert_eq!(ImageState::load(&states[0]), state(L::GENERAL, Some(Kind::Graphics)));
    }

    #[test]
    fn a_released_image_is_taken_by_its_receiver() {
        let released = ImageState {
            released_to: Some(Kind::Transfer),
            ..state(L::GENERAL, Some(Kind::Graphics))
        };
        let states = atomics(&[released]);
        assert_eq!(taken(&states, 0..1, L::GENERAL, Kind::Transfer), [(0, 1, released)]);
        assert_eq!(ImageState::load(&states[0]), state(L::GENERAL, Some(Kind::Transfer)));
    }

    /// An access of another family's image panics before any mip moved.
    #[test]
    fn taking_another_familys_image_panics_before_anything_moves() {
        let g = state(L::GENERAL, Some(Kind::Graphics));
        let c = state(L::GENERAL, Some(Kind::Compute));
        let states = atomics(&[g, c]);
        let panic = std::panic::catch_unwind(|| taken(&states, 0..2, L::GENERAL, Kind::Graphics));
        let message = *panic.unwrap_err().downcast::<String>().unwrap();
        assert!(message.contains("`cmd.release::<Graphics>(view)`"), "{message}");
        assert_eq!(ImageState::load(&states[0]), g, "nothing moved");
    }

    #[test]
    fn host_layouts_leave_the_owner() {
        let c = state(L::GENERAL, Some(Kind::Compute));
        let states = atomics(&[ImageState::UNDEFINED, c, c]);
        let mut runs = Vec::new();
        take_host_layouts(&states, (0..3).into(), L::GENERAL, |mips, old| {
            runs.push((mips.start, mips.end, old))
        });
        assert_eq!(runs, [(0, 1, L::UNDEFINED)]);
        assert_eq!(ImageState::load(&states[0]), state(L::GENERAL, None));
        assert_eq!(ImageState::load(&states[2]), c);
    }
}

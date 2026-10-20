//! An image's tracked state, one `AtomicU64` per view or mip: its layout, the kind of queue family that owns it, the kind it was released to, and whether a swapchain image waits for its acquire; how recording an access takes it
use std::{
    range::Range,
    sync::atomic::{AtomicU64, Ordering},
};

use crate::{queue::Kind, vk};

/// Packed into one `u64`: the layout in bits 0..32, the owner in 32..34, the kind it was
/// released to in 34..36, and bit 36 for an acquired swapchain image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ImageState {
    pub(crate) layout: vk::ImageLayout,
    /// Images are `EXCLUSIVE`, so another family's recording needs a release first.
    pub(crate) owner: Option<Kind>,
    pub(crate) released_to: Option<Kind>,
    /// A swapchain image whose first access waits for the acquire's semaphore.
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
    pub(crate) const UNDEFINED: ImageState = ImageState {
        layout: vk::ImageLayout::UNDEFINED,
        owner: None,
        released_to: None,
        acquired: false,
    };

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

    pub(crate) fn atomic(self) -> AtomicU64 {
        AtomicU64::new(self.pack())
    }

    pub(crate) fn load(state: &AtomicU64) -> Self {
        Self::unpack(state.load(Ordering::Relaxed))
    }
}

/// Calls `f` per state and the mips it covers; `states` are an image's, one for all its mips or
/// one per mip of the image (mip `i` at index `i`).
fn for_each(states: &[AtomicU64], mips: Range<u32>, mut f: impl FnMut(&AtomicU64, Range<u32>)) {
    if let [state] = states {
        return f(state, mips);
    }
    for mip in mips {
        f(&states[mip as usize], (mip..mip + 1).into());
    }
}

/// Whether the `kind` family owns no mip of the image whose `states` these are any more: each
/// is released, owned by another family or by none.
pub(crate) fn released_by(states: &[AtomicU64], kind: Kind) -> bool {
    states.iter().all(|state| {
        let state = ImageState::load(state);
        state.owner != Some(kind) || state.released_to.is_some()
    })
}

/// Panics unless a recording of `kind` may access every mip of `mips`: nobody owns it, `kind`
/// does, or its owner released it to `kind`.
#[track_caller]
pub(crate) fn check_states(states: &[AtomicU64], mips: Range<u32>, kind: Kind) {
    for_each(states, mips, |state, _| {
        let state = ImageState::load(state);
        if let Some(owner) = state.owner
            && owner != kind
            && state.released_to != Some(kind)
        {
            panic!(
                "an image the {owner:?} family owns is accessed on the {kind:?} family: record \
                 `cmd.release::<{kind:?}>(view)` on a {owner:?} timeline first, and wait for it"
            );
        }
    });
}

/// Moves the states of `mips`, checked with `check_states`, to `layout`, owned by `kind`, and
/// calls `changed(mips, old)` per run of neighbouring mips with the same old state that differs
/// from the new one.
pub(crate) fn take_checked_states(
    states: &[AtomicU64],
    mips: Range<u32>,
    layout: vk::ImageLayout,
    kind: Kind,
    mut changed: impl FnMut(Range<u32>, ImageState),
) {
    let new = ImageState {
        layout,
        owner: Some(kind),
        released_to: None,
        acquired: false,
    };
    let mut run: Option<(Range<u32>, ImageState)> = None;
    for_each(states, mips, |state, mips| {
        let old = ImageState::unpack(state.swap(new.pack(), Ordering::Relaxed));
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

/// Checks every mip of `mips` first, then flags those `kind` owns released to `to` and calls
/// `released(mips, layout)` per run of neighbouring owned mips in the same layout.
#[track_caller]
pub(crate) fn release_states(
    states: &[AtomicU64],
    mips: Range<u32>,
    kind: Kind,
    to: Kind,
    mut released: impl FnMut(Range<u32>, vk::ImageLayout),
) {
    for_each(states, mips, |state, _| {
        let state = ImageState::load(state);
        let mine = state.owner.is_none_or(|owner| owner == kind);
        assert!(
            mine && state.released_to.is_none(),
            "only the family that owns an image releases it: this one is owned by {:?}{}",
            state.owner,
            match state.released_to {
                Some(to) => format!(" and released to {to:?}"),
                None => String::new(),
            }
        );
    });
    let mut run: Option<(Range<u32>, vk::ImageLayout)> = None;
    for_each(states, mips, |state, mips| {
        let old = ImageState::load(state);
        if old.owner.is_none() {
            if let Some((run_mips, layout)) = run.take() {
                released(run_mips, layout);
            }
            return;
        }
        let new = ImageState {
            released_to: Some(to),
            ..old
        };
        state.store(new.pack(), Ordering::Relaxed);
        match &mut run {
            Some((run_mips, layout)) if *layout == old.layout => run_mips.end = mips.end,
            _ => {
                if let Some((run_mips, layout)) = run.replace((mips, old.layout)) {
                    released(run_mips, layout);
                }
            }
        }
    });
    if let Some((run_mips, layout)) = run {
        released(run_mips, layout);
    }
}

/// Moves the layouts of `mips` to `layout` for host image copies, leaving their owners alone,
/// and calls `changed(mips, old)` per run of mips that had another layout.
#[cfg(feature = "host-image-copy")]
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

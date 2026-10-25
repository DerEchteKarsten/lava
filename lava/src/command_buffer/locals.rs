//! A recording's working copies of the access states of the objects it touches (`Locals`): a timeline has one recording at a time, so its share of an object's `AccessState` belongs to that recording while it records; the first touch locks the object once and copies its share in (pruned), every later access works on the copy without a lock, and `publish` writes the copies back once, when the recording is submitted (or replayed)

use std::hash::{BuildHasherDefault, Hasher};
use std::sync::{Arc, Mutex};

use hashbrown::HashMap;

use crate::command_buffer::access_state::{AccessState, Tracker, Visibility};
use crate::object::{Header, Keep, Object};
use crate::queue::lock;

/// A copy of one object's share of its access state.
struct Local {
    /// Keeps the state's owner (a structure's is its buffer's) alive until `publish`: a
    /// referencing buffer reached as an ancestor isn't kept by the recording otherwise.
    owner: Arc<dyn Object>,
    local: AccessState,
}

/// Hashes the pointer of a state: the low bits are alignment, the rest spread by a multiply.
#[derive(Default)]
struct PointerHasher(u64);

impl Hasher for PointerHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, _: &[u8]) {
        unreachable!("only pointers are hashed")
    }

    fn write_usize(&mut self, n: usize) {
        self.0 = ((n >> 4) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

/// The working copies of a recording, by the address of the object's state.
pub(crate) struct Locals {
    tracker: Tracker,
    entries: Vec<Local>,
    /// Every copy by the address of its object's state: the fallback of the tag in the
    /// object's header (`Keep::tag`, which carries the copy's index), which another timeline's
    /// recording may overwrite.
    index: HashMap<usize, u32, BuildHasherDefault<PointerHasher>>,
    /// The objects the recording names, kept until its submission completed.
    pub(crate) keep: Keep,
}

impl Locals {
    pub(crate) fn new(tracker: Tracker) -> Self {
        Locals {
            tracker,
            entries: Vec::new(),
            index: HashMap::default(),
            keep: Keep::new(tracker.timeline as usize, tracker.value),
        }
    }

    pub(crate) fn tracker(&self) -> &Tracker {
        &self.tracker
    }

    /// The recording's copy of `owner`'s state (an object that has one, else a panic).
    pub(crate) fn of<T: Object>(&mut self, owner: &Arc<T>) -> &mut AccessState {
        let state = owner.access_state().expect("an object with tracked accesses");
        self.entry(owner.header(), state, || owner.clone())
    }

    /// The recording's copy of `owner`'s state, or none for an object without one.
    pub(crate) fn of_dyn(&mut self, owner: &Arc<dyn Object>) -> Option<&mut AccessState> {
        let state = owner.access_state()?;
        Some(self.entry(owner.header(), state, || owner.clone()))
    }

    /// The copy of `state`, by the index the recording's tag in `header` carries (one load for
    /// the common case), else by the map; the first touch imports it and keeps the object.
    fn entry(
        &mut self,
        header: &Header,
        state: &Mutex<AccessState>,
        owner: impl FnOnce() -> Arc<dyn Object>,
    ) -> &mut AccessState {
        if let Some(i) = self.keep.local_index(header) {
            return &mut self.entries[i].local;
        }
        let key = state as *const Mutex<AccessState> as usize;
        let i = match self.index.get(&key) {
            Some(&i) => i as usize,
            None => {
                let i = self.entries.len();
                self.index.insert(key, i as u32);
                let owner = owner();
                self.keep.object_with_index(header, &owner, i);
                self.entries.push(Local {
                    owner,
                    local: lock(state).import(&self.tracker),
                });
                i
            }
        };
        &mut self.entries[i].local
    }

    /// Writes every copy back into its object, replacing the timeline's share, with the
    /// recording's barriers (`vis`) taken into its writes, and forgets them.
    pub(crate) fn publish(&mut self, vis: &Visibility) {
        for mut entry in self.entries.drain(..) {
            entry.local.settle(&self.tracker, vis);
            let state = entry.owner.access_state().expect("imported from it");
            lock(state).publish(&self.tracker, entry.local);
            // Kept still, but with no copy to find.
            self.keep.forget_index(entry.owner.header());
        }
        self.index.clear();
    }
}

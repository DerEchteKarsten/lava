//! Per-frame resources: one value per frame slot of a timeline, reached only through the `Submission` whose `begin` waited for that slot's last submission, so host writes into them never race the GPU
use bytemuck::{AnyBitPattern, NoUninit};

use crate::{
    buffer::{Buffer, HostVisible},
    error::Result,
    queue::{QueueKind, Submission, Timeline},
};

/// One `T` per frame slot of a timeline (buffers, rings, query pools, anything), indexed by
/// [`Submission::slot`]. A begun submission's slot is free (`begin` waited for its last
/// submission), so what [`Self::get_mut`] and [`Self::host_mut`] give is the GPU's no longer:
/// the convention of one range per frame slot, as a type.
///
/// Its values belong to the timeline it was made for: a submission of another timeline (or a
/// later one at the same index with another slot count) panics.
pub struct PerFrame<T> {
    slots: Box<[T]>,
    timeline: usize,
}

impl<T> PerFrame<T> {
    /// A value per frame slot of `timeline`, `make(slot)` each.
    pub fn new<K: QueueKind>(
        timeline: &Timeline<K>,
        mut make: impl FnMut(usize) -> Result<T>,
    ) -> Result<Self> {
        let slots = (0..timeline.in_flight() as usize)
            .map(&mut make)
            .collect::<Result<_>>()?;
        Ok(PerFrame {
            slots,
            timeline: timeline.index(),
        })
    }

    #[track_caller]
    fn slot<K: QueueKind>(&self, submission: &Submission<'_, K>) -> usize {
        let (timeline, slots) = submission.timeline();
        assert!(
            timeline == self.timeline && slots == self.slots.len(),
            "a `PerFrame` is indexed by submissions of the timeline it was made for"
        );
        submission.slot()
    }

    /// The value of `submission`'s frame slot.
    #[track_caller]
    pub fn get<K: QueueKind>(&self, submission: &Submission<'_, K>) -> &T {
        &self.slots[self.slot(submission)]
    }

    /// The value of `submission`'s frame slot, which no submission in flight reaches.
    #[track_caller]
    pub fn get_mut<K: QueueKind>(&mut self, submission: &Submission<'_, K>) -> &mut T {
        let slot = self.slot(submission);
        &mut self.slots[slot]
    }

    /// Every slot's value, in slot order (to resize or drop them all).
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.slots.iter_mut()
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.slots.iter()
    }

    /// How many slots it has, the timeline's `in_flight`.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

impl<T: NoUninit + AnyBitPattern, M: HostVisible> PerFrame<Buffer<T, M>> {
    /// The mapping of `submission`'s slot's buffer, which no earlier submission of the slot
    /// still reads ([`Buffer::host_mut`], with its check of GPU writes).
    #[track_caller]
    pub fn host_mut<K: QueueKind>(&mut self, submission: &Submission<'_, K>) -> &mut [T] {
        self.get_mut(submission).host_mut()
    }
}

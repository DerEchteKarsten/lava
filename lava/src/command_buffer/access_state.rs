//! The accesses later commands may have to wait for, kept in each buffer and image (`AccessState`, one per buffer or image, entries tagged with their timeline) and, for the reads of every acceleration structure a TLAS build or trace makes, per timeline: what its submissions read and wrote (layout transitions and acquires count as writes), each write with what barriers already made it visible to (its own pairs, and the memory barriers its recording logged after it), how recording a command adds to them, how they shrink once the GPU finished them, and the host-access check of a buffer's incomplete writes

use std::range::Range;

use crate::vk::{self, AccessFlags2 as A, PipelineStageFlags2 as S2};
use smallvec::SmallVec;

use crate::command_buffer::Barriers;
use crate::queue::timeline::{INDICES, completed};

/// How many writes of one timeline an object keeps before the completed ones fold into one.
pub(crate) const MAX_WRITES: usize = 16;

/// How many scopes a folded write keeps; dropping one only costs a barrier.
const MAX_VISIBLE: usize = 4;

/// What a recording knows of its timeline while it tracks accesses.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Tracker {
    pub(crate) timeline: u8,
    /// The value its submission signals; what its accesses are tagged with.
    pub(crate) value: u64,
    /// Entries of its timeline at or below it are a former owner's of the index, on another
    /// queue maybe, which a semaphore wait orders: ignored.
    pub(crate) base: u64,
    /// Its timeline's completed value when the recording began: reads up to it are done.
    pub(crate) completed: u64,
}

/// A read, with the timeline value of the latest submission it stands for.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Read {
    pub(crate) timeline: u8,
    pub(crate) value: u64,
    pub(crate) range: Range<u64>,
    pub(crate) stage: S2,
    pub(crate) access: A,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Write {
    pub(crate) timeline: u8,
    pub(crate) value: u64,
    pub(crate) range: Range<u64>,
    pub(crate) stage: S2,
    pub(crate) access: A,
    /// The destination scopes of the barriers recorded since, each visible to its own pairs.
    pub(crate) visible: SmallVec<[(S2, A); 2]>,
    /// The recording (its value) and the length of its barrier log when `visible` last took
    /// that log's memory barriers in: later ones may make it visible too.
    pub(crate) applied: (u64, u32),
}

impl Write {
    fn new(t: &Tracker, seq: u32, range: Range<u64>, stage: S2, access: A) -> Self {
        Write {
            timeline: t.timeline,
            value: t.value,
            range,
            stage,
            access,
            visible: SmallVec::new(),
            applied: (t.value, seq),
        }
    }

    /// Whether one barrier made the write visible to `access` at `stage`; two can't combine.
    pub(crate) fn visible_to(&self, stage: S2, access: A) -> bool {
        (self.visible.iter())
            .any(|&(stages, accesses)| stages.contains(stage) && accesses.contains(access))
    }

    /// A scope merges only into one of the same stages or accesses, or replaces one it contains:
    /// only then is the merged scope exactly the pairs of both.
    pub(crate) fn make_visible(&mut self, stage: S2, access: A) {
        if self.visible_to(stage, access) {
            return;
        }
        (self.visible).retain(|&mut (stages, accesses)| {
            !(stage.contains(stages) && access.contains(accesses))
        });
        for (stages, accesses) in &mut self.visible {
            if *stages == stage {
                *accesses |= access;
                return;
            }
            if *accesses == access {
                *stages |= stage;
                return;
            }
        }
        self.visible.push((stage, access));
    }

    /// Takes in the memory barriers of the recording `t` logged since it last looked (`log`,
    /// this recording's barriers in order): a barrier makes every write in its source scope
    /// visible to its destination, also writes it wasn't recorded for. A write of an earlier
    /// recording comes before all of this one's barriers.
    fn apply(&mut self, t: &Tracker, log: &[Barriers]) {
        let from = match self.applied {
            (value, seq) if value == t.value => seq as usize,
            _ => 0,
        };
        for barriers in log.iter().skip(from) {
            if let Some(m) = barriers.memory
                && m.src_stage_mask.contains(self.stage)
                && m.src_access_mask.contains(self.access)
            {
                self.make_visible(m.dst_stage_mask, m.dst_access_mask);
            }
        }
        self.applied = (t.value, log.len() as u32);
    }

    /// Keeps only the pairs `other` is visible to as well (each scope is a box of stages and
    /// accesses, so the pairs both see are the boxes' intersections), at most `MAX_VISIBLE`.
    fn intersect_visible(&mut self, other: &[(S2, A)]) {
        let mine = std::mem::take(&mut self.visible);
        for &(stages, accesses) in &mine {
            for &(other_stages, other_accesses) in other {
                let (s, a) = (stages & other_stages, accesses & other_accesses);
                if !s.is_empty() && !a.is_empty() {
                    self.make_visible(s, a);
                }
            }
        }
        self.visible.truncate(MAX_VISIBLE);
    }
}

fn overlap(a: Range<u64>, b: Range<u64>) -> bool {
    a.start.max(b.start) < a.end.min(b.end)
}

fn covers(outer: Range<u64>, inner: Range<u64>) -> bool {
    outer.start <= inner.start && inner.end <= outer.end
}

/// The accesses of one buffer (ranges of device addresses) or one image (one range for the
/// whole image: its accesses are per image, as its barriers' memory dependencies are) by every
/// timeline that recorded one.
#[derive(Clone, Debug, Default)]
pub(crate) struct AccessState {
    pub(crate) reads: SmallVec<[Read; 2]>,
    pub(crate) writes: SmallVec<[Write; 2]>,
}

/// Where an image's accesses lie: every access covers the image.
pub(crate) const WHOLE: Range<u64> = Range { start: 0, end: 1 };

impl AccessState {
    /// Drops the entries `t` no longer needs: its timeline's former owner's, and its reads that
    /// completed.
    fn prune(&mut self, t: &Tracker) {
        let mine = |timeline: u8, value: u64| timeline == t.timeline && value <= t.base;
        self.reads.retain(|r| {
            !mine(r.timeline, r.value) && !(r.timeline == t.timeline && r.value <= t.completed)
        });
        self.writes.retain(|w| !mine(w.timeline, w.value));
    }

    /// What an access of `range` at `stage` with `access` waits for on `t`'s timeline: the
    /// source stages and accesses of its hazards, `None` for none (a hazard may have no stage,
    /// a present's transition). A write waits for the reads it overlaps, any access for the
    /// overlapping writes no barrier made visible to it.
    pub(crate) fn hazards(
        &mut self,
        t: &Tracker,
        log: &[Barriers],
        range: Range<u64>,
        stage: S2,
        access: A,
    ) -> Option<(S2, A)> {
        self.prune(t);
        let mut hazard: Option<(S2, A)> = None;
        let mut add = |s: S2, a: A| {
            let (stages, accesses) = hazard.get_or_insert((S2::NONE, A::NONE));
            *stages |= s;
            *accesses |= a;
        };
        if access.intersects(WRITE_ACCESS) {
            for r in self.reads.iter().filter(|r| r.timeline == t.timeline) {
                if overlap(r.range, range) {
                    add(r.stage, r.access);
                }
            }
        }
        for w in self.writes.iter_mut().filter(|w| w.timeline == t.timeline) {
            if !overlap(w.range, range) {
                continue;
            }
            w.apply(t, log);
            if !w.visible_to(stage, access) {
                add(w.stage, w.access);
            }
        }
        hazard
    }

    /// The stages and accesses of every access of `t`'s timeline: what a layout transition or
    /// a release waits for.
    pub(crate) fn sources(&mut self, t: &Tracker) -> (S2, A) {
        self.prune(t);
        let (mut stage, mut access) = (S2::NONE, A::NONE);
        let reads = self.reads.iter().map(|r| (r.timeline, r.stage, r.access));
        let writes = self.writes.iter().map(|w| (w.timeline, w.stage, w.access));
        for (timeline, s, a) in reads.chain(writes) {
            if timeline == t.timeline {
                stage |= s;
                access |= a;
            }
        }
        (stage, access)
    }

    /// Tracks an access of a command `t` recorded after the barriers of `log`, its own
    /// included. An older access is only forgotten once a newer write covers its whole range; a
    /// newer read doesn't wait for older reads, so it takes over those it covers.
    pub(crate) fn record(
        &mut self,
        t: &Tracker,
        log: &[Barriers],
        range: Range<u64>,
        stage: S2,
        access: A,
    ) {
        let mine = |timeline: u8| timeline == t.timeline;
        // The command's barriers (the log's last) may have made its writes visible: taken in
        // now, so the visibility outlives the recording's log.
        for w in self.writes.iter_mut().filter(|w| mine(w.timeline)) {
            w.apply(t, log);
        }
        if access.intersects(WRITE_ACCESS) {
            (self.writes).retain(|w| !(mine(w.timeline) && covers(range, w.range)));
            let seq = log.len() as u32;
            self.writes.push(Write::new(t, seq, range, stage, access));
            self.reads
                .retain(|r| !(mine(r.timeline) && covers(range, r.range)));
            self.fold(t, log);
        }
        if access.intersects(READ_ACCESS) {
            let mut read = Read {
                timeline: t.timeline,
                value: t.value,
                range,
                stage,
                access,
            };
            self.reads.retain(|r| {
                let covered = mine(r.timeline) && covers(range, r.range);
                if covered {
                    read.stage |= r.stage;
                    read.access |= r.access;
                }
                !covered
            });
            self.reads.push(read);
        }
    }

    /// Once `t`'s timeline has more than `MAX_WRITES` writes here, folds the completed ones
    /// into one over all their ranges, which keeps the visibility all of them had (their own
    /// pairs, with `log`'s barriers taken in first).
    fn fold(&mut self, t: &Tracker, log: &[Barriers]) {
        let mine = (self.writes.iter())
            .filter(|w| w.timeline == t.timeline)
            .count();
        if mine <= MAX_WRITES {
            return;
        }
        let mut folded: Option<Write> = None;
        self.writes.retain_mut(|w| {
            let fold = w.timeline == t.timeline && w.value <= t.completed;
            if fold {
                w.apply(t, log);
                match &mut folded {
                    None => folded = Some(w.clone()),
                    Some(f) => {
                        f.range = Range {
                            start: f.range.start.min(w.range.start),
                            end: f.range.end.max(w.range.end),
                        };
                        f.stage |= w.stage;
                        f.access |= w.access;
                        f.value = f.value.max(w.value);
                        f.intersect_visible(&w.visible);
                    }
                }
            }
            !fold
        });
        if let Some(folded) = folded {
            self.writes.push(folded);
        }
    }

    /// A layout transition or queue family acquire of the image by `t`, after the barriers of
    /// `log` (its own included): it is the image's latest write, at the destination stages of
    /// its barriers and visible there (`scopes`). Every earlier access of `t`'s timeline happens
    /// before it (the barrier waited for them, or for a semaphore that did), so they are
    /// dropped.
    pub(crate) fn transition(&mut self, t: &Tracker, log: &[Barriers], scopes: &[(S2, A)]) {
        self.forget(t);
        let stage = scopes.iter().fold(S2::NONE, |stage, &(s, _)| stage | s);
        let mut write = Write::new(t, log.len() as u32, WHOLE, stage, A::NONE);
        for &(stage, access) in scopes {
            write.make_visible(stage, access);
        }
        self.writes.push(write);
    }

    /// Drops every access of `t`'s timeline: a release hands the image to another family, which
    /// waits for this submission as a whole.
    pub(crate) fn forget(&mut self, t: &Tracker) {
        self.reads.retain(|r| r.timeline != t.timeline);
        self.writes.retain(|w| w.timeline != t.timeline);
    }

    /// The latest value of a submitted write overlapping `range` that its timeline hasn't
    /// completed, with its timeline: what host access waits for. Writes recorded but not
    /// submitted yet don't count (the GPU hasn't seen them), former owners' writes of an index
    /// count as complete.
    pub(crate) fn incomplete_write(&self, range: Range<u64>) -> Option<(usize, u64)> {
        self.incomplete_write_of(range, |i| {
            let index = &INDICES[i];
            (completed(i), index.base(), index.submitted())
        })
    }

    /// `incomplete_write` with each timeline's completed, base and submitted values from
    /// `values`.
    pub(crate) fn incomplete_write_of(
        &self,
        range: Range<u64>,
        values: impl Fn(usize) -> (u64, u64, u64),
    ) -> Option<(usize, u64)> {
        (self.writes.iter())
            .filter(|w| {
                let (completed, base, submitted) = values(w.timeline as usize);
                w.value > completed.max(base) && w.value <= submitted && overlap(w.range, range)
            })
            .map(|w| (w.timeline as usize, w.value))
            .max_by_key(|&(_, value)| value)
    }
}

/// The reads of every acceleration structure (the BLASes a TLAS build or trace reaches, which
/// lava can't list) of one timeline, and its acceleration structure writes they wait for:
/// kept per timeline beside the structures' own accesses.
#[derive(Clone, Debug, Default)]
pub(crate) struct StructureAccesses {
    reads: Option<Read>,
    /// Every acceleration structure write of the timeline, merged: they don't cover each other.
    writes: Option<Write>,
}

impl StructureAccesses {
    /// What a read of every structure waits for: the structure writes no barrier made visible.
    pub(crate) fn read_hazards(
        &mut self,
        t: &Tracker,
        log: &[Barriers],
        stage: S2,
        access: A,
    ) -> Option<(S2, A)> {
        let w = self.writes.as_mut()?;
        w.apply(t, log);
        (!w.visible_to(stage, access)).then_some((w.stage, w.access))
    }

    /// What a structure write waits for: every pending read of all structures.
    pub(crate) fn write_hazards(&self, t: &Tracker) -> Option<(S2, A)> {
        let r = self.reads.as_ref()?;
        (r.value > t.completed).then_some((r.stage, r.access))
    }

    pub(crate) fn record_read(&mut self, t: &Tracker, stage: S2, access: A) {
        match &mut self.reads {
            Some(r) if r.value > t.completed => {
                r.stage |= stage;
                r.access |= access;
                r.value = t.value;
            }
            reads => {
                *reads = Some(Read {
                    timeline: t.timeline,
                    value: t.value,
                    range: WHOLE,
                    stage,
                    access,
                })
            }
        }
    }

    /// A structure write: its barrier waited for every structure read, and later reads run at
    /// the build stage after it.
    /// The merged write is visible to nothing yet: the pairs both see, of which the new write
    /// has none.
    pub(crate) fn record_write(&mut self, t: &Tracker, log: &[Barriers], stage: S2, access: A) {
        self.reads = None;
        let mut write = Write::new(t, log.len() as u32, WHOLE, stage, access);
        if let Some(old) = self.writes.take()
            && old.timeline == t.timeline
            && old.value > t.base
        {
            write.stage |= old.stage;
            write.access |= old.access;
        }
        self.writes = Some(write);
    }
}

/// The union of these `AccessFlags2`, as a constant (ash's `|` isn't `const`).
macro_rules! access_set {
    ($($flag:ident,)*) => {
        vk::AccessFlags2::from_raw(0 $(| vk::AccessFlags2::$flag.as_raw())*)
    };
}

/// Every access that writes.
pub(crate) const WRITE_ACCESS: vk::AccessFlags2 = access_set!(
    SHADER_WRITE,
    COLOR_ATTACHMENT_WRITE,
    DEPTH_STENCIL_ATTACHMENT_WRITE,
    TRANSFER_WRITE,
    HOST_WRITE,
    MEMORY_WRITE,
    SHADER_STORAGE_WRITE,
    TRANSFORM_FEEDBACK_WRITE_EXT,
    TRANSFORM_FEEDBACK_COUNTER_WRITE_EXT,
    ACCELERATION_STRUCTURE_WRITE_KHR,
    COMMAND_PREPROCESS_WRITE_NV,
    VIDEO_DECODE_WRITE_KHR,
    VIDEO_ENCODE_WRITE_KHR,
    MICROMAP_WRITE_EXT,
    OPTICAL_FLOW_WRITE_NV,
);

/// Every access that reads.
pub(crate) const READ_ACCESS: vk::AccessFlags2 = access_set!(
    INDIRECT_COMMAND_READ,
    INDEX_READ,
    VERTEX_ATTRIBUTE_READ,
    UNIFORM_READ,
    INPUT_ATTACHMENT_READ,
    SHADER_READ,
    COLOR_ATTACHMENT_READ,
    DEPTH_STENCIL_ATTACHMENT_READ,
    TRANSFER_READ,
    HOST_READ,
    MEMORY_READ,
    SHADER_SAMPLED_READ,
    SHADER_STORAGE_READ,
    SHADER_BINDING_TABLE_READ_KHR,
    TRANSFORM_FEEDBACK_COUNTER_READ_EXT,
    CONDITIONAL_RENDERING_READ_EXT,
    COLOR_ATTACHMENT_READ_NONCOHERENT_EXT,
    ACCELERATION_STRUCTURE_READ_KHR,
    FRAGMENT_DENSITY_MAP_READ_EXT,
    FRAGMENT_SHADING_RATE_ATTACHMENT_READ_KHR,
    COMMAND_PREPROCESS_READ_NV,
    DESCRIPTOR_BUFFER_READ_EXT,
    INVOCATION_MASK_READ_HUAWEI,
    VIDEO_DECODE_READ_KHR,
    VIDEO_ENCODE_READ_KHR,
    MICROMAP_READ_EXT,
    OPTICAL_FLOW_READ_NV,
);

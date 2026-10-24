//! The accesses later commands may have to wait for: what a timeline's submissions read and wrote (layout transitions and acquires count as writes), each write with what barriers already made it visible to, how recording a command adds to them, and how they shrink once the GPU finished them or a resource is destroyed, and the scan host access makes for incomplete writes of a buffer

use std::range::Range;

use crate::vk;
use smallvec::SmallVec;

use crate::command_buffer::{
    Barriers, BufferAccess, EVERY_ACCELERATION_STRUCTURE, EVERY_BUFFER, ImageAccess,
};

pub(crate) const MAX_PENDING_BUFFER_WRITES: usize = 64;

#[derive(Debug, Default, Clone)]
pub(crate) struct PendingAccesses {
    pub(crate) buffer_reads: SmallVec<[PendingRead<BufferAccess>; 8]>,
    pub(crate) image_reads: SmallVec<[PendingRead<ImageAccess>; 8]>,
    pub(crate) buffer_writes: SmallVec<[PendingWrite<BufferAccess>; 8]>,
    pub(crate) image_writes: SmallVec<[PendingWrite<ImageAccess>; 8]>,
}

/// A read with the timeline value of the latest submission it stands for.
#[derive(Debug, Clone)]
pub(crate) struct PendingRead<A> {
    pub(crate) read: A,
    pub(crate) value: u64,
}

impl<A> std::ops::Deref for PendingRead<A> {
    type Target = A;
    fn deref(&self) -> &A {
        &self.read
    }
}

/// How many scopes a folded write keeps; dropping one only costs a barrier.
const MAX_VISIBLE: usize = 4;

#[derive(Debug, Clone)]
pub(crate) struct PendingWrite<A> {
    pub(crate) write: A,
    pub(crate) value: u64,
    /// The destination scopes of the barriers recorded since, each visible to its own pairs.
    pub(crate) visible: SmallVec<[(vk::PipelineStageFlags2, vk::AccessFlags2); 2]>,
}

impl<A> PendingWrite<A> {
    pub(crate) fn new(write: A, value: u64) -> Self {
        Self {
            write,
            value,
            visible: SmallVec::new(),
        }
    }

    /// Whether one barrier made the write visible to `access` at `stage`; two can't combine.
    pub(crate) fn visible_to(
        &self,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
    ) -> bool {
        (self.visible.iter())
            .any(|&(stages, accesses)| stages.contains(stage) && accesses.contains(access))
    }

    /// A scope merges only into one of the same stages or accesses, or replaces one it contains:
    /// only then is the merged scope exactly the pairs of both.
    pub(crate) fn make_visible(
        &mut self,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
    ) {
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

    /// Keeps only the pairs `other` is visible to as well (each scope is a box of stages and
    /// accesses, so the pairs both see are the boxes' intersections), at most `MAX_VISIBLE`.
    fn intersect_visible(&mut self, other: &[(vk::PipelineStageFlags2, vk::AccessFlags2)]) {
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

impl PendingAccesses {
    /// Updates the accesses after `barriers` of the command of timeline value `value`: a memory
    /// barrier covers every write in its source scope, also those it wasn't recorded for, and a
    /// layout transition or queue family acquire is the image's latest write, at its barrier's
    /// destination stages and visible there. Every earlier access of the image happens before
    /// it (the barrier waited for them, or for a semaphore that did), so they are dropped.
    pub(crate) fn made_visible(&mut self, barriers: &Barriers, value: u64) {
        let memory = barriers.memory.map(|m| {
            (
                m.src_stage_mask,
                m.src_access_mask,
                m.dst_stage_mask,
                m.dst_access_mask,
            )
        });
        if let Some((src_stage, src_access, dst_stage, dst_access)) = memory {
            for w in &mut self.buffer_writes {
                if src_stage.contains(w.write.stage) && src_access.contains(w.write.access) {
                    w.make_visible(dst_stage, dst_access);
                }
            }
        }
        if let Some((src_stage, src_access, dst_stage, dst_access)) = memory {
            for w in &mut self.image_writes {
                if src_stage.contains(w.write.stage) && src_access.contains(w.write.access) {
                    w.make_visible(dst_stage, dst_access);
                }
            }
        }
        // Every barrier of `images` transitions a layout.
        let moved = barriers.acquires.iter().chain(&barriers.images);
        for (i, barrier) in moved.clone().enumerate() {
            let image = barrier.image;
            if moved.clone().take(i).any(|b| b.image == image) {
                continue;
            }
            let range = barrier.subresource_range;
            let mut write = PendingWrite::new(
                ImageAccess {
                    stage: vk::PipelineStageFlags2::NONE,
                    access: vk::AccessFlags2::NONE,
                    image,
                    layout: barrier.new_layout,
                    aspect: range.aspect_mask,
                    mip_range: (range.base_mip_level..range.base_mip_level + range.level_count)
                        .into(),
                },
                value,
            );
            for b in moved.clone().skip(i).filter(|b| b.image == image) {
                let mips = b.subresource_range;
                let mip_range = &mut write.write.mip_range;
                mip_range.start = mip_range.start.min(mips.base_mip_level);
                mip_range.end = mip_range.end.max(mips.base_mip_level + mips.level_count);
                write.write.stage |= b.dst_stage_mask;
                write.make_visible(b.dst_stage_mask, b.dst_access_mask);
            }
            self.forget_image(image);
            self.image_writes.push(write);
        }
    }

    /// Drops the buffer and image reads up to `completed`, and once there are too many buffer
    /// writes, folds the completed ones into one write of `EVERY_BUFFER`, which still needs
    /// visibility, except where a barrier made every folded write visible already.
    pub(crate) fn retire(&mut self, completed: u64) {
        self.buffer_reads.retain(|read| read.value > completed);
        self.image_reads.retain(|read| read.value > completed);
        if self.buffer_writes.len() <= MAX_PENDING_BUFFER_WRITES {
            return;
        }
        let every = (self.buffer_writes.iter()).position(|w| w.write.range == EVERY_BUFFER);
        let mut folded = every.map(|i| self.buffer_writes.swap_remove(i));
        self.buffer_writes.retain(|w| {
            let fold = w.value <= completed;
            if fold {
                match &mut folded {
                    None => folded = Some(w.clone()),
                    Some(f) => {
                        f.write.stage |= w.write.stage;
                        f.write.access |= w.write.access;
                        f.intersect_visible(&w.visible);
                    }
                }
            }
            !fold
        });
        if let Some(mut folded) = folded {
            folded.write.range = EVERY_BUFFER;
            folded.value = 0;
            self.buffer_writes.push(folded);
        }
    }

    pub(crate) fn forget_image(&mut self, image: vk::Image) {
        self.image_reads.retain(|read| read.image != image);
        self.image_writes.retain(|write| write.write.image != image);
    }

    /// Tracks the accesses of a command of the submission of timeline value `value`. An older
    /// buffer access is only forgotten once a newer write covers its whole range; a newer read
    /// doesn't wait for older reads, so it takes over those it covers.
    pub(crate) fn record(
        &mut self,
        value: u64,
        image_accesses: impl Iterator<Item = ImageAccess>,
        buffer_accesses: impl Iterator<Item = BufferAccess>,
    ) {
        for access in buffer_accesses {
            let covers = |a: &BufferAccess| {
                access.range.start <= a.range.start && a.range.end <= access.range.end
            };
            let writes = access.access.intersects(WRITE_ACCESS);
            let reads = access.access.intersects(READ_ACCESS);

            if writes {
                self.buffer_writes.retain(|a| !covers(&a.write));
                self.buffer_writes.push(PendingWrite::new(access, value));
                // Its barrier waited for every structure read, and later reads run at the build
                // stage after it.
                let structure_write = access
                    .access
                    .contains(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR);
                self.buffer_reads.retain(|a| {
                    !covers(a) && !(structure_write && a.range == EVERY_ACCELERATION_STRUCTURE)
                });
            }
            if reads {
                let mut read = access;
                self.buffer_reads.retain(|a| {
                    let covered = covers(a);
                    if covered {
                        read.stage |= a.stage;
                        read.access |= a.access;
                    }
                    !covered
                });
                self.buffer_reads.push(PendingRead { read, value });
            }
        }

        for access in image_accesses {
            let same_image = |a: &ImageAccess| a.image == access.image;
            let writes = access.access.intersects(WRITE_ACCESS);
            let reads = access.access.intersects(READ_ACCESS);

            if writes {
                self.image_writes.retain(|a| !same_image(&a.write));
                self.image_writes
                    .push(PendingWrite::new(access.clone(), value));
                self.image_reads.retain(|a| !same_image(a));
            }
            if reads {
                let mut read = access.clone();
                self.image_reads.retain(|a| {
                    let same = same_image(a);
                    if same {
                        read.stage |= a.stage;
                        read.access |= a.access;
                    }
                    !same
                });
                self.image_reads.push(PendingRead { read, value });
            }
        }
    }
}

/// The latest value of the `writes` (ranges with the values of their submissions) that
/// overlap `range` and that `completed` doesn't cover. `EVERY_BUFFER` overlaps every range but
/// only ever holds completed writes; `EVERY_ACCELERATION_STRUCTURE` is empty.
pub(crate) fn incomplete_write(
    writes: impl Iterator<Item = (Range<u64>, u64)>,
    range: Range<u64>,
    completed: u64,
) -> Option<u64> {
    writes
        .filter(|(write, value)| {
            *value > completed && write.start.max(range.start) < write.end.min(range.end)
        })
        .map(|(_, value)| value)
        .max()
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

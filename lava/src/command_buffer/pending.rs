//! The accesses later commands may have to wait for: what a queue's submissions read and wrote, each write with what barriers already made it visible to, how recording a command adds to them, and how they shrink once the GPU finished them or a resource is destroyed

use crate::vk;
use smallvec::SmallVec;

use crate::command_buffer::{
    Barriers, BufferAccess, EVERY_ACCELERATION_STRUCTURE, EVERY_BUFFER, ImageAccess,
};

/// How many pending buffer writes are kept before the completed ones are folded into one.
pub(crate) const MAX_PENDING_BUFFER_WRITES: usize = 64;

#[derive(Debug, Default, Clone)]
pub(crate) struct PendingAccesses {
    pub(crate) buffer_reads: SmallVec<[PendingRead; 8]>,
    pub(crate) image_reads: SmallVec<[ImageAccess; 8]>,
    pub(crate) buffer_writes: SmallVec<[PendingWrite<BufferAccess>; 8]>,
    pub(crate) image_writes: SmallVec<[PendingWrite<ImageAccess>; 8]>,
    /// The epoch of the frame being recorded, which the accesses it records are tagged with.
    pub(crate) epoch: u64,
}

/// A buffer read later writes may have to wait for, with the epoch of the frame that made it:
/// once that completed, nothing has to.
#[derive(Debug, Clone)]
pub(crate) struct PendingRead {
    pub(crate) read: BufferAccess,
    pub(crate) epoch: u64,
}

impl std::ops::Deref for PendingRead {
    type Target = BufferAccess;
    fn deref(&self) -> &BufferAccess {
        &self.read
    }
}

/// A write later commands may have to wait for, and what the barriers recorded since already
/// made it visible to: those accesses need no barrier for it anymore.
#[derive(Debug, Clone)]
pub(crate) struct PendingWrite<A> {
    pub(crate) write: A,
    /// The epoch of the frame that made it.
    pub(crate) epoch: u64,
    /// The destination scopes of those barriers: each made the write visible to its accesses
    /// at its stages (every pair of them), but not to its accesses at another barrier's stages.
    pub(crate) visible: SmallVec<[(vk::PipelineStageFlags2, vk::AccessFlags2); 2]>,
}

impl<A> PendingWrite<A> {
    pub(crate) fn new(write: A, epoch: u64) -> Self {
        Self {
            write,
            epoch,
            visible: SmallVec::new(),
        }
    }

    /// Whether one barrier made the write visible to `access` at `stage`: two barriers, one to
    /// sampled reads in fragment shaders and one to storage reads in compute shaders, don't
    /// make it visible to sampled reads in compute shaders.
    pub(crate) fn visible_to(
        &self,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
    ) -> bool {
        (self.visible.iter()).any(|&(stages, accesses)| {
            stages.contains(stage) && accesses.contains(access)
        })
    }

    /// Remembers that a barrier made the write visible to `access` at `stage`. A scope merges
    /// into one of the same stages or the same accesses, or replaces one it contains: only then
    /// is the merged scope exactly the pairs of both.
    pub(crate) fn make_visible(&mut self, stage: vk::PipelineStageFlags2, access: vk::AccessFlags2) {
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
}

impl PendingAccesses {
    /// Remembers what `barriers` made the pending writes in their source scope visible to.
    /// A memory barrier covers every write of its source stages and access types, also those
    /// it wasn't recorded for.
    pub(crate) fn made_visible(&mut self, barriers: &Barriers) {
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
        for w in &mut self.image_writes {
            let image = w.write.image;
            let transitions = barriers
                .images
                .iter()
                .filter(|b| b.image == image)
                .map(|b| {
                    (
                        b.src_stage_mask,
                        b.src_access_mask,
                        b.dst_stage_mask,
                        b.dst_access_mask,
                    )
                });
            for (src_stage, src_access, dst_stage, dst_access) in
                memory.into_iter().chain(transitions)
            {
                if src_stage.contains(w.write.stage) && src_access.contains(w.write.access) {
                    w.make_visible(dst_stage, dst_access);
                }
            }
        }
    }

    /// Drops what the GPU finished, every epoch up to `completed`: the buffer reads, which no
    /// later write has to wait for any more, and, once there are many, the buffer writes, which
    /// later accesses still have to be made visible to: they fold into one write of every
    /// address (`EVERY_BUFFER`). Image accesses stay; there is one of each kind per image.
    pub(crate) fn retire(&mut self, completed: u64) {
        self.buffer_reads.retain(|read| read.epoch > completed);
        if self.buffer_writes.len() <= MAX_PENDING_BUFFER_WRITES {
            return;
        }
        let (mut stage, mut access) = (vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE);
        self.buffer_writes.retain(|w| {
            let fold = w.epoch <= completed && w.write.range != EVERY_BUFFER;
            if fold {
                stage |= w.write.stage;
                access |= w.write.access;
            }
            !fold
        });
        if stage.is_empty() && access.is_empty() {
            return;
        }
        match (self.buffer_writes.iter_mut()).find(|w| w.write.range == EVERY_BUFFER) {
            Some(every) => {
                every.write.stage |= stage;
                every.write.access |= access;
                every.visible.clear();
            }
            None => {
                let write = BufferAccess {
                    stage,
                    access,
                    range: EVERY_BUFFER,
                };
                self.buffer_writes.push(PendingWrite::new(write, 0));
            }
        }
    }

    /// Forgets the accesses of an image that is being destroyed.
    pub(crate) fn forget_image(&mut self, image: vk::Image) {
        self.image_reads.retain(|read| read.image != image);
        self.image_writes.retain(|write| write.write.image != image);
    }

    /// Forgets the accesses within the addresses of a buffer that is being destroyed.
    pub(crate) fn forget_buffer(&mut self, range: std::ops::Range<u64>) {
        let outside = |access: &BufferAccess| !range.contains(&access.range.start);
        self.buffer_reads.retain(|read| outside(&read.read));
        self.buffer_writes.retain(|write| outside(&write.write));
    }

    /// Adds the accesses of another queue's submissions, which this queue's have to wait for
    /// too. An access both have is only waited for twice.
    pub(crate) fn append(&mut self, other: PendingAccesses) {
        self.buffer_reads.extend(other.buffer_reads);
        self.image_reads.extend(other.image_reads);
        self.buffer_writes.extend(other.buffer_writes);
        self.image_writes.extend(other.image_writes);
    }

    /// Remembers the accesses of a command that was just recorded, so later commands can be
    /// synchronised against them.
    ///
    /// An older buffer access is only forgotten once a newer one covers its whole range: a
    /// partially overlapped write still has bytes nobody has synchronised with. A covered write
    /// is forgotten by a newer write, a covered read by a newer write, whose barrier waited for
    /// it. A newer read doesn't wait for older reads, so it takes over their stages and accesses.
    pub(crate) fn record(
        &mut self,
        image_acceses: impl Iterator<Item = ImageAccess>,
        buffer_acceses: impl Iterator<Item = BufferAccess>,
    ) {
        for access in buffer_acceses {
            let covers = |a: &BufferAccess| {
                access.range.start <= a.range.start && a.range.end <= access.range.end
            };
            let writes = access.access.intersects(all_write_access());
            let reads = access.access.intersects(all_read_access());

            if writes {
                self.buffer_writes.retain(|a| !covers(&a.write));
                self.buffer_writes
                    .push(PendingWrite::new(access.clone(), self.epoch));
                // An acceleration structure write waited for every read of every structure,
                // and every later one runs at the same build stage, after that barrier.
                let structure_write = access
                    .access
                    .contains(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR);
                self.buffer_reads.retain(|a| {
                    !covers(a) && !(structure_write && a.range == EVERY_ACCELERATION_STRUCTURE)
                });
            }
            if reads {
                let mut read = access.clone();
                self.buffer_reads.retain(|a| {
                    let covered = covers(a);
                    if covered {
                        read.stage |= a.stage;
                        read.access |= a.access;
                    }
                    !covered
                });
                self.buffer_reads.push(PendingRead {
                    read,
                    epoch: self.epoch,
                });
            }
        }

        for access in image_acceses {
            let same_image = |a: &ImageAccess| a.image == access.image;
            let writes = access.access.intersects(all_write_access());
            let reads = access.access.intersects(all_read_access());

            if writes {
                self.image_writes.retain(|a| !same_image(&a.write));
                self.image_writes
                    .push(PendingWrite::new(access.clone(), self.epoch));
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
                self.image_reads.push(read);
            }
        }
    }
}

pub(crate) fn all_write_access() -> vk::AccessFlags2 {
    vk::AccessFlags2::SHADER_WRITE
        | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
        | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
        | vk::AccessFlags2::TRANSFER_WRITE
        | vk::AccessFlags2::HOST_WRITE
        | vk::AccessFlags2::MEMORY_WRITE
        | vk::AccessFlags2::SHADER_STORAGE_WRITE
        | vk::AccessFlags2::TRANSFORM_FEEDBACK_WRITE_EXT
        | vk::AccessFlags2::TRANSFORM_FEEDBACK_COUNTER_WRITE_EXT
        | vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR
        | vk::AccessFlags2::COMMAND_PREPROCESS_WRITE_NV
        | vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR
        | vk::AccessFlags2::VIDEO_ENCODE_WRITE_KHR
        | vk::AccessFlags2::MICROMAP_WRITE_EXT
        | vk::AccessFlags2::OPTICAL_FLOW_WRITE_NV
}

pub(crate) fn all_read_access() -> vk::AccessFlags2 {
    vk::AccessFlags2::INDIRECT_COMMAND_READ
        | vk::AccessFlags2::INDEX_READ
        | vk::AccessFlags2::VERTEX_ATTRIBUTE_READ
        | vk::AccessFlags2::UNIFORM_READ
        | vk::AccessFlags2::INPUT_ATTACHMENT_READ
        | vk::AccessFlags2::SHADER_READ
        | vk::AccessFlags2::COLOR_ATTACHMENT_READ
        | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
        | vk::AccessFlags2::TRANSFER_READ
        | vk::AccessFlags2::HOST_READ
        | vk::AccessFlags2::MEMORY_READ
        | vk::AccessFlags2::SHADER_SAMPLED_READ
        | vk::AccessFlags2::SHADER_STORAGE_READ
        | vk::AccessFlags2::SHADER_BINDING_TABLE_READ_KHR
        | vk::AccessFlags2::TRANSFORM_FEEDBACK_COUNTER_READ_EXT
        | vk::AccessFlags2::CONDITIONAL_RENDERING_READ_EXT
        | vk::AccessFlags2::COLOR_ATTACHMENT_READ_NONCOHERENT_EXT
        | vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR
        | vk::AccessFlags2::FRAGMENT_DENSITY_MAP_READ_EXT
        | vk::AccessFlags2::FRAGMENT_SHADING_RATE_ATTACHMENT_READ_KHR
        | vk::AccessFlags2::COMMAND_PREPROCESS_READ_NV
        | vk::AccessFlags2::DESCRIPTOR_BUFFER_READ_EXT
        | vk::AccessFlags2::INVOCATION_MASK_READ_HUAWEI
        | vk::AccessFlags2::VIDEO_DECODE_READ_KHR
        | vk::AccessFlags2::VIDEO_ENCODE_READ_KHR
        | vk::AccessFlags2::MICROMAP_READ_EXT
        | vk::AccessFlags2::OPTICAL_FLOW_READ_NV
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command_buffer::tests::{buffer, image, image_read, image_write, pending, read, write};
    use crate::vk::Handle;
    use std::iter;
    use vk::{AccessFlags2 as A, ImageLayout as L, PipelineStageFlags2 as S};

    #[test]
    fn accesses_are_recorded_by_kind() {
        let read_write = buffer(
            S::COMPUTE_SHADER,
            A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE,
            0..10,
        );
        let pending = pending(&[read(20..30), write(40..50), read_write], &[]);
        assert_eq!(pending.buffer_reads.len(), 2);
        assert_eq!(pending.buffer_writes.len(), 2);
    }

    #[test]
    fn a_covering_access_replaces_the_older_ones() {
        let pending = pending(&[write(10..20), write(30..40), write(0..100)], &[]);
        assert_eq!(pending.buffer_writes.len(), 1);
        assert_eq!(pending.buffer_writes[0].write.range, (0u64..100).into());

        // Same range again: still a single entry, lists don't grow frame over frame.
        let mut pending = pending;
        for _ in 0..10 {
            pending.record(iter::empty(), [write(0..100), read(0..100)].into_iter());
        }
        assert_eq!(pending.buffer_writes.len(), 1);
        assert_eq!(pending.buffer_reads.len(), 1);
    }

    #[test]
    fn forgetting_drops_only_the_accesses_of_the_destroyed_object() {
        let mut p = pending(
            &[read(0..10), write(10..20), read(100..110), write(120..130)],
            &[image_read(1), image_write(2), image_read(3), image_write(3)],
        );
        p.forget_buffer(100..200);
        p.forget_image(vk::Image::from_raw(3));
        assert_eq!(p.buffer_reads.len(), 1);
        assert_eq!(p.buffer_reads[0].range, (0u64..10).into());
        assert_eq!(p.buffer_writes.len(), 1);
        assert_eq!(p.buffer_writes[0].write.range, (10u64..20).into());
        assert_eq!(p.image_reads.len(), 1);
        assert_eq!(p.image_reads[0].image, vk::Image::from_raw(1));
        assert_eq!(p.image_writes.len(), 1);
        assert_eq!(p.image_writes[0].write.image, vk::Image::from_raw(2));
    }

    #[test]
    fn image_accesses_replace_older_ones_of_the_same_image_only() {
        let other_stage = image(1, S::TRANSFER, A::TRANSFER_WRITE, L::GENERAL, L::GENERAL);
        let pending = pending(&[], &[image_write(1), image_write(2), other_stage]);
        assert_eq!(pending.image_writes.len(), 2);
        let first = pending
            .image_writes
            .iter()
            .find(|a| a.write.image == vk::Image::from_raw(1))
            .unwrap();
        assert_eq!(first.write.stage, S::TRANSFER);
    }

    #[test]
    fn a_write_forgets_the_reads_it_waited_for() {
        let p = pending(&[], &[image_read(1), image_write(1)]);
        assert!(p.image_reads.is_empty());

        let p = pending(&[read(0..10), read(20..30), write(0..15)], &[]);
        assert_eq!(p.buffer_reads.len(), 1);
        assert_eq!(p.buffer_reads[0].range, (20u64..30).into());
    }

    #[test]
    fn visible_scopes_merge_only_where_that_is_exact() {
        let mut write = PendingWrite::new((), 0);
        write.make_visible(S::FRAGMENT_SHADER, A::SHADER_SAMPLED_READ);
        write.make_visible(S::COMPUTE_SHADER, A::SHADER_SAMPLED_READ);
        assert_eq!(write.visible.len(), 1, "same accesses: one scope");
        write.make_visible(S::COMPUTE_SHADER, A::SHADER_STORAGE_READ);
        assert_eq!(write.visible.len(), 2);
        assert!(!write.visible_to(S::FRAGMENT_SHADER, A::SHADER_STORAGE_READ));
        // A scope containing another replaces it.
        let wide_stages = S::VERTEX_SHADER | S::FRAGMENT_SHADER | S::COMPUTE_SHADER;
        write.make_visible(wide_stages, A::SHADER_SAMPLED_READ | A::SHADER_STORAGE_READ);
        assert_eq!(write.visible.len(), 1);
        assert!(write.visible_to(S::VERTEX_SHADER, A::SHADER_STORAGE_READ));
    }
}

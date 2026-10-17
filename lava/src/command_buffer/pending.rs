//! The accesses later commands may have to wait for: what a timeline's submissions read and wrote, each write with what barriers already made it visible to, how recording a command adds to them, and how they shrink once the GPU finished them or a resource is destroyed

use crate::vk;
use smallvec::SmallVec;

use crate::command_buffer::{
    Barriers, BufferAccess, EVERY_ACCELERATION_STRUCTURE, EVERY_BUFFER, ImageAccess,
};

pub(crate) const MAX_PENDING_BUFFER_WRITES: usize = 64;

#[derive(Debug, Default, Clone)]
pub(crate) struct PendingAccesses {
    pub(crate) buffer_reads: SmallVec<[PendingRead; 8]>,
    pub(crate) image_reads: SmallVec<[ImageAccess; 8]>,
    pub(crate) buffer_writes: SmallVec<[PendingWrite<BufferAccess>; 8]>,
    pub(crate) image_writes: SmallVec<[PendingWrite<ImageAccess>; 8]>,
    /// The timeline value of the submission being recorded.
    pub(crate) value: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct PendingRead {
    pub(crate) read: BufferAccess,
    pub(crate) value: u64,
}

impl std::ops::Deref for PendingRead {
    type Target = BufferAccess;
    fn deref(&self) -> &BufferAccess {
        &self.read
    }
}

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
        (self.visible.iter()).any(|&(stages, accesses)| {
            stages.contains(stage) && accesses.contains(access)
        })
    }

    /// A scope merges only into one of the same stages or accesses, or replaces one it contains:
    /// only then is the merged scope exactly the pairs of both.
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
    /// A memory barrier covers every write in its source scope, also those it wasn't recorded for.
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

    /// Drops the buffer reads up to `completed`, and once there are too many buffer writes,
    /// folds the completed ones into one write of `EVERY_BUFFER`, which still needs visibility.
    pub(crate) fn retire(&mut self, completed: u64) {
        self.buffer_reads.retain(|read| read.value > completed);
        if self.buffer_writes.len() <= MAX_PENDING_BUFFER_WRITES {
            return;
        }
        let (mut stage, mut access) = (vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE);
        self.buffer_writes.retain(|w| {
            let fold = w.value <= completed && w.write.range != EVERY_BUFFER;
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

    pub(crate) fn forget_image(&mut self, image: vk::Image) {
        self.image_reads.retain(|read| read.image != image);
        self.image_writes.retain(|write| write.write.image != image);
    }

    /// An older buffer access is only forgotten once a newer write covers its whole range; a
    /// newer read doesn't wait for older reads, so it takes over those it covers.
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
                    .push(PendingWrite::new(access.clone(), self.value));
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
                    value: self.value,
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
                    .push(PendingWrite::new(access.clone(), self.value));
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

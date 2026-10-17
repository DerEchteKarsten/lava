//! The device's queues: per family lava uses, at most `MAX_TIMELINES` queues, each locked while it is submitted to and counting what is pinned to it
use std::sync::{Mutex, MutexGuard, atomic::AtomicU32};

use crate::vk;
use smallvec::SmallVec;

use crate::{
    queue::{Kind, MAX_TIMELINES, lock},
    state::Ctx,
    command_buffer::pending::PendingAccesses,
    vkobjects::physical_device::QueueFamily,
};

/// The queues of every family lava created queues of, by [`Kind`].
#[derive(Debug)]
pub(crate) struct DeviceQueues {
    families: [Option<FamilyQueues>; 3],
}

/// The queues lava created of one family.
#[derive(Debug)]
pub(crate) struct FamilyQueues {
    pub(crate) index: u32,
    pub(crate) flags: vk::QueueFlags,
    /// The valid bits of its timestamps (0: it has none).
    pub(crate) timestamp_valid_bits: u32,
    pub(crate) queues: Box<[QueueHandle]>,
}

/// One queue: Vulkan wants submissions to it externally synchronised.
#[derive(Debug)]
pub(crate) struct QueueHandle {
    pub(crate) queue: Mutex<vk::Queue>,
    /// What is pinned to it, so the next one takes the least shared queue.
    pub(crate) timelines: AtomicU32,
    /// The accesses of what was submitted to it that later submissions may have to wait for.
    pub(crate) pending: Mutex<PendingAccesses>,
}

/// How many queues lava creates of `family`.
pub(crate) fn queue_count(family: &QueueFamily) -> u32 {
    family.num_queues.min(MAX_TIMELINES as u32)
}

impl FamilyQueues {
    fn new(device: &ash::Device, family: &QueueFamily) -> Self {
        Self {
            index: family.index,
            flags: family.handel.queue_flags,
            timestamp_valid_bits: family.handel.timestamp_valid_bits,
            queues: (0..queue_count(family))
                .map(|i| QueueHandle {
                    queue: Mutex::new(unsafe { device.get_device_queue(family.index, i) }),
                    timelines: AtomicU32::new(0),
                    pending: Mutex::default(),
                })
                .collect(),
        }
    }
}

impl DeviceQueues {
    /// The queues `create_device` created of each family, by kind.
    pub(crate) fn new(device: &ash::Device, families: [Option<&QueueFamily>; 3]) -> Self {
        Self {
            families: families.map(|family| family.map(|family| FamilyQueues::new(device, family))),
        }
    }

    /// The family of `kind`, if lava created queues of it.
    pub(crate) fn family(&self, kind: Kind) -> Option<&FamilyQueues> {
        self.families[kind.slot()].as_ref()
    }

    pub(crate) fn graphics(&self) -> &FamilyQueues {
        self.family(Kind::Graphics)
            .expect("every device has a graphics family")
    }

    /// Every family lava created queues of, graphics first.
    pub(crate) fn iter(&self) -> impl Iterator<Item = &FamilyQueues> {
        self.families.iter().flatten()
    }

    /// The indices of those families, which buffers are shared between.
    pub(crate) fn family_indices(&self) -> SmallVec<[u32; 3]> {
        self.iter().map(|family| family.index).collect()
    }
}

/// Every queue, locked in order: nothing can be submitted or presented while they are held.
pub(crate) fn lock_all_queues() -> Vec<MutexGuard<'static, vk::Queue>> {
    let queues = Ctx::get().queues.iter().flat_map(|family| &family.queues);
    queues.map(|queue| lock(&queue.queue)).collect()
}

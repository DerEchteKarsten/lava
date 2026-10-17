//! The device's queues: per family lava uses, at most `MAX_TIMELINES` queues, each locked while it is submitted to and counting the timelines pinned to it
use std::sync::{
    Mutex, MutexGuard,
    atomic::{AtomicU32, Ordering},
};

use crate::vk;
use smallvec::SmallVec;

use crate::{
    queue::{Kind, MAX_TIMELINES, lock},
    state::Ctx,
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
    pub(crate) kind: Kind,
    pub(crate) index: u32,
    /// The valid bits of its timestamps (0: it has none).
    pub(crate) timestamp_valid_bits: u32,
    /// The granularity of its image copies (1×1×1 on graphics and compute families).
    pub(crate) image_granularity: vk::Extent3D,
    pub(crate) queues: Box<[QueueHandle]>,
}

/// One queue: Vulkan wants submissions to it externally synchronised.
#[derive(Debug)]
pub(crate) struct QueueHandle {
    pub(crate) queue: Mutex<vk::Queue>,
    /// The timelines pinned to it, so the next one takes the least shared queue.
    timelines: AtomicU32,
}

/// How many queues lava creates of `family`.
pub(crate) fn queue_count(family: &QueueFamily) -> u32 {
    family.num_queues.min(MAX_TIMELINES as u32)
}

/// Held while timelines are pinned, so two never count the same queue as the least shared.
static PINNING: Mutex<()> = Mutex::new(());

impl FamilyQueues {
    fn new(device: &ash::Device, kind: Kind, family: &QueueFamily) -> Self {
        Self {
            kind,
            index: family.index,
            timestamp_valid_bits: family.handel.timestamp_valid_bits,
            image_granularity: family.handel.min_image_transfer_granularity,
            queues: (0..queue_count(family))
                .map(|i| QueueHandle {
                    queue: Mutex::new(unsafe { device.get_device_queue(family.index, i) }),
                    timelines: AtomicU32::new(0),
                })
                .collect(),
        }
    }

    /// Pins a timeline to the queue the fewest timelines share, which it returns.
    pub(crate) fn pin(&self) -> usize {
        let _pinning = lock(&PINNING);
        let count = |i: &usize| self.queues[*i].timelines.load(Ordering::Relaxed);
        let queue = (0..self.queues.len())
            .min_by_key(count)
            .expect("a queue family has at least one queue");
        self.queues[queue].timelines.fetch_add(1, Ordering::Relaxed);
        queue
    }

    /// A timeline pinned to `queue` is gone.
    pub(crate) fn unpin(&self, queue: usize) {
        let _pinning = lock(&PINNING);
        self.queues[queue].timelines.fetch_sub(1, Ordering::Relaxed);
    }
}

impl DeviceQueues {
    /// The queues `create_device` created of each family, by kind.
    pub(crate) fn new(device: &ash::Device, families: [Option<&QueueFamily>; 3]) -> Self {
        let kinds = [Kind::Graphics, Kind::Compute, Kind::Transfer];
        let mut families = families.into_iter().zip(kinds);
        Self {
            families: std::array::from_fn(|_| {
                let (family, kind) = families.next().expect("one per kind");
                family.map(|family| FamilyQueues::new(device, kind, family))
            }),
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

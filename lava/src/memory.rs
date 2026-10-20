//! GPU memory: what lava's allocator holds (its live bytes counted at each allocation and free, its blocks' capacity), each heap's size, and with `memory-budget` its usage and budget
use std::sync::atomic::{AtomicU64, Ordering};

use crate::vk;

#[cfg(feature = "memory-budget")]
use crate::features::{Gated, On, gate};
#[cfg(feature = "memory-budget")]
use crate::state::Functions;
use crate::state::Ctx;

/// Bytes of live allocations, counted by `Ctx::allocator()`'s `allocate` and `free`.
pub(crate) static ALLOCATED: AtomicU64 = AtomicU64::new(0);

/// GPU memory: what lava's allocator holds, and what each heap has in use.
#[derive(Clone, Debug, Default)]
pub struct MemoryReport {
    /// Bytes of live allocations.
    pub allocated: u64,
    /// Bytes of the memory blocks they are sub-allocated from.
    pub reserved: u64,
    pub heaps: Vec<HeapBudget>,
}

#[derive(Clone, Debug)]
pub struct HeapBudget {
    pub size: u64,
    pub device_local: bool,
    /// Bytes in use by this process (`VK_EXT_memory_budget`).
    #[cfg(feature = "memory-budget")]
    pub usage: Gated<gate::MemoryBudget, u64>,
    /// Bytes the process can use without overcommitting (`VK_EXT_memory_budget`).
    #[cfg(feature = "memory-budget")]
    pub budget: Gated<gate::MemoryBudget, u64>,
}

/// Cheap enough for every frame: two totals of the allocator, the heaps lava read at init, and
/// with `memory-budget` one query of the budget.
pub fn memory_report() -> MemoryReport {
    let _call = crate::state::call();
    let reserved = Ctx::allocator().capacity();
    let memory = &Ctx::physical_device().mem_properties;
    #[cfg(feature = "memory-budget")]
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    #[cfg(feature = "memory-budget")]
    if gate::MemoryBudget::on() {
        let mut properties = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        unsafe {
            Functions::instance()
                .get_physical_device_memory_properties2(Ctx::physical_device().handel, &mut properties)
        };
    }
    let heaps = (0..memory.memory_heap_count as usize)
        .map(|i| HeapBudget {
            size: memory.memory_heaps[i].size,
            device_local: memory.memory_heaps[i]
                .flags
                .contains(vk::MemoryHeapFlags::DEVICE_LOCAL),
            #[cfg(feature = "memory-budget")]
            usage: crate::features::gate::<gate::MemoryBudget, _>(|| budget.heap_usage[i]),
            #[cfg(feature = "memory-budget")]
            budget: crate::features::gate::<gate::MemoryBudget, _>(|| budget.heap_budget[i]),
        })
        .collect();
    MemoryReport {
        allocated: ALLOCATED.load(Ordering::Relaxed),
        reserved,
        heaps,
    }
}

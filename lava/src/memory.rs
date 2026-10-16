//! GPU memory: what lava's allocator holds, and each heap's size, and with `memory-budget` its usage and budget
use crate::vk;

#[cfg(feature = "memory-budget")]
use crate::features::{Gated, On, gate};
use crate::state::{Ctx, Functions};

/// GPU memory: what lava's allocator holds, and what each heap has in use.
#[derive(Clone, Debug, Default)]
pub struct MemoryReport {
    /// Bytes of live allocations.
    pub allocated: u64,
    /// Bytes of the memory blocks they are sub-allocated from.
    pub reserved: u64,
    pub blocks: usize,
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

pub fn memory_report() -> MemoryReport {
    let report = Ctx::allocator().generate_report();
    #[cfg(feature = "memory-budget")]
    let has_budget = gate::MemoryBudget::on();
    #[cfg(not(feature = "memory-budget"))]
    let has_budget = false;
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let mut properties = vk::PhysicalDeviceMemoryProperties2::default();
    if has_budget {
        properties = properties.push_next(&mut budget);
    }
    unsafe {
        Functions::instance()
            .get_physical_device_memory_properties2(Ctx::physical_device().handel, &mut properties)
    };
    let memory = properties.memory_properties;
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
        allocated: report.total_allocated_bytes,
        reserved: report.total_capacity_bytes,
        blocks: report.blocks.len(),
        heaps,
    }
}

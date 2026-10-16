//! GPU memory: what lava's allocator holds, and each heap's usage and budget
use crate::vk;

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
    /// Bytes in use by this process; `None` without `VK_EXT_memory_budget`.
    pub usage: Option<u64>,
    /// Bytes the process can use without overcommitting; the heap size without the extension.
    pub budget: u64,
}

pub fn memory_report() -> MemoryReport {
    let report = Ctx::allocator().generate_report();
    let has_budget = Ctx::features().memory_budget;
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
            usage: has_budget.then(|| budget.heap_usage[i]),
            budget: if has_budget {
                budget.heap_budget[i]
            } else {
                memory.memory_heaps[i].size
            },
        })
        .collect();
    MemoryReport {
        allocated: report.total_allocated_bytes,
        reserved: report.total_capacity_bytes,
        blocks: report.blocks.len(),
        heaps,
    }
}

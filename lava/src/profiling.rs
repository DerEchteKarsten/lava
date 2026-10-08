//! GPU timestamp and pipeline-statistics queries per frame slot
use std::sync::atomic::{AtomicBool, Ordering};

use ash::vk;

use crate::{
    error::Result,
    state::{Ctx, Functions},
};

const TIMESTAMPS: u32 = 256;
const STATISTICS: u32 = 128;

/// Invocation counts of one pass. `task` and `mesh` stay 0 without `mesh_queries`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PipelineStats {
    pub task: u64,
    pub mesh: u64,
    pub fragment: u64,
    pub compute: u64,
}

/// A timed region of one submission, in ns since its root scope ("frame") started.
#[derive(Clone, Debug)]
pub struct GpuScope {
    pub name: &'static str,
    pub depth: u16,
    pub start_ns: u64,
    pub end_ns: u64,
    /// Only for passes, and only when the device supports pipeline statistics queries.
    pub stats: Option<PipelineStats>,
}

struct Scope {
    name: &'static str,
    depth: u16,
    /// Its start timestamp; the end is the next one.
    timestamp: u32,
    statistic: Option<u32>,
}

/// The query pools of one frame slot and the scopes recorded into them. A scope takes two
/// timestamps; a scope that doesn't fit is skipped.
pub struct FrameQueries {
    timestamps: vk::QueryPool,
    statistics: Option<vk::QueryPool>,
    valid_bits: u64,
    scopes: Vec<Scope>,
    /// Indices into `scopes` of the open scopes, `None` for a skipped one.
    open: Vec<Option<usize>>,
    next_timestamp: u32,
    next_statistic: u32,
    results: Vec<GpuScope>,
}

static OVERFLOW_WARNED: AtomicBool = AtomicBool::new(false);

impl FrameQueries {
    pub(crate) fn new(queue_family: u32) -> Result<Self> {
        let device = Ctx::device();
        let features = Ctx::features();
        let timestamps = unsafe {
            device.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(TIMESTAMPS),
                None,
            )?
        };
        // Clipping counts aren't allowed in a query that is active while mesh shaders draw.
        let mut flags = vk::QueryPipelineStatisticFlags::FRAGMENT_SHADER_INVOCATIONS
            | vk::QueryPipelineStatisticFlags::COMPUTE_SHADER_INVOCATIONS;
        if features.mesh_queries {
            flags |= vk::QueryPipelineStatisticFlags::TASK_SHADER_INVOCATIONS_EXT
                | vk::QueryPipelineStatisticFlags::MESH_SHADER_INVOCATIONS_EXT;
        }
        let statistics = if features.pipeline_statistics {
            Some(unsafe {
                device.create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::PIPELINE_STATISTICS)
                        .pipeline_statistics(flags)
                        .query_count(STATISTICS),
                    None,
                )?
            })
        } else {
            None
        };
        let bits = Ctx::physical_device().queue_families[queue_family as usize]
            .handel
            .timestamp_valid_bits;
        Ok(Self {
            timestamps,
            statistics,
            valid_bits: if bits >= 64 { !0 } else { (1 << bits) - 1 },
            scopes: Vec::new(),
            open: Vec::new(),
            next_timestamp: 0,
            next_statistic: 0,
            results: Vec::new(),
        })
    }

    /// Resets the pools at the start of a command buffer.
    pub(crate) fn reset(&mut self, cmd: vk::CommandBuffer) {
        let device = Ctx::device();
        unsafe {
            device.cmd_reset_query_pool(cmd, self.timestamps, 0, TIMESTAMPS);
            if let Some(statistics) = self.statistics {
                device.cmd_reset_query_pool(cmd, statistics, 0, STATISTICS);
            }
        }
        self.scopes.clear();
        self.open.clear();
        self.next_timestamp = 0;
        self.next_statistic = 0;
    }

    /// Opens a scope; `statistics` also counts its invocations, so it must not contain
    /// another scope with statistics.
    pub(crate) fn begin(&mut self, cmd: vk::CommandBuffer, name: &'static str, statistics: bool) {
        if self.next_timestamp + 2 > TIMESTAMPS {
            if !OVERFLOW_WARNED.swap(true, Ordering::Relaxed) {
                tracing::warn!("more than {} GPU scopes in a frame", TIMESTAMPS / 2);
            }
            self.open.push(None);
            return;
        }
        let timestamp = self.next_timestamp;
        self.next_timestamp += 2;
        let statistic = self
            .statistics
            .filter(|_| statistics && self.next_statistic < STATISTICS)
            .map(|_| {
                self.next_statistic += 1;
                self.next_statistic - 1
            });
        let device = Ctx::device();
        unsafe {
            device.cmd_write_timestamp2(
                cmd,
                vk::PipelineStageFlags2::NONE,
                self.timestamps,
                timestamp,
            );
            if let (Some(pool), Some(query)) = (self.statistics, statistic) {
                device.cmd_begin_query(cmd, pool, query, vk::QueryControlFlags::empty());
            }
        }
        self.open.push(Some(self.scopes.len()));
        self.scopes.push(Scope {
            name,
            depth: self.open.len() as u16 - 1,
            timestamp,
            statistic,
        });
    }

    pub(crate) fn end(&mut self, cmd: vk::CommandBuffer) {
        let Some(Some(index)) = self.open.pop() else {
            return;
        };
        let scope = &self.scopes[index];
        let device = Ctx::device();
        unsafe {
            if let (Some(pool), Some(query)) = (self.statistics, scope.statistic) {
                device.cmd_end_query(cmd, pool, query);
            }
            device.cmd_write_timestamp2(
                cmd,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                self.timestamps,
                scope.timestamp + 1,
            );
        }
    }

    /// Reads the results of the last submission, which must have finished.
    pub(crate) fn read(&mut self) -> Result<()> {
        self.results.clear();
        if self.scopes.is_empty() {
            return Ok(());
        }
        let device = Ctx::device();
        let mut ticks = vec![0u64; self.next_timestamp as usize];
        unsafe {
            device.get_query_pool_results(
                self.timestamps,
                0,
                &mut ticks,
                vk::QueryResultFlags::TYPE_64,
            )?
        };
        // One element per query, wide enough for every flag.
        let mut values = vec![[0u64; 4]; self.next_statistic as usize];
        if let Some(pool) = self.statistics
            && self.next_statistic > 0
        {
            unsafe {
                device.get_query_pool_results(
                    pool,
                    0,
                    &mut values,
                    vk::QueryResultFlags::TYPE_64,
                )?
            };
        }
        let period = Ctx::physical_device().limits.timestamp_period as f64;
        let base = ticks[self.scopes[0].timestamp as usize] & self.valid_bits;
        let ns = |tick: u64| ((tick & self.valid_bits).saturating_sub(base) as f64 * period) as u64;
        self.results
            .extend(self.scopes.iter().map(|scope| GpuScope {
                name: scope.name,
                depth: scope.depth,
                start_ns: ns(ticks[scope.timestamp as usize]),
                end_ns: ns(ticks[scope.timestamp as usize + 1]),
                stats: scope.statistic.map(|query| {
                    // In flag bit order: fragment, compute, then task and mesh.
                    let v = values[query as usize];
                    PipelineStats {
                        fragment: v[0],
                        compute: v[1],
                        task: v[2],
                        mesh: v[3],
                    }
                }),
            }));
        Ok(())
    }

    pub(crate) fn results(&self) -> Option<&[GpuScope]> {
        (!self.results.is_empty()).then_some(&self.results)
    }
}

impl Drop for FrameQueries {
    fn drop(&mut self) {
        let device = Ctx::device();
        unsafe {
            device.destroy_query_pool(self.timestamps, None);
            if let Some(statistics) = self.statistics {
                device.destroy_query_pool(statistics, None);
            }
        }
    }
}

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
        reserved: report.total_reserved_bytes,
        blocks: report.blocks.len(),
        heaps,
    }
}

//! GPU profiling per frame slot: timestamps, pipeline statistics and hardware counters per pass, shader clock regions, and GPU memory
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use ash::vk;

use crate::{
    error::Result,
    state::{Ctx, Functions},
};

pub mod clock;
pub mod counters;

pub use clock::RegionTime;
pub use counters::{Counter, CounterUnit, MAX_COUNTERS};

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
    /// Only for passes and renderings: the values of `FrameTimings::counters`, in order.
    pub counters: Option<[f64; MAX_COUNTERS]>,
}

/// What a profiled frame slot read back about the frame it submitted before.
#[derive(Clone, Debug, Default)]
pub struct FrameTimings {
    /// The root scope "frame" first, then passes and `begin_scope` groups in recording order.
    pub scopes: Vec<GpuScope>,
    /// The counters the scopes' values belong to (`counters::selected_counters` when the
    /// frame slot was created).
    pub counters: Arc<[&'static Counter]>,
    pub regions: Vec<RegionTime>,
}

pub(crate) fn init() -> Result<()> {
    clock::init()?;
    counters::init()
}

struct Scope {
    name: &'static str,
    depth: u16,
    /// Its start timestamp; the end is the next one.
    timestamp: u32,
    statistic: Option<u32>,
    performance: Option<u32>,
}

/// The counter query pool of a frame slot. It holds the profiling lock while it exists.
struct Performance {
    pool: vk::QueryPool,
    counters: Arc<[&'static Counter]>,
    /// Scratch for the results, one per counter and query.
    results: Vec<vk::PerformanceCounterResultKHR>,
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
    performance: Option<Performance>,
    next_performance: u32,
    /// `counters::generation()` when this was created; a newer selection needs new pools.
    pub(crate) generation: u64,
    results: FrameTimings,
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
        let generation = counters::generation();
        let selected = counters::selected_counters();
        // Counters were enumerated for the graphics family; the profiling lock must be held
        // from before the command buffer is begun until its queries are done.
        let performance = if !selected.is_empty()
            && queue_family == Ctx::gfx_queue_index()
            && counters::acquire_lock()
        {
            let indices: Vec<u32> = selected.iter().map(|counter| counter.index).collect();
            let mut info = vk::QueryPoolPerformanceCreateInfoKHR::default()
                .queue_family_index(queue_family)
                .counter_indices(&indices);
            let pool = unsafe {
                device.create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::PERFORMANCE_QUERY_KHR)
                        .query_count(STATISTICS)
                        .push_next(&mut info),
                    None,
                )
            };
            match pool {
                Ok(pool) => Some(Performance {
                    pool,
                    results: Vec::with_capacity(STATISTICS as usize * selected.len()),
                    counters: selected,
                }),
                Err(err) => {
                    counters::release_lock();
                    return Err(err.into());
                }
            }
        } else {
            None
        };
        clock::add_profiled_slots(1);
        Ok(Self {
            timestamps,
            statistics,
            valid_bits: if bits >= 64 { !0 } else { (1 << bits) - 1 },
            scopes: Vec::new(),
            open: Vec::new(),
            next_timestamp: 0,
            next_statistic: 0,
            results: FrameTimings {
                counters: performance
                    .as_ref()
                    .map_or_else(|| Arc::new([]) as Arc<[_]>, |p| p.counters.clone()),
                ..Default::default()
            },
            performance,
            next_performance: 0,
            generation,
        })
    }

    /// Resets the pools at the start of a command buffer. The counter pool is reset on the
    /// host: a command buffer that begins its queries may not reset them.
    pub(crate) fn reset(&mut self, cmd: vk::CommandBuffer) {
        let device = Ctx::device();
        unsafe {
            device.cmd_reset_query_pool(cmd, self.timestamps, 0, TIMESTAMPS);
            if let Some(statistics) = self.statistics {
                device.cmd_reset_query_pool(cmd, statistics, 0, STATISTICS);
            }
            if let Some(performance) = &self.performance {
                device.reset_query_pool(performance.pool, 0, STATISTICS);
            }
        }
        self.scopes.clear();
        self.open.clear();
        self.next_timestamp = 0;
        self.next_statistic = 0;
        self.next_performance = 0;
    }

    /// Opens a scope; `statistics` also counts its invocations and reads the counters, so it
    /// must not contain another scope with statistics.
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
        let performance = self
            .performance
            .as_ref()
            .filter(|_| statistics && self.next_performance < STATISTICS)
            .map(|_| {
                self.next_performance += 1;
                self.next_performance - 1
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
            if let (Some(performance), Some(query)) = (&self.performance, performance) {
                device.cmd_begin_query(
                    cmd,
                    performance.pool,
                    query,
                    vk::QueryControlFlags::empty(),
                );
            }
        }
        self.open.push(Some(self.scopes.len()));
        self.scopes.push(Scope {
            name,
            depth: self.open.len() as u16 - 1,
            timestamp,
            statistic,
            performance,
        });
    }

    pub(crate) fn end(&mut self, cmd: vk::CommandBuffer) {
        let Some(Some(index)) = self.open.pop() else {
            return;
        };
        let scope = &self.scopes[index];
        let device = Ctx::device();
        unsafe {
            if let (Some(performance), Some(query)) = (&self.performance, scope.performance) {
                device.cmd_end_query(cmd, performance.pool, query);
            }
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
    /// Makes the shader clock counters readable by the host once the submission finished.
    pub(crate) fn finish(&self, cmd: vk::CommandBuffer) {
        let barrier = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ)];
        let info = vk::DependencyInfo::default().memory_barriers(&barrier);
        unsafe { Ctx::device().cmd_pipeline_barrier2(cmd, &info) };
    }

    pub(crate) fn read(&mut self) -> Result<()> {
        self.results.scopes.clear();
        self.results.regions = clock::read();
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
        let count = self.performance.as_ref().map_or(0, |p| p.counters.len());
        if let Some(performance) = &mut self.performance
            && self.next_performance > 0
        {
            // One result per counter and query; no flags: 64-bit, availability and partial
            // results are not allowed for counters.
            let results = &mut performance.results;
            results.clear();
            results.resize(
                self.next_performance as usize * count,
                vk::PerformanceCounterResultKHR::default(),
            );
            let stride = size_of::<vk::PerformanceCounterResultKHR>() * count;
            unsafe {
                (device.fp_v1_0().get_query_pool_results)(
                    device.handle(),
                    performance.pool,
                    0,
                    self.next_performance,
                    results.len() * size_of::<vk::PerformanceCounterResultKHR>(),
                    results.as_mut_ptr().cast(),
                    stride as u64,
                    vk::QueryResultFlags::empty(),
                )
                .result()?
            };
        }
        let performance = &self.performance;
        let counter_values = |query: u32| {
            let performance = performance.as_ref()?;
            let results = &performance.results[query as usize * count..][..count];
            let mut values = [0.0; MAX_COUNTERS];
            for (value, (counter, result)) in
                values.iter_mut().zip(performance.counters.iter().zip(results))
            {
                *value = counter.value(result);
            }
            Some(values)
        };
        let period = Ctx::physical_device().limits.timestamp_period as f64;
        let base = ticks[self.scopes[0].timestamp as usize] & self.valid_bits;
        let ns = |tick: u64| ((tick & self.valid_bits).saturating_sub(base) as f64 * period) as u64;
        self.results
            .scopes
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
                counters: scope.performance.and_then(counter_values),
            }));
        Ok(())
    }

    pub(crate) fn results(&self) -> Option<&FrameTimings> {
        (!self.results.scopes.is_empty()).then_some(&self.results)
    }
}

impl Drop for FrameQueries {
    fn drop(&mut self) {
        clock::add_profiled_slots(-1);
        let device = Ctx::device();
        unsafe {
            device.destroy_query_pool(self.timestamps, None);
            if let Some(statistics) = self.statistics {
                device.destroy_query_pool(statistics, None);
            }
            if let Some(performance) = &self.performance {
                device.destroy_query_pool(performance.pool, None);
                counters::release_lock();
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

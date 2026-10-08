//! GPU timestamp and pipeline-statistics queries per frame slot, and the shader clock counters that `profile.slang` fills
use std::sync::{
    Mutex, OnceLock,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

use ash::vk;

use crate::{
    bindings::PASS_MAP,
    bindless::Bindless,
    buffer::Buffer,
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

/// Subgroup clock ticks the shaders of one pass and stage spent since the frame before, as
/// counted by `PROFILE` in the shader. Ticks are the device's own unit; only ratios between
/// passes and stages are meaningful.
#[derive(Clone, Debug)]
pub struct ShaderTime {
    pub pass: &'static str,
    pub stage: &'static str,
    /// Summed over all subgroups, which run in parallel, so it can exceed the frame.
    pub ticks: u64,
    pub subgroups: u64,
}

/// What a profiled frame slot read back about the frame it submitted before.
#[derive(Clone, Debug, Default)]
pub struct FrameTimings {
    /// The root scope "frame" first, then passes and `begin_scope` groups in recording order.
    pub scopes: Vec<GpuScope>,
    pub shaders: Vec<ShaderTime>,
}

/// Mirrors `profile.slang`.
const FLAG_WORDS: usize = 8;
const STAGES: [&str; 4] = ["compute", "task", "mesh", "vertex"];
const STRIPES: usize = 64;
const STRIPE_WORDS: usize = 8;

/// The buffer `profile.slang` adds to, in descriptor set 2. Word 0 counts the profiling
/// `FrameQueries`; the shaders skip their atomics while it is 0.
static COUNTERS: OnceLock<Buffer<u64>> = OnceLock::new();
/// Per pass and stage, the ticks and subgroups at the last read. Shared by all frame slots,
/// since the counters only grow.
static LAST_READ: Mutex<Vec<[u64; 2]>> = Mutex::new(Vec::new());

fn counter(index: usize) -> &'static AtomicU64 {
    let counters = COUNTERS.get().expect("shader clocks are initialised");
    unsafe { &*(counters.range(..).element_ptr(index) as *const AtomicU64) }
}

/// Creates the shader clock counters and binds them, if the device has subgroup clocks.
pub(crate) fn init() -> Result<()> {
    if !Ctx::features().shader_clock {
        return Ok(());
    }
    let len = FLAG_WORDS + PASS_MAP.len() * STAGES.len() * STRIPES * STRIPE_WORDS;
    let counters = Buffer::<u64>::new(len, true)?;
    counters.range(..).copy_from(&vec![0; counters.len()]);
    let info = [vk::DescriptorBufferInfo::default()
        .buffer(counters.handle)
        .range(vk::WHOLE_SIZE)];
    let write = vk::WriteDescriptorSet::default()
        .dst_set(Bindless::profile_set())
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(&info);
    unsafe { Ctx::device().update_descriptor_sets(&[write], &[]) };
    let _ = COUNTERS.set(counters);
    Ok(())
}

/// The ticks every pass and stage spent since the last call, from any frame slot.
fn read_shader_clocks() -> Vec<ShaderTime> {
    if COUNTERS.get().is_none() {
        return Vec::new();
    }
    let mut last = LAST_READ.lock().unwrap();
    last.resize(PASS_MAP.len() * STAGES.len(), [0; 2]);
    let mut times = Vec::new();
    for (slot, last) in last.iter_mut().enumerate() {
        let mut now = [0; 2];
        for stripe in 0..STRIPES {
            let base = FLAG_WORDS + (slot * STRIPES + stripe) * STRIPE_WORDS;
            now[0] += counter(base).load(Ordering::Relaxed);
            now[1] += counter(base + 1).load(Ordering::Relaxed);
        }
        let (ticks, subgroups) = (now[0] - last[0], now[1] - last[1]);
        *last = now;
        if subgroups > 0 {
            times.push(ShaderTime {
                pass: PASS_MAP[slot / STAGES.len()].name,
                stage: STAGES[slot % STAGES.len()],
                ticks,
                subgroups,
            });
        }
    }
    times
}

/// Specialization info telling `profile.slang` which pass a pipeline belongs to.
pub(crate) fn pass_constant(pass_index: &u32) -> vk::SpecializationInfo<'_> {
    const ENTRIES: [vk::SpecializationMapEntry; 1] = [vk::SpecializationMapEntry {
        constant_id: 900,
        offset: 0,
        size: 4,
    }];
    vk::SpecializationInfo::default()
        .map_entries(&ENTRIES)
        .data(bytemuck::bytes_of(pass_index))
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
        if COUNTERS.get().is_some() {
            counter(0).fetch_add(1, Ordering::Relaxed);
        }
        Ok(Self {
            timestamps,
            statistics,
            valid_bits: if bits >= 64 { !0 } else { (1 << bits) - 1 },
            scopes: Vec::new(),
            open: Vec::new(),
            next_timestamp: 0,
            next_statistic: 0,
            results: FrameTimings::default(),
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
        self.results.shaders = read_shader_clocks();
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
            }));
        Ok(())
    }

    pub(crate) fn results(&self) -> Option<&FrameTimings> {
        (!self.results.scopes.is_empty()).then_some(&self.results)
    }
}

impl Drop for FrameQueries {
    fn drop(&mut self) {
        if COUNTERS.get().is_some() {
            counter(0).fetch_sub(1, Ordering::Relaxed);
        }
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

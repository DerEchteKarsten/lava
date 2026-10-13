//! Pass queries: pipeline statistics and hardware performance counters (`VK_KHR_performance_query`) per pass and rendering of a frame slot, the device's counters, and the profiling lock
use ash::vk::TaggedStructure;
use std::sync::{Arc, Mutex, OnceLock};

use ash::vk;

use crate::{
    error::{Error, Result},
    state::{Ctx, Functions},
};

/// The most counters a selection can hold; `PassResult::values` has room for this many.
pub const MAX_COUNTERS: usize = 8;

/// What a counter's value measures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CounterUnit {
    Generic,
    Percentage,
    Nanoseconds,
    Bytes,
    BytesPerSecond,
    Kelvin,
    Watts,
    Volts,
    Amps,
    Hertz,
    Cycles,
}

impl CounterUnit {
    /// Whether values describe a rate or state over the measured time (averaged over it)
    /// rather than an amount (summed).
    pub fn is_rate(self) -> bool {
        matches!(
            self,
            Self::Percentage
                | Self::BytesPerSecond
                | Self::Kelvin
                | Self::Watts
                | Self::Volts
                | Self::Amps
                | Self::Hertz
        )
    }

    fn from_vk(unit: vk::PerformanceCounterUnitKHR) -> Self {
        use vk::PerformanceCounterUnitKHR as U;
        match unit {
            U::PERCENTAGE => Self::Percentage,
            U::NANOSECONDS => Self::Nanoseconds,
            U::BYTES => Self::Bytes,
            U::BYTES_PER_SECOND => Self::BytesPerSecond,
            U::KELVIN => Self::Kelvin,
            U::WATTS => Self::Watts,
            U::VOLTS => Self::Volts,
            U::AMPS => Self::Amps,
            U::HERTZ => Self::Hertz,
            U::CYCLES => Self::Cycles,
            _ => Self::Generic,
        }
    }
}

/// A counter of the graphics queue family.
#[derive(Debug)]
pub struct Counter {
    pub name: String,
    pub category: String,
    pub description: String,
    pub unit: CounterUnit,
    /// Its index in the device's list, which query pools are created with.
    pub(crate) index: u32,
    storage: vk::PerformanceCounterStorageKHR,
}

impl Counter {
    /// The value of one query result.
    pub(crate) fn value(&self, result: &vk::PerformanceCounterResultKHR) -> f64 {
        use vk::PerformanceCounterStorageKHR as S;
        unsafe {
            match self.storage {
                S::INT32 => result.int32 as f64,
                S::INT64 => result.int64 as f64,
                S::UINT32 => result.uint32 as f64,
                S::UINT64 => result.uint64 as f64,
                S::FLOAT32 => result.float32 as f64,
                _ => result.float64,
            }
        }
    }
}

static COUNTERS: OnceLock<Vec<Counter>> = OnceLock::new();
/// How many query pools use the profiling lock; it is held while there is one.
static LOCK_USERS: Mutex<usize> = Mutex::new(0);

fn to_string(chars: &[std::ffi::c_char]) -> String {
    unsafe { std::ffi::CStr::from_ptr(chars.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

/// Lists the counters of the graphics queue family, if the device has performance queries.
pub(crate) fn init() -> Result<()> {
    let Some((instance, _)) = Functions::performance_query() else {
        return Ok(());
    };
    let (device, family) = (Ctx::physical_device().handel, Ctx::gfx_queue_index());
    let len = unsafe {
        instance
            .enumerate_physical_device_queue_family_performance_query_counters_len(device, family)?
    };
    let mut counters = vec![vk::PerformanceCounterKHR::default(); len];
    let mut descriptions = vec![vk::PerformanceCounterDescriptionKHR::default(); len];
    unsafe {
        instance.enumerate_physical_device_queue_family_performance_query_counters(
            device,
            family,
            &mut counters,
            &mut descriptions,
        )?
    };
    let counters = counters
        .iter()
        .zip(&descriptions)
        .enumerate()
        .map(|(index, (counter, description))| Counter {
            name: to_string(&description.name),
            category: to_string(&description.category),
            description: to_string(&description.description),
            unit: CounterUnit::from_vk(counter.unit),
            index: index as u32,
            storage: counter.storage,
        })
        .collect();
    let _ = COUNTERS.set(counters);
    Ok(())
}

/// Every counter the graphics queue family offers; empty before `lava::init` and without
/// `VK_KHR_performance_query`.
pub fn counters() -> &'static [Counter] {
    COUNTERS.get().map_or(&[], Vec::as_slice)
}

/// How many submissions of a frame it takes to read `selection`. Only selections of one pass
/// can be recorded.
pub fn passes(selection: &[&Counter]) -> u32 {
    let Some((instance, _)) = Functions::performance_query() else {
        return 0;
    };
    let indices: Vec<u32> = selection.iter().map(|counter| counter.index).collect();
    let info = vk::QueryPoolPerformanceCreateInfoKHR::default()
        .queue_family_index(Ctx::gfx_queue_index())
        .counter_indices(&indices);
    unsafe {
        instance.get_physical_device_queue_family_performance_query_passes(
            Ctx::physical_device().handel,
            &info,
        )
    }
}

/// Takes the profiling lock for one more query pool; false if another process holds it.
fn acquire_lock() -> bool {
    let mut users = LOCK_USERS.lock().unwrap();
    if *users == 0 {
        let (_, device) = Functions::performance_query().expect("performance queries are enabled");
        let info = vk::AcquireProfilingLockInfoKHR::default().timeout(0);
        if let Err(err) = unsafe { device.acquire_profiling_lock(&info) } {
            tracing::warn!("no performance counters: the profiling lock is taken ({err})");
            return false;
        }
    }
    *users += 1;
    true
}

/// Gives the lock back for a query pool that no command buffer uses anymore.
fn release_lock() {
    let mut users = LOCK_USERS.lock().unwrap();
    *users -= 1;
    if *users == 0 {
        let (_, device) = Functions::performance_query().expect("performance queries are enabled");
        unsafe { device.release_profiling_lock() };
    }
}

const QUERIES: u32 = 128;

/// Invocation counts of one pass. `task` and `mesh` stay 0 without `mesh_queries`.
///
/// Also the layout of a query's results: the statistics come in the order of their flag
/// bits (fragment, compute, task, mesh), one `u64` each, and the pool is read straight into
/// a `Vec` of these.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct PipelineStats {
    pub fragment: u64,
    pub compute: u64,
    pub task: u64,
    pub mesh: u64,
}

/// What the queries of one pass or rendering counted.
#[derive(Clone, Debug)]
pub struct PassResult {
    pub name: &'static str,
    /// Only when the device supports pipeline statistics queries.
    pub stats: Option<PipelineStats>,
    /// The values of `PassCounters::counters`, in order, if the profiling lock was free.
    pub values: Option<[f64; MAX_COUNTERS]>,
}

/// What a frame slot's pass queries read back about the frame it submitted before.
#[derive(Clone, Debug, Default)]
pub struct PassCounters {
    /// The selection the slot records (`FrameSlot::set_counters`).
    pub counters: Arc<[&'static Counter]>,
    /// In recording order.
    pub passes: Vec<PassResult>,
}

/// The pipeline statistics and counter pools of one frame slot. The counter pool holds the
/// profiling lock while it exists.
pub struct PassQueries {
    statistics: Option<vk::QueryPool>,
    performance: Option<vk::QueryPool>,
    /// The names of the passes recorded since the reset; pass `i` uses query `i`.
    names: Vec<&'static str>,
    /// Whether a pass's queries are open.
    active: bool,
    statistic_values: Vec<PipelineStats>,
    performance_values: Vec<vk::PerformanceCounterResultKHR>,
    results: PassCounters,
}

impl PassQueries {
    /// Fails if `selection` has more than `MAX_COUNTERS` or needs more than one pass.
    pub(crate) fn new(queue_family: u32, selection: Arc<[&'static Counter]>) -> Result<Self> {
        if selection.len() > MAX_COUNTERS {
            return Err(Error::message(format!(
                "{} counters selected, at most {MAX_COUNTERS} fit",
                selection.len()
            )));
        }
        if !selection.is_empty() && passes(&selection) != 1 {
            return Err(Error::message(format!(
                "the counters need {} passes, only selections of one can be recorded",
                passes(&selection)
            )));
        }
        let device = Ctx::device();
        let features = Ctx::features();
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
                        .query_count(QUERIES),
                    None,
                )?
            })
        } else {
            None
        };
        // Counters were enumerated for the graphics family; the profiling lock must be held
        // from before the command buffer is begun until its queries are done.
        let performance =
            if !selection.is_empty() && queue_family == Ctx::gfx_queue_index() && acquire_lock() {
                let indices: Vec<u32> = selection.iter().map(|counter| counter.index).collect();
                let mut info = vk::QueryPoolPerformanceCreateInfoKHR::default()
                    .queue_family_index(queue_family)
                    .counter_indices(&indices);
                let pool = unsafe {
                    device.create_query_pool(
                        &vk::QueryPoolCreateInfo::default()
                            .query_type(vk::QueryType::PERFORMANCE_QUERY_KHR)
                            .query_count(QUERIES)
                            .push(&mut info),
                        None,
                    )
                };
                match pool {
                    Ok(pool) => Some(pool),
                    Err(err) => {
                        release_lock();
                        if let Some(statistics) = statistics {
                            unsafe { device.destroy_query_pool(statistics, None) };
                        }
                        return Err(err.into());
                    }
                }
            } else {
                None
            };
        Ok(Self {
            statistics,
            performance,
            names: Vec::new(),
            active: false,
            statistic_values: Vec::new(),
            performance_values: Vec::new(),
            results: PassCounters {
                counters: selection,
                passes: Vec::new(),
            },
        })
    }

    /// The selection this was created with.
    pub(crate) fn selection(&self) -> &Arc<[&'static Counter]> {
        &self.results.counters
    }

    /// Resets the pools at the start of a command buffer. The counter pool is reset on the
    /// host: a command buffer that begins its queries may not reset them.
    pub(crate) fn reset(&mut self, cmd: vk::CommandBuffer) {
        let device = Ctx::device();
        unsafe {
            if let Some(statistics) = self.statistics {
                device.cmd_reset_query_pool(cmd, statistics, 0, QUERIES);
            }
            if let Some(performance) = self.performance {
                device.reset_query_pool(performance, 0, QUERIES);
            }
        }
        self.names.clear();
    }

    /// Begins the queries of a pass; passes don't nest. Passes past `QUERIES` aren't counted.
    pub(crate) fn begin(&mut self, cmd: vk::CommandBuffer, name: &'static str) {
        let query = self.names.len() as u32;
        self.active = query < QUERIES;
        if !self.active {
            return;
        }
        self.names.push(name);
        let device = Ctx::device();
        for pool in [self.statistics, self.performance].into_iter().flatten() {
            unsafe { device.cmd_begin_query(cmd, pool, query, vk::QueryControlFlags::empty()) };
        }
    }

    pub(crate) fn end(&mut self, cmd: vk::CommandBuffer) {
        if !std::mem::take(&mut self.active) {
            return;
        }
        let query = self.names.len() as u32 - 1;
        let device = Ctx::device();
        for pool in [self.performance, self.statistics].into_iter().flatten() {
            unsafe { device.cmd_end_query(cmd, pool, query) };
        }
    }

    /// Reads the results of the last submission, which must have finished.
    pub(crate) fn read(&mut self) -> Result<()> {
        self.results.passes.clear();
        let queries = self.names.len();
        if queries == 0 {
            return Ok(());
        }
        let device = Ctx::device();
        // One element per query, wide enough for every flag.
        self.statistic_values
            .resize(queries, PipelineStats::default());
        if let Some(pool) = self.statistics {
            unsafe {
                device.get_query_pool_results(
                    pool,
                    0,
                    &mut self.statistic_values,
                    vk::QueryResultFlags::TYPE_64,
                )?
            };
        }
        let count = self.results.counters.len();
        self.performance_values
            .resize(queries * count, vk::PerformanceCounterResultKHR::default());
        if let Some(pool) = self.performance {
            // One result per counter and query; no flags: 64-bit, availability and partial
            // results are not allowed for counters.
            let size = size_of::<vk::PerformanceCounterResultKHR>();
            unsafe {
                (device.fp_v1_0().get_query_pool_results)(
                    device.handle(),
                    pool,
                    0,
                    queries as u32,
                    self.performance_values.len() * size,
                    self.performance_values.as_mut_ptr().cast(),
                    (size * count) as u64,
                    vk::QueryResultFlags::empty(),
                )
                .result()?
            };
        }
        for (query, name) in self.names.iter().enumerate() {
            let values = self.performance.map(|_| {
                let results = &self.performance_values[query * count..][..count];
                let mut values = [0.0; MAX_COUNTERS];
                for (value, (counter, result)) in values
                    .iter_mut()
                    .zip(self.results.counters.iter().zip(results))
                {
                    *value = counter.value(result);
                }
                values
            });
            self.results.passes.push(PassResult {
                name,
                stats: self.statistic_values.get(query).cloned(),
                values,
            });
        }
        Ok(())
    }

    pub(crate) fn results(&self) -> Option<&PassCounters> {
        (!self.results.passes.is_empty()).then_some(&self.results)
    }
}

impl Drop for PassQueries {
    fn drop(&mut self) {
        let device = Ctx::device();
        unsafe {
            if let Some(statistics) = self.statistics {
                device.destroy_query_pool(statistics, None);
            }
            if let Some(performance) = self.performance {
                device.destroy_query_pool(performance, None);
                release_lock();
            }
        }
    }
}

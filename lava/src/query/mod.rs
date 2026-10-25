//! GPU queries: user-owned pools of one kind each (timestamps, and with their features pipeline and compute statistics, occlusion, hardware counters, compacted sizes), the kinds each queue family records as a type (`RecordedOn`), each query taken once per recording and the pool reset on the host when a recording first touches it, read without waiting, calibrated against the host's clock (cached), and destroyed once the pool and every submission that used it are gone (`PoolOwner`)
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "calibrated-timestamps")]
use std::{
    sync::Mutex,
    time::{Duration, Instant},
};

use crate::vk;

#[allow(unused_imports)]
use crate::{
    command_buffer::Recording,
    error::{Error, Result},
    features::{Feature, Gated, gate},
    queue::{HasCompute, HasGraphics, QueueKind, timeline::poll},
    sealed::Sealed,
    state::{Ctx, Functions},
};

#[cfg(feature = "performance-query")]
pub mod counters;
pub mod observer;

pub use crate::vk::QueryPipelineStatisticFlags;
#[cfg(feature = "performance-query")]
pub use counters::{Counter, CounterScope, CounterUnit, MAX_COUNTERS};
pub use observer::{QueryRecorder, ScopeInfo, ScopeObserver};

/// What a pool's queries measure. Only lava implements it.
pub trait QueryKind: Sealed {
    #[doc(hidden)]
    const TYPE: vk::QueryType;
}

/// The query kinds a recording on a family of kind `Q` can begin or write (sealed through
/// [`QueryKind`]): timestamps on every family (one without a clock panics), occlusion queries,
/// pipeline statistics and hardware counters on graphics, compute statistics on graphics and
/// compute. Vulkan refuses the others (`vkCmdBeginQuery`'s queue family rules), so they don't
/// compile:
///
/// ```compile_fail
/// # use lava::{command_buffer::CommandBuffer, query::{Occlusion, QueryPool}, queue::Compute};
/// fn count(cmd: &mut CommandBuffer<Compute>, pool: &QueryPool<Occlusion>) {
///     cmd.query(pool, 0, |_| {});
/// }
/// ```
///
/// ```compile_fail
/// # use lava::{command_buffer::CommandBuffer, query::{ComputeStatistics, QueryPool}, queue::Transfer};
/// fn count(cmd: &mut CommandBuffer<Transfer>, pool: &QueryPool<ComputeStatistics>) {
///     cmd.query(pool, 0, |_| {});
/// }
/// ```
///
/// ```compile_fail
/// # use lava::{command_buffer::CommandBuffer, query::{PipelineStatistics, QueryPool}, queue::Compute};
/// fn count(cmd: &mut CommandBuffer<Compute>, pool: &QueryPool<PipelineStatistics>) {
///     cmd.query(pool, 0, |_| {});
/// }
/// ```
///
/// ```no_run
/// # use lava::{command_buffer::CommandBuffer, query::{ComputeStatistics, Occlusion, QueryPool}, queue::{Compute, Graphics}};
/// fn count(cmd: &mut CommandBuffer<Graphics>, pool: &QueryPool<Occlusion>) {
///     cmd.query(pool, 0, |_| {});
/// }
/// fn count_compute(cmd: &mut CommandBuffer<Compute>, pool: &QueryPool<ComputeStatistics>) {
///     cmd.query(pool, 0, |_| {});
/// }
/// ```
pub trait RecordedOn<Q: QueueKind>: QueryKind {}

/// Kinds whose queries are begun and ended around commands (`CommandBuffer::query`).
pub trait Scoped: QueryKind {
    #[doc(hidden)]
    fn control_flags(&self) -> vk::QueryControlFlags {
        vk::QueryControlFlags::empty()
    }
    /// The narrowest scope of its counters (`COMMAND` for kinds that aren't counters).
    #[doc(hidden)]
    fn counter_scope(&self) -> vk::PerformanceCounterScopeKHR {
        vk::PerformanceCounterScopeKHR::COMMAND
    }
    /// The submissions it takes to count every one of its counters (1 for kinds that aren't
    /// counters).
    #[doc(hidden)]
    fn counter_passes(&self) -> u32 {
        1
    }
}

/// Kinds with one `u64` per query.
pub trait SingleValue: QueryKind {}

/// The device's clock when a command reached a stage (`CommandBuffer::write_timestamp`).
pub struct Timestamp {
    /// The valid bits of the narrowest family with timestamps, as a mask: differences of
    /// timestamps written on any family are right below its wrap.
    valid_bits: u64,
    /// The last calibration of the device's clock against the host's (`instant`).
    #[cfg(feature = "calibrated-timestamps")]
    calibration: Mutex<Option<Calibration>>,
}

/// A device tick and the host's clock when the device read it.
#[cfg(feature = "calibrated-timestamps")]
#[derive(Clone, Copy)]
struct Calibration {
    tick: u64,
    at: Instant,
}

/// Invocation and primitive counts of the commands in a query, recorded on the graphics family.
#[cfg(feature = "pipeline-statistics")]
pub struct PipelineStatistics {
    flags: vk::QueryPipelineStatisticFlags,
}

/// The compute shader invocations of the commands in a query: the one statistic a compute family
/// counts too.
#[cfg(feature = "pipeline-statistics")]
pub struct ComputeStatistics;

/// Samples that passed the depth and stencil tests in a query.
pub struct Occlusion {
    precise: bool,
}

/// Hardware counters (`VK_KHR_performance_query`) of the commands in a query.
#[cfg(feature = "performance-query")]
pub struct Performance {
    /// The counters of each query, in result order.
    counters: Box<[&'static Counter]>,
    /// The narrowest scope among them.
    scope: vk::PerformanceCounterScopeKHR,
    /// The submissions it takes to count all of them.
    passes: u32,
}

/// The compacted size of an acceleration structure (`CommandBuffer::query_compacted_sizes`).
#[cfg(feature = "acceleration-structure")]
pub struct CompactedSize;

/// The serialized size of an acceleration structure (`CommandBuffer::query_serialized_sizes`).
#[cfg(feature = "acceleration-structure")]
pub struct SerializedSize;

macro_rules! kinds {
    ($($kind:ident: $ty:ident $(+ $trait:ident)*;)*) => {$(
        impl Sealed for $kind {}
        impl QueryKind for $kind {
            const TYPE: vk::QueryType = vk::QueryType::$ty;
        }
        $(impl $trait for $kind {})*
    )*};
}

kinds! {
    Timestamp: TIMESTAMP + SingleValue;
    Occlusion: OCCLUSION + SingleValue;
}
impl<Q: QueueKind> RecordedOn<Q> for Timestamp {}
impl<Q: HasGraphics> RecordedOn<Q> for Occlusion {}

#[cfg(feature = "pipeline-statistics")]
kinds! {
    PipelineStatistics: PIPELINE_STATISTICS + Scoped;
    ComputeStatistics: PIPELINE_STATISTICS + Scoped + SingleValue;
}
#[cfg(feature = "pipeline-statistics")]
impl<Q: HasGraphics> RecordedOn<Q> for PipelineStatistics {}
#[cfg(feature = "pipeline-statistics")]
impl<Q: HasCompute> RecordedOn<Q> for ComputeStatistics {}

#[cfg(feature = "performance-query")]
kinds! {
    Performance: PERFORMANCE_QUERY_KHR;
}
#[cfg(feature = "performance-query")]
impl Scoped for Performance {
    fn counter_scope(&self) -> vk::PerformanceCounterScopeKHR {
        self.scope
    }
    fn counter_passes(&self) -> u32 {
        self.passes
    }
}
#[cfg(feature = "performance-query")]
impl<Q: HasGraphics> RecordedOn<Q> for Performance {}

#[cfg(feature = "acceleration-structure")]
kinds! {
    CompactedSize: ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR + SingleValue;
    SerializedSize: ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR + SingleValue;
}

impl Scoped for Occlusion {
    fn control_flags(&self) -> vk::QueryControlFlags {
        if self.precise {
            vk::QueryControlFlags::PRECISE
        } else {
            vk::QueryControlFlags::empty()
        }
    }
}

/// `count` queries of kind `K`. Recorded through a shared reference, like images, so a
/// rendering can borrow it. A pool holds one recording's results: the first command of a
/// recording that touches it resets it, and each of its queries takes one timestamp, query or
/// size per recording.
pub struct QueryPool<K: QueryKind> {
    pub(crate) handle: vk::QueryPool,
    count: u32,
    /// The recording that last touched it: its value shifted past its timeline's 3 bits, 0 for
    /// none.
    last: AtomicU64,
    /// One bit per query that recording used.
    used: Box<[AtomicU64]>,
    pub(crate) kind: K,
    /// What destroys the pool, which recordings keep until their submissions completed.
    owner: std::sync::Arc<PoolOwner>,
}

/// A query pool's `VkQueryPool`, destroyed once the pool and every submission that used it are
/// gone.
struct PoolOwner {
    handle: vk::QueryPool,
    /// A performance pool holds the profiling lock.
    #[cfg(feature = "performance-query")]
    locked: bool,
    /// Last, so it counts the pool until its drop finished.
    _live: crate::global::Live,
}

impl Drop for PoolOwner {
    fn drop(&mut self) {
        unsafe { Ctx::device().destroy_query_pool(self.handle, None) };
        #[cfg(feature = "performance-query")]
        if self.locked {
            counters::release_lock();
        }
    }
}

impl<K: QueryKind> QueryPool<K> {
    fn create(count: u32, info: vk::QueryPoolCreateInfo, kind: K) -> Result<Self> {
        let live = crate::global::Live::new();
        assert!(count > 0, "a query pool holds at least one query");
        let info = info.query_type(K::TYPE).query_count(count);
        let handle = unsafe { Ctx::device().create_query_pool(&info, None)? };
        // Unavailable until written, so it can be polled before its first recording.
        unsafe { Ctx::device().reset_query_pool(handle, 0, count) };
        Ok(Self {
            handle,
            count,
            last: AtomicU64::new(0),
            used: (0..count.div_ceil(64)).map(|_| AtomicU64::new(0)).collect(),
            kind,
            owner: std::sync::Arc::new(PoolOwner {
                handle,
                #[cfg(feature = "performance-query")]
                locked: K::TYPE == vk::QueryType::PERFORMANCE_QUERY_KHR,
                _live: live,
            }),
        })
    }

    pub fn count(&self) -> u32 {
        self.count
    }

    /// Called by every command that touches the pool: resets it on the host the first time a
    /// recording does, except in a later pass of the capture that profiles it, which keeps the
    /// earlier passes' counters.
    fn record(&self, rec: &mut Recording) {
        const _: () = assert!(crate::queue::MAX_TIMELINES <= 8);
        let packed = rec.value << 3 | rec.timeline as u64;
        let last = self.last.swap(packed, Ordering::Relaxed);
        if last == packed {
            return;
        }
        // The submission keeps the pool until it completed.
        rec.locals.keep.other(self.owner.clone());
        for word in &self.used {
            word.store(0, Ordering::Relaxed);
        }
        if rec
            .profile
            .is_some_and(|p| p.pool == self.handle && p.pass > 0)
        {
            return;
        }
        let (timeline, value) = ((last & 7) as usize, last >> 3);
        assert!(
            last == 0 || poll(timeline).is_ok_and(|done| done >= value),
            "a query pool is recorded into while an earlier submission that recorded into it \
             still runs: keep one pool per frame slot"
        );
        unsafe { Ctx::device().reset_query_pool(self.handle, 0, self.count) };
    }

    /// Takes queries `first..first + count` for the recording `rec`, resetting the pool first if
    /// `rec` is the first to touch it. Panics if they aren't in the pool or `rec` took one of
    /// them already.
    #[track_caller]
    pub(crate) fn take(&self, rec: &mut Recording, first: u32, count: u32) {
        let end = first.checked_add(count).filter(|&end| end <= self.count);
        let Some(end) = end else {
            panic!(
                "queries {first}.. ({count} of them) of a pool of {}",
                self.count
            );
        };
        self.record(rec);
        for query in first..end {
            let bit = 1 << (query % 64);
            let taken = self.used[query as usize / 64].fetch_or(bit, Ordering::Relaxed);
            assert!(
                taken & bit == 0,
                "query {query} of a pool is used twice in one recording: a query holds one \
                 timestamp, query or size per recording"
            );
        }
    }

    fn read_raw<T>(
        &self,
        first: u32,
        queries: usize,
        out: &mut [T],
        stride: usize,
    ) -> Result<bool> {
        let end = (first as usize).checked_add(queries);
        assert!(
            end.is_some_and(|end| end <= self.count as usize),
            "{queries} queries from {first} on of a pool of {}",
            self.count
        );
        if queries == 0 {
            return Ok(true);
        }
        // Counter results take no flags: 64-bit, availability and partial results are refused.
        let flags = if K::TYPE == vk::QueryType::PERFORMANCE_QUERY_KHR {
            vk::QueryResultFlags::empty()
        } else {
            vk::QueryResultFlags::TYPE_64
        };
        let device = Ctx::device();
        let result = unsafe {
            (device.fp_v1_0().get_query_pool_results)(
                device.handle(),
                self.handle,
                first,
                queries as u32,
                size_of_val(out),
                out.as_mut_ptr().cast(),
                stride as u64,
                flags,
            )
        };
        match result {
            vk::Result::SUCCESS => Ok(true),
            vk::Result::NOT_READY => Ok(false),
            err => Err(err.into()),
        }
    }
}

impl<K: SingleValue> QueryPool<K> {
    /// Never waits: writes the results of queries `first..first + out.len()` and returns true
    /// once the recording that wrote them finished, else returns false.
    pub fn read(&self, first: u32, out: &mut [u64]) -> Result<bool> {
        self.read_raw(first, out.len(), out, size_of::<u64>())
    }
}

impl QueryPool<Timestamp> {
    pub fn timestamps(count: u32) -> Result<Self> {
        let bits = (Ctx::get().queues.iter())
            .map(|family| family.timestamp_valid_bits)
            .filter(|&bits| bits > 0)
            .min()
            .unwrap_or(64);
        let kind = Timestamp {
            valid_bits: if bits >= 64 { !0 } else { (1 << bits) - 1 },
            #[cfg(feature = "calibrated-timestamps")]
            calibration: Mutex::new(None),
        };
        Self::create(count, vk::QueryPoolCreateInfo::default(), kind)
    }

    /// Nanoseconds from timestamp `start` to `end`, which is later, also across a wrap of the
    /// device's clock.
    pub fn elapsed_ns(&self, start: u64, end: u64) -> u64 {
        let ticks = end.wrapping_sub(start) & self.kind.valid_bits;
        let period = Ctx::device_properties().limits.timestamp_period as f64;
        (ticks as f64 * period) as u64
    }

    /// When the device's clock read `tick`, on the host's clock, `None` if calibrating failed.
    /// Calibrates the clocks at most once per [`CALIBRATION_PERIOD`] per pool, retrying a few
    /// times while the calibration's deviation is over [`CALIBRATION_DEVIATION`] (keeping the
    /// best one).
    #[cfg(feature = "calibrated-timestamps")]
    pub fn instant(&self, tick: u64) -> Gated<gate::CalibratedTimestamps, Option<Instant>> {
        crate::features::gated::<gate::CalibratedTimestamps, _>(|| {
            let calibration = {
                let mut cached = crate::queue::lock(&self.kind.calibration);
                match *cached {
                    Some(c) if c.at.elapsed() < CALIBRATION_PERIOD => c,
                    _ => {
                        let fresh = calibrate()?;
                        *cached = Some(fresh);
                        fresh
                    }
                }
            };
            let valid_bits = self.kind.valid_bits;
            let period = Ctx::device_properties().limits.timestamp_period as f64;
            let ns = |ticks: u64| Duration::from_nanos((ticks as f64 * period) as u64);
            // A tick before the calibration is a positive masked distance, unless the clock
            // wrapped in between, which reads as the other way round.
            let behind = calibration.tick.wrapping_sub(tick) & valid_bits;
            Some(if behind <= valid_bits >> 1 {
                calibration.at - ns(behind)
            } else {
                calibration.at + ns(tick.wrapping_sub(calibration.tick) & valid_bits)
            })
        })
    }
}

/// How long [`QueryPool::instant`] keeps a calibration: the clocks drift apart by parts per
/// million, a microsecond per second at most.
#[cfg(feature = "calibrated-timestamps")]
pub const CALIBRATION_PERIOD: Duration = Duration::from_secs(1);

/// The uncertainty of a calibration [`QueryPool::instant`] accepts at once: the driver's
/// `maxDeviation` plus the host's time between asking and getting the device's tick.
#[cfg(feature = "calibrated-timestamps")]
pub const CALIBRATION_DEVIATION: Duration = Duration::from_micros(50);

/// The device's clock against the host's: the best of up to four tries, stopping at the first
/// within [`CALIBRATION_DEVIATION`]. `None` if every try failed.
#[cfg(feature = "calibrated-timestamps")]
fn calibrate() -> Option<Calibration> {
    let functions = Functions::calibrated_timestamps();
    let info = [vk::CalibratedTimestampInfoKHR::default().time_domain(vk::TimeDomainKHR::DEVICE)];
    let mut best: Option<(Duration, Calibration)> = None;
    for _ in 0..4 {
        let before = Instant::now();
        let calibrated = unsafe { functions.get_calibrated_timestamps(&info) };
        let after = Instant::now();
        let (ticks, max_deviation) = match calibrated {
            Ok(calibrated) => calibrated,
            Err(err) => {
                tracing::warn!("failed to calibrate the GPU timestamps: {err}");
                continue;
            }
        };
        let deviation = Duration::from_nanos(max_deviation) + (after - before) / 2;
        let calibration = Calibration {
            tick: ticks[0],
            at: before + (after - before) / 2,
        };
        if best.is_none_or(|(d, _)| deviation < d) {
            best = Some((deviation, calibration));
        }
        if deviation <= CALIBRATION_DEVIATION {
            break;
        }
    }
    best.map(|(_, calibration)| calibration)
}

/// Every flag up to mesh shader invocations, plus one spare.
#[cfg(feature = "pipeline-statistics")]
const STATISTICS: usize = 14;

/// What one pipeline statistics query counted, for the flags of its pool. Results are read
/// straight into it: one `u64` per flag, in bit order, from offset 0.
#[cfg(feature = "pipeline-statistics")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct PipelineStats {
    values: [u64; STATISTICS],
    flags: vk::QueryPipelineStatisticFlags,
}

#[cfg(feature = "pipeline-statistics")]
impl PipelineStats {
    /// `None` if the pool doesn't count `statistic`.
    pub fn get(&self, statistic: vk::QueryPipelineStatisticFlags) -> Option<u64> {
        assert_eq!(
            statistic.as_raw().count_ones(),
            1,
            "one statistic at a time"
        );
        self.flags.contains(statistic).then(|| {
            let below = self.flags.as_raw() & (statistic.as_raw() - 1);
            self.values[below.count_ones() as usize]
        })
    }

    pub fn flags(&self) -> vk::QueryPipelineStatisticFlags {
        self.flags
    }
}

#[cfg(feature = "pipeline-statistics")]
impl QueryPool<PipelineStatistics> {
    /// Queries that count `statistics`, on the graphics family ([`ComputeStatistics`] counts on
    /// compute families too). Task and mesh shader invocations need `mesh-shader-queries`;
    /// queries that are active while mesh shaders draw may not count input assembly, vertex,
    /// geometry, tessellation or clipping statistics.
    ///
    /// Panics without any `statistics`.
    pub fn statistics(count: u32, statistics: vk::QueryPipelineStatisticFlags) -> Result<Self> {
        assert!(
            !statistics.is_empty(),
            "a pipeline statistics pool counts at least one statistic"
        );
        gate::PipelineStatistics::check()?;
        let mesh = vk::QueryPipelineStatisticFlags::TASK_SHADER_INVOCATIONS_EXT
            | vk::QueryPipelineStatisticFlags::MESH_SHADER_INVOCATIONS_EXT;
        if statistics.intersects(mesh) {
            #[cfg(feature = "mesh-shader-queries")]
            gate::MeshShaderQueries::check()?;
            #[cfg(not(feature = "mesh-shader-queries"))]
            return Err(crate::features::MissingFeature {
                feature: "mesh-shader-queries",
            }
            .into());
        }
        assert!(statistics.as_raw().count_ones() as usize <= STATISTICS);
        let info = vk::QueryPoolCreateInfo::default().pipeline_statistics(statistics);
        let kind = PipelineStatistics { flags: statistics };
        Self::create(count, info, kind)
    }

    /// Never waits: writes the results of queries `first..first + out.len()` and returns true
    /// once the recording that wrote them finished, else returns false.
    pub fn read(&self, first: u32, out: &mut [PipelineStats]) -> Result<bool> {
        let stride = size_of::<PipelineStats>();
        let ready = self.read_raw(first, out.len(), out, stride)?;
        out.iter_mut()
            .for_each(|stats| stats.flags = self.kind.flags);
        Ok(ready)
    }
}

#[cfg(feature = "pipeline-statistics")]
impl QueryPool<ComputeStatistics> {
    /// Queries that count compute shader invocations, on graphics and compute families.
    pub fn compute_statistics(count: u32) -> Result<Self> {
        gate::PipelineStatistics::check()?;
        let statistics = vk::QueryPipelineStatisticFlags::COMPUTE_SHADER_INVOCATIONS;
        let info = vk::QueryPoolCreateInfo::default().pipeline_statistics(statistics);
        Self::create(count, info, ComputeStatistics)
    }
}

impl QueryPool<Occlusion> {
    /// Queries that count the samples of their draws. A query that saw samples may report any
    /// non-zero count (`precise_occlusion`, with `precise-occlusion`, counts them exactly).
    pub fn occlusion(count: u32) -> Result<Self> {
        let kind = Occlusion { precise: false };
        Self::create(count, vk::QueryPoolCreateInfo::default(), kind)
    }

    /// Queries that count the samples of their draws exactly.
    #[cfg(feature = "precise-occlusion")]
    pub fn precise_occlusion(count: u32) -> Result<Self> {
        gate::PreciseOcclusion::check()?;
        let kind = Occlusion { precise: true };
        Self::create(count, vk::QueryPoolCreateInfo::default(), kind)
    }
}

#[cfg(feature = "performance-query")]
impl QueryPool<Performance> {
    /// Queries of the hardware counters `counters` (at most [`MAX_COUNTERS`]), recorded on the
    /// graphics family. Holds the profiling lock until it is destroyed, and fails if another
    /// process holds it.
    ///
    /// A selection that fits one submission ([`counters::fits`]) is recorded like any query,
    /// one of more passes only by [`Timeline::capture`](crate::queue::Timeline::capture). Counters
    /// of [`CounterScope::CommandBuffer`] count whole recordings
    /// ([`Submission::submit_profiled`](crate::queue::Submission::submit_profiled) or a capture),
    /// those of [`CounterScope::RenderPass`] can't begin or end inside a rendering, those of
    /// [`CounterScope::Command`] work anywhere.
    ///
    /// Panics for a `count` of 0, for no or more than [`MAX_COUNTERS`] counters, and for counters
    /// of an earlier lava context.
    pub fn performance(count: u32, counters: &[&'static Counter]) -> Result<Self> {
        let _call = crate::state::call();
        gate::PerformanceQuery::check()?;
        assert!(count > 0, "a query pool holds at least one query");
        assert!(
            !counters.is_empty() && counters.len() <= MAX_COUNTERS,
            "{} counters selected, 1 to {MAX_COUNTERS} fit",
            counters.len()
        );
        counters::check_current(counters);
        let passes = counters::passes(counters);
        {
            let mut users = crate::queue::lock(&counters::LOCK_USERS);
            if *users == 0 {
                let info = vk::AcquireProfilingLockInfoKHR::default().timeout(0);
                if let Err(err) = unsafe {
                    Functions::performance_query()
                        .1
                        .acquire_profiling_lock(&info)
                } {
                    return Err(
                        Error::from(err).context("the profiling lock is taken by another process")
                    );
                }
            }
            *users += 1;
        }
        let indices: Vec<u32> = counters.iter().map(|counter| counter.index).collect();
        let mut performance = vk::QueryPoolPerformanceCreateInfoKHR::default()
            .queue_family_index(Ctx::get().queues.graphics().index)
            .counter_indices(&indices);
        let info = vk::QueryPoolCreateInfo::default().push_next(&mut performance);
        let scope = (counters.iter().map(|counter| counter.scope))
            .min()
            .expect("at least one counter")
            .vk();
        let kind = Performance {
            counters: counters.into(),
            scope,
            passes,
        };
        Self::create(count, info, kind).inspect_err(|_| counters::release_lock())
    }

    /// In result order.
    pub fn counters(&self) -> &[&'static Counter] {
        &self.kind.counters
    }

    /// The submissions it takes to count all its counters: 1 unless it was made for a
    /// [`Timeline::capture`](crate::queue::Timeline::capture).
    pub fn passes(&self) -> u32 {
        self.kind.passes
    }

    /// Never waits: writes the counter values of queries `first..` (`counters().len()` per
    /// query, `out.len()` a multiple of that) and returns true once the recording that wrote
    /// them finished, else returns false.
    pub fn read(&self, first: u32, out: &mut [f64]) -> Result<bool> {
        let width = self.kind.counters.len();
        assert_eq!(out.len() % width, 0, "whole queries of {width} counters");
        const _: () = assert!(size_of::<vk::PerformanceCounterResultKHR>() == size_of::<f64>());
        let results: &mut [vk::PerformanceCounterResultKHR] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast(), out.len()) };
        let queries = out.len() / width;
        let ready = self.read_raw(first, queries, results, width * size_of::<f64>())?;
        if ready {
            use vk::PerformanceCounterStorageKHR as S;
            for (i, value) in out.iter_mut().enumerate() {
                let result: vk::PerformanceCounterResultKHR =
                    unsafe { std::mem::transmute(*value) };
                *value = unsafe {
                    match self.kind.counters[i % width].storage {
                        S::INT32 => result.int32 as f64,
                        S::INT64 => result.int64 as f64,
                        S::UINT32 => result.uint32 as f64,
                        S::UINT64 => result.uint64 as f64,
                        S::FLOAT32 => result.float32 as f64,
                        _ => result.float64,
                    }
                };
            }
        }
        Ok(ready)
    }
}

#[cfg(feature = "acceleration-structure")]
impl QueryPool<CompactedSize> {
    pub fn compacted_sizes(count: u32) -> Result<Self> {
        crate::vkobjects::acceleration_structure::functions()?;
        Self::create(count, vk::QueryPoolCreateInfo::default(), CompactedSize)
    }
}

#[cfg(feature = "acceleration-structure")]
impl QueryPool<SerializedSize> {
    pub fn serialized_sizes(count: u32) -> Result<Self> {
        crate::vkobjects::acceleration_structure::functions()?;
        Self::create(count, vk::QueryPoolCreateInfo::default(), SerializedSize)
    }
}

//! GPU queries: user-owned pools of one kind each (timestamps, and with their features pipeline statistics, occlusion, hardware counters, compacted sizes), reset on the host when a recording first touches them, read without waiting, and destroyed once no submission can reach them
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "calibrated-timestamps")]
use std::time::{Duration, Instant};

use crate::vk;

#[allow(unused_imports)]
use crate::{
    command_buffer::Recording,
    error::{Error, Result},
    features::{Feature, Gated, gate},
    lifetime,
    queue::{Kind, timeline::poll},
    sealed::Sealed,
    state::{Ctx, Functions},
};

#[cfg(feature = "performance-query")]
pub mod counters;
pub mod observer;

pub use crate::vk::QueryPipelineStatisticFlags;
#[cfg(feature = "performance-query")]
pub use counters::{Counter, CounterUnit, MAX_COUNTERS};
pub use observer::{QueryRecorder, ScopeInfo, ScopeObserver};

/// What a pool's queries measure. Only lava implements it.
pub trait QueryKind: Sealed {
    #[doc(hidden)]
    const TYPE: vk::QueryType;
    /// Panics unless a recording on a family of `kind` may begin a query of this pool.
    #[doc(hidden)]
    fn check_kind(&self, kind: Kind) {
        let _ = kind;
    }
}

/// Kinds whose queries are begun and ended around commands (`CommandBuffer::query`).
pub trait Scoped: QueryKind {
    #[doc(hidden)]
    fn control_flags(&self) -> vk::QueryControlFlags {
        vk::QueryControlFlags::empty()
    }
}

/// Kinds with one `u64` per query.
pub trait SingleValue: QueryKind {}

/// The device's clock when a command reached a stage (`CommandBuffer::write_timestamp`).
pub struct Timestamp {
    /// The valid bits of the family the timestamps were last written on, as a mask.
    pub(crate) valid_bits: AtomicU64,
}

/// Invocation and primitive counts of the commands in a query.
#[cfg(feature = "pipeline-statistics")]
pub struct PipelineStatistics {
    flags: vk::QueryPipelineStatisticFlags,
}

/// Samples that passed the depth and stencil tests in a query.
pub struct Occlusion {
    precise: bool,
}

/// Hardware counters (`VK_KHR_performance_query`) of the commands in a query.
#[cfg(feature = "performance-query")]
pub struct Performance {
    /// The counters of each query, in result order.
    counters: Box<[&'static Counter]>,
}

/// The compacted size of an acceleration structure (`CommandBuffer::query_compacted_sizes`).
#[cfg(feature = "acceleration-structure")]
pub struct CompactedSize;

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

#[cfg(feature = "pipeline-statistics")]
const GRAPHICS_STATISTICS: vk::QueryPipelineStatisticFlags = vk::QueryPipelineStatisticFlags::from_raw(
    !vk::QueryPipelineStatisticFlags::COMPUTE_SHADER_INVOCATIONS.as_raw(),
);

#[cfg(feature = "pipeline-statistics")]
impl Sealed for PipelineStatistics {}
#[cfg(feature = "pipeline-statistics")]
impl QueryKind for PipelineStatistics {
    const TYPE: vk::QueryType = vk::QueryType::PIPELINE_STATISTICS;
    fn check_kind(&self, kind: Kind) {
        assert!(
            kind == Kind::Graphics || !self.flags.intersects(GRAPHICS_STATISTICS),
            "pipeline statistics of graphics stages are counted on the graphics family, not {kind:?}"
        );
    }
}
#[cfg(feature = "pipeline-statistics")]
impl Scoped for PipelineStatistics {}

#[cfg(feature = "performance-query")]
impl Sealed for Performance {}
#[cfg(feature = "performance-query")]
impl QueryKind for Performance {
    const TYPE: vk::QueryType = vk::QueryType::PERFORMANCE_QUERY_KHR;
    fn check_kind(&self, kind: Kind) {
        assert_eq!(
            kind,
            Kind::Graphics,
            "hardware counters are only recorded on the graphics family"
        );
    }
}
#[cfg(feature = "performance-query")]
impl Scoped for Performance {}
#[cfg(feature = "acceleration-structure")]
kinds! {
    CompactedSize: ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR + SingleValue;
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
/// rendering can borrow it.
pub struct QueryPool<K: QueryKind> {
    pub(crate) handle: vk::QueryPool,
    count: u32,
    /// The recording that last touched it: its value shifted past its timeline's 3 bits, 0 for
    /// none.
    last: AtomicU64,
    pub(crate) kind: K,
}

impl<K: QueryKind> QueryPool<K> {
    fn create(count: u32, info: vk::QueryPoolCreateInfo, kind: K) -> Result<Self> {
        assert!(count > 0, "a query pool holds at least one query");
        let info = info.query_type(K::TYPE).query_count(count);
        let handle = unsafe { Ctx::device().create_query_pool(&info, None)? };
        // Unavailable until written, so it can be polled before its first recording.
        unsafe { Ctx::device().reset_query_pool(handle, 0, count) };
        Ok(Self {
            handle,
            count,
            last: AtomicU64::new(0),
            kind,
        })
    }

    pub fn count(&self) -> u32 {
        self.count
    }

    /// Called by every command that touches the pool: resets it on the host the first time a
    /// recording does.
    pub(crate) fn record(&self, rec: &Recording) {
        const _: () = assert!(crate::queue::MAX_TIMELINES <= 8);
        let packed = rec.value << 3 | rec.timeline as u64;
        let last = self.last.swap(packed, Ordering::Relaxed);
        if last == packed {
            return;
        }
        let (timeline, value) = ((last & 7) as usize, last >> 3);
        assert!(
            last == 0 || poll(timeline) >= value,
            "a query pool is recorded into while an earlier submission that recorded into it \
             still runs: keep one pool per frame slot"
        );
        unsafe { Ctx::device().reset_query_pool(self.handle, 0, self.count) };
    }

    fn read_raw<T>(&self, first: u32, queries: u32, out: &mut [T], stride: usize) -> Result<bool> {
        assert!(
            first + queries <= self.count,
            "queries {first}..{} of a pool of {}",
            first + queries,
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
                queries,
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
        self.read_raw(first, out.len() as u32, out, size_of::<u64>())
    }
}

impl QueryPool<Timestamp> {
    pub fn timestamps(count: u32) -> Result<Self> {
        let kind = Timestamp {
            valid_bits: AtomicU64::new(0),
        };
        Self::create(count, vk::QueryPoolCreateInfo::default(), kind)
    }

    /// Nanoseconds from timestamp `start` to `end`, which is later, also across a wrap of the
    /// device's clock.
    pub fn elapsed_ns(&self, start: u64, end: u64) -> u64 {
        let ticks = end.wrapping_sub(start) & self.kind.valid_bits.load(Ordering::Relaxed);
        let period = Ctx::device_properties().limits.timestamp_period as f64;
        (ticks as f64 * period) as u64
    }

    /// When the device's clock read `tick`, on the host's clock, `None` if calibrating failed.
    /// Calibrates once per call.
    #[cfg(feature = "calibrated-timestamps")]
    pub fn instant(&self, tick: u64) -> Gated<gate::CalibratedTimestamps, Option<Instant>> {
        gate::<gate::CalibratedTimestamps, _>(|| {
            let functions = Functions::calibrated_timestamps();
            let info = [vk::CalibratedTimestampInfoKHR::default().time_domain(vk::TimeDomainKHR::DEVICE)];
            let before = Instant::now();
            let (timestamps, _max_deviation) = unsafe { functions.get_calibrated_timestamps(&info) }
                .inspect_err(|err| tracing::warn!("failed to calibrate the GPU timestamps: {err}"))
                .ok()?;
            let after = Instant::now();
            let (now_ticks, now) = (timestamps[0], before + (after - before) / 2);
            let valid_bits = self.kind.valid_bits.load(Ordering::Relaxed);
            let period = Ctx::device_properties().limits.timestamp_period as f64;
            let ns = |ticks: u64| Duration::from_nanos((ticks as f64 * period) as u64);
            // A tick before the calibration is a positive masked distance, unless the clock
            // wrapped in between, which reads as the other way round.
            let behind = now_ticks.wrapping_sub(tick) & valid_bits;
            Some(if behind <= valid_bits >> 1 {
                now - ns(behind)
            } else {
                now + ns(tick.wrapping_sub(now_ticks) & valid_bits)
            })
        })
    }
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
    /// Queries that count `statistics`. Task and mesh shader invocations need
    /// `mesh-shader-queries`; queries that are active while mesh shaders draw may not count
    /// input assembly, vertex, geometry, tessellation or clipping statistics.
    pub fn statistics(count: u32, statistics: vk::QueryPipelineStatisticFlags) -> Result<Self> {
        gate::PipelineStatistics::check()?;
        let mesh = vk::QueryPipelineStatisticFlags::TASK_SHADER_INVOCATIONS_EXT
            | vk::QueryPipelineStatisticFlags::MESH_SHADER_INVOCATIONS_EXT;
        if statistics.intersects(mesh) {
            #[cfg(feature = "mesh-shader-queries")]
            gate::MeshShaderQueries::check()?;
            #[cfg(not(feature = "mesh-shader-queries"))]
            return Err(Error::message(
                "task and mesh shader invocations need lava's `mesh-shader-queries` feature",
            ));
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
        let ready = self.read_raw(first, out.len() as u32, out, stride)?;
        out.iter_mut()
            .for_each(|stats| stats.flags = self.kind.flags);
        Ok(ready)
    }
}

impl QueryPool<Occlusion> {
    /// Queries that count the samples of their draws. A query that saw samples may report any
    /// non-zero count ([`Self::precise_occlusion`] counts them exactly).
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
    /// Queries of the hardware counters `counters` (at most [`MAX_COUNTERS`], read in one
    /// submission: [`counters::passes`] is 1), recorded on the graphics family. Holds the
    /// profiling lock until it is destroyed, and fails if another process holds it.
    pub fn performance(count: u32, counters: &[&'static Counter]) -> Result<Self> {
        gate::PerformanceQuery::check()?;
        if counters.is_empty() || counters.len() > MAX_COUNTERS {
            return Err(Error::message(format!(
                "{} counters selected, 1 to {MAX_COUNTERS} fit",
                counters.len()
            )));
        }
        let passes = counters::passes(counters);
        if passes != 1 {
            return Err(Error::message(format!(
                "the counters need {passes} submissions, only selections of one can be recorded"
            )));
        }
        {
            let mut users = counters::LOCK_USERS.lock().unwrap();
            if *users == 0 {
                let info = vk::AcquireProfilingLockInfoKHR::default().timeout(0);
                if let Err(err) = unsafe { Functions::performance_query().1.acquire_profiling_lock(&info) } {
                    return Err(Error::message(format!("the profiling lock is taken by another process ({err})")));
                }
            }
            *users += 1;
        }
        let indices: Vec<u32> = counters.iter().map(|counter| counter.index).collect();
        let mut performance = vk::QueryPoolPerformanceCreateInfoKHR::default()
            .queue_family_index(Ctx::get().queues.graphics().index)
            .counter_indices(&indices);
        let info = vk::QueryPoolCreateInfo::default().push_next(&mut performance);
        let kind = Performance {
            counters: counters.into(),
        };
        Self::create(count, info, kind).inspect_err(|_| counters::release_lock())
    }

    /// In result order.
    pub fn counters(&self) -> &[&'static Counter] {
        &self.kind.counters
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
        let queries = (out.len() / width) as u32;
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

impl<K: QueryKind> Drop for QueryPool<K> {
    fn drop(&mut self) {
        let (pool, locked) = (self.handle, K::TYPE == vk::QueryType::PERFORMANCE_QUERY_KHR);
        lifetime::retire(move || {
            unsafe { Ctx::device().destroy_query_pool(pool, None) };
            #[cfg(feature = "performance-query")]
            if locked {
                counters::release_lock();
            }
            let _ = locked;
        });
    }
}

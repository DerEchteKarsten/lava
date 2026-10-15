//! GPU queries: user-owned pools of one kind each (timestamps, pipeline statistics, occlusion, hardware counters, compacted sizes), reset on the host when a recording first touches them, read without waiting, and destroyed once no frame can reach them
//!
//! A pool holds the results of one recording. The first command of a recording that touches a
//! pool resets it on the host, which needs the frames that recorded into it before to be done:
//! keep one pool per frame in flight. Its results are readable once that recording finished
//! ([`QueryPool::read`] never waits). A [`ScopeObserver`] that `CommandBuffer::measure` attaches
//! can record queries at the edges of every scope and pass of a recording.
use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

use ash::vk;
use ash::vk::TaggedStructure;

use crate::{
    epoch,
    error::{Error, Result},
    sealed::Sealed,
    state::{Ctx, Functions},
};

pub mod counters;
pub mod observer;

pub use ash::vk::QueryPipelineStatisticFlags;
pub use counters::{Counter, CounterUnit, MAX_COUNTERS};
pub use observer::{QueryRecorder, ScopeInfo, ScopeObserver};

pub(crate) fn init() -> Result<()> {
    counters::init()
}

/// What a pool's queries measure, and what the pool needs to know about them. Only lava
/// implements it.
pub trait QueryKind: Sealed {
    #[doc(hidden)]
    const TYPE: vk::QueryType;
}

/// Kinds whose queries are begun and ended around commands (`CommandBuffer::query`).
pub trait Scoped: QueryKind {
    /// What a query of this pool is begun with.
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
    valid_bits: AtomicU64,
}

/// Invocation and primitive counts of the commands in a query.
pub struct PipelineStatistics {
    /// What each query counts.
    flags: vk::QueryPipelineStatisticFlags,
}

/// Samples that passed the depth and stencil tests in a query.
pub struct Occlusion {
    /// Whether queries count samples exactly.
    precise: bool,
}

/// Hardware counters (`VK_KHR_performance_query`) of the commands in a query.
pub struct Performance {
    /// The counters of each query, in result order.
    counters: Box<[&'static Counter]>,
}

/// The compacted size of an acceleration structure (`CommandBuffer::query_compacted_sizes`).
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
    PipelineStatistics: PIPELINE_STATISTICS + Scoped;
    Occlusion: OCCLUSION + SingleValue;
    Performance: PERFORMANCE_QUERY_KHR + Scoped;
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
    /// The epoch of the recording that last touched it, 0 for none.
    epoch: AtomicU64,
    pub(crate) kind: K,
}

impl<K: QueryKind> QueryPool<K> {
    /// A pool of `count` queries of `kind`, whose type-specific creation info is in `info`.
    fn create(count: u32, info: vk::QueryPoolCreateInfo, kind: K) -> Result<Self> {
        assert!(count > 0, "a query pool holds at least one query");
        let info = info.query_type(K::TYPE).query_count(count);
        let handle = unsafe { Ctx::device().create_query_pool(&info, None)? };
        // Unavailable until written, so it can be polled before its first recording.
        unsafe { Ctx::device().reset_query_pool(handle, 0, count) };
        Ok(Self {
            handle,
            count,
            epoch: AtomicU64::new(0),
            kind,
        })
    }

    pub fn count(&self) -> u32 {
        self.count
    }

    /// Called by every command that touches the pool: resets it on the host the first time a
    /// recording does, which needs the recordings before to be done.
    pub(crate) fn record(&self, epoch: u64) {
        let last = self.epoch.swap(epoch, Ordering::Relaxed);
        if last == epoch {
            return;
        }
        assert!(
            last <= epoch::completed(),
            "a query pool is recorded into while an earlier frame that recorded into it still runs: \
             keep one pool per frame in flight"
        );
        unsafe { Ctx::device().reset_query_pool(self.handle, 0, self.count) };
    }

    /// Reads the results of `queries` queries from `first` on into `out`, `stride` bytes per
    /// query, if they are all available.
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

    /// Resets the pool if needed, and remembers the valid bits of `family`, which writes it.
    pub(crate) fn record_on(&self, epoch: u64, family: u32) {
        self.record(epoch);
        let bits = Ctx::physical_device().queue_families[family as usize]
            .handel
            .timestamp_valid_bits;
        assert!(bits > 0, "queue family {family} has no timestamps");
        let mask = if bits >= 64 { !0 } else { (1 << bits) - 1 };
        self.kind.valid_bits.store(mask, Ordering::Relaxed);
    }

    /// Nanoseconds from timestamp `start` to `end`, which is later, also across a wrap of the
    /// device's clock.
    pub fn elapsed_ns(&self, start: u64, end: u64) -> u64 {
        let ticks = end.wrapping_sub(start) & self.kind.valid_bits.load(Ordering::Relaxed);
        let period = Ctx::device_properties().limits.timestamp_period as f64;
        (ticks as f64 * period) as u64
    }

    /// When the device's clock read `tick`, on the host's clock: from
    /// `VK_KHR_calibrated_timestamps` (`Features::calibrated_timestamps`), `None` without it.
    /// Calibrates once per call.
    pub fn instant(&self, tick: u64) -> Option<Instant> {
        let (now_ticks, now) = calibrate()?;
        let valid_bits = self.kind.valid_bits.load(Ordering::Relaxed);
        let period = Ctx::device_properties().limits.timestamp_period as f64;
        let ns = |ticks: u64| Duration::from_nanos((ticks as f64 * period) as u64);
        // A tick before the calibration is a positive distance in the masked ticks, unless the
        // clock wrapped in between, which reads as the other way round.
        let behind = now_ticks.wrapping_sub(tick) & valid_bits;
        Some(if behind <= valid_bits >> 1 {
            now - ns(behind)
        } else {
            now + ns(tick.wrapping_sub(now_ticks) & valid_bits)
        })
    }
}

/// The device's clock (its raw timestamp) and the host's at the same moment, from
/// `VK_KHR_calibrated_timestamps`: the host time is the middle of the call.
fn calibrate() -> Option<(u64, Instant)> {
    let functions = Functions::calibrated_timestamps()?;
    let info = [vk::CalibratedTimestampInfoKHR::default().time_domain(vk::TimeDomainKHR::DEVICE)];
    let before = Instant::now();
    let (timestamps, _max_deviation) = unsafe { functions.get_calibrated_timestamps(&info) }
        .inspect_err(|err| tracing::warn!("failed to calibrate the GPU timestamps: {err}"))
        .ok()?;
    let after = Instant::now();
    Some((timestamps[0], before + (after - before) / 2))
}

/// The most statistics one query can count: every flag up to mesh shader invocations, plus
/// one spare.
const STATISTICS: usize = 14;

/// What one pipeline statistics query counted, for the flags of its pool.
///
/// Also the layout a query's results are read into: they come in the order of their flag bits,
/// one `u64` each, from offset 0, and the pool is read straight into a slice of these.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct PipelineStats {
    values: [u64; STATISTICS],
    flags: vk::QueryPipelineStatisticFlags,
}

impl PipelineStats {
    /// The count of one statistic, `None` if the pool doesn't count it.
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

impl QueryPool<PipelineStatistics> {
    /// Queries that count `statistics`. Task and mesh shader invocations need
    /// `Features::mesh_queries`; queries that are active while mesh shaders draw may not count
    /// input assembly, vertex, geometry, tessellation or clipping statistics.
    pub fn statistics(count: u32, statistics: vk::QueryPipelineStatisticFlags) -> Result<Self> {
        let features = Ctx::features();
        if !features.pipeline_statistics {
            return Err(Error::message(
                "the device has no pipeline statistics queries",
            ));
        }
        let mesh = vk::QueryPipelineStatisticFlags::TASK_SHADER_INVOCATIONS_EXT
            | vk::QueryPipelineStatisticFlags::MESH_SHADER_INVOCATIONS_EXT;
        if statistics.intersects(mesh) && !features.mesh_queries {
            return Err(Error::message(
                "the device can't count task and mesh shader invocations",
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
    /// Queries that count the samples of their draws. Without `precise` (or without
    /// `Features::precise_occlusion`), a query that saw samples may report any non-zero count.
    pub fn occlusion(count: u32, precise: bool) -> Result<Self> {
        let kind = Occlusion {
            precise: precise && Ctx::features().precise_occlusion,
        };
        Self::create(count, vk::QueryPoolCreateInfo::default(), kind)
    }
}

impl QueryPool<Performance> {
    /// Queries of the hardware counters `counters` (at most [`MAX_COUNTERS`], read in one
    /// submission: [`counters::passes`] is 1), recorded on the graphics family. Holds the
    /// profiling lock until it is destroyed, and fails if another process holds it.
    pub fn performance(count: u32, counters: &[&'static Counter]) -> Result<Self> {
        if Functions::performance_query().is_none() {
            return Err(Error::message("the device has no performance queries"));
        }
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
        // The lock must be held from before a command buffer that uses the pool is begun until
        // its queries are done.
        if !counters::acquire_lock() {
            return Err(Error::message(
                "the profiling lock is taken by another process",
            ));
        }
        let indices: Vec<u32> = counters.iter().map(|counter| counter.index).collect();
        let mut performance = vk::QueryPoolPerformanceCreateInfoKHR::default()
            .queue_family_index(Ctx::gfx_queue_index())
            .counter_indices(&indices);
        let info = vk::QueryPoolCreateInfo::default().push(&mut performance);
        let kind = Performance {
            counters: counters.into(),
        };
        Self::create(count, info, kind).inspect_err(|_| counters::release_lock())
    }

    /// The counters of each query, in result order.
    pub fn counters(&self) -> &[&'static Counter] {
        &self.kind.counters
    }

    /// Never waits: writes the counter values of queries `first..` (`counters().len()` per
    /// query, `out.len()` a multiple of that) and returns true once the recording that wrote
    /// them finished, else returns false.
    pub fn read(&self, first: u32, out: &mut [f64]) -> Result<bool> {
        let width = self.kind.counters.len();
        assert_eq!(out.len() % width, 0, "whole queries of {width} counters");
        // A result is an 8-byte union, so it is read in place and converted.
        const _: () = assert!(size_of::<vk::PerformanceCounterResultKHR>() == size_of::<f64>());
        let results: &mut [vk::PerformanceCounterResultKHR] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast(), out.len()) };
        let queries = (out.len() / width) as u32;
        let ready = self.read_raw(first, queries, results, width * size_of::<f64>())?;
        if ready {
            for (i, value) in out.iter_mut().enumerate() {
                let result: vk::PerformanceCounterResultKHR =
                    unsafe { std::mem::transmute(*value) };
                *value = self.kind.counters[i % width].value(&result);
            }
        }
        Ok(ready)
    }
}

impl QueryPool<CompactedSize> {
    /// Room for the compacted sizes of `count` acceleration structures.
    pub fn compacted_sizes(count: u32) -> Result<Self> {
        crate::vkobjects::acceleration_structure::functions()?;
        Self::create(count, vk::QueryPoolCreateInfo::default(), CompactedSize)
    }
}

impl<K: QueryKind> Drop for QueryPool<K> {
    fn drop(&mut self) {
        let (pool, locked) = (self.handle, K::TYPE == vk::QueryType::PERFORMANCE_QUERY_KHR);
        epoch::retire(move || {
            unsafe { Ctx::device().destroy_query_pool(pool, None) };
            if locked {
                counters::release_lock();
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vk::QueryPipelineStatisticFlags as S;

    #[test]
    fn statistics_are_found_by_their_flag_among_the_counted_ones() {
        let stats = PipelineStats {
            values: [7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            flags: S::FRAGMENT_SHADER_INVOCATIONS
                | S::COMPUTE_SHADER_INVOCATIONS
                | S::MESH_SHADER_INVOCATIONS_EXT,
        };
        assert_eq!(stats.get(S::FRAGMENT_SHADER_INVOCATIONS), Some(7));
        assert_eq!(stats.get(S::COMPUTE_SHADER_INVOCATIONS), Some(8));
        assert_eq!(stats.get(S::MESH_SHADER_INVOCATIONS_EXT), Some(9));
        assert_eq!(stats.get(S::TASK_SHADER_INVOCATIONS_EXT), None);
    }

    #[test]
    fn every_statistic_up_to_mesh_invocations_fits() {
        let all = S::from_raw((S::MESH_SHADER_INVOCATIONS_EXT.as_raw() << 1) - 1);
        assert!(all.as_raw().count_ones() as usize <= STATISTICS);
    }
}

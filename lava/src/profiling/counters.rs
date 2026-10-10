//! Hardware performance counters (`VK_KHR_performance_query`): the device's counters, the selection every profiled frame slot records per pass, and the profiling lock
use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicU64, Ordering},
};

use ash::vk;

use crate::{
    error::{Error, Result},
    state::{Ctx, Functions},
};

/// The most counters a selection can hold; `GpuScope::counters` has room for this many values.
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
static SELECTION: Mutex<Option<Arc<[&'static Counter]>>> = Mutex::new(None);
/// Counts the changes of `SELECTION`, so frame slots notice one without locking.
static GENERATION: AtomicU64 = AtomicU64::new(0);
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
        instance.enumerate_physical_device_queue_family_performance_query_counters_len(
            device, family,
        )?
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

/// Makes profiled frame slots begun from now on record `selection` per pass and rendering
/// (empty: none). Fails if it has more than `MAX_COUNTERS` or needs more than one pass.
pub fn set_counters(selection: &[&'static Counter]) -> Result<()> {
    if selection.len() > MAX_COUNTERS {
        return Err(Error::message(format!(
            "{} counters selected, at most {MAX_COUNTERS} fit",
            selection.len()
        )));
    }
    if !selection.is_empty() {
        let passes = passes(selection);
        if passes != 1 {
            return Err(Error::message(format!(
                "the counters need {passes} passes, only selections of one can be recorded"
            )));
        }
    }
    *SELECTION.lock().unwrap() = Some(selection.into());
    GENERATION.fetch_add(1, Ordering::Release);
    Ok(())
}

/// The counters profiled frame slots record.
pub fn selected_counters() -> Arc<[&'static Counter]> {
    SELECTION.lock().unwrap().clone().unwrap_or_else(|| Arc::new([]))
}

pub(crate) fn generation() -> u64 {
    GENERATION.load(Ordering::Acquire)
}

/// Takes the profiling lock for one more query pool; false if another process holds it.
pub(crate) fn acquire_lock() -> bool {
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
pub(crate) fn release_lock() {
    let mut users = LOCK_USERS.lock().unwrap();
    *users -= 1;
    if *users == 0 {
        let (_, device) = Functions::performance_query().expect("performance queries are enabled");
        unsafe { device.release_profiling_lock() };
    }
}

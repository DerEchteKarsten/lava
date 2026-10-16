//! Hardware performance counters (`VK_KHR_performance_query`, the `performance-query` feature): the graphics family's counters, how many passes a selection needs, and the profiling lock counter pools hold
use std::sync::{Mutex, OnceLock};

use crate::vk;

use crate::{
    error::Result,
    features::{Gated, On, gate},
    state::{Ctx, Functions},
};

/// The most counters a performance query pool can hold.
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
    if !gate::PerformanceQuery::on() {
        return Ok(());
    }
    let (instance, _) = Functions::performance_query();
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

/// Every counter the graphics queue family offers.
pub fn counters() -> Gated<gate::PerformanceQuery, &'static [Counter]> {
    gate::<gate::PerformanceQuery, _>(|| COUNTERS.get().map_or(&[][..], Vec::as_slice))
}

/// How many submissions of a frame it takes to read `selection`, counters of
/// [`counters()`] (0 without performance queries, which have no counters). Only selections of
/// one pass can be recorded.
pub fn passes(selection: &[&Counter]) -> u32 {
    if !gate::PerformanceQuery::on() {
        return 0;
    }
    let (instance, _) = Functions::performance_query();
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

/// Takes the profiling lock for one more counter pool; false if another process holds it.
pub(crate) fn acquire_lock() -> bool {
    let mut users = LOCK_USERS.lock().unwrap();
    if *users == 0 {
        let (_, device) = Functions::performance_query();
        let info = vk::AcquireProfilingLockInfoKHR::default().timeout(0);
        if let Err(err) = unsafe { device.acquire_profiling_lock(&info) } {
            tracing::warn!("no performance counters: the profiling lock is taken ({err})");
            return false;
        }
    }
    *users += 1;
    true
}

/// Gives the lock back for a counter pool that no submission uses any more.
pub(crate) fn release_lock() {
    let mut users = LOCK_USERS.lock().unwrap();
    *users -= 1;
    if *users == 0 {
        let (_, device) = Functions::performance_query();
        unsafe { device.release_profiling_lock() };
    }
}

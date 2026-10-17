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
    /// Whether values are a rate or state over the measured time (averaged) rather than an
    /// amount (summed).
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
    pub(crate) storage: vk::PerformanceCounterStorageKHR,
}

static COUNTERS: OnceLock<Vec<Counter>> = OnceLock::new();
/// How many query pools use the profiling lock; it is held while there is one.
pub(crate) static LOCK_USERS: Mutex<usize> = Mutex::new(0);

fn to_string(chars: &[std::ffi::c_char]) -> String {
    unsafe { std::ffi::CStr::from_ptr(chars.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

pub(crate) fn init() -> Result<()> {
    if !gate::PerformanceQuery::on() {
        return Ok(());
    }
    let (instance, _) = Functions::performance_query();
    let (device, family) = (Ctx::physical_device().handel, Ctx::get().queues.graphics().index);
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
        .map(|(index, (counter, description))| {
            use vk::PerformanceCounterUnitKHR as U;
            let unit = match counter.unit {
                U::PERCENTAGE => CounterUnit::Percentage,
                U::NANOSECONDS => CounterUnit::Nanoseconds,
                U::BYTES => CounterUnit::Bytes,
                U::BYTES_PER_SECOND => CounterUnit::BytesPerSecond,
                U::KELVIN => CounterUnit::Kelvin,
                U::WATTS => CounterUnit::Watts,
                U::VOLTS => CounterUnit::Volts,
                U::AMPS => CounterUnit::Amps,
                U::HERTZ => CounterUnit::Hertz,
                U::CYCLES => CounterUnit::Cycles,
                _ => CounterUnit::Generic,
            };
            Counter {
                name: to_string(&description.name),
                category: to_string(&description.category),
                description: to_string(&description.description),
                unit,
                index: index as u32,
                storage: counter.storage,
            }
        })
        .collect();
    let _ = COUNTERS.set(counters);
    Ok(())
}

/// Every counter the graphics queue family offers.
pub fn counters() -> Gated<gate::PerformanceQuery, &'static [Counter]> {
    gate::<gate::PerformanceQuery, _>(|| COUNTERS.get().map_or(&[][..], Vec::as_slice))
}

/// How many submissions it takes to read `selection` (0 without performance queries). Only
/// selections of one pass can be recorded.
pub fn passes(selection: &[&Counter]) -> u32 {
    if !gate::PerformanceQuery::on() {
        return 0;
    }
    let (instance, _) = Functions::performance_query();
    let indices: Vec<u32> = selection.iter().map(|counter| counter.index).collect();
    let info = vk::QueryPoolPerformanceCreateInfoKHR::default()
        .queue_family_index(Ctx::get().queues.graphics().index)
        .counter_indices(&indices);
    unsafe {
        instance.get_physical_device_queue_family_performance_query_passes(
            Ctx::physical_device().handel,
            &info,
        )
    }
}

pub(crate) fn release_lock() {
    let mut users = LOCK_USERS.lock().unwrap();
    *users -= 1;
    if *users == 0 {
        let (_, device) = Functions::performance_query();
        unsafe { device.release_profiling_lock() };
    }
}

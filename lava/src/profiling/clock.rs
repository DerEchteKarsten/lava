//! Shader clock regions: the counters `CLOCK_REGION` in `profile.slang` adds subgroup clock ticks to, read per pass and region
use std::sync::{
    Mutex, OnceLock,
    atomic::{AtomicU64, Ordering},
};

use ash::vk;

use crate::{
    bindings::PASS_MAP, bindless::Bindless, buffer::Buffer, error::Result, slang_compile,
    state::Ctx,
};

/// Clock ticks a pass's region spent since the frame before. Ticks are the device's own unit,
/// about core clock cycles.
#[derive(Clone, Debug)]
pub struct RegionTime {
    pub pass: &'static str,
    pub region: &'static str,
    /// Summed over all subgroups that ran the region, which run in parallel, so it can exceed
    /// the frame.
    pub ticks: u64,
    pub subgroups: u64,
}

/// Mirrors `profile.slang`.
const FLAG_WORDS: usize = 8;
const MAX_REGIONS: usize = slang_compile::CLOCK_REGION_IDS.end as usize
    - slang_compile::CLOCK_REGION_IDS.start as usize;
const STRIPES: usize = 64;
const STRIPE_WORDS: usize = 8;
/// A counter holds the ticks in its low bits and the subgroups above them.
const TICK_BITS: u32 = 40;

/// The buffer `profile.slang` adds to, in descriptor set 2. Word 0 counts the profiling
/// `FrameQueries`; the shaders skip their atomics while it is 0.
static COUNTERS: OnceLock<Buffer<u64>> = OnceLock::new();
/// Per pass and region, the sum of its stripes' counters at the last read. Shared by all frame
/// slots, since the counters only grow.
static LAST_READ: Mutex<Vec<u64>> = Mutex::new(Vec::new());

fn counter(index: usize) -> &'static AtomicU64 {
    let counters = COUNTERS.get().expect("shader clocks are initialised");
    unsafe { &*(counters.range(..).element_ptr(index) as *const AtomicU64) }
}

/// Creates the counters and binds them, if the device has subgroup clocks.
pub(crate) fn init() -> Result<()> {
    if !Ctx::features().shader_clock {
        return Ok(());
    }
    let len = FLAG_WORDS + PASS_MAP.len() * MAX_REGIONS * STRIPES * STRIPE_WORDS;
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

/// Counts a profiled frame slot in or out; the shaders count only while one exists.
pub(crate) fn add_profiled_slots(delta: i64) {
    if COUNTERS.get().is_some() {
        counter(0).fetch_add(delta as u64, Ordering::Relaxed);
    }
}

/// The ticks every region spent since the last call, from any frame slot.
pub(crate) fn read() -> Vec<RegionTime> {
    if COUNTERS.get().is_none() {
        return Vec::new();
    }
    let mut last = LAST_READ.lock().unwrap();
    last.resize(PASS_MAP.len() * MAX_REGIONS, 0);
    let mut times = Vec::new();
    for (pass, entry) in PASS_MAP.iter().enumerate() {
        for (region, name) in entry.regions.iter().enumerate() {
            let slot = pass * MAX_REGIONS + region;
            let mut now = 0u64;
            for stripe in 0..STRIPES {
                let base = FLAG_WORDS + (slot * STRIPES + stripe) * STRIPE_WORDS;
                now = now.wrapping_add(counter(base).load(Ordering::Relaxed));
            }
            // The sums wrap, but the difference is exact while the ticks since the last read
            // fit `TICK_BITS` and the subgroups the bits above.
            let since = now.wrapping_sub(last[slot]);
            last[slot] = now;
            let (ticks, subgroups) = (since & ((1 << TICK_BITS) - 1), since >> TICK_BITS);
            if subgroups > 0 {
                times.push(RegionTime {
                    pass: entry.name,
                    region: name,
                    ticks,
                    subgroups,
                });
            }
        }
    }
    times
}

//! Shader clock regions: the switch that bakes them into clocked passes, and the counters `CLOCK_REGION` in `profile.slang` adds subgroup clock ticks to, read per pass and region
use std::sync::{
    Mutex, OnceLock,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

use ash::vk;

use crate::{
    bindings::PASS_MAP, bindless::Bindless, buffer::Buffer, error::Result, slang_compile,
    state::Ctx,
};

/// Clock ticks a pass's region spent since the last read. Ticks are the device's own unit,
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
const MAX_REGIONS: usize = slang_compile::CLOCK_REGION_IDS.end as usize
    - slang_compile::CLOCK_REGION_IDS.start as usize;
const STRIPES: usize = 64;
const STRIPE_WORDS: usize = 8;
/// A counter holds the ticks in its low bits and the subgroups above them.
const TICK_BITS: u32 = 40;

static ENABLED: AtomicBool = AtomicBool::new(false);
/// The buffer `profile.slang` adds to, in descriptor set 2.
static COUNTERS: OnceLock<Buffer<u64>> = OnceLock::new();
/// Per pass and region, the sum of its stripes' counters at the last read.
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
    let len = PASS_MAP.len() * MAX_REGIONS * STRIPES * STRIPE_WORDS;
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

/// Makes the passes recorded from now on count their regions, if the device has subgroup
/// clocks. Off, the generated `constant_bindings` bake no region names, so the driver drops
/// the regions; each state is its own pipeline.
pub fn set_enabled(enabled: bool) {
    ENABLED.store(enabled && COUNTERS.get().is_some(), Ordering::Relaxed);
}

pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Makes the counters readable by the host once the submission finished.
pub(crate) fn host_barrier(cmd: vk::CommandBuffer) {
    let barrier = [vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::HOST)
        .dst_access_mask(vk::AccessFlags2::HOST_READ)];
    let info = vk::DependencyInfo::default().memory_barriers(&barrier);
    unsafe { Ctx::device().cmd_pipeline_barrier2(cmd, &info) };
}

/// Calls `f` with the ticks of every region that ran since the last call.
pub fn read(mut f: impl FnMut(RegionTime)) {
    if COUNTERS.get().is_none() {
        return;
    }
    let mut last = LAST_READ.lock().unwrap();
    last.resize(PASS_MAP.len() * MAX_REGIONS, 0);
    for (pass, entry) in PASS_MAP.iter().enumerate() {
        for (region, name) in entry.regions.iter().enumerate() {
            let slot = pass * MAX_REGIONS + region;
            let mut now = 0u64;
            for stripe in 0..STRIPES {
                let base = (slot * STRIPES + stripe) * STRIPE_WORDS;
                now = now.wrapping_add(counter(base).load(Ordering::Relaxed));
            }
            // The sums wrap, but the difference is exact while the ticks since the last read
            // fit `TICK_BITS` and the subgroups the bits above.
            let since = now.wrapping_sub(last[slot]);
            last[slot] = now;
            let (ticks, subgroups) = (since & ((1 << TICK_BITS) - 1), since >> TICK_BITS);
            if subgroups > 0 {
                f(RegionTime {
                    pass: entry.name,
                    region: name,
                    ticks,
                    subgroups,
                });
            }
        }
    }
}

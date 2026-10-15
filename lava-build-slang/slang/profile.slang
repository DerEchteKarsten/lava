#pragma once
// Shader clock regions for lava's profiler (`lava/src/profiling/clock.rs`), a debugging aid:
// `CLOCK_REGION("name")` at the start of a block adds, per subgroup, the clock ticks until the
// block ends to that region's counter of the pass, while `clock::set_enabled(true)`. The
// profiler reports each region's ticks and subgroups. One region per block; at most 8 per pass.
// Not in fragment shaders: a fragment shader that writes a buffer loses early depth testing.
//
// Only a pass that `#define USE_CLOCKS` before its first include of this file counts; in every
// other pass `CLOCK_REGION` is empty, so shared code can keep its regions. lava-build-slang's Slang
// session defines `PROFILING` for lava with its `profiling` feature, and `PROFILE_PASS`, the
// pass's index in `PASS_MAP`, for every pass.

#ifdef USE_CLOCKS
// The hashes of the pass's region names in slot order, 0 for unused slots, and 0 for all of
// them while clocks are off (and without `PROFILING`), which folds every region away. Declared
// with `USE_CLOCKS` alone, so a pass's generated bindings don't depend on the feature:
// lava-build-slang bakes the hashes into its `constant_bindings`, which takes no parameters for
// them. The driver folds the comparisons in `clock_region_slot`.
[vk::constant_id(901)] const uint CLOCK_REGION_0 = 0;
[vk::constant_id(902)] const uint CLOCK_REGION_1 = 0;
[vk::constant_id(903)] const uint CLOCK_REGION_2 = 0;
[vk::constant_id(904)] const uint CLOCK_REGION_3 = 0;
[vk::constant_id(905)] const uint CLOCK_REGION_4 = 0;
[vk::constant_id(906)] const uint CLOCK_REGION_5 = 0;
[vk::constant_id(907)] const uint CLOCK_REGION_6 = 0;
[vk::constant_id(908)] const uint CLOCK_REGION_7 = 0;
#endif

#if defined(PROFILING) && defined(USE_CLOCKS)

// Per pass, region and stripe one counter: the summed ticks in its low `CLOCK_TICK_BITS` bits and the number of
// subgroups above them, so a subgroup adds both with one atomic. Mirrored by `clock.rs`.
[vk::binding(0, 2)] RWStructuredBuffer<uint64_t> ProfileCounters;

static const uint CLOCK_MAX_REGIONS = 8;
// Subgroups spread their atomics over this many copies of a counter, each in its own cache
// line (`CLOCK_STRIPE_WORDS`): the L2 serialises the atomics to a line, about 200 cycles
// each, so with 64 stripes a pass of 30k subgroups pays 0.1 ms per atomic per subgroup.
static const uint CLOCK_STRIPES = 64;
static const uint CLOCK_STRIPE_WORDS = 8;
static const uint CLOCK_TICK_BITS = 40;

uint clock_region_slot(uint hash)
{
    if (hash == CLOCK_REGION_0) return 0;
    if (hash == CLOCK_REGION_1) return 1;
    if (hash == CLOCK_REGION_2) return 2;
    if (hash == CLOCK_REGION_3) return 3;
    if (hash == CLOCK_REGION_4) return 4;
    if (hash == CLOCK_REGION_5) return 5;
    if (hash == CLOCK_REGION_6) return 6;
    if (hash == CLOCK_REGION_7) return 7;
    return CLOCK_MAX_REGIONS;
}

// Subgroup scope: only differences within one subgroup mean something.
uint64_t subgroup_clock()
{
    let clock = spirv_asm {
        OpCapability ShaderClockKHR;
        OpExtension "SPV_KHR_shader_clock";
        result : $$uint2 = OpReadClockKHR Subgroup
    };
    return (uint64_t(clock.y) << 32) | clock.x;
}

void clock_region_end(uint hash, uint64_t start)
{
    let region = clock_region_slot(hash);
    if (!WaveIsFirstLane() || region == CLOCK_MAX_REGIONS)
        return;
    let ticks = subgroup_clock() - start;
    let stripe = uint(start) % CLOCK_STRIPES;
    let slot = ((PROFILE_PASS * CLOCK_MAX_REGIONS + region) * CLOCK_STRIPES + stripe) * CLOCK_STRIPE_WORDS;
    InterlockedAdd(ProfileCounters[slot], ticks + (1ull << CLOCK_TICK_BITS));
}

#define CLOCK_REGION(name) let clock_region_start = subgroup_clock(); defer clock_region_end(uint(getStringHash(name)), clock_region_start);
#else
#define CLOCK_REGION(name)
#endif

#pragma once
// Shader clock instrumentation for lava's profiler (`lava/src/profiling.rs`). `PROFILE(stage)`
// at the top of an entry point adds, per subgroup, the clock ticks from there until the entry
// point returns to its pass's counters, while a profiled frame slot exists. Not in fragment
// shaders: a fragment shader that writes a buffer loses early depth testing, even while the
// profiler is off.
//
// Only a pass that `#define USE_CLOCKS` before its first include of this file gets the clocks
// (and their counter binding); in every other pass `PROFILE` is empty, so shared code can call
// it. lava's Slang session defines `PROFILING` with its `profiling` feature, and `PROFILE_PASS`,
// the pass's index in `PASS_MAP`, for every pass.
#if defined(PROFILING) && defined(USE_CLOCKS)

// `PROFILE_FLAG_WORDS` words of flags (word 0: profiled frame slots alive), then per pass, stage
// and stripe one counter: the summed ticks in its low `PROFILE_TICK_BITS` bits and the number
// of subgroups above them, so a subgroup adds both with one atomic. Mirrored by `profiling.rs`.
[vk::binding(0, 2)] RWStructuredBuffer<uint64_t> ProfileCounters;

static const uint PROFILE_FLAG_WORDS = 8;
static const uint PROFILE_STAGES = 4;
// Subgroups spread their atomics over this many copies of a counter, each in its own cache
// line (`PROFILE_STRIPE_WORDS`): the L2 serialises the atomics to a line, about 200 cycles
// each, so with 64 stripes a pass of 30k subgroups pays 0.1 ms per atomic per subgroup.
static const uint PROFILE_STRIPES = 64;
static const uint PROFILE_STRIPE_WORDS = 8;
static const uint PROFILE_TICK_BITS = 40;

static const uint PROFILE_COMPUTE = 0;
static const uint PROFILE_TASK = 1;
static const uint PROFILE_MESH = 2;
static const uint PROFILE_VERTEX = 3;

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

void profile_end(uint stage, uint64_t start)
{
    if (!WaveIsFirstLane() || ProfileCounters[0] == 0)
        return;
    let ticks = subgroup_clock() - start;
    let stripe = uint(start) % PROFILE_STRIPES;
    let slot = PROFILE_FLAG_WORDS + ((PROFILE_PASS * PROFILE_STAGES + stage) * PROFILE_STRIPES + stripe) * PROFILE_STRIPE_WORDS;
    InterlockedAdd(ProfileCounters[slot], ticks + (1ull << PROFILE_TICK_BITS));
}

#define PROFILE(stage) let profile_start = subgroup_clock(); defer profile_end(stage, profile_start);
#else
#define PROFILE(stage)
#endif

#pragma once
// Shader clock instrumentation for lava's profiler (`lava/src/profiling.rs`). `PROFILE(stage)`
// at the top of an entry point adds, per subgroup, the clock ticks from there until the entry
// point returns to its pass's counters, while a profiled frame slot exists. Not in fragment
// shaders: a fragment shader that writes a buffer loses early depth testing, even while the
// profiler is off.

// Set by lava for every pipeline: the pass's index in `PASS_MAP`.
[vk::constant_id(900)] const uint PROFILE_PASS = 0;

// `PROFILE_FLAG_WORDS` words of flags (word 0: profiled frame slots alive), then per pass, stage
// and stripe the summed ticks and the number of subgroups. Mirrored by `profiling.rs`.
[vk::binding(0, 2)] RWStructuredBuffer<uint64_t> ProfileCounters;

static const uint PROFILE_FLAG_WORDS = 8;
static const uint PROFILE_STAGES = 4;
// Subgroups spread their atomics over this many copies of a counter, each in its own cache
// line (`PROFILE_STRIPE_WORDS`), or contention makes them cost more than the shader.
static const uint PROFILE_STRIPES = 64;
static const uint PROFILE_STRIPE_WORDS = 8;

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
    InterlockedAdd(ProfileCounters[slot], ticks);
    InterlockedAdd(ProfileCounters[slot + 1], 1);
}

// lava defines `PROFILING` with its `profiling` feature; without it the clocks aren't compiled.
#ifdef PROFILING
#define PROFILE(stage) let profile_start = subgroup_clock(); defer profile_end(stage, profile_start);
#else
#define PROFILE(stage)
#endif

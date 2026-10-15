#pragma once
// Shader clock regions, a debugging aid: `CLOCK_REGION(clocks, field)` at the start of a block
// adds, per subgroup, the clock ticks until the block ends to the `Clock` `clocks->field`. The
// application declares a struct of `Clock`s, passes its address (`clocks`, a pointer in the
// push constants), and reads the fields on the host with the generated `Clock::take` (a struct
// whose fields are all `Clock`s also gets `regions()`, every field by name); a null `clocks`
// skips the region, which is the switch. One region per block. The macro takes the struct pointer rather than the field,
// because a field pointer derived from null isn't null.
// Not in fragment shaders: a fragment shader that writes a buffer loses early depth testing.
//
// Regions only count where `USE_CLOCKS` is defined (an application defines it for all its
// shaders); elsewhere `CLOCK_REGION` is empty, while `Clock` stays declared, so the generated
// bindings don't depend on it. Needs `VK_KHR_shader_clock`'s subgroup clock.

// Subgroups spread their atomics over this many copies of a counter, each in its own cache
// line (`CLOCK_STRIPE_WORDS`): the L2 serialises the atomics to a line, about 200 cycles
// each, so with 64 stripes a pass of 30k subgroups pays 0.1 ms per atomic per subgroup.
static const uint CLOCK_STRIPES = 64;
static const uint CLOCK_STRIPE_WORDS = 8;
// Each counter holds the summed ticks in its low `CLOCK_TICK_BITS` bits and the number of
// subgroups above them, so a subgroup adds both with one atomic.
static const uint CLOCK_TICK_BITS = 40;

// One region's counters: stripe `s` is `counters[s * CLOCK_STRIPE_WORDS]`.
struct Clock {
    uint64_t counters[CLOCK_STRIPES * CLOCK_STRIPE_WORDS];
}

#ifdef USE_CLOCKS
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

void clock_region_end(Clock* clock, uint64_t start)
{
    if (!WaveIsFirstLane())
        return;
    let ticks = subgroup_clock() - start;
    let stripe = uint(start) % CLOCK_STRIPES;
    InterlockedAdd(clock.counters[stripe * CLOCK_STRIPE_WORDS], ticks + (1ull << CLOCK_TICK_BITS));
}

#define CLOCK_REGION(clocks, field) \
    let clock_region_clocks = (clocks); \
    let clock_region_start = clock_region_clocks != nullptr ? subgroup_clock() : 0; \
    defer { if (clock_region_clocks != nullptr) clock_region_end(&clock_region_clocks.field, clock_region_start); }
#else
#define CLOCK_REGION(clocks, field)
#endif

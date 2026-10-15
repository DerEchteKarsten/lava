// Test pass (compute, clock regions): two regions, each around a write of out[i].
#include "bindless.slang"
#include "profile.slang"

struct TestClocks {
    Clock first;
    Clock second;
}

struct TestClockBindings {
    TestClocks* clocks;
    MutBuf<uint> out;
}

[[vk::push_constant]] TestClockBindings b;

[shader("compute")]
[numthreads(64, 1, 1)]
void test_clock(uint dtid: SV_DispatchThreadID) {
    {
        CLOCK_REGION(b.clocks, first);
        b.out.val[dtid] = dtid;
    }
    {
        CLOCK_REGION(b.clocks, second);
        b.out.val[dtid] += 1;
    }
}

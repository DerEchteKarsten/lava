// Test pass (compute, required subgroup size): `[WaveSize(32)]` writes each thread's subgroup size; devices that can't require 32 get no pipeline.
#include "bindless.slang"

struct TestWaveSizeBindings {
    MutBuf<uint> out;
}

[[vk::push_constant]] TestWaveSizeBindings b;

[shader("compute")]
[numthreads(64, 1, 1)]
[WaveSize(32)]
void test_wave_size(uint i: SV_DispatchThreadID) {
    b.out.val[i] = WaveGetLaneCount();
}

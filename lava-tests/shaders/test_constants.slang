// Test pass (compute, specialization constants): fills `target` with `brightness` in red and
// writes out[i] = offset + i * multiplier for the first `count` elements.
#include "bindless.slang"

SCONST(MutBuf<uint>, out)
SCONST(MutImage2D<float4>, target)
[SpecializationConstant] const uint multiplier = 1;
[SpecializationConstant] const float brightness = 1.0;

struct TestConstantsBindings {
    uint offset;
    uint count;
}

[[vk::push_constant]] TestConstantsBindings b;

[shader("compute")]
[numthreads(8, 8, 1)]
void test_constants(uint2 dtid: SV_DispatchThreadID) {
    target.Store(dtid, float4(brightness, 0.0, 0.0, 1.0));
    if (dtid.y == 0 && dtid.x < b.count) {
        out.val[dtid.x] = b.offset + dtid.x * multiplier;
    }
}

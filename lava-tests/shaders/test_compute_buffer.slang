// Test pass (compute, buffers): dst[i] = src[i] * multiplier + i for the first `count` elements.
#include "bindless.slang"

struct TestComputeBufferBindings {
    Buf<uint> src;
    MutBuf<uint> dst;
    uint count;
    uint multiplier;
}

[[vk::push_constant]] TestComputeBufferBindings b;

[shader("compute")]
[numthreads(64, 1, 1)]
void test_compute_buffer(uint dtid: SV_DispatchThreadID) {
    if (dtid >= b.count) {
        return;
    }
    b.dst.val[dtid] = b.src.val[dtid] * b.multiplier + dtid;
}

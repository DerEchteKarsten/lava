// Test pass (compute, storage image): writes a UV gradient with a checkerboard in blue.
#include "bindless.slang"

struct TestComputeImageBindings {
    MutImage2D<float4> target;
    float4 tint;
    // Top-left pixel and size of the rectangle that is written; pixels outside stay untouched.
    uint2 offset;
    uint2 size;
}

[[vk::push_constant]] TestComputeImageBindings b;

[shader("compute")]
[numthreads(8, 8, 1)]
void test_compute_image(uint2 dtid: SV_DispatchThreadID) {
    if (any(dtid >= b.size)) {
        return;
    }
    let uv = (float2(dtid) + 0.5) / float2(b.size);
    let checker = float(((dtid.x / 8) + (dtid.y / 8)) % 2);
    b.target.Store(dtid + b.offset, float4(uv, checker, 1.0) * b.tint);
}

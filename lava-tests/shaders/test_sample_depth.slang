// Test pass (compute, sampled depth): copies the depth of each texel of a depth image into a buffer, row by row.
#include "bindless.slang"

struct TestSampleDepthBindings {
    Tex2D<float> depth;
    MutBuf<float> out;
}

[[vk::push_constant]] TestSampleDepthBindings b;

[shader("compute")]
[numthreads(8, 8, 1)]
void test_sample_depth(uint2 dtid: SV_DispatchThreadID) {
    let size = b.depth.GetDimensions();
    if (any(dtid >= size)) {
        return;
    }
    b.out.val[dtid.y * size.x + dtid.x] = b.depth.Load(dtid);
}

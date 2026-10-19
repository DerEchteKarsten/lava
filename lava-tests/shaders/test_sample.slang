// Test pass (compute, samplers): samples a sampled slot through two samplers and a comparison sampler into a buffer.
#include "bindless.slang"

struct TestSampleBindings {
    MutBuf<float4> out;
    Sampler first;
    Sampler second;
    // Skipped where its index is `NULL_HANDLE`; the slot then needn't hold a depth image.
    Sampler compare;
    // A slot of the sampled set, so reserved slots can be read too.
    uint slot;
    float2 uv;
    float reference;
}

[[vk::push_constant]] TestSampleBindings b;

[shader("compute")]
[numthreads(1, 1, 1)]
void test_sample() {
    let color = Tex2D<float4>(b.slot, NULL_HANDLE);
    b.out.val[0] = color.SampleLevel(b.first, b.uv, 0.0);
    b.out.val[1] = color.SampleLevel(b.second, b.uv, 0.0);
    if (b.compare.index != NULL_HANDLE) {
        let depth = Tex2D<float>(b.slot, NULL_HANDLE);
        b.out.val[2] = float4(depth.SampleCmpLevelZero(b.compare, b.uv, b.reference));
    }
}

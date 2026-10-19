// Test pass (compute, image handles): doubles a combined image in place, triples one at a reserved pair of slots, samples three `DynImg`s (sampled, storage, null) and reads each mip of a combined mip chain through `Img.Mip`.
#include "bindless.slang"

struct TestCombinedBindings {
    // Read through its sampled descriptor, written through its storage one.
    MutTex2D<float4> target;
    // A combined handle of a reserved pair of slots as a plain `uint`, or `NULL_HANDLE`.
    uint pair;
    // Mip 0 of a chain bound with `bind_mips_combined`.
    Tex2D<float4> mips;
    DynImg<float4> sampled;
    DynImg<float4> storage;
    DynImg<float4> none;
    Sampler sampler;
    // The three samples, then texel (0, 0) of each mip.
    MutBuf<float4> out;
    uint levels;
}

[[vk::push_constant]] TestCombinedBindings b;

[shader("compute")]
[numthreads(4, 4, 1)]
void test_combined(uint2 dtid: SV_DispatchThreadID) {
    b.target.Store(dtid, b.target.Load(dtid) * 2.0);
    if (b.pair != NULL_HANDLE) {
        let pair = MutTex2D<float4>(b.pair);
        pair.Store(dtid, pair.Load(dtid) * 3.0);
    }
    if (any(dtid != 0)) {
        return;
    }
    b.out.val[0] = b.sampled.SampleLevel(b.sampler, float2(0.5), 0.0);
    b.out.val[1] = b.storage.SampleLevel(b.sampler, float2(0.5), 0.0);
    b.out.val[2] = b.none.SampleLevel(b.sampler, float2(0.5), 0.0);
    for (uint level = 0; level < b.levels; level++) {
        b.out.val[3 + level] = b.mips.Mip(level).Load(uint2(0));
    }
}

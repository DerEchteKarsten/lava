// Test pass (compute, image operations): gathers, explicit-gradient samples and mip loads of a sampled mip chain, level samples of a combined 2D and 3D image, and every atomic on `uint` and `int` storage images, into buffers.
#include "bindless.slang"

struct TestImageOpsBindings {
    // A 4x4 chain of 3 mips, each mip one colour but mip 0, whose texels differ.
    Tex2D<float4> chain;
    // Read through its sampled descriptor.
    MutTex2D<float4> combined;
    // 1x1, `R32Uint` and `R32Sint`, cleared to zero.
    MutImage2D<uint> counters;
    // 2x2x2: texel (1, 1, 1) is written with what a sample of texel (0, 0, 0) reads.
    MutTex3D<float4> volume;
    MutImage2D<int> signed;
    Sampler nearest;
    // Gather, GatherRed, GatherGreen, GatherBlue, GatherAlpha, SampleGrad, Load at mip 1,
    // combined.SampleLevel.
    MutBuf<float4> out;
    // The atomics' results, then the final texels.
    MutBuf<int> atomics;
}

[[vk::push_constant]] TestImageOpsBindings b;

[shader("compute")]
[numthreads(1, 1, 1)]
void test_image_ops() {
    // Between texels (0, 0), (1, 0), (0, 1) and (1, 1) of mip 0.
    let uv = float2(0.25, 0.25);
    b.out.val[0] = b.chain.Gather(b.nearest, uv);
    b.out.val[1] = b.chain.GatherRed(b.nearest, uv);
    b.out.val[2] = b.chain.GatherGreen(b.nearest, uv);
    b.out.val[3] = b.chain.GatherBlue(b.nearest, uv);
    b.out.val[4] = b.chain.GatherAlpha(b.nearest, uv);
    // Derivatives of a whole texture pick the last mip.
    b.out.val[5] = b.chain.SampleGrad(b.nearest, float2(0.5), float2(1.0, 0.0), float2(0.0, 1.0));
    b.out.val[6] = b.chain.Load(uint2(0), 1);
    b.out.val[7] = b.combined.SampleLevel(b.nearest, float2(0.5), 0.0);
    b.volume.Store(uint3(1), b.volume.SampleLevel(b.nearest, float3(0.25), 0.0));

    let p = uint2(0);
    b.atomics.val[0] = int(b.counters.InterlockedAdd(p, 5));
    b.atomics.val[1] = int(b.counters.InterlockedMax(p, 9));
    b.atomics.val[2] = int(b.counters.InterlockedMin(p, 7));
    b.atomics.val[3] = int(b.counters.InterlockedOr(p, 8));
    b.atomics.val[4] = int(b.counters.InterlockedAnd(p, 6));
    b.atomics.val[5] = int(b.counters.InterlockedXor(p, 3));
    b.atomics.val[6] = int(b.counters.InterlockedExchange(p, 42));
    b.atomics.val[7] = int(b.counters.InterlockedCompareExchange(p, 42, 100));
    b.atomics.val[8] = int(b.counters.InterlockedCompareExchange(p, 1, 200));
    b.atomics.val[9] = int(b.counters.Load(p));
    b.atomics.val[10] = b.signed.InterlockedMin(p, -4);
    b.atomics.val[11] = b.signed.InterlockedMax(p, -9);
    b.atomics.val[12] = b.signed.InterlockedAdd(p, -1);
    b.atomics.val[13] = b.signed.Load(p);
}

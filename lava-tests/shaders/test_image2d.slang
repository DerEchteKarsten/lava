// Test pass (compute, Slang image shapes): reads a read-only storage image (`Image2D`) through its storage descriptor, and a storage and a sampled image through handles inside a struct behind a `Buf`, writing every texel it read to a buffer.
#include "bindless.slang"

struct TestImagePair {
    Image2D<float4> storage;
    Tex2D<float4> sampled;
}

struct TestImage2dBindings {
    Image2D<float4> image;
    Buf<TestImagePair> images;
    Sampler sampler;
    MutBuf<float4> out;
}

[[vk::push_constant]] TestImage2dBindings b;

[shader("compute")]
[numthreads(4, 1, 1)]
void test_image2d(uint i: SV_DispatchThreadID) {
    let p = uint2(i % 2, i / 2);
    let images = b.images.val[0];
    b.out.val[i] = b.image.Load(p);
    b.out.val[4 + i] = images.storage.Load(p);
    b.out.val[8 + i] = images.sampled.SampleLevel(b.sampler, (float2(p) + 0.5) / 2.0, 0.0);
}

// Test pass (compute): samples the centre of every layer of a 2D array image into a buffer.
#include "bindless.slang"

struct TestSampleArrayBindings {
    Tex2DArray<float4> image;
    Sampler sampler;
    MutBuf<float4> texels;
    uint layers;
}

[[vk::push_constant]] TestSampleArrayBindings b;

[shader("compute")]
[numthreads(1, 1, 1)]
void test_sample_array() {
    let size = b.image.GetDimensions();
    for (uint layer = 0; layer < b.layers; layer++) {
        b.texels.val[layer] = b.image.SampleLevel(b.sampler, float3(0.5, 0.5, float(layer)), 0.0);
    }
    // The layer count the descriptor shows, after the samples.
    b.texels.val[b.layers] = float4(float(size.z), 0.0, 0.0, 0.0);
}

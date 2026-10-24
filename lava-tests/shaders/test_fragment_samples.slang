// Test pass (vertex + fragment, implicit-derivative samples): a full-target triangle that samples a mip chain with a bias (each pixel shows the mip the bias picks), or a combined image with `Sample` for a negative bias.
#include "bindless.slang"

struct TestFragmentSamplesBindings {
    Tex2D<float4> chain;
    MutTex2D<float4> combined;
    Sampler sampler;
    float bias;
}

[[vk::push_constant]] TestFragmentSamplesBindings b;

struct VertexOut {
    float4 position : SV_Position;
    float2 uv;
}

[shader("vertex")]
VertexOut test_fragment_samples_vertex(uint vertex : SV_VertexID) {
    let uv = float2((vertex << 1) & 2, vertex & 2);
    VertexOut out;
    out.position = float4(uv * 2.0 - 1.0, 0.5, 1.0);
    out.uv = uv;
    return out;
}

[shader("fragment")]
float4 test_fragment_samples_fragment(VertexOut in) : SV_Target {
    if (b.bias < 0.0) {
        return b.combined.Sample(b.sampler, in.uv);
    }
    return b.chain.SampleBias(b.sampler, in.uv, b.bias);
}

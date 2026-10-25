// Bloom (compute): one upsample step, the smaller mip filtered and added onto the larger.
#include "common.slang"

struct BloomUpBindings {
    Tex2D<float4> src;
    MutImage2D<float4> dst;
    Sampler sampler;
    uint2 size;
}

[[vk::push_constant]] BloomUpBindings b;

[shader("compute")]
[numthreads(8, 8, 1)]
void bloom_up(uint2 id: SV_DispatchThreadID) {
    if (any(id >= b.size)) {
        return;
    }
    let uv = (float2(id) + 0.5) / float2(b.size);
    b.dst.Store(id, b.dst.Load(id) + b.src.SampleLevel(b.sampler, uv, 0.0) * 0.6);
}

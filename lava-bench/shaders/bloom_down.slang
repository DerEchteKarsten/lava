// Bloom (compute): one downsample step, four bilinear taps of the larger mip into the next.
#include "common.slang"

struct BloomDownBindings {
    Tex2D<float4> src;
    MutImage2D<float4> dst;
    Sampler sampler;
    uint2 size;
}

[[vk::push_constant]] BloomDownBindings b;

[shader("compute")]
[numthreads(8, 8, 1)]
void bloom_down(uint2 id: SV_DispatchThreadID) {
    if (any(id >= b.size)) {
        return;
    }
    let texel = 1.0 / float2(b.size);
    let uv = (float2(id) + 0.5) * texel;
    let o = texel * 0.25;
    let sum = b.src.SampleLevel(b.sampler, uv + float2(-o.x, -o.y), 0.0)
        + b.src.SampleLevel(b.sampler, uv + float2(o.x, -o.y), 0.0)
        + b.src.SampleLevel(b.sampler, uv + float2(-o.x, o.y), 0.0)
        + b.src.SampleLevel(b.sampler, uv + float2(o.x, o.y), 0.0);
    b.dst.Store(id, sum * 0.25);
}

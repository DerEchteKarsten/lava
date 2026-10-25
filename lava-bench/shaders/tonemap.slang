// Tonemap (compute): ACES-fitted HDR to an LDR storage image.
#include "common.slang"

struct TonemapBindings {
    Tex2D<float4> hdr;
    MutImage2D<float4> ldr;
    uint2 size;
}

[[vk::push_constant]] TonemapBindings b;

float3 aces(float3 x) {
    return saturate((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void tonemap(uint2 id: SV_DispatchThreadID) {
    if (any(id >= b.size)) {
        return;
    }
    b.ldr.Store(id, float4(aces(b.hdr.Load(id).rgb), 1.0));
}

// Image processing scenario, histogram (compute): adds the luminance of a band of rows to 256 bins with atomics, through a raw pointer, so dispatches over different bands commute.
#include "common.slang"

struct HistogramBindings {
    Tex2D<float4> src;
    uint* bins;
    uint2 size;
    uint first_row;
}

[[vk::push_constant]] HistogramBindings b;

groupshared uint local_bins[256];

/// Each workgroup of 256 counts into group-shared bins, then adds them to the global ones.
[shader("compute")]
[numthreads(16, 16, 1)]
void histogram(uint2 id: SV_DispatchThreadID, uint local: SV_GroupIndex) {
    local_bins[local] = 0;
    GroupMemoryBarrierWithGroupSync();
    let p = uint2(id.x, id.y + b.first_row);
    if (all(p < b.size)) {
        let luma = dot(b.src.Load(p).rgb, float3(0.2126, 0.7152, 0.0722));
        InterlockedAdd(local_bins[min(uint(luma * 255.0), 255)], 1);
    }
    GroupMemoryBarrierWithGroupSync();
    if (local_bins[local] != 0) {
        InterlockedAdd(b.bins[local], local_bins[local]);
    }
}

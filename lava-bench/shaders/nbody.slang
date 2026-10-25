// Async compute scenario, simulation (compute): one step of an all-pairs gravitational n-body system, tiled through group-shared memory.
#include "common.slang"

struct NbodyBindings {
    Buf<float4> positions;
    Buf<float4> velocities;
    MutBuf<float4> positions_out;
    MutBuf<float4> velocities_out;
    uint count;
    float dt;
}

[[vk::push_constant]] NbodyBindings b;

groupshared float4 tile[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void nbody(uint id: SV_DispatchThreadID, uint local: SV_GroupIndex) {
    let p = b.positions.val[min(id, b.count - 1)];
    float3 acceleration = 0.0;
    for (uint start = 0; start < b.count; start += 256) {
        tile[local] = start + local < b.count ? b.positions.val[start + local] : float4(0.0);
        GroupMemoryBarrierWithGroupSync();
        for (uint j = 0; j < 256; j++) {
            let d = tile[j].xyz - p.xyz;
            let inv = rsqrt(dot(d, d) + 0.01);
            acceleration += d * (tile[j].w * inv * inv * inv);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (id >= b.count) {
        return;
    }
    let v = b.velocities.val[id].xyz + acceleration * b.dt;
    b.velocities_out.val[id] = float4(v, 0.0);
    b.positions_out.val[id] = float4(p.xyz + v * b.dt, p.w);
}

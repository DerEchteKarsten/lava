// Fluid scenario, advection (compute): moves the velocity field (xyz, dye in w) along itself, semi-Lagrangian, and injects a jet.
#include "common.slang"

struct FluidAdvectBindings {
    Tex3D<float4> velocity;
    MutImage3D<float4> velocity_out;
    Sampler sampler;
    uint3 size;
    float dt;
    float time;
}

[[vk::push_constant]] FluidAdvectBindings b;

[shader("compute")]
[numthreads(4, 4, 4)]
void fluid_advect(uint3 id: SV_DispatchThreadID) {
    if (any(id >= b.size)) {
        return;
    }
    let size = float3(b.size);
    let uvw = (float3(id) + 0.5) / size;
    let v = b.velocity.Load(id);
    var moved = b.velocity.SampleLevel(b.sampler, uvw - v.xyz * b.dt / size, 0.0) * 0.999;
    let jet = float3(0.5 + 0.2 * sin(b.time), 0.1, 0.5) * size;
    if (distance(float3(id), jet) < 4.0) {
        moved = float4(0.0, 20.0, 5.0 * cos(b.time), 1.0);
    }
    b.velocity_out.Store(id, moved);
}

// Fluid scenario, divergence (compute): central differences of the velocity field.
#include "common.slang"

struct FluidDivergenceBindings {
    Tex3D<float4> velocity;
    MutImage3D<float> divergence;
    uint3 size;
}

[[vk::push_constant]] FluidDivergenceBindings b;

float3 at(int3 p) {
    return b.velocity.Load(uint3(clamp(p, int3(0), int3(b.size) - 1))).xyz;
}

[shader("compute")]
[numthreads(4, 4, 4)]
void fluid_divergence(uint3 id: SV_DispatchThreadID) {
    if (any(id >= b.size)) {
        return;
    }
    let p = int3(id);
    let d = (at(p + int3(1, 0, 0)).x - at(p - int3(1, 0, 0)).x
        + at(p + int3(0, 1, 0)).y - at(p - int3(0, 1, 0)).y
        + at(p + int3(0, 0, 1)).z - at(p - int3(0, 0, 1)).z) * 0.5;
    b.divergence.Store(id, d);
}

// Fluid scenario, projection (compute): subtracts the pressure gradient, making the velocity field divergence-free.
#include "common.slang"

struct FluidProjectBindings {
    Tex3D<float4> velocity;
    Tex3D<float> pressure;
    MutImage3D<float4> velocity_out;
    uint3 size;
}

[[vk::push_constant]] FluidProjectBindings b;

float at(int3 p) {
    return b.pressure.Load(uint3(clamp(p, int3(0), int3(b.size) - 1)));
}

[shader("compute")]
[numthreads(4, 4, 4)]
void fluid_project(uint3 id: SV_DispatchThreadID) {
    if (any(id >= b.size)) {
        return;
    }
    let p = int3(id);
    let gradient = float3(at(p + int3(1, 0, 0)) - at(p - int3(1, 0, 0)),
        at(p + int3(0, 1, 0)) - at(p - int3(0, 1, 0)),
        at(p + int3(0, 0, 1)) - at(p - int3(0, 0, 1))) * 0.5;
    let v = b.velocity.Load(id);
    b.velocity_out.Store(id, float4(v.xyz - gradient, v.w));
}

// Fluid scenario, pressure solve (compute): one Jacobi iteration of the pressure Poisson equation.
#include "common.slang"

struct FluidJacobiBindings {
    Tex3D<float> pressure;
    Tex3D<float> divergence;
    MutImage3D<float> pressure_out;
    uint3 size;
}

[[vk::push_constant]] FluidJacobiBindings b;

float at(int3 p) {
    return b.pressure.Load(uint3(clamp(p, int3(0), int3(b.size) - 1)));
}

[shader("compute")]
[numthreads(4, 4, 4)]
void fluid_jacobi(uint3 id: SV_DispatchThreadID) {
    if (any(id >= b.size)) {
        return;
    }
    let p = int3(id);
    let sum = at(p + int3(1, 0, 0)) + at(p - int3(1, 0, 0)) + at(p + int3(0, 1, 0))
        + at(p - int3(0, 1, 0)) + at(p + int3(0, 0, 1)) + at(p - int3(0, 0, 1));
    b.pressure_out.Store(id, (sum - b.divergence.Load(id)) / 6.0);
}

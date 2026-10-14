// Test pass (compute, 3D sampled image): copies one depth slice of a volume into a storage image.
#include "bindless.slang"

struct TestComputeVolumeBindings {
    Tex3D<float4> volume;
    MutImage2D<float4> target;
    uint slice;
}

[[vk::push_constant]] TestComputeVolumeBindings b;

[shader("compute")]
[numthreads(8, 8, 1)]
void test_compute_volume(uint2 dtid: SV_DispatchThreadID) {
    if (any(dtid >= b.volume.GetDimensions().xy)) {
        return;
    }
    b.target.Store(dtid, b.volume.Load(uint3(dtid, b.slice)));
}

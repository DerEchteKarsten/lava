// Ray tracing scenario, skinning (compute): deforms one mesh's rest vertices into the positions its BLAS is refitted from.
#include "common.slang"

struct SkinBindings {
    Buf<MeshVertex> rest;
    MutBuf<float3> positions;
    uint count;
    uint mesh;
    float time;
}

[[vk::push_constant]] SkinBindings b;

[shader("compute")]
[numthreads(64, 1, 1)]
void skin(uint id: SV_DispatchThreadID) {
    if (id >= b.count) {
        return;
    }
    let p = b.rest.val[id].position;
    let wobble = 1.0 + 0.15 * sin(b.time * 3.0 + p.y * 4.0 + float(b.mesh));
    b.positions.val[id] = float3(p.x * wobble, p.y, p.z * wobble);
}

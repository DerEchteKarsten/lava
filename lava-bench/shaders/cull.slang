// GPU-driven scenario, culling (compute): frustum-tests instance spheres and appends an indexed indirect draw per visible instance.
#include "common.slang"

struct DrawIndexedCommand {
    uint index_count;
    uint instance_count;
    uint first_index;
    int vertex_offset;
    uint first_instance;
}

struct CullBindings {
    float4 planes[6];
    Buf<CullInstance> instances;
    MutBuf<DrawIndexedCommand> draws;
    MutBuf<uint> count;
    uint instance_count;
    uint index_count;
}

[[vk::push_constant]] CullBindings b;

[shader("compute")]
[numthreads(64, 1, 1)]
void cull(uint id: SV_DispatchThreadID) {
    if (id >= b.instance_count) {
        return;
    }
    let sphere = b.instances.val[id].sphere;
    for (uint i = 0; i < 6; i++) {
        if (dot(b.planes[i].xyz, sphere.xyz) + b.planes[i].w < -sphere.w) {
            return;
        }
    }
    uint slot;
    InterlockedAdd(b.count.val[0], 1, slot);
    DrawIndexedCommand draw;
    draw.index_count = b.index_count;
    draw.instance_count = 1;
    draw.first_index = 0;
    draw.vertex_offset = 0;
    draw.first_instance = id;
    b.draws.val[slot] = draw;
}

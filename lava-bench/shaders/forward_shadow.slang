// Forward scenario, shadow map (vertex only): the depth of one mesh instance from the light.
#include "common.slang"

struct ForwardShadowBindings {
    float4x4 light_view_proj;
    Buf<MeshVertex> vertices;
    Buf<DrawInstance> instances;
    uint instance;
}

[[vk::push_constant]] ForwardShadowBindings b;

[shader("vertex")]
float4 forward_shadow_vertex(uint index: SV_VertexID) : SV_Position {
    let position = b.vertices.val[index].position;
    let model = b.instances.val[b.instance].model;
    return mul(b.light_view_proj, mul(model, float4(position, 1.0)));
}

// GPU-driven scenario, draw (vertex + fragment): the instance a culled indirect draw names (its first instance).
#include "common.slang"

struct DrivenDrawBindings {
    float4x4 view_proj;
    Buf<MeshVertex> vertices;
    Buf<CullInstance> instances;
}

[[vk::push_constant]] DrivenDrawBindings b;

struct DrivenVSOutput {
    float4 position : SV_Position;
    float3 normal;
    nointerpolation uint instance;
};

[shader("vertex")]
DrivenVSOutput driven_draw_vertex(uint index: SV_VertexID, uint instance: SV_VulkanInstanceID) {
    let vertex = b.vertices.val[index];
    let sphere = b.instances.val[instance].sphere;
    DrivenVSOutput output;
    output.position = mul(b.view_proj, float4(sphere.xyz + vertex.position * sphere.w, 1.0));
    output.normal = vertex.normal;
    output.instance = instance;
    return output;
}

[shader("fragment")]
float4 driven_draw_fragment(in DrivenVSOutput input) : SV_Target {
    let tint = float3(hash01(input.instance), hash01(input.instance + 1), hash01(input.instance + 2));
    return float4(tint * (0.3 + 0.7 * saturate(dot(input.normal, normalize(float3(0.3, 1.0, 0.2))))), 1.0);
}

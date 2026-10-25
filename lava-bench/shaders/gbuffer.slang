// Deferred scenario, G-buffer (vertex + fragment): one mesh instance into three colour targets (albedo, normal, material) and depth.
#include "common.slang"

struct GbufferBindings {
    float4x4 view_proj;
    Buf<MeshVertex> vertices;
    Buf<DrawInstance> instances;
    uint instance;
}

[[vk::push_constant]] GbufferBindings b;

struct GbufferVSOutput {
    float4 position : SV_Position;
    float3 normal;
    float2 uv;
    float4 color;
};

[shader("vertex")]
GbufferVSOutput gbuffer_vertex(uint index: SV_VertexID) {
    let vertex = b.vertices.val[index];
    let instance = b.instances.val[b.instance];
    GbufferVSOutput output;
    output.position = mul(b.view_proj, mul(instance.model, float4(vertex.position, 1.0)));
    output.normal = mul(instance.model, float4(vertex.normal, 0.0)).xyz;
    output.uv = vertex.uv;
    output.color = instance.color;
    return output;
}

struct GbufferOutput {
    [[vk::location(0)]] float4 albedo : SV_Target0;
    [[vk::location(1)]] float4 normal : SV_Target1;
    [[vk::location(2)]] float4 material : SV_Target2;
};

[shader("fragment")]
GbufferOutput gbuffer_fragment(in GbufferVSOutput input) {
    GbufferOutput output;
    let checker = (uint(input.uv.x * 16.0) + uint(input.uv.y * 8.0)) & 1;
    output.albedo = input.color * (checker == 0 ? 1.0 : 0.6);
    output.normal = float4(normalize(input.normal), 0.0);
    output.material = float4(input.uv, 0.5, 1.0);
    return output;
}

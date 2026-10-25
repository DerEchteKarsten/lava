// Forward scenario, main pass (vertex + fragment): one textured, shadowed mesh instance, its material read through a referencing buffer of textures and samplers.
#include "common.slang"

struct Material {
    Tex2D<float4> albedo;
    Sampler sampler;
    float4 tint;
}

struct ForwardMainBindings {
    float4x4 view_proj;
    float4x4 light_view_proj;
    Buf<MeshVertex> vertices;
    Buf<DrawInstance> instances;
    Buf<Material> materials;
    Tex2D<float> shadow;
    Sampler shadow_sampler;
    uint instance;
    uint material;
}

[[vk::push_constant]] ForwardMainBindings b;

struct ForwardVSOutput {
    float4 position : SV_Position;
    float3 normal;
    float2 uv;
    float4 light;
    float4 color;
};

[shader("vertex")]
ForwardVSOutput forward_main_vertex(uint index: SV_VertexID) {
    let vertex = b.vertices.val[index];
    let instance = b.instances.val[b.instance];
    let world = mul(instance.model, float4(vertex.position, 1.0));
    ForwardVSOutput output;
    output.position = mul(b.view_proj, world);
    output.normal = mul(instance.model, float4(vertex.normal, 0.0)).xyz;
    output.uv = vertex.uv;
    output.light = mul(b.light_view_proj, world);
    output.color = instance.color;
    return output;
}

[shader("fragment")]
float4 forward_main_fragment(in ForwardVSOutput input) : SV_Target {
    let material = b.materials.val[b.material];
    let albedo = material.albedo.Sample(material.sampler, input.uv) * material.tint * input.color;
    let light = input.light.xyz / input.light.w;
    let uv = light.xy * 0.5 + 0.5;
    let stored = b.shadow.SampleLevel(b.shadow_sampler, uv, 0.0);
    // Reverse depth: nearer is larger, so a stored depth above ours is a closer occluder.
    let lit = stored > light.z + 0.002 ? 0.2 : 1.0;
    let n_dot_l = saturate(dot(normalize(input.normal), normalize(float3(0.4, 1.0, 0.3))));
    return float4(albedo.rgb * (0.1 + lit * n_dot_l * 2.0), 1.0);
}

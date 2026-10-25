// Deferred scenario, lighting (compute): point lights over the G-buffer into an HDR storage image.
#include "common.slang"

struct PointLight {
    float4 position_radius;
    float4 color;
}

struct LightingBindings {
    float4x4 inv_view_proj;
    Tex2D<float4> albedo;
    Tex2D<float4> normal;
    Tex2D<float4> material;
    Tex2D<float> depth;
    MutImage2D<float4> hdr;
    uint2 size;
    Buf<PointLight> lights;
    uint light_count;
}

[[vk::push_constant]] LightingBindings b;

[shader("compute")]
[numthreads(8, 8, 1)]
void lighting(uint2 id: SV_DispatchThreadID) {
    if (any(id >= b.size)) {
        return;
    }
    let depth = b.depth.Load(id);
    if (depth == 0.0) {
        b.hdr.Store(id, float4(0.02, 0.02, 0.03, 1.0));
        return;
    }
    let ndc = float2(id + 0.5) / float2(b.size) * 2.0 - 1.0;
    let world = mul(b.inv_view_proj, float4(ndc, depth, 1.0));
    let position = world.xyz / world.w;
    let albedo = b.albedo.Load(id).rgb;
    let normal = b.normal.Load(id).xyz;
    let material = b.material.Load(id);
    var color = albedo * 0.05;
    for (uint i = 0; i < b.light_count; i++) {
        let light = b.lights.val[i];
        let to_light = light.position_radius.xyz - position;
        let distance = length(to_light);
        let falloff = saturate(1.0 - distance / light.position_radius.w);
        color += albedo * light.color.rgb * saturate(dot(normal, to_light / distance)) * falloff * falloff * (0.5 + material.z);
    }
    b.hdr.Store(id, float4(color, 1.0));
}

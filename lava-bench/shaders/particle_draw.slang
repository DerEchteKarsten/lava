// Particles scenario, draw (vertex + fragment): additive camera-facing quads of the live particles in sorted order, an instance each.
#include "common.slang"

struct Particle {
    float3 position;
    float life;
    float3 velocity;
    float seed;
}

struct ParticleDrawBindings {
    float4x4 view_proj;
    Buf<Particle> particles;
    Buf<uint> order;
    float size;
}

[[vk::push_constant]] ParticleDrawBindings b;

struct ParticleVSOutput {
    float4 position : SV_Position;
    float2 corner;
    float4 color;
};

[shader("vertex")]
ParticleVSOutput particle_draw_vertex(uint vertex: SV_VertexID, uint instance: SV_InstanceID) {
    let p = b.particles.val[b.order.val[instance]];
    let corners = float2[6](float2(-1, -1), float2(1, -1), float2(1, 1), float2(-1, -1), float2(1, 1), float2(-1, 1));
    let corner = corners[vertex];
    var clip = mul(b.view_proj, float4(p.position, 1.0));
    clip.xy += corner * b.size * float2(0.5625, 1.0);
    ParticleVSOutput output;
    output.position = clip;
    output.corner = corner;
    output.color = float4(1.0, 0.5 + p.seed * 0.5, 0.2, 1.0) * saturate(p.life) * 0.05;
    return output;
}

[shader("fragment")]
float4 particle_draw_fragment(in ParticleVSOutput input) : SV_Target {
    let falloff = saturate(1.0 - dot(input.corner, input.corner));
    return input.color * falloff;
}

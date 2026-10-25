// Async compute scenario, draw (vertex + fragment): additive camera-facing quads of the bodies, an instance each.
#include "common.slang"

struct NbodyDrawBindings {
    float4x4 view_proj;
    Buf<float4> positions;
    float size;
}

[[vk::push_constant]] NbodyDrawBindings b;

struct NbodyVSOutput {
    float4 position : SV_Position;
    float2 corner;
};

[shader("vertex")]
NbodyVSOutput nbody_draw_vertex(uint vertex: SV_VertexID, uint instance: SV_InstanceID) {
    let corners = float2[6](float2(-1, -1), float2(1, -1), float2(1, 1), float2(-1, -1), float2(1, 1), float2(-1, 1));
    let corner = corners[vertex];
    var clip = mul(b.view_proj, float4(b.positions.val[instance].xyz, 1.0));
    clip.xy += corner * b.size * float2(0.5625, 1.0);
    NbodyVSOutput output;
    output.position = clip;
    output.corner = corner;
    return output;
}

[shader("fragment")]
float4 nbody_draw_fragment(in NbodyVSOutput input) : SV_Target {
    return float4(0.3, 0.5, 1.0, 1.0) * saturate(1.0 - dot(input.corner, input.corner)) * 0.2;
}

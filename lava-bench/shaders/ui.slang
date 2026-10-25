// UI (vertex + fragment): alpha-blended textured quads, one instance per quad.
#include "common.slang"

struct UiQuad {
    float2 position;
    float2 size;
    float4 color;
    float4 uv_rect;
}

struct UiBindings {
    Buf<UiQuad> quads;
    Tex2D<float4> atlas;
    Sampler sampler;
    uint first;
}

[[vk::push_constant]] UiBindings b;

struct UiVSOutput {
    float4 position : SV_Position;
    float2 uv;
    float4 color;
};

[shader("vertex")]
UiVSOutput ui_vertex(uint vertex: SV_VertexID, uint instance: SV_InstanceID) {
    let quad = b.quads.val[b.first + instance];
    // Two clockwise triangles (lava's front faces) of the unit square.
    let corners = float2[6](float2(0, 0), float2(1, 0), float2(1, 1), float2(0, 0), float2(1, 1), float2(0, 1));
    let corner = corners[vertex];
    UiVSOutput output;
    output.position = float4((quad.position + corner * quad.size) * 2.0 - 1.0, 0.0, 1.0);
    output.uv = lerp(quad.uv_rect.xy, quad.uv_rect.zw, corner);
    output.color = quad.color;
    return output;
}

[shader("fragment")]
float4 ui_fragment(in UiVSOutput input) : SV_Target {
    return b.atlas.Sample(b.sampler, input.uv) * input.color;
}

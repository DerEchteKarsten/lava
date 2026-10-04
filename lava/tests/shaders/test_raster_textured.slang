// Test pass (vertex + fragment, sampled image): textured triangles, meant for indexed draws.
#include "bindless.slang"

struct TestTexturedVertex {
    float2 position;
    float2 uv;
}

struct TestRasterTexturedBindings {
    Buf<TestTexturedVertex> vertices;
    Tex2D<float4> tex;
    // Index into lava's immutable sampler heap.
    uint sampler_index;
}

[[vk::push_constant]] TestRasterTexturedBindings b;

struct TestTexturedVSOutput {
    float4 position : SV_Position;
    float2 uv;
};

[shader("vertex")]
TestTexturedVSOutput test_raster_textured_vertex(uint index: SV_VertexID) {
    let vertex = b.vertices.val[index];
    TestTexturedVSOutput output;
    output.position = float4(vertex.position, 0.5, 1.0);
    output.uv = vertex.uv;
    return output;
}

[shader("fragment")]
float4 test_raster_textured_fragment(in TestTexturedVSOutput input) : SV_Target {
    return float4(b.tex.Sample(input.uv, b.sampler_index).rgb, 1.0);
}

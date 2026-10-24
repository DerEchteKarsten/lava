// Test pass (vertex + fragment, specialization constants): coloured triangles pulled from a vertex buffer, their colour scaled by `brightness` and one channel dropped by `dropped`, both baked into the pipeline.
#include "bindless.slang"

struct TestVertex {
    // Clip-space position; z is the depth (larger is nearer, lava tests with GREATER).
    float4 position;
    float4 color;
}

[SpecializationConstant] const float brightness = 1.0;
// The colour channel (0 to 2) the fragments drop, or 3 for none.
[SpecializationConstant] const uint dropped = 3;

struct TestRasterSpecializedBindings {
    Buf<TestVertex> vertices;
}

[[vk::push_constant]] TestRasterSpecializedBindings b;

struct TestRasterSpecializedVSOutput {
    float4 position : SV_Position;
    float4 color;
};

[shader("vertex")]
TestRasterSpecializedVSOutput test_raster_specialized_vertex(uint index: SV_VertexID) {
    let vertex = b.vertices.val[index];
    TestRasterSpecializedVSOutput output;
    output.position = vertex.position;
    output.color = float4(vertex.color.rgb * brightness, vertex.color.a);
    return output;
}

[shader("fragment")]
float4 test_raster_specialized_fragment(in TestRasterSpecializedVSOutput input) : SV_Target {
    var color = input.color;
    if (dropped < 3) {
        color[dropped] = 0.0;
    }
    return color;
}

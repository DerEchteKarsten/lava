// Test pass (vertex + fragment, flat shading): coloured primitives pulled from a vertex buffer whose colour is the provoking vertex's; points are one pixel.
#include "bindless.slang"

struct TestVertex {
    // Clip-space position; z is the depth (larger is nearer, lava tests with GREATER).
    float4 position;
    float4 color;
}

struct TestFlatBindings {
    Buf<TestVertex> vertices;
}

[[vk::push_constant]] TestFlatBindings b;

struct TestFlatVSOutput {
    float4 position : SV_Position;
    nointerpolation float4 color;
    float point_size : SV_PointSize;
};

[shader("vertex")]
TestFlatVSOutput test_flat_vertex(uint index: SV_VertexID) {
    let vertex = b.vertices.val[index];
    TestFlatVSOutput output;
    output.position = vertex.position;
    output.color = vertex.color;
    output.point_size = 1.0;
    return output;
}

// The vertex outputs but the point size, which fragment shaders can't take.
struct TestFlatFSInput {
    float4 position : SV_Position;
    nointerpolation float4 color;
};

[shader("fragment")]
float4 test_flat_fragment(in TestFlatFSInput input) : SV_Target {
    return input.color;
}

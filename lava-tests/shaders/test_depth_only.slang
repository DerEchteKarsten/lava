// Test pass (vertex only): draws the depth of triangles pulled from a vertex buffer, no colour.
#include "bindless.slang"

struct TestVertex {
    float4 position;
    float4 color;
}

struct TestDepthOnlyBindings {
    Buf<TestVertex> vertices;
}

[[vk::push_constant]] TestDepthOnlyBindings b;

[shader("vertex")]
float4 test_depth_only_vertex(uint index: SV_VertexID) : SV_Position {
    return b.vertices.val[index].position;
}

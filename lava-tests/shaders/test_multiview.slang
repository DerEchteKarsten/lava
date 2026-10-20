// Test pass (vertex + fragment): a multiview rendering's triangles, coloured per view
// (`SV_ViewID`): view 0 the vertex colour, the others `other`.
#include "bindless.slang"

struct TestVertex {
    float4 position;
    float4 color;
}

struct TestMultiviewBindings {
    Buf<TestVertex> vertices;
    float4 other;
}

[[vk::push_constant]] TestMultiviewBindings b;

struct TestMultiviewOutput {
    float4 position : SV_Position;
    float4 color;
};

[shader("vertex")]
TestMultiviewOutput test_multiview_vertex(uint index: SV_VertexID, uint view: SV_ViewID) {
    let vertex = b.vertices.val[index];
    TestMultiviewOutput output;
    output.position = vertex.position;
    output.color = view == 0 ? vertex.color : b.other;
    return output;
}

[shader("fragment")]
float4 test_multiview_fragment(in TestMultiviewOutput input) : SV_Target {
    return input.color;
}

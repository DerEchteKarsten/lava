// Test pass (vertex + fragment): a layered rendering's triangles, instance `i` drawn into layer
// `i` (`SV_RenderTargetArrayIndex` from the vertex shader, which needs `shader-output-layer`).
#include "bindless.slang"

struct TestVertex {
    float4 position;
    float4 color;
}

struct TestLayeredBindings {
    Buf<TestVertex> vertices;
    // The colour of layer 1's triangle; layer 0's is the vertex colour.
    float4 second;
}

[[vk::push_constant]] TestLayeredBindings b;

struct TestLayeredOutput {
    float4 position : SV_Position;
    float4 color;
    uint layer : SV_RenderTargetArrayIndex;
};

[shader("vertex")]
TestLayeredOutput test_layered_vertex(uint instance: SV_InstanceID, uint index: SV_VertexID) {
    let vertex = b.vertices.val[index];
    TestLayeredOutput output;
    output.position = vertex.position;
    output.color = instance == 0 ? vertex.color : b.second;
    output.layer = instance;
    return output;
}

[shader("fragment")]
float4 test_layered_fragment(in TestLayeredOutput input) : SV_Target {
    return input.color;
}

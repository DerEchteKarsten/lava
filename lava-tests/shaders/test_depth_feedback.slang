// Test pass (vertex + fragment, gpu_sync): triangles pulled from a vertex buffer whose fragments write into red the depth they load at their pixel from a depth image, the read-only depth attachment of their own rendering.
#include "bindless.slang"

struct TestVertex {
    // Clip-space position; z is the depth (larger is nearer, lava tests with GREATER).
    float4 position;
    float4 color;
}

struct TestDepthFeedbackBindings {
    Buf<TestVertex> vertices;
    Tex2D<float> depth;
}

[[vk::push_constant]] TestDepthFeedbackBindings b;

struct TestDepthFeedbackVSOutput {
    float4 position : SV_Position;
};

[shader("vertex")]
TestDepthFeedbackVSOutput test_depth_feedback_vertex(uint index: SV_VertexID) {
    TestDepthFeedbackVSOutput output;
    output.position = b.vertices.val[index].position;
    return output;
}

[shader("fragment")]
float4 test_depth_feedback_fragment(in TestDepthFeedbackVSOutput input) : SV_Target {
    let depth = b.depth.Load(uint2(input.position.xy));
    return float4(depth, 0.0, 0.0, 1.0);
}

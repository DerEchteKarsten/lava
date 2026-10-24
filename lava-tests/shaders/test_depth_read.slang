// Test pass (vertex + fragment, a runtime-typed read of the rendering's own depth attachment): triangles whose colour is the depth `depth` holds at each pixel, an image the pass reaches only through a `DynImg` handle (so its bindings name it in their accesses).
#include "bindless.slang"

struct TestVertex {
    // Clip-space position; z is the depth (larger is nearer, lava tests with GREATER).
    float4 position;
    float4 color;
}

struct TestDepthReadBindings {
    Buf<TestVertex> vertices;
    DynImg<float> depth;
    Sampler sampler;
    // The render target's size in pixels.
    float2 size;
}

[[vk::push_constant]] TestDepthReadBindings b;

struct TestDepthReadVSOutput {
    float4 position : SV_Position;
};

[shader("vertex")]
TestDepthReadVSOutput test_depth_read_vertex(uint index: SV_VertexID) {
    TestDepthReadVSOutput output;
    output.position = b.vertices.val[index].position;
    return output;
}

[shader("fragment")]
float4 test_depth_read_fragment(in TestDepthReadVSOutput input) : SV_Target {
    let depth = b.depth.SampleLevel(b.sampler, input.position.xy / b.size, 0.0);
    return float4(depth, depth, depth, 1.0);
}

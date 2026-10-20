// Test pass (vertex + fragment): like `test_raster`, with Vulkan's vertex and instance indices,
// which include a draw's first vertex, vertex offset and first instance.
#include "bindless.slang"

struct TestVertex {
    float4 position;
    float4 color;
}

struct TestDrawParamsBindings {
    Buf<TestVertex> vertices;
    // Clip-space shift applied once per instance index.
    float2 instance_offset;
}

[[vk::push_constant]] TestDrawParamsBindings b;

struct TestDrawParamsOutput {
    float4 position : SV_Position;
    float4 color;
};

[shader("vertex")]
TestDrawParamsOutput test_draw_params_vertex(uint instance: SV_VulkanInstanceID, uint index: SV_VulkanVertexID) {
    let vertex = b.vertices.val[index];
    TestDrawParamsOutput output;
    output.position = vertex.position + float4(b.instance_offset * float(instance), 0.0, 0.0);
    output.color = vertex.color;
    return output;
}

[shader("fragment")]
float4 test_draw_params_fragment(in TestDrawParamsOutput input) : SV_Target {
    return input.color;
}

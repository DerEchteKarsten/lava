// Test pass (vertex + fragment): coloured triangles pulled from a vertex buffer.
#include "bindless.slang"

struct TestVertex {
    // Clip-space position; z is the depth (larger is nearer, lava tests with GREATER).
    float4 position;
    float4 color;
}

struct TestRasterBindings {
    Buf<TestVertex> vertices;
    // Clip-space shift applied once per instance index.
    float2 instance_offset;
}

[[vk::push_constant]] TestRasterBindings b;

struct TestRasterVSOutput {
    float4 position : SV_Position;
    float4 color;
};

[shader("vertex")]
TestRasterVSOutput test_raster_vertex(uint instance: SV_InstanceID, uint index: SV_VertexID) {
    let vertex = b.vertices.val[index];
    TestRasterVSOutput output;
    output.position = vertex.position + float4(b.instance_offset * float(instance), 0.0, 0.0);
    output.color = vertex.color;
    return output;
}

[shader("fragment")]
float4 test_raster_fragment(in TestRasterVSOutput input) : SV_Target {
    return input.color;
}

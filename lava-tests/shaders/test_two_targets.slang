// Test pass (vertex + fragment, two colour attachments): coloured triangles written to location 0 and their inverted colour to location 1.
#include "bindless.slang"

struct TestVertex {
    // Clip-space position; z is the depth (larger is nearer, lava tests with GREATER).
    float4 position;
    float4 color;
}

struct TestTwoTargetsBindings {
    Buf<TestVertex> vertices;
}

[[vk::push_constant]] TestTwoTargetsBindings b;

struct TestTwoTargetsVSOutput {
    float4 position : SV_Position;
    float4 color;
};

[shader("vertex")]
TestTwoTargetsVSOutput test_two_targets_vertex(uint index: SV_VertexID) {
    let vertex = b.vertices.val[index];
    TestTwoTargetsVSOutput output;
    output.position = vertex.position;
    output.color = vertex.color;
    return output;
}

struct TestTwoTargetsOutput {
    [[vk::location(0)]] float4 first : SV_Target0;
    // Slang numbers outputs from 0 whatever their `SV_Target`: the second attachment is location 1.
    [[vk::location(1)]] float4 second : SV_Target1;
};

[shader("fragment")]
TestTwoTargetsOutput test_two_targets_fragment(in TestTwoTargetsVSOutput input) {
    TestTwoTargetsOutput output;
    output.first = input.color;
    output.second = float4(1.0 - input.color.rgb, 1.0);
    return output;
}

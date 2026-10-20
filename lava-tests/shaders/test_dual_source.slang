// Test pass (vertex + fragment): `test_raster`'s triangles with a second fragment output for
// dual-source blending (`BlendFactor::SRC1_*`): `color` and the blend factor `factor`.
#include "bindless.slang"

struct TestVertex {
    float4 position;
    float4 color;
}

struct TestDualSourceBindings {
    Buf<TestVertex> vertices;
    float4 factor;
}

[[vk::push_constant]] TestDualSourceBindings b;

struct TestDualSourceVSOutput {
    float4 position : SV_Position;
    float4 color;
};

struct TestDualSourceTargets {
    [[vk::location(0)]] float4 color : SV_Target0;
    [[vk::location(0), vk::index(1)]] float4 factor : SV_Target1;
};

[shader("vertex")]
TestDualSourceVSOutput test_dual_source_vertex(uint index: SV_VertexID) {
    let vertex = b.vertices.val[index];
    TestDualSourceVSOutput output;
    output.position = vertex.position;
    output.color = vertex.color;
    return output;
}

[shader("fragment")]
TestDualSourceTargets test_dual_source_fragment(in TestDualSourceVSOutput input) {
    TestDualSourceTargets targets;
    targets.color = input.color;
    targets.factor = b.factor;
    return targets;
}

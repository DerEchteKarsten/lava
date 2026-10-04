// Test pass (mesh + fragment): every mesh workgroup emits one triangle, shifted per group.
#include "bindless.slang"

struct TestMeshBindings {
    float4 color;
    // Clip-space shift applied once per workgroup index.
    float2 group_offset;
    // Half-size of the triangle in clip space.
    float scale;
}

[[vk::push_constant]] TestMeshBindings b;

struct TestMeshVertex {
    float4 position : SV_Position;
    float4 color;
};

static const float2 TRIANGLE[3] = { float2(0.0, -1.0), float2(1.0, 1.0), float2(-1.0, 1.0) };

[shader("mesh")]
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void test_mesh(out indices uint3 triangles[1], out vertices TestMeshVertex vertices[3], uint3 gid: SV_GroupID) {
    SetMeshOutputCounts(3, 1);
    triangles[0] = uint3(0, 1, 2);
    for (int i = 0; i < 3; i++) {
        let corner = TRIANGLE[i] * b.scale + b.group_offset * float(gid.x);
        vertices[i].position = float4(corner, 0.5, 1.0);
        // Vary the colour per corner so interpolation is visible.
        vertices[i].color = b.color * float4(float(i == 0), float(i == 1), float(i == 2), 1.0) + float4(0.2, 0.2, 0.2, 0.0);
    }
}

[shader("fragment")]
float4 test_mesh_fragment(in TestMeshVertex input) : SV_Target {
    return input.color;
}

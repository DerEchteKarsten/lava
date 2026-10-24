// Test pass (task + mesh + fragment): one task workgroup launches `columns` mesh workgroups through its payload, and mesh workgroup i fills column i (of width `width` in clip space, from the left edge) with `color`.
#include "bindless.slang"

struct TestTaskBindings {
    float4 color;
    uint columns;
    float width;
}

[[vk::push_constant]] TestTaskBindings b;

struct TestTaskPayload {
    uint columns;
};

groupshared TestTaskPayload payload;

[shader("amplification")]
[numthreads(1, 1, 1)]
void test_task_task() {
    payload.columns = b.columns;
    DispatchMesh(b.columns, 1, 1, payload);
}

struct TestTaskVertex {
    float4 position : SV_Position;
    float4 color;
};

[shader("mesh")]
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void test_task_mesh(
    in payload TestTaskPayload columns,
    out indices uint3 triangles[2],
    out vertices TestTaskVertex vertices[4],
    uint3 gid: SV_GroupID
) {
    SetMeshOutputCounts(4, 2);
    triangles[0] = uint3(0, 1, 2);
    triangles[1] = uint3(2, 1, 3);
    let left = -1.0 + b.width * float(gid.x);
    let corners = float2[4](float2(left, -1.0), float2(left + b.width, -1.0), float2(left, 1.0), float2(left + b.width, 1.0));
    for (int i = 0; i < 4; i++) {
        vertices[i].position = float4(corners[i], 0.5, 1.0);
        // Groups the payload doesn't count draw red, which the test would see.
        vertices[i].color = gid.x < columns.columns ? b.color : float4(1.0, 0.0, 0.0, 1.0);
    }
}

[shader("fragment")]
float4 test_task_fragment(in TestTaskVertex input) : SV_Target {
    return input.color;
}

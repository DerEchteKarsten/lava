// Test pass (compute): writes the instances of the ray tracing test scene on the GPU, the same
// ones `gpu_ray_tracing.rs` writes on the host.
#include "bindless.slang"
#include "ray_tracing.slang"

struct TestWriteInstancesBindings {
    MutBuf<AccelerationStructureInstance> instances;
    uint64_t wall;
    uint64_t triangle;
    uint64_t sphere;
}

[[vk::push_constant]] TestWriteInstancesBindings b;

float3x4 place(float scale, float3 position) {
    return float3x4(scale, 0.0, 0.0, position.x,
                    0.0, scale, 0.0, position.y,
                    0.0, 0.0, scale, position.z);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test_write_instances() {
    b.instances.val[0] = AccelerationStructureInstance(place(1.0, float3(0.0)), b.wall, 0);
    b.instances.val[1] = AccelerationStructureInstance(place(1.0, float3(-0.4, -0.3, -0.5)), b.triangle, 1);
    b.instances.val[2] = AccelerationStructureInstance(place(0.25, float3(0.35, 0.35, -0.5)), b.sphere, 2, 0xFF, 1);
    b.instances.val[3] = AccelerationStructureInstance(place(1.0, float3(0.45, -0.45, -0.5)), b.triangle, 3, 0x02);
}

// Test pass (ray tracing pipeline, shader execution reordering): traces into a hit object,
// reorders the threads by it, then runs its closest hit or miss shader, which colour the pixel
// like `test_ray_query.slang`'s triangles.
#include "bindless.slang"
#include "ray_tracing.slang"
#include "include/ray_scene.slang"

struct TestReorderBindings {
    MutImage2D<float4> target;
    Accel tlas;
    uint2 size;
}

[[vk::push_constant]] TestReorderBindings b;

struct Payload {
    float3 color;
}

[shader("raygeneration")]
void raygen() {
    let id = DispatchRaysIndex().xy;
    Payload payload = { float3(0.0) };
    let tlas = b.tlas.get();
    HitObject hit = HitObject::TraceRay(tlas, RAY_FLAG_NONE, RAY_MASK, 0, 0, 0, primary_ray(id, b.size), payload);
    ReorderThread(hit);
    HitObject::Invoke(tlas, hit, payload);
    b.target.Store(id, float4(payload.color, 1.0));
}

[shader("miss")]
void miss(inout Payload payload) {
    payload.color = float3(0.0);
}

[shader("closesthit")]
void hit(inout Payload payload, BuiltInTriangleIntersectionAttributes attributes) {
    payload.color = palette(InstanceID()) * (PrimitiveIndex() == 0 ? 1.0 : 0.6);
}

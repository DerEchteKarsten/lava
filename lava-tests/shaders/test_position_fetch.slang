// Test pass (ray tracing pipeline, position fetch): colours each hit by the object-space
// position of its triangle's second vertex, which the closest hit reads from the structure
// (`BuildFlags::allow_data_access`), mapped from -1..1 to 0..1.
#include "bindless.slang"
#include "ray_tracing.slang"
#include "include/ray_scene.slang"

struct TestPositionFetchBindings {
    MutImage2D<float4> target;
    Accel tlas;
    uint2 size;
}

[[vk::push_constant]] TestPositionFetchBindings b;

struct Payload {
    float3 color;
}

[shader("raygeneration")]
void raygen() {
    let id = DispatchRaysIndex().xy;
    Payload payload = { float3(0.0) };
    TraceRay(b.tlas.get(), RAY_FLAG_NONE, RAY_MASK, 0, 0, 0, primary_ray(id, b.size), payload);
    b.target.Store(id, float4(payload.color, 1.0));
}

[shader("miss")]
void miss(inout Payload payload) {
    payload.color = float3(0.0);
}

[shader("closesthit")]
void hit(inout Payload payload, BuiltInTriangleIntersectionAttributes attributes) {
    payload.color = HitTriangleVertexPosition(1) * 0.5 + 0.5;
}

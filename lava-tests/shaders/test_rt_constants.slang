// Test pass (ray tracing pipeline, specialization constants): primary rays, forced opaque, into a TLAS an `SCONST` names; hits are coloured by instance and misses by the `sky` constant, so each constant value is its own pipeline.
#include "bindless.slang"
#include "ray_tracing.slang"
#include "include/ray_scene.slang"

SCONST(Accel, tlas)
[SpecializationConstant] const uint sky = 0;

struct TestRtConstantsBindings {
    MutImage2D<float4> target;
    uint2 size;
}

[[vk::push_constant]] TestRtConstantsBindings b;

struct Payload {
    float3 color;
}

[shader("raygeneration")]
void raygen() {
    let id = DispatchRaysIndex().xy;
    Payload payload = { float3(0.0) };
    TraceRay(tlas.get(), RAY_FLAG_FORCE_OPAQUE, RAY_MASK, 0, 0, 0, primary_ray(id, b.size), payload);
    b.target.Store(id, float4(payload.color, 1.0));
}

[shader("miss")]
void sky_miss(inout Payload payload) {
    payload.color = float3(0.0, float(sky) / 255.0, 0.0);
}

[shader("closesthit")]
void wall_hit(inout Payload payload, BuiltInTriangleIntersectionAttributes attributes) {
    payload.color = palette(InstanceID());
}

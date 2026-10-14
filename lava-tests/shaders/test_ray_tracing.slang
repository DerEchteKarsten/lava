// Test pass (ray tracing pipeline): every shader kind. Two miss shaders (sky, shadow), a
// triangle hit group whose any-hit cuts holes into non-opaque geometry, a procedural sphere hit
// group, a callable that colours by instance, and shadow rays from the closest hits.
#include "bindless.slang"
#include "ray_tracing.slang"
#include "include/ray_scene.slang"

struct TestRayTracingBindings {
    MutImage2D<float4> target;
    Accel tlas;
    uint2 size;
}

[[vk::push_constant]] TestRayTracingBindings b;

struct Payload {
    float3 color;
}

struct ShadowPayload {
    uint lit;
}

struct Shade {
    uint instance;
    float3 color;
}

struct SphereAttributes {
    float3 normal;
}

static const float3 SKY = float3(0.2, 0.6, 0.3);

/// How much light reaches `position`: a shadow ray through the shadow miss shader.
float light(float3 position) {
    RayDesc ray;
    ray.Origin = position + TO_LIGHT * 0.001;
    ray.Direction = TO_LIGHT;
    ray.TMin = 0.0;
    ray.TMax = 10.0;
    ShadowPayload shadow = { 0 };
    // Forced opaque, so no any-hit shader sees this payload.
    let flags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_FORCE_OPAQUE;
    TraceRay(b.tlas.get(), flags, RAY_MASK, 0, 0, 1, ray, shadow);
    return shadow.lit != 0 ? 1.0 : 0.4;
}

float3 shade(float3 position) {
    Shade shade = { InstanceID(), float3(0.0) };
    CallShader(0, shade);
    return shade.color * light(position);
}

float3 hit_position() {
    return WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
}

[shader("raygeneration")]
[MaxRecursion(2)]
void raygen() {
    let id = DispatchRaysIndex().xy;
    Payload payload = { float3(0.0) };
    TraceRay(b.tlas.get(), RAY_FLAG_NONE, RAY_MASK, 0, 0, 0, primary_ray(id, b.size), payload);
    b.target.Store(id, float4(payload.color, 1.0));
}

[shader("miss")]
void sky(inout Payload payload) {
    payload.color = SKY;
}

[shader("miss")]
[Group(1)]
void unshadowed(inout ShadowPayload shadow) {
    shadow.lit = 1;
}

[shader("anyhit")]
void holes(inout Payload payload, BuiltInTriangleIntersectionAttributes attributes) {
    let cell = frac(hit_position().xy * 2.0);
    if (all(cell < 0.2)) {
        IgnoreHit();
    }
}

[shader("closesthit")]
void triangle_hit(inout Payload payload, BuiltInTriangleIntersectionAttributes attributes) {
    payload.color = shade(hit_position());
}

[shader("intersection")]
[Group(1)]
void sphere() {
    let t = hit_sphere(ObjectRayOrigin(), ObjectRayDirection());
    if (t >= RayTMin()) {
        SphereAttributes attributes = { normalize(ObjectRayOrigin() + ObjectRayDirection() * t) };
        ReportHit(t, 0, attributes);
    }
}

[shader("closesthit")]
[Group(1)]
void sphere_hit(inout Payload payload, SphereAttributes attributes) {
    // Brighter where the sphere faces the camera.
    payload.color = shade(hit_position()) * (0.5 - 0.5 * attributes.normal.z);
}

[shader("callable")]
void instance_colour(inout Shade shade) {
    shade.color = palette(shade.instance);
}

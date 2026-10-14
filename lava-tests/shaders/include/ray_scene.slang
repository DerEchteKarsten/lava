#pragma once
// The scene of lava's ray tracing tests (`gpu_headless.rs` builds it): orthographic rays along +Z
// onto a wall at z = 0, with triangles and a sphere in front of it.

/// The ray mask of every test ray: an instance with mask 0x02 is invisible.
static const uint RAY_MASK = 0x01;
/// Towards the light, for shadow rays.
static const float3 TO_LIGHT = normalize(float3(-0.4, -0.4, -1.0));

/// The primary ray of pixel `id`: from z = -2 along +Z through x, y in [-1, 1].
RayDesc primary_ray(uint2 id, uint2 size) {
    let xy = (float2(id) + 0.5) / float2(size) * 2.0 - 1.0;
    RayDesc ray;
    ray.Origin = float3(xy, -2.0);
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMin = 0.0;
    ray.TMax = 10.0;
    return ray;
}

/// The nearest hit of a ray with the unit sphere at the origin (the AABB's object space), or
/// a negative value.
float hit_sphere(float3 origin, float3 direction) {
    let a = dot(direction, direction);
    let b = dot(origin, direction);
    let c = dot(origin, origin) - 1.0;
    let discriminant = b * b - a * c;
    if (discriminant < 0.0) {
        return -1.0;
    }
    return (-b - sqrt(discriminant)) / a;
}

/// One colour per instance custom index.
float3 palette(uint index) {
    switch (index) {
    case 0: return float3(0.8, 0.8, 0.8);
    case 1: return float3(1.0, 0.2, 0.1);
    case 2: return float3(0.1, 0.4, 1.0);
    default: return float3(1.0, 0.0, 1.0);
    }
}

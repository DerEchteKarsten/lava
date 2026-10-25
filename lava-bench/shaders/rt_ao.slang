// Ray tracing scenario, ambient occlusion (compute, ray queries): a primary ray per pixel and two short occlusion rays from its hit.
#include "common.slang"

struct RtAoBindings {
    float4x4 inv_view_proj;
    float4 eye;
    MutImage2D<float4> target;
    Accel tlas;
    uint2 size;
}

[[vk::push_constant]] RtAoBindings b;

static const uint AO_RAYS = 2;

[shader("compute")]
[numthreads(8, 8, 1)]
void rt_ao(uint2 id: SV_DispatchThreadID) {
    if (any(id >= b.size)) {
        return;
    }
    let ndc = (float2(id) + 0.5) / float2(b.size) * 2.0 - 1.0;
    let far = mul(b.inv_view_proj, float4(ndc, 0.5, 1.0));
    RayDesc ray;
    ray.Origin = b.eye.xyz;
    ray.Direction = normalize(far.xyz / far.w - b.eye.xyz);
    ray.TMin = 0.01;
    ray.TMax = 1000.0;
    RayQuery<RAY_FLAG_FORCE_OPAQUE> primary;
    primary.TraceRayInline(b.tlas.get(), RAY_FLAG_NONE, 0xff, ray);
    primary.Proceed();
    if (primary.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
        b.target.Store(id, float4(0.1, 0.1, 0.15, 1.0));
        return;
    }
    let hit = ray.Origin + ray.Direction * primary.CommittedRayT();
    let instance = primary.CommittedInstanceID();
    float open = 0.0;
    for (uint i = 0; i < AO_RAYS; i++) {
        let h = hash(id.y * b.size.x + id.x + i * 0x9e3779b9u);
        let d = normalize(float3(hash01(h), hash01(h + 1), hash01(h + 2)) * 2.0 - 1.0 - ray.Direction);
        RayDesc occlusion;
        occlusion.Origin = hit - ray.Direction * 0.01;
        occlusion.Direction = d;
        occlusion.TMin = 0.01;
        occlusion.TMax = 1.0;
        RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
        query.TraceRayInline(b.tlas.get(), RAY_FLAG_NONE, 0xff, occlusion);
        query.Proceed();
        open += query.CommittedStatus() == COMMITTED_NOTHING ? 1.0 : 0.0;
    }
    let tint = float3(hash01(instance), hash01(instance + 1), hash01(instance + 2));
    b.target.Store(id, float4(tint * open / float(AO_RAYS), 1.0));
}

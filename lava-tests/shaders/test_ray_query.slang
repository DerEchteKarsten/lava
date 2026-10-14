// Test pass (compute, ray queries): colours each pixel by the instance its primary ray hits,
// testing AABB candidates against a sphere.
#include "bindless.slang"
#include "include/ray_scene.slang"

struct TestRayQueryBindings {
    MutImage2D<float4> target;
    Accel tlas;
    uint2 size;
}

[[vk::push_constant]] TestRayQueryBindings b;

[shader("compute")]
[numthreads(8, 8, 1)]
void test_ray_query(uint2 id: SV_DispatchThreadID) {
    if (any(id >= b.size)) {
        return;
    }
    let ray = primary_ray(id, b.size);
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(b.tlas.get(), RAY_FLAG_NONE, RAY_MASK, ray);
    while (query.Proceed()) {
        if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
            let t = hit_sphere(query.CandidateObjectRayOrigin(), query.CandidateObjectRayDirection());
            if (t >= ray.TMin && t <= query.CommittedRayT()) {
                query.CommitProceduralPrimitiveHit(t);
            }
        } else {
            query.CommitNonOpaqueTriangleHit();
        }
    }
    var color = float3(0.0);
    switch (query.CommittedStatus()) {
    case COMMITTED_TRIANGLE_HIT:
        // Darker on the second triangle of a geometry, so primitive indices show.
        color = palette(query.CommittedInstanceID()) * (query.CommittedPrimitiveIndex() == 0 ? 1.0 : 0.6);
        break;
    case COMMITTED_PROCEDURAL_PRIMITIVE_HIT:
        color = palette(query.CommittedInstanceID());
        break;
    default:
        break;
    }
    b.target.Store(id, float4(color, 1.0));
}

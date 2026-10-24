// Test pass (compute, ray queries): the ray query test's trace, through a TLAS and ray mask named inside a struct behind a `Buf` (a `u64` there, whose trace accesses the bindings get by hand).
#include "bindless.slang"
#include "include/ray_scene.slang"

struct TestSceneRef {
    Accel tlas;
    uint mask;
    uint unused;
}

struct TestAccelStructBindings {
    MutImage2D<float4> target;
    Buf<TestSceneRef> scene;
    uint2 size;
}

[[vk::push_constant]] TestAccelStructBindings b;

[shader("compute")]
[numthreads(8, 8, 1)]
void test_accel_struct(uint2 id: SV_DispatchThreadID) {
    if (any(id >= b.size)) {
        return;
    }
    let scene = b.scene.val[0];
    let ray = primary_ray(id, b.size);
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(scene.tlas.get(), RAY_FLAG_NONE, scene.mask, ray);
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

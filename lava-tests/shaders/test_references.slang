// Test pass (compute, referencing structs): samples images through the samplers of an array of nested referencing structs behind a `Buf`, reads a buffer and a `DynImg` they name, and writes through their write references (an `[Unordered]` atomic, and a store of what it returned).
#include "bindless.slang"

struct RefMaterial {
    Tex2D<float4> color;
    Sampler sampler;
}

struct RefCounter {
    [Unordered] MutBuf<uint> hits;
    MutBuf<uint> slots;
}

struct RefNode {
    RefMaterial materials[2];
    Buf<float> weights;
    RefCounter counter;
    DynImg<float4> dyn;
    float scale;
}

struct TestReferencesBindings {
    Buf<RefNode> nodes;
    MutBuf<float4> out;
}

[[vk::push_constant]] TestReferencesBindings b;

[shader("compute")]
[numthreads(1, 1, 1)]
void test_references() {
    let n = b.nodes.val[0];
    let uv = float2(0.5, 0.5);
    b.out.val[0] = n.materials[0].color.SampleLevel(n.materials[0].sampler, uv, 0.0);
    b.out.val[1] = n.materials[1].color.SampleLevel(n.materials[1].sampler, uv, 0.0);
    b.out.val[2] = float4(n.weights.val[0], n.weights.val[1], n.scale, 0.0);
    b.out.val[3] = n.dyn.IsNull() ? float4(-1.0) : n.dyn.SampleLevel(n.materials[0].sampler, uv, 0.0);
    uint old;
    InterlockedAdd(n.counter.hits.val[0], 1, old);
    n.counter.slots.val[0] = old + 10;
}

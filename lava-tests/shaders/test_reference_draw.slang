// Test pass (vertex + fragment): a triangle over the whole target whose fragments write through the write references of structs behind `Buf`s: an `[Unordered]` atomic count, and an ordered store.
#include "bindless.slang"

struct RefHits {
    [Unordered] MutBuf<uint> hits;
}

struct RefSlot {
    MutBuf<uint> slot;
}

struct TestReferenceDrawBindings {
    Buf<RefHits> hits;
    Buf<RefSlot> slot;
}

[[vk::push_constant]] TestReferenceDrawBindings b;

[shader("vertex")]
float4 test_reference_draw_vertex(uint index: SV_VertexID) : SV_Position {
    let uv = float2((index << 1) & 2, index & 2);
    return float4(uv * 2.0 - 1.0, 0.5, 1.0);
}

[shader("fragment")]
float4 test_reference_draw_fragment() : SV_Target {
    InterlockedAdd(b.hits.val[0].hits.val[0], 1);
    b.slot.val[0].slot.val[0] = 1;
    return float4(1.0);
}

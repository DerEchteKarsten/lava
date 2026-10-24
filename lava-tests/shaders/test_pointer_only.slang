// Test pass (compute, raw pointer): reaches `data` only through an address without an access; `write` sets data[i] = value + i after `spin` iterations of busy work (so a read that doesn't wait for it sees old data), else out[i] = data[i], for the first `count` elements.
#include "bindless.slang"

struct TestPointerOnlyBindings {
    uint* data;
    MutBuf<uint> out;
    uint count;
    uint value;
    uint write;
    uint spin;
}

[[vk::push_constant]] TestPointerOnlyBindings b;

[shader("compute")]
[numthreads(64, 1, 1)]
void test_pointer_only(uint dtid: SV_DispatchThreadID) {
    if (dtid >= b.count) {
        return;
    }
    if (b.write != 0) {
        uint hash = dtid;
        for (uint i = 0; i < b.spin; i++) {
            hash = hash * 1664525u + 1013904223u;
        }
        // Never true, but the compiler can't tell: the loop stays.
        let extra = hash == 0xFFFFFFFFu && dtid == 0xFFFFFFFFu ? 1u : 0u;
        b.data[dtid] = b.value + dtid + extra;
    } else {
        b.out.val[dtid] = b.data[dtid];
    }
}

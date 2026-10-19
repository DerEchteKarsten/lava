// Test pass (compute, push data): copies a push-constant struct of the full 256 bytes into a buffer.
#include "bindless.slang"

struct TestPushDataBindings {
    uint4 values[15];
    MutBuf<uint4> out;
    uint2 last;
}

[[vk::push_constant]] TestPushDataBindings b;

[shader("compute")]
[numthreads(16, 1, 1)]
void test_push_data(uint i: SV_DispatchThreadID) {
    if (i < 15) {
        b.out.val[i] = b.values[i];
    } else if (i == 15) {
        b.out.val[i] = uint4(b.last, 0, 0);
    }
}

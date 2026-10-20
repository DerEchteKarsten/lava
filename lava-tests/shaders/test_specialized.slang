// Test pass (compute, a specialization constant that reaches nothing): out[i] = i * factor for
// the first `count` elements, whose pipelines several threads make at once.
#include "bindless.slang"

[SpecializationConstant] const uint factor = 1;

struct TestSpecializedBindings {
    MutBuf<uint> out;
    uint count;
}

[[vk::push_constant]] TestSpecializedBindings b;

[shader("compute")]
[numthreads(64, 1, 1)]
void test_specialized(uint i: SV_DispatchThreadID) {
    if (i < b.count) {
        b.out.val[i] = i * factor;
    }
}

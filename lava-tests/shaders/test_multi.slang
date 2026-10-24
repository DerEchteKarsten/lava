// Test pass file (several compute entry points, dispatch bases, exported constants): `fill` and `double` are two passes, `TestMultiFill` and `TestMultiDouble`, sharing the push constants and the constants exported to Rust.
#include "bindless.slang"

[RustConst] static const uint MULTI_FILL_ADD = 3;
[RustConst] static const int MULTI_NEGATIVE = -2;
[RustConst] static const float MULTI_HALF = 0.5;
[RustConst] static const bool MULTI_YES = true;
[RustConst] static const uint64_t MULTI_BIG = 0x100000000;
static const uint MULTI_PRIVATE = 7;

struct TestMultiBindings {
    MutBuf<uint> out;
    uint count;
}

[[vk::push_constant]] TestMultiBindings b;

/// out[i] = i + MULTI_FILL_ADD for every thread's i, workgroup bases included.
[shader("compute")]
[numthreads(8, 1, 1)]
void fill(uint i: SV_DispatchThreadID) {
    if (i < b.count) {
        b.out.val[i] = i + MULTI_FILL_ADD + MULTI_PRIVATE - 7;
    }
}

/// out[i] *= 2, on a 4x2 workgroup over the rows of 8.
[shader("compute")]
[numthreads(4, 2, 1)]
void double(uint2 id: SV_DispatchThreadID) {
    let i = id.y * 8 + id.x;
    if (i < b.count) {
        b.out.val[i] *= 2;
    }
}

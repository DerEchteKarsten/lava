// Hi-Z (compute): one reduction step, the farthest (reverse depth: smallest) of each 2x2 block of the larger level.
#include "common.slang"

struct HizBindings {
    Tex2D<float> src;
    MutImage2D<float> dst;
    uint2 src_size;
    uint2 size;
}

[[vk::push_constant]] HizBindings b;

[shader("compute")]
[numthreads(8, 8, 1)]
void hiz(uint2 id: SV_DispatchThreadID) {
    if (any(id >= b.size)) {
        return;
    }
    let last = b.src_size - 1;
    let p = id * 2;
    let d = min(min(b.src.Load(min(p, last)), b.src.Load(min(p + uint2(1, 0), last))),
                min(b.src.Load(min(p + uint2(0, 1), last)), b.src.Load(min(p + uint2(1, 1), last))));
    b.dst.Store(id, d);
}

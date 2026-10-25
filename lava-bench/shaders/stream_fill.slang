// Streaming scenario, transient fill (compute): fills a freshly created buffer with hashes of its seed.
#include "common.slang"

struct StreamFillBindings {
    MutBuf<uint> dst;
    uint count;
    uint seed;
}

[[vk::push_constant]] StreamFillBindings b;

[shader("compute")]
[numthreads(256, 1, 1)]
void stream_fill(uint id: SV_DispatchThreadID) {
    if (id < b.count) {
        b.dst.val[id] = hash(b.seed * 65536 + id);
    }
}

// Checksum (compute): out[slot] += a hash of each word of `src` and its index, for `--check`.
#include "common.slang"

struct ChecksumBufferBindings {
    Buf<uint> src;
    MutBuf<uint> out;
    uint count;
    uint slot;
}

[[vk::push_constant]] ChecksumBufferBindings b;

[shader("compute")]
[numthreads(256, 1, 1)]
void checksum_buffer(uint id: SV_DispatchThreadID) {
    if (id >= b.count) {
        return;
    }
    InterlockedAdd(b.out.val[b.slot], checksum_term(b.src.val[id], id));
}

// Checksum (compute): out[slot] += a hash of each texel component of a 2D image (mip 0), for `--check`.
#include "common.slang"

struct ChecksumImageBindings {
    Tex2D<float4> image;
    MutBuf<uint> out;
    uint2 size;
    uint slot;
}

[[vk::push_constant]] ChecksumImageBindings b;

[shader("compute")]
[numthreads(8, 8, 1)]
void checksum_image(uint2 id: SV_DispatchThreadID) {
    if (any(id >= b.size)) {
        return;
    }
    let texel = asuint(b.image.Load(id));
    let index = (id.y * b.size.x + id.x) * 4;
    let sum = checksum_term(texel.x, index) + checksum_term(texel.y, index + 1)
        + checksum_term(texel.z, index + 2) + checksum_term(texel.w, index + 3);
    InterlockedAdd(b.out.val[b.slot], sum);
}

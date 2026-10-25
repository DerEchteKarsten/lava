// Checksum (compute): out[slot] += a hash of each texel component of a 3D image, for `--check`.
#include "common.slang"

struct ChecksumVolumeBindings {
    Tex3D<float4> volume;
    MutBuf<uint> out;
    uint3 size;
    uint slot;
}

[[vk::push_constant]] ChecksumVolumeBindings b;

[shader("compute")]
[numthreads(4, 4, 4)]
void checksum_volume(uint3 id: SV_DispatchThreadID) {
    if (any(id >= b.size)) {
        return;
    }
    let texel = asuint(b.volume.Load(id));
    let index = ((id.z * b.size.y + id.y) * b.size.x + id.x) * 4;
    let sum = checksum_term(texel.x, index) + checksum_term(texel.y, index + 1)
        + checksum_term(texel.z, index + 2) + checksum_term(texel.w, index + 3);
    InterlockedAdd(b.out.val[b.slot], sum);
}

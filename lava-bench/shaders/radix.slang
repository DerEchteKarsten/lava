// Particles scenario, radix sort (compute, one pass per entry point): one 8-bit digit pass of a stable key/value sort is a per-tile digit count, one scan of the counts and a scatter. Tiles are 256 keys, one workgroup each.
#include "common.slang"

struct RadixBindings {
    Buf<uint> keys_in;
    Buf<uint> values_in;
    MutBuf<uint> keys_out;
    MutBuf<uint> values_out;
    MutBuf<uint> histogram;
    uint count;
    uint shift;
}

[[vk::push_constant]] RadixBindings b;

static const uint TILE = 256;

uint tiles() {
    return (b.count + TILE - 1) / TILE;
}

groupshared uint digits[256];

/// histogram[digit * tiles + tile] = how many of the tile's keys have that digit.
[shader("compute")]
[numthreads(256, 1, 1)]
void count(uint id: SV_DispatchThreadID, uint local: SV_GroupIndex, uint3 group: SV_GroupID) {
    digits[local] = 0;
    GroupMemoryBarrierWithGroupSync();
    if (id < b.count) {
        InterlockedAdd(digits[(b.keys_in.val[id] >> b.shift) & 0xff], 1);
    }
    GroupMemoryBarrierWithGroupSync();
    b.histogram.val[local * tiles() + group.x] = digits[local];
}

groupshared uint sums[1024];

/// One workgroup of 1024 threads turns the histogram into exclusive offsets, each thread a
/// contiguous run of it.
[shader("compute")]
[numthreads(1024, 1, 1)]
void scan(uint local: SV_GroupIndex) {
    let n = 256 * tiles();
    let run = (n + 1023) / 1024;
    let first = local * run;
    let last = min(first + run, n);
    uint sum = 0;
    for (uint i = first; i < last; i++) {
        sum += b.histogram.val[i];
    }
    sums[local] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint offset = 1; offset < 1024; offset *= 2) {
        let add = local >= offset ? sums[local - offset] : 0;
        GroupMemoryBarrierWithGroupSync();
        sums[local] += add;
        GroupMemoryBarrierWithGroupSync();
    }
    var prefix = sums[local] - sum;
    for (uint i = first; i < last; i++) {
        let count = b.histogram.val[i];
        b.histogram.val[i] = prefix;
        prefix += count;
    }
}

/// Each key goes to its digit's offset plus the number of keys before it in its tile with the
/// same digit, which keeps the sort stable.
[shader("compute")]
[numthreads(256, 1, 1)]
void scatter(uint id: SV_DispatchThreadID, uint local: SV_GroupIndex, uint3 group: SV_GroupID) {
    let key = id < b.count ? b.keys_in.val[id] : 0xffffffffu;
    let digit = (key >> b.shift) & 0xff;
    digits[local] = id < b.count ? digit : 256;
    GroupMemoryBarrierWithGroupSync();
    if (id >= b.count) {
        return;
    }
    uint rank = 0;
    for (uint i = 0; i < local; i++) {
        rank += digits[i] == digit ? 1 : 0;
    }
    let dst = b.histogram.val[digit * tiles() + group.x] + rank;
    b.keys_out.val[dst] = key;
    b.values_out.val[dst] = b.values_in.val[id];
}

// Particles scenario (compute, one pass per entry point): emit and simulate particles, then compact the live ones to the front of a key/value list (depth keys, dead ones last) and write the indirect draw of the live ones. Tiles are 256 particles, one workgroup each.
#include "common.slang"

struct Particle {
    float3 position;
    float life;
    float3 velocity;
    float seed;
}

struct DrawCommand {
    uint vertex_count;
    uint instance_count;
    uint first_vertex;
    uint first_instance;
}

struct ParticlesBindings {
    float4x4 view_proj;
    MutBuf<Particle> particles;
    MutBuf<uint> tile_counts;
    MutBuf<uint> keys;
    MutBuf<uint> values;
    MutBuf<DrawCommand> draw;
    uint count;
    uint frame;
    float dt;
}

[[vk::push_constant]] ParticlesBindings b;

static const uint TILE = 256;

[shader("compute")]
[numthreads(256, 1, 1)]
void emit(uint id: SV_DispatchThreadID) {
    if (id >= b.count) {
        return;
    }
    var p = b.particles.val[id];
    if (p.life > 0.0 || hash01(id ^ (b.frame * 0x9e3779b9u)) > 0.05) {
        return;
    }
    let h = hash(id * 3 + b.frame);
    p.position = float3(hash01(h) - 0.5, 0.0, hash01(h + 1) - 0.5) * 2.0;
    p.velocity = float3(hash01(h + 2) - 0.5, 4.0 + hash01(h + 3) * 2.0, hash01(h + 4) - 0.5) * 2.0;
    p.life = 1.0 + hash01(h + 5) * 2.0;
    p.seed = hash01(h + 6);
    b.particles.val[id] = p;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void simulate(uint id: SV_DispatchThreadID) {
    if (id >= b.count) {
        return;
    }
    var p = b.particles.val[id];
    if (p.life <= 0.0) {
        return;
    }
    p.velocity += float3(0.0, -9.81, 0.0) * b.dt;
    p.velocity *= 0.995;
    p.position += p.velocity * b.dt;
    if (p.position.y < 0.0) {
        p.position.y = -p.position.y;
        p.velocity.y = -p.velocity.y * 0.6;
    }
    p.life -= b.dt;
    b.particles.val[id] = p;
}

groupshared uint tile_alive;

[shader("compute")]
[numthreads(256, 1, 1)]
void compact_count(uint id: SV_DispatchThreadID, uint local: SV_GroupIndex, uint3 group: SV_GroupID) {
    if (local == 0) {
        tile_alive = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    if (id < b.count && b.particles.val[id].life > 0.0) {
        InterlockedAdd(tile_alive, 1);
    }
    GroupMemoryBarrierWithGroupSync();
    if (local == 0) {
        b.tile_counts.val[group.x] = tile_alive;
    }
}

groupshared uint scan[1024];

/// An inclusive scan of `scan[0..n)` by `n` threads (Hillis-Steele), the thread's own entry.
uint scan_inclusive(uint local, uint n) {
    for (uint offset = 1; offset < n; offset *= 2) {
        let add = local >= offset ? scan[local - offset] : 0;
        GroupMemoryBarrierWithGroupSync();
        scan[local] += add;
        GroupMemoryBarrierWithGroupSync();
    }
    return scan[local];
}

/// One workgroup of 1024 threads: the tiles' live counts become exclusive offsets, the total
/// the draw's instance count.
[shader("compute")]
[numthreads(1024, 1, 1)]
void compact_scan(uint local: SV_GroupIndex) {
    let tiles = (b.count + TILE - 1) / TILE;
    let own = local < tiles ? b.tile_counts.val[local] : 0;
    scan[local] = own;
    GroupMemoryBarrierWithGroupSync();
    let inclusive = scan_inclusive(local, 1024);
    if (local < tiles) {
        b.tile_counts.val[local] = inclusive - own;
    }
    if (local == 1023) {
        DrawCommand draw;
        draw.vertex_count = 6;
        draw.instance_count = inclusive;
        draw.first_vertex = 0;
        draw.first_instance = 0;
        b.draw.val[0] = draw;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void compact_scatter(uint id: SV_DispatchThreadID, uint local: SV_GroupIndex, uint3 group: SV_GroupID) {
    let alive = id < b.count && b.particles.val[id].life > 0.0;
    scan[local] = alive ? 1 : 0;
    GroupMemoryBarrierWithGroupSync();
    let rank = scan_inclusive(local, TILE) - (alive ? 1 : 0);
    if (id >= b.count) {
        return;
    }
    let offset = b.tile_counts.val[group.x];
    let total = b.draw.val[0].instance_count;
    uint dst;
    uint key;
    if (alive) {
        dst = offset + rank;
        let clip = mul(b.view_proj, float4(b.particles.val[id].position, 1.0));
        key = asuint(max(clip.w, 0.0));
    } else {
        // Dead particles after every live one, in particle order.
        dst = total + (group.x * TILE - offset) + (local - rank);
        key = 0xffffffffu;
    }
    b.keys.val[dst] = key;
    b.values.val[dst] = id;
}

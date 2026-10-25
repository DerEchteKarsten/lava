// ML scenario (compute, one pass per entry point): the kernels of a transformer step and its backward pass, on float matrices that are sub-ranges of shared buffers. Every pass reads `a`, `b` and `bias` and writes `c`.
#include "common.slang"

struct MlBindings {
    Buf<float> a;
    Buf<float> b;
    MutBuf<float> c;
    Buf<float> bias;
    // c[m x n] (+)= op(a)[m x k] * op(b)[k x n], row-major with leading dimensions lda, ldb
    // and ldc; workgroups along z step each matrix by its batch stride.
    uint m;
    uint n;
    uint k;
    uint lda;
    uint ldb;
    uint ldc;
    uint a_batch;
    uint b_batch;
    uint c_batch;
    /// Bit 0 transposes a, bit 1 b, bit 2 accumulates into c, bit 3 masks the future (softmax).
    uint flags;
    float scale;
}

[[vk::push_constant]] MlBindings b;

static const uint TILE = 16;
groupshared float tile_a[TILE][TILE];
groupshared float tile_b[TILE][TILE];

float load_a(uint batch, uint row, uint col) {
    if (row >= b.m || col >= b.k) {
        return 0.0;
    }
    let base = batch * b.a_batch;
    return (b.flags & 1) != 0 ? b.a.val[base + col * b.lda + row] : b.a.val[base + row * b.lda + col];
}

float load_b(uint batch, uint row, uint col) {
    if (row >= b.k || col >= b.n) {
        return 0.0;
    }
    let base = batch * b.b_batch;
    return (b.flags & 2) != 0 ? b.b.val[base + col * b.ldb + row] : b.b.val[base + row * b.ldb + col];
}

[shader("compute")]
[numthreads(16, 16, 1)]
void matmul(uint3 group: SV_GroupID, uint3 local: SV_GroupThreadID) {
    let row = group.y * TILE + local.y;
    let col = group.x * TILE + local.x;
    let batch = group.z;
    float sum = 0.0;
    for (uint t = 0; t < b.k; t += TILE) {
        tile_a[local.y][local.x] = load_a(batch, row, t + local.x);
        tile_b[local.y][local.x] = load_b(batch, t + local.y, col);
        GroupMemoryBarrierWithGroupSync();
        for (uint i = 0; i < TILE; i++) {
            sum += tile_a[local.y][i] * tile_b[i][local.x];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (row < b.m && col < b.n) {
        let index = batch * b.c_batch + row * b.ldc + col;
        let value = sum * b.scale;
        b.c.val[index] = (b.flags & 4) != 0 ? b.c.val[index] + value : value;
    }
}

groupshared float row_reduce[256];

/// The sum of every thread's `value` in a workgroup of 256.
float reduce_sum(uint local, float value) {
    row_reduce[local] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128; stride > 0; stride /= 2) {
        if (local < stride) {
            row_reduce[local] += row_reduce[local + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    let sum = row_reduce[0];
    GroupMemoryBarrierWithGroupSync();
    return sum;
}

/// The largest `value` of a workgroup of 256.
float reduce_max(uint local, float value) {
    row_reduce[local] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128; stride > 0; stride /= 2) {
        if (local < stride) {
            row_reduce[local] = max(row_reduce[local], row_reduce[local + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    let result = row_reduce[0];
    GroupMemoryBarrierWithGroupSync();
    return result;
}

/// In place over the m rows of n (at most 256) of c, one workgroup per row; flag 8 masks
/// columns after the row's position within its n x n block (causal attention).
[shader("compute")]
[numthreads(256, 1, 1)]
void softmax(uint3 group: SV_GroupID, uint local: SV_GroupIndex) {
    let row = group.x;
    let masked = (b.flags & 8) != 0 && local > row % b.n;
    let inside = local < b.n && !masked;
    let index = row * b.ldc + local;
    let x = inside ? b.c.val[index] : -1e30;
    let largest = reduce_max(local, x);
    let e = inside ? exp(x - largest) : 0.0;
    let sum = reduce_sum(local, e);
    if (local < b.n) {
        b.c.val[index] = e / sum;
    }
}

/// c = (a - mean) / deviation * b + bias over rows of n (at most 256), one workgroup per row.
[shader("compute")]
[numthreads(256, 1, 1)]
void layernorm(uint3 group: SV_GroupID, uint local: SV_GroupIndex) {
    let index = group.x * b.n + local;
    let x = local < b.n ? b.a.val[index] : 0.0;
    let mean = reduce_sum(local, x) / float(b.n);
    let d = local < b.n ? x - mean : 0.0;
    let variance = reduce_sum(local, d * d) / float(b.n);
    if (local < b.n) {
        b.c.val[index] = d * rsqrt(variance + 1e-5) * b.b.val[local] + b.bias.val[local];
    }
}

float gelu(float x) {
    return 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
}

/// c = gelu(a), m * n elements.
[shader("compute")]
[numthreads(256, 1, 1)]
void gelu(uint id: SV_DispatchThreadID) {
    if (id < b.m * b.n) {
        b.c.val[id] = gelu(b.a.val[id]);
    }
}

/// c = a + bias, m * n elements.
[shader("compute")]
[numthreads(256, 1, 1)]
void add(uint id: SV_DispatchThreadID) {
    if (id < b.m * b.n) {
        b.c.val[id] = b.a.val[id] + b.bias.val[id];
    }
}

/// c *= gelu'(a), m * n elements.
[shader("compute")]
[numthreads(256, 1, 1)]
void gelu_back(uint id: SV_DispatchThreadID) {
    if (id >= b.m * b.n) {
        return;
    }
    let x = b.a.val[id];
    let t = tanh(0.7978845608 * (x + 0.044715 * x * x * x));
    let derivative = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * 0.7978845608 * (1.0 + 0.134145 * x * x);
    b.c.val[id] *= derivative;
}

/// In place: c = a * (c - sum(c * a)) per row, the softmax gradient (a the probabilities).
[shader("compute")]
[numthreads(256, 1, 1)]
void softmax_back(uint3 group: SV_GroupID, uint local: SV_GroupIndex) {
    let index = group.x * b.ldc + local;
    let p = local < b.n ? b.a.val[index] : 0.0;
    let g = local < b.n ? b.c.val[index] : 0.0;
    let dot = reduce_sum(local, p * g);
    if (local < b.n) {
        b.c.val[index] = p * (g - dot);
    }
}

/// In place: c *= b / deviation of a's row, the layer norm's gradient without its mean terms.
[shader("compute")]
[numthreads(256, 1, 1)]
void layernorm_back(uint3 group: SV_GroupID, uint local: SV_GroupIndex) {
    let index = group.x * b.n + local;
    let x = local < b.n ? b.a.val[index] : 0.0;
    let mean = reduce_sum(local, x) / float(b.n);
    let d = local < b.n ? x - mean : 0.0;
    let variance = reduce_sum(local, d * d) / float(b.n);
    if (local < b.n) {
        b.c.val[index] *= b.b.val[local] * rsqrt(variance + 1e-5);
    }
}

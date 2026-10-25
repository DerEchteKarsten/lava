// What several benchmark passes share: mesh vertices, draw instances, hashes.
#pragma once
#include "bindless.slang"

struct MeshVertex {
    float3 position;
    float3 normal;
    float2 uv;
}

struct DrawInstance {
    float4x4 model;
    float4 color;
}

/// An instance of the GPU-driven scenario: its bounding sphere (centre, radius).
struct CullInstance {
    float4 sphere;
}

/// A well-mixed 32-bit hash of `x`.
uint hash(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

/// A hash of `x` in [0, 1).
float hash01(uint x) {
    return float(hash(x) >> 8) / 16777216.0;
}

/// The checksum term of word `value` at position `index`: the sum of all terms is independent
/// of the order threads add them in.
uint checksum_term(uint value, uint index) {
    return hash(value ^ hash(index));
}

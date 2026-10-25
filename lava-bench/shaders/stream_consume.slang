// Streaming scenario, consumer (compute): sums every uploaded buffer, every transient buffer and the smallest mips of every streamed texture, all named by a referencing buffer the frame rewrites.
#include "common.slang"

struct StreamInputs {
    Buf<uint> uploaded[32];
    Buf<uint> transients[16];
    Tex2D<float4> textures[8];
    Sampler sampler;
    // Pads the struct to its 8-byte stride, which lava-build-slang asks for.
    uint pad;
}

struct StreamConsumeBindings {
    Buf<StreamInputs> inputs;
    MutBuf<uint> out;
    uint count;
}

[[vk::push_constant]] StreamConsumeBindings b;

[shader("compute")]
[numthreads(256, 1, 1)]
void stream_consume(uint id: SV_DispatchThreadID) {
    if (id >= b.count) {
        return;
    }
    let inputs = b.inputs.val[0];
    uint sum = 0;
    for (uint i = 0; i < 32; i++) {
        sum += inputs.uploaded[i].val[id];
    }
    for (uint i = 0; i < 16; i++) {
        sum += inputs.transients[i].val[id * 4];
    }
    let uv = float2(hash01(id), hash01(id + 7));
    for (uint i = 0; i < 8; i++) {
        sum += asuint(inputs.textures[i].SampleLevel(inputs.sampler, uv, 4.0).x);
    }
    b.out.val[id] = sum;
}

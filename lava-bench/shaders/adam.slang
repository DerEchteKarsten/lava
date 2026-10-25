// ML scenario, optimizer (compute): one Adam step of every weight from its gradient, the moments interleaved per weight.
#include "common.slang"

struct AdamBindings {
    MutBuf<float> params;
    Buf<float> grads;
    MutBuf<float2> moments;
    uint count;
    float lr;
    /// 1 - beta1^t and 1 - beta2^t.
    float correction1;
    float correction2;
}

[[vk::push_constant]] AdamBindings b;

[shader("compute")]
[numthreads(256, 1, 1)]
void adam(uint id: SV_DispatchThreadID) {
    if (id >= b.count) {
        return;
    }
    let g = clamp(b.grads.val[id], -1.0, 1.0);
    var m = b.moments.val[id];
    m.x = 0.9 * m.x + 0.1 * g;
    m.y = 0.999 * m.y + 0.001 * g * g;
    b.moments.val[id] = m;
    let step = (m.x / b.correction1) / (sqrt(m.y / b.correction2) + 1e-8);
    b.params.val[id] -= b.lr * step;
}

// TAA resolve (compute): the lit frame plus bloom blended into last frame's history; writes the new history and the LDR output.
#include "common.slang"

struct TaaBindings {
    Tex2D<float4> current;
    Tex2D<float4> bloom;
    Tex2D<float4> history;
    MutImage2D<float4> history_out;
    MutImage2D<float4> output;
    Sampler sampler;
    uint2 size;
}

[[vk::push_constant]] TaaBindings b;

[shader("compute")]
[numthreads(8, 8, 1)]
void taa(uint2 id: SV_DispatchThreadID) {
    if (any(id >= b.size)) {
        return;
    }
    let uv = (float2(id) + 0.5) / float2(b.size);
    let current = b.current.Load(id) + b.bloom.SampleLevel(b.sampler, uv, 0.0) * 0.2;
    // Clamp the history to the current neighbourhood's range.
    var low = current;
    var high = current;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            let p = uint2(clamp(int2(id) + int2(x, y), int2(0), int2(b.size) - 1));
            let c = b.current.Load(p);
            low = min(low, c);
            high = max(high, c);
        }
    }
    let history = clamp(b.history.Load(id), low, high);
    let resolved = lerp(history, current, 0.1);
    b.history_out.Store(id, resolved);
    b.output.Store(id, float4(resolved.rgb / (1.0 + resolved.rgb), 1.0));
}

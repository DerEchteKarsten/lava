// Image processing scenario (compute, one pass per entry point): the filters of a photo or video chain, each from `src` into `dst` (storage) of `size` texels; `params` are the filter's own.
#include "common.slang"

struct FilterBindings {
    Tex2D<float4> src;
    MutImage2D<float4> dst;
    Sampler sampler;
    uint2 size;
    float4 params;
}

[[vk::push_constant]] FilterBindings b;

bool outside(uint2 id) {
    return any(id >= b.size);
}

/// A test picture: gradients, rings and noise.
[shader("compute")]
[numthreads(8, 8, 1)]
void generate(uint2 id: SV_DispatchThreadID) {
    if (outside(id)) {
        return;
    }
    let uv = float2(id) / float2(b.size);
    let ring = 0.5 + 0.5 * sin(length(uv - 0.5) * 60.0);
    let noise = hash01(id.y * b.size.x + id.x);
    b.dst.Store(id, float4(uv.x, uv.y * ring, 0.3 + 0.4 * noise, 1.0));
}

/// A Gaussian blur along `params.xy` (a texel step), `params.z` the radius in texels.
float4 blur(uint2 id) {
    let radius = int(b.params.z);
    let sigma = max(b.params.z * 0.5, 0.5);
    float4 sum = 0.0;
    float weight = 0.0;
    let uv = (float2(id) + 0.5) / float2(b.size);
    for (int i = -radius; i <= radius; i++) {
        let w = exp(-float(i * i) / (2.0 * sigma * sigma));
        sum += b.src.SampleLevel(b.sampler, uv + b.params.xy * float(i), 0.0) * w;
        weight += w;
    }
    return sum / weight;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void blur_h(uint2 id: SV_DispatchThreadID) {
    if (!outside(id)) {
        b.dst.Store(id, blur(id));
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void blur_v(uint2 id: SV_DispatchThreadID) {
    if (!outside(id)) {
        b.dst.Store(id, blur(id));
    }
}

/// Lift, gamma and gain (`params.xyz`) and saturation (`params.w`).
[shader("compute")]
[numthreads(8, 8, 1)]
void grade(uint2 id: SV_DispatchThreadID) {
    if (outside(id)) {
        return;
    }
    let c = b.src.Load(id).rgb;
    let graded = pow(saturate(c * b.params.z + b.params.x), b.params.y);
    let luma = dot(graded, float3(0.2126, 0.7152, 0.0722));
    b.dst.Store(id, float4(lerp(float3(luma), graded, b.params.w), 1.0));
}

/// An unsharp mask of strength `params.x`.
[shader("compute")]
[numthreads(8, 8, 1)]
void sharpen(uint2 id: SV_DispatchThreadID) {
    if (outside(id)) {
        return;
    }
    let last = b.size - 1;
    let c = b.src.Load(id);
    let n = b.src.Load(uint2(id.x, max(id.y, 1) - 1)) + b.src.Load(uint2(id.x, min(id.y + 1, last.y)))
        + b.src.Load(uint2(max(id.x, 1) - 1, id.y)) + b.src.Load(uint2(min(id.x + 1, last.x), id.y));
    b.dst.Store(id, max(c + (c * 4.0 - n) * b.params.x, 0.0));
}

/// A vignette of strength `params.x` and a chromatic aberration of `params.y` texels.
[shader("compute")]
[numthreads(8, 8, 1)]
void vignette(uint2 id: SV_DispatchThreadID) {
    if (outside(id)) {
        return;
    }
    let uv = (float2(id) + 0.5) / float2(b.size);
    let offset = (uv - 0.5) * b.params.y / float2(b.size);
    let r = b.src.SampleLevel(b.sampler, uv + offset, 0.0).r;
    let g = b.src.SampleLevel(b.sampler, uv, 0.0).g;
    let bl = b.src.SampleLevel(b.sampler, uv - offset, 0.0).b;
    let fade = 1.0 - b.params.x * dot(uv - 0.5, uv - 0.5) * 2.0;
    b.dst.Store(id, float4(float3(r, g, bl) * fade, 1.0));
}

/// Exposure `params.x`, then a Reinhard curve.
[shader("compute")]
[numthreads(8, 8, 1)]
void output(uint2 id: SV_DispatchThreadID) {
    if (outside(id)) {
        return;
    }
    let c = b.src.Load(id).rgb * b.params.x;
    b.dst.Store(id, float4(c / (1.0 + c), 1.0));
}

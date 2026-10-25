// Test pass (compute, generated struct layouts): copies a referencing struct of every mapped field type (gaps, a tail, half and 16-bit fields, 64-bit vectors, bool vectors, matrices square and not, an array over 32, handles, a keyword name) into its plain twin, its handles as `uint`s, changing each field, and reads a pointee of handles and one of `float3`s.
#include "bindless.slang"

struct Inner {
    uint64_t wide;
    uint narrow;
}

struct Layouts {
    uint a;
    uint64_t b;
    half h;
    float after_half;
    int16_t s;
    uint8_t u;
    int64_t3 l3;
    uint64_t2 u2;
    bool3 flags;
    float3x3 m33;
    float3x4 m34;
    float arr[40];
    Tex2D<float4> tex;
    DynImg<float4> dyn;
    uint type;
    Inner inner;
    float4 tail_vec;
    uint tail;
}

/// `Layouts` with its handles as plain `uint`s: the GPU writes no references.
struct LayoutsOut {
    uint a;
    uint64_t b;
    half h;
    float after_half;
    int16_t s;
    uint8_t u;
    int64_t3 l3;
    uint64_t2 u2;
    bool3 flags;
    float3x3 m33;
    float3x4 m34;
    float arr[40];
    uint tex;
    uint dyn;
    uint type;
    Inner inner;
    float4 tail_vec;
    uint tail;
}

struct TestLayoutsBindings {
    Buf<Layouts> input;
    MutBuf<LayoutsOut> output;
    Buf<Tex2D<float4>> handles;
    Buf<float3> points;
    MutBuf<float> sums;
    float3x3 rotation;
    half scale;
}

[[vk::push_constant]] TestLayoutsBindings b;

[shader("compute")]
[numthreads(1, 1, 1)]
void test_layouts() {
    var l = b.input.val[0];
    l.a += 1;
    l.b += 1;
    l.h = l.h * b.scale;
    l.after_half += 1.0;
    l.s -= 1;
    l.u += 1;
    l.l3 += int64_t3(1, 2, 3);
    l.u2 *= 2;
    l.flags = !l.flags;
    l.m33 = mul(b.rotation, l.m33);
    l.m34[2][3] += 1.0;
    for (int i = 0; i < 40; i++) {
        l.arr[i] += float(i);
    }
    l.tex = b.handles.val[1];
    l.type += 10;
    l.inner.wide += 1;
    l.inner.narrow += 1;
    l.tail_vec.w += 1.0;
    l.tail += 1;
    LayoutsOut o;
    o.a = l.a;
    o.b = l.b;
    o.h = l.h;
    o.after_half = l.after_half;
    o.s = l.s;
    o.u = l.u;
    o.l3 = l.l3;
    o.u2 = l.u2;
    o.flags = l.flags;
    o.m33 = l.m33;
    o.m34 = l.m34;
    o.arr = l.arr;
    o.tex = l.tex.handle;
    o.dyn = l.dyn.handle;
    o.type = l.type;
    o.inner = l.inner;
    o.tail_vec = l.tail_vec;
    o.tail = l.tail;
    b.output.val[0] = o;
    b.sums.val[0] = b.points.val[0].x + b.points.val[1].y + b.points.val[2].z;
}

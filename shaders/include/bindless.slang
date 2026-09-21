#pragma once
#define NUM_SAMPLERS 2
[vk::binding(0, 0)] SamplerState SamplerHeap[];
[vk::binding(NUM_SAMPLERS, 0)] __DynamicResource TextureHeap[];
[vk::binding(0, 1)] __DynamicResource RWTextureHeap[];

public static const int FMT_INFER    = 0;
public static const int FMT_RGBA32F  = 1;
public static const int FMT_RGBA16F  = 2;
public static const int FMT_RG32F    = 3;
public static const int FMT_R32F     = 6;
public static const int FMT_RGBA8    = 10;
public static const int FMT_R32UI    = 37;

public struct Img<E : ITexelElement, Shape : __ITextureShape, let U : int, let F : int> {
    public uint32_t SampledIndex;
    public uint32_t StorageIndex;
}

__generic<E : ITexelElement, Shape : __ITextureShape, let F : int>
public extension Img<E, Shape, 0, F> {
    typealias T = _Texture<E, Shape, 0, 0, 0, 0, 0, 0, F>;

    [ForceInline] public E Load(vector<uint, Shape.dimensions> p) {
        return TextureHeap[SampledIndex].asOpaqueDescriptor<T>()[p];
    }
    [ForceInline] public E Sample(vector<float, Shape.dimensions> uv, uint sampler = 0) {
        return TextureHeap[SampledIndex].asOpaqueDescriptor<T>().Sample(SamplerHeap[sampler], uv);
    }
}

__generic<E : ITexelElement, Shape : __ITextureShape, let F : int>
public extension Img<E, Shape, 1, F> {
    typealias T = _Texture<E, Shape, 0, 0, 0, 1, 0, 0, F>;

    [ForceInline] public E Load(vector<uint, Shape.dimensions> p) {
        return RWTextureHeap[StorageIndex].asOpaqueDescriptor<T>()[p];
    }
}

__generic<E : ITexelElement, Shape : __ITextureShape, let F : int>
public extension Img<E, Shape, 2, F> {
    typealias T = _Texture<E, Shape, 0, 0, 0, 1, 0, 0, F>;

    [ForceInline] public E Load(vector<uint, Shape.dimensions> p) {
        return RWTextureHeap[StorageIndex].asOpaqueDescriptor<T>()[p];
    }
    [ForceInline] public void Store(vector<uint, Shape.dimensions> p, E v) {
        RWTextureHeap[StorageIndex].asOpaqueDescriptor<T>().Store(p, v);
    }
}

public typealias Tex2D<E : ITexelElement = float4, let F : int = 0>      = Img<E, __Shape2D, 0, F>;
public typealias Tex3D<E : ITexelElement = float4, let F : int = 0>      = Img<E, __Shape3D, 0, F>;
public typealias Image2D<E : ITexelElement = float4, let F : int = 0>    = Img<E, __Shape2D, 1, F>;
public typealias Image3D<E : ITexelElement = float4, let F : int = 0>    = Img<E, __Shape3D, 1, F>;
public typealias MutImage2D<E : ITexelElement = float4, let F : int = 0> = Img<E, __Shape2D, 2, F>;
public typealias MutImage3D<E : ITexelElement = float4, let F : int = 0> = Img<E, __Shape3D, 2, F>;

public struct Buf<T>    { public T *val; }
public struct MutBuf<T> { public T *val; }

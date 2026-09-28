#pragma once
#define NUM_SAMPLERS 2
[vk::binding(0, 0)] SamplerState SamplerHeap[];
[vk::binding(NUM_SAMPLERS, 0)] __DynamicResource TextureHeap[];
[vk::binding(0, 1)] __DynamicResource RWTextureHeap[];


public struct Img<E : ITexelElement, Shape : __ITextureShape, let U : int> {
    public uint32_t SampledIndex;
    public uint32_t StorageIndex;
}

__generic<E : ITexelElement, Shape : __ITextureShape>
public extension Img<E, Shape, 0> {
    typealias T = _Texture<E, Shape, 0, 0, 0, 0, 0, 0, 0>;

    [ForceInline] public E Load(vector<uint, Shape.dimensions> p) {
        return TextureHeap[SampledIndex].asOpaqueDescriptor<T>()[p];
    }
    [ForceInline] public E Sample(vector<float, Shape.dimensions> uv, uint sampler = 0) {
        return TextureHeap[SampledIndex].asOpaqueDescriptor<T>().Sample(SamplerHeap[sampler], uv);
    }
}

__generic<E : ITexelElement, Shape : __ITextureShape>
public extension Img<E, Shape, 1> {
    typealias T = _Texture<E, Shape, 0, 0, 0, 1, 0, 0, 0>;

    [ForceInline] public E Load(vector<uint, Shape.dimensions> p) {
        return RWTextureHeap[StorageIndex].asOpaqueDescriptor<T>()[p];
    }
}

__generic<E : ITexelElement, Shape : __ITextureShape>
public extension Img<E, Shape, 2> {
    typealias T = _Texture<E, Shape, 0, 0, 0, 1, 0, 0, 0>;

    [ForceInline] public E Load(vector<uint, Shape.dimensions> p) {
        return RWTextureHeap[StorageIndex].asOpaqueDescriptor<T>()[p];
    }
    [ForceInline] public void Store(vector<uint, Shape.dimensions> p, E v) {
        RWTextureHeap[StorageIndex].asOpaqueDescriptor<T>().Store(p, v);
    }
}

public typealias Tex2D<E : ITexelElement = float4>      = Img<E, __Shape2D, 0>;
public typealias Tex3D<E : ITexelElement = float4>      = Img<E, __Shape3D, 0>;
public typealias Image2D<E : ITexelElement = float4>    = Img<E, __Shape2D, 1>;
public typealias Image3D<E : ITexelElement = float4>    = Img<E, __Shape3D, 1>;
public typealias MutImage2D<E : ITexelElement = float4> = Img<E, __Shape2D, 2>;
public typealias MutImage3D<E : ITexelElement = float4> = Img<E, __Shape3D, 2>;

public struct Buf<T>    { public T *val; }
public struct MutBuf<T> { public T *val; }

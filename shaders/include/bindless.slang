#pragma once
#define NUM_SAMPLERS 2
[vk::binding(0, 0)] SamplerState SamplerHeap[];
[vk::binding(NUM_SAMPLERS, 0)] __DynamicResource TextureHeap[];
[vk::binding(0, 1)] __DynamicResource RWTextureHeap[];

interface IImageDescriptor {
    static const bool isWriteable;
};

__generic<E : ITexelElement, Shape : __ITextureShape, int isArray, int isMS, int sampleCount, int access, int isShadow, int format>
    extension _Texture<E, Shape, isArray, isMS, sampleCount, access, isShadow, 0, format> : IImageDescriptor {
    static const bool isWriteable = access == 1;
};

public struct Image<T : IImageDescriptor & __IDynamicResourceCastable> {
    uint32_t pad;
    uint32_t DescriptorIndex;
    __init(uint32_t i)
    {
        pad = 0;
        DescriptorIndex = i;
    }
}

public struct Texture<T : IImageDescriptor & __IDynamicResourceCastable> {
    uint32_t DescriptorIndex;
    uint32_t pad;
    __init(uint32_t i)
    {
        pad = 0;
        DescriptorIndex = i;
    }
}

public struct MutImage<T : IImageDescriptor & __IDynamicResourceCastable> {
    uint32_t pad;
    uint32_t DescriptorIndex;
    __init(uint32_t i)
    {
        pad = 0;
        DescriptorIndex = i;
    }
}

public struct MutTexture<T : IImageDescriptor & __IDynamicResourceCastable> {
    uint32_t DescriptorIndex;
    uint32_t pad;
    __init(uint32_t i)
    {
        pad = 0;
        DescriptorIndex = i;
    }
}

__generic<E : ITexelElement, Shape : __ITextureShape, let format : int>
public extension Image<_Texture<E, Shape, 0, 0, 0, 1, 0, 0, format>> {
    typealias T = _Texture<E, Shape, 0, 0, 0, 1, 0, 0, format>;

    [ForceInline]
    public E Load(vector<uint, Shape.dimensions> pos) {
        return RWTextureHeap[DescriptorIndex].asOpaqueDescriptor<T>()[pos];
    }
}

__generic<E : ITexelElement, Shape : __ITextureShape, let format : int>
public extension MutImage<_Texture<E, Shape, 0, 0, 0, 1, 0, 0, format>> {
    typealias T = _Texture<E, Shape, 0, 0, 0, 1, 0, 0, format>;

    [ForceInline]
    public E Load(vector<uint, Shape.dimensions> pos) {
        return RWTextureHeap[DescriptorIndex].asOpaqueDescriptor<T>()[pos];
    }

    [ForceInline]
    public void Store(vector<uint, Shape.dimensions> pos, E value) {
        RWTextureHeap[DescriptorIndex].asOpaqueDescriptor<T>().Store(pos, value);
    }
}

__generic<E : ITexelElement, Shape : __ITextureShape, let format : int>
public extension Texture<_Texture<E, Shape, 0, 0, 0, 0, 0, 0, format>> {
    typealias T = _Texture<E, Shape, 0, 0, 0, 0, 0, 0, format>;

    [ForceInline]
    public E Load(vector<uint, Shape.dimensions> pos) {
        return TextureHeap[DescriptorIndex].asOpaqueDescriptor<T>()[pos];
    }

    [ForceInline]
    public E Sample(vector<float, Shape.dimensions> pos) {
        return TextureHeap[DescriptorIndex].asOpaqueDescriptor<T>().Sample(SamplerHeap[0], pos);
    }
}

public typealias Image2D<F : ITexelElement = float4> = Image<RWTexture2D<F>>;
public typealias Image3D<F : ITexelElement = float4> = Image<RWTexture3D<F>>;

public typealias MutImage2D<F : ITexelElement = float4> = MutImage<RWTexture2D<F>>;
public typealias MutImage3D<F : ITexelElement = float4> = MutImage<RWTexture3D<F>>;

public typealias Tex2D<F : ITexelElement = float4> = Texture<Texture2D<F>>;
public typealias Tex3D<F : ITexelElement = float4> = Texture<Texture3D<F>>;

public struct Buf<T> {
    public T *val;
}

public struct MutBuf<T> {
    public T *val;
}

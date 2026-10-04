#pragma once
#define NUM_SAMPLERS 3
[vk::binding(0, 0)] SamplerState SamplerHeap[];
[vk::binding(NUM_SAMPLERS, 0)] __DynamicResource TextureHeap[];
[vk::binding(0, 1)] __DynamicResource RWTextureHeap[];

/// Index of an unset bindless slot; matches `lava::bindless::NULL_HANDLE`.
public static const uint32_t NULL_HANDLE = 0xFFFFFFFF;


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
        return TextureHeap[NonUniformResourceIndex(SampledIndex)].asOpaqueDescriptor<T>().Sample(SamplerHeap[sampler], uv);
    }
}

/// Size in texels of mip 0, for every kind of `Img`: read through the index its usage has.
__generic<E : ITexelElement, let U : int>
public extension Img<E, __Shape2D, U> {
    public uint2 GetDimensions() {
        uint width;
        uint height;
        if (U == 0) {
            TextureHeap[NonUniformResourceIndex(SampledIndex)].asOpaqueDescriptor<_Texture<E, __Shape2D, 0, 0, 0, 0, 0, 0, 0>>().GetDimensions(width, height);
        } else {
            RWTextureHeap[NonUniformResourceIndex(StorageIndex)].asOpaqueDescriptor<_Texture<E, __Shape2D, 0, 0, 0, 1, 0, 0, 0>>().GetDimensions(width, height);
        }
        return uint2(width, height);
    }
}

__generic<E : ITexelElement, let U : int>
public extension Img<E, __Shape3D, U> {
    public uint3 GetDimensions() {
        uint width;
        uint height;
        uint depth;
        if (U == 0) {
            TextureHeap[NonUniformResourceIndex(SampledIndex)].asOpaqueDescriptor<_Texture<E, __Shape3D, 0, 0, 0, 0, 0, 0, 0>>().GetDimensions(width, height, depth);
        } else {
            RWTextureHeap[NonUniformResourceIndex(StorageIndex)].asOpaqueDescriptor<_Texture<E, __Shape3D, 0, 0, 0, 1, 0, 0, 0>>().GetDimensions(width, height, depth);
        }
        return uint3(width, height, depth);
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

/// A 2D image whose kind is only known at runtime, e.g. a handle stored in vertex data.
/// Reads the sampled image if the handle has one, the storage image otherwise.
public struct DynImg<E : ITexelElement = float4> {
    public uint32_t SampledIndex;
    public uint32_t StorageIndex;

    typealias Sampled = _Texture<E, __Shape2D, 0, 0, 0, 0, 0, 0, 0>;
    typealias Storage = _Texture<E, __Shape2D, 0, 0, 0, 1, 0, 0, 0>;

    /// Size in texels, or zero for a null image.
    public uint2 GetDimensions() {
        uint width = 0;
        uint height = 0;
        if (SampledIndex != NULL_HANDLE) {
            TextureHeap[NonUniformResourceIndex(SampledIndex)].asOpaqueDescriptor<Sampled>().GetDimensions(width, height);
        } else if (StorageIndex != NULL_HANDLE) {
            RWTextureHeap[NonUniformResourceIndex(StorageIndex)].asOpaqueDescriptor<Storage>().GetDimensions(width, height);
        }
        return uint2(width, height);
    }

    /// True if neither index is set; `Sample` then returns zero.
    [ForceInline] public bool IsNull() {
        return SampledIndex == NULL_HANDLE && StorageIndex == NULL_HANDLE;
    }

    /// Filtered sample for sampled images, nearest texel for storage images.
    public E Sample(float2 uv, uint sampler = 0) {
        if (SampledIndex != NULL_HANDLE) {
            return TextureHeap[NonUniformResourceIndex(SampledIndex)].asOpaqueDescriptor<Sampled>().Sample(SamplerHeap[sampler], uv);
        } else if (StorageIndex != NULL_HANDLE) {
            let image = RWTextureHeap[NonUniformResourceIndex(StorageIndex)].asOpaqueDescriptor<Storage>();
            uint width;
            uint height;
            image.GetDimensions(width, height);
            let size = uint2(width, height);
            return image[min(uint2(saturate(uv) * float2(size)), size - 1)];
        } else {
            E zero;
            return zero;
        }
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

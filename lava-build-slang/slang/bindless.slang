#pragma once
// Shaders index lava's descriptor heaps directly (`SPV_EXT_descriptor_heap`): the resource heap
// holds the sampled images, then the storage images, and the sampler heap the samplers
// (`lava::bindless::Sampler`). A heap index needs no `NonUniformResourceIndex`: heap accesses
// are non-uniform unless declared otherwise.

/// Index of an unset bindless slot; matches `lava::bindless::NULL_HANDLE`.
public static const uint32_t NULL_HANDLE = 0xFFFFFFFF;

/// A sampler the application created (`lava::bindless::Sampler`), by its slot in the sampler
/// heap: lava's `SamplerHandle`.
public struct Sampler {
    public uint32_t index;

    [ForceInline] internal SamplerState get() {
        SamplerState sampler = SamplerDescriptorHeap[index];
        return sampler;
    }
    [ForceInline] internal SamplerComparisonState getCmp() {
        SamplerComparisonState sampler = SamplerDescriptorHeap[index];
        return sampler;
    }
}

public struct Img<E : ITexelElement, Shape : __ITextureShape, let U : int> {
    public uint32_t SampledIndex;
    public uint32_t StorageIndex;
}

/// The descriptors behind an `Img`'s indices.
__generic<E : ITexelElement, Shape : __ITextureShape, let U : int>
public extension Img<E, Shape, U> {
    typealias SampledTexture = _Texture<E, Shape, 0, 0, 0, 0, 0, 0, 0>;
    typealias StorageTexture = _Texture<E, Shape, 0, 0, 0, 1, 0, 0, 0>;

    [ForceInline] internal SampledTexture sampled() {
        SampledTexture texture = ResourceDescriptorHeap[SampledIndex];
        return texture;
    }
    [ForceInline] internal StorageTexture storage() {
        StorageTexture texture = ResourceDescriptorHeap[StorageIndex];
        return texture;
    }
}

__generic<E : ITexelElement, Shape : __ITextureShape>
public extension Img<E, Shape, 0> {
    [ForceInline] public E Load(vector<uint, Shape.dimensions> p) {
        return sampled()[p];
    }
    [ForceInline] public E Sample(Sampler sampler, vector<float, Shape.dimensions> uv) {
        return sampled().Sample(sampler.get(), uv);
    }
    /// Sample at an explicit mip level, which compute shaders need.
    [ForceInline] public E SampleLevel(Sampler sampler, vector<float, Shape.dimensions> uv, float level) {
        return sampled().SampleLevel(sampler.get(), uv, level);
    }
    /// With a comparison sampler (`SamplerDesc::compare`): the filtered share of texels that
    /// pass `reference op texel` (percentage-closer filtering).
    [ForceInline] public float SampleCmp(Sampler sampler, vector<float, Shape.dimensions> uv, float reference) {
        return sampled().SampleCmp(sampler.getCmp(), uv, reference);
    }
    /// `SampleCmp` at mip 0, which compute shaders need.
    [ForceInline] public float SampleCmpLevelZero(Sampler sampler, vector<float, Shape.dimensions> uv, float reference) {
        return sampled().SampleCmpLevelZero(sampler.getCmp(), uv, reference);
    }
}

/// Size in texels of mip 0, for every kind of `Img`: read through the index its usage has.
__generic<E : ITexelElement, let U : int>
public extension Img<E, __Shape2D, U> {
    public uint2 GetDimensions() {
        uint width;
        uint height;
        if (U == 0) {
            sampled().GetDimensions(width, height);
        } else {
            storage().GetDimensions(width, height);
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
            sampled().GetDimensions(width, height, depth);
        } else {
            storage().GetDimensions(width, height, depth);
        }
        return uint3(width, height, depth);
    }
}

__generic<E : ITexelElement, Shape : __ITextureShape>
public extension Img<E, Shape, 1> {
    [ForceInline] public E Load(vector<uint, Shape.dimensions> p) {
        return storage()[p];
    }
}

__generic<E : ITexelElement, Shape : __ITextureShape>
public extension Img<E, Shape, 2> {
    [ForceInline] public E Load(vector<uint, Shape.dimensions> p) {
        return storage()[p];
    }
    [ForceInline] public void Store(vector<uint, Shape.dimensions> p, E v) {
        storage().Store(p, v);
    }
}

/// Bound in both sets: reads go through the sampler, which fetches far faster than a typed
/// load of a storage image does on Intel, and writes through the storage descriptor.
__generic<E : ITexelElement, Shape : __ITextureShape>
public extension Img<E, Shape, 3> {
    [ForceInline] public E Load(vector<uint, Shape.dimensions> p) {
        return sampled()[p];
    }
    [ForceInline] public void Store(vector<uint, Shape.dimensions> p, E v) {
        storage().Store(p, v);
    }
}

/// A 2D image whose kind is only known at runtime, e.g. a handle stored in vertex data.
/// Reads the sampled image if the handle has one, the storage image otherwise.
public struct DynImg<E : ITexelElement = float4> {
    public uint32_t SampledIndex;
    public uint32_t StorageIndex;

    typealias Sampled = _Texture<E, __Shape2D, 0, 0, 0, 0, 0, 0, 0>;
    typealias Storage = _Texture<E, __Shape2D, 0, 0, 0, 1, 0, 0, 0>;

    [ForceInline] internal Sampled sampled() {
        Sampled texture = ResourceDescriptorHeap[SampledIndex];
        return texture;
    }
    [ForceInline] internal Storage storage() {
        Storage texture = ResourceDescriptorHeap[StorageIndex];
        return texture;
    }

    /// Size in texels, or zero for a null image.
    public uint2 GetDimensions() {
        uint width = 0;
        uint height = 0;
        if (SampledIndex != NULL_HANDLE) {
            sampled().GetDimensions(width, height);
        } else if (StorageIndex != NULL_HANDLE) {
            storage().GetDimensions(width, height);
        }
        return uint2(width, height);
    }

    /// True if neither index is set; `Sample` then returns zero.
    [ForceInline] public bool IsNull() {
        return SampledIndex == NULL_HANDLE && StorageIndex == NULL_HANDLE;
    }

    /// Filtered sample for sampled images, nearest texel for storage images.
    public E Sample(Sampler sampler, float2 uv) {
        if (SampledIndex != NULL_HANDLE) {
            return sampled().Sample(sampler.get(), uv);
        } else if (StorageIndex != NULL_HANDLE) {
            let image = storage();
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
public typealias MutTex2D<E : ITexelElement = float4>   = Img<E, __Shape2D, 3>;

/// A buffer the pass only reads, which lava tracks as a read: its pointer is read-only, so a
/// write through it doesn't compile. `MutBuf` is read and written.
public struct Buf<T>    { public Ptr<T, Access.Read> val; }
public struct MutBuf<T> { public T *val; }

/// A top-level acceleration structure by its device address: lava's `&Tlas` in push constants,
/// a `u64` inside structs (register it with `BindingOutput::acceleration_structure_read`). Never
/// an `SCONST`: addresses change from run to run.
public struct Accel {
    public uint64_t address;
    [ForceInline] public RaytracingAccelerationStructure get() {
        return RaytracingAccelerationStructure(address);
    }
}

/// Marks the `uint64_t` behind an `SCONST`; lava-build-slang reads the type from it.
[__AttributeUsage(_AttributeTargets.Var)]
public struct SConstAttribute { string type; };

/// A specialization constant of an 8-byte handle type (`Buf<T>`, `MutBuf<T>`, any `Img`,
/// `DynImg`): lava bakes it into the pipeline, which the generated `constant_bindings` sets.
/// Scalars need no macro: `[SpecializationConstant] const uint name = 0;`. Every distinct value
/// builds its own pipeline, so only use them for values that never change.
#define SCONST(T, name) [SpecializationConstant][SConst(#T)] const uint64_t name##_sconst = 0; static T name = bit_cast<T>(name##_sconst);

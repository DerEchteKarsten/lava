#pragma once
// Shaders index lava's descriptor heaps directly (`SPV_EXT_descriptor_heap`): the resource heap
// holds image descriptors, the sampler heap the samplers (`lava::bindless::Sampler`). A heap
// index needs no `NonUniformResourceIndex`: heap accesses are non-uniform unless declared
// otherwise.
//
// An image handle (`lava::bindless::ImageHandle`) is one `uint32_t`: bits 0 to 29 are the slot
// of its first descriptor, bit 30 that descriptor's kind (0 sampled, 1 storage), bit 31 how many
// there are (0 one, 1 two: sampled at the slot, storage at the next). `NULL_HANDLE`, zero
// (sampled slot 0, which lava never hands out), names nothing, so zeroed structs hold null
// handles; a zeroed `Sampler` is slot 0 of the sampler heap, lava's default sampler.

/// The handle of no image; matches `lava::bindless::ImageHandle::NULL`.
public static const uint32_t NULL_HANDLE = 0;
static const uint32_t HANDLE_SLOT = 0x3FFFFFFF;

/// The slot of a handle's sampled descriptor, if `handleHasSampled`.
[ForceInline] internal uint32_t handleSampledSlot(uint32_t handle) {
    return handle & HANDLE_SLOT;
}
/// The slot of a handle's storage descriptor, if `handleHasStorage`.
[ForceInline] internal uint32_t handleStorageSlot(uint32_t handle) {
    return (handle & HANDLE_SLOT) + (handle >> 31);
}
[ForceInline] internal bool handleHasSampled(uint32_t handle) {
    return handle != NULL_HANDLE && (handle & (1u << 30)) == 0;
}
[ForceInline] internal bool handleHasStorage(uint32_t handle) {
    return handle != NULL_HANDLE && (handle >> 30) != 0;
}

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

/// An image the pass reads through the descriptors its kind `U` names: 0 sampled (`Tex2D`), 1
/// and 2 storage (`Image2D`, `MutImage2D`), 3 both (`MutTex2D`).
public struct Img<E : ITexelElement, Shape : __ITextureShape, let U : int> {
    public uint32_t handle;

    [ForceInline] public __init(uint32_t handle) {
        this.handle = handle;
    }

    /// The image behind an `SCONST`.
    [ForceInline] public static Img<E, Shape, U> fromConst(uint64_t value) {
        return Img<E, Shape, U>(uint32_t(value));
    }

    [ForceInline] public bool IsNull() {
        return handle == NULL_HANDLE;
    }

    /// Mip `level` of a mip chain this is mip 0 of (`bind_mips_*`), whose handles lie the
    /// handle's width apart.
    [ForceInline] public Img<E, Shape, U> Mip(uint32_t level) {
        let slot = (handle & HANDLE_SLOT) + level * (1 + (handle >> 31));
        return Img<E, Shape, U>((handle & ~HANDLE_SLOT) | slot);
    }

    typealias SampledTexture = _Texture<E, Shape, 0, 0, 0, 0, 0, 0, 0>;
    typealias StorageTexture = _Texture<E, Shape, 0, 0, 0, 1, 0, 0, 0>;

    [ForceInline] internal SampledTexture sampled() {
        SampledTexture texture = ResourceDescriptorHeap[handleSampledSlot(handle)];
        return texture;
    }
    [ForceInline] internal StorageTexture storage() {
        StorageTexture texture = ResourceDescriptorHeap[handleStorageSlot(handle)];
        return texture;
    }
}

__generic<E : ITexelElement, Shape : __ITextureShape>
public extension Img<E, Shape, 0> {
    /// The texel at `p` of the first mip the handle names (`Mip(level)` for another). A null
    /// handle reads an undefined value: check `IsNull()` where one can be null.
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
    /// Sample with explicit derivatives of `uv`, which pick the mip.
    [ForceInline] public E SampleGrad(Sampler sampler, vector<float, Shape.dimensions> uv, vector<float, Shape.dimensions> ddx, vector<float, Shape.dimensions> ddy) {
        return sampled().SampleGrad(sampler.get(), uv, ddx, ddy);
    }
    /// Sample at the mip the derivatives pick plus `bias` (fragment shaders).
    [ForceInline] public E SampleBias(Sampler sampler, vector<float, Shape.dimensions> uv, float bias) {
        return sampled().SampleBias(sampler.get(), uv, bias);
    }
}

/// Loads and gathers of 2D images read through the sampled descriptor.
__generic<E : ITexelElement>
public extension Img<E, __Shape2D, 0> {
    /// The texel at `p` of mip `level`.
    [ForceInline] public E Load(uint2 p, uint level) {
        return sampled().Load(int3(int2(p), int(level)));
    }
    /// The red components of the four texels bilinear filtering at `uv` reads (mip 0), in the
    /// order (-, +), (+, +), (+, -), (-, -) of their offsets from `uv`.
    [ForceInline] public vector<E.Element, 4> Gather(Sampler sampler, float2 uv) {
        return sampled().Gather(sampler.get(), uv);
    }
    [ForceInline] public vector<E.Element, 4> GatherRed(Sampler sampler, float2 uv) {
        return sampled().GatherRed(sampler.get(), uv);
    }
    [ForceInline] public vector<E.Element, 4> GatherGreen(Sampler sampler, float2 uv) {
        return sampled().GatherGreen(sampler.get(), uv);
    }
    [ForceInline] public vector<E.Element, 4> GatherBlue(Sampler sampler, float2 uv) {
        return sampled().GatherBlue(sampler.get(), uv);
    }
    [ForceInline] public vector<E.Element, 4> GatherAlpha(Sampler sampler, float2 uv) {
        return sampled().GatherAlpha(sampler.get(), uv);
    }
}

__generic<E : ITexelElement>
public extension Img<E, __Shape3D, 0> {
    /// The texel at `p` of mip `level`.
    [ForceInline] public E Load(uint3 p, uint level) {
        return sampled().Load(int4(int3(p), int(level)));
    }
}

/// Size in texels of mip 0, for every kind of `Img`: read through the descriptor it reads.
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

/// Bound with both descriptors (`bind_combined`): reads go through the sampled one, which
/// fetches far faster than a typed load of a storage image does on Intel, and writes through
/// the storage one.
__generic<E : ITexelElement, Shape : __ITextureShape>
public extension Img<E, Shape, 3> {
    [ForceInline] public E Load(vector<uint, Shape.dimensions> p) {
        return sampled()[p];
    }
    [ForceInline] public void Store(vector<uint, Shape.dimensions> p, E v) {
        storage().Store(p, v);
    }
    [ForceInline] public E Sample(Sampler sampler, vector<float, Shape.dimensions> uv) {
        return sampled().Sample(sampler.get(), uv);
    }
    [ForceInline] public E SampleLevel(Sampler sampler, vector<float, Shape.dimensions> uv, float level) {
        return sampled().SampleLevel(sampler.get(), uv, level);
    }
}

// Atomics on the texels of `uint` and `int` images written through their storage descriptor
// (`MutImage2D<uint>`, `MutTex2D<int>`, ...; an `R32Uint` or `R32Sint` image, the format SPIR-V
// atomics need declared: 37 and 27 are Slang's `r32ui` and `r32i`): each returns the texel's
// value before it.
#define LAVA_IMAGE_ATOMICS(T, U, FORMAT) \
__generic<Shape : __ITextureShape> \
public extension Img<T, Shape, U> { \
    typealias AtomicTexture = _Texture<T, Shape, 0, 0, 0, 1, 0, 0, FORMAT>; \
    [ForceInline] internal AtomicTexture atomic() { \
        AtomicTexture texture = ResourceDescriptorHeap[handleStorageSlot(handle)]; \
        return texture; \
    } \
    [ForceInline] public T InterlockedAdd(vector<uint, Shape.dimensions> p, T value) { \
        T original; ::InterlockedAdd(atomic()[p], value, original); return original; \
    } \
    [ForceInline] public T InterlockedMin(vector<uint, Shape.dimensions> p, T value) { \
        T original; ::InterlockedMin(atomic()[p], value, original); return original; \
    } \
    [ForceInline] public T InterlockedMax(vector<uint, Shape.dimensions> p, T value) { \
        T original; ::InterlockedMax(atomic()[p], value, original); return original; \
    } \
    [ForceInline] public T InterlockedAnd(vector<uint, Shape.dimensions> p, T value) { \
        T original; ::InterlockedAnd(atomic()[p], value, original); return original; \
    } \
    [ForceInline] public T InterlockedOr(vector<uint, Shape.dimensions> p, T value) { \
        T original; ::InterlockedOr(atomic()[p], value, original); return original; \
    } \
    [ForceInline] public T InterlockedXor(vector<uint, Shape.dimensions> p, T value) { \
        T original; ::InterlockedXor(atomic()[p], value, original); return original; \
    } \
    [ForceInline] public T InterlockedExchange(vector<uint, Shape.dimensions> p, T value) { \
        T original; ::InterlockedExchange(atomic()[p], value, original); return original; \
    } \
    /** Writes `value` if the texel is `compare`. */ \
    [ForceInline] public T InterlockedCompareExchange(vector<uint, Shape.dimensions> p, T compare, T value) { \
        T original; ::InterlockedCompareExchange(atomic()[p], compare, value, original); return original; \
    } \
}
LAVA_IMAGE_ATOMICS(uint, 2, 37)
LAVA_IMAGE_ATOMICS(int, 2, 27)
LAVA_IMAGE_ATOMICS(uint, 3, 37)
LAVA_IMAGE_ATOMICS(int, 3, 27)

/// A 2D array image (`lava::image::D2Array`, every layer of the view it was bound from) the pass
/// reads through the descriptors its kind `U` names, like `Img`: 0 sampled (`Tex2DArray`), 1 and
/// 2 storage (`Image2DArray`, `MutImage2DArray`). Coordinates end with the layer.
public struct ImgArray<E : ITexelElement, let U : int> {
    public uint32_t handle;

    [ForceInline] public __init(uint32_t handle) {
        this.handle = handle;
    }

    /// The image behind an `SCONST`.
    [ForceInline] public static ImgArray<E, U> fromConst(uint64_t value) {
        return ImgArray<E, U>(uint32_t(value));
    }

    [ForceInline] public bool IsNull() {
        return handle == NULL_HANDLE;
    }

    typealias SampledTexture = _Texture<E, __Shape2D, 1, 0, 0, 0, 0, 0, 0>;
    typealias StorageTexture = _Texture<E, __Shape2D, 1, 0, 0, 1, 0, 0, 0>;

    [ForceInline] internal SampledTexture sampled() {
        SampledTexture texture = ResourceDescriptorHeap[handleSampledSlot(handle)];
        return texture;
    }
    [ForceInline] internal StorageTexture storage() {
        StorageTexture texture = ResourceDescriptorHeap[handleStorageSlot(handle)];
        return texture;
    }

    /// Size in texels of mip 0 and the layer count, read through the descriptor it reads.
    public uint3 GetDimensions() {
        uint width;
        uint height;
        uint layers;
        if (U == 0) {
            sampled().GetDimensions(width, height, layers);
        } else {
            storage().GetDimensions(width, height, layers);
        }
        return uint3(width, height, layers);
    }
}

__generic<E : ITexelElement>
public extension ImgArray<E, 0> {
    /// The texel at `p.xy` of layer `p.z`.
    [ForceInline] public E Load(uint3 p) {
        return sampled()[p];
    }
    /// The texel at `p.xy` of layer `p.z`, mip `level`.
    [ForceInline] public E Load(uint3 p, uint level) {
        return sampled().Load(int4(int3(p), int(level)));
    }
    /// Samples layer `uvl.z` at `uvl.xy`.
    [ForceInline] public E Sample(Sampler sampler, float3 uvl) {
        return sampled().Sample(sampler.get(), uvl);
    }
    [ForceInline] public E SampleLevel(Sampler sampler, float3 uvl, float level) {
        return sampled().SampleLevel(sampler.get(), uvl, level);
    }
    [ForceInline] public float SampleCmp(Sampler sampler, float3 uvl, float reference) {
        return sampled().SampleCmp(sampler.getCmp(), uvl, reference);
    }
    [ForceInline] public float SampleCmpLevelZero(Sampler sampler, float3 uvl, float reference) {
        return sampled().SampleCmpLevelZero(sampler.getCmp(), uvl, reference);
    }
}

__generic<E : ITexelElement>
public extension ImgArray<E, 1> {
    [ForceInline] public E Load(uint3 p) {
        return storage()[p];
    }
}

__generic<E : ITexelElement>
public extension ImgArray<E, 2> {
    [ForceInline] public E Load(uint3 p) {
        return storage()[p];
    }
    [ForceInline] public void Store(uint3 p, E v) {
        storage().Store(p, v);
    }
}

/// A 2D image whose kind is only known at runtime, e.g. a handle stored in vertex data.
/// Reads the sampled descriptor if the handle has one, the storage descriptor otherwise.
public struct DynImg<E : ITexelElement = float4> {
    public uint32_t handle;

    [ForceInline] public __init(uint32_t handle) {
        this.handle = handle;
    }

    typealias Sampled = _Texture<E, __Shape2D, 0, 0, 0, 0, 0, 0, 0>;
    typealias Storage = _Texture<E, __Shape2D, 0, 0, 0, 1, 0, 0, 0>;

    [ForceInline] internal Sampled sampled() {
        Sampled texture = ResourceDescriptorHeap[handleSampledSlot(handle)];
        return texture;
    }
    [ForceInline] internal Storage storage() {
        Storage texture = ResourceDescriptorHeap[handleStorageSlot(handle)];
        return texture;
    }

    /// Size in texels, or zero for a null image.
    public uint2 GetDimensions() {
        uint width = 0;
        uint height = 0;
        if (handleHasSampled(handle)) {
            sampled().GetDimensions(width, height);
        } else if (handleHasStorage(handle)) {
            storage().GetDimensions(width, height);
        }
        return uint2(width, height);
    }

    /// True for `NULL_HANDLE`; `Sample` then returns zero.
    [ForceInline] public bool IsNull() {
        return handle == NULL_HANDLE;
    }

    /// Filtered sample for sampled images, nearest texel for storage images.
    public E Sample(Sampler sampler, float2 uv) {
        if (handleHasSampled(handle)) {
            return sampled().Sample(sampler.get(), uv);
        }
        return nearest(uv);
    }

    /// `Sample` at an explicit mip level of a sampled image, which compute shaders need.
    public E SampleLevel(Sampler sampler, float2 uv, float level) {
        if (handleHasSampled(handle)) {
            return sampled().SampleLevel(sampler.get(), uv, level);
        }
        return nearest(uv);
    }

    /// The storage image's texel at `uv`, or zero without one.
    internal E nearest(float2 uv) {
        if (handleHasStorage(handle)) {
            let image = storage();
            uint width;
            uint height;
            image.GetDimensions(width, height);
            let size = uint2(width, height);
            return image[min(uint2(saturate(uv) * float2(size)), size - 1)];
        } else {
            return E(E.Element(0));
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
public typealias MutTex3D<E : ITexelElement = float4>   = Img<E, __Shape3D, 3>;
public typealias Tex2DArray<E : ITexelElement = float4>      = ImgArray<E, 0>;
public typealias Image2DArray<E : ITexelElement = float4>    = ImgArray<E, 1>;
public typealias MutImage2DArray<E : ITexelElement = float4> = ImgArray<E, 2>;

/// A buffer the pass only reads, which lava tracks as a read: its pointer is read-only, so a
/// write through it doesn't compile. `MutBuf` is read and written.
public struct Buf<T> {
    public Ptr<T, Access.Read> val;

    /// The buffer behind an `SCONST`.
    [ForceInline] public static Buf<T> fromConst(uint64_t value) {
        return bit_cast<Buf<T>>(value);
    }
}
public struct MutBuf<T> {
    public T *val;

    /// The buffer behind an `SCONST`.
    [ForceInline] public static MutBuf<T> fromConst(uint64_t value) {
        return bit_cast<MutBuf<T>>(value);
    }
}

/// A top-level acceleration structure by its device address: lava's `&Tlas` in push constants
/// and `SCONST`s, a `u64` inside structs (pass `BufferAccess::trace(&tlas)` in the bindings'
/// `Accesses`).
public struct Accel {
    public uint64_t address;

    /// The structure behind an `SCONST`.
    [ForceInline] public static Accel fromConst(uint64_t value) {
        Accel accel;
        accel.address = value;
        return accel;
    }
    [ForceInline] public RaytracingAccelerationStructure get() {
        return RaytracingAccelerationStructure(address);
    }
}

/// Exports a `static const` scalar (`bool`, `int`, `uint`, `int64_t`, `uint64_t`, `float`) of a
/// pass or its includes to Rust: lava-build-slang generates `pub const NAME: T = value;` beside
/// the passes, so shaders and the application share one value
/// (`[RustConst] static const uint MESHLET_STRIDE = 64;`).
[__AttributeUsage(_AttributeTargets.Var)]
public struct RustConstAttribute {};

/// Marks the `uint64_t` behind an `SCONST`; lava-build-slang reads the type from it.
[__AttributeUsage(_AttributeTargets.Var)]
public struct SConstAttribute { string type; };

/// A specialization constant of a handle type (`Buf<T>`, `MutBuf<T>`, any `Img`, `Accel`):
/// lava bakes it into the pipeline, set by the generated constants struct of the pass. It is a
/// `uint64_t` constant either way, an image handle zero-extended. Scalars need no macro:
/// `[SpecializationConstant] const uint name = 0;`. Every distinct value builds its own
/// pipeline, so only use them for values that never change. Addresses (`Buf`, `MutBuf`,
/// `Accel`) differ from run to run, so the driver's shader cache never hits for such a
/// pipeline and every start compiles it again; a bindless image handle that is fixed for the
/// app's lifetime is the case they fit. `T` is split on commas, so a generic image needs an
/// alias.
#define SCONST(T, name) [SpecializationConstant][SConst(#T)] const uint64_t name##_sconst = 0; static T name = T.fromConst(name##_sconst);

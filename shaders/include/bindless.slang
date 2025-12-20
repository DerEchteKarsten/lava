#define NUM_SAMPLERS 1
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
    uint32_t DescriptorSet;
    uint32_t DescriptorIndex;

    public property bool IsNull { get { return DescriptorIndex == ~0; } }

    public property T Instance {
        get { if(DescriptorSet == 0) {
            return TextureHeap[DescriptorIndex].as<T>();
        } else {
            return RWTextureHeap[DescriptorIndex].as<T>();
        }}
    }
    public property T NonUniformInstance {
        get { if (DescriptorSet == 0) {
                return TextureHeap[NonUniformResourceIndex(DescriptorIndex)].as<T>();
            } else {
                return RWTextureHeap[NonUniformResourceIndex(DescriptorIndex)].as<T>();;
            }}
    }
}

public struct MutImage<T : IImageDescriptor & __IDynamicResourceCastable> {
    uint32_t DescriptorSet;
    uint32_t DescriptorIndex;

    public property bool IsNull { get { return DescriptorIndex == ~0; } }

    public property T Instance {
        get { if (DescriptorSet == 0) {
                return TextureHeap[DescriptorIndex].as<T>();
            } else {
                return RWTextureHeap[DescriptorIndex].as<T>();
            }}
    }
    public property T NonUniformInstance {
        get { if (DescriptorSet == 0) {
                return TextureHeap[NonUniformResourceIndex(DescriptorIndex)].as<T>();
            } else {
                return RWTextureHeap[NonUniformResourceIndex(DescriptorIndex)].as<T>(); ;
            }}
    }
}

__generic<E : ITexelElement, Shape : __ITextureShape, let format : int>
public extension Image<_Texture<E, Shape, 0, 0, 0, 0, 0, 0, format>> {
    typealias TextureType = _Texture<E, Shape, 0, 0, 0, 0, 0, 0, format>;

    [ForceInline]
    public E Load(vector<uint, Shape.dimensions> pos) {
        return Instance[pos];
    }
}

__generic<E : ITexelElement, Shape : __ITextureShape, let format : int>
public extension MutImage<_Texture<E, Shape, 0, 0, 0, 1, 0, 0, format>> {
    typealias TextureType = _Texture<E, Shape, 0, 0, 0, 1, 0, 0, format>;

    [ForceInline]
    public E Load(vector<uint, Shape.dimensions> pos) {
        return Instance[pos];
    }
    [ForceInline]
    public void Store(vector<uint, Shape.dimensions> pos, E value) {
        Instance[pos] = value;
    }
}

#define DESTRUCT_VEC1(var) var.x
#define DESTRUCT_VEC2(var) var.x, var.y
#define DESTRUCT_VEC3(var) var.x, var.y, var.z

#define DEF_TEXTURE_SIZE_EXT(shape, isArray, dim)                                                                                                                                                       \
    __generic<T : ITexelElement, let access : int, let isShadow : int, let isCombined : int, let format : int> \
    public extension _Texture<T, shape, isArray, 0, 0, access, isShadow, isCombined, format> {                                                                                                                                                                                                   \
        public property uint##dim Size {                                                                                                                                                                                               \
            get {                                                                                                                                                                                       \
                uint##dim res;                                                                                                                                                                          \
                GetDimensions(DESTRUCT_VEC##dim(res));                                                                                                                                                  \
                return res;                                                                                                                                                                             \
            }                                                                                                                                                                                           \
        }                                                                                                                                                                                               \
    }

public typealias Image2D<F : ITexelElement = float4> = Image<Texture2D<F>>;
public typealias Image3D<F : ITexelElement = float4> = Image<Texture3D<F>>;

public typealias MutImage2D<F : ITexelElement = float4> = MutImage<RWTexture2D<F>>;
public typealias MutImage3D<F : ITexelElement = float4> = MutImage<RWTexture3D<F>>;

DEF_TEXTURE_SIZE_EXT(__Shape2D, 0, 2);
DEF_TEXTURE_SIZE_EXT(__Shape2D, 1, 3);
DEF_TEXTURE_SIZE_EXT(__Shape3D, 0, 3);

DEF_TEXTURE_SIZE_EXT(__ShapeCube, 0, 2);

public struct Buf<T> {
    public T *val;
}

public struct MutBuf<T> {
    public T *val;
}
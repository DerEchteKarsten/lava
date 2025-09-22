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

public struct ImageHandle<T : IImageDescriptor & __IDynamicResourceCastable> {
    uint64_t DescriptorIndex;

    public property bool IsNull { get { return DescriptorIndex == ~0; } }

    public property T Instance {
        [ForceInline]
        get { return (T.isWriteable ? RWTextureHeap : TextureHeap)[DescriptorIndex].as<T>(); }
    }
    public property T NonUniformInstance {
        [ForceInline]
        get { return (T.isWriteable ? RWTextureHeap : TextureHeap)[NonUniformResourceIndex(DescriptorIndex)].as<T>(); }
    }
}

__generic<E : ITexelElement, Shape : __ITextureShape, let format : int>
public extension ImageHandle<_Texture<E, Shape, 0, 0, 0, 1, 0, 0, format>> {
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

public typealias TextureHandle2D<F: ITexelElement = float4> = ImageHandle<Texture2D<F>>;
public typealias TextureHandle3D<F: ITexelElement = float4> = ImageHandle<Texture3D<F>>;
public typealias ImageHandle2D<F: ITexelElement = float4> = ImageHandle<RWTexture2D<F>>;
public typealias ImageHandle3D<F: ITexelElement = float4> = ImageHandle<RWTexture3D<F>>;

public typealias RWTextureCube<F: ITexelElement = float4> = _Texture<F, __ShapeCube, 0, 0, 0, 1, 0, 0, 0>;

public typealias TextureHandleCube<F: ITexelElement = float4> = ImageHandle<TextureCube<F>>;
public typealias ImageHandleCube<F: ITexelElement = float4> = ImageHandle<RWTextureCube<F>>;

public typealias ImageHandleArray2D<F: ITexelElement = float4> = ImageHandle<RWTexture2DArray<F>>;

#define DESTRUCT_VEC1(var) var.x
#define DESTRUCT_VEC2(var) var.x, var.y
#define DESTRUCT_VEC3(var) var.x, var.y, var.z

#define DEF_TEXTURE_SIZE_EXT(shape, isArray, dim) \
    __generic<T: ITexelElement, let access : int, let isShadow : int, let isCombined : int, let format : int> \
    public extension _Texture<T, shape, isArray, 0, 0, access, isShadow, isCombined, format> { \
        public property uint##dim Size { \
            get { \
                uint##dim res; \
                GetDimensions(DESTRUCT_VEC##dim(res)); \
                return res; \
            } \
        } \
    }

DEF_TEXTURE_SIZE_EXT(__Shape2D, 0, 2);
DEF_TEXTURE_SIZE_EXT(__Shape2D, 1, 3);
DEF_TEXTURE_SIZE_EXT(__Shape3D, 0, 3);

DEF_TEXTURE_SIZE_EXT(__ShapeCube, 0, 2);

//! Type-level buffer usage markers mapping to Vulkan buffer usage flags
use ash::vk;

pub trait BufferUsage: 'static + Copy + Clone {
    const VK: vk::BufferUsageFlags;
}

#[derive(Clone, Copy, Debug)]
pub struct Unknown;

#[derive(Clone, Copy, Debug)]
pub struct Storage;
#[derive(Clone, Copy, Debug)]
pub struct Uniform;
#[derive(Clone, Copy, Debug)]
pub struct Vertex;
#[derive(Clone, Copy, Debug)]
pub struct Index;
#[derive(Clone, Copy, Debug)]
pub struct Indirect;

#[derive(Clone, Copy, Debug)]
pub struct StorageUniform;
#[derive(Clone, Copy, Debug)]
pub struct StorageIndirect;
#[derive(Clone, Copy, Debug)]
pub struct UniformIndirect;
#[derive(Clone, Copy, Debug)]
pub struct StorageUniformIndirect;
#[derive(Clone, Copy, Debug)]
pub struct VertexIndex;
#[derive(Clone, Copy, Debug)]
pub struct StorageVertex;
#[derive(Clone, Copy, Debug)]
pub struct StorageIndex;
#[derive(Clone, Copy, Debug)]
pub struct UniformVertex;
#[derive(Clone, Copy, Debug)]
pub struct UniformIndex;

impl BufferUsage for Unknown {
    const VK: vk::BufferUsageFlags = vk::BufferUsageFlags::empty();
}

impl BufferUsage for Storage {
    const VK: vk::BufferUsageFlags = vk::BufferUsageFlags::STORAGE_BUFFER;
}
impl BufferUsage for Uniform {
    const VK: vk::BufferUsageFlags = vk::BufferUsageFlags::UNIFORM_BUFFER;
}
impl BufferUsage for Vertex {
    const VK: vk::BufferUsageFlags = vk::BufferUsageFlags::VERTEX_BUFFER;
}
impl BufferUsage for Index {
    const VK: vk::BufferUsageFlags = vk::BufferUsageFlags::INDEX_BUFFER;
}
impl BufferUsage for Indirect {
    const VK: vk::BufferUsageFlags = vk::BufferUsageFlags::INDIRECT_BUFFER;
}

impl BufferUsage for StorageUniform {
    const VK: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
        vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
            | vk::BufferUsageFlags::UNIFORM_BUFFER.as_raw(),
    );
}
impl BufferUsage for StorageIndirect {
    const VK: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
        vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
            | vk::BufferUsageFlags::INDIRECT_BUFFER.as_raw(),
    );
}
impl BufferUsage for UniformIndirect {
    const VK: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
        vk::BufferUsageFlags::UNIFORM_BUFFER.as_raw()
            | vk::BufferUsageFlags::INDIRECT_BUFFER.as_raw(),
    );
}
impl BufferUsage for StorageUniformIndirect {
    const VK: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
        vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
            | vk::BufferUsageFlags::UNIFORM_BUFFER.as_raw()
            | vk::BufferUsageFlags::INDIRECT_BUFFER.as_raw(),
    );
}
impl BufferUsage for VertexIndex {
    const VK: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
        vk::BufferUsageFlags::VERTEX_BUFFER.as_raw() | vk::BufferUsageFlags::INDEX_BUFFER.as_raw(),
    );
}
impl BufferUsage for StorageVertex {
    const VK: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
        vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
            | vk::BufferUsageFlags::VERTEX_BUFFER.as_raw(),
    );
}
impl BufferUsage for StorageIndex {
    const VK: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
        vk::BufferUsageFlags::STORAGE_BUFFER.as_raw() | vk::BufferUsageFlags::INDEX_BUFFER.as_raw(),
    );
}
impl BufferUsage for UniformVertex {
    const VK: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
        vk::BufferUsageFlags::UNIFORM_BUFFER.as_raw()
            | vk::BufferUsageFlags::VERTEX_BUFFER.as_raw(),
    );
}
impl BufferUsage for UniformIndex {
    const VK: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
        vk::BufferUsageFlags::UNIFORM_BUFFER.as_raw() | vk::BufferUsageFlags::INDEX_BUFFER.as_raw(),
    );
}

/// Marker for buffers that can be bound as uniform buffers.
pub trait IsUniform: BufferUsage {}
/// Marker for buffers that can be bound as storage buffers.
pub trait IsStorage: BufferUsage {}
/// Marker for buffers usable as vertex buffers.
pub trait IsVertex: BufferUsage {}
/// Marker for buffers usable as index buffers.
pub trait IsIndex: BufferUsage {}
/// Marker for buffers usable as indirect buffers.
pub trait IsIndirect: BufferUsage {}

impl IsStorage for Storage {}
impl IsStorage for StorageUniform {}
impl IsStorage for StorageIndirect {}
impl IsStorage for StorageUniformIndirect {}
impl IsStorage for StorageVertex {}
impl IsStorage for StorageIndex {}

impl IsUniform for Uniform {}
impl IsUniform for StorageUniform {}
impl IsUniform for UniformIndirect {}
impl IsUniform for StorageUniformIndirect {}
impl IsUniform for UniformVertex {}
impl IsUniform for UniformIndex {}

impl IsVertex for Vertex {}
impl IsVertex for VertexIndex {}
impl IsVertex for StorageVertex {}
impl IsVertex for UniformVertex {}

impl IsIndex for Index {}
impl IsIndex for VertexIndex {}
impl IsIndex for StorageIndex {}
impl IsIndex for UniformIndex {}

impl IsIndirect for Indirect {}
impl IsIndirect for StorageIndirect {}
impl IsIndirect for UniformIndirect {}
impl IsIndirect for StorageUniformIndirect {}

#[cfg(test)]
mod tests {
    use super::*;
    use vk::BufferUsageFlags as F;

    #[test]
    fn single_usages_map_to_their_flag() {
        assert_eq!(Unknown::VK, F::empty());
        assert_eq!(Storage::VK, F::STORAGE_BUFFER);
        assert_eq!(Uniform::VK, F::UNIFORM_BUFFER);
        assert_eq!(Vertex::VK, F::VERTEX_BUFFER);
        assert_eq!(Index::VK, F::INDEX_BUFFER);
        assert_eq!(Indirect::VK, F::INDIRECT_BUFFER);
    }

    #[test]
    fn combined_usages_are_the_union_of_their_parts() {
        assert_eq!(StorageUniform::VK, F::STORAGE_BUFFER | F::UNIFORM_BUFFER);
        assert_eq!(StorageIndirect::VK, F::STORAGE_BUFFER | F::INDIRECT_BUFFER);
        assert_eq!(UniformIndirect::VK, F::UNIFORM_BUFFER | F::INDIRECT_BUFFER);
        assert_eq!(
            StorageUniformIndirect::VK,
            F::STORAGE_BUFFER | F::UNIFORM_BUFFER | F::INDIRECT_BUFFER
        );
        assert_eq!(VertexIndex::VK, F::VERTEX_BUFFER | F::INDEX_BUFFER);
        assert_eq!(StorageVertex::VK, F::STORAGE_BUFFER | F::VERTEX_BUFFER);
        assert_eq!(StorageIndex::VK, F::STORAGE_BUFFER | F::INDEX_BUFFER);
        assert_eq!(UniformVertex::VK, F::UNIFORM_BUFFER | F::VERTEX_BUFFER);
        assert_eq!(UniformIndex::VK, F::UNIFORM_BUFFER | F::INDEX_BUFFER);
    }

    /// The `Is*` marker traits must agree with the flags: a type may only claim a capability
    /// its Vulkan usage actually contains.
    #[test]
    fn marker_traits_match_the_flags() {
        fn storage<U: IsStorage>() -> F {
            U::VK
        }
        fn uniform<U: IsUniform>() -> F {
            U::VK
        }
        fn vertex<U: IsVertex>() -> F {
            U::VK
        }
        fn index<U: IsIndex>() -> F {
            U::VK
        }
        fn indirect<U: IsIndirect>() -> F {
            U::VK
        }

        for flags in [
            storage::<Storage>(),
            storage::<StorageUniform>(),
            storage::<StorageIndirect>(),
            storage::<StorageUniformIndirect>(),
            storage::<StorageVertex>(),
            storage::<StorageIndex>(),
        ] {
            assert!(flags.contains(F::STORAGE_BUFFER));
        }
        for flags in [
            uniform::<Uniform>(),
            uniform::<StorageUniform>(),
            uniform::<UniformIndirect>(),
            uniform::<StorageUniformIndirect>(),
            uniform::<UniformVertex>(),
            uniform::<UniformIndex>(),
        ] {
            assert!(flags.contains(F::UNIFORM_BUFFER));
        }
        for flags in [
            vertex::<Vertex>(),
            vertex::<VertexIndex>(),
            vertex::<StorageVertex>(),
            vertex::<UniformVertex>(),
        ] {
            assert!(flags.contains(F::VERTEX_BUFFER));
        }
        for flags in [
            index::<Index>(),
            index::<VertexIndex>(),
            index::<StorageIndex>(),
            index::<UniformIndex>(),
        ] {
            assert!(flags.contains(F::INDEX_BUFFER));
        }
        for flags in [
            indirect::<Indirect>(),
            indirect::<StorageIndirect>(),
            indirect::<UniformIndirect>(),
            indirect::<StorageUniformIndirect>(),
        ] {
            assert!(flags.contains(F::INDIRECT_BUFFER));
        }
    }
}

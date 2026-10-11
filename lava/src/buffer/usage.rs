//! Type-level buffer usage markers, one per combination of usages, mapping to Vulkan buffer usage flags
use ash::vk;

/// Every buffer also gets `SHADER_DEVICE_ADDRESS`, `TRANSFER_SRC` and `TRANSFER_DST` (`Buffer::new`):
/// leaving the transfer usages out of buffers that don't need them made no measurable difference.
pub trait BufferUsage: 'static + Copy + Clone {
    const VK: vk::BufferUsageFlags;
}

/// Marker for buffers that can be bound as storage buffers.
pub trait IsStorage: BufferUsage {}
/// Marker for buffers that can be bound as uniform buffers.
pub trait IsUniform: BufferUsage {}
/// Marker for buffers usable as vertex buffers.
pub trait IsVertex: BufferUsage {}
/// Marker for buffers usable as index buffers.
pub trait IsIndex: BufferUsage {}
/// Marker for buffers usable as indirect buffers.
pub trait IsIndirect: BufferUsage {}

/// No usage but the ones every buffer has.
#[derive(Clone, Copy, Debug)]
pub struct Unknown;
impl BufferUsage for Unknown {
    const VK: vk::BufferUsageFlags = vk::BufferUsageFlags::empty();
}

/// Defines one usage per `Name: Flag...;` line: its Vulkan flags and a marker trait per flag.
macro_rules! usages {
    (@flag Storage) => { vk::BufferUsageFlags::STORAGE_BUFFER };
    (@flag Uniform) => { vk::BufferUsageFlags::UNIFORM_BUFFER };
    (@flag Vertex) => { vk::BufferUsageFlags::VERTEX_BUFFER };
    (@flag Index) => { vk::BufferUsageFlags::INDEX_BUFFER };
    (@flag Indirect) => { vk::BufferUsageFlags::INDIRECT_BUFFER };

    (@trait $name:ident Storage) => { impl IsStorage for $name {} };
    (@trait $name:ident Uniform) => { impl IsUniform for $name {} };
    (@trait $name:ident Vertex) => { impl IsVertex for $name {} };
    (@trait $name:ident Index) => { impl IsIndex for $name {} };
    (@trait $name:ident Indirect) => { impl IsIndirect for $name {} };

    ($($name:ident: $($flag:ident)+;)*) => {
        $(
            #[derive(Clone, Copy, Debug)]
            pub struct $name;
            impl BufferUsage for $name {
                const VK: vk::BufferUsageFlags =
                    vk::BufferUsageFlags::from_raw(0 $(| usages!(@flag $flag).as_raw())+);
            }
            $(usages!(@trait $name $flag);)+
        )*

        #[cfg(test)]
        const ALL: &[(&str, vk::BufferUsageFlags)] = &[$((stringify!($name), $name::VK)),*];
    };
}

// Vulkan allows every combination of buffer usages.
usages! {
    Storage: Storage;
    Uniform: Uniform;
    StorageUniform: Storage Uniform;
    Vertex: Vertex;
    StorageVertex: Storage Vertex;
    UniformVertex: Uniform Vertex;
    StorageUniformVertex: Storage Uniform Vertex;
    Index: Index;
    StorageIndex: Storage Index;
    UniformIndex: Uniform Index;
    StorageUniformIndex: Storage Uniform Index;
    VertexIndex: Vertex Index;
    StorageVertexIndex: Storage Vertex Index;
    UniformVertexIndex: Uniform Vertex Index;
    StorageUniformVertexIndex: Storage Uniform Vertex Index;
    Indirect: Indirect;
    StorageIndirect: Storage Indirect;
    UniformIndirect: Uniform Indirect;
    StorageUniformIndirect: Storage Uniform Indirect;
    VertexIndirect: Vertex Indirect;
    StorageVertexIndirect: Storage Vertex Indirect;
    UniformVertexIndirect: Uniform Vertex Indirect;
    StorageUniformVertexIndirect: Storage Uniform Vertex Indirect;
    IndexIndirect: Index Indirect;
    StorageIndexIndirect: Storage Index Indirect;
    UniformIndexIndirect: Uniform Index Indirect;
    StorageUniformIndexIndirect: Storage Uniform Index Indirect;
    VertexIndexIndirect: Vertex Index Indirect;
    StorageVertexIndexIndirect: Storage Vertex Index Indirect;
    UniformVertexIndexIndirect: Uniform Vertex Index Indirect;
    StorageUniformVertexIndexIndirect: Storage Uniform Vertex Index Indirect;
}

#[cfg(test)]
mod tests {
    use super::*;
    use vk::BufferUsageFlags as F;

    const FLAGS: [(&str, F); 5] = [
        ("Storage", F::STORAGE_BUFFER),
        ("Uniform", F::UNIFORM_BUFFER),
        ("Vertex", F::VERTEX_BUFFER),
        ("Index", F::INDEX_BUFFER),
        ("Indirect", F::INDIRECT_BUFFER),
    ];

    #[test]
    fn names_list_their_flags() {
        for (name, flags) in ALL {
            let expected: String = FLAGS
                .iter()
                .filter(|(_, flag)| flags.contains(*flag))
                .map(|(name, _)| *name)
                .collect();
            assert_eq!(*name, expected);
        }
    }

    #[test]
    fn every_combination_exists_once() {
        let mut all: Vec<F> = ALL.iter().map(|(_, flags)| *flags).collect();
        all.sort();
        all.dedup();
        assert_eq!(all.len(), (1 << FLAGS.len()) - 1);
        assert_eq!(Unknown::VK, F::empty());
    }

    #[test]
    fn marker_traits_match_the_flags() {
        fn storage<U: IsStorage>() -> F {
            U::VK
        }
        fn index<U: IsIndex>() -> F {
            U::VK
        }
        fn uniform<U: IsUniform>() -> F {
            U::VK
        }
        assert_eq!(
            storage::<StorageIndirect>(),
            F::STORAGE_BUFFER | F::INDIRECT_BUFFER
        );
        assert_eq!(index::<VertexIndex>(), F::VERTEX_BUFFER | F::INDEX_BUFFER);
        assert_eq!(
            uniform::<UniformVertexIndirect>(),
            F::UNIFORM_BUFFER | F::VERTEX_BUFFER | F::INDIRECT_BUFFER
        );
    }
}

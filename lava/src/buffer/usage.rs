//! Buffer usages: one marker type per combination of usages, mapping to Vulkan buffer usage flags, and `Dynamic`, whose usage is only known at runtime
use crate::vk;

/// Every buffer also gets device address and transfer src/dst usage, and acceleration
/// structure build input with that feature (`Buffer::raw`).
pub trait BufferUsage: crate::sealed::Sealed + 'static + Copy + Send + Sync + std::fmt::Debug {
    fn flags(&self) -> vk::BufferUsageFlags;
    /// Panics unless the usage has every flag of `required`; a static usage's does nothing.
    fn check(&self, required: vk::BufferUsageFlags);
}

/// Every usage but `Dynamic`.
pub trait StaticUsage: BufferUsage + Default {
    const VK: vk::BufferUsageFlags;
}

pub trait IsStorage: BufferUsage {}
pub trait IsUniform: BufferUsage {}
pub trait IsVertex: BufferUsage {}
pub trait IsIndex: BufferUsage {}
pub trait IsIndirect: BufferUsage {}

/// No usage but the ones every buffer has.
#[derive(Clone, Copy, Debug, Default)]
pub struct Unknown;
impl StaticUsage for Unknown {
    const VK: vk::BufferUsageFlags = vk::BufferUsageFlags::empty();
}
impl crate::sealed::Sealed for Unknown {}
impl BufferUsage for Unknown {
    fn flags(&self) -> vk::BufferUsageFlags {
        Self::VK
    }
    fn check(&self, _: vk::BufferUsageFlags) {}
}

/// The shader binding table of a ray tracing pipeline.
#[derive(Clone, Copy, Debug, Default)]
pub struct ShaderBindingTable;
impl StaticUsage for ShaderBindingTable {
    const VK: vk::BufferUsageFlags = vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR;
}
impl crate::sealed::Sealed for ShaderBindingTable {}
impl BufferUsage for ShaderBindingTable {
    fn flags(&self) -> vk::BufferUsageFlags {
        Self::VK
    }
    fn check(&self, _: vk::BufferUsageFlags) {}
}

/// The memory of an acceleration structure, which owns its buffer.
#[derive(Clone, Copy, Debug, Default)]
pub struct AccelerationStructureStorage;
impl StaticUsage for AccelerationStructureStorage {
    const VK: vk::BufferUsageFlags = vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR;
}
impl crate::sealed::Sealed for AccelerationStructureStorage {}
impl BufferUsage for AccelerationStructureStorage {
    fn flags(&self) -> vk::BufferUsageFlags {
        Self::VK
    }
    fn check(&self, _: vk::BufferUsageFlags) {}
}

/// A usage only known at runtime. It meets every usage bound, and `check` panics where the
/// bound wouldn't have compiled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dynamic(pub vk::BufferUsageFlags);

impl crate::sealed::Sealed for Dynamic {}
impl BufferUsage for Dynamic {
    fn flags(&self) -> vk::BufferUsageFlags {
        self.0
    }
    fn check(&self, required: vk::BufferUsageFlags) {
        assert!(
            self.0.contains(required),
            "a buffer with usage {:?} lacks {:?}",
            self.0,
            required & !self.0
        );
    }
}

impl IsStorage for Dynamic {}
impl IsUniform for Dynamic {}
impl IsVertex for Dynamic {}
impl IsIndex for Dynamic {}
impl IsIndirect for Dynamic {}

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
            #[derive(Clone, Copy, Debug, Default)]
            pub struct $name;
            impl StaticUsage for $name {
                const VK: vk::BufferUsageFlags =
                    vk::BufferUsageFlags::from_raw(0 $(| usages!(@flag $flag).as_raw())+);
            }
            impl crate::sealed::Sealed for $name {}
            impl BufferUsage for $name {
                fn flags(&self) -> vk::BufferUsageFlags {
                    Self::VK
                }
                fn check(&self, _: vk::BufferUsageFlags) {}
            }
            $(usages!(@trait $name $flag);)+
        )*
    };
}

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

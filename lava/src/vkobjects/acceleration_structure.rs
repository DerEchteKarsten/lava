//! Ray tracing acceleration structures: `Blas`/`Tlas` owning their memory, their geometry (triangles, AABBs, instances), builds, refits and compaction with automatic barriers
use std::{marker::PhantomData, range::Range};

use crate::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Affine3A, Vec3, Vec4};
use lava_macros::validation_trace;
use smallvec::{SmallVec, smallvec};

use crate::{
    buffer::{Buffer, Gpu, slice::BufferSlice},
    command_buffer::{BufferAccess, CommandBuffer, EVERY_ACCELERATION_STRUCTURE},
    lifetime,
    queue::HasCompute,
    error::Result,
    features::{Feature, gate},
    query::{CompactedSize, QueryPool},
    queue::lock,
    global::Live,
    state::{Ctx, Functions},
};
use std::sync::Mutex;

const BUILD: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR;
const READ: vk::AccessFlags2 = vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR;
const WRITE: vk::AccessFlags2 = vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR;
/// `MEMORY_READ` is for Mesa's ANV (26.2.2): after a structure traced earlier in the same
/// submission was rebuilt, `ACCELERATION_STRUCTURE_READ` alone lets the next trace see stale
/// nodes (`a_tlas_rebuilt_between_two_traces_waits_for_the_first`).
const TRACE_READ: vk::AccessFlags2 =
    vk::AccessFlags2::from_raw(READ.as_raw() | vk::AccessFlags2::MEMORY_READ.as_raw());
/// What `vkCreateAccelerationStructure2KHR` requires of a structure's address and size.
const STRUCTURE_ALIGNMENT: u64 = 256;

/// `MissingFeature` where lava checks it.
pub(crate) fn functions() -> Result<&'static ash::khr::acceleration_structure::Device> {
    gate::AccelerationStructure::check()?;
    Ok(Functions::acceleration_structure())
}

/// Unchecked: a command's structure only exists past the check.
fn command_functions() -> &'static ash::khr::acceleration_structure::Device {
    Functions::acceleration_structure()
}

/// The alignment of a scratch buffer's address (`minAccelerationStructureScratchOffsetAlignment`).
/// The scratch sizes of the structures are multiples of it, so a batch's add up.
pub fn scratch_alignment() -> u64 {
    Ctx::physical_device()
        .acceleration_structure_properties
        .map_or(1, |p| {
            p.min_acceleration_structure_scratch_offset_alignment as u64
        })
}

/// The read a trace or TLAS build makes of the BLASes it reaches, which lava can't list.
pub(crate) fn every_structure_read(stage: vk::PipelineStageFlags2) -> BufferAccess {
    let access = if stage == BUILD { READ } else { TRACE_READ };
    BufferAccess {
        unordered: false,
        stage,
        access,
        range: EVERY_ACCELERATION_STRUCTURE,
    }
}

pub struct Described {
    pub geometry: vk::AccelerationStructureGeometryKHR<'static>,
    pub primitives: u32,
    pub inputs: [Range<u64>; 3],
}

/// What an update has to keep of a geometry's last build.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Shape {
    ty: vk::GeometryTypeKHR,
    flags: vk::GeometryFlagsKHR,
    primitives: u32,
    vertex_format: vk::Format,
    max_vertex: u32,
    index_type: vk::IndexType,
    transform: bool,
}

impl Shape {
    fn of(described: &Described) -> Self {
        let geometry = &described.geometry;
        let mut shape = Shape {
            ty: geometry.geometry_type,
            flags: geometry.flags,
            primitives: described.primitives,
            vertex_format: vk::Format::UNDEFINED,
            max_vertex: 0,
            index_type: vk::IndexType::NONE_KHR,
            transform: false,
        };
        if geometry.geometry_type == vk::GeometryTypeKHR::TRIANGLES {
            // SAFETY: a triangle geometry's data is `triangles`, its addresses device addresses.
            let triangles = unsafe { geometry.geometry.triangles };
            shape.vertex_format = triangles.vertex_format;
            shape.max_vertex = triangles.max_vertex;
            shape.index_type = triangles.index_type;
            shape.transform = unsafe { triangles.transform_data.device_address } != 0;
        }
        shape
    }
}

pub trait Describe {
    fn describe(&self, out: &mut SmallVec<[Described; 1]>);
}

/// The level of an acceleration structure: [`Bottom`] (geometry) or [`Top`] (instances).
pub trait Level: crate::sealed::Sealed + 'static {
    const TY: vk::AccelerationStructureTypeKHR;
    /// What a build of this level reads.
    type Geometry<'a>: Describe;
}

/// A bottom-level structure: triangles or AABBs.
pub struct Bottom;
/// A top-level structure: instances of bottom-level ones.
pub struct Top;
impl crate::sealed::Sealed for Bottom {}
impl crate::sealed::Sealed for Top {}
impl Level for Bottom {
    const TY: vk::AccelerationStructureTypeKHR = vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL;
    type Geometry<'a> = BlasGeometry<'a>;
}
impl Level for Top {
    const TY: vk::AccelerationStructureTypeKHR = vk::AccelerationStructureTypeKHR::TOP_LEVEL;
    type Geometry<'a> = Instances<'a>;
}

pub type Blas = AccelerationStructure<Bottom>;
pub type Tlas = AccelerationStructure<Top>;

/// How a structure is built. The default prefers fast traces.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BuildFlags {
    /// Prefer fast builds over fast traces.
    pub fast_build: bool,
    /// Allow [`CommandBuffer::update`].
    pub allow_update: bool,
    /// Allow [`CommandBuffer::query_compacted_sizes`] and [`CommandBuffer::compact`].
    pub allow_compaction: bool,
}

impl BuildFlags {
    fn vk(self) -> vk::BuildAccelerationStructureFlagsKHR {
        type F = vk::BuildAccelerationStructureFlagsKHR;
        let mut flags = if self.fast_build {
            F::PREFER_FAST_BUILD
        } else {
            F::PREFER_FAST_TRACE
        };
        if self.allow_update {
            flags |= F::ALLOW_UPDATE;
        }
        if self.allow_compaction {
            flags |= F::ALLOW_COMPACTION;
        }
        flags
    }
}

/// An acceleration structure and the memory it owns, which builds write through a shared
/// reference. A TLAS that still names a dropped BLAS must not be traced ([`Instances::new`]).
pub struct AccelerationStructure<L: Level> {
    handle: vk::AccelerationStructureKHR,
    address: u64,
    flags: BuildFlags,
    /// Per geometry: the most primitives a build may have.
    max_primitives: SmallVec<[u32; 4]>,
    build_scratch: u64,
    update_scratch: u64,
    /// Made by `compacted` of this structure: its memory only holds the compacted structure, so
    /// it can't be built again.
    compacted_from: Option<vk::AccelerationStructureKHR>,
    /// The geometries of the last recorded build (or compaction into it), which an update keeps;
    /// empty before the first.
    built: Mutex<SmallVec<[Shape; 1]>>,
    /// Dropped after the structure, so destroyed after it.
    buffer: Buffer<u8>,
    _level: PhantomData<L>,
    /// Last, so it counts the structure until its drop finished.
    _live: Live,
}

fn round_up(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment) * alignment
}

impl<L: Level> AccelerationStructure<L> {
    fn with_geometries(
        geometries: &[vk::AccelerationStructureGeometryKHR<'_>],
        max_primitives: SmallVec<[u32; 4]>,
        flags: BuildFlags,
    ) -> Result<Self> {
        let live = Live::new();
        let info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(L::TY)
            .flags(flags.vk())
            .geometries(geometries);
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            functions()?.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &info,
                &max_primitives,
                &mut sizes,
            )
        };
        let alignment = scratch_alignment();
        Self::create(
            sizes.acceleration_structure_size,
            flags,
            max_primitives,
            round_up(sizes.build_scratch_size, alignment),
            if flags.allow_update {
                round_up(sizes.update_scratch_size, alignment)
            } else {
                0
            },
            None,
            live,
        )
    }

    fn create(
        size: u64,
        flags: BuildFlags,
        max_primitives: SmallVec<[u32; 4]>,
        build_scratch: u64,
        update_scratch: u64,
        compacted_from: Option<vk::AccelerationStructureKHR>,
        live: Live,
    ) -> Result<Self> {
        let functions = functions()?;
        let buffer = Buffer::with_flags(
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
            Gpu,
            round_up(size, STRUCTURE_ALIGNMENT),
            Some(STRUCTURE_ALIGNMENT as u32),
        )?;
        // The buffer has no storage usage.
        let create_info = vk::AccelerationStructureCreateInfo2KHR::default()
            .address_range(buffer.range(..).address_range())
            .address_flags(
                vk::AddressCommandFlagsKHR::UNKNOWN_STORAGE_BUFFER_USAGE
                    | vk::AddressCommandFlagsKHR::FULLY_BOUND,
            )
            .ty(L::TY);
        let handle = unsafe {
            Functions::device_address_commands()
                .create_acceleration_structure2(&create_info, None)?
        };
        let address = unsafe {
            functions.get_acceleration_structure_device_address(
                &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                    .acceleration_structure(handle),
            )
        };
        Ok(Self {
            handle,
            address,
            flags,
            max_primitives,
            build_scratch,
            update_scratch,
            compacted_from,
            built: Mutex::new(SmallVec::new()),
            buffer,
            _level: PhantomData,
            _live: live,
        })
    }

    /// The same structure named `name` (its memory too), for debuggers, the validation layer
    /// (with debug utils, `Config::validation`) and allocator reports.
    pub fn named(mut self, name: &str) -> Self {
        self.buffer.set_name(name);
        Functions::set_debug_name(name, self.handle);
        self
    }

    /// Its device address, which shaders and instances use. Not its buffer's address.
    pub fn address(&self) -> u64 {
        self.address
    }

    /// The bytes its memory takes.
    pub fn size(&self) -> u64 {
        self.buffer.size()
    }

    pub fn flags(&self) -> BuildFlags {
        self.flags
    }

    /// The scratch bytes a build needs, a multiple of [`scratch_alignment`].
    pub fn build_scratch_size(&self) -> u64 {
        self.build_scratch
    }

    /// The scratch bytes an update needs, a multiple of [`scratch_alignment`]; 0 without
    /// `allow_update`.
    pub fn update_scratch_size(&self) -> u64 {
        self.update_scratch
    }

    /// An unbuilt structure of `size` bytes (read from a `QueryPool<CompactedSize>`) with this
    /// one's flags, for [`CommandBuffer::compact`] of this one. It can be updated but never built.
    pub fn compacted(&self, size: u64) -> Result<Self> {
        Self::create(
            size,
            self.flags,
            self.max_primitives.clone(),
            self.build_scratch,
            self.update_scratch,
            Some(self.handle),
            Live::new(),
        )
    }

    /// Whether it was made by `compacted`, so only `compact` and `update` write it.
    pub fn is_compacted(&self) -> bool {
        self.compacted_from.is_some()
    }

    #[track_caller]
    fn check_built(&self, what: &str) {
        assert!(!lock(&self.built).is_empty(), "{what} a structure that was built before");
    }

    pub(crate) fn access(
        &self,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
    ) -> BufferAccess {
        BufferAccess {
            unordered: false,
            stage,
            access,
            range: self.buffer.range(..).get_range(),
        }
    }
}

impl Blas {
    /// An unbuilt BLAS for builds of `geometry`, with as many primitives per geometry at most.
    pub fn new(geometry: &BlasGeometry<'_>, flags: BuildFlags) -> Result<Self> {
        let mut described = SmallVec::new();
        geometry.describe(&mut described);
        let geometries: SmallVec<[_; 1]> = described.iter().map(|d| d.geometry).collect();
        let primitives = described.iter().map(|d| d.primitives).collect();
        Self::with_geometries(&geometries, primitives, flags)
    }
}

impl Tlas {
    /// An unbuilt TLAS for builds of up to `max_instances` instances.
    pub fn new(max_instances: u32, flags: BuildFlags) -> Result<Self> {
        let geometry = instance_geometry(0);
        Self::with_geometries(&[geometry], smallvec![max_instances], flags)
    }

    /// The accesses of a pass that traces it: its memory and the BLASes it names.
    pub fn trace_accesses(&self, stage: vk::PipelineStageFlags2) -> [BufferAccess; 2] {
        [self.access(stage, TRACE_READ), every_structure_read(stage)]
    }
}

impl<L: Level> Drop for AccelerationStructure<L> {
    fn drop(&mut self) {
        let handle = self.handle;
        lifetime::retire(move || {
            if let Ok(functions) = functions() {
                unsafe { functions.destroy_acceleration_structure(handle, None) };
            }
        });
    }
}

/// A vertex type whose first bytes are its position, in `FORMAT`. Its size is the stride.
pub trait VertexPosition: Copy + Pod {
    const FORMAT: vk::Format;
}
impl VertexPosition for [f32; 3] {
    const FORMAT: vk::Format = vk::Format::R32G32B32_SFLOAT;
}
impl VertexPosition for Vec3 {
    const FORMAT: vk::Format = vk::Format::R32G32B32_SFLOAT;
}
impl VertexPosition for [f32; 4] {
    const FORMAT: vk::Format = vk::Format::R32G32B32_SFLOAT;
}
impl VertexPosition for Vec4 {
    const FORMAT: vk::Format = vk::Format::R32G32B32_SFLOAT;
}

/// The index types a triangle geometry can use, those of indexed draws.
pub use crate::command_buffer::IndexFormat;

/// The bytes of a vertex format's smallest component, which vertex addresses and strides are
/// multiples of; 1 for formats it doesn't know.
fn component_bytes(format: vk::Format) -> u64 {
    use vk::Format as F;
    match format {
        F::R32G32_SFLOAT | F::R32G32B32_SFLOAT | F::R32G32B32A32_SFLOAT => 4,
        F::R16G16_SFLOAT
        | F::R16G16B16A16_SFLOAT
        | F::R16G16_SNORM
        | F::R16G16B16A16_SNORM
        | F::R16G16_UNORM
        | F::R16G16B16A16_UNORM => 2,
        _ => 1,
    }
}

/// A row-major 3x4 transform (`VkTransformMatrixKHR`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct TransformMatrix(pub [[f32; 4]; 3]);

impl From<Affine3A> for TransformMatrix {
    fn from(t: Affine3A) -> Self {
        let (m, p) = (t.matrix3, t.translation);
        Self(std::array::from_fn(|row| {
            [m.x_axis[row], m.y_axis[row], m.z_axis[row], p[row]]
        }))
    }
}

/// An axis-aligned box of a procedural geometry (`VkAabbPositionsKHR`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

fn geometry_flags(opaque: bool) -> vk::GeometryFlagsKHR {
    if opaque {
        vk::GeometryFlagsKHR::OPAQUE
    } else {
        vk::GeometryFlagsKHR::empty()
    }
}

fn const_address(address: u64) -> vk::DeviceOrHostAddressConstKHR {
    vk::DeviceOrHostAddressConstKHR {
        device_address: address,
    }
}

/// A triangle geometry: every three vertices, or every three indices, make a triangle. Opaque
/// unless [`Self::opaque`] says otherwise (then any-hit shaders run).
#[derive(Clone, Copy)]
pub struct Triangles<'a> {
    data: vk::AccelerationStructureGeometryTrianglesDataKHR<'static>,
    primitives: u32,
    opaque: bool,
    inputs: [Range<u64>; 3],
    _marker: PhantomData<&'a ()>,
}

impl<'a> Triangles<'a> {
    pub fn new<V: VertexPosition>(vertices: BufferSlice<'a, V>) -> Self {
        assert!(
            vertices.len() % 3 == 0,
            "{} vertices don't make triangles",
            vertices.len()
        );
        Self::with_indices(
            vertices,
            vk::IndexType::NONE_KHR,
            0,
            (vertices.len() / 3) as u32,
            (0..0).into(),
        )
    }

    pub fn indexed<V: VertexPosition, I: IndexFormat>(
        vertices: BufferSlice<'a, V>,
        indices: BufferSlice<'a, I>,
    ) -> Self {
        assert!(
            indices.len() % 3 == 0,
            "{} indices don't make triangles",
            indices.len()
        );
        Self::with_indices(
            vertices,
            I::TYPE,
            indices.address(),
            (indices.len() / 3) as u32,
            indices.get_range(),
        )
    }

    fn with_indices<V: VertexPosition>(
        vertices: BufferSlice<'a, V>,
        index_type: vk::IndexType,
        index_address: u64,
        primitives: u32,
        index_range: Range<u64>,
    ) -> Self {
        assert!(!vertices.is_empty(), "a triangle geometry needs vertices");
        let component = component_bytes(V::FORMAT);
        assert!(
            vertices.address() % component == 0 && size_of::<V>() as u64 % component == 0,
            "the vertices' address and stride have to be multiples of the {component} bytes of a \
             {:?} component",
            V::FORMAT
        );
        Self {
            data: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                .vertex_format(V::FORMAT)
                .vertex_data(const_address(vertices.address()))
                .vertex_stride(size_of::<V>() as u64)
                .max_vertex(vertices.len() as u32 - 1)
                .index_type(index_type)
                .index_data(const_address(index_address)),
            primitives,
            opaque: true,
            inputs: [vertices.get_range(), index_range, (0..0).into()],
            _marker: PhantomData,
        }
    }

    /// Transforms the vertices by the first matrix of `transform`, whose address has to be a
    /// multiple of 16.
    pub fn transform(
        mut self,
        transform: BufferSlice<'a, TransformMatrix>,
    ) -> Self {
        assert!(!transform.is_empty(), "the transform slice is empty");
        assert!(
            transform.address() % 16 == 0,
            "a transform's address has to be a multiple of 16"
        );
        self.data = self.data.transform_data(const_address(transform.address()));
        self.inputs[2] = transform.range(..1).get_range();
        self
    }

    pub fn opaque(mut self, opaque: bool) -> Self {
        self.opaque = opaque;
        self
    }
}

/// A procedural geometry: one primitive per box, hit through intersection shaders (or a ray
/// query's candidates). Opaque unless [`Self::opaque`] says otherwise.
#[derive(Clone, Copy)]
pub struct Aabbs<'a> {
    address: u64,
    primitives: u32,
    opaque: bool,
    range: Range<u64>,
    _marker: PhantomData<&'a ()>,
}

impl<'a> Aabbs<'a> {
    pub fn new(boxes: BufferSlice<'a, Aabb>) -> Self {
        assert!(
            boxes.address() % 8 == 0,
            "an AABB buffer's address has to be a multiple of 8"
        );
        Self {
            address: boxes.address(),
            primitives: boxes.len() as u32,
            opaque: true,
            range: boxes.get_range(),
            _marker: PhantomData,
        }
    }

    pub fn opaque(mut self, opaque: bool) -> Self {
        self.opaque = opaque;
        self
    }
}

/// What a BLAS is built from: triangle or procedural geometries, never both.
#[derive(Clone, Copy)]
pub enum BlasGeometry<'a> {
    Triangles(&'a [Triangles<'a>]),
    Aabbs(&'a [Aabbs<'a>]),
}

impl Describe for BlasGeometry<'_> {
    fn describe(&self, out: &mut SmallVec<[Described; 1]>) {
        match *self {
            BlasGeometry::Triangles(triangles) => out.extend(triangles.iter().map(|t| {
                Described {
                    geometry: vk::AccelerationStructureGeometryKHR::default()
                        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                        .geometry(vk::AccelerationStructureGeometryDataKHR { triangles: t.data })
                        .flags(geometry_flags(t.opaque)),
                    primitives: t.primitives,
                    inputs: t.inputs,
                }
            })),
            BlasGeometry::Aabbs(boxes) => out.extend(boxes.iter().map(|b| {
                Described {
                    geometry: vk::AccelerationStructureGeometryKHR::default()
                        .geometry_type(vk::GeometryTypeKHR::AABBS)
                        .geometry(vk::AccelerationStructureGeometryDataKHR {
                            aabbs: vk::AccelerationStructureGeometryAabbsDataKHR::default()
                                .data(const_address(b.address))
                                .stride(size_of::<Aabb>() as u64),
                        })
                        .flags(geometry_flags(b.opaque)),
                    primitives: b.primitives,
                    inputs: [b.range, (0..0).into(), (0..0).into()],
                }
            })),
        }
    }
}

/// An instance of a BLAS in a TLAS (`VkAccelerationStructureInstanceKHR`, a layout of the
/// shader contract, so passes can write them).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct Instance {
    pub transform: TransformMatrix,
    /// The custom index in bits 0..24, the mask in 24..32.
    pub custom_index_and_mask: u32,
    /// The hit group offset in bits 0..24, the flags in 24..32.
    pub sbt_offset_and_flags: u32,
    pub blas: u64,
}

const LOW_24: u32 = (1 << 24) - 1;

impl Instance {
    /// `blas` placed by `transform`, seen by every ray mask, with custom index, hit group
    /// offset and flags 0. It keeps nothing alive: see [`Instances::new`].
    pub fn new(blas: &Blas, transform: Affine3A) -> Self {
        Self::from_address(blas.address(), transform)
    }

    /// Like [`Self::new`] for the BLAS at `address` ([`AccelerationStructure::address`]).
    pub fn from_address(address: u64, transform: Affine3A) -> Self {
        Self {
            transform: transform.into(),
            custom_index_and_mask: 0xFF << 24,
            sbt_offset_and_flags: 0,
            blas: address,
        }
    }

    /// What `InstanceID()` returns; at most 24 bits.
    pub fn custom_index(mut self, index: u32) -> Self {
        assert!(index <= LOW_24, "custom index {index} doesn't fit 24 bits");
        self.custom_index_and_mask = (self.custom_index_and_mask & !LOW_24) | index;
        self
    }

    /// Rays whose mask shares no bit with it miss the instance.
    pub fn mask(mut self, mask: u8) -> Self {
        self.custom_index_and_mask = (self.custom_index_and_mask & LOW_24) | (mask as u32) << 24;
        self
    }

    /// The index of the hit group its geometries use (`[Group(n)]`); at most 24 bits.
    pub fn sbt_offset(mut self, offset: u32) -> Self {
        assert!(
            offset <= LOW_24,
            "hit group offset {offset} doesn't fit 24 bits"
        );
        self.sbt_offset_and_flags = (self.sbt_offset_and_flags & !LOW_24) | offset;
        self
    }

    pub fn flags(mut self, flags: vk::GeometryInstanceFlagsKHR) -> Self {
        let raw = flags.as_raw();
        assert!(raw <= 0xFF, "instance flags {flags:?} don't fit 8 bits");
        self.sbt_offset_and_flags = (self.sbt_offset_and_flags & LOW_24) | raw << 24;
        self
    }
}

fn instance_geometry(address: u64) -> vk::AccelerationStructureGeometryKHR<'static> {
    vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                .data(const_address(address)),
        })
}

/// What a TLAS is built from: a slice of [`Instance`]s, which the host or a pass may write.
#[derive(Clone, Copy)]
pub struct Instances<'a> {
    address: u64,
    count: u32,
    range: Range<u64>,
    _marker: PhantomData<&'a ()>,
}

impl<'a> Instances<'a> {
    /// The build input of a TLAS: `instances`, written by the host or a pass.
    ///
    /// # Safety
    /// Every BLAS the instances name (by address; 0 is an inactive instance) must stay alive
    /// while a TLAS built from them is traced. lava doesn't track them: a dropped BLAS is
    /// destroyed once the submissions in flight at its drop are done, so rebuild the TLAS
    /// without it before tracing again.
    pub unsafe fn new(instances: BufferSlice<'a, Instance>) -> Self {
        assert!(
            instances.address() % 16 == 0,
            "an instance buffer's address has to be a multiple of 16"
        );
        Self {
            address: instances.address(),
            count: instances.len() as u32,
            range: instances.get_range(),
            _marker: PhantomData,
        }
    }
}

impl Describe for Instances<'_> {
    fn describe(&self, out: &mut SmallVec<[Described; 1]>) {
        out.push(Described {
            geometry: instance_geometry(self.address),
            primitives: self.count,
            inputs: [self.range, (0..0).into(), (0..0).into()],
        });
    }
}

impl<K: HasCompute> CommandBuffer<'_, K> {
    /// Builds every structure in one `vkCmdBuildAccelerationStructuresKHR`, each from its
    /// `build_scratch_size()` of `scratch` after the previous one's. `scratch`'s address has to
    /// be a multiple of [`scratch_alignment`].
    #[validation_trace]
    pub fn build<L: Level>(
        &mut self,
        builds: &[(&AccelerationStructure<L>, L::Geometry<'_>)],
        scratch: BufferSlice<'_, u8>,
    ) {
        self.record_builds(builds, scratch, false);
    }

    /// Refits every structure in place for its geometry, which has the primitive counts of its
    /// last build. Needs `allow_update`; the scratch is laid out by `update_scratch_size()`.
    #[validation_trace]
    pub fn update<L: Level>(
        &mut self,
        updates: &[(&AccelerationStructure<L>, L::Geometry<'_>)],
        scratch: BufferSlice<'_, u8>,
    ) {
        self.record_builds(updates, scratch, true);
    }

    fn record_builds<L: Level>(
        &mut self,
        builds: &[(&AccelerationStructure<L>, L::Geometry<'_>)],
        scratch: BufferSlice<'_, u8>,
        update: bool,
    ) {
        let alignment = scratch_alignment();
        assert!(
            scratch.address() % alignment == 0,
            "a scratch buffer's address has to be a multiple of {alignment}"
        );

        let mut described: SmallVec<[SmallVec<[Described; 1]>; 1]> = SmallVec::new();
        let mut shapes: SmallVec<[SmallVec<[Shape; 1]>; 1]> = SmallVec::new();
        let mut accesses: SmallVec<[BufferAccess; 8]> = SmallVec::new();
        let mut offset = 0;
        for (i, (structure, geometry)) in builds.iter().enumerate() {
            let same = builds[..i].iter().position(|(other, _)| other.handle == structure.handle);
            if let Some(first) = same {
                panic!("builds {first} and {i} of one batch write the same structure");
            }
            let mut geometries = SmallVec::new();
            geometry.describe(&mut geometries);
            assert_eq!(
                geometries.len(),
                structure.max_primitives.len(),
                "a structure is built from as many geometries as it was created for"
            );
            for (g, max) in geometries.iter().zip(&structure.max_primitives) {
                assert!(
                    g.primitives <= *max,
                    "a geometry of {} primitives doesn't fit a structure created for {max}",
                    g.primitives
                );
            }
            assert!(
                !update || structure.flags.allow_update,
                "only a structure with `allow_update` can be updated"
            );
            assert!(
                update || !structure.is_compacted(),
                "a compacted structure can't be built again: its memory only holds the \
                 compacted structure (build the one it was compacted from, or a new one)"
            );
            let shape: SmallVec<[Shape; 1]> = geometries.iter().map(Shape::of).collect();
            if update {
                let built = lock(&structure.built);
                assert!(!built.is_empty(), "an update refits a structure that was built before");
                if let Some((g, (last, new))) =
                    built.iter().zip(&shape).enumerate().find(|(_, (last, new))| last != new)
                {
                    panic!(
                        "an update keeps everything of the last build but positions: geometry \
                         {g} was built as {last:?}, not {new:?}"
                    );
                }
            }
            shapes.push(shape);
            let size = if update {
                structure.update_scratch
            } else {
                structure.build_scratch
            };
            let chunk: Range<u64> =
                (scratch.address() + offset..scratch.address() + offset + size).into();
            offset += size;

            let dst = if update { READ | WRITE } else { WRITE };
            accesses.push(structure.access(BUILD, dst));
            accesses.push(BufferAccess {
                unordered: false,
                stage: BUILD,
                access: READ | WRITE,
                range: chunk,
            });
            for input in geometries
                .iter()
                .flat_map(|g| g.inputs)
                .filter(|r| !r.is_empty())
            {
                accesses.push(BufferAccess {
                    unordered: false,
                    stage: BUILD,
                    access: vk::AccessFlags2::SHADER_READ,
                    range: input,
                });
            }
            described.push(geometries);
        }
        assert!(
            offset <= scratch.size(),
            "the builds need {offset} scratch bytes, the slice has {}",
            scratch.size()
        );
        if L::TY == vk::AccelerationStructureTypeKHR::TOP_LEVEL {
            accesses.push(every_structure_read(BUILD));
        }
        self.rec.flush_pending(std::iter::empty(), accesses.iter().cloned());
        for ((structure, _), shape) in builds.iter().zip(shapes) {
            *lock(&structure.built) = shape;
        }

        let geometries: SmallVec<[SmallVec<[vk::AccelerationStructureGeometryKHR; 1]>; 1]> =
            described
                .iter()
                .map(|d| d.iter().map(|g| g.geometry).collect())
                .collect();
        let ranges: SmallVec<[SmallVec<[vk::AccelerationStructureBuildRangeInfoKHR; 1]>; 1]> =
            described
                .iter()
                .map(|d| {
                    d.iter()
                        .map(|g| vk::AccelerationStructureBuildRangeInfoKHR {
                            primitive_count: g.primitives,
                            ..Default::default()
                        })
                        .collect()
                })
                .collect();
        let mut offset = 0;
        let infos: SmallVec<[_; 1]> = builds
            .iter()
            .zip(&geometries)
            .map(|((structure, _), geometries)| {
                let info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                    .ty(L::TY)
                    .flags(structure.flags.vk())
                    .geometries(geometries)
                    .dst_acceleration_structure(structure.handle)
                    .scratch_data(vk::DeviceOrHostAddressKHR {
                        device_address: scratch.address() + offset,
                    });
                offset += if update {
                    structure.update_scratch
                } else {
                    structure.build_scratch
                };
                if update {
                    info.mode(vk::BuildAccelerationStructureModeKHR::UPDATE)
                        .src_acceleration_structure(structure.handle)
                } else {
                    info.mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                }
            })
            .collect();
        // Not ash 0.38's wrapper, which collects these into a `Vec`.
        let ranges: SmallVec<[_; 1]> = ranges.iter().map(|r| r.as_ptr()).collect();
        unsafe {
            (command_functions().fp().cmd_build_acceleration_structures_khr)(
                self.rec.handle,
                infos.len() as u32,
                infos.as_ptr(),
                ranges.as_ptr(),
            )
        };
    }

    /// Writes the compacted sizes of `structures` (each with `allow_compaction`, built before)
    /// into the queries of `sizes` from `first` on.
    #[validation_trace]
    pub fn query_compacted_sizes<L: Level>(
        &mut self,
        structures: &[&AccelerationStructure<L>],
        sizes: &QueryPool<CompactedSize>,
        first: u32,
    ) {
        let count = u32::try_from(structures.len()).unwrap_or(u32::MAX);
        let end = first.checked_add(count);
        assert!(
            end.is_some_and(|end| end <= sizes.count()),
            "{count} sizes from {first} on don't fit {} queries",
            sizes.count()
        );
        assert!(
            structures.iter().all(|s| s.flags.allow_compaction),
            "only structures with `allow_compaction` have a compacted size"
        );
        for structure in structures {
            structure.check_built("compacted sizes are of");
        }
        let accesses: SmallVec<[_; 8]> = structures.iter().map(|s| s.access(BUILD, READ)).collect();
        self.rec.flush_pending(std::iter::empty(), accesses.iter().cloned());
        sizes.take(&self.rec, first, count);
        let handles: SmallVec<[_; 8]> = structures.iter().map(|s| s.handle).collect();
        unsafe {
            command_functions().cmd_write_acceleration_structures_properties(
                self.rec.handle,
                &handles,
                vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
                sizes.handle,
                first,
            );
        }
    }

    /// Copies `src` (built, with `allow_compaction`) into `dst`, made by `src.compacted(size)`;
    /// `dst` can then be updated like `src`.
    /// `src` can be dropped right after.
    #[validation_trace]
    pub fn compact<L: Level>(
        &mut self,
        src: &AccelerationStructure<L>,
        dst: &AccelerationStructure<L>,
    ) {
        assert!(
            src.flags.allow_compaction,
            "only a structure with `allow_compaction` can be compacted"
        );
        assert!(
            dst.compacted_from == Some(src.handle),
            "`compact` copies into a structure made by `compacted` of its source"
        );
        src.check_built("`compact` copies");
        let accesses = [src.access(BUILD, READ), dst.access(BUILD, WRITE)];
        self.rec.flush_pending(std::iter::empty(), accesses.iter().cloned());
        *lock(&dst.built) = lock(&src.built).clone();
        unsafe {
            command_functions().cmd_copy_acceleration_structure(
                self.rec.handle,
                &vk::CopyAccelerationStructureInfoKHR::default()
                    .src(src.handle)
                    .dst(dst.handle)
                    .mode(vk::CopyAccelerationStructureModeKHR::COMPACT),
            )
        };
    }
}

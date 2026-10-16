//! Ray tracing acceleration structures: `Blas`/`Tlas` owning their memory, their geometry (triangles, AABBs, instances), builds, refits and compaction with automatic barriers
use std::{marker::PhantomData, range::Range};

use crate::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Affine3A, Vec3, Vec4};
use lava_macros::validation_trace;
use smallvec::{SmallVec, smallvec};

use crate::{
    buffer::{
        Buffer,
        slice::BufferSlice,
        usage::{self, BufferUsage, IsStorage},
    },
    command_buffer::{BufferAccess, CommandBuffer, EVERY_ACCELERATION_STRUCTURE},
    epoch,
    error::{Error, Result},
    query::{CompactedSize, QueryPool},
    state::{Ctx, Functions},
};

const BUILD: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR;
const READ: vk::AccessFlags2 = vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR;
const WRITE: vk::AccessFlags2 = vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR;
/// What a trace or ray query reads acceleration structures with. `MEMORY_READ` is for Mesa's
/// ANV (26.2.2 on the B390): after a structure traced earlier in the same submission was
/// rebuilt, `ACCELERATION_STRUCTURE_READ` alone leaves the next trace seeing stale nodes (only
/// HDC flushes, `anv_pipe_invalidate_bits_for_access_flags`), while `MEMORY_READ` invalidates
/// every cache (`a_tlas_rebuilt_between_two_traces_waits_for_the_first`). It only widens
/// barriers that follow a structure write anyway.
const TRACE_READ: vk::AccessFlags2 =
    vk::AccessFlags2::from_raw(READ.as_raw() | vk::AccessFlags2::MEMORY_READ.as_raw());
/// What `vkCreateAccelerationStructure2KHR` requires of a structure's address and size.
const STRUCTURE_ALIGNMENT: u64 = 256;

pub(crate) fn functions() -> Result<&'static ash::khr::acceleration_structure::Device> {
    Functions::acceleration_structure().ok_or_else(|| {
        Error::message(
            "acceleration structures aren't supported by this device (`Ctx::features().acceleration_structure`)",
        )
    })
}

fn command_functions() -> &'static ash::khr::acceleration_structure::Device {
    functions().unwrap_or_else(|err| panic!("{err}"))
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

/// The read a trace or a TLAS build makes of every acceleration structure: the BLASes it
/// reaches, which lava can't list.
pub(crate) fn every_structure_read(stage: vk::PipelineStageFlags2) -> BufferAccess {
    let access = if stage == BUILD { READ } else { TRACE_READ };
    BufferAccess {
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

/// An acceleration structure and the memory it owns. Shaders reach a [`Tlas`] by its address
/// (a `u64` in the shader contract); instances name a [`Blas`] by its address. Builds write it through a
/// shared reference: the command buffer orders them with the traces. Destroyed once no frame
/// can reach it any more (`epoch`); a TLAS that still names a dropped BLAS must not be traced
/// after that.
pub struct AccelerationStructure<L: Level> {
    handle: vk::AccelerationStructureKHR,
    address: u64,
    flags: BuildFlags,
    /// Per geometry: the most primitives a build may have.
    max_primitives: SmallVec<[u32; 4]>,
    build_scratch: u64,
    update_scratch: u64,
    /// Dropped after the structure, so destroyed after it.
    buffer: Buffer<u8, usage::AccelerationStructureStorage>,
    _level: PhantomData<L>,
}

fn round_up(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment) * alignment
}

impl<L: Level> AccelerationStructure<L> {
    /// An unbuilt structure for builds of `geometries` with at most `max_primitives` each.
    fn with_geometries(
        geometries: &[vk::AccelerationStructureGeometryKHR<'_>],
        max_primitives: SmallVec<[u32; 4]>,
        flags: BuildFlags,
    ) -> Result<Self> {
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
        )
    }

    fn create(
        size: u64,
        flags: BuildFlags,
        max_primitives: SmallVec<[u32; 4]>,
        build_scratch: u64,
        update_scratch: u64,
    ) -> Result<Self> {
        let functions = functions()?;
        let buffer = Buffer::raw(
            usage::AccelerationStructureStorage,
            false,
            round_up(size, STRUCTURE_ALIGNMENT),
            Some(STRUCTURE_ALIGNMENT as u32),
        )?;
        let range = buffer.range(..);
        let create_info = vk::AccelerationStructureCreateInfo2KHR::default()
            .address_range(range.address_range())
            .address_flags(range.address_flags())
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
            buffer,
            _level: PhantomData,
        })
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
    /// one's flags, for [`CommandBuffer::compact`].
    pub fn compacted(&self, size: u64) -> Result<Self> {
        Self::create(
            size,
            self.flags,
            self.max_primitives.clone(),
            self.build_scratch,
            self.update_scratch,
        )
    }

    /// An access of its memory by an acceleration structure command or a shader.
    pub(crate) fn access(
        &self,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
    ) -> BufferAccess {
        BufferAccess {
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

/// Destroyed once no frame can reach it any more (`epoch`).
impl<L: Level> Drop for AccelerationStructure<L> {
    fn drop(&mut self) {
        let handle = self.handle;
        epoch::retire(move || {
            if let Ok(functions) = functions() {
                unsafe { functions.destroy_acceleration_structure(handle, None) };
            }
        });
    }
}

// ---- geometry ---------------------------------------------------------------------------

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

/// An index type a triangle geometry can use.
pub trait IndexFormat: crate::sealed::Sealed + Copy + Pod {
    const TYPE: vk::IndexType;
}
impl crate::sealed::Sealed for u16 {}
impl IndexFormat for u16 {
    const TYPE: vk::IndexType = vk::IndexType::UINT16;
}
impl crate::sealed::Sealed for u32 {}
impl IndexFormat for u32 {
    const TYPE: vk::IndexType = vk::IndexType::UINT32;
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
    pub fn new<V: VertexPosition, U: BufferUsage>(vertices: BufferSlice<'a, V, U>) -> Self {
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

    pub fn indexed<V: VertexPosition, I: IndexFormat, U: BufferUsage, UI: BufferUsage>(
        vertices: BufferSlice<'a, V, U>,
        indices: BufferSlice<'a, I, UI>,
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

    fn with_indices<V: VertexPosition, U: BufferUsage>(
        vertices: BufferSlice<'a, V, U>,
        index_type: vk::IndexType,
        index_address: u64,
        primitives: u32,
        index_range: Range<u64>,
    ) -> Self {
        assert!(!vertices.is_empty(), "a triangle geometry needs vertices");
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
    pub fn transform<U: BufferUsage>(
        mut self,
        transform: BufferSlice<'a, TransformMatrix, U>,
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
    pub fn new<U: BufferUsage>(boxes: BufferSlice<'a, Aabb, U>) -> Self {
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
    /// offset and flags 0.
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
    pub fn new<U: BufferUsage>(instances: BufferSlice<'a, Instance, U>) -> Self {
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

// ---- commands ---------------------------------------------------------------------------

impl CommandBuffer {
    /// Builds every structure from its geometry in one `vkCmdBuildAccelerationStructuresKHR`,
    /// so the GPU may build them in parallel. Each takes its `build_scratch_size()` of
    /// `scratch` after the previous one's, whose address has to be a multiple of
    /// [`scratch_alignment`]. A geometry may have fewer primitives than at creation. A TLAS
    /// build waits for every pending BLAS build.
    #[validation_trace]
    pub fn build<L: Level, U: IsStorage>(
        &mut self,
        builds: &[(&AccelerationStructure<L>, L::Geometry<'_>)],
        scratch: BufferSlice<'_, u8, U>,
    ) {
        self.record_builds(builds, scratch, false);
    }

    /// Refits every structure in place for its geometry, which has the primitive counts of
    /// its last build (vertices and instances may have moved). Needs `allow_update`; the
    /// scratch is laid out like [`Self::build`]'s, by `update_scratch_size()`.
    #[validation_trace]
    pub fn update<L: Level, U: IsStorage>(
        &mut self,
        updates: &[(&AccelerationStructure<L>, L::Geometry<'_>)],
        scratch: BufferSlice<'_, u8, U>,
    ) {
        self.record_builds(updates, scratch, true);
    }

    fn record_builds<L: Level, U: IsStorage>(
        &mut self,
        builds: &[(&AccelerationStructure<L>, L::Geometry<'_>)],
        scratch: BufferSlice<'_, u8, U>,
        update: bool,
    ) {
        scratch.usage_check(vk::BufferUsageFlags::STORAGE_BUFFER);
        let alignment = scratch_alignment();
        assert!(
            scratch.address() % alignment == 0,
            "a scratch buffer's address has to be a multiple of {alignment}"
        );

        let mut described: SmallVec<[SmallVec<[Described; 1]>; 1]> = SmallVec::new();
        let mut accesses: SmallVec<[BufferAccess; 8]> = SmallVec::new();
        let mut offset = 0;
        for (structure, geometry) in builds {
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
        self.flush_pending(std::iter::empty(), accesses.iter().cloned());

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
        // The function pointer directly: ash 0.38's wrapper collects these pointers into a `Vec`.
        let ranges: SmallVec<[_; 1]> = ranges.iter().map(|r| r.as_ptr()).collect();
        unsafe {
            (command_functions().fp().cmd_build_acceleration_structures_khr)(
                self.handle,
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
        let count = structures.len() as u32;
        assert!(
            first + count <= sizes.count(),
            "{count} sizes from {first} on don't fit {} queries",
            sizes.count()
        );
        assert!(
            structures.iter().all(|s| s.flags.allow_compaction),
            "only structures with `allow_compaction` have a compacted size"
        );
        let accesses: SmallVec<[_; 8]> = structures.iter().map(|s| s.access(BUILD, READ)).collect();
        self.flush_pending(std::iter::empty(), accesses.iter().cloned());
        sizes.record(self.epoch);
        let handles: SmallVec<[_; 8]> = structures.iter().map(|s| s.handle).collect();
        unsafe {
            command_functions().cmd_write_acceleration_structures_properties(
                self.handle,
                &handles,
                vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
                sizes.handle,
                first,
            );
        }
    }

    /// Copies `src` (built, with `allow_compaction`) into `dst`, made by `src.compacted(size)`.
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
        let accesses = [src.access(BUILD, READ), dst.access(BUILD, WRITE)];
        self.flush_pending(std::iter::empty(), accesses.iter().cloned());
        unsafe {
            command_functions().cmd_copy_acceleration_structure(
                self.handle,
                &vk::CopyAccelerationStructureInfoKHR::default()
                    .src(src.handle)
                    .dst(dst.handle)
                    .mode(vk::CopyAccelerationStructureModeKHR::COMPACT),
            )
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat3, Quat};

    #[test]
    fn mirrors_have_the_layout_of_the_vulkan_structs() {
        assert_eq!(
            size_of::<Instance>(),
            size_of::<vk::AccelerationStructureInstanceKHR>()
        );
        assert_eq!(
            align_of::<Instance>(),
            align_of::<vk::AccelerationStructureInstanceKHR>()
        );
        assert_eq!(
            size_of::<TransformMatrix>(),
            size_of::<vk::TransformMatrixKHR>()
        );
        assert_eq!(size_of::<Aabb>(), size_of::<vk::AabbPositionsKHR>());
        assert_eq!(std::mem::offset_of!(Instance, blas), 56);
    }

    #[test]
    fn transforms_are_stored_by_rows() {
        let t = Affine3A::from_mat3_translation(
            Mat3::from_cols_array(&[1., 2., 3., 4., 5., 6., 7., 8., 9.]),
            Vec3::new(10., 11., 12.),
        );
        assert_eq!(
            TransformMatrix::from(t).0,
            [[1., 4., 7., 10.], [2., 5., 8., 11.], [3., 6., 9., 12.]]
        );
        let r = Affine3A::from_quat(Quat::from_rotation_z(1.0));
        let m = TransformMatrix::from(r).0;
        let v = Vec3::X;
        let row = |i: usize| Vec4::from(m[i]).truncate().dot(v);
        assert!((Vec3::new(row(0), row(1), row(2)) - r.transform_vector3(v)).length() < 1e-6);
    }

    #[test]
    fn instance_fields_pack_into_their_bits() {
        let instance = Instance::from_address(0xABCD, Affine3A::IDENTITY);
        assert_eq!(instance.custom_index_and_mask, 0xFF00_0000);
        let instance = instance
            .custom_index(0x12_3456)
            .mask(0x81)
            .sbt_offset(0xFF_FFFF)
            .flags(vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE)
            .custom_index(7);
        assert_eq!(instance.custom_index_and_mask, 0x8100_0007);
        assert_eq!(instance.sbt_offset_and_flags, 0x04FF_FFFF);
        assert_eq!(instance.sbt_offset(1).sbt_offset_and_flags, 0x0400_0001);
        assert_eq!(instance.blas, 0xABCD);
    }

    #[test]
    #[should_panic(expected = "doesn't fit 24 bits")]
    fn custom_indices_past_24_bits_panic() {
        Instance::from_address(0, Affine3A::IDENTITY).custom_index(1 << 24);
    }

    #[test]
    #[should_panic(expected = "doesn't fit 24 bits")]
    fn hit_group_offsets_past_24_bits_panic() {
        Instance::from_address(0, Affine3A::IDENTITY).sbt_offset(1 << 24);
    }

    #[test]
    fn build_flags_prefer_fast_traces_by_default() {
        type F = vk::BuildAccelerationStructureFlagsKHR;
        assert_eq!(BuildFlags::default().vk(), F::PREFER_FAST_TRACE);
        let all = BuildFlags {
            fast_build: true,
            allow_update: true,
            allow_compaction: true,
        };
        assert_eq!(
            all.vk(),
            F::PREFER_FAST_BUILD | F::ALLOW_UPDATE | F::ALLOW_COMPACTION
        );
    }
}

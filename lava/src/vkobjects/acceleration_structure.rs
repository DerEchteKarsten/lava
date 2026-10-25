//! Ray tracing acceleration structures: `Blas`/`Tlas` in their own memory or an arena's, the shapes they were created for (from geometry or counts), their geometry (triangles of sealed vertex formats, AABBs, and instances: written with their BLASes as references (`InstanceInfo`), which a TLAS build takes for the TLAS so traces keep and wait for them, or raw (`RawInstance`, `unsafe Instances::from_raw`), which waits for every structure), builds and refits (batched to fit a `Scratch`), copies, serialization and compaction with automatic barriers and debug scopes, and the built state (applied once a recording reaches the GPU) that traces check
use std::{
    marker::PhantomData,
    range::Range,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
};

use crate::half::f16_bits;
use crate::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Affine3A, Vec2, Vec3, Vec3A, Vec4};
use lava_macros::validation_trace;
use smallvec::SmallVec;

use crate::{
    bindless::Ranges,
    buffer::{Buffer, BufferInternal, Gpu, slice::BufferSlice},
    command_buffer::{
        BufferAccess, CommandBuffer, EVERY_ACCELERATION_STRUCTURE, Recording,
        access_state::AccessState,
    },
    error::{Error, Result},
    features::{Feature, gate},
    global::Live,
    object::{Header, Object, Strong},
    query::{CompactedSize, QueryKind, QueryPool, SerializedSize},
    queue::{HasCompute, lock},
    reference::KeepAlive,
    state::{Ctx, Functions},
};

pub use crate::vkobjects::compactor::Compactor;

const BUILD: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR;
const READ: vk::AccessFlags2 = vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR;
const WRITE: vk::AccessFlags2 = vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR;
/// What `vkCreateAccelerationStructure2KHR` requires of a structure's address and size, and
/// copies to and from memory of their buffer addresses.
const STRUCTURE_ALIGNMENT: u64 = 256;

/// What a trace reads of a structure: `ACCELERATION_STRUCTURE_READ`, plus `MEMORY_READ` on
/// Mesa's ANV (26.2.2), where after a structure traced earlier in the same submission was
/// rebuilt, the structure read alone lets the next trace see stale nodes
/// (`a_tlas_rebuilt_between_two_traces_waits_for_the_first`). Decided once at init
/// (`PhysicalDevice::stale_trace_reads`).
pub(crate) fn trace_read() -> vk::AccessFlags2 {
    match Ctx::physical_device().stale_trace_reads {
        true => READ | vk::AccessFlags2::MEMORY_READ,
        false => READ,
    }
}

/// `MissingFeature` where lava checks it.
pub(crate) fn functions() -> Result<&'static ash::khr::acceleration_structure::Device> {
    gate::AccelerationStructure::check()?;
    Ok(Functions::acceleration_structure())
}

/// Unchecked: a command's structure only exists past the check.
fn command_functions() -> &'static ash::khr::acceleration_structure::Device {
    Functions::acceleration_structure()
}

fn properties() -> &'static vk::PhysicalDeviceAccelerationStructurePropertiesKHR<'static> {
    // Structures exist only past the feature's check, which leaves the properties set.
    (Ctx::physical_device()
        .acceleration_structure_properties
        .as_ref())
    .expect("acceleration structure properties of a device with the feature")
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

/// The read a trace or TLAS build makes of the BLASes it reaches, which lava can't list:
/// `access` at `stage` of [`EVERY_ACCELERATION_STRUCTURE`].
pub(crate) fn every_structure_read<'a>(
    stage: vk::PipelineStageFlags2,
    access: vk::AccessFlags2,
) -> BufferAccess<'a> {
    BufferAccess {
        unordered: false,
        stage,
        access,
        range: EVERY_ACCELERATION_STRUCTURE,
        buffer: None,
        structure: None,
    }
}

/// An access that is none: what a trace of a TLAS whose BLASes it references adds.
fn no_access<'a>(stage: vk::PipelineStageFlags2) -> BufferAccess<'a> {
    BufferAccess {
        unordered: false,
        stage,
        access: vk::AccessFlags2::NONE,
        range: (0..0).into(),
        buffer: None,
        structure: None,
    }
}

/// What lava reads from geometry: only lava's types implement [`Describe`].
mod describe {
    use super::*;

    /// One geometry of a build: what Vulkan reads, its shape, and the buffer ranges it reads.
    pub struct Described<'a> {
        pub(crate) geometry: vk::AccelerationStructureGeometryKHR<'static>,
        pub(crate) shape: Shape,
        pub(crate) inputs: [Option<BufferSlice<'a, u8>>; 3],
    }

    /// The geometries of a build ([`Level::Geometry`]).
    pub trait Describe<'a> {
        fn describe(&self, out: &mut SmallVec<[Described<'a>; 1]>);
        /// Where a TLAS build's instances come from; none for a BLAS's geometry.
        fn instance_source(&self) -> Option<InstanceSource<'a>> {
            None
        }
    }

    /// The instances of a TLAS build: the buffer they were written to with their BLASes as
    /// references, or none for raw instances.
    #[derive(Clone, Copy)]
    pub struct InstanceSource<'a> {
        pub(crate) referenced: Option<&'a std::sync::Arc<crate::buffer::BufferInternal>>,
    }
}
use describe::{Describe, Described, InstanceSource};

/// One geometry as Vulkan sizes and builds it: everything but its addresses. A structure keeps
/// its creation's shapes, with the counts as maxima, and its last build's, which an update keeps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Shape {
    ty: vk::GeometryTypeKHR,
    flags: vk::GeometryFlagsKHR,
    primitives: u32,
    vertex_format: vk::Format,
    max_vertex: u32,
    index_type: vk::IndexType,
    transform: bool,
}

impl Shape {
    fn instances(count: u32) -> Self {
        Shape {
            ty: vk::GeometryTypeKHR::INSTANCES,
            flags: vk::GeometryFlagsKHR::empty(),
            primitives: count,
            vertex_format: vk::Format::UNDEFINED,
            max_vertex: 0,
            index_type: vk::IndexType::NONE_KHR,
            transform: false,
        }
    }

    fn aabbs(boxes: u32, opaque: bool) -> Self {
        Shape {
            ty: vk::GeometryTypeKHR::AABBS,
            flags: geometry_flags(opaque),
            ..Self::instances(boxes)
        }
    }

    /// Whether a build of this shape fits a structure created for `creation`: the same
    /// geometry, with at most its primitives and highest vertex.
    fn fits(&self, creation: &Shape) -> bool {
        let counted = Shape {
            primitives: creation.primitives,
            max_vertex: creation.max_vertex,
            ..*self
        };
        counted == *creation
            && self.primitives <= creation.primitives
            && self.max_vertex <= creation.max_vertex
    }
}

/// Whether a structure was built, shared with the recordings that build it.
pub(crate) struct BuildState {
    /// A build was recorded by a recording that hasn't failed, or reached the GPU: what a
    /// trace checks, one atomic load.
    recorded: AtomicBool,
    /// The shapes of the last build that reached the GPU (or compaction, copy or
    /// deserialization into it; none known after a deserialization); `None` before.
    built: Mutex<Option<SmallVec<[Shape; 1]>>>,
}

impl BuildState {
    fn new() -> Arc<Self> {
        Arc::new(BuildState {
            recorded: AtomicBool::new(false),
            built: Mutex::new(None),
        })
    }
}

/// The builds of a recording, which become the structures' built state once it reaches the GPU.
pub(crate) type RecordedBuilds = Vec<(Arc<BuildState>, SmallVec<[Shape; 1]>)>;

impl Recording<'_> {
    fn structure_built(&mut self, state: &Arc<BuildState>, shapes: SmallVec<[Shape; 1]>) {
        state.recorded.store(true, Ordering::Release);
        self.structure_builds.push((state.clone(), shapes));
    }

    /// The shapes of the structure's last build this recording recorded, else of the last that
    /// reached the GPU; `None` if neither happened.
    fn last_shapes(&self, state: &Arc<BuildState>) -> Option<SmallVec<[Shape; 1]>> {
        let recorded = self
            .structure_builds
            .iter()
            .rev()
            .find(|(s, _)| Arc::ptr_eq(s, state));
        match recorded {
            Some((_, shapes)) => Some(shapes.clone()),
            None => lock(&state.built).clone(),
        }
    }

    /// Applies the recording's builds once it reached the GPU (`reached`); else forgets them,
    /// so a structure only this recording built counts as unbuilt again.
    pub(crate) fn finish_structure_builds(&mut self, reached: bool) {
        let builds = std::mem::take(&mut self.structure_builds);
        if reached {
            for (state, shapes) in builds {
                *lock(&state.built) = Some(shapes);
            }
        } else {
            for (state, _) in builds {
                let built = lock(&state.built).is_some();
                state.recorded.store(built, Ordering::Release);
            }
        }
    }
}

/// The level of an acceleration structure: [`Bottom`] (geometry) or [`Top`] (instances).
#[allow(private_bounds)]
pub trait Level: crate::sealed::Sealed + 'static {
    const TY: vk::AccelerationStructureTypeKHR;
    /// What a build of this level reads.
    type Geometry<'a>: Describe<'a>;
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

/// Whether builds favour fast traces, fast builds, or neither (the driver's choice).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum BuildPreference {
    /// Slower builds for faster traces (`PREFER_FAST_TRACE`).
    #[default]
    FastTrace,
    /// Faster builds for slower traces (`PREFER_FAST_BUILD`).
    FastBuild,
    /// No preference: the driver's balance.
    None,
}

/// How a structure is built. The default prefers fast traces.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BuildFlags {
    pub prefer: BuildPreference,
    /// Allow [`CommandBuffer::update`].
    pub allow_update: bool,
    /// Allow [`CommandBuffer::query_compacted_sizes`] and [`CommandBuffer::compact`].
    pub allow_compaction: bool,
    /// Let hit shaders and ray queries read the vertex positions of the triangles they hit
    /// (`ALLOW_DATA_ACCESS`, `HitTriangleVertexPosition` in `ray_tracing.slang`); a creation
    /// with it returns `MissingFeature` on a device without position fetch.
    #[cfg(feature = "ray-tracing-position-fetch")]
    pub allow_data_access: bool,
}

impl BuildFlags {
    fn vk(self) -> vk::BuildAccelerationStructureFlagsKHR {
        type F = vk::BuildAccelerationStructureFlagsKHR;
        let mut flags = match self.prefer {
            BuildPreference::FastTrace => F::PREFER_FAST_TRACE,
            BuildPreference::FastBuild => F::PREFER_FAST_BUILD,
            BuildPreference::None => F::empty(),
        };
        if self.allow_update {
            flags |= F::ALLOW_UPDATE;
        }
        if self.allow_compaction {
            flags |= F::ALLOW_COMPACTION;
        }
        #[cfg(feature = "ray-tracing-position-fetch")]
        if self.allow_data_access {
            flags |= F::ALLOW_DATA_ACCESS;
        }
        flags
    }

    fn check(self) -> Result<()> {
        #[cfg(feature = "ray-tracing-position-fetch")]
        if self.allow_data_access {
            gate::RayTracingPositionFetch::check()?;
        }
        Ok(())
    }
}

/// How a structure came to be, which decides what may write it.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Origin {
    /// Sized for builds of its shapes.
    Created,
    /// Made by `compacted` of the structure with this handle: its memory only holds the
    /// compacted structure, so it can't be built again.
    Compacted(vk::AccelerationStructureKHR),
    /// Made by `clone_target`, sized like its source: buildable where that was.
    Copied { buildable: bool },
    /// Made for serialized data, whose shapes lava doesn't know: only traced, copied and
    /// serialized.
    Deserialized,
}

/// The memory a structure lives in.
enum Memory {
    /// Its own buffer, dropped after the structure, so destroyed after it.
    Own(Buffer<u8>),
    /// A range of an arena's buffer (in `STRUCTURE_ALIGNMENT` units), given back once the
    /// structure is destroyed.
    Arena {
        arena: Arc<ArenaInner>,
        units: std::ops::Range<u32>,
    },
}

/// Where a new structure goes.
#[derive(Clone, Copy)]
enum Placement<'a> {
    Own,
    Arena(&'a StructureArena),
}

/// An acceleration structure and its memory (its own, or a range of a [`StructureArena`]),
/// which builds write through a shared reference. A TLAS that still names a dropped BLAS must
/// not be traced ([`Instances::new`]).
pub struct AccelerationStructure<L: Level> {
    pub(crate) internal: Arc<StructureInternal>,
    _level: PhantomData<L>,
}

/// An acceleration structure as lava owns it, shared by its [`AccelerationStructure`] and
/// whatever else keeps it, destroyed with the last owner.
pub(crate) struct StructureInternal {
    pub(crate) handle: vk::AccelerationStructureKHR,
    address: u64,
    flags: BuildFlags,
    /// Per geometry, the shape it was created for, its counts the most a build may have; empty
    /// for a deserialized structure.
    shapes: SmallVec<[Shape; 1]>,
    build_scratch: u64,
    update_scratch: u64,
    origin: Origin,
    state: Arc<BuildState>,
    /// The device addresses of its memory, which barriers track.
    pub(crate) range: Range<u64>,
    memory: Memory,
    /// A TLAS's BLASes: what its last recorded build's instances referenced.
    keep_alive: OnceLock<Mutex<KeepAlive>>,
    /// Its last recorded build was from raw instances, whose BLASes lava doesn't know.
    raw_instances: AtomicBool,
    header: Header,
}

impl std::fmt::Debug for StructureInternal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (f.debug_struct("StructureInternal"))
            .field("handle", &self.handle)
            .field("address", &self.address)
            .finish_non_exhaustive()
    }
}

impl Object for StructureInternal {
    fn header(&self) -> &Header {
        &self.header
    }

    fn keep_alive(&self) -> Option<&Mutex<KeepAlive>> {
        self.keep_alive.get()
    }

    fn access_state(&self) -> Option<&Mutex<AccessState>> {
        Some(&self.buffer().state)
    }

    fn referencing_range(&self) -> Option<Range<u64>> {
        self.keep_alive.get()?;
        Some(self.range)
    }
}

impl StructureInternal {
    /// The buffer its memory lies in.
    fn buffer(&self) -> &Arc<BufferInternal> {
        match &self.memory {
            Memory::Own(buffer) => &buffer.internal,
            Memory::Arena { arena, .. } => &arena.buffer.internal,
        }
    }

    pub(crate) fn access(
        self: &Arc<Self>,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
    ) -> BufferAccess<'_> {
        BufferAccess {
            unordered: false,
            stage,
            access,
            range: self.range,
            buffer: Some(self.buffer()),
            structure: Some(self),
        }
    }

    /// Whether its last recorded build was from raw instances.
    pub(crate) fn from_raw_instances(&self) -> bool {
        self.raw_instances.load(Ordering::Relaxed)
    }

    /// Takes the BLASes `instances` references (none for raw instances) for its own, as a
    /// build from them is recorded.
    #[track_caller]
    fn take_instances(self: &Arc<Self>, source: InstanceSource<'_>) {
        let new = match source.referenced {
            Some(buffer) => {
                let Some(instances) = buffer.keep_alive.get() else {
                    panic!("a TLAS is built from instances that were never written")
                };
                let instances = lock(instances);
                for blas in instances.children.iter() {
                    if let Strong::Structure(blas) = &blas.object {
                        assert!(
                            blas.state.recorded.load(Ordering::Acquire),
                            "a TLAS is built from an instance of a BLAS that was never built: \
                             record its build first"
                        );
                    }
                }
                instances.clone_for(self)
            }
            None => KeepAlive::new::<Instance>(),
        };
        self.replace_instances(new, source.referenced.is_none());
    }

    /// Takes `src`'s BLASes, as a copy or compaction of it into this TLAS is recorded.
    fn copy_instances(self: &Arc<Self>, src: &StructureInternal) {
        if let Some(instances) = src.keep_alive.get() {
            let new = lock(instances).clone_for(self);
            self.replace_instances(new, src.from_raw_instances());
        }
    }

    /// Its BLASes are `new` from now on, or unknown (`raw`).
    fn replace_instances(self: &Arc<Self>, new: KeepAlive, raw: bool) {
        let keep_alive = self
            .keep_alive
            .get_or_init(|| Mutex::new(KeepAlive::new::<Instance>()));
        self.raw_instances.store(raw, Ordering::Relaxed);
        // The BLASes of the last build go once nothing references them (a recording that
        // traces this TLAS keeps them through its submission).
        let old = std::mem::replace(&mut *lock(keep_alive), new);
        old.unregister(self);
    }
}

fn round_up(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment) * alignment
}

/// `Error::OverLimit` if `requested` is above `max`.
fn limit(limit: &'static str, requested: u64, max: u64) -> Result<()> {
    match requested <= max {
        true => Ok(()),
        false => Err(Error::OverLimit {
            limit,
            requested,
            max,
        }),
    }
}

impl<L: Level> AccelerationStructure<L> {
    /// A structure for builds of `described` (its shapes the maxima), sized by Vulkan.
    fn with_geometries(
        described: &[Described<'_>],
        flags: BuildFlags,
        placement: Placement,
    ) -> Result<Self> {
        let live = Live::new();
        functions()?;
        flags.check()?;
        let properties = properties();
        limit(
            "maxGeometryCount",
            described.len() as u64,
            properties.max_geometry_count,
        )?;
        let primitives: u64 = described.iter().map(|d| d.shape.primitives as u64).sum();
        match L::TY {
            vk::AccelerationStructureTypeKHR::TOP_LEVEL => limit(
                "maxInstanceCount",
                primitives,
                properties.max_instance_count,
            )?,
            _ => limit(
                "maxPrimitiveCount",
                primitives,
                properties.max_primitive_count,
            )?,
        }
        let geometries: SmallVec<[_; 1]> = described.iter().map(|d| d.geometry).collect();
        let max_primitives: SmallVec<[u32; 1]> =
            described.iter().map(|d| d.shape.primitives).collect();
        let info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(L::TY)
            .flags(flags.vk())
            .geometries(&geometries);
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            command_functions().get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &info,
                &max_primitives,
                &mut sizes,
            )
        };
        let alignment = scratch_alignment();
        let update_scratch = match flags.allow_update {
            true => round_up(sizes.update_scratch_size, alignment),
            false => 0,
        };
        Self::create(
            sizes.acceleration_structure_size,
            flags,
            described.iter().map(|d| d.shape).collect(),
            round_up(sizes.build_scratch_size, alignment),
            update_scratch,
            Origin::Created,
            placement,
            live,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn create(
        size: u64,
        flags: BuildFlags,
        shapes: SmallVec<[Shape; 1]>,
        build_scratch: u64,
        update_scratch: u64,
        origin: Origin,
        placement: Placement,
        live: Live,
    ) -> Result<Self> {
        // Every caller checked the feature, or holds a structure, which exists only past it.
        let functions = command_functions();
        let size = round_up(size.max(1), STRUCTURE_ALIGNMENT);
        let (memory, range) = match placement {
            Placement::Own => {
                let buffer = Buffer::with_flags(
                    vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
                    Gpu,
                    size,
                    Some(STRUCTURE_ALIGNMENT),
                )?;
                let range = buffer.range(..).get_range();
                (Memory::Own(buffer), range)
            }
            Placement::Arena(arena) => {
                let units = arena.take(size)?;
                let start = arena.inner.buffer.address() + units.start as u64 * STRUCTURE_ALIGNMENT;
                let memory = Memory::Arena {
                    arena: arena.inner.clone(),
                    units,
                };
                (memory, (start..start + size).into())
            }
        };
        // The buffer has no storage usage.
        let create_info = vk::AccelerationStructureCreateInfo2KHR::default()
            .address_range(vk::DeviceAddressRangeKHR {
                address: range.start,
                size: range.end - range.start,
            })
            .address_flags(
                vk::AddressCommandFlagsKHR::UNKNOWN_STORAGE_BUFFER_USAGE
                    | vk::AddressCommandFlagsKHR::FULLY_BOUND,
            )
            .ty(L::TY);
        let created = unsafe {
            Functions::device_address_commands().create_acceleration_structure2(&create_info, None)
        };
        let handle = match created {
            Ok(handle) => handle,
            Err(err) => {
                // An arena range goes back to the arena.
                if let Memory::Arena { arena, units } = memory {
                    lock(&arena.free).give(units);
                }
                return Err(err.into());
            }
        };
        let address = unsafe {
            functions.get_acceleration_structure_device_address(
                &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                    .acceleration_structure(handle),
            )
        };
        Ok(Self {
            internal: Arc::new(StructureInternal {
                handle,
                address,
                flags,
                shapes,
                build_scratch,
                update_scratch,
                origin,
                state: BuildState::new(),
                range,
                memory,
                keep_alive: OnceLock::new(),
                raw_instances: AtomicBool::new(false),
                header: Header::new(live),
            }),
            _level: PhantomData,
        })
    }

    /// The same structure named `name` (its own memory too), for debuggers, the validation
    /// layer (with debug utils, `Config::validation`) and allocator reports.
    pub fn named(mut self, name: &str) -> Self {
        if let Some(internal) = Arc::get_mut(&mut self.internal)
            && let Memory::Own(buffer) = &mut internal.memory
        {
            buffer.set_name(name);
        }
        Functions::set_debug_name(name, self.internal.handle);
        self
    }

    /// Its device address, which shaders and instances use. Not its buffer's address.
    pub fn address(&self) -> u64 {
        self.internal.address
    }

    /// The bytes its memory takes.
    pub fn size(&self) -> u64 {
        self.internal.range.end - self.internal.range.start
    }

    pub fn flags(&self) -> BuildFlags {
        self.internal.flags
    }

    /// The scratch bytes a build needs, a multiple of [`scratch_alignment`]; 0 for structures
    /// that can't be built (compacted, deserialized).
    pub fn build_scratch_size(&self) -> u64 {
        self.internal.build_scratch
    }

    /// The scratch bytes an update needs, a multiple of [`scratch_alignment`]; 0 without
    /// `allow_update`.
    pub fn update_scratch_size(&self) -> u64 {
        self.internal.update_scratch
    }

    /// Whether a build (or copy into it) was recorded by a recording that didn't fail, or
    /// reached the GPU: what traces, compaction and copies need.
    pub fn is_built(&self) -> bool {
        self.internal.state.recorded.load(Ordering::Acquire)
    }

    /// Whether it was made by `compacted`, so only `compact` and `update` write it.
    pub fn is_compacted(&self) -> bool {
        matches!(self.internal.origin, Origin::Compacted(_))
    }

    /// Whether `build` may write it: it was sized for builds.
    pub fn is_buildable(&self) -> bool {
        match self.internal.origin {
            Origin::Created => true,
            Origin::Copied { buildable } => buildable,
            Origin::Compacted(_) | Origin::Deserialized => false,
        }
    }

    #[track_caller]
    fn compaction_check(&self) {
        assert!(
            self.internal.flags.allow_compaction,
            "only a structure built with `allow_compaction` can be compacted"
        );
        assert!(
            self.is_built(),
            "compaction needs a structure that was built before"
        );
    }

    /// An unbuilt structure of `size` bytes (read from a `QueryPool<CompactedSize>`) with this
    /// one's flags, for [`CommandBuffer::compact`] of this one, which has `allow_compaction`
    /// and was built (panics otherwise). It can be updated but never built. A [`Compactor`]
    /// does every step.
    #[track_caller]
    pub fn compacted(&self, size: u64) -> Result<Self> {
        self.compacted_in(size, Placement::Own)
    }

    #[track_caller]
    fn compacted_in(&self, size: u64, placement: Placement) -> Result<Self> {
        self.compaction_check();
        Self::create(
            size,
            self.internal.flags,
            self.internal.shapes.clone(),
            0,
            self.internal.update_scratch,
            Origin::Compacted(self.internal.handle),
            placement,
            Live::new(),
        )
    }

    /// An unbuilt structure like this one (its size, flags and shapes), for
    /// [`CommandBuffer::copy_structure`] of it: buildable and updatable where this one is.
    pub fn clone_target(&self) -> Result<Self> {
        self.clone_target_in(Placement::Own)
    }

    fn clone_target_in(&self, placement: Placement) -> Result<Self> {
        Self::create(
            self.size(),
            self.internal.flags,
            self.internal.shapes.clone(),
            self.internal.build_scratch,
            self.internal.update_scratch,
            Origin::Copied {
                buildable: self.is_buildable(),
            },
            placement,
            Live::new(),
        )
    }

    /// An unbuilt structure for [`CommandBuffer::deserialize`] of `serialized`, the data
    /// [`CommandBuffer::serialize`] wrote (at least its 56-byte header): `Error::Incompatible`
    /// unless this device and driver read it (`vkGetDeviceAccelerationStructureCompatibilityKHR`).
    /// lava doesn't know the shapes it was built from, so it can be traced, copied and
    /// serialized, but not built, updated or compacted. A TLAS names BLASes by their addresses
    /// in the data: deserialize those first, and rebuild instead where they moved.
    #[track_caller]
    pub fn deserialized(serialized: &[u8]) -> Result<Self> {
        Self::deserialized_in(serialized, Placement::Own)
    }

    #[track_caller]
    fn deserialized_in(serialized: &[u8], placement: Placement) -> Result<Self> {
        let live = Live::new();
        let functions = functions()?;
        assert!(
            serialized.len() >= SERIALIZED_HEADER,
            "serialized acceleration structures start with a {SERIALIZED_HEADER}-byte header, \
             not {} bytes",
            serialized.len()
        );
        let version: &[u8; 2 * vk::UUID_SIZE] = serialized[..2 * vk::UUID_SIZE].try_into().unwrap();
        let compatibility = unsafe {
            functions.get_device_acceleration_structure_compatibility(
                &vk::AccelerationStructureVersionInfoKHR::default().version_data(version),
            )
        };
        if compatibility != vk::AccelerationStructureCompatibilityKHR::COMPATIBLE {
            return Err(Error::Incompatible);
        }
        let size = u64::from_ne_bytes(serialized[40..48].try_into().unwrap());
        Self::create(
            size,
            BuildFlags::default(),
            SmallVec::new(),
            0,
            0,
            Origin::Deserialized,
            placement,
            live,
        )
    }

    pub(crate) fn access(
        &self,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
    ) -> BufferAccess<'_> {
        self.internal.access(stage, access)
    }
}

/// The bytes of a serialized structure's header: the driver and compatibility UUIDs, the
/// serialized and deserialized sizes and the count of BLAS handles that follow it.
const SERIALIZED_HEADER: usize = 2 * vk::UUID_SIZE + 3 * 8;

impl Blas {
    /// An unbuilt BLAS for builds of `geometry`: the same geometries, with at most their
    /// primitives and vertices. [`Self::with_capacity`] sizes one from counts alone.
    pub fn new(geometry: &BlasGeometry<'_>, flags: BuildFlags) -> Result<Self> {
        let mut described = SmallVec::new();
        geometry.describe(&mut described);
        Self::with_geometries(&described, flags, Placement::Own)
    }

    /// An unbuilt BLAS for builds of geometries like `desc`'s, sized from their formats and
    /// counts without buffers.
    pub fn with_capacity(desc: &BlasDesc<'_>, flags: BuildFlags) -> Result<Self> {
        Self::with_geometries(&desc.described(), flags, Placement::Own)
    }
}

impl Tlas {
    /// An unbuilt TLAS for builds of up to `max_instances` instances.
    pub fn new(max_instances: u32, flags: BuildFlags) -> Result<Self> {
        Self::with_geometries(
            &[instances_described(0, max_instances)],
            flags,
            Placement::Own,
        )
    }

    /// The accesses of a pass that traces it: its memory (through which a command reaches the
    /// BLASes its instances referenced), and for a TLAS built from raw instances a read of every
    /// structure. Panics unless a build of it was recorded ([`AccelerationStructure::is_built`]):
    /// a trace of an unbuilt TLAS reads uninitialised memory.
    #[track_caller]
    pub fn trace_accesses(&self, stage: vk::PipelineStageFlags2) -> [BufferAccess<'_>; 2] {
        assert!(
            self.is_built(),
            "a trace of a TLAS that was never built: record its build first"
        );
        let read = trace_read();
        let blases = match self.internal.from_raw_instances() {
            true => every_structure_read(stage, read),
            false => no_access(stage),
        };
        [self.access(stage, read), blases]
    }
}

impl Drop for StructureInternal {
    fn drop(&mut self) {
        if let Some(keep_alive) = self.keep_alive.get() {
            lock(keep_alive).unregister_all();
        }
        let handle = self.handle;
        let arena = match &self.memory {
            Memory::Own(_) => None,
            Memory::Arena { arena, units } => Some((arena.clone(), units.clone())),
        };
        unsafe { command_functions().destroy_acceleration_structure(handle, None) };
        if let Some((arena, units)) = arena {
            lock(&arena.free).give(units);
        }
    }
}

/// Memory for many acceleration structures in one buffer, which they take ranges of (first
/// fit, 256-byte aligned) and give back once they are destroyed: fewer buffers and allocations
/// than one per structure. Its structures keep the buffer alive, so the arena can be dropped
/// before them.
pub struct StructureArena {
    inner: Arc<ArenaInner>,
    /// Last, so it counts the arena until its drop finished.
    _live: Live,
}

struct ArenaInner {
    buffer: Buffer<u8>,
    /// Free ranges in `STRUCTURE_ALIGNMENT` units.
    free: Mutex<Ranges>,
}

impl StructureArena {
    /// An arena of `bytes` (rounded up to 256).
    pub fn new(bytes: u64) -> Result<Self> {
        let live = Live::new();
        functions()?;
        let units = u32::try_from(bytes.div_ceil(STRUCTURE_ALIGNMENT))
            .map_err(|_| Error::other(format!("an arena of {bytes} bytes is larger than 1 TiB")))?;
        let buffer = Buffer::with_flags(
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
            Gpu,
            units as u64 * STRUCTURE_ALIGNMENT,
            Some(STRUCTURE_ALIGNMENT),
        )?;
        Ok(StructureArena {
            inner: Arc::new(ArenaInner {
                buffer,
                free: Mutex::new(Ranges::all(units)),
            }),
            _live: live,
        })
    }

    /// The same arena with its memory named `name` (before structures are placed in it).
    pub fn named(mut self, name: &str) -> Self {
        if let Some(inner) = Arc::get_mut(&mut self.inner) {
            inner.buffer.set_name(name);
        }
        self
    }

    /// The bytes its buffer holds.
    pub fn size(&self) -> u64 {
        self.inner.buffer.size()
    }

    fn take(&self, size: u64) -> Result<std::ops::Range<u32>> {
        let full = || Error::ArenaFull {
            requested: size,
            capacity: self.size(),
        };
        let units = u32::try_from(size.div_ceil(STRUCTURE_ALIGNMENT)).map_err(|_| full())?;
        let start = lock(&self.inner.free).take(units).ok_or_else(full)?;
        Ok(start..start + units)
    }

    /// [`Blas::new`] in the arena; `Error::ArenaFull` without room.
    pub fn blas(&self, geometry: &BlasGeometry<'_>, flags: BuildFlags) -> Result<Blas> {
        let mut described = SmallVec::new();
        geometry.describe(&mut described);
        Blas::with_geometries(&described, flags, Placement::Arena(self))
    }

    /// [`Blas::with_capacity`] in the arena.
    pub fn blas_with_capacity(&self, desc: &BlasDesc<'_>, flags: BuildFlags) -> Result<Blas> {
        Blas::with_geometries(&desc.described(), flags, Placement::Arena(self))
    }

    /// [`Tlas::new`] in the arena.
    pub fn tlas(&self, max_instances: u32, flags: BuildFlags) -> Result<Tlas> {
        Tlas::with_geometries(
            &[instances_described(0, max_instances)],
            flags,
            Placement::Arena(self),
        )
    }

    /// [`AccelerationStructure::compacted`] in the arena.
    #[track_caller]
    pub fn compacted<L: Level>(
        &self,
        source: &AccelerationStructure<L>,
        size: u64,
    ) -> Result<AccelerationStructure<L>> {
        source.compacted_in(size, Placement::Arena(self))
    }

    /// [`AccelerationStructure::clone_target`] in the arena.
    pub fn clone_target<L: Level>(
        &self,
        source: &AccelerationStructure<L>,
    ) -> Result<AccelerationStructure<L>> {
        source.clone_target_in(Placement::Arena(self))
    }

    /// [`AccelerationStructure::deserialized`] in the arena.
    #[track_caller]
    pub fn deserialized<L: Level>(&self, serialized: &[u8]) -> Result<AccelerationStructure<L>> {
        AccelerationStructure::deserialized_in(serialized, Placement::Arena(self))
    }
}

/// A buffer for the scratch memory of builds and updates, aligned to [`scratch_alignment`].
/// [`CommandBuffer::build_all`] splits builds into batches that fit it.
pub struct Scratch {
    buffer: Buffer<u8>,
}

impl Scratch {
    /// At least `bytes` of scratch memory (rounded up to the alignment).
    pub fn new(bytes: u64) -> Result<Self> {
        let alignment = scratch_alignment();
        let buffer = Buffer::raw(Gpu, round_up(bytes.max(1), alignment), Some(alignment))?;
        Ok(Scratch { buffer })
    }

    /// Enough scratch for one build or update of the largest of `structures`, the least
    /// [`CommandBuffer::build_all`] needs.
    pub fn for_any<L: Level>(structures: &[&AccelerationStructure<L>]) -> Result<Self> {
        let sizes = structures
            .iter()
            .map(|s| s.internal.build_scratch.max(s.internal.update_scratch));
        Self::new(sizes.max().unwrap_or(0))
    }

    /// The same scratch with its memory named `name`.
    pub fn named(self, name: &str) -> Self {
        Scratch {
            buffer: self.buffer.named(name),
        }
    }

    /// The bytes it holds.
    pub fn size(&self) -> u64 {
        self.buffer.size()
    }

    /// All of it, for [`CommandBuffer::build`] and [`CommandBuffer::update`].
    pub fn slice(&self) -> BufferSlice<'_, u8> {
        self.buffer.range(..)
    }
}

/// A vertex position format of triangle geometries: the formats every device with
/// acceleration structures builds from. Two-component positions have a z of 0, and a fourth
/// component is ignored.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum VertexFormat {
    Float32x2,
    Float32x3,
    Float16x2,
    Float16x4,
    Snorm16x2,
    Snorm16x4,
}

impl VertexFormat {
    fn vk(self) -> vk::Format {
        use vk::Format as F;
        match self {
            VertexFormat::Float32x2 => F::R32G32_SFLOAT,
            VertexFormat::Float32x3 => F::R32G32B32_SFLOAT,
            VertexFormat::Float16x2 => F::R16G16_SFLOAT,
            VertexFormat::Float16x4 => F::R16G16B16A16_SFLOAT,
            VertexFormat::Snorm16x2 => F::R16G16_SNORM,
            VertexFormat::Snorm16x4 => F::R16G16B16A16_SNORM,
        }
    }

    /// The bytes of a component, which vertex addresses and strides are multiples of.
    fn component_bytes(self) -> u64 {
        match self {
            VertexFormat::Float32x2 | VertexFormat::Float32x3 => 4,
            _ => 2,
        }
    }
}

/// A vertex type whose first bytes are its position, in `FORMAT`; its size is the stride. Sealed:
/// lava's position types, which cover every [`VertexFormat`].
pub trait VertexPosition: crate::sealed::Sealed + Copy + 'static {
    const FORMAT: VertexFormat;
}

macro_rules! vertex_positions {
    ($($ty:ty: $format:ident;)*) => {$(
        impl crate::sealed::Sealed for $ty {}
        impl VertexPosition for $ty {
            const FORMAT: VertexFormat = VertexFormat::$format;
        }
    )*};
}
vertex_positions! {
    [f32; 2]: Float32x2;
    Vec2: Float32x2;
    [f32; 3]: Float32x3;
    Vec3: Float32x3;
    // 16 bytes apart; the fourth component (or padding) is ignored.
    [f32; 4]: Float32x3;
    Vec4: Float32x3;
    Vec3A: Float32x3;
    Half2: Float16x2;
    Half4: Float16x4;
    Snorm16x2: Snorm16x2;
    Snorm16x4: Snorm16x4;
}

/// Two half floats (IEEE binary16 bits): an x, y position (z = 0).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct Half2(pub [u16; 2]);

/// Four half floats (IEEE binary16 bits): a position and an ignored fourth component, 8 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct Half4(pub [u16; 4]);

/// Two signed normalized 16-bit integers: an x, y position in -1 to 1 (z = 0).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct Snorm16x2(pub [i16; 2]);

/// Four signed normalized 16-bit integers: a position in -1 to 1 and an ignored fourth
/// component, 8 bytes. Scale vertices into the unit cube and the instance transform back.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct Snorm16x4(pub [i16; 4]);

impl Half2 {
    pub fn new(position: Vec2) -> Self {
        Half2([f16_bits(position.x), f16_bits(position.y)])
    }
}

impl Half4 {
    /// `position` rounded to half floats (to nearest, ties to even; infinite beyond 65504).
    pub fn new(position: Vec3) -> Self {
        Half4([
            f16_bits(position.x),
            f16_bits(position.y),
            f16_bits(position.z),
            0,
        ])
    }
}

impl Snorm16x2 {
    pub fn new(position: Vec2) -> Self {
        Snorm16x2([snorm16(position.x), snorm16(position.y)])
    }
}

impl Snorm16x4 {
    /// `position`, clamped to -1 to 1, as the nearest multiples of 1/32767.
    pub fn new(position: Vec3) -> Self {
        Snorm16x4([
            snorm16(position.x),
            snorm16(position.y),
            snorm16(position.z),
            0,
        ])
    }
}

fn snorm16(value: f32) -> i16 {
    (value.clamp(-1.0, 1.0) * 32767.0).round() as i16
}

/// The index types a triangle geometry can use, those of indexed draws.
pub use crate::command_buffer::IndexFormat;

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

impl From<TransformMatrix> for Affine3A {
    fn from(TransformMatrix(rows): TransformMatrix) -> Self {
        let column = |c: usize| Vec3A::new(rows[0][c], rows[1][c], rows[2][c]);
        Affine3A::from_cols(column(0), column(1), column(2), column(3))
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
    inputs: [Option<BufferSlice<'a, u8>>; 3],
}

impl<'a> Triangles<'a> {
    #[track_caller]
    pub fn new<V: VertexPosition>(vertices: BufferSlice<'a, V>) -> Self {
        assert!(
            vertices.len().is_multiple_of(3),
            "{} vertices don't make triangles",
            vertices.len()
        );
        Self::with_indices(
            vertices,
            vk::IndexType::NONE_KHR,
            0,
            (vertices.len() / 3) as u32,
            None,
        )
    }

    #[track_caller]
    pub fn indexed<V: VertexPosition, I: IndexFormat>(
        vertices: BufferSlice<'a, V>,
        indices: BufferSlice<'a, I>,
    ) -> Self {
        assert!(
            indices.len().is_multiple_of(3),
            "{} indices don't make triangles",
            indices.len()
        );
        Self::with_indices(
            vertices,
            I::TYPE,
            indices.address(),
            (indices.len() / 3) as u32,
            Some(indices.bytes()),
        )
    }

    #[track_caller]
    fn with_indices<V: VertexPosition>(
        vertices: BufferSlice<'a, V>,
        index_type: vk::IndexType,
        index_address: u64,
        primitives: u32,
        indices: Option<BufferSlice<'a, u8>>,
    ) -> Self {
        assert!(!vertices.is_empty(), "a triangle geometry needs vertices");
        let component = V::FORMAT.component_bytes();
        assert!(
            vertices.address().is_multiple_of(component)
                && (size_of::<V>() as u64).is_multiple_of(component),
            "the vertices' address and stride have to be multiples of the {component} bytes of a \
             {:?} component",
            V::FORMAT
        );
        Self {
            data: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                .vertex_format(V::FORMAT.vk())
                .vertex_data(const_address(vertices.address()))
                .vertex_stride(size_of::<V>() as u64)
                .max_vertex(vertices.len() as u32 - 1)
                .index_type(index_type)
                .index_data(const_address(index_address)),
            primitives,
            opaque: true,
            inputs: [Some(vertices.bytes()), indices, None],
        }
    }

    /// Transforms the vertices by the first matrix of `transform`, whose address has to be a
    /// multiple of 16.
    #[track_caller]
    pub fn transform(mut self, transform: BufferSlice<'a, TransformMatrix>) -> Self {
        assert!(!transform.is_empty(), "the transform slice is empty");
        assert!(
            transform.address().is_multiple_of(16),
            "a transform's address has to be a multiple of 16"
        );
        self.data = self.data.transform_data(const_address(transform.address()));
        self.inputs[2] = Some(transform.range(..1).bytes());
        self
    }

    pub fn opaque(mut self, opaque: bool) -> Self {
        self.opaque = opaque;
        self
    }

    fn shape(&self) -> Shape {
        Shape {
            ty: vk::GeometryTypeKHR::TRIANGLES,
            flags: geometry_flags(self.opaque),
            primitives: self.primitives,
            vertex_format: self.data.vertex_format,
            max_vertex: self.data.max_vertex,
            index_type: self.data.index_type,
            // SAFETY: lava writes device addresses.
            transform: unsafe { self.data.transform_data.device_address } != 0,
        }
    }
}

/// A procedural geometry: one primitive per box, hit through intersection shaders (or a ray
/// query's candidates). Opaque unless [`Self::opaque`] says otherwise.
#[derive(Clone, Copy)]
pub struct Aabbs<'a> {
    boxes: BufferSlice<'a, u8>,
    primitives: u32,
    opaque: bool,
}

impl<'a> Aabbs<'a> {
    #[track_caller]
    pub fn new(boxes: BufferSlice<'a, Aabb>) -> Self {
        assert!(
            boxes.address().is_multiple_of(8),
            "an AABB buffer's address has to be a multiple of 8"
        );
        Self {
            primitives: boxes.len() as u32,
            boxes: boxes.bytes(),
            opaque: true,
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

fn triangles_geometry(
    data: vk::AccelerationStructureGeometryTrianglesDataKHR<'static>,
    opaque: bool,
) -> vk::AccelerationStructureGeometryKHR<'static> {
    vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .geometry(vk::AccelerationStructureGeometryDataKHR { triangles: data })
        .flags(geometry_flags(opaque))
}

fn aabbs_geometry(address: u64, opaque: bool) -> vk::AccelerationStructureGeometryKHR<'static> {
    vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::AABBS)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            aabbs: vk::AccelerationStructureGeometryAabbsDataKHR::default()
                .data(const_address(address))
                .stride(size_of::<Aabb>() as u64),
        })
        .flags(geometry_flags(opaque))
}

impl<'a> Describe<'a> for BlasGeometry<'a> {
    fn describe(&self, out: &mut SmallVec<[Described<'a>; 1]>) {
        match *self {
            BlasGeometry::Triangles(triangles) => out.extend(triangles.iter().map(|t| Described {
                geometry: triangles_geometry(t.data, t.opaque),
                shape: t.shape(),
                inputs: t.inputs,
            })),
            BlasGeometry::Aabbs(boxes) => out.extend(boxes.iter().map(|b| Described {
                geometry: aabbs_geometry(b.boxes.address(), b.opaque),
                shape: Shape::aabbs(b.primitives, b.opaque),
                inputs: [Some(b.boxes), None, None],
            })),
        }
    }
}

/// A triangle geometry described by its formats and counts alone, for [`Blas::with_capacity`]:
/// builds then take [`Triangles`] of the same vertex and index types, transform and opacity,
/// with at most as many vertices and triangles.
#[derive(Clone, Copy, Debug)]
pub struct TrianglesDesc {
    format: VertexFormat,
    stride: u64,
    vertices: u32,
    index_type: vk::IndexType,
    triangles: u32,
    transform: bool,
    opaque: bool,
}

impl TrianglesDesc {
    /// Up to `vertices` `V`s, every three a triangle.
    #[track_caller]
    pub fn new<V: VertexPosition>(vertices: u32) -> Self {
        assert!(
            vertices.is_multiple_of(3),
            "{vertices} vertices don't make triangles"
        );
        Self::with::<V>(vertices, vk::IndexType::NONE_KHR, vertices / 3)
    }

    /// Up to `vertices` `V`s and `triangles` triangles of `I` indices.
    #[track_caller]
    pub fn indexed<V: VertexPosition, I: IndexFormat>(vertices: u32, triangles: u32) -> Self {
        Self::with::<V>(vertices, I::TYPE, triangles)
    }

    #[track_caller]
    fn with<V: VertexPosition>(vertices: u32, index_type: vk::IndexType, triangles: u32) -> Self {
        assert!(vertices > 0, "a triangle geometry needs vertices");
        TrianglesDesc {
            format: V::FORMAT,
            stride: size_of::<V>() as u64,
            vertices,
            index_type,
            triangles,
            transform: false,
            opaque: true,
        }
    }

    /// Built with a transform ([`Triangles::transform`]).
    pub fn transformed(mut self) -> Self {
        self.transform = true;
        self
    }

    pub fn opaque(mut self, opaque: bool) -> Self {
        self.opaque = opaque;
        self
    }
}

/// A procedural geometry of up to `boxes` boxes, for [`Blas::with_capacity`].
#[derive(Clone, Copy, Debug)]
pub struct AabbsDesc {
    boxes: u32,
    opaque: bool,
}

impl AabbsDesc {
    pub fn new(boxes: u32) -> Self {
        AabbsDesc {
            boxes,
            opaque: true,
        }
    }

    pub fn opaque(mut self, opaque: bool) -> Self {
        self.opaque = opaque;
        self
    }
}

/// The geometries a BLAS is created for by [`Blas::with_capacity`], like a [`BlasGeometry`]
/// without its buffers.
#[derive(Clone, Copy, Debug)]
pub enum BlasDesc<'a> {
    Triangles(&'a [TrianglesDesc]),
    Aabbs(&'a [AabbsDesc]),
}

impl BlasDesc<'_> {
    fn described(&self) -> SmallVec<[Described<'static>; 1]> {
        let none = || [None; 3];
        match *self {
            BlasDesc::Triangles(triangles) => triangles
                .iter()
                .map(|t| {
                    // Sizes don't read addresses, only whether a transform is there.
                    let data = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                        .vertex_format(t.format.vk())
                        .vertex_stride(t.stride)
                        .max_vertex(t.vertices - 1)
                        .index_type(t.index_type)
                        .transform_data(const_address(t.transform as u64 * 16));
                    Described {
                        geometry: triangles_geometry(data, t.opaque),
                        shape: Shape {
                            ty: vk::GeometryTypeKHR::TRIANGLES,
                            flags: geometry_flags(t.opaque),
                            primitives: t.triangles,
                            vertex_format: t.format.vk(),
                            max_vertex: t.vertices - 1,
                            index_type: t.index_type,
                            transform: t.transform,
                        },
                        inputs: none(),
                    }
                })
                .collect(),
            BlasDesc::Aabbs(boxes) => boxes
                .iter()
                .map(|b| Described {
                    geometry: aabbs_geometry(0, b.opaque),
                    shape: Shape::aabbs(b.boxes, b.opaque),
                    inputs: none(),
                })
                .collect(),
        }
    }
}

/// How an instance's triangles are hit (`VkGeometryInstanceFlagsKHR`), combined with `|`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct InstanceFlags(u8);

impl InstanceFlags {
    pub const NONE: Self = Self(0);
    /// Both faces are hit, whatever the ray's cull flags.
    pub const TRIANGLE_FACING_CULL_DISABLE: Self = Self(1);
    /// Counter-clockwise triangles face forward instead of clockwise ones.
    pub const TRIANGLE_FLIP_FACING: Self = Self(2);
    /// Every geometry is opaque: no any-hit shaders run.
    pub const FORCE_OPAQUE: Self = Self(4);
    /// No geometry is opaque: any-hit shaders run for all.
    pub const FORCE_NO_OPAQUE: Self = Self(8);

    pub const fn bits(self) -> u8 {
        self.0
    }

    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

impl std::ops::BitOr for InstanceFlags {
    type Output = Self;
    fn bitor(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl std::ops::BitOrAssign for InstanceFlags {
    fn bitor_assign(&mut self, other: Self) {
        self.0 |= other.0;
    }
}

/// An instance of a BLAS in a TLAS as lava writes it (`VkAccelerationStructureInstanceKHR`):
/// a [`RawInstance`] whose BLAS the buffer it is in keeps alive, written only with it by an
/// [`InstanceInfo`] (`host_write_referencing`, `Uploader::write_referencing`). A TLAS built
/// from such instances keeps their BLASes, and commands that trace it wait for their builds.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::AnyBitPattern)]
pub struct Instance(RawInstance);

// SAFETY: one reference, the BLAS `InstanceInfo` writes with its address.
unsafe impl crate::reference::ReferencingData for Instance {
    const REFERENCES: usize = 1;
    const WRITES: u64 = 0;
    const UNORDERED: u64 = 0;
}

impl std::ops::Deref for Instance {
    type Target = RawInstance;
    fn deref(&self) -> &RawInstance {
        &self.0
    }
}

/// An [`Instance`] as what it references: its BLAS, placed and flagged by the `with_` methods
/// of a [`RawInstance`].
#[derive(Clone, Copy)]
pub struct InstanceInfo<'a> {
    blas: &'a Blas,
    raw: RawInstance,
}

impl<'a> InstanceInfo<'a> {
    /// `blas` placed by `transform`, seen by every ray mask, with custom index, hit group
    /// offset and flags 0.
    pub fn new(blas: &'a Blas, transform: Affine3A) -> Self {
        InstanceInfo {
            blas,
            raw: RawInstance::new(blas.address(), transform),
        }
    }

    /// The instance with `f` applied to its raw fields ([`RawInstance::with_mask`], ...); its
    /// BLAS stays.
    pub fn with(mut self, f: impl FnOnce(RawInstance) -> RawInstance) -> Self {
        self.raw = f(self.raw);
        self.raw.blas = self.blas.address();
        self
    }

    /// See [`RawInstance::with_custom_index`].
    #[track_caller]
    pub fn with_custom_index(self, index: u32) -> Self {
        self.with(|raw| raw.with_custom_index(index))
    }

    /// See [`RawInstance::with_mask`].
    pub fn with_mask(self, mask: u8) -> Self {
        self.with(|raw| raw.with_mask(mask))
    }

    /// See [`RawInstance::with_sbt_offset`].
    #[track_caller]
    pub fn with_sbt_offset(self, offset: u32) -> Self {
        self.with(|raw| raw.with_sbt_offset(offset))
    }

    pub fn with_flags(self, flags: InstanceFlags) -> Self {
        self.with(|raw| raw.with_flags(flags))
    }

    pub fn with_transform(self, transform: Affine3A) -> Self {
        self.with(|raw| raw.with_transform(transform))
    }
}

impl<'a> crate::reference::ReferenceInfo<'a> for InstanceInfo<'a> {
    type Data = Instance;
    fn write(self, data: &mut Instance, references: &mut [crate::reference::Reference<'a>]) {
        *data = Instance(self.raw);
        references[0] = crate::reference::Reference::structure(self.blas);
    }
}

/// `VkAccelerationStructureInstanceKHR` as plain data, a layout of the shader contract (Slang's
/// `AccelerationStructureInstance`), so passes can write them: TLASes are built from them only
/// through `unsafe` [`Instances::from_raw`]. Its fields are public for that layout; read them
/// through the getters and set them with the `with_` methods.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct RawInstance {
    pub transform: TransformMatrix,
    /// The custom index in bits 0..24, the mask in 24..32.
    pub custom_index_and_mask: u32,
    /// The hit group offset in bits 0..24, the flags in 24..32.
    pub sbt_offset_and_flags: u32,
    pub blas: u64,
}

const LOW_24: u32 = (1 << 24) - 1;

impl RawInstance {
    /// The BLAS at `address` ([`AccelerationStructure::address`]; 0 for an inactive instance)
    /// placed by `transform`, seen by every ray mask, with custom index, hit group offset and
    /// flags 0. It keeps nothing alive: see [`Instances::from_raw`].
    pub fn new(address: u64, transform: Affine3A) -> Self {
        Self {
            transform: transform.into(),
            custom_index_and_mask: 0xFF << 24,
            sbt_offset_and_flags: 0,
            blas: address,
        }
    }

    /// What `InstanceID()` returns; at most 24 bits.
    #[track_caller]
    pub fn with_custom_index(mut self, index: u32) -> Self {
        assert!(index <= LOW_24, "custom index {index} doesn't fit 24 bits");
        self.custom_index_and_mask = (self.custom_index_and_mask & !LOW_24) | index;
        self
    }

    /// Rays whose mask shares no bit with it miss the instance.
    pub fn with_mask(mut self, mask: u8) -> Self {
        self.custom_index_and_mask = (self.custom_index_and_mask & LOW_24) | (mask as u32) << 24;
        self
    }

    /// The index of the hit group its geometries use (`[Group(n)]`); at most 24 bits.
    #[track_caller]
    pub fn with_sbt_offset(mut self, offset: u32) -> Self {
        assert!(
            offset <= LOW_24,
            "hit group offset {offset} doesn't fit 24 bits"
        );
        self.sbt_offset_and_flags = (self.sbt_offset_and_flags & !LOW_24) | offset;
        self
    }

    pub fn with_flags(mut self, flags: InstanceFlags) -> Self {
        self.sbt_offset_and_flags = (self.sbt_offset_and_flags & LOW_24) | (flags.0 as u32) << 24;
        self
    }

    pub fn with_transform(mut self, transform: Affine3A) -> Self {
        self.transform = transform.into();
        self
    }

    pub fn custom_index(&self) -> u32 {
        self.custom_index_and_mask & LOW_24
    }

    pub fn mask(&self) -> u8 {
        (self.custom_index_and_mask >> 24) as u8
    }

    pub fn sbt_offset(&self) -> u32 {
        self.sbt_offset_and_flags & LOW_24
    }

    pub fn flags(&self) -> InstanceFlags {
        InstanceFlags((self.sbt_offset_and_flags >> 24) as u8)
    }

    /// The address of the BLAS it places; 0 for an inactive instance.
    pub fn blas_address(&self) -> u64 {
        self.blas
    }

    pub fn transform(&self) -> Affine3A {
        self.transform.into()
    }
}

fn instances_described(address: u64, count: u32) -> Described<'static> {
    Described {
        geometry: vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                    .data(const_address(address)),
            }),
        shape: Shape::instances(count),
        inputs: [None; 3],
    }
}

/// What a TLAS is built from: a slice of [`Instance`]s, whose BLASes the TLAS keeps, or of
/// [`RawInstance`]s a pass may write.
#[derive(Clone, Copy)]
pub struct Instances<'a> {
    instances: BufferSlice<'a, u8>,
    count: u32,
    raw: bool,
}

impl<'a> Instances<'a> {
    /// The build input of a TLAS: `instances`, written with their BLASes, which a build keeps
    /// for the TLAS. Panics unless every BLAS they name has a build recorded.
    #[track_caller]
    pub fn new(instances: BufferSlice<'a, Instance>) -> Self {
        Self::of(instances.bytes(), instances.len(), false)
    }

    /// The build input of a TLAS from instances a pass may have written.
    ///
    /// # Safety
    /// Every BLAS the instances name (by address; 0 is an inactive instance) must be built and
    /// alive when a build from them executes on the GPU, which reads them, and stay alive while
    /// a TLAS built from them is traced. lava doesn't know them: a dropped BLAS is destroyed
    /// once the submissions in flight at its drop are done, so rebuild the TLAS without it
    /// before tracing again. (Builds and traces of such TLASes wait for every structure build.)
    #[track_caller]
    pub unsafe fn from_raw(instances: BufferSlice<'a, RawInstance>) -> Self {
        Self::of(instances.bytes(), instances.len(), true)
    }

    #[track_caller]
    fn of(instances: BufferSlice<'a, u8>, count: usize, raw: bool) -> Self {
        assert!(
            instances.address().is_multiple_of(16),
            "an instance buffer's address has to be a multiple of 16"
        );
        Self {
            count: count as u32,
            instances,
            raw,
        }
    }
}

impl<'a> Describe<'a> for Instances<'a> {
    fn describe(&self, out: &mut SmallVec<[Described<'a>; 1]>) {
        let mut described = instances_described(self.instances.address(), self.count);
        described.inputs[0] = Some(self.instances);
        out.push(described);
    }

    fn instance_source(&self) -> Option<InstanceSource<'a>> {
        Some(InstanceSource {
            referenced: (!self.raw).then_some(self.instances.buffer),
        })
    }
}

/// Panics if two of `handles` are the same structure.
#[track_caller]
fn check_distinct(handles: impl Iterator<Item = vk::AccelerationStructureKHR>, what: &str) {
    let mut handles: SmallVec<[(u64, usize); 8]> = handles
        .enumerate()
        .map(|(i, h)| (vk::Handle::as_raw(h), i))
        .collect();
    handles.sort_unstable();
    if let Some(pair) = handles.windows(2).find(|pair| pair[0].0 == pair[1].0) {
        let (first, second) = (pair[0].1.min(pair[1].1), pair[0].1.max(pair[1].1));
        panic!("{what} {first} and {second} of one batch write the same structure");
    }
}

impl<K: HasCompute> CommandBuffer<'_, K> {
    /// Builds every structure in one `vkCmdBuildAccelerationStructuresKHR`, each from its
    /// `build_scratch_size()` of `scratch` after the previous one's. `scratch`'s address has to
    /// be a multiple of [`scratch_alignment`]; [`Self::build_all`] splits builds that don't fit
    /// one scratch buffer. Each geometry has the shape its structure was created for (type,
    /// formats, index type, transform, opacity) with at most its primitives and vertices.
    /// Nothing is recorded for no builds.
    #[validation_trace]
    pub fn build<L: Level>(
        &mut self,
        builds: &[(&AccelerationStructure<L>, L::Geometry<'_>)],
        scratch: BufferSlice<'_, u8>,
    ) {
        self.record_builds(builds, scratch, false);
    }

    /// Refits every structure in place for its geometry, which has the shape of its last build
    /// (primitive and vertex counts included): only positions, transforms and instances may
    /// change. Needs `allow_update`; the scratch is laid out by `update_scratch_size()`.
    #[validation_trace]
    pub fn update<L: Level>(
        &mut self,
        updates: &[(&AccelerationStructure<L>, L::Geometry<'_>)],
        scratch: BufferSlice<'_, u8>,
    ) {
        self.record_builds(updates, scratch, true);
    }

    /// [`Self::build`] in as few batches as fit `scratch`, one after another (each waits for
    /// the previous one's scratch use). Panics for a structure whose scratch alone is larger.
    #[validation_trace]
    pub fn build_all<L: Level>(
        &mut self,
        builds: &[(&AccelerationStructure<L>, L::Geometry<'_>)],
        scratch: &Scratch,
    ) {
        self.record_batches(builds, scratch, false);
    }

    /// [`Self::update`] in as few batches as fit `scratch`, like [`Self::build_all`].
    #[validation_trace]
    pub fn update_all<L: Level>(
        &mut self,
        updates: &[(&AccelerationStructure<L>, L::Geometry<'_>)],
        scratch: &Scratch,
    ) {
        self.record_batches(updates, scratch, true);
    }

    #[track_caller]
    fn record_batches<L: Level>(
        &mut self,
        builds: &[(&AccelerationStructure<L>, L::Geometry<'_>)],
        scratch: &Scratch,
        update: bool,
    ) {
        check_distinct(builds.iter().map(|(s, _)| s.internal.handle), "builds");
        let (mut start, mut used) = (0, 0);
        for (i, (structure, _)) in builds.iter().enumerate() {
            let size = match update {
                true => structure.internal.update_scratch,
                false => structure.internal.build_scratch,
            };
            assert!(
                size <= scratch.size(),
                "build {i} needs {size} scratch bytes, more than the scratch's {}",
                scratch.size()
            );
            if used + size > scratch.size() {
                self.record_builds(&builds[start..i], scratch.slice(), update);
                (start, used) = (i, 0);
            }
            used += size;
        }
        self.record_builds(&builds[start..], scratch.slice(), update);
    }

    #[track_caller]
    fn record_builds<L: Level>(
        &mut self,
        builds: &[(&AccelerationStructure<L>, L::Geometry<'_>)],
        scratch: BufferSlice<'_, u8>,
        update: bool,
    ) {
        if builds.is_empty() {
            return;
        }
        let alignment = scratch_alignment();
        assert!(
            scratch.address().is_multiple_of(alignment),
            "a scratch buffer's address has to be a multiple of {alignment}"
        );
        check_distinct(builds.iter().map(|(s, _)| s.internal.handle), "builds");

        let mut described: SmallVec<[SmallVec<[Described; 1]>; 1]> = SmallVec::new();
        let mut accesses: SmallVec<[BufferAccess<'_>; 8]> = SmallVec::new();
        // Each build's scratch address.
        let mut scratches: SmallVec<[u64; 1]> = SmallVec::new();
        let mut offset = 0;
        for (i, (structure, geometry)) in builds.iter().enumerate() {
            let mut geometries = SmallVec::new();
            geometry.describe(&mut geometries);
            if update {
                assert!(
                    structure.internal.flags.allow_update
                        && structure.internal.origin != Origin::Deserialized,
                    "build {i}: only a structure with `allow_update` can be updated"
                );
                let Some(last) = self.rec.last_shapes(&structure.internal.state) else {
                    panic!("build {i}: an update refits a structure that was built before");
                };
                assert_eq!(
                    geometries.len(),
                    last.len(),
                    "build {i}: an update has the geometries of the last build"
                );
                if let Some((g, (last, new))) = (last.iter())
                    .zip(geometries.iter().map(|g| &g.shape))
                    .enumerate()
                    .find(|(_, (last, new))| last != new)
                {
                    panic!(
                        "build {i}: an update keeps everything of the last build but positions: \
                         geometry {g} was built as {last:?}, not {new:?}"
                    );
                }
            } else {
                assert!(
                    structure.is_buildable(),
                    "build {i}: a compacted or deserialized structure can't be built again: its \
                     memory only holds what was copied into it (build the one it was made from, \
                     or a new one)"
                );
                assert_eq!(
                    geometries.len(),
                    structure.internal.shapes.len(),
                    "build {i}: a structure is built from as many geometries as it was created for"
                );
                for (g, (new, creation)) in geometries
                    .iter()
                    .zip(&structure.internal.shapes)
                    .enumerate()
                {
                    assert!(
                        new.shape.fits(creation),
                        "build {i}: geometry {g} is {:?}, which doesn't fit the {creation:?} the \
                         structure was created for (the same geometry with at most its \
                         primitives and highest vertex)",
                        new.shape
                    );
                }
            }
            let size = match update {
                true => structure.internal.update_scratch,
                false => structure.internal.build_scratch,
            };
            let chunk = BufferSlice::<u8>::new(scratch.buffer, scratch.address() + offset, size);
            scratches.push(chunk.address());
            offset += size;

            let dst = if update { READ | WRITE } else { WRITE };
            accesses.push(structure.access(BUILD, dst));
            accesses.push(chunk.access(BUILD, READ | WRITE));
            for input in geometries
                .iter()
                .flat_map(|g| g.inputs)
                .flatten()
                .filter(|r| !r.is_empty())
            {
                accesses.push(input.access(BUILD, vk::AccessFlags2::SHADER_READ));
            }
            described.push(geometries);
        }
        assert!(
            offset <= scratch.size(),
            "the builds need {offset} scratch bytes, the slice has {}",
            scratch.size()
        );
        // A TLAS build reaches the BLASes its instances reference through their buffer; raw
        // instances name BLASes lava doesn't know.
        let sources: SmallVec<[_; 1]> = (builds.iter())
            .filter_map(|(structure, geometry)| Some((structure, geometry.instance_source()?)))
            .collect();
        if sources.iter().any(|(_, s)| s.referenced.is_none()) {
            accesses.push(every_structure_read(BUILD, READ));
        }
        for (structure, source) in sources {
            structure.internal.take_instances(source);
        }
        let name = match update {
            true => "update acceleration structures",
            false => "build acceleration structures",
        };
        self.rec.enter(name, false);
        self.rec
            .flush_pending(std::iter::empty(), accesses.iter().cloned());
        for ((structure, _), geometries) in builds.iter().zip(&described) {
            let shapes = geometries.iter().map(|g| g.shape).collect();
            self.rec.structure_built(&structure.internal.state, shapes);
        }

        let geometries: SmallVec<[SmallVec<[vk::AccelerationStructureGeometryKHR; 1]>; 1]> =
            (described.iter())
                .map(|d| d.iter().map(|g| g.geometry).collect())
                .collect();
        let ranges: SmallVec<[SmallVec<[vk::AccelerationStructureBuildRangeInfoKHR; 1]>; 1]> =
            (described.iter())
                .map(|d| {
                    d.iter()
                        .map(|g| vk::AccelerationStructureBuildRangeInfoKHR {
                            primitive_count: g.shape.primitives,
                            ..Default::default()
                        })
                        .collect()
                })
                .collect();
        let infos: SmallVec<[_; 1]> = (builds.iter().zip(&geometries).zip(&scratches))
            .map(|(((structure, _), geometries), &scratch)| {
                let info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                    .ty(L::TY)
                    .flags(structure.internal.flags.vk())
                    .geometries(geometries)
                    .dst_acceleration_structure(structure.internal.handle)
                    .scratch_data(vk::DeviceOrHostAddressKHR {
                        device_address: scratch,
                    });
                if update {
                    info.mode(vk::BuildAccelerationStructureModeKHR::UPDATE)
                        .src_acceleration_structure(structure.internal.handle)
                } else {
                    info.mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                }
            })
            .collect();
        // Not ash 0.38's wrapper, which collects these into a `Vec`.
        let ranges: SmallVec<[_; 1]> = ranges.iter().map(|r| r.as_ptr()).collect();
        unsafe {
            (command_functions()
                .fp()
                .cmd_build_acceleration_structures_khr)(
                self.rec.handle,
                infos.len() as u32,
                infos.as_ptr(),
                ranges.as_ptr(),
            )
        };
        self.rec.exit(name, false);
    }

    /// Writes the compacted sizes of `structures` (each with `allow_compaction`, built before)
    /// into the queries of `sizes` from `first` on. Nothing is recorded for no structures.
    #[validation_trace]
    pub fn query_compacted_sizes<L: Level>(
        &mut self,
        structures: &[&AccelerationStructure<L>],
        sizes: &QueryPool<CompactedSize>,
        first: u32,
    ) {
        for structure in structures {
            assert!(
                structure.internal.flags.allow_compaction,
                "only structures with `allow_compaction` have a compacted size"
            );
        }
        self.write_sizes(structures, sizes, first, "query compacted sizes");
    }

    /// Writes the bytes [`Self::serialize`] writes of each of `structures` (built before) into
    /// the queries of `sizes` from `first` on. Nothing is recorded for no structures.
    #[validation_trace]
    pub fn query_serialized_sizes<L: Level>(
        &mut self,
        structures: &[&AccelerationStructure<L>],
        sizes: &QueryPool<SerializedSize>,
        first: u32,
    ) {
        self.write_sizes(structures, sizes, first, "query serialized sizes");
    }

    #[track_caller]
    fn write_sizes<L: Level, Q: QueryKind>(
        &mut self,
        structures: &[&AccelerationStructure<L>],
        sizes: &QueryPool<Q>,
        first: u32,
        name: &'static str,
    ) {
        if structures.is_empty() {
            return;
        }
        for structure in structures {
            self.check_built(structure, "sizes are of");
        }
        let count = u32::try_from(structures.len()).unwrap_or(u32::MAX);
        let accesses: SmallVec<[_; 8]> = structures.iter().map(|s| s.access(BUILD, READ)).collect();
        self.rec.enter(name, false);
        self.rec
            .flush_pending(std::iter::empty(), accesses.iter().cloned());
        sizes.take(&mut self.rec, first, count);
        let handles: SmallVec<[_; 8]> = structures.iter().map(|s| s.internal.handle).collect();
        unsafe {
            command_functions().cmd_write_acceleration_structures_properties(
                self.rec.handle,
                &handles,
                Q::TYPE,
                sizes.handle,
                first,
            );
        }
        self.rec.exit(name, false);
    }

    #[track_caller]
    fn check_built<L: Level>(&self, structure: &AccelerationStructure<L>, what: &str) {
        assert!(
            self.rec.last_shapes(&structure.internal.state).is_some() || structure.is_built(),
            "{what} a structure that was built before"
        );
    }

    /// Copies `src` (built, with `allow_compaction`) into `dst`, made by `src.compacted(size)`;
    /// `dst` can then be updated like `src`. `src` can be dropped right after.
    #[validation_trace]
    pub fn compact<L: Level>(
        &mut self,
        src: &AccelerationStructure<L>,
        dst: &AccelerationStructure<L>,
    ) {
        assert!(
            src.internal.flags.allow_compaction,
            "only a structure with `allow_compaction` can be compacted"
        );
        assert!(
            dst.internal.origin == Origin::Compacted(src.internal.handle),
            "`compact` copies into a structure made by `compacted` of its source"
        );
        let mode = vk::CopyAccelerationStructureModeKHR::COMPACT;
        self.copy(src, dst, mode, "compact acceleration structure");
    }

    /// Copies `src` (built) into `dst`, made by `src.clone_target()` (or of a structure with
    /// its size, flags and shapes): `dst` is then what `src` is, built, and can be updated (and
    /// rebuilt) where `src` can. `src` can be dropped right after.
    #[validation_trace]
    pub fn copy_structure<L: Level>(
        &mut self,
        src: &AccelerationStructure<L>,
        dst: &AccelerationStructure<L>,
    ) {
        assert!(
            matches!(dst.internal.origin, Origin::Copied { .. })
                && dst.size() >= src.size()
                && dst.internal.flags == src.internal.flags
                && dst.internal.shapes == src.internal.shapes,
            "`copy_structure` copies into a structure made by `clone_target` of its source"
        );
        let mode = vk::CopyAccelerationStructureModeKHR::CLONE;
        self.copy(src, dst, mode, "copy acceleration structure");
    }

    #[track_caller]
    fn copy<L: Level>(
        &mut self,
        src: &AccelerationStructure<L>,
        dst: &AccelerationStructure<L>,
        mode: vk::CopyAccelerationStructureModeKHR,
        name: &'static str,
    ) {
        let Some(shapes) = self.rec.last_shapes(&src.internal.state).or_else(|| {
            // Built by another timeline's recording in flight, which the caller waits for.
            src.is_built().then(SmallVec::new)
        }) else {
            panic!("`{name}` copies a structure that was built before: record its build first");
        };
        let accesses = [src.access(BUILD, READ), dst.access(BUILD, WRITE)];
        self.rec.enter(name, false);
        self.rec
            .flush_pending(std::iter::empty(), accesses.iter().cloned());
        self.rec.structure_built(&dst.internal.state, shapes);
        dst.internal.copy_instances(&src.internal);
        unsafe {
            command_functions().cmd_copy_acceleration_structure(
                self.rec.handle,
                &vk::CopyAccelerationStructureInfoKHR::default()
                    .src(src.internal.handle)
                    .dst(dst.internal.handle)
                    .mode(mode),
            )
        };
        self.rec.exit(name, false);
    }

    /// Writes `src` (built) into `dst` in the device's serialized format (its size from
    /// [`Self::query_serialized_sizes`]), whose header [`AccelerationStructure::deserialized`]
    /// checks, for a cache of BLASes on disk. `dst`'s address has to be a multiple of 256.
    #[validation_trace]
    pub fn serialize<L: Level>(
        &mut self,
        src: &AccelerationStructure<L>,
        dst: BufferSlice<'_, u8>,
    ) {
        self.check_built(src, "`serialize` writes");
        assert!(
            dst.address().is_multiple_of(STRUCTURE_ALIGNMENT)
                && dst.size() as usize >= SERIALIZED_HEADER,
            "a serialized structure goes to an address that is a multiple of 256, into at \
             least its {SERIALIZED_HEADER}-byte header"
        );
        let accesses = [
            src.access(BUILD, READ),
            dst.access(BUILD, vk::AccessFlags2::TRANSFER_WRITE),
        ];
        let name = "serialize acceleration structure";
        self.rec.enter(name, false);
        self.rec
            .flush_pending(std::iter::empty(), accesses.iter().cloned());
        unsafe {
            command_functions().cmd_copy_acceleration_structure_to_memory(
                self.rec.handle,
                &vk::CopyAccelerationStructureToMemoryInfoKHR::default()
                    .src(src.internal.handle)
                    .dst(vk::DeviceOrHostAddressKHR {
                        device_address: dst.address(),
                    })
                    .mode(vk::CopyAccelerationStructureModeKHR::SERIALIZE),
            )
        };
        self.rec.exit(name, false);
    }

    /// Reads serialized data from `src` (written by [`Self::serialize`], at an address that is
    /// a multiple of 256) into `dst`, made by [`AccelerationStructure::deserialized`] of its
    /// header; `dst` is built then.
    #[validation_trace]
    pub fn deserialize<L: Level>(
        &mut self,
        src: BufferSlice<'_, u8>,
        dst: &AccelerationStructure<L>,
    ) {
        assert!(
            dst.internal.origin == Origin::Deserialized,
            "`deserialize` writes a structure made by `AccelerationStructure::deserialized`"
        );
        assert!(
            src.address().is_multiple_of(STRUCTURE_ALIGNMENT)
                && src.size() as usize >= SERIALIZED_HEADER,
            "serialized data is read from an address that is a multiple of 256, at least its \
             {SERIALIZED_HEADER}-byte header"
        );
        let accesses = [
            src.access(BUILD, vk::AccessFlags2::TRANSFER_READ),
            dst.access(BUILD, WRITE),
        ];
        let name = "deserialize acceleration structure";
        self.rec.enter(name, false);
        self.rec
            .flush_pending(std::iter::empty(), accesses.iter().cloned());
        self.rec
            .structure_built(&dst.internal.state, SmallVec::new());
        // Its instances name BLASes lava doesn't know.
        if L::TY == vk::AccelerationStructureTypeKHR::TOP_LEVEL {
            dst.internal
                .replace_instances(KeepAlive::new::<Instance>(), true);
        }
        unsafe {
            command_functions().cmd_copy_memory_to_acceleration_structure(
                self.rec.handle,
                &vk::CopyMemoryToAccelerationStructureInfoKHR::default()
                    .src(const_address(src.address()))
                    .dst(dst.internal.handle)
                    .mode(vk::CopyAccelerationStructureModeKHR::DESERIALIZE),
            )
        };
        self.rec.exit(name, false);
    }
}

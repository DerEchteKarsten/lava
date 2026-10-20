//! Command recording for generated passes: the typed `CommandBuffer<K>` over the untyped `Recording`, barriers and queue family ownership releases, renderings and dispatches, scopes and queries (transfers live with the slices and views they take, `drain_uploads` with the staging rings in `buffer/staging.rs`, acceleration structure commands with the structures, `present` with `PresentSurface` in `present.rs`)
use std::{
    fmt::Debug, marker::PhantomData, ops::Deref, panic::AssertUnwindSafe, ptr::NonNull,
    range::Range, sync::atomic::AtomicU64,
};

use crate::vk::{self, IndexType, PipelineStageFlags2 as S2};
#[cfg(feature = "acceleration-structure")]
use crate::vkobjects::acceleration_structure::Tlas;
use crate::{
    buffer::slice::BufferSlice,
    image::{
        format::Format,
        kind::{ColorTarget, DepthTarget, ImageRef, SampledView, StorageView},
    },
    pipeline::{MAX_PUSH_CONSTANTS_SIZE, RasterKeyRef, pass_map, pipelines},
    present::Present,
    query::{
        Occlusion, QueryKind, QueryPool, QueryRecorder, ScopeInfo, ScopeObserver, Scoped, Timestamp,
    },
    queue::{FamilyInfo, HasCompute, HasGraphics, Kind, QueueKind},
    state::{Ctx, Functions, with_debug_name},
};
use bytemuck::{Pod, Zeroable};
use glam::{I64Vec2, IVec2, UVec2};
use lava_macros::validation_trace;
use smallvec::SmallVec;

use crate::buffer::staging::Staging;

mod binding_output;
pub(crate) mod image_state;
pub(crate) mod pending;
mod raster_state;
use crate::unsized_struct::unsized_struct;
pub use binding_output::{BindingOutput, Bindings, kind};
use image_state::{ImageState, check_states, release_states, released_by, take_checked_states};
use pending::{PendingAccesses, all_read_access, all_write_access};
pub use raster_state::*;

/// A pass's access of a buffer range, made by `BufferSlice::access`.
#[derive(Debug, Clone, PartialEq)]
pub struct BufferAccess {
    pub stage: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
    pub range: Range<u64>,
}

/// A read of every acceleration structure (the BLASes a TLAS build or trace reaches): empty, so
/// it only conflicts with acceleration structure writes (`BufferAccess::overlaps`).
pub(crate) const EVERY_ACCELERATION_STRUCTURE: Range<u64> = Range {
    start: u64::MAX,
    end: u64::MAX,
};

/// The write completed buffer writes fold into (`PendingAccesses::retire`); no buffer starts at
/// address 0, so none covers it.
pub(crate) const EVERY_BUFFER: Range<u64> = Range {
    start: 0,
    end: u64::MAX,
};

impl BufferAccess {
    fn overlaps(&self, other: &BufferAccess) -> bool {
        let writes_structure = |a: &BufferAccess| {
            a.access
                .contains(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
        };
        if self.range == EVERY_ACCELERATION_STRUCTURE {
            writes_structure(other)
        } else if other.range == EVERY_ACCELERATION_STRUCTURE {
            writes_structure(self)
        } else {
            self.range.start.max(other.range.start) < self.range.end.min(other.range.end)
        }
    }
}

/// What a command does to mips of an image, and the layout it needs them in.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ImageAccess {
    pub(crate) stage: vk::PipelineStageFlags2,
    pub(crate) access: vk::AccessFlags2,
    pub(crate) image: vk::Image,
    pub(crate) layout: vk::ImageLayout,
    pub(crate) aspect: vk::ImageAspectFlags,
    pub(crate) mip_range: Range<u32>,
}

/// An image access with the tracked states of its mips, which only move when the command is
/// recorded. Made by `ImageView::access` (shaders' accesses, always in `GENERAL`) and the
/// constructors of `Accesses`; lava alone picks other layouts, so the tracking stays true.
#[derive(Debug, Clone)]
pub struct ImageUse<'a> {
    pub(crate) access: ImageAccess,
    /// The image's tracked states: one for all its mips, or one per mip of the image.
    pub(crate) states: &'a [AtomicU64],
}

impl ImageUse<'_> {
    pub fn image(&self) -> vk::Image {
        self.access.image
    }

    pub fn stage(&self) -> vk::PipelineStageFlags2 {
        self.access.stage
    }

    pub fn access(&self) -> vk::AccessFlags2 {
        self.access.access
    }

    /// The layout the command wants the mips in.
    pub fn layout(&self) -> vk::ImageLayout {
        self.access.layout
    }

    pub fn mip_range(&self) -> Range<u32> {
        self.access.mip_range
    }
}

pub(crate) struct Recording<'p> {
    pub(crate) handle: vk::CommandBuffer,
    pub(crate) pending_accesses: &'p mut PendingAccesses,
    pub(crate) timeline: usize,
    pub(crate) value: u64,
    pub(crate) family: FamilyInfo,
    /// Set by `measure` for the duration of its closure, its lifetime erased.
    observer: Option<NonNull<dyn ScopeObserver>>,
    depth: u16,
    /// One bit per kind (`query_bit`): one query of each kind can be active.
    pub(crate) active_queries: u32,
    /// Every barrier the tracking already counts, so a recording that never reaches the GPU is
    /// replaced by one of just these, which keeps the tracking true. The frame slot's, reused.
    pub(crate) barrier_log: &'p mut Vec<Barriers>,
    /// Acquired swapchain images with the stages of their first access.
    pub(crate) acquired: SmallVec<[(vk::Image, vk::PipelineStageFlags2); 1]>,
    pub(crate) presents: SmallVec<[Present; 1]>,
    /// The timeline's staging ring, which `drain_uploads` records.
    pub(crate) staging: Option<&'p Staging>,
}

/// A command buffer being recorded for a family of kind `K`; its methods are the commands `K`
/// can run.
pub struct CommandBuffer<'p, K: QueueKind> {
    pub(crate) rec: Recording<'p>,
    _kind: PhantomData<fn() -> K>,
}

impl<'p, K: QueueKind> CommandBuffer<'p, K> {
    pub(crate) fn new(rec: Recording<'p>) -> Self {
        Self {
            rec,
            _kind: PhantomData,
        }
    }
}

pub(crate) fn query_bit(kind: vk::QueryType) -> u32 {
    match kind {
        vk::QueryType::OCCLUSION => 1,
        vk::QueryType::PIPELINE_STATISTICS => 2,
        vk::QueryType::PERFORMANCE_QUERY_KHR => 4,
        _ => 0,
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Default)]
pub struct ShaderHash {
    pub entry: &'static str,
    pub file: &'static str,
}

const INLINE_DRAWS: usize = 16;

/// A rendering being built (`CommandBuffer::raster`), which borrows its draws until
/// [`Self::record`].
pub struct RasterBuilder<'a, 'p, const N: usize> {
    cmd_buf: &'a mut Recording<'p>,
    extent: UVec2,
    viewport: Viewport,
    scissor: vk::Rect2D,
    color: [Attachment<'a>; N],
    depth_format: vk::Format,
    stencil_format: vk::Format,
    depth_attachment: vk::ImageView,
    /// Only a read; `record` adds the write.
    depth_access: Option<ImageUse<'a>>,
    clear_depth: Option<vk::ClearValue>,
    depth_discard: Discard,
    draws: SmallVec<[NonNull<DrawOutput<'static>>; INLINE_DRAWS]>,
    marker: PhantomData<&'a DrawOutput<'a>>,
}

/// A color attachment of a rendering, cleared to `clear` first unless that is `None`, else
/// loaded, and stored after the draws; [`Self::discard_before`] and [`Self::discard_after`]
/// skip the load and the store.
pub struct Attachment<'a> {
    format: vk::Format,
    view: vk::ImageView,
    extent: UVec2,
    clear: Option<vk::ClearValue>,
    discard: Discard,
    access: ImageUse<'a>,
}

/// The contents a rendering drops of an attachment: `before` loads nothing (`DONT_CARE`, for
/// an attachment the draws cover wherever it is read later), `after` stores nothing
/// (`DONT_CARE`, for one only the rendering itself reads, like a depth buffer of one pass).
/// Either leaves the texels of the render area undefined, which costs tiled GPUs no memory
/// traffic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Discard {
    before: bool,
    after: bool,
}

impl Discard {
    fn load_op(self, clear: Option<vk::ClearValue>) -> vk::AttachmentLoadOp {
        match (clear, self.before) {
            (Some(_), _) => vk::AttachmentLoadOp::CLEAR,
            (None, true) => vk::AttachmentLoadOp::DONT_CARE,
            (None, false) => vk::AttachmentLoadOp::LOAD,
        }
    }

    #[track_caller]
    fn before(&mut self, clear: Option<vk::ClearValue>) {
        assert!(
            clear.is_none(),
            "a cleared attachment loads nothing anyway: `discard_before` is for one that isn't"
        );
        self.before = true;
    }
}

impl<'a> Attachment<'a> {
    /// Renders into `image`, a view of one mip.
    pub fn color<T: ColorTarget + 'a>(
        image: T,
        clear: Option<<T::Format as Format>::Texels>,
    ) -> Self {
        let image = image.view();
        image.aspect_check(vk::ImageAspectFlags::COLOR);
        image.usage_check(vk::ImageUsageFlags::COLOR_ATTACHMENT);
        one_mip(image.mip_range);
        let access = image.access_in(
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                | if clear.is_none() {
                    vk::AccessFlags2::COLOR_ATTACHMENT_READ
                } else {
                    vk::AccessFlags2::empty()
                },
            // `GENERAL` can keep a driver from compressing a framebuffer.
            vk::ImageLayout::ATTACHMENT_OPTIMAL,
        );
        Self {
            format: image.format().vk_format(),
            view: image.view,
            extent: image.extent,
            clear: clear.map(|e| image.format().clear_value(e).into()),
            discard: Discard::default(),
            access,
        }
    }

    /// Doesn't load the attachment (`DONT_CARE`): the texels of the render area are undefined
    /// until the draws write them. Panics for a cleared attachment, which loads nothing.
    #[track_caller]
    pub fn discard_before(mut self) -> Self {
        self.discard.before(self.clear);
        // A `DONT_CARE` load writes.
        self.access.access.access &= !vk::AccessFlags2::COLOR_ATTACHMENT_READ;
        self
    }

    /// Doesn't store the attachment (`DONT_CARE`): its texels in the render area are undefined
    /// after the rendering, whose draws can still blend with them.
    pub fn discard_after(mut self) -> Self {
        self.discard.after = true;
        self
    }

    fn store_op(&self) -> vk::AttachmentStoreOp {
        match self.discard.after {
            true => vk::AttachmentStoreOp::DONT_CARE,
            false => vk::AttachmentStoreOp::STORE,
        }
    }
}

/// An attachment is one mip of an image (Vulkan's attachment views have one level).
#[track_caller]
fn one_mip(mips: Range<u32>) {
    assert!(
        mips.end - mips.start == 1,
        "an attachment is one mip, not mips {}..{}: render into a `mip_view` of a `MipChain`, or \
         into an image of one mip",
        mips.start,
        mips.end
    );
}

unsized_struct! {
    /// A draw into `N` color attachments that reads `E` index or indirect buffers. The rendering
    /// borrows it, so build draws added in a loop before the rendering.
    pub struct Draw<'a>[N, E];
    /// A draw as a rendering borrows it, unsized.
    pub struct DrawOutput {
        bindings: &'a BindingOutput<'a, ()>,
        /// The pass's shader stages (`PassKind::stages`), where its declared accesses happen.
        stages: vk::PipelineStageFlags2,
        command: DrawCommand,
        state: RasterState<0>,
        samples: Option<(&'a QueryPool<Occlusion>, u32)>,
        accesses: Option<Accesses<'a>>,
    } sections {
        extra: [BufferAccess; E],
        blends: [Blend; N],
        write_masks: [ColorComponentFlags; N],
    }
}

impl<'a> DrawOutput<'a> {
    fn image_accesses(&self) -> impl Iterator<Item = ImageUse<'a>> + Clone {
        let (declared, _) = declared(self.accesses, self.stages);
        self.bindings.sections().images().chain(declared)
    }

    fn buffer_accesses(&self) -> impl Iterator<Item = BufferAccess> + Clone + '_ {
        let (_, declared) = declared(self.accesses, self.stages);
        (self.bindings.sections().buffers())
            .chain(self.extra().iter().cloned())
            .chain(declared)
    }
}

/// Panics if a draw accesses what another draw of the rendering writes: draws are ordered only
/// by depth tests and blending, and nothing else passes between them. Attachments aren't draws'
/// accesses, so they are left out.
fn check_hazards<'a>(name: &str, draws: impl Iterator<Item = &'a DrawOutput<'a>> + Clone) {
    let writes = |access: vk::AccessFlags2| access.intersects(all_write_access());
    let hazard = |i: usize, j: usize, what: std::fmt::Arguments| -> ! {
        panic!(
            "draws {i} and {j} of the rendering `{name}` both access {what}, which draw {i} \
             writes: draws of one rendering are ordered only by depth tests and blending, so \
             record them in two renderings"
        )
    };
    for (i, draw) in draws.clone().enumerate() {
        for write in draw.buffer_accesses().filter(|b| writes(b.access)) {
            let mut others = draws.clone().enumerate().filter(|&(j, _)| j != i);
            if let Some((j, _)) =
                others.find(|(_, d)| d.buffer_accesses().any(|b| b.overlaps(&write)))
            {
                hazard(i, j, format_args!("the addresses {:?}", write.range));
            }
        }
        for write in draw
            .image_accesses()
            .filter(|image| writes(image.access.access))
        {
            let w = &write.access;
            let overlaps = |image: ImageUse| {
                let mips = image.access.mip_range;
                image.access.image == w.image
                    && mips.start < w.mip_range.end
                    && w.mip_range.start < mips.end
            };
            let mut others = draws.clone().enumerate().filter(|&(j, _)| j != i);
            if let Some((j, _)) = others.find(|(_, d)| d.image_accesses().any(overlaps)) {
                hazard(
                    i,
                    j,
                    format_args!("mips {:?} of the image {:?}", w.mip_range, w.image),
                );
            }
        }
    }
}

impl<'a, const N: usize, const E: usize> Draw<'a, N, E> {
    fn with(
        bindings: &'a BindingOutput<'a, ()>,
        stages: vk::PipelineStageFlags2,
        command: DrawCommand,
        extra: [BufferAccess; E],
        state: RasterState<N>,
        accesses: Option<Accesses<'a>>,
    ) -> Self {
        let (state, blends, write_masks) = state.split();
        Self::from_parts(
            bindings,
            stages,
            command,
            state,
            None,
            accesses,
            extra,
            blends,
            write_masks,
        )
    }

    /// Counts the samples that pass the depth and stencil tests into query `index` of `pool`.
    pub fn count_samples(mut self, pool: &'a QueryPool<Occlusion>, index: u32) -> Self {
        assert!(
            index < pool.count(),
            "query {index} of a pool of {}",
            pool.count()
        );
        self.samples = Some((pool, index));
        self
    }
}

impl<'a, const N: usize> Draw<'a, N, 0> {
    pub fn new(
        bindings: &'a BindingOutput<'a, kind::RasterVertex>,
        vertices: u32,
        instances: u32,
        state: RasterState<N>,
        accesses: Option<Accesses<'a>>,
    ) -> Self {
        let command = DrawCommand::Draw {
            vertices,
            instances,
        };
        Self::with(
            bindings.erase(),
            kind::RasterVertex::STAGES,
            command,
            [],
            state,
            accesses,
        )
    }

    /// Launches `groups` task (or, without a task shader, mesh) workgroups.
    #[cfg(feature = "mesh-shader")]
    pub fn launch(
        bindings: &'a BindingOutput<'a, kind::RasterMesh>,
        groups: [u32; 3],
        state: RasterState<N>,
        accesses: Option<Accesses<'a>>,
    ) -> Self {
        let pass = &pass_map().passes[bindings.sections().pass];
        let task = matches!(
            pass.kind,
            crate::pipeline::PassKind::RasterMesh { amp: Some(_), .. }
        );
        // Mesh bindings exist only past the feature's check.
        let p = &Ctx::physical_device().mesh_shader_properties;
        let (stage, max, total) = match task {
            true => (
                "task workgroups",
                p.max_task_work_group_count,
                p.max_task_work_group_total_count,
            ),
            false => (
                "mesh workgroups",
                p.max_mesh_work_group_count,
                p.max_mesh_work_group_total_count,
            ),
        };
        check_groups(groups, max, total as u64, stage);
        Self::with(
            bindings.erase(),
            kind::RasterMesh::STAGES,
            DrawCommand::Launch(groups),
            [],
            state,
            accesses,
        )
    }
}

impl<'a, const N: usize> Draw<'a, N, 1> {
    pub fn indexed(
        bindings: &'a BindingOutput<'a, kind::RasterVertex>,
        index_buffer: BufferSlice<'a, u32>,
        instances: u32,
        state: RasterState<N>,
        accesses: Option<Accesses<'a>>,
    ) -> Self {
        let index_access = index_buffer.access(
            vk::PipelineStageFlags2::INDEX_INPUT,
            vk::AccessFlags2::INDEX_READ,
        );
        let command = DrawCommand::Indexed {
            indices: index_buffer.address_range(),
            flags: index_buffer.address_flags(),
            count: index_buffer.len() as u32,
            instances,
        };
        Self::with(
            bindings.erase(),
            kind::RasterVertex::STAGES,
            command,
            [index_access],
            state,
            accesses,
        )
    }

    pub fn indirect(
        bindings: &'a BindingOutput<'a, kind::RasterVertex>,
        buffer: BufferSlice<'a, DrawIndirectCommand>,
        state: RasterState<N>,
        accesses: Option<Accesses<'a>>,
    ) -> Self {
        let command = DrawCommand::Indirect {
            draws: buffer.strided_range(),
            flags: buffer.address_flags(),
            count: buffer.len() as u32,
        };
        Self::with(
            bindings.erase(),
            kind::RasterVertex::STAGES,
            command,
            [indirect_access(buffer)],
            state,
            accesses,
        )
    }

    /// Launches the task (or, without a task shader, mesh) workgroup counts in `buffer`.
    #[cfg(feature = "mesh-shader")]
    pub fn launch_indirect(
        bindings: &'a BindingOutput<'a, kind::RasterMesh>,
        buffer: BufferSlice<'a, DispatchIndirectCommand>,
        state: RasterState<N>,
        accesses: Option<Accesses<'a>>,
    ) -> Self {
        let command = DrawCommand::LaunchIndirect {
            draws: buffer.strided_range(),
            flags: buffer.address_flags(),
            count: buffer.len() as u32,
        };
        Self::with(
            bindings.erase(),
            kind::RasterMesh::STAGES,
            command,
            [indirect_access(buffer)],
            state,
            accesses,
        )
    }
}

impl<'a, const N: usize> Draw<'a, N, 2> {
    pub fn indirect_count(
        bindings: &'a BindingOutput<'a, kind::RasterVertex>,
        buffer: BufferSlice<'a, DrawIndirectCommand>,
        count_buffer: BufferSlice<'a, u32>,
        state: RasterState<N>,
        accesses: Option<Accesses<'a>>,
    ) -> Self {
        let extra = [indirect_access(buffer), indirect_access(count_buffer)];
        let command = DrawCommand::IndirectCount {
            draws: buffer.strided_range(),
            flags: buffer.address_flags(),
            count: count_buffer.address_range(),
            count_flags: count_buffer.address_flags(),
            max_count: buffer.len() as u32,
        };
        Self::with(
            bindings.erase(),
            kind::RasterVertex::STAGES,
            command,
            extra,
            state,
            accesses,
        )
    }
}

#[derive(Debug)]
pub(crate) enum DrawCommand {
    Draw {
        vertices: u32,
        instances: u32,
    },
    Indexed {
        indices: vk::DeviceAddressRangeKHR,
        flags: vk::AddressCommandFlagsKHR,
        count: u32,
        instances: u32,
    },
    Indirect {
        draws: vk::StridedDeviceAddressRangeKHR,
        flags: vk::AddressCommandFlagsKHR,
        count: u32,
    },
    IndirectCount {
        draws: vk::StridedDeviceAddressRangeKHR,
        flags: vk::AddressCommandFlagsKHR,
        count: vk::DeviceAddressRangeKHR,
        count_flags: vk::AddressCommandFlagsKHR,
        max_count: u32,
    },
    #[cfg(feature = "mesh-shader")]
    Launch([u32; 3]),
    #[cfg(feature = "mesh-shader")]
    LaunchIndirect {
        draws: vk::StridedDeviceAddressRangeKHR,
        flags: vk::AddressCommandFlagsKHR,
        count: u32,
    },
}

#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct DrawIndirectCommand {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct DispatchIndirectCommand {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// The rays of an indirect trace (`VkTraceRaysIndirectCommandKHR`).
#[cfg(feature = "ray-tracing-pipeline")]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct TraceRaysIndirectCommand {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

#[derive(Clone, Copy)]
pub struct Scissor {
    pub offset: IVec2,
    pub extent: UVec2,
}

#[derive(Clone, Copy)]
pub struct Viewport {
    pub offset: IVec2,
    pub extent: UVec2,
}

fn rect(offset: IVec2, extent: UVec2) -> vk::Rect2D {
    vk::Rect2D {
        offset: vk::Offset2D {
            x: offset.x,
            y: offset.y,
        },
        extent: vk::Extent2D {
            width: extent.x,
            height: extent.y,
        },
    }
}

fn iter_draws<'a>(
    draws: &SmallVec<[NonNull<DrawOutput<'static>>; INLINE_DRAWS]>,
) -> impl Iterator<Item = &'a DrawOutput<'a>> + Clone {
    (draws.iter()).map(|draw| unsafe { &*(draw.as_ptr() as *const DrawOutput<'a>) })
}

impl<'a, const N: usize> RasterBuilder<'a, '_, N> {
    /// Depth-tests the draws against `image`, a view of one mip (nearer = greater). `clear`
    /// resets the depth first.
    pub fn depth_attachment<T: DepthTarget + 'a>(
        &mut self,
        image: T,
        clear: Option<<T::Format as Format>::Texels>,
    ) -> &mut Self {
        let image = image.view();
        image.aspect_check(vk::ImageAspectFlags::DEPTH);
        image.usage_check(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT);
        one_mip(image.mip_range);
        fits(self.extent, image.extent, "depth");
        let format = image.format();
        self.depth_format = format.vk_format();
        self.stencil_format = match format.aspects().contains(vk::ImageAspectFlags::STENCIL) {
            true => format.vk_format(),
            false => vk::Format::UNDEFINED,
        };
        self.depth_attachment = image.view;
        self.clear_depth = clear.map(|e| format.clear_value(e).into());
        self.depth_discard = Discard::default();
        self.depth_access = Some(image.access_in(
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
            vk::ImageLayout::ATTACHMENT_OPTIMAL,
        ));
        self
    }

    /// Doesn't load the depth attachment (`DONT_CARE`), like [`Attachment::discard_before`].
    /// Panics without a depth attachment or for a cleared one.
    #[track_caller]
    pub fn discard_depth_before(&mut self) -> &mut Self {
        assert!(self.depth_access.is_some(), "`discard_depth_before` needs a `depth_attachment`");
        self.depth_discard.before(self.clear_depth);
        self
    }

    /// Doesn't store the depth attachment (`DONT_CARE`), like [`Attachment::discard_after`]:
    /// for a depth buffer only this rendering tests against. Panics without a depth attachment.
    #[track_caller]
    pub fn discard_depth_after(&mut self) -> &mut Self {
        assert!(self.depth_access.is_some(), "`discard_depth_after` needs a `depth_attachment`");
        self.depth_discard.after = true;
        self
    }

    /// Replaces the default viewport, the whole render area.
    pub fn viewport(&mut self, viewport: Viewport) -> &mut Self {
        assert!(
            viewport.extent.cmpgt(UVec2::ZERO).all(),
            "a viewport of extent {} has no pixels",
            viewport.extent
        );
        self.viewport = viewport;
        self
    }

    /// Replaces the default scissor, the whole render area. There is one, as there is one
    /// viewport.
    pub fn scissor(&mut self, scissor: Scissor) -> &mut Self {
        let end = scissor.offset.as_i64vec2() + scissor.extent.as_i64vec2();
        assert!(
            scissor.offset.cmpge(IVec2::ZERO).all()
                && end.cmple(I64Vec2::splat(i32::MAX as i64)).all(),
            "a scissor starts at a non-negative offset and ends below 2^31, not {} texels at {}",
            scissor.extent,
            scissor.offset
        );
        self.scissor = rect(scissor.offset, scissor.extent);
        self
    }

    pub fn draw<const E: usize>(&mut self, draw: &'a Draw<'a, N, E>) -> &mut Self {
        let draw = (draw.deref()) as *const DrawOutput<'a> as *mut DrawOutput<'static>;
        self.draws.push(unsafe { NonNull::new_unchecked(draw) });
        self
    }

    /// Records the rendering under the GPU scope `name`, its draws in order. Only depth tests,
    /// depth writes and blending see earlier draws, so no draw may read what an earlier one writes.
    #[validation_trace]
    pub fn record(&mut self, name: &'static str) {
        let _span = tracing::info_span!("draw", pass = name).entered();
        // One mip can't be in two layouts in one rendering, so an attachment whose mip a draw
        // also accesses is rendered in `GENERAL`, its descriptor's layout.
        let in_layout = |mut access: ImageUse<'a>| {
            let a = &access.access;
            let overlaps = |i: ImageUse| {
                let mips = i.access.mip_range;
                i.access.image == a.image
                    && mips.start < a.mip_range.end
                    && a.mip_range.start < mips.end
            };
            if iter_draws(&self.draws).any(|draw| draw.image_accesses().any(overlaps)) {
                access.access.layout = vk::ImageLayout::GENERAL;
            }
            access
        };
        let color_accesses = self.color.each_ref().map(|c| in_layout(c.access.clone()));
        let depth_access = self.depth_access.clone().map(|mut depth| {
            let writes = |d: &DrawOutput| {
                (d.state.depth_test.is_some() && d.state.depth_write) || d.state.stencil.is_some()
            };
            // `DONT_CARE` loads and stores write too.
            let discards = self.depth_discard.before || self.depth_discard.after;
            if self.clear_depth.is_some() || discards || iter_draws(&self.draws).any(writes) {
                depth.access.access |= vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE;
            }
            in_layout(depth)
        });
        let draws = iter_draws(&self.draws);
        check_hazards(name, draws.clone());
        self.cmd_buf.flush_pending(
            draws
                .clone()
                .flat_map(|draw| draw.image_accesses())
                .chain(color_accesses.iter().cloned())
                .chain(depth_access.clone()),
            draws.flat_map(|draw| draw.buffer_accesses()),
        );

        let mut color_layouts = color_accesses.iter().map(|access| access.access.layout);
        let color_attachments = self.color.each_ref().map(|color| {
            vk::RenderingAttachmentInfo::default()
                .image_layout(color_layouts.next().expect("a layout per attachment"))
                .image_view(color.view)
                .load_op(color.discard.load_op(color.clear))
                .store_op(color.store_op())
                .clear_value(color.clear.unwrap_or_default())
        });
        let mut rendering_info = vk::RenderingInfo::default()
            .color_attachments(&color_attachments)
            .layer_count(1)
            .render_area(rect(IVec2::ZERO, self.extent))
            .view_mask(0);

        let depth_info;
        if let Some(depth) = &depth_access {
            let written = depth
                .access
                .access
                .contains(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE);
            let store_op = match (self.depth_discard.after, written) {
                (true, _) => vk::AttachmentStoreOp::DONT_CARE,
                (false, true) => vk::AttachmentStoreOp::STORE,
                (false, false) => vk::AttachmentStoreOp::NONE,
            };
            depth_info = vk::RenderingAttachmentInfo::default()
                .image_layout(depth.access.layout)
                .image_view(self.depth_attachment)
                .load_op(self.depth_discard.load_op(self.clear_depth))
                .store_op(store_op)
                .clear_value(self.clear_depth.unwrap_or_default());
            rendering_info = rendering_info.depth_attachment(&depth_info);
            if self.stencil_format != vk::Format::UNDEFINED {
                rendering_info = rendering_info.stencil_attachment(&depth_info);
            }
        }

        let device = Ctx::device();
        let handle = self.cmd_buf.handle;
        self.cmd_buf.enter(name, true);
        unsafe {
            device.cmd_begin_rendering(handle, &rendering_info);
            device.cmd_set_viewport_with_count(
                handle,
                &[vk::Viewport {
                    x: self.viewport.offset.x as f32,
                    y: self.viewport.offset.y as f32,
                    width: self.viewport.extent.x as f32,
                    height: self.viewport.extent.y as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            device.cmd_set_scissor_with_count(handle, &[self.scissor]);
        }
        let color_formats = self.color.each_ref().map(|c| c.format);
        let rendering = Rendering {
            color_attachments: N,
            depth: depth_access.is_some(),
            stencil: self.stencil_format != vk::Format::UNDEFINED,
        };
        let dynamic = Dynamic::of(Ctx::features_ref());
        let mut state: Option<DrawState> = None;
        let mut assembly: Assembly = None;
        let mut bound: Option<(usize, RasterKeyRef)> = None;
        for draw in iter_draws(&self.draws) {
            if let Some((pool, query)) = draw.samples {
                self.cmd_buf.begin_query(pool, query);
            }
            // The validation layer warns about direct draws of nothing, and an empty index
            // range can't be bound without `nullDescriptor`. Their queries count 0.
            if !matches!(
                draw.command,
                DrawCommand::Draw { instances: 0, .. }
                    | DrawCommand::Indexed { instances: 0, .. }
                    | DrawCommand::Indexed { count: 0, .. }
            ) {
                let sections = draw.bindings.sections();
                let draw_state = DrawState {
                    state: &draw.state,
                    blends: draw.blends(),
                    write_masks: draw.write_masks(),
                };
                #[cfg(bake_blend)]
                let (blends, write_masks) = match dynamic.blend {
                    true => (&[][..], &[][..]),
                    false => (draw_state.blends, draw_state.write_masks),
                };
                let key = RasterKeyRef {
                    color_formats: &color_formats,
                    depth_format: self.depth_format,
                    stencil_format: self.stencil_format,
                    constants: sections.constants,
                    baked: Baked::of(draw_state.state, &dynamic),
                    #[cfg(bake_blend)]
                    blends,
                    #[cfg(bake_blend)]
                    write_masks,
                };
                if bound != Some((sections.pass, key)) {
                    let pipeline = pipelines().raster(&key, sections.pass);
                    unsafe {
                        device.cmd_bind_pipeline(handle, vk::PipelineBindPoint::GRAPHICS, pipeline)
                    };
                    bound = Some((sections.pass, key));
                }
                self.cmd_buf.push_constants(sections.push_constants);
                let vertex = match draw.command {
                    #[cfg(feature = "mesh-shader")]
                    DrawCommand::Launch(_) | DrawCommand::LaunchIndirect { .. } => false,
                    _ => true,
                };
                draw_state.record(
                    state.as_ref(),
                    rendering,
                    vertex,
                    &mut assembly,
                    &dynamic,
                    handle,
                );
                state = Some(draw_state);
                unsafe {
                    match draw.command {
                        DrawCommand::Draw {
                            vertices,
                            instances,
                        } => device.cmd_draw(handle, vertices, instances, 0, 0),
                        DrawCommand::Indexed {
                            indices,
                            flags,
                            count,
                            instances,
                        } => {
                            Functions::device_address_commands().cmd_bind_index_buffer3(
                                handle,
                                &vk::BindIndexBuffer3InfoKHR::default()
                                    .address_range(indices)
                                    .address_flags(flags)
                                    .index_type(IndexType::UINT32),
                            );
                            device.cmd_draw_indexed(handle, count, instances, 0, 0, 0);
                        }
                        DrawCommand::Indirect {
                            draws,
                            flags,
                            count,
                        } => Functions::device_address_commands().cmd_draw_indirect2(
                            handle,
                            &vk::DrawIndirect2InfoKHR::default()
                                .address_range(draws)
                                .address_flags(flags)
                                .draw_count(count),
                        ),
                        DrawCommand::IndirectCount {
                            draws,
                            flags,
                            count,
                            count_flags,
                            max_count,
                        } => Functions::device_address_commands().cmd_draw_indirect_count2(
                            handle,
                            &vk::DrawIndirectCount2InfoKHR::default()
                                .address_range(draws)
                                .address_flags(flags)
                                .count_address_range(count)
                                .count_address_flags(count_flags)
                                .max_draw_count(max_count),
                        ),
                        #[cfg(feature = "mesh-shader")]
                        DrawCommand::Launch([x, y, z]) => {
                            Functions::mesh().cmd_draw_mesh_tasks(handle, x, y, z)
                        }
                        #[cfg(feature = "mesh-shader")]
                        DrawCommand::LaunchIndirect {
                            draws,
                            flags,
                            count,
                        } => Functions::device_address_commands().cmd_draw_mesh_tasks_indirect2(
                            handle,
                            &vk::DrawIndirect2InfoKHR::default()
                                .address_range(draws)
                                .address_flags(flags)
                                .draw_count(count),
                        ),
                    }
                }
            }
            if let Some((pool, query)) = draw.samples {
                self.cmd_buf.end_query(pool, query);
            }
        }
        unsafe { device.cmd_end_rendering(handle) };
        self.cmd_buf.exit(name, true);
        self.draws.clear();
    }
}

#[track_caller]
fn fits(extent: UVec2, attachment: UVec2, what: &str) {
    assert!(
        extent.cmple(attachment).all(),
        "a rendering of extent {extent} doesn't fit its {what} attachment of {attachment}"
    );
}

/// Panics unless `groups` are within the device's limits per dimension and in total.
#[track_caller]
fn check_groups(groups: [u32; 3], max: [u32; 3], total: u64, what: &str) {
    assert!(
        groups.iter().zip(max).all(|(&g, m)| g <= m),
        "{groups:?} {what} exceed the device's {max:?}"
    );
    let product: u64 = groups.iter().map(|&g| g as u64).product();
    assert!(
        product <= total,
        "{groups:?} {what} are {product}, more than the device's {total}"
    );
}

fn indirect_access<T: Copy + Pod>(buffer: BufferSlice<'_, T>) -> BufferAccess {
    buffer.access(
        vk::PipelineStageFlags2::DRAW_INDIRECT,
        vk::AccessFlags2::INDIRECT_COMMAND_READ,
    )
}

impl<'p, K: HasGraphics> CommandBuffer<'p, K> {
    /// Starts a rendering of the area `extent` into the `color` attachments, with viewport and
    /// scissor covering it. Every attachment is at least as large.
    pub fn raster<'a, const N: usize>(
        &'a mut self,
        extent: UVec2,
        color: [Attachment<'a>; N],
    ) -> RasterBuilder<'a, 'p, N> {
        assert!(
            extent.cmpgt(UVec2::ZERO).all(),
            "a rendering of extent {extent} has no pixels"
        );
        for attachment in &color {
            fits(extent, attachment.extent, "color");
        }
        RasterBuilder {
            cmd_buf: &mut self.rec,
            extent,
            viewport: Viewport {
                offset: IVec2::ZERO,
                extent,
            },
            scissor: rect(IVec2::ZERO, extent),
            color,
            depth_format: vk::Format::UNDEFINED,
            stencil_format: vk::Format::UNDEFINED,
            depth_attachment: vk::ImageView::null(),
            depth_access: None,
            clear_depth: None,
            depth_discard: Discard::default(),
            draws: SmallVec::new(),
            marker: PhantomData,
        }
    }
}

impl<K: HasCompute> CommandBuffer<'_, K> {
    fn compute_private(
        &mut self,
        bindings: &BindingOutput<'_, kind::Compute>,
        dispatch: [u32; 3],
        indirect_buffer: Option<BufferSlice<'_, DispatchIndirectCommand>>,
        accesses: Option<Accesses<'_>>,
    ) {
        let sections = bindings.sections();
        let pass = &pass_map().passes[sections.pass];
        let _span = tracing::info_span!("compute", pass = pass.name).entered();
        let (images, buffers) = declared(accesses, pass.kind.stages());
        let rec = &mut self.rec;
        rec.flush_pending(
            sections.images().chain(images),
            (sections.buffers())
                .chain(indirect_buffer.map(indirect_access))
                .chain(buffers),
        );
        rec.push_constants(sections.push_constants);

        let pipeline = pipelines().compute(sections.pass, sections.constants);

        rec.enter(pass.name, true);
        unsafe {
            Ctx::device().cmd_bind_pipeline(rec.handle, vk::PipelineBindPoint::COMPUTE, pipeline);
            if let Some(slice) = indirect_buffer {
                Functions::device_address_commands().cmd_dispatch_indirect2(
                    rec.handle,
                    &vk::DispatchIndirect2InfoKHR::default()
                        .address_range(slice.address_range())
                        .address_flags(slice.address_flags()),
                );
            } else {
                Ctx::device().cmd_dispatch(rec.handle, dispatch[0], dispatch[1], dispatch[2]);
            }
        }
        rec.exit(pass.name, true);
    }

    /// Like [`Self::compute`], with the workgroup counts read from `buffer` on the GPU.
    #[validation_trace]
    pub fn compute_indirect(
        &mut self,
        bindings: &BindingOutput<'_, kind::Compute>,
        buffer: BufferSlice<'_, DispatchIndirectCommand>,
        accesses: Option<Accesses<'_>>,
    ) {
        self.compute_private(bindings, [0, 0, 0], Some(buffer), accesses);
    }

    /// Dispatches `dispatch` workgroups of a compute pass that also reaches `accesses`.
    #[validation_trace]
    pub fn compute(
        &mut self,
        bindings: &BindingOutput<'_, kind::Compute>,
        dispatch: [u32; 3],
        accesses: Option<Accesses<'_>>,
    ) {
        let max = Ctx::device_properties().limits.max_compute_work_group_count;
        check_groups(dispatch, max, u64::MAX, "compute workgroups");
        self.compute_private(bindings, dispatch, None, accesses);
    }

    #[cfg(feature = "ray-tracing-pipeline")]
    fn raytrace_private(
        &mut self,
        bindings: &BindingOutput<'_, kind::RayTracing>,
        rays: [u32; 3],
        indirect_buffer: Option<BufferSlice<'_, TraceRaysIndirectCommand>>,
        accesses: Option<Accesses<'_>>,
    ) {
        let sections = bindings.sections();
        let pass = &pass_map().passes[sections.pass];
        let _span = tracing::info_span!("raytrace", pass = pass.name).entered();
        let (images, buffers) = declared(accesses, pass.kind.stages());
        let rec = &mut self.rec;
        rec.flush_pending(
            sections.images().chain(images),
            (sections.buffers())
                .chain(indirect_buffer.map(indirect_access))
                .chain(buffers),
        );
        rec.push_constants(sections.push_constants);

        rec.enter(pass.name, true);
        let handle = rec.handle;
        pipelines().ray_tracing(sections.pass, sections.constants, |pipeline| unsafe {
            // Past the pass's check: `ray_tracing` panics for hand-made bindings without it.
            let functions = Functions::raytracing_pipeline();
            Ctx::device().cmd_bind_pipeline(
                handle,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                pipeline.pipeline,
            );
            let [raygen, miss, hit, callable] = &pipeline.sbt.regions;
            match indirect_buffer {
                Some(slice) => functions.cmd_trace_rays_indirect(
                    handle,
                    raygen,
                    miss,
                    hit,
                    callable,
                    slice.address(),
                ),
                None => functions.cmd_trace_rays(
                    handle, raygen, miss, hit, callable, rays[0], rays[1], rays[2],
                ),
            }
        });
        rec.exit(pass.name, true);
    }

    /// Traces `rays` (width, height, depth) raygen invocations of a pass that also reaches
    /// `accesses`.
    #[cfg(feature = "ray-tracing-pipeline")]
    #[validation_trace]
    pub fn raytrace(
        &mut self,
        bindings: &BindingOutput<'_, kind::RayTracing>,
        rays: [u32; 3],
        accesses: Option<Accesses<'_>>,
    ) {
        let limits = &Ctx::device_properties().limits;
        let max = std::array::from_fn(|i| {
            let size = limits.max_compute_work_group_count[i] as u64;
            (size * limits.max_compute_work_group_size[i] as u64).min(u32::MAX as u64) as u32
        });
        // The bindings exist only past the feature's check.
        let total = (Ctx::physical_device().ray_tracing_pipeline_properties)
            .map_or(0, |p| p.max_ray_dispatch_invocation_count as u64);
        check_groups(rays, max, total, "rays");
        self.raytrace_private(bindings, rays, None, accesses);
    }

    /// Like [`Self::raytrace`], with the ray counts read from `buffer` on the GPU.
    #[cfg(feature = "ray-tracing-pipeline")]
    #[validation_trace]
    pub fn raytrace_indirect(
        &mut self,
        bindings: &BindingOutput<'_, kind::RayTracing>,
        buffer: BufferSlice<'_, TraceRaysIndirectCommand>,
        accesses: Option<Accesses<'_>>,
    ) {
        self.raytrace_private(bindings, [0, 0, 0], Some(buffer), accesses);
    }
}

impl<K: QueueKind> CommandBuffer<'_, K> {
    /// Debugging aid: a full barrier the tracking doesn't know of. If a glitch goes away with
    /// it, an automatic barrier before it is missing.
    pub fn flush_all(&mut self) {
        let all = vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE;
        let memory = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(all)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(all)];
        let info = vk::DependencyInfo::default().memory_barriers(&memory);
        unsafe { Ctx::device().cmd_pipeline_barrier2(self.rec.handle, &info) };
    }

    /// Records `f` under a debug label, a scope the observer of [`Self::measure`] sees.
    pub fn scope<R>(&mut self, name: &'static str, f: impl FnOnce(&mut Self) -> R) -> R {
        self.rec.enter(name, false);
        let result = f(self);
        self.rec.exit(name, false);
        result
    }

    /// Records `f` as the scope `name` with `observer` attached to it and every scope and pass
    /// inside. Doesn't nest.
    pub fn measure<R>(
        &mut self,
        observer: Option<&mut dyn ScopeObserver>,
        name: &'static str,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let Some(observer) = observer else {
            return self.scope(name, f);
        };
        assert!(self.rec.observer.is_none(), "`measure` doesn't nest");
        // SAFETY: only the lifetime is erased. The pointer is cleared before `observer`'s borrow
        // ends, also when `f` panics, and until then only this command buffer reaches it.
        let observer = unsafe {
            std::mem::transmute::<NonNull<dyn ScopeObserver + '_>, NonNull<dyn ScopeObserver>>(
                NonNull::from(observer),
            )
        };
        self.rec.observer = Some(observer);
        self.rec.depth = 0;
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| self.scope(name, f)));
        self.rec.observer = None;
        result.unwrap_or_else(|panic| std::panic::resume_unwind(panic))
    }

    /// Writes the clock once the commands recorded before reached `stage`: `NONE` for when they
    /// started, `ALL_COMMANDS` for when they finished.
    pub fn write_timestamp(
        &mut self,
        pool: &QueryPool<Timestamp>,
        index: u32,
        stage: vk::PipelineStageFlags2,
    ) {
        self.rec.write_timestamp(pool, index, stage);
    }

    /// Records `f` inside query `index` of `pool`; one query of a kind can be active at a time.
    /// A performance query's pool has to exist before the recording begins.
    pub fn query<Q: Scoped, R>(
        &mut self,
        pool: &QueryPool<Q>,
        index: u32,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        assert!(
            index < pool.count(),
            "query {index} of a pool of {}",
            pool.count()
        );
        self.rec.begin_query(pool, index);
        let result = f(self);
        self.rec.end_query(pool, index);
        result
    }
}

impl<K: QueueKind> CommandBuffer<'_, K> {
    /// Records the release of `image` to the family of kind `D`, whose next access acquires it.
    /// That submission has to wait for this one (`Submission::wait_for`), which lava doesn't check.
    #[validation_trace]
    pub fn release<D: QueueKind>(&mut self, image: impl ImageRef) {
        if let Err(missing) = D::check() {
            panic!("`release` to a family lava doesn't have: {missing}");
        }
        let view = image.view();
        let aspect = view.format().aspects();
        (self.rec).release(view.image, view.states, view.mip_range, aspect, D::KIND);
    }
}

impl Recording<'_> {
    /// Records the release of `mips` of `image`, whose tracked states are `states`, to the
    /// family of kind `to`; mips nobody owns stay unowned. Panics unless this family owns them.
    pub(crate) fn release(
        &mut self,
        image: vk::Image,
        states: &[AtomicU64],
        mips: Range<u32>,
        aspect: vk::ImageAspectFlags,
        to: Kind,
    ) {
        if to == self.family.kind {
            return;
        }
        let (from, to_index) = (self.family.index(self.family.kind), self.family.index(to));
        // It waits for this timeline's accesses of the image; the acquire waits for it through
        // the semaphore its submission signals.
        let pending = &self.pending_accesses;
        let tracked = (pending.image_reads.iter().map(|r| &r.read))
            .chain(pending.image_writes.iter().map(|w| &w.write))
            .filter(|a| a.image == image);
        let (mut stage, mut access) = (S2::NONE, vk::AccessFlags2::NONE);
        for a in tracked {
            stage |= a.stage;
            access |= a.access;
        }
        if stage.is_empty() {
            stage = S2::ALL_COMMANDS;
            access = vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE;
        }
        let mut images = SmallVec::new();
        release_states(states, mips, self.family.kind, to, |mips, layout| {
            images.push(
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(stage)
                    .src_access_mask(access)
                    .dst_stage_mask(S2::NONE)
                    .dst_access_mask(vk::AccessFlags2::NONE)
                    .old_layout(layout)
                    .new_layout(layout)
                    .src_queue_family_index(from)
                    .dst_queue_family_index(to_index)
                    .image(image)
                    .subresource_range(subresource_range(aspect, mips)),
            )
        });
        if images.is_empty() {
            return;
        }
        self.barrier_log.push(Barriers {
            acquired: SmallVec::new(),
            acquires: SmallVec::new(),
            memory: None,
            images,
        });
        self.pipeline_barrier(self.barrier_log.last().expect("just pushed"));
        // The family it goes to waits for this submission as a whole. Mips this family still
        // owns keep the image's accesses, which their later accesses wait for.
        if released_by(states, self.family.kind) {
            self.pending_accesses.forget_image(image);
        }
    }
}

impl<'p> Recording<'p> {
    pub(crate) fn new(
        handle: vk::CommandBuffer,
        family: FamilyInfo,
        pending_accesses: &'p mut PendingAccesses,
        timeline: usize,
        value: u64,
        barrier_log: &'p mut Vec<Barriers>,
        staging: Option<&'p Staging>,
    ) -> Self {
        Self {
            handle,
            pending_accesses,
            timeline,
            value,
            family,
            observer: None,
            depth: 0,
            active_queries: 0,
            barrier_log,
            acquired: SmallVec::new(),
            presents: SmallVec::new(),
            staging,
        }
    }

    pub(crate) fn pipeline_barrier(&self, barriers: &Barriers) {
        // Barriers of one call are unordered, so acquires go first in their own.
        if !barriers.acquires.is_empty() {
            let info = vk::DependencyInfo::default().image_memory_barriers(&barriers.acquires);
            unsafe { Ctx::device().cmd_pipeline_barrier2(self.handle, &info) };
        }
        let mut info = vk::DependencyInfo::default();
        let memory = barriers.memory.map(|memory| [memory]);
        if let Some(memory) = &memory {
            info = info.memory_barriers(memory);
        }
        if !barriers.images.is_empty() {
            info = info.image_memory_barriers(&barriers.images);
        }
        if memory.is_some() || !barriers.images.is_empty() {
            unsafe { Ctx::device().cmd_pipeline_barrier2(self.handle, &info) };
        }
    }

    fn push_constants(&mut self, constants: &[u8]) {
        if constants.is_empty() {
            return;
        }
        // Push data comes in whole words.
        let mut padded = [0u8; MAX_PUSH_CONSTANTS_SIZE];
        let data = if constants.len() % 4 == 0 {
            constants
        } else {
            padded[..constants.len()].copy_from_slice(constants);
            &padded[..constants.len().next_multiple_of(4)]
        };
        let info = vk::PushDataInfoEXT::default().data(vk::HostAddressRangeConstEXT {
            address: data.as_ptr().cast(),
            size: data.len(),
        });
        unsafe { Functions::descriptor_heap().cmd_push_data(self.handle, &info) };
    }

    /// Records the barriers a command with these accesses needs and tracks its accesses.
    pub(crate) fn flush_pending<'a>(
        &mut self,
        image_acceses: impl Iterator<Item = ImageUse<'a>> + Clone,
        buffer_acceses: impl Iterator<Item = BufferAccess> + Clone,
    ) {
        let barriers = compute_barriers(
            &self.pending_accesses,
            &self.family,
            image_acceses.clone(),
            buffer_acceses.clone(),
        );
        if let Some(barriers) = barriers {
            self.acquired.extend_from_slice(&barriers.acquired);
            self.pending_accesses.made_visible(&barriers, self.value);
            self.barrier_log.push(barriers);
            let barriers = self.barrier_log.last().expect("just pushed");
            self.pipeline_barrier(barriers);
        }

        (self.pending_accesses).record(
            self.value,
            image_acceses.map(|image| image.access),
            buffer_acceses,
        );
    }

    fn enter(&mut self, name: &'static str, pass: bool) {
        if let Some(debug_utils) = Functions::debug_utils() {
            with_debug_name(name, |name| {
                let label = vk::DebugUtilsLabelEXT::default().label_name(name);
                unsafe { debug_utils.cmd_begin_debug_utils_label(self.handle, &label) };
            });
        }
        if let Some(mut observer) = self.observer {
            let scope = ScopeInfo {
                name,
                depth: self.depth,
                pass,
            };
            self.depth += 1;
            // SAFETY: see `CommandBuffer::measure`.
            unsafe { observer.as_mut() }.on_enter(&mut QueryRecorder { cmd: self }, &scope);
        }
    }

    fn exit(&mut self, name: &'static str, pass: bool) {
        if let Some(mut observer) = self.observer {
            self.depth -= 1;
            let scope = ScopeInfo {
                name,
                depth: self.depth,
                pass,
            };
            // SAFETY: see `CommandBuffer::measure`.
            unsafe { observer.as_mut() }.on_exit(&mut QueryRecorder { cmd: self }, &scope);
        }
        if let Some(debug_utils) = Functions::debug_utils() {
            unsafe { debug_utils.cmd_end_debug_utils_label(self.handle) };
        }
    }

    pub(crate) fn write_timestamp(
        &mut self,
        pool: &QueryPool<Timestamp>,
        index: u32,
        stage: vk::PipelineStageFlags2,
    ) {
        assert!(
            index < pool.count(),
            "query {index} of a pool of {}",
            pool.count()
        );
        pool.record(self);
        let bits = self.family.timestamp_valid_bits;
        assert!(
            bits > 0,
            "the {:?} family has no timestamps",
            self.family.kind
        );
        let mask = if bits >= 64 { !0 } else { (1 << bits) - 1 };
        pool.kind
            .valid_bits
            .store(mask, std::sync::atomic::Ordering::Relaxed);
        unsafe { Ctx::device().cmd_write_timestamp2(self.handle, stage, pool.handle, index) };
    }

    pub(crate) fn begin_query<Q: Scoped>(&mut self, pool: &QueryPool<Q>, index: u32) {
        let bit = query_bit(Q::TYPE);
        assert!(
            self.active_queries & bit == 0,
            "a query of this kind is already active, and queries of one kind don't nest"
        );
        pool.kind.check_kind(self.family.kind);
        pool.record(self);
        let flags = pool.kind.control_flags();
        unsafe { Ctx::device().cmd_begin_query(self.handle, pool.handle, index, flags) };
        self.active_queries |= bit;
    }

    pub(crate) fn end_query<Q: QueryKind>(&mut self, pool: &QueryPool<Q>, index: u32) {
        let bit = query_bit(Q::TYPE);
        assert!(
            self.active_queries & bit != 0,
            "no query of this kind is active"
        );
        unsafe { Ctx::device().cmd_end_query(self.handle, pool.handle, index) };
        self.active_queries &= !bit;
    }
}

const INLINE_BARRIERS: usize = 8;

#[derive(Clone, Debug)]
pub(crate) struct Barriers {
    /// Acquired swapchain images with the stages that wait for their acquire.
    pub(crate) acquired: SmallVec<[(vk::Image, vk::PipelineStageFlags2); 1]>,
    pub(crate) acquires: SmallVec<[vk::ImageMemoryBarrier2<'static>; 2]>,
    pub(crate) memory: Option<vk::MemoryBarrier2<'static>>,
    /// A rendering transitions all its attachments and the images its draws read at once.
    pub(crate) images: SmallVec<[vk::ImageMemoryBarrier2<'static>; INLINE_BARRIERS]>,
}

/// The resources a pass reaches without naming them in its bindings (through a bindless handle
/// or a pointer in a buffer), tracked at the command's shader stages.
#[derive(Clone, Copy, Default)]
pub struct Accesses<'a> {
    pub images: &'a [ImageUse<'a>],
    pub buffers: &'a [BufferAccess],
}

impl<'a> Accesses<'a> {
    pub fn new(images: &'a [ImageUse<'a>], buffers: &'a [BufferAccess]) -> Self {
        Self { images, buffers }
    }

    pub fn images(images: &'a [ImageUse<'a>]) -> Self {
        Self {
            images,
            buffers: &[],
        }
    }

    pub fn buffers(buffers: &'a [BufferAccess]) -> Self {
        Self {
            images: &[],
            buffers,
        }
    }
}

fn declared<'a>(
    accesses: Option<Accesses<'a>>,
    stage: vk::PipelineStageFlags2,
) -> (
    impl Iterator<Item = ImageUse<'a>> + Clone,
    impl Iterator<Item = BufferAccess> + Clone,
) {
    let Accesses { images, buffers } = accesses.unwrap_or_default();
    let images = images.iter().cloned().map(move |mut image| {
        image.access.stage = stage;
        image
    });
    let buffers = buffers.iter().cloned().map(move |mut buffer| {
        buffer.stage = stage;
        buffer
    });
    (images, buffers)
}

/// The image accesses of [`Accesses`]; the command sets the stage.
impl<'a> ImageUse<'a> {
    /// The pass samples `image` (`Tex2D`, or a `DynImg` with a sampled index).
    pub fn sampled_read(image: impl SampledView + 'a) -> Self {
        let image = image.view();
        image.usage_check(vk::ImageUsageFlags::SAMPLED);
        image.access(S2::NONE, vk::AccessFlags2::SHADER_SAMPLED_READ)
    }

    /// The pass loads from `image` as a storage image (`Image2D`, or a `DynImg` without a
    /// sampled index).
    pub fn storage_read(image: impl StorageView + 'a) -> Self {
        let image = image.view();
        image.usage_check(vk::ImageUsageFlags::STORAGE);
        image.access(S2::NONE, vk::AccessFlags2::SHADER_STORAGE_READ)
    }

    /// The pass loads from and stores to `image` (`MutImage2D`).
    pub fn storage_write(image: impl StorageView + 'a) -> Self {
        let image = image.view();
        image.usage_check(vk::ImageUsageFlags::STORAGE);
        let access = vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE;
        image.access(S2::NONE, access)
    }
}

/// The buffer accesses of [`Accesses`]; the command sets the stage.
impl BufferAccess {
    /// The pass reads `buffer` through a pointer (`Buf`).
    pub fn read<V: Copy + Pod>(buffer: BufferSlice<'_, V>) -> Self {
        buffer.access(S2::NONE, vk::AccessFlags2::SHADER_STORAGE_READ)
    }

    /// The pass reads and writes `buffer` through a pointer (`MutBuf`).
    pub fn write<V: Copy + Pod>(buffer: BufferSlice<'_, V>) -> Self {
        let access = vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE;
        buffer.access(S2::NONE, access)
    }

    /// The pass traces `tlas` through an address it reads from a buffer (an `Accel` inside a
    /// struct): its memory and the BLASes it names.
    #[cfg(feature = "acceleration-structure")]
    pub fn trace(tlas: &Tlas) -> [Self; 2] {
        tlas.trace_accesses(S2::NONE)
    }
}

/// The barriers a command with these accesses needs after `pending`. It moves each image's
/// tracked state, so call it once per command, when the command is recorded.
pub(crate) fn compute_barriers<'a>(
    pending: &PendingAccesses,
    family: &FamilyInfo,
    image_acceses: impl Iterator<Item = ImageUse<'a>> + Clone,
    buffer_acceses: impl Iterator<Item = BufferAccess>,
) -> Option<Barriers> {
    let mut src_stage = vk::PipelineStageFlags2::NONE;
    let mut dst_stage = vk::PipelineStageFlags2::NONE;
    let mut src_access = vk::AccessFlags2::empty();
    let mut dst_access = vk::AccessFlags2::empty();

    for buffer_access in buffer_acceses {
        let buffer_access = &buffer_access;
        let is_write = buffer_access.access.intersects(all_write_access());
        let is_read = buffer_access.access.intersects(all_read_access());

        if is_write {
            for pending in pending
                .buffer_reads
                .iter()
                .filter(|r| r.overlaps(buffer_access))
            {
                src_stage |= pending.stage;
                src_access |= pending.access;
                dst_stage |= buffer_access.stage;
                dst_access |= buffer_access.access;
            }
        }
        for w in pending.buffer_writes.iter().filter(|w| {
            w.write.overlaps(buffer_access)
                && !w.visible_to(buffer_access.stage, buffer_access.access)
        }) {
            if is_read || is_write {
                src_stage |= w.write.stage;
                src_access |= w.write.access;
                dst_stage |= buffer_access.stage;
                dst_access |= buffer_access.access;
            }
        }
    }

    for image in image_acceses.clone() {
        // The first access of an acquired swapchain image (which has one state) waits for the
        // acquire's semaphore, which comes after every earlier access of it.
        if let [state] = image.states
            && ImageState::load(state).acquired
        {
            continue;
        }
        let image_access = image.access;
        let is_write = image_access.access.intersects(all_write_access());
        let is_read = image_access.access.intersects(all_read_access());

        if is_write {
            for pending in pending
                .image_reads
                .iter()
                .filter(|r| r.image == image_access.image)
            {
                src_stage |= pending.stage;
                src_access |= pending.access;
                dst_stage |= image_access.stage;
                dst_access |= image_access.access;
            }
        }
        for w in pending.image_writes.iter().filter(|w| {
            w.write.image == image_access.image
                && !w.visible_to(image_access.stage, image_access.access)
        }) {
            if is_read || is_write {
                src_stage |= w.write.stage;
                src_access |= w.write.access;
                dst_stage |= image_access.stage;
                dst_access |= image_access.access;
            }
        }
    }

    let kind = family.kind;
    // Every image is checked before any state moves: a panic then leaves the tracking as the
    // barriers logged so far describe it.
    for image in image_acceses.clone() {
        check_states(image.states, image.access.mip_range, kind);
    }
    let all = vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE;
    let mut image_barriers: SmallVec<[vk::ImageMemoryBarrier2; INLINE_BARRIERS]> = SmallVec::new();
    let mut acquires = SmallVec::new();
    let mut acquired = SmallVec::new();
    for image in image_acceses {
        let access = &image.access;
        let to_stage = if access.stage.is_empty() {
            S2::ALL_COMMANDS
        } else {
            access.stage
        };
        let to_access = if access.access.is_empty() {
            all
        } else {
            access.access
        };
        take_checked_states(
            image.states,
            access.mip_range,
            access.layout,
            kind,
            |mips, old| {
                // The first access of an acquired swapchain image waits for the acquire at its
                // stages, which the transition then starts from.
                let waited = old.acquired.then(|| {
                    acquired.push((access.image, to_stage));
                    to_stage
                });
                // Released to this family: the acquire half of the transfer.
                let acquire = old.released_to == Some(kind) && old.owner.is_some();
                if acquire && let Some(owner) = old.owner {
                    acquires.push(
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(S2::NONE)
                            .src_access_mask(vk::AccessFlags2::NONE)
                            .dst_stage_mask(to_stage)
                            .dst_access_mask(to_access)
                            .old_layout(old.layout)
                            .new_layout(old.layout)
                            .src_queue_family_index(family.index(owner))
                            .dst_queue_family_index(family.index(kind))
                            .image(access.image)
                            .subresource_range(subresource_range(access.aspect, mips)),
                    );
                }
                if old.layout == access.layout {
                    return;
                }
                // The transition waits for the image's tracked accesses. Without any, a
                // submission it waited for may still use it, so it waits for everything, unless
                // no command ever accessed it.
                let (mut src_stage, mut src_access) = (S2::NONE, vk::AccessFlags2::NONE);
                for p in (pending.image_reads.iter().map(|r| &r.read))
                    .chain(pending.image_writes.iter().map(|w| &w.write))
                    .filter(|p| p.image == access.image)
                {
                    src_stage |= p.stage;
                    src_access |= p.access;
                }
                let from_present = old.layout == vk::ImageLayout::PRESENT_SRC_KHR;
                let fresh = old == ImageState::UNDEFINED;
                if !fresh && (src_stage.is_empty() || from_present) {
                    src_stage = S2::ALL_COMMANDS;
                    src_access = if from_present {
                        vk::AccessFlags2::NONE
                    } else {
                        all
                    };
                }
                // It comes after the acquire, a barrier call of its own, at its stages.
                if acquire {
                    src_stage |= to_stage;
                }
                // The semaphore wait chains with the transition at the stages it waits at.
                if let Some(stage) = waited {
                    src_stage = stage;
                    src_access = vk::AccessFlags2::NONE;
                }
                image_barriers.push(
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(src_stage)
                        .src_access_mask(src_access)
                        .dst_stage_mask(to_stage)
                        .dst_access_mask(match access.layout {
                            vk::ImageLayout::PRESENT_SRC_KHR => vk::AccessFlags2::NONE,
                            _ => to_access,
                        })
                        .old_layout(old.layout)
                        .new_layout(access.layout)
                        // Not an ownership transfer (ash's default would be family 0 to 0).
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(access.image)
                        .subresource_range(subresource_range(access.aspect, mips)),
                );
            },
        );
    }

    let hazard = !src_stage.is_empty() || !dst_stage.is_empty();
    if !hazard && image_barriers.is_empty() && acquired.is_empty() && acquires.is_empty() {
        return None;
    }

    // A hazard with an access of no stage (`present`) waits for, or holds back, everything.
    if src_stage.is_empty() {
        src_stage = vk::PipelineStageFlags2::ALL_COMMANDS;
    }
    if dst_stage.is_empty() {
        dst_stage = vk::PipelineStageFlags2::ALL_COMMANDS;
    }
    if src_access.is_empty() {
        src_access = all;
    }

    Some(Barriers {
        acquired,
        acquires,
        memory: hazard.then(|| {
            vk::MemoryBarrier2::default()
                .src_stage_mask(src_stage)
                .src_access_mask(src_access)
                .dst_stage_mask(dst_stage)
                .dst_access_mask(dst_access)
        }),
        images: image_barriers,
    })
}

fn subresource_range(aspect: vk::ImageAspectFlags, mips: Range<u32>) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: aspect,
        base_array_layer: 0,
        layer_count: 1,
        base_mip_level: mips.start,
        level_count: mips.end - mips.start,
    }
}

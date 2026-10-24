//! Command recording for generated passes: the typed `CommandBuffer<K>` over the untyped `Recording`, barriers and queue family ownership releases, renderings (render areas, depth or stencil attachments, layered and multiview) and their draws (direct, indexed, indirect from any `StartsWith` command records, counted, multi-draws, mesh launches), dispatches (direct, from a workgroup base, indirect; the last bound compute and ray tracing pipeline kept), scopes and queries (transfers live with the slices and views they take, `drain_uploads` with the staging rings in `buffer/staging.rs`, acceleration structure commands with the structures, `present` with `PresentSurface` in `present.rs`)
use std::{
    fmt::Debug, marker::PhantomData, ops::Deref, panic::AssertUnwindSafe, ptr::NonNull,
    range::Range, sync::atomic::AtomicU64,
};

use crate::vk::{self, PipelineStageFlags2 as S2};
#[cfg(feature = "acceleration-structure")]
use crate::vkobjects::acceleration_structure::Tlas;
use crate::{
    buffer::slice::BufferSlice,
    image::{
        dim::Dim2D,
        format::Format,
        kind::{ColorTarget, DepthTarget, ImageRef, SampledView, StencilTarget, StorageView},
        slice::ImageView,
    },
    pipeline::{MAX_PUSH_CONSTANTS_SIZE, RasterKeyRef, pass_map, pipelines},
    present::Present,
    query::{
        Occlusion, QueryPool, RecordedOn, ScopeInfo, ScopeObserver, Scoped, Timestamp,
        observer::{Observer, Typed},
    },
    queue::{FamilyInfo, HasCompute, HasGraphics, Kind, QueueKind},
    state::{Ctx, Functions, with_debug_name},
};
use bytemuck::{NoUninit, Pod, Zeroable};
use glam::UVec2;
#[cfg(feature = "multi-draw")]
use crate::features::{Gated, gate};
use lava_macros::validation_trace;
use smallvec::SmallVec;

use crate::buffer::staging::Staging;
#[cfg(feature = "ray-tracing-pipeline")]
use crate::vkobjects::rt_pipeline::BoundRayTracing;

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
    /// Draws of one rendering may make it alongside each other's (`unordered`).
    pub(crate) unordered: bool,
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
    /// The same access, made by draws of one rendering in any order: a rendering doesn't
    /// refuse two draws whose accesses of a range are both unordered, though one writes it
    /// (`RasterBuilder::record` refuses other pairs). Nothing orders such draws' accesses, so
    /// the application guarantees their result doesn't depend on it: atomics that commute
    /// (counters, `InterlockedAdd`, min and max), or writes of elements no other draw of the
    /// rendering reads or writes. Commands of other renderings or passes are ordered with it
    /// by barriers as usual.
    pub fn unordered(mut self) -> Self {
        self.unordered = true;
        self
    }

    /// Whether it is [`Self::unordered`].
    pub fn is_unordered(&self) -> bool {
        self.unordered
    }

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
    observer: Option<NonNull<dyn Observer>>,
    depth: u16,
    /// The active query of each scoped kind (`query_slot`), with its pool: one of each kind can
    /// be active.
    pub(crate) open_queries: [Option<(vk::QueryPool, u32)>; 3],
    /// The hardware counter pool the recording uses: one per recording.
    performance_pool: Option<vk::QueryPool>,
    /// The counter query `Submission::submit_profiled` or `Timeline::capture` wraps the whole
    /// recording in.
    pub(crate) profile: Option<Profile>,
    /// Set while that query begins or ends, the one place counters of command-buffer scope can.
    profiling: bool,
    /// Between `vkCmdBeginRendering` and `vkCmdEndRendering`.
    in_rendering: bool,
    /// Every barrier the tracking already counts, so a recording that never reaches the GPU is
    /// replaced by one of just these, which keeps the tracking true. The frame slot's, reused.
    pub(crate) barrier_log: &'p mut Vec<Barriers>,
    /// Acquired swapchain images with the stages of their first access.
    pub(crate) acquired: SmallVec<[(vk::Image, vk::PipelineStageFlags2); 1]>,
    pub(crate) presents: SmallVec<[Present; 1]>,
    /// The timeline's staging ring, which `drain_uploads` records.
    pub(crate) staging: Option<&'p Staging>,
    /// The compute and ray tracing pipeline bound last, so repeated dispatches skip the lookup.
    bound_compute: Option<Bound<vk::Pipeline>>,
    #[cfg(feature = "ray-tracing-pipeline")]
    bound_ray_tracing: Option<Bound<BoundRayTracing>>,
    /// The acceleration structure builds (and copies into structures) it recorded, applied
    /// once it reaches the GPU (`finish_structure_builds`).
    #[cfg(feature = "acceleration-structure")]
    pub(crate) structure_builds: crate::vkobjects::acceleration_structure::RecordedBuilds,
}

/// A pipeline a recording bound, with the pass and constant bytes it was looked up for.
struct Bound<P> {
    pass: usize,
    constants: SmallVec<[u8; 16]>,
    pipeline: P,
}

impl<P: Copy> Bound<P> {
    /// The pipeline of `pass` with `constants`: the bound one if it was looked up for them,
    /// else `lookup`'s, then bound by `bind` unless it is the bound one (another pass's
    /// specialization may share it).
    fn get(
        slot: &mut Option<Self>,
        pass: usize,
        constants: &[u8],
        lookup: impl FnOnce() -> P,
        same: impl Fn(&P, &P) -> bool,
        bind: impl FnOnce(&P),
    ) -> P {
        if let Some(bound) = slot
            && bound.pass == pass
            && *bound.constants == *constants
        {
            return bound.pipeline;
        }
        let pipeline = lookup();
        if !slot.as_ref().is_some_and(|bound| same(&bound.pipeline, &pipeline)) {
            bind(&pipeline);
        }
        *slot = Some(Bound {
            pass,
            constants: SmallVec::from_slice(constants),
            pipeline,
        });
        pipeline
    }
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

/// The query of a hardware counter pool around a whole recording: query `index` of `pool`, in
/// counter pass `pass` (a capture's later passes keep the pool's earlier counters).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Profile {
    pub(crate) pool: vk::QueryPool,
    pub(crate) index: u32,
    pub(crate) pass: u32,
}

/// Where a recording keeps the active query of a scoped kind.
fn query_slot(kind: vk::QueryType) -> usize {
    match kind {
        vk::QueryType::OCCLUSION => 0,
        vk::QueryType::PIPELINE_STATISTICS => 1,
        vk::QueryType::PERFORMANCE_QUERY_KHR => 2,
        _ => unreachable!("{kind:?} queries aren't scoped"),
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
    area: Rect,
    viewport: Rect,
    scissor: Rect,
    /// The layers a layered rendering draws into (`layers`).
    layer_count: u32,
    /// The views of a multiview rendering (`view_mask`), 0 for none.
    view_mask: u32,
    color: [Attachment<'a>; N],
    depth_stencil: Option<DepthStencil<'a>>,
    draws: DrawList<'a>,
}

/// The draws a rendering borrows: pointers, tied to `'a` by the marker, so draws can be
/// temporaries of the recording statement that drop before the builder (a `SmallVec` of
/// references would need them alive at its `Drop`).
struct DrawList<'a> {
    draws: SmallVec<[NonNull<DrawOutput<'static>>; INLINE_DRAWS]>,
    marker: PhantomData<&'a DrawOutput<'a>>,
}

impl<'a> DrawList<'a> {
    fn push(&mut self, draw: &'a DrawOutput<'a>) {
        let draw = draw as *const DrawOutput<'a> as *mut DrawOutput<'static>;
        // SAFETY: from a reference.
        self.draws.push(unsafe { NonNull::new_unchecked(draw) });
    }

    /// The draws, borrowed for no longer than the list, which lives within `'a`.
    fn iter(&self) -> impl Iterator<Item = &DrawOutput<'a>> + Clone + '_ {
        // SAFETY: every pointer came from a `&'a DrawOutput<'a>` in `push`.
        (self.draws.iter()).map(|draw| unsafe { &*(draw.as_ptr() as *const DrawOutput<'a>) })
    }
}

/// A color attachment of a rendering, cleared to `clear` first unless that is `None`, else
/// loaded, and stored after the draws; [`Self::discard_before`] and [`Self::discard_after`]
/// skip the load and the store.
pub struct Attachment<'a> {
    format: vk::Format,
    view: vk::ImageView,
    extent: UVec2,
    layers: u32,
    clear: Option<vk::ClearValue>,
    discard: Discard,
    access: ImageUse<'a>,
}

/// The depth or stencil attachment of a rendering, both for a combined format attached as depth.
struct DepthStencil<'a> {
    view: vk::ImageView,
    format: vk::Format,
    /// Attached as the depth, the stencil attachment, or both.
    depth: bool,
    stencil: bool,
    layers: u32,
    /// Only a read; `record` adds the write.
    access: ImageUse<'a>,
    clear: Option<vk::ClearValue>,
    discard: Discard,
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
    /// Renders into `image`, a view of one mip: one layer, or the layers of an array a layered
    /// or multiview rendering draws into.
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
            view: image.vk_view(),
            extent: image.extent,
            layers: image.layer_count(),
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
        command: DrawCommand<'a>,
        state: RasterState<0>,
        samples: Option<(&'a QueryPool<Occlusion>, u32)>,
    } sections {
        extra: [BufferAccess; E],
        blends: [Blend; N],
        write_masks: [ColorComponentFlags; N],
    }
}

impl<'a> DrawOutput<'a> {
    fn image_accesses(&self) -> impl Iterator<Item = ImageUse<'a>> + Clone {
        self.bindings.sections().images()
    }

    fn buffer_accesses(&self) -> impl Iterator<Item = BufferAccess> + Clone + '_ {
        (self.bindings.sections().buffers()).chain(self.extra().iter().cloned())
    }

    /// A vertex draw, which has a topology, rather than a mesh launch.
    fn is_vertex(&self) -> bool {
        self.command.is_vertex()
    }
}

/// Panics if a draw accesses what another draw of the rendering writes: draws are ordered only
/// by depth tests and blending, and nothing else passes between them. Attachments aren't draws'
/// accesses, so they are left out, and so are pairs of [`BufferAccess::unordered`] accesses.
fn check_hazards<'a>(name: &str, draws: impl Iterator<Item = &'a DrawOutput<'a>> + Clone) {
    let writes = |access: vk::AccessFlags2| access.intersects(all_write_access());
    let hazard = |i: usize, j: usize, what: std::fmt::Arguments| -> ! {
        panic!(
            "draws {i} and {j} of the rendering `{name}` both access {what}, which draw {i} \
             writes: draws of one rendering are ordered only by depth tests and blending, so \
             record them in two renderings (or mark accesses whose order doesn't matter \
             `BufferAccess::unordered`)"
        )
    };
    for (i, draw) in draws.clone().enumerate() {
        for write in draw.buffer_accesses().filter(|b| writes(b.access)) {
            let mut others = draws.clone().enumerate().filter(|&(j, _)| j != i);
            let conflicts = |b: &BufferAccess| b.overlaps(&write) && !(b.unordered && write.unordered);
            if let Some((j, _)) = others.find(|(_, d)| d.buffer_accesses().any(|b| conflicts(&b))) {
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
    #[track_caller]
    fn with(
        bindings: &'a BindingOutput<'a, ()>,
        command: DrawCommand<'a>,
        extra: [BufferAccess; E],
        state: RasterState<N>,
    ) -> Self {
        let (state, blends, write_masks) = state.split();
        state.check(&blends, command.is_vertex());
        Self::from_parts(
            bindings,
            command,
            state,
            None,
            extra,
            blends,
            write_masks,
        )
    }

    /// Counts the samples that pass the depth and stencil tests into query `index` of `pool`.
    pub fn count_samples(mut self, pool: &'a QueryPool<Occlusion>, index: u32) -> Self {
        self.samples = Some((pool, index));
        self
    }

    /// Starts instance indices at `first` (`SV_StartInstanceLocation`; Slang's `SV_InstanceID`
    /// doesn't include it, `SV_VulkanInstanceID` does). Panics unless the draw is direct
    /// (`new`, `indexed`, `multi`, `multi_indexed`): indirect commands carry their own.
    #[track_caller]
    pub fn first_instance(mut self, first: u32) -> Self {
        match &mut self.command {
            DrawCommand::Draw { first_instance, .. }
            | DrawCommand::Indexed { first_instance, .. }
            | DrawCommand::Multi { first_instance, .. }
            | DrawCommand::MultiIndexed { first_instance, .. } => *first_instance = first,
            other => panic!("`first_instance` is for direct draws, not {}", other.name()),
        }
        self
    }

    /// Starts vertex indices at `first` (`SV_StartVertexLocation`; Slang's `SV_VertexID`
    /// doesn't include it, `SV_VulkanVertexID` does). Panics unless the draw is `Draw::new`.
    #[track_caller]
    pub fn first_vertex(mut self, first: u32) -> Self {
        match &mut self.command {
            DrawCommand::Draw { first_vertex, .. } => *first_vertex = first,
            other => panic!("`first_vertex` is for `Draw::new`, not {}", other.name()),
        }
        self
    }

    /// Adds `offset` to every index before it picks a vertex. Panics unless the draw is
    /// `Draw::indexed` (a multi-draw's offsets are per draw, in `MultiDrawIndexed`).
    #[track_caller]
    pub fn vertex_offset(mut self, offset: i32) -> Self {
        match &mut self.command {
            DrawCommand::Indexed { vertex_offset, .. } => *vertex_offset = offset,
            other => panic!("`vertex_offset` is for `Draw::indexed`, not {}", other.name()),
        }
        self
    }
}

/// The index buffer of an indexed draw.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Indices {
    range: vk::DeviceAddressRangeKHR,
    flags: vk::AddressCommandFlagsKHR,
    ty: vk::IndexType,
}

impl Indices {
    fn of<I: IndexFormat>(buffer: BufferSlice<'_, I>) -> (Self, BufferAccess) {
        let access = buffer.access(
            vk::PipelineStageFlags2::INDEX_INPUT,
            vk::AccessFlags2::INDEX_READ,
        );
        let indices = Self {
            range: buffer.address_range(),
            flags: buffer.address_flags(),
            ty: I::TYPE,
        };
        (indices, access)
    }

    unsafe fn bind(&self, cmd: vk::CommandBuffer) {
        unsafe {
            Functions::device_address_commands().cmd_bind_index_buffer3(
                cmd,
                &vk::BindIndexBuffer3InfoKHR::default()
                    .address_range(self.range)
                    .address_flags(self.flags)
                    .index_type(self.ty),
            )
        };
    }
}

/// The commands of an indirect draw in `buffer`, `T`'s stride apart.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Commands {
    range: vk::StridedDeviceAddressRangeKHR,
    flags: vk::AddressCommandFlagsKHR,
    count: u32,
}

impl Commands {
    fn of<C: Pod, T: StartsWith<C>>(buffer: BufferSlice<'_, T>) -> (Self, BufferAccess) {
        const {
            assert!(
                size_of::<T>() >= size_of::<C>() && size_of::<T>() % 4 == 0,
                "an indirect command's stride holds the command and is a multiple of 4 bytes"
            )
        };
        let commands = Self {
            range: buffer.strided_range(),
            flags: buffer.address_flags(),
            count: buffer.len() as u32,
        };
        (commands, indirect_access(buffer))
    }

    fn info(&self) -> vk::DrawIndirect2InfoKHR<'static> {
        vk::DrawIndirect2InfoKHR::default()
            .address_range(self.range)
            .address_flags(self.flags)
            .draw_count(self.count)
    }

    fn count_info(&self, count: &Count) -> vk::DrawIndirectCount2InfoKHR<'static> {
        vk::DrawIndirectCount2InfoKHR::default()
            .address_range(self.range)
            .address_flags(self.flags)
            .count_address_range(count.range)
            .count_address_flags(count.flags)
            .max_draw_count(self.count)
    }
}

/// The buffer an indirect draw reads its draw count from.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Count {
    range: vk::DeviceAddressRangeKHR,
    flags: vk::AddressCommandFlagsKHR,
}

impl Count {
    fn of(buffer: BufferSlice<'_, u32>) -> (Self, BufferAccess) {
        let count = Self {
            range: buffer.address_range(),
            flags: buffer.address_flags(),
        };
        (count, indirect_access(buffer))
    }
}

impl<'a, const N: usize> Draw<'a, N, 0> {
    /// Draws `vertices` vertices of `instances` instances ([`Self::first_vertex`] and
    /// [`Self::first_instance`] start elsewhere).
    #[track_caller]
    pub fn new(
        bindings: &'a BindingOutput<'a, kind::RasterVertex>,
        vertices: u32,
        instances: u32,
        state: RasterState<N>,
    ) -> Self {
        let command = DrawCommand::Draw {
            vertices,
            instances,
            first_vertex: 0,
            first_instance: 0,
        };
        Self::with(
            bindings.erase(),
            command,
            [],
            state,
        )
    }

    /// Draws each of `draws` (a range of vertices) with `instances` instances in one command
    /// (`VK_EXT_multi_draw`), at most `maxMultiDrawCount`.
    #[cfg(feature = "multi-draw")]
    #[track_caller]
    pub fn multi(
        bindings: &'a BindingOutput<'a, kind::RasterVertex>,
        draws: &'a [MultiDraw],
        instances: u32,
        state: RasterState<N>,
    ) -> Gated<gate::MultiDraw, Self> {
        crate::features::gate::<gate::MultiDraw, _>(|| {
            check_multi_draws(draws.len());
            let command = DrawCommand::Multi {
                draws,
                instances,
                first_instance: 0,
            };
            Self::with(
                bindings.erase(),
                command,
                [],
                state,
            )
        })
    }

    /// Launches `groups` task (or, without a task shader, mesh) workgroups.
    #[cfg(feature = "mesh-shader")]
    #[track_caller]
    pub fn launch(
        bindings: &'a BindingOutput<'a, kind::RasterMesh>,
        groups: [u32; 3],
        state: RasterState<N>,
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
            DrawCommand::Launch(groups),
            [],
            state,
        )
    }
}

impl<'a, const N: usize> Draw<'a, N, 1> {
    /// Draws the vertices `index_buffer` names (`u16` or `u32` indices), `instances` times
    /// ([`Self::vertex_offset`] and [`Self::first_instance`] shift them).
    #[track_caller]
    pub fn indexed<I: IndexFormat>(
        bindings: &'a BindingOutput<'a, kind::RasterVertex>,
        index_buffer: BufferSlice<'a, I>,
        instances: u32,
        state: RasterState<N>,
    ) -> Self {
        let (indices, index_access) = Indices::of(index_buffer);
        let command = DrawCommand::Indexed {
            indices,
            count: index_buffer.len() as u32,
            instances,
            vertex_offset: 0,
            first_instance: 0,
        };
        Self::with(
            bindings.erase(),
            command,
            [index_access],
            state,
        )
    }

    /// Draws each of `draws` (a range of `index_buffer` and a vertex offset) with `instances`
    /// instances in one command (`VK_EXT_multi_draw`), at most `maxMultiDrawCount`. Panics for
    /// a draw outside the index buffer.
    #[cfg(feature = "multi-draw")]
    #[track_caller]
    pub fn multi_indexed<I: IndexFormat>(
        bindings: &'a BindingOutput<'a, kind::RasterVertex>,
        index_buffer: BufferSlice<'a, I>,
        draws: &'a [MultiDrawIndexed],
        instances: u32,
        state: RasterState<N>,
    ) -> Gated<gate::MultiDraw, Self> {
        crate::features::gate::<gate::MultiDraw, _>(|| {
            check_multi_draws(draws.len());
            let len = index_buffer.len() as u64;
            for (i, draw) in draws.iter().enumerate() {
                assert!(
                    draw.first_index as u64 + draw.index_count as u64 <= len,
                    "draw {i} reads indices {}..{} of an index buffer of {len}",
                    draw.first_index,
                    draw.first_index as u64 + draw.index_count as u64
                );
            }
            let (indices, index_access) = Indices::of(index_buffer);
            let command = DrawCommand::MultiIndexed {
                indices,
                draws,
                instances,
                first_instance: 0,
            };
            Self::with(
                bindings.erase(),
                command,
                [index_access],
                state,
            )
        })
    }

    /// Draws every command of `buffer`: a slice of commands, or of any `T` that starts with one
    /// ([`StartsWith`]), `T`'s size apart.
    #[track_caller]
    pub fn indirect<T: StartsWith<DrawIndirectCommand>>(
        bindings: &'a BindingOutput<'a, kind::RasterVertex>,
        buffer: BufferSlice<'a, T>,
        state: RasterState<N>,
    ) -> Self {
        let (commands, access) = Commands::of::<DrawIndirectCommand, T>(buffer);
        Self::with(
            bindings.erase(),
            DrawCommand::Indirect(commands),
            [access],
            state,
        )
    }

    /// Launches the task (or, without a task shader, mesh) workgroup counts of every command in
    /// `buffer` (any `T` that starts with one, like [`Self::indirect`]).
    #[cfg(feature = "mesh-shader")]
    #[track_caller]
    pub fn launch_indirect<T: StartsWith<DispatchIndirectCommand>>(
        bindings: &'a BindingOutput<'a, kind::RasterMesh>,
        buffer: BufferSlice<'a, T>,
        state: RasterState<N>,
    ) -> Self {
        let (commands, access) = Commands::of::<DispatchIndirectCommand, T>(buffer);
        Self::with(
            bindings.erase(),
            DrawCommand::LaunchIndirect(commands),
            [access],
            state,
        )
    }
}

impl<'a, const N: usize> Draw<'a, N, 2> {
    /// Draws the first `count_buffer[0]` commands of `buffer`, at most all of them.
    #[track_caller]
    pub fn indirect_count<T: StartsWith<DrawIndirectCommand>>(
        bindings: &'a BindingOutput<'a, kind::RasterVertex>,
        buffer: BufferSlice<'a, T>,
        count_buffer: BufferSlice<'a, u32>,
        state: RasterState<N>,
    ) -> Self {
        let (commands, access) = Commands::of::<DrawIndirectCommand, T>(buffer);
        let (count, count_access) = Count::of(count_buffer);
        Self::with(
            bindings.erase(),
            DrawCommand::IndirectCount(commands, count),
            [access, count_access],
            state,
        )
    }

    /// Draws every indexed command of `buffer` with the indices of `index_buffer`.
    #[track_caller]
    pub fn indexed_indirect<I: IndexFormat, T: StartsWith<DrawIndexedIndirectCommand>>(
        bindings: &'a BindingOutput<'a, kind::RasterVertex>,
        index_buffer: BufferSlice<'a, I>,
        buffer: BufferSlice<'a, T>,
        state: RasterState<N>,
    ) -> Self {
        let (indices, index_access) = Indices::of(index_buffer);
        let (commands, access) = Commands::of::<DrawIndexedIndirectCommand, T>(buffer);
        Self::with(
            bindings.erase(),
            DrawCommand::IndexedIndirect(indices, commands),
            [index_access, access],
            state,
        )
    }

    /// Launches the first `count_buffer[0]` commands of `buffer`, at most all of them.
    #[cfg(feature = "mesh-shader")]
    #[track_caller]
    pub fn launch_indirect_count<T: StartsWith<DispatchIndirectCommand>>(
        bindings: &'a BindingOutput<'a, kind::RasterMesh>,
        buffer: BufferSlice<'a, T>,
        count_buffer: BufferSlice<'a, u32>,
        state: RasterState<N>,
    ) -> Self {
        let (commands, access) = Commands::of::<DispatchIndirectCommand, T>(buffer);
        let (count, count_access) = Count::of(count_buffer);
        Self::with(
            bindings.erase(),
            DrawCommand::LaunchIndirectCount(commands, count),
            [access, count_access],
            state,
        )
    }
}

impl<'a, const N: usize> Draw<'a, N, 3> {
    /// Draws the first `count_buffer[0]` indexed commands of `buffer`, at most all of them.
    #[track_caller]
    pub fn indexed_indirect_count<I: IndexFormat, T: StartsWith<DrawIndexedIndirectCommand>>(
        bindings: &'a BindingOutput<'a, kind::RasterVertex>,
        index_buffer: BufferSlice<'a, I>,
        buffer: BufferSlice<'a, T>,
        count_buffer: BufferSlice<'a, u32>,
        state: RasterState<N>,
    ) -> Self {
        let (indices, index_access) = Indices::of(index_buffer);
        let (commands, access) = Commands::of::<DrawIndexedIndirectCommand, T>(buffer);
        let (count, count_access) = Count::of(count_buffer);
        Self::with(
            bindings.erase(),
            DrawCommand::IndexedIndirectCount(indices, commands, count),
            [index_access, access, count_access],
            state,
        )
    }
}

#[cfg(feature = "multi-draw")]
#[track_caller]
fn check_multi_draws(count: usize) {
    // Multi-draw bindings exist only past the feature's check.
    let max = Ctx::physical_device().max_multi_draw_count;
    assert!(
        count <= max as usize,
        "{count} draws are more than the device's {max} per multi-draw (`maxMultiDrawCount`)"
    );
}

#[derive(Debug)]
pub(crate) enum DrawCommand<'a> {
    Draw {
        vertices: u32,
        instances: u32,
        first_vertex: u32,
        first_instance: u32,
    },
    Indexed {
        indices: Indices,
        count: u32,
        instances: u32,
        vertex_offset: i32,
        first_instance: u32,
    },
    Indirect(Commands),
    IndirectCount(Commands, Count),
    IndexedIndirect(Indices, Commands),
    IndexedIndirectCount(Indices, Commands, Count),
    // Made only with `multi-draw`.
    #[cfg_attr(not(feature = "multi-draw"), allow(dead_code))]
    Multi {
        draws: &'a [MultiDraw],
        instances: u32,
        first_instance: u32,
    },
    #[cfg_attr(not(feature = "multi-draw"), allow(dead_code))]
    MultiIndexed {
        indices: Indices,
        draws: &'a [MultiDrawIndexed],
        instances: u32,
        first_instance: u32,
    },
    #[cfg(feature = "mesh-shader")]
    Launch([u32; 3]),
    #[cfg(feature = "mesh-shader")]
    LaunchIndirect(Commands),
    #[cfg(feature = "mesh-shader")]
    LaunchIndirectCount(Commands, Count),
}

impl DrawCommand<'_> {
    fn is_vertex(&self) -> bool {
        match self {
            #[cfg(feature = "mesh-shader")]
            DrawCommand::Launch(_)
            | DrawCommand::LaunchIndirect(_)
            | DrawCommand::LaunchIndirectCount(..) => false,
            _ => true,
        }
    }

    /// Draws nothing: the validation layer warns about direct draws of nothing, and an empty
    /// index range can't be bound without `nullDescriptor`. Their queries count 0.
    fn is_empty(&self) -> bool {
        match self {
            DrawCommand::Draw { instances, .. } => *instances == 0,
            DrawCommand::Indexed {
                instances, count, ..
            } => *instances == 0 || *count == 0,
            DrawCommand::Multi {
                draws, instances, ..
            } => *instances == 0 || draws.is_empty(),
            DrawCommand::MultiIndexed {
                draws, instances, ..
            } => *instances == 0 || draws.is_empty(),
            _ => false,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            DrawCommand::Draw { .. } => "`Draw::new`",
            DrawCommand::Indexed { .. } => "`Draw::indexed`",
            DrawCommand::Indirect(_) => "`Draw::indirect`",
            DrawCommand::IndirectCount(..) => "`Draw::indirect_count`",
            DrawCommand::IndexedIndirect(..) => "`Draw::indexed_indirect`",
            DrawCommand::IndexedIndirectCount(..) => "`Draw::indexed_indirect_count`",
            DrawCommand::Multi { .. } => "`Draw::multi`",
            DrawCommand::MultiIndexed { .. } => "`Draw::multi_indexed`",
            #[cfg(feature = "mesh-shader")]
            DrawCommand::Launch(_) => "`Draw::launch`",
            #[cfg(feature = "mesh-shader")]
            DrawCommand::LaunchIndirect(_) => "`Draw::launch_indirect`",
            #[cfg(feature = "mesh-shader")]
            DrawCommand::LaunchIndirectCount(..) => "`Draw::launch_indirect_count`",
        }
    }

    /// Records the draw; its pipeline, push constants and state are recorded.
    unsafe fn record(&self, cmd: vk::CommandBuffer) {
        let device = Ctx::device();
        let address = Functions::device_address_commands();
        unsafe {
            match *self {
                DrawCommand::Draw {
                    vertices,
                    instances,
                    first_vertex,
                    first_instance,
                } => device.cmd_draw(cmd, vertices, instances, first_vertex, first_instance),
                DrawCommand::Indexed {
                    indices,
                    count,
                    instances,
                    vertex_offset,
                    first_instance,
                } => {
                    indices.bind(cmd);
                    device.cmd_draw_indexed(cmd, count, instances, 0, vertex_offset, first_instance);
                }
                DrawCommand::Indirect(commands) => address.cmd_draw_indirect2(cmd, &commands.info()),
                DrawCommand::IndirectCount(commands, count) => {
                    address.cmd_draw_indirect_count2(cmd, &commands.count_info(&count))
                }
                DrawCommand::IndexedIndirect(indices, commands) => {
                    indices.bind(cmd);
                    address.cmd_draw_indexed_indirect2(cmd, &commands.info());
                }
                DrawCommand::IndexedIndirectCount(indices, commands, count) => {
                    indices.bind(cmd);
                    address.cmd_draw_indexed_indirect_count2(cmd, &commands.count_info(&count));
                }
                #[cfg(feature = "multi-draw")]
                DrawCommand::Multi {
                    draws,
                    instances,
                    first_instance,
                } => {
                    // `MultiDraw` has `VkMultiDrawInfoEXT`'s layout.
                    (Functions::multi_draw().fp().cmd_draw_multi_ext)(
                        cmd,
                        draws.len() as u32,
                        draws.as_ptr().cast(),
                        instances,
                        first_instance,
                        size_of::<MultiDraw>() as u32,
                    );
                }
                #[cfg(feature = "multi-draw")]
                DrawCommand::MultiIndexed {
                    indices,
                    draws,
                    instances,
                    first_instance,
                } => {
                    indices.bind(cmd);
                    // `MultiDrawIndexed` has `VkMultiDrawIndexedInfoEXT`'s layout; each draw
                    // has its own vertex offset.
                    (Functions::multi_draw().fp().cmd_draw_multi_indexed_ext)(
                        cmd,
                        draws.len() as u32,
                        draws.as_ptr().cast(),
                        instances,
                        first_instance,
                        size_of::<MultiDrawIndexed>() as u32,
                        std::ptr::null(),
                    );
                }
                #[cfg(not(feature = "multi-draw"))]
                DrawCommand::Multi { .. } | DrawCommand::MultiIndexed { .. } => {
                    unreachable!("multi-draws are made only with `multi-draw`")
                }
                #[cfg(feature = "mesh-shader")]
                DrawCommand::Launch([x, y, z]) => Functions::mesh().cmd_draw_mesh_tasks(cmd, x, y, z),
                #[cfg(feature = "mesh-shader")]
                DrawCommand::LaunchIndirect(commands) => {
                    address.cmd_draw_mesh_tasks_indirect2(cmd, &commands.info())
                }
                #[cfg(feature = "mesh-shader")]
                DrawCommand::LaunchIndirectCount(commands, count) => {
                    address.cmd_draw_mesh_tasks_indirect_count2(cmd, &commands.count_info(&count))
                }
            }
        }
    }
}

/// A type whose first bytes are an indirect command `C`, so a slice of it is a buffer of
/// commands `size_of::<Self>()` apart: indirect draws and dispatches take a slice of any
/// `T: StartsWith<C>`, like a culling pass's draw records with the command first.
///
/// # Safety
/// `Self` is `#[repr(C)]` (or `#[repr(transparent)]` over such a type) and holds a `C` at offset
/// 0. Its size being a multiple of 4 and at least `C`'s is checked at compile time.
pub unsafe trait StartsWith<C: Pod>: NoUninit {}

/// `VkDrawIndirectCommand`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
#[repr(C)]
pub struct DrawIndirectCommand {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}
// SAFETY: every type starts with itself.
unsafe impl StartsWith<DrawIndirectCommand> for DrawIndirectCommand {}

/// `VkDrawIndexedIndirectCommand`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
#[repr(C)]
pub struct DrawIndexedIndirectCommand {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub first_instance: u32,
}
// SAFETY: every type starts with itself.
unsafe impl StartsWith<DrawIndexedIndirectCommand> for DrawIndexedIndirectCommand {}

/// `VkDispatchIndirectCommand`, also the workgroup counts of an indirect mesh launch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
#[repr(C)]
pub struct DispatchIndirectCommand {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}
// SAFETY: every type starts with itself.
unsafe impl StartsWith<DispatchIndirectCommand> for DispatchIndirectCommand {}

/// The rays of an indirect trace (`VkTraceRaysIndirectCommandKHR`).
#[cfg(feature = "ray-tracing-pipeline")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
#[repr(C)]
pub struct TraceRaysIndirectCommand {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}
// SAFETY: every type starts with itself.
#[cfg(feature = "ray-tracing-pipeline")]
unsafe impl StartsWith<TraceRaysIndirectCommand> for TraceRaysIndirectCommand {}

/// One draw of [`Draw::multi`]: `vertex_count` vertices from `first_vertex` (`VkMultiDrawInfoEXT`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
#[repr(C)]
pub struct MultiDraw {
    pub first_vertex: u32,
    pub vertex_count: u32,
}

/// One draw of [`Draw::multi_indexed`]: `index_count` indices from `first_index`, each plus
/// `vertex_offset` (`VkMultiDrawIndexedInfoEXT`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
#[repr(C)]
pub struct MultiDrawIndexed {
    pub first_index: u32,
    pub index_count: u32,
    pub vertex_offset: i32,
}

/// An index type: draws (`Draw::indexed`) and triangle geometries take `u16` and `u32` indices.
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

impl<'a, const N: usize> RasterBuilder<'a, '_, N> {
    /// Depth-tests the draws against `image`, a view of one mip (nearer = greater), and for a
    /// combined depth-stencil format stencil-tests them too. `clear` resets it first. A
    /// rendering has one depth-stencil image: this replaces an earlier `depth_attachment` or
    /// `stencil_attachment`.
    #[track_caller]
    pub fn depth_attachment<T: DepthTarget + 'a>(
        &mut self,
        image: T,
        clear: Option<<T::Format as Format>::Texels>,
    ) -> &mut Self {
        let image = image.view();
        image.aspect_check(vk::ImageAspectFlags::DEPTH);
        let stencil = image.format().aspects().contains(vk::ImageAspectFlags::STENCIL);
        let clear = clear.map(|e| image.format().clear_value(e).into());
        self.depth_stencil(image, clear, true, stencil, "depth")
    }

    /// Stencil-tests the draws against `image`, a view of one mip of a stencil-only format
    /// (`S8Uint`), or of a combined one whose depth the rendering leaves alone. `clear` resets
    /// it first. A rendering has one depth-stencil image: this replaces an earlier
    /// `depth_attachment` (a combined format's stencil comes with that) or `stencil_attachment`.
    #[track_caller]
    pub fn stencil_attachment<T: StencilTarget + 'a>(
        &mut self,
        image: T,
        clear: Option<<T::Format as Format>::Texels>,
    ) -> &mut Self {
        let image = image.view();
        image.aspect_check(vk::ImageAspectFlags::STENCIL);
        let clear = clear.map(|e| image.format().clear_value(e).into());
        self.depth_stencil(image, clear, false, true, "stencil")
    }

    #[track_caller]
    fn depth_stencil<F: Format, U: crate::image::usage::ImageUsage, D: Dim2D>(
        &mut self,
        image: ImageView<'a, F, U, D>,
        clear: Option<vk::ClearValue>,
        depth: bool,
        stencil: bool,
        what: &str,
    ) -> &mut Self {
        image.usage_check(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT);
        one_mip(image.mip_range);
        fits(self.area, image.extent, what);
        self.depth_stencil = Some(DepthStencil {
            view: image.vk_view(),
            format: image.format().vk_format(),
            depth,
            stencil,
            layers: image.layer_count(),
            access: image.access_in(
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
                vk::ImageLayout::ATTACHMENT_OPTIMAL,
            ),
            clear,
            discard: Discard::default(),
        });
        self
    }

    /// Doesn't load the depth-stencil attachment (`DONT_CARE`), like
    /// [`Attachment::discard_before`]. Panics without one or for a cleared one.
    #[track_caller]
    pub fn discard_depth_before(&mut self) -> &mut Self {
        let ds = (self.depth_stencil.as_mut())
            .expect("`discard_depth_before` needs a `depth_attachment` or `stencil_attachment`");
        ds.discard.before(ds.clear);
        self
    }

    /// Doesn't store the depth-stencil attachment (`DONT_CARE`), like
    /// [`Attachment::discard_after`]: for a depth buffer only this rendering tests against.
    /// Panics without one.
    #[track_caller]
    pub fn discard_depth_after(&mut self) -> &mut Self {
        let ds = (self.depth_stencil.as_mut())
            .expect("`discard_depth_after` needs a `depth_attachment` or `stencil_attachment`");
        ds.discard.after = true;
        self
    }

    /// Replaces the default viewport of the rendering's draws, its area; a draw's own
    /// (`RasterState::viewport`) replaces it for that draw.
    #[track_caller]
    pub fn viewport(&mut self, viewport: Rect) -> &mut Self {
        viewport.check_viewport();
        self.viewport = viewport;
        self
    }

    /// Replaces the default scissor of the rendering's draws, its area; a draw's own
    /// (`RasterState::scissor`) replaces it for that draw.
    #[track_caller]
    pub fn scissor(&mut self, scissor: Rect) -> &mut Self {
        scissor.check_scissor("scissor");
        self.scissor = scissor;
        self
    }

    /// Draws into `count` layers of every attachment (a layered rendering): each primitive goes
    /// to the layer its vertex or mesh shader picks (`SV_RenderTargetArrayIndex`; vertex
    /// shaders need `shader-output-layer`), so every attachment is an array view of at least
    /// `count` layers. At most `maxFramebufferLayers`; not with a [`Self::view_mask`].
    #[track_caller]
    pub fn layers(&mut self, count: u32) -> &mut Self {
        let max = Ctx::device_properties().limits.max_framebuffer_layers;
        assert!(
            (1..=max).contains(&count),
            "a rendering draws into 1 to {max} layers (`maxFramebufferLayers`), not {count}"
        );
        self.layer_count = count;
        self
    }

    /// Draws every draw once per set bit of `mask` (multiview), into the layer of the bit's
    /// index; shaders read the index as `SV_ViewID`. Every attachment is an array view with a
    /// layer for the highest bit. At most `maxMultiviewViewCount` views; mesh passes need the
    /// device's `multiviewMeshShader`. Part of each pipeline's key.
    #[track_caller]
    pub fn view_mask(&mut self, mask: u32) -> &mut Self {
        let max = Ctx::physical_device().vk11_properties.max_multiview_view_count;
        assert!(
            mask != 0 && 32 - mask.leading_zeros() <= max,
            "a view mask has views 0 to {} (`maxMultiviewViewCount`), not {mask:#b}",
            max.saturating_sub(1)
        );
        self.view_mask = mask;
        self
    }

    pub fn draw<const E: usize>(&mut self, draw: &'a Draw<'a, N, E>) -> &mut Self {
        self.draws.push(draw.deref());
        self
    }

    /// The layers every attachment needs: the highest view's with a view mask, else the
    /// rendering's layer count.
    #[track_caller]
    fn check_layers(&self) {
        assert!(
            self.view_mask == 0 || self.layer_count == 1,
            "a multiview rendering (`view_mask`) isn't layered (`layers`) too"
        );
        let needed = match self.view_mask {
            0 => self.layer_count,
            mask => 32 - mask.leading_zeros(),
        };
        let layers = (self.color.iter().map(|c| c.layers)).chain(self.depth_stencil.as_ref().map(|d| d.layers));
        for layers in layers {
            assert!(
                layers >= needed,
                "an attachment of {layers} layers in a rendering that draws into {needed}: give \
                 it an array view (`ImageView::layers`) of as many"
            );
        }
        if self.view_mask != 0 && !Ctx::physical_device().multiview_mesh_shader {
            assert!(
                self.draws.iter().all(|d| d.is_vertex()),
                "the device draws no mesh passes in multiview renderings (`multiviewMeshShader`)"
            );
        }
    }

    /// Panics if a draw blends into an attachment whose format can't blend
    /// (`COLOR_ATTACHMENT_BLEND`).
    #[track_caller]
    fn check_blends(&self) {
        let blending = |i: usize| {
            self.draws.iter().any(|d| d.blends()[i].equation().is_some())
        };
        for (i, color) in self.color.iter().enumerate() {
            if blending(i) {
                let features = Ctx::format_features(color.format);
                assert!(
                    features.contains(vk::FormatFeatureFlags2::COLOR_ATTACHMENT_BLEND),
                    "a draw blends into color attachment {i}, whose format {:?} doesn't blend on \
                     this device: draw it with `Blend::Replace`",
                    color.format
                );
            }
        }
    }

    /// Records the rendering under the GPU scope `name`, its draws in order. Only depth tests,
    /// depth writes and blending see earlier draws, so no draw may read what an earlier one writes.
    #[validation_trace]
    pub fn record(&mut self, name: &'static str) {
        let _span = tracing::info_span!("draw", pass = name).entered();
        self.check_layers();
        self.check_blends();
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
            if self.draws.iter().any(|draw| draw.image_accesses().any(overlaps)) {
                access.access.layout = vk::ImageLayout::GENERAL;
            }
            access
        };
        let color_accesses = self.color.each_ref().map(|c| in_layout(c.access.clone()));
        let ds_access = self.depth_stencil.as_ref().map(|ds| {
            let mut access = ds.access.clone();
            let writes = |d: &DrawOutput| {
                (ds.depth && d.state.depth_test.is_some() && d.state.depth_write)
                    || (ds.stencil && d.state.stencil.is_some())
            };
            // `DONT_CARE` loads and stores write too.
            let discards = ds.discard.before || ds.discard.after;
            if ds.clear.is_some() || discards || self.draws.iter().any(writes) {
                access.access.access |= vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE;
            }
            in_layout(access)
        });
        let draws = self.draws.iter();
        check_hazards(name, draws.clone());
        self.cmd_buf.flush_pending(
            draws
                .clone()
                .flat_map(|draw| draw.image_accesses())
                .chain(color_accesses.iter().cloned())
                .chain(ds_access.clone()),
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
            .layer_count(self.layer_count)
            .render_area(self.area.vk())
            .view_mask(self.view_mask);

        let ds_info;
        let (mut depth_format, mut stencil_format) = (vk::Format::UNDEFINED, vk::Format::UNDEFINED);
        if let (Some(ds), Some(access)) = (&self.depth_stencil, &ds_access) {
            let written = (access.access.access)
                .contains(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE);
            let store_op = match (ds.discard.after, written) {
                (true, _) => vk::AttachmentStoreOp::DONT_CARE,
                (false, true) => vk::AttachmentStoreOp::STORE,
                (false, false) => vk::AttachmentStoreOp::NONE,
            };
            ds_info = vk::RenderingAttachmentInfo::default()
                .image_layout(access.access.layout)
                .image_view(ds.view)
                .load_op(ds.discard.load_op(ds.clear))
                .store_op(store_op)
                .clear_value(ds.clear.unwrap_or_default());
            if ds.depth {
                rendering_info = rendering_info.depth_attachment(&ds_info);
                depth_format = ds.format;
            }
            if ds.stencil {
                rendering_info = rendering_info.stencil_attachment(&ds_info);
                stencil_format = ds.format;
            }
        }

        let device = Ctx::device();
        let handle = self.cmd_buf.handle;
        self.cmd_buf.enter(name, true);
        unsafe {
            device.cmd_begin_rendering(handle, &rendering_info);
        }
        self.cmd_buf.in_rendering = true;
        let color_formats = self.color.each_ref().map(|c| c.format);
        let rendering = Rendering {
            color_attachments: N,
            depth: depth_format != vk::Format::UNDEFINED,
            stencil: stencil_format != vk::Format::UNDEFINED,
            viewport: self.viewport,
            scissor: self.scissor,
        };
        let dynamic = *Ctx::dynamic();
        let mut state: Option<DrawState> = None;
        let mut assembly: Assembly = None;
        let mut bound: Option<(usize, RasterKeyRef)> = None;
        for draw in self.draws.iter() {
            if let Some((pool, query)) = draw.samples {
                self.cmd_buf.begin_query(pool, query);
            }
            if !draw.command.is_empty() {
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
                let vertex = draw.is_vertex();
                let key = RasterKeyRef {
                    color_formats: &color_formats,
                    depth_format,
                    stencil_format,
                    view_mask: self.view_mask,
                    constants: sections.constants,
                    baked: Baked::of(draw_state.state, &dynamic, vertex),
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
                draw_state.record(
                    state.as_ref(),
                    rendering,
                    vertex,
                    &mut assembly,
                    &dynamic,
                    handle,
                );
                state = Some(draw_state);
                unsafe { draw.command.record(handle) };
            }
            if let Some((pool, query)) = draw.samples {
                self.cmd_buf.end_query(pool, query);
            }
        }
        unsafe { device.cmd_end_rendering(handle) };
        self.cmd_buf.in_rendering = false;
        self.cmd_buf.exit(name, true);
        self.draws.draws.clear();
    }
}

/// Panics unless the render area lies inside an attachment of `attachment` texels.
#[track_caller]
fn fits(area: Rect, attachment: UVec2, what: &str) {
    let end = area.offset.as_i64vec2() + area.extent.as_i64vec2();
    assert!(
        end.cmple(attachment.as_i64vec2()).all(),
        "a rendering of {} texels at {} doesn't fit its {what} attachment of {attachment}",
        area.extent,
        area.offset
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

/// How a compute dispatch gets its workgroups.
enum Dispatch {
    Groups([u32; 3]),
    /// The first workgroup's ID, and the counts.
    Base([u32; 3], [u32; 3]),
    Indirect(vk::DeviceAddressRangeKHR, BufferAccess),
}

/// The address range of the first command of `buffer` an indirect dispatch or trace reads, and
/// its access.
#[track_caller]
fn first_command<C: Pod, T: StartsWith<C>>(
    buffer: BufferSlice<'_, T>,
) -> (vk::DeviceAddressRangeKHR, BufferAccess) {
    assert!(
        !buffer.is_empty(),
        "an indirect dispatch or trace reads its command from an empty slice"
    );
    const {
        assert!(
            size_of::<T>() >= size_of::<C>() && size_of::<T>() % 4 == 0,
            "an indirect command's stride holds the command and is a multiple of 4 bytes"
        )
    };
    let range = buffer.address_range();
    let first = vk::DeviceAddressRangeKHR {
        address: range.address,
        size: range.size.min(size_of::<C>() as u64),
    };
    (first, indirect_access(buffer))
}

fn address_flags() -> vk::AddressCommandFlagsKHR {
    vk::AddressCommandFlagsKHR::STORAGE_BUFFER_USAGE | vk::AddressCommandFlagsKHR::FULLY_BOUND
}

fn indirect_access<T: Copy + 'static>(buffer: BufferSlice<'_, T>) -> BufferAccess {
    buffer.access(
        vk::PipelineStageFlags2::DRAW_INDIRECT,
        vk::AccessFlags2::INDIRECT_COMMAND_READ,
    )
}

impl<'p, K: HasGraphics> CommandBuffer<'p, K> {
    /// Starts a rendering of `area` (an extent at the origin, or a [`Rect`]) into the `color`
    /// attachments, with viewport and scissor covering it. Every attachment covers the area.
    #[track_caller]
    pub fn raster<'a, const N: usize>(
        &'a mut self,
        area: impl Into<Rect>,
        color: [Attachment<'a>; N],
    ) -> RasterBuilder<'a, 'p, N> {
        let area = area.into();
        assert!(
            area.extent.cmpgt(UVec2::ZERO).all(),
            "a rendering of extent {} has no pixels",
            area.extent
        );
        area.check_scissor("render area");
        for attachment in &color {
            fits(area, attachment.extent, "color");
        }
        RasterBuilder {
            cmd_buf: &mut self.rec,
            area,
            viewport: area,
            scissor: area,
            layer_count: 1,
            view_mask: 0,
            color,
            depth_stencil: None,
            draws: DrawList {
                draws: SmallVec::new(),
                marker: PhantomData,
            },
        }
    }
}

impl<K: HasCompute> CommandBuffer<'_, K> {
    fn compute_private(
        &mut self,
        bindings: &BindingOutput<'_, kind::Compute>,
        dispatch: Dispatch,
    ) {
        let sections = bindings.sections();
        let pass = &pass_map().passes[sections.pass];
        let _span = tracing::info_span!("compute", pass = pass.name).entered();
        let indirect = match &dispatch {
            Dispatch::Indirect(_, access) => Some(access.clone()),
            _ => None,
        };
        let rec = &mut self.rec;
        rec.flush_pending(sections.images(), sections.buffers().chain(indirect));
        rec.push_constants(sections.push_constants);

        rec.enter(pass.name, true);
        let handle = rec.handle;
        Bound::get(
            &mut rec.bound_compute,
            sections.pass,
            sections.constants,
            || pipelines().compute(sections.pass, sections.constants),
            |a, b| a == b,
            |pipeline| unsafe {
                Ctx::device().cmd_bind_pipeline(handle, vk::PipelineBindPoint::COMPUTE, *pipeline)
            },
        );
        unsafe {
            match dispatch {
                Dispatch::Indirect(range, _) => {
                    Functions::device_address_commands().cmd_dispatch_indirect2(
                        rec.handle,
                        &vk::DispatchIndirect2InfoKHR::default()
                            .address_range(range)
                            .address_flags(address_flags()),
                    )
                }
                Dispatch::Groups([x, y, z]) => Ctx::device().cmd_dispatch(rec.handle, x, y, z),
                Dispatch::Base([bx, by, bz], [x, y, z]) => {
                    Ctx::device().cmd_dispatch_base(rec.handle, bx, by, bz, x, y, z)
                }
            }
        }
        rec.exit(pass.name, true);
    }

    /// Like [`Self::compute`], with the workgroup counts read from the first command of
    /// `buffer` on the GPU (a slice of commands, or of any `T` that starts with one,
    /// [`StartsWith`]).
    #[validation_trace]
    pub fn compute_indirect<T: StartsWith<DispatchIndirectCommand>>(
        &mut self,
        bindings: &BindingOutput<'_, kind::Compute>,
        buffer: BufferSlice<'_, T>,
    ) {
        let (range, access) = first_command::<DispatchIndirectCommand, T>(buffer);
        self.compute_private(bindings, Dispatch::Indirect(range, access));
    }

    /// Dispatches `groups` workgroups of a compute pass (the generated `Pass::groups(threads)`
    /// rounds a thread count up to them).
    #[validation_trace]
    pub fn compute(&mut self, bindings: &BindingOutput<'_, kind::Compute>, groups: [u32; 3]) {
        let max = Ctx::device_properties().limits.max_compute_work_group_count;
        check_groups(groups, max, u64::MAX, "compute workgroups");
        self.compute_private(bindings, Dispatch::Groups(groups));
    }

    /// Dispatches `groups` workgroups whose workgroup IDs start at `base` (`vkCmdDispatchBase`:
    /// `SV_GroupID` and `SV_DispatchThreadID` include it), e.g. to split a large dispatch.
    /// Panics if `base + groups` exceeds the device's `maxComputeWorkGroupCount`.
    #[validation_trace]
    pub fn compute_base(
        &mut self,
        bindings: &BindingOutput<'_, kind::Compute>,
        base: [u32; 3],
        groups: [u32; 3],
    ) {
        let max = Ctx::device_properties().limits.max_compute_work_group_count;
        let end: [u32; 3] = std::array::from_fn(|i| base[i].saturating_add(groups[i]));
        assert!(
            end.iter().zip(max).all(|(&e, m)| e <= m),
            "workgroups {base:?} to {end:?} exceed the device's {max:?}"
        );
        self.compute_private(bindings, Dispatch::Base(base, groups));
    }

    #[cfg(feature = "ray-tracing-pipeline")]
    fn raytrace_private(
        &mut self,
        bindings: &BindingOutput<'_, kind::RayTracing>,
        rays: [u32; 3],
        indirect_buffer: Option<(vk::DeviceAddressRangeKHR, BufferAccess)>,
    ) {
        let sections = bindings.sections();
        let pass = &pass_map().passes[sections.pass];
        let _span = tracing::info_span!("raytrace", pass = pass.name).entered();
        let rec = &mut self.rec;
        rec.flush_pending(
            sections.images(),
            (sections.buffers()).chain(indirect_buffer.as_ref().map(|(_, access)| access.clone())),
        );
        rec.push_constants(sections.push_constants);

        rec.enter(pass.name, true);
        let handle = rec.handle;
        let pipeline = Bound::get(
            &mut rec.bound_ray_tracing,
            sections.pass,
            sections.constants,
            || pipelines().ray_tracing(sections.pass, sections.constants),
            |a, b| a.pipeline == b.pipeline,
            |pipeline| unsafe {
                Ctx::device().cmd_bind_pipeline(
                    handle,
                    vk::PipelineBindPoint::RAY_TRACING_KHR,
                    pipeline.pipeline,
                )
            },
        );
        unsafe {
            // Past the pass's check: `ray_tracing` panics for hand-made bindings without it.
            let functions = Functions::raytracing_pipeline();
            let [raygen, miss, hit, callable] = &pipeline.regions;
            match indirect_buffer {
                Some((range, _)) => functions.cmd_trace_rays_indirect(
                    handle,
                    raygen,
                    miss,
                    hit,
                    callable,
                    range.address,
                ),
                None => functions.cmd_trace_rays(
                    handle, raygen, miss, hit, callable, rays[0], rays[1], rays[2],
                ),
            }
        }
        rec.exit(pass.name, true);
    }

    /// Traces `rays` (width, height, depth) raygen invocations of a pass.
    #[cfg(feature = "ray-tracing-pipeline")]
    #[validation_trace]
    pub fn raytrace(
        &mut self,
        bindings: &BindingOutput<'_, kind::RayTracing>,
        rays: [u32; 3],
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
        self.raytrace_private(bindings, rays, None);
    }

    /// Like [`Self::raytrace`], with the ray counts read from the first command of `buffer`
    /// on the GPU (any `T` that starts with one, like [`Self::compute_indirect`]).
    #[cfg(feature = "ray-tracing-pipeline")]
    #[validation_trace]
    pub fn raytrace_indirect<T: StartsWith<TraceRaysIndirectCommand>>(
        &mut self,
        bindings: &BindingOutput<'_, kind::RayTracing>,
        buffer: BufferSlice<'_, T>,
    ) {
        let first = first_command::<TraceRaysIndirectCommand, T>(buffer);
        self.raytrace_private(bindings, [0, 0, 0], Some(first));
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
        observer: Option<&mut dyn ScopeObserver<K>>,
        name: &'static str,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let Some(observer) = observer else {
            return self.scope(name, f);
        };
        assert!(self.rec.observer.is_none(), "`measure` doesn't nest");
        let mut typed = Typed(observer);
        // SAFETY: only the lifetime is erased. The pointer is cleared before `typed` and the
        // observer's borrow end, also when `f` panics, and until then only this command buffer
        // reaches it.
        let observer = unsafe {
            std::mem::transmute::<NonNull<dyn Observer + '_>, NonNull<dyn Observer>>(
                NonNull::from(&mut typed as &mut dyn Observer),
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

    /// Records `f` inside query `index` of `pool`, a kind this family records ([`RecordedOn`]);
    /// one query of a kind can be active at a time. A performance query's pool has to exist
    /// before the recording begins.
    pub fn query<Q: Scoped + RecordedOn<K>, R>(
        &mut self,
        pool: &QueryPool<Q>,
        index: u32,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
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
        self.check_unpresented(image);
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
            open_queries: [None; 3],
            performance_pool: None,
            profile: None,
            profiling: false,
            in_rendering: false,
            barrier_log,
            acquired: SmallVec::new(),
            presents: SmallVec::new(),
            staging,
            bound_compute: None,
            #[cfg(feature = "ray-tracing-pipeline")]
            bound_ray_tracing: None,
            #[cfg(feature = "acceleration-structure")]
            structure_builds: Vec::new(),
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
        if !self.presents.is_empty() {
            for access in image_acceses.clone() {
                self.check_unpresented(access.access.image);
            }
        }
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

    pub(crate) fn enter(&mut self, name: &'static str, pass: bool) {
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
            unsafe { observer.as_mut() }.on_enter(self, &scope);
        }
    }

    pub(crate) fn exit(&mut self, name: &'static str, pass: bool) {
        if let Some(mut observer) = self.observer {
            self.depth -= 1;
            let scope = ScopeInfo {
                name,
                depth: self.depth,
                pass,
            };
            // SAFETY: see `CommandBuffer::measure`.
            unsafe { observer.as_mut() }.on_exit(self, &scope);
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
        let bits = self.family.timestamp_valid_bits;
        assert!(
            bits > 0,
            "the {:?} family has no timestamps",
            self.family.kind
        );
        pool.take(self, index, 1);
        let mask = if bits >= 64 { !0 } else { (1 << bits) - 1 };
        pool.kind
            .valid_bits
            .store(mask, std::sync::atomic::Ordering::Relaxed);
        unsafe { Ctx::device().cmd_write_timestamp2(self.handle, stage, pool.handle, index) };
    }

    /// The active query of kind `Q`.
    pub(crate) fn open_query<Q: Scoped>(&self) -> Option<(vk::QueryPool, u32)> {
        self.open_queries[query_slot(Q::TYPE)]
    }

    /// Panics where hardware counters of `kind` can't begin or end: those of command-buffer
    /// scope, or of more than one pass, only around the whole recording (`profiling`), those of
    /// render-pass scope not inside a rendering; and a recording has one counter pool.
    fn check_counters<Q: Scoped>(&mut self, pool: vk::QueryPool, kind: &Q) {
        if Q::TYPE != vk::QueryType::PERFORMANCE_QUERY_KHR {
            return;
        }
        match self.performance_pool {
            Some(used) => assert!(
                used == pool,
                "a recording records one pool of hardware counters (lava doesn't enable \
                 `performanceCounterMultipleQueryPools`)"
            ),
            None => self.performance_pool = Some(pool),
        }
        let passes = kind.counter_passes();
        assert!(
            passes == 1 || self.profiling,
            "the counters of this pool take {passes} submissions: record them with \
             `Timeline::capture`"
        );
        match kind.counter_scope() {
            vk::PerformanceCounterScopeKHR::COMMAND_BUFFER => assert!(
                self.profiling,
                "counters of command-buffer scope count a whole recording: pass their pool to \
                 `Submission::submit_profiled` or `Timeline::capture`"
            ),
            vk::PerformanceCounterScopeKHR::RENDER_PASS => assert!(
                !self.in_rendering,
                "counters of render-pass scope can't begin or end inside a rendering"
            ),
            _ => {}
        }
    }

    pub(crate) fn begin_query<Q: Scoped>(&mut self, pool: &QueryPool<Q>, index: u32) {
        let slot = query_slot(Q::TYPE);
        if let Some((_, active)) = self.open_queries[slot] {
            panic!(
                "query {active} of a pool of this kind is active, and queries of one kind don't \
                 nest"
            );
        }
        self.check_counters(pool.handle, &pool.kind);
        pool.take(self, index, 1);
        let flags = pool.kind.control_flags();
        unsafe { Ctx::device().cmd_begin_query(self.handle, pool.handle, index, flags) };
        self.open_queries[slot] = Some((pool.handle, index));
    }

    pub(crate) fn end_query<Q: Scoped>(&mut self, pool: &QueryPool<Q>, index: u32) {
        let slot = query_slot(Q::TYPE);
        match self.open_queries[slot] {
            Some(active) if active == (pool.handle, index) => {}
            Some((active_pool, active)) => panic!(
                "query {index} of {} pool is ended while query {active} of {} is the active one \
                 of its kind",
                if active_pool == pool.handle { "the" } else { "a" },
                if active_pool == pool.handle { "it" } else { "another pool" },
            ),
            None => panic!("query {index} is ended, but no query of its kind is active"),
        }
        self.check_counters(pool.handle, &pool.kind);
        unsafe { Ctx::device().cmd_end_query(self.handle, pool.handle, index) };
        self.open_queries[slot] = None;
    }

    /// Begins or ends the query of `Submission::submit_profiled` or `Timeline::capture`, as the
    /// recording's first or last command.
    #[cfg(feature = "performance-query")]
    pub(crate) fn profile_query(
        &mut self,
        pool: &QueryPool<crate::query::Performance>,
        index: u32,
        begin: bool,
    ) {
        self.profiling = true;
        match begin {
            true => self.begin_query(pool, index),
            false => self.end_query(pool, index),
        }
        self.profiling = false;
    }

    /// The queries a finished recording left active, besides its profile's.
    pub(crate) fn left_open(&self) -> Option<u32> {
        let profile = self.profile.map(|p| (p.pool, p.index));
        let mut open = self.open_queries.iter().flatten();
        open.find(|&&query| Some(query) != profile).map(|&(_, index)| index)
    }

    /// Panics if `image` was presented earlier in the recording.
    fn check_unpresented(&self, image: vk::Image) {
        assert!(
            !self.presents.iter().any(|p| p.image == image),
            "a swapchain image is accessed after `cmd.present` of it: present must be the \
             image's last command"
        );
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

/// The resources a pass reaches without naming them in its push constants or specialization
/// constants (through a bindless handle or a pointer in a buffer), given to its generated
/// `push_bindings` and tracked at the pass's shader stages like the bindings' own accesses.
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

/// The image accesses of [`Accesses`]; the bindings set the stage.
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

/// The buffer accesses of [`Accesses`]; the bindings set the stage.
impl BufferAccess {
    /// The pass reads `buffer` through a pointer (`Buf`).
    pub fn read<V: Copy + 'static>(buffer: BufferSlice<'_, V>) -> Self {
        buffer.access(S2::NONE, vk::AccessFlags2::SHADER_STORAGE_READ)
    }

    /// The pass reads and writes `buffer` through a pointer (`MutBuf`).
    pub fn write<V: Copy + 'static>(buffer: BufferSlice<'_, V>) -> Self {
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

/// Every layer of `mips`: an image's tracked states cover all its layers.
fn subresource_range(aspect: vk::ImageAspectFlags, mips: Range<u32>) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: aspect,
        base_array_layer: 0,
        layer_count: vk::REMAINING_ARRAY_LAYERS,
        base_mip_level: mips.start,
        level_count: mips.end - mips.start,
    }
}

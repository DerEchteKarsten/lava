//! Command recording for generated passes: barriers, renderings and dispatches, scopes and queries (transfers live with the slices and views they take, acceleration structure commands with the structures, `present` with the swapchain)
use std::{
    fmt::Debug,
    panic::AssertUnwindSafe,
    ptr::NonNull,
    range::Range,
    sync::atomic::{AtomicU32, Ordering},
};

use crate::{
    bindless::Bindless,
    buffer::{
        self,
        slice::BufferSlice,
        usage::{IsIndex, IsIndirect},
    },
    error::Result,
    image::{
        dim::Dim,
        format::{ColorAspect, DepthAspect, Format},
        slice::ImageView,
        usage::{IsColorAttachment, IsDepthAttachment, IsSampled, IsStorage},
    },
    pipeline::{MAX_PUSH_CONSTANTS_SIZE, RasterHash, pass_map, pipelines},
    query::{Occlusion, QueryKind, QueryPool, QueryRecorder, ScopeInfo, ScopeObserver, Scoped, Timestamp},
    state::{Ctx, Functions},
    vkobjects::{
        acceleration_structure::Tlas,
        queue::{PendingAccesses, PendingWrite, Stage},
    },
};
use ash::vk::{self, IndexType};
use bytemuck::{Pod, Zeroable};
use glam::{IVec2, UVec2};
use lava_macros::validation_trace;
use smallvec::{SmallVec, smallvec};

mod binding_output;
use binding_output::Sections;
pub use binding_output::{BindingOutput, Bindings, kind};

/// A pass's access of a buffer range, made by `BufferSlice::access` (a binding generator's
/// part of [`Bindings::new`]).
#[derive(Debug, Clone, PartialEq)]
pub struct BufferAccess {
    pub stage: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
    pub range: Range<u64>,
}

/// The range of a read of every acceleration structure: a TLAS build or a trace reaches
/// BLASes lava can't list (instances may be written on the GPU). It is empty, so it overlaps
/// no buffer range, covers none and is covered by none; [`BufferAccess::overlaps`] makes it
/// conflict with acceleration structure writes instead, which only acceleration structure
/// commands make, all at the build stage.
pub(crate) const EVERY_ACCELERATION_STRUCTURE: Range<u64> = Range {
    start: u64::MAX,
    end: u64::MAX,
};

impl BufferAccess {
    /// Whether the two accesses touch the same memory.
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

#[derive(Debug, Clone, PartialEq)]
pub struct ImageAccess {
    pub stage: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
    pub image: vk::Image,
    pub layout: vk::ImageLayout,
    pub aspect: vk::ImageAspectFlags,
    pub mip_range: Range<u32>,
}

/// An image access of a command that isn't recorded yet, with the tracked layouts of its mips,
/// which only move when the command is recorded (see [`compute_barriers`]). Made by
/// `ImageView::access` (a binding generator's part of [`Bindings::new`]).
#[derive(Debug, Clone)]
pub struct ImageUse<'a> {
    pub access: ImageAccess,
    /// One layout for all of `access.mip_range`, or one per mip.
    pub(crate) layouts: &'a [AtomicU32],
}

impl ImageUse<'_> {
    /// Moves the tracked layouts of the access's mips to the one it asks for, and calls
    /// `transition(mips, old_layout)` once per run of neighbouring mips that had the same old
    /// layout and need a transition.
    pub(crate) fn take_layouts(&self, mut transition: impl FnMut(Range<u32>, vk::ImageLayout)) {
        let new = self.access.layout;
        let swap = |layout: &AtomicU32| {
            vk::ImageLayout::from_raw(layout.swap(new.as_raw() as u32, Ordering::Relaxed) as i32)
        };
        let mips = self.access.mip_range;
        if let [layout] = self.layouts {
            let old = swap(layout);
            if old != new {
                transition(mips, old);
            }
            return;
        }
        let mut run: Option<(u32, vk::ImageLayout)> = None;
        for (mip, layout) in (mips.start..).zip(self.layouts) {
            let old = swap(layout);
            match run {
                Some((_, run_old)) if run_old == old => {}
                Some((start, run_old)) => {
                    if run_old != new {
                        transition((start..mip).into(), run_old);
                    }
                    run = Some((mip, old));
                }
                None => run = Some((mip, old)),
            }
        }
        if let Some((start, old)) = run
            && old != new
        {
            transition((start..mips.end).into(), old);
        }
    }
}

pub struct CommandBuffer {
    pub(crate) handle: vk::CommandBuffer,
    pub(crate) pending_accesses: PendingAccesses,
    /// The epoch of the frame being recorded, which the query pools it touches remember.
    pub(crate) epoch: u64,
    pub(crate) family: u32,
    /// What `measure` attached, for the duration of its closure (its lifetime erased).
    observer: Option<NonNull<dyn ScopeObserver>>,
    /// How many scopes the observer is inside.
    depth: u16,
    /// The kinds of the active queries (`query_bit`): one of each can be active.
    pub(crate) active_queries: u32,
    /// The barriers of [`Self::record_access`]es, recorded with the next command's.
    deferred: Option<Barriers>,
    /// The draw list renderings reuse (`RasterBuilder::draws`), empty between them.
    draws: Vec<Draw<'static>>,
}

/// A bit per kind of query that is begun and ended, of which one can be active at a time.
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

/// The most color attachments a rendering can have.
pub const MAX_COLOR_ATTACHMENTS: usize = 8;

/// One rendering being built (`CommandBuffer::raster`): its `N` color attachments, an optional
/// depth attachment and the draws so far, which [`Self::record`] records.
pub struct RasterBuilder<'a, const N: usize> {
    cmd_buf: &'a mut CommandBuffer,
    extent: UVec2,
    viewport: Viewport,
    scissors: SmallVec<[vk::Rect2D; 1]>,
    color: [Attachment<'a>; N],
    depth_format: vk::Format,
    stencil_format: vk::Format,
    depth_attachment: vk::ImageView,
    /// Only a read; [`Self::depth_access`] adds the write.
    depth_access: Option<ImageUse<'a>>,
    clear_depth: Option<vk::ClearValue>,
    /// The command buffer's draw list, lent to this rendering.
    draws: Vec<Draw<'a>>,
}

/// A color attachment of a rendering (`CommandBuffer::raster`): the view drawn to, cleared to
/// `clear` first unless that is `None`.
pub struct Attachment<'a> {
    format: vk::Format,
    view: vk::ImageView,
    clear: Option<vk::ClearValue>,
    access: ImageUse<'a>,
}

impl<'a> Attachment<'a> {
    pub fn color<F: Format, U>(image: ImageView<'a, F, U>, clear: Option<F::Texels>) -> Self
    where
        U: IsColorAttachment,
        F: ColorAspect,
    {
        image.aspect_check(vk::ImageAspectFlags::COLOR);
        image.usage_check(vk::ImageUsageFlags::COLOR_ATTACHMENT);
        let access = image.access(
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                | if clear.is_none() {
                    vk::AccessFlags2::COLOR_ATTACHMENT_READ
                } else {
                    vk::AccessFlags2::empty()
                },
            // Attachments are drawn to in this layout; every other access asks for `GENERAL`,
            // which costs nothing when sampling but can keep a driver from compressing a
            // framebuffer.
            vk::ImageLayout::ATTACHMENT_OPTIMAL,
        );
        Self {
            format: image.format().vk_format(),
            view: image.view,
            clear: clear.map(|e| image.format().clear_value(e).into()),
            access,
        }
    }
}

/// How a color attachment combines a fragment's output `src` with the stored value `dst`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Blend {
    /// `src`, unblended.
    Replace,
    /// `src.rgb * src.a + dst.rgb * (1 - src.a)`; alpha becomes `src.a`.
    Alpha,
    /// `src + dst`.
    Add,
    /// `dst * (1 - src)`: `src` is how much of `dst` the fragment hides.
    Attenuate,
    /// `dst`: the draw leaves the attachment untouched.
    Skip,
}

/// The pipeline state of one draw into `COLOR_ATTACHMENTS` color attachments.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RasterState<const COLOR_ATTACHMENTS: usize> {
    blends: [Blend; COLOR_ATTACHMENTS],
    depth_write: bool,
    backface_culling: bool,
    wire_frame: bool,
}

impl<const COLOR_ATTACHMENTS: usize> Default for RasterState<COLOR_ATTACHMENTS> {
    fn default() -> Self {
        Self {
            blends: [Blend::Replace; COLOR_ATTACHMENTS],
            depth_write: true,
            backface_culling: true,
            wire_frame: false,
        }
    }
}

impl<const COLOR_ATTACHMENTS: usize> RasterState<COLOR_ATTACHMENTS> {
    /// One blend per color attachment, in attachment order.
    pub fn blends(mut self, blends: [Blend; COLOR_ATTACHMENTS]) -> Self {
        self.blends = blends;
        self
    }

    /// Whether fragments that pass the depth test write their depth.
    pub fn depth_write(mut self, depth_write: bool) -> Self {
        self.depth_write = depth_write;
        self
    }

    pub fn backface_culling(mut self, backface_culling: bool) -> Self {
        self.backface_culling = backface_culling;
        self
    }

    pub fn wire_frame(mut self, wire_frame: bool) -> Self {
        self.wire_frame = wire_frame;
        self
    }

    /// The same state in a list of any rendering's draws: the blends past the first
    /// `COLOR_ATTACHMENTS` are never read.
    fn widen(self) -> RasterState<MAX_COLOR_ATTACHMENTS> {
        let mut blends = [Blend::Replace; MAX_COLOR_ATTACHMENTS];
        blends[..COLOR_ATTACHMENTS].copy_from_slice(&self.blends);
        RasterState {
            blends,
            depth_write: self.depth_write,
            backface_culling: self.backface_culling,
            wire_frame: self.wire_frame,
        }
    }
}

struct Draw<'a> {
    bindings: &'a BindingOutput<'a, ()>,
    /// The draw's index or indirect buffers, read at other stages than the pass's shaders.
    extra: SmallVec<[BufferAccess; 2]>,
    command: DrawCommand,
    state: RasterState<MAX_COLOR_ATTACHMENTS>,
    /// The occlusion query counting its samples (`RasterBuilder::count_samples`).
    samples: Option<(&'a QueryPool<Occlusion>, u32)>,
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
    Launch([u32; 3]),
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

/// `draws`, emptied, for draws of another lifetime, keeping its allocation (an in-place
/// collect of nothing).
fn recycle<'b>(mut draws: Vec<Draw<'_>>) -> Vec<Draw<'b>> {
    draws.clear();
    draws.into_iter().map(|_| unreachable!()).collect()
}

impl<'a, const N: usize> RasterBuilder<'a, N> {
    /// Depth-tests the draws against `image` (nearer = greater). `clear` resets the depth first.
    pub fn depth_attachment<F: Format, U>(
        &mut self,
        image: ImageView<'a, F, U>,
        clear: Option<F::Texels>,
    ) -> &mut Self
    where
        U: IsDepthAttachment,
        F: DepthAspect,
    {
        image.aspect_check(vk::ImageAspectFlags::DEPTH);
        image.usage_check(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT);
        let format = image.format();
        self.depth_format = format.vk_format();
        if format.aspects().contains(vk::ImageAspectFlags::STENCIL) {
            self.stencil_format = format.vk_format();
        }
        self.depth_attachment = image.view;
        self.clear_depth = clear.map(|e| format.clear_value(e).into());
        self.depth_access = Some(image.access(
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
            vk::ImageLayout::ATTACHMENT_OPTIMAL,
        ));
        self
    }

    /// Replaces the default viewport, the whole render area.
    pub fn viewport(&mut self, viewport: Viewport) -> &mut Self {
        self.viewport = viewport;
        self
    }

    /// Replaces the default scissor, the whole render area.
    pub fn scissors(&mut self, scissors: &[Scissor]) -> &mut Self {
        self.scissors = scissors.iter().map(|s| rect(s.offset, s.extent)).collect();
        self
    }

    pub fn draw(
        &mut self,
        bindings: &'a BindingOutput<'a, kind::RasterVertex>,
        vertices: u32,
        instances: u32,
        state: RasterState<N>,
    ) -> &mut Self {
        let command = DrawCommand::Draw {
            vertices,
            instances,
        };
        self.push(bindings.erase(), SmallVec::new(), command, state)
    }

    pub fn draw_indexed<IDX: IsIndex>(
        &mut self,
        bindings: &'a BindingOutput<'a, kind::RasterVertex>,
        index_buffer: BufferSlice<'a, u32, IDX>,
        instances: u32,
        state: RasterState<N>,
    ) -> &mut Self {
        index_buffer.usage_check(vk::BufferUsageFlags::INDEX_BUFFER);
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
        self.push(bindings.erase(), smallvec![index_access], command, state)
    }

    pub fn draw_indirect<IND: IsIndirect>(
        &mut self,
        bindings: &'a BindingOutput<'a, kind::RasterVertex>,
        buffer: BufferSlice<'a, DrawIndirectCommand, IND>,
        state: RasterState<N>,
    ) -> &mut Self {
        let command = DrawCommand::Indirect {
            draws: buffer.strided_range(),
            flags: buffer.address_flags(),
            count: buffer.len() as u32,
        };
        self.push(
            bindings.erase(),
            smallvec![indirect_access(buffer)],
            command,
            state,
        )
    }

    pub fn draw_indirect_count<IND: IsIndirect, CNT: IsIndirect>(
        &mut self,
        bindings: &'a BindingOutput<'a, kind::RasterVertex>,
        buffer: BufferSlice<'a, DrawIndirectCommand, IND>,
        count_buffer: BufferSlice<'a, u32, CNT>,
        state: RasterState<N>,
    ) -> &mut Self {
        let extra = smallvec![indirect_access(buffer), indirect_access(count_buffer)];
        let command = DrawCommand::IndirectCount {
            draws: buffer.strided_range(),
            flags: buffer.address_flags(),
            count: count_buffer.address_range(),
            count_flags: count_buffer.address_flags(),
            max_count: buffer.len() as u32,
        };
        self.push(bindings.erase(), extra, command, state)
    }

    pub fn launch(
        &mut self,
        bindings: &'a BindingOutput<'a, kind::RasterMesh>,
        groups: [u32; 3],
        state: RasterState<N>,
    ) -> &mut Self {
        self.push(
            bindings.erase(),
            SmallVec::new(),
            DrawCommand::Launch(groups),
            state,
        )
    }

    /// Launches the task (or, without a task shader, mesh) workgroup counts in `buffer`.
    pub fn launch_indirect<IND: IsIndirect>(
        &mut self,
        bindings: &'a BindingOutput<'a, kind::RasterMesh>,
        buffer: BufferSlice<'a, DispatchIndirectCommand, IND>,
        state: RasterState<N>,
    ) -> &mut Self {
        let command = DrawCommand::LaunchIndirect {
            draws: buffer.strided_range(),
            flags: buffer.address_flags(),
            count: buffer.len() as u32,
        };
        self.push(
            bindings.erase(),
            smallvec![indirect_access(buffer)],
            command,
            state,
        )
    }

    fn push(
        &mut self,
        bindings: &'a BindingOutput<'a, ()>,
        extra: SmallVec<[BufferAccess; 2]>,
        command: DrawCommand,
        state: RasterState<N>,
    ) -> &mut Self {
        self.draws.push(Draw {
            bindings,
            extra,
            command,
            state: state.widen(),
            samples: None,
        });
        self
    }

    /// Counts the samples of the draw added last that pass the depth and stencil tests into
    /// query `index` of `pool`.
    pub fn count_samples(&mut self, pool: &'a QueryPool<Occlusion>, index: u32) -> &mut Self {
        assert!(
            index < pool.count(),
            "query {index} of a pool of {}",
            pool.count()
        );
        let draw = self
            .draws
            .last_mut()
            .expect("`count_samples` follows a draw");
        draw.samples = Some((pool, index));
        self
    }

    /// The depth attachment's access: also a write when it is cleared or a draw writes depth.
    fn depth_access(&self) -> Option<ImageUse<'a>> {
        let mut depth = self.depth_access.clone()?;
        if self.clear_depth.is_some() || self.draws.iter().any(|d| d.state.depth_write) {
            depth.access.access |= vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE;
        }
        Some(depth)
    }

    /// The pipeline key of a draw with `state` in this rendering.
    fn pipeline_key(&self, state: &RasterState<MAX_COLOR_ATTACHMENTS>) -> RasterHash {
        RasterHash {
            color_formats: self.color.iter().map(|c| c.format).collect(),
            depth_format: self.depth_format,
            stencil_format: self.stencil_format,
            blends: SmallVec::from_slice(&state.blends[..N]),
            depth_write: state.depth_write,
            backface_culling: state.backface_culling,
            wire_frame: state.wire_frame,
        }
    }

    /// Records the rendering under the GPU scope `name`: the attachments are cleared or loaded,
    /// then the draws run in order. Depth tests, depth writes and blending see the earlier draws'
    /// fragments, but nothing else is synchronised between them, so no draw may read what an
    /// earlier one writes to a buffer or storage image. Records the draws once: the builder has
    /// none afterwards.
    #[validation_trace]
    pub fn record(&mut self, name: &'static str) {
        let _span = tracing::info_span!("draw", pass = name).entered();
        let depth_access = self.depth_access();
        let draws = &self.draws;
        self.cmd_buf.flush_pending(
            draws
                .iter()
                .flat_map(|draw| draw.bindings.sections().images())
                .chain(self.color.iter().map(|c| c.access.clone()))
                .chain(depth_access.clone()),
            draws.iter().flat_map(|draw| {
                (draw.bindings.sections().buffers()).chain(draw.extra.iter().cloned())
            }),
        );

        let color_attachments = self.color.each_ref().map(|color| {
            let ret = vk::RenderingAttachmentInfo::default()
                .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                .image_view(color.view)
                .store_op(vk::AttachmentStoreOp::STORE);
            if let Some(clear_value) = color.clear {
                ret.clear_value(clear_value)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
            } else {
                ret.load_op(vk::AttachmentLoadOp::LOAD)
            }
        });
        let mut rendering_info = vk::RenderingInfo::default()
            .color_attachments(&color_attachments)
            .layer_count(1)
            .render_area(rect(IVec2::ZERO, self.extent))
            .view_mask(0);

        let mut depth_info;
        if let Some(depth) = &depth_access {
            let written = depth
                .access
                .access
                .contains(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE);
            depth_info = vk::RenderingAttachmentInfo::default()
                .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                .image_view(self.depth_attachment)
                .store_op(if written {
                    vk::AttachmentStoreOp::STORE
                } else {
                    vk::AttachmentStoreOp::NONE
                })
                .load_op(vk::AttachmentLoadOp::LOAD);
            if let Some(clear_value) = self.clear_depth {
                depth_info = depth_info
                    .clear_value(clear_value)
                    .load_op(vk::AttachmentLoadOp::CLEAR);
            }
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
            device.cmd_set_viewport(
                handle,
                0,
                &[vk::Viewport {
                    x: self.viewport.offset.x as f32,
                    y: self.viewport.offset.y as f32,
                    width: self.viewport.extent.x as f32,
                    height: self.viewport.extent.y as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            device.cmd_set_scissor(handle, 0, &self.scissors);
        }
        for draw in &self.draws {
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
                let key = self.pipeline_key(&draw.state);
                record_draw(self.cmd_buf, &key, &draw.command, draw.bindings.sections());
            }
            if let Some((pool, query)) = draw.samples {
                self.cmd_buf.end_query(pool, query);
            }
        }
        unsafe { device.cmd_end_rendering(handle) };
        self.cmd_buf.exit(name, true);
        // The next rendering reuses the list (one dropped without being recorded loses it).
        self.cmd_buf.draws = recycle(std::mem::take(&mut self.draws));
    }
}

/// Binds the pipeline `key` of a draw's pass and records the draw.
fn record_draw(
    cmd_buf: &mut CommandBuffer,
    key: &RasterHash,
    command: &DrawCommand,
    sections: Sections,
) {
    let (device, handle) = (Ctx::device(), cmd_buf.handle);
    {
        let pipeline = pipelines().raster(key, sections.constants, sections.pass);
        cmd_buf.push_constants(sections.push_constants);
        unsafe {
            device.cmd_bind_pipeline(handle, vk::PipelineBindPoint::GRAPHICS, pipeline);
            match *command {
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
                    address_commands().cmd_bind_index_buffer3(
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
                } => address_commands().cmd_draw_indirect2(
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
                } => address_commands().cmd_draw_indirect_count2(
                    handle,
                    &vk::DrawIndirectCount2InfoKHR::default()
                        .address_range(draws)
                        .address_flags(flags)
                        .count_address_range(count)
                        .count_address_flags(count_flags)
                        .max_draw_count(max_count),
                ),
                DrawCommand::Launch([x, y, z]) => {
                    mesh_functions().cmd_draw_mesh_tasks(handle, x, y, z)
                }
                DrawCommand::LaunchIndirect {
                    draws,
                    flags,
                    count,
                } => address_commands().cmd_draw_mesh_tasks_indirect2(
                    handle,
                    &vk::DrawIndirect2InfoKHR::default()
                        .address_range(draws)
                        .address_flags(flags)
                        .draw_count(count),
                ),
            }
        }
    }
}

fn address_commands() -> &'static ash::khr::device_address_commands::Device {
    Functions::device_address_commands()
}

fn mesh_functions() -> &'static ash::ext::mesh_shader::Device {
    Functions::mesh().expect(
        "mesh shaders aren't supported by this device (`Ctx::features().mesh`), so a mesh pass can't be launched",
    )
}

/// The access of an indirect command read from `buffer`, which has to be `IsIndirect`.
fn indirect_access<T: Copy + Pod, U: IsIndirect>(buffer: BufferSlice<'_, T, U>) -> BufferAccess {
    buffer.usage_check(vk::BufferUsageFlags::INDIRECT_BUFFER);
    buffer.access(
        vk::PipelineStageFlags2::DRAW_INDIRECT,
        vk::AccessFlags2::INDIRECT_COMMAND_READ,
    )
}

impl CommandBuffer {
    /// Starts a rendering of the area `extent` into the `color` attachments, with viewport and
    /// scissor covering it. Every draw has a blend per color attachment.
    pub fn raster<'a, const N: usize>(
        &'a mut self,
        extent: UVec2,
        color: [Attachment<'a>; N],
    ) -> RasterBuilder<'a, N> {
        const { assert!(N <= MAX_COLOR_ATTACHMENTS) };
        let draws = recycle(std::mem::take(&mut self.draws));
        RasterBuilder {
            cmd_buf: self,
            extent,
            viewport: Viewport {
                offset: IVec2::ZERO,
                extent,
            },
            scissors: smallvec![rect(IVec2::ZERO, extent)],
            color,
            depth_format: vk::Format::UNDEFINED,
            stencil_format: vk::Format::UNDEFINED,
            depth_attachment: vk::ImageView::null(),
            depth_access: None,
            clear_depth: None,
            draws,
        }
    }

    /// Declares that the next command reaches `access` without naming it, at `stage`: through
    /// a bindless handle or a pointer the pass reads from a buffer. The access is tracked like
    /// the command's own: its barrier joins the next command's (or the end of the recording),
    /// and later commands wait for it. Declare a rendering's accesses before it. An image
    /// declared in another layout than the next command uses it in gets two transitions in one
    /// barrier, which is invalid.
    pub fn record_access(&mut self, access: Access<'_>, stage: Stage) {
        use vk::PipelineStageFlags2 as S;
        let stage = match stage {
            Stage::All => S::ALL_COMMANDS,
            Stage::Compute => S::COMPUTE_SHADER,
            Stage::RayTracing => S::RAY_TRACING_SHADER_KHR,
            Stage::Raster => S::PRE_RASTERIZATION_SHADERS | S::FRAGMENT_SHADER,
            Stage::Transfer => {
                panic!("`record_access` declares shader accesses; transfers name what they touch")
            }
        };
        let (image, buffers) = match access.0 {
            AccessKind::Image(mut image) => {
                image.access.stage = stage;
                (Some(image), SmallVec::new())
            }
            AccessKind::Buffers(mut buffers) => {
                buffers.iter_mut().for_each(|buffer| buffer.stage = stage);
                (None, buffers)
            }
        };
        let barriers = compute_barriers(
            &self.pending_accesses,
            image.iter().cloned(),
            buffers.iter().cloned(),
        );
        if let Some(barriers) = barriers {
            self.pending_accesses.made_visible(&barriers);
            match &mut self.deferred {
                Some(deferred) => deferred.merge(barriers),
                None => self.deferred = Some(barriers),
            }
        }
        self.pending_accesses
            .record(image.map(|image| image.access).into_iter(), buffers.into_iter());
    }

    /// Records the barriers of [`Self::record_access`]es no command took.
    pub(crate) fn flush_deferred(&mut self) {
        if let Some(barriers) = self.deferred.take() {
            self.pipeline_barrier(&barriers);
        }
    }

    fn pipeline_barrier(&self, barriers: &Barriers) {
        let mut info = vk::DependencyInfo::default();
        let memory = barriers.memory.map(|memory| [memory]);
        if let Some(memory) = &memory {
            info = info.memory_barriers(memory);
        }
        if !barriers.images.is_empty() {
            info = info.image_memory_barriers(&barriers.images);
        }
        unsafe { Ctx::device().cmd_pipeline_barrier2(self.handle, &info) };
    }


    fn compute_private<IND: IsIndirect>(
        &mut self,
        bindings: &BindingOutput<'_, kind::Compute>,
        dispatch: [u32; 3],
        indirect_buffer: Option<BufferSlice<'_, DispatchIndirectCommand, IND>>,
    ) {
        let sections = bindings.sections();
        let pass = &pass_map().passes[sections.pass];
        let _span = tracing::info_span!("compute", pass = pass.name).entered();
        self.flush_pending(
            sections.images(),
            sections
                .buffers()
                .chain(indirect_buffer.map(indirect_access)),
        );
        self.push_constants(sections.push_constants);

        let pipeline = pipelines().compute(sections.pass, sections.constants);

        self.enter(pass.name, true);
        unsafe {
            Ctx::device().cmd_bind_pipeline(self.handle, vk::PipelineBindPoint::COMPUTE, pipeline);
            if let Some(slice) = indirect_buffer {
                address_commands().cmd_dispatch_indirect2(
                    self.handle,
                    &vk::DispatchIndirect2InfoKHR::default()
                        .address_range(slice.address_range())
                        .address_flags(slice.address_flags()),
                );
            } else {
                Ctx::device().cmd_dispatch(self.handle, dispatch[0], dispatch[1], dispatch[2]);
            }
        }
        self.exit(pass.name, true);
    }

    #[validation_trace]
    pub fn compute_indirect<IND: IsIndirect>(
        &mut self,
        bindings: &BindingOutput<'_, kind::Compute>,
        buffer: BufferSlice<'_, DispatchIndirectCommand, IND>,
    ) {
        self.compute_private(bindings, [0, 0, 0], Some(buffer));
    }

    #[validation_trace]
    pub fn compute(&mut self, bindings: &BindingOutput<'_, kind::Compute>, dispatch: [u32; 3]) {
        self.compute_private::<buffer::usage::Indirect>(bindings, dispatch, None);
    }

    fn raytrace_private<IND: IsIndirect>(
        &mut self,
        bindings: &BindingOutput<'_, kind::RayTracing>,
        rays: [u32; 3],
        indirect_buffer: Option<BufferSlice<'_, TraceRaysIndirectCommand, IND>>,
    ) {
        let sections = bindings.sections();
        let pass = &pass_map().passes[sections.pass];
        let _span = tracing::info_span!("raytrace", pass = pass.name).entered();
        self.flush_pending(
            sections.images(),
            sections
                .buffers()
                .chain(indirect_buffer.map(indirect_access)),
        );
        self.push_constants(sections.push_constants);

        self.enter(pass.name, true);
        let functions = Functions::raytracing_pipeline()
            .expect("a pass that can be recorded has its pipeline functions");
        pipelines().ray_tracing(sections.pass, sections.constants, |pipeline| unsafe {
            Ctx::device().cmd_bind_pipeline(
                self.handle,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                pipeline.pipeline,
            );
            let [raygen, miss, hit, callable] = &pipeline.sbt.regions;
            match indirect_buffer {
                Some(slice) => functions.cmd_trace_rays_indirect(
                    self.handle,
                    raygen,
                    miss,
                    hit,
                    callable,
                    slice.address(),
                ),
                None => functions.cmd_trace_rays(
                    self.handle,
                    raygen,
                    miss,
                    hit,
                    callable,
                    rays[0],
                    rays[1],
                    rays[2],
                ),
            }
        });
        self.exit(pass.name, true);
    }

    /// Traces `rays` (width, height, depth) raygen invocations of a ray tracing pass.
    #[validation_trace]
    pub fn raytrace(&mut self, bindings: &BindingOutput<'_, kind::RayTracing>, rays: [u32; 3]) {
        self.raytrace_private::<buffer::usage::Indirect>(bindings, rays, None);
    }

    /// Like [`Self::raytrace`], with the ray counts read from `buffer` on the GPU.
    #[validation_trace]
    pub fn raytrace_indirect<IND: IsIndirect>(
        &mut self,
        bindings: &BindingOutput<'_, kind::RayTracing>,
        buffer: BufferSlice<'_, TraceRaysIndirectCommand, IND>,
    ) {
        self.raytrace_private(bindings, [0, 0, 0], Some(buffer));
    }

    /// Pushes a pass's push constants from offset 0; the bytes past them keep what earlier
    /// passes pushed, which the pass doesn't read.
    fn push_constants(&mut self, constants: &[u8]) {
        if constants.is_empty() {
            return;
        }
        unsafe {
            Ctx::device().cmd_push_constants(
                self.handle,
                Bindless::layout(),
                vk::ShaderStageFlags::ALL,
                0,
                constants,
            )
        };
    }

    pub(crate) fn flush_pending<'a>(
        &mut self,
        image_acceses: impl Iterator<Item = ImageUse<'a>> + Clone,
        buffer_acceses: impl Iterator<Item = BufferAccess> + Clone,
    ) {
        let barriers = compute_barriers(
            &self.pending_accesses,
            image_acceses.clone(),
            buffer_acceses.clone(),
        );
        if let Some(barriers) = &barriers {
            self.pending_accesses.made_visible(barriers);
        }
        let barriers = match (self.deferred.take(), barriers) {
            (Some(mut deferred), Some(barriers)) => {
                deferred.merge(barriers);
                Some(deferred)
            }
            (deferred, barriers) => deferred.or(barriers),
        };
        if let Some(barriers) = &barriers {
            self.pipeline_barrier(barriers);
        }

        self.pending_accesses
            .record(image_acceses.map(|image| image.access), buffer_acceses);
    }

    /// Debugging aid: a full barrier that makes every command recorded before it finish, and
    /// its writes visible to every later command, before anything after it starts. Leaves the
    /// tracked accesses alone, so the automatic barriers stay what they would be without it:
    /// if a glitch goes away with this call, an automatic barrier before it is missing.
    pub fn flush_all(&mut self) {
        let all = vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE;
        let memory = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(all)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(all)];
        let info = vk::DependencyInfo::default().memory_barriers(&memory);
        unsafe { Ctx::device().cmd_pipeline_barrier2(self.handle, &info) };
    }

    /// Records `f` as a named group: a debug label for RenderDoc/Nsight in debug builds, and a
    /// scope the observer of [`Self::measure`] sees. Every pass and rendering is such a scope on
    /// its own.
    pub fn scope<R>(&mut self, name: &'static str, f: impl FnOnce(&mut Self) -> R) -> R {
        self.enter(name, false);
        let result = f(self);
        self.exit(name, false);
        result
    }

    /// Records `f` as the scope `name` with `observer` attached, which sees it and every scope
    /// and pass inside it (see [`ScopeObserver`]); with `None` just as a scope. Doesn't nest.
    pub fn measure<R>(
        &mut self,
        observer: Option<&mut dyn ScopeObserver>,
        name: &'static str,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let Some(observer) = observer else {
            return self.scope(name, f);
        };
        assert!(self.observer.is_none(), "`measure` doesn't nest");
        // SAFETY: only the lifetime is erased. The pointer is cleared before `observer`'s borrow
        // ends, also when `f` panics, and until then only this command buffer reaches it.
        let observer = unsafe {
            std::mem::transmute::<NonNull<dyn ScopeObserver + '_>, NonNull<dyn ScopeObserver>>(
                NonNull::from(observer),
            )
        };
        self.observer = Some(observer);
        self.depth = 0;
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| self.scope(name, f)));
        self.observer = None;
        result.unwrap_or_else(|panic| std::panic::resume_unwind(panic))
    }

    /// Opens a scope: its debug label, and the observer's `on_enter`.
    fn enter(&mut self, name: &'static str, pass: bool) {
        Functions::cmd_start_label(&self.handle, name);
        if let Some(mut observer) = self.observer {
            let scope = ScopeInfo {
                name,
                depth: self.depth,
                pass,
            };
            self.depth += 1;
            // SAFETY: see `measure`.
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
            // SAFETY: see `measure`.
            unsafe { observer.as_mut() }.on_exit(&mut QueryRecorder { cmd: self }, &scope);
        }
        Functions::cmd_end_label(&self.handle);
    }

    /// Writes the device's clock into query `index` of `pool` once the commands recorded before
    /// reached `stage`: `NONE` for when they started, `ALL_COMMANDS` for when they finished.
    pub fn write_timestamp(
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
        pool.record_on(self.epoch, self.family);
        unsafe { Ctx::device().cmd_write_timestamp2(self.handle, stage, pool.handle, index) };
    }

    /// Records `f` inside query `index` of `pool`, which counts what its commands do. One query
    /// of a kind can be active at a time, also counting an observer's (`QueryRecorder::is_active`).
    /// A performance query's pool has to exist before the recording begins.
    pub fn query<K: Scoped, R>(
        &mut self,
        pool: &QueryPool<K>,
        index: u32,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        assert!(
            index < pool.count(),
            "query {index} of a pool of {}",
            pool.count()
        );
        self.begin_query(pool, index);
        let result = f(self);
        self.end_query(pool, index);
        result
    }

    pub(crate) fn begin_query<K: Scoped>(&mut self, pool: &QueryPool<K>, index: u32) {
        let bit = query_bit(K::TYPE);
        assert!(
            self.active_queries & bit == 0,
            "a query of this kind is already active, and queries of one kind don't nest"
        );
        if K::TYPE == vk::QueryType::PERFORMANCE_QUERY_KHR {
            assert_eq!(
                self.family,
                Ctx::gfx_queue_index(),
                "hardware counters are only recorded on the graphics family"
            );
        }
        pool.record(self.epoch);
        let flags = pool.kind.control_flags();
        unsafe { Ctx::device().cmd_begin_query(self.handle, pool.handle, index, flags) };
        self.active_queries |= bit;
    }

    pub(crate) fn end_query<K: QueryKind>(&mut self, pool: &QueryPool<K>, index: u32) {
        let bit = query_bit(K::TYPE);
        assert!(self.active_queries & bit != 0, "no query of this kind is active");
        unsafe { Ctx::device().cmd_end_query(self.handle, pool.handle, index) };
        self.active_queries &= !bit;
    }

    /// A command buffer recording into `handle` for the frame of `epoch` on queue `family`.
    pub(crate) fn new(
        handle: vk::CommandBuffer,
        pending_accesses: PendingAccesses,
        epoch: u64,
        family: u32,
    ) -> Self {
        Self {
            handle,
            pending_accesses,
            epoch,
            family,
            observer: None,
            depth: 0,
            active_queries: 0,
            deferred: None,
            draws: Vec::new(),
        }
    }

    pub(crate) fn begin(&mut self) -> Result<()> {
        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe { Ctx::device().begin_command_buffer(self.handle, &begin_info)? };
        Bindless::bind(&self.handle);
        // Passes push only their own push constants; this sets the rest of the range once, so
        // no byte of it is ever undefined.
        self.push_constants(&[0; MAX_PUSH_CONSTANTS_SIZE]);
        Ok(())
    }

    pub(crate) fn end(&mut self) -> Result<()> {
        unsafe { Ctx::device().end_command_buffer(self.handle)? };
        Ok(())
    }
}

/// Barriers one command needs before it may run, given what is still pending.
pub(crate) struct Barriers {
    /// The dependency on the pending accesses the command conflicts with; `None` without a
    /// hazard, when only a layout transition is needed, since that carries its own stages.
    pub(crate) memory: Option<vk::MemoryBarrier2<'static>>,
    /// Layout transitions; empty when every image already is in the requested layout.
    pub(crate) images: SmallVec<[vk::ImageMemoryBarrier2<'static>; 1]>,
}

impl Barriers {
    /// Adds `other`'s dependencies: one memory barrier from both sources to both destinations,
    /// and both lists of transitions.
    fn merge(&mut self, other: Barriers) {
        self.memory = match (self.memory, other.memory) {
            (Some(a), Some(b)) => Some(
                a.src_stage_mask(a.src_stage_mask | b.src_stage_mask)
                    .src_access_mask(a.src_access_mask | b.src_access_mask)
                    .dst_stage_mask(a.dst_stage_mask | b.dst_stage_mask)
                    .dst_access_mask(a.dst_access_mask | b.dst_access_mask),
            ),
            (a, b) => a.or(b),
        };
        self.images.extend(other.images);
    }
}

/// A resource a pass reaches without naming it in its push constants, through a bindless
/// handle or a pointer it reads from a buffer, which [`CommandBuffer::record_access`] declares.
pub struct Access<'a>(AccessKind<'a>);

enum AccessKind<'a> {
    Image(ImageUse<'a>),
    Buffers(SmallVec<[BufferAccess; 2]>),
}

impl<'a> Access<'a> {
    fn image<F: Format, U: crate::image::usage::ImageUsage, D: Dim>(
        image: ImageView<'a, F, U, D>,
        access: vk::AccessFlags2,
    ) -> Self {
        // `record_access` sets the stage.
        let stage = vk::PipelineStageFlags2::NONE;
        Self(AccessKind::Image(image.access(stage, access, vk::ImageLayout::GENERAL)))
    }

    /// The pass samples `image` (`Tex2D`, or a `DynImg` with a sampled index).
    pub fn sampled_read<F: Format, U: IsSampled, D: Dim>(image: ImageView<'a, F, U, D>) -> Self {
        image.usage_check(vk::ImageUsageFlags::SAMPLED);
        Self::image(image, vk::AccessFlags2::SHADER_SAMPLED_READ)
    }

    /// The pass loads from `image` as a storage image (`Image2D`, or a `DynImg` without a
    /// sampled index).
    pub fn storage_read<F: Format, U: IsStorage, D: Dim>(image: ImageView<'a, F, U, D>) -> Self {
        image.usage_check(vk::ImageUsageFlags::STORAGE);
        Self::image(image, vk::AccessFlags2::SHADER_STORAGE_READ)
    }

    /// The pass loads from and stores to `image` (`MutImage2D`).
    pub fn storage_write<F: Format, U: IsStorage, D: Dim>(image: ImageView<'a, F, U, D>) -> Self {
        image.usage_check(vk::ImageUsageFlags::STORAGE);
        let access = vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE;
        Self::image(image, access)
    }

    /// The pass reads `buffer` through a pointer (`Buf`).
    pub fn buffer_read<V: Copy + Pod, U: buffer::usage::IsStorage>(
        buffer: BufferSlice<'_, V, U>,
    ) -> Self {
        buffer.usage_check(vk::BufferUsageFlags::STORAGE_BUFFER);
        let stage = vk::PipelineStageFlags2::NONE;
        let access = buffer.access(stage, vk::AccessFlags2::SHADER_STORAGE_READ);
        Self(AccessKind::Buffers(smallvec![access]))
    }

    /// The pass reads and writes `buffer` through a pointer (`MutBuf`).
    pub fn buffer_write<V: Copy + Pod, U: buffer::usage::IsStorage>(
        buffer: BufferSlice<'_, V, U>,
    ) -> Self {
        buffer.usage_check(vk::BufferUsageFlags::STORAGE_BUFFER);
        let access = buffer.access(
            vk::PipelineStageFlags2::NONE,
            vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
        );
        Self(AccessKind::Buffers(smallvec![access]))
    }

    /// The pass traces `tlas` through an address it reads from a buffer (an `Accel` inside a
    /// struct): its memory and the BLASes it names.
    pub fn trace(tlas: &Tlas) -> Self {
        let accesses = tlas.trace_accesses(vk::PipelineStageFlags2::NONE);
        Self(AccessKind::Buffers(SmallVec::from_buf(accesses)))
    }
}


/// Works out the barriers needed to run a command with the given accesses after `pending`.
///
/// A hazard exists when the command touches something a pending access wrote (read-after-write,
/// write-after-write) or writes something a pending access read. Buffers conflict when their
/// byte ranges overlap, images when they are the same image. An image whose tracked layout
/// differs from the requested one additionally gets a layout transition. Returns `None` when
/// neither applies.
///
/// It moves each image's tracked layout to the requested one, in the order of the accesses, so
/// call it once per command, when the command is recorded. Command buffers that use the same
/// image must still be recorded one after the other and submitted in that order.
pub(crate) fn compute_barriers<'a>(
    pending: &PendingAccesses,
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

    for image_access in image_acceses.clone().map(|image| image.access) {
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

    let mut image_barriers: SmallVec<[vk::ImageMemoryBarrier2; 1]> = SmallVec::new();
    for image in image_acceses {
        image.take_layouts(|mips, old_layout| {
            image_barriers.push(transition_barrier(pending, &image.access, mips, old_layout))
        });
    }

    let hazard = !src_stage.is_empty() || !dst_stage.is_empty();
    if !hazard && image_barriers.is_empty() {
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
        src_access = vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE;
    }

    Some(Barriers {
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

/// The barrier that moves `mips` of the image `image_access` accesses from `old_layout` into
/// the layout it asks for.
fn transition_barrier(
    pending: &PendingAccesses,
    image_access: &ImageAccess,
    mips: Range<u32>,
    old_layout: vk::ImageLayout,
) -> vk::ImageMemoryBarrier2<'static> {
    // The transition waits for the image's tracked accesses. Without any, the image may
    // still be in use by an earlier submission, or wait for a semaphore when it comes from
    // the presentation engine, so it waits for everything.
    let mut src_stage = vk::PipelineStageFlags2::NONE;
    let mut src_access = vk::AccessFlags2::NONE;
    for p in pending
        .image_reads
        .iter()
        .chain(pending.image_writes.iter().map(|w| &w.write))
        .filter(|r| r.image == image_access.image)
    {
        src_stage |= p.stage;
        src_access |= p.access;
    }
    if src_stage.is_empty() || old_layout == vk::ImageLayout::PRESENT_SRC_KHR {
        src_stage = vk::PipelineStageFlags2::ALL_COMMANDS;
        src_access = vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE;
    }

    let mut dst_stage = image_access.stage;
    let mut dst_access = image_access.access;
    if dst_stage.is_empty() {
        dst_stage = vk::PipelineStageFlags2::ALL_COMMANDS;
    }
    if dst_access.is_empty() {
        dst_access = vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE;
    }

    if old_layout == vk::ImageLayout::PRESENT_SRC_KHR {
        src_access = vk::AccessFlags2::NONE;
    }
    if image_access.layout == vk::ImageLayout::PRESENT_SRC_KHR {
        dst_access = vk::AccessFlags2::NONE;
    }

    vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)
        .old_layout(old_layout)
        .new_layout(image_access.layout)
        .image(image_access.image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: image_access.aspect,
            base_array_layer: 0,
            layer_count: 1,
            base_mip_level: mips.start,
            level_count: mips.end - mips.start,
        })
}

impl PendingAccesses {
    /// Remembers what `barriers` made the pending writes in their source scope visible to.
    /// A memory barrier covers every write of its source stages and access types, also those
    /// it wasn't recorded for.
    pub(crate) fn made_visible(&mut self, barriers: &Barriers) {
        let memory = barriers.memory.map(|m| {
            (
                m.src_stage_mask,
                m.src_access_mask,
                m.dst_stage_mask,
                m.dst_access_mask,
            )
        });
        if let Some((src_stage, src_access, dst_stage, dst_access)) = memory {
            for w in &mut self.buffer_writes {
                if src_stage.contains(w.write.stage) && src_access.contains(w.write.access) {
                    w.visible_stages |= dst_stage;
                    w.visible_access |= dst_access;
                }
            }
        }
        for w in &mut self.image_writes {
            let image = w.write.image;
            let transitions = barriers
                .images
                .iter()
                .filter(|b| b.image == image)
                .map(|b| {
                    (
                        b.src_stage_mask,
                        b.src_access_mask,
                        b.dst_stage_mask,
                        b.dst_access_mask,
                    )
                });
            for (src_stage, src_access, dst_stage, dst_access) in
                memory.into_iter().chain(transitions)
            {
                if src_stage.contains(w.write.stage) && src_access.contains(w.write.access) {
                    w.visible_stages |= dst_stage;
                    w.visible_access |= dst_access;
                }
            }
        }
    }

    /// Forgets the accesses of an image that is being destroyed.
    pub(crate) fn forget_image(&mut self, image: vk::Image) {
        self.image_reads.retain(|read| read.image != image);
        self.image_writes.retain(|write| write.write.image != image);
    }

    /// Forgets the accesses within the addresses of a buffer that is being destroyed.
    pub(crate) fn forget_buffer(&mut self, range: std::ops::Range<u64>) {
        let outside = |access: &BufferAccess| !range.contains(&access.range.start);
        self.buffer_reads.retain(|read| outside(read));
        self.buffer_writes.retain(|write| outside(&write.write));
    }

    /// Adds the accesses of another queue's submissions, which this queue's have to wait for
    /// too. An access both have is only waited for twice.
    pub(crate) fn append(&mut self, other: PendingAccesses) {
        self.buffer_reads.extend(other.buffer_reads);
        self.image_reads.extend(other.image_reads);
        self.buffer_writes.extend(other.buffer_writes);
        self.image_writes.extend(other.image_writes);
    }

    /// Remembers the accesses of a command that was just recorded, so later commands can be
    /// synchronised against them.
    ///
    /// An older buffer access is only forgotten once a newer one covers its whole range: a
    /// partially overlapped write still has bytes nobody has synchronised with. A covered write
    /// is forgotten by a newer write, a covered read by a newer write, whose barrier waited for
    /// it. A newer read doesn't wait for older reads, so it takes over their stages and accesses.
    pub(crate) fn record(
        &mut self,
        image_acceses: impl Iterator<Item = ImageAccess>,
        buffer_acceses: impl Iterator<Item = BufferAccess>,
    ) {
        for access in buffer_acceses {
            let covers = |a: &BufferAccess| {
                access.range.start <= a.range.start && a.range.end <= access.range.end
            };
            let writes = access.access.intersects(all_write_access());
            let reads = access.access.intersects(all_read_access());

            if writes {
                self.buffer_writes.retain(|a| !covers(&a.write));
                self.buffer_writes.push(PendingWrite::new(access.clone()));
                // An acceleration structure write waited for every read of every structure,
                // and every later one runs at the same build stage, after that barrier.
                let structure_write = access
                    .access
                    .contains(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR);
                self.buffer_reads.retain(|a| {
                    !covers(a) && !(structure_write && a.range == EVERY_ACCELERATION_STRUCTURE)
                });
            }
            if reads {
                let mut read = access.clone();
                self.buffer_reads.retain(|a| {
                    let covered = covers(a);
                    if covered {
                        read.stage |= a.stage;
                        read.access |= a.access;
                    }
                    !covered
                });
                self.buffer_reads.push(read);
            }
        }

        for access in image_acceses {
            let same_image = |a: &ImageAccess| a.image == access.image;
            let writes = access.access.intersects(all_write_access());
            let reads = access.access.intersects(all_read_access());

            if writes {
                self.image_writes.retain(|a| !same_image(&a.write));
                self.image_writes.push(PendingWrite::new(access.clone()));
                self.image_reads.retain(|a| !same_image(a));
            }
            if reads {
                let mut read = access.clone();
                self.image_reads.retain(|a| {
                    let same = same_image(a);
                    if same {
                        read.stage |= a.stage;
                        read.access |= a.access;
                    }
                    !same
                });
                self.image_reads.push(read);
            }
        }
    }
}

fn all_write_access() -> vk::AccessFlags2 {
    vk::AccessFlags2::SHADER_WRITE
        | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
        | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
        | vk::AccessFlags2::TRANSFER_WRITE
        | vk::AccessFlags2::HOST_WRITE
        | vk::AccessFlags2::MEMORY_WRITE
        | vk::AccessFlags2::SHADER_STORAGE_WRITE
        | vk::AccessFlags2::TRANSFORM_FEEDBACK_WRITE_EXT
        | vk::AccessFlags2::TRANSFORM_FEEDBACK_COUNTER_WRITE_EXT
        | vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR
        | vk::AccessFlags2::COMMAND_PREPROCESS_WRITE_NV
        | vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR
        | vk::AccessFlags2::VIDEO_ENCODE_WRITE_KHR
        | vk::AccessFlags2::MICROMAP_WRITE_EXT
        | vk::AccessFlags2::OPTICAL_FLOW_WRITE_NV
}

fn all_read_access() -> vk::AccessFlags2 {
    vk::AccessFlags2::INDIRECT_COMMAND_READ
        | vk::AccessFlags2::INDEX_READ
        | vk::AccessFlags2::VERTEX_ATTRIBUTE_READ
        | vk::AccessFlags2::UNIFORM_READ
        | vk::AccessFlags2::INPUT_ATTACHMENT_READ
        | vk::AccessFlags2::SHADER_READ
        | vk::AccessFlags2::COLOR_ATTACHMENT_READ
        | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
        | vk::AccessFlags2::TRANSFER_READ
        | vk::AccessFlags2::HOST_READ
        | vk::AccessFlags2::MEMORY_READ
        | vk::AccessFlags2::SHADER_SAMPLED_READ
        | vk::AccessFlags2::SHADER_STORAGE_READ
        | vk::AccessFlags2::SHADER_BINDING_TABLE_READ_KHR
        | vk::AccessFlags2::TRANSFORM_FEEDBACK_COUNTER_READ_EXT
        | vk::AccessFlags2::CONDITIONAL_RENDERING_READ_EXT
        | vk::AccessFlags2::COLOR_ATTACHMENT_READ_NONCOHERENT_EXT
        | vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR
        | vk::AccessFlags2::FRAGMENT_DENSITY_MAP_READ_EXT
        | vk::AccessFlags2::FRAGMENT_SHADING_RATE_ATTACHMENT_READ_KHR
        | vk::AccessFlags2::COMMAND_PREPROCESS_READ_NV
        | vk::AccessFlags2::DESCRIPTOR_BUFFER_READ_EXT
        | vk::AccessFlags2::INVOCATION_MASK_READ_HUAWEI
        | vk::AccessFlags2::VIDEO_DECODE_READ_KHR
        | vk::AccessFlags2::VIDEO_ENCODE_READ_KHR
        | vk::AccessFlags2::MICROMAP_READ_EXT
        | vk::AccessFlags2::OPTICAL_FLOW_READ_NV
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::usage::ImageUsage;
    use std::iter;
    use crate::image::{
        format::{self, D32Sfloat, D32SfloatS8Uint, R8G8B8A8Unorm, StaticFormat},
        usage::{self, ColorAttachment, DepthAttachment, StaticUsage},
    };
    use ash::vk::Handle;
    use std::marker::PhantomData;
    use std::sync::atomic::AtomicU32;
    use vk::{AccessFlags2 as A, ImageLayout as L, PipelineStageFlags2 as S};

    // ---- helpers -------------------------------------------------------------------------

    fn buffer(stage: S, access: A, range: std::ops::Range<u64>) -> BufferAccess {
        BufferAccess {
            stage,
            access,
            range: range.into(),
        }
    }

    fn read(range: std::ops::Range<u64>) -> BufferAccess {
        buffer(S::VERTEX_SHADER, A::SHADER_STORAGE_READ, range)
    }

    fn write(range: std::ops::Range<u64>) -> BufferAccess {
        buffer(S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE, range)
    }

    /// An access of image `id`, whose tracked layout is `old_layout` until it is recorded.
    fn image(id: u64, stage: S, access: A, old_layout: L, layout: L) -> ImageUse<'static> {
        ImageUse {
            access: ImageAccess {
                stage,
                access,
                image: vk::Image::from_raw(id),
                layout,
                aspect: vk::ImageAspectFlags::COLOR,
                mip_range: (0..1).into(),
            },
            layouts: std::slice::from_ref(Box::leak(Box::new(self::layout(old_layout)))),
        }
    }

    fn image_read(id: u64) -> ImageUse<'static> {
        image(
            id,
            S::FRAGMENT_SHADER,
            A::SHADER_SAMPLED_READ,
            L::GENERAL,
            L::GENERAL,
        )
    }

    fn image_write(id: u64) -> ImageUse<'static> {
        image(
            id,
            S::COMPUTE_SHADER,
            A::SHADER_STORAGE_WRITE,
            L::GENERAL,
            L::GENERAL,
        )
    }

    fn pending(buffers: &[BufferAccess], images: &[ImageUse]) -> PendingAccesses {
        let mut pending = PendingAccesses::default();
        pending.record(
            images.iter().map(|image| image.access.clone()),
            buffers.iter().cloned(),
        );
        pending
    }

    fn buffer_barrier(pending: &PendingAccesses, access: &BufferAccess) -> Option<Barriers> {
        compute_barriers(pending, iter::empty(), iter::once(access.clone()))
    }

    fn image_barrier(pending: &PendingAccesses, access: &ImageUse) -> Option<Barriers> {
        compute_barriers(pending, iter::once(access.clone()), iter::empty())
    }

    /// A command buffer that is never handed to Vulkan: enough for state-only code paths.
    fn offline_cmd() -> CommandBuffer {
        CommandBuffer::new(vk::CommandBuffer::null(), PendingAccesses::default(), 0, 0)
    }

    fn view<F: StaticFormat, U: StaticUsage>(layout: &AtomicU32) -> ImageView<'_, F, U> {
        dynamic_view(layout, F::default(), U::default())
    }

    fn dynamic_view<F: Format, U: ImageUsage>(
        layout: &AtomicU32,
        format: F,
        usage: U,
    ) -> ImageView<'_, F, U> {
        ImageView {
            image: vk::Image::from_raw(1),
            view: vk::ImageView::from_raw(2),
            mip_range: (0..1).into(),
            extent: EXTENT,
            handle: crate::bindless::BindlessHandle::none(),
            layouts: std::slice::from_ref(layout),
            format,
            usage,
            dim: crate::image::D2,
        }
    }

    fn layout(layout: L) -> AtomicU32 {
        AtomicU32::new(layout.as_raw() as u32)
    }

    // ---- access classification -----------------------------------------------------------

    #[test]
    fn read_and_write_masks_are_disjoint_and_cover_the_accesses_lava_uses() {
        assert!((all_read_access() & all_write_access()).is_empty());
        for access in [
            A::SHADER_STORAGE_READ,
            A::SHADER_SAMPLED_READ,
            A::TRANSFER_READ,
            A::INDEX_READ,
            A::INDIRECT_COMMAND_READ,
            A::COLOR_ATTACHMENT_READ,
            A::DEPTH_STENCIL_ATTACHMENT_READ,
        ] {
            assert!(all_read_access().contains(access), "{access:?}");
        }
        for access in [
            A::SHADER_STORAGE_WRITE,
            A::TRANSFER_WRITE,
            A::COLOR_ATTACHMENT_WRITE,
            A::DEPTH_STENCIL_ATTACHMENT_WRITE,
        ] {
            assert!(all_write_access().contains(access), "{access:?}");
        }
    }

    // ---- buffer hazards ------------------------------------------------------------------

    #[test]
    fn first_access_needs_no_barrier() {
        let nothing = PendingAccesses::default();
        assert!(buffer_barrier(&nothing, &read(0..64)).is_none());
        assert!(buffer_barrier(&nothing, &write(0..64)).is_none());
        assert!(image_barrier(&nothing, &image_write(1)).is_none());
    }

    #[test]
    fn read_after_write_needs_a_barrier_from_the_writer_to_the_reader() {
        let barriers = buffer_barrier(&pending(&[write(0..100)], &[]), &read(10..20)).unwrap();
        assert_eq!(barriers.memory.unwrap().src_stage_mask, S::COMPUTE_SHADER);
        assert_eq!(
            barriers.memory.unwrap().src_access_mask,
            A::SHADER_STORAGE_WRITE
        );
        assert_eq!(barriers.memory.unwrap().dst_stage_mask, S::VERTEX_SHADER);
        assert_eq!(
            barriers.memory.unwrap().dst_access_mask,
            A::SHADER_STORAGE_READ
        );
        assert!(barriers.images.is_empty());
    }

    #[test]
    fn write_after_write_needs_a_barrier() {
        let transfer = buffer(S::TRANSFER, A::TRANSFER_WRITE, 50..60);
        let barriers = buffer_barrier(&pending(&[write(0..100)], &[]), &transfer).unwrap();
        assert_eq!(barriers.memory.unwrap().src_stage_mask, S::COMPUTE_SHADER);
        assert_eq!(barriers.memory.unwrap().dst_stage_mask, S::TRANSFER);
        assert_eq!(barriers.memory.unwrap().dst_access_mask, A::TRANSFER_WRITE);
    }

    #[test]
    fn write_after_read_needs_a_barrier_from_the_reader() {
        let barriers = buffer_barrier(&pending(&[read(0..100)], &[]), &write(0..8)).unwrap();
        assert_eq!(barriers.memory.unwrap().src_stage_mask, S::VERTEX_SHADER);
        assert_eq!(
            barriers.memory.unwrap().src_access_mask,
            A::SHADER_STORAGE_READ
        );
        assert_eq!(barriers.memory.unwrap().dst_stage_mask, S::COMPUTE_SHADER);
    }

    // ---- acceleration structures ---------------------------------------------------------

    const BUILD: S = S::ACCELERATION_STRUCTURE_BUILD_KHR;

    fn structure_write(range: std::ops::Range<u64>) -> BufferAccess {
        buffer(BUILD, A::ACCELERATION_STRUCTURE_WRITE_KHR, range)
    }

    /// What a trace (or TLAS build) adds for the BLASes it reaches.
    fn every_structure_read() -> BufferAccess {
        BufferAccess {
            stage: S::COMPUTE_SHADER,
            access: A::ACCELERATION_STRUCTURE_READ_KHR,
            range: EVERY_ACCELERATION_STRUCTURE,
        }
    }

    #[test]
    fn a_trace_waits_for_the_build_of_any_structure() {
        let barriers = buffer_barrier(
            &pending(&[structure_write(0..256)], &[]),
            &every_structure_read(),
        )
        .unwrap()
        .memory
        .unwrap();
        assert_eq!(
            (barriers.src_stage_mask, barriers.src_access_mask),
            (BUILD, A::ACCELERATION_STRUCTURE_WRITE_KHR)
        );
        assert_eq!(
            (barriers.dst_stage_mask, barriers.dst_access_mask),
            (S::COMPUTE_SHADER, A::ACCELERATION_STRUCTURE_READ_KHR)
        );
    }

    #[test]
    fn a_trace_after_a_rebuild_between_two_traces_waits_for_it() {
        let tlas = 4096..8192;
        let trace = || {
            [
                buffer(
                    S::COMPUTE_SHADER,
                    A::ACCELERATION_STRUCTURE_READ_KHR,
                    tlas.clone(),
                ),
                every_structure_read(),
            ]
        };
        let build = |scratch: std::ops::Range<u64>| {
            [
                structure_write(tlas.clone()),
                buffer(
                    BUILD,
                    A::ACCELERATION_STRUCTURE_READ_KHR | A::ACCELERATION_STRUCTURE_WRITE_KHR,
                    scratch,
                ),
                buffer(BUILD, A::SHADER_READ, 0..64),
                BufferAccess {
                    stage: BUILD,
                    access: A::ACCELERATION_STRUCTURE_READ_KHR,
                    range: EVERY_ACCELERATION_STRUCTURE,
                },
            ]
        };
        let mut pending = PendingAccesses::default();
        flush(&mut pending, &[], &build(100_000..101_000));
        assert!(flush(&mut pending, &[], &trace()).is_some());
        assert!(flush(&mut pending, &[], &build(200_000..201_000)).is_some());
        let barrier = flush(&mut pending, &[], &trace())
            .and_then(|b| b.memory)
            .expect("the second trace waits for the rebuild");
        assert!(barrier.src_stage_mask.contains(BUILD));
        assert!(
            barrier
                .src_access_mask
                .contains(A::ACCELERATION_STRUCTURE_WRITE_KHR)
        );
    }

    #[test]
    fn a_trace_ignores_unrelated_storage_writes() {
        let pending = pending(&[write(0..u64::MAX - 1)], &[]);
        assert!(buffer_barrier(&pending, &every_structure_read()).is_none());
    }

    #[test]
    fn a_structure_write_waits_for_pending_traces() {
        let barriers = buffer_barrier(
            &pending(&[every_structure_read()], &[]),
            &structure_write(0..256),
        )
        .unwrap()
        .memory
        .unwrap();
        assert_eq!(barriers.src_stage_mask, S::COMPUTE_SHADER);
        assert_eq!(barriers.dst_stage_mask, BUILD);
        // Other writes don't.
        assert!(buffer_barrier(&pending(&[every_structure_read()], &[]), &write(0..256)).is_none());
    }

    #[test]
    fn a_structure_write_retires_the_read_of_every_structure() {
        let mut pending = pending(&[every_structure_read(), read(0..64)], &[]);
        pending.record(iter::empty(), iter::once(structure_write(1024..2048)));
        assert!(
            pending
                .buffer_reads
                .iter()
                .all(|r| r.range != EVERY_ACCELERATION_STRUCTURE)
        );
        // The unrelated read stays, and a later write to it still waits.
        assert!(buffer_barrier(&pending, &write(0..8)).is_some());
    }

    #[test]
    fn the_read_of_every_structure_absorbs_no_other_read() {
        let pending = pending(&[read(0..64), every_structure_read()], &[]);
        assert_eq!(pending.buffer_reads.len(), 2);
        let barriers = buffer_barrier(&pending, &buffer(S::TRANSFER, A::TRANSFER_WRITE, 0..8))
            .unwrap()
            .memory
            .unwrap();
        assert_eq!(barriers.src_stage_mask, S::VERTEX_SHADER);
    }

    #[test]
    fn read_after_read_needs_no_barrier() {
        assert!(buffer_barrier(&pending(&[read(0..100)], &[]), &read(0..100)).is_none());
    }

    #[test]
    fn disjoint_and_adjacent_ranges_do_not_conflict() {
        let pending = pending(&[write(100..200)], &[]);
        assert!(buffer_barrier(&pending, &read(0..100)).is_none());
        assert!(buffer_barrier(&pending, &write(200..300)).is_none());
        assert!(buffer_barrier(&pending, &read(1000..2000)).is_none());
        // One shared byte is enough.
        assert!(buffer_barrier(&pending, &read(0..101)).is_some());
        assert!(buffer_barrier(&pending, &read(199..300)).is_some());
    }

    #[test]
    fn barrier_merges_every_conflicting_pending_access() {
        let pending = pending(
            &[
                write(0..10),
                buffer(S::TRANSFER, A::TRANSFER_WRITE, 10..20),
                buffer(S::FRAGMENT_SHADER, A::SHADER_STORAGE_WRITE, 500..600),
            ],
            &[],
        );
        let barriers = buffer_barrier(&pending, &read(0..20)).unwrap();
        assert_eq!(
            barriers.memory.unwrap().src_stage_mask,
            S::COMPUTE_SHADER | S::TRANSFER
        );
        assert_eq!(
            barriers.memory.unwrap().src_access_mask,
            A::SHADER_STORAGE_WRITE | A::TRANSFER_WRITE
        );
    }

    #[test]
    fn read_write_access_conflicts_with_pending_reads_and_writes() {
        let read_write = buffer(
            S::COMPUTE_SHADER,
            A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE,
            0..10,
        );
        assert!(buffer_barrier(&pending(&[read(0..10)], &[]), &read_write).is_some());
        assert!(buffer_barrier(&pending(&[write(0..10)], &[]), &read_write).is_some());
    }

    // ---- access recording ----------------------------------------------------------------

    #[test]
    fn accesses_are_recorded_by_kind() {
        let read_write = buffer(
            S::COMPUTE_SHADER,
            A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE,
            0..10,
        );
        let pending = pending(&[read(20..30), write(40..50), read_write], &[]);
        assert_eq!(pending.buffer_reads.len(), 2);
        assert_eq!(pending.buffer_writes.len(), 2);
    }

    #[test]
    fn a_covering_access_replaces_the_older_ones() {
        let pending = pending(&[write(10..20), write(30..40), write(0..100)], &[]);
        assert_eq!(pending.buffer_writes.len(), 1);
        assert_eq!(pending.buffer_writes[0].write.range, (0u64..100).into());

        // Same range again: still a single entry, lists don't grow frame over frame.
        let mut pending = pending;
        for _ in 0..10 {
            pending.record(iter::empty(), [write(0..100), read(0..100)].into_iter());
        }
        assert_eq!(pending.buffer_writes.len(), 1);
        assert_eq!(pending.buffer_reads.len(), 1);
    }

    /// Regression: a write that only partially overlapped an older pending write used to
    /// erase it, so a later read of the untouched part got no barrier.
    #[test]
    fn a_partially_overlapping_write_keeps_the_older_write_pending() {
        let pending = pending(
            &[
                write(0..100),
                buffer(S::TRANSFER, A::TRANSFER_WRITE, 50..60),
            ],
            &[],
        );
        assert_eq!(pending.buffer_writes.len(), 2);

        let barriers =
            buffer_barrier(&pending, &read(0..10)).expect("the first write is unsynchronised");
        assert_eq!(barriers.memory.unwrap().src_stage_mask, S::COMPUTE_SHADER);

        let barriers = buffer_barrier(&pending, &read(52..58)).unwrap();
        assert!(
            barriers
                .memory
                .unwrap()
                .src_stage_mask
                .contains(S::TRANSFER)
        );
    }

    #[test]
    fn forgetting_drops_only_the_accesses_of_the_destroyed_object() {
        let mut p = pending(
            &[read(0..10), write(10..20), read(100..110), write(120..130)],
            &[image_read(1), image_write(2), image_read(3), image_write(3)],
        );
        p.forget_buffer(100..200);
        p.forget_image(vk::Image::from_raw(3));
        assert_eq!(p.buffer_reads.len(), 1);
        assert_eq!(p.buffer_reads[0].range, (0u64..10).into());
        assert_eq!(p.buffer_writes.len(), 1);
        assert_eq!(p.buffer_writes[0].write.range, (10u64..20).into());
        assert_eq!(p.image_reads.len(), 1);
        assert_eq!(p.image_reads[0].image, vk::Image::from_raw(1));
        assert_eq!(p.image_writes.len(), 1);
        assert_eq!(p.image_writes[0].write.image, vk::Image::from_raw(2));
    }

    #[test]
    fn image_accesses_replace_older_ones_of_the_same_image_only() {
        let other_stage = image(1, S::TRANSFER, A::TRANSFER_WRITE, L::GENERAL, L::GENERAL);
        let pending = pending(&[], &[image_write(1), image_write(2), other_stage]);
        assert_eq!(pending.image_writes.len(), 2);
        let first = pending
            .image_writes
            .iter()
            .find(|a| a.write.image == vk::Image::from_raw(1))
            .unwrap();
        assert_eq!(first.write.stage, S::TRANSFER);
    }

    /// Regression: a read replaced the older reads of the same image, so a fragment read
    /// followed by a compute read let a later write wait for the compute read only.
    #[test]
    fn a_write_waits_for_every_earlier_read() {
        let compute_read = image(
            1,
            S::COMPUTE_SHADER,
            A::SHADER_STORAGE_READ,
            L::GENERAL,
            L::GENERAL,
        );
        let p = pending(&[], &[image_read(1), compute_read]);
        assert_eq!(p.image_reads.len(), 1);
        let barriers = image_barrier(&p, &image_write(1)).unwrap();
        assert_eq!(
            barriers.memory.unwrap().src_stage_mask,
            S::FRAGMENT_SHADER | S::COMPUTE_SHADER
        );

        let compute_read = buffer(S::COMPUTE_SHADER, A::SHADER_STORAGE_READ, 0..100);
        let p = pending(&[read(0..10), compute_read], &[]);
        assert_eq!(p.buffer_reads.len(), 1);
        let barriers = buffer_barrier(&p, &write(0..10)).unwrap();
        assert_eq!(
            barriers.memory.unwrap().src_stage_mask,
            S::VERTEX_SHADER | S::COMPUTE_SHADER
        );
    }

    #[test]
    fn a_write_forgets_the_reads_it_waited_for() {
        let p = pending(&[], &[image_read(1), image_write(1)]);
        assert!(p.image_reads.is_empty());

        let p = pending(&[read(0..10), read(20..30), write(0..15)], &[]);
        assert_eq!(p.buffer_reads.len(), 1);
        assert_eq!(p.buffer_reads[0].range, (20u64..30).into());
    }

    // ---- image hazards and layout transitions --------------------------------------------

    #[test]
    fn image_hazards_are_per_image() {
        let pending = pending(&[], &[image_write(1)]);
        let barriers = image_barrier(&pending, &image_read(1)).unwrap();
        assert_eq!(barriers.memory.unwrap().src_stage_mask, S::COMPUTE_SHADER);
        assert_eq!(barriers.memory.unwrap().dst_stage_mask, S::FRAGMENT_SHADER);
        assert!(barriers.images.is_empty(), "same layout: no transition");

        assert!(image_barrier(&pending, &image_read(2)).is_none());
    }

    #[test]
    fn image_write_after_read_needs_a_barrier_but_read_after_read_does_not() {
        let pending = pending(&[], &[image_read(1)]);
        assert!(image_barrier(&pending, &image_write(1)).is_some());
        assert!(image_barrier(&pending, &image_read(1)).is_none());
    }

    #[test]
    fn layout_change_emits_a_transition_for_the_whole_image() {
        let access = image(7, S::TRANSFER, A::TRANSFER_WRITE, L::UNDEFINED, L::GENERAL);
        let barriers = image_barrier(&PendingAccesses::default(), &access).unwrap();
        assert_eq!(barriers.images.len(), 1);
        let transition = &barriers.images[0];
        assert_eq!(transition.image, vk::Image::from_raw(7));
        assert_eq!(
            (transition.old_layout, transition.new_layout),
            (L::UNDEFINED, L::GENERAL)
        );
        assert_eq!(transition.dst_stage_mask, S::TRANSFER);
        assert_eq!(transition.dst_access_mask, A::TRANSFER_WRITE);
        assert_eq!(
            transition.subresource_range.aspect_mask,
            vk::ImageAspectFlags::COLOR
        );
        // The access's mips: the helper's image has one.
        let range = transition.subresource_range;
        assert_eq!((range.base_mip_level, range.level_count), (0, 1));
        assert_eq!((range.base_array_layer, range.layer_count), (0, 1));
    }

    /// Mips of a per-mip image that are in different layouts get one transition per run of
    /// neighbours in the same layout, and the mips already in the wanted layout none.
    #[test]
    fn transitions_cover_runs_of_mips_in_the_same_layout() {
        let layouts = [
            L::UNDEFINED,
            L::UNDEFINED,
            L::GENERAL,
            vk::ImageLayout::ATTACHMENT_OPTIMAL,
            L::UNDEFINED,
        ]
        .map(layout);
        let access = |mips: std::ops::Range<u32>, wanted: L| {
            let mut access = image(
                7,
                S::COMPUTE_SHADER,
                A::SHADER_SAMPLED_READ,
                L::GENERAL,
                wanted,
            );
            access.access.mip_range = mips.clone().into();
            access.layouts = &layouts[mips.start as usize..mips.end as usize];
            access
        };
        let transitions = |access: &ImageUse| {
            let barriers = image_barrier(&PendingAccesses::default(), access);
            barriers.map_or(vec![], |barriers| {
                let ranges = barriers.images.iter().map(|b| {
                    let range = b.subresource_range;
                    (range.base_mip_level, range.level_count, b.old_layout)
                });
                ranges.collect()
            })
        };
        assert_eq!(
            transitions(&access(0..5, L::GENERAL)),
            [
                (0, 2, L::UNDEFINED),
                (3, 1, vk::ImageLayout::ATTACHMENT_OPTIMAL),
                (4, 1, L::UNDEFINED)
            ]
        );
        assert_eq!(transitions(&access(0..5, L::GENERAL)), []);
        // A view of one mip moves only that mip.
        assert_eq!(
            transitions(&access(3..4, vk::ImageLayout::ATTACHMENT_OPTIMAL)),
            [(3, 1, L::GENERAL)]
        );
        assert_eq!(
            transitions(&access(2..5, L::GENERAL)),
            [(3, 1, vk::ImageLayout::ATTACHMENT_OPTIMAL)]
        );
    }

    #[test]
    fn transition_waits_for_the_pending_accesses_of_that_image() {
        let pending = pending(&[], &[image_write(7), image_write(8)]);
        let access = image(
            7,
            S::TRANSFER,
            A::TRANSFER_READ,
            L::GENERAL,
            L::TRANSFER_SRC_OPTIMAL,
        );
        let barriers = image_barrier(&pending, &access).unwrap();
        let transition = &barriers.images[0];
        assert!(transition.src_stage_mask.contains(S::COMPUTE_SHADER));
        assert!(transition.src_access_mask.contains(A::SHADER_STORAGE_WRITE));
    }

    #[test]
    fn transition_waits_only_for_the_tracked_accesses_of_its_image() {
        let pending = pending(&[], &[image_read(7), image_write(8)]);
        let access = image(
            7,
            S::COLOR_ATTACHMENT_OUTPUT,
            A::COLOR_ATTACHMENT_WRITE,
            L::GENERAL,
            L::ATTACHMENT_OPTIMAL,
        );
        let transition = &image_barrier(&pending, &access).unwrap().images[0];
        assert_eq!(transition.src_stage_mask, S::FRAGMENT_SHADER);
        assert_eq!(transition.src_access_mask, A::SHADER_SAMPLED_READ);
    }

    #[test]
    fn a_transition_without_a_hazard_needs_no_memory_barrier() {
        let access = image(7, S::TRANSFER, A::TRANSFER_WRITE, L::UNDEFINED, L::GENERAL);
        let barriers = image_barrier(&PendingAccesses::default(), &access).unwrap();
        assert!(barriers.memory.is_none());
        assert_eq!(barriers.images.len(), 1);
        // A hazard with the image makes one, and the transition still has its own stages.
        let pending = pending(&[], &[image_write(7)]);
        let access = image(
            7,
            S::TRANSFER,
            A::TRANSFER_READ,
            L::GENERAL,
            L::TRANSFER_SRC_OPTIMAL,
        );
        let barriers = image_barrier(&pending, &access).unwrap();
        assert_eq!(barriers.memory.unwrap().src_stage_mask, S::COMPUTE_SHADER);
        assert_eq!(barriers.images[0].src_stage_mask, S::COMPUTE_SHADER);
    }

    #[test]
    fn transition_of_an_untracked_image_waits_for_everything() {
        let access = image(
            7,
            S::COLOR_ATTACHMENT_OUTPUT,
            A::COLOR_ATTACHMENT_WRITE,
            L::GENERAL,
            L::ATTACHMENT_OPTIMAL,
        );
        let transition = &image_barrier(&PendingAccesses::default(), &access)
            .unwrap()
            .images[0];
        assert_eq!(transition.src_stage_mask, S::ALL_COMMANDS);
    }

    // ---- writes a barrier already made visible -------------------------------------------

    /// Records `access` after the barriers it needs, like `CommandBuffer::flush_pending`.
    fn flush(
        pending: &mut PendingAccesses,
        images: &[ImageUse],
        buffers: &[BufferAccess],
    ) -> Option<Barriers> {
        let barriers = compute_barriers(pending, images.iter().cloned(), buffers.iter().cloned());
        if let Some(barriers) = &barriers {
            pending.made_visible(barriers);
        }
        pending.record(
            images.iter().map(|image| image.access.clone()),
            buffers.iter().cloned(),
        );
        barriers
    }

    #[test]
    fn reads_a_barrier_already_made_a_write_visible_to_need_no_second_one() {
        let mut pending = pending(&[write(0..100)], &[]);
        assert!(flush(&mut pending, &[], &[read(0..10)]).is_some());
        assert!(
            flush(&mut pending, &[], &[read(50..60)]).is_none(),
            "same stage and access"
        );

        let indirect = buffer(S::DRAW_INDIRECT, A::INDIRECT_COMMAND_READ, 0..10);
        let barriers =
            flush(&mut pending, &[], &[indirect]).expect("not visible to indirect reads yet");
        assert_eq!(barriers.memory.unwrap().src_stage_mask, S::COMPUTE_SHADER);
    }

    #[test]
    fn a_new_write_is_not_visible_to_anything() {
        let mut pending = pending(&[write(0..100)], &[]);
        flush(&mut pending, &[], &[read(0..100)]);
        flush(&mut pending, &[], &[write(0..100)]);
        assert!(flush(&mut pending, &[], &[read(0..100)]).is_some());
    }

    #[test]
    fn image_writes_become_visible_through_their_transition() {
        let mut pending = pending(&[], &[image_write(1)]);
        let sampled = image(
            1,
            S::FRAGMENT_SHADER,
            A::SHADER_SAMPLED_READ,
            L::GENERAL,
            L::READ_ONLY_OPTIMAL,
        );
        assert!(flush(&mut pending, &[sampled.clone()], &[]).is_some());
        // Recording it moved the tracked layout, so the same access needs no transition again.
        assert!(flush(&mut pending, &[sampled], &[]).is_none());
    }

    #[test]
    fn transition_to_present_has_no_destination_access() {
        let access = image(7, S::empty(), A::empty(), L::GENERAL, L::PRESENT_SRC_KHR);
        let barriers = image_barrier(&pending(&[], &[image_write(7)]), &access).unwrap();
        let transition = &barriers.images[0];
        assert_eq!(transition.new_layout, L::PRESENT_SRC_KHR);
        assert_eq!(transition.dst_access_mask, A::NONE);
        assert!(!transition.dst_stage_mask.is_empty());
    }

    #[test]
    fn transition_from_present_has_no_source_access() {
        let access = image(
            7,
            S::TRANSFER,
            A::TRANSFER_WRITE,
            L::PRESENT_SRC_KHR,
            L::GENERAL,
        );
        let barriers = image_barrier(&PendingAccesses::default(), &access).unwrap();
        assert_eq!(barriers.images[0].src_access_mask, A::NONE);
    }

    #[test]
    fn only_images_that_change_layout_are_transitioned() {
        let accesses = [
            image_read(1),
            image(2, S::TRANSFER, A::TRANSFER_WRITE, L::UNDEFINED, L::GENERAL),
        ];
        let barriers = compute_barriers(
            &PendingAccesses::default(),
            accesses.into_iter(),
            iter::empty(),
        )
        .unwrap();
        assert_eq!(barriers.images.len(), 1);
        assert_eq!(barriers.images[0].image, vk::Image::from_raw(2));
    }

    /// Recording, not building, moves an image's tracked layout: accesses can be recorded in
    /// another order than they were built, and one built but never recorded changes nothing.
    #[test]
    fn layouts_move_when_accesses_are_recorded() {
        let state = layout(L::UNDEFINED);
        let image = view::<R8G8B8A8Unorm, ColorAttachment>(&state);
        let sampled = image.access(S::FRAGMENT_SHADER, A::SHADER_SAMPLED_READ, L::GENERAL);
        let attachment = image.access(
            S::COLOR_ATTACHMENT_OUTPUT,
            A::COLOR_ATTACHMENT_WRITE,
            vk::ImageLayout::ATTACHMENT_OPTIMAL,
        );
        drop(image.access(S::TRANSFER, A::TRANSFER_READ, L::TRANSFER_SRC_OPTIMAL));

        let nothing = PendingAccesses::default();
        let transition = |access: &ImageUse| {
            let barriers =
                compute_barriers(&nothing, iter::once(access.clone()), iter::empty()).unwrap();
            let [transition] = barriers.images.as_slice() else {
                panic!("one transition");
            };
            (transition.old_layout, transition.new_layout)
        };
        assert_eq!(
            transition(&attachment),
            (L::UNDEFINED, vk::ImageLayout::ATTACHMENT_OPTIMAL)
        );
        assert_eq!(
            transition(&sampled),
            (vk::ImageLayout::ATTACHMENT_OPTIMAL, L::GENERAL)
        );
    }

    #[test]
    fn flush_without_hazards_only_records_and_needs_no_device() {
        let mut cmd = offline_cmd();
        cmd.flush_pending(
            [image_write(1)].into_iter(),
            [write(0..10), read(20..30)].into_iter(),
        );
        assert_eq!(cmd.pending_accesses.buffer_writes.len(), 1);
        assert_eq!(cmd.pending_accesses.buffer_reads.len(), 1);
        assert_eq!(cmd.pending_accesses.image_writes.len(), 1);
    }

    // ---- blits ---------------------------------------------------------------------------

    // ---- raster builder ------------------------------------------------------------------

    const EXTENT: UVec2 = UVec2::new(640, 480);

    /// Bindings for draws that are never recorded.
    fn no_bindings<K>() -> Bindings<'static, K, 4, 0, 0, 0> {
        Bindings::new(0, &0u32, [], [])
    }

    fn slice<T: Pod, U: crate::buffer::usage::StaticUsage>(
        gpu_ptr: u64,
        len: u64,
    ) -> BufferSlice<'static, T, U> {
        BufferSlice {
            address: gpu_ptr,
            size: len * size_of::<T>() as u64,
            usage: U::default(),
            _marker: PhantomData,
        }
    }

    #[test]
    fn raster_covers_the_extent_without_attachments() {
        let mut cmd = offline_cmd();
        let builder = cmd.raster(EXTENT, []);
        assert!(builder.color.is_empty());
        assert_eq!(builder.depth_format, vk::Format::UNDEFINED);
        assert_eq!(builder.stencil_format, vk::Format::UNDEFINED);
        assert!(builder.depth_access().is_none());
        assert_eq!(
            (builder.viewport.offset, builder.viewport.extent),
            (IVec2::ZERO, EXTENT)
        );
        assert_eq!(builder.scissors.as_slice(), [rect(IVec2::ZERO, EXTENT)]);
    }

    #[test]
    fn scissors_replace_the_default_one() {
        let mut cmd = offline_cmd();
        let mut builder = cmd.raster(EXTENT, []);
        builder.scissors(&[Scissor {
            offset: IVec2::new(-3, 7),
            extent: UVec2::new(64, 32),
        }]);
        let [scissor] = builder.scissors.as_slice() else {
            panic!("one scissor");
        };
        assert_eq!((scissor.offset.x, scissor.offset.y), (-3, 7));
        assert_eq!((scissor.extent.width, scissor.extent.height), (64, 32));
    }

    #[test]
    fn raster_state_defaults_to_filled_backface_culled_depth_writing_draws() {
        let state = RasterState::<2>::default();
        assert!(state.backface_culling);
        assert!(!state.wire_frame);
        assert!(state.depth_write);
        assert_eq!(state.blends, [Blend::Replace; 2]);
    }

    #[test]
    fn raster_state_setters_end_up_in_the_pipeline_key() {
        let mut cmd = offline_cmd();
        let builder = cmd.raster(EXTENT, []);
        let state = RasterState::<0>::default()
            .backface_culling(false)
            .wire_frame(true)
            .depth_write(false)
            .widen();
        let key = builder.pipeline_key(&state);
        assert!(!key.backface_culling);
        assert!(key.wire_frame);
        assert!(!key.depth_write);
        assert_ne!(
            key,
            builder.pipeline_key(&RasterState::<0>::default().widen()),
            "different state must not share a pipeline"
        );
    }

    #[test]
    fn blend_modes_get_their_own_pipelines() {
        let state = layout(L::UNDEFINED);
        let mut cmd = offline_cmd();
        let target = view::<R8G8B8A8Unorm, ColorAttachment>(&state);
        let builder = cmd.raster(EXTENT, [Attachment::color(target, None)]);
        let keys = [
            Blend::Replace,
            Blend::Alpha,
            Blend::Add,
            Blend::Attenuate,
            Blend::Skip,
        ]
        .map(|blend| builder.pipeline_key(&RasterState::default().blends([blend]).widen()));
        for (i, key) in keys.iter().enumerate() {
            assert!(!keys[i + 1..].contains(key), "{key:?}");
        }
    }

    #[test]
    fn without_blends_every_attachment_is_replaced() {
        let (a, b) = (layout(L::UNDEFINED), layout(L::UNDEFINED));
        let mut cmd = offline_cmd();
        let builder = cmd.raster(
            EXTENT,
            [
                Attachment::color(view::<R8G8B8A8Unorm, ColorAttachment>(&a), None),
                Attachment::color(view::<R8G8B8A8Unorm, ColorAttachment>(&b), None),
            ],
        );
        let key = builder.pipeline_key(&RasterState::<2>::default().widen());
        assert_eq!(key.blends.as_slice(), [Blend::Replace; 2]);
        let explicit = RasterState::default().blends([Blend::Replace; 2]).widen();
        assert_eq!(key, builder.pipeline_key(&explicit));
    }

    #[test]
    fn color_attachment_records_format_view_and_access() {
        let state = layout(L::UNDEFINED);
        let target = view::<R8G8B8A8Unorm, ColorAttachment>(&state);
        let color = Attachment::color(target, Some([0.0, 0.5, 1.0, 1.0]));

        assert_eq!(color.format, vk::Format::R8G8B8A8_UNORM);
        assert_eq!(color.view, vk::ImageView::from_raw(2));
        assert_eq!(
            unsafe { color.clear.unwrap().color.float32 },
            [0.0, 0.5, 1.0, 1.0]
        );
        let access = &color.access.access;
        assert_eq!(access.stage, S::COLOR_ATTACHMENT_OUTPUT);
        // Cleared attachments are not read.
        assert_eq!(access.access, A::COLOR_ATTACHMENT_WRITE);
        assert_eq!(access.layout, vk::ImageLayout::ATTACHMENT_OPTIMAL);
    }

    /// An image is only in the attachment layout while it is drawn to: sampling it in
    /// between is an access in `GENERAL`, so the next rendering transitions it back.
    #[test]
    fn attachments_are_transitioned_into_the_attachment_layout() {
        let color = layout(L::GENERAL);
        let depth = layout(L::GENERAL);
        let nothing = PendingAccesses::default();
        // The layout transitions recording the rendering makes, without a device.
        let draw = |cmd: &mut CommandBuffer| {
            let target = view::<R8G8B8A8Unorm, ColorAttachment>(&color);
            let mut builder = cmd.raster(EXTENT, [Attachment::color(target, None)]);
            builder.depth_attachment(view::<D32Sfloat, DepthAttachment>(&depth), None);
            let depth = builder.depth_access();
            let accesses = builder.color.iter().map(|c| c.access.clone()).chain(depth);
            compute_barriers(&nothing, accesses, iter::empty()).map_or(vec![], |barriers| {
                let transitions = barriers.images.iter();
                transitions.map(|t| (t.old_layout, t.new_layout)).collect()
            })
        };
        let mut cmd = offline_cmd();
        assert_eq!(
            draw(&mut cmd),
            [(L::GENERAL, vk::ImageLayout::ATTACHMENT_OPTIMAL); 2]
        );
        // A second rendering finds them there.
        assert_eq!(draw(&mut cmd), []);
        let sampled = view::<R8G8B8A8Unorm, ColorAttachment>(&color).access(
            S::FRAGMENT_SHADER,
            A::SHADER_SAMPLED_READ,
            L::GENERAL,
        );
        let mut old = None;
        sampled.take_layouts(|_, layout| old = Some(layout));
        assert_eq!(old, Some(vk::ImageLayout::ATTACHMENT_OPTIMAL));
        assert_eq!(
            draw(&mut cmd),
            [(L::GENERAL, vk::ImageLayout::ATTACHMENT_OPTIMAL)]
        );
    }

    #[test]
    fn loaded_color_attachment_is_also_read() {
        let state = layout(L::GENERAL);
        let color = Attachment::color(view::<R8G8B8A8Unorm, ColorAttachment>(&state), None);
        assert!(color.clear.is_none());
        assert_eq!(
            color.access.access.access,
            A::COLOR_ATTACHMENT_WRITE | A::COLOR_ATTACHMENT_READ
        );
    }

    #[test]
    fn read_only_depth_attachment_tests_without_writing() {
        let state = layout(L::GENERAL);
        let mut cmd = offline_cmd();
        let read_only = RasterState::default().depth_write(false);
        let none = no_bindings();
        let mut builder = cmd.raster(EXTENT, []);
        builder
            .depth_attachment(view::<D32Sfloat, DepthAttachment>(&state), None)
            .draw(&none, 3, 1, read_only.clone());

        assert_eq!(builder.depth_format, vk::Format::D32_SFLOAT);
        assert_eq!(builder.stencil_format, vk::Format::UNDEFINED);
        assert!(builder.clear_depth.is_none());
        let access = builder.depth_access().unwrap().access;
        assert_eq!(access.access, A::DEPTH_STENCIL_ATTACHMENT_READ);
        assert_eq!(
            access.stage,
            S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS
        );
        // Depth writes are pipeline state, so read-only draws get their own pipeline.
        assert_ne!(
            builder.pipeline_key(&read_only.widen()),
            builder.pipeline_key(&RasterState::<0>::default().widen())
        );
    }

    /// Clearing modifies the attachment even if every draw only tests against it.
    #[test]
    fn cleared_read_only_depth_attachment_is_still_a_write_access() {
        let state = layout(L::UNDEFINED);
        let mut cmd = offline_cmd();
        let none = no_bindings();
        let mut builder = cmd.raster(EXTENT, []);
        builder
            .depth_attachment(view::<D32Sfloat, DepthAttachment>(&state), Some([0.25]))
            .draw(&none, 3, 1, RasterState::default().depth_write(false));
        assert_eq!(
            unsafe { builder.clear_depth.unwrap().depth_stencil.depth },
            0.25
        );
        assert_eq!(
            builder.depth_access().unwrap().access.access,
            A::DEPTH_STENCIL_ATTACHMENT_READ | A::DEPTH_STENCIL_ATTACHMENT_WRITE
        );
    }

    #[test]
    fn dynamic_attachments_work_like_static_ones() {
        let (color, depth) = (layout(L::UNDEFINED), layout(L::UNDEFINED));
        let mut cmd = offline_cmd();
        let clear = format::ClearValue::Float([0.0, 0.5, 1.0, 1.0]);
        let target = dynamic_view(
            &color,
            format::Dynamic(vk::Format::R8G8B8A8_UNORM),
            usage::Dynamic(vk::ImageUsageFlags::COLOR_ATTACHMENT),
        );
        let mut builder = cmd.raster(EXTENT, [Attachment::color(target, Some(clear))]);
        builder.depth_attachment(
            dynamic_view(
                &depth,
                format::Dynamic(vk::Format::D32_SFLOAT_S8_UINT),
                DepthAttachment,
            ),
            None,
        );
        assert_eq!(builder.color[0].format, vk::Format::R8G8B8A8_UNORM);
        assert_eq!(
            unsafe { builder.color[0].clear.unwrap().color.float32 },
            [0.0, 0.5, 1.0, 1.0]
        );
        assert_eq!(builder.depth_format, vk::Format::D32_SFLOAT_S8_UINT);
        assert_eq!(builder.stencil_format, vk::Format::D32_SFLOAT_S8_UINT);
    }

    #[test]
    #[should_panic(expected = "lacks COLOR_ATTACHMENT")]
    fn dynamic_usage_without_the_attachment_flag_panics() {
        let state = layout(L::UNDEFINED);
        let _ = Attachment::color(
            dynamic_view(
                &state,
                R8G8B8A8Unorm,
                usage::Dynamic(vk::ImageUsageFlags::SAMPLED),
            ),
            None,
        );
    }

    #[test]
    #[should_panic(expected = "lacks the aspects DEPTH")]
    fn dynamic_colour_format_as_depth_attachment_panics() {
        let state = layout(L::UNDEFINED);
        let mut cmd = offline_cmd();
        cmd.raster(EXTENT, []).depth_attachment(
            dynamic_view(
                &state,
                format::Dynamic(vk::Format::R8G8B8A8_UNORM),
                DepthAttachment,
            ),
            None,
        );
    }

    #[test]
    fn one_depth_writing_draw_makes_the_attachment_a_write_access() {
        let state = layout(L::UNDEFINED);
        let mut cmd = offline_cmd();
        let none = no_bindings();
        let mut builder = cmd.raster(EXTENT, []);
        builder
            .depth_attachment(view::<D32Sfloat, DepthAttachment>(&state), None)
            .draw(&none, 3, 1, RasterState::default())
            .draw(&none, 3, 1, RasterState::default().depth_write(false));
        assert_eq!(
            builder.depth_access().unwrap().access.access,
            A::DEPTH_STENCIL_ATTACHMENT_READ | A::DEPTH_STENCIL_ATTACHMENT_WRITE
        );
    }

    /// Regression: the stencil format was never set, so pipelines for depth-stencil
    /// attachments were created with a mismatching stencil format.
    #[test]
    fn depth_stencil_attachment_sets_the_stencil_format_too() {
        let state = layout(L::UNDEFINED);
        let mut cmd = offline_cmd();
        let mut builder = cmd.raster(EXTENT, []);
        builder.depth_attachment(
            view::<D32SfloatS8Uint, DepthAttachment>(&state),
            Some((0.0, 0)),
        );
        assert_eq!(builder.depth_format, vk::Format::D32_SFLOAT_S8_UINT);
        assert_eq!(builder.stencil_format, vk::Format::D32_SFLOAT_S8_UINT);
    }

    /// A declared access is tracked at once; its barrier waits for the next command's.
    #[test]
    fn declared_accesses_are_tracked_and_their_barriers_held_back() {
        use crate::buffer::usage::Storage;
        let mut cmd = offline_cmd();
        let buffer = slice::<u32, Storage>(0x1000, 4);
        cmd.record_access(Access::buffer_write(buffer), Stage::Compute);
        assert!(cmd.deferred.is_none(), "a first access needs no barrier");
        assert_eq!(cmd.pending_accesses.buffer_writes.len(), 1);

        cmd.record_access(Access::buffer_read(buffer), Stage::Raster);
        let memory = cmd.deferred.take().unwrap().memory.unwrap();
        assert_eq!(memory.src_stage_mask, S::COMPUTE_SHADER);
        assert_eq!(
            memory.dst_stage_mask,
            S::PRE_RASTERIZATION_SHADERS | S::FRAGMENT_SHADER
        );
        assert_eq!(memory.dst_access_mask, A::SHADER_STORAGE_READ);
    }

    /// The draw list is the command buffer's, lent to each rendering and given back by
    /// `record`: draws can be added in a loop, and the next rendering starts with none but
    /// keeps the allocation.
    #[test]
    fn renderings_reuse_the_command_buffers_draw_list() {
        let mut cmd = offline_cmd();
        let none = no_bindings();
        let mut builder = cmd.raster(EXTENT, []);
        for _ in 0..5 {
            builder.draw(&none, 3, 1, RasterState::default());
        }
        assert_eq!(builder.draws.len(), 5);
        let capacity = builder.draws.capacity();
        // What `record` does after recording.
        builder.cmd_buf.draws = recycle(std::mem::take(&mut builder.draws));
        let builder = cmd.raster(EXTENT, []);
        assert!(builder.draws.is_empty());
        assert_eq!(builder.draws.capacity(), capacity);
    }

    #[test]
    fn draws_keep_their_order_and_access_their_command_buffers() {
        use crate::buffer::usage::{Index, Indirect};
        let mut cmd = offline_cmd();
        let (vertex, mesh) = (no_bindings(), no_bindings());
        let mut builder = cmd.raster(EXTENT, []);
        builder
            .draw_indexed(
                &vertex,
                slice::<u32, Index>(0x1000, 6),
                2,
                RasterState::default(),
            )
            .launch_indirect(
                &mesh,
                slice::<DispatchIndirectCommand, Indirect>(0x2000, 1),
                RasterState::default(),
            );
        let [indexed, launch] = builder.draws.as_slice() else {
            panic!("two draws");
        };
        let DrawCommand::Indexed {
            indices,
            flags,
            count: 6,
            instances: 2,
        } = indexed.command
        else {
            panic!("not the indexed draw: {:?}", indexed.command);
        };
        assert_eq!((indices.address, indices.size), (0x1000, 24));
        assert_eq!(
            flags,
            vk::AddressCommandFlagsKHR::UNKNOWN_STORAGE_BUFFER_USAGE
                | vk::AddressCommandFlagsKHR::FULLY_BOUND
        );
        assert_eq!(indexed.bindings.sections().buffers().len(), 0);
        let index = &indexed.extra[0];
        assert_eq!((index.stage, index.access), (S::INDEX_INPUT, A::INDEX_READ));
        assert_eq!(index.range, (0x1000u64..0x1018).into());

        let DrawCommand::LaunchIndirect {
            draws, count: 1, ..
        } = launch.command
        else {
            panic!("not the indirect launch: {:?}", launch.command);
        };
        assert_eq!((draws.address, draws.size, draws.stride), (0x2000, 12, 12));
        let indirect = &launch.extra[0];
        assert_eq!(
            (indirect.stage, indirect.access),
            (S::DRAW_INDIRECT, A::INDIRECT_COMMAND_READ)
        );
        assert_eq!(indirect.range.start, 0x2000);
    }

    // ---- layouts read by the GPU ---------------------------------------------------------

    #[test]
    fn indirect_commands_match_the_vulkan_structs() {
        assert_eq!(
            size_of::<DrawIndirectCommand>(),
            size_of::<vk::DrawIndirectCommand>()
        );
        assert_eq!(
            size_of::<DispatchIndirectCommand>(),
            size_of::<vk::DispatchIndirectCommand>()
        );

        let draw = DrawIndirectCommand {
            vertex_count: 1,
            instance_count: 2,
            first_vertex: 3,
            first_instance: 4,
        };
        let raw: vk::DrawIndirectCommand = unsafe { std::mem::transmute(draw) };
        assert_eq!(
            (
                raw.vertex_count,
                raw.instance_count,
                raw.first_vertex,
                raw.first_instance
            ),
            (1, 2, 3, 4)
        );
        let dispatch = DispatchIndirectCommand { x: 5, y: 6, z: 7 };
        let raw: vk::DispatchIndirectCommand = unsafe { std::mem::transmute(dispatch) };
        assert_eq!((raw.x, raw.y, raw.z), (5, 6, 7));
    }
}

//! Command recording for generated passes: barriers, renderings and dispatches (transfers live with the slices and views they take, acceleration structure commands with the structures, `present` with the swapchain)
use std::{
    fmt::Debug,
    iter,
    marker::PhantomData,
    ops::{IntoBounds, RangeBounds},
    range::Range,
    sync::atomic::{AtomicU32, Ordering},
};

use crate::{
    bindings::{
        MAX_PASS_BUFFERS, MAX_PASS_IMAGES, MAX_PUSH_CONSTANTS_SIZE,
        MAX_SPECIALIZATION_CONSTANTS_SIZE, PASS_MAP,
    },
    bindless::Bindless,
    buffer::{
        self,
        slice::BufferSlice,
        usage::{BufferUsage, IsIndex, IsIndirect},
    },
    error::Result,
    image::{
        dim::Dim,
        format::{ColorAspect, DepthAspect, Format},
        slice::ImageView,
        usage::{ImageUsage, IsColorAttachment, IsDepthAttachment},
    },
    pipeline::{Constants, RasterHash, pipelines},
    profiling::{PassQueries, TimestampQueries},
    state::{Ctx, Functions},
    vkobjects::queue::{PendingAccesses, PendingWrite},
};
use ash::vk::{self, IndexType};
use bytemuck::{Pod, Zeroable, bytes_of};
use glam::{IVec2, UVec2};
use lava_macros::validation_trace;
use smallvec::{SmallVec, smallvec};

#[derive(Debug, Clone)]
pub(crate) struct BufferAccess {
    pub(crate) stage: vk::PipelineStageFlags2,
    pub(crate) access: vk::AccessFlags2,
    pub(crate) range: Range<u64>,
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
            !self.range.intersect(other.range).is_empty()
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ImageAccess {
    pub(crate) stage: vk::PipelineStageFlags2,
    pub(crate) access: vk::AccessFlags2,
    pub(crate) image: vk::Image,
    pub(crate) layout: vk::ImageLayout,
    pub(crate) aspect: vk::ImageAspectFlags,
    pub(crate) mip_range: Range<u32>,
}

/// An image access of a command that isn't recorded yet, with the tracked layouts of its mips,
/// which only move when the command is recorded (see [`compute_barriers`]).
#[derive(Debug, Clone)]
pub(crate) struct ImageUse<'a> {
    pub(crate) access: ImageAccess,
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
    pub(crate) timestamps: Option<TimestampQueries>,
    pub(crate) counters: Option<PassQueries>,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Default)]
pub struct ShaderHash {
    pub entry: &'static str,
    pub file: &'static str,
}

/// Marker types for the kind of pass a [`BindingOutput`] was made for, so that it can only be
/// recorded the way its shaders run.
pub mod kind {
    pub struct Compute;
    pub struct RayTracing;
    pub struct RasterVertex;
    pub struct RasterMesh;
}

/// A pass's push constants, specialization constants and the resources it accesses, made by
/// the generated `Pass::push_bindings` (and `constant_bindings` for a pass with
/// specialization constants). It borrows its images' tracked layouts until it is recorded.
pub struct BindingOutput<'a, K> {
    /// Index into `PASS_MAP`.
    pub(crate) pass: usize,
    /// The pass's push-constant struct, zero-padded; always pushed whole.
    pub(crate) push_constants: [u8; MAX_PUSH_CONSTANTS_SIZE],
    /// The pass's specialization constants (laid out by `PassEntry.constants`), zero-padded;
    /// part of the key of its pipeline.
    pub(crate) constants: [u8; MAX_SPECIALIZATION_CONSTANTS_SIZE],
    pub(crate) images: SmallVec<[ImageUse<'a>; MAX_PASS_IMAGES]>,
    pub(crate) buffers: SmallVec<[BufferAccess; MAX_PASS_BUFFERS]>,
    /// Shader stages of the pass; accesses registered later use the same stages.
    pub(crate) stage: vk::PipelineStageFlags2,
    kind: PhantomData<K>,
}

impl<'a, K> BindingOutput<'a, K> {
    pub(crate) fn new<G: Pod, const I: usize, const B: usize>(
        pass: usize,
        bindings: &G,
        stage: vk::PipelineStageFlags2,
        images: [ImageUse<'a>; I],
        buffers: [BufferAccess; B],
    ) -> Self {
        assert!(size_of::<G>() <= MAX_PUSH_CONSTANTS_SIZE);
        let mut push_constants = [0; MAX_PUSH_CONSTANTS_SIZE];
        push_constants[..size_of::<G>()].copy_from_slice(bytes_of(bindings));
        Self {
            pass,
            push_constants,
            constants: [0; MAX_SPECIALIZATION_CONSTANTS_SIZE],
            images: images.into_iter().collect(),
            buffers: buffers.into_iter().collect(),
            stage,
            kind: PhantomData,
        }
    }

    /// Sets the specialization constants and adds the accesses of the resources they name.
    pub(crate) fn with_constants<G: Pod, const I: usize, const B: usize>(
        mut self,
        constants: &G,
        images: [ImageUse<'a>; I],
        buffers: [BufferAccess; B],
    ) -> Self {
        self.constants[..size_of::<G>()].copy_from_slice(bytes_of(constants));
        self.images.extend(images);
        self.buffers.extend(buffers);
        self
    }

    /// Also shortens the borrow, so a longer-lived binding can join a rendering. A `SmallVec`
    /// can't coerce to a shorter lifetime by itself: it is invariant in its items.
    fn erase<'s>(self) -> BindingOutput<'s, ()>
    where
        'a: 's,
    {
        BindingOutput {
            pass: self.pass,
            push_constants: self.push_constants,
            constants: self.constants,
            // SAFETY: only the lifetime changes, which doesn't affect the layout, and it gets
            // shorter ('a: 's) on a value owned here, so nothing can store a shorter-lived
            // borrow where the longer one is expected.
            images: unsafe {
                std::mem::transmute::<
                    SmallVec<[ImageUse<'a>; MAX_PASS_IMAGES]>,
                    SmallVec<[ImageUse<'s>; MAX_PASS_IMAGES]>,
                >(self.images)
            },
            buffers: self.buffers,
            stage: self.stage,
            kind: PhantomData,
        }
    }
}

/// Registering resources the pass reaches without naming them in its push constants, e.g.
/// through a bindless handle or pointer stored in a buffer. Each call adds the access to the
/// barriers recorded before the pass.
impl<'a, K> BindingOutput<'a, K> {
    /// The pass samples `image` (`Tex2D`, or a `DynImg` with a sampled index).
    pub fn sampled_read<F: Format, U: crate::image::usage::IsSampled, D: Dim>(
        self,
        image: ImageView<'a, F, U, D>,
    ) -> Self {
        image.usage_check(vk::ImageUsageFlags::SAMPLED);
        self.with_image(image, vk::AccessFlags2::SHADER_SAMPLED_READ)
    }

    /// The pass loads from `image` as a storage image (`Image2D`, or a `DynImg` without a
    /// sampled index).
    pub fn storage_read<F: Format, U: crate::image::usage::IsStorage, D: Dim>(
        self,
        image: ImageView<'a, F, U, D>,
    ) -> Self {
        image.usage_check(vk::ImageUsageFlags::STORAGE);
        self.with_image(image, vk::AccessFlags2::SHADER_STORAGE_READ)
    }

    /// The pass loads from and stores to `image` (`MutImage2D`).
    pub fn storage_write<F: Format, U: crate::image::usage::IsStorage, D: Dim>(
        self,
        image: ImageView<'a, F, U, D>,
    ) -> Self {
        image.usage_check(vk::ImageUsageFlags::STORAGE);
        self.with_image(
            image,
            vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
        )
    }

    /// The pass traces `tlas` through an address it reads from a buffer (an `Accel` inside a
    /// struct): its memory and the BLASes it names.
    pub fn acceleration_structure_read(
        mut self,
        tlas: &crate::vkobjects::acceleration_structure::Tlas,
    ) -> Self {
        self.buffers.extend(tlas.trace_accesses(self.stage));
        self
    }

    /// The pass reads `buffer` through a pointer (`Buf`).
    pub fn buffer_read<V: Copy + Pod, U: crate::buffer::usage::IsStorage>(
        self,
        buffer: BufferSlice<'_, V, U>,
    ) -> Self {
        buffer.usage_check(vk::BufferUsageFlags::STORAGE_BUFFER);
        self.with_buffer(buffer, vk::AccessFlags2::SHADER_STORAGE_READ)
    }

    /// The pass reads and writes `buffer` through a pointer (`MutBuf`).
    pub fn buffer_write<V: Copy + Pod, U: crate::buffer::usage::IsStorage>(
        self,
        buffer: BufferSlice<'_, V, U>,
    ) -> Self {
        buffer.usage_check(vk::BufferUsageFlags::STORAGE_BUFFER);
        self.with_buffer(
            buffer,
            vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
        )
    }

    fn with_image<F: Format, U: ImageUsage, D: Dim>(
        mut self,
        image: ImageView<'a, F, U, D>,
        access: vk::AccessFlags2,
    ) -> Self {
        self.images
            .push(image.access(self.stage, access, vk::ImageLayout::GENERAL));
        self
    }

    fn with_buffer<V: Copy + Pod, U: BufferUsage>(
        mut self,
        buffer: BufferSlice<'_, V, U>,
        access: vk::AccessFlags2,
    ) -> Self {
        self.buffers.push(buffer.access(self.stage, access));
        self
    }
}

/// One rendering being built: its `COLOR_ATTACHMENTS` and its `DRAWS` so far, which
/// [`Self::record`] records.
pub struct RasterBuilder<'a, const DRAWS: usize, const COLOR_ATTACHMENTS: usize> {
    cmd_buf: &'a mut CommandBuffer,
    extent: UVec2,
    viewport: Viewport,
    scissors: SmallVec<[vk::Rect2D; 1]>,
    color_formats: [vk::Format; COLOR_ATTACHMENTS],
    color_attachments: [(vk::ImageView, Option<vk::ClearValue>); COLOR_ATTACHMENTS],
    color_accesses: [ImageUse<'a>; COLOR_ATTACHMENTS],
    depth_format: vk::Format,
    stencil_format: vk::Format,
    depth_attachment: vk::ImageView,
    /// Only a read; [`Self::depth_access`] adds the write.
    depth_access: Option<ImageUse<'a>>,
    clear_depth: Option<vk::ClearValue>,
    draws: [Draw<'a, COLOR_ATTACHMENTS>; DRAWS],
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
}

struct Draw<'a, const COLOR_ATTACHMENTS: usize> {
    bindings: BindingOutput<'a, ()>,
    command: DrawCommand,
    state: RasterState<COLOR_ATTACHMENTS>,
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

/// `array` with `item` added at the end.
fn append<T, const N: usize>(array: [T; N], item: T) -> [T; N + 1] {
    let mut items = array.into_iter().chain(iter::once(item));
    std::array::from_fn(|_| items.next().unwrap())
}

impl<'a, const COLOR_ATTACHMENTS: usize> RasterBuilder<'a, 0, COLOR_ATTACHMENTS> {
    /// Color attachments come before the draws, since every draw has a blend for each.
    pub fn color_attachment<F: Format, U>(
        self,
        image: ImageView<'a, F, U>,
        clear: Option<F::Texels>,
    ) -> RasterBuilder<'a, 0, { COLOR_ATTACHMENTS + 1 }>
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
        RasterBuilder {
            cmd_buf: self.cmd_buf,
            extent: self.extent,
            viewport: self.viewport,
            scissors: self.scissors,
            color_formats: append(self.color_formats, image.format().vk_format()),
            color_attachments: append(
                self.color_attachments,
                (
                    image.view,
                    clear.map(|e| image.format().clear_value(e).into()),
                ),
            ),
            color_accesses: append(self.color_accesses, access),
            depth_format: self.depth_format,
            stencil_format: self.stencil_format,
            depth_attachment: self.depth_attachment,
            depth_access: self.depth_access,
            clear_depth: self.clear_depth,
            draws: [],
        }
    }
}

impl<'a, const DRAWS: usize, const COLOR_ATTACHMENTS: usize>
    RasterBuilder<'a, DRAWS, COLOR_ATTACHMENTS>
{
    /// Depth-tests the draws against `image` (nearer = greater). `clear` resets the depth first.
    pub fn depth_attachment<F: Format, U>(
        mut self,
        image: ImageView<'a, F, U>,
        clear: Option<F::Texels>,
    ) -> Self
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
    pub fn viewport(mut self, viewport: Viewport) -> Self {
        self.viewport = viewport;
        self
    }

    /// Replaces the default scissor, the whole render area.
    pub fn scissors(mut self, scissors: &[Scissor]) -> Self {
        self.scissors = scissors.iter().map(|s| rect(s.offset, s.extent)).collect();
        self
    }

    pub fn draw<'b: 'a>(
        self,
        bindings: BindingOutput<'b, kind::RasterVertex>,
        vertices: u32,
        instances: u32,
        state: RasterState<COLOR_ATTACHMENTS>,
    ) -> RasterBuilder<'a, { DRAWS + 1 }, COLOR_ATTACHMENTS> {
        let command = DrawCommand::Draw {
            vertices,
            instances,
        };
        self.push(bindings.erase(), command, state)
    }

    pub fn draw_indexed<'b: 'a, IDX: IsIndex>(
        self,
        bindings: BindingOutput<'b, kind::RasterVertex>,
        index_buffer: BufferSlice<'a, u32, IDX>,
        instances: u32,
        state: RasterState<COLOR_ATTACHMENTS>,
    ) -> RasterBuilder<'a, { DRAWS + 1 }, COLOR_ATTACHMENTS> {
        index_buffer.usage_check(vk::BufferUsageFlags::INDEX_BUFFER);
        let mut bindings = bindings.erase();
        bindings.buffers.push(index_buffer.access(
            vk::PipelineStageFlags2::INDEX_INPUT,
            vk::AccessFlags2::INDEX_READ,
        ));
        let command = DrawCommand::Indexed {
            indices: index_buffer.address_range(),
            flags: index_buffer.address_flags(),
            count: index_buffer.len() as u32,
            instances,
        };
        self.push(bindings, command, state)
    }

    pub fn draw_indirect<'b: 'a, IND: IsIndirect>(
        self,
        bindings: BindingOutput<'b, kind::RasterVertex>,
        buffer: BufferSlice<'a, DrawIndirectCommand, IND>,
        state: RasterState<COLOR_ATTACHMENTS>,
    ) -> RasterBuilder<'a, { DRAWS + 1 }, COLOR_ATTACHMENTS> {
        let mut bindings = bindings.erase();
        bindings.buffers.push(indirect_access(buffer));
        let command = DrawCommand::Indirect {
            draws: buffer.strided_range(),
            flags: buffer.address_flags(),
            count: buffer.len() as u32,
        };
        self.push(bindings, command, state)
    }

    pub fn draw_indirect_count<'b: 'a, IND: IsIndirect, CNT: IsIndirect>(
        self,
        bindings: BindingOutput<'b, kind::RasterVertex>,
        buffer: BufferSlice<'a, DrawIndirectCommand, IND>,
        count_buffer: BufferSlice<'a, u32, CNT>,
        state: RasterState<COLOR_ATTACHMENTS>,
    ) -> RasterBuilder<'a, { DRAWS + 1 }, COLOR_ATTACHMENTS> {
        let mut bindings = bindings.erase();
        bindings.buffers.push(indirect_access(buffer));
        bindings.buffers.push(indirect_access(count_buffer));
        let command = DrawCommand::IndirectCount {
            draws: buffer.strided_range(),
            flags: buffer.address_flags(),
            count: count_buffer.address_range(),
            count_flags: count_buffer.address_flags(),
            max_count: buffer.len() as u32,
        };
        self.push(bindings, command, state)
    }

    pub fn launch<'b: 'a>(
        self,
        bindings: BindingOutput<'b, kind::RasterMesh>,
        groups: [u32; 3],
        state: RasterState<COLOR_ATTACHMENTS>,
    ) -> RasterBuilder<'a, { DRAWS + 1 }, COLOR_ATTACHMENTS> {
        self.push(bindings.erase(), DrawCommand::Launch(groups), state)
    }

    /// Launches the task (or, without a task shader, mesh) workgroup counts in `buffer`.
    pub fn launch_indirect<'b: 'a, IND: IsIndirect>(
        self,
        bindings: BindingOutput<'b, kind::RasterMesh>,
        buffer: BufferSlice<'a, DispatchIndirectCommand, IND>,
        state: RasterState<COLOR_ATTACHMENTS>,
    ) -> RasterBuilder<'a, { DRAWS + 1 }, COLOR_ATTACHMENTS> {
        let mut bindings = bindings.erase();
        bindings.buffers.push(indirect_access(buffer));
        let command = DrawCommand::LaunchIndirect {
            draws: buffer.strided_range(),
            flags: buffer.address_flags(),
            count: buffer.len() as u32,
        };
        self.push(bindings, command, state)
    }

    fn push(
        self,
        bindings: BindingOutput<'a, ()>,
        command: DrawCommand,
        state: RasterState<COLOR_ATTACHMENTS>,
    ) -> RasterBuilder<'a, { DRAWS + 1 }, COLOR_ATTACHMENTS> {
        let draw = Draw {
            bindings,
            command,
            state,
        };
        RasterBuilder {
            draws: append(self.draws, draw),
            cmd_buf: self.cmd_buf,
            extent: self.extent,
            viewport: self.viewport,
            scissors: self.scissors,
            color_formats: self.color_formats,
            color_attachments: self.color_attachments,
            color_accesses: self.color_accesses,
            depth_format: self.depth_format,
            stencil_format: self.stencil_format,
            depth_attachment: self.depth_attachment,
            depth_access: self.depth_access,
            clear_depth: self.clear_depth,
        }
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
    fn pipeline_key(
        &self,
        state: &RasterState<COLOR_ATTACHMENTS>,
        constants: &Constants,
    ) -> RasterHash {
        RasterHash {
            color_formats: SmallVec::from_slice(&self.color_formats),
            depth_format: self.depth_format,
            stencil_format: self.stencil_format,
            blends: SmallVec::from_slice(&state.blends),
            depth_write: state.depth_write,
            backface_culling: state.backface_culling,
            wire_frame: state.wire_frame,
            constants: *constants,
        }
    }

    /// Records the rendering under the GPU scope `name`: the attachments are cleared or loaded,
    /// then the draws run in order. Depth tests, depth writes and blending see the earlier draws'
    /// fragments, but nothing else is synchronised between them, so no draw may read what an
    /// earlier one writes to a buffer or storage image.
    #[validation_trace]
    pub fn record(self, name: &'static str) {
        let _span = tracing::info_span!("draw", pass = name).entered();
        let depth_access = self.depth_access();
        self.cmd_buf.flush_pending(
            self.draws
                .iter()
                .flat_map(|draw| draw.bindings.images.iter())
                .chain(&self.color_accesses)
                .chain(&depth_access),
            self.draws
                .iter()
                .flat_map(|draw| draw.bindings.buffers.iter()),
        );

        let color_attachments = self.color_attachments.map(|(view, clear)| {
            let ret = vk::RenderingAttachmentInfo::default()
                .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                .image_view(view)
                .store_op(vk::AttachmentStoreOp::STORE);
            if let Some(clear_value) = clear {
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
        self.cmd_buf.begin_pass(name);
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
            // The validation layer warns about direct draws of nothing, and an empty index
            // range can't be bound without `nullDescriptor`.
            if let DrawCommand::Draw { instances: 0, .. }
            | DrawCommand::Indexed { instances: 0, .. }
            | DrawCommand::Indexed { count: 0, .. } = draw.command
            {
                continue;
            }
            let key = self.pipeline_key(&draw.state, &draw.bindings.constants);
            let pipeline = pipelines().raster(&key, draw.bindings.pass);
            self.cmd_buf.push_constants(&draw.bindings.push_constants);
            unsafe {
                device.cmd_bind_pipeline(handle, vk::PipelineBindPoint::GRAPHICS, pipeline);
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
        unsafe { device.cmd_end_rendering(handle) };
        self.cmd_buf.end_pass();
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
    /// Starts a rendering of the area `extent`, with viewport and scissor covering it.
    pub fn raster(&mut self, extent: UVec2) -> RasterBuilder<'_, 0, 0> {
        RasterBuilder {
            cmd_buf: self,
            extent,
            viewport: Viewport {
                offset: IVec2::ZERO,
                extent,
            },
            scissors: smallvec![rect(IVec2::ZERO, extent)],
            color_formats: [],
            color_attachments: [],
            color_accesses: [],
            depth_format: vk::Format::UNDEFINED,
            stencil_format: vk::Format::UNDEFINED,
            depth_attachment: vk::ImageView::null(),
            depth_access: None,
            clear_depth: None,
            draws: [],
        }
    }

    fn compute_private<IND: IsIndirect>(
        &mut self,
        mut bindings: BindingOutput<'_, kind::Compute>,
        dispatch: [u32; 3],
        indirect_buffer: Option<BufferSlice<'_, DispatchIndirectCommand, IND>>,
    ) {
        let pass = &PASS_MAP[bindings.pass];
        let _span = tracing::info_span!("compute", pass = pass.name).entered();
        bindings
            .buffers
            .extend(indirect_buffer.map(indirect_access));
        self.flush_pending(bindings.images.iter(), bindings.buffers.iter());
        self.push_constants(&bindings.push_constants);

        let pipeline = pipelines().compute(bindings.pass, &bindings.constants);

        self.begin_pass(pass.name);
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
        self.end_pass();
    }

    #[validation_trace]
    pub fn compute_indirect<IND: IsIndirect>(
        &mut self,
        bindings: BindingOutput<'_, kind::Compute>,
        buffer: BufferSlice<'_, DispatchIndirectCommand, IND>,
    ) {
        self.compute_private(bindings, [0, 0, 0], Some(buffer));
    }

    #[validation_trace]
    pub fn compute(&mut self, bindings: BindingOutput<'_, kind::Compute>, dispatch: [u32; 3]) {
        self.compute_private::<buffer::usage::Indirect>(bindings, dispatch, None);
    }

    fn raytrace_private<IND: IsIndirect>(
        &mut self,
        mut bindings: BindingOutput<'_, kind::RayTracing>,
        rays: [u32; 3],
        indirect_buffer: Option<BufferSlice<'_, TraceRaysIndirectCommand, IND>>,
    ) {
        let pass = &PASS_MAP[bindings.pass];
        let _span = tracing::info_span!("raytrace", pass = pass.name).entered();
        bindings
            .buffers
            .extend(indirect_buffer.map(indirect_access));
        self.flush_pending(bindings.images.iter(), bindings.buffers.iter());
        self.push_constants(&bindings.push_constants);

        self.begin_pass(pass.name);
        let functions = Functions::raytracing_pipeline()
            .expect("a pass that can be recorded has its pipeline functions");
        pipelines().ray_tracing(bindings.pass, &bindings.constants, |pipeline| unsafe {
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
        self.end_pass();
    }

    /// Traces `rays` (width, height, depth) raygen invocations of a ray tracing pass.
    #[validation_trace]
    pub fn raytrace(&mut self, bindings: BindingOutput<'_, kind::RayTracing>, rays: [u32; 3]) {
        self.raytrace_private::<buffer::usage::Indirect>(bindings, rays, None);
    }

    /// Like [`Self::raytrace`], with the ray counts read from `buffer` on the GPU.
    #[validation_trace]
    pub fn raytrace_indirect<IND: IsIndirect>(
        &mut self,
        bindings: BindingOutput<'_, kind::RayTracing>,
        buffer: BufferSlice<'_, TraceRaysIndirectCommand, IND>,
    ) {
        self.raytrace_private(bindings, [0, 0, 0], Some(buffer));
    }

    fn push_constants(&mut self, constants: &[u8; MAX_PUSH_CONSTANTS_SIZE]) {
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

    pub(crate) fn flush_pending<'a: 'b, 'b>(
        &mut self,
        image_acceses: impl Iterator<Item = &'b ImageUse<'a>> + Clone,
        buffer_acceses: impl Iterator<Item = &'b BufferAccess> + Clone,
    ) {
        let barriers = compute_barriers(
            &self.pending_accesses,
            image_acceses.clone(),
            buffer_acceses.clone(),
        );
        if let Some(barriers) = barriers {
            let mut info = vk::DependencyInfo::default();
            let memory = barriers.memory.map(|memory| [memory]);
            if let Some(memory) = &memory {
                info = info.memory_barriers(memory);
            }
            if !barriers.images.is_empty() {
                info = info.image_memory_barriers(&barriers.images);
            }
            unsafe {
                Ctx::device().cmd_pipeline_barrier2(self.handle, &info);
            }
            self.pending_accesses.made_visible(&barriers);
        }

        self.pending_accesses
            .record(image_acceses.map(|image| &image.access), buffer_acceses);
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

    /// Opens a named group of the commands recorded until [`Self::end_scope`]. Profiled
    /// frames time it (see `FrameSlot::set_profiling`), and debug builds label it for
    /// RenderDoc/Nsight. Every pass is such a scope on its own.
    pub fn begin_scope(&mut self, name: &'static str) {
        Functions::cmd_start_label(&self.handle, name);
        if let Some(timestamps) = &mut self.timestamps {
            timestamps.begin(self.handle, name);
        }
    }

    pub fn end_scope(&mut self) {
        if let Some(timestamps) = &mut self.timestamps {
            timestamps.end(self.handle);
        }
        Functions::cmd_end_label(&self.handle);
    }

    /// A scope that the pass queries count too.
    fn begin_pass(&mut self, name: &'static str) {
        self.begin_scope(name);
        if let Some(counters) = &mut self.counters {
            counters.begin(self.handle, name);
        }
    }

    fn end_pass(&mut self) {
        if let Some(counters) = &mut self.counters {
            counters.end(self.handle);
        }
        self.end_scope();
    }

    pub(crate) fn begin(&mut self) -> Result<()> {
        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe { Ctx::device().begin_command_buffer(self.handle, &begin_info)? };
        Bindless::bind(&self.handle);
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
pub(crate) fn compute_barriers<'a: 'b, 'b>(
    pending: &PendingAccesses,
    image_acceses: impl Iterator<Item = &'b ImageUse<'a>> + Clone,
    buffer_acceses: impl Iterator<Item = &'b BufferAccess>,
) -> Option<Barriers> {
    let mut src_stage = vk::PipelineStageFlags2::NONE;
    let mut dst_stage = vk::PipelineStageFlags2::NONE;
    let mut src_access = vk::AccessFlags2::empty();
    let mut dst_access = vk::AccessFlags2::empty();

    for buffer_access in buffer_acceses {
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

    for image_access in image_acceses.clone().map(|image| &image.access) {
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
    pub(crate) fn record<'a>(
        &mut self,
        image_acceses: impl Iterator<Item = &'a ImageAccess>,
        buffer_acceses: impl Iterator<Item = &'a BufferAccess>,
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
    use crate::image::{
        format::{self, D32Sfloat, D32SfloatS8Uint, R8G8B8A8Unorm, StaticFormat},
        usage::{self, ColorAttachment, DepthAttachment, Sampled, StaticUsage},
    };
    use ash::vk::Handle;
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
        pending.record(images.iter().map(|image| &image.access), buffers.iter());
        pending
    }

    fn buffer_barrier(pending: &PendingAccesses, access: &BufferAccess) -> Option<Barriers> {
        compute_barriers(pending, iter::empty(), iter::once(access))
    }

    fn image_barrier(pending: &PendingAccesses, access: &ImageUse) -> Option<Barriers> {
        compute_barriers(pending, iter::once(access), iter::empty())
    }

    /// A command buffer that is never handed to Vulkan: enough for state-only code paths.
    fn offline_cmd() -> CommandBuffer {
        CommandBuffer {
            handle: vk::CommandBuffer::null(),
            pending_accesses: PendingAccesses::default(),
            timestamps: None,
            counters: None,
        }
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
        pending.record(iter::empty(), iter::once(&structure_write(1024..2048)));
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
            pending.record(iter::empty(), [write(0..100), read(0..100)].iter());
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
        let barriers = compute_barriers(pending, images.iter(), buffers.iter());
        if let Some(barriers) = &barriers {
            pending.made_visible(barriers);
        }
        pending.record(images.iter().map(|image| &image.access), buffers.iter());
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
        let barriers =
            compute_barriers(&PendingAccesses::default(), accesses.iter(), iter::empty()).unwrap();
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
            let barriers = compute_barriers(&nothing, iter::once(access), iter::empty()).unwrap();
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
        cmd.flush_pending([image_write(1)].iter(), [write(0..10), read(20..30)].iter());
        assert_eq!(cmd.pending_accesses.buffer_writes.len(), 1);
        assert_eq!(cmd.pending_accesses.buffer_reads.len(), 1);
        assert_eq!(cmd.pending_accesses.image_writes.len(), 1);
    }

    // ---- blits ---------------------------------------------------------------------------

    // ---- raster builder ------------------------------------------------------------------

    const EXTENT: UVec2 = UVec2::new(640, 480);

    /// Bindings for draws that are never recorded.
    fn no_bindings<K>() -> BindingOutput<'static, K> {
        BindingOutput::new(0, &0u32, S::VERTEX_SHADER, [], [])
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
        let builder = cmd.raster(EXTENT);
        assert!(builder.color_formats.is_empty());
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
        let builder = cmd.raster(EXTENT).scissors(&[Scissor {
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

    const NO_CONSTANTS: Constants = [0; MAX_SPECIALIZATION_CONSTANTS_SIZE];

    #[test]
    fn raster_state_setters_end_up_in_the_pipeline_key() {
        let mut cmd = offline_cmd();
        let builder = cmd.raster(EXTENT);
        let state = RasterState::default()
            .backface_culling(false)
            .wire_frame(true)
            .depth_write(false);
        let key = builder.pipeline_key(&state, &NO_CONSTANTS);
        assert!(!key.backface_culling);
        assert!(key.wire_frame);
        assert!(!key.depth_write);
        assert_ne!(
            key,
            builder.pipeline_key(&RasterState::default(), &NO_CONSTANTS),
            "different state must not share a pipeline"
        );
    }

    #[test]
    fn blend_modes_get_their_own_pipelines() {
        let state = layout(L::UNDEFINED);
        let mut cmd = offline_cmd();
        let builder = cmd
            .raster(EXTENT)
            .color_attachment(view::<R8G8B8A8Unorm, ColorAttachment>(&state), None);
        let keys = [
            Blend::Replace,
            Blend::Alpha,
            Blend::Add,
            Blend::Attenuate,
            Blend::Skip,
        ]
        .map(|blend| builder.pipeline_key(&RasterState::default().blends([blend]), &NO_CONSTANTS));
        for (i, key) in keys.iter().enumerate() {
            assert!(!keys[i + 1..].contains(key), "{key:?}");
        }
    }

    #[test]
    fn without_blends_every_attachment_is_replaced() {
        let (a, b) = (layout(L::UNDEFINED), layout(L::UNDEFINED));
        let mut cmd = offline_cmd();
        let builder = cmd
            .raster(EXTENT)
            .color_attachment(view::<R8G8B8A8Unorm, ColorAttachment>(&a), None)
            .color_attachment(view::<R8G8B8A8Unorm, ColorAttachment>(&b), None);
        let key = builder.pipeline_key(&RasterState::default(), &NO_CONSTANTS);
        assert_eq!(key.blends.as_slice(), [Blend::Replace; 2]);
        let explicit = RasterState::default().blends([Blend::Replace; 2]);
        assert_eq!(key, builder.pipeline_key(&explicit, &NO_CONSTANTS));
    }

    #[test]
    fn color_attachment_records_format_view_and_access() {
        let state = layout(L::UNDEFINED);
        let mut cmd = offline_cmd();
        let builder = cmd.raster(EXTENT).color_attachment(
            view::<R8G8B8A8Unorm, ColorAttachment>(&state),
            Some([0.0, 0.5, 1.0, 1.0]),
        );

        assert_eq!(
            builder.color_formats.as_slice(),
            [vk::Format::R8G8B8A8_UNORM]
        );
        let (attachment, clear) = builder.color_attachments[0];
        assert_eq!(attachment, vk::ImageView::from_raw(2));
        assert_eq!(
            unsafe { clear.unwrap().color.float32 },
            [0.0, 0.5, 1.0, 1.0]
        );

        let access = &builder.color_accesses[0].access;
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
            let builder = cmd
                .raster(EXTENT)
                .color_attachment(view::<R8G8B8A8Unorm, ColorAttachment>(&color), None)
                .depth_attachment(view::<D32Sfloat, DepthAttachment>(&depth), None);
            let depth = builder.depth_access();
            let accesses = builder.color_accesses.iter().chain(&depth);
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
        let mut cmd = offline_cmd();
        let builder = cmd
            .raster(EXTENT)
            .color_attachment(view::<R8G8B8A8Unorm, ColorAttachment>(&state), None);
        assert!(builder.color_attachments[0].1.is_none());
        assert_eq!(
            builder.color_accesses[0].access.access,
            A::COLOR_ATTACHMENT_WRITE | A::COLOR_ATTACHMENT_READ
        );
    }

    #[test]
    fn read_only_depth_attachment_tests_without_writing() {
        let state = layout(L::GENERAL);
        let mut cmd = offline_cmd();
        let read_only = RasterState::default().depth_write(false);
        let builder = cmd
            .raster(EXTENT)
            .depth_attachment(view::<D32Sfloat, DepthAttachment>(&state), None)
            .draw(no_bindings(), 3, 1, read_only.clone());

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
            builder.pipeline_key(&read_only, &NO_CONSTANTS),
            builder.pipeline_key(&RasterState::default(), &NO_CONSTANTS)
        );
    }

    /// Clearing modifies the attachment even if every draw only tests against it.
    #[test]
    fn cleared_read_only_depth_attachment_is_still_a_write_access() {
        let state = layout(L::UNDEFINED);
        let mut cmd = offline_cmd();
        let builder = cmd
            .raster(EXTENT)
            .depth_attachment(view::<D32Sfloat, DepthAttachment>(&state), Some([0.25]))
            .draw(
                no_bindings(),
                3,
                1,
                RasterState::default().depth_write(false),
            );
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
        let builder = cmd
            .raster(EXTENT)
            .color_attachment(
                dynamic_view(
                    &color,
                    format::Dynamic(vk::Format::R8G8B8A8_UNORM),
                    usage::Dynamic(vk::ImageUsageFlags::COLOR_ATTACHMENT),
                ),
                Some(clear),
            )
            .depth_attachment(
                dynamic_view(
                    &depth,
                    format::Dynamic(vk::Format::D32_SFLOAT_S8_UINT),
                    DepthAttachment,
                ),
                None,
            );
        assert_eq!(
            builder.color_formats.as_slice(),
            [vk::Format::R8G8B8A8_UNORM]
        );
        let (_, color_clear) = builder.color_attachments[0];
        assert_eq!(
            unsafe { color_clear.unwrap().color.float32 },
            [0.0, 0.5, 1.0, 1.0]
        );
        assert_eq!(builder.depth_format, vk::Format::D32_SFLOAT_S8_UINT);
        assert_eq!(builder.stencil_format, vk::Format::D32_SFLOAT_S8_UINT);
    }

    #[test]
    #[should_panic(expected = "lacks COLOR_ATTACHMENT")]
    fn dynamic_usage_without_the_attachment_flag_panics() {
        let state = layout(L::UNDEFINED);
        let mut cmd = offline_cmd();
        let _ = cmd.raster(EXTENT).color_attachment(
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
        let _ = cmd.raster(EXTENT).depth_attachment(
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
        let builder = cmd
            .raster(EXTENT)
            .depth_attachment(view::<D32Sfloat, DepthAttachment>(&state), None)
            .draw(no_bindings(), 3, 1, RasterState::default())
            .draw(
                no_bindings(),
                3,
                1,
                RasterState::default().depth_write(false),
            );
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
        let builder = cmd.raster(EXTENT).depth_attachment(
            view::<D32SfloatS8Uint, DepthAttachment>(&state),
            Some((0.0, 0)),
        );
        assert_eq!(builder.depth_format, vk::Format::D32_SFLOAT_S8_UINT);
        assert_eq!(builder.stencil_format, vk::Format::D32_SFLOAT_S8_UINT);
    }

    #[test]
    fn draws_keep_their_order_and_access_their_command_buffers() {
        use crate::buffer::usage::{Index, Indirect};
        let mut cmd = offline_cmd();
        let builder = cmd
            .raster(EXTENT)
            .draw_indexed(
                no_bindings(),
                slice::<u32, Index>(0x1000, 6),
                2,
                RasterState::default(),
            )
            .launch_indirect(
                no_bindings(),
                slice::<DispatchIndirectCommand, Indirect>(0x2000, 1),
                RasterState::default(),
            );
        let [indexed, launch] = &builder.draws;
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
        let index = &indexed.bindings.buffers[0];
        assert_eq!((index.stage, index.access), (S::INDEX_INPUT, A::INDEX_READ));
        assert_eq!(index.range, (0x1000u64..0x1018).into());

        let DrawCommand::LaunchIndirect {
            draws, count: 1, ..
        } = launch.command
        else {
            panic!("not the indirect launch: {:?}", launch.command);
        };
        assert_eq!((draws.address, draws.size, draws.stride), (0x2000, 12, 12));
        let indirect = &launch.bindings.buffers[0];
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

    #[cfg(feature = "test-passes")]
    mod generated {
        use super::*;
        use crate::bindings::{
            CTestComputeBufferBindings, CTestComputeImageBindings, CTestRasterBindings,
            TestComputeBuffer, TestComputeImage, TestRaster, TestVertex,
        };
        use crate::buffer::usage::Storage as StorageBuffer;
        use crate::image::usage::Storage as StorageImage;
        use crate::pipeline::PassKind;

        fn storage<T: Pod>(gpu_ptr: u64, len: u64) -> BufferSlice<'static, T, StorageBuffer> {
            slice(gpu_ptr, len)
        }

        fn push_constants<G: Pod, K>(bindings: &BindingOutput<'_, K>) -> G {
            bytemuck::pod_read_unaligned(&bindings.push_constants[..size_of::<G>()])
        }

        #[test]
        fn test_passes_are_registered_with_their_kind() {
            let compute =
                TestComputeBuffer::push_bindings(storage::<u32>(0, 1), storage::<u32>(0, 1), 1, 1)
                    .pass;
            assert!(matches!(
                PASS_MAP[compute].kind,
                PassKind::Compute {
                    entry: "test_compute_buffer\0"
                }
            ));
            let raster =
                TestRaster::push_bindings(storage::<TestVertex>(0, 1), glam::Vec2::ZERO).pass;
            assert!(matches!(
                PASS_MAP[raster].kind,
                PassKind::RasterVertex {
                    vertex: "test_raster_vertex\0",
                    fragment: "test_raster_fragment\0"
                }
            ));
            assert_eq!(PASS_MAP[raster].source, "shaders/test_raster.slang");
        }

        #[test]
        fn push_constants_are_the_struct_zero_padded() {
            let bindings = TestComputeBuffer::push_bindings(
                storage::<u32>(0x1000, 16),
                storage::<u32>(0x2000, 16),
                16,
                3,
            );
            let size = size_of::<CTestComputeBufferBindings>();
            assert!(bindings.push_constants[size..].iter().all(|&b| b == 0));
        }

        #[test]
        fn buffer_fields_become_device_addresses_and_accesses() {
            let bindings = TestComputeBuffer::push_bindings(
                storage::<u32>(0x1000, 16),
                storage::<u32>(0x2000, 16),
                16,
                3,
            );
            let constants: CTestComputeBufferBindings = push_constants(&bindings);
            assert_eq!(constants.src, 0x1000);
            assert_eq!(constants.dst, 0x2000);
            assert_eq!((constants.count, constants.multiplier), (16, 3));

            let [src, dst] = bindings.buffers.as_slice() else {
                panic!("two buffers");
            };
            // `Buf` is read-only, `MutBuf` is read-write.
            assert_eq!(src.access, A::SHADER_STORAGE_READ);
            assert_eq!(dst.access, A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE);
            assert_eq!(src.stage, S::COMPUTE_SHADER);
            assert_eq!(src.range, (0x1000u64..0x1040).into());
            assert_eq!(dst.range, (0x2000u64..0x2040).into());
        }

        #[test]
        fn raster_passes_access_their_buffers_in_both_shader_stages() {
            let bindings = TestRaster::push_bindings(
                storage::<TestVertex>(0x4000, 3),
                glam::Vec2::new(0.5, 0.0),
            );
            let constants: CTestRasterBindings = push_constants(&bindings);
            assert_eq!(constants.vertices, 0x4000);
            assert_eq!(
                bindings.buffers[0].stage,
                S::VERTEX_SHADER | S::FRAGMENT_SHADER
            );
            assert_eq!(bindings.buffers[0].access, A::SHADER_STORAGE_READ);
        }

        #[test]
        fn registered_accesses_are_appended_with_the_pass_stages() {
            let state = layout(L::UNDEFINED);
            let extra = view::<R8G8B8A8Unorm, StorageImage>(&state);
            let bindings = TestRaster::push_bindings(
                storage::<TestVertex>(0x4000, 3),
                glam::Vec2::new(0.5, 0.0),
            )
            .storage_read(extra)
            .buffer_read(storage::<u32>(0x5000, 4))
            .buffer_write(storage::<u32>(0x6000, 4));

            let stages = S::VERTEX_SHADER | S::FRAGMENT_SHADER;
            let [image] = bindings.images.as_slice() else {
                panic!("one image");
            };
            assert_eq!(image.access.access, A::SHADER_STORAGE_READ);
            assert_eq!(image.access.stage, stages);
            assert_eq!(image.access.layout, L::GENERAL);

            // The pass's own buffer stays first, registered ones follow in call order.
            let [vertices, read, write] = bindings.buffers.as_slice() else {
                panic!("three buffers");
            };
            assert_eq!(vertices.range.start, 0x4000);
            assert_eq!(read.range, (0x5000u64..0x5010).into());
            assert_eq!(read.access, A::SHADER_STORAGE_READ);
            assert_eq!(write.range.start, 0x6000);
            assert_eq!(
                write.access,
                A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE
            );
            assert_eq!((read.stage, write.stage), (stages, stages));
        }

        #[test]
        fn image_fields_become_bindless_handles_and_general_layout_accesses() {
            let state = layout(L::UNDEFINED);
            let mut target = view::<R8G8B8A8Unorm, StorageImage>(&state);
            target.handle.descriptor_index_set1 = 5;
            let bindings = TestComputeImage::push_bindings(
                target,
                glam::Vec4::ONE,
                UVec2::ZERO,
                UVec2::new(8, 8),
            )
            .constant_bindings();
            let constants: CTestComputeImageBindings = push_constants(&bindings);
            assert_eq!(constants.target.descriptor_index_set1, 5);
            let [image] = bindings.images.as_slice() else {
                panic!("one image");
            };
            let access = &image.access;
            assert_eq!(
                access.access,
                A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE
            );
            assert_eq!(access.stage, S::COMPUTE_SHADER);
            assert_eq!(access.layout, L::GENERAL);
        }
    }
}

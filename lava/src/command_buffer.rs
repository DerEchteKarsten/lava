use std::{
    cell::{Cell, LazyCell},
    collections::HashMap,
    ffi::CStr,
    fmt::Debug,
    iter,
    ops::{IntoBounds, RangeBounds},
    range::Range,
    sync::{Mutex, OnceLock, RwLock, atomic::AtomicU64},
};

use crate::{
    bindings::NUM_RAYTRACING_PIPELINES,
    bindless::Bindless,
    buffer::slice::BufferSlice,
    image::{
        format::Format,
        slice::{ImageSlice, ImageView},
        usage::{IsColorAttachment, IsDepthAttachment, UsageSet},
    },
    state::{Ctx, Functions},
    vkobjects::{
        queue::PendingAccesses,
        rt_pipeline::{RayTracingShaderCreateInfo, RayTracingShaderGroup, RaytracingPipeline},
    },
};
use ash::vk::{self, BufferCopy, Handle, IndexType};
use bytemuck::{Pod, Zeroable, bytes_of};
use glam::{IVec2, UVec2};
use lava_macros::validation_trace;
use notify::EventHandler;
use smallvec::SmallVec;

#[derive(Debug)]
pub(crate) struct BufferAccess {
    pub(crate) stage: vk::PipelineStageFlags2,
    pub(crate) access: vk::AccessFlags2,
    pub(crate) range: Range<u64>,
}

#[derive(Debug)]
pub(crate) struct ImageAccess {
    pub(crate) stage: vk::PipelineStageFlags2,
    pub(crate) access: vk::AccessFlags2,
    pub(crate) image: vk::ImageView,
}

#[derive(Debug)]
pub struct CommandBuffer {
    pub(crate) handle: vk::CommandBuffer,
    pub(crate) pending_accesses: PendingAccesses,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Default)]
pub struct ShaderHash {
    pub entry: &'static str,
    pub file: &'static str,
}

pub const SPIRV_PATH: &'static str = concat!(env!("OUT_DIR"), "/shaders.spv");
pub const SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders.spv"));
pub const PIPELINES: Option<PipelineManager> = None;

pub struct PipelineManager {
    pub module: Mutex<vk::ShaderModule>,
    pub compute_pipelines: RwLock<[vk::Pipeline; NUM_RAYTRACING_PIPELINES]>,
    pub raytracing_pipelines: RwLock<[RaytracingPipeline; NUM_RAYTRACING_PIPELINES]>,
    pub raster_pipelines: RwLock<HashMap<RasterHash, vk::Pipeline>>,
}

impl PipelineManager {
    pub fn new() -> Self {
        let module = create_module(SPIRV);

        Self {
            module: Mutex::new(module),
            compute_pipelines,
            raytracing_pipelines,
            raster_pipelines: RwLock::new(HashMap::new()),
        }
    }
}

struct FileWatcher;

impl EventHandler for FileWatcher {
    fn handle_event(&mut self, e: notify::Result<notify::Event>) {
        match e {
            Ok(_) => {
                tracing::info!(target: "shader_file_watcher", "Reloading Shaders");
                match std::fs::read(SPIRV_PATH) {
                    Ok(spirv) => {
                        *(PIPELINES.unwrap().module.lock().unwrap()) = create_module(&spirv)
                    }
                    Err(e) => tracing::error!(target: "shader_file_watcher", "{}", e),
                }
            }
            Err(e) => {
                tracing::error!(target: "shader_file_watcher", "{}", e);
            }
        };
    }
}

pub fn init() {
    {
        let config = notify::Config::default();
        let file_watcher = notify::Watcher::new(FileWatcher, config).unwrap();
    }
}

fn create_module(bytes: &[u8]) -> vk::ShaderModule {
    let decoded_code = ash::util::read_spv(&mut std::io::Cursor::new(bytes)).unwrap();
    let create_info = vk::ShaderModuleCreateInfo::default().code(&decoded_code);

    unsafe {
        Ctx::device()
            .create_shader_module(&create_info, None)
            .unwrap()
    }
}

fn create_shader_stage<'a>(
    entry: &'a str,
    bytes: &[u8],
    stage: vk::ShaderStageFlags,
) -> (vk::ShaderModule, vk::PipelineShaderStageCreateInfo<'a>) {
    let module = create_module(bytes);
    (module, make_shader_stage(entry, stage, module))
}

fn make_shader_stage<'a>(
    entry: &'a str,
    stage: vk::ShaderStageFlags,
    module: vk::ShaderModule,
) -> vk::PipelineShaderStageCreateInfo<'a> {
    vk::PipelineShaderStageCreateInfo::default()
        .stage(stage)
        .module(module)
        .name(CStr::from_bytes_with_nul(entry.as_bytes()).unwrap())
}

pub trait ComputePass {
    type GpuBinding: Binding;
    const STATIC_PIPELINE: usize;

    fn get() -> vk::Pipeline {
        PIPELINES.unwrap().compute_pipelines.read().unwrap()[Self::STATIC_PIPELINE]
    }
}
pub trait RayTracingPass {
    type GpuBinding: Binding;
    const STATIC_PIPELINE: usize;

    fn get() -> (vk::Pipeline, [vk::StridedDeviceAddressRegionKHR; 3]) {
        let p = &PIPELINES
            .as_ref()
            .unwrap()
            .raytracing_pipelines
            .read()
            .unwrap()[Self::STATIC_PIPELINE];
        (
            p.pipeline,
            [p.sbt.raygen_region, p.sbt.miss_region, p.sbt.hit_region],
        )
    }
}

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub struct RasterHash {
    backface_culling: bool,
    wire_frame: bool,
    color_formats: SmallVec<[vk::Format; 4]>,
    depth_format: vk::Format,
    stencil_format: vk::Format,
    pipeline_index: usize,
}

pub trait RasterVertexShaderPass: RasterPass {
    fn get(hash: &RasterHash) -> vk::Pipeline {}
}

pub trait RasterMeshShaderPass: RasterPass {
    fn get(hash: &RasterHash) -> vk::Pipeline {}
}

fn create_raster_pipeline(
    stages: &[vk::PipelineShaderStageCreateInfo<'_>],
    hash: &RasterHash,
) -> vk::Pipeline {
    let _span = tracing::info_span!("create raster pipeline");

    let mut create_info = vk::GraphicsPipelineCreateInfo::default();
    let ia = vk::PipelineInputAssemblyStateCreateInfo::default()
        .primitive_restart_enable(false)
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_attribute_descriptions(&[])
        .vertex_binding_descriptions(&[]);
    create_info = create_info
        .input_assembly_state(&ia)
        .vertex_input_state(&vertex_input_state);

    let mut rendering = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&hash.color_formats)
        .depth_attachment_format(hash.depth_format)
        .stencil_attachment_format(hash.stencil_format)
        .view_mask(0);
    let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
        .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

    let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1)
        .sample_shading_enable(false)
        .min_sample_shading(1.0)
        .alpha_to_coverage_enable(false)
        .alpha_to_one_enable(false)
        .sample_mask(&[]);

    let color_blend_attachments = hash
        .color_formats
        .iter()
        .map(|_| {
            vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(true)
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
                .alpha_blend_op(vk::BlendOp::ADD)
        })
        .collect::<Vec<_>>();
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(color_blend_attachments.as_slice())
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .blend_constants([1.0, 1.0, 1.0, 1.0]);

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .scissor_count(1)
        .viewport_count(1);
    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .line_width(1.0)
        .polygon_mode(if hash.wire_frame {
            vk::PolygonMode::LINE
        } else {
            vk::PolygonMode::FILL
        })
        .cull_mode(if hash.backface_culling {
            vk::CullModeFlags::BACK
        } else {
            vk::CullModeFlags::NONE
        })
        .front_face(vk::FrontFace::CLOCKWISE)
        .depth_bias_enable(true);
    let depth_stencil_state = if hash.depth_format != vk::Format::UNDEFINED {
        vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_bounds_test_enable(false)
            .depth_compare_op(vk::CompareOp::GREATER)
            .depth_test_enable(true)
            .depth_write_enable(true)
            .min_depth_bounds(1.0)
            .max_depth_bounds(0.0)
            .stencil_test_enable(false)
    } else {
        vk::PipelineDepthStencilStateCreateInfo::default()
    };

    create_info = create_info
        .stages(&stages)
        .layout(Bindless::layout())
        .dynamic_state(&dynamic_state)
        .multisample_state(&multisampling)
        .color_blend_state(&color_blend_state)
        .rasterization_state(&rasterization_state)
        .viewport_state(&viewport_state)
        .depth_stencil_state(&depth_stencil_state)
        .base_pipeline_handle(vk::Pipeline::null())
        .base_pipeline_index(-1)
        .push_next(&mut rendering);

    let pipeline = unsafe {
        Ctx::device()
            .create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
            .unwrap()
    }[0];

    pipeline
}

pub trait Binding: Pod {
    type CpuBinding<'a>;
    const N: usize;
    const M: usize;
    fn from_cpu_binding<'a>(binding: &Self::CpuBinding<'a>) -> Self;
    fn buffer_accesses<'a>(
        binding: &Self::CpuBinding<'a>,
        stage: vk::PipelineStageFlags2,
    ) -> [BufferAccess; Self::N];

    fn image_accesses<'a>(
        binding: &Self::CpuBinding<'a>,
        stage: vk::PipelineStageFlags2,
    ) -> [ImageAccess; Self::M];
}

#[derive(Debug)]
pub enum PushConstant {
    BindlessImage(u64),
    BufferPointer(u64),
    Constants(Vec<u8>),
}

#[repr(i32)]
pub enum Filter {
    Nearest = 0,
    Liniear = 1,
}

pub trait RasterPass {
    type GpuBinding: Binding;
    const PIPELINE_INDEX: usize;
}

pub struct RasterBuilder<'command_buffer_ref, 'command_buffer_resources, S: RasterPass> {
    hash: RasterHash,
    color_attachments: SmallVec<[(vk::ImageView, Option<vk::ClearValue>); 2]>,
    color_accesses: SmallVec<[ImageAccess; 2]>,
    depth_attachment: vk::ImageView,
    depth_access: Option<ImageAccess>,
    clear_depth: Option<vk::ClearValue>,
    write_depth: bool,
    cmd_buf: &'command_buffer_ref mut CommandBuffer,
    binding:
        Option<<<S as RasterPass>::GpuBinding as Binding>::CpuBinding<'command_buffer_resources>>,
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

#[derive(Clone, Copy)]
pub enum RasterVertexDispatch<'a> {
    Draw {
        vertex_count: u32,
        instance_count: u32,
    },
    DrawIndexed {
        index_buffer: BufferSlice<'a, u32>,
        instance_count: u32,
    },
    DrawIndirect {
        buffer: BufferSlice<'a, DrawIndirectCommand>,
    },
    DrawIndirectCount {
        buffer: BufferSlice<'a, DrawIndirectCommand>,
        count_buffer: BufferSlice<'a, u32>,
    },
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Scissor {
    pub offset: IVec2,
    pub extent: UVec2,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Viewport {
    pub offset: IVec2,
    pub extent: UVec2,
}

impl<'a, 'b, S: RasterVertexShaderPass> RasterBuilder<'a, 'b, S> {
    #[validation_trace]
    pub fn draw_with_dynstates(
        self,
        dispatch: RasterVertexDispatch<'b>,
        total_extent: UVec2,
        scissors: &[Scissor],
        viewport: Viewport,
    ) {
        let pipeline = S::get(&self.hash);
        self.draw_private(
            pipeline,
            Some(dispatch),
            total_extent,
            [0, 0, 0],
            unsafe { std::mem::transmute(scissors) },
            viewport,
        );
    }
    #[validation_trace]
    pub fn draw(self, total_extent: UVec2, dispatch: RasterVertexDispatch<'b>) {
        self.draw_with_dynstates(
            dispatch,
            total_extent,
            &[Scissor {
                extent: total_extent,
                offset: IVec2::ZERO,
            }],
            Viewport {
                extent: total_extent,
                offset: IVec2::ZERO,
            },
        );
    }
}

impl<'a, 'b, S: RasterMeshShaderPass> RasterBuilder<'a, 'b, S> {
    #[validation_trace]
    pub fn launch_with_dynstates(
        self,
        x: u32,
        y: u32,
        z: u32,
        extend: UVec2,
        scissors: &[Scissor],
        viewport: Viewport,
    ) {
        let pipeline = S::get(&self.hash);
        self.draw_private(
            pipeline,
            None,
            extend,
            [x, y, z],
            unsafe { std::mem::transmute(scissors) },
            viewport,
        );
    }

    #[validation_trace]
    pub fn launch(self, x: u32, y: u32, z: u32, extent: UVec2) {
        self.launch_with_dynstates(
            x,
            y,
            z,
            extent,
            &[Scissor {
                extent,
                offset: IVec2::ZERO,
            }],
            Viewport {
                extent,
                offset: IVec2::ZERO,
            },
        );
    }
}

impl<'a, 'b, S: RasterPass> RasterBuilder<'a, 'b, S> {
    pub fn backface_culling(mut self, backface_culling: bool) -> Self {
        self.hash.backface_culling = backface_culling;
        self
    }
    pub fn wire_frame(mut self, wire_frame: bool) -> Self {
        self.hash.wire_frame = wire_frame;
        self
    }
    pub fn color_attachment<F: Format, U>(
        mut self,
        image: ImageView<'b, F, U>,
        clear: Option<[F::Texel; 4]>,
    ) -> Self
    where
        U: IsColorAttachment,
    {
        assert!(F::ASPECTS.contains(vk::ImageAspectFlags::COLOR));
        self.hash.color_formats.push(F::format());
        let no_clear = clear.is_none();
        self.color_attachments
            .push((image.view, clear.map(|e| F::clear_value(e))));
        self.color_accesses.push(ImageAccess {
            access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                | if no_clear {
                    vk::AccessFlags2::COLOR_ATTACHMENT_READ
                } else {
                    vk::AccessFlags2::empty()
                },
            stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            image: image.view,
        });
        self
    }

    pub fn depth_attachment<F: Format, U>(
        mut self,
        image: ImageView<'b, F, U>,
        clear: Option<[F::Texel; 4]>,
        write: bool,
    ) -> Self
    where
        U: IsDepthAttachment,
    {
        assert!(F::ASPECTS.contains(vk::ImageAspectFlags::DEPTH));
        self.hash.depth_format = F::format();
        self.depth_attachment = image.view;
        self.clear_depth = clear.map(|e| F::clear_value(e));
        self.write_depth = write;
        self.depth_access = Some(ImageAccess {
            access: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                | if self.write_depth {
                    vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
                } else {
                    vk::AccessFlags2::empty()
                },
            stage: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            image: image.view,
        });
        self
    }

    fn draw_private(
        self,
        pipeline: vk::Pipeline,
        dispatch: Option<RasterVertexDispatch<'b>>,
        total_extent: UVec2,
        launch: [u32; 3],
        scissors: &[vk::Rect2D],
        viewport: Viewport,
    ) {
        let _span = tracing::info_span!("draw");

        let stage = if dispatch.is_some() {
            vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::VERTEX_SHADER
        } else {
            vk::PipelineStageFlags2::MESH_SHADER_EXT
        };
        let image_accesses = S::GpuBinding::image_accesses(self.binding, stage);
        let buffer_accesses = S::GpuBinding::buffer_accesses(self.binding, stage);

        let mut access1 = None;
        let mut access2 = None;

        if let Some(dispatch) = &dispatch {
            match dispatch {
                RasterVertexDispatch::DrawIndirect { buffer } => {
                    access1 = Some(BufferAccess {
                        access: vk::AccessFlags2::INDIRECT_COMMAND_READ,
                        stage: vk::PipelineStageFlags2::DRAW_INDIRECT,
                        range: buffer.get_range(),
                    });
                }
                RasterVertexDispatch::DrawIndirectCount {
                    buffer,
                    count_buffer,
                } => {
                    access1 = Some(BufferAccess {
                        access: vk::AccessFlags2::INDIRECT_COMMAND_READ,
                        stage: vk::PipelineStageFlags2::DRAW_INDIRECT,
                        range: buffer.get_range(),
                    });
                    access2 = Some(BufferAccess {
                        access: vk::AccessFlags2::INDIRECT_COMMAND_READ,
                        stage: vk::PipelineStageFlags2::DRAW_INDIRECT,
                        range: count_buffer.get_range(),
                    });
                }
                RasterVertexDispatch::DrawIndexed { index_buffer, .. } => {
                    access1 = Some(BufferAccess {
                        access: vk::AccessFlags2::INDEX_READ,
                        stage: vk::PipelineStageFlags2::INDEX_INPUT,
                        range: index_buffer.get_range(),
                    });
                }
                _ => {}
            }
        }

        self.cmd_buf.flush_pending(
            image_accesses
                .iter()
                .chain(self.color_accesses.iter())
                .chain(self.depth_access.iter()),
            buffer_accesses
                .iter()
                .chain(access1.iter())
                .chain(access2.iter()),
        );

        self.cmd_buf
            .push_constants::<S::GpuBinding>(self.binding.as_ref().unwrap());

        let color_attachments = self
            .color_attachments
            .iter()
            .map(|e| {
                let ret = vk::RenderingAttachmentInfo::default()
                    .image_layout(vk::ImageLayout::GENERAL)
                    .image_view(e.0)
                    .store_op(vk::AttachmentStoreOp::STORE);
                if let Some(clear_value) = e.1 {
                    ret.clear_value(clear_value)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                } else {
                    ret.load_op(vk::AttachmentLoadOp::LOAD)
                }
            })
            .collect::<Vec<_>>();
        let mut rendering_info = vk::RenderingInfo::default()
            .color_attachments(color_attachments.as_slice())
            .layer_count(1)
            .render_area(vk::Rect2D {
                extent: vk::Extent2D {
                    height: total_extent.y,
                    width: total_extent.x,
                },
                offset: vk::Offset2D { x: 0, y: 0 },
            })
            .view_mask(0);

        let mut render_info1;

        if self.depth_attachment != vk::ImageView::null() {
            render_info1 = vk::RenderingAttachmentInfo::default()
                .image_layout(vk::ImageLayout::GENERAL)
                .image_view(self.depth_attachment)
                .store_op(vk::AttachmentStoreOp::NONE)
                .load_op(vk::AttachmentLoadOp::LOAD);
            if let Some(clear_value) = self.clear_depth {
                render_info1 = render_info1
                    .clear_value(clear_value)
                    .load_op(vk::AttachmentLoadOp::CLEAR);
            }
            if self.write_depth {
                render_info1 = render_info1.store_op(vk::AttachmentStoreOp::STORE);
            }
            rendering_info = rendering_info.depth_attachment(&render_info1);
        }

        unsafe {
            Ctx::device().cmd_begin_rendering(self.cmd_buf.handle, &rendering_info);
            Ctx::device().cmd_bind_pipeline(
                self.cmd_buf.handle,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline,
            );

            Ctx::device().cmd_set_viewport(
                self.cmd_buf.handle,
                0,
                &[vk::Viewport {
                    x: viewport.offset.x as f32,
                    y: viewport.offset.y as f32,
                    width: viewport.extent.x as f32,
                    height: viewport.extent.y as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            Ctx::device().cmd_set_scissor(self.cmd_buf.handle, 0, scissors);
            if let Some(dispatch) = dispatch {
                match dispatch {
                    RasterVertexDispatch::Draw {
                        vertex_count,
                        instance_count,
                    } => Ctx::device().cmd_draw(
                        self.cmd_buf.handle,
                        vertex_count,
                        instance_count,
                        0,
                        0,
                    ),
                    RasterVertexDispatch::DrawIndirect { buffer } => Ctx::device()
                        .cmd_draw_indirect(
                            self.cmd_buf.handle,
                            buffer.handle,
                            buffer.offset(),
                            buffer.len() as u32,
                            size_of::<vk::DrawIndirectCommand>() as u32,
                        ),
                    RasterVertexDispatch::DrawIndirectCount {
                        buffer,
                        count_buffer,
                    } => Ctx::device().cmd_draw_indirect_count(
                        self.cmd_buf.handle,
                        buffer.handle,
                        buffer.offset(),
                        count_buffer.handle,
                        count_buffer.offset(),
                        u32::MAX,
                        size_of::<vk::DrawIndirectCommand>() as u32,
                    ),

                    RasterVertexDispatch::DrawIndexed {
                        index_buffer,
                        instance_count,
                    } => {
                        Ctx::device().cmd_bind_index_buffer(
                            self.cmd_buf.handle,
                            index_buffer.handle,
                            index_buffer.offset(),
                            IndexType::UINT32,
                        );
                        Ctx::device().cmd_draw_indexed(
                            self.cmd_buf.handle,
                            index_buffer.len() as u32,
                            instance_count,
                            0,
                            0,
                            0,
                        )
                    }
                }
            } else {
                Functions::mesh().unwrap().cmd_draw_mesh_tasks(
                    self.cmd_buf.handle,
                    launch[0],
                    launch[1],
                    launch[2],
                );
            }
            Ctx::device().cmd_end_rendering(self.cmd_buf.handle);
        };
    }
    pub fn bind(mut self, b: <<S as RasterPass>::GpuBinding as Binding>::CpuBinding<'b>) -> Self {
        self.binding = Some(b);
        self
    }
}

pub struct ComputeBuilder<'command_buffer_ref, 'command_buffer_resources, S: ComputePass> {
    cmd_buffer: &'command_buffer_ref mut CommandBuffer,
    binding:
        Option<<<S as ComputePass>::GpuBinding as Binding>::CpuBinding<'command_buffer_resources>>,
}

impl<'command_buffer_ref, 'command_buffer_resources, S: ComputePass>
    ComputeBuilder<'command_buffer_ref, 'command_buffer_resources, S>
{
    #[validation_trace]
    pub fn bind(
        mut self,
        b: <<S as ComputePass>::GpuBinding as Binding>::CpuBinding<'command_buffer_resources>,
    ) -> Self {
        self.binding = Some(b);
        self
    }

    fn build(
        self,
        dispatch: [u32; 3],
        indirect_buffer: Option<BufferSlice<'command_buffer_resources, DrawIndirectCommand>>,
    ) {
        let _span = tracing::info_span!("compute");
        let buffer_access = S::GpuBinding::buffer_accesses(
            self.binding.as_ref().unwrap(),
            vk::PipelineStageFlags2::COMPUTE_SHADER,
        );
        let image_access = S::GpuBinding::image_accesses(
            self.binding.as_ref().unwrap(),
            vk::PipelineStageFlags2::COMPUTE_SHADER,
        );

        let access1 = indirect_buffer.map(|b| BufferAccess {
            access: vk::AccessFlags2::INDIRECT_COMMAND_READ,
            stage: vk::PipelineStageFlags2::DRAW_INDIRECT,
            range: b.get_range(),
        });
        self.cmd_buffer.flush_pending(
            image_access.iter(),
            buffer_access.iter().chain(access1.iter()),
        );
        self.cmd_buffer
            .push_constants::<S::GpuBinding>(self.binding.as_ref().unwrap());

        let pipeline = S::get();

        unsafe {
            Ctx::device().cmd_bind_pipeline(
                self.cmd_buffer.handle,
                vk::PipelineBindPoint::COMPUTE,
                pipeline,
            );
            if let Some(slice) = indirect_buffer {
                Ctx::device().cmd_dispatch_indirect(
                    self.cmd_buffer.handle,
                    slice.handle,
                    slice.offset(),
                );
            } else {
                Ctx::device().cmd_dispatch(
                    self.cmd_buffer.handle,
                    dispatch[0],
                    dispatch[1],
                    dispatch[2],
                );
            }
        }
    }

    #[validation_trace]
    pub fn dispatch_indirect(
        self,
        buffer: BufferSlice<'command_buffer_resources, DrawIndirectCommand>,
    ) {
        self.build([0, 0, 0], Some(buffer));
    }

    #[validation_trace]
    pub fn dispatch(self, x: u32, y: u32, z: u32) {
        self.build([x, y, z], None);
    }
}

pub struct RayTracingBuilder<'a, 'b, S: RayTracingPass> {
    binding: Option<<<S as RayTracingPass>::GpuBinding as Binding>::CpuBinding<'b>>,
    cmd_buffer: &'a mut CommandBuffer,
}

impl<'a, 'b, S: RayTracingPass> RayTracingBuilder<'a, 'b, S> {
    #[validation_trace]
    pub fn bind(
        mut self,
        b: <<S as RayTracingPass>::GpuBinding as Binding>::CpuBinding<'b>,
    ) -> Self {
        self.binding = Some(b);
        self
    }

    fn build(self, dispatch: [u32; 2])
    where
        [(); Self::N],
        [(); Self::M],
    {
        let image_accesses = S::GpuBinding::image_accesses(
            self.binding.as_ref().unwrap(),
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );
        let buffer_accesses = S::GpuBinding::buffer_accesses(
            self.binding.as_ref().unwrap(),
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );

        self.cmd_buffer
            .flush_pending(image_accesses.iter(), buffer_accesses.iter());
        self.cmd_buffer
            .push_constants::<S::GpuBinding>(self.binding.as_ref().unwrap());

        let (pipeline, sbt) = S::get();
        unsafe {
            Ctx::device().cmd_bind_pipeline(
                self.cmd_buffer.handle,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                pipeline,
            );
            let call_region = vk::StridedDeviceAddressRegionKHR::default();
            Functions::raytracing_pipeline().unwrap().cmd_trace_rays(
                self.cmd_buffer.handle,
                &sbt[0],
                &sbt[1],
                &sbt[2],
                &call_region,
                dispatch[0],
                dispatch[1],
                1,
            );
        };
    }

    #[validation_trace]
    pub fn dispatch(self, x: u32, y: u32) {
        self.build([x, y]);
    }
}

impl CommandBuffer {
    #[validation_trace]
    pub fn fill_buffer<'a, T: Copy + Pod>(&mut self, buffer: BufferSlice<'a, T>, data: u32) {
        let _span = tracing::info_span!("fill_buffer");
        self.flush_pending(
            std::iter::empty(),
            std::iter::once(&BufferAccess {
                access: vk::AccessFlags2::TRANSFER_WRITE,
                stage: vk::PipelineStageFlags2::TRANSFER,
                range: buffer.get_range(),
            }),
        );
        unsafe {
            Ctx::device().cmd_fill_buffer(
                self.handle,
                buffer.handle,
                buffer.offset(),
                buffer.size,
                data,
            )
        };
    }
    #[validation_trace]
    pub fn clear_image<'a, F: Format, U: UsageSet>(
        &'a mut self,
        image: ImageView<'a, F, U>,
        clear_color: [F::Texel; 4],
    ) {
        let _span = tracing::info_span!("clear_image");
        self.flush_pending(
            std::iter::once(&ImageAccess {
                access: vk::AccessFlags2::TRANSFER_WRITE,
                stage: vk::PipelineStageFlags2::TRANSFER,
                image: image.view,
            }),
            std::iter::empty(),
        );
        unsafe {
            Ctx::device().cmd_clear_color_image(
                self.handle,
                image.image,
                vk::ImageLayout::GENERAL,
                &F::clear_value(clear_color).color,
                &[image.subresource_range()],
            )
        };
    }
    #[validation_trace]
    pub fn update_buffer<'a, T: Copy + Pod>(&mut self, buffer: BufferSlice<'a, T>, data: &T) {
        let _span = tracing::info_span!("update_buffer_element");
        self.flush_pending(
            iter::empty(),
            iter::once(&BufferAccess {
                stage: vk::PipelineStageFlags2::TRANSFER,
                access: vk::AccessFlags2::TRANSFER_WRITE,
                range: buffer.get_range(),
            }),
        );
        unsafe {
            Ctx::device().cmd_update_buffer(
                self.handle,
                buffer.handle,
                buffer.offset(),
                bytemuck::bytes_of(data),
            )
        };
    }
    #[validation_trace]

    pub fn blit_image<'a, F: Format, U: UsageSet, F2: Format, U2: UsageSet>(
        &mut self,
        src: ImageSlice<'a, F, U>,
        dst: ImageSlice<'a, F2, U2>,
        filter: Filter,
    ) {
        let _span = tracing::info_span!("blit_image");
        self.flush_pending(
            [
                ImageAccess {
                    access: vk::AccessFlags2::TRANSFER_READ,
                    stage: vk::PipelineStageFlags2::TRANSFER,
                    image: src.view.view,
                },
                ImageAccess {
                    access: vk::AccessFlags2::TRANSFER_WRITE,
                    stage: vk::PipelineStageFlags2::TRANSFER,
                    image: dst.view.view,
                },
            ]
            .iter(),
            iter::empty(),
        );

        let regions = [vk::ImageBlit {
            src_offsets: [
                src.offset,
                vk::Offset3D {
                    x: src.extend.width as i32,
                    y: src.extend.height as i32,
                    z: 1,
                },
            ],
            dst_offsets: [
                dst.offset,
                vk::Offset3D {
                    x: dst.extend.width as i32,
                    y: dst.extend.height as i32,
                    z: 1,
                },
            ],
            src_subresource: src.view.subresource_layers(src.view.mip_range.start),
            dst_subresource: dst.view.subresource_layers(dst.view.mip_range.start),
        }];
        unsafe {
            Ctx::device().cmd_blit_image(
                self.handle,
                src.view.image,
                vk::ImageLayout::GENERAL,
                dst.view.image,
                vk::ImageLayout::GENERAL,
                &regions,
                vk::Filter::from_raw(filter as i32),
            );
        }
    }
    #[validation_trace]

    pub fn copy_buffer<'a, T: Copy + Pod>(
        &mut self,
        src: BufferSlice<'a, T>,
        dst: BufferSlice<'a, T>,
    ) {
        self.copy_buffer_regions(src, dst, &[src.region(dst)]);
    }
    #[validation_trace]

    pub fn copy_buffer_regions<'a, T: Copy + Pod>(
        &mut self,
        src: BufferSlice<'a, T>,
        dst: BufferSlice<'a, T>,
        regions: &[BufferCopy],
    ) {
        let _span = tracing::info_span!("copy_buffer");
        self.flush_pending(
            iter::empty(),
            [
                BufferAccess {
                    access: vk::AccessFlags2::TRANSFER_READ,
                    stage: vk::PipelineStageFlags2::TRANSFER,
                    range: src.get_range(),
                },
                BufferAccess {
                    access: vk::AccessFlags2::TRANSFER_WRITE,
                    stage: vk::PipelineStageFlags2::TRANSFER,
                    range: dst.get_range(),
                },
            ]
            .iter(),
        );
        unsafe { Ctx::device().cmd_copy_buffer(self.handle, src.handle, dst.handle, regions) };
    }

    #[validation_trace]
    pub fn copy_buffer_to_image<'a, T: Copy + Pod, F: Format, U: UsageSet>(
        &mut self,
        src: BufferSlice<'a, T>,
        dst: ImageSlice<'a, F, U>,
    ) {
        let _span = tracing::info_span!("copy_buffer_to_image");
        self.flush_pending(
            iter::once(&ImageAccess {
                access: vk::AccessFlags2::TRANSFER_WRITE,
                stage: vk::PipelineStageFlags2::TRANSFER,
                image: dst.view.view,
            }),
            iter::once(&BufferAccess {
                access: vk::AccessFlags2::TRANSFER_READ,
                stage: vk::PipelineStageFlags2::TRANSFER,
                range: src.get_range(),
            }),
        );
        let regions = [vk::BufferImageCopy {
            image_extent: dst.extend,
            image_subresource: dst.view.subresource_layers(dst.view.mip_range.start),
            buffer_image_height: 0,
            buffer_offset: src.offset(),
            buffer_row_length: 0,
            image_offset: dst.offset,
        }];
        unsafe {
            Ctx::device().cmd_copy_buffer_to_image(
                self.handle,
                src.handle,
                dst.view.image,
                vk::ImageLayout::GENERAL,
                &regions,
            )
        };
    }

    #[validation_trace]
    pub fn raster<'a, 'c: 'a, S: RasterPass>(&'c mut self) -> RasterBuilder<'a, 'c, S> {
        RasterBuilder {
            write_depth: true,
            clear_depth: None,
            cmd_buf: self,
            color_accesses: SmallVec::new(),
            color_attachments: SmallVec::new(),
            depth_attachment: vk::ImageView::null(),
            binding: None,
            hash: RasterHash {
                backface_culling: true,
                color_formats: SmallVec::new(),
                depth_format: vk::Format::UNDEFINED,
                stencil_format: vk::Format::UNDEFINED,
                wire_frame: false,
                pipeline_index: S::PIPELINE_INDEX,
            },
            depth_access: None,
        }
    }

    #[validation_trace]
    pub fn compute<'a, 'c: 'a, S: ComputePass>(&'c mut self) -> ComputeBuilder<'a, 'c, S> {
        ComputeBuilder {
            cmd_buffer: self,
            binding: None,
        }
    }
    #[validation_trace]
    pub fn raytrace<'a, 'c: 'a, S: RayTracingPass>(&'c mut self) -> RayTracingBuilder<'a, 'c, S> {
        RayTracingBuilder {
            binding: None,
            cmd_buffer: self,
        }
    }
    #[validation_trace]
    pub fn present<'a, F: Format, U: UsageSet>(&'a mut self, swapchain_image: ImageView<'a, F, U>) {
        let _span = tracing::info_span!("present_barriers");
        self.flush_pending(
            iter::once(&ImageAccess {
                access: vk::AccessFlags2::empty(),
                stage: vk::PipelineStageFlags2::empty(),
                image: swapchain_image.view,
            }),
            iter::empty(),
        );
    }

    fn push_constants<'b, B: Binding>(&mut self, binding: &B::CpuBinding<'b>) {
        let binding = B::from_cpu_binding(binding);
        let constants = bytes_of(&binding);
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
        image_acceses: impl Iterator<Item = &'a ImageAccess> + Clone,
        buffer_acceses: impl Iterator<Item = &'a BufferAccess> + Clone,
    ) {
        let mut src_stage = vk::PipelineStageFlags2::NONE;
        let mut dst_stage = vk::PipelineStageFlags2::NONE;
        let mut src_access = vk::AccessFlags2::empty();
        let mut dst_access = vk::AccessFlags2::empty();

        for buffer_access in buffer_acceses.clone() {
            let is_write = buffer_access.access.contains(all_write_access());
            let is_read = buffer_access.access.contains(all_read_access());

            if is_write {
                for pending in self
                    .pending_accesses
                    .buffer_reads
                    .iter()
                    .filter(|r| !r.range.intersect(buffer_access.range).is_empty())
                {
                    src_stage |= pending.stage;
                    dst_stage |= buffer_access.stage;
                }
            }
            for w in self
                .pending_accesses
                .buffer_writes
                .iter()
                .filter(|w| !w.range.intersect(buffer_access.range).is_empty())
            {
                if is_read || is_write {
                    src_stage |= w.stage;
                    src_access |= w.access;
                    dst_stage |= buffer_access.stage;
                    dst_access |= buffer_access.access;
                }
            }
        }

        for image_access in image_acceses {
            let is_write = image_access.access.intersects(all_write_access());
            let is_read = image_access.access.intersects(all_read_access());

            if is_write {
                for pending in self
                    .pending_accesses
                    .image_reads
                    .iter()
                    .filter(|r| r.image == image_access.image)
                {
                    src_stage |= pending.stage;
                    dst_stage |= image_access.stage;
                }
            }
            for w in self
                .pending_accesses
                .image_writes
                .iter()
                .filter(|r| r.image == image_access.image)
            {
                if is_read || is_write {
                    src_stage |= w.stage;
                    src_access |= w.access;
                    dst_stage |= image_access.stage;
                    dst_access |= image_access.access;
                }
            }
        }

        let mut info = vk::DependencyInfo::default();
    }

    pub(crate) fn begin(&mut self) {
        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe { Ctx::device().begin_command_buffer(self.handle, &begin_info) }.unwrap();
        Bindless::bind(&self.handle);
    }

    pub(crate) fn end(&mut self) {
        unsafe { Ctx::device().end_command_buffer(self.handle) }.unwrap();
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

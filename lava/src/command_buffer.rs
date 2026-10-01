use std::{
    collections::HashMap,
    ffi::CStr,
    fmt::Debug,
    iter,
    marker::PhantomData,
    ops::{IntoBounds, RangeBounds},
    range::Range,
    sync::{OnceLock, RwLock},
};

use crate::{
    bindings::{NUM_COMPUTE_PIPELINES, NUM_RASTER_PIPELINES, NUM_RAY_TRACING_PIPELINES, PASS_MAP},
    bindless::Bindless,
    buffer::{
        slice::BufferSlice,
        usage::{IsIndex, IsIndirect},
    },
    error::Result,
    image::{
        format::{ColorAspect, DepthAspect, Format},
        slice::{ImageSlice, ImageView},
        usage::{ImageUsage, IsColorAttachment, IsDepthAttachment},
    },
    state::{Ctx, Functions},
    vkobjects::{
        queue::PendingAccesses,
        rt_pipeline::{RayTracingShaderCreateInfo, RayTracingShaderGroup, RaytracingPipeline},
    },
};
use ash::vk::{self, BufferCopy, IndexType};
use bytemuck::{NoUninit, Pod, Zeroable, bytes_of};
use glam::{IVec2, UVec2};
use lava_macros::validation_trace;
use notify::EventHandler;
use smallvec::{SmallVec, smallvec};

#[derive(Debug, Clone)]
pub(crate) struct BufferAccess {
    pub(crate) stage: vk::PipelineStageFlags2,
    pub(crate) access: vk::AccessFlags2,
    pub(crate) range: Range<u64>,
}

#[derive(Debug, Clone)]
pub(crate) struct ImageAccess {
    pub(crate) stage: vk::PipelineStageFlags2,
    pub(crate) access: vk::AccessFlags2,
    pub(crate) image: vk::Image,
    pub(crate) layout: vk::ImageLayout,
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

pub const PASS_DIR: &str = concat!(env!("OUT_DIR"), "/passes");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassKind {
    Compute {
        entry: &'static str,
    },
    RayTracing {
        ray_gen: &'static str,
        ray_any: &'static str,
        ray_closest: &'static str,
    },
    RasterVertex {
        vertex: &'static str,
        fragment: &'static str,
    },
    RasterMesh {
        amp: Option<&'static str>,
        mesh: &'static str,
        fragment: &'static str,
    },
}

pub struct PassEntry {
    pub path: &'static str,
    pub kind: PassKind,
    pub index: usize,
}

static PIPELINES: OnceLock<PipelineManager> = OnceLock::new();

pub(crate) fn pipelines() -> &'static PipelineManager {
    PIPELINES.get().expect(
        "pipelines were not initialized; call `command_buffer::init()` after `lava::init()`",
    )
}

pub struct PipelineManager {
    pass_modules: RwLock<Vec<vk::ShaderModule>>,
    pub compute_pipelines: RwLock<[vk::Pipeline; NUM_COMPUTE_PIPELINES]>,
    pub raytracing_pipelines: RwLock<[Option<RaytracingPipeline>; NUM_RAY_TRACING_PIPELINES]>,
    pub raster_pipelines: RwLock<[HashMap<RasterHash, vk::Pipeline>; NUM_RASTER_PIPELINES]>,
}

impl PipelineManager {
    fn new() -> Self {
        let modules: Vec<vk::ShaderModule> = PASS_MAP
            .iter()
            .map(|pass| create_module(&read_pass_spirv(pass.path)))
            .collect();

        let manager = Self {
            pass_modules: RwLock::new(modules),
            compute_pipelines: RwLock::new([vk::Pipeline::null(); NUM_COMPUTE_PIPELINES]),
            raytracing_pipelines: RwLock::new([None; NUM_RAY_TRACING_PIPELINES]),
            raster_pipelines: RwLock::new(std::array::from_fn::<_, NUM_RASTER_PIPELINES, _>(
                |_| HashMap::new(),
            )),
        };

        for i in 0..PASS_MAP.len() {
            manager.reload_pass(i);
        }

        manager
    }

    fn reload_pass(&self, pass_index: usize) {
        let pass = &PASS_MAP[pass_index];
        let module = create_module(&read_pass_spirv(&pass.path));

        let Ok(mut modules) = self.pass_modules.write() else {
            tracing::error!("failed to acquire lock on pass modules");
            return;
        };
        let old = std::mem::replace(&mut modules[pass_index], module);
        unsafe { Ctx::device().destroy_shader_module(old, None) };

        match pass.kind {
            PassKind::Compute { entry } => {
                let pipeline = create_compute_pipeline(module, entry);
                let Ok(mut pipelines) = self.compute_pipelines.write() else {
                    tracing::error!("failed to acquire lock on compute pipelines");
                    return;
                };
                pipelines[pass.index] = pipeline;
            }
            PassKind::RayTracing {
                ray_any,
                ray_closest,
                ray_gen,
            } => {
                let pipeline = create_raytracing_pipeline(module, ray_any, ray_closest, ray_gen);
                let Ok(mut pipelines) = self.raytracing_pipelines.write() else {
                    tracing::error!("failed to acquire lock on ray tracing pipelines");
                    return;
                };
                pipelines[pass.index] = Some(pipeline);
            }
            PassKind::RasterVertex { .. } | PassKind::RasterMesh { .. } => {
                let Ok(mut pipelines) = self.raster_pipelines.write() else {
                    tracing::error!("failed to acquire lock on raster pipelines");
                    return;
                };
                for (k, v) in pipelines[pass.index].iter_mut() {
                    let new = create_raster_pipeline(module, k, pass_index);
                    *v = new;
                }
            }
        }
    }
}

fn read_pass_spirv(file: &str) -> Vec<u8> {
    let path = std::path::Path::new(PASS_DIR).join(file);
    std::fs::read(&path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

fn create_compute_pipeline(module: vk::ShaderModule, entry: &str) -> vk::Pipeline {
    let stage = make_shader_stage(entry, vk::ShaderStageFlags::COMPUTE, module);
    let create_info = vk::ComputePipelineCreateInfo::default()
        .layout(Bindless::layout())
        .stage(stage);
    let pipeline = unsafe {
        Ctx::device()
            .create_compute_pipelines(vk::PipelineCache::null(), &[create_info], None)
            .unwrap()
    }[0];
    Functions::set_debug_name(entry, pipeline);
    pipeline
}

fn create_raytracing_pipeline(
    module: vk::ShaderModule,
    raygen_entry: &str,
    hit_entry: &str,
    miss_entry: &str,
) -> RaytracingPipeline {
    let raygen = make_shader_stage(raygen_entry, vk::ShaderStageFlags::RAYGEN_KHR, module);
    let hit = make_shader_stage(hit_entry, vk::ShaderStageFlags::CLOSEST_HIT_KHR, module);
    let miss = make_shader_stage(miss_entry, vk::ShaderStageFlags::MISS_KHR, module);

    RaytracingPipeline::new(
        Bindless::layout(),
        &[
            RayTracingShaderCreateInfo {
                stages: std::slice::from_ref(&raygen),
                group: RayTracingShaderGroup::RayGen,
            },
            RayTracingShaderCreateInfo {
                stages: std::slice::from_ref(&hit),
                group: RayTracingShaderGroup::Hit,
            },
            RayTracingShaderCreateInfo {
                stages: std::slice::from_ref(&miss),
                group: RayTracingShaderGroup::Miss,
            },
        ],
    )
    .unwrap()
}

pub fn init() {
    use notify::Watcher;

    let manager = PIPELINES.get_or_init(PipelineManager::new);

    let manager: &'static PipelineManager = manager;

    static WATCHER: OnceLock<notify::RecommendedWatcher> = OnceLock::new();
    let _ = WATCHER.get_or_init(|| {
        let mut watcher = notify::recommended_watcher(FileWatcher { manager }).unwrap();
        watcher
            .watch(
                std::path::Path::new(PASS_DIR),
                notify::RecursiveMode::NonRecursive,
            )
            .unwrap();
        watcher
    });
}

struct FileWatcher {
    manager: &'static PipelineManager,
}

impl EventHandler for FileWatcher {
    fn handle_event(&mut self, e: notify::Result<notify::Event>) {
        let event = match e {
            Ok(event) => event,
            Err(e) => {
                tracing::error!(target: "shader_file_watcher", "{e}");
                return;
            }
        };

        for path in &event.paths {
            let Some(pass_index) = PASS_MAP
                .iter()
                .position(|pass| pass.path == path.to_string_lossy())
            else {
                continue;
            };

            tracing::info!(target: "shader_file_watcher", "Reloading pass `{:#?}`", path);
            self.manager.reload_pass(pass_index);
        }
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

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub struct RasterHash {
    backface_culling: bool,
    wire_frame: bool,
    color_formats: SmallVec<[vk::Format; 4]>,
    depth_format: vk::Format,
    stencil_format: vk::Format,
}

fn get_raster_pipeline(hash: &RasterHash, pass_index: usize) -> vk::Pipeline {
    let manager = pipelines();
    let map = &mut manager.raster_pipelines.write().unwrap()[PASS_MAP[pass_index].index];
    if let Some(pipeline) = map.get(hash) {
        return *pipeline;
    }
    let module = manager.pass_modules.read().unwrap()[pass_index];
    let pipeline = create_raster_pipeline(module, hash, pass_index);
    map.insert(hash.clone(), pipeline);
    pipeline
}

fn create_raster_pipeline(
    module: vk::ShaderModule,
    hash: &RasterHash,
    pass_index: usize,
) -> vk::Pipeline {
    let _span = tracing::info_span!("create raster pipeline");

    let pass = &PASS_MAP[pass_index];
    let stages = match pass.kind {
        PassKind::RasterVertex { vertex, fragment } => {
            smallvec![
                make_shader_stage(vertex, vk::ShaderStageFlags::VERTEX, module),
                make_shader_stage(fragment, vk::ShaderStageFlags::FRAGMENT, module),
            ]
        }
        PassKind::RasterMesh {
            amp,
            mesh,
            fragment,
        } => {
            let mut stages = SmallVec::<[vk::PipelineShaderStageCreateInfo<'_>; 3]>::new();
            if let Some(task) = amp {
                stages.push(make_shader_stage(
                    task,
                    vk::ShaderStageFlags::TASK_EXT,
                    module,
                ));
            }
            stages.push(make_shader_stage(
                mesh,
                vk::ShaderStageFlags::MESH_EXT,
                module,
            ));
            stages.push(make_shader_stage(
                fragment,
                vk::ShaderStageFlags::FRAGMENT,
                module,
            ));
            stages
        }
        other => panic!("pass {other:?} is not a raster pass"),
    };

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

pub struct BindingOutput<GpuBinding: Pod, T: PassType, const Images: usize, const Buffers: usize> {
    pub(crate) images: [ImageAccess; Images],
    pub(crate) buffers: [BufferAccess; Buffers],
    pub(crate) gpu_bindings: GpuBinding,
    pub(crate) _marker: PhantomData<T>,
}

pub trait PassType {
    const PASS_INDEX: usize;
}

pub trait ComputePass: PassType {}
pub trait RaytracingPass: PassType {}
pub trait RasterPass: PassType {}
pub trait RasterVertexPass: RasterPass {}
pub trait RasterMeshPass: RasterPass {}

#[repr(i32)]
pub enum Filter {
    Nearest = 0,
    Liniear = 1,
}

pub struct RasterBuilder<'command_buffer_ref> {
    hash: RasterHash,
    color_attachments: SmallVec<[(vk::ImageView, Option<vk::ClearValue>); 2]>,
    color_accesses: SmallVec<[ImageAccess; 2]>,
    depth_attachment: vk::ImageView,
    depth_access: Option<ImageAccess>,
    clear_depth: Option<vk::ClearValue>,
    write_depth: bool,
    cmd_buf: &'command_buffer_ref mut CommandBuffer,
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

impl<'a> RasterBuilder<'a> {
    /// Helper that performs the shared raster setup: flush pending buffer
    /// accesses, bind constants, begin rendering, bind the pipeline and set
    /// viewport/scissor, then run the given draw closure inside the render
    /// pass.
    fn render<GpuBinding: Pod, T: RasterPass, const Images: usize, const Buffers: usize>(
        self,
        bindings: BindingOutput<GpuBinding, T, Images, Buffers>,
        access1: Option<BufferAccess>,
        access2: Option<BufferAccess>,
        total_extent: UVec2,
        scissors: &[vk::Rect2D],
        viewport: Viewport,
        draw: impl FnOnce(vk::CommandBuffer),
    ) {
        let _span = tracing::info_span!("draw");

        let pipeline = get_raster_pipeline(&self.hash, T::PASS_INDEX);

        self.cmd_buf.flush_pending(
            bindings
                .images
                .iter()
                .chain(self.color_accesses.iter())
                .chain(self.depth_access.iter()),
            bindings
                .buffers
                .iter()
                .chain(access1.iter())
                .chain(access2.iter()),
        );

        self.cmd_buf.push_constants(&bindings.gpu_bindings);

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
            draw(self.cmd_buf.handle);
            Ctx::device().cmd_end_rendering(self.cmd_buf.handle);
        };
    }

    fn full_extent_dynstates(total_extent: UVec2) -> ([Scissor; 1], Viewport) {
        (
            [Scissor {
                extent: total_extent,
                offset: IVec2::ZERO,
            }],
            Viewport {
                extent: total_extent,
                offset: IVec2::ZERO,
            },
        )
    }

    /// Issue a plain (non-indexed) draw.
    #[validation_trace]
    pub fn draw_with_dynstates<
        GpuBinding: Pod,
        T: RasterVertexPass,
        const Images: usize,
        const Buffers: usize,
    >(
        self,
        bindings: BindingOutput<GpuBinding, T, Images, Buffers>,
        total_extent: UVec2,
        vertex_count: u32,
        instance_count: u32,
        scissors: &[Scissor],
        viewport: Viewport,
    ) {
        self.render(
            bindings,
            None,
            None,
            total_extent,
            unsafe { std::mem::transmute(scissors) },
            viewport,
            |cmd| unsafe {
                Ctx::device().cmd_draw(cmd, vertex_count, instance_count, 0, 0);
            },
        );
    }

    #[validation_trace]
    pub fn draw<GpuBinding: Pod, T: RasterVertexPass, const Images: usize, const Buffers: usize>(
        self,
        bindings: BindingOutput<GpuBinding, T, Images, Buffers>,
        total_extent: UVec2,
        vertex_count: u32,
        instance_count: u32,
    ) {
        let (scissors, viewport) = Self::full_extent_dynstates(total_extent);
        self.draw_with_dynstates(
            bindings,
            total_extent,
            vertex_count,
            instance_count,
            &scissors,
            viewport,
        );
    }

    /// Issue an indexed draw. The index buffer must carry an [`IsIndex`] usage.
    #[validation_trace]
    pub fn draw_indexed_with_dynstates<
        GpuBinding: Pod,
        T: RasterVertexPass,
        IDX: IsIndex,
        const Images: usize,
        const Buffers: usize,
    >(
        self,
        bindings: BindingOutput<GpuBinding, T, Images, Buffers>,
        total_extent: UVec2,
        index_buffer: BufferSlice<'a, u32, IDX>,
        instance_count: u32,
        scissors: &[Scissor],
        viewport: Viewport,
    ) {
        let index_access = BufferAccess {
            access: vk::AccessFlags2::INDEX_READ,
            stage: vk::PipelineStageFlags2::INDEX_INPUT,
            range: index_buffer.get_range(),
        };
        let id = index_buffer.handle;
        let offset = index_buffer.offset();
        let count = index_buffer.len() as u32;
        self.render(
            bindings,
            Some(index_access),
            None,
            total_extent,
            unsafe { std::mem::transmute(scissors) },
            viewport,
            move |cmd| unsafe {
                Ctx::device().cmd_bind_index_buffer(cmd, id, offset, IndexType::UINT32);
                Ctx::device().cmd_draw_indexed(cmd, count, instance_count, 0, 0, 0);
            },
        );
    }

    #[validation_trace]
    pub fn draw_indexed<
        GpuBinding: Pod,
        T: RasterVertexPass,
        IDX: IsIndex,
        const Images: usize,
        const Buffers: usize,
    >(
        self,
        bindings: BindingOutput<GpuBinding, T, Images, Buffers>,
        total_extent: UVec2,
        index_buffer: BufferSlice<'a, u32, IDX>,
        instance_count: u32,
    ) {
        let (scissors, viewport) = Self::full_extent_dynstates(total_extent);
        self.draw_indexed_with_dynstates(
            bindings,
            total_extent,
            index_buffer,
            instance_count,
            &scissors,
            viewport,
        );
    }

    /// Issue an indirect draw. The buffer must carry an [`IsIndirect`] usage.
    #[validation_trace]
    pub fn draw_indirect_with_dynstates<
        GpuBinding: Pod,
        T: RasterVertexPass,
        IND: IsIndirect,
        const Images: usize,
        const Buffers: usize,
    >(
        self,
        bindings: BindingOutput<GpuBinding, T, Images, Buffers>,
        total_extent: UVec2,
        buffer: BufferSlice<'a, DrawIndirectCommand, IND>,
        scissors: &[Scissor],
        viewport: Viewport,
    ) {
        let indirect_access = BufferAccess {
            access: vk::AccessFlags2::INDIRECT_COMMAND_READ,
            stage: vk::PipelineStageFlags2::DRAW_INDIRECT,
            range: buffer.get_range(),
        };
        let id = buffer.handle;
        let offset = buffer.offset();
        let draw_count = buffer.len() as u32;
        self.render(
            bindings,
            Some(indirect_access),
            None,
            total_extent,
            unsafe { std::mem::transmute(scissors) },
            viewport,
            move |cmd| unsafe {
                Ctx::device().cmd_draw_indirect(
                    cmd,
                    id,
                    offset,
                    draw_count,
                    size_of::<vk::DrawIndirectCommand>() as u32,
                );
            },
        );
    }

    #[validation_trace]
    pub fn draw_indirect<
        GpuBinding: Pod,
        T: RasterVertexPass,
        IND: IsIndirect,
        const Images: usize,
        const Buffers: usize,
    >(
        self,
        bindings: BindingOutput<GpuBinding, T, Images, Buffers>,
        total_extent: UVec2,
        buffer: BufferSlice<'a, DrawIndirectCommand, IND>,
    ) {
        let (scissors, viewport) = Self::full_extent_dynstates(total_extent);
        self.draw_indirect_with_dynstates(bindings, total_extent, buffer, &scissors, viewport);
    }

    /// Issue an indirect draw whose draw count is also read from a buffer.
    #[validation_trace]
    pub fn draw_indirect_count_with_dynstates<
        GpuBinding: Pod,
        T: RasterVertexPass,
        IND: IsIndirect,
        const Images: usize,
        const Buffers: usize,
    >(
        self,
        bindings: BindingOutput<GpuBinding, T, Images, Buffers>,
        total_extent: UVec2,
        buffer: BufferSlice<'a, DrawIndirectCommand, IND>,
        count_buffer: BufferSlice<'a, u32>,
        scissors: &[Scissor],
        viewport: Viewport,
    ) {
        let indirect_access = BufferAccess {
            access: vk::AccessFlags2::INDIRECT_COMMAND_READ,
            stage: vk::PipelineStageFlags2::DRAW_INDIRECT,
            range: buffer.get_range(),
        };
        let count_access = BufferAccess {
            access: vk::AccessFlags2::INDIRECT_COMMAND_READ,
            stage: vk::PipelineStageFlags2::DRAW_INDIRECT,
            range: count_buffer.get_range(),
        };
        let id = buffer.handle;
        let offset = buffer.offset();
        let count_id = count_buffer.handle;
        let count_offset = count_buffer.offset();
        self.render(
            bindings,
            Some(indirect_access),
            Some(count_access),
            total_extent,
            unsafe { std::mem::transmute(scissors) },
            viewport,
            move |cmd| unsafe {
                Ctx::device().cmd_draw_indirect_count(
                    cmd,
                    id,
                    offset,
                    count_id,
                    count_offset,
                    u32::MAX,
                    size_of::<vk::DrawIndirectCommand>() as u32,
                );
            },
        );
    }

    #[validation_trace]
    pub fn draw_indirect_count<
        GpuBinding: Pod,
        T: RasterVertexPass,
        IND: IsIndirect,
        const Images: usize,
        const Buffers: usize,
    >(
        self,
        bindings: BindingOutput<GpuBinding, T, Images, Buffers>,
        total_extent: UVec2,
        buffer: BufferSlice<'a, DrawIndirectCommand, IND>,
        count_buffer: BufferSlice<'a, u32>,
    ) {
        let (scissors, viewport) = Self::full_extent_dynstates(total_extent);
        self.draw_indirect_count_with_dynstates(
            bindings,
            total_extent,
            buffer,
            count_buffer,
            &scissors,
            viewport,
        );
    }

    #[validation_trace]
    pub fn launch_with_dynstates<
        GpuBinding: Pod,
        T: RasterMeshPass,
        const Images: usize,
        const Buffers: usize,
    >(
        self,
        bindings: BindingOutput<GpuBinding, T, Images, Buffers>,
        x: u32,
        y: u32,
        z: u32,
        extend: UVec2,
        scissors: &[Scissor],
        viewport: Viewport,
    ) {
        self.render(
            bindings,
            None,
            None,
            extend,
            unsafe { std::mem::transmute(scissors) },
            viewport,
            |cmd| unsafe {
                Functions::mesh().unwrap().cmd_draw_mesh_tasks(cmd, x, y, z);
            },
        );
    }

    #[validation_trace]
    pub fn launch<GpuBinding: Pod, T: RasterMeshPass, const Images: usize, const Buffers: usize>(
        self,
        bindings: BindingOutput<GpuBinding, T, Images, Buffers>,
        x: u32,
        y: u32,
        z: u32,
        extent: UVec2,
    ) {
        let (scissors, viewport) = Self::full_extent_dynstates(extent);
        self.launch_with_dynstates(bindings, x, y, z, extent, &scissors, viewport);
    }

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
        image: ImageView<'a, F, U>,
        clear: Option<F::Texels>,
    ) -> Self
    where
        U: IsColorAttachment,
        F: ColorAspect,
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
            image: image.image,
            layout: vk::ImageLayout::GENERAL,
        });
        self
    }

    pub fn depth_attachment<'b, F: Format, U>(
        mut self,
        image: ImageView<'b, F, U>,
        clear: Option<F::Texels>,
        write: bool,
    ) -> Self
    where
        U: IsDepthAttachment,
        F: DepthAspect,
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
            image: image.image,
            layout: vk::ImageLayout::GENERAL,
        });
        self
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
    pub fn clear_image<'a, F: Format, U: ImageUsage>(
        &'a mut self,
        image: ImageView<'a, F, U>,
        clear_color: F::Texels,
    ) where
        F: ColorAspect,
    {
        let _span = tracing::info_span!("clear_image");
        self.flush_pending(
            std::iter::once(&ImageAccess {
                access: vk::AccessFlags2::TRANSFER_WRITE,
                stage: vk::PipelineStageFlags2::TRANSFER,
                image: image.image,
                layout: vk::ImageLayout::GENERAL,
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
    pub fn blit_image<'a, F: Format, U: ImageUsage, F2: Format, U2: ImageUsage>(
        &mut self,
        src: ImageSlice<'a, F, U>,
        dst: ImageSlice<'a, F2, U2>,
        filter: Filter,
    ) where
        F: ColorAspect,
        F2: ColorAspect,
    {
        let _span = tracing::info_span!("blit_image");
        self.flush_pending(
            [
                ImageAccess {
                    access: vk::AccessFlags2::TRANSFER_READ,
                    stage: vk::PipelineStageFlags2::TRANSFER,
                    image: src.view.image,
                    layout: vk::ImageLayout::GENERAL,
                },
                ImageAccess {
                    access: vk::AccessFlags2::TRANSFER_WRITE,
                    stage: vk::PipelineStageFlags2::TRANSFER,
                    image: dst.view.image,
                    layout: vk::ImageLayout::GENERAL,
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
    pub fn copy_buffer_to_image<'a, T: Copy + Pod, F: Format, U: ImageUsage>(
        &mut self,
        src: BufferSlice<'a, T>,
        dst: ImageSlice<'a, F, U>,
    ) where
        F: ColorAspect,
    {
        let _span = tracing::info_span!("copy_buffer_to_image");
        self.flush_pending(
            iter::once(&ImageAccess {
                access: vk::AccessFlags2::TRANSFER_WRITE,
                stage: vk::PipelineStageFlags2::TRANSFER,
                image: dst.view.image,
                layout: vk::ImageLayout::GENERAL,
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
    pub fn raster<'a>(&'a mut self) -> RasterBuilder<'a> {
        RasterBuilder {
            write_depth: true,
            clear_depth: None,
            cmd_buf: self,
            color_accesses: SmallVec::new(),
            color_attachments: SmallVec::new(),
            depth_attachment: vk::ImageView::null(),
            hash: RasterHash {
                backface_culling: true,
                color_formats: SmallVec::new(),
                depth_format: vk::Format::UNDEFINED,
                stencil_format: vk::Format::UNDEFINED,
                wire_frame: false,
            },
            depth_access: None,
        }
    }

    fn compute_private<
        'b,
        GpuBinding: Pod,
        T: ComputePass,
        const Images: usize,
        const Buffers: usize,
    >(
        &mut self,
        bindings: BindingOutput<GpuBinding, T, Images, Buffers>,
        dispatch: [u32; 3],
        indirect_buffer: Option<BufferSlice<'b, DrawIndirectCommand>>,
    ) {
        let _span = tracing::info_span!("compute");
        let access1 = indirect_buffer.map(|b| BufferAccess {
            access: vk::AccessFlags2::INDIRECT_COMMAND_READ,
            stage: vk::PipelineStageFlags2::DRAW_INDIRECT,
            range: b.get_range(),
        });
        self.flush_pending(
            bindings.images.iter(),
            bindings.buffers.iter().chain(access1.iter()),
        );
        self.push_constants(&bindings.gpu_bindings);

        let pipeline = pipelines().compute_pipelines.read().unwrap()[PASS_MAP[T::PASS_INDEX].index];

        unsafe {
            Ctx::device().cmd_bind_pipeline(self.handle, vk::PipelineBindPoint::COMPUTE, pipeline);
            if let Some(slice) = indirect_buffer {
                Ctx::device().cmd_dispatch_indirect(self.handle, slice.handle, slice.offset());
            } else {
                Ctx::device().cmd_dispatch(self.handle, dispatch[0], dispatch[1], dispatch[2]);
            }
        }
    }

    #[validation_trace]
    pub fn compute_indirect<
        'b,
        GpuBinding: Pod,
        T: ComputePass,
        const Images: usize,
        const Buffers: usize,
    >(
        &mut self,
        bindings: BindingOutput<GpuBinding, T, Images, Buffers>,
        buffer: BufferSlice<'b, DrawIndirectCommand>,
    ) {
        self.compute_private(bindings, [0, 0, 0], Some(buffer));
    }

    #[validation_trace]
    pub fn compute<
        'b,
        GpuBinding: Pod,
        T: ComputePass,
        const Images: usize,
        const Buffers: usize,
    >(
        &mut self,
        bindings: BindingOutput<GpuBinding, T, Images, Buffers>,
        dispatch: [u32; 3],
    ) {
        self.compute_private(bindings, dispatch, None);
    }

    pub fn raytrace<
        'b,
        GpuBinding: Pod,
        T: RaytracingPass,
        const Images: usize,
        const Buffers: usize,
    >(
        &mut self,
        bindings: BindingOutput<GpuBinding, T, Images, Buffers>,
        x: u32,
        y: u32,
    ) {
        self.flush_pending(bindings.images.iter(), bindings.buffers.iter());
        self.push_constants(&bindings.gpu_bindings);

        let lock = pipelines().raytracing_pipelines.read().unwrap();
        let pipeline = lock[PASS_MAP[T::PASS_INDEX].index].as_ref().unwrap();
        unsafe {
            Ctx::device().cmd_bind_pipeline(
                self.handle,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                pipeline.pipeline,
            );
            let call_region = vk::StridedDeviceAddressRegionKHR::default();
            Functions::raytracing_pipeline().unwrap().cmd_trace_rays(
                self.handle,
                &pipeline.sbt.raygen_region,
                &pipeline.sbt.hit_region,
                &pipeline.sbt.miss_region,
                &call_region,
                x,
                y,
                1,
            );
        };
    }

    #[validation_trace]
    pub fn present<'a, F: Format, U: ImageUsage>(
        &'a mut self,
        swapchain_image: ImageView<'a, F, U>,
    ) {
        let _span = tracing::info_span!("present_barriers");
        self.flush_pending(
            iter::once(&ImageAccess {
                access: vk::AccessFlags2::empty(),
                stage: vk::PipelineStageFlags2::empty(),
                image: swapchain_image.image,
                layout: vk::ImageLayout::PRESENT_SRC_KHR,
            }),
            iter::empty(),
        );
    }

    fn push_constants<'b, T: NoUninit>(&mut self, binding: &T) {
        let constants = bytes_of(binding);
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
            let is_write = buffer_access.access.intersects(all_write_access());
            let is_read = buffer_access.access.intersects(all_read_access());

            if is_write {
                for pending in self
                    .pending_accesses
                    .buffer_reads
                    .iter()
                    .filter(|r| !r.range.intersect(buffer_access.range).is_empty())
                {
                    src_stage |= pending.stage;
                    src_access |= pending.access;
                    dst_stage |= buffer_access.stage;
                    dst_access |= buffer_access.access;
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

        for image_access in image_acceses.clone() {
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
                    src_access |= pending.access;
                    dst_stage |= image_access.stage;
                    dst_access |= image_access.access;
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

        let mut image_barriers: SmallVec<[vk::ImageMemoryBarrier2; 1]> = SmallVec::new();
        let mut has_transition = false;
        for image_access in image_acceses.clone() {
            if image_access.layout != vk::ImageLayout::GENERAL
                && image_access.layout != vk::ImageLayout::PRESENT_SRC_KHR
            {
                continue;
            }

            let mut old_layout = vk::ImageLayout::GENERAL;
            let mut src_stage = vk::PipelineStageFlags2::ALL_COMMANDS;
            let mut src_access = vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE;
            for p in self
                .pending_accesses
                .image_reads
                .iter()
                .chain(self.pending_accesses.image_writes.iter())
                .filter(|r| r.image == image_access.image)
            {
                src_stage |= p.stage;
                src_access |= p.access;
            }
            for (img, layout) in &self.pending_accesses.image_layouts {
                if *img == image_access.image {
                    old_layout = *layout;
                }
            }

            let needs_transition = old_layout != image_access.layout
                && (old_layout == vk::ImageLayout::PRESENT_SRC_KHR
                    || image_access.layout == vk::ImageLayout::PRESENT_SRC_KHR);
            if !needs_transition {
                continue;
            }

            let mut dst_stage = image_access.stage;
            let mut dst_access = image_access.access;
            if dst_stage.is_empty() {
                dst_stage = vk::PipelineStageFlags2::ALL_COMMANDS;
            }
            if dst_access.is_empty() {
                dst_access = vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE;
            }

            image_barriers.push(
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(src_stage)
                    .src_access_mask(src_access)
                    .dst_stage_mask(dst_stage)
                    .dst_access_mask(dst_access)
                    .old_layout(old_layout)
                    .new_layout(image_access.layout)
                    .image(image_access.image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_array_layer: 0,
                        layer_count: 1,
                        base_mip_level: 0,
                        level_count: 1,
                        ..Default::default()
                    }),
            );
            has_transition = true;
        }

        if (src_stage.is_empty() && dst_stage.is_empty()) && !has_transition {
            self.record_accesses(image_acceses, buffer_acceses);
            return;
        }

        if src_stage.is_empty() {
            src_stage = vk::PipelineStageFlags2::ALL_COMMANDS;
        }
        if dst_stage.is_empty() {
            dst_stage = vk::PipelineStageFlags2::ALL_COMMANDS;
        }
        if src_access.is_empty() {
            src_access = vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE;
        }

        let barrier = vk::MemoryBarrier2::default()
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access);
        let barriers = [barrier];
        let mut info = vk::DependencyInfo::default().memory_barriers(&barriers);
        if has_transition {
            info = info.image_memory_barriers(&image_barriers);
        }

        unsafe {
            Ctx::device().cmd_pipeline_barrier2(self.handle, &info);
        }

        self.record_accesses(image_acceses, buffer_acceses);
    }

    fn record_accesses<'a>(
        &mut self,
        image_acceses: impl Iterator<Item = &'a ImageAccess>,
        buffer_acceses: impl Iterator<Item = &'a BufferAccess>,
    ) {
        for access in buffer_acceses {
            let overlaps = |a: &BufferAccess| !a.range.intersect(access.range).is_empty();
            let writes = access.access.intersects(all_write_access());
            let reads = access.access.intersects(all_read_access());

            if writes {
                self.pending_accesses.buffer_writes.retain(|a| !overlaps(a));
                self.pending_accesses.buffer_writes.push(access.clone());
            }
            if reads {
                self.pending_accesses.buffer_reads.retain(|a| !overlaps(a));
                self.pending_accesses.buffer_reads.push(access.clone());
            }
        }

        for access in image_acceses {
            let same_image = |a: &ImageAccess| a.image == access.image;
            let writes = access.access.intersects(all_write_access());
            let reads = access.access.intersects(all_read_access());

            if writes {
                self.pending_accesses
                    .image_writes
                    .retain(|a| !same_image(a));
                self.pending_accesses.image_writes.push(access.clone());
            }
            if reads {
                self.pending_accesses.image_reads.retain(|a| !same_image(a));
                self.pending_accesses.image_reads.push(access.clone());
            }

            let layouts = &mut self.pending_accesses.image_layouts;
            let raw = access.image;
            if let Some((_, layout)) = layouts.iter_mut().find(|(img, _)| *img == raw) {
                *layout = access.layout;
            } else {
                layouts.push((raw, access.layout));
            }
        }
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

use std::{
    cell::LazyCell,
    collections::HashMap,
    ffi::CStr,
    mem::MaybeUninit,
    sync::{Arc, LazyLock, Mutex, MutexGuard, Once, OnceLock},
};

use crate::{
    bindless::Bindless, command_buffer::{DrawIndexedIndirectCommand, DrawIndirectCommand}, state::{Ctx, Functions}, vkobjects::{
        buffer::{self, Buffer, Location},
        image::Image,
        rt_pipeline::{
            RayTracingShaderCreateInfo, RayTracingShaderGroup, RaytracingPipeline,
            ShaderBindingTable,
        },
    }
};
use anyhow::Result;
use ash::vk::{self};
use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};

#[derive(Clone, Hash, PartialEq, Eq, Default)]
pub struct ComputePipelineHandle {
    pub path: ShaderPath,
}

impl ComputePipelineHandle {
    pub fn dispatch(&self, cmd: &vk::CommandBuffer, x: u32, y: u32, z: u32) {
        let pipeline = PipelineCache::get().get_compute_pipeline(self);
        unsafe {
            Ctx::device().cmd_bind_pipeline(*cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            Ctx::device().cmd_dispatch(*cmd, x, y, z);
        }
    }
    pub fn dispatch_indirect(&self, cmd: &vk::CommandBuffer, buffer: vk::Buffer, offset: u32) {
        let pipeline = PipelineCache::get().get_compute_pipeline(self);
        unsafe {
            Ctx::device().cmd_bind_pipeline(*cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            Ctx::device().cmd_dispatch_indirect(*cmd, buffer, offset as u64);
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Default)]
pub struct RayTracingPipelineHandle {
    pub path: ShaderPath,
}

impl RayTracingPipelineHandle {
    pub fn launch(&self, cmd: &vk::CommandBuffer, x: u32, y: u32) {
        let mut binding = PipelineCache::get();
        let pipeline = binding.get_raytracing_pipeline(self);
        unsafe {
            Ctx::device().cmd_bind_pipeline(
                *cmd,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                pipeline.pipeline,
            );
            let call_region = vk::StridedDeviceAddressRegionKHR::default();
            Functions::raytracing_pipeline().unwrap().cmd_trace_rays(
                *cmd,
                &pipeline.sbt.raygen_region,
                &pipeline.sbt.miss_region,
                &pipeline.sbt.hit_region,
                &call_region,
                x,
                y,
                1,
            );
        };
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct ShaderPath {
    pub path: &'static str,
    pub entry: &'static str,
}

impl Default for ShaderPath {
    fn default() -> Self {
        Self {
            entry: "main",
            path: "",
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Default)]
pub struct RasterPipelineHandle {
    pub fragment: ShaderPath,
    pub backface_culling: bool,
    pub model: PipelineModel,
}

#[derive(Clone, Hash, Eq, PartialEq)]
pub enum PipelineModel {
    Mesh {
        task: Option<ShaderPath>,
        mesh: ShaderPath,
    },
    Vertex {
        vertex: ShaderPath,
        vertex_buffer: bool,
    },
}

impl Default for PipelineModel {
    fn default() -> Self {
        Self::Vertex {
            vertex: ShaderPath::default(),
            vertex_buffer: false,
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct RasterPipelineHash {
    handle: RasterPipelineHandle,
    color_formats: Vec<vk::Format>,
    depth_format: vk::Format,
    stencil_format: vk::Format,
}

#[derive(Clone, Copy)]
pub enum RasterDispatch {
    LaunchMesh {
        x: u32,
        y: u32,
        z: u32,
    },
    Draw {
        vertex_count: u32,
        instance_count: u32,
    },
    DrawIndexed {
        triangle_count: u32,
        instance_count: u32,
    },
    DrawIndirect {
        buffer: vk::Buffer,
        offset: u32,
        count: u32,
    },
    DrawIndexedIndirect {
        buffer: vk::Buffer,
        offset: u32,
        count: u32,
    },
    DrawIndirectCount {
        buffer: vk::Buffer,
        offset: u32,
        count_buffer: vk::Buffer,
        count_offset: u32,
    },
    DrawIndexedIndirectCount {
        buffer: vk::Buffer,
        offset: u32,
        count_buffer: vk::Buffer,
        count_offset: u32,
    },
}

impl RasterDispatch {
    pub fn launch_mesh(x: u32, y: u32, z: u32) -> Self {
        Self::LaunchMesh { x, y, z }
    }
    pub fn draw(vertex_count: u32, instance_count: u32) -> Self {
        Self::Draw {
            vertex_count,
            instance_count,
        }
    }
    pub fn indexed(triangle_count: u32, instance_count: u32) -> Self {
        Self::DrawIndexed {
            triangle_count,
            instance_count,
        }
    }
    pub fn indirect<T: Copy+Pod, L: Location>(
        buffer: &Buffer<T, L>,
        offset: u32,
        count: u32,
    ) -> Self {
        Self::DrawIndirect {
            buffer: buffer.handle,
            offset,
            count,
        }
    }
    pub fn indexed_indirect<T: Copy+Pod, L: Location>(
        buffer: &Buffer<T, L>,
        offset: u32,
        count: u32,
    ) -> Self {
        Self::DrawIndexedIndirect {
            buffer: buffer.handle,
            offset,
            count,
        }
    }
    pub fn indirect_count<T: Copy+Pod, L: Location>(
        buffer: &Buffer<T, L>,
        offset: u32,
        count_buffer: vk::Buffer,
        count_offset: u32,
    ) -> Self {
        Self::DrawIndirectCount {
            buffer: buffer.handle,
            offset,
            count_buffer,
            count_offset,
        }
    }
    pub fn indexed_indirect_count<L: Location>(
        buffer: &Buffer<DrawIndexedIndirectCommand, L>,
        offset: u32,
        count_buffer: vk::Buffer,
        count_offset: u32,
    ) -> Self {
        Self::DrawIndexedIndirectCount {
            buffer: buffer.handle,
            offset,
            count_buffer,
            count_offset,
        }
    }
}

#[derive(Debug, Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
}

impl RasterPipelineHandle {
    pub fn dispatch(
        &self,
        cmd: vk::CommandBuffer,
        color_attachments: &[(Image, Option<[f32; 4]>)],
        depth_attachment: Option<&Image>,
        stencil_attachment: Option<&Image>,
        vertex_buffer: Option<&vk::Buffer>,
        width: u32,
        height: u32,
        dispatch: RasterDispatch,
    ) {
        let mut cache = PipelineCache::get();
        let color_formats = color_attachments
            .iter()
            .map(|e| e.0.format)
            .collect::<Vec<_>>();
        let depth_format = depth_attachment
            .and_then(|d| Some(d.format))
            .unwrap_or(vk::Format::UNDEFINED);
        let stencil_format = stencil_attachment
            .and_then(|d| Some(d.format))
            .unwrap_or(vk::Format::UNDEFINED);

        let pipeline = cache.get_raster_pipeline(self, color_formats, depth_format, stencil_format);

        let color_attachments = color_attachments
            .iter()
            .map(|e| {
                let ret = vk::RenderingAttachmentInfo::default()
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .image_view(e.0.view)
                    .store_op(vk::AttachmentStoreOp::STORE);
                if let Some(clear_color) = e.1 {
                    ret.clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: clear_color,
                        },
                    })
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                } else {
                    ret.load_op(vk::AttachmentLoadOp::LOAD)
                }
            })
            .collect::<Vec<_>>();
        let mut rendering_info = vk::RenderingInfo::default()
            .color_attachments(color_attachments.as_slice())
            .layer_count(1)
            .render_area(
                vk::Rect2D::default()
                    .offset(vk::Offset2D { x: 0, y: 0 })
                    .extent(vk::Extent2D { width, height }),
            )
            .view_mask(0);

        let render_info1;
        let render_info2;

        if let Some(depth_attachment) = &depth_attachment {
            render_info1 = vk::RenderingAttachmentInfo::default()
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .image_view(depth_attachment.view)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 0.0,
                        stencil: 0,
                    },
                })
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE);
            rendering_info = rendering_info.depth_attachment(&render_info1);
        }
        if let Some(stencil_attachment) = &stencil_attachment {
            render_info2 = vk::RenderingAttachmentInfo::default()
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image_view(stencil_attachment.view)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 0.0,
                        stencil: 0,
                    },
                })
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE);
            rendering_info = rendering_info.stencil_attachment(&render_info2);
        }

        unsafe {
            Ctx::device().cmd_begin_rendering(cmd, &rendering_info);
            Ctx::device().cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);

            Ctx::device().cmd_set_viewport(
                cmd,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: Ctx::window_width().unwrap_or(0) as f32,
                    height: Ctx::window_height().unwrap_or(0) as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            Ctx::device().cmd_set_scissor(
                cmd,
                0,
                &[vk::Rect2D {
                    extent: vk::Extent2D {
                        width: Ctx::window_width().unwrap_or(0),
                        height: Ctx::window_height().unwrap_or(0),
                    },
                    offset: vk::Offset2D { x: 0, y: 0 },
                }],
            );
            match &self.model {
                PipelineModel::Mesh { task, mesh } => match dispatch {
                    RasterDispatch::LaunchMesh { x, y, z } => {
                        Functions::mesh().unwrap().cmd_draw_mesh_tasks(cmd, x, y, z);
                    }
                    _ => panic!("Invalid dispatch for mesh pipeline"),
                },
                PipelineModel::Vertex {
                    vertex,
                    vertex_buffer: use_vertex_buffer,
                } => {
                    if let Some(vertex_buffer) = vertex_buffer {
                        Ctx::device().cmd_bind_vertex_buffers(cmd, 0, &[*vertex_buffer], &[0]);
                    }
                    match dispatch {
                        RasterDispatch::Draw {
                            vertex_count,
                            instance_count,
                        } => Ctx::device().cmd_draw(cmd, vertex_count, instance_count, 0, 0),
                        RasterDispatch::DrawIndexed {
                            triangle_count,
                            instance_count,
                        } => Ctx::device().cmd_draw_indexed(
                            cmd,
                            triangle_count * 3,
                            instance_count,
                            0,
                            0,
                            0,
                        ),
                        RasterDispatch::DrawIndirect {
                            buffer,
                            offset,
                            count,
                        } => Ctx::device().cmd_draw_indirect(
                            cmd,
                            buffer,
                            offset as u64,
                            count,
                            size_of::<vk::DrawIndirectCommand>() as u32,
                        ),
                        RasterDispatch::DrawIndexedIndirect {
                            buffer,
                            offset,
                            count,
                        } => Ctx::device().cmd_draw_indexed_indirect(
                            cmd,
                            buffer,
                            offset as u64,
                            count,
                            size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                        ),
                        RasterDispatch::DrawIndirectCount {
                            buffer,
                            offset,
                            count_buffer,
                            count_offset,
                        } => Ctx::device().cmd_draw_indirect_count(
                            cmd,
                            buffer,
                            offset as u64,
                            count_buffer,
                            count_offset as u64,
                            u32::MAX,
                            size_of::<vk::DrawIndirectCommand>() as u32,
                        ),
                        RasterDispatch::DrawIndexedIndirectCount {
                            buffer,
                            offset,
                            count_buffer,
                            count_offset,
                        } => Ctx::device().cmd_draw_indexed_indirect_count(
                            cmd,
                            buffer,
                            offset as u64,
                            count_buffer,
                            count_offset as u64,
                            u32::MAX,
                            size_of::<vk::DrawIndirectCommand>() as u32,
                        ),
                        RasterDispatch::LaunchMesh { x, y, z } => {
                            panic!("Invalid dispatch for vertex pipeline")
                        }
                    }
                }
            }
            Ctx::device().cmd_end_rendering(cmd);
        };
    }
}

pub struct PipelineCache {
    compute_pipelines: HashMap<ComputePipelineHandle, vk::Pipeline>,
    raster_pipelines: HashMap<RasterPipelineHash, vk::Pipeline>,
    raytracing_pipelines: HashMap<RayTracingPipelineHandle, RaytracingPipeline>,
    shader_cache: HashMap<String, vk::ShaderModule>,
}

pub static CACHE: LazyLock<Mutex<PipelineCache>> =
    LazyLock::new(|| Mutex::new(PipelineCache::new()));

use json::JsonValue;
use std::io::Read;

impl PipelineCache {
    pub fn get<'a>() -> MutexGuard<'a, PipelineCache> {
        CACHE.lock().unwrap()
    }

    pub fn create_shader_module(&mut self, code_path: &str) -> Result<vk::ShaderModule> {
        match self.shader_cache.get(code_path) {
            Some(module) => Ok(module.clone()),
            None => {
                let mut code = std::fs::File::open(code_path)?;
                let decoded_code = ash::util::read_spv(&mut code)?;
                let create_info = vk::ShaderModuleCreateInfo::default().code(&decoded_code);

                let module = unsafe { Ctx::device().create_shader_module(&create_info, None)? };

                self.shader_cache
                    .insert(code_path.to_string(), module.clone());
                Ok(module)
            }
        }
    }

    fn create_shader_stage<'a>(
        &mut self,
        code_path: &str,
        main: &'a str,
        stage: vk::ShaderStageFlags,
    ) -> Result<vk::PipelineShaderStageCreateInfo<'a>> {
        let module = self.create_shader_module(code_path)?;
        Ok(vk::PipelineShaderStageCreateInfo::default()
            .stage(stage)
            .module(module)
            .name(CStr::from_bytes_with_nul(main.as_bytes())?))
    }

    pub fn new() -> Self {
        Self {
            compute_pipelines: HashMap::new(),
            shader_cache: HashMap::new(),
            raytracing_pipelines: HashMap::new(),
            raster_pipelines: HashMap::new(),
        }
    }

    pub fn get_compute_pipeline(&mut self, handle: &ComputePipelineHandle) -> vk::Pipeline {
        match self.compute_pipelines.get(handle) {
            Some(pipeline) => pipeline.clone(),
            None => {
                let entry = format!("{}\0", handle.path.entry);
                let path = format!("./core/shaders/bin/{}.slang.spv", handle.path.path);

                let create_info = vk::ComputePipelineCreateInfo::default()
                    .layout(Bindless::layout())
                    .stage(
                        self.create_shader_stage(&path, &entry, vk::ShaderStageFlags::COMPUTE)
                            .unwrap(),
                    );
                let pipeline = unsafe {
                    Ctx::device()
                        .create_compute_pipelines(vk::PipelineCache::null(), &[create_info], None)
                        .unwrap()
                }[0];
                Functions::set_debug_name(&handle.path.path, pipeline);
                self.compute_pipelines
                    .insert(handle.clone(), pipeline.clone());
                pipeline
            }
        }
    }

    pub fn get_raytracing_pipeline<'a, 'b>(
        &'b mut self,
        handle: &'a RayTracingPipelineHandle,
    ) -> &'a RaytracingPipeline
    where
        'b: 'a,
    {
        if self.raytracing_pipelines.contains_key(handle) {
            self.raytracing_pipelines.get(handle).unwrap()
        } else {
            let entry = format!("{}\0", handle.path.entry);
            let path = format!("./shaders/bin/{}.slang.spv", handle.path.path,);

            let pipeline = RaytracingPipeline::new(
                Bindless::layout(),
                &[
                    RayTracingShaderCreateInfo {
                        group: RayTracingShaderGroup::RayGen,
                        source: &[(&path, &entry, vk::ShaderStageFlags::RAYGEN_KHR)],
                    },
                    RayTracingShaderCreateInfo {
                        group: RayTracingShaderGroup::Hit,
                        source: &[(
                            "shaders/bin/default_hit",
                            "main\0",
                            vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                        )],
                    },
                    RayTracingShaderCreateInfo {
                        group: RayTracingShaderGroup::Miss,
                        source: &[(
                            "shaders/bin/default_miss",
                            "main\0",
                            vk::ShaderStageFlags::MISS_KHR,
                        )],
                    },
                ],
            )
            .unwrap();
            Functions::set_debug_name(&handle.path.path, pipeline.pipeline);
            self.raytracing_pipelines.insert(handle.clone(), pipeline);
            self.raytracing_pipelines.get(handle).unwrap()
        }
    }

    pub fn get_raster_pipeline(
        &mut self,
        handle: &RasterPipelineHandle,
        color_formats: Vec<vk::Format>,
        depth_format: vk::Format,
        stencil_format: vk::Format,
    ) -> vk::Pipeline {
        let hash = RasterPipelineHash {
            handle: handle.clone(),
            color_formats,
            depth_format,
            stencil_format,
        };
        match self.raster_pipelines.get(&hash) {
            Some(pipeline) => pipeline.clone(),
            None => {
                let mut create_info = vk::GraphicsPipelineCreateInfo::default();

                let mesh_entry = if let PipelineModel::Mesh { task, mesh } = &handle.model {
                    Some(format!("{}\0", mesh.entry))
                } else {
                    None
                };
                let amplicfication_entry = if let PipelineModel::Mesh { task, mesh } = &handle.model
                {
                    task.as_ref().map(|task| format!("{}\0", task.entry))
                } else {
                    None
                };
                let vertex_entry = if let PipelineModel::Vertex {
                    vertex,
                    vertex_buffer,
                } = &handle.model
                {
                    Some(format!("{}\0", vertex.entry))
                } else {
                    None
                };
                let mut rendering = vk::PipelineRenderingCreateInfo::default()
                    .color_attachment_formats(&hash.color_formats)
                    .depth_attachment_format(depth_format)
                    .stencil_attachment_format(stencil_format)
                    .view_mask(0);

                let fragment_entry = format!("{}\0", handle.fragment.entry);
                let fragment_path =
                    format!("./core/shaders/bin/{}.slang.spv", handle.fragment.path);
                let mut stages = vec![
                    self.create_shader_stage(
                        &fragment_path,
                        &fragment_entry,
                        vk::ShaderStageFlags::FRAGMENT,
                    )
                    .unwrap(),
                ];
                let vertex_input_state;
                let input_assembly;
                let vertex_bindings;
                let vertex_attribute_descriptions;
                match &handle.model {
                    PipelineModel::Mesh { task, mesh } => {
                        let mesh_path = format!("./core/shaders/bin/{}.slang.spv", mesh.path);
                        stages.push(
                            self.create_shader_stage(
                                &mesh_path,
                                mesh_entry.as_ref().unwrap(),
                                vk::ShaderStageFlags::MESH_EXT,
                            )
                            .unwrap(),
                        );
                        if let Some(task) = task {
                            let amplicfication_path =
                                format!("./core/shaders/bin/{}.slang.spv", task.path);
                            stages.push(
                                self.create_shader_stage(
                                    &amplicfication_path,
                                    amplicfication_entry.as_ref().unwrap(),
                                    vk::ShaderStageFlags::TASK_EXT,
                                )
                                .unwrap(),
                            );
                        }
                    }
                    PipelineModel::Vertex {
                        vertex,
                        vertex_buffer,
                    } => {
                        let vertex_path = format!("./core/shaders/bin/{}.slang.spv", vertex.path);
                        stages.push(
                            self.create_shader_stage(
                                &vertex_path,
                                &vertex_entry.as_ref().unwrap(),
                                vk::ShaderStageFlags::VERTEX,
                            )
                            .unwrap(),
                        );
                        if *vertex_buffer {
                            vertex_bindings = [vk::VertexInputBindingDescription::default()
                                .binding(0)
                                .stride(size_of::<Vertex>() as u32)
                                .input_rate(vk::VertexInputRate::VERTEX)];

                            vertex_attribute_descriptions = [
                                vk::VertexInputAttributeDescription::default()
                                    .binding(0)
                                    .location(0)
                                    .format(vk::Format::R32G32B32_SFLOAT)
                                    .offset(0),
                                vk::VertexInputAttributeDescription::default()
                                    .binding(0)
                                    .location(1)
                                    .format(vk::Format::R32G32B32_SFLOAT)
                                    .offset(12),
                                vk::VertexInputAttributeDescription::default()
                                    .binding(0)
                                    .location(2)
                                    .format(vk::Format::R32G32_SFLOAT)
                                    .offset(24),
                            ];

                            vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
                                .vertex_binding_descriptions(&vertex_bindings)
                                .vertex_attribute_descriptions(&vertex_attribute_descriptions);

                            input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
                                .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                                .primitive_restart_enable(false);
                            create_info = create_info
                                .vertex_input_state(&vertex_input_state)
                                .input_assembly_state(&input_assembly);
                        }else {
                            vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
                                    .vertex_attribute_descriptions(&[])
                                    .vertex_binding_descriptions(&[]);
                            input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
                                .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                                .primitive_restart_enable(false);
                            create_info = create_info
                                .vertex_input_state(&vertex_input_state)
                                .input_assembly_state(&input_assembly);
                        }
                    }
                }

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
                    .map(|e| {
                        vk::PipelineColorBlendAttachmentState::default()
                            .blend_enable(false)
                            .color_write_mask(vk::ColorComponentFlags::RGBA)
                            .alpha_blend_op(vk::BlendOp::ADD)
                            .color_blend_op(vk::BlendOp::ADD)
                            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                            .dst_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                            .src_alpha_blend_factor(vk::BlendFactor::SRC_ALPHA)
                            .src_color_blend_factor(vk::BlendFactor::SRC_COLOR)
                    })
                    .collect::<Vec<_>>();
                let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
                    .attachments(color_blend_attachments.as_slice())
                    .logic_op_enable(false)
                    .logic_op(vk::LogicOp::COPY)
                    .blend_constants([0.0, 0.0, 0.0, 0.0]);

                let viewport_state = vk::PipelineViewportStateCreateInfo::default()
                    .scissor_count(1)
                    .viewport_count(1);
                let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
                    .depth_clamp_enable(false)
                    .rasterizer_discard_enable(false)
                    .line_width(1.0)
                    .polygon_mode(vk::PolygonMode::FILL)
                    .cull_mode(if handle.backface_culling {
                        vk::CullModeFlags::BACK
                    } else {
                        vk::CullModeFlags::NONE
                    })
                    .front_face(vk::FrontFace::CLOCKWISE)
                    .depth_bias_enable(true);
                let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
                    .depth_bounds_test_enable(false)
                    .depth_compare_op(vk::CompareOp::GREATER)
                    .depth_test_enable(true)
                    .depth_write_enable(true)
                    .min_depth_bounds(1.0)
                    .max_depth_bounds(0.0)
                    .stencil_test_enable(false);

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

                self.raster_pipelines.insert(hash, pipeline.clone());
                pipeline
            }
        }
    }
}

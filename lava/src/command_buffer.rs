use std::{any, cell::{LazyCell, OnceCell}, collections::HashMap, ffi::CStr, fmt::Debug, marker::PhantomData, ops::{Index, IndexMut}, sync::{Mutex, OnceLock}};

use anyhow::Result;
use ash::vk::{self, PipelineStageFlags2, ShaderStageFlags};
use bytemuck::{Pod, Zeroable, bytes_of};

use crate::{
    bindless::Bindless,
    state::{Ctx, Functions},
    vkobjects::{
        buffer::{Buffer, CpuBuffer, GpuBuffer, Location, StorageBuffer},
        image::Image, rt_pipeline::{RayTracingShaderCreateInfo, RayTracingShaderGroup, RaytracingPipeline},
    },
};

pub struct CommandBuffer<'a> {
    pub(crate) handle: vk::CommandBuffer,
    pub(crate) resource_hashes: &'a mut HashMap<ResourceHandle, ResourceState>,
}


#[derive(Clone, Copy, Hash, PartialEq, Eq, Default)]
pub struct ShaderHash {
    pub entry: &'static str,
    pub file: &'static str,
} 


fn create_module(bytes: &[u8]) -> vk::ShaderModule {
    let decoded_code = ash::util::read_spv(&mut std::io::Cursor::new(bytes)).unwrap();
    let create_info = vk::ShaderModuleCreateInfo::default().code(&decoded_code);

    unsafe { Ctx::device().create_shader_module(&create_info, None).unwrap() }
}

fn create_shader_stage<'a>(entry: &'a str, bytes: &[u8], stage: vk::ShaderStageFlags) -> (vk::ShaderModule, vk::PipelineShaderStageCreateInfo<'a>) {
    let module = create_module(bytes);
    (module, make_shader_stage(entry, stage, module))
}


fn make_shader_stage<'a>(entry: &'a str, stage: vk::ShaderStageFlags, module: vk::ShaderModule) -> vk::PipelineShaderStageCreateInfo<'a> {
    vk::PipelineShaderStageCreateInfo::default()
        .stage(stage)
        .module(module)
        .name(CStr::from_bytes_with_nul(entry.as_bytes()).unwrap())
}

pub trait ComputePass {
    type GpuBinding: Binding;
    const ENTRY: &'static str;
    const BYTES: &[u8];

    fn cache() -> &'static OnceLock<vk::Pipeline>;

    fn get() -> vk::Pipeline {
        Self::cache().get_or_init(|| {
            let (module, stage) = create_shader_stage(Self::ENTRY, Self::BYTES, vk::ShaderStageFlags::COMPUTE);
            let create_info = vk::ComputePipelineCreateInfo::default()
                .layout(Bindless::layout())
                .stage(stage);
            let pipeline = unsafe {
                Ctx::device()
                .create_compute_pipelines(vk::PipelineCache::null(), &[create_info], None)
                .unwrap()
            }[0];
            Functions::set_debug_name(Self::ENTRY, pipeline);
            unsafe { Ctx::device().destroy_shader_module(module, None) };
            pipeline
        }).clone()
    }
}

pub trait RayTracingPass {
    type GpuBinding: Binding;
    const RAYGEN: &'static str;
    const HIT: &'static str;
    const MISS: &'static str;
    const BYTES: &[u8];

    fn cache() -> &'static OnceLock<RaytracingPipeline>;

    fn get<'a>() -> &'a RaytracingPipeline {
        Self::cache().get_or_init(|| {
            let module = create_module(Self::BYTES);
            let raygen = make_shader_stage(Self::RAYGEN, vk::ShaderStageFlags::RAYGEN_KHR, module); 
            let hit = make_shader_stage(Self::HIT, vk::ShaderStageFlags::CLOSEST_HIT_KHR, module); 
            let miss = make_shader_stage(Self::RAYGEN, vk::ShaderStageFlags::MISS_KHR, module); 

            let pipeline = RaytracingPipeline::new(
                Bindless::layout(),
                &[
                    RayTracingShaderCreateInfo {
                        group: RayTracingShaderGroup::RayGen,
                        stages: &[raygen]
                    },
                    RayTracingShaderCreateInfo {
                        group: RayTracingShaderGroup::Hit,
                        stages: &[hit]
                    },
                    RayTracingShaderCreateInfo {
                        group: RayTracingShaderGroup::Miss,
                        stages: &[miss]
                    },
                ],
            )
            .unwrap();
            Functions::set_debug_name(&Self::RAYGEN, pipeline.pipeline);
            unsafe { Ctx::device().destroy_shader_module(module, None) };
            pipeline
        })
    }
}


#[derive(Hash, PartialEq, Eq, Clone)]
pub struct RasterHash {
    backface_culling: bool,
    vertex_buffer: bool,
    color_formats: Vec<vk::Format>,
    depth_format: vk::Format,
    stencil_format: vk::Format,
} 

pub trait RasterVertexShaderPass: RasterPass {
    const VERTEX: &'static str;
    const FRAGMENT: &'static str;
    const BYTES: &[u8];
    type Vertex: Pod;

    fn module_cache() -> &'static OnceLock<vk::ShaderModule>;
    
    fn pipeline_cache() -> &'static Mutex<LazyCell<HashMap<RasterHash, vk::Pipeline>>>;

    fn get(hash: &RasterHash) -> vk::Pipeline {
        let p_cache = Self::pipeline_cache();
        let mutex = p_cache.lock();
        mutex.unwrap().entry(hash.clone()).or_insert({
            let m_cache = Self::module_cache();
            let module = m_cache.get_or_init(|| create_module(Self::BYTES));
            let stages = vec![
                make_shader_stage(Self::FRAGMENT, vk::ShaderStageFlags::FRAGMENT, *module),
                make_shader_stage(Self::VERTEX, vk::ShaderStageFlags::VERTEX, *module),
            ];

            let vertex_bindings; let vertex_attribute_descriptions;
            let (vertex_input_state, input_assembly) = if hash.vertex_buffer {
                vertex_bindings = [vk::VertexInputBindingDescription::default()
                    .binding(0)
                    .stride(size_of::<Self::Vertex>() as u32)
                    .input_rate(vk::VertexInputRate::VERTEX)];
                vertex_attribute_descriptions= [
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
                (vk::PipelineVertexInputStateCreateInfo::default()
                    .vertex_binding_descriptions(&vertex_bindings)
                    .vertex_attribute_descriptions(&vertex_attribute_descriptions)
                ,vk::PipelineInputAssemblyStateCreateInfo::default()
                    .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                    .primitive_restart_enable(false)
                )
            } else {
                (vk::PipelineVertexInputStateCreateInfo::default()
                    .vertex_attribute_descriptions(&[])
                    .vertex_binding_descriptions(&[]),
                vk::PipelineInputAssemblyStateCreateInfo::default()
                    .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                    .primitive_restart_enable(false))
            };
            create_raster_pipeline(&stages, hash, Some((vertex_input_state, input_assembly)))
        }).clone()
    }
}


pub trait RasterMeshShaderPass: RasterPass {
    const MESH: &'static str;
    const FRAGMENT: &'static str;
    const TASK: Option<&'static str>;
    const BYTES: &[u8];

    fn module_cache() -> &'static OnceLock<vk::ShaderModule>;
    fn pipeline_cache() -> &'static Mutex<LazyCell<HashMap<RasterHash, vk::Pipeline>>>;

    fn get(hash: &RasterHash) -> vk::Pipeline {
        let pipeline_cache = Self::pipeline_cache();
        let mutex = pipeline_cache.lock();
        let module_cache = Self::module_cache();
        mutex.unwrap().entry(hash.clone()).or_insert({
            let module = module_cache.get_or_init(|| create_module(Self::BYTES));
            let mut stages = vec![
                make_shader_stage(Self::FRAGMENT, vk::ShaderStageFlags::FRAGMENT, *module),
                make_shader_stage(Self::MESH, vk::ShaderStageFlags::MESH_EXT, *module),
            ];
            if let Some(hash) = Self::TASK {
                stages.push(make_shader_stage(hash, vk::ShaderStageFlags::TASK_EXT, *module));
            }
            create_raster_pipeline(&stages, hash, None)
        }).clone()
    }
}


fn create_raster_pipeline(stages: &[vk::PipelineShaderStageCreateInfo<'_>], hash: &RasterHash, ia: Option<(vk::PipelineVertexInputStateCreateInfo, vk::PipelineInputAssemblyStateCreateInfo)>) -> vk::Pipeline{
    let mut create_info = vk::GraphicsPipelineCreateInfo::default();
    if let Some((vertex_input_state, input_assembly)) = &ia {
        create_info = create_info
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly);
    }
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
        .cull_mode(if hash.backface_culling {
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

    pipeline

}

pub trait Binding: Pod {
    type CpuBinding<'a>;
    fn from_cpu_binding<'a>(binding: &Self::CpuBinding<'a>) -> Self;
    fn resources<'a>(binding: &Self::CpuBinding<'a>, stage: ash::vk::PipelineStageFlags2) -> Vec<(ResourceHandle, ResourceState)>;
}

#[derive(Debug)]
pub enum PushConstant {
    BindlessImage(u64),
    BufferPointer(u64),
    Constants(Vec<u8>),
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub enum ResourceHandle {
    Buffer(vk::Buffer),
    Image((vk::ImageView, vk::Image)),
}

#[derive(Clone, Copy, Default, Debug)]
pub struct ResourceState {
    pub stages: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
    pub layout: vk::ImageLayout,
    pub aspect: vk::ImageAspectFlags,
}


pub trait RasterPass {
    type GpuBinding: Binding;
}

pub struct RasterBuilder<'a, 'b, 'c, S: RasterPass> {
    hash: RasterHash,
    color_attachments: Vec<(Image, Option<[f32;4]>)>,
    depth_attachment: Option<Image>,
    vertex_buffer: Option<vk::Buffer>,
    index_buffer: Option<vk::Buffer>,
    resource_states: Vec<(ResourceHandle, ResourceState)>,
    cmd_buf: &'a mut CommandBuffer<'b>,
    binding: Option<<<S as RasterPass>::GpuBinding as Binding>::CpuBinding<'c>>,
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

#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct DrawIndexedIndirectCommand {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub first_instance: u32,
}


#[derive(Clone, Copy)]
pub enum RasterVertexDispatch {
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


impl RasterVertexDispatch {
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
    pub fn indirect<T: Copy + Pod, L: Location>(
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
    pub fn indexed_indirect<T: Copy + Pod, L: Location>(
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
    pub fn indirect_count<T: Copy + Pod, L: Location>(
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

impl<'a, 'b, 'c, S: RasterVertexShaderPass> RasterBuilder<'a,'b,'c,S> {
    pub fn index_buffer<L: Location>(mut self, buffer: &Buffer<u32, L>) -> Self {
        assert!(self.index_buffer.is_none());
        self.index_buffer = Some(buffer.handle);
        self.resource_states.push((
            ResourceHandle::Buffer(buffer.handle),
            ResourceState {
                access: vk::AccessFlags2::INDEX_READ,
                layout: vk::ImageLayout::UNDEFINED,
                stages: vk::PipelineStageFlags2::INDEX_INPUT,
                aspect: vk::ImageAspectFlags::NONE,
            },
        ));
        self
    }
    pub fn vertex_buffer(mut self, buffer: &Buffer<S::Vertex, GpuBuffer>) -> Self {
        assert!(self.vertex_buffer.is_none());
        assert!(size_of::<S::Vertex>() > 0);
        self.vertex_buffer = Some(buffer.handle);
        self.resource_states.push((
            ResourceHandle::Buffer(buffer.handle),
            ResourceState {
                access: vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
                aspect: vk::ImageAspectFlags::NONE,
                layout: vk::ImageLayout::UNDEFINED,
                stages: vk::PipelineStageFlags2::VERTEX_INPUT,
            }
        ));
        self
    }
    pub fn draw(mut self, dispatch: RasterVertexDispatch, width: u32, height: u32) {
        self.hash.color_formats = self.color_attachments
        .iter()
            .map(|e| e.0.format)
            .collect::<Vec<_>>();
        self.hash.depth_format = self.depth_attachment
            .as_ref()
            .and_then(|d| Some(d.format))
            .unwrap_or(vk::Format::UNDEFINED);
        self.hash.vertex_buffer = self.vertex_buffer.is_some();
        let pipeline = S::get(&self.hash);
        self.draw_private(pipeline, Some(dispatch), width, height, [0,0,0]);
    }
    pub fn draw_fullscreen(self, dispatch: RasterVertexDispatch) {
        self.draw(dispatch, Ctx::window_width().unwrap(), Ctx::window_height().unwrap());
    }
}


impl<'a, 'b, 'c, S: RasterMeshShaderPass> RasterBuilder<'a,'b,'c,S> {
    pub fn launch(mut self, x: u32, y: u32, z: u32, width: u32, height: u32) {
        self.hash.color_formats = self.color_attachments
            .iter()
            .map(|e| e.0.format)
            .collect::<Vec<_>>();
        self.hash.depth_format = self.depth_attachment
            .as_ref()
            .and_then(|d| Some(d.format))
            .unwrap_or(vk::Format::UNDEFINED);
        let pipeline = S::get(&self.hash);
        self.draw_private(pipeline, None, width, height, [x,y,z]);
    }
    pub fn launch_fullscrean(self, x: u32, y: u32, z: u32) {
        self.launch(x,y,z, Ctx::window_width().unwrap(), Ctx::window_height().unwrap());
    }
}


impl<'a, 'b, 'c, S: RasterPass> RasterBuilder<'a, 'b, 'c, S> {
    pub fn backface_culling(mut self, backface_culling: bool) -> Self {
        self.hash.backface_culling = backface_culling;
        self
    }

    pub fn color_attachment(mut self, image: &Image, clear: Option<[f32; 4]>) -> Self {
        self
            .color_attachments
            .push((image.clone(), clear));
        self.resource_states.push((
            ResourceHandle::Image((image.view, image.handle)),
            ResourceState {
                access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                stages: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                aspect: vk::ImageAspectFlags::COLOR,
            },
        ));
        self
    }

    pub fn depth_attachment(mut self, image: &Image) -> Self {
        assert!(self.depth_attachment.is_none());
        self.depth_attachment = Some(image.clone());
        self.resource_states.push((
            ResourceHandle::Image((image.view, image.handle)),
            ResourceState {
                access: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
                    | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
                layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                stages: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                aspect: vk::ImageAspectFlags::DEPTH,
            },
        ));
        self
    }

    fn draw_private(mut self, pipeline: vk::Pipeline, dispatch: Option<RasterVertexDispatch>, width: u32, height: u32, launch: [u32; 3]) {
        let buffers = if let Some(dispatch) = &dispatch {
        match dispatch {
            RasterVertexDispatch::DrawIndexedIndirect {
                buffer,
                offset: _,
                count: _,
            } => vec![buffer],
            RasterVertexDispatch::DrawIndexedIndirectCount {
                buffer,
                offset: _,
                count_buffer,
                count_offset: _,
            } => vec![buffer, count_buffer],
            RasterVertexDispatch::DrawIndirect {
                buffer,
                offset: _,
                count: _,
            } => vec![buffer],
            RasterVertexDispatch::DrawIndirectCount {
                buffer,
                offset: _,
                count_buffer,
                count_offset: _,
            } => vec![buffer, count_buffer],
            _ => vec![],
        }}
        else {
            vec![]
        };
        for buffer in buffers {
            self.resource_states.push((
                ResourceHandle::Buffer(*buffer),
                ResourceState {
                    access: vk::AccessFlags2::INDIRECT_COMMAND_READ,
                    layout: vk::ImageLayout::UNDEFINED,
                    stages: vk::PipelineStageFlags2::DRAW_INDIRECT,
                    aspect: vk::ImageAspectFlags::NONE,
                },
            ));
        }
        let stage = if dispatch.is_some() {
            vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::VERTEX_SHADER
        }else {
            vk::PipelineStageFlags2::MESH_SHADER_EXT
        };
        let mut shader_resouces = S::GpuBinding::resources(self.binding.as_ref().unwrap(), stage);
        self.resource_states.append(&mut shader_resouces);
        self.cmd_buf.barriers(self.resource_states);
        self.cmd_buf.push_constants::<S::GpuBinding>(self.binding.as_ref().unwrap());

        let color_attachments = self.color_attachments
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

        if let Some(depth_attachment) = &self.depth_attachment {
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

        unsafe {
            Ctx::device().cmd_begin_rendering(self.cmd_buf.handle, &rendering_info);
            Ctx::device().cmd_bind_pipeline(self.cmd_buf.handle, vk::PipelineBindPoint::GRAPHICS, pipeline);

            Ctx::device().cmd_set_viewport(
                self.cmd_buf.handle,
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
                self.cmd_buf.handle,
                0,
                &[vk::Rect2D {
                    extent: vk::Extent2D {
                        width: Ctx::window_width().unwrap_or(0),
                        height: Ctx::window_height().unwrap_or(0),
                    },
                    offset: vk::Offset2D { x: 0, y: 0 },
                }],
            );
            if let Some(dispatch) = dispatch {
                if let Some(vertex_buffer) = self.vertex_buffer {
                    Ctx::device().cmd_bind_vertex_buffers(self.cmd_buf.handle, 0, &[vertex_buffer], &[0]);
                }
                match dispatch {
                    RasterVertexDispatch::Draw {
                        vertex_count,
                        instance_count,
                    } => Ctx::device().cmd_draw(self.cmd_buf.handle, vertex_count, instance_count, 0, 0),
                    RasterVertexDispatch::DrawIndexed {
                        triangle_count,
                        instance_count,
                    } => Ctx::device().cmd_draw_indexed(
                        self.cmd_buf.handle,
                        triangle_count * 3,
                        instance_count,
                        0,
                        0,
                        0,
                    ),
                    RasterVertexDispatch::DrawIndirect {
                        buffer,
                        offset,
                        count,
                    } => Ctx::device().cmd_draw_indirect(
                        self.cmd_buf.handle,
                        buffer,
                        offset as u64,
                        count,
                        size_of::<vk::DrawIndirectCommand>() as u32,
                    ),
                    RasterVertexDispatch::DrawIndexedIndirect {
                        buffer,
                        offset,
                        count,
                    } => Ctx::device().cmd_draw_indexed_indirect(
                        self.cmd_buf.handle,
                        buffer,
                        offset as u64,
                        count,
                        size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                    ),
                    RasterVertexDispatch::DrawIndirectCount {
                        buffer,
                        offset,
                        count_buffer,
                        count_offset,
                    } => Ctx::device().cmd_draw_indirect_count(
                        self.cmd_buf.handle,
                        buffer,
                        offset as u64,
                        count_buffer,
                        count_offset as u64,
                        u32::MAX,
                        size_of::<vk::DrawIndirectCommand>() as u32,
                    ),
                    RasterVertexDispatch::DrawIndexedIndirectCount {
                        buffer,
                        offset,
                        count_buffer,
                        count_offset,
                    } => Ctx::device().cmd_draw_indexed_indirect_count(
                        self.cmd_buf.handle,
                        buffer,
                        offset as u64,
                        count_buffer,
                        count_offset as u64,
                        u32::MAX,
                        size_of::<vk::DrawIndirectCommand>() as u32,
                    ),
                }
            }else {
                Functions::mesh().unwrap().cmd_draw_mesh_tasks(self.cmd_buf.handle, launch[0], launch[1], launch[2]);
            }
            Ctx::device().cmd_end_rendering(self.cmd_buf.handle);
        };
    }

    pub fn bind(mut self, b: <<S as RasterPass>::GpuBinding as Binding>::CpuBinding<'c>) -> Self {
        self.binding = Some(b);
        self
    }}

pub struct ComputeBuilder<'a, 'b, 'c, S: ComputePass> {
    cmd_buffer: &'a mut CommandBuffer<'b>,
    binding: Option<<<S as ComputePass>::GpuBinding as Binding>::CpuBinding<'c>>,
}

impl<'a, 'b, 'c, S: ComputePass> ComputeBuilder<'a, 'b, 'c, S> {
    pub fn bind(mut self, b: <<S as ComputePass>::GpuBinding as Binding>::CpuBinding<'c>) -> Self {
        self.binding = Some(b);
        self
    }

        
    fn build(self, dispatch: [u32; 3], indirect_buffer: Option<(vk::Buffer, u32)>) {
        let mut resources = S::GpuBinding::resources(self.binding.as_ref().unwrap(), vk::PipelineStageFlags2::COMPUTE_SHADER);
        if let Some(indirect) = indirect_buffer {
            resources.push((
                ResourceHandle::Buffer(indirect.0),
                ResourceState {
                    access: vk::AccessFlags2::INDIRECT_COMMAND_READ,
                    stages: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    ..Default::default()
                },
            ));
        }
        self.cmd_buffer.barriers(resources);
        self.cmd_buffer.push_constants::<S::GpuBinding>(self.binding.as_ref().unwrap());

        let pipeline = S::get();

        unsafe {
            Ctx::device().cmd_bind_pipeline(self.cmd_buffer.handle, vk::PipelineBindPoint::COMPUTE, pipeline);
            if let Some((buffer, offset)) = indirect_buffer {
                Ctx::device().cmd_dispatch_indirect(self.cmd_buffer.handle, buffer, offset as u64);
            } else {
                Ctx::device().cmd_dispatch(self.cmd_buffer.handle, dispatch[0], dispatch[1], dispatch[2]);
            }
        }
    }

    pub fn dispatch_indirect<L: Location, T: Copy + Pod>(
        mut self,
        buffer: &Buffer<T, L>,
        offset: u32,
    ) {
        self.build([0, 0, 0], Some((buffer.handle, offset as u32)));
    }

    pub fn dispatch(self, x: u32, y: u32, z: u32) {
        self.build([x, y, z], None);
    }
    pub fn dispatch_fullscreen(self) {
        self.build(
            [
                Ctx::window_width().unwrap().div_ceil(8),
                Ctx::window_height().unwrap().div_ceil(8),
                1,
            ],
            None,
        );
    }
    pub fn dispatch_fractional_fullscreen(self, x: u32, y: u32) {
        self.build(
            [
                Ctx::window_width().unwrap().div_ceil(x),
                Ctx::window_height().unwrap().div_ceil(y),
                1,
            ],
            None,
        );
    }
}

pub struct RayTracingBuilder<'a, 'b, 'c, S: RayTracingPass> {
    binding: Option<<<S as RayTracingPass>::GpuBinding as Binding>::CpuBinding<'c>>,
    cmd_buffer: &'a mut CommandBuffer<'b>,
}

impl<'a, 'b, 'c, S: RayTracingPass> RayTracingBuilder<'a, 'b, 'c, S> {
    pub fn bind(mut self, b: <<S as RayTracingPass>::GpuBinding as Binding>::CpuBinding<'c>) -> Self {
        self.binding = Some(b);
        self
    }

    fn build(self, dispatch: [u32; 2]) {
        let resources = S::GpuBinding::resources(self.binding.as_ref().unwrap(), vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR);
        self.cmd_buffer.barriers(resources);
        self.cmd_buffer.push_constants::<S::GpuBinding>(self.binding.as_ref().unwrap());

        let pipeline = S::get();
        unsafe {
            Ctx::device().cmd_bind_pipeline(
                self.cmd_buffer.handle,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                pipeline.pipeline,
            );
            let call_region = vk::StridedDeviceAddressRegionKHR::default();
            Functions::raytracing_pipeline().unwrap().cmd_trace_rays(
                self.cmd_buffer.handle,
                &pipeline.sbt.raygen_region,
                &pipeline.sbt.miss_region,
                &pipeline.sbt.hit_region,
                &call_region,
                dispatch[0],
                dispatch[1],
                1,
            );
        };
    }

    pub fn dispatch(self, x: u32, y: u32) {
        self.build([x, y]);
    }
    pub fn dispatch_fullscreen(self) {
        self.build([Ctx::window_width().unwrap(), Ctx::window_height().unwrap()]);
    }
    pub fn dispatch_fractional_fullscreen(self, x: u32, y: u32) {
        self.build([
            Ctx::window_width().unwrap().div_ceil(x),
            Ctx::window_height().unwrap().div_ceil(y),
        ]);
    }
}

impl<'b> CommandBuffer<'b> {
    pub fn fill_buffer<T: Copy + Pod, L: Location>(
        &mut self,
        buffer: &Buffer<T, L>,
        offset: u32,
        data: u32,
    ) {
        self.barriers(vec![(
            ResourceHandle::Buffer(buffer.handle),
            ResourceState {
                access: vk::AccessFlags2::TRANSFER_WRITE,
                stages: vk::PipelineStageFlags2::TRANSFER,
                ..Default::default()
            },
        )]);

        unsafe {
            Ctx::device().cmd_fill_buffer(
                self.handle,
                buffer.handle,
                offset as u64,
                buffer.size - offset as u64,
                data,
            )
        };
    }
    pub fn update_buffer_element<T: Copy + Pod, L: Location>(
        &mut self,
        buffer: &Buffer<T, L>,
        element: usize,
        data: &T,
    ) {
        self.barriers(vec![(
            ResourceHandle::Buffer(buffer.handle),
            ResourceState {
                access: vk::AccessFlags2::TRANSFER_WRITE,
                stages: vk::PipelineStageFlags2::TRANSFER,
                ..Default::default()
            },
        )]);

        unsafe {
            Ctx::device().cmd_update_buffer(
                self.handle,
                buffer.handle,
                (element * size_of::<T>()) as u64,
                bytemuck::bytes_of(data),
            )
        };
    }
    pub fn copy_buffer<T: Copy + Pod, L: Location, B: Location>(
        &mut self,
        src: &Buffer<T, L>,
        dst: &Buffer<T, B>,
        num_elements: usize,
        src_offset: u32,
        dst_offset: u32,
    ) {
        self.barriers(vec![
            (
                ResourceHandle::Buffer(src.handle),
                ResourceState {
                    access: vk::AccessFlags2::TRANSFER_READ,
                    stages: vk::PipelineStageFlags2::TRANSFER,
                    ..Default::default()
                },
            ),
            (
                ResourceHandle::Buffer(src.handle),
                ResourceState {
                    access: vk::AccessFlags2::TRANSFER_WRITE,
                    stages: vk::PipelineStageFlags2::TRANSFER,
                    ..Default::default()
                },
            ),
        ]);
        let num_bytes = num_elements * size_of::<T>();
        unsafe {
            Ctx::device().cmd_copy_buffer(
                self.handle,
                src.handle,
                dst.handle,
                &[vk::BufferCopy {
                    src_offset: src_offset as u64,
                    dst_offset: dst_offset as u64,
                    size: num_bytes as u64,
                }],
            )
        };
    }

    pub fn read_buffer<T: Copy + Pod>(
        &mut self,
        buffer: &Buffer<T, GpuBuffer>,
        staging: &Buffer<T, CpuBuffer>,
        num_elements: usize,
        offset: usize,
    ) -> Vec<T> {
        if !cfg!(debug_assertions) {
            log::warn!("Using read_buffer in release can cause performance problems!");
        }
        self.copy_buffer(
            buffer,
            staging,
            num_elements,
            offset as u32 * size_of::<T>() as u32,
            0,
        );
        self.barriers(vec![(
            ResourceHandle::Buffer(staging.handle),
            ResourceState {
                access: vk::AccessFlags2::HOST_READ,
                stages: vk::PipelineStageFlags2::HOST,
                ..Default::default()
            },
        )]);
        staging.read_len(num_elements)
    }

    pub fn raster<'a, 'c, S: RasterPass>(&'a mut self) -> RasterBuilder<'a, 'b, 'c, S> {
        RasterBuilder {
            cmd_buf: self,
            color_attachments: Vec::new(),
            depth_attachment: None,
            index_buffer: None,
            resource_states: Vec::new(),
            vertex_buffer: None,
            binding: None,
            hash: RasterHash { backface_culling: true, vertex_buffer: false, color_formats: Vec::new(), depth_format: vk::Format::UNDEFINED, stencil_format: vk::Format::UNDEFINED }
        }
    }

    pub fn compute<'a, 'c, S: ComputePass>(&'a mut self) -> ComputeBuilder<'a, 'b, 'c, S> {
        ComputeBuilder {
            cmd_buffer: self,
            binding: None,
        }
    }
    pub fn raytrace<'a, 'c, S: RayTracingPass>(&'a mut self) -> RayTracingBuilder<'a, 'b, 'c, S> {
        RayTracingBuilder {
            binding: None,
            cmd_buffer: self,
        }
    }

    pub fn present(&mut self, swapchain_image: Image) {
        self.barriers(vec![(
            ResourceHandle::Image((swapchain_image.view, swapchain_image.handle)),
            ResourceState {
                access: vk::AccessFlags2::empty(),
                aspect: vk::ImageAspectFlags::COLOR,
                layout: vk::ImageLayout::PRESENT_SRC_KHR,
                stages: vk::PipelineStageFlags2::NONE,
            },
        )]);
    }

    fn push_constants<'a, B: Binding>(&mut self, binding: &B::CpuBinding<'a>) {
        let binding = B::from_cpu_binding(binding);
        let constants = bytes_of(&binding);
        unsafe { Ctx::device().cmd_push_constants(self.handle, Bindless::layout(), vk::ShaderStageFlags::ALL, 0, constants) };
    }

    fn barriers(&mut self, resources: Vec<(ResourceHandle, ResourceState)>) {
        let mut image_barriers = Vec::new();
        let mut buffer_barriers = Vec::new();
        for (resource, new) in resources {
            let prev = self
                .resource_hashes
                .get(&resource)
                .copied()
                .unwrap_or(ResourceState {
                    stages: vk::PipelineStageFlags2::TOP_OF_PIPE,
                    access: vk::AccessFlags2::NONE,
                    layout: vk::ImageLayout::UNDEFINED,
                    aspect: vk::ImageAspectFlags::COLOR,
                });
            // fast path: same layout/access/queue and no write->read hazard => no barrier
            let read_to_read = prev.access.contains(vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_SAMPLED_READ)
                && !prev.access.contains(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                && new.access.contains(vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_SAMPLED_READ)
                && !new.access.contains(vk::AccessFlags2::SHADER_STORAGE_WRITE);
            let same_layout = prev.layout == new.layout;
            let first_use = prev.stages.contains(vk::PipelineStageFlags2::TOP_OF_PIPE);

            let need_barrier = !read_to_read || !same_layout || !first_use;

            if need_barrier {
                // src/dst stages & access: from prev -> next
                let src_stage_mask = prev.stages;
                let dst_stage_mask = new.stages;

                match resource {
                    ResourceHandle::Buffer(buffer) => {
                        buffer_barriers.push(vk::BufferMemoryBarrier2 {
                            src_access_mask: prev.access,
                            dst_access_mask: new.access,
                            src_stage_mask,
                            dst_stage_mask,
                            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                            buffer: buffer,
                            offset: 0,
                            size: vk::WHOLE_SIZE,
                            ..Default::default()
                        })
                    }
                    ResourceHandle::Image((_, image)) => {
                        image_barriers.push(vk::ImageMemoryBarrier2 {
                            src_access_mask: prev.access,
                            dst_access_mask: new.access,
                            src_stage_mask,
                            dst_stage_mask,
                            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                            image: image,
                            old_layout: prev.layout,
                            new_layout: new.layout,
                            subresource_range: vk::ImageSubresourceRange {
                                aspect_mask: new.aspect,
                                base_array_layer: 0,
                                base_mip_level: 0,
                                layer_count: 1,
                                level_count: 1,
                            },
                            ..Default::default()
                        })
                    }
                };
            }
            self.resource_hashes.insert(resource.clone(), new.clone());
        }
        if !image_barriers.is_empty() || !buffer_barriers.is_empty() {
            let dependency_info = vk::DependencyInfo::default()
                .buffer_memory_barriers(&buffer_barriers)
                .image_memory_barriers(&image_barriers)
                .dependency_flags(vk::DependencyFlags::BY_REGION);
            unsafe { Ctx::device().cmd_pipeline_barrier2(self.handle, &dependency_info) };
        }
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

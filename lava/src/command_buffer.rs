use std::collections::HashMap;

use ash::vk;
use glam::Mat4;

use crate::{
    bindless::Bindless,
    pipelines::{
        ComputePipelineHandle, PipelineModel, RasterDispatch, RasterPipelineHandle, RayTracingPipelineHandle, ShaderPath, Vertex
    },
    state::Ctx,
    vkobjects::{
        buffer::{Buffer, Location},
        image::Image,
        rt_pipeline::alinged_size,
    },
};

enum Command {
    Raster {
        pipeline: RasterPipelineHandle,
        dispatch: RasterDispatch,
        width: u32,
        height: u32,
    },
    Raytracing {
        pipeline: RayTracingPipelineHandle,
        dispatch: [u32; 2],
    },
    Compute {
        pipeline: ComputePipelineHandle,
        dispatch: [u32; 3],
    },
    Present,
}

#[derive(Debug)]
pub enum PushConstant {
    BindlessImage(u64),
    BufferPointer(u64),
    Constants(Vec<u8>),
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub enum ResourceHandle {
    Buffer(vk::Buffer),
    Image((vk::ImageView, vk::Image)),
}

#[derive(Clone, Copy)]
pub struct ResourceState {
    stages: vk::PipelineStageFlags2,
    access: vk::AccessFlags2,
    layout: vk::ImageLayout,
    aspect: vk::ImageAspectFlags,
}

pub struct Action {
    command: Command,
    push_constants: Option<Vec<PushConstant>>,
    color_attachments: Option<Vec<(Image, Option<[f32; 4]>)>>,
    depth_attachments: Option<Image>,
    vertex_buffer: Option<vk::Buffer>,
    index_buffer: Option<vk::Buffer>,
    resources: Vec<(ResourceHandle, ResourceState)>,
}

pub struct CommandBuffer {
    pub(crate) handle: vk::CommandBuffer,
    pub(crate) commands: Vec<Action>,
    pub(crate) resource_hashes: HashMap<ResourceHandle, ResourceState>,
}

#[derive(Default)]
pub struct RasterBuilder {
    pipeline_handle: RasterPipelineHandle,
    color_attachments: Vec<(Image, Option<[f32; 4]>)>,
    depth_attachments: Option<Image>,
    vertex_buffer: Option<vk::Buffer>,
    index_buffer: Option<vk::Buffer>,
}

pub trait IntoShaderResourceHandle {
    fn push_constant(&self) -> PushConstant;
    fn resource_handle(&self) -> Option<ResourceHandle>;
    fn aspect(&self) -> vk::ImageAspectFlags;
    fn preferd_default_layout(&self) -> Option<vk::ImageLayout>;
}

impl<T: Copy> IntoShaderResourceHandle for Buffer<T> {
    fn push_constant(&self) -> PushConstant {
        PushConstant::BufferPointer(self.ptr())
    }
    fn resource_handle(&self) -> Option<ResourceHandle> {
        self.buffer.as_ref().map(|b| ResourceHandle::Buffer(b.buffer))
    }
    fn preferd_default_layout(&self) -> Option<vk::ImageLayout> {
        None
    }
    fn aspect(&self) -> vk::ImageAspectFlags {
        vk::ImageAspectFlags::NONE
    }
}

impl IntoShaderResourceHandle for Image {
    fn push_constant(&self) -> PushConstant {
        PushConstant::BindlessImage(self.bindless_handle.expect("Image is neither a texture nor a storage image, consider adding either vk::Sampled or vk::StorageImage to your image usage flags.") as u64)
    }
    fn resource_handle(&self) -> Option<ResourceHandle> {
        Some(ResourceHandle::Image((self.view, self.image)))
    }
    fn preferd_default_layout(&self) -> Option<vk::ImageLayout> {
        if self.usage.contains(vk::ImageUsageFlags::STORAGE) {
            Some(vk::ImageLayout::GENERAL)
        } else {
            None
        }
    }
    fn aspect(&self) -> vk::ImageAspectFlags {
        if self.format == vk::Format::D32_SFLOAT || self.format == vk::Format::D16_UNORM || self.format == vk::Format::D16_UNORM_S8_UINT || self.format == vk::Format::D24_UNORM_S8_UINT || self.format == vk::Format::D32_SFLOAT_S8_UINT {
            vk::ImageAspectFlags::DEPTH
        }else {
            vk::ImageAspectFlags::COLOR
        }
    }
}

impl IntoShaderResourceHandle for u64 {
    fn push_constant(&self) -> PushConstant {
        PushConstant::BufferPointer(*self)
    }
    fn resource_handle(&self) -> Option<ResourceHandle> {
        None
    }
    fn preferd_default_layout(&self) -> Option<vk::ImageLayout> {
        None
    }
    fn aspect(&self) -> vk::ImageAspectFlags {
        vk::ImageAspectFlags::NONE
    }
}

pub struct CommandBuilder<'a, T: Default> {
    push_constants: Vec<PushConstant>,
    resources: Vec<(ResourceHandle, ResourceState)>,
    sub_builder: T,
    cmd_buffer: &'a mut CommandBuffer,
}

impl<'a, T: Default> CommandBuilder<'a, T> {
    pub fn resource_access(
        mut self,
        value: &impl IntoShaderResourceHandle,
        access: vk::AccessFlags2,
        layout: vk::ImageLayout,
    ) -> Self {
        self.push_constants.push(value.push_constant());
        if let Some(v) = value.resource_handle() {
            let mut stages =
                vk::PipelineStageFlags2::VERTEX_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER;
            if Ctx::features().raytracing {
                stages |= vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR;
            }
            self.resources.push((
                v,
                ResourceState {
                    access,
                    layout,
                    stages,
                    aspect: value.aspect(),
                },
            ));
        }
        self
    }
    pub fn read(self, read: &impl IntoShaderResourceHandle) -> Self {
        self.resource_access(
            read,
            vk::AccessFlags2::SHADER_READ,
            read.preferd_default_layout()
                .unwrap_or(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        )
    }
    pub fn readwrite(self, read: &impl IntoShaderResourceHandle) -> Self {
        self.resource_access(
            read,
            vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
            vk::ImageLayout::GENERAL,
        )
    }
    pub fn write(self, write: &impl IntoShaderResourceHandle) -> Self {
        self.resource_access(
            write,
            vk::AccessFlags2::SHADER_WRITE,
            vk::ImageLayout::GENERAL,
        )
    }
    pub fn constant<A: Clone>(mut self, value: &A) -> Self {
        let mut slice = [value.clone()];
        let mut vec = Vec::new();
        unsafe {
            vec.extend_from_slice(std::slice::from_raw_parts(
                slice.as_mut_ptr() as *mut u8,
                size_of::<A>(),
            ))
        };
        self.push_constants.push(PushConstant::Constants(vec));
        self
    }
}

impl<'a> CommandBuilder<'a, RasterBuilder> {
    pub fn fragment(mut self, path: &'static str, entry: &'static str) -> Self {
        self.sub_builder.pipeline_handle.fragment = crate::pipelines::ShaderPath { path, entry };
        self
    }
    pub fn fragment_path(mut self, path: &'static str) -> Self {
        self.sub_builder.pipeline_handle.fragment = crate::pipelines::ShaderPath {
            path,
            entry: "main",
        };
        self
    }

    pub fn vertex(mut self, path: &'static str, entry: &'static str) -> Self {
        self.sub_builder.pipeline_handle.model = crate::pipelines::PipelineModel::Vertex {
            vertex: ShaderPath { entry, path },
        };
        self
    }
    pub fn vertex_path(mut self, path: &'static str) -> Self {
        self.sub_builder.pipeline_handle.model = crate::pipelines::PipelineModel::Vertex {
            vertex: ShaderPath {
                entry: "main",
                path,
            },
        };
        self
    }

    pub fn mesh(mut self, path: &'static str, entry: &'static str) -> Self {
        if let PipelineModel::Mesh { task, mesh } = &mut self.sub_builder.pipeline_handle.model {
            mesh.entry = entry;
            mesh.path = path;
        }else {
            self.sub_builder.pipeline_handle.model = PipelineModel::Mesh { task: None, mesh: ShaderPath { path, entry } }
        }
        self
    }
    pub fn task(mut self, path: &'static str, entry: &'static str) -> Self {
        if let PipelineModel::Mesh { task, mesh } = &mut self.sub_builder.pipeline_handle.model {
            *task = Some(ShaderPath { entry, path })
        }else {
            self.sub_builder.pipeline_handle.model = PipelineModel::Mesh { task: Some(ShaderPath { path, entry }), mesh: ShaderPath {path: "", entry: ""} }
        }
        self
    }

    pub fn color_attachment(mut self, image: &Image, clear: Option<[f32; 4]>) -> Self {
        self.sub_builder
            .color_attachments
            .push((image.clone(), clear));
        self.resources.push((
            ResourceHandle::Image((image.view, image.image)),
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
        assert!(self.sub_builder.depth_attachments.is_none());
        self.sub_builder.depth_attachments = Some(image.clone());
        self.resources.push((
            ResourceHandle::Image((image.view, image.image)),
            ResourceState {
                access: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
                    | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
                layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                stages: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
                aspect: vk::ImageAspectFlags::DEPTH,
            },
        ));
        self
    }

    pub fn vertex_buffer(mut self, buffer: &Buffer<Vertex>) -> Self {
        assert!(self.sub_builder.vertex_buffer.is_none());
        let buffer = buffer.buffer.as_ref().map(|e| e.buffer).unwrap();
        self.sub_builder.vertex_buffer = Some(buffer);
        self.resources.push((
            ResourceHandle::Buffer(buffer),
            ResourceState {
                access: vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
                layout: vk::ImageLayout::UNDEFINED,
                stages: vk::PipelineStageFlags2::VERTEX_INPUT,
                aspect: vk::ImageAspectFlags::NONE,
            },
        ));
        self
    }

    pub fn index_buffer(mut self, buffer: &Buffer<u32>) -> Self {
        assert!(self.sub_builder.index_buffer.is_none());
        let buffer = buffer.buffer.as_ref().map(|e| e.buffer).unwrap();
        self.sub_builder.index_buffer = Some(buffer);
        self.resources.push((
            ResourceHandle::Buffer(buffer),
            ResourceState {
                access: vk::AccessFlags2::INDEX_READ,
                layout: vk::ImageLayout::UNDEFINED,
                stages: vk::PipelineStageFlags2::INDEX_INPUT,
                aspect: vk::ImageAspectFlags::NONE,
            },
        ));
        self
    }


    pub fn draw(mut self, dispatch: RasterDispatch, width: u32, height: u32) {
        let buffers = match &dispatch {
            RasterDispatch::DrawIndexedIndirect { buffer, offset, count } => vec![buffer],
            RasterDispatch::DrawIndexedIndirectCount { buffer, offset, count_buffer, count_offset } => vec![buffer, count_buffer],
            RasterDispatch::DrawIndirect { buffer, offset, count } => vec![buffer],
            RasterDispatch::DrawIndirectCount { buffer, offset, count_buffer, count_offset } => vec![buffer, count_buffer],
            _ => vec![],
        };
        for buffer in buffers {
            self.resources.push((
                ResourceHandle::Buffer(*buffer),
                ResourceState {
                    access: vk::AccessFlags2::INDIRECT_COMMAND_READ,
                    layout: vk::ImageLayout::UNDEFINED,
                    stages: vk::PipelineStageFlags2::DRAW_INDIRECT,
                    aspect: vk::ImageAspectFlags::NONE,
                },
            ));
        }
        self.cmd_buffer.commands.push(Action {
            color_attachments: Some(self.sub_builder.color_attachments),
            depth_attachments: self.sub_builder.depth_attachments,
            command: Command::Raster{
                pipeline:  self.sub_builder.pipeline_handle,
                dispatch,
                width,
                height,
            },
            vertex_buffer: self.sub_builder.vertex_buffer,
            push_constants: Some(self.push_constants),
            resources: self.resources,
            index_buffer: self.sub_builder.index_buffer,
        });
    }

    pub fn draw_fullscreen(self, dispatch: RasterDispatch) {
        self.draw(
            dispatch,
            Ctx::window_width().unwrap(),
            Ctx::window_height().unwrap(),
        );
    }
}

#[derive(Default)]
pub struct ComputeBuilder {
    pipeline_handle: ComputePipelineHandle,
}

impl<'a> CommandBuilder<'a, ComputeBuilder> {
    pub fn shader(mut self, path: &'static str, entry: &'static str) -> Self {
        self.sub_builder.pipeline_handle.path = ShaderPath { path, entry };
        self
    }
    pub fn shader_path(mut self, path: &'static str) -> Self {
        self.sub_builder.pipeline_handle.path = ShaderPath {
            path,
            entry: "main",
        };
        self
    }

    fn build(self, dispatch: [u32; 3]) {
        self.cmd_buffer.commands.push(Action {
            color_attachments: None,
            depth_attachments: None,
            command: Command::Compute{
                pipeline: self.sub_builder.pipeline_handle,
                dispatch,
            },
            push_constants: Some(self.push_constants),
            resources: self.resources,
            vertex_buffer: None,
            index_buffer: None,
        });
    }

    pub fn dispatch(self, x: u32, y: u32, z: u32) {
        self.build([x, y, z]);
    }
    pub fn dispatch_fullscreen(self) {
        self.build([
            Ctx::window_width().unwrap().div_ceil(8),
            Ctx::window_height().unwrap().div_ceil(8),
            1,
        ]);
    }
    pub fn dispatch_fractional_fullscreen(self, x: u32, y: u32) {
        self.build([
            Ctx::window_width().unwrap().div_ceil(x),
            Ctx::window_height().unwrap().div_ceil(y),
            1,
        ]);
    }
}

#[derive(Default)]
struct RayTracingBuilder {
    pipeline_handle: RayTracingPipelineHandle,
}

impl<'a> CommandBuilder<'a, RayTracingBuilder> {
    pub fn shader(mut self, path: &'static str, entry: &'static str) -> Self {
        self.sub_builder.pipeline_handle.path = ShaderPath { path, entry };
        self
    }
    pub fn shader_path(mut self, path: &'static str) -> Self {
        self.sub_builder.pipeline_handle.path = ShaderPath {
            path,
            entry: "main",
        };
        self
    }

    fn build(self, dispatch: [u32; 2]) {
        self.cmd_buffer.commands.push(Action {
            color_attachments: None,
            depth_attachments: None,
            command: Command::Raytracing {
                pipeline: self.sub_builder.pipeline_handle,
                dispatch,
            },
            push_constants: Some(self.push_constants),
            resources: self.resources,
            vertex_buffer: None,
            index_buffer: None,
        });
    }

    pub fn dispatch(self, x: u32, y: u32) {
        self.build([x, y]);
    }
    pub fn dispatch_fullscreen(self) {
        self.build([
            Ctx::window_width().unwrap(),
            Ctx::window_height().unwrap(),
        ]);
    }
    pub fn dispatch_fractional_fullscreen(self, x: u32, y: u32) {
        self.build([
            Ctx::window_width().unwrap().div_ceil(x),
            Ctx::window_height().unwrap().div_ceil(y),
        ]);
    }
}

impl CommandBuffer {
    pub fn raster<'a>(&'a mut self) -> CommandBuilder<'a, RasterBuilder> {
        CommandBuilder {
            resources: Vec::new(),
            sub_builder: RasterBuilder::default(),
            cmd_buffer: self,
            push_constants: Vec::new(),
        }
    }
    pub fn compute<'a>(&'a mut self) -> CommandBuilder<'a, ComputeBuilder> {
        CommandBuilder {
            resources: Vec::new(),
            sub_builder: ComputeBuilder::default(),
            cmd_buffer: self,
            push_constants: Vec::new(),
        }
    }
    pub fn raytrace<'a>(&'a mut self) -> CommandBuilder<'a, RayTracingBuilder> {
        CommandBuilder {
            resources: Vec::new(),
            sub_builder: RayTracingBuilder::default(),
            cmd_buffer: self,
            push_constants: Vec::new(),
        }
    }

    pub fn present(&mut self, swapchain_image: Image) {
        self.commands.push(Action {
            color_attachments: None,
            depth_attachments: None,
            push_constants: None,
            command: Command::Present,
            resources: vec![(
                ResourceHandle::Image((swapchain_image.view, swapchain_image.image)),
                ResourceState {
                    access: vk::AccessFlags2::empty(),
                    aspect: vk::ImageAspectFlags::COLOR,
                    layout: vk::ImageLayout::PRESENT_SRC_KHR,
                    stages: vk::PipelineStageFlags2::NONE,
                },
            )],
            vertex_buffer: None,
            index_buffer: None,
        });
    }

    pub fn record(&mut self) {
        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe { Ctx::device().begin_command_buffer(self.handle, &begin_info) }.unwrap();
        Bindless::bind(&self.handle);

        for (i, aktion) in self.commands.iter().enumerate() {
            let mut data = vec![0; Ctx::physical_device().limits.max_push_constants_size as usize];
            let mut index = 0;
            if let Some(constants) = &aktion.push_constants {
                for c in constants {
                    let mut bytes = vec![];
                    match c {
                        PushConstant::BindlessImage(n) => bytes.extend_from_slice(&n.to_ne_bytes()),
                        PushConstant::BufferPointer(n) => bytes.extend_from_slice(&n.to_ne_bytes()),
                        PushConstant::Constants(con) => bytes.extend(con),
                    };
                    data[index..(index + bytes.len())].copy_from_slice(&bytes);
                    index += bytes.len();
                }
            }

            unsafe {
                Ctx::device().cmd_push_constants(
                    self.handle,
                    Bindless::layout(),
                    vk::ShaderStageFlags::ALL,
                    0,
                    &data,
                )
            };

            let mut image_barriers = Vec::new();
            let mut buffer_barriers = Vec::new();
            for (resource, new) in &aktion.resources {
                let prev = self
                    .resource_hashes
                    .get(&resource)
                    .copied()
                    .unwrap_or(ResourceState {
                        stages: vk::PipelineStageFlags2::empty(),
                        access: vk::AccessFlags2::empty(),
                        layout: vk::ImageLayout::UNDEFINED,
                        aspect: vk::ImageAspectFlags::COLOR,
                    });
                // fast path: same layout/access/queue and no write->read hazard => no barrier
                let read_to_read = prev.access.contains(vk::AccessFlags2::SHADER_READ)
                    && !prev.access.intersects(vk::AccessFlags2::SHADER_WRITE)
                    && new.access.contains(vk::AccessFlags2::SHADER_READ)
                    && !new.access.intersects(vk::AccessFlags2::SHADER_WRITE);
                let same_layout = prev.layout == new.layout;

                let need_barrier = !read_to_read || !same_layout;

                if need_barrier {
                    // src/dst stages & access: from prev -> next
                    let src_stage_mask = if prev.stages.is_empty() {
                        vk::PipelineStageFlags2::TOP_OF_PIPE
                    } else {
                        prev.stages
                    };
                    let dst_stage_mask = if new.stages.is_empty() {
                        vk::PipelineStageFlags2::BOTTOM_OF_PIPE
                    } else {
                        new.stages
                    };

                    match resource {
                        ResourceHandle::Buffer(buffer) => {
                            buffer_barriers.push(vk::BufferMemoryBarrier2 {
                                src_access_mask: prev.access,
                                dst_access_mask: new.access,
                                src_stage_mask,
                                dst_stage_mask,
                                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                                buffer: *buffer,
                                offset: 0,
                                size: vk::WHOLE_SIZE,
                                ..Default::default()
                            })
                        }
                        ResourceHandle::Image((view, image)) => {
                            image_barriers.push(vk::ImageMemoryBarrier2 {
                                src_access_mask: prev.access,
                                dst_access_mask: new.access,
                                src_stage_mask,
                                dst_stage_mask,
                                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                                image: *image,
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

            match &aktion.command {
                Command::Compute { pipeline, dispatch: [x,y,z] } => pipeline.dispatch(&self.handle, *x, *y, *z),
                Command::Raster { pipeline, dispatch, width, height} => {
                    if let Some(index_buffer) = aktion.index_buffer {
                        unsafe { Ctx::device().cmd_bind_index_buffer(self.handle, index_buffer, 0, vk::IndexType::UINT32) };
                    }
                    pipeline.dispatch(
                        self.handle,
                        aktion.color_attachments.as_ref().unwrap(),
                        aktion.depth_attachments.as_ref(),
                        None,
                        aktion.vertex_buffer.as_ref(),
                            *width,
                            *height,
                            
                            *dispatch
                    )
                },
                Command::Raytracing { pipeline, dispatch } => {
                    pipeline.launch(&self.handle, dispatch[0], dispatch[1])
                }
                Command::Present => {}
            }
        }

        unsafe { Ctx::device().end_command_buffer(self.handle) }.unwrap();
    }
}

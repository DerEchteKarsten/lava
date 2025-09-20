use std::collections::HashMap;

use ash::vk;

use crate::{bindless::Bindless, pipelines::{ComputePipelineHandle, PipelineModel, RasterPipelineHandle, RayTracingPipelineHandle, ShaderPath}, state::Ctx, vkobjects::{buffer::{Buffer, DynamicBuffer}, image::Image}};


enum Command {
    Raster(RasterPipelineHandle),
    Raytracing(RayTracingPipelineHandle),
    Compute(ComputePipelineHandle),
    Present,
}

enum PushConstant {
    BindlessImage(u32),
    BufferPointer(u64),
    Constants(Vec<u8>),
}

pub enum ResourceHandle {
    Buffer(vk::Buffer),
    Image(vk::ImageView),
}

pub struct ResourceState {
    pipeline_barrier_src_stages: vk::PipelineStageFlags2,
    last_access: vk::AccessFlags2,
    layout: vk::ImageLayout,
}

pub struct Action {
    dispatch: [u32; 5],
    command: Command,
    push_constants: Option<Vec<PushConstant>>,
    color_attachments: Option<Vec<(Image, Option<[f32; 4]>)>>,
    depth_attachments: Option<Image>,
    shader_resources: Option<Vec<(ResourceHandle, vk::AccessFlags2)>>,
}

pub struct CommandBuffer {
    pub(crate) handle: vk::CommandBuffer,
    pub(crate) commands: Vec<Action>,
    pub(crate) resource_hashes: HashMap<ResourceHandle, ResourceState>
}

#[derive(Default)]
struct RasterBuilder {
    pipeline_handle: RasterPipelineHandle,
    color_attachments: Vec<(Image, Option<[f32; 4]>)>,
    depth_attachments: Option<Image>,
    dispatch: [u32; 5],
}

trait IntoShaderResourceHandle {
    fn to(&self) -> PushConstant;
    fn vk(&self) -> Option<ResourceHandle>;
}

impl IntoShaderResourceHandle for Buffer {
    fn to(&self) -> PushConstant {
        PushConstant::BufferPointer(self.address)
    }
    fn vk(&self) -> Option<ResourceHandle> {
        Some(ResourceHandle::Buffer(self.buffer))
    }
}

impl IntoShaderResourceHandle for DynamicBuffer {
    fn to(&self) -> PushConstant {
        PushConstant::BufferPointer(self.ptr())
    }
    fn vk(&self) -> Option<ResourceHandle> {
        Some(ResourceHandle::Buffer(self.buffer.buffer))
    }
}

impl IntoShaderResourceHandle for Image {
    fn to(&self) -> PushConstant {
        PushConstant::BindlessImage(self.bindless_handle)
    }
    fn vk(&self) -> Option<ResourceHandle> {
        Some(ResourceHandle::Image(self.view))
    }
}

impl IntoShaderResourceHandle for u64 {
    fn to(&self) -> PushConstant {
        PushConstant::BufferPointer(*self)
    }
    fn vk(&self) -> Option<ResourceHandle> {
        None
    }
}



pub struct CommandBuilder<'a, T: Default> {
    push_constants: Vec<PushConstant>,
    resources: Vec<(ResourceHandle, vk::AccessFlags2)>,
    sub_builder: T,
    cmd_buffer: &'a mut CommandBuffer,
}

impl<'a, T: Default> CommandBuilder<'a, T> {
    pub fn resource_access(mut self, value: &impl IntoShaderResourceHandle, access: vk::AccessFlags2) -> Self {
        self.push_constants.push(value.to());
        if let Some(v) = value.vk() {
            self.resources.push((v, access));
        }
        self
    }
    pub fn read(self, read: &impl IntoShaderResourceHandle) -> Self {
        self.resource_access(read, vk::AccessFlags2::SHADER_READ)
    }
    pub fn readwrite(self, read: &impl IntoShaderResourceHandle) -> Self {
        self.resource_access(read, vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE)
    }
    pub fn write(self, read: &impl IntoShaderResourceHandle) -> Self {
        self.resource_access(read, vk::AccessFlags2::SHADER_WRITE)
    }
    pub fn constant<A: Clone>(mut self, value: &A) -> Self {
        let mut slice = [value.clone()];
        let byte_slice = unsafe { Vec::from_raw_parts(slice.as_mut_ptr() as *mut u8, size_of::<A>(), size_of::<A>()) };
        self.push_constants.push(PushConstant::Constants(byte_slice));
        self
    }
}

impl<'a> CommandBuilder<'a, RasterBuilder> {
    pub fn fragment(mut self, path: &'static str, entry: &'static str) -> Self {
        self.sub_builder.pipeline_handle.fragment = crate::pipelines::ShaderPath { path, entry };
        self
    }
    pub fn fragment_path(mut self, path: &'static str) -> Self {
        self.sub_builder.pipeline_handle.fragment = crate::pipelines::ShaderPath { path, entry: "main" };
        self
    }

    pub fn vertex(mut self, path: &'static str, entry: &'static str) -> Self {
        self.sub_builder.pipeline_handle.model = crate::pipelines::PipelineModel::Vertex { vertex: ShaderPath { entry, path} };
        self
    }
    pub fn vertex_path(mut self, path: &'static str) -> Self {
        self.sub_builder.pipeline_handle.model = crate::pipelines::PipelineModel::Vertex { vertex: ShaderPath { entry: "main", path} };
        self
    }


    pub fn mesh(mut self, path: &'static str, entry: &'static str) -> Self {
        if let PipelineModel::Mesh{task, mesh} = &mut self.sub_builder.pipeline_handle.model {
            mesh.entry = entry;
            mesh.path = path;
        }
        self
    }
    pub fn task(mut self, path: &'static str, entry: &'static str) -> Self {
        if let PipelineModel::Mesh{task, mesh} = &mut self.sub_builder.pipeline_handle.model {
            *task = Some(ShaderPath {
                entry,
                path,
            })
        }
        self
    }

    pub fn color_attachment(mut self, image: &Image, clear: Option<[f32; 4]>) -> Self {
        self.sub_builder.color_attachments.push((image.clone(), clear));
        self.resources.push((ResourceHandle::Image(image.view), vk::AccessFlags2::COLOR_ATTACHMENT_WRITE));
        self
    } 

    pub fn depth_attachment(mut self, image: &Image) -> Self {
        assert!(self.sub_builder.depth_attachments.is_none());
        self.sub_builder.depth_attachments = Some(image.clone());
        self.resources.push((ResourceHandle::Image(image.view), vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE_KHR | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ_KHR));
        self
    } 

    fn build(self, dispatch: [u32; 5]) {
        self.cmd_buffer.commands.push(Action {
            color_attachments: Some(self.sub_builder.color_attachments),
            depth_attachments: self.sub_builder.depth_attachments,
            dispatch,
            command: Command::Raster(self.sub_builder.pipeline_handle),
            push_constants: Some(self.push_constants),
            shader_resources: Some(self.resources),
        });
    }

    pub fn draw(self, x: u32, y: u32, z: u32, width: u32, height: u32) {
        self.build([x,y,z,width,height]);
    }

    pub fn draw_fullscreen(self, x: u32, y: u32, z: u32) {
        self.build([x,y,z,Ctx::window_width().unwrap(), Ctx::window_height().unwrap()]);
    }

    pub fn draw_instances_fullscreen(self, vertex_count: u32, instance_count: u32) {
        self.build([vertex_count, instance_count,0,Ctx::window_width().unwrap(), Ctx::window_height().unwrap()]);
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
        self.sub_builder.pipeline_handle.path = ShaderPath { path, entry: "main" };
        self
    }

    fn build(self, dispatch: [u32; 5]) {
        self.cmd_buffer.commands.push(Action {
            color_attachments: None,
            depth_attachments: None,
            command: Command::Compute(self.sub_builder.pipeline_handle),
            dispatch,
            push_constants: Some(self.push_constants),
            shader_resources: Some(self.resources)
        });
    }

    pub fn dispatch(self, x: u32, y: u32, z: u32) {
        self.build([x,y,z,0,0]);
    }
    pub fn dispatch_fullscreen(self) {
        self.build([Ctx::window_width().unwrap().div_ceil(8),Ctx::window_height().unwrap().div_ceil(8),1,0,0]);
    }
    pub fn dispatch_fractional_fullscreen(self, x: u32, y: u32) {
        self.build([Ctx::window_width().unwrap().div_ceil(x),Ctx::window_height().unwrap().div_ceil(y),1,0,0]);
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
        self.sub_builder.pipeline_handle.path = ShaderPath { path, entry: "main" };
        self
    }

    fn build(self, dispatch: [u32; 5]) {
        self.cmd_buffer.commands.push(Action {
            color_attachments: None,
            depth_attachments: None,
            command: Command::Raytracing(self.sub_builder.pipeline_handle),
            dispatch,
            push_constants: Some(self.push_constants),
            shader_resources: Some(self.resources)
        });
    }

    pub fn dispatch(self, x: u32, y: u32) {
        self.build([x,y,0,0,0]);
    }
    pub fn dispatch_fullscreen(self) {
        self.build([Ctx::window_width().unwrap(),Ctx::window_height().unwrap(),1,0,0]);
    }
    pub fn dispatch_fractional_fullscreen(self, x: u32, y: u32) {
        self.build([Ctx::window_width().unwrap().div_ceil(x),Ctx::window_height().unwrap().div_ceil(y),1,0,0]);
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
                    data[index..bytes.len()].copy_from_slice(&bytes);
                    index += bytes.len();
                }
            }

            unsafe { Ctx::device().cmd_push_constants(self.handle, Bindless::layout(), vk::ShaderStageFlags::ALL, 0, &data) };

            match &aktion.command {
                Command::Compute(pipeline) => {
                    pipeline.dispatch(&self.handle, aktion.dispatch[0], aktion.dispatch[1], aktion.dispatch[2]);
                },
                Command::Raster(pipeline) => {
                    pipeline.dispatch(self.handle, aktion.color_attachments.as_ref().unwrap(), aktion.depth_attachments.as_ref(), None, aktion.dispatch[3], aktion.dispatch[4],  aktion.dispatch[0],  aktion.dispatch[1],  aktion.dispatch[2]);
                },
                Command::Raytracing(pipeline) => {
                    pipeline.launch(&self.handle, aktion.dispatch[0], aktion.dispatch[1]);
                }
                Command::Present => {}
            }
        }
    }
}

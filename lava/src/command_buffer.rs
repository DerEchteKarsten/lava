use std::{
    cell::LazyCell,
    collections::HashMap,
    sync::{LazyLock, Mutex},
};

use ash::vk::{self, Handle};
use glam::Mat4;
use json::JsonValue;

use crate::{
    bindless::Bindless,
    pipelines::{
        ComputePipelineHandle, PipelineCache, PipelineModel, RasterDispatch, RasterPipelineHandle,
        RayTracingPipelineHandle, ShaderPath, Vertex,
    },
    state::Ctx,
    vkobjects::{
        buffer::{Buffer, CpuBuffer, Growable, Location, Size},
        image::Image,
        rt_pipeline::alinged_size,
    },
};

#[derive(Default)]
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
    FillBuffer {
        buffer: vk::Buffer,
        data: u32,
        offset: u32,
    },
    CopyBuffer {
        src: vk::Buffer,
        dst: vk::Buffer,
        src_offset: u32,
        dst_offset: u32,
        num_bytes: u32,
    },
    #[default]
    Present,
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
    stages: vk::PipelineStageFlags2,
    access: vk::AccessFlags2,
    layout: vk::ImageLayout,
    aspect: vk::ImageAspectFlags,
}

#[derive(Debug)]
enum LayoutBlock {
    Constant { size: u32 },
    Type { name: String },
}

#[derive(Default)]
pub struct Action {
    command: Command,
    push_constants: Option<Vec<PushConstant>>,
    #[cfg(debug_assertions)]
    layout_validation: Vec<LayoutBlock>,
    color_attachments: Option<Vec<(Image, Option<[f32; 4]>)>>,
    depth_attachments: Option<Image>,
    vertex_buffer: Option<vk::Buffer>,
    index_buffer: Option<vk::Buffer>,
    resources: Vec<(ResourceHandle, ResourceState)>,
}

pub struct CommandBuffer<'a> {
    pub(crate) handle: vk::CommandBuffer,
    pub(crate) commands: Vec<Action>,
    pub(crate) resource_hashes: &'a mut HashMap<ResourceHandle, ResourceState>,
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
    fn type_name(&self) -> String;
}

impl<T: Copy> IntoShaderResourceHandle for Buffer<T> {
    fn push_constant(&self) -> PushConstant {
        PushConstant::BufferPointer(self.ptr)
    }
    fn resource_handle(&self) -> Option<ResourceHandle> {
        if self.buffer.is_null() {
            None
        }else {
            Some(ResourceHandle::Buffer(self.buffer))
        }
    }
    fn preferd_default_layout(&self) -> Option<vk::ImageLayout> {
        None
    }
    fn aspect(&self) -> vk::ImageAspectFlags {
        vk::ImageAspectFlags::NONE
    }
    fn type_name(&self) -> String {
        let type_name = std::any::type_name::<T>()
            .rsplit("::")
            .next()
            .unwrap()
            .to_string();

        match type_name.as_str() {
            "u8" => "uint8_t",
            "u32" => "uint",
            "i32" => "int",
            "f32" => "float",
            "f64" => "double",
            "Vec2" => "vector",
            "Vec3" => "vector",
            "Vec4" => "vector",
            "Mat3" => "matrix",
            "Mat4" => "matrix",
            _ => type_name.as_str(),
        }
        .to_string()
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
        if self.format == vk::Format::D32_SFLOAT
            || self.format == vk::Format::D16_UNORM
            || self.format == vk::Format::D16_UNORM_S8_UINT
            || self.format == vk::Format::D24_UNORM_S8_UINT
            || self.format == vk::Format::D32_SFLOAT_S8_UINT
        {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        }
    }
    fn type_name(&self) -> String {
        if self.usage.contains(vk::ImageUsageFlags::STORAGE) {
            "ImageHandle".to_string()
        } else if self.usage.contains(vk::ImageUsageFlags::SAMPLED) {
            "TextureHandle".to_string()
        } else {
            //Most likely storage Image
            "ImageHandle".to_string()
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
    fn type_name(&self) -> String {
        "".to_string()
    }
}

pub struct CommandBuilder<'a, 'b, T: Default> {
    push_constants: Vec<PushConstant>,
    resources: Vec<(ResourceHandle, ResourceState)>,
    sub_builder: T,
    cmd_buffer: &'a mut CommandBuffer<'b>,
    #[cfg(debug_assertions)]
    layout_validation: Vec<LayoutBlock>,
}

impl<'a, 'b, T: Default> CommandBuilder<'a, 'b, T> {
    pub fn resource_access(
        mut self,
        value: &impl IntoShaderResourceHandle,
        access: vk::AccessFlags2,
        layout: vk::ImageLayout,
    ) -> Self {
        self.push_constants.push(value.push_constant());
        if cfg!(debug_assertions) {
            self.layout_validation.push(LayoutBlock::Type {
                name: value.type_name(),
            });
        }
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
        self.layout_validation.push(LayoutBlock::Constant {
            size: size_of::<A>() as u32,
        });
        self
    }
}

impl<'a, 'b> CommandBuilder<'a, 'b, RasterBuilder> {
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

    pub fn backface_culling(mut self, backface_culling: bool) -> Self {
        self.sub_builder.pipeline_handle.backface_culling = backface_culling;
        self
    }

    pub fn mesh(mut self, path: &'static str, entry: &'static str) -> Self {
        if let PipelineModel::Mesh { task, mesh } = &mut self.sub_builder.pipeline_handle.model {
            mesh.entry = entry;
            mesh.path = path;
        } else {
            self.sub_builder.pipeline_handle.model = PipelineModel::Mesh {
                task: None,
                mesh: ShaderPath { path, entry },
            }
        }
        self
    }
    pub fn task(mut self, path: &'static str, entry: &'static str) -> Self {
        if let PipelineModel::Mesh { task, mesh } = &mut self.sub_builder.pipeline_handle.model {
            *task = Some(ShaderPath { entry, path })
        } else {
            self.sub_builder.pipeline_handle.model = PipelineModel::Mesh {
                task: Some(ShaderPath { path, entry }),
                mesh: ShaderPath {
                    path: "",
                    entry: "",
                },
            }
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

    pub fn vertex_buffer<S: Size>(mut self, buffer: &Buffer<Vertex, S>) -> Self {
        assert!(self.sub_builder.vertex_buffer.is_none());
        if buffer.buffer.is_null() {
            return self;
        }
        self.sub_builder.vertex_buffer = Some(buffer.buffer);
        self.resources.push((
            ResourceHandle::Buffer(buffer.buffer),
            ResourceState {
                access: vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
                layout: vk::ImageLayout::UNDEFINED,
                stages: vk::PipelineStageFlags2::VERTEX_INPUT,
                aspect: vk::ImageAspectFlags::NONE,
            },
        ));
        self
    }

    pub fn index_buffer<S: Size>(mut self, buffer: &Buffer<u32, S>) -> Self {
        assert!(self.sub_builder.index_buffer.is_none());
        if buffer.buffer.is_null() {
            return self;
        }
        self.sub_builder.index_buffer = Some(buffer.buffer);
        self.resources.push((
            ResourceHandle::Buffer(buffer.buffer),
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
            RasterDispatch::DrawIndexedIndirect {
                buffer,
                offset,
                count,
            } => vec![buffer],
            RasterDispatch::DrawIndexedIndirectCount {
                buffer,
                offset,
                count_buffer,
                count_offset,
            } => vec![buffer, count_buffer],
            RasterDispatch::DrawIndirect {
                buffer,
                offset,
                count,
            } => vec![buffer],
            RasterDispatch::DrawIndirectCount {
                buffer,
                offset,
                count_buffer,
                count_offset,
            } => vec![buffer, count_buffer],
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
            #[cfg(debug_assertions)]
            layout_validation: self.layout_validation,
            color_attachments: Some(self.sub_builder.color_attachments),
            depth_attachments: self.sub_builder.depth_attachments,
            command: Command::Raster {
                pipeline: self.sub_builder.pipeline_handle,
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

impl<'a, 'b> CommandBuilder<'a, 'b, ComputeBuilder> {
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
            #[cfg(debug_assertions)]
            layout_validation: self.layout_validation,
            command: Command::Compute {
                pipeline: self.sub_builder.pipeline_handle,
                dispatch,
            },
            push_constants: Some(self.push_constants),
            resources: self.resources,
            ..Default::default()
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
pub struct RayTracingBuilder {
    pipeline_handle: RayTracingPipelineHandle,
}

impl<'a, 'b> CommandBuilder<'a, 'b, RayTracingBuilder> {
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
            #[cfg(debug_assertions)]
            layout_validation: self.layout_validation,
            command: Command::Raytracing {
                pipeline: self.sub_builder.pipeline_handle,
                dispatch,
            },
            push_constants: Some(self.push_constants),
            resources: self.resources,
            ..Default::default()
        });
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

static JSON_CACHE: Mutex<LazyCell<HashMap<String, JsonValue>>> =
    Mutex::new(LazyCell::new(|| HashMap::new()));

impl<'b> CommandBuffer<'b> {
    pub fn fill_buffer<T: Copy, S: Size, L: Location>(
        &mut self,
        buffer: &Buffer<T, S, L>,
        offset: u32,
        data: u32,
    ) {
        if buffer.buffer.is_null() {
            return;
        }

        self.commands.push(Action {
            command: Command::FillBuffer {
                buffer: buffer.buffer,
                data,
                offset,
            },
            resources: vec![(
                ResourceHandle::Buffer(buffer.buffer),
                ResourceState {
                    access: vk::AccessFlags2::TRANSFER_WRITE,
                    stages: vk::PipelineStageFlags2::TRANSFER,
                    ..Default::default()
                },
            )],
            ..Default::default()
        });
    }
    pub fn copy_buffer<T: Copy, L: Location, S: Size, J: Size, B: Location>(
        &mut self,
        src: &Buffer<T, S, L>,
        dst: &Buffer<T, J, B>,
        num_elements: usize,
        src_offset: u32,
        dst_offset: u32,
    ) {
        let num_bytes = num_elements * size_of::<T>();
        if src.buffer.is_null() || dst.buffer.is_null() || src.size < src_offset as u64 + num_bytes as u64 || dst.size < dst_offset as u64 + num_bytes as u64 {
            return;
        }

        self.commands.push(Action {
            command: Command::CopyBuffer {
                src: src.buffer,
                dst: dst.buffer,
                src_offset,
                dst_offset,
                num_bytes: num_bytes as u32,
            },
            resources: vec![(
                ResourceHandle::Buffer(src.buffer),
                ResourceState {
                    access: vk::AccessFlags2::TRANSFER_READ,
                    stages: vk::PipelineStageFlags2::TRANSFER,
                    ..Default::default()
                },
            ),(
                ResourceHandle::Buffer(src.buffer),
                ResourceState {
                    access: vk::AccessFlags2::TRANSFER_WRITE,
                    stages: vk::PipelineStageFlags2::TRANSFER,
                    ..Default::default()
                },
            )],
            ..Default::default()
        });
        
    }

    pub fn copy_dyn_buffer<T: Copy, L: Location, S: Size + Growable, H: Size, B: Location>(
        &mut self,
        src: &Buffer<T, H, L>,
        dst: &mut Buffer<T, S, B>,
        num_elements: usize,
        src_offset: u32,
        dst_offset: u32,
    ) {
        let num_bytes = num_elements * size_of::<T>();
        dst.grow_to_size(num_bytes as u64 + dst_offset as u64);
        self.copy_buffer(src, dst, num_elements, src_offset, dst_offset);
    }

    pub fn raster<'a>(&'a mut self) -> CommandBuilder<'a, 'b, RasterBuilder> {
        CommandBuilder {
            resources: Vec::new(),
            sub_builder: RasterBuilder::default(),
            cmd_buffer: self,
            push_constants: Vec::new(),
            #[cfg(debug_assertions)]
            layout_validation: Vec::new(),
        }
    }
    pub fn compute<'a>(&'a mut self) -> CommandBuilder<'a, 'b, ComputeBuilder> {
        CommandBuilder {
            resources: Vec::new(),
            sub_builder: ComputeBuilder::default(),
            cmd_buffer: self,
            push_constants: Vec::new(),
            #[cfg(debug_assertions)]
            layout_validation: Vec::new(),
        }
    }
    pub fn raytrace<'a>(&'a mut self) -> CommandBuilder<'a, 'b, RayTracingBuilder> {
        CommandBuilder {
            resources: Vec::new(),
            sub_builder: RayTracingBuilder::default(),
            cmd_buffer: self,
            push_constants: Vec::new(),
            #[cfg(debug_assertions)]
            layout_validation: Vec::new(),
        }
    }

    pub fn present(&mut self, swapchain_image: Image) {
        self.commands.push(Action {
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
            ..Default::default()
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

            if cfg!(debug_assertions) && aktion.push_constants.is_some() {
                let shaders = match &aktion.command {
                    Command::Compute { pipeline, dispatch } => vec![pipeline.path.clone()],
                    Command::Raster {
                        pipeline,
                        dispatch,
                        width,
                        height,
                    } => {
                        let mut shaders = vec![pipeline.fragment.clone()];
                        match &pipeline.model {
                            PipelineModel::Mesh { task, mesh } => {
                                if let Some(task) = task {
                                    shaders.push(task.clone());
                                }
                                shaders.push(mesh.clone())
                            }
                            PipelineModel::Vertex { vertex } => shaders.push(vertex.clone()),
                        }
                        shaders
                    }
                    Command::Raytracing { pipeline, dispatch } => vec![pipeline.path.clone()],
                    _ => vec![],
                };

                let mut cache = JSON_CACHE.lock().unwrap();
                for shader_path in shaders {
                    let path = format!("./core/shaders/bin/{}.slang.json", shader_path.path);
                    let json = cache
                        .entry(path.clone())
                        .or_insert(json::parse(&std::fs::read_to_string(&path).unwrap()).unwrap());

                    let binding = json["parameters"]
                        .members()
                        .find(|m| m["binding"]["kind"].as_str().unwrap() == "pushConstantBuffer")
                        .expect(&format!(
                            "No Push Constant block found in shader {}",
                            shader_path.path
                        ));

                    let binding = &binding["type"]["elementVarLayout"]["type"];
                    assert!(
                        binding["kind"] == "struct",
                        "Push Constant block must be a struct"
                    );

                    let members = binding["fields"].members().collect::<Vec<_>>();
                    assert!(
                        members.len() >= aktion.push_constants.as_ref().unwrap().len(),
                        "Push constant struct must have at least as many members as are in push constants"
                    );

                    let mut offset = 0;
                    let mut byte_offset = 0;
                    for block in &aktion.layout_validation {
                        match block {
                            LayoutBlock::Constant { size } => {
                                let constant_end = byte_offset + *size;
                                let member = members[offset];
                                while {
                                    let member = members[offset];
                                    let member_type = member["type"]["kind"].as_str().unwrap();
                                    let member_name = member["type"]["name"].as_str().unwrap_or("");
                                    member_type != "pointer"
                                        && member_name != "ImageHandle"
                                        && member_name != "TextureHandle"
                                        && offset < members.len()
                                } {
                                    let member_size = member["binding"]["size"].as_u32().unwrap();
                                    offset += 1;
                                    byte_offset += member_size;
                                    assert!(
                                        byte_offset <= constant_end,
                                        "Exspected Constants block to end at {} bytes, but it didnt!",
                                        constant_end
                                    );
                                }
                            }
                            LayoutBlock::Type { name } => {
                                let member = members[offset];
                                let member_type = member["type"]["kind"].as_str().unwrap();
                                let member_offset = member["binding"]["offset"].as_u32().unwrap();
                                assert!(
                                    member_offset == byte_offset,
                                    "Expected member to be at offset {}, found it at offset {}",
                                    byte_offset,
                                    member_offset
                                );
                                let member_field_name = member["name"].as_str().unwrap();
                                let member_name = if member_type == "pointer" {
                                    member["type"]["valueType"].as_str().unwrap()
                                } else if member_type == "struct" {
                                    member["type"]["name"].as_str().unwrap()
                                } else {
                                    assert!(
                                        false,
                                        "Expected pointer or struct, found {}, at field {}",
                                        member_type, member_field_name
                                    );
                                    ""
                                };

                                assert!(
                                    member_name == *name,
                                    "Expected typename {}, found {} for field {}",
                                    name,
                                    member_name,
                                    member_field_name
                                );
                                offset += 1;
                                byte_offset += 8;
                            }
                        }
                    }
                }
            }

            match &aktion.command {
                Command::Compute {
                    pipeline,
                    dispatch: [x, y, z],
                } => pipeline.dispatch(&self.handle, *x, *y, *z),
                Command::Raster {
                    pipeline,
                    dispatch,
                    width,
                    height,
                } => {
                    if let Some(index_buffer) = aktion.index_buffer {
                        unsafe {
                            Ctx::device().cmd_bind_index_buffer(
                                self.handle,
                                index_buffer,
                                0,
                                vk::IndexType::UINT32,
                            )
                        };
                    }
                    pipeline.dispatch(
                        self.handle,
                        aktion.color_attachments.as_ref().unwrap(),
                        aktion.depth_attachments.as_ref(),
                        None,
                        aktion.vertex_buffer.as_ref(),
                        *width,
                        *height,
                        *dispatch,
                    )
                }
                Command::Raytracing { pipeline, dispatch } => {
                    pipeline.launch(&self.handle, dispatch[0], dispatch[1])
                }
                Command::FillBuffer {
                    buffer,
                    data,
                    offset,
                } => unsafe {
                    Ctx::device().cmd_fill_buffer(
                        self.handle,
                        *buffer,
                        *offset as u64,
                        vk::WHOLE_SIZE,
                        *data,
                    );
                },
                Command::CopyBuffer { src, dst, src_offset, dst_offset, num_bytes } => unsafe {
                    Ctx::device().cmd_copy_buffer(self.handle, *src, *dst, &[vk::BufferCopy{
                        src_offset: *src_offset as u64,
                        dst_offset: *dst_offset as u64,
                        size: *num_bytes as u64,
                    }]);
                }
                Command::Present => {}
            }
        }

        unsafe { Ctx::device().end_command_buffer(self.handle) }.unwrap();
    }
}

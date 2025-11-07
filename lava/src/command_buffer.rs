use std::{
    any::TypeId,
    cell::LazyCell,
    collections::HashMap,
    fmt::Debug,
    sync::{LazyLock, Mutex},
};

use ash::vk::{self, Handle};
use bytemuck::{Pod, Zeroable, bytes_of};
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
        buffer::{Buffer, CpuBuffer, GpuBuffer, Location, StorageBuffer},
        image::Image,
        rt_pipeline::alinged_size,
    },
};

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

#[macro_export]
macro_rules! c {
    ($($val:expr),+ $(,)?) => {{
        use bytemuck::bytes_of;
        let mut vec = Vec::new();

        $(
            vec.extend_from_slice(bytes_of(&$val));
        )+
        vec
    }};
}

#[derive(Debug)]
pub enum LayoutBlock {
    Constant { size: u32 },
    Type { name: String },
}
pub struct CommandBuffer<'a> {
    pub(crate) handle: vk::CommandBuffer,
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

impl<T: Copy + Pod, L: Location> IntoShaderResourceHandle for Buffer<T, L> {
    fn push_constant(&self) -> PushConstant {
        PushConstant::BufferPointer(self.address)
    }
    fn resource_handle(&self) -> Option<ResourceHandle> {
        Some(ResourceHandle::Buffer(self.handle))
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

impl<T: Copy + Pod> IntoShaderResourceHandle for StorageBuffer<T> {
    fn aspect(&self) -> vk::ImageAspectFlags {
        self.buffer.aspect()
    }
    fn preferd_default_layout(&self) -> Option<vk::ImageLayout> {
        self.buffer.preferd_default_layout()
    }
    fn push_constant(&self) -> PushConstant {
        self.buffer.push_constant()
    }
    fn resource_handle(&self) -> Option<ResourceHandle> {
        self.buffer.resource_handle()
    }
    fn type_name(&self) -> String {
        self.buffer.type_name()
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
    pub fn constants(mut self, bytes: Vec<u8>) -> Self {
        let size = bytes.len() as u32;
        self.push_constants.push(PushConstant::Constants(bytes));
        self.layout_validation.push(LayoutBlock::Constant {
            size,
        });
        self
    }
}

impl<'a, 'b> CommandBuilder<'a, 'b, RasterBuilder> {
    pub fn fragment(mut self, path: &'static str, entry: &'static str) -> Self {
        self.sub_builder.pipeline_handle.fragment = crate::pipelines::ShaderPath { path, entry };
        self
    }
    pub fn fragment_path(self, path: &'static str) -> Self {
        self.fragment(path, "main")
    }

    pub fn vertex(mut self, path: &'static str, entry: &'static str) -> Self {
        self.sub_builder.pipeline_handle.model = crate::pipelines::PipelineModel::Vertex {
            vertex: ShaderPath { entry, path },
            vertex_buffer: false,
        };
        self
    }
    pub fn vertex_path(self, path: &'static str) -> Self {
        self.vertex(path, "main")
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
                stages: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                aspect: vk::ImageAspectFlags::DEPTH,
            },
        ));
        self
    }

    pub fn vertex_buffer<L: Location>(mut self, buffer: &Buffer<Vertex, L>) -> Self {
        assert!(self.sub_builder.vertex_buffer.is_none());
        self.sub_builder.vertex_buffer = Some(buffer.handle);
        if let PipelineModel::Vertex {
            vertex,
            vertex_buffer,
        } = &mut self.sub_builder.pipeline_handle.model
        {
            *vertex_buffer = true
        }
        self.resources.push((
            ResourceHandle::Buffer(buffer.handle),
            ResourceState {
                access: vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
                layout: vk::ImageLayout::UNDEFINED,
                stages: vk::PipelineStageFlags2::VERTEX_INPUT,
                aspect: vk::ImageAspectFlags::NONE,
            },
        ));
        self
    }

    pub fn index_buffer<L: Location>(mut self, buffer: &Buffer<u32, L>) -> Self {
        assert!(self.sub_builder.index_buffer.is_none());
        self.sub_builder.index_buffer = Some(buffer.handle);
        self.resources.push((
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
        self.cmd_buffer.barriers(self.resources);
        self.cmd_buffer.push_constants(&self.push_constants);
        if cfg!(debug_assertions) {
            let mut shaders = vec![self.sub_builder.pipeline_handle.fragment.clone()];
            match self.sub_builder.pipeline_handle.model {
                PipelineModel::Vertex { ref vertex, .. } => shaders.push(vertex.clone()),
                PipelineModel::Mesh { ref mesh, ref task } => {
                    shaders.push(mesh.clone());
                    if let Some(task) = task {
                        shaders.push(task.clone());
                    }
                },
            }

            self.cmd_buffer
                .type_check(&self.push_constants, &shaders, &self.layout_validation).unwrap();
        }
        self.sub_builder.pipeline_handle.dispatch(
            self.cmd_buffer.handle,
            self.sub_builder.color_attachments.as_ref(),
            self.sub_builder.depth_attachments.as_ref(),
            None,
            self.sub_builder.vertex_buffer.as_ref(),
            width,
            height,
            dispatch,
        );
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

    fn build(self, dispatch: [u32; 3], indirect_buffer: Option<(vk::Buffer, u32)>) {
        self.cmd_buffer.barriers(self.resources);
        self.cmd_buffer.push_constants(&self.push_constants);
        if cfg!(debug_assertions) {
            self.cmd_buffer.type_check(
                &self.push_constants,
                &vec![self.sub_builder.pipeline_handle.path.clone()],
                &self.layout_validation,
            ).unwrap();
        }
        if let Some(buffer) = indirect_buffer {
            self.sub_builder.pipeline_handle.dispatch_indirect(&self.cmd_buffer.handle, buffer.0, buffer.1);
        }else {
            self.sub_builder.pipeline_handle.dispatch(
                &self.cmd_buffer.handle,
                dispatch[0],
                dispatch[1],
                dispatch[2],
            );
        }
    }

    pub fn dispatch_indirect<L: Location, T: Copy + Pod>(mut self, buffer: &Buffer<T, L>, offset: u32) {
        self.resources.push((
            ResourceHandle::Buffer(buffer.handle),
            ResourceState {
                access: vk::AccessFlags2::INDIRECT_COMMAND_READ,
                stages: vk::PipelineStageFlags2::COMPUTE_SHADER,
                ..Default::default()
            }
        ));
        self.build([0, 0, 0], Some((buffer.handle, offset as u32)));
    }

    pub fn dispatch(self, x: u32, y: u32, z: u32) {
        self.build([x, y, z], None);
    }
    pub fn dispatch_fullscreen(self) {
        self.build([
            Ctx::window_width().unwrap().div_ceil(8),
            Ctx::window_height().unwrap().div_ceil(8),
            1,
        ], None);
    }
    pub fn dispatch_fractional_fullscreen(self, x: u32, y: u32) {
        self.build([
            Ctx::window_width().unwrap().div_ceil(x),
            Ctx::window_height().unwrap().div_ceil(y),
            1,
        ], None);
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
        self.cmd_buffer.barriers(self.resources);
        self.cmd_buffer.push_constants(&self.push_constants);
        if cfg!(debug_assertions) {
            self.cmd_buffer.type_check(
                &self.push_constants,
                &vec![self.sub_builder.pipeline_handle.path.clone()],
                &self.layout_validation,
            ).unwrap();
        }
        self.sub_builder
            .pipeline_handle
            .launch(&self.cmd_buffer.handle, dispatch[0], dispatch[1]);
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

pub type LayoutResult = Result<(), LayoutError>;

pub struct LayoutError {
    file: String,
    entry: String,
    field: String,
    _ty: LayoutErrorType,
}

pub enum LayoutErrorType {
    ConstantsTooLarge{
        expected_at: usize,
    },
    WrongType {
        expected: String,
        found: String,
    },
    WrongTypeName {
        expected: String,
        found: String,
    },
    WrongOffset {
        expected: u32,
        found: u32,
    },
}

impl Debug for LayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self._ty {
            LayoutErrorType::ConstantsTooLarge { expected_at } => format!("expect constants block to end at {}, but it didnt", expected_at),
            LayoutErrorType::WrongType { ref expected, ref found } => format!("expected field {} to be of type {}, found {}", self.field, expected, found),
            LayoutErrorType::WrongTypeName { ref expected, ref found } => format!("expected field {} type name {}, found {}", self.field, expected, found),
            LayoutErrorType::WrongOffset { expected, found } => format!("expected field {} to be at offset {}, found it at {}", self.field, expected, found),
        };
        write!(f, "Layout error in shader {} at entry point {}: {}", self.file, self.entry, msg)?;
        Ok(())
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
        data: &T
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

    pub fn read_buffer<T: Copy + Pod + Debug>(
        &mut self,
        buffer: &Buffer<T, GpuBuffer>,
        staging: &Buffer<T, CpuBuffer>,
        num_elements: usize,
        offset: usize,
    ) -> Vec<T>{
        if !cfg!(debug_assertions) {
            log::warn!("Using read_buffer in release can cause performance problems!");
        }
        self.copy_buffer(buffer, staging, num_elements, offset as u32 * size_of::<T>() as u32, 0);
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
        self.barriers(vec![(
            ResourceHandle::Image((swapchain_image.view, swapchain_image.image)),
            ResourceState {
                access: vk::AccessFlags2::empty(),
                aspect: vk::ImageAspectFlags::COLOR,
                layout: vk::ImageLayout::PRESENT_SRC_KHR,
                stages: vk::PipelineStageFlags2::NONE,
            },
        )]);
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
                            buffer: buffer,
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

    fn push_constants(&self, constants: &Vec<PushConstant>) {
        let mut data = vec![0; Ctx::physical_device().limits.max_push_constants_size as usize];
        let mut index = 0;
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

        unsafe {
            Ctx::device().cmd_push_constants(
                self.handle,
                Bindless::layout(),
                vk::ShaderStageFlags::ALL,
                0,
                &data,
            )
        };
    }

    pub fn type_check(
        &self,
        push_constants: &Vec<PushConstant>,
        shaders: &Vec<ShaderPath>,
        layout_validation: &Vec<LayoutBlock>,
    ) -> LayoutResult {
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
                members.len() >= push_constants.len(),
                "Push constant struct must have at least as many members as are in push constants"
            );

            let mut offset = 0;
            let mut byte_offset = 0;
            let mut error = LayoutError{
                file: shader_path.path.to_string(),
                entry: shader_path.entry.to_string(),
                field: "".to_string(),
                _ty: LayoutErrorType::ConstantsTooLarge { expected_at: 0 },   
            };
            for block in layout_validation {
                match block {
                    LayoutBlock::Constant { size } => {
                        let constant_end = byte_offset + *size;
                        while {
                            let member = members[offset];
                            let member_type = member["type"]["kind"].as_str().unwrap();
                            let member_name = member["type"]["name"].as_str().unwrap_or("");
                            member_type != "pointer"
                                && member_name != "ImageHandle"
                                && member_name != "TextureHandle"
                                && offset < members.len()
                        } {
                            let member = members[offset];
                            let member_size = member["binding"]["size"].as_u32().unwrap();
                            offset += 1;
                            byte_offset += member_size;
                            if byte_offset > constant_end {
                                error._ty = LayoutErrorType::ConstantsTooLarge { expected_at: constant_end as usize };
                                return Err(error);
                            }
                        }
                    }
                    LayoutBlock::Type { name } => {
                        let member = members[offset];
                        let member_type = member["type"]["kind"].as_str().unwrap();
                        let member_offset = member["binding"]["offset"].as_u32().unwrap();
                        let member_field_name = member["name"].as_str().unwrap();
                        error.field = member_field_name.to_string();
                        let member_name = if member_type == "pointer" {
                            member["type"]["valueType"].as_str().unwrap()
                        } else if member_type == "struct" {
                            member["type"]["name"].as_str().unwrap()
                        } else {
                            error._ty = LayoutErrorType::WrongType { expected: "Pointer or Struct".to_owned(), found: member_type.to_string() };
                            return Err(error);
                            ""
                        };
                        if member_offset != byte_offset {
                            error._ty = LayoutErrorType::WrongOffset { expected: byte_offset, found: member_offset };
                            return Err(error);
                        }
                        if member_name != *name {
                            error._ty = LayoutErrorType::WrongTypeName { expected: name.to_string(), found: member_name.to_string() };
                            return Err(error);
                        }
                        offset += 1;
                        byte_offset += 8;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn begin(&mut self) {
        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe { Ctx::device().begin_command_buffer(self.handle, &begin_info) }.unwrap();
        Bindless::bind(&self.handle);
    }

    pub fn end(&mut self) {
        unsafe { Ctx::device().end_command_buffer(self.handle) }.unwrap();
    }
}

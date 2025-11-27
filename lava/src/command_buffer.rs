use std::{any, cell::LazyCell, collections::HashMap, fmt::Debug, marker::PhantomData, ops::{Index, IndexMut}, sync::Mutex};

use ash::vk::{self, PipelineStageFlags2};
use bytemuck::{Pod, Zeroable, bytes_of};

use crate::{
    bindless::Bindless,
    pipelines::{
        ComputePipelineHandle, PipelineModel, RasterDispatch, RasterPipelineHandle,
        RayTracingPipelineHandle, Vertex,
    },
    state::Ctx,
    vkobjects::{
        buffer::{Buffer, CpuBuffer, GpuBuffer, Location, StorageBuffer},
        image::Image,
    },
};

pub struct CommandBuffer<'a> {
    pub(crate) handle: vk::CommandBuffer,
    pub(crate) resource_hashes: &'a mut HashMap<ResourceHandle, ResourceState>,
}

pub trait Shader {
    type GpuBinding: Binding;
    const STAGE: vk::PipelineStageFlags2;
    const ENTRY: &'static str;
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

pub struct RasterBuilder<'a, 'b, 'c, S: Shader> {
    pipeline_handle: RasterPipelineHandle,
    color_attachments: Vec<(Image, Option<[f32; 4]>)>,
    depth_attachments: Option<Image>,
    vertex_buffer: Option<vk::Buffer>,
    index_buffer: Option<vk::Buffer>,
    resource_states: Vec<(ResourceHandle, ResourceState)>,
    cmd_buf: &'a mut CommandBuffer<'b>,
    binding: Option<<<S as Shader>::GpuBinding as Binding>::CpuBinding<'c>>,
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

impl<'a, 'b, 'c, S: Shader> RasterBuilder<'a, 'b, 'c, S> {
    pub fn backface_culling(mut self, backface_culling: bool) -> Self {
        self.pipeline_handle.backface_culling = backface_culling;
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
        assert!(self.depth_attachments.is_none());
        self.depth_attachments = Some(image.clone());
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

    pub fn vertex_buffer<L: Location>(mut self, buffer: &Buffer<Vertex, L>) -> Self {
        assert!(self.vertex_buffer.is_none());
        self.vertex_buffer = Some(buffer.handle);
        if let PipelineModel::Vertex {
            vertex: _,
            vertex_buffer,
        } = &mut self.pipeline_handle.model
        {
            *vertex_buffer = true
        }
        self.resource_states.push((
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

    pub fn draw(mut self, dispatch: RasterDispatch, width: u32, height: u32) {
        let buffers = match &dispatch {
            RasterDispatch::DrawIndexedIndirect {
                buffer,
                offset: _,
                count: _,
            } => vec![buffer],
            RasterDispatch::DrawIndexedIndirectCount {
                buffer,
                offset: _,
                count_buffer,
                count_offset: _,
            } => vec![buffer, count_buffer],
            RasterDispatch::DrawIndirect {
                buffer,
                offset: _,
                count: _,
            } => vec![buffer],
            RasterDispatch::DrawIndirectCount {
                buffer,
                offset: _,
                count_buffer,
                count_offset: _,
            } => vec![buffer, count_buffer],
            _ => vec![],
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
        let mut shader_resouces = S::GpuBinding::resources(self.binding.as_ref().unwrap(), PipelineStageFlags2::TOP_OF_PIPE);
        self.resource_states.append(&mut shader_resouces);
        self.cmd_buf.barriers(self.resource_states);
        self.cmd_buf.push_constants::<S::GpuBinding>(self.binding.as_ref().unwrap());

        self.pipeline_handle.dispatch(
            self.cmd_buf.handle,
            self.color_attachments.as_ref(),
            self.depth_attachments.as_ref(),
            None,
            self.vertex_buffer.as_ref(),
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

    pub fn bind(mut self, b: <<S as Shader>::GpuBinding as Binding>::CpuBinding<'c>) -> Self {
        self.binding = Some(b);
        self
    }}

pub struct ComputeBuilder<'a, 'b, 'c, S: Shader> {
    cmd_buffer: &'a mut CommandBuffer<'b>,
    binding: Option<<<S as Shader>::GpuBinding as Binding>::CpuBinding<'c>>,
}

impl<'a, 'b, 'c, S: Shader> ComputeBuilder<'a, 'b, 'c, S> {
    pub fn bind(mut self, b: <<S as Shader>::GpuBinding as Binding>::CpuBinding<'c>) -> Self {
        self.binding = Some(b);
        self
    }

        
    fn build(self, dispatch: [u32; 3], indirect_buffer: Option<(vk::Buffer, u32)>) {
        let mut resources = S::GpuBinding::resources(self.binding.as_ref().unwrap(), S::STAGE);
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
        let pipeline_handle = ComputePipelineHandle {
            entry: S::ENTRY
        };
        if let Some(buffer) = indirect_buffer {
            pipeline_handle.dispatch_indirect(
                &self.cmd_buffer.handle,
                buffer.0,
                buffer.1,
            );
        } else {
            pipeline_handle.dispatch(
                &self.cmd_buffer.handle,
                dispatch[0],
                dispatch[1],
                dispatch[2],
            );
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

pub struct RayTracingBuilder<'a, 'b, 'c, S: Shader> {
    binding: Option<<<S as Shader>::GpuBinding as Binding>::CpuBinding<'c>>,
    cmd_buffer: &'a mut CommandBuffer<'b>,
}

impl<'a, 'b, 'c, S: Shader> RayTracingBuilder<'a, 'b, 'c, S> {
    pub fn bind(mut self, b: <<S as Shader>::GpuBinding as Binding>::CpuBinding<'c>) -> Self {
        self.binding = Some(b);
        self
    }

    fn build(self, dispatch: [u32; 2]) {
        let resources = S::GpuBinding::resources(self.binding.as_ref().unwrap(), S::STAGE);
        self.cmd_buffer.barriers(resources);
        self.cmd_buffer.push_constants::<S::GpuBinding>(self.binding.as_ref().unwrap());
        RayTracingPipelineHandle {
            entry: any::type_name::<S>().split("::").last().unwrap()
        }.launch(&self.cmd_buffer.handle, dispatch[0], dispatch[1]);
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

    pub fn read_buffer<T: Copy + Pod + Debug>(
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

    pub fn raster_vertex<'a, 'c, Fragment: Shader, Vertex: Shader>(&'a mut self) -> RasterBuilder<'a, 'b, 'c, Fragment> {
        RasterBuilder {
            cmd_buf: self,
            color_attachments: Vec::new(),
            depth_attachments: None,
            index_buffer: None,
            pipeline_handle: RasterPipelineHandle { fragment: Fragment::ENTRY, backface_culling: false, model: PipelineModel::Vertex { vertex: Vertex::ENTRY, vertex_buffer: false } },
            resource_states: Vec::new(),
            vertex_buffer: None,
            binding: None
        }
    }

    pub fn raster_mesh<'a, 'c, Fragment: Shader, Mesh: Shader>(&'a mut self) -> RasterBuilder<'a, 'b, 'c, Fragment> {
        RasterBuilder {
            cmd_buf: self,
            color_attachments: Vec::new(),
            depth_attachments: None,
            index_buffer: None,
            pipeline_handle: RasterPipelineHandle { fragment: Fragment::ENTRY, backface_culling: false, model: PipelineModel::Mesh { task: None, mesh: Mesh::ENTRY } },
            resource_states: Vec::new(),
            vertex_buffer: None,
            binding: None
        }
    }

    pub fn raster_task<'a, 'c, Fragment: Shader, Mesh: Shader, Task: Shader>(&'a mut self) -> RasterBuilder<'a, 'b, 'c, Fragment> {
        RasterBuilder {
            cmd_buf: self,
            color_attachments: Vec::new(),
            depth_attachments: None,
            index_buffer: None,
            pipeline_handle: RasterPipelineHandle { fragment: Fragment::ENTRY, backface_culling: false, model: PipelineModel::Mesh { task: Some(Task::ENTRY), mesh: Mesh::ENTRY } },
            resource_states: Vec::new(),
            vertex_buffer: None,
            binding: None,
        }
    }

    pub fn compute<'a, 'c, S: Shader>(&'a mut self) -> ComputeBuilder<'a, 'b, 'c, S> {
        ComputeBuilder {
            cmd_buffer: self,
            binding: None,
        }
    }
    pub fn raytrace<'a, 'c, S: Shader>(&'a mut self) -> RayTracingBuilder<'a, 'b, 'c, S> {
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

    pub fn push_constants<'a, B: Binding>(&mut self, binding: &B::CpuBinding<'a>) {
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

//! The baseline's thin, non-tracking ash layer: per queue a command pool and buffer per frame slot, a timeline semaphore and a timestamp pool, and a command buffer wrapper whose methods map one-to-one to Vulkan calls

use std::ops::Range;
use std::time::Instant;

use lava::queue::Kind;
use lava::vk;

/// A stage and the accesses made there, one side of a barrier.
pub type Access = (vk::PipelineStageFlags2, vk::AccessFlags2);

pub const NONE: Access = (vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE);
pub const COMPUTE_READ: Access = (
    vk::PipelineStageFlags2::COMPUTE_SHADER,
    vk::AccessFlags2::from_raw(
        vk::AccessFlags2::SHADER_SAMPLED_READ.as_raw() | vk::AccessFlags2::SHADER_STORAGE_READ.as_raw(),
    ),
);
pub const COMPUTE_WRITE: Access = (
    vk::PipelineStageFlags2::COMPUTE_SHADER,
    vk::AccessFlags2::SHADER_STORAGE_WRITE,
);
pub const COMPUTE_RW: Access = (
    vk::PipelineStageFlags2::COMPUTE_SHADER,
    vk::AccessFlags2::from_raw(
        vk::AccessFlags2::SHADER_STORAGE_WRITE.as_raw()
            | vk::AccessFlags2::SHADER_STORAGE_READ.as_raw()
            | vk::AccessFlags2::SHADER_SAMPLED_READ.as_raw(),
    ),
);
pub const VERTEX_READ: Access = (
    vk::PipelineStageFlags2::VERTEX_SHADER,
    vk::AccessFlags2::SHADER_STORAGE_READ,
);
pub const FRAGMENT_READ: Access = (
    vk::PipelineStageFlags2::FRAGMENT_SHADER,
    vk::AccessFlags2::from_raw(
        vk::AccessFlags2::SHADER_SAMPLED_READ.as_raw() | vk::AccessFlags2::SHADER_STORAGE_READ.as_raw(),
    ),
);
pub const GRAPHICS_READ: Access = (
    vk::PipelineStageFlags2::from_raw(
        vk::PipelineStageFlags2::VERTEX_SHADER.as_raw()
            | vk::PipelineStageFlags2::FRAGMENT_SHADER.as_raw(),
    ),
    vk::AccessFlags2::from_raw(
        vk::AccessFlags2::SHADER_SAMPLED_READ.as_raw() | vk::AccessFlags2::SHADER_STORAGE_READ.as_raw(),
    ),
);
pub const INDIRECT_READ: Access = (
    vk::PipelineStageFlags2::DRAW_INDIRECT,
    vk::AccessFlags2::INDIRECT_COMMAND_READ,
);
pub const COLOR_WRITE: Access = (
    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
    vk::AccessFlags2::from_raw(
        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE.as_raw()
            | vk::AccessFlags2::COLOR_ATTACHMENT_READ.as_raw(),
    ),
);
pub const DEPTH_WRITE: Access = (
    vk::PipelineStageFlags2::from_raw(
        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS.as_raw()
            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS.as_raw(),
    ),
    vk::AccessFlags2::from_raw(
        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE.as_raw()
            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ.as_raw(),
    ),
);
pub const TRANSFER_WRITE: Access = (
    vk::PipelineStageFlags2::ALL_TRANSFER,
    vk::AccessFlags2::TRANSFER_WRITE,
);
pub const TRANSFER_READ: Access = (
    vk::PipelineStageFlags2::ALL_TRANSFER,
    vk::AccessFlags2::TRANSFER_READ,
);
pub const HOST_READ: Access = (vk::PipelineStageFlags2::HOST, vk::AccessFlags2::HOST_READ);
pub const BUILD_WRITE: Access = (
    vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
    vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
);
/// What builds read: structures, and geometry and instances as shader reads.
pub const BUILD_READ: Access = (
    vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
    vk::AccessFlags2::from_raw(
        vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR.as_raw()
            | vk::AccessFlags2::SHADER_READ.as_raw(),
    ),
);

/// Two accesses joined, the stages and accesses of both.
pub const fn or(a: Access, b: Access) -> Access {
    (
        vk::PipelineStageFlags2::from_raw(a.0.as_raw() | b.0.as_raw()),
        vk::AccessFlags2::from_raw(a.1.as_raw() | b.1.as_raw()),
    )
}

/// A global memory barrier from `src` to `dst`.
pub fn mem(src: Access, dst: Access) -> vk::MemoryBarrier2<'static> {
    vk::MemoryBarrier2::default()
        .src_stage_mask(src.0)
        .src_access_mask(src.1)
        .dst_stage_mask(dst.0)
        .dst_access_mask(dst.1)
}

/// A layout transition of `mips` of every layer of a colour image.
pub fn img(
    image: vk::Image,
    mips: Range<u32>,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
    src: Access,
    dst: Access,
) -> vk::ImageMemoryBarrier2<'static> {
    img_aspect(image, vk::ImageAspectFlags::COLOR, mips, old, new, src, dst)
}

/// A layout transition of `mips` of every layer of `aspect`.
pub fn img_aspect(
    image: vk::Image,
    aspect: vk::ImageAspectFlags,
    mips: Range<u32>,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
    src: Access,
    dst: Access,
) -> vk::ImageMemoryBarrier2<'static> {
    vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src.0)
        .src_access_mask(src.1)
        .dst_stage_mask(dst.0)
        .dst_access_mask(dst.1)
        .old_layout(old)
        .new_layout(new)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level: mips.start,
            level_count: mips.end - mips.start,
            base_array_layer: 0,
            layer_count: vk::REMAINING_ARRAY_LAYERS,
        })
}

/// The address flags lava gives every command: storage usage, fully bound.
pub const ADDRESS_FLAGS: vk::AddressCommandFlagsKHR = vk::AddressCommandFlagsKHR::from_raw(
    vk::AddressCommandFlagsKHR::STORAGE_BUFFER_USAGE.as_raw()
        | vk::AddressCommandFlagsKHR::FULLY_BOUND.as_raw(),
);

/// `size` bytes at `address`.
pub fn range(address: u64, size: u64) -> vk::DeviceAddressRangeKHR {
    vk::DeviceAddressRangeKHR { address, size }
}

/// One queue of the baseline: a command pool and buffer per frame slot, a timeline semaphore
/// and two timestamps per slot.
pub struct RawQueue {
    pub kind: Kind,
    pools: Vec<vk::CommandPool>,
    buffers: Vec<vk::CommandBuffer>,
    /// The semaphore value each slot's last submission signals.
    slot_values: Vec<u64>,
    pub semaphore: vk::Semaphore,
    pub value: u64,
    pub stamps: vk::QueryPool,
    slot: usize,
}

impl RawQueue {
    pub fn new(kind: Kind, slots: usize) -> Self {
        let device = lava::raw::device();
        let family = lava::Ctx::queue_family(kind).expect("a family of that kind");
        let pools: Vec<_> = (0..slots)
            .map(|_| unsafe {
                device
                    .create_command_pool(
                        &vk::CommandPoolCreateInfo::default()
                            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                            .queue_family_index(family),
                        None,
                    )
                    .unwrap()
            })
            .collect();
        let buffers = (pools.iter())
            .map(|&pool| unsafe {
                device
                    .allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_pool(pool)
                            .level(vk::CommandBufferLevel::PRIMARY)
                            .command_buffer_count(1),
                    )
                    .unwrap()[0]
            })
            .collect();
        let mut timeline = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let semaphore = unsafe {
            device
                .create_semaphore(
                    &vk::SemaphoreCreateInfo::default().push_next(&mut timeline),
                    None,
                )
                .unwrap()
        };
        let stamps = unsafe {
            device
                .create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(2 * slots as u32),
                    None,
                )
                .unwrap()
        };
        Self {
            kind,
            pools,
            buffers,
            slot_values: vec![0; slots],
            semaphore,
            value: 0,
            stamps,
            slot: slots - 1,
        }
    }

    pub fn slots(&self) -> usize {
        self.pools.len()
    }

    /// Blocks until the last submission completed.
    pub fn block(&self) {
        block(self.semaphore, self.value);
    }

    /// The GPU ticks between the start and end timestamps of `slot`'s last submission, which
    /// completed.
    pub fn stamps(&self, slot: usize) -> [u64; 2] {
        let mut out = [0u64; 2];
        unsafe {
            lava::raw::device()
                .get_query_pool_results(
                    self.stamps,
                    2 * slot as u32,
                    &mut out,
                    vk::QueryResultFlags::TYPE_64,
                )
                .unwrap()
        };
        out
    }

    /// Records and submits one command buffer in the next slot, waiting for `waits` (semaphore,
    /// value, stage) and signalling the next value. Returns the slot and the record time.
    pub fn submit(
        &mut self,
        waits: &[(vk::Semaphore, u64, vk::PipelineStageFlags2)],
        record: impl FnOnce(&mut RawCmd),
    ) -> (usize, u64) {
        let device = lava::raw::device();
        self.slot = (self.slot + 1) % self.pools.len();
        let slot = self.slot;
        block(self.semaphore, self.slot_values[slot]);
        let cmd = self.buffers[slot];
        let elapsed;
        unsafe {
            device
                .reset_command_pool(self.pools[slot], vk::CommandPoolResetFlags::empty())
                .unwrap();
            device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();
            device.cmd_reset_query_pool(cmd, self.stamps, 2 * slot as u32, 2);
            if self.kind != Kind::Transfer {
                lava::raw::bind_heaps(cmd);
            }
            device.cmd_write_timestamp2(
                cmd,
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                self.stamps,
                2 * slot as u32,
            );
            let start = Instant::now();
            record(&mut RawCmd { cmd });
            elapsed = start.elapsed().as_nanos() as u64;
            device.cmd_write_timestamp2(
                cmd,
                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                self.stamps,
                2 * slot as u32 + 1,
            );
            device.end_command_buffer(cmd).unwrap();
        }
        self.value += 1;
        self.slot_values[slot] = self.value;
        let wait_infos: Vec<_> = (waits.iter())
            .map(|&(semaphore, value, stage)| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(semaphore)
                    .value(value)
                    .stage_mask(stage)
            })
            .collect();
        let signal = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.semaphore)
            .value(self.value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let buffers = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&wait_infos)
            .command_buffer_infos(&buffers)
            .signal_semaphore_infos(&signal);
        let queue = lava::raw::queue(self.kind, 0).expect("a queue of that kind");
        unsafe {
            device
                .queue_submit2(*queue, &[submit], vk::Fence::null())
                .unwrap()
        };
        (slot, elapsed)
    }
}

impl Drop for RawQueue {
    fn drop(&mut self) {
        let device = lava::raw::device();
        self.block();
        unsafe {
            for &pool in &self.pools {
                device.destroy_command_pool(pool, None);
            }
            device.destroy_semaphore(self.semaphore, None);
            device.destroy_query_pool(self.stamps, None);
        }
    }
}

/// Blocks until `semaphore` reaches `value`.
pub fn block(semaphore: vk::Semaphore, value: u64) {
    if value == 0 {
        return;
    }
    let semaphores = [semaphore];
    let values = [value];
    unsafe {
        lava::raw::device()
            .wait_semaphores(
                &vk::SemaphoreWaitInfo::default()
                    .semaphores(&semaphores)
                    .values(&values),
                u64::MAX,
            )
            .unwrap()
    };
}

/// A colour attachment of `begin_rendering`: its view, and its clear colour or `None` to load it.
pub struct Color {
    pub view: vk::ImageView,
    pub clear: Option<[f32; 4]>,
}

/// A command buffer that is recording; each method is one Vulkan command.
pub struct RawCmd {
    pub cmd: vk::CommandBuffer,
}

fn device() -> &'static ash::Device {
    lava::raw::device()
}

impl RawCmd {
    pub fn barrier(&mut self, memory: &[vk::MemoryBarrier2], images: &[vk::ImageMemoryBarrier2]) {
        let info = vk::DependencyInfo::default()
            .memory_barriers(memory)
            .image_memory_barriers(images);
        unsafe { device().cmd_pipeline_barrier2(self.cmd, &info) };
    }

    /// One global memory barrier.
    pub fn mem(&mut self, src: Access, dst: Access) {
        self.barrier(&[mem(src, dst)], &[]);
    }

    pub fn images(&mut self, images: &[vk::ImageMemoryBarrier2]) {
        self.barrier(&[], images);
    }

    pub fn bind_compute(&mut self, pipeline: vk::Pipeline) {
        unsafe { device().cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::COMPUTE, pipeline) };
    }

    pub fn bind_graphics(&mut self, pipeline: vk::Pipeline) {
        unsafe { device().cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::GRAPHICS, pipeline) };
    }

    /// Pushes `data` as push data from offset 0, padded to whole words like lava pads it.
    pub fn push<T: bytemuck::NoUninit>(&mut self, data: &T) {
        let bytes = bytemuck::bytes_of(data);
        let mut padded = [0u8; 256];
        let data = match bytes.len() % 4 {
            0 => bytes,
            _ => {
                padded[..bytes.len()].copy_from_slice(bytes);
                &padded[..bytes.len().next_multiple_of(4)]
            }
        };
        let info = vk::PushDataInfoEXT::default().data(vk::HostAddressRangeConstEXT {
            address: data.as_ptr().cast(),
            size: data.len(),
        });
        unsafe { lava::raw::descriptor_heap().cmd_push_data(self.cmd, &info) };
    }

    pub fn dispatch(&mut self, [x, y, z]: [u32; 3]) {
        unsafe { device().cmd_dispatch(self.cmd, x, y, z) };
    }

    pub fn begin_rendering(
        &mut self,
        extent: [u32; 2],
        colors: &[Color],
        depth: Option<(vk::ImageView, Option<f32>)>,
    ) {
        let attachments: Vec<_> = (colors.iter())
            .map(|c| {
                vk::RenderingAttachmentInfo::default()
                    .image_view(c.view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(match c.clear {
                        Some(_) => vk::AttachmentLoadOp::CLEAR,
                        None => vk::AttachmentLoadOp::LOAD,
                    })
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: c.clear.unwrap_or_default(),
                        },
                    })
            })
            .collect();
        let depth = depth.map(|(view, clear)| {
            vk::RenderingAttachmentInfo::default()
                .image_view(view)
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(match clear {
                    Some(_) => vk::AttachmentLoadOp::CLEAR,
                    None => vk::AttachmentLoadOp::LOAD,
                })
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: clear.unwrap_or(0.0),
                        stencil: 0,
                    },
                })
        });
        let area = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D {
                width: extent[0],
                height: extent[1],
            },
        };
        let mut info = vk::RenderingInfo::default()
            .render_area(area)
            .layer_count(1)
            .color_attachments(&attachments);
        if let Some(depth) = &depth {
            info = info.depth_attachment(depth);
        }
        unsafe {
            device().cmd_begin_rendering(self.cmd, &info);
            // The viewport and scissor are the pipelines' only dynamic state.
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: extent[0] as f32,
                height: extent[1] as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            device().cmd_set_viewport_with_count(self.cmd, &[viewport]);
            device().cmd_set_scissor_with_count(self.cmd, &[area]);
        }
    }

    pub fn end_rendering(&mut self) {
        unsafe { device().cmd_end_rendering(self.cmd) };
    }

    pub fn bind_index_u32(&mut self, address: u64, size: u64) {
        let info = vk::BindIndexBuffer3InfoKHR::default()
            .address_range(range(address, size))
            .address_flags(ADDRESS_FLAGS)
            .index_type(vk::IndexType::UINT32);
        unsafe { lava::raw::device_address_commands().cmd_bind_index_buffer3(self.cmd, &info) };
    }

    pub fn draw(&mut self, vertices: u32, instances: u32) {
        unsafe { device().cmd_draw(self.cmd, vertices, instances, 0, 0) };
    }

    pub fn draw_indexed(&mut self, indices: u32, instances: u32) {
        unsafe { device().cmd_draw_indexed(self.cmd, indices, instances, 0, 0, 0) };
    }

    /// Draws of the `T`s at `address` (`stride` apart), as many as the `u32` at `count` says.
    pub fn draw_indexed_indirect_count(
        &mut self,
        address: u64,
        stride: u64,
        max: u32,
        count: u64,
    ) {
        let info = vk::DrawIndirectCount2InfoKHR::default()
            .address_range(vk::StridedDeviceAddressRangeKHR {
                address,
                size: stride * max as u64,
                stride,
            })
            .address_flags(ADDRESS_FLAGS)
            .count_address_range(range(count, 4))
            .count_address_flags(ADDRESS_FLAGS)
            .max_draw_count(max);
        unsafe {
            lava::raw::device_address_commands().cmd_draw_indexed_indirect_count2(self.cmd, &info)
        };
    }

    /// One draw of the `VkDrawIndirectCommand` at `address`.
    pub fn draw_indirect(&mut self, address: u64) {
        let info = vk::DrawIndirect2InfoKHR::default()
            .address_range(vk::StridedDeviceAddressRangeKHR {
                address,
                size: 16,
                stride: 16,
            })
            .address_flags(ADDRESS_FLAGS)
            .draw_count(1);
        unsafe { lava::raw::device_address_commands().cmd_draw_indirect2(self.cmd, &info) };
    }

    pub fn fill(&mut self, address: u64, size: u64, value: u32) {
        unsafe {
            lava::raw::device_address_commands().cmd_fill_memory(
                self.cmd,
                &range(address, size),
                ADDRESS_FLAGS,
                value,
            )
        };
    }

    /// Copies of `(src, dst, size)` byte ranges.
    pub fn copy(&mut self, regions: &[(u64, u64, u64)]) {
        let regions: Vec<_> = (regions.iter())
            .map(|&(src, dst, size)| {
                vk::DeviceMemoryCopyKHR::default()
                    .src_range(range(src, size))
                    .src_flags(ADDRESS_FLAGS)
                    .dst_range(range(dst, size))
                    .dst_flags(ADDRESS_FLAGS)
            })
            .collect();
        let info = vk::CopyDeviceMemoryInfoKHR::default().regions(&regions);
        unsafe { lava::raw::device_address_commands().cmd_copy_memory(self.cmd, &info) };
    }

    /// Copies tightly packed texels at `address` into all of `mip` of a colour image in
    /// `TRANSFER_DST_OPTIMAL`.
    pub fn copy_to_image(
        &mut self,
        address: u64,
        size: u64,
        image: vk::Image,
        mip: u32,
        extent: [u32; 2],
    ) {
        let regions = [vk::DeviceMemoryImageCopyKHR::default()
            .address_range(range(address, size))
            .address_flags(ADDRESS_FLAGS)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: mip,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image_extent(vk::Extent3D {
                width: extent[0],
                height: extent[1],
                depth: 1,
            })];
        let info = vk::CopyDeviceMemoryImageInfoKHR::default()
            .image(image)
            .regions(&regions);
        unsafe {
            lava::raw::device_address_commands().cmd_copy_memory_to_image(self.cmd, &info)
        };
    }

    /// A linear blit of all of mip `src` to all of mip `dst` of one colour image, `src` in
    /// `TRANSFER_SRC_OPTIMAL` and `dst` in `TRANSFER_DST_OPTIMAL`.
    pub fn blit_mip(&mut self, image: vk::Image, src: u32, dst: u32, extent: [u32; 2]) {
        let size = |mip: u32| vk::Offset3D {
            x: (extent[0] >> mip).max(1) as i32,
            y: (extent[1] >> mip).max(1) as i32,
            z: 1,
        };
        let layers = |mip| vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: mip,
            base_array_layer: 0,
            layer_count: 1,
        };
        let regions = [vk::ImageBlit2::default()
            .src_subresource(layers(src))
            .src_offsets([vk::Offset3D::default(), size(src)])
            .dst_subresource(layers(dst))
            .dst_offsets([vk::Offset3D::default(), size(dst)])];
        let info = vk::BlitImageInfo2::default()
            .src_image(image)
            .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .dst_image(image)
            .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .regions(&regions)
            .filter(vk::Filter::LINEAR);
        unsafe { device().cmd_blit_image2(self.cmd, &info) };
    }

    /// Clears all of a colour image in `TRANSFER_DST_OPTIMAL`.
    pub fn clear(&mut self, image: vk::Image, color: [f32; 4]) {
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: vk::REMAINING_MIP_LEVELS,
            base_array_layer: 0,
            layer_count: vk::REMAINING_ARRAY_LAYERS,
        };
        unsafe {
            device().cmd_clear_color_image(
                self.cmd,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue { float32: color },
                &[range],
            )
        };
    }
}

/// A buffer of the baseline's own, device-local, for what lava's buffers lack the usage for
/// (acceleration structure storage).
pub struct RawBuffer {
    pub buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    pub address: u64,
}

impl RawBuffer {
    pub fn new(size: u64, usage: vk::BufferUsageFlags) -> Self {
        let device = device();
        let usage = usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let buffer = unsafe {
            device
                .create_buffer(&vk::BufferCreateInfo::default().size(size).usage(usage), None)
                .unwrap()
        };
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let properties = unsafe {
            lava::raw::instance().get_physical_device_memory_properties(lava::raw::physical_device())
        };
        let memory_type = (0..properties.memory_type_count)
            .find(|&i| {
                requirements.memory_type_bits & (1 << i) != 0
                    && (properties.memory_types[i as usize].property_flags)
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .expect("a device-local memory type");
        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let memory = unsafe {
            device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(requirements.size)
                        .memory_type_index(memory_type)
                        .push_next(&mut flags),
                    None,
                )
                .unwrap()
        };
        unsafe { device.bind_buffer_memory(buffer, memory, 0).unwrap() };
        let address = unsafe {
            device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        };
        Self {
            buffer,
            memory,
            address,
        }
    }
}

impl Drop for RawBuffer {
    fn drop(&mut self) {
        unsafe {
            device().destroy_buffer(self.buffer, None);
            device().free_memory(self.memory, None);
        }
    }
}

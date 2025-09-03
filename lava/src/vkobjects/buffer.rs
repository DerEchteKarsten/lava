use std::{ffi::c_void, fmt::Debug, ptr::NonNull, sync::Mutex};

use anyhow::{Error, Result};
use ash::vk;
use derivative::Derivative;
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc},
    MemoryLocation,
};
use image::{DynamicImage, GenericImageView};

use crate::renderer::bindless::{BindlessDescriptorHeap, DescriptorResourceHandle};

use super::{
    image::{get_aspects, Image, ImageType},
    Context,
};

#[derive(Derivative)]
#[derivative(Eq, PartialEq, Debug)]
pub struct Buffer {
    pub buffer: vk::Buffer,
    #[derivative(PartialEq = "ignore")]
    pub allocation: Option<Mutex<Allocation>>,
    pub address: vk::DeviceAddress,
    pub size: vk::DeviceSize,
    pub usage: vk::BufferUsageFlags,
}

impl Clone for Buffer {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer,
            usage: self.usage,
            address: self.address,
            size: self.size,
            allocation: None,
        }
    }
}

impl Buffer {
    pub fn handle(&self) -> BufferHandle {
        BufferHandle {
            buffer: self.buffer,
            address: self.address,
            size: self.size,
            usage: self.usage,
        }
    }
}

#[derive(Default, Clone, Copy)]
pub struct BufferHandle {
    pub buffer: vk::Buffer,
    pub address: vk::DeviceAddress,
    pub size: vk::DeviceSize,
    pub usage: vk::BufferUsageFlags,
}

impl BufferType for Buffer {
    fn get_address(&self) -> vk::DeviceAddress {
        self.address
    }
    fn get_size(&self) -> vk::DeviceSize {
        self.size
    }
    fn get_usage(&self) -> vk::BufferUsageFlags {
        self.usage
    }
    fn to_vk(&self) -> vk::Buffer {
        self.buffer
    }
}
impl BufferType for BufferHandle {
    fn get_address(&self) -> vk::DeviceAddress {
        self.address
    }
    fn get_size(&self) -> vk::DeviceSize {
        self.size
    }
    fn get_usage(&self) -> vk::BufferUsageFlags {
        self.usage
    }
    fn to_vk(&self) -> vk::Buffer {
        self.buffer
    }
}

pub trait BufferType {
    fn get_address(&self) -> vk::DeviceAddress;
    fn get_size(&self) -> vk::DeviceSize;
    fn to_vk(&self) -> vk::Buffer;
    fn get_usage(&self) -> vk::BufferUsageFlags;
    fn copy_to_image(
        &self,
        cmd: &vk::CommandBuffer,
        dst: &impl ImageType,
        layout: vk::ImageLayout,
        buffer_offset: u64,
    ) {
        let region = vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: get_aspects(dst.get_format()),
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_extent(vk::Extent3D {
                width: dst.get_extent().width,
                height: dst.get_extent().height,
                depth: 1,
            })
            .buffer_offset(buffer_offset);

        unsafe {
            Context::get().device.cmd_copy_buffer_to_image(
                *cmd,
                self.to_vk(),
                dst.get_image(),
                layout,
                std::slice::from_ref(&region),
            );
        };
    }

    fn copy(&self, cmd: &vk::CommandBuffer, dst_buffer: &impl BufferType) {
        unsafe {
            let region = vk::BufferCopy::default().size(self.get_size());
            Context::get().device.cmd_copy_buffer(
                *cmd,
                self.to_vk(),
                dst_buffer.to_vk(),
                std::slice::from_ref(&region),
            )
        };
    }
}

impl Buffer {
    pub fn new_aligned(
        usage: vk::BufferUsageFlags,
        memory_location: MemoryLocation,
        size: vk::DeviceSize,
        alignment: Option<u64>,
    ) -> Result<Self> {
        let ctx = Context::get();
        let create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);
        let buffer = unsafe { ctx.device.create_buffer(&create_info, None)? };
        let mut requirements = unsafe { ctx.device.get_buffer_memory_requirements(buffer) };
        if let Some(a) = alignment {
            requirements.alignment = a;
        }

        let allocation = {
            let mut allocator = ctx.allocator.lock().unwrap();
            (*allocator).allocate(&AllocationCreateDesc {
                name: "buffer",
                requirements,
                location: memory_location,
                linear: true,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })
        }?;

        unsafe {
            ctx.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?
        };
        let addr_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);

        Ok(Self {
            buffer,
            allocation: Some(Mutex::new(allocation)),
            address: unsafe { ctx.device.get_buffer_device_address(&addr_info) },
            size,
            usage,
        })
    }

    pub fn new(
        usage: vk::BufferUsageFlags,
        memory_location: MemoryLocation,
        size: vk::DeviceSize,
    ) -> Result<Self> {
        Self::new_aligned(usage, memory_location, size, None)
    }

    pub fn copy_data_to_buffer<T: Copy>(&self, data: &[T]) -> Result<()> {
        let mut allocation = self
            .allocation
            .as_ref()
            .ok_or(Error::msg("Buffer not Owned"))?
            .lock()
            .unwrap();

        presser::copy_from_slice_to_offset(data, &mut (*allocation), 0)?;

        Ok(())
    }

    pub fn copy_data_to_buffer_offset<T: Copy>(&self, data: &[T], offset: u64) -> Result<()> {
        let mut allocation = self
            .allocation
            .as_ref()
            .ok_or(Error::msg("Buffer not Owned"))?
            .lock()
            .unwrap();
        presser::copy_from_slice_to_offset(data, &mut (*allocation), offset as usize)?;

        Ok(())
    }

    pub fn copy_value_to_buffer_offset<T: Copy>(&self, data: &T, offset: u64) -> Result<()> {
        let mut allocation = self
            .allocation
            .as_ref()
            .ok_or(Error::msg("Buffer not Owned"))?
            .lock()
            .unwrap();
        presser::copy_to_offset(data, &mut (*allocation), offset as usize)?;

        Ok(())
    }

    pub fn copy_data_to_aligned_buffer<T: Copy>(&self, data: &[T], alignment: u32) -> Result<()> {
        let mut allocation = self
            .allocation
            .as_ref()
            .ok_or(Error::msg("Buffer not Owned"))?
            .lock()
            .unwrap();
        presser::copy_from_slice_to_offset_with_align(
            data,
            &mut (*allocation),
            0,
            alignment as usize,
        )?;

        Ok(())
    }

    pub fn destroy(&mut self) -> Result<()> {
        let ctx = Context::get();
        unsafe { ctx.device.destroy_buffer(self.buffer, None) };
        let mut allocator = ctx.allocator.lock().unwrap();
        let allocation = self
            .allocation
            .take()
            .ok_or(Error::msg("Buffer not Owned"))?;

        (*allocator).free(allocation.into_inner()?).unwrap();
        Ok(())
    }

    pub fn read<T: Clone>(&self, num_elements: usize) -> Result<Vec<T>> {
        let allocation = self
            .allocation
            .as_ref()
            .ok_or(Error::msg("Buffer not Owned"))?;
        let allocation = allocation.lock().unwrap();
        let slice = allocation.mapped_slice().unwrap();
        let mut vec = Vec::new();
        vec.extend_from_slice(
            &unsafe { std::mem::transmute::<&[u8], &[T]>(slice) }[0..num_elements],
        );
        Ok(vec)
    }

    // pub fn from_data_with_size<T: Copy>(
    //     ctx: &mut Context,
    //     usage: vk::BufferUsageFlags,
    //     data: &[T],
    //     size: u64,
    // ) -> Result<Buffer> {
    //     let staging_buffer = Self::new(
    //         ctx,
    //         vk::BufferUsageFlags::TRANSFER_SRC,
    //         MemoryLocation::CpuToGpu,
    //         size,
    //     )?;
    //     staging_buffer.copy_data_to_buffer(data)?;

    //     let buffer = Self::new(
    //         ctx,
    //         usage | vk::BufferUsageFlags::TRANSFER_DST,
    //         MemoryLocation::GpuOnly,
    //         size,
    //     )?;

    //     ctx.execute_one_time_commands(|cmd_buffer| {
    //         staging_buffer.copy(&ctx, cmd_buffer, &buffer);
    //     })?;

    //     staging_buffer.destroy(ctx);

    //     Ok(buffer)
    // }

    // pub fn from_data<T: Copy>(
    //     ctx: &mut Context,
    //     usage: vk::BufferUsageFlags,
    //     data: &[T],
    // ) -> Result<Buffer> {
    //     let size = size_of_val(data) as _;
    //     Self::from_data_with_size(ctx, usage, data, size)
    // }
}

#[derive(Default)]
pub struct RawDynamicBuffer {
    pub buffer: vk::Buffer,
    address: vk::DeviceAddress,
    allocation: Option<Allocation>,
}

impl Clone for RawDynamicBuffer {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer,
            address: self.address,
            allocation: None,
        }
    }
}

#[derive(Clone)]
pub struct DynamicBuffer {
    pub buffer: RawDynamicBuffer,
    pub capacity: u64,
    pub usage: vk::BufferUsageFlags,
    pub size: u64,
    pub memory_location: MemoryLocation,
    pub override_alignment: Option<u64>,
    pub bindless_handle: DescriptorResourceHandle,
}

impl BufferType for DynamicBuffer {
    fn get_address(&self) -> vk::DeviceAddress {
        self.buffer.address
    }
    fn get_size(&self) -> vk::DeviceSize {
        self.capacity
    }
    fn get_usage(&self) -> vk::BufferUsageFlags {
        self.usage
    }
    fn to_vk(&self) -> vk::Buffer {
        self.buffer.buffer
    }
}

impl DynamicBuffer {
    pub fn new(
        usage: vk::BufferUsageFlags,
        memory_location: MemoryLocation,
        capacity: u64,
        override_alignment: Option<u64>,
    ) -> Result<Self> {
        let mut s = Self {
            memory_location,
            buffer: RawDynamicBuffer::default(),
            capacity,
            size: 0,
            usage: usage
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            override_alignment,
            bindless_handle: DescriptorResourceHandle(0),
        };
        let buffer = s.create_buffer()?;
        s.buffer = buffer;
        s.bindless_handle = BindlessDescriptorHeap::get().allocate_buffer_handle(&s);
        Ok(s)
    }

    fn create_buffer(&mut self) -> Result<RawDynamicBuffer> {
        let ctx = Context::get();
        let create_info = vk::BufferCreateInfo::default()
            .size(self.capacity)
            .usage(self.usage);
        let buffer = unsafe { ctx.device.create_buffer(&create_info, None)? };
        let mut requirements = unsafe { ctx.device.get_buffer_memory_requirements(buffer) };
        if let Some(a) = self.override_alignment {
            requirements.alignment = a;
        }

        let allocation = {
            let mut allocator = ctx.allocator.lock().unwrap();
            (*allocator).allocate(&AllocationCreateDesc {
                name: "buffer",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: true,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })
        }?;

        unsafe {
            ctx.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?
        };
        let addr_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
        let address = unsafe { ctx.device.get_buffer_device_address(&addr_info) };
        Ok(RawDynamicBuffer {
            address,
            allocation: Some(allocation),
            buffer,
        })
    }

    pub fn grow_to_size(&mut self, size: u64) -> Result<()> {
        let old_size = self.size;
        self.size = size;
        if size > self.capacity {
            let ctx = Context::get();
            self.capacity = self.size.next_power_of_two();

            let buffer = self.create_buffer()?;
            if old_size != 0 {
                ctx.execute_one_time_commands(|cmd| {
                    unsafe {
                        ctx.device.cmd_copy_buffer(
                            *cmd,
                            self.buffer.buffer,
                            buffer.buffer,
                            &[vk::BufferCopy {
                                size: old_size,
                                src_offset: 0,
                                dst_offset: 0,
                            }],
                        )
                    };
                })?;
            }
            unsafe { ctx.device.destroy_buffer(self.buffer.buffer, None) };
            {
                let mut allocator = ctx.allocator.lock().unwrap();
                (*allocator)
                    .free(
                        self.buffer
                            .allocation
                            .take()
                            .ok_or(Error::msg("Buffer not Owned"))?,
                    )
                    .unwrap();
            }
            self.buffer = buffer;
            BindlessDescriptorHeap::get().update_buffer_handle(self, self.bindless_handle);
        }
        Ok(())
    }

    pub fn cmd_grow_to_size(&mut self, size: u64, cmd: &vk::CommandBuffer) -> Result<()> {
        let old_size = self.size;
        self.size = size;
        if size > self.capacity {
            let ctx = Context::get();
            self.capacity = self.size.next_power_of_two();

            let buffer = self.create_buffer()?;
            unsafe {
                ctx.device.cmd_copy_buffer(
                    *cmd,
                    self.buffer.buffer,
                    buffer.buffer,
                    &[vk::BufferCopy {
                        size: old_size,
                        src_offset: 0,
                        dst_offset: 0,
                    }],
                )
            };
            unsafe { ctx.device.destroy_buffer(self.buffer.buffer, None) };
            {
                let mut allocator = ctx.allocator.lock().unwrap();
                (*allocator)
                    .free(
                        self.buffer
                            .allocation
                            .take()
                            .ok_or(Error::msg("Buffer not Owned"))?,
                    )
                    .unwrap();
            }
            self.buffer = buffer;
            BindlessDescriptorHeap::get().update_buffer_handle(self, self.bindless_handle);
        }
        Ok(())
    }

    pub fn copy_from(
        &mut self,
        src_buffer: &impl BufferType,
        offset: u64,
        size: u64,
    ) -> Result<()> {
        if self.size < size + offset {
            self.grow_to_size(size + offset)?;
        }
        Context::get().execute_one_time_commands(|cmd| {
            unsafe {
                Context::get().device.cmd_copy_buffer(
                    *cmd,
                    src_buffer.to_vk(),
                    self.buffer.buffer,
                    &[vk::BufferCopy {
                        src_offset: 0,
                        size,
                        dst_offset: offset,
                    }],
                )
            };
        })
    }
    pub fn copy_from_cmd(
        &mut self,
        src_buffer: &impl BufferType,
        cmd: &vk::CommandBuffer,
        offset: u64,
        size: u64,
    ) -> Result<()> {
        if self.size < size + offset {
            self.grow_to_size(size + offset)?;
        }
        unsafe {
            Context::get().device.cmd_copy_buffer(
                *cmd,
                src_buffer.to_vk(),
                self.buffer.buffer,
                &[vk::BufferCopy {
                    src_offset: 0,
                    size: size,
                    dst_offset: offset,
                }],
            )
        };
        Ok(())
    }

    pub fn push<T: Copy>(&mut self, staging_buffer: &Buffer, data: &[T]) {
        let offset = self.size;
        let size = data.len() * size_of::<T>();

        let old_size = self.size;
        if self.size < size as u64 + offset {
            self.grow_to_size(size as u64 + offset).unwrap();
            log::debug!("Resized Buffer form {} to {}", old_size, self.size);
        }

        for i in 0..size.div_ceil(staging_buffer.size as usize) {
            staging_buffer
                .copy_data_to_buffer(
                    &data[i * staging_buffer.size as usize / size_of::<T>()
                        ..data
                            .len()
                            .min((i + 1) * staging_buffer.size as usize / size_of::<T>())],
                )
                .unwrap();
            self.copy_from(
                staging_buffer,
                offset + i as u64 * staging_buffer.size as u64,
                (staging_buffer.size as u64).min(size as u64),
            )
            .unwrap();
        }
    }
}

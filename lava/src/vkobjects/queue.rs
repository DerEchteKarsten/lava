use anyhow::Result;
use ash::vk;

use crate::state::Ctx;

#[derive(Debug)]
pub struct Queue {
    pub family_index: u32,
    pub handle: vk::Queue,
    pub percistent_command_pool: vk::CommandPool,
}

#[derive(Debug)]
pub struct CommandBuffer {
    pub handle: vk::CommandBuffer,
}

impl CommandBuffer {
    pub fn record<'a, R, F: Fn(&'a vk::CommandBuffer) -> R>(&'a self, f: &F) -> Result<R> {
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { Ctx::device()
                    .begin_command_buffer(self.handle, &begin_info) }?;
        let result = f(&self.handle);
        unsafe { Ctx::device().end_command_buffer(self.handle)? };
        Ok(result)
    }
}

impl Queue {
    pub fn new(device: &ash::Device, family_index: u32) -> Result<Self> {
        let handle = unsafe { device.get_device_queue(family_index, 0) };
        let percistent_command_pool = unsafe { device.create_command_pool(&vk::CommandPoolCreateInfo {
            queue_family_index: family_index,
            ..Default::default()
        }, None) }?;
        
        Ok(Self {
            handle,
            family_index,
            percistent_command_pool
        })
    }

    pub fn cmd(&self) -> CommandBuffer {
        CommandBuffer {
            handle: unsafe {
                Ctx::device().allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_buffer_count(1)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_pool(self.percistent_command_pool),
                ).unwrap()
            }[0]
        }
    }

    pub fn execute_command_wait<R, F: FnOnce(&vk::CommandBuffer) -> R>(
        &self,
        executor: F,
    ) -> Result<R> {
        unsafe {
            let command_buffer = unsafe {
                Ctx::device().allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_buffer_count(1)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_pool(self.percistent_command_pool),
                )?
            }[0];

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            Ctx::device()
                .begin_command_buffer(command_buffer, &begin_info)?;

            let executor_result = executor(&command_buffer);

            Ctx::device().end_command_buffer(command_buffer)?;

            let fence = 
                Ctx::device()
                    .create_fence(&vk::FenceCreateInfo::default(), None)?;

            let cmd_buffer_submit_info =
                vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer);

            let submit_info = vk::SubmitInfo2::default()
                .command_buffer_infos(std::slice::from_ref(&cmd_buffer_submit_info));

            Ctx::device().queue_submit2(
                Ctx::queue().handle,
                std::slice::from_ref(&submit_info),
                fence,
            )?;

            Ctx::device().wait_for_fences(&[fence], true, u64::MAX)?;
            
            Ctx::device()
                .free_command_buffers(self.percistent_command_pool, &[command_buffer]);
            
            Ok(executor_result)
        }
    }
}
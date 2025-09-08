use anyhow::Result;
use ash::vk;

use crate::state::Ctx;

pub struct Queue {
    pub family_index: usize,
    pub handle: vk::Queue,
    pub percistent_command_pool: vk::CommandPool,
}

pub struct CommandBuffer {
    handle: vk::CommandBuffer,
}

impl CommandBuffer {
    pub fn record<R, F: FnOnce(&vk::CommandBuffer) -> R>(&self, f: F) -> Result<()> {
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { Ctx::device()
                    .begin_command_buffer(self.handle, &begin_info) }
        let result = f(&self.handle);
        unsafe { Ctx::device().end_command_buffer(self.handle)? };
        result
    }
}

impl Queue {
    pub fn new(family_index: usize) -> Self {
        let handle = unsafe { Ctx::device().get_device_queue(family_index, 0) };
        let percistent_command_pool = unsafe { Ctx::device().create_command_pool(vk::CommandPoolCreateInfo {
            queue_family_index: family_index as u32,
            ..Default::default()
        }, None) };
        
        Self {
            handle,
            family_index,
            percistent_command_pool
        }
    }

    pub fn cmd(&self) -> CommandBuffer {
        let command_buffer = unsafe {
            Ctx::device().allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_buffer_count(1)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_pool(self.percistent_command_pool),
            )?
        }[0];
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

            self.device.queue_submit2(
                self.transfer_queue,
                std::slice::from_ref(&submit_info),
                fence,
            )?;

            self.device.wait_for_fences(&[fence], true, u64::MAX)?;
            
            self.device
                .free_command_buffers(self.percistent_command_pool, &[command_buffer]);
            
            Ok(executor_result)
        }
    }
}
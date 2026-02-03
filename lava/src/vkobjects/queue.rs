use std::u64;

use anyhow::Result;
use ash::vk;

use crate::{command_buffer::CommandBuffer, state::{Ctx, STATE}};

#[derive(Debug)]
pub struct Queue {
    pub family_index: u32,
    pub handle: vk::Queue,
    pub percistent_command_pool: vk::CommandPool,
}

impl Queue {
    pub fn new(device: &ash::Device, family_index: u32) -> Result<Self> {
        let handle = unsafe { device.get_device_queue(family_index, 0) };
        let percistent_command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo {
                    queue_family_index: family_index,
                    ..Default::default()
                },
                None,
            )
        }?;

        Ok(Self {
            handle,
            family_index,
            percistent_command_pool,
        })
    }

    fn execute_command<R, F: FnOnce(&mut CommandBuffer) -> R>(
        &self,
        executor: F,
    ) -> Result<(R, vk::Fence, vk::CommandBuffer)> {
        unsafe {
            let command_buffer = Ctx::device().allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_buffer_count(1)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_pool(self.percistent_command_pool),
            )?[0];

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            Ctx::device().begin_command_buffer(command_buffer, &begin_info)?;

            let mut resource_hashes = STATE.get().unwrap().resource_cache.lock().unwrap();

            let mut cmd_buffer = CommandBuffer {
                handle: command_buffer,
                resource_hashes: &mut resource_hashes
            };
            let executor_result = executor(&mut cmd_buffer);

            Ctx::device().end_command_buffer(command_buffer)?;

            let fence = Ctx::device().create_fence(&vk::FenceCreateInfo::default(), None)?;

            let cmd_buffer_submit_info =
                vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer);

            let submit_info = vk::SubmitInfo2::default()
                .command_buffer_infos(std::slice::from_ref(&cmd_buffer_submit_info));

            Ctx::device().queue_submit2(
                Ctx::queue().handle,
                std::slice::from_ref(&submit_info),
                fence,
            )?;

            Ok((executor_result, fence, command_buffer))
        }
    }

    pub fn execute_command_wait<R, F: FnOnce(&mut CommandBuffer) -> R>(
        &self,
        executor: F,
    ) -> Result<R> {
        let (res, fence, command_buffer) = self.execute_command(executor)?;
        unsafe {
            Ctx::device().wait_for_fences(&[fence], true, u64::MAX);
            Ctx::device().free_command_buffers(self.percistent_command_pool, &[command_buffer]);
        }
        Ok(res)
    }

    pub async fn execute_command_async<R, F: FnOnce(&mut CommandBuffer) -> R>(
        &self,
        executor: F,
    ) -> Result<R> {
        let (res, fence, command_buffer) = self.execute_command(executor)?;
        unsafe {
            FenceFuture {
                fence
            }.await;
            Ctx::device().free_command_buffers(self.percistent_command_pool, &[command_buffer]);
        }
        Ok(res)
    }
    
}

struct FenceFuture { 
    fence: vk::Fence,
}

impl Future for FenceFuture {
    type Output = ();
    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Self::Output> {
        if unsafe { Ctx::device().get_fence_status(self.fence).unwrap() } {
            std::task::Poll::Ready(())
        }else  {
            std::task::Poll::Pending
        }
    }
}

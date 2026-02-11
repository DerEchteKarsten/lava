use std::{
    cell::UnsafeCell, collections::HashMap, marker::PhantomData, rc::Rc, sync::atomic::Ordering,
    u64,
};

use anyhow::{Result, anyhow};
use ash::{prelude::VkResult, vk};

use crate::{
    command_buffer::CommandBuffer,
    state::{Ctx, Functions, STATE},
    vkobjects::{physical_device::QueueFamily, swapchain::Swapchain},
};

#[derive(Debug)]
pub struct Semaphore<T: SemaphoreType + ?Sized> {
    pub(crate) handle: vk::Semaphore,
    marker: PhantomData<T>,
}

impl<T: SemaphoreType> Default for Semaphore<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: SemaphoreType + ?Sized> Drop for Semaphore<T> {
    fn drop(&mut self) {
        unsafe { Ctx::device().destroy_semaphore(self.handle, None) };
    }
}

#[derive(Clone, Copy)]
pub enum SemaphoreInfo {
    Timeline(vk::Semaphore, u64),
    Binary(vk::Semaphore),
}
impl SemaphoreInfo {
    fn to_vk<'a>(&'a self, stage: vk::PipelineStageFlags2) -> vk::SemaphoreSubmitInfo<'a> {
        let sub = vk::SemaphoreSubmitInfo::default().stage_mask(stage);
        match self {
            SemaphoreInfo::Binary(handle) => sub.semaphore(*handle),
            SemaphoreInfo::Timeline(handle, value) => sub.semaphore(*handle).value(*value),
        }
    }
}

trait SemaphoreType {
    fn create() -> Semaphore<Self>;
}

#[derive(Debug)]
pub struct Timeline;
impl SemaphoreType for Timeline {
    fn create() -> Semaphore<Self> {
        let mut timeline = vk::SemaphoreTypeCreateInfo {
            initial_value: 0,
            semaphore_type: vk::SemaphoreType::TIMELINE,
            ..Default::default()
        };
        let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut timeline);
        let handle = unsafe { Ctx::device().create_semaphore(&create_info, None) }.unwrap();
        Semaphore {
            handle,
            marker: PhantomData,
        }
    }
}

#[derive(Debug)]
pub struct Binary;
impl SemaphoreType for Binary {
    fn create() -> Semaphore<Self> {
        let create_info = vk::SemaphoreCreateInfo::default();
        let handle = unsafe { Ctx::device().create_semaphore(&create_info, None) }.unwrap();
        Semaphore {
            handle,
            marker: PhantomData,
        }
    }
}

impl<T: SemaphoreType> Semaphore<T> {
    pub fn new() -> Self {
        T::create()
    }
}

impl Semaphore<Timeline> {
    pub fn block_until_value(&self, value: u64) {
        let binding = [self.handle];
        let values = [value];
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&binding)
            .values(&values);
        unsafe { Ctx::device().wait_semaphores(&wait_info, u64::MAX).unwrap() }
    }
    pub fn info(&self, value: u64) -> SemaphoreInfo {
        SemaphoreInfo::Timeline(self.handle, value)
    }
}

impl Semaphore<Binary> {
    pub fn info(&self) -> SemaphoreInfo {
        SemaphoreInfo::Binary(self.handle)
    }
}

pub struct Fence {
    handle: vk::Fence,
}
impl Drop for Fence {
    fn drop(&mut self) {
        unsafe { Ctx::device().destroy_fence(self.handle, None) };
    }
}

impl Fence {
    pub fn new() -> Self {
        let create_info = vk::FenceCreateInfo::default();
        let handle = unsafe { Ctx::device().create_fence(&create_info, None) }.unwrap();
        Self { handle }
    }
    pub fn reset(&self) {
        unsafe { Ctx::device().reset_fences(&[self.handle]) }.unwrap();
    }
    pub fn wait(&self) {
        unsafe { Ctx::device().wait_for_fences(&[self.handle], true, u64::MAX) }.unwrap();
    }
    pub fn wait_async(&self) -> FenceFuture {
        FenceFuture { fence: self.handle }
    }
}

#[derive(Debug)]
pub struct Queue {
    handle: vk::Queue,
    familie: u32,
}

#[derive(Debug)]
pub struct CommandPool {
    handle: vk::CommandPool,
}
impl CommandPool {
    pub fn reset(&mut self) {
        unsafe {
            Ctx::device().reset_command_pool(self.handle, vk::CommandPoolResetFlags::empty())
        }
        .unwrap();
    }
    pub fn create_command_buffer(&mut self) -> CommandBufferMemory {
        let allocate_info = vk::CommandBufferAllocateInfo {
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            command_pool: self.handle,
            ..Default::default()
        };
        let handle = unsafe {
            Ctx::device()
                .allocate_command_buffers(&allocate_info)
                .unwrap()
        }[0];
        CommandBufferMemory {
            pool: self.handle,
            handle,
        }
    }
}
impl Drop for CommandPool {
    fn drop(&mut self) {
        unsafe { Ctx::device().destroy_command_pool(self.handle, None) };
    }
}

#[derive(Debug)]
pub struct CommandBufferMemory {
    pool: vk::CommandPool,
    handle: vk::CommandBuffer,
}

impl Drop for CommandBufferMemory {
    fn drop(&mut self) {
        unsafe { Ctx::device().free_command_buffers(self.pool, &[self.handle]) };
    }
}

impl Queue {
    pub fn new(familie: u32, idx: u32) -> Result<Self> {
        Ok(Self {
            handle: unsafe { Ctx::device().get_device_queue(familie, idx) },
            familie,
        })
    }

    pub fn create_pool(&self) -> CommandPool {
        CommandPool {
            handle: unsafe {
                Ctx::device().create_command_pool(
                    &vk::CommandPoolCreateInfo {
                        queue_family_index: self.familie,
                        ..Default::default()
                    },
                    None,
                )
            }
            .unwrap(),
        }
    }

    pub fn execute_command<F: FnOnce(&mut CommandBuffer)>(
        &self,
        buffer: &mut CommandBufferMemory,
        fence: Option<&Fence>,
        wait_on: &[SemaphoreInfo],
        signal: &[SemaphoreInfo],
        executor: F,
    ) -> Result<()> {
        unsafe {
            let mut cmd_buffer = CommandBuffer {
                handle: buffer.handle,
                last_stage: vk::PipelineStageFlags2::TOP_OF_PIPE,
                resource_hashes: HashMap::new(),
            };

            cmd_buffer.begin();
            executor(&mut cmd_buffer);
            cmd_buffer.end();

            let cmd_buffer_submit_info =
                vk::CommandBufferSubmitInfo::default().command_buffer(buffer.handle);
            let wait_infos: Vec<_> = wait_on
                .iter()
                .map(|sem| sem.to_vk(cmd_buffer.last_stage))
                .collect();
            let signal_infos: Vec<_> = signal
                .iter()
                .map(|sem| sem.to_vk(vk::PipelineStageFlags2::ALL_COMMANDS))
                .collect();

            let submit_info = vk::SubmitInfo2::default()
                .command_buffer_infos(std::slice::from_ref(&cmd_buffer_submit_info))
                .wait_semaphore_infos(&wait_infos)
                .signal_semaphore_infos(&signal_infos);

            Ctx::device().queue_submit2(
                self.handle,
                std::slice::from_ref(&submit_info),
                fence.map(|e| e.handle).unwrap_or(vk::Fence::null()),
            )?;
            Ok(())
        }
    }

    pub fn present(&self, swapchain: &Swapchain, image_index: u32, wait_on: &[&Semaphore<Binary>]) -> Result<bool> {
        let sc = [swapchain.handle];
        let ii = [image_index];
        let waits: Vec<_> = wait_on.iter().map(|sem| sem.handle).collect();
        let present_info = vk::PresentInfoKHR::default()
            .swapchains(&sc)
            .image_indices(&ii)
            .wait_semaphores(waits.as_slice());
        match unsafe { Functions::swapchain().queue_present(self.handle, &present_info) } {
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                Ok(true)
            }
            Ok(false) => Ok(false),
            Err(e) => Err(anyhow!("present failed: {e:?}")),
        }
    }
}

pub struct FenceFuture {
    fence: vk::Fence,
}

impl Future for FenceFuture {
    type Output = ();
    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        if unsafe { Ctx::device().get_fence_status(self.fence).unwrap() } {
            std::task::Poll::Ready(())
        } else {
            std::task::Poll::Pending
        }
    }
}

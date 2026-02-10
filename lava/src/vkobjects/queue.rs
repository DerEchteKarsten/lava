use std::{marker::PhantomData, u64};

use anyhow::Result;
use ash::vk;

use crate::{
    command_buffer::CommandBuffer,
    state::{Ctx, STATE},
};

#[derive(Debug)]
pub struct Queue {
    pub handle: vk::Queue,
    pub command_pools: Vec<vk::CommandPool>,
}

impl Drop for Queue {
    fn drop(&mut self) {
        for c in &self.command_pools {
            unsafe { Ctx::device().destroy_command_pool(*c, None) };
        }
    }
}

pub struct Semaphore<T: SemaphoreType + ?Sized> {
    handle: vk::Semaphore,
    marker: PhantomData<T>
}

impl<T: SemaphoreType + ?Sized> Drop for Semaphore<T> {
    fn drop(&mut self) {
        unsafe { Ctx::device().destroy_semaphore(self.handle, None) };
    }
}

enum SemaphoreWait {
    Timeline(vk::Semaphore, u64),
    Binary(vk::Semaphore),
}
impl SemaphoreWait{
    fn to_vk<'a>(&'a self, stage: vk::PipelineStageFlags2) -> vk::SemaphoreSubmitInfo<'a> {
        let sub = vk::SemaphoreSubmitInfo::default()
            .stage_mask(stage); 
        match self {
            SemaphoreWait::Binary(handle) => sub.semaphore(*handle),
            SemaphoreWait::Timeline(handle, value) => sub.semaphore(*handle).value(*value) 
        }
    }
}

trait SemaphoreType {
    fn create() -> Semaphore<Self>;
}

struct Timeline;
impl SemaphoreType for Timeline {
    fn create() -> Semaphore<Self> {
        let mut timeline = vk::SemaphoreTypeCreateInfo {
            initial_value: 0,
            semaphore_type: vk::SemaphoreType::TIMELINE,
            ..Default::default()
        };
        let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut timeline);
        let handle = unsafe { Ctx::device().create_semaphore(&create_info, None) }.unwrap();
        Semaphore { handle, marker: PhantomData }
    }
}
struct Binary;
impl SemaphoreType for Binary {
    fn create() -> Semaphore<Self> {
        let create_info = vk::SemaphoreCreateInfo::default();
        let handle = unsafe { Ctx::device().create_semaphore(&create_info, None) }.unwrap();
        Semaphore { handle, marker: PhantomData }
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
    pub fn wait_value(&self, value: u64) -> SemaphoreWait {
        SemaphoreWait::Timeline(self.handle, value)
    } 
}

impl Semaphore<Binary> {
    pub fn wait_value(&self) -> SemaphoreWait {
        SemaphoreWait::Binary(self.handle)
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
        Self {
            handle
        }
    }

    pub fn wait(&self) {
        unsafe { Ctx::device().wait_for_fences(&[self.handle], true, u64::MAX) }.unwrap();
    }
    pub fn wait_async(&self) -> FenceFuture {
        FenceFuture { fence: self.handle }
    }
}

pub struct CommandBufferMemory {
    pool: vk::CommandPool,
    buffer: vk::CommandBuffer,
}

impl CommandBufferMemory {
    pub fn done(self) {
        unsafe { Ctx::device().free_command_buffers(self.pool, &[self.buffer]) };
    }
}


impl Queue {
    pub fn new(family_index: u32, num: u32, num_pools: usize) -> Result<Self> {
        let handle = unsafe { Ctx::device().get_device_queue(family_index, num) };
        let command_pools = (0..num_pools).map(|_| unsafe {
            Ctx::device().create_command_pool(
                &vk::CommandPoolCreateInfo {
                    queue_family_index: family_index,
                    ..Default::default()
                },
                None,
            )
        }.unwrap()).collect();

        Ok(Self {
            handle,
            command_pools,
        })
    }

    pub fn clear(&self, pool: usize) {
        unsafe {
            Ctx::device().reset_command_pool(self.command_pools[pool], vk::CommandPoolResetFlags::empty());
        }
    }

    pub fn execute_command<F: FnOnce(&mut CommandBuffer)>(
        &self,
        pool: usize,
        executor: F,
        fence: Option<Fence>,
        wait_on: &[SemaphoreWait],
        signal: &[SemaphoreWait],
    ) -> Result<()> {
        unsafe {
            let command_buffer = Ctx::device().allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_buffer_count(1)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_pool(self.command_pools[pool]),
            )?[0];

            let mut resource_hashes = STATE.get().unwrap().resource_cache.lock().unwrap();
            let mut cmd_buffer = CommandBuffer {
                handle: command_buffer,
                last_stage: vk::PipelineStageFlags2::TOP_OF_PIPE,
                resource_hashes: &mut resource_hashes,
            };

            cmd_buffer.begin();
            executor(&mut cmd_buffer);
            cmd_buffer.end();
            
            let cmd_buffer_submit_info =
                vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer);
            let wait_infos: Vec<_> = wait_on.iter().map(|sem| sem.to_vk(cmd_buffer.last_stage)).collect();
            let signal_infos: Vec<_> = signal.iter().map(|sem| sem.to_vk(vk::PipelineStageFlags2::ALL_COMMANDS)).collect();

            let submit_info = vk::SubmitInfo2::default()
                .command_buffer_infos(std::slice::from_ref(&cmd_buffer_submit_info))
                .wait_semaphore_infos(&wait_infos)
                .signal_semaphore_infos(&signal_infos);

            Ctx::device().queue_submit2(
                self.handle,
                std::slice::from_ref(&submit_info),
                fence.map(|e|e.handle).unwrap_or(vk::Fence::null()),
            )?;
            Ok(())
        }
    }
}

struct FenceFuture {
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

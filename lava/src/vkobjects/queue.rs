use std::{
    collections::HashMap,
    fmt::Debug,
    future::Future,
    marker::PhantomData,
    sync::atomic::{AtomicBool, Ordering},
};

use crate::error::{Error, Result};
use ash::vk;
use lava_macros::validation_trace;
use smallvec::SmallVec;

use crate::{
    command_buffer::{BufferAccess, CommandBuffer, ImageAccess},
    state::{CALLSITE, Ctx, Functions},
    vkobjects::swapchain::Swapchain,
};

#[derive(Debug)]
pub struct Semaphore<T: SemaphoreType + ?Sized> {
    pub(crate) handle: vk::Semaphore,
    marker: PhantomData<T>,
}

impl<T: SemaphoreType> Default for Semaphore<T> {
    fn default() -> Self {
        Self::new().expect("failed to create default semaphore")
    }
}

impl<T: SemaphoreType + ?Sized> Drop for Semaphore<T> {
    fn drop(&mut self) {
        unsafe { Ctx::device().destroy_semaphore(self.handle, None) };
    }
}

pub struct Event {
    pub(crate) handle: vk::Event,
}

impl Event {
    #[validation_trace]
    pub fn new() -> Result<Self> {
        let create_info = vk::EventCreateInfo::default();
        Ok(Self {
            handle: unsafe { Ctx::device().create_event(&create_info, None)? },
        })
    }
    #[validation_trace]
    pub fn wait(&self) -> Result<()> {
        loop {
            if unsafe { Ctx::device().get_event_status(self.handle)? } {
                unsafe { Ctx::device().set_event(self.handle)? };
                break;
            }
            std::thread::yield_now();
        }
        Ok(())
    }
    #[validation_trace]
    pub fn set(&self) -> Result<()> {
        unsafe { Ctx::device().set_event(self.handle)? };
        Ok(())
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        unsafe { Ctx::device().destroy_event(self.handle, None) };
    }
}

#[derive(Clone, Copy)]
pub enum SemaphoreInfo {
    Timeline(vk::Semaphore, u64),
    Binary(vk::Semaphore),
}
impl SemaphoreInfo {
    pub(crate) fn to_vk<'a>(
        &'a self,
        stage: vk::PipelineStageFlags2,
    ) -> vk::SemaphoreSubmitInfo<'a> {
        let sub = vk::SemaphoreSubmitInfo::default().stage_mask(stage);
        match self {
            SemaphoreInfo::Binary(handle) => sub.semaphore(*handle),
            SemaphoreInfo::Timeline(handle, value) => sub.semaphore(*handle).value(*value),
        }
    }
}

pub trait SemaphoreType {
    fn create() -> Result<Semaphore<Self>>;
}

#[derive(Debug)]
pub struct Timeline;
impl SemaphoreType for Timeline {
    fn create() -> Result<Semaphore<Self>> {
        let mut timeline = vk::SemaphoreTypeCreateInfo {
            initial_value: 0,
            semaphore_type: vk::SemaphoreType::TIMELINE,
            ..Default::default()
        };
        let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut timeline);
        let handle = unsafe { Ctx::device().create_semaphore(&create_info, None)? };
        Ok(Semaphore {
            handle,
            marker: PhantomData,
        })
    }
}

#[derive(Debug)]
pub struct Binary;
impl SemaphoreType for Binary {
    fn create() -> Result<Semaphore<Self>> {
        let create_info = vk::SemaphoreCreateInfo::default();
        let handle = unsafe { Ctx::device().create_semaphore(&create_info, None)? };
        Ok(Semaphore {
            handle,
            marker: PhantomData,
        })
    }
}

impl<T: SemaphoreType> Semaphore<T> {
    #[validation_trace]
    pub fn new() -> Result<Self> {
        T::create()
    }
}

impl Semaphore<Timeline> {
    #[validation_trace]
    pub fn block_until_value(&self, value: u64) -> Result<()> {
        let binding = [self.handle];
        let values = [value];
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&binding)
            .values(&values);
        unsafe { Ctx::device().wait_semaphores(&wait_info, u64::MAX)? };
        Ok(())
    }
    #[validation_trace]
    pub fn info(&self, value: u64) -> SemaphoreInfo {
        SemaphoreInfo::Timeline(self.handle, value)
    }
}

impl Semaphore<Binary> {
    #[validation_trace]
    pub fn info(&self) -> SemaphoreInfo {
        SemaphoreInfo::Binary(self.handle)
    }
}

pub struct Fence {
    pub(crate) handle: vk::Fence,
}
impl Drop for Fence {
    fn drop(&mut self) {
        unsafe { Ctx::device().destroy_fence(self.handle, None) };
    }
}

impl Default for Fence {
    fn default() -> Self {
        Self::new().expect("failed to create default fence")
    }
}

impl Fence {
    #[validation_trace]
    pub fn new() -> Result<Self> {
        let create_info = vk::FenceCreateInfo::default();
        let handle = unsafe { Ctx::device().create_fence(&create_info, None)? };
        Ok(Self { handle })
    }
    #[validation_trace]
    pub fn reset(&self) -> Result<()> {
        unsafe { Ctx::device().reset_fences(&[self.handle])? };
        Ok(())
    }
    #[validation_trace]
    pub fn wait(&self) -> Result<()> {
        unsafe { Ctx::device().wait_for_fences(&[self.handle], true, u64::MAX)? };
        Ok(())
    }
}

pub trait QueueFamilie: Debug {
    fn index() -> u32;
    fn is_free() -> &'static [AtomicBool];
}

#[derive(Debug)]
pub struct Transfer;
#[derive(Debug)]
pub struct Present;
#[derive(Debug)]
pub struct Gfx;

impl QueueFamilie for Transfer {
    fn index() -> u32 {
        Ctx::transfer_queue_index()
    }
    fn is_free() -> &'static [AtomicBool] {
        Ctx::get()
            .transfer_queues_in_use
            .as_ref()
            .unwrap_or(&Ctx::get().gfx_queues_in_use)
    }
}
impl QueueFamilie for Present {
    fn index() -> u32 {
        Ctx::present_queue_index()
    }
    fn is_free() -> &'static [AtomicBool] {
        Ctx::get()
            .present_queues_in_use
            .as_ref()
            .unwrap_or(&Ctx::get().gfx_queues_in_use)
    }
}
impl QueueFamilie for Gfx {
    fn index() -> u32 {
        Ctx::gfx_queue_index()
    }
    fn is_free() -> &'static [AtomicBool] {
        &Ctx::get().gfx_queues_in_use
    }
}

#[derive(Debug)]
pub struct Queue<Q: QueueFamilie> {
    pub(crate) handle: vk::Queue,
    familie: u32,
    idx: u32,
    _marker: PhantomData<Q>,
}

impl<Q: QueueFamilie> Drop for Queue<Q> {
    fn drop(&mut self) {
        Q::is_free()[self.idx as usize].store(false, Ordering::Relaxed);
    }
}

#[derive(Debug)]
pub struct CommandPool {
    handle: vk::CommandPool,
}
impl CommandPool {
    #[validation_trace]
    pub fn reset(&self) -> Result<()> {
        unsafe {
            Ctx::device().reset_command_pool(self.handle, vk::CommandPoolResetFlags::empty())?
        };
        Ok(())
    }
    #[validation_trace]
    pub fn create_command_buffer(&self) -> Result<CommandBufferMemory> {
        let allocate_info = vk::CommandBufferAllocateInfo {
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            command_pool: self.handle,
            ..Default::default()
        };
        let handle = unsafe { Ctx::device().allocate_command_buffers(&allocate_info)? }[0];
        Ok(CommandBufferMemory {
            pool: self.handle,
            handle,
        })
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        unsafe { Ctx::device().destroy_command_pool(self.handle, None) };
    }
}

#[derive(Debug)]
pub struct CommandBufferMemory {
    pub(crate) pool: vk::CommandPool,
    pub(crate) handle: vk::CommandBuffer,
}

impl Drop for CommandBufferMemory {
    fn drop(&mut self) {
        unsafe {
            Ctx::device().free_command_buffers(self.pool, &[self.handle]);
        }
    }
}

#[derive(Debug, Default)]
pub struct PendingAccesses {
    pub(crate) buffer_reads: SmallVec<[BufferAccess; 8]>,
    pub(crate) image_reads: SmallVec<[ImageAccess; 8]>,
    pub(crate) buffer_writes: SmallVec<[BufferAccess; 8]>,
    pub(crate) image_writes: SmallVec<[ImageAccess; 8]>,
    pub(crate) image_layouts: SmallVec<[(vk::Image, vk::ImageLayout); 4]>,
}

impl<Q: QueueFamilie> Queue<Q> {
    #[validation_trace]
    pub fn new() -> Result<Self> {
        let mut handle = None;
        for (i, slot) in Q::is_free().iter().enumerate() {
            if slot
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                handle = Some((i as u32, unsafe {
                    Ctx::device().get_device_queue(Q::index(), i as u32)
                }));
                break;
            }
        }
        let (idx, handle) = handle.ok_or(Error::message("All queues used up"))?;
        Ok(Self {
            handle,
            idx,
            familie: Q::index(),
            _marker: PhantomData,
        })
    }
    #[validation_trace]
    pub fn create_pool(&self) -> Result<CommandPool> {
        Ok(CommandPool {
            handle: unsafe {
                Ctx::device().create_command_pool(
                    &vk::CommandPoolCreateInfo {
                        queue_family_index: self.familie,
                        ..Default::default()
                    },
                    None,
                )?
            },
        })
    }

    #[validation_trace]
    pub fn execute_command<F: FnOnce(&mut CommandBuffer)>(
        &self,
        pending: PendingAccesses,
        buffer: &CommandBufferMemory,
        fence: Option<&Fence>,
        wait_on: &[SemaphoreInfo],
        signal: &[SemaphoreInfo],
        executor: F,
    ) -> Result<PendingAccesses> {
        unsafe {
            let mut cmd_buffer = CommandBuffer {
                handle: buffer.handle,
                pending_accesses: pending,
            };

            cmd_buffer.begin()?;
            let prev = CALLSITE.replace(None);
            executor(&mut cmd_buffer);
            CALLSITE.set(prev);
            cmd_buffer.end()?;

            let cmd_buffer_submit_info =
                vk::CommandBufferSubmitInfo::default().command_buffer(buffer.handle);
            let wait_infos: SmallVec<[_; 1]> = wait_on
                .iter()
                .map(|sem| sem.to_vk(vk::PipelineStageFlags2::ALL_COMMANDS))
                .collect();
            let mut last_stage = vk::PipelineStageFlags2::empty();
            for i in &cmd_buffer.pending_accesses.image_reads {
                last_stage |= i.stage;
            }
            for i in &cmd_buffer.pending_accesses.image_writes {
                last_stage |= i.stage;
            }
            for i in &cmd_buffer.pending_accesses.buffer_reads {
                last_stage |= i.stage;
            }
            for i in &cmd_buffer.pending_accesses.buffer_writes {
                last_stage |= i.stage;
            }
            let signal_infos: SmallVec<[_; 1]> =
                signal.iter().map(|sem| sem.to_vk(last_stage)).collect();

            let submit_info = vk::SubmitInfo2::default()
                .command_buffer_infos(std::slice::from_ref(&cmd_buffer_submit_info))
                .wait_semaphore_infos(&wait_infos)
                .signal_semaphore_infos(&signal_infos);

            Ctx::device().queue_submit2(
                self.handle,
                std::slice::from_ref(&submit_info),
                fence.map(|e| e.handle).unwrap_or(vk::Fence::null()),
            )?;

            Ok(cmd_buffer.pending_accesses)
        }
    }

    #[validation_trace]
    pub fn present(
        &self,
        swapchain: &Swapchain,
        image_index: u32,
        wait_on: &[&Semaphore<Binary>],
    ) -> Result<bool> {
        let sc = [swapchain.handle];
        let ii = [image_index];
        let waits: Vec<_> = wait_on.iter().map(|sem| sem.handle).collect();
        let present_info = vk::PresentInfoKHR::default()
            .swapchains(&sc)
            .image_indices(&ii)
            .wait_semaphores(waits.as_slice());
        match unsafe { Functions::swapchain().queue_present(self.handle, &present_info) } {
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(true),
            Ok(false) => Ok(false),
            Err(e) => Err(Error::Vulkan(e)),
        }
    }
}

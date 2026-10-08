//! Queue submission and sync: typed queues, semaphores, fences, frame slots, presenting
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
    command_buffer::{self, BufferAccess, CommandBuffer, ImageAccess},
    profiling::{FrameQueries, FrameTimings},
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
        &Ctx::get().gfx_queues_in_use
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

pub struct FrameSlot {
    retired: Vec<Box<dyn Send + Sync>>,
    buffer: CommandBufferMemory,
    pool: CommandPool,
    fence: Fence,
    submitted: bool,
    family: u32,
    profiling: bool,
    queries: Option<FrameQueries>,
}

impl FrameSlot {
    #[validation_trace]
    pub fn new<Q: QueueFamilie>(queue: &Queue<Q>) -> Result<Self> {
        let pool = queue.create_pool()?;
        let buffer = pool.create_command_buffer()?;
        Ok(Self {
            retired: Vec::new(),
            buffer,
            pool,
            fence: Fence::new()?,
            submitted: false,
            family: queue.familie,
            profiling: false,
            queries: None,
        })
    }

    /// Times the scopes of the frames begun from now on; see [`Frame::last_timings`].
    pub fn set_profiling(&mut self, profiling: bool) {
        self.profiling = profiling;
    }

    #[validation_trace]
    pub fn begin(&mut self) -> Result<Frame<'_>> {
        if self.submitted {
            // `wait`: the profiler doesn't count it as CPU work.
            tracing::info_span!("wait for frame slot", wait = true).in_scope(|| self.fence.wait())?;
            if let Some(queries) = &mut self.queries {
                queries.read()?;
            }
            self.fence.reset()?;
            self.pool.reset()?;
            self.submitted = false;
            self.retired.clear();
        }
        // The pools are idle now.
        if !self.profiling {
            self.queries = None;
        } else if self.queries.is_none() {
            self.queries = Some(FrameQueries::new(self.family)?);
        }
        let mut frame = Frame { slot: self };
        for pipeline in command_buffer::apply_pending_reloads() {
            frame.retire(pipeline);
        }
        Ok(frame)
    }
}

pub struct Frame<'a> {
    slot: &'a mut FrameSlot,
}

impl Frame<'_> {
    pub fn retire<T: Send + Sync + 'static>(&mut self, value: T) {
        self.slot.retired.push(Box::new(value));
    }

    /// The GPU scopes and shader clock times of the frame this slot submitted before, if it
    /// was profiled.
    pub fn last_timings(&self) -> Option<&FrameTimings> {
        self.slot.queries.as_ref()?.results()
    }

    #[validation_trace]
    pub fn execute<Q: QueueFamilie, F: FnOnce(&mut CommandBuffer)>(
        self,
        queue: &Queue<Q>,
        pending: PendingAccesses,
        wait_on: &[SemaphoreInfo],
        signal: &[SemaphoreInfo],
        executor: F,
    ) -> Result<PendingAccesses> {
        let slot = self.slot;
        let result = queue.execute_command(
            pending,
            &slot.buffer,
            Some(&slot.fence),
            &mut slot.queries,
            wait_on,
            signal,
            executor,
        );
        match result {
            Ok(_) => slot.submitted = true,
            Err(_) => slot.pool.reset()?,
        }
        result
    }
}

impl Drop for FrameSlot {
    fn drop(&mut self) {
        if self.submitted
            && let Err(err) = self.fence.wait()
        {
            tracing::error!(%err, "failed to wait for frame slot fence");
        }
    }
}

#[derive(Debug, Default)]
pub struct PendingAccesses {
    pub(crate) buffer_reads: SmallVec<[BufferAccess; 8]>,
    pub(crate) image_reads: SmallVec<[ImageAccess; 8]>,
    pub(crate) buffer_writes: SmallVec<[PendingWrite<BufferAccess>; 8]>,
    pub(crate) image_writes: SmallVec<[PendingWrite<ImageAccess>; 8]>,
}

/// A write later commands may have to wait for, and the stages and accesses a barrier recorded
/// since already made it visible to: those need no barrier for it anymore.
#[derive(Debug, Clone)]
pub(crate) struct PendingWrite<A> {
    pub(crate) write: A,
    pub(crate) visible_stages: vk::PipelineStageFlags2,
    pub(crate) visible_access: vk::AccessFlags2,
}

impl<A> PendingWrite<A> {
    pub(crate) fn new(write: A) -> Self {
        Self {
            write,
            visible_stages: vk::PipelineStageFlags2::NONE,
            visible_access: vk::AccessFlags2::NONE,
        }
    }

    pub(crate) fn visible_to(
        &self,
        stage: vk::PipelineStageFlags2,
        access: vk::AccessFlags2,
    ) -> bool {
        self.visible_stages.contains(stage) && self.visible_access.contains(access)
    }
}

impl PendingAccesses {
    /// Union of the pipeline stages of every access still pending, i.e. the stages a
    /// submission has to finish before its signal semaphores may fire.
    pub(crate) fn last_stage(&self) -> vk::PipelineStageFlags2 {
        let images = self
            .image_reads
            .iter()
            .chain(self.image_writes.iter().map(|w| &w.write));
        let buffers = self
            .buffer_reads
            .iter()
            .chain(self.buffer_writes.iter().map(|w| &w.write));
        images
            .map(|i| i.stage)
            .chain(buffers.map(|b| b.stage))
            .fold(vk::PipelineStageFlags2::empty(), |acc, stage| acc | stage)
    }
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

    /// Records `executor` into `buffer` and submits it. With `queries`, the recording is
    /// timed: the root scope "frame" around it, plus every pass and `begin_scope`.
    #[validation_trace]
    pub fn execute_command<F: FnOnce(&mut CommandBuffer)>(
        &self,
        pending: PendingAccesses,
        buffer: &CommandBufferMemory,
        fence: Option<&Fence>,
        queries: &mut Option<FrameQueries>,
        wait_on: &[SemaphoreInfo],
        signal: &[SemaphoreInfo],
        executor: F,
    ) -> Result<PendingAccesses> {
        unsafe {
            let mut cmd_buffer = CommandBuffer {
                handle: buffer.handle,
                pending_accesses: pending,
                queries: queries.take(),
            };

            cmd_buffer.begin()?;
            if let Some(queries) = &mut cmd_buffer.queries {
                queries.reset(buffer.handle);
            }
            cmd_buffer.begin_scope("frame");
            let prev = CALLSITE.replace(None);
            executor(&mut cmd_buffer);
            CALLSITE.set(prev);
            cmd_buffer.end_scope();
            if let Some(queries) = &cmd_buffer.queries {
                queries.finish(buffer.handle);
            }
            *queries = cmd_buffer.queries.take();
            cmd_buffer.end()?;

            let cmd_buffer_submit_info =
                vk::CommandBufferSubmitInfo::default().command_buffer(buffer.handle);
            let wait_infos: SmallVec<[_; 1]> = wait_on
                .iter()
                .map(|sem| sem.to_vk(vk::PipelineStageFlags2::ALL_COMMANDS))
                .collect();
            let last_stage = cmd_buffer.pending_accesses.last_stage();
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
        let _span = tracing::info_span!("present", wait = true).entered();
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

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;

    #[test]
    fn binary_semaphore_info_has_no_value() {
        let info = SemaphoreInfo::Binary(vk::Semaphore::from_raw(7));
        let submit = info.to_vk(vk::PipelineStageFlags2::COMPUTE_SHADER);
        assert_eq!(submit.semaphore, vk::Semaphore::from_raw(7));
        assert_eq!(submit.value, 0);
        assert_eq!(submit.stage_mask, vk::PipelineStageFlags2::COMPUTE_SHADER);
    }

    #[test]
    fn timeline_semaphore_info_carries_its_value() {
        let info = SemaphoreInfo::Timeline(vk::Semaphore::from_raw(9), 12);
        let submit = info.to_vk(vk::PipelineStageFlags2::ALL_COMMANDS);
        assert_eq!(submit.semaphore, vk::Semaphore::from_raw(9));
        assert_eq!(submit.value, 12);
        assert_eq!(submit.stage_mask, vk::PipelineStageFlags2::ALL_COMMANDS);
    }

    #[test]
    fn last_stage_is_empty_without_pending_accesses() {
        assert!(PendingAccesses::default().last_stage().is_empty());
    }

    #[test]
    fn last_stage_is_the_union_over_all_pending_accesses() {
        use vk::PipelineStageFlags2 as S;
        let buffer = |stage| BufferAccess {
            stage,
            access: vk::AccessFlags2::empty(),
            range: (0u64..4).into(),
        };
        let image = |stage| ImageAccess {
            stage,
            access: vk::AccessFlags2::empty(),
            image: vk::Image::null(),
            layout: vk::ImageLayout::GENERAL,
            aspect: vk::ImageAspectFlags::COLOR,
            old_layout: vk::ImageLayout::GENERAL,
        };
        let mut pending = PendingAccesses::default();
        pending.buffer_reads.push(buffer(S::VERTEX_SHADER));
        pending
            .buffer_writes
            .push(PendingWrite::new(buffer(S::COMPUTE_SHADER)));
        pending.image_reads.push(image(S::FRAGMENT_SHADER));
        pending
            .image_writes
            .push(PendingWrite::new(image(S::COLOR_ATTACHMENT_OUTPUT)));
        assert_eq!(
            pending.last_stage(),
            S::VERTEX_SHADER | S::COMPUTE_SHADER | S::FRAGMENT_SHADER | S::COLOR_ATTACHMENT_OUTPUT
        );
    }
}

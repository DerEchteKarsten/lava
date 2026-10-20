//! Uploads into memory the host can't map: writes of `Gpu` buffers and images staged in `Upload` buffers and queued until the next submission records their copies before its closure (`from_slice`, `write`, `Image::upload`), and the per-frame-slot staging ring of in-recording uploads (`cmd.upload`)
use std::{
    range::Range,
    sync::{
        Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};

use bytemuck::Pod;
use lava_macros::validation_trace;

use crate::{
    buffer::{
        Buffer, Gpu, Upload,
        slice::{BufferSlice, MAX_UPDATE},
    },
    command_buffer::{BufferAccess, CommandBuffer, ImageAccess, ImageUse, image_state::ImageState},
    error::{Error, Result},
    image::{
        Image, Views,
        dim::Dim,
        expected_data_len,
        format::{ColorAspect, Format, bc_block},
        usage::IsTransferDst,
    },
    queue::{Kind, QueueKind, lock},
    state::Functions,
    vk,
};

/// A staged write waiting for a submission, or recorded into one (`flushed`, its timeline and
/// value) that hasn't reached the GPU yet.
struct Queued {
    staging: Buffer<u8, Upload>,
    target: Target,
    flushed: Option<(usize, u64)>,
}

enum Target {
    /// The address the staged bytes go to.
    Buffer(u64),
    Image(ImageCopy),
}

/// A copy of the staged texels into one whole mip.
struct ImageCopy {
    image: vk::Image,
    /// The image's tracked states, one for every mip or one per mip, whose address stays put
    /// while the image moves; valid until the image is dropped, which forgets the copy first.
    states: *const AtomicU64,
    states_len: usize,
    aspect: vk::ImageAspectFlags,
    mip_levels: u32,
    level: u32,
    extent: vk::Extent3D,
}

// SAFETY: `states` points at atomics, which every thread may access.
unsafe impl Send for ImageCopy {}

impl ImageCopy {
    fn states(&self) -> &[AtomicU64] {
        // SAFETY: see `states`; the queue's lock is held while it is used.
        unsafe { std::slice::from_raw_parts(self.states, self.states_len) }
    }

    /// The states and mips the copy accesses: all of them where one state covers every mip.
    fn accessed(&self) -> (&[AtomicU64], Range<u32>) {
        match self.states() {
            [state] => (std::slice::from_ref(state), (0..self.mip_levels).into()),
            states => {
                let level = self.level as usize;
                (
                    &states[level..level + 1],
                    (self.level..self.level + 1).into(),
                )
            }
        }
    }

    /// Whether a recording of `kind` may take it: the family that owns or receives it, or
    /// graphics for an image nobody owns yet, so an upload doesn't hand it to another family.
    fn recordable_on(&self, kind: Kind) -> bool {
        let (states, _) = self.accessed();
        states.iter().all(|state| {
            let state = ImageState::load(state);
            match state.owner {
                None => kind == Kind::Graphics,
                Some(owner) => owner == kind || state.released_to == Some(kind),
            }
        })
    }
}

static QUEUE: Mutex<Vec<Queued>> = Mutex::new(Vec::new());
/// Entries of `QUEUE`, so submissions and drops skip its lock while nothing is queued.
static QUEUED: AtomicUsize = AtomicUsize::new(0);

fn push(staging: Buffer<u8, Upload>, target: Target) {
    lock(&QUEUE).push(Queued {
        staging,
        target,
        flushed: None,
    });
    QUEUED.fetch_add(1, Ordering::Release);
}

/// Takes the entries `remove` picks out of the queue; their staging buffers are dropped once
/// the lock is released.
fn remove(mut remove: impl FnMut(&Queued) -> bool) {
    if QUEUED.load(Ordering::Acquire) == 0 {
        return;
    }
    let removed: Vec<Queued> = {
        let mut queue = lock(&QUEUE);
        let removed: Vec<_> = queue.extract_if(.., |q| remove(q)).collect();
        QUEUED.fetch_sub(removed.len(), Ordering::Release);
        removed
    };
    drop(removed);
}

/// Forgets the writes into a dropped buffer's `range`.
pub(crate) fn forget_buffer(range: Range<u64>) {
    remove(|q| matches!(q.target, Target::Buffer(address) if range.contains(&address)));
}

/// Forgets the uploads into a dropped image, before its tracked states go.
pub(crate) fn forget_image(image: vk::Image) {
    remove(|q| matches!(&q.target, Target::Image(copy) if copy.image == image));
}

/// Records the copies of every queued write the recording can take, in the order they were
/// queued, and marks them as recorded into its submission.
pub(crate) fn flush<K: QueueKind>(cmd: &mut CommandBuffer<'_, K>) {
    if QUEUED.load(Ordering::Acquire) == 0 {
        return;
    }
    let mut queue = lock(&QUEUE);
    let submission = (cmd.rec.timeline, cmd.rec.value);
    for queued in queue.iter_mut().filter(|q| q.flushed.is_none()) {
        let src = queued.staging.byte_range(..);
        match &queued.target {
            &Target::Buffer(address) => {
                let dst = BufferSlice::<u8>::new(address, src.size());
                cmd.copy_buffer(src, dst);
            }
            Target::Image(copy) => {
                if !copy.recordable_on(K::KIND) {
                    continue;
                }
                record_image_copy(cmd, src, copy);
            }
        }
        queued.flushed = Some(submission);
    }
}

fn record_image_copy<K: QueueKind>(
    cmd: &mut CommandBuffer<'_, K>,
    src: BufferSlice<'_, u8>,
    copy: &ImageCopy,
) {
    let (states, mip_range) = copy.accessed();
    let image = ImageUse {
        access: ImageAccess {
            stage: vk::PipelineStageFlags2::TRANSFER,
            access: vk::AccessFlags2::TRANSFER_WRITE,
            image: copy.image,
            layout: vk::ImageLayout::GENERAL,
            aspect: copy.aspect,
            mip_range,
        },
        states,
    };
    cmd.rec.flush_pending(
        std::iter::once(image),
        std::iter::once(src.access(
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_READ,
        )),
    );
    let regions = [vk::DeviceMemoryImageCopyKHR::default()
        .address_range(src.address_range())
        .address_flags(src.address_flags())
        .image_subresource(vk::ImageSubresourceLayers {
            aspect_mask: copy.aspect,
            mip_level: copy.level,
            base_array_layer: 0,
            layer_count: 1,
        })
        .image_layout(vk::ImageLayout::GENERAL)
        .image_offset(vk::Offset3D::default())
        .image_extent(copy.extent)];
    unsafe {
        Functions::device_address_commands().cmd_copy_memory_to_image(
            cmd.rec.handle,
            &vk::CopyDeviceMemoryImageInfoKHR::default()
                .image(copy.image)
                .regions(&regions),
        )
    };
}

/// After the submission of `value` on `timeline`: drops the staging buffers of the writes it
/// recorded once it `reached` the GPU (lifetime destroys them after it), else queues them again.
pub(crate) fn finish(timeline: usize, value: u64, reached: bool) {
    let flushed = |q: &Queued| q.flushed == Some((timeline, value));
    if reached {
        return remove(flushed);
    }
    if QUEUED.load(Ordering::Acquire) == 0 {
        return;
    }
    for queued in lock(&QUEUE).iter_mut().filter(|q| flushed(q)) {
        queued.flushed = None;
    }
}

/// A staging buffer holding `bytes`.
fn staged(bytes: &[u8]) -> Result<Buffer<u8, Upload>> {
    // Aligned to 4 bytes, which a transfer family's image copies need.
    let staging = Buffer::<u8, Upload>::raw(Upload, bytes.len() as u64, Some(4))?;
    let ptr = staging.mapping().expect("upload memory is mapped");
    // SAFETY: no submission has seen the new buffer.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len()) };
    Ok(staging)
}

impl<T: Copy + Pod> Buffer<T, Gpu> {
    /// A device-local buffer holding `data`: written right away where its memory is mapped
    /// (integrated GPUs), else staged like [`Self::write`]. Empty data is an error, like an
    /// empty buffer.
    #[validation_trace]
    pub fn from_slice(data: &[T]) -> Result<Self> {
        let buffer = Self::new(data.len())?;
        let coherent = buffer
            .allocation
            .memory_properties()
            .contains(vk::MemoryPropertyFlags::HOST_COHERENT);
        match buffer.mapping() {
            // SAFETY: no submission has seen the new buffer.
            Some(ptr) if coherent => unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len())
            },
            _ => buffer.write(0, data)?,
        }
        Ok(buffer)
    }

    /// Writes `data` at element `offset`, in submission order: it is copied into a staging
    /// buffer now, and the next submission of any timeline copies it into this one before its
    /// closure records anything, so earlier submissions read the old data and later ones the
    /// new, also while frames in flight read the buffer. Another timeline sees it only after a
    /// `Wait` on that submission. Panics unless `data` fits.
    #[validation_trace]
    pub fn write(&self, offset: usize, data: &[T]) -> Result<()> {
        assert!(
            offset
                .checked_add(data.len())
                .is_some_and(|end| end <= self.len()),
            "writing {} elements at {offset} into a buffer of {}",
            data.len(),
            self.len()
        );
        if data.is_empty() {
            return Ok(());
        }
        let staging = staged(bytemuck::cast_slice(data))?;
        push(
            staging,
            Target::Buffer(self.address + (offset * size_of::<T>()) as u64),
        );
        Ok(())
    }
}

impl<F: Format + ColorAspect, U: IsTransferDst, D: Dim, V: Views> Image<F, U, D, V> {
    /// Writes the whole mip `level` from `data`, tightly packed, in submission order like
    /// [`Buffer::write`]: the next submission of a timeline of the family that owns the image
    /// (graphics while nobody does) copies it. `copy_from` writes right away, with host image
    /// copies.
    #[validation_trace]
    pub fn upload(&self, level: u32, data: &[u8]) -> Result<()> {
        self.aspect_check(vk::ImageAspectFlags::COLOR);
        self.usage_check(vk::ImageUsageFlags::TRANSFER_DST);
        if level >= self.mip_levels() {
            return Err(Error::message(format!(
                "mip level {level} is out of range for an image with {} levels",
                self.mip_levels()
            )));
        }
        let extent = D::extent_3d(self.mip_extent(level));
        let format = self.format();
        let expected = expected_data_len(extent, format.texel_size(), bc_block(format.vk_format()));
        if expected != Some(data.len()) {
            return Err(Error::message(format!(
                "mip level {level} holds {expected:?} bytes but {} were given",
                data.len()
            )));
        }
        let staging = staged(data)?;
        let states = self.whole_view().states;
        push(
            staging,
            Target::Image(ImageCopy {
                image: self.whole_view().image,
                states: states.as_ptr(),
                states_len: states.len(),
                aspect: vk::ImageAspectFlags::COLOR,
                mip_levels: self.mip_levels(),
                level,
                extent,
            }),
        );
        Ok(())
    }
}

/// A frame slot's staging memory for `cmd.upload`: one linear allocator, reset when the slot's
/// last submission finished, which grows by replacing its buffer with a larger one (the old one
/// lives until the submissions that copy from it finished).
#[derive(Default)]
pub(crate) struct StagingRing {
    buffer: Option<Buffer<u8, Upload>>,
    used: u64,
}

/// The smallest buffer of a staging ring.
const MIN_RING: u64 = 1 << 20;

impl StagingRing {
    pub(crate) fn reset(&mut self) {
        self.used = 0;
    }

    /// `bytes` copied into the ring, at a 16-byte aligned address.
    fn stage(&mut self, bytes: &[u8]) -> Result<BufferSlice<'_, u8>> {
        let size = bytes.len() as u64;
        let start = self.used.next_multiple_of(16);
        let capacity = self.buffer.as_ref().map_or(0, |b| b.size());
        let start = if start + size <= capacity {
            start
        } else {
            let capacity = (capacity * 2).max(size.next_power_of_two()).max(MIN_RING);
            self.buffer = Some(Buffer::raw(Upload, capacity, Some(16))?);
            0
        };
        self.used = start + size;
        let buffer = self.buffer.as_ref().expect("allocated above");
        let ptr = buffer.mapping().expect("upload memory is mapped");
        // SAFETY: the slot's earlier submissions that read the ring finished, and this
        // recording's earlier uploads lie before `start`.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.add(start as usize), bytes.len())
        };
        Ok(buffer.byte_range(start as usize..(start + size) as usize))
    }
}

impl<K: QueueKind> CommandBuffer<'_, K> {
    /// Writes `data` to the start of `dst`, ordered like every other command: up to 64 KiB of
    /// whole 4-byte words at a 4-byte aligned address from the command buffer itself, more
    /// through the frame slot's staging memory. Fails only when that can't grow.
    #[validation_trace]
    pub fn upload<T: Copy + Pod>(&mut self, dst: BufferSlice<'_, T>, data: &[T]) -> Result<()> {
        assert!(
            data.len() <= dst.len(),
            "uploading {} elements into a slice of {}",
            data.len(),
            dst.len()
        );
        if data.is_empty() {
            return Ok(());
        }
        let bytes: &[u8] = bytemuck::cast_slice(data);
        let dst = dst.byte_range(..bytes.len());
        if bytes.len() <= MAX_UPDATE && bytes.len() % 4 == 0 && dst.address() % 4 == 0 {
            self.rec.flush_pending(
                std::iter::empty(),
                std::iter::once(BufferAccess {
                    stage: vk::PipelineStageFlags2::TRANSFER,
                    access: vk::AccessFlags2::TRANSFER_WRITE,
                    range: dst.get_range(),
                }),
            );
            unsafe {
                Functions::device_address_commands().cmd_update_memory(
                    self.rec.handle,
                    &dst.address_range(),
                    dst.address_flags(),
                    bytes,
                )
            };
            return Ok(());
        }
        let src = self.rec.staging.stage(bytes)?;
        let src = BufferSlice::<u8>::new(src.address(), src.size());
        self.copy_buffer(src, dst);
        Ok(())
    }
}

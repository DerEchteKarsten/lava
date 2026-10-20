//! Explicit uploads into memory the host can't map: an `Uploads` batch the application fills (`init`, `from_slice`, `write`, `image`) and records with `cmd.record_uploads` on the timeline and at the point it chooses, releasing the images to the batch's destination family; filled buffers written in place where their memory is mapped (`rebar`); lazily allocated staging chunks; and the per-frame-slot staging ring of in-recording uploads (`cmd.upload`)
use std::{
    marker::PhantomData,
    range::Range,
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};

use bytemuck::Pod;
use lava_macros::validation_trace;
use smallvec::SmallVec;

use crate::{
    buffer::{
        Buffer, Gpu, Location, USAGE, Upload,
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
    queue::{Graphics, Kind, QueueKind, lock},
    state::{Ctx, Functions},
    vk,
};

/// A staging chunk: `Upload` memory that staged writes are appended to.
type Chunk = Buffer<u8, Upload>;

/// The smallest staging chunk; a larger write gets a chunk of its own size.
pub(crate) const MIN_CHUNK: u64 = 1 << 20;
/// The alignment of staged bytes: a transfer family's image copies need 4, texel blocks 16.
const STAGING_ALIGN: u64 = 16;

/// A staged write of batch `batch`, from `range` of `chunk`, waiting for its batch to be
/// recorded, or recorded into a submission (`recorded`, its timeline and value) that hasn't
/// reached the GPU yet.
pub(crate) struct Entry<C> {
    pub(crate) batch: u64,
    pub(crate) chunk: Arc<C>,
    pub(crate) range: Range<u64>,
    pub(crate) target: Target,
    pub(crate) recorded: Option<(usize, u64)>,
}

pub(crate) enum Target {
    /// The address the staged bytes go to.
    Buffer(u64),
    Image(ImageCopy),
}

/// A copy of the staged texels into one whole mip.
pub(crate) struct ImageCopy {
    image: vk::Image,
    /// The image's tracked states, one for every mip or one per mip, whose address stays put
    /// while the image moves; valid until the image is dropped, which forgets the copy first.
    states: *const AtomicU64,
    states_len: usize,
    aspect: vk::ImageAspectFlags,
    /// The mips whose tracked state the copy moves: all of them where one state covers every
    /// mip, else `level` alone.
    mips: Range<u32>,
    level: u32,
    extent: vk::Extent3D,
}

// SAFETY: `states` points at atomics, which every thread may access.
unsafe impl Send for ImageCopy {}

impl ImageCopy {
    fn states(&self) -> &[AtomicU64] {
        // SAFETY: see `states`; the registry's lock is held while it is used.
        unsafe { std::slice::from_raw_parts(self.states, self.states_len) }
    }

    /// The family other than `kind` that owns a mip the copy accesses and hasn't released it
    /// to `kind`, if one does.
    fn foreign_owner(&self, kind: Kind) -> Option<Kind> {
        let states = self.states();
        let accessed = match states {
            [_] => states,
            _ => &states[self.mips.start as usize..self.mips.end as usize],
        };
        accessed.iter().find_map(|state| {
            let state = ImageState::load(state);
            let owner = state.owner?;
            (owner != kind && state.released_to != Some(kind)).then_some(owner)
        })
    }
}

/// Every live batch's staged writes, kept only so that dropping a target or a batch forgets
/// them and a failed submission returns them to their batch: nothing in it is recorded unless
/// its batch is (`record_uploads`).
pub(crate) struct Entries<C>(pub(crate) Vec<Entry<C>>);

impl<C> Entries<C> {
    /// The entries of `batch` no submission recorded, in the order they were staged, marked
    /// as recorded into `submission`.
    pub(crate) fn record(
        &mut self,
        batch: u64,
        submission: (usize, u64),
    ) -> impl Iterator<Item = &Entry<C>> {
        (self.0.iter_mut())
            .filter(move |e| e.batch == batch && e.recorded.is_none())
            .map(move |e| {
                e.recorded = Some(submission);
                &*e
            })
    }

    /// The number of entries of `batch` no submission recorded.
    pub(crate) fn unrecorded(&self, batch: u64) -> usize {
        (self.0.iter()).filter(|e| e.batch == batch && e.recorded.is_none()).count()
    }

    /// Takes the entries `remove` picks out.
    pub(crate) fn remove(&mut self, mut remove: impl FnMut(&Entry<C>) -> bool) -> Vec<Entry<C>> {
        self.0.extract_if(.., |e| remove(e)).collect()
    }

    /// After `submission`: takes the entries it recorded once it `reached` the GPU, else
    /// returns them to their batches, or takes them too where their batch was dropped.
    pub(crate) fn finish(&mut self, submission: (usize, u64), reached: bool) -> Vec<Entry<C>> {
        let recorded = |e: &Entry<C>| e.recorded == Some(submission);
        if reached {
            return self.remove(recorded);
        }
        let removed = self.remove(|e| recorded(e) && e.batch == DROPPED);
        for entry in self.0.iter_mut().filter(|e| recorded(e)) {
            entry.recorded = None;
        }
        removed
    }

    /// Takes the entries of a dropped `batch` no submission recorded, and marks the recorded
    /// ones as left by it, so a submission that doesn't reach the GPU drops them.
    pub(crate) fn drop_batch(&mut self, batch: u64) -> Vec<Entry<C>> {
        let removed = self.remove(|e| e.batch == batch && e.recorded.is_none());
        for entry in self.0.iter_mut().filter(|e| e.batch == batch) {
            entry.batch = DROPPED;
        }
        removed
    }
}

static ENTRIES: Mutex<Entries<Chunk>> = Mutex::new(Entries(Vec::new()));
/// The number of `ENTRIES`, so submissions and drops skip its lock while there are none.
static COUNT: AtomicUsize = AtomicUsize::new(0);
/// The id of the next batch.
static NEXT_BATCH: AtomicU64 = AtomicU64::new(0);
/// The batch of recorded entries whose batch was dropped.
pub(crate) const DROPPED: u64 = u64::MAX;

fn push(entry: Entry<Chunk>) {
    lock(&ENTRIES).0.push(entry);
    COUNT.fetch_add(1, Ordering::Release);
}

/// Takes the entries `remove` picks out; their chunks are dropped once the lock is released,
/// since dropping a buffer takes it.
fn remove(remove: impl FnMut(&Entry<Chunk>) -> bool) {
    if COUNT.load(Ordering::Acquire) == 0 {
        return;
    }
    let removed = lock(&ENTRIES).remove(remove);
    COUNT.fetch_sub(removed.len(), Ordering::Release);
    drop(removed);
}

/// Forgets the writes into a dropped buffer's `range`.
pub(crate) fn forget_buffer(range: Range<u64>) {
    remove(|e| matches!(e.target, Target::Buffer(address) if range.contains(&address)));
}

/// Forgets the writes into a dropped image, before its tracked states go.
pub(crate) fn forget_image(image: vk::Image) {
    remove(|e| matches!(&e.target, Target::Image(copy) if copy.image == image));
}

/// After the submission of `value` on `timeline`: drops the entries it recorded once it
/// `reached` the GPU (lifetime destroys their chunks after it), else returns them to their
/// batches, whose next `record_uploads` takes them again.
pub(crate) fn finish(timeline: usize, value: u64, reached: bool) {
    if COUNT.load(Ordering::Acquire) == 0 {
        return;
    }
    let removed = lock(&ENTRIES).finish((timeline, value), reached);
    COUNT.fetch_sub(removed.len(), Ordering::Release);
    drop(removed);
}

/// Where a batch stages its next write: its newest chunk, which lives while an entry staged in
/// it does (an idle batch holds no staging memory), and how much of it is used.
pub(crate) struct Cursor<C> {
    chunk: Weak<C>,
    capacity: u64,
    used: u64,
}

impl<C> Default for Cursor<C> {
    fn default() -> Self {
        Self {
            chunk: Weak::new(),
            capacity: 0,
            used: 0,
        }
    }
}

impl<C> Cursor<C> {
    /// The chunk and offset of `size` staged bytes: appended to the newest chunk where they
    /// fit, else at the start of a new one, `new(capacity)`, of `MIN_CHUNK` bytes. A larger
    /// write gets a chunk of exactly its size, and later ones still append to the newest.
    pub(crate) fn place(
        &mut self,
        size: u64,
        new: impl FnOnce(u64) -> Result<C>,
    ) -> Result<(Arc<C>, u64)> {
        let start = self.used.next_multiple_of(STAGING_ALIGN);
        if let Some(chunk) = self.chunk.upgrade()
            && start + size <= self.capacity
        {
            self.used = start + size;
            return Ok((chunk, start));
        }
        if size > MIN_CHUNK {
            return Ok((Arc::new(new(size)?), 0));
        }
        let chunk = Arc::new(new(MIN_CHUNK)?);
        (self.chunk, self.capacity, self.used) = (Arc::downgrade(&chunk), MIN_CHUNK, size);
        Ok((chunk, 0))
    }
}

/// A batch of staged writes into buffers and images, which its owner records with
/// [`CommandBuffer::record_uploads`] on the timeline and at the point of the recording it
/// chooses: lava records nothing it isn't asked to. `D` is the family that uses the uploaded
/// data, which the images it copies into are released to where it is recorded on another
/// family:
/// ```no_run
/// # fn f(
/// #     transfer: &mut lava::queue::Timeline<lava::queue::Transfer>,
/// #     vertices: &[[f32; 3]],
/// # ) -> lava::Result<()> {
/// use lava::buffer::Uploads;
/// let mut uploads = Uploads::new();
/// let mesh = uploads.from_slice(vertices)?;
/// let done = transfer.begin()?.submit(|cmd| cmd.record_uploads(&uploads))?;
/// // A graphics submission that reads `mesh` waits for `done`.
/// # Ok(()) }
/// ```
///
/// Staging memory comes in chunks of at least 1 MiB, allocated at the first staged write and
/// freed once the submissions that copy from them completed. A batch can be filled and
/// recorded again and again; dropping it discards what no submission recorded, and dropping a
/// target forgets its writes.
pub struct Uploads<D: QueueKind = Graphics> {
    id: u64,
    staging: Cursor<Chunk>,
    _destination: PhantomData<fn() -> D>,
}

impl<D: QueueKind> std::fmt::Debug for Uploads<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (f.debug_struct("Uploads"))
            .field("destination", &D::KIND)
            .field("len", &self.len())
            .finish()
    }
}

impl Uploads<Graphics> {
    /// An empty batch for the graphics family; it allocates nothing.
    pub fn new() -> Self {
        Self::with_destination()
    }
}

impl Default for Uploads<Graphics> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D: QueueKind> Drop for Uploads<D> {
    fn drop(&mut self) {
        if COUNT.load(Ordering::Acquire) == 0 {
            return;
        }
        // Dropped once the lock is released, like `remove`'s.
        let removed = lock(&ENTRIES).drop_batch(self.id);
        COUNT.fetch_sub(removed.len(), Ordering::Release);
        drop(removed);
    }
}

impl<D: QueueKind> Uploads<D> {
    fn with_destination() -> Self {
        Self {
            id: NEXT_BATCH.fetch_add(1, Ordering::Relaxed),
            staging: Cursor::default(),
            _destination: PhantomData,
        }
    }

    /// An empty batch for the family of kind `D`; `MissingFeature` where lava has none.
    pub fn to() -> Result<Self> {
        D::check()?;
        Ok(Self::with_destination())
    }

    /// The number of staged writes no submission recorded.
    pub fn len(&self) -> usize {
        if COUNT.load(Ordering::Acquire) == 0 {
            return 0;
        }
        lock(&ENTRIES).unrecorded(self.id)
    }

    /// Whether a `record_uploads` of the batch would record nothing.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Stages `size` bytes written by `fill` into its mapping, for `target`.
    fn stage<T: Pod>(
        &mut self,
        len: usize,
        target: Target,
        fill: impl FnOnce(&mut [T]),
    ) -> Result<()> {
        let size = (len * size_of::<T>()) as u64;
        let (chunk, start) = self
            .staging
            .place(size, |capacity| Buffer::raw(Upload, capacity, Some(STAGING_ALIGN as u32)))?;
        let ptr = chunk.mapping().expect("upload memory is mapped");
        // SAFETY: the chunk holds `start + size` bytes; no submission has seen this range,
        // which is never staged into again.
        let data = unsafe {
            let ptr = ptr.add(start as usize).cast::<T>();
            assert!(ptr.is_aligned(), "staged {}s at a misaligned {ptr:?}", std::any::type_name::<T>());
            std::slice::from_raw_parts_mut(ptr, len)
        };
        fill(data);
        push(Entry {
            batch: self.id,
            chunk,
            range: (start..start + size).into(),
            target,
            recorded: None,
        });
        Ok(())
    }

    /// A device-local buffer of `len` elements that `fill` writes, given the buffer's address
    /// (for data that embeds addresses into itself). With `rebar` it is allocated from the
    /// device-local memory the host can map (the BAR heap is all of VRAM there, and like
    /// `Shared` it doesn't spill to system memory; plain device-local memory is the fallback).
    /// Where its memory is mapped (`rebar`, integrated GPUs) `fill` writes it in place and the
    /// batch gets nothing to record; elsewhere `fill` writes staging memory and the batch one
    /// copy. Either way the buffer is a [`Gpu`] one without host access.
    #[validation_trace]
    pub fn init<T: Copy + Pod>(
        &mut self,
        len: usize,
        fill: impl FnOnce(u64, &mut [T]),
    ) -> Result<Buffer<T, Gpu>> {
        let bytes = len.checked_mul(size_of::<T>()).ok_or_else(|| {
            Error::message(format!("{len} elements of {} bytes overflow", size_of::<T>()))
        })? as u64;
        let align = Some(align_of::<T>() as u32);
        let buffer = match Ctx::features().rebar {
            true => Buffer::with_memory_types(USAGE, Gpu, bytes, align, Location::Shared)
                .or_else(|_| Buffer::new(len))?,
            false => Buffer::new(len)?,
        };
        let coherent = (buffer.allocation.memory_properties())
            .contains(vk::MemoryPropertyFlags::HOST_COHERENT);
        match buffer.mapping() {
            Some(ptr) if coherent => {
                // SAFETY: no submission has seen the new buffer, whose mapping holds `len`.
                fill(buffer.address, unsafe { std::slice::from_raw_parts_mut(ptr, len) })
            }
            _ => self.stage(len, Target::Buffer(buffer.address), |data| {
                fill(buffer.address, data)
            })?,
        }
        Ok(buffer)
    }

    /// A device-local buffer holding `data`, made like [`Self::init`]. Empty data is an error,
    /// like an empty buffer.
    #[validation_trace]
    pub fn from_slice<T: Copy + Pod>(&mut self, data: &[T]) -> Result<Buffer<T, Gpu>> {
        self.init(data.len(), |_, out| out.copy_from_slice(data))
    }

    /// Stages `data` for element `offset` of `buffer`, any buffer: its copy is ordered like
    /// every command where the batch is recorded, so earlier commands read the old data and
    /// later ones the new, also while frames in flight read the buffer. It is staged on every
    /// device: to edit host-visible memory in place, use a `Shared` buffer's `host_mut`, one
    /// range per frame slot. Panics unless `data` fits.
    #[validation_trace]
    pub fn write<T: Copy + Pod, M: super::Memory>(
        &mut self,
        buffer: &Buffer<T, M>,
        offset: usize,
        data: &[T],
    ) -> Result<()> {
        assert!(
            offset.checked_add(data.len()).is_some_and(|end| end <= buffer.len()),
            "writing {} elements at {offset} into a buffer of {}",
            data.len(),
            buffer.len()
        );
        if data.is_empty() {
            return Ok(());
        }
        let address = buffer.address + (offset * size_of::<T>()) as u64;
        self.stage(data.len(), Target::Buffer(address), |out| out.copy_from_slice(data))
    }

    /// Stages the whole mip `level` of `image` from `data`, tightly packed. Recorded on a
    /// family other than `D`, the batch releases the image to `D` (`record_uploads`).
    /// `Image::copy_from` writes right away instead, with host image copies.
    #[validation_trace]
    pub fn image<F: Format + ColorAspect, U: IsTransferDst, Dm: Dim, V: Views>(
        &mut self,
        image: &Image<F, U, Dm, V>,
        level: u32,
        data: &[u8],
    ) -> Result<()> {
        let copy = image_copy(image, level, data.len())?;
        self.stage(data.len(), Target::Image(copy), |out| out.copy_from_slice(data))
    }
}

/// The copy of `len` bytes into the whole mip `level` of `image`, checked: a colour image with
/// transfer destination usage, a mip it has, and the bytes the mip holds.
fn image_copy<F: Format + ColorAspect, U: IsTransferDst, D: Dim, V: Views>(
    image: &Image<F, U, D, V>,
    level: u32,
    len: usize,
) -> Result<ImageCopy> {
    image.aspect_check(vk::ImageAspectFlags::COLOR);
    image.usage_check(vk::ImageUsageFlags::TRANSFER_DST);
    if level >= image.mip_levels() {
        return Err(Error::message(format!(
            "mip level {level} is out of range for an image with {} levels",
            image.mip_levels()
        )));
    }
    let extent = D::extent_3d(image.mip_extent(level));
    let format = image.format();
    let expected = expected_data_len(extent, format.texel_size(), bc_block(format.vk_format()));
    if expected != Some(len) {
        return Err(Error::message(format!(
            "mip level {level} holds {expected:?} bytes but {len} were given"
        )));
    }
    let states = image.whole_view().states;
    let mips = match states.len() {
        1 => (0..image.mip_levels()).into(),
        _ => (level..level + 1).into(),
    };
    Ok(ImageCopy {
        image: image.whole_view().image,
        states: states.as_ptr(),
        states_len: states.len(),
        aspect: vk::ImageAspectFlags::COLOR,
        mips,
        level,
        extent,
    })
}

/// An image a recording of a batch copied into, and the mips it copied (bit `i` for mip `i`),
/// which it releases to the batch's destination.
struct Copied {
    image: vk::Image,
    states: *const AtomicU64,
    states_len: usize,
    aspect: vk::ImageAspectFlags,
    mips: u64,
}

impl<K: QueueKind> CommandBuffer<'_, K> {
    /// Records the copies of the writes of `uploads` no submission recorded yet, here, in the
    /// order they were staged; barrier tracking orders them against the commands around them.
    /// Where `K` isn't the batch's destination `D`, every image it copied into is released to
    /// `D` after the last copy, and `D`'s first access acquires it: the receiving submission has
    /// to wait for this one (`Submission::wait_for`), which lava doesn't check, and this
    /// recording can't use those images after it, so record the batch after its own uses.
    /// Panics, before anything is recorded, if a copy goes into an image another family owns:
    /// record such a batch on that family. If the submission doesn't reach the GPU, its writes
    /// go back to the batch.
    #[validation_trace]
    pub fn record_uploads<D: QueueKind>(&mut self, uploads: &Uploads<D>) {
        if COUNT.load(Ordering::Acquire) == 0 {
            return;
        }
        let mut entries = lock(&ENTRIES);
        let foreign = (entries.0.iter())
            .filter(|e| e.batch == uploads.id && e.recorded.is_none())
            .find_map(|e| match &e.target {
                Target::Image(copy) => copy.foreign_owner(K::KIND),
                Target::Buffer(_) => None,
            });
        if let Some(owner) = foreign {
            drop(entries);
            panic!(
                "`record_uploads` on the {kind:?} family copies into an image the {owner:?}                  family owns: record the batch on a {owner:?} timeline",
                kind = K::KIND
            );
        }
        let mut copied = SmallVec::<[Copied; 4]>::new();
        let submission = (self.rec.timeline, self.rec.value);
        for entry in entries.record(uploads.id, submission) {
            let src = BufferSlice::<u8>::new(
                entry.chunk.address() + entry.range.start,
                entry.range.end - entry.range.start,
            );
            match &entry.target {
                &Target::Buffer(address) => {
                    self.copy_buffer(src, BufferSlice::<u8>::new(address, src.size()))
                }
                Target::Image(copy) => {
                    record_image_copy(self, src, copy);
                    let mips = (copy.mips.start..copy.mips.end).fold(0, |m, i| m | 1 << i);
                    match copied.iter_mut().find(|c| c.image == copy.image) {
                        Some(c) => c.mips |= mips,
                        None => copied.push(Copied {
                            image: copy.image,
                            states: copy.states,
                            states_len: copy.states_len,
                            aspect: copy.aspect,
                            mips,
                        }),
                    }
                }
            }
        }
        if D::KIND != K::KIND {
            for c in &copied {
                // SAFETY: as `ImageCopy::states`; the lock is still held.
                let states = unsafe { std::slice::from_raw_parts(c.states, c.states_len) };
                // One release per run of copied mips.
                let mut mip = 0;
                while mip < 64 {
                    if c.mips >> mip & 1 == 0 {
                        mip += 1;
                        continue;
                    }
                    let end = mip + (c.mips >> mip).trailing_ones();
                    (self.rec).release(c.image, states, (mip..end).into(), c.aspect, D::KIND);
                    mip = end;
                }
            }
        }
        drop(entries);
    }
}

fn record_image_copy<K: QueueKind>(
    cmd: &mut CommandBuffer<'_, K>,
    src: BufferSlice<'_, u8>,
    copy: &ImageCopy,
) {
    let image = ImageUse {
        access: ImageAccess {
            stage: vk::PipelineStageFlags2::TRANSFER,
            access: vk::AccessFlags2::TRANSFER_WRITE,
            image: copy.image,
            layout: vk::ImageLayout::GENERAL,
            aspect: copy.aspect,
            mip_range: copy.mips,
        },
        states: copy.states(),
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

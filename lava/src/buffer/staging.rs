//! Uploads into memory the host can't map through a staging ring per timeline (`Staging`, sized at `Timeline::new`, none on `rebar` devices, where every `Gpu` buffer is mapped): `Uploader` handles cloned from the timeline fill it from any thread (`init`, `from_slice`, `write`, `image`, in place where the target is mapped), `cmd.drain_uploads` records every published copy at the point of the recording the application chooses and releases the images to their destination family (each copy holds its target, and one whose target only the ring still holds is skipped), and the timeline's values free the drained ring space
use std::{
    collections::VecDeque,
    fmt,
    ops::Range,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering::SeqCst},
    },
    time::{Duration, Instant},
};

use bytemuck::{AnyBitPattern, NoUninit};
use lava_macros::validation_trace;
use smallvec::SmallVec;
use std::mem::MaybeUninit;

use crate::{
    buffer::{Buffer, BufferInternal, Gpu, Memory, Upload, slice::BufferSlice},
    command_buffer::{CommandBuffer, ImageAccess, ImageUse, image_state::ImageState},
    error::{Error, Result},
    image::{
        Image, ImageInternal, Views,
        dim::Dim,
        format::{ColorAspect, Format},
        slice::{copy_block_of, rows_len},
        usage::IsTransferDst,
    },
    object::Strong,
    queue::{Kind, QueueKind, lock, timeline},
    reference::{KeepAlive, Reference, ReferenceInfo, ReferencingData},
    state::Functions,
    vk,
};

/// The alignment of staged bytes: a transfer family's image copies need 4, texel blocks 16.
pub(crate) const STAGING_ALIGN: u64 = 16;

const POSITION_BITS: u32 = 48;
const POSITION_MASK: u64 = (1 << POSITION_BITS) - 1;
/// One writer counted in, in the packed head.
const WRITER: u64 = 1 << POSITION_BITS;

/// The positions of a staging ring, without its memory (unit-tested on their own). Positions
/// are monotonic: a position's byte offset is the position modulo the size.
pub(crate) struct Ring {
    size: u64,
    /// Bits 0..48: the head position; bits 48..64: the writers counted in (reserved, not yet
    /// published).
    head: AtomicU64,
    /// The position below which nothing is in use.
    tail: AtomicU64,
    /// The start of the reservation that counted the first writer of the current run in: no
    /// counted-in writer's region lies below it. A stale value from an earlier run is smaller,
    /// so it stays a bound.
    floor: AtomicU64,
}

/// A reserved region of a ring.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Reservation {
    /// The reserved positions, the padding to the ring's end included where the data wrapped.
    pub(crate) range: Range<u64>,
    /// The byte offset of the data in the ring.
    pub(crate) offset: u64,
}

impl Ring {
    pub(crate) fn new(size: u64) -> Self {
        Self {
            size,
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
            floor: AtomicU64::new(0),
        }
    }

    pub(crate) fn size(&self) -> u64 {
        self.size
    }

    /// The bytes in use: reserved, published or in flight.
    pub(crate) fn used(&self) -> u64 {
        (self.head.load(SeqCst) & POSITION_MASK) - self.tail.load(SeqCst)
    }

    /// Reserves `len` bytes (a multiple of `STAGING_ALIGN`, at most the size) at the head and
    /// counts the writer in; `None` while they don't fit before the tail.
    pub(crate) fn reserve(&self, len: u64) -> Option<Reservation> {
        debug_assert!(len.is_multiple_of(STAGING_ALIGN) && len <= self.size);
        let mut packed = self.head.load(SeqCst);
        loop {
            let (pos, writers) = (packed & POSITION_MASK, packed >> POSITION_BITS);
            let offset = pos % self.size;
            let start = match offset + len > self.size {
                true => pos + (self.size - offset),
                false => pos,
            };
            let end = start + len;
            if end - self.tail.load(SeqCst) > self.size {
                return None;
            }
            assert!(
                writers < 0xFFFF && end <= POSITION_MASK,
                "the staging ring's counters overflowed"
            );
            let new = end | (writers + 1) << POSITION_BITS;
            match self.head.compare_exchange_weak(packed, new, SeqCst, SeqCst) {
                Ok(_) => {
                    if writers == 0 {
                        self.floor.store(pos, SeqCst);
                    }
                    return Some(Reservation {
                        range: pos..end,
                        offset: start % self.size,
                    });
                }
                Err(current) => packed = current,
            }
        }
    }

    /// Counts a writer out, once its copy is published (or abandoned: nothing reads an
    /// unpublished region, so the tail may pass it).
    pub(crate) fn count_out(&self) {
        self.head.fetch_sub(WRITER, SeqCst);
    }

    /// Moves the tail up to the lowest start among `in_use` (the published and in-flight
    /// regions), or to the head when there are none; never past a writer still counted in.
    pub(crate) fn reclaim(&self, in_use: impl Iterator<Item = u64>) {
        let packed = self.head.load(SeqCst);
        let (pos, writers) = (packed & POSITION_MASK, packed >> POSITION_BITS);
        let floor = self.floor.load(SeqCst);
        let mut tail = in_use.min().unwrap_or(pos);
        if writers > 0 {
            tail = tail.min(floor);
        }
        self.tail.fetch_max(tail, SeqCst);
    }
}

/// A copy of the staged texels into one whole mip.
pub(crate) struct ImageCopy {
    image: Arc<ImageInternal>,
    aspect: vk::ImageAspectFlags,
    /// The mips whose tracked state the copy moves: all of them where one state covers every
    /// mip, else `level` alone.
    mips: Range<u32>,
    level: u32,
    extent: vk::Extent3D,
    /// Every layer of the image, the data one layer after another.
    layers: u32,
}

impl ImageCopy {
    /// The family other than `kind` that owns a mip the copy accesses and hasn't released it
    /// to `kind`, if one does.
    fn foreign_owner(&self, kind: Kind) -> Option<Kind> {
        let states = &*self.image.states;
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

/// What a staged write goes to, kept alive by it until its copy completed.
enum Target {
    /// The buffer and the address in it the staged bytes go to.
    Buffer(Arc<BufferInternal>, u64),
    Image(ImageCopy),
}

impl Target {
    /// Whether only the ring holds it: nothing reads it any more, so its copy is skipped.
    fn orphaned(&self) -> bool {
        match self {
            Target::Buffer(buffer, _) => Arc::strong_count(buffer) == 1,
            Target::Image(copy) => Arc::strong_count(&copy.image) == 1,
        }
    }
}

/// A published write: its reservation, where its data lies in the ring, and where it goes.
pub(crate) struct Staged {
    reserved: Range<u64>,
    offset: u64,
    size: u64,
    target: Target,
    /// The family that uses the data, which its image is released to when it is drained on
    /// another.
    destination: Kind,
    /// What the elements it overwrites referenced that nothing references any more: alive
    /// until the copy completed, since submissions before it read the old elements.
    #[allow(dead_code, reason = "held until the copy completed")]
    released: Vec<Strong>,
}

/// A timeline's staging ring: `Upload` memory that `Uploader`s fill and `drain_uploads`
/// copies from.
pub(crate) struct Staging {
    buffer: Buffer<u8, Upload>,
    ring: Ring,
    timeline: usize,
    /// Set when the timeline drops: writes fail from then on.
    closed: AtomicBool,
    /// Published copies no recording drained yet, in publication order. Locked before
    /// `in_flight`.
    pending: Mutex<Vec<Staged>>,
    /// Drained copies with the value of the submission that recorded them, in that order; their
    /// regions are free once it completed.
    in_flight: Mutex<VecDeque<(Staged, u64)>>,
}

impl Staging {
    /// A ring of `size` bytes (rounded up to whole `STAGING_ALIGN`s) for timeline `timeline`.
    pub(crate) fn new(timeline: usize, size: u64) -> Result<Arc<Self>> {
        let size = size.next_multiple_of(STAGING_ALIGN);
        let buffer = Buffer::raw(Upload, size, Some(STAGING_ALIGN))?;
        Ok(Arc::new(Self {
            buffer,
            ring: Ring::new(size),
            timeline,
            closed: AtomicBool::new(false),
            pending: Mutex::new(Vec::new()),
            in_flight: Mutex::new(VecDeque::new()),
        }))
    }

    /// Closes the ring when its timeline drops: no write reaches it from then on.
    pub(crate) fn close(&self) {
        self.closed.store(true, SeqCst);
    }

    /// Frees the regions of the submissions that completed (`done`, the timeline's completed
    /// value) and everything else no longer in use.
    fn reclaim(&self, done: u64) {
        let pending = lock(&self.pending);
        let mut in_flight = lock(&self.in_flight);
        in_flight.retain(|(_, value)| *value > done);
        let starts = (pending.iter().map(|s| s.reserved.start))
            .chain(in_flight.iter().map(|(s, _)| s.reserved.start));
        self.ring.reclaim(starts);
    }

    /// A reservation of `size` bytes, counted in; waits up to `timeout` for space.
    fn reserve(&self, size: u64, timeout: Duration) -> Result<Reservation> {
        let len = size.next_multiple_of(STAGING_ALIGN);
        let full = || Error::StagingFull {
            requested: size,
            capacity: self.ring.size(),
        };
        if len > self.ring.size() {
            return Err(full());
        }
        let deadline = Instant::now().checked_add(timeout);
        loop {
            if self.closed.load(SeqCst) {
                return Err(Error::other("the timeline of this uploader was dropped"));
            }
            if let Some(reservation) = self.ring.reserve(len) {
                return Ok(reservation);
            }
            self.reclaim(timeline::poll(self.timeline)?);
            if let Some(reservation) = self.ring.reserve(len) {
                return Ok(reservation);
            }
            let remaining = match deadline {
                None => Duration::MAX,
                Some(deadline) => match deadline.checked_duration_since(Instant::now()) {
                    Some(remaining) if !remaining.is_zero() => remaining,
                    _ => return Err(full()),
                },
            };
            // Space comes from the oldest submission in flight, else from a drain.
            let oldest = (lock(&self.in_flight).front())
                .map(|(_, value)| *value)
                .filter(|&value| value > timeline::completed(self.timeline));
            match oldest {
                Some(value) => {
                    timeline::wait_value_for(self.timeline, value, remaining)?;
                }
                None => std::thread::sleep(remaining.min(Duration::from_millis(1))),
            }
        }
    }

    fn publish(&self, staged: Staged) {
        lock(&self.pending).push(staged);
    }

    /// After the submission of `value` didn't reach the GPU: its copies go back to the front of
    /// `pending`, their space still reserved, so the next drain records them.
    pub(crate) fn requeue(&self, value: u64) {
        let mut pending = lock(&self.pending);
        let mut in_flight = lock(&self.in_flight);
        if !in_flight.iter().any(|(_, v)| *v == value) {
            return;
        }
        let (back, keep): (Vec<_>, Vec<_>) = in_flight.drain(..).partition(|(_, v)| *v == value);
        in_flight.extend(keep);
        let mut back: Vec<Staged> = back.into_iter().map(|(staged, _)| staged).collect();
        back.append(&mut pending);
        *pending = back;
    }
}

/// Counts a writer out when dropped, also when its fill panicked.
struct CountOut<'a>(&'a Ring);

impl Drop for CountOut<'_> {
    fn drop(&mut self) {
        self.0.count_out();
    }
}

/// What a `drain_uploads` recorded, and how full the ring is after it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UploadStats {
    /// The copies it recorded.
    pub copies: usize,
    /// The bytes they copy.
    pub bytes: u64,
    /// The ring bytes in use after it: reserved, published or in flight.
    pub used: u64,
    /// The ring's size; 0 without a ring.
    pub capacity: u64,
}

/// A handle on a timeline's staging ring that writes buffers and images from any thread
/// (`Timeline::uploader`): `Clone + Send + Sync`, so one can go to every loader. Its writes
/// are copied where that timeline's recording calls [`CommandBuffer::drain_uploads`]:
/// ```no_run
/// # fn f(
/// #     transfer: &mut lava::queue::Timeline<lava::queue::Transfer>,
/// #     vertices: &[[f32; 3]],
/// # ) -> lava::Result<()> {
/// let uploader = transfer.uploader_for::<lava::queue::Graphics>()?;
/// let mesh = uploader.from_slice(vertices)?;
/// let done = transfer.begin()?.submit(|cmd| {
///     cmd.drain_uploads();
/// })?;
/// // A graphics submission that reads `mesh` waits for `done`.
/// # Ok(()) }
/// ```
/// Without a ring (`rebar` devices, where every `Gpu` buffer is mapped, or a timeline made
/// with a staging size of 0) buffers are written in place and images can't be uploaded. A
/// write waits up to its timeout for ring space, then fails with `Error::StagingFull`.
#[derive(Clone)]
pub struct Uploader {
    staging: Option<Arc<Staging>>,
    destination: Kind,
    timeout: Duration,
    /// Counts the uploader, which uploads into lava's state without another object.
    _live: crate::global::Live,
}

impl fmt::Debug for Uploader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (f.debug_struct("Uploader"))
            .field("destination", &self.destination)
            .field("capacity", &self.capacity())
            .field("timeout", &self.timeout)
            .finish()
    }
}

impl Uploader {
    pub(crate) fn new(staging: Option<Arc<Staging>>, destination: Kind) -> Self {
        Self {
            staging,
            destination,
            timeout: Duration::from_secs(1),
            _live: crate::global::Live::new(),
        }
    }

    /// The same uploader waiting up to `timeout` for ring space (1 s by default;
    /// `Duration::ZERO` fails right away).
    pub fn with_timeout(self, timeout: Duration) -> Self {
        Self { timeout, ..self }
    }

    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// The family that uses the uploaded data.
    pub fn destination(&self) -> Kind {
        self.destination
    }

    /// Whether its writes go through a staging ring; without one buffers are written in place
    /// and images can't be uploaded.
    pub fn stages(&self) -> bool {
        self.staging.is_some()
    }

    /// The ring's size in bytes; 0 without one.
    pub fn capacity(&self) -> u64 {
        self.staging.as_ref().map_or(0, |s| s.ring.size())
    }

    /// The copies no drain recorded yet, of every uploader of the ring.
    pub fn pending(&self) -> usize {
        (self.staging.as_ref()).map_or(0, |s| lock(&s.pending).len())
    }

    fn ring(&self) -> Result<&Arc<Staging>> {
        self.staging.as_ref().ok_or_else(|| {
            Error::other(
                "this uploader's timeline has no staging ring: on a `rebar` device images are \
                 uploaded with `Image::copy_from` (`HostCopy` usage); elsewhere give \
                 `Timeline::new` a staging size",
            )
        })
    }

    /// Stages `len` `T`s written by `fill`, for `target`.
    /// Stages `len` `T`s that `fill` writes (returning what the staged copy keeps alive) for
    /// `target`.
    fn stage<T: Copy>(
        &self,
        len: usize,
        target: Target,
        fill: impl FnOnce(&mut [MaybeUninit<T>]) -> Vec<Strong>,
    ) -> Result<()> {
        let staging = self.ring()?;
        let size = (len * size_of::<T>()) as u64;
        let reservation = staging.reserve(size, self.timeout)?;
        let counted = CountOut(&staging.ring);
        let ptr = staging.buffer.mapping().expect("upload memory is mapped");
        // SAFETY: the reserved region is the writer's alone until it is published and drained.
        let data = unsafe {
            let ptr = ptr
                .add(reservation.offset as usize)
                .cast::<MaybeUninit<T>>();
            assert!(
                ptr.is_aligned(),
                "staged {}s at a misaligned {ptr:?}",
                std::any::type_name::<T>()
            );
            std::slice::from_raw_parts_mut(ptr, len)
        };
        let released = fill(data);
        staging.publish(Staged {
            reserved: reservation.range,
            offset: reservation.offset,
            size,
            target,
            destination: self.destination,
            released,
        });
        drop(counted);
        Ok(())
    }

    /// A device-local buffer of `len` elements that `fill` writes, given the buffer's address
    /// (for data that embeds addresses into itself). Where its memory is mapped (`rebar`
    /// devices, integrated GPUs) `fill` writes it in place and nothing is left to drain;
    /// elsewhere `fill` writes the ring and the next drain copies it. Either way the buffer is
    /// a [`Gpu`] one without host access. `fill` sees what the memory held before, so the
    /// elements are `AnyBitPattern`; [`Self::from_slice`] takes any `NoUninit` ones.
    #[validation_trace]
    pub fn init<T: NoUninit + AnyBitPattern>(
        &self,
        len: usize,
        fill: impl FnOnce(u64, &mut [T]),
    ) -> Result<Buffer<T, Gpu>> {
        self.init_uninit(len, |address, data| {
            // SAFETY: any bytes are a valid `T`, and mapped memory holds bytes.
            fill(address, unsafe {
                &mut *(data as *mut [MaybeUninit<T>] as *mut [T])
            })
        })
    }

    fn init_uninit<T: NoUninit>(
        &self,
        len: usize,
        fill: impl FnOnce(u64, &mut [MaybeUninit<T>]),
    ) -> Result<Buffer<T, Gpu>> {
        let buffer = Buffer::<T, Gpu>::new(len)?;
        match buffer.coherent_mapping() {
            // SAFETY: no submission has seen the new buffer, whose mapping holds `len`.
            Some(ptr) => fill(buffer.address(), unsafe {
                std::slice::from_raw_parts_mut(ptr.cast::<MaybeUninit<T>>(), len)
            }),
            None => {
                let target = Target::Buffer(buffer.internal.clone(), buffer.address());
                self.stage(len, target, |data| {
                    fill(buffer.address(), data);
                    Vec::new()
                })?
            }
        }
        Ok(buffer)
    }

    /// A device-local buffer holding `data`, made like [`Self::init`]. Empty data is an error,
    /// like an empty buffer.
    #[validation_trace]
    pub fn from_slice<T: NoUninit>(&self, data: &[T]) -> Result<Buffer<T, Gpu>> {
        self.init_uninit(data.len(), |_, out| write_slice(out, data))
    }

    /// Writes `data` at element `offset` of `buffer`, any buffer. With a ring it is staged, and
    /// its copy is ordered like every command where the ring is drained: earlier commands read
    /// the old data, later ones the new, also while frames in flight read the buffer. Without
    /// a ring it is written into the buffer's mapping right away, which a frame in flight may
    /// be reading: double-buffer such data. A buffer that isn't mapped is an error then.
    /// Panics unless `data` fits.
    #[validation_trace]
    pub fn write<T: NoUninit, M: Memory>(
        &self,
        buffer: &Buffer<T, M>,
        offset: usize,
        data: &[T],
    ) -> Result<()> {
        assert!(
            offset
                .checked_add(data.len())
                .is_some_and(|end| end <= buffer.len()),
            "writing {} elements at {offset} into a buffer of {}",
            data.len(),
            buffer.len()
        );
        if data.is_empty() {
            return Ok(());
        }
        if self.staging.is_some() {
            let address = buffer.address() + (offset * size_of::<T>()) as u64;
            let target = Target::Buffer(buffer.internal.clone(), address);
            return self.stage(data.len(), target, |out| {
                write_slice(out, data);
                Vec::new()
            });
        }
        let Some(ptr) = buffer.coherent_mapping() else {
            return Err(Error::other(
                "the buffer's memory isn't mapped and the uploader's timeline has no staging \
                 ring: give `Timeline::new` a staging size",
            ));
        };
        // SAFETY: the mapping holds `buffer.len()` elements and `data` fits at `offset`; a
        // frame in flight reading it is the documented race.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.add(offset), data.len()) };
        Ok(())
    }

    /// Stages the whole mip `level` of every layer of `image` from `data`, tightly packed, one
    /// layer after another; needs a ring.
    /// Drained on a family other than the uploader's destination, the image is released to
    /// it. `Image::copy_from` writes right away instead, with host image copies. Panics for a
    /// mip the image doesn't have and data of another length than the mip's.
    #[validation_trace]
    pub fn image<F: Format + ColorAspect, U: IsTransferDst, D: Dim, V: Views>(
        &self,
        image: &Image<F, U, D, V>,
        level: u32,
        data: &[u8],
    ) -> Result<()> {
        self.ring()?;
        let copy = image_copy(image, level, data.len())?;
        self.stage(data.len(), Target::Image(copy), |out| {
            write_slice(out, data);
            Vec::new()
        })
    }

    /// Writes the elements of `items` (each its data and references, a generated
    /// `{Name}ReferenceInfo`) from element `first` on into `buffer`, staged like
    /// [`Self::write`] (or in place without a ring): the only uploader write of a referencing
    /// type. The buffer keeps what they reference alive from now on; what the elements it
    /// overwrites referenced and nothing else does any more stays alive until the copy
    /// completed (in place: is let go right away). Panics unless the elements fit.
    #[validation_trace]
    pub fn write_referencing<'a, T: ReferencingData, M: Memory>(
        &self,
        buffer: &Buffer<T, M>,
        first: usize,
        items: impl IntoIterator<Item = impl ReferenceInfo<'a, Data = T>>,
    ) -> Result<()> {
        let len = buffer.len();
        let elements: Vec<(T, SmallVec<[Reference<'a>; 4]>)> = items
            .into_iter()
            .map(|item| item.with_output(|data, references| (*data, references.into())))
            .collect();
        assert!(
            first
                .checked_add(elements.len())
                .is_some_and(|end| end <= len),
            "writing {} elements at {first} into a buffer of {len}",
            elements.len()
        );
        if elements.is_empty() {
            return Ok(());
        }
        let internal = &buffer.internal;
        let replace = || {
            let keep_alive = internal
                .keep_alive
                .get_or_init(|| Mutex::new(KeepAlive::new::<T>()));
            let mut keep_alive = lock(keep_alive);
            let mut released = Vec::new();
            for (i, (_, references)) in (first..).zip(&elements) {
                keep_alive.replace(internal, len, i, references, &mut released);
            }
            released
        };
        if self.staging.is_some() {
            let address = buffer.address() + (first * size_of::<T>()) as u64;
            let target = Target::Buffer(internal.clone(), address);
            return self.stage(elements.len(), target, |out| {
                for (slot, (data, _)) in out.iter_mut().zip(&elements) {
                    slot.write(*data);
                }
                replace()
            });
        }
        let Some(ptr) = buffer.coherent_mapping() else {
            return Err(Error::other(
                "the buffer's memory isn't mapped and the uploader's timeline has no staging \
                 ring: give `Timeline::new` a staging size",
            ));
        };
        for (i, (data, _)) in (first..).zip(&elements) {
            // SAFETY: inside the mapping, which holds `len` elements; a frame in flight
            // reading it is the documented race.
            unsafe { ptr.add(i).write(*data) };
        }
        drop(replace());
        Ok(())
    }
}

/// Copies `data` into `out`, which is as long.
fn write_slice<T: Copy>(out: &mut [MaybeUninit<T>], data: &[T]) {
    assert_eq!(out.len(), data.len());
    // SAFETY: both hold `data.len()` elements and don't overlap (`out` is exclusive).
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), out.as_mut_ptr().cast::<T>(), data.len())
    };
}

/// The copy of `len` bytes into the whole mip `level` of `image`, checked: a colour image with
/// transfer destination usage, a mip it has, and the bytes the mip holds (panics otherwise).
#[track_caller]
fn image_copy<F: Format + ColorAspect, U: IsTransferDst, D: Dim, V: Views>(
    image: &Image<F, U, D, V>,
    level: u32,
    len: usize,
) -> Result<ImageCopy> {
    image.format().aspect_check(vk::ImageAspectFlags::COLOR);
    image.usage_check(vk::ImageUsageFlags::TRANSFER_DST);
    assert!(
        level < image.mip_levels(),
        "mip level {level} is out of range for an image with {} levels",
        image.mip_levels()
    );
    let extent = D::extent_3d(image.mip_extent(level));
    let format = image.format();
    let layers = image.layers();
    let block = copy_block_of(&format, vk::ImageAspectFlags::COLOR);
    // Staged bytes start on `STAGING_ALIGN`, and a copy's address is a multiple of its block.
    assert!(
        STAGING_ALIGN.is_multiple_of(block.bytes as u64),
        "staged uploads into {:?} aren't supported: its {}-byte texel blocks don't divide the \
         ring's {STAGING_ALIGN}-byte alignment",
        format.vk_format(),
        block.bytes
    );
    let expected = rows_len(block, extent, layers, (0, 0));
    assert!(
        expected == len,
        "mip level {level} of {layers} layers holds {expected} bytes but {len} were given"
    );
    let mips = match image.internal.states.len() {
        1 => 0..image.mip_levels(),
        _ => level..level + 1,
    };
    Ok(ImageCopy {
        image: image.internal.clone(),
        aspect: vk::ImageAspectFlags::COLOR,
        mips,
        level,
        extent,
        layers,
    })
}

/// An image a drain copied into, the mips it copied (bit `i` for mip `i`) and the family it
/// releases them to.
struct Copied<'a> {
    image: &'a ImageInternal,
    aspect: vk::ImageAspectFlags,
    mips: u64,
    to: Kind,
}

impl<K: QueueKind> CommandBuffer<'_, K> {
    /// Records the copies of every write its timeline's uploaders published so far, here, in
    /// publication order; barrier tracking orders them against the commands around them.
    /// Every image an uploader for another family `D` copied into is released to `D` after
    /// the last copy, and `D`'s first access acquires it: the receiving submission has to
    /// wait for this one (`Submission::wait_for`), which lava doesn't check, and this
    /// recording can't use those images after it, so drain after its own uses. Panics, before
    /// anything is recorded, if a copy goes into an image another family owns: upload it
    /// through that family's timeline. If the submission doesn't reach the GPU, the copies
    /// are published again. Without a ring it records nothing and returns zeros.
    #[validation_trace]
    pub fn drain_uploads(&mut self) -> UploadStats {
        let Some(staging) = self.rec.staging else {
            return UploadStats::default();
        };
        let mut pending = lock(&staging.pending);
        let foreign = pending.iter().find_map(|s| match &s.target {
            Target::Image(copy) => copy.foreign_owner(K::KIND),
            _ => None,
        });
        if let Some(owner) = foreign {
            drop(pending);
            panic!(
                "`drain_uploads` on the {kind:?} family copies into an image the {owner:?} \
                 family owns: upload it through a {owner:?} timeline",
                kind = K::KIND
            );
        }
        let mut copied = SmallVec::<[Copied; 4]>::new();
        let mut stats = UploadStats {
            capacity: staging.ring.size(),
            ..Default::default()
        };
        let ring = &staging.buffer.internal;
        for staged in pending.iter() {
            if staged.target.orphaned() {
                continue;
            }
            let src = BufferSlice::<u8>::new(ring, ring.address + staged.offset, staged.size);
            match &staged.target {
                Target::Buffer(buffer, address) => {
                    self.copy_staged(src, BufferSlice::<u8>::new(buffer, *address, staged.size))
                }
                Target::Image(copy) => {
                    record_image_copy(self, src, copy);
                    if staged.destination != K::KIND {
                        let mips = (copy.mips.start..copy.mips.end).fold(0, |m, i| m | 1 << i);
                        let same = |c: &&mut Copied| {
                            std::ptr::eq(c.image, &*copy.image) && c.to == staged.destination
                        };
                        match copied.iter_mut().find(same) {
                            Some(c) => c.mips |= mips,
                            None => copied.push(Copied {
                                image: &copy.image,
                                aspect: copy.aspect,
                                mips,
                                to: staged.destination,
                            }),
                        }
                    }
                }
            }
            stats.copies += 1;
            stats.bytes += staged.size;
        }
        for c in &copied {
            // One release per run of copied mips.
            let mut mip = 0;
            while mip < 64 {
                if c.mips >> mip & 1 == 0 {
                    mip += 1;
                    continue;
                }
                let end = mip + (c.mips >> mip).trailing_ones();
                (self.rec).release(c.image, (mip..end).into(), c.aspect, c.to);
                mip = end;
            }
        }
        drop(copied);
        let value = self.rec.value;
        let mut in_flight = lock(&staging.in_flight);
        let done = timeline::completed(staging.timeline);
        in_flight.retain(|(_, v)| *v > done);
        in_flight.extend(pending.drain(..).map(|s| (s, value)));
        staging
            .ring
            .reclaim(in_flight.iter().map(|(s, _)| s.reserved.start));
        stats.used = staging.ring.used();
        stats
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
            image: copy.image.image,
            layout: vk::ImageLayout::GENERAL,
            aspect: copy.aspect,
            mip_range: copy.mips.clone().into(),
        },
        internal: &copy.image,
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
            layer_count: copy.layers,
        })
        .image_layout(vk::ImageLayout::GENERAL)
        .image_offset(vk::Offset3D::default())
        .image_extent(copy.extent)];
    unsafe {
        Functions::device_address_commands().cmd_copy_memory_to_image(
            cmd.rec.handle,
            &vk::CopyDeviceMemoryImageInfoKHR::default()
                .image(copy.image.image)
                .regions(&regions),
        )
    };
}

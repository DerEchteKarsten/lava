//! Uploads into memory the host can't map through a staging ring per timeline (`Staging`, sized at `Timeline::new`, none on `rebar` devices, where every `Gpu` buffer is mapped): `Uploader` handles cloned from the timeline fill it from any thread (`init`, `from_slice`, `write`, `image`, in place where the target is mapped), `cmd.drain_uploads` records every published copy at the point of the recording the application chooses and releases the images to their destination family, and the timeline's values free the drained ring space
use std::{
    collections::VecDeque,
    fmt,
    ops::Range,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::SeqCst},
    },
    time::{Duration, Instant},
};

use bytemuck::Pod;
use lava_macros::validation_trace;
use smallvec::SmallVec;

use crate::{
    buffer::{Buffer, Gpu, Memory, Upload, slice::BufferSlice},
    command_buffer::{CommandBuffer, ImageAccess, ImageUse, image_state::ImageState},
    error::{Error, Result},
    image::{
        Image, Views,
        dim::Dim,
        format::{ColorAspect, Format},
        slice::{copy_block_of, rows_len},
        usage::IsTransferDst,
    },
    queue::{Kind, QueueKind, lock, timeline},
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
        debug_assert!(len % STAGING_ALIGN == 0 && len <= self.size);
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
    /// Every layer of the image, the data one layer after another.
    layers: u32,
}

// SAFETY: `states` points at atomics, which every thread may access.
unsafe impl Send for ImageCopy {}

impl ImageCopy {
    fn states(&self) -> &[AtomicU64] {
        // SAFETY: see `states`; the ring's `pending` lock is held while it is used.
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

enum Target {
    /// The address the staged bytes go to.
    Buffer(u64),
    Image(ImageCopy),
    /// Its target was dropped while the copy was in flight; it is never recorded again.
    Dropped,
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
    /// The entries of both lists, so drops skip the locks while there are none.
    entries: AtomicUsize,
}

/// The number of live rings, so drops skip every lock while there are none.
static RINGS: AtomicUsize = AtomicUsize::new(0);

impl Staging {
    /// A ring of `size` bytes (rounded up to whole `STAGING_ALIGN`s) for timeline `timeline`,
    /// registered in its index.
    pub(crate) fn new(timeline: usize, size: u64) -> Result<Arc<Self>> {
        let size = size.next_multiple_of(STAGING_ALIGN);
        let buffer = Buffer::raw(Upload, size, Some(STAGING_ALIGN as u32))?;
        let staging = Arc::new(Self {
            buffer,
            ring: Ring::new(size),
            timeline,
            closed: AtomicBool::new(false),
            pending: Mutex::new(Vec::new()),
            in_flight: Mutex::new(VecDeque::new()),
            entries: AtomicUsize::new(0),
        });
        *lock(&timeline::INDICES[timeline].staging) = Some(staging.clone());
        RINGS.fetch_add(1, SeqCst);
        Ok(staging)
    }

    /// Unregisters the ring when its timeline drops: no write reaches it from then on.
    pub(crate) fn close(&self) {
        self.closed.store(true, SeqCst);
        *lock(&timeline::INDICES[self.timeline].staging) = None;
        RINGS.fetch_sub(1, SeqCst);
    }

    /// Frees the regions of the submissions that completed (`done`, the timeline's completed
    /// value) and everything else no longer in use.
    fn reclaim(&self, done: u64) {
        let pending = lock(&self.pending);
        let mut in_flight = lock(&self.in_flight);
        let before = in_flight.len();
        in_flight.retain(|(_, value)| *value > done);
        self.entries.fetch_sub(before - in_flight.len(), SeqCst);
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
        self.entries.fetch_add(1, SeqCst);
    }

    /// Forgets the pending copies whose target `matches` and neutralises the in-flight ones.
    fn forget(&self, mut matches: impl FnMut(&Target) -> bool) {
        let mut pending = lock(&self.pending);
        let before = pending.len();
        pending.retain(|s| !matches(&s.target));
        self.entries.fetch_sub(before - pending.len(), SeqCst);
        for (staged, _) in lock(&self.in_flight).iter_mut() {
            if matches(&staged.target) {
                staged.target = Target::Dropped;
            }
        }
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
        let before = back.len();
        let mut back: Vec<Staged> = (back.into_iter())
            .map(|(staged, _)| staged)
            .filter(|staged| !matches!(staged.target, Target::Dropped))
            .collect();
        self.entries.fetch_sub(before - back.len(), SeqCst);
        back.append(&mut pending);
        *pending = back;
    }
}

/// The live rings, at most one per timeline.
fn rings() -> SmallVec<[Arc<Staging>; 2]> {
    (timeline::INDICES.iter())
        .filter_map(|index| lock(&index.staging).clone())
        .collect()
}

fn forget(mut matches: impl FnMut(&Target) -> bool) {
    if RINGS.load(SeqCst) == 0 {
        return;
    }
    for ring in rings() {
        if ring.entries.load(SeqCst) > 0 {
            ring.forget(&mut matches);
        }
    }
}

/// Forgets the copies into a dropped buffer's `range`.
pub(crate) fn forget_buffer(range: Range<u64>) {
    forget(|t| matches!(t, Target::Buffer(address) if range.contains(address)));
}

/// Forgets the copies into a dropped image, before its tracked states go.
pub(crate) fn forget_image(image: vk::Image) {
    forget(|t| matches!(t, Target::Image(copy) if copy.image == image));
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
    fn stage<T: Pod>(&self, len: usize, target: Target, fill: impl FnOnce(&mut [T])) -> Result<()> {
        let staging = self.ring()?;
        let size = (len * size_of::<T>()) as u64;
        let reservation = staging.reserve(size, self.timeout)?;
        let counted = CountOut(&staging.ring);
        let ptr = staging.buffer.mapping().expect("upload memory is mapped");
        // SAFETY: the reserved region is the writer's alone until it is published and drained.
        let data = unsafe {
            let ptr = ptr.add(reservation.offset as usize).cast::<T>();
            assert!(
                ptr.is_aligned(),
                "staged {}s at a misaligned {ptr:?}",
                std::any::type_name::<T>()
            );
            std::slice::from_raw_parts_mut(ptr, len)
        };
        fill(data);
        staging.publish(Staged {
            reserved: reservation.range,
            offset: reservation.offset,
            size,
            target,
            destination: self.destination,
        });
        drop(counted);
        Ok(())
    }

    /// A device-local buffer of `len` elements that `fill` writes, given the buffer's address
    /// (for data that embeds addresses into itself). Where its memory is mapped (`rebar`
    /// devices, integrated GPUs) `fill` writes it in place and nothing is left to drain;
    /// elsewhere `fill` writes the ring and the next drain copies it. Either way the buffer is
    /// a [`Gpu`] one without host access.
    #[validation_trace]
    pub fn init<T: Copy + Pod>(
        &self,
        len: usize,
        fill: impl FnOnce(u64, &mut [T]),
    ) -> Result<Buffer<T, Gpu>> {
        let buffer = Buffer::<T, Gpu>::new(len)?;
        match buffer.coherent_mapping() {
            // SAFETY: no submission has seen the new buffer, whose mapping holds `len`.
            Some(ptr) => fill(buffer.address(), unsafe {
                std::slice::from_raw_parts_mut(ptr, len)
            }),
            None => self.stage(len, Target::Buffer(buffer.address()), |data| {
                fill(buffer.address(), data)
            })?,
        }
        Ok(buffer)
    }

    /// A device-local buffer holding `data`, made like [`Self::init`]. Empty data is an error,
    /// like an empty buffer.
    #[validation_trace]
    pub fn from_slice<T: Copy + Pod>(&self, data: &[T]) -> Result<Buffer<T, Gpu>> {
        self.init(data.len(), |_, out| out.copy_from_slice(data))
    }

    /// Writes `data` at element `offset` of `buffer`, any buffer. With a ring it is staged, and
    /// its copy is ordered like every command where the ring is drained: earlier commands read
    /// the old data, later ones the new, also while frames in flight read the buffer. Without
    /// a ring it is written into the buffer's mapping right away, which a frame in flight may
    /// be reading: double-buffer such data. A buffer that isn't mapped is an error then.
    /// Panics unless `data` fits.
    #[validation_trace]
    pub fn write<T: Copy + Pod, M: Memory>(
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
            return self.stage(data.len(), Target::Buffer(address), |out| {
                out.copy_from_slice(data)
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
            out.copy_from_slice(data)
        })
    }
}

/// The copy of `len` bytes into the whole mip `level` of `image`, checked: a colour image with
/// transfer destination usage, a mip it has, and the bytes the mip holds (panics otherwise).
#[track_caller]
fn image_copy<F: Format + ColorAspect, U: IsTransferDst, D: Dim, V: Views>(
    image: &Image<F, U, D, V>,
    level: u32,
    len: usize,
) -> Result<ImageCopy> {
    image.aspect_check(vk::ImageAspectFlags::COLOR);
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
        STAGING_ALIGN % block.bytes as u64 == 0,
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
    let states = image.whole_view().states;
    let mips = match states.len() {
        1 => 0..image.mip_levels(),
        _ => level..level + 1,
    };
    Ok(ImageCopy {
        image: image.whole_view().image,
        states: states.as_ptr(),
        states_len: states.len(),
        aspect: vk::ImageAspectFlags::COLOR,
        mips,
        level,
        extent,
        layers,
    })
}

/// An image a drain copied into, the mips it copied (bit `i` for mip `i`) and the family it
/// releases them to.
struct Copied {
    image: vk::Image,
    states: *const AtomicU64,
    states_len: usize,
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
        for staged in pending.iter() {
            let src = BufferSlice::<u8>::new(staging.buffer.address() + staged.offset, staged.size);
            match &staged.target {
                &Target::Buffer(address) => {
                    self.copy_buffer(src, BufferSlice::<u8>::new(address, staged.size))
                }
                Target::Image(copy) => {
                    record_image_copy(self, src, copy);
                    if staged.destination != K::KIND {
                        let mips = (copy.mips.start..copy.mips.end).fold(0, |m, i| m | 1 << i);
                        let same =
                            |c: &&mut Copied| c.image == copy.image && c.to == staged.destination;
                        match copied.iter_mut().find(same) {
                            Some(c) => c.mips |= mips,
                            None => copied.push(Copied {
                                image: copy.image,
                                states: copy.states,
                                states_len: copy.states_len,
                                aspect: copy.aspect,
                                mips,
                                to: staged.destination,
                            }),
                        }
                    }
                }
                Target::Dropped => continue,
            }
            stats.copies += 1;
            stats.bytes += staged.size;
        }
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
                (self.rec).release(c.image, states, (mip..end).into(), c.aspect, c.to);
                mip = end;
            }
        }
        let value = self.rec.value;
        let mut in_flight = lock(&staging.in_flight);
        let done = timeline::completed(staging.timeline);
        let before = in_flight.len();
        in_flight.retain(|(_, v)| *v > done);
        staging.entries.fetch_sub(before - in_flight.len(), SeqCst);
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
            image: copy.image,
            layout: vk::ImageLayout::GENERAL,
            aspect: copy.aspect,
            mip_range: copy.mips.clone().into(),
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
            layer_count: copy.layers,
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

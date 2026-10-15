//! API sketches for STATE.md entries and `BindingOutput` size reductions (pseudo-Rust in a block comment, so it needn't compile)
/*
// ------------------------------------------------------------------------------------------
// 1. Pipelines are created synchronously on first use, under a write lock, without a cache.
// ------------------------------------------------------------------------------------------
// (a) A VkPipelineCache, saved next to the binary: cheap, no API change, and the driver's own
//     cache already makes repeat starts fast, so this only helps a shader change at start.
pub struct PipelineManager {
    cache: vk::PipelineCache, // created from `$OUT_DIR/pipeline.cache` if it exists
    // ...
}
impl Drop for PipelineManager {
    fn drop(&mut self) {
        // `get_pipeline_cache_data` -> written back to the file.
    }
}

// (b) Warm-up before the first frame: every pass's pipelines with the constants and raster
//     states the engine uses, built in parallel on worker threads. The `RasterHash` is what
//     the command buffer would compute; core lists them once at startup.
pub fn warm_up(requests: &[PipelineRequest]) {
    requests.par_iter().for_each(|r| match r {
        PipelineRequest::Compute { pass, constants } => { pipelines().compute(pass.index, constants); }
        PipelineRequest::Raster { pass, hash } => { pipelines().raster(hash, pass.index); }
    });
}
pub enum PipelineRequest {
    Compute { pass: &'static PassEntry, constants: Constants },
    Raster { pass: &'static PassEntry, hash: RasterHash },
}

// (c) Only the write lock is a problem for *other* passes while one compiles: `cached` holds
//     it during `create()`. A per-pass lock (an array of `Mutex<HashMap<..>>` instead of one
//     `RwLock` over the array) lets passes compile in parallel and readers of other passes
//     through; that is the smallest change and removes the stall for everything but the
//     pass being created.
// specialized_compute_pipelines: [Mutex<HashMap<Constants, vk::Pipeline>>; NUM_SPECIALIZED_COMPUTE_PIPELINES],
// raster_pipelines: [Mutex<HashMap<RasterHash, vk::Pipeline>>; NUM_RASTER_PIPELINES],

// ------------------------------------------------------------------------------------------
// 2. Smaller `BindingOutput`s. The push constants stay as they are: 256 bytes inline, pushed
//    whole (BvhCull uses about 232 of them).
// ------------------------------------------------------------------------------------------
// Today (bytes):                         | largest real use in core
//   pass                              8  |
//   push_constants                  256  | ~232 (BvhCull)
//   constants: [u8; 128] + len      136  | ~52 (BvhCull, with 8 clock regions)
//   images: SmallVec<[ImageUse; 8]> 464  | 5 (Composite); ImageUse is 56 bytes
//   buffers: SmallVec<[BufferAccess; 16]> 528 | 10 (BvhCull); BufferAccess is 32 bytes
//   stage                             8  |
//   total                         ~1400
//
// What each option leaves (the options combine):
//   (a) pooled rest                  ~280   (no API change)
//   (b) interned constants          ~1280   (no API change)
//   (c) compact access records      ~1080   (`images()`/`buffers()` change)
//   (d) smallvec `union`            ~1384   (no API change)
//   (b) + (c) + (d)                  ~940
//   (a) + (b) + (c)                  ~280, with a ~690 byte box instead of ~1130
//   (e) per-pass const generics      exact per pass, but every signature naming it changes

// ---- (a) Everything but the push constants in a recycled box ------------------------------
// The accesses and constants move into a box taken from a thread-local free list and given
// back (cleared) on drop. Moving a `BindingOutput` copies 280 bytes; the box's contents never
// move. No allocation at steady state; the box stays in cache because the list is LIFO.
pub struct BindingOutput<'a, K> {
    pass: usize,
    push_constants: [u8; MAX_PUSH_CONSTANTS_SIZE], // unchanged
    stage: vk::PipelineStageFlags2,
    rest: Pooled<'a>,
    kind: PhantomData<K>,
}

struct Rest<'a> {
    constants: Constants,
    images: SmallVec<[ImageUse<'a>; MAX_PASS_IMAGES]>,
    buffers: SmallVec<[BufferAccess; MAX_PASS_BUFFERS]>,
}

/// A `Box<Rest>` from the pool. The pool holds only empty boxes, so the lifetime is changed
/// only on empty contents: on `take` (nothing inside to outlive) and on drop (after `clear`).
struct Pooled<'a>(ManuallyDrop<Box<Rest<'a>>>);

thread_local! {
    static POOL: RefCell<Vec<Box<Rest<'static>>>> = const { RefCell::new(Vec::new()) };
}
/// Bounds a consumer thread's pool when bindings are made on one thread and dropped on
/// another (the maker then allocates each time; core makes and records on one thread).
const POOL_CAP: usize = 64;

impl<'a> Pooled<'a> {
    fn take() -> Self {
        let rest = POOL.with_borrow_mut(Vec::pop).unwrap_or_else(|| Box::new(Rest::EMPTY));
        // SAFETY: the box is empty.
        Self(ManuallyDrop::new(unsafe { transmute::<Box<Rest<'static>>, Box<Rest<'a>>>(rest) }))
    }
}

impl Drop for Pooled<'_> {
    fn drop(&mut self) {
        let mut rest = unsafe { ManuallyDrop::take(&mut self.0) };
        rest.images.clear();
        rest.buffers.clear(); // a spilled SmallVec keeps its heap part: the next big pass reuses it
        rest.constants = Constants::NONE;
        // SAFETY: the box is empty.
        let rest = unsafe { transmute::<Box<Rest<'_>>, Box<Rest<'static>>>(rest) };
        // `try_with`: the thread-local may already be gone while the thread exits.
        let _ = POOL.try_with(|pool| {
            let mut pool = pool.borrow_mut();
            if pool.len() < POOL_CAP {
                pool.push(rest);
            }
        });
    }
}

impl<'a, K> BindingOutput<'a, K> {
    pub fn new<G: Pod, const I: usize, const B: usize>(
        pass: usize,
        bindings: &G,
        stage: vk::PipelineStageFlags2,
        images: [ImageUse<'a>; I],
        buffers: [BufferAccess; B],
    ) -> Self {
        let mut rest = Pooled::take();
        rest.images.extend(images);
        rest.buffers.extend(buffers);
        Self { pass, push_constants: padded(bindings), stage, rest, kind: PhantomData }
    }

    // `with_constants`, `sampled_read`, ... write through `self.rest` instead of `self`;
    // `images()`, `buffers()`, `constants()` read through it. Signatures unchanged.

    fn erase<'s>(self) -> BindingOutput<'s, ()>
    where
        'a: 's,
    {
        // Still a transmute (Rest is invariant in 'a, like today's SmallVec), now of one box
        // pointer instead of the two vectors.
        // BindingOutput { rest: unsafe { transmute::<Pooled<'a>, Pooled<'s>>(self.rest) }, .. }
    }
}
// Costs: a thread-local access and a pop/push per binding, a `Drop` impl, one more pointer
// hop at record time (to a box that was just written). RasterBuilder's draw array, rebuilt
// on every draw call, shrinks with it: a `Draw` becomes ~280 bytes plus command and state.
// Variant without thread-locals: one global `crossbeam::ArrayQueue<Box<Rest<'static>>>`
// (atomics per take/give, no per-thread imbalance).

// ---- (b) Interned specialization constants ------------------------------------------------
// `with_constants` looks the bytes up in a per-pass set and keeps a reference to the interned
// copy: 16 bytes instead of 136. The set is bounded like the pipelines (every distinct value
// builds one; a pass warns at 16), so the copies are leaked. The pipeline caches then key on
// the pointer, so record time hashes 8 bytes instead of up to 128: the hash of the bytes moves
// from record time to `with_constants`, it isn't added.
#[derive(Clone, Copy)]
pub(crate) struct Constants(&'static [u8]);

impl Constants {
    pub(crate) const NONE: Self = Self(&[]);

    pub(crate) fn new(pass: usize, bytes: &[u8]) -> Self {
        let set = &interned()[pass]; // RwLock<HashSet<&'static [u8]>>, sized from the pass map
        if let Some(&interned) = set.read().get(bytes) {
            return Self(interned);
        }
        let mut set = set.write();
        // Re-check under the write lock: another thread may have inserted it meanwhile.
        Self(*set.get_or_insert_with(bytes, |b| Box::leak(Box::from(b))))
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        self.0 // `BindingOutput::constants()` keeps returning `&[u8]`
    }
}

/// Interned per pass, so equal pointers <=> equal bytes within a pass, and pipeline keys are
/// always per pass.
impl PartialEq for Constants {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self.0, other.0)
    }
}
impl Hash for Constants {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.as_ptr().hash(state);
    }
}

impl<'a, K> BindingOutput<'a, K> {
    pub fn with_constants<G: Pod, const I: usize, const B: usize>(
        mut self,
        constants: &G,
        images: [ImageUse<'a>; I],
        buffers: [BufferAccess; B],
    ) -> Self {
        self.constants = Constants::new(self.pass, bytes_of(constants)); // the pass is known here
        // ...
        todo!()
    }
}
// Hot reload rebuilds the cached specializations under the same keys, so interned entries
// stay valid. Later step if wanted: the interned entry holds the compute/ray tracing
// pipeline itself (`&'static Specialization { bytes, pipeline: AtomicU64 }`), which a reload
// swaps, and recording skips the cache lookup altogether.

// ---- (c) Compact access records (superseded by (f) at the end) ---------------------------
// Inside a `BindingOutput`, every access of the pass runs at `self.stage` (the generated
// accesses and every `sampled_read`/`buffer_write`/... use it), so the records drop their
// stage, and images keep a thin pointer to their tracked layouts.
//   images:  56 -> 32 bytes each (8 inline: 464 -> 272)
//   buffers: 32 -> 24 bytes each (16 inline: 528 -> 400)

/// An image access of a pass, at the pass's stage.
struct PassImage<'a> {
    image: vk::Image,          // 8
    access: vk::AccessFlags2,  // 8: arbitrary on purpose (other binding generators), so no u8 enum
    layout: vk::ImageLayout,   // 4: also arbitrary, values like 1000xxxxxx
    first_mip: u8,             // 1
    mip_count: u8,             // 1
    aspect: u8,                // 1: COLOR/DEPTH/STENCIL bits
    per_mip: bool,             // 1: `layouts` has `mip_count` entries, else one
    layouts: NonNull<AtomicU32>, // 8, thin
    _borrow: PhantomData<&'a [AtomicU32]>,
}

/// A buffer access of a pass, at the pass's stage.
struct PassBuffer {
    access: vk::AccessFlags2, // 8
    range: Range<u64>,        // 16
}

impl<'a, K> BindingOutput<'a, K> {
    pub fn new<G: Pod, const I: usize, const B: usize>(
        pass: usize,
        bindings: &G,
        stage: vk::PipelineStageFlags2,
        images: [ImageUse<'a>; I],     // the public constructor types stay
        buffers: [BufferAccess; B],
    ) -> Self {
        debug_assert!(images.iter().all(|i| i.access.stage == stage)); // likewise for buffers
        // store `PassImage::from(use)` / `PassBuffer::from(access)`
    }

    // API change: the accessors can't hand out slices of the public types any more. They
    // return them by value, with the stage filled back in (lava-tests reads `.stage`/`.access`).
    pub fn images(&self) -> impl ExactSizeIterator<Item = ImageUse<'a>> + Clone { .. }
    pub fn buffers(&self) -> impl ExactSizeIterator<Item = BufferAccess> + Clone { .. }
}

// The accesses with another stage don't belong to the pass: the index and indirect buffers of
// `draw_indexed`/`draw_indirect(_count)`/`launch_indirect`/`compute_indirect`. They move to the
// command that has them (`Draw.extra: SmallVec<[BufferAccess; 2]>`, `compute_private`'s
// argument) instead of being pushed into the binding's buffers.
//
// `flush_pending`/`compute_barriers` take the accesses by value instead of by reference:
pub(crate) fn flush_pending<'a>(
    &mut self,
    image_accesses: impl Iterator<Item = ImageUse<'a>> + Clone,
    buffer_accesses: impl Iterator<Item = BufferAccess> + Clone,
);
// and the raster `record` chains each draw's `bindings.buffers()` with its `extra`.

// ---- (d) smallvec's `union` feature -------------------------------------------------------
// One line in the workspace Cargo.toml; drops the enum tag of every inline SmallVec (8 bytes
// each, 16 per binding). Affects every SmallVec in the workspace the same way, never worse.
smallvec = { version = "1.15.1", features = ["serde", "const_generics", "union"] }

// ---- (e) Per-pass sizes as const generics (not recommended) -------------------------------
// The generator knows each pass's push size and access counts, so the type could carry them:
pub struct BindingOutput<'a, K, const I: usize = 8, const B: usize = 16> {
    images: ArrayVec<ImageUse<'a>, { I + 2 }>,   // + room for sampled_read & co.
    buffers: ArrayVec<BufferAccess, { B + 2 }>,
    // ...
}
// `Composite::push_bindings(..) -> BindingOutput<'a, kind::Compute, 5, 0>`.
// Against it: every function taking a `BindingOutput<'_, K>` (lava's own `compute`, `draw`,
// ..., lava-tests' helpers) becomes generic over I and B; the registration methods need a
// fixed spare capacity (or change the type, i.e. `{ I + 1 }` per call); RasterBuilder's
// draw array needs one element type again, so draws would erase into today's layout anyway.

// ---- (f) CompressedAccess: 20 bytes per access, images and buffers in one list ------------
// What a binding's access doesn't have to store, because the binding already knows it:
//   stage   `BindingOutput.stage` (every access of the pass runs there)
//   access  one bit: read, or read and write
//   layout  always GENERAL (attachments, the only other layout, aren't binding accesses)
//   aspect  colour (depth formats have no storage usage)
//   mips    at most 32: a 33rd mip needs an extent of 2^32, which a u32 can't hold, so 5 bits
//           each for the first mip and the count (stored minus one) are exact, not a cap
//
//   bytes 0..4    header: kind, read/write, mip range (bit layout below)
//   bytes 4..12   image: `vk::Image` handle         buffer: base device address
//   bytes 12..20  image: pointer to the first
//                 tracked layout of the range       buffer: size in bytes
//
// `repr(C, packed(4))` makes it 20 bytes; with natural alignment the u64s would pad it to 24.
// x86 loads unaligned u64s at no cost; fields are only ever read by value, never referenced.
#[derive(Clone, Copy)]
#[repr(C, packed(4))]
pub struct CompressedAccess<'a> {
    header: u32,
    /// `vk::Image::as_raw()` or the buffer's base address.
    handle_or_address: u64,
    /// `*const AtomicU32` or the buffer's size.
    layouts_or_size: u64,
    _borrow: PhantomData<&'a AtomicU32>,
}
const _: () = assert!(size_of::<CompressedAccess>() == 20);
// The pointer makes it !Send/!Sync; it is a shared borrow of atomics, so both are sound and
// keep `BindingOutput` `Send` as it is today.
unsafe impl Send for CompressedAccess<'_> {}
unsafe impl Sync for CompressedAccess<'_> {}

// Header bits. Bits 0, 1 and 16..26 are the design above; bits 2..6 are what four cases in
// today's code need (see "Gaps" below), in the header's spare bits, so the size stays 20.
const IMAGE: u32 = 1 << 0;            // else buffer
const WRITE: u32 = 1 << 1;            // else read
const ONE_LAYOUT: u32 = 1 << 2;       // gap 1: one tracked layout for the whole mip range
const DEPTH: u32 = 1 << 3;            // gap 2: depth aspect (sampled depth attachment)
const STENCIL: u32 = 1 << 4;          // gap 2: stencil aspect (depth-stencil formats)
const ACCELERATION_STRUCTURE: u32 = 1 << 5; // gap 3: a trace's read of a TLAS
const FIRST_MIP_SHIFT: u32 = 16;      // bits 16..21
const MIP_COUNT_SHIFT: u32 = 21;      // bits 21..26, count - 1
const MIP_MASK: u32 = 0b1_1111;

impl<'a> CompressedAccess<'a> {
    pub(crate) fn image(view_image: vk::Image, layouts: &'a [AtomicU32], mips: Range<u32>,
                        aspect: vk::ImageAspectFlags, write: bool) -> Self {
        debug_assert!(mips.start < 32 && mips.end - mips.start <= 32 && !mips.is_empty());
        debug_assert!(layouts.len() == 1 || layouts.len() == mips.len());
        let mut header = IMAGE
            | (mips.start << FIRST_MIP_SHIFT)
            | ((mips.end - mips.start - 1) << MIP_COUNT_SHIFT);
        header |= if write { WRITE } else { 0 };
        header |= if layouts.len() == 1 { ONE_LAYOUT } else { 0 };
        header |= if aspect.contains(DEPTH_ASPECT) { DEPTH } else { 0 };
        header |= if aspect.contains(STENCIL_ASPECT) { STENCIL } else { 0 };
        Self {
            header,
            handle_or_address: view_image.as_raw(),
            layouts_or_size: layouts.as_ptr() as u64,
            _borrow: PhantomData,
        }
    }

    pub(crate) fn buffer(range: Range<u64>, write: bool) -> Self {
        Self {
            header: if write { WRITE } else { 0 },
            handle_or_address: range.start,
            layouts_or_size: range.end - range.start,
            _borrow: PhantomData,
        }
    }

    pub fn is_image(self) -> bool { self.header & IMAGE != 0 }
    pub fn is_write(self) -> bool { self.header & WRITE != 0 }
    pub fn mip_range(self) -> Range<u32> {
        let first = (self.header >> FIRST_MIP_SHIFT) & MIP_MASK;
        let count = ((self.header >> MIP_COUNT_SHIFT) & MIP_MASK) + 1;
        (first..first + count).into()
    }

    /// The flags the barriers see. Sync2's SHADER_READ/SHADER_WRITE cover sampled and storage
    /// accesses alike, so dropping today's SAMPLED_READ/STORAGE_READ distinction loses
    /// nothing a barrier needs.
    fn flags(self) -> vk::AccessFlags2 {
        if self.header & ACCELERATION_STRUCTURE != 0 {
            TRACE_READ // ACCELERATION_STRUCTURE_READ | MEMORY_READ (see acceleration_structure.rs)
        } else if self.is_write() {
            vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE
        } else {
            vk::AccessFlags2::SHADER_READ
        }
    }

    /// Expanded at record time, with the binding's stage, into what `compute_barriers` takes.
    pub(crate) fn image_use(self, stage: vk::PipelineStageFlags2) -> ImageUse<'a> {
        let mips = self.mip_range();
        let len = if self.header & ONE_LAYOUT != 0 { 1 } else { mips.len() };
        let layouts = unsafe {
            // SAFETY: made from a `&'a [AtomicU32]` of this length in `image`.
            slice::from_raw_parts(self.layouts_or_size as *const AtomicU32, len)
        };
        ImageUse {
            access: ImageAccess {
                stage,
                access: self.flags(),
                image: vk::Image::from_raw(self.handle_or_address),
                layout: vk::ImageLayout::GENERAL,
                aspect: self.aspect(), // COLOR unless DEPTH/STENCIL
                mip_range: mips,
            },
            layouts,
        }
    }

    pub(crate) fn buffer_access(self, stage: vk::PipelineStageFlags2) -> BufferAccess {
        BufferAccess {
            stage,
            access: self.flags(),
            range: (self.handle_or_address..self.handle_or_address + self.layouts_or_size).into(),
        }
    }
}

// The binding holds one list: 24 inline (today's 8 + 16) at 20 bytes is 480 instead of 992.
pub struct BindingOutput<'a, K> {
    pass: usize,
    push_constants: [u8; MAX_PUSH_CONSTANTS_SIZE], // unchanged
    constants: Constants,
    accesses: SmallVec<[CompressedAccess<'a>; MAX_PASS_ACCESSES]>, // MAX_PASS_ACCESSES = 24
    stage: vk::PipelineStageFlags2,
    kind: PhantomData<K>,
}
// ~904 bytes instead of ~1400 (~784 with (b)'s interned constants, ~776 with (d) as well).

// Binding generator API: the views and slices make compressed accesses directly, and the
// generated code passes one array instead of two.
impl<'a, F: Format, U: ImageUsage, D: Dim> ImageView<'a, F, U, D> {
    /// `Tex2D`, `Image2D` and a `DynImg`'s registration.
    pub fn read(&self) -> CompressedAccess<'a> {
        CompressedAccess::image(self.image, self.layouts, self.mip_range, self.format.aspects(), false)
    }
    /// `MutImage2D`, `MutTex2D`.
    pub fn write(&self) -> CompressedAccess<'a> { /* same with true */ }
    // `access(stage, access, layout) -> ImageUse` stays for transfers and attachments.
}
impl<'a, T: Pod, U: BufferUsage> BufferSlice<'a, T, U> {
    pub fn read(&self) -> CompressedAccess<'a> { CompressedAccess::buffer(self.get_range(), false) }
    pub fn write(&self) -> CompressedAccess<'a> { CompressedAccess::buffer(self.get_range(), true) }
}
impl Tlas {
    /// Its memory and every structure it reaches (gap 3).
    pub fn trace_accesses(&self) -> [CompressedAccess<'_>; 2] {
        [
            CompressedAccess::buffer(self.range(), false).with(ACCELERATION_STRUCTURE),
            CompressedAccess::buffer(EVERY_ACCELERATION_STRUCTURE, false).with(ACCELERATION_STRUCTURE),
        ]
    }
}

impl<'a, K> BindingOutput<'a, K> {
    pub fn new<G: Pod, const N: usize>(
        pass: usize,
        bindings: &G,
        stage: vk::PipelineStageFlags2,
        accesses: [CompressedAccess<'a>; N],
    ) -> Self { .. }

    pub fn with_constants<G: Pod, const N: usize>(
        mut self,
        constants: &G,
        accesses: [CompressedAccess<'a>; N],
    ) -> Self { .. }

    // `sampled_read(view)` pushes `view.read()`, `storage_write(view)` pushes `view.write()`,
    // `buffer_read(slice)` pushes `slice.read()`, ...: signatures unchanged.

    /// Replaces `images()`/`buffers()` (lava-tests' binding tests read these).
    pub fn accesses(&self) -> &[CompressedAccess<'a>] { &self.accesses }
}

// Generated code, e.g. Composite:
//     BindingOutput::new(
//         4,
//         &CompositeGpu { .. },
//         vk::PipelineStageFlags2::COMPUTE_SHADER,
//         [gbuffer.read(), depth.read(), accum.write(), revealage.write(), out.write()],
//     )

// Recording expands them with the binding's stage; transfers and attachments keep passing
// full `ImageUse`/`BufferAccess`es, so `flush_pending` takes values instead of references:
fn compute_private(&mut self, bindings: BindingOutput<'_, kind::Compute>, ..) {
    let stage = bindings.stage;
    let images = bindings.accesses.iter().filter(|a| a.is_image()).map(|a| a.image_use(stage));
    let buffers = bindings.accesses.iter().filter(|a| !a.is_image()).map(|a| a.buffer_access(stage))
        .chain(indirect_buffer.map(indirect_access)); // gap 4
    self.flush_pending(images, buffers);
    // ...
}

// Gaps: four cases in today's code that the header's two bits don't cover:
// 1. An image with one view (`Single`: render targets, textures) has ONE tracked layout for
//    all its mips, so the mip range doesn't give the length of the layout array: a whole view
//    of a 10-mip texture has `layouts.len() == 1`. Only a `MipChain` has one per mip.
//    -> ONE_LAYOUT bit.
// 2. Depth images are sampled: Composite reads the depth attachment as `Tex2D`, and barriers
//    of a depth image need its depth (and, for depth-stencil formats, stencil) aspect.
//    -> DEPTH/STENCIL bits (format.aspects()).
// 3. A trace reads its TLAS (`Accel` fields, `acceleration_structure_read`) with
//    ACCELERATION_STRUCTURE_READ | MEMORY_READ, which SHADER_READ doesn't cover: without it
//    a TLAS built earlier in the command buffer gets no barrier before the trace.
//    -> ACCELERATION_STRUCTURE bit; the "every structure" read stays address u64::MAX, size 0.
// 4. `draw_indexed`, `draw_indirect(_count)`, `launch_indirect`, `compute_indirect` and
//    `raytrace_indirect` push their index/indirect buffer into the binding's buffers today,
//    at INDEX_INPUT / DRAW_INDIRECT with INDEX_READ / INDIRECT_COMMAND_READ, not the pass's
//    stage. -> they stay full `BufferAccess`es next to the command (`Draw.extra:
//    SmallVec<[BufferAccess; 2]>`, chained in `compute_private`/`raytrace_private` as
//    already above), instead of joining the binding's list.
*/

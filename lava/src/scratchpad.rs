//! API sketches for the "find solutions" entries of STATE.md that touch lava (pseudo-Rust in a
//! block comment, so it needn't compile). The reasoning is in the chat summary.
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
specialized_compute_pipelines: [Mutex<HashMap<Constants, vk::Pipeline>>; NUM_SPECIALIZED_COMPUTE_PIPELINES],
raster_pipelines: [Mutex<HashMap<RasterHash, vk::Pipeline>>; NUM_RASTER_PIPELINES],

// ------------------------------------------------------------------------------------------
// 2. `PendingAccesses` is never pruned of destroyed resources; each command scans pending x accesses.
// ------------------------------------------------------------------------------------------
// (a) Prune through the drop hooks lava already has (`image::on_drop`, `buffer::on_drop`):
//     the hook records the dropped handle in a global list, and `Frame::execute` drains it
//     into the `PendingAccesses` it hands back, removing every entry of those handles.
//     Buffers are tracked by address range, so `buffer::on_drop` passes the erased buffer and
//     its address range is removed.
static DROPPED: Mutex<(Vec<vk::Image>, Vec<Range<u64>>)>;
impl PendingAccesses {
    fn prune(&mut self, dropped: &(Vec<vk::Image>, Vec<Range<u64>>)) {
        self.image_reads.retain(|r| !dropped.0.contains(&r.image));
        self.image_writes.retain(|w| !dropped.0.contains(&w.write.image));
        self.buffer_reads.retain(|r| !dropped.1.iter().any(|d| d.overlaps(&r.range)));
        // ...
    }
}
// (b) The scan: pending holds about a dozen entries per frame in core (a few images, the
//     render buffers), so pending x accesses is tens of comparisons per command. It only
//     grows with leaked entries, which (a) stops. If it ever matters, the lists can be keyed
//     by resource (a SmallVec map from `vk::Image` / buffer address to its last access),
//     which makes the lookup per access O(1) and the retention in (a) a `remove`.

// ------------------------------------------------------------------------------------------
// 3. One keepalive for every lava object: a slab of (count, layout) words, objects stay
//    inline in their owners, views and slices point at the word instead of borrowing.
//    Replaces `Frame::retire`, the hot-reload `keep` list and core's "retire a grown image
//    before its slot is reused" rule. (STATE.md, Ideas. Reasoning in the chat of 2026-10-13.)
// ------------------------------------------------------------------------------------------

// --- The slab -----------------------------------------------------------------------------
// One 8-byte word per object: the count of handles alive (owner, views, slices, frame-slot
// uses) and, for images, the tracked layout, which today lives in `Single.layout` /
// `PerMip.layouts` and every view borrows. Chunks of 1024 are never moved or freed, so a
// pointer stays valid for the process' life; words are reused through a free list once their
// count hits 0 (a word is never reused while a handle holds it, so no generation). The first
// chunk comes with `lava::init`, more under the mutex when the free list runs dry (object
// creation, not the hot path). Nothing in here needs a `Ctx`, so the unit tests use it too.
#[repr(C, align(8))]
struct Word {
    /// Low 31 bits: handles alive. High bit: the owner is gone and the object waits in
    /// `ORPHANS` (see below). 0 = free.
    count: AtomicU32,
    /// `vk::ImageLayout` as u32; unused for buffers, swapchains and pipelines.
    layout: AtomicU32,
}
const ORPHANED: u32 = 1 << 31;
struct Slab {
    chunks: Mutex<Vec<Box<[Word; 1024]>>>,
    /// Runs of `n` consecutive words for a `MipChain` (one layout per mip, the first word's
    /// count is the object's); singles for everything else.
    free: Mutex<Vec<(NonNull<Word>, u32)>>,
}
static SLAB: Slab;
fn alloc(words: u32) -> NonNull<Word>;      // count = 1 (the owner's), layouts UNDEFINED
fn release(word: NonNull<Word>, words: u32); // onto the free list, by the destroyer

// --- A handle on the word --------------------------------------------------------------
// `Clone` = Relaxed increment, `Drop` = Release decrement; the decrement that reaches
// `ORPHANED | 0` destroys (Acquire fence first, like `Arc`): it takes the object out of
// `ORPHANS` and runs its destruction there. Any thread may be the one (a frame slot's
// release on the render thread is the common case; Vulkan allows destruction anywhere and
// the allocator is behind a `Mutex`).
#[derive(Clone)]
pub(crate) struct Count(NonNull<Word>);

// --- Owners stay what they are ----------------------------------------------------------
// `Image`, `Buffer`, `Swapchain` and (hot reload) `GenericPipeline` keep their handles,
// allocation, views and bindless handles inline and gain one field: `count: Count` (its
// first word; a `MipChain` owns a run of `mip_levels` words). The only change is `Drop`:
impl Drop for Image<..> {
    fn drop(&mut self) {
        // Nobody else holds it (fast path: 1 -> 0, destroy here, at once, as today).
        if self.count.0.count.fetch_or(ORPHANED, AcqRel) & !ORPHANED == 1 { destroy(self); return; }
        // Frames or views still use it: move the Vulkan parts out and park them. The
        // handle that drops last destroys them (`Count::drop` above).
        ORPHANS.lock().push((self.count.0, Orphan::Image {
            image, allocation: take(&mut self.allocation), views, handles, words: mip_levels,
        }));
    }
}
// (The `fetch_or` sets the bit and tells us the old count in one atomic, so a decrement
// racing with it either sees the bit, and destroys, or doesn't, and we do.)
static ORPHANS: Mutex<Vec<(NonNull<Word>, Orphan)>>;  // short: things dropped mid-flight
enum Orphan {
    Image { image: vk::Image, allocation: Allocation, views: SmallVec<[vk::ImageView; 1]>,
            handles: SmallVec<[BindlessHandle; 1]>, words: u32 },
    Buffer { handle: vk::Buffer, allocation: Allocation },
    Swapchain { handle: vk::SwapchainKHR, render_finished: Vec<vk::Semaphore> },
    #[cfg(feature = "hot-reload")] Pipeline(GenericPipeline),
}
// Destruction is one function for both paths: Vulkan destroy calls, the allocation back,
// the `on_drop` hooks with the bindless handles (so core's free list only ever gets a slot
// no pending frame reads), the handle or range onto the `DROPPED` list that prunes
// `PendingAccesses` (proposal 2), then `release(word, words)`.

// --- Views and slices: no lifetime, one count each -------------------------------------
pub struct ImageView<F, U, D> {          // was ImageView<'a, F, U, D>
    count: Count,                        // the image's first word; mip i's layout is word i
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub mip_range: Range<u32>,
    pub extent: D::Extent,
    pub handle: BindlessHandle,
    format: F, usage: U, dim: D,
}
impl ImageView { fn layout(&self, mip: u32) -> &AtomicU32 { &word(self.count, mip).layout } }
pub struct BufferSlice<T, U> {           // was BufferSlice<'a, T, U>, Copy
    count: Count,
    address: u64, size: u64, usage: U,
    _marker: PhantomData<[T]>,
}
// Both `Clone` (an increment), not `Copy`. `whole_view()`, `mip_view(i)`, `range(..)` cost
// one Relaxed increment and one Release decrement each; core makes a few dozen per frame.
// A view may live in a resource across frames and keeps the image from being destroyed,
// which is what the lifetime guaranteed. Host access stays on the owner (`host`,
// `host_mut`, `host_ptr`), so `&mut Buffer` is still the only way at the mapping.
// A swapchain image gets a word per image (the swapchain owns the run), its views count
// that word, and the swapchain's own word is counted by `present`; `recreate` drops the
// old swapchain, which parks itself until the last frame that drew or presented it
// releases. Core's `Swapchain.retired` and its semaphore bookkeeping go.

// --- Command buffers and frame slots ------------------------------------------------------
pub struct BindingOutput<K> { images: SmallVec<[ImageUse; ..]>, .. }   // no 'a, and no
pub struct RasterBuilder<DRAWS, COLOR_ATTACHMENTS> { .. }               // `erase` transmute
pub struct CommandBuffer {
    /// Every object an access named, once (`ptr_eq` against the recent ones; the same
    /// image is named by many passes). Moved to the slot on submit.
    used: Vec<Count>,
}
// `flush_pending` pushes each access's `count.clone()`; `keep` the pipeline's; `present`
// the swapchain image's and the swapchain's.
pub struct FrameSlot {
    /// The last submission's `used`: cleared (= decremented) right after its fence in
    /// `begin`, where `retired` and `pipelines` are cleared today. Reused: no allocation.
    in_flight: Vec<Count>,
}
impl Frame<'_> {
    /// What commands can't see: a buffer reached by address through another buffer (a
    /// mesh, a material), an image reached through a bindless handle in a buffer. One
    /// count until this frame's fence. Replaces `retire`, which goes.
    pub fn uses(&mut self, object: &impl Counted);    // Image, Buffer, ImageView, BufferSlice
}
// core: the instance extraction `uses` each instance's mesh buffer and material slot (a
// few thousand Relaxed increments per frame; measure). Textures are reachable by every
// frame through their slots, so they stay a convention (a `GpuTexture` dropped mid-frame)
// until streaming makes the per-frame texture set explicit.

// --- Prior art --------------------------------------------------------------------------
// The count outside the object, the object inline: Objective-C's runtime keeps refcounts in
// a global side table keyed by object pointer (`SideTable`, and Swift's `RefCount` spills
// to one), `boost::intrusive_ptr`/`triomphe` put the count inside the object instead; a
// frame slot holding counts until its fence is epoch-based reclamation (crossbeam-epoch:
// a pinned epoch defers frees until every reader of it is gone), and the parked `ORPHANS`
// are its garbage list. Vulkan frameworks mostly do the frame-stamp version: Granite and
// Unreal's RHI queue a dropped resource under the current frame number and free it when
// that frame's fence passes; wgpu records a last-used submission index per resource.
// The refcount version handles several frame loops (windowed and headless, a second
// queue) without a shared frame counter, which is why it fits lava better.

// --- Open points ------------------------------------------------------------------------
// - `Copy` is lost on `BufferSlice`: call sites that reuse one need `.clone()` or a new
//   `range(..)`. Most make a fresh one per use already.
// - `Word` is 8 bytes, a chunk 8 KB; a `MipChain` of n mips takes a run of n words, so the
//   free list keeps runs by length (or just carves fresh runs and reuses singles).
// - Destruction may happen on a loader thread (an image dropped there with no frame using
//   it, as today) or on the render thread (parked objects, in frame order after the fence),
//   never while the GPU can touch the object.
// - The ORPHANS list is a `Mutex<Vec>` touched only when an owner dies mid-flight and when
//   a count reaches `ORPHANED | 0`; the lookup by word pointer is a scan of a short list.
*/

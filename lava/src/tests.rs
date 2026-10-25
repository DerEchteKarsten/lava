//! lava's unit tests: the barrier tracking, the host-access check, buffer memory locations, image states, image usage types, texel blocks, format components, the format table's texel sizes, copy sizes with row pitches, 64-bit clear values and format compatibility classes, runtime formats' and usages' checks, buffer and image cast checks, mip extents and boxes, the slot allocator (double frees included) and image handle bits, the staging ring's positions, the descriptor heap layout, feature detection (every feature's extensions detect it), feature prerequisites (`consistent`), `without`/`missing_from`, the capability scan and the manifest's cargo names, pipeline cache headers, raster pipeline keys (borrowed keys hash like owned ones) and baked draw state, queue family picking, device ranking, wait stage masks, error mapping, calls before init, the layouts the GPU reads (instances byte for byte, binding table regions), topology classes and render areas, half floats, copy region overlaps, and `unsized_struct!`'s read-back
use std::{
    iter,
    sync::{Arc, atomic::AtomicU64},
};

use crate::vk::{self, AccessFlags2 as A, Handle, ImageLayout as L, PipelineStageFlags2 as S};

use crate::{
    bindless::{Combined, HeapLayout, ImageHandle, Ranges, Sampled, Storage},
    buffer::staging::{Reservation, Ring, STAGING_ALIGN, Uploader},
    buffer::{BufferInternal, BufferMemory},
    command_buffer::{
        Barriers, BufferAccess, ImageAccess, ImageUse,
        access_state::{AccessState, MAX_WRITES, StructureAccesses, Tracker},
        compute_barriers,
        image_state::{ImageState, check_states, release_states, released_by, take_checked_states},
        made_visible, record, walk,
    },
    features::{DeviceFacts, FeatureChain, Features, Needs},
    global::Live,
    image::{
        ImageInternal, ImageMemory, ViewCache, ViewSlot,
        format::{self, Block, ClearValue, Format},
        slice::rows_len,
        usage::{ImageUsage, StaticUsage, Usage, usage},
    },
    object::{DIRTY_LAYOUT, Header, Object, Strong, mark_dirty},
    queue::{FamilyInfo, Kind},
    reference::{KeepAlive, Reference, Referenced, ReferencingData},
    vkobjects::physical_device::{QueueFamily, pick_queue_families},
};

const GRAPHICS: FamilyInfo = FamilyInfo {
    kind: Kind::Graphics,
    timestamp_valid_bits: 64,
    image_granularity: vk::Extent3D {
        width: 1,
        height: 1,
        depth: 1,
    },
    indices: [0, 1, 2],
};

/// The message `f` panics with; fails the test if it doesn't panic.
#[track_caller]
fn panic_message<R>(f: impl FnOnce() -> R) -> String {
    let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) else {
        panic!("it should panic");
    };
    (payload.downcast_ref::<String>().cloned())
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

thread_local! {
    /// The buffer of a test's accesses: every test runs on a thread of its own, so each gets a
    /// fresh one. Never dropped (its `Drop` needs a context).
    static BUFFER: &'static Arc<BufferInternal> = Box::leak(Box::new(Arc::new(BufferInternal {
        handle: vk::Buffer::null(),
        address: 0,
        num_bytes: 1 << 40,
        memory: BufferMemory::Allocation(Default::default()),
        state: Default::default(),
        keep_alive: Default::default(),
        header: Header::new(Live::internal()),
    })));
}

fn test_buffer() -> &'static Arc<BufferInternal> {
    BUFFER.with(|buffer| *buffer)
}

fn buffer(stage: S, access: A, range: std::ops::Range<u64>) -> BufferAccess<'static> {
    BufferAccess {
        unordered: false,
        stage,
        access,
        range: range.into(),
        buffer: Some(test_buffer()),
        #[cfg(feature = "acceleration-structure")]
        structure: None,
    }
}

fn read(range: std::ops::Range<u64>) -> BufferAccess<'static> {
    buffer(S::VERTEX_SHADER, A::SHADER_STORAGE_READ, range)
}

fn write(range: std::ops::Range<u64>) -> BufferAccess<'static> {
    buffer(S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE, range)
}

fn state(layout: L, owner: Option<Kind>) -> ImageState {
    ImageState {
        layout,
        owner,
        released_to: None,
        acquired: false,
    }
}

/// An image `id` with tracked states starting as `states`, never dropped (its `Drop` needs a
/// context).
fn tracked(id: u64, states: impl IntoIterator<Item = ImageState>) -> &'static Arc<ImageInternal> {
    Box::leak(Box::new(Arc::new(ImageInternal {
        image: vk::Image::from_raw(id),
        memory: ImageMemory::Allocation(Default::default()),
        format: vk::Format::R8G8B8A8_UNORM,
        aspects: vk::ImageAspectFlags::COLOR,
        usage: vk::ImageUsageFlags::SAMPLED,
        mip_levels: 1,
        layers: 1,
        create_flags: vk::ImageCreateFlags::empty(),
        cache: ViewCache::default(),
        whole: ViewSlot::new(vk::ImageView::null(), ImageHandle::NULL),
        mips: Vec::new(),
        states: states.into_iter().map(ImageState::atomic).collect(),
        access: Default::default(),
        header: Header::new(Live::internal()),
    })))
}

/// An access of image `id` with its own tracked state, which starts as `old`.
fn image(id: u64, stage: S, access: A, old: ImageState, layout: L) -> ImageUse<'static> {
    image_use(id, tracked(id, [old]), stage, access, layout)
}

/// An access of mip 0 of image `id`, whose tracked states are `internal`'s.
fn image_use<'a>(
    id: u64,
    internal: &'a Arc<ImageInternal>,
    stage: S,
    access: A,
    layout: L,
) -> ImageUse<'a> {
    ImageUse {
        access: ImageAccess {
            stage,
            access,
            image: vk::Image::from_raw(id),
            layout,
            aspect: vk::ImageAspectFlags::COLOR,
            mip_range: (0..1).into(),
        },
        internal,
    }
}

/// A recording of value `value` on timeline 0, which completed `completed`.
fn tracker(value: u64, completed: u64) -> Tracker {
    Tracker {
        timeline: 0,
        value,
        base: 0,
        completed,
    }
}

/// What a recording keeps while it tracks accesses: its tracker, its barrier log and its
/// timeline's acceleration structure accesses.
struct Rec {
    tracker: Tracker,
    log: Vec<Barriers>,
    structures: StructureAccesses,
}

impl Rec {
    fn new() -> Self {
        Self::at(tracker(1, 0))
    }

    fn at(tracker: Tracker) -> Self {
        Rec {
            tracker,
            log: Vec::new(),
            structures: StructureAccesses::default(),
        }
    }

    /// Tracks `images` and `buffers` without their barriers.
    fn record(&mut self, images: &[ImageUse], buffers: &[BufferAccess]) {
        record(
            &self.tracker,
            &self.log,
            &mut self.structures,
            images.iter().cloned(),
            buffers.iter().cloned(),
        );
    }
}

/// Records `images` and `buffers` after the barriers they need, like `Recording::flush_pending`.
fn flush_barriers(
    rec: &mut Rec,
    images: &[ImageUse],
    buffers: &[BufferAccess],
) -> Option<Barriers> {
    let barriers = compute_barriers(
        &rec.tracker,
        &rec.log,
        &mut rec.structures,
        &GRAPHICS,
        images.iter().cloned(),
        buffers.iter().cloned(),
    );
    if let Some(barriers) = &barriers {
        rec.log.push(barriers.clone());
        made_visible(&rec.tracker, &rec.log, images.iter().cloned());
    }
    rec.record(images, buffers);
    barriers
}

/// `flush_barriers`' memory barrier's source and its number of image barriers.
fn flush(rec: &mut Rec, images: &[ImageUse], buffers: &[BufferAccess]) -> Option<(S, A, usize)> {
    flush_barriers(rec, images, buffers).map(|b| {
        let memory = b.memory.map_or((S::NONE, A::NONE), |m| {
            (m.src_stage_mask, m.src_access_mask)
        });
        (memory.0, memory.1, b.images.len())
    })
}

#[test]
fn buffer_hazards_wait_for_the_overlapping_accesses() {
    let mut pending = Rec::new();
    assert_eq!(
        flush(&mut pending, &[], &[write(0..100)]),
        None,
        "nothing pending"
    );
    // Read after write, from the writer's stage.
    assert_eq!(
        flush(&mut pending, &[], &[read(10..20)]),
        Some((S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE, 0))
    );
    // The barrier made the write visible to that stage and access already.
    assert_eq!(flush(&mut pending, &[], &[read(50..60)]), None);
    // Disjoint and adjacent ranges don't conflict, read after read neither.
    assert_eq!(flush(&mut pending, &[], &[write(100..200)]), None);
    // Write after read waits for the reader (and the older write).
    assert_eq!(
        flush(&mut pending, &[], &[write(0..15)]),
        Some((
            S::VERTEX_SHADER | S::COMPUTE_SHADER,
            A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE,
            0
        ))
    );
}

/// A partly overlapping write keeps the older write pending; reads of completed submissions
/// are dropped.
#[test]
fn tracked_accesses_shrink_only_when_covered_or_completed() {
    let mut state = AccessState::default();
    let t = tracker(1, 0);
    let (write, transfer) = (
        (S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE),
        (S::TRANSFER, A::TRANSFER_WRITE),
    );
    state.record(&t, &[], (0..100).into(), write.0, write.1);
    state.record(&t, &[], (50..60).into(), transfer.0, transfer.1);
    assert_eq!(state.writes.len(), 2);
    state.record(&t, &[], (0..100).into(), write.0, write.1);
    assert_eq!(state.writes.len(), 1, "a covering write replaces both");
    state.record(
        &tracker(2, 0),
        &[],
        (0..10).into(),
        S::VERTEX_SHADER,
        A::SHADER_STORAGE_READ,
    );
    state.sources(&tracker(3, 1));
    assert_eq!(state.reads.len(), 1);
    state.sources(&tracker(3, 2));
    assert!(state.reads.is_empty());
}

/// Each timeline sees only its own accesses, and a timeline that took an index over ignores
/// what its former owners left (`Tracker::base`).
#[test]
fn timelines_track_their_own_accesses() {
    let mut state = AccessState::default();
    let write = |t: &Tracker, state: &mut AccessState| {
        state.record(
            t,
            &[],
            (0..16).into(),
            S::COMPUTE_SHADER,
            A::SHADER_STORAGE_WRITE,
        )
    };
    let read_hazards = |t: &Tracker, state: &mut AccessState| {
        state.hazards(
            t,
            &[],
            (0..16).into(),
            S::COMPUTE_SHADER,
            A::SHADER_STORAGE_READ,
        )
    };
    let other = Tracker {
        timeline: 1,
        ..tracker(1, 0)
    };
    write(&other, &mut state);
    assert_eq!(
        read_hazards(&tracker(1, 0), &mut state),
        None,
        "another timeline's"
    );
    assert!(read_hazards(&other, &mut state).is_some());
    let next_owner = Tracker {
        base: 1,
        ..tracker(2, 1)
    };
    let next_owner = Tracker {
        timeline: 1,
        ..next_owner
    };
    assert_eq!(
        read_hazards(&next_owner, &mut state),
        None,
        "a former owner's"
    );
    assert!(state.writes.is_empty());
}

/// A barrier's visibility outlives its recording's log: a buffer written and read once in one
/// recording needs no barrier for the same read in the next.
#[test]
fn visibility_outlives_its_recording() {
    let mut first = Rec::new();
    flush(&mut first, &[], &[write(0..16)]);
    assert!(flush(&mut first, &[], &[read(0..16)]).is_some());
    let mut next = Rec::at(tracker(2, 0));
    assert_eq!(flush(&mut next, &[], &[read(0..16)]), None);
}

/// Two barriers, to sampled reads in fragment shaders and to storage reads in compute shaders,
/// don't make a write visible to sampled reads in compute shaders.
#[test]
fn write_visibility_is_per_barrier() {
    let image = tracked(1, [state(L::GENERAL, Some(Kind::Graphics))]);
    let at = |stage, access| image_use(1, image, stage, access, L::GENERAL);
    let mut pending = Rec::new();
    flush(
        &mut pending,
        &[at(S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE)],
        &[],
    );
    assert!(
        flush(
            &mut pending,
            &[at(S::FRAGMENT_SHADER, A::SHADER_SAMPLED_READ)],
            &[]
        )
        .is_some()
    );
    assert!(
        flush(
            &mut pending,
            &[at(S::COMPUTE_SHADER, A::SHADER_STORAGE_READ)],
            &[]
        )
        .is_some()
    );
    assert!(
        flush(
            &mut pending,
            &[at(S::COMPUTE_SHADER, A::SHADER_SAMPLED_READ)],
            &[]
        )
        .is_some()
    );
    assert!(
        flush(
            &mut pending,
            &[at(S::FRAGMENT_SHADER, A::SHADER_SAMPLED_READ)],
            &[]
        )
        .is_none()
    );
}

/// Mips in different states get one transition per run of neighbours with the same old state.
#[test]
fn image_states_move_in_runs_of_mips() {
    let g = |layout| state(layout, Some(Kind::Graphics));
    let states: Vec<AtomicU64> = [
        ImageState::UNDEFINED,
        ImageState::UNDEFINED,
        g(L::GENERAL),
        g(L::ATTACHMENT_OPTIMAL),
    ]
    .map(ImageState::atomic)
    .into();
    let mut runs = Vec::new();
    take_checked_states(
        &states,
        (0..4).into(),
        L::GENERAL,
        Kind::Graphics,
        |mips, old| runs.push((mips.start, mips.end, old.layout)),
    );
    assert_eq!(runs, [(0, 2, L::UNDEFINED), (3, 4, L::ATTACHMENT_OPTIMAL)]);
    assert_eq!(ImageState::load(&states[0]), g(L::GENERAL));
    for s in [g(L::PRESENT_SRC_KHR), state(L::GENERAL, None)] {
        assert_eq!(ImageState::unpack(s.pack()), s);
    }
}

/// An image another family owns can't be accessed until its owner released it, and the
/// receiver's first access acquires it in a barrier call of its own.
#[test]
fn ownership_moves_through_release_and_acquire() {
    let states = [state(L::GENERAL, Some(Kind::Compute)).atomic()];
    let message = panic_message(|| check_states(&states, (0..1).into(), Kind::Graphics));
    assert!(
        message.contains("the Compute family owns is accessed on the Graphics family"),
        "{message}"
    );
    let mut released = Vec::new();
    release_states(
        &states,
        (0..1).into(),
        Kind::Compute,
        Kind::Graphics,
        |mips, layout| released.push((mips.start, layout)),
    );
    assert_eq!(released, [(0, L::GENERAL)]);
    check_states(&states, (0..1).into(), Kind::Graphics);

    let access = image(
        3,
        S::FRAGMENT_SHADER,
        A::SHADER_SAMPLED_READ,
        ImageState::load(&states[0]),
        L::GENERAL,
    );
    let barriers = compute_barriers(
        &tracker(1, 0),
        &[],
        &mut StructureAccesses::default(),
        &GRAPHICS,
        iter::once(access.clone()),
        iter::empty(),
    )
    .unwrap();
    let [acquire] = barriers.acquires.as_slice() else {
        panic!("one acquire")
    };
    assert_eq!(
        (
            acquire.src_queue_family_index,
            acquire.dst_queue_family_index
        ),
        (1, 0)
    );
    assert!(barriers.images.is_empty(), "same layout");
    assert_eq!(
        ImageState::load(&access.states()[0]).owner,
        Some(Kind::Graphics)
    );
}

/// Releasing one mip of a chain keeps the image's pending accesses for the mips the family
/// still owns (`released_by`); they go once every mip is released.
#[test]
fn releasing_one_mip_keeps_the_others() {
    let states: Vec<AtomicU64> = [
        state(L::GENERAL, Some(Kind::Graphics)),
        state(L::GENERAL, Some(Kind::Graphics)),
        ImageState::UNDEFINED,
    ]
    .map(ImageState::atomic)
    .into();
    let mut released = Vec::new();
    release_states(
        &states,
        (1..2).into(),
        Kind::Graphics,
        Kind::Transfer,
        |mips, _| released.push((mips.start, mips.end)),
    );
    assert_eq!(released, [(1, 2)]);
    assert_eq!(ImageState::load(&states[0]).released_to, None);
    assert!(
        !released_by(&states, Kind::Graphics),
        "mip 0 is still owned"
    );
    release_states(
        &states,
        (0..1).into(),
        Kind::Graphics,
        Kind::Transfer,
        |_, _| {},
    );
    assert!(
        released_by(&states, Kind::Graphics),
        "mip 2 is unowned, the others released"
    );
}

/// A layout transition is the image's latest write, at its barrier's destination: a later read
/// at another stage waits for that stage, which chains after the transition, not for the
/// attachment write before it.
#[test]
fn transitions_are_writes_at_their_destination() {
    let image = tracked(1, [ImageState::UNDEFINED]);
    let at = |stage, access, layout| image_use(1, image, stage, access, layout);
    let mut pending = Rec::new();
    let target = at(
        S::COLOR_ATTACHMENT_OUTPUT,
        A::COLOR_ATTACHMENT_WRITE,
        L::ATTACHMENT_OPTIMAL,
    );
    flush(&mut pending, &[target], &[]);
    let sampled = flush_barriers(
        &mut pending,
        &[at(S::FRAGMENT_SHADER, A::SHADER_SAMPLED_READ, L::GENERAL)],
        &[],
    )
    .unwrap();
    let [transition] = sampled.images.as_slice() else {
        panic!("one transition")
    };
    assert_eq!(
        (transition.src_stage_mask, transition.dst_stage_mask),
        (S::COLOR_ATTACHMENT_OUTPUT, S::FRAGMENT_SHADER)
    );
    {
        let access = image.access.lock().unwrap();
        let [write] = access.writes.as_slice() else {
            panic!("one write")
        };
        assert_eq!((write.stage, write.access), (S::FRAGMENT_SHADER, A::NONE));
        assert!(write.visible_to(S::FRAGMENT_SHADER, A::SHADER_SAMPLED_READ));
    }

    let compute = flush_barriers(
        &mut pending,
        &[at(S::COMPUTE_SHADER, A::SHADER_SAMPLED_READ, L::GENERAL)],
        &[],
    )
    .unwrap();
    let memory = compute.memory.expect("a memory barrier");
    assert_eq!(
        memory.src_stage_mask,
        S::FRAGMENT_SHADER,
        "chains after the transition"
    );
    assert!(compute.images.is_empty());
}

/// An acquire is a write as well: a copy after an acquired image was sampled waits for it. A
/// transition right after the acquire comes after the acquire's stages.
#[test]
fn acquires_are_writes_and_transitions_follow_them() {
    let released = ImageState {
        released_to: Some(Kind::Graphics),
        ..state(L::GENERAL, Some(Kind::Compute))
    };
    let image = tracked(2, [released]);
    let mut pending = Rec::new();
    let sampled = flush_barriers(
        &mut pending,
        &[image_use(
            2,
            image,
            S::FRAGMENT_SHADER,
            A::SHADER_SAMPLED_READ,
            L::GENERAL,
        )],
        &[],
    )
    .unwrap();
    assert_eq!(sampled.acquires.len(), 1);
    let copy = flush(
        &mut pending,
        &[image_use(
            2,
            image,
            S::TRANSFER,
            A::TRANSFER_READ,
            L::GENERAL,
        )],
        &[],
    );
    assert_eq!(
        copy.map(|(stage, ..)| stage),
        Some(S::FRAGMENT_SHADER),
        "the copy waits for the acquire"
    );

    // A stale access of the image on this timeline doesn't stand in for the acquire.
    let image = tracked(3, [released]);
    let mut pending = Rec::new();
    pending.record(
        &[image_use(
            3,
            image,
            S::COMPUTE_SHADER,
            A::SHADER_STORAGE_WRITE,
            L::GENERAL,
        )],
        &[],
    );
    let target = image_use(
        3,
        image,
        S::COLOR_ATTACHMENT_OUTPUT,
        A::COLOR_ATTACHMENT_WRITE,
        L::ATTACHMENT_OPTIMAL,
    );
    let barriers = flush_barriers(&mut pending, &[target], &[]).unwrap();
    let ([acquire], [transition]) = (barriers.acquires.as_slice(), barriers.images.as_slice())
    else {
        panic!("an acquire and a transition")
    };
    assert!(transition.src_stage_mask.contains(acquire.dst_stage_mask));
}

/// The family that released an image can't use it any more: the receiver owns it now.
#[test]
fn a_released_image_is_the_receivers() {
    let states = [state(L::GENERAL, Some(Kind::Graphics)).atomic()];
    release_states(
        &states,
        (0..1).into(),
        Kind::Graphics,
        Kind::Compute,
        |_, _| {},
    );
    let panic = std::panic::catch_unwind(|| check_states(&states, (0..1).into(), Kind::Graphics))
        .unwrap_err();
    let message = panic.downcast_ref::<String>().unwrap();
    assert!(
        message.contains("released to the Compute family"),
        "{message}"
    );
    check_states(&states, (0..1).into(), Kind::Compute);
}

/// An image no command accessed has nothing to wait for; one with no tracked accesses may still
/// be in use by a submission this one waited for; an acquired swapchain image waits for its
/// semaphore, not for the accesses before its present.
#[test]
fn transitions_wait_only_for_what_can_still_run() {
    let mut pending = Rec::new();
    let fresh = flush_barriers(
        &mut pending,
        &[image(
            4,
            S::TRANSFER,
            A::TRANSFER_WRITE,
            ImageState::UNDEFINED,
            L::GENERAL,
        )],
        &[],
    )
    .unwrap();
    assert_eq!(
        (
            fresh.images[0].src_stage_mask,
            fresh.images[0].src_access_mask
        ),
        (S::NONE, A::NONE)
    );
    assert!(fresh.memory.is_none());

    let used = state(L::ATTACHMENT_OPTIMAL, Some(Kind::Graphics));
    let old = flush_barriers(
        &mut pending,
        &[image(5, S::TRANSFER, A::TRANSFER_WRITE, used, L::GENERAL)],
        &[],
    )
    .unwrap();
    assert_eq!(old.images[0].src_stage_mask, S::ALL_COMMANDS);

    let image = tracked(6, [state(L::GENERAL, Some(Kind::Graphics))]);
    let states: &[AtomicU64] = &image.states;
    let target = |access| {
        image_use(
            6,
            image,
            S::COLOR_ATTACHMENT_OUTPUT,
            access,
            L::ATTACHMENT_OPTIMAL,
        )
    };
    flush(&mut pending, &[target(A::COLOR_ATTACHMENT_WRITE)], &[]);
    states[0].store(
        ImageState {
            acquired: true,
            ..ImageState::UNDEFINED
        }
        .pack(),
        std::sync::atomic::Ordering::Relaxed,
    );
    let acquired = flush_barriers(&mut pending, &[target(A::COLOR_ATTACHMENT_WRITE)], &[]).unwrap();
    assert!(
        acquired.memory.is_none(),
        "the acquire's semaphore covers the last frame's writes"
    );
    assert_eq!(
        acquired.images[0].src_stage_mask,
        S::COLOR_ATTACHMENT_OUTPUT
    );
    assert_eq!(acquired.acquired.len(), 1);
    // Its old contents are discarded: the transition starts from `UNDEFINED`, not from the
    // layout its last frame left.
    assert_eq!(acquired.images[0].old_layout, L::UNDEFINED);
}

/// Image reads of completed submissions are dropped like buffer reads, and once a buffer has
/// more than `MAX_WRITES` writes, its completed ones fold into one over all their ranges, which
/// keeps the visibility all of them had (barriers logged since they were recorded included).
#[test]
fn completed_accesses_retire_and_folds_keep_their_visibility() {
    let sampled = image(
        7,
        S::FRAGMENT_SHADER,
        A::SHADER_SAMPLED_READ,
        state(L::GENERAL, Some(Kind::Graphics)),
        L::GENERAL,
    );
    Rec::at(tracker(3, 0)).record(&[sampled.clone()], &[]);
    let mut access = sampled.internal.access.lock().unwrap();
    access.sources(&tracker(4, 2));
    assert_eq!(access.reads.len(), 1);
    access.sources(&tracker(4, 3));
    assert!(access.reads.is_empty());
    drop(access);

    let mut pending = Rec::new();
    for i in 0..=MAX_WRITES as u64 {
        flush(&mut pending, &[], &[write(i * 16..i * 16 + 16)]);
        flush(&mut pending, &[], &[read(i * 16..i * 16 + 16)]);
    }
    // The next recording, once the first completed: its write folds the completed ones.
    let end = (MAX_WRITES as u64 + 1) * 16;
    let mut next = Rec::at(tracker(2, 1));
    flush(&mut next, &[], &[write(end..end + 16)]);
    {
        let state = test_buffer().state.lock().unwrap();
        let [_, folded] = state.writes.as_slice() else {
            panic!("the new write and the folded one")
        };
        assert_eq!((folded.range.start, folded.range.end), (0, end));
        assert!(folded.visible_to(S::VERTEX_SHADER, A::SHADER_STORAGE_READ));
    }
    assert_eq!(
        flush(&mut next, &[], &[read(0..16)]),
        None,
        "already visible"
    );
    assert!(
        flush(
            &mut next,
            &[],
            &[buffer(S::TRANSFER, A::TRANSFER_READ, 0..16)]
        )
        .is_some()
    );
}

/// The ring hands out positions at its head: a reservation that doesn't fit before the tail
/// fails, wrapping pads to the ring's end, the tail follows what is in use, and reclaiming
/// never passes a writer still counted in.
#[test]
fn staging_ring_positions_wrap_fill_and_reclaim() {
    let ring = Ring::new(64);
    let reserve = |len| ring.reserve(len).map(|r| (r.range, r.offset));
    assert_eq!(reserve(32), Some((0..32, 0)));
    assert_eq!(reserve(16), Some((32..48, 32)));
    assert_eq!(reserve(16), Some((48..64, 48)), "fits exactly, no padding");
    assert_eq!(reserve(16), None, "full");
    assert_eq!(ring.used(), 64);
    ring.count_out();
    ring.count_out();
    ring.count_out();
    // The first region is still in use: the tail moves to its start, which frees nothing.
    ring.reclaim([0, 48].into_iter());
    assert_eq!(reserve(16), None);
    ring.reclaim([48].into_iter());
    assert_eq!(ring.used(), 16);
    assert_eq!(reserve(32), Some((64..96, 0)), "wrapped to offset 0");
    ring.count_out();
    // Nothing in use and no writer: the tail reaches the head.
    ring.reclaim(iter::empty());
    assert_eq!(ring.used(), 0);
    assert_eq!(reserve(16), Some((96..112, 32)));
    ring.count_out();
    ring.reclaim(iter::empty());
    // 32 bytes don't fit the 16 left before the end: the reservation includes that padding.
    assert_eq!(
        reserve(64),
        None,
        "the padding leaves no room for a whole ring"
    );
    assert_eq!(reserve(32), Some((112..160, 0)));
    ring.count_out();
    ring.reclaim(iter::empty());
    assert_eq!(ring.used(), 0);

    // A writer counted in pins the tail at its reservation (its region isn't published yet),
    // also when later writers published theirs or nothing else is in use.
    let writing = ring.reserve(16).unwrap();
    assert_eq!(
        writing,
        Reservation {
            range: 160..176,
            offset: 32
        }
    );
    let published = ring.reserve(16).unwrap();
    ring.count_out();
    ring.reclaim([published.range.start].into_iter());
    assert_eq!(ring.used(), 32, "the tail stayed at the writer's start");
    ring.reclaim(iter::empty());
    assert_eq!(ring.used(), 32);
    ring.count_out();
    ring.reclaim(iter::empty());
    assert_eq!(ring.used(), 0);
    assert_eq!(STAGING_ALIGN, 16);
}

#[test]
fn usages_are_their_flags_in_any_order() {
    use vk::ImageUsageFlags as U;
    fn same<T>(_: T, _: T) {}
    same(
        <usage![TransferDst, Sampled]>::default(),
        <usage![Sampled, TransferDst]>::default(),
    );
    same(
        <usage![Storage]>::default(),
        Usage::<false, false, false, true, false, false, false, false>,
    );
    assert!(<usage![Storage]>::default().create_flags().is_empty());
    assert_eq!(
        <usage![Storage, Sampled, MutableFormat]>::default().create_flags(),
        vk::ImageCreateFlags::MUTABLE_FORMAT | vk::ImageCreateFlags::EXTENDED_USAGE
    );
    assert_eq!(
        <usage![Storage, MutableFormat]>::VK,
        U::STORAGE,
        "no usage flag of its own"
    );
    assert_eq!(
        <usage![TransferDst, Sampled]>::VK,
        U::SAMPLED | U::TRANSFER_DST
    );
    assert_eq!(
        <usage![ColorAttachment, Storage, TransferSrc, TransferDst]>::default().flags(),
        U::COLOR_ATTACHMENT | U::STORAGE | U::TRANSFER_SRC | U::TRANSFER_DST
    );
    assert_eq!(
        <usage![DepthAttachment, Sampled]>::VK,
        U::DEPTH_STENCIL_ATTACHMENT | U::SAMPLED
    );
    #[cfg(feature = "host-image-copy")]
    assert_eq!(
        <usage![HostCopy, Sampled]>::VK,
        U::SAMPLED | U::HOST_TRANSFER_EXT
    );
    assert_eq!(
        format!("{:?}", <usage![TransferSrc]>::default()),
        "Usage(TRANSFER_SRC)"
    );
}

#[test]
fn bindless_slots_are_first_fit_and_freed_ranges_merge() {
    let mut ranges = Ranges::default();
    ranges.give(0..10);
    assert_eq!((ranges.take(3), ranges.take(4)), (Some(0), Some(3)));
    assert_eq!(ranges.take(4), None);
    ranges.give(0..3);
    assert_eq!(ranges.take(2), Some(0), "the first hole that fits");
    for slot in [1, 0] {
        ranges.give(slot..slot + 1);
    }
    ranges.give(3..7);
    assert_eq!(
        ranges.take(10),
        Some(0),
        "freed one at a time, the slots merged back"
    );
}

#[test]
fn heaps_hold_their_descriptors_then_the_reserved_range() {
    // Intel's resource heap: 64-byte descriptors, nothing reserved, 2 GiB at most.
    let heap = HeapLayout::new(64, 64, 1 << 16, 0, 64, 1 << 31);
    assert_eq!((heap.stride, heap.count), (64, 1 << 16));
    assert_eq!(heap.offset(3), 3 * 64);
    assert_eq!((heap.reserved_offset, heap.size), (64 << 16, 64 << 16));

    // Strides round up to the descriptor alignment, the reserved range to its own.
    let heap = HeapLayout::new(48, 32, 10, 100, 512, 1 << 20);
    assert_eq!((heap.stride, heap.count), (64, 10));
    assert_eq!((heap.reserved_offset, heap.size), (1024, 1124));

    // A small heap holds fewer descriptors, so the padding and reserved range still fit.
    let heap = HeapLayout::new(64, 64, 1 << 16, 1000, 256, 1 << 16);
    assert_eq!(heap.count, 1004);
    assert_eq!(heap.reserved_offset % 256, 0);
    assert!(heap.reserved_offset >= heap.offset(heap.count) && heap.size <= 1 << 16);
}

#[test]
fn image_handles_encode_their_slots_and_kind() {
    let sampled = ImageHandle::<Sampled>::new(5);
    assert_eq!(sampled.bits(), 5);
    assert_eq!(
        (sampled.sampled_slot(), sampled.storage_slot()),
        (Some(5), None)
    );
    let storage = ImageHandle::<Storage>::new(5);
    assert_eq!(storage.bits(), 5 | 1 << 30);
    assert_eq!(
        (storage.sampled_slot(), storage.storage_slot()),
        (None, Some(5))
    );
    let combined = ImageHandle::<Combined>::new(5);
    assert_eq!(combined.bits(), 5 | 1 << 31);
    assert_eq!(
        (combined.sampled_slot(), combined.storage_slot()),
        (Some(5), Some(6))
    );

    // Mips are a handle's width apart, and keep its kind.
    assert_eq!(sampled.mip(3).sampled_slot(), Some(8));
    assert_eq!(storage.mip(3).storage_slot(), Some(8));
    let mip = combined.mip(3);
    assert_eq!(
        (mip.sampled_slot(), mip.storage_slot()),
        (Some(11), Some(12))
    );

    // Retyping keeps the bits, so a combined handle read as either kind finds its descriptor.
    assert_eq!(combined.as_sampled().bits(), combined.bits());
    assert_eq!(combined.as_storage().storage_slot(), Some(6));
    assert_eq!(
        combined.erase().narrow::<Sampled>(),
        Some(combined.as_sampled())
    );
    assert_eq!(
        combined.erase().narrow::<Storage>(),
        Some(combined.as_storage())
    );
    assert_eq!(sampled.erase().narrow::<Storage>(), None);
    assert_eq!(sampled.erase().narrow::<Combined>(), None);
    assert_eq!(storage.erase().narrow::<Sampled>(), None);

    let null = ImageHandle::<Combined>::NULL;
    assert_eq!(null.bits(), 0);
    let message = panic_message(|| ImageHandle::<Sampled>::new(0));
    assert!(message.contains("slot 0 is reserved"), "{message}");
    assert!(<ImageHandle<Combined> as bytemuck::Zeroable>::zeroed() == null);
    assert!(null.is_null() && ImageHandle::<Sampled>::default().is_null());
    assert_eq!((null.sampled_slot(), null.storage_slot()), (None, None));
    assert_eq!(null.erase().narrow::<Sampled>(), None);
    assert!(
        !ImageHandle::<Sampled>::new((1 << 30) - 1).is_null(),
        "the last slot is a handle"
    );
}

#[test]
fn passes_need_the_features_of_their_capabilities() {
    let capability = |id: u32| [2 << 16 | 17, id];
    let header = [0x0723_0203u32, 0x0001_0600, 0, 1, 0];
    // An `OpExtension` of two words between them, then `OpMemoryModel`, which ends the scan.
    let extension = [3 << 16 | 10, 0x5f565053, 0];
    let words = (header.into_iter())
        .chain([1, 39].into_iter().flat_map(capability))
        .chain(extension)
        .chain(capability(5283))
        .chain([3 << 16 | 14, 0, 1])
        .chain(capability(4479));
    let spirv: Vec<u8> = words.flat_map(u32::to_le_bytes).collect();
    let capabilities: Vec<u32> = crate::features::spirv_capabilities(&spirv).collect();
    assert_eq!(
        capabilities,
        [1, 39, 5283],
        "the scan stops after the capabilities"
    );
    // `Shader` (1) needs nothing, `Int8` (39) `shader-int8`, `MeshShadingEXT` (5283)
    // `mesh-shader`; `RayTracingKHR` (4479) comes after the scan stopped.
    let mut expected = Needs {
        features: Features::NONE,
        compiled_out: Vec::new(),
    };
    #[cfg(feature = "shader-int8")]
    {
        expected.features.shader_int8 = true;
    }
    #[cfg(not(feature = "shader-int8"))]
    expected.compiled_out.push("shader-int8");
    #[cfg(feature = "mesh-shader")]
    {
        expected.features.mesh_shader = true;
    }
    #[cfg(not(feature = "mesh-shader"))]
    expected.compiled_out.push("mesh-shader");
    assert_eq!(Features::of_spirv(&spirv), expected);
    assert_eq!(
        Features::of_spirv(&spirv).features,
        named(&["shader-int8", "mesh-shader"]),
        "every other feature stays off"
    );
}

/// What lava enables is what it requires, and enabling every optional feature reads back as
/// supported.
#[test]
fn enabled_features_read_back_as_supported() {
    let queried = FeatureChain::default();
    let enabling = |features: &Features| FeatureChain::enabling(features, &queried);
    assert!(
        enabling(&Features::default())
            .lava_missing(&|_| true)
            .is_empty()
    );
    let all = Features::from_bits(iter::repeat(true));
    let facts = DeviceFacts {
        rebar: true,
        strict_lines: true,
        dedicated_compute: true,
        dedicated_transfer: true,
        sparse_queue: true,
    };
    assert_eq!(enabling(&all).optional(&|_| true, facts), all.consistent());
}

#[test]
fn queue_families_are_graphics_and_the_dedicated_ones() {
    use vk::QueueFlags as Q;
    let family = |index, flags, num_queues, timestamps: bool| QueueFamily {
        index,
        properties: vk::QueueFamilyProperties {
            queue_flags: flags,
            queue_count: num_queues,
            timestamp_valid_bits: if timestamps { 64 } else { 0 },
            ..Default::default()
        },
    };
    let families = [
        family(0, Q::COMPUTE | Q::TRANSFER, 8, true),
        family(1, Q::GRAPHICS | Q::COMPUTE, 0, true),
        family(2, Q::GRAPHICS | Q::COMPUTE | Q::TRANSFER, 16, true),
        family(3, Q::TRANSFER, 2, false),
    ];
    let picked = pick_queue_families(&families).unwrap();
    assert_eq!(picked.graphics.index, 2, "family 1 has no queues");
    assert_eq!(picked.compute.unwrap().index, 0);
    assert_eq!(picked.transfer.unwrap().index, 3);
    assert!(pick_queue_families(&families[3..]).is_none());
}

#[cfg(feature = "ray-tracing-pipeline")]
#[test]
fn binding_table_regions_are_aligned_and_empty_ones_have_no_size() {
    use crate::vkobjects::rt_pipeline::sbt_layout;
    let layout = sbt_layout([1, 3, 0, 1], 20, 32, 64);
    assert_eq!(layout.stride, 32);
    assert_eq!(layout.regions, [(0, 32), (64, 96), (160, 0), (192, 32)]);
    assert_eq!(layout.size, 224);
}

#[cfg(feature = "acceleration-structure")]
#[test]
fn instances_have_the_layout_of_the_vulkan_struct() {
    use crate::vkobjects::acceleration_structure::{Instance, InstanceFlags, RawInstance};
    use std::mem::offset_of;
    assert_eq!(
        size_of::<Instance>(),
        size_of::<vk::AccelerationStructureInstanceKHR>()
    );
    assert_eq!(size_of::<RawInstance>(), size_of::<Instance>());
    assert_eq!(
        offset_of!(RawInstance, transform),
        offset_of!(vk::AccelerationStructureInstanceKHR, transform)
    );
    assert_eq!(
        offset_of!(RawInstance, custom_index_and_mask),
        offset_of!(
            vk::AccelerationStructureInstanceKHR,
            instance_custom_index_and_mask
        )
    );
    assert_eq!(
        offset_of!(RawInstance, sbt_offset_and_flags),
        offset_of!(
            vk::AccelerationStructureInstanceKHR,
            instance_shader_binding_table_record_offset_and_flags
        )
    );
    assert_eq!(
        offset_of!(RawInstance, blas),
        offset_of!(
            vk::AccelerationStructureInstanceKHR,
            acceleration_structure_reference
        )
    );
    let transform = glam::Affine3A::from_translation(glam::Vec3::new(1.0, 2.0, 3.0));
    let flags = InstanceFlags::FORCE_OPAQUE | InstanceFlags::TRIANGLE_FLIP_FACING;
    let instance = RawInstance::new(0xABC, transform)
        .with_custom_index(5)
        .with_mask(0x0F)
        .with_sbt_offset(7)
        .with_flags(flags);
    assert_eq!(instance.custom_index_and_mask, 0x0F00_0005);
    assert_eq!(instance.sbt_offset_and_flags, 0x0600_0007);
    assert_eq!(
        (
            instance.custom_index(),
            instance.mask(),
            instance.sbt_offset(),
            instance.flags()
        ),
        (5, 0x0F, 7, flags)
    );
    assert_eq!(
        (instance.blas_address(), instance.transform()),
        (0xABC, transform)
    );
    assert_eq!(
        flags.bits(),
        vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE.as_raw() as u8 | 2
    );
    assert!(
        flags.contains(InstanceFlags::FORCE_OPAQUE)
            && !flags.contains(InstanceFlags::FORCE_NO_OPAQUE)
    );

    // Byte for byte what ash's struct holds for the same values: the transform's rows, the
    // 24-bit fields in the low bits of their words, the 8-bit ones in the high, the address.
    let vulkan = vk::AccelerationStructureInstanceKHR {
        transform: vk::TransformMatrixKHR {
            matrix: [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 3.0],
        },
        instance_custom_index_and_mask: vk::Packed24_8::new(5, 0x0F),
        instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
            7,
            (vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE
                | vk::GeometryInstanceFlagsKHR::TRIANGLE_FLIP_FACING)
                .as_raw() as u8,
        ),
        acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
            device_handle: 0xABC,
        },
    };
    // SAFETY: a `repr(C)` struct of 64 bytes without padding, read as bytes.
    let vulkan_bytes = unsafe {
        std::slice::from_raw_parts(
            (&raw const vulkan).cast::<u8>(),
            size_of::<vk::AccelerationStructureInstanceKHR>(),
        )
    };
    assert_eq!(bytemuck::bytes_of(&instance), vulkan_bytes);
}

#[test]
fn half_floats_round_to_nearest_even() {
    use crate::half::{F16, f16_bits};
    let cases = [
        (0.0, 0x0000),
        (-0.0, 0x8000),
        (1.0, 0x3C00),
        (-2.0, 0xC000),
        (0.5, 0x3800),
        (65504.0, 0x7BFF),
        (65520.0, 0x7C00),
        (1e9, 0x7C00),
        (f32::INFINITY, 0x7C00),
        (2f32.powi(-14), 0x0400),
        (2f32.powi(-24), 0x0001),
        (2f32.powi(-25), 0x0000),
        (1.5 * 2f32.powi(-25), 0x0001),
        // 1 + 2^-11 is halfway between 1 and the next half: ties to even, 1.
        (1.0 + 2f32.powi(-11), 0x3C00),
        (1.0 + 3.0 * 2f32.powi(-11), 0x3C02),
        (0.1, 0x2E66),
    ];
    for (value, bits) in cases {
        assert_eq!(f16_bits(value), bits, "{value}");
    }
    assert_eq!(f16_bits(f32::NAN) & 0x7C00, 0x7C00);
    assert_ne!(f16_bits(f32::NAN) & 0x3FF, 0);
    // Back to `f32` exactly, subnormals and infinities too.
    for (value, bits) in [
        (0.0, 0),
        (-2.0, 0xC000),
        (0.5, 0x3800),
        (65504.0, 0x7BFF),
        (2f32.powi(-24), 1),
        (-(2f32.powi(-24)), 0x8001),
        (f32::INFINITY, 0x7C00),
        (0.099975586, 0x2E66),
    ] {
        assert_eq!(F16(bits).to_f32(), value, "{bits:#x}");
        assert_eq!(F16::new(value).0, bits);
    }
    assert!(F16::new(f32::NAN).to_f32().is_nan());
}

#[test]
fn copy_regions_may_not_write_what_others_read_or_write() {
    use crate::buffer::slice::check_copy_overlaps;
    let r = |a: u64, b: u64| -> std::range::Range<u64> { (a..b).into() };
    let check = |regions: &[(std::range::Range<u64>, std::range::Range<u64>)]| {
        std::panic::catch_unwind(|| check_copy_overlaps(regions.iter().copied()))
            .map_err(|e| e.downcast_ref::<String>().cloned().unwrap_or_default())
    };
    assert!(check(&[(r(0, 8), r(8, 16)), (r(16, 24), r(24, 32))]).is_ok());
    // Sources may overlap each other; touching ranges don't overlap.
    assert!(check(&[(r(0, 8), r(16, 24)), (r(4, 12), r(24, 32))]).is_ok());
    let message = check(&[(r(0, 8), r(16, 24)), (r(4, 12), r(20, 28))]).unwrap_err();
    assert!(message.contains("regions 0 and 1 overlap"), "{message}");
    let message = check(&[(r(0, 8), r(16, 24)), (r(20, 28), r(32, 40))]).unwrap_err();
    assert!(message.contains("regions 0 and 1 overlap"), "{message}");
    let message = check(&[(r(0, 8), r(4, 12))]).unwrap_err();
    assert!(message.contains("region 0 reads and writes"), "{message}");
    // A destination after a long source it overlaps.
    let message = check(&[(r(0, 100), r(200, 208)), (r(300, 308), r(50, 58))]).unwrap_err();
    assert!(message.contains("overlap"), "{message}");
}

#[test]
fn unsized_structs_read_back_every_field_and_section() {
    use std::marker::PhantomData;
    crate::unsized_struct::unsized_struct! {
        struct Sized<'a, K>[NA, NB, NC, ND];
        struct Unsized {
            first: u8,
            name: &'a str,
        } sections {
            bytes: [u8; NA],
            words: [u64; NB],
            nothing: [u32; NC],
            halves: [u16; ND],
        }
    }
    let sized: Sized<'_, PhantomData<()>, 3, 2, 0, 1> =
        Sized::from_parts(7, "lava", [1, 2, 3], [4, u64::MAX], [], [6]);
    let unsized_: &Unsized<'_, PhantomData<()>> = &sized;
    assert_eq!((unsized_.first, unsized_.name), (7, "lava"));
    assert_eq!(
        (unsized_.bytes(), unsized_.words()),
        (&[1, 2, 3][..], &[4, u64::MAX][..])
    );
    assert!(unsized_.nothing().is_empty());
    assert_eq!(unsized_.halves(), [6]);
    assert_eq!(size_of_val(unsized_), size_of_val(&sized));
}

#[test]
fn locations_allocate_from_their_memory_types() {
    use crate::buffer::{Dynamic, Location, Memory};
    use gpu_allocator::MemoryLocation as M;
    use vk::MemoryPropertyFlags as P;
    let mut properties = vk::PhysicalDeviceMemoryProperties {
        memory_type_count: 4,
        ..Default::default()
    };
    // Device-local, the BAR window, system memory, cached system memory.
    let flags = [
        P::DEVICE_LOCAL,
        P::DEVICE_LOCAL | P::HOST_VISIBLE | P::HOST_COHERENT,
        P::HOST_VISIBLE | P::HOST_COHERENT,
        P::HOST_VISIBLE | P::HOST_COHERENT | P::HOST_CACHED,
    ];
    for (i, flags) in flags.into_iter().enumerate() {
        properties.memory_types[i].property_flags = flags;
    }
    assert_eq!(Location::Gpu.allocator_location(), M::GpuOnly);
    assert_eq!(Location::Upload.allocator_location(), M::CpuToGpu);
    assert_eq!(Location::Shared.allocator_location(), M::CpuToGpu);
    assert_eq!(Location::Readback.allocator_location(), M::GpuToCpu);
    for location in [Location::Gpu, Location::Upload, Location::Readback] {
        assert_eq!(
            location.memory_type_bits(0b1111, &properties),
            0b1111,
            "{location:?}"
        );
    }
    assert_eq!(
        Location::Shared.memory_type_bits(0b1111, &properties),
        0b0010,
        "only device-local and host-visible"
    );
    assert_eq!(
        Location::Shared.memory_type_bits(0b1101, &properties),
        0,
        "fails rather than spill"
    );
    assert!(!Location::Gpu.host_visible() && Location::Readback.host_visible());
    Dynamic(Location::Upload).host_check();
    let message = panic_message(|| Dynamic(Location::Gpu).host_check());
    assert!(
        message.contains("the host can't access a buffer of Gpu memory"),
        "{message}"
    );
}

/// Host access waits for the latest submitted write overlapping its range that its timeline
/// hasn't completed.
#[test]
fn host_access_waits_for_overlapping_incomplete_writes() {
    let mut state = AccessState::default();
    let write = (S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE);
    state.record(&tracker(3, 0), &[], (0..16).into(), write.0, write.1);
    state.record(&tracker(5, 0), &[], (16..32).into(), write.0, write.1);
    state.record(&tracker(9, 0), &[], (32..48).into(), write.0, write.1);
    state.record(
        &tracker(9, 0),
        &[],
        (64..80).into(),
        S::COMPUTE_SHADER,
        A::SHADER_STORAGE_READ,
    );
    // Timeline 0 submitted up to 8 (so 9 is still being recorded) and completed `completed`.
    let scan = |range: std::ops::Range<u64>, completed| {
        state.incomplete_write_of(range.into(), |_| (completed, 0, 8))
    };
    assert_eq!(
        scan(0..64, 0),
        Some((0, 5)),
        "the latest submitted overlapping value"
    );
    assert_eq!(scan(8..12, 2), Some((0, 3)));
    assert_eq!(scan(8..12, 3), None, "completed");
    assert_eq!(
        scan(32..64, 0),
        None,
        "recorded but not submitted, or touching"
    );
    assert_eq!(scan(64..80, 0), None, "reads aren't waited for");
}

#[test]
fn host_image_copy_needs_no_rebar() {
    let mut features = Features::from_bits(iter::repeat(true));
    features.rebar = false;
    let consistent = features.consistent();
    #[cfg(feature = "host-image-copy")]
    assert!(consistent.host_image_copy);
    let _ = consistent;
}

/// Uploaders go to loader threads and are shared between them.
const _: fn() = || {
    fn shared<T: Send + Sync + Clone>() {}
    shared::<Uploader>();
};

/// A pipeline cache loads only with this device's header: its size, version, vendor, device
/// and cache UUID.
#[test]
fn pipeline_caches_load_only_for_their_device() {
    use crate::pipeline::cache_header_matches;
    let mut properties = vk::PhysicalDeviceProperties::default();
    (properties.vendor_id, properties.device_id) = (0x8086, 0x56a0);
    properties.pipeline_cache_uuid = [7; 16];
    let header = |size: u32, version: u32, vendor: u32, uuid: [u8; 16]| {
        let words = [size, version, vendor, 0x56a0]
            .into_iter()
            .flat_map(u32::to_le_bytes);
        let mut data: Vec<u8> = words.chain(uuid).collect();
        data.extend([0; 64]);
        data
    };
    assert!(cache_header_matches(
        &header(32, 1, 0x8086, [7; 16]),
        &properties
    ));
    assert!(
        !cache_header_matches(&header(32, 1, 0x1002, [7; 16]), &properties),
        "vendor"
    );
    assert!(
        !cache_header_matches(&header(32, 1, 0x8086, [8; 16]), &properties),
        "driver"
    );
    assert!(
        !cache_header_matches(&header(32, 2, 0x8086, [7; 16]), &properties),
        "version"
    );
    assert!(
        !cache_header_matches(&header(16, 1, 0x8086, [7; 16]), &properties),
        "size"
    );
    assert!(
        !cache_header_matches(&header(32, 1, 0x8086, [7; 16])[..20], &properties),
        "cut"
    );
    assert!(!cache_header_matches(&[], &properties));
}

/// Discrete GPUs first, then integrated ones, then virtual and CPU ones; the driver's order
/// within a rank.
#[test]
fn devices_are_tried_by_type() {
    use vk::PhysicalDeviceType as T;
    let types = [
        T::CPU,
        T::OTHER,
        T::INTEGRATED_GPU,
        T::DISCRETE_GPU,
        T::VIRTUAL_GPU,
        T::DISCRETE_GPU,
    ];
    assert_eq!(crate::state::device_order(&types), [3, 5, 2, 4, 0, 1]);
    assert!(crate::state::device_rank(T::CPU) > crate::state::device_rank(T::INTEGRATED_GPU));
    assert_eq!(crate::state::device_order(&[]), Vec::<usize>::new());
}

/// Each family's stages, masked by the features: ray tracing shaders only with ray tracing
/// pipelines.
#[test]
fn wait_stages_are_masked_by_features() {
    use crate::queue::{AllStages, ComputeStage, GraphicsStage, StageSet, TransferStage};
    let none = Features::NONE;
    assert_eq!(AllStages.mask(&none), S::ALL_COMMANDS);
    assert_eq!(GraphicsStage::Raster.mask(&none), S::ALL_GRAPHICS);
    assert_eq!(
        ComputeStage::Compute.mask(&none),
        S::COMPUTE_SHADER | S::DRAW_INDIRECT
    );
    assert_eq!(TransferStage::Transfer.mask(&none), S::ALL_TRANSFER);
    assert_eq!(GraphicsStage::RayTracing.mask(&none), S::DRAW_INDIRECT);
    #[cfg(feature = "ray-tracing-pipeline")]
    {
        let rt = Features {
            ray_tracing_pipeline: true,
            ..Features::NONE
        };
        let traced = S::RAY_TRACING_SHADER_KHR | S::DRAW_INDIRECT;
        assert_eq!(ComputeStage::RayTracing.mask(&rt), traced);
        assert_eq!(GraphicsStage::RayTracing.mask(&rt), traced);
    }
}

/// Errors map Vulkan's typed results, keep their context apart from their source, and leave
/// typed variants without context.
#[test]
fn errors_are_typed_and_display_once() {
    use crate::Error;
    use std::error::Error as _;
    assert!(matches!(
        Error::from(vk::Result::ERROR_DEVICE_LOST),
        Error::DeviceLost
    ));
    assert!(matches!(
        Error::from(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY),
        Error::OutOfMemory
    ));
    assert!(matches!(
        Error::from(gpu_allocator::AllocationError::OutOfMemory),
        Error::OutOfMemory
    ));
    assert!(matches!(
        Error::from(vk::Result::ERROR_OUT_OF_DATE_KHR),
        Error::SurfaceOutOfDate
    ));
    let wrapped = Error::from(vk::Result::ERROR_UNKNOWN).context("the pipelines of `Pass`");
    assert_eq!(wrapped.to_string(), "the pipelines of `Pass`");
    let source = wrapped.source().expect("the Vulkan error").to_string();
    assert!(source.contains("ERROR_UNKNOWN"), "{source}");
    assert!(matches!(
        Error::DeviceLost.context("anything"),
        Error::DeviceLost
    ));
    let slots = Error::OutOfBindlessSlots {
        heap: crate::bindless::Heap::Sampler,
        count: 2,
    };
    assert_eq!(
        slots.to_string(),
        "the bindless sampler heap has no 2 contiguous free slots"
    );
    assert!(slots.source().is_none());
}

/// Without `lava::init`, lava's calls panic with one message, and shutdown does nothing.
#[test]
fn calls_before_init_panic() {
    let panicked = std::panic::catch_unwind(crate::state::call).unwrap_err();
    let message = panicked
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_default();
    assert!(message.contains("lava is not initialised"), "{message}");
    assert!(crate::shutdown().is_ok());
}

#[test]
fn topology_classes_and_render_areas() {
    use crate::command_buffer::{PrimitiveTopology as T, Rect, topology_class};
    let classes = [
        (T::POINT_LIST, T::POINT_LIST),
        (T::LINE_LIST, T::LINE_LIST),
        (T::LINE_STRIP, T::LINE_LIST),
        (T::LINE_STRIP_WITH_ADJACENCY, T::LINE_LIST),
        (T::TRIANGLE_LIST, T::TRIANGLE_LIST),
        (T::TRIANGLE_STRIP, T::TRIANGLE_LIST),
        (T::TRIANGLE_FAN, T::TRIANGLE_LIST),
        (T::TRIANGLE_LIST_WITH_ADJACENCY, T::TRIANGLE_LIST),
        (T::PATCH_LIST, T::PATCH_LIST),
    ];
    for (topology, class) in classes {
        assert_eq!(topology_class(topology), class, "{topology:?}");
    }
    let area = Rect::from(glam::UVec2::new(4, 8));
    assert_eq!(area, Rect::new(glam::IVec2::ZERO, glam::UVec2::new(4, 8)));
    assert_eq!(area.vk().extent.height, 8);
}

#[test]
fn compressed_formats_have_their_blocks() {
    use vk::Format as F;
    let block = |w, h, bytes| {
        Some(Block {
            width: w,
            height: h,
            bytes,
        })
    };
    assert_eq!(format::block(F::BC1_RGB_SRGB_BLOCK), block(4, 4, 8));
    assert_eq!(format::block(F::BC7_UNORM_BLOCK), block(4, 4, 16));
    assert_eq!(format::block(F::ETC2_R8G8B8_UNORM_BLOCK), block(4, 4, 8));
    assert_eq!(format::block(F::ETC2_R8G8B8A1_SRGB_BLOCK), block(4, 4, 8));
    assert_eq!(format::block(F::ETC2_R8G8B8A8_UNORM_BLOCK), block(4, 4, 16));
    assert_eq!(format::block(F::EAC_R11_SNORM_BLOCK), block(4, 4, 8));
    assert_eq!(format::block(F::EAC_R11G11_UNORM_BLOCK), block(4, 4, 16));
    assert_eq!(format::block(F::ASTC_5X4_UNORM_BLOCK), block(5, 4, 16));
    assert_eq!(format::block(F::ASTC_12X10_SRGB_BLOCK), block(12, 10, 16));
    assert_eq!(format::block(F::R8G8B8A8_UNORM), None);
    // Every compressed format lava has a type for has a block and a feature.
    for raw in 131..=184 {
        let format = F::from_raw(raw);
        assert!(format::block(format).is_some(), "{format:?}");
        assert!(format::compression(format).is_some(), "{format:?}");
    }
    assert_eq!(
        format::compression(F::ASTC_4X4_SRGB_BLOCK),
        Some("astc-compression")
    );
    assert_eq!(
        format::compression(F::EAC_R11G11_SNORM_BLOCK),
        Some("etc2-compression")
    );
    // Copies move one aspect: a combined format's depth or stencil alone.
    let aspect = |a| {
        format::copy_block(F::D32_SFLOAT_S8_UINT, 5, a)
            .unwrap()
            .bytes
    };
    assert_eq!(aspect(vk::ImageAspectFlags::DEPTH), 4);
    assert_eq!(aspect(vk::ImageAspectFlags::STENCIL), 1);
    assert_eq!(
        format::copy_block(F::R8G8B8_UNORM, 3, vk::ImageAspectFlags::COLOR)
            .unwrap()
            .bytes,
        3
    );
    assert_eq!(
        format::copy_block(F::UNDEFINED, 0, vk::ImageAspectFlags::COLOR),
        None
    );
}

#[test]
fn copy_sizes_count_whole_blocks_and_rows() {
    let extent = |w, h, d| vk::Extent3D {
        width: w,
        height: h,
        depth: d,
    };
    let texel = Block {
        width: 1,
        height: 1,
        bytes: 4,
    };
    let bc = Block {
        width: 4,
        height: 4,
        bytes: 16,
    };
    assert_eq!(rows_len(texel, extent(3, 2, 1), 1, (0, 0)), 3 * 2 * 4);
    assert_eq!(
        rows_len(texel, extent(3, 2, 1), 2, (0, 0)),
        2 * 3 * 2 * 4,
        "layers one after another"
    );
    // Rows 8 texels apart: the last row ends after its 3 texels.
    assert_eq!(rows_len(texel, extent(3, 2, 1), 1, (8, 0)), (8 + 3) * 4);
    // Slices 4 rows apart, two of them.
    assert_eq!(
        rows_len(texel, extent(3, 2, 2), 1, (8, 4)),
        ((4 + 1) * 8 + 3) * 4
    );
    // A 6x6 box of 4x4 blocks takes 2x2 blocks; a mip narrower than its block takes one.
    assert_eq!(rows_len(bc, extent(6, 6, 1), 1, (0, 0)), 4 * 16);
    assert_eq!(rows_len(bc, extent(2, 2, 1), 1, (0, 0)), 16);
    assert_eq!(
        rows_len(bc, extent(4, 4, 1), 1, (16, 0)),
        16,
        "one row of blocks"
    );
    let astc = Block {
        width: 5,
        height: 4,
        bytes: 16,
    };
    assert_eq!(rows_len(astc, extent(10, 8, 1), 1, (0, 0)), 2 * 2 * 16);
}

#[test]
fn clear_values_pack_64_bit_components_into_two_words() {
    use crate::image::format::{R64G64B64A64Uint, R64G64Sint, R64G64Uint, R64Sfloat, R64Uint};
    assert_eq!(
        R64Uint.clear_value([0x1_0000_0002]),
        ClearValue::Uint([2, 1, 0, 0])
    );
    assert_eq!(
        R64G64Uint.clear_value([u64::MAX, 5]),
        ClearValue::Uint([u32::MAX, u32::MAX, 5, 0])
    );
    assert_eq!(
        R64G64Sint.clear_value([-1, 2]),
        ClearValue::Sint([-1, -1, 2, 0])
    );
    let ClearValue::Float(words) = R64Sfloat.clear_value([1.5]) else {
        panic!()
    };
    let bits = words.map(f32::to_bits);
    assert_eq!((bits[1] as u64) << 32 | bits[0] as u64, 1.5f64.to_bits());
    assert_eq!(
        R64G64B64A64Uint.clear_value([1, 2, 0, 0]),
        ClearValue::Uint([1, 0, 2, 0])
    );
    let message = panic_message(|| R64G64B64A64Uint.clear_value([1, 2, 3, 0]));
    assert!(
        message.contains("a clear value holds two 64-bit components"),
        "{message}"
    );
}

#[test]
fn reinterpreting_views_keep_the_compatibility_class() {
    use crate::image::format::*;
    fn class<F: Format>(f: F) -> format::Class {
        format::class(f.vk_format(), f.texel_size(), f.aspects())
    }
    assert_eq!(class(R8G8B8A8Srgb), class(R8G8B8A8Unorm));
    assert_eq!(class(R32Uint), class(R32Sfloat));
    assert_eq!(class(B10G11R11UfloatPack32), class(R8G8B8A8Uint));
    assert_eq!(class(E5B9G9R9UfloatPack32), class(R32Sfloat));
    assert_ne!(class(R16G16Sfloat), class(R16Sfloat));
    assert_ne!(
        class(A8Unorm),
        class(R8Unorm),
        "8-bit alpha is a class of its own"
    );
    assert_ne!(class(D32Sfloat), class(R32Sfloat));
    assert_ne!(class(D24UnormS8Uint), class(X8D24UnormPack32));
    #[cfg(feature = "bc-compression")]
    {
        assert_eq!(class(BC7UnormBlock), class(BC7SrgbBlock));
        assert_eq!(class(BC4UnormBlock), class(BC4SnormBlock));
        assert_ne!(class(BC1RgbUnormBlock), class(BC1RgbaUnormBlock));
        assert_ne!(
            class(BC7UnormBlock),
            class(R32G32B32A32Uint),
            "no block-texel views"
        );
    }
}

#[test]
fn formats_count_their_components() {
    use crate::image::format::*;
    fn texels<F: HasTexels<T>, T>(_: F) {}
    texels::<_, [f32; 3]>(E5B9G9R9UfloatPack32);
    texels::<_, [f32; 4]>(A8Unorm);
    texels::<_, [f32; 4]>(A4R4G4B4UnormPack16);
    texels::<_, [f32; 4]>(A1B5G5R5UnormPack16);
    for format in [
        vk::Format::A4R4G4B4_UNORM_PACK16,
        vk::Format::A4B4G4R4_UNORM_PACK16,
        vk::Format::A8_UNORM_KHR,
        vk::Format::A1B5G5R5_UNORM_PACK16_KHR,
    ] {
        assert!(format::info(format).is_some(), "{format:?} has a type");
    }
}

/// `features` by their cargo names (`"rebar"` too); names lava was compiled without are left
/// out.
fn named(names: &[&str]) -> Features {
    let all = Features::from_bits(iter::repeat(true)).missing_from(&Features::NONE);
    Features::from_bits(all.iter().map(|name| names.contains(name)))
}

/// Every feature this build has, by cargo name, `"rebar"` first.
fn feature_names() -> Vec<&'static str> {
    Features::from_bits(iter::repeat(true)).missing_from(&Features::NONE)
}

/// Whether `features` has `name`, which this build compiles.
fn has_feature(features: Features, name: &str) -> bool {
    feature_names().contains(&name) && named(&[name]).without(features) == Features::NONE
}

#[test]
fn features_subtract_and_name_what_is_missing() {
    let both = named(&["mesh-shader", "logic-op"]);
    assert_eq!(both.without(named(&["logic-op"])), named(&["mesh-shader"]));
    assert_eq!(both.without(both), Features::NONE);
    assert_eq!(both.without(Features::NONE), both);
    #[cfg(all(feature = "mesh-shader", feature = "logic-op"))]
    {
        assert_eq!(both.missing_from(&named(&["logic-op"])), ["mesh-shader"]);
        // In declaration order, whatever order they were asked in.
        assert_eq!(
            named(&["logic-op", "mesh-shader"]).missing_from(&Features::NONE),
            ["mesh-shader", "logic-op"]
        );
    }
    assert!(Features::NONE.missing_from(&Features::NONE).is_empty());
    assert!(both.missing_from(&both).is_empty());
    let all = Features::from_bits(iter::repeat(true));
    assert_eq!(all.missing_from(&Features::NONE), feature_names());
    assert_eq!(feature_names()[0], "rebar");
}

/// Each feature that builds on others is dropped without any of them, kept with all of them;
/// every other feature stands alone.
#[test]
fn consistent_features_drop_those_without_their_prerequisites() {
    let needs: &[(&str, &[&str])] = &[
        ("ray-query", &["acceleration-structure"]),
        ("ray-tracing-pipeline", &["acceleration-structure"]),
        (
            "ray-traversal-primitive-culling",
            &["ray-query", "acceleration-structure"],
        ),
        (
            "mesh-shader-queries",
            &["mesh-shader", "pipeline-statistics"],
        ),
        ("ray-tracing-position-fetch", &["acceleration-structure"]),
        (
            "ray-tracing-invocation-reorder",
            &["ray-tracing-pipeline", "acceleration-structure"],
        ),
        ("depth-clamp-control", &["depth-clamp"]),
        ("dynamic-depth-clip", &["depth-clip-enable"]),
        (
            "dynamic-depth-clip-negative-one-to-one",
            &["depth-clip-control"],
        ),
        (
            "dynamic-conservative-rasterization",
            &["conservative-rasterization"],
        ),
        ("dynamic-provoking-vertex", &["provoking-vertex"]),
        ("sparse-residency-buffer", &["sparse-binding"]),
        ("sparse-residency-image-2d", &["sparse-binding"]),
        ("sparse-residency-image-3d", &["sparse-binding"]),
    ];
    let compiled = feature_names();
    for &(feature, prerequisites) in needs {
        if !iter::once(&feature)
            .chain(prerequisites)
            .all(|f| compiled.contains(f))
        {
            continue;
        }
        let with: Vec<&str> = iter::once(feature)
            .chain(prerequisites.iter().copied())
            .collect();
        assert_eq!(named(&with).consistent(), named(&with), "{feature}");
        for missing in prerequisites {
            let without: Vec<&str> = with.iter().copied().filter(|f| f != missing).collect();
            assert!(
                !has_feature(named(&without).consistent(), feature),
                "{feature} without {missing}"
            );
        }
    }
    // Ray traversal culling also comes with ray tracing pipelines instead of ray queries.
    #[cfg(feature = "ray-tracing-pipeline")]
    let pipelines = named(&[
        "ray-traversal-primitive-culling",
        "ray-tracing-pipeline",
        "acceleration-structure",
    ]);
    #[cfg(feature = "ray-tracing-pipeline")]
    assert_eq!(pipelines.consistent(), pipelines);
    for name in feature_names() {
        let alone = named(&[name]).consistent();
        match needs.iter().any(|(feature, _)| *feature == name) {
            true => assert_eq!(alone, Features::NONE, "{name} alone"),
            false => assert_eq!(alone, named(&[name]), "{name} alone"),
        }
    }
    // Every feature is possible, but ray traversal culling compiled without what brings it.
    #[cfg(any(
        not(feature = "ray-traversal-primitive-culling"),
        feature = "ray-query",
        feature = "ray-tracing-pipeline"
    ))]
    {
        let all = Features::from_bits(iter::repeat(true));
        assert_eq!(all.consistent(), all);
    }
}

/// The extensions `extensions()` enables for a set of features are all `optional()` needs to
/// detect it on a device that has its bits, and its bits detect nothing else (but what the
/// device facts decide).
#[test]
fn extensions_name_what_detecting_the_features_needs() {
    let facts = DeviceFacts {
        rebar: true,
        strict_lines: true,
        dedicated_compute: true,
        dedicated_transfer: true,
        sparse_queue: true,
    };
    let by_facts = named(&["rebar", "async-compute", "async-transfer"]);
    let queried = FeatureChain::default();
    let names = feature_names();
    // Every feature but those this build can't turn on (ray traversal culling compiled
    // without ray queries and pipelines).
    let possible = Features::from_bits(iter::repeat(true)).consistent();
    for &name in names.iter().filter(|&&name| has_feature(possible, name)) {
        // The smallest consistent set with it: everything, less what it does without. Removed
        // from the back, so ray traversal culling keeps ray queries (a ray tracing pipeline
        // would need the device's `rayTraversalPrimitiveCulling`, which `queried` lacks).
        let mut set = possible;
        for &other in names.iter().rev().filter(|&&other| other != name) {
            let fewer = set.without(named(&[other]));
            if has_feature(fewer.consistent(), name) {
                set = fewer.consistent();
            }
        }
        let extensions = set.extensions();
        let has = |extension: &std::ffi::CStr| extensions.contains(&extension);
        let detected = FeatureChain::enabling(&set, &queried).optional(&has, facts);
        assert_eq!(
            set.without(detected),
            Features::NONE,
            "{name}: {:?} undetected with {extensions:?}",
            set.missing_from(&detected)
        );
        // Ray queries imply the culling capability (`RayTraversalPrimitiveCullingKHR`).
        let implied = match has_feature(set, "ray-query") {
            true => named(&["ray-traversal-primitive-culling"]),
            false => Features::NONE,
        };
        let extra = detected.without(set).without(by_facts).without(implied);
        assert_eq!(
            extra,
            Features::NONE,
            "{name}: {:?} detected too",
            extra.missing_from(&Features::NONE)
        );
    }
}

/// Every cargo name lava's tables use is a feature of `lava/Cargo.toml`, with its `assert-`.
#[test]
fn cargo_names_are_features_of_the_manifest() {
    let manifest = include_str!("../Cargo.toml");
    let section = manifest
        .split_once("\n[features]\n")
        .expect("lava has features")
        .1;
    let section = section.split("\n[").next().unwrap();
    let declared: Vec<&str> = (section.lines())
        .filter_map(|line| line.split_once('='))
        .map(|(key, _)| key.trim())
        .filter(|key| !key.is_empty() && !key.starts_with('#') && !key.starts_with('"'))
        .collect();
    assert!(declared.contains(&"default"), "{declared:?}");
    let declares = |name: &str| declared.contains(&name);
    for (capability, cargo) in crate::features::SPIRV_CAPABILITIES {
        assert!(declares(cargo), "capability {capability}: `{cargo}`");
        assert!(declares(&format!("assert-{cargo}")), "assert-{cargo}");
    }
    for name in feature_names().into_iter().skip(1) {
        assert!(declares(name), "`{name}`");
        assert!(declares(&format!("assert-{name}")), "assert-{name}");
    }
    assert!(
        !declares("rebar"),
        "rebar is a device fact, no cargo feature"
    );
}

#[test]
fn freeing_slots_twice_panics() {
    // Slots 4..10 free.
    let partly_taken = || {
        let mut ranges = Ranges::all(10);
        assert_eq!(ranges.take(4), Some(0));
        ranges
    };
    partly_taken().give(0..4);
    let message = panic_message(|| partly_taken().give(5..6));
    assert!(
        message.contains("bindless slots 5..6 are freed twice"),
        "{message}"
    );
    // Overlapping a free range at either end, or spanning one, is a double free too.
    for range in [3..5, 9..12, 2..11] {
        let message = panic_message(|| partly_taken().give(range.clone()));
        assert!(message.contains("freed twice"), "{range:?}: {message}");
    }
    // Free 2..4 and 6..8.
    let holes = || {
        let mut ranges = Ranges::all(10);
        assert_eq!(ranges.take(10), Some(0));
        ranges.give(2..4);
        ranges.give(6..8);
        ranges
    };
    let message = panic_message(|| holes().give(1..9));
    assert!(message.contains("1..9 are freed twice"), "{message}");
    // Next to free ranges is fine, and an empty range is nothing.
    let mut ranges = holes();
    ranges.give(4..6);
    ranges.give(5..5);
    assert_eq!(ranges.take(6), Some(2), "2..8 merged");
}

/// A borrowed key hashes and compares like the key it stands for, so lookups find it without
/// copying.
#[test]
fn borrowed_raster_keys_find_their_owned_keys() {
    use crate::command_buffer::{Baked, PrimitiveTopology};
    #[cfg(bake_blend)]
    use crate::command_buffer::{Blend, ColorComponentFlags};
    use crate::pipeline::{RasterKey, RasterKeyRef};
    use hashbrown::{Equivalent, HashMap};
    use std::hash::BuildHasher;
    let formats = [vk::Format::R8G8B8A8_UNORM, vk::Format::R16G16B16A16_SFLOAT];
    let key = |colors: &'static [vk::Format], constants: &'static [u8], topology| RasterKeyRef {
        color_formats: colors,
        depth_format: vk::Format::D32_SFLOAT,
        stencil_format: vk::Format::UNDEFINED,
        view_mask: 0,
        constants,
        baked: Baked {
            topology,
            ..Baked::default()
        },
        #[cfg(bake_blend)]
        blends: &[Blend::Replace, Blend::Alpha][..colors.len()],
        #[cfg(bake_blend)]
        write_masks: &[ColorComponentFlags::RGBA; 2][..colors.len()],
    };
    let formats: &'static [vk::Format] = Box::leak(Box::new(formats));
    let keys = [
        key(&formats[..1], &[], PrimitiveTopology::TRIANGLE_LIST),
        key(formats, &[], PrimitiveTopology::TRIANGLE_LIST),
        key(
            &formats[..1],
            &[1, 0, 0, 0],
            PrimitiveTopology::TRIANGLE_LIST,
        ),
        key(&formats[..1], &[], PrimitiveTopology::LINE_LIST),
    ];
    let hasher = hashbrown::DefaultHashBuilder::default();
    let mut map: HashMap<RasterKey, usize> = HashMap::new();
    for (i, key) in keys.iter().enumerate() {
        let owned = key.owned();
        assert_eq!(hasher.hash_one(key), hasher.hash_one(&owned), "{key:?}");
        assert!(key.equivalent(&owned));
        map.insert(owned, i);
    }
    for (i, key) in keys.iter().enumerate() {
        assert_eq!(map.get(key), Some(&i), "{key:?}");
        for (j, other) in keys.iter().enumerate() {
            assert_eq!(key.equivalent(&other.owned()), i == j, "{key:?} {other:?}");
        }
    }
}

/// A pipeline bakes the draw state the device can't set dynamically, and only that.
#[test]
fn pipelines_bake_only_the_state_without_dynamic_state() {
    use crate::command_buffer::{Baked, Dynamic, PrimitiveTopology as T, RasterState};
    let dynamic = |on: bool| Dynamic {
        blend: on,
        polygon_mode: on,
        depth_clamp: on,
        depth_clamp_range: on,
        depth_clip: on,
        depth_clip_negative_one_to_one: on,
        logic_op: on,
        conservative: on,
        line_mode: on,
        line_stipple_enable: on,
        line_stipple: on,
        provoking_vertex: on,
        topology_unrestricted: on,
    };
    let (mut lines, _, _) = RasterState::<1>::default().topology(T::LINE_STRIP).split();
    let (triangles, _, _) = RasterState::<1>::default().split();
    // Vertex pipelines bake the topology's class unless any topology is dynamic; mesh
    // pipelines have none.
    assert_eq!(
        Baked::of(&lines, &dynamic(false), true).topology,
        T::LINE_LIST
    );
    assert_eq!(
        Baked::of(&triangles, &dynamic(false), true).topology,
        T::TRIANGLE_LIST
    );
    assert_eq!(Baked::of(&lines, &dynamic(true), true), Baked::default());
    assert_eq!(
        Baked::of(&lines, &dynamic(false), false).topology,
        Baked::default().topology
    );
    // With every state dynamic, draws that differ in it share a pipeline.
    lines.polygon_mode = crate::command_buffer::PolygonMode::LINE;
    lines.depth_clip = Some(false);
    lines.logic_op = Some(crate::command_buffer::LogicOp::XOR);
    lines.provoking_vertex = vk::ProvokingVertexModeEXT::LAST_VERTEX;
    lines.conservative = vk::ConservativeRasterizationModeEXT::OVERESTIMATE;
    lines.line_mode = vk::LineRasterizationModeKHR::BRESENHAM;
    assert_eq!(
        Baked::of(&lines, &dynamic(true), false),
        Baked::of(&triangles, &dynamic(true), false)
    );
    let baked = Baked::of(&lines, &dynamic(false), false);
    #[cfg(bake_polygon_mode)]
    assert_eq!(baked.polygon_mode, crate::command_buffer::PolygonMode::LINE);
    #[cfg(bake_depth_clip)]
    assert!(!baked.depth_clip);
    #[cfg(bake_logic_op)]
    assert_eq!(baked.logic_op, Some(crate::command_buffer::LogicOp::XOR));
    #[cfg(bake_provoking_vertex)]
    assert_eq!(
        baked.provoking_vertex,
        vk::ProvokingVertexModeEXT::LAST_VERTEX
    );
    #[cfg(bake_conservative_rasterization)]
    assert_eq!(
        baked.conservative,
        vk::ConservativeRasterizationModeEXT::OVERESTIMATE
    );
    #[cfg(bake_line_mode)]
    assert_eq!(baked.line_mode, vk::LineRasterizationModeKHR::BRESENHAM);
    let _ = baked;
    // Depth clipping bakes its effective value: clipped unless clamped.
    #[cfg(bake_depth_clip)]
    assert!(Baked::of(&triangles, &dynamic(false), false).depth_clip);
}

/// The dynamic state follows the features: none without them, all with all of them.
#[test]
fn dynamic_state_follows_the_features() {
    use crate::command_buffer::Dynamic;
    let none = Dynamic::of(&Features::NONE, true);
    let all = Dynamic::of(&Features::from_bits(iter::repeat(true)), true);
    let fields = |d: Dynamic| {
        [
            d.blend,
            d.polygon_mode,
            d.depth_clamp,
            d.depth_clamp_range,
            d.depth_clip,
            d.depth_clip_negative_one_to_one,
            d.logic_op,
            d.conservative,
            d.line_mode,
            d.line_stipple_enable,
            d.line_stipple,
            d.provoking_vertex,
            d.topology_unrestricted,
        ]
    };
    assert_eq!(fields(none), [false; 13]);
    // A dynamic topology leaves its class only with extended dynamic state 3 enabled.
    assert!(!Dynamic::of(&named(&["logic-op"]), true).topology_unrestricted);
    #[cfg(all(
        feature = "dynamic-blend",
        feature = "dynamic-polygon-mode",
        feature = "dynamic-depth-clamp",
        feature = "depth-clamp-control",
        feature = "dynamic-depth-clip",
        feature = "dynamic-depth-clip-negative-one-to-one",
        feature = "dynamic-logic-op",
        feature = "dynamic-conservative-rasterization",
        feature = "dynamic-line-mode",
        feature = "dynamic-line-stipple",
        feature = "line-stipple",
        feature = "dynamic-provoking-vertex"
    ))]
    {
        assert_eq!(fields(all), [true; 13]);
        let all = Features::from_bits(iter::repeat(true));
        assert!(!Dynamic::of(&all, false).topology_unrestricted);
        assert!(Dynamic::of(&named(&["dynamic-blend"]), true).topology_unrestricted);
    }
    let _ = all;
}

/// Mips halve every dimension to at least 1; boxes lie inside a mip from a non-negative offset.
#[test]
fn mips_halve_and_boxes_lie_inside_them() {
    use crate::image::dim::{D2, D2Array, D3, Dim, box_inside, box_is_empty};
    use glam::{UVec2, UVec3};
    assert_eq!(D2::mip_extent(UVec2::new(64, 16), 0), UVec2::new(64, 16));
    assert_eq!(D2::mip_extent(UVec2::new(64, 16), 3), UVec2::new(8, 2));
    assert_eq!(D2::mip_extent(UVec2::new(64, 16), 5), UVec2::new(2, 1));
    assert_eq!(D2::mip_extent(UVec2::new(5, 3), 1), UVec2::new(2, 1));
    assert_eq!(
        D2::mip_extent(UVec2::new(64, 16), 40),
        UVec2::ONE,
        "no shift overflow"
    );
    assert_eq!(D2Array::mip_extent(UVec2::new(8, 8), 2), UVec2::new(2, 2));
    assert_eq!(D3::mip_extent(UVec3::new(16, 8, 4), 2), UVec3::new(4, 2, 1));
    assert_eq!(D3::mip_extent(UVec3::new(16, 8, 4), 3), UVec3::new(2, 1, 1));

    let offset = |x, y, z| vk::Offset3D { x, y, z };
    let extent = |width, height, depth| vk::Extent3D {
        width,
        height,
        depth,
    };
    let mip = extent(8, 4, 1);
    assert!(box_inside(offset(0, 0, 0), mip, mip));
    assert!(box_inside(offset(4, 2, 0), extent(4, 2, 1), mip));
    assert!(!box_inside(offset(5, 2, 0), extent(4, 2, 1), mip));
    assert!(!box_inside(offset(0, 0, 1), extent(1, 1, 1), mip));
    assert!(!box_inside(offset(-1, 0, 0), extent(1, 1, 1), mip));
    assert!(
        box_inside(offset(8, 4, 1), extent(0, 0, 0), mip),
        "an empty box at the end"
    );
    assert!(
        !box_inside(
            offset(i32::MAX, 0, 0),
            extent(u32::MAX, 1, 1),
            extent(u32::MAX, 1, 1)
        ),
        "no overflow"
    );
    assert!(box_is_empty(extent(0, 4, 1)) && !box_is_empty(mip));
}

/// Buffer casts keep whole, aligned elements; image casts keep the format and a usage subset.
#[test]
fn casts_check_sizes_alignments_formats_and_usages() {
    use crate::buffer::slice::cast_check as buffer_cast;
    use crate::image::slice::cast_check as image_cast;
    use vk::{Format as F, ImageUsageFlags as U};
    buffer_cast::<[u32; 3]>(16, 24);
    buffer_cast::<u64>(0, 0);
    let message = panic_message(|| buffer_cast::<[u32; 3]>(16, 20));
    assert!(
        message.contains("20 bytes aren't a whole number of 12-byte elements"),
        "{message}"
    );
    let message = panic_message(|| buffer_cast::<u64>(4, 8));
    assert!(
        message.contains("address 0x4 isn't aligned to the 8 bytes"),
        "{message}"
    );
    image_cast(
        (F::R8G8B8A8_UNORM, U::SAMPLED | U::STORAGE),
        (F::R8G8B8A8_UNORM, U::SAMPLED),
    );
    let message = panic_message(|| {
        image_cast(
            (F::R8G8B8A8_UNORM, U::SAMPLED),
            (F::R8G8B8A8_UNORM, U::STORAGE),
        )
    });
    assert!(message.contains("can't cast an image"), "{message}");
    let message = panic_message(|| {
        image_cast(
            (F::R8G8B8A8_UNORM, U::SAMPLED),
            (F::R8G8B8A8_SRGB, U::SAMPLED),
        )
    });
    assert!(message.contains("to format R8G8B8A8_SRGB"), "{message}");
}

/// Runtime formats and usages check at runtime what the static types check at compile time.
#[test]
fn dynamic_formats_and_usages_check_what_static_ones_guarantee() {
    use crate::image::{format::*, usage};
    use vk::{ImageAspectFlags as Aspect, ImageUsageFlags as U};
    let color = Dynamic(vk::Format::R8G8B8A8_UNORM);
    assert_eq!(color.texel_size(), R8G8B8A8Unorm.texel_size());
    assert_eq!(color.aspects(), Aspect::COLOR);
    color.aspect_check(Aspect::COLOR);
    color.texel_check::<[f32; 4]>();
    assert_eq!(
        color.clear_value(ClearValue::Float([1.0; 4])),
        ClearValue::Float([1.0; 4])
    );
    let message = panic_message(|| color.aspect_check(Aspect::DEPTH));
    assert!(message.contains("lacks the aspects DEPTH"), "{message}");
    let message = panic_message(|| color.texel_check::<[u32; 4]>());
    assert!(
        message.contains("doesn't have texels of type [u32; 4]"),
        "{message}"
    );
    let message = panic_message(|| color.clear_value(ClearValue::Uint([1; 4])));
    assert!(message.contains("can't be cleared to Uint"), "{message}");

    let depth = Dynamic(vk::Format::D32_SFLOAT_S8_UINT);
    assert_eq!(depth.aspects(), Aspect::DEPTH | Aspect::STENCIL);
    depth.aspect_check(Aspect::STENCIL);
    depth.clear_value(ClearValue::DepthStencil {
        depth: 1.0,
        stencil: 0,
    });
    let message = panic_message(|| depth.clear_value(ClearValue::Float([0.0; 4])));
    assert!(message.contains("can't be cleared"), "{message}");
    let message = panic_message(|| Dynamic(vk::Format::UNDEFINED).aspects());
    assert!(
        message.contains("no format type for UNDEFINED"),
        "{message}"
    );
    assert!(!Dynamic(vk::Format::UNDEFINED).known() && color.known());

    let dynamic = usage::Dynamic(U::SAMPLED | U::TRANSFER_DST);
    assert_eq!(dynamic.flags(), U::SAMPLED | U::TRANSFER_DST);
    assert!(dynamic.create_flags().is_empty(), "never a mutable format");
    dynamic.check(U::SAMPLED);
    let message = panic_message(|| dynamic.check(U::STORAGE | U::SAMPLED));
    assert!(message.contains("lacks"), "{message}");
}

/// Texel sizes of the format table, against the sizes the Vulkan spec gives them.
#[test]
fn the_format_table_has_the_spec_texel_sizes() {
    use vk::Format as F;
    let sizes = [
        (F::R8_UNORM, 1),
        (F::R8G8_SNORM, 2),
        (F::R8G8B8_UNORM, 3),
        (F::R8G8B8A8_SRGB, 4),
        (F::B8G8R8A8_UNORM, 4),
        (F::A2B10G10R10_UNORM_PACK32, 4),
        (F::R16_SFLOAT, 2),
        (F::R16G16B16A16_UINT, 8),
        (F::R32_SFLOAT, 4),
        (F::R32G32B32_SFLOAT, 12),
        (F::R32G32B32A32_SINT, 16),
        (F::R64_UINT, 8),
        (F::R64G64B64A64_SFLOAT, 32),
        (F::B10G11R11_UFLOAT_PACK32, 4),
        (F::E5B9G9R9_UFLOAT_PACK32, 4),
        (F::A4R4G4B4_UNORM_PACK16, 2),
        (F::D16_UNORM, 2),
        (F::D32_SFLOAT, 4),
        (F::S8_UINT, 1),
    ];
    for (format, size) in sizes {
        let info = format::info(format).unwrap_or_else(|| panic!("{format:?} has a type"));
        assert_eq!(info.texel_size, size, "{format:?}");
        let aspect = crate::image::slice::single_aspect(info.aspects);
        assert_eq!(
            format::copy_block(format, info.texel_size, aspect).map(|b| b.bytes),
            Some(size as u32),
            "{format:?} copies its texels"
        );
    }
    // Compressed formats have no texel size: their blocks have the bytes.
    #[cfg(feature = "bc-compression")]
    assert_eq!(format::info(F::BC7_UNORM_BLOCK).unwrap().texel_size, 0);
}

// ---- referencing buffers ----------------------------------------------------------------------

/// A buffer of 1 TiB of addresses at `address`, never dropped (its `Drop` needs a context).
fn fake_buffer(address: u64) -> &'static Arc<BufferInternal> {
    Box::leak(Box::new(Arc::new(BufferInternal {
        handle: vk::Buffer::null(),
        address,
        num_bytes: 1 << 40,
        memory: BufferMemory::Allocation(Default::default()),
        state: Default::default(),
        keep_alive: Default::default(),
        header: Header::new(Live::internal()),
    })))
}

fn to_buffer(
    buffer: &'static Arc<BufferInternal>,
    range: std::ops::Range<u64>,
) -> Reference<'static> {
    Reference {
        object: Referenced::Buffer(buffer),
        range: range.into(),
    }
}

/// An element with two references, the second a write reference.
#[derive(Clone, Copy, bytemuck::AnyBitPattern)]
#[repr(C)]
struct TwoReferences([u64; 2]);

unsafe impl ReferencingData for TwoReferences {
    const REFERENCES: usize = 2;
    const WRITES: u64 = 0b10;
    const UNORDERED: u64 = 0;
}

fn keep_alive_of(
    parent: &'static Arc<BufferInternal>,
) -> std::sync::MutexGuard<'static, KeepAlive> {
    let keep_alive = parent
        .keep_alive
        .get_or_init(|| std::sync::Mutex::new(KeepAlive::new::<TwoReferences>()));
    keep_alive.lock().unwrap()
}

/// A referencing buffer keeps each distinct child once, counted per reference, registers itself
/// as the child's parent, and lets a child go when no element references it any more; write
/// references make the child a writer and the buffer `writes_below`.
#[test]
fn referencing_buffers_count_their_children() {
    let parent = fake_buffer(1 << 40);
    let (a, b) = (fake_buffer(2 << 40), fake_buffer(3 << 40));
    let mut released = Vec::new();
    {
        let mut keep_alive = keep_alive_of(parent);
        keep_alive.replace(
            parent,
            4,
            0,
            &[to_buffer(a, 0..16), to_buffer(b, 0..16)],
            &mut released,
        );
        keep_alive.replace(
            parent,
            4,
            1,
            &[to_buffer(a, 16..32), to_buffer(a, 32..48)],
            &mut released,
        );
        assert_eq!(keep_alive.children.iter().count(), 2);
        let child_a = keep_alive
            .children
            .iter()
            .find(|c| c.object.key() == Strong::Buffer(a.clone()).key())
            .unwrap();
        assert_eq!(
            (child_a.range.start, child_a.range.end),
            (0, 48),
            "the referenced ranges' union"
        );
        assert_eq!(child_a.writes, 1);
        assert_eq!(
            keep_alive.writers.len(),
            2,
            "a and b are both written through"
        );
    }
    assert!(released.is_empty());
    assert_eq!(
        a.header
            .parent_count
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );
    assert!(
        parent
            .header
            .writes_below
            .load(std::sync::atomic::Ordering::Relaxed)
    );
    // Element 0 references b alone, element 1 too: a is let go.
    {
        let mut keep_alive = keep_alive_of(parent);
        keep_alive.replace(
            parent,
            4,
            0,
            &[to_buffer(b, 0..16), to_buffer(b, 0..16)],
            &mut released,
        );
        assert!(released.is_empty(), "element 1 still references a");
        keep_alive.replace(
            parent,
            4,
            1,
            &[to_buffer(b, 0..16), to_buffer(b, 0..16)],
            &mut released,
        );
        assert_eq!(keep_alive.children.iter().count(), 1);
        assert_eq!(keep_alive.writers.len(), 1);
    }
    assert_eq!(released.len(), 1);
    assert_eq!(
        a.header
            .parent_count
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
    // A reference to the buffer itself keeps nothing alive.
    let mut keep_alive = keep_alive_of(parent);
    keep_alive.replace(
        parent,
        4,
        2,
        &[to_buffer(parent, 0..8), to_buffer(b, 0..8)],
        &mut released,
    );
    assert_eq!(keep_alive.children.iter().count(), 1);
}

/// Children past the fourth are found through a hash index, and freed entries are reused.
#[test]
fn children_past_four_are_hashed() {
    let parent = fake_buffer(10 << 40);
    let children: Vec<_> = (0..7).map(|i| fake_buffer((11 + i) << 40)).collect();
    let mut keep_alive = keep_alive_of(parent);
    let mut released = Vec::new();
    for (i, pair) in children.chunks(2).enumerate() {
        let second = pair.get(1).unwrap_or(&pair[0]);
        keep_alive.replace(
            parent,
            8,
            i,
            &[to_buffer(pair[0], 0..4), to_buffer(second, 0..4)],
            &mut released,
        );
    }
    assert_eq!(keep_alive.children.iter().count(), 7);
    // Every one found again: each element written again keeps the same children.
    for (i, pair) in children.chunks(2).enumerate() {
        let second = pair.get(1).unwrap_or(&pair[0]);
        keep_alive.replace(
            parent,
            8,
            i,
            &[to_buffer(pair[0], 0..4), to_buffer(second, 0..4)],
            &mut released,
        );
    }
    assert_eq!(keep_alive.children.iter().count(), 7);
    assert!(released.is_empty());
    // One let go, a new one takes its entry.
    keep_alive.replace(
        parent,
        8,
        0,
        &[to_buffer(children[1], 0..4), to_buffer(children[1], 0..4)],
        &mut released,
    );
    assert_eq!(released.len(), 1);
    let newcomer = fake_buffer(30 << 40);
    keep_alive.replace(
        parent,
        8,
        4,
        &[to_buffer(newcomer, 0..4), to_buffer(newcomer, 0..4)],
        &mut released,
    );
    assert_eq!(keep_alive.children.iter().count(), 7);
}

/// A write marks the object's ancestors, stopping at one that has the bits already; a walk of a
/// dirty buffer reaches its dirty children and its writers and leaves it clean.
#[test]
fn dirty_bits_climb_and_walks_clear_them() {
    use std::sync::atomic::Ordering::Relaxed;
    let (outer, inner) = (fake_buffer(40 << 40), fake_buffer(41 << 40));
    let (read, written) = (fake_buffer(42 << 40), fake_buffer(43 << 40));
    let mut released = Vec::new();
    keep_alive_of(inner).replace(
        inner,
        1,
        0,
        &[to_buffer(read, 0..4), to_buffer(written, 0..4)],
        &mut released,
    );
    // The outer buffer's write reference is to itself, which keeps nothing alive.
    keep_alive_of(outer).replace(
        outer,
        1,
        0,
        &[to_buffer(inner, 0..4), to_buffer(outer, 0..4)],
        &mut released,
    );
    assert!(
        outer.header.writes_below.load(Relaxed),
        "a writer below, in the inner buffer"
    );
    mark_dirty(&**read, 1 << 3);
    assert_eq!(inner.header.dirty.load(Relaxed), 1 << 3);
    assert_eq!(outer.header.dirty.load(Relaxed), 1 << 3);
    mark_dirty(&**written, DIRTY_LAYOUT);
    assert_eq!(outer.header.dirty.load(Relaxed), 1 << 3 | DIRTY_LAYOUT);

    let mut reached = smallvec::SmallVec::new();
    let mask = 1 << 3 | DIRTY_LAYOUT;
    walk(
        &**outer,
        outer.keep_alive().unwrap(),
        true,
        mask,
        &mut reached,
    );
    let keys: Vec<usize> = reached.iter().map(|r| r.object.key()).collect();
    for child in [inner, read, written] {
        assert!(keys.contains(&Strong::Buffer(child.clone()).key()));
    }
    let is = |r: &&crate::command_buffer::Reached, b: &'static Arc<BufferInternal>| {
        r.object.key() == Strong::Buffer(b.clone()).key()
    };
    assert!(
        reached.iter().find(|r| is(r, written)).unwrap().write,
        "a write reference"
    );
    assert!(!reached.iter().find(|r| is(r, read)).unwrap().write);
    for object in [outer, inner, read, written] {
        assert_eq!(
            object.header.dirty.load(Relaxed) & mask,
            0,
            "clean after the walk"
        );
    }
    // Clean, but a writer below: a walk still reaches the writer.
    let mut reached = smallvec::SmallVec::new();
    walk(
        &**outer,
        outer.keep_alive().unwrap(),
        false,
        mask,
        &mut reached,
    );
    assert_eq!(reached.len(), 1);
    assert_eq!(
        reached[0].object.key(),
        Strong::Buffer(written.clone()).key()
    );
}

/// What a walk reaches goes through the barrier computation like a named access: an image
/// outside `GENERAL` is transitioned back. And a transition away from `GENERAL` marks the
/// buffers that reference the image, whose next walk reaches it.
#[test]
fn reached_images_are_transitioned_and_transitions_mark_their_buffers() {
    use std::sync::atomic::Ordering::Relaxed;
    let image = tracked(50, [state(L::GENERAL, Some(Kind::Graphics))]);
    let parent = fake_buffer(50 << 40);
    let mut released = Vec::new();
    let to_image = Reference {
        object: Referenced::Image(image),
        range: (0..1).into(),
    };
    keep_alive_of(parent).replace(
        parent,
        1,
        0,
        &[to_image, to_buffer(parent, 0..4)],
        &mut released,
    );
    // Rendered into: the transition to the attachment layout marks the buffer.
    let mut rec = Rec::new();
    let attachment = image_use(
        50,
        image,
        S::COLOR_ATTACHMENT_OUTPUT,
        A::COLOR_ATTACHMENT_WRITE,
        L::ATTACHMENT_OPTIMAL,
    );
    flush_barriers(&mut rec, &[attachment], &[]).unwrap();
    assert_ne!(parent.header.dirty.load(Relaxed) & DIRTY_LAYOUT, 0);
    // A command that names the buffer reaches the image, which goes back to `GENERAL`.
    let mut reached = smallvec::SmallVec::new();
    walk(
        &**parent,
        parent.keep_alive().unwrap(),
        true,
        1 | DIRTY_LAYOUT,
        &mut reached,
    );
    let images: Vec<ImageUse> = reached
        .iter()
        .filter_map(crate::command_buffer::Reached::image_use)
        .collect();
    assert_eq!(images.len(), 1);
    let barriers = flush_barriers(&mut rec, &images, &[]).unwrap();
    let [transition] = barriers.images.as_slice() else {
        panic!("one transition")
    };
    assert_eq!(
        (transition.old_layout, transition.new_layout),
        (L::ATTACHMENT_OPTIMAL, L::GENERAL)
    );
    assert_eq!(
        parent.header.dirty.load(Relaxed) & (1 | DIRTY_LAYOUT),
        0,
        "clean again"
    );
}

/// A command that reads a child through a clean referencing buffer records only its read of
/// the buffer: a later write of the child waits for that read, and a transition of a referenced
/// image too.
#[test]
fn writes_of_referenced_objects_wait_for_reads_through_their_buffers() {
    let child = fake_buffer(60 << 40);
    let parent = fake_buffer(61 << 40);
    let image = tracked(60, [state(L::GENERAL, Some(Kind::Graphics))]);
    let mut released = Vec::new();
    let to_image = Reference {
        object: Referenced::Image(image),
        range: (0..1).into(),
    };
    keep_alive_of(parent).replace(
        parent,
        1,
        0,
        &[to_buffer(child, 0..4), to_buffer(parent, 0..4)],
        &mut released,
    );
    let grandparent = fake_buffer(62 << 40);
    let image_parent = fake_buffer(63 << 40);
    keep_alive_of(image_parent).replace(
        image_parent,
        1,
        0,
        &[to_image, to_buffer(image_parent, 0..4)],
        &mut released,
    );
    keep_alive_of(grandparent).replace(
        grandparent,
        1,
        0,
        &[to_buffer(parent, 0..4), to_buffer(grandparent, 0..4)],
        &mut released,
    );
    let access = |b: &'static Arc<BufferInternal>, stage, access| BufferAccess {
        unordered: false,
        stage,
        access,
        range: (b.address..b.address + 4).into(),
        buffer: Some(b),
        #[cfg(feature = "acceleration-structure")]
        structure: None,
    };
    let mut rec = Rec::new();
    // A fragment shader reads the child through the grandparent, a vertex shader the image
    // through its buffer: only the named buffers' reads are recorded.
    let read = A::SHADER_STORAGE_READ;
    rec.record(&[], &[access(grandparent, S::FRAGMENT_SHADER, read)]);
    rec.record(&[], &[access(image_parent, S::VERTEX_SHADER, read)]);
    let copy = access(child, S::TRANSFER, A::TRANSFER_WRITE);
    assert_eq!(
        flush(&mut rec, &[], &[copy]),
        Some((S::FRAGMENT_SHADER, read, 0)),
        "the write of the child waits for the read two buffers up"
    );
    let attachment = image_use(
        60,
        image,
        S::COLOR_ATTACHMENT_OUTPUT,
        A::COLOR_ATTACHMENT_WRITE,
        L::ATTACHMENT_OPTIMAL,
    );
    let barriers = flush_barriers(&mut rec, &[attachment], &[]).unwrap();
    let [transition] = barriers.images.as_slice() else {
        panic!("one transition")
    };
    assert!(
        transition.src_stage_mask.contains(S::VERTEX_SHADER),
        "{:?}",
        transition.src_stage_mask
    );
    // A child nothing references waits for nothing of the kind.
    let alone = fake_buffer(64 << 40);
    let copy = access(alone, S::TRANSFER, A::TRANSFER_WRITE);
    assert_eq!(flush(&mut rec, &[], &[copy]), None);
}

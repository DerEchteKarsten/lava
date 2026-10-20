//! lava's unit tests: the barrier tracking, the host-access check, buffer memory locations, image states, image usage types, the slot allocator, the staging ring's positions, the descriptor heap layout, feature detection, queue family picking and the layouts the GPU reads
use std::{
    iter,
    sync::atomic::AtomicU64,
};

use crate::vk::{self, AccessFlags2 as A, Handle, ImageLayout as L, PipelineStageFlags2 as S};

use crate::{
    buffer::staging::{Reservation, Ring, STAGING_ALIGN, Uploader},
    bindless::{Combined, HeapLayout, ImageHandle, Ranges, Sampled, Storage},
    command_buffer::{
        Barriers, BufferAccess, EVERY_BUFFER, ImageAccess, ImageUse, compute_barriers,
        image_state::{ImageState, check_states, release_states, released_by, take_checked_states},
        pending::{MAX_PENDING_BUFFER_WRITES, PendingAccesses},
    },
    features::{DeviceFacts, FeatureChain, Features},
    image::usage::{ImageUsage, StaticUsage, Usage, usage},
    queue::{FamilyInfo, Kind},
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

fn buffer(stage: S, access: A, range: std::ops::Range<u64>) -> BufferAccess {
    BufferAccess {
        stage,
        access,
        range: range.into(),
    }
}

fn read(range: std::ops::Range<u64>) -> BufferAccess {
    buffer(S::VERTEX_SHADER, A::SHADER_STORAGE_READ, range)
}

fn write(range: std::ops::Range<u64>) -> BufferAccess {
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

/// An access of image `id` with its own tracked state, which starts as `old`.
fn image(id: u64, stage: S, access: A, old: ImageState, layout: L) -> ImageUse<'static> {
    let states = std::slice::from_ref(Box::leak(Box::new(old.atomic())));
    image_use(id, states, stage, access, layout)
}

/// An access of mip 0 of image `id`, whose tracked states are `states`.
fn image_use<'a>(id: u64, states: &'a [AtomicU64], stage: S, access: A, layout: L) -> ImageUse<'a> {
    ImageUse {
        access: ImageAccess {
            stage,
            access,
            image: vk::Image::from_raw(id),
            layout,
            aspect: vk::ImageAspectFlags::COLOR,
            mip_range: (0..1).into(),
        },
        states,
    }
}

/// Records `images` and `buffers` after the barriers they need, like `Recording::flush_pending`.
fn flush_barriers(pending: &mut PendingAccesses, images: &[ImageUse], buffers: &[BufferAccess]) -> Option<Barriers> {
    let barriers = compute_barriers(pending, &GRAPHICS, images.iter().cloned(), buffers.iter().cloned());
    if let Some(barriers) = &barriers {
        pending.made_visible(barriers, 1);
    }
    pending.record(1, images.iter().map(|i| i.access.clone()), buffers.iter().cloned());
    barriers
}

/// `flush_barriers`' memory barrier's source and its number of image barriers.
fn flush(pending: &mut PendingAccesses, images: &[ImageUse], buffers: &[BufferAccess]) -> Option<(S, A, usize)> {
    flush_barriers(pending, images, buffers).map(|b| {
        let memory = b.memory.map_or((S::NONE, A::NONE), |m| (m.src_stage_mask, m.src_access_mask));
        (memory.0, memory.1, b.images.len())
    })
}

#[test]
fn buffer_hazards_wait_for_the_overlapping_accesses() {
    let mut pending = PendingAccesses::default();
    assert_eq!(flush(&mut pending, &[], &[write(0..100)]), None, "nothing pending");
    // Read after write, from the writer's stage.
    assert_eq!(flush(&mut pending, &[], &[read(10..20)]), Some((S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE, 0)));
    // The barrier made the write visible to that stage and access already.
    assert_eq!(flush(&mut pending, &[], &[read(50..60)]), None);
    // Disjoint and adjacent ranges don't conflict, read after read neither.
    assert_eq!(flush(&mut pending, &[], &[write(100..200)]), None);
    // Write after read waits for the reader (and the older write).
    assert_eq!(flush(&mut pending, &[], &[write(0..15)]), Some((S::VERTEX_SHADER | S::COMPUTE_SHADER, A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE, 0)));
}

/// A partly overlapping write keeps the older write pending; reads of completed submissions
/// are dropped.
#[test]
fn pending_accesses_shrink_only_when_covered_or_completed() {
    let mut pending = PendingAccesses::default();
    pending.record(1, iter::empty(), [write(0..100), buffer(S::TRANSFER, A::TRANSFER_WRITE, 50..60)].into_iter());
    assert_eq!(pending.buffer_writes.len(), 2);
    pending.record(1, iter::empty(), [write(0..100)].into_iter());
    assert_eq!(pending.buffer_writes.len(), 1, "a covering write replaces both");
    pending.record(2, iter::empty(), [read(0..10)].into_iter());
    pending.retire(1);
    assert_eq!(pending.buffer_reads.len(), 1);
    pending.retire(2);
    assert!(pending.buffer_reads.is_empty());
}

/// Two barriers, to sampled reads in fragment shaders and to storage reads in compute shaders,
/// don't make a write visible to sampled reads in compute shaders.
#[test]
fn write_visibility_is_per_barrier() {
    let general = state(L::GENERAL, Some(Kind::Graphics));
    let at = |stage, access| image(1, stage, access, general, L::GENERAL);
    let mut pending = PendingAccesses::default();
    flush(&mut pending, &[at(S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE)], &[]);
    assert!(flush(&mut pending, &[at(S::FRAGMENT_SHADER, A::SHADER_SAMPLED_READ)], &[]).is_some());
    assert!(flush(&mut pending, &[at(S::COMPUTE_SHADER, A::SHADER_STORAGE_READ)], &[]).is_some());
    assert!(flush(&mut pending, &[at(S::COMPUTE_SHADER, A::SHADER_SAMPLED_READ)], &[]).is_some());
    assert!(flush(&mut pending, &[at(S::FRAGMENT_SHADER, A::SHADER_SAMPLED_READ)], &[]).is_none());
}

/// Mips in different states get one transition per run of neighbours with the same old state.
#[test]
fn image_states_move_in_runs_of_mips() {
    let g = |layout| state(layout, Some(Kind::Graphics));
    let states: Vec<AtomicU64> =
        [ImageState::UNDEFINED, ImageState::UNDEFINED, g(L::GENERAL), g(L::ATTACHMENT_OPTIMAL)]
            .map(ImageState::atomic)
            .into();
    let mut runs = Vec::new();
    take_checked_states(&states, (0..4).into(), L::GENERAL, Kind::Graphics, |mips, old| {
        runs.push((mips.start, mips.end, old.layout))
    });
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
    assert!(std::panic::catch_unwind(|| check_states(&states, (0..1).into(), Kind::Graphics)).is_err());
    let mut released = Vec::new();
    release_states(&states, (0..1).into(), Kind::Compute, Kind::Graphics, |mips, layout| {
        released.push((mips.start, layout))
    });
    assert_eq!(released, [(0, L::GENERAL)]);
    check_states(&states, (0..1).into(), Kind::Graphics);

    let access = image(3, S::FRAGMENT_SHADER, A::SHADER_SAMPLED_READ, ImageState::load(&states[0]), L::GENERAL);
    let barriers = compute_barriers(&PendingAccesses::default(), &GRAPHICS, iter::once(access.clone()), iter::empty()).unwrap();
    let [acquire] = barriers.acquires.as_slice() else { panic!("one acquire") };
    assert_eq!((acquire.src_queue_family_index, acquire.dst_queue_family_index), (1, 0));
    assert!(barriers.images.is_empty(), "same layout");
    assert_eq!(ImageState::load(&access.states[0]).owner, Some(Kind::Graphics));
}

/// Releasing one mip of a chain keeps the image's pending accesses for the mips the family
/// still owns (`released_by`); they go once every mip is released.
#[test]
fn releasing_one_mip_keeps_the_others() {
    let states: Vec<AtomicU64> =
        [state(L::GENERAL, Some(Kind::Graphics)), state(L::GENERAL, Some(Kind::Graphics)), ImageState::UNDEFINED]
            .map(ImageState::atomic)
            .into();
    let mut released = Vec::new();
    release_states(&states, (1..2).into(), Kind::Graphics, Kind::Transfer, |mips, _| {
        released.push((mips.start, mips.end))
    });
    assert_eq!(released, [(1, 2)]);
    assert_eq!(ImageState::load(&states[0]).released_to, None);
    assert!(!released_by(&states, Kind::Graphics), "mip 0 is still owned");
    release_states(&states, (0..1).into(), Kind::Graphics, Kind::Transfer, |_, _| {});
    assert!(released_by(&states, Kind::Graphics), "mip 2 is unowned, the others released");
}

/// A layout transition is the image's latest write, at its barrier's destination: a later read
/// at another stage waits for that stage, which chains after the transition, not for the
/// attachment write before it.
#[test]
fn transitions_are_writes_at_their_destination() {
    let states = [ImageState::UNDEFINED.atomic()];
    let at = |stage, access, layout| image_use(1, &states, stage, access, layout);
    let mut pending = PendingAccesses::default();
    let target = at(S::COLOR_ATTACHMENT_OUTPUT, A::COLOR_ATTACHMENT_WRITE, L::ATTACHMENT_OPTIMAL);
    flush(&mut pending, &[target], &[]);
    let sampled = flush_barriers(&mut pending, &[at(S::FRAGMENT_SHADER, A::SHADER_SAMPLED_READ, L::GENERAL)], &[]).unwrap();
    let [transition] = sampled.images.as_slice() else { panic!("one transition") };
    assert_eq!((transition.src_stage_mask, transition.dst_stage_mask), (S::COLOR_ATTACHMENT_OUTPUT, S::FRAGMENT_SHADER));
    let [write] = pending.image_writes.as_slice() else { panic!("one write") };
    assert_eq!((write.write.stage, write.write.access), (S::FRAGMENT_SHADER, A::NONE));
    assert!(write.visible_to(S::FRAGMENT_SHADER, A::SHADER_SAMPLED_READ));

    let compute = flush_barriers(&mut pending, &[at(S::COMPUTE_SHADER, A::SHADER_SAMPLED_READ, L::GENERAL)], &[]).unwrap();
    let memory = compute.memory.expect("a memory barrier");
    assert_eq!(memory.src_stage_mask, S::FRAGMENT_SHADER, "chains after the transition");
    assert!(compute.images.is_empty());
}

/// An acquire is a write as well: a copy after an acquired image was sampled waits for it. A
/// transition right after the acquire comes after the acquire's stages.
#[test]
fn acquires_are_writes_and_transitions_follow_them() {
    let released = ImageState { released_to: Some(Kind::Graphics), ..state(L::GENERAL, Some(Kind::Compute)) };
    let states = [released.atomic()];
    let mut pending = PendingAccesses::default();
    let sampled = flush_barriers(&mut pending, &[image_use(2, &states, S::FRAGMENT_SHADER, A::SHADER_SAMPLED_READ, L::GENERAL)], &[]).unwrap();
    assert_eq!(sampled.acquires.len(), 1);
    let copy = flush(&mut pending, &[image_use(2, &states, S::TRANSFER, A::TRANSFER_READ, L::GENERAL)], &[]);
    assert_eq!(copy.map(|(stage, ..)| stage), Some(S::FRAGMENT_SHADER), "the copy waits for the acquire");

    // A stale access of the image on this timeline doesn't stand in for the acquire.
    let states = [released.atomic()];
    let mut pending = PendingAccesses::default();
    pending.record(1, iter::once(image_use(3, &states, S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE, L::GENERAL).access), iter::empty());
    let target = image_use(3, &states, S::COLOR_ATTACHMENT_OUTPUT, A::COLOR_ATTACHMENT_WRITE, L::ATTACHMENT_OPTIMAL);
    let barriers = flush_barriers(&mut pending, &[target], &[]).unwrap();
    let ([acquire], [transition]) = (barriers.acquires.as_slice(), barriers.images.as_slice()) else { panic!("an acquire and a transition") };
    assert!(transition.src_stage_mask.contains(acquire.dst_stage_mask));
}

/// The family that released an image can't use it any more: the receiver owns it now.
#[test]
fn a_released_image_is_the_receivers() {
    let states = [state(L::GENERAL, Some(Kind::Graphics)).atomic()];
    release_states(&states, (0..1).into(), Kind::Graphics, Kind::Compute, |_, _| {});
    let panic = std::panic::catch_unwind(|| check_states(&states, (0..1).into(), Kind::Graphics)).unwrap_err();
    let message = panic.downcast_ref::<String>().unwrap();
    assert!(message.contains("released to the Compute family"), "{message}");
    check_states(&states, (0..1).into(), Kind::Compute);
}

/// An image no command accessed has nothing to wait for; one with no tracked accesses may still
/// be in use by a submission this one waited for; an acquired swapchain image waits for its
/// semaphore, not for the accesses before its present.
#[test]
fn transitions_wait_only_for_what_can_still_run() {
    let mut pending = PendingAccesses::default();
    let fresh = flush_barriers(&mut pending, &[image(4, S::TRANSFER, A::TRANSFER_WRITE, ImageState::UNDEFINED, L::GENERAL)], &[]).unwrap();
    assert_eq!((fresh.images[0].src_stage_mask, fresh.images[0].src_access_mask), (S::NONE, A::NONE));
    assert!(fresh.memory.is_none());

    let used = state(L::ATTACHMENT_OPTIMAL, Some(Kind::Graphics));
    let old = flush_barriers(&mut pending, &[image(5, S::TRANSFER, A::TRANSFER_WRITE, used, L::GENERAL)], &[]).unwrap();
    assert_eq!(old.images[0].src_stage_mask, S::ALL_COMMANDS);

    let states = [state(L::GENERAL, Some(Kind::Graphics)).atomic()];
    let target = |access| image_use(6, &states, S::COLOR_ATTACHMENT_OUTPUT, access, L::ATTACHMENT_OPTIMAL);
    flush(&mut pending, &[target(A::COLOR_ATTACHMENT_WRITE)], &[]);
    states[0].store(ImageState { acquired: true, ..ImageState::UNDEFINED }.pack(), std::sync::atomic::Ordering::Relaxed);
    let acquired = flush_barriers(&mut pending, &[target(A::COLOR_ATTACHMENT_WRITE)], &[]).unwrap();
    assert!(acquired.memory.is_none(), "the acquire's semaphore covers the last frame's writes");
    assert_eq!(acquired.images[0].src_stage_mask, S::COLOR_ATTACHMENT_OUTPUT);
    assert_eq!(acquired.acquired.len(), 1);
}

/// Image reads of completed submissions are dropped like buffer reads, and completed buffer
/// writes fold into one of every buffer, which keeps the visibility all of them had.
#[test]
fn completed_accesses_retire_and_folds_keep_their_visibility() {
    let mut pending = PendingAccesses::default();
    let sampled = image(7, S::FRAGMENT_SHADER, A::SHADER_SAMPLED_READ, state(L::GENERAL, Some(Kind::Graphics)), L::GENERAL);
    pending.record(3, iter::once(sampled.access), iter::empty());
    pending.retire(2);
    assert_eq!(pending.image_reads.len(), 1);
    pending.retire(3);
    assert!(pending.image_reads.is_empty());

    let mut pending = PendingAccesses::default();
    for i in 0..=MAX_PENDING_BUFFER_WRITES as u64 {
        flush(&mut pending, &[], &[write(i * 16..i * 16 + 16)]);
        flush(&mut pending, &[], &[read(i * 16..i * 16 + 16)]);
    }
    pending.retire(1);
    let [every] = pending.buffer_writes.as_slice() else { panic!("one folded write") };
    assert_eq!(every.write.range, EVERY_BUFFER);
    assert!(every.visible_to(S::VERTEX_SHADER, A::SHADER_STORAGE_READ));
    assert_eq!(flush(&mut pending, &[], &[read(0..16)]), None, "already visible");
    assert!(flush(&mut pending, &[], &[buffer(S::TRANSFER, A::TRANSFER_READ, 0..16)]).is_some());
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
    assert_eq!(reserve(64), None, "the padding leaves no room for a whole ring");
    assert_eq!(reserve(32), Some((112..160, 0)));
    ring.count_out();
    ring.reclaim(iter::empty());
    assert_eq!(ring.used(), 0);

    // A writer counted in pins the tail at its reservation (its region isn't published yet),
    // also when later writers published theirs or nothing else is in use.
    let writing = ring.reserve(16).unwrap();
    assert_eq!(writing, Reservation { range: 160..176, offset: 32 });
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
    same(<usage![TransferDst, Sampled]>::default(), <usage![Sampled, TransferDst]>::default());
    same(<usage![Storage]>::default(), Usage::<false, false, false, true, false, false, false>);
    assert_eq!(<usage![TransferDst, Sampled]>::VK, U::SAMPLED | U::TRANSFER_DST);
    assert_eq!(
        <usage![ColorAttachment, Storage, TransferSrc, TransferDst]>::default().flags(),
        U::COLOR_ATTACHMENT | U::STORAGE | U::TRANSFER_SRC | U::TRANSFER_DST
    );
    assert_eq!(<usage![DepthAttachment, Sampled]>::VK, U::DEPTH_STENCIL_ATTACHMENT | U::SAMPLED);
    #[cfg(feature = "host-image-copy")]
    assert_eq!(<usage![HostCopy, Sampled]>::VK, U::SAMPLED | U::HOST_TRANSFER_EXT);
    assert_eq!(format!("{:?}", <usage![TransferSrc]>::default()), "Usage(TRANSFER_SRC)");
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
    assert_eq!(ranges.take(10), Some(0), "freed one at a time, the slots merged back");
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
    assert_eq!((sampled.sampled_slot(), sampled.storage_slot()), (Some(5), None));
    let storage = ImageHandle::<Storage>::new(5);
    assert_eq!(storage.bits(), 5 | 1 << 30);
    assert_eq!((storage.sampled_slot(), storage.storage_slot()), (None, Some(5)));
    let combined = ImageHandle::<Combined>::new(5);
    assert_eq!(combined.bits(), 5 | 1 << 31);
    assert_eq!((combined.sampled_slot(), combined.storage_slot()), (Some(5), Some(6)));

    // Mips are a handle's width apart, and keep its kind.
    assert_eq!(sampled.mip(3).sampled_slot(), Some(8));
    assert_eq!(storage.mip(3).storage_slot(), Some(8));
    let mip = combined.mip(3);
    assert_eq!((mip.sampled_slot(), mip.storage_slot()), (Some(11), Some(12)));

    // Retyping keeps the bits, so a combined handle read as either kind finds its descriptor.
    assert_eq!(combined.as_sampled().bits(), combined.bits());
    assert_eq!(combined.as_storage().storage_slot(), Some(6));
    assert_eq!(combined.erase().narrow::<Sampled>(), Some(combined.as_sampled()));
    assert_eq!(combined.erase().narrow::<Storage>(), Some(combined.as_storage()));
    assert_eq!(sampled.erase().narrow::<Storage>(), None);
    assert_eq!(sampled.erase().narrow::<Combined>(), None);
    assert_eq!(storage.erase().narrow::<Sampled>(), None);

    let null = ImageHandle::<Combined>::NULL;
    assert_eq!(null.bits(), !0);
    assert!(null.is_null() && ImageHandle::<Sampled>::default().is_null());
    assert_eq!((null.sampled_slot(), null.storage_slot()), (None, None));
    assert_eq!(null.erase().narrow::<Sampled>(), None);
    assert!(!ImageHandle::<Sampled>::new((1 << 30) - 1).is_null(), "the last slot is a handle");
}

#[test]
fn passes_need_the_features_of_their_capabilities() {
    let capability = |id: u32| [2 << 16 | 17, id];
    let header = [0x0723_0203u32, 0x0001_0600, 0, 1, 0];
    let words = header.into_iter().chain([1, 39, 5283].into_iter().flat_map(capability));
    let spirv: Vec<u8> = words.flat_map(u32::to_le_bytes).collect();
    let needs = Features::of_spirv(&spirv);
    #[cfg(feature = "shader-int8")]
    assert!(needs.features.shader_int8);
    #[cfg(feature = "mesh-shader")]
    assert!(needs.features.mesh_shader);
    #[cfg(not(feature = "mesh-shader"))]
    assert!(needs.compiled_out.contains(&"mesh-shader"));
    let _ = needs;
}

/// What lava enables is what it requires, and enabling every optional feature reads back as
/// supported.
#[test]
fn enabled_features_read_back_as_supported() {
    assert!(FeatureChain::enabling(&Features::default()).lava_missing(&|_| true).is_empty());
    let all = Features::from_bits(iter::repeat(true));
    let facts = DeviceFacts {
        rebar: true,
        strict_lines: true,
        dedicated_compute: true,
        dedicated_transfer: true,
    };
    assert_eq!(FeatureChain::enabling(&all).optional(&|_| true, facts), all);
}

#[test]
fn queue_families_are_graphics_and_the_dedicated_ones() {
    use vk::QueueFlags as Q;
    let family = |index, flags, num_queues, timestamps: bool| QueueFamily {
        index,
        num_queues,
        handel: vk::QueueFamilyProperties {
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
    use crate::vkobjects::acceleration_structure::Instance;
    assert_eq!(size_of::<Instance>(), size_of::<vk::AccelerationStructureInstanceKHR>());
    let instance = Instance::from_address(0xABC, glam::Affine3A::IDENTITY)
        .custom_index(5)
        .mask(0x0F)
        .sbt_offset(7);
    assert_eq!(instance.custom_index_and_mask, 0x0F00_0005);
    assert_eq!(instance.sbt_offset_and_flags & 0xFF_FFFF, 7);
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
    assert_eq!((unsized_.bytes(), unsized_.words()), (&[1, 2, 3][..], &[4, u64::MAX][..]));
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
        assert_eq!(location.memory_type_bits(0b1111, &properties), 0b1111, "{location:?}");
    }
    assert_eq!(Location::Shared.memory_type_bits(0b1111, &properties), 0b0010, "only device-local and host-visible");
    assert_eq!(Location::Shared.memory_type_bits(0b1101, &properties), 0, "fails rather than spill");
    assert!(!Location::Gpu.host_visible() && Location::Readback.host_visible());
    Dynamic(Location::Upload).host_check();
    assert!(std::panic::catch_unwind(|| Dynamic(Location::Gpu).host_check()).is_err());
}

#[test]
fn host_access_waits_for_overlapping_incomplete_writes() {
    use crate::command_buffer::{EVERY_ACCELERATION_STRUCTURE, EVERY_BUFFER, pending::incomplete_write};
    let writes = [((0..16).into(), 3), ((16..32).into(), 5), (EVERY_BUFFER, 0), (EVERY_ACCELERATION_STRUCTURE, 9)];
    let scan = |range: std::ops::Range<u64>, completed| incomplete_write(writes.iter().copied(), range.into(), completed);
    assert_eq!(scan(0..64, 0), Some(5), "the latest overlapping value");
    assert_eq!(scan(8..12, 2), Some(3));
    assert_eq!(scan(8..12, 3), None, "completed");
    assert_eq!(scan(32..64, 0), None, "touching isn't overlapping");
    assert_eq!(scan(16..20, 4), Some(5));
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

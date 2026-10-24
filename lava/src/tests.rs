//! lava's unit tests: the barrier tracking, the host-access check, buffer memory locations, image states, image usage types, texel blocks, copy sizes with row pitches, 64-bit clear values and format compatibility classes, the slot allocator, the staging ring's positions, the descriptor heap layout, feature detection and the capability scan, pipeline cache headers, queue family picking, device ranking, wait stage masks, error mapping, calls before init, the layouts the GPU reads, topology classes and render areas, half floats, and copy region overlaps
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
    image::{
        format::{self, Block, ClearValue, Format},
        slice::rows_len,
        usage::{ImageUsage, StaticUsage, Usage, usage},
    },
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
        unordered: false,
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
    // Its old contents are discarded: the transition starts from `UNDEFINED`, not from the
    // layout its last frame left.
    assert_eq!(acquired.images[0].old_layout, L::UNDEFINED);
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
    same(<usage![Storage]>::default(), Usage::<false, false, false, true, false, false, false, false>);
    assert!(<usage![Storage]>::default().create_flags().is_empty());
    assert_eq!(
        <usage![Storage, Sampled, MutableFormat]>::default().create_flags(),
        vk::ImageCreateFlags::MUTABLE_FORMAT | vk::ImageCreateFlags::EXTENDED_USAGE
    );
    assert_eq!(<usage![Storage, MutableFormat]>::VK, U::STORAGE, "no usage flag of its own");
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
    assert_eq!(capabilities, [1, 39, 5283], "the scan stops after the capabilities");
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
    let queried = FeatureChain::default();
    let enabling = |features: &Features| FeatureChain::enabling(features, &queried);
    assert!(enabling(&Features::default()).lava_missing(&|_| true).is_empty());
    let all = Features::from_bits(iter::repeat(true));
    let facts = DeviceFacts {
        rebar: true,
        strict_lines: true,
        dedicated_compute: true,
        dedicated_transfer: true,
        sparse_queue: true,
    };
    assert_eq!(enabling(&all).optional(&|_| true, facts), all);
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
    use crate::vkobjects::acceleration_structure::{Instance, InstanceFlags};
    use std::mem::offset_of;
    assert_eq!(size_of::<Instance>(), size_of::<vk::AccelerationStructureInstanceKHR>());
    assert_eq!(offset_of!(Instance, transform), offset_of!(vk::AccelerationStructureInstanceKHR, transform));
    assert_eq!(
        offset_of!(Instance, custom_index_and_mask),
        offset_of!(vk::AccelerationStructureInstanceKHR, instance_custom_index_and_mask)
    );
    assert_eq!(
        offset_of!(Instance, sbt_offset_and_flags),
        offset_of!(vk::AccelerationStructureInstanceKHR, instance_shader_binding_table_record_offset_and_flags)
    );
    assert_eq!(
        offset_of!(Instance, blas),
        offset_of!(vk::AccelerationStructureInstanceKHR, acceleration_structure_reference)
    );
    let transform = glam::Affine3A::from_translation(glam::Vec3::new(1.0, 2.0, 3.0));
    let flags = InstanceFlags::FORCE_OPAQUE | InstanceFlags::TRIANGLE_FLIP_FACING;
    let instance = Instance::from_address(0xABC, transform)
        .with_custom_index(5)
        .with_mask(0x0F)
        .with_sbt_offset(7)
        .with_flags(flags);
    assert_eq!(instance.custom_index_and_mask, 0x0F00_0005);
    assert_eq!(instance.sbt_offset_and_flags, 0x0600_0007);
    assert_eq!(
        (instance.custom_index(), instance.mask(), instance.sbt_offset(), instance.flags()),
        (5, 0x0F, 7, flags)
    );
    assert_eq!((instance.blas_address(), instance.transform()), (0xABC, transform));
    assert_eq!(flags.bits(), vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE.as_raw() as u8 | 2);
    assert!(flags.contains(InstanceFlags::FORCE_OPAQUE) && !flags.contains(InstanceFlags::FORCE_NO_OPAQUE));
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
    for (value, bits) in [(0.0, 0), (-2.0, 0xC000), (0.5, 0x3800), (65504.0, 0x7BFF), (2f32.powi(-24), 1), (-(2f32.powi(-24)), 0x8001), (f32::INFINITY, 0x7C00), (0.099975586, 0x2E66)] {
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

/// A pipeline cache loads only with this device's header: its size, version, vendor, device
/// and cache UUID.
#[test]
fn pipeline_caches_load_only_for_their_device() {
    use crate::pipeline::cache_header_matches;
    let mut properties = vk::PhysicalDeviceProperties::default();
    (properties.vendor_id, properties.device_id) = (0x8086, 0x56a0);
    properties.pipeline_cache_uuid = [7; 16];
    let header = |size: u32, version: u32, vendor: u32, uuid: [u8; 16]| {
        let words = [size, version, vendor, 0x56a0].into_iter().flat_map(u32::to_le_bytes);
        let mut data: Vec<u8> = words.chain(uuid).collect();
        data.extend([0; 64]);
        data
    };
    assert!(cache_header_matches(&header(32, 1, 0x8086, [7; 16]), &properties));
    assert!(!cache_header_matches(&header(32, 1, 0x1002, [7; 16]), &properties), "vendor");
    assert!(!cache_header_matches(&header(32, 1, 0x8086, [8; 16]), &properties), "driver");
    assert!(!cache_header_matches(&header(32, 2, 0x8086, [7; 16]), &properties), "version");
    assert!(!cache_header_matches(&header(16, 1, 0x8086, [7; 16]), &properties), "size");
    assert!(!cache_header_matches(&header(32, 1, 0x8086, [7; 16])[..20], &properties), "cut");
    assert!(!cache_header_matches(&[], &properties));
}

/// Discrete GPUs first, then integrated ones, then virtual and CPU ones; the driver's order
/// within a rank.
#[test]
fn devices_are_tried_by_type() {
    use vk::PhysicalDeviceType as T;
    let types = [T::CPU, T::OTHER, T::INTEGRATED_GPU, T::DISCRETE_GPU, T::VIRTUAL_GPU, T::DISCRETE_GPU];
    assert_eq!(crate::state::device_order(&types), [3, 5, 2, 4, 0, 1]);
    assert!(crate::state::device_rank(T::CPU) > crate::state::device_rank(T::INTEGRATED_GPU));
    assert_eq!(crate::state::device_order(&[]), Vec::<usize>::new());
}

/// Each family's stages, masked by the features: ray tracing shaders only with ray tracing
/// pipelines.
#[test]
fn wait_stages_are_masked_by_features() {
    use crate::queue::{AllStages, ComputeStage, GraphicsStage, StageSet, TransferStage};
    let none = Features::DEFAULT;
    assert_eq!(AllStages.mask(&none), S::ALL_COMMANDS);
    assert_eq!(GraphicsStage::Raster.mask(&none), S::ALL_GRAPHICS);
    assert_eq!(ComputeStage::Compute.mask(&none), S::COMPUTE_SHADER | S::DRAW_INDIRECT);
    assert_eq!(TransferStage::Transfer.mask(&none), S::ALL_TRANSFER);
    assert_eq!(GraphicsStage::RayTracing.mask(&none), S::DRAW_INDIRECT);
    #[cfg(feature = "ray-tracing-pipeline")]
    {
        let rt = Features {
            ray_tracing_pipeline: true,
            ..Features::DEFAULT
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
    assert!(matches!(Error::from(vk::Result::ERROR_DEVICE_LOST), Error::DeviceLost));
    assert!(matches!(Error::from(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY), Error::OutOfMemory));
    assert!(matches!(
        Error::from(gpu_allocator::AllocationError::OutOfMemory),
        Error::OutOfMemory
    ));
    assert!(matches!(Error::from(vk::Result::ERROR_OUT_OF_DATE_KHR), Error::SurfaceOutOfDate));
    let wrapped = Error::from(vk::Result::ERROR_UNKNOWN).context("the pipelines of `Pass`");
    assert_eq!(wrapped.to_string(), "the pipelines of `Pass`");
    let source = wrapped.source().expect("the Vulkan error").to_string();
    assert!(source.contains("ERROR_UNKNOWN"), "{source}");
    assert!(matches!(Error::DeviceLost.context("anything"), Error::DeviceLost));
    let slots = Error::OutOfBindlessSlots {
        heap: crate::bindless::Heap::Sampler,
        count: 2,
    };
    assert_eq!(slots.to_string(), "the bindless sampler heap has no 2 contiguous free slots");
    assert!(slots.source().is_none());
}

/// Without `lava::init`, lava's calls panic with one message, and shutdown does nothing.
#[test]
fn calls_before_init_panic() {
    let panicked = std::panic::catch_unwind(crate::state::call).unwrap_err();
    let message = panicked.downcast_ref::<String>().cloned().unwrap_or_default();
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
    let block = |w, h, bytes| Some(Block { width: w, height: h, bytes });
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
    assert_eq!(format::compression(F::ASTC_4X4_SRGB_BLOCK), Some("astc-compression"));
    assert_eq!(format::compression(F::EAC_R11G11_SNORM_BLOCK), Some("etc2-compression"));
    // Copies move one aspect: a combined format's depth or stencil alone.
    let aspect = |a| format::copy_block(F::D32_SFLOAT_S8_UINT, 5, a).unwrap().bytes;
    assert_eq!(aspect(vk::ImageAspectFlags::DEPTH), 4);
    assert_eq!(aspect(vk::ImageAspectFlags::STENCIL), 1);
    assert_eq!(format::copy_block(F::R8G8B8_UNORM, 3, vk::ImageAspectFlags::COLOR).unwrap().bytes, 3);
    assert_eq!(format::copy_block(F::UNDEFINED, 0, vk::ImageAspectFlags::COLOR), None);
}

#[test]
fn copy_sizes_count_whole_blocks_and_rows() {
    let extent = |w, h, d| vk::Extent3D { width: w, height: h, depth: d };
    let texel = Block { width: 1, height: 1, bytes: 4 };
    let bc = Block { width: 4, height: 4, bytes: 16 };
    assert_eq!(rows_len(texel, extent(3, 2, 1), 1, (0, 0)), 3 * 2 * 4);
    assert_eq!(rows_len(texel, extent(3, 2, 1), 2, (0, 0)), 2 * 3 * 2 * 4, "layers one after another");
    // Rows 8 texels apart: the last row ends after its 3 texels.
    assert_eq!(rows_len(texel, extent(3, 2, 1), 1, (8, 0)), (8 + 3) * 4);
    // Slices 4 rows apart, two of them.
    assert_eq!(rows_len(texel, extent(3, 2, 2), 1, (8, 4)), ((4 + 1) * 8 + 3) * 4);
    // A 6x6 box of 4x4 blocks takes 2x2 blocks; a mip narrower than its block takes one.
    assert_eq!(rows_len(bc, extent(6, 6, 1), 1, (0, 0)), 4 * 16);
    assert_eq!(rows_len(bc, extent(2, 2, 1), 1, (0, 0)), 16);
    assert_eq!(rows_len(bc, extent(4, 4, 1), 1, (16, 0)), 16, "one row of blocks");
    let astc = Block { width: 5, height: 4, bytes: 16 };
    assert_eq!(rows_len(astc, extent(10, 8, 1), 1, (0, 0)), 2 * 2 * 16);
}

#[test]
fn clear_values_pack_64_bit_components_into_two_words() {
    use crate::image::format::{R64G64Sint, R64G64Uint, R64Sfloat, R64Uint, R64G64B64A64Uint};
    assert_eq!(
        R64Uint.clear_value([0x1_0000_0002]),
        ClearValue::Uint([2, 1, 0, 0])
    );
    assert_eq!(
        R64G64Uint.clear_value([u64::MAX, 5]),
        ClearValue::Uint([u32::MAX, u32::MAX, 5, 0])
    );
    assert_eq!(R64G64Sint.clear_value([-1, 2]), ClearValue::Sint([-1, -1, 2, 0]));
    let ClearValue::Float(words) = R64Sfloat.clear_value([1.5]) else { panic!() };
    let bits = words.map(f32::to_bits);
    assert_eq!((bits[1] as u64) << 32 | bits[0] as u64, 1.5f64.to_bits());
    assert_eq!(R64G64B64A64Uint.clear_value([1, 2, 0, 0]), ClearValue::Uint([1, 0, 2, 0]));
    let third = std::panic::catch_unwind(|| R64G64B64A64Uint.clear_value([1, 2, 3, 0]));
    assert!(third.is_err(), "a clear value holds two 64-bit components");
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
    assert_ne!(class(A8Unorm), class(R8Unorm), "8-bit alpha is a class of its own");
    assert_ne!(class(D32Sfloat), class(R32Sfloat));
    assert_ne!(class(D24UnormS8Uint), class(X8D24UnormPack32));
    #[cfg(feature = "bc-compression")]
    {
        assert_eq!(class(BC7UnormBlock), class(BC7SrgbBlock));
        assert_eq!(class(BC4UnormBlock), class(BC4SnormBlock));
        assert_ne!(class(BC1RgbUnormBlock), class(BC1RgbaUnormBlock));
        assert_ne!(class(BC7UnormBlock), class(R32G32B32A32Uint), "no block-texel views");
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

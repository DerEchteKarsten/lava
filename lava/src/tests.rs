//! lava's unit tests: the barrier tracking, the host-access check, buffer memory locations, image states, image usage types, the slot allocator, the descriptor heap layout, feature detection, queue family picking and the layouts the GPU reads
use std::{iter, sync::atomic::AtomicU64};

use crate::vk::{self, AccessFlags2 as A, Handle, ImageLayout as L, PipelineStageFlags2 as S};

use crate::{
    bindless::{Combined, HeapLayout, ImageHandle, Ranges, Sampled, Storage},
    command_buffer::{
        BufferAccess, ImageAccess, ImageUse, compute_barriers,
        image_state::{ImageState, check_states, release_states, take_checked_states},
        pending::PendingAccesses,
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
    ImageUse {
        access: ImageAccess {
            stage,
            access,
            image: vk::Image::from_raw(id),
            layout,
            aspect: vk::ImageAspectFlags::COLOR,
            mip_range: (0..1).into(),
        },
        states: std::slice::from_ref(Box::leak(Box::new(old.atomic()))),
    }
}

/// Records `images` and `buffers` after the barriers they need, like `Recording::flush_pending`.
fn flush(pending: &mut PendingAccesses, images: &[ImageUse], buffers: &[BufferAccess]) -> Option<(S, A, usize)> {
    let barriers = compute_barriers(pending, &GRAPHICS, images.iter().cloned(), buffers.iter().cloned());
    if let Some(barriers) = &barriers {
        pending.made_visible(barriers);
    }
    pending.record(0, images.iter().map(|i| i.access.clone()), buffers.iter().cloned());
    barriers.map(|b| {
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

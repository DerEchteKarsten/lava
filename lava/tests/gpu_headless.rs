//! Headless GPU tests: buffers, images, sync, command recording, and golden images per pass type
//!
//! These run on the real GPU with the validation layer on; every test fails if the layer
//! reports an error. The passes come from `lava/tests/shaders`, not from the engine.
mod common;

use common::{buffer_with, golden::assert_golden, gpu, zeroed_buffer};
use glam::{IVec2, UVec2, Vec2, Vec4};
use lava::{
    bindings::{
        TestComputeBuffer, TestComputeImage, TestMesh, TestRaster, TestRasterTextured,
        TestTexturedVertex, TestVertex,
    },
    bindless::NULL_HANDLE,
    buffer::{
        Buffer,
        usage::{Index, Indirect, Storage},
    },
    command_buffer::{
        CommandBuffer, DispatchIndirectCommand, DrawIndirectCommand, Filter, Scissor, Viewport,
    },
    image::{
        Image,
        format::{BC7UnormBlock, D32Sfloat, D32SfloatS8Uint, R8G8B8A8Unorm},
        slice::AsImage,
        usage::{
            ColorAttachment, ColorAttachmentStorage, DepthAttachment, Sampled, SampledStorage,
            Storage as StorageImage,
        },
    },
    state::Ctx,
    vkobjects::queue::{
        Binary, Event, Fence, FrameSlot, Gfx, PendingAccesses, Queue, Semaphore, Timeline,
    },
};

// ---- helpers ---------------------------------------------------------------------------------

/// Side length of the square render targets.
const SIZE: u32 = 64;
const EXTENT: UVec2 = UVec2::splat(SIZE);
const BLACK: [u8; 4] = [0, 0, 0, 255];

type Target = Image<R8G8B8A8Unorm, ColorAttachmentStorage>;

/// Clears a fresh target to opaque black, lets `record` draw into it, and reads it back.
fn render(gpu: &common::Gpu, record: impl FnOnce(&mut CommandBuffer, &Target)) -> Vec<u8> {
    let target = Target::new(SIZE, SIZE).unwrap();
    let readback = zeroed_buffer::<u8, Storage>((SIZE * SIZE * 4) as usize);
    gpu.submit(|cmd| {
        cmd.clear_image(target.whole_view(), [0.0, 0.0, 0.0, 1.0]);
        record(cmd, &target);
        cmd.copy_image_to_buffer(target.whole(), readback.range(..));
    });
    readback.range(..).as_slice().to_vec()
}

fn pixel(pixels: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * SIZE + x) * 4) as usize;
    pixels[i..i + 4].try_into().unwrap()
}

fn close(a: [u8; 4], b: [u8; 4]) -> bool {
    a.iter().zip(b).all(|(a, b)| a.abs_diff(b) <= 2)
}

#[track_caller]
fn assert_pixel(pixels: &[u8], x: u32, y: u32, expected: [u8; 4]) {
    let actual = pixel(pixels, x, y);
    assert!(
        close(actual, expected),
        "pixel ({x}, {y}) is {actual:?}, expected {expected:?}"
    );
}

fn vertex(x: f32, y: f32, z: f32, color: [f32; 4]) -> TestVertex {
    TestVertex {
        position: Vec4::new(x, y, z, 1.0),
        color: Vec4::from_array(color),
    }
}

const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
const BLUE: [f32; 4] = [0.0, 0.0, 1.0, 1.0];
const YELLOW: [f32; 4] = [1.0, 1.0, 0.0, 1.0];

/// Two triangles: a front-facing one (clockwise on screen) in the left half with one colour
/// per corner, and a back-facing yellow one in the right half.
fn two_triangles() -> Buffer<TestVertex> {
    buffer_with(&[
        vertex(-0.5, -0.8, 0.5, RED),
        vertex(-0.1, 0.8, 0.5, GREEN),
        vertex(-0.9, 0.8, 0.5, BLUE),
        vertex(0.5, -0.8, 0.5, YELLOW),
        vertex(0.1, 0.8, 0.5, YELLOW),
        vertex(0.9, 0.8, 0.5, YELLOW),
    ])
}

/// A pixel well inside the left (front-facing) and the right (back-facing) triangle.
const IN_LEFT: (u32, u32) = (16, 44);
const IN_RIGHT: (u32, u32) = (48, 44);

/// Deterministic RGBA8 test pattern with a distinct colour per texel.
fn pattern(width: u32, height: u32) -> Vec<u8> {
    (0..height)
        .flat_map(|y| {
            (0..width)
                .flat_map(move |x| [(x * 31) as u8, (y * 29) as u8, (x + y * width) as u8, 255])
        })
        .collect()
}

fn texel(pixels: &[u8], width: u32, x: u32, y: u32) -> &[u8] {
    let i = ((y * width + x) * 4) as usize;
    &pixels[i..i + 4]
}

// ---- context ---------------------------------------------------------------------------------

#[test]
fn init_twice_is_an_error() {
    let _gpu = gpu();
    assert!(lava::init(None, false, false).is_err());
}

#[test]
fn context_exposes_the_graphics_queue_family() {
    let _gpu = gpu();
    assert!(Ctx::num_gfx_queues() >= 1);
    // Presentation shares the graphics family.
    assert_eq!(Ctx::present_queue_index(), Ctx::gfx_queue_index());
    // Headless: no swapchain was requested.
    assert!(!Ctx::features().present);
}

// ---- buffers ---------------------------------------------------------------------------------

#[test]
fn buffer_has_the_requested_size() {
    let _gpu = gpu();
    let buffer = Buffer::<[u32; 3], Storage>::new(5, true).unwrap();
    assert_eq!(buffer.len(), 5);
    assert_eq!(buffer.size(), 60);
    assert_eq!(buffer.range(..).len(), 5);
    assert_ne!(buffer.address, 0, "buffers are addressable from shaders");
    assert_eq!(buffer.range(2..).gpu_ptr, buffer.address + 24);
}

#[test]
fn host_writes_are_visible_through_slices_and_indexing() {
    let _gpu = gpu();
    let mut buffer = Buffer::<u32, Storage>::new(8, true).unwrap();
    buffer.range(..).copy_from(&[1, 2, 3, 4, 5, 6, 7, 8]);
    buffer[0] = 10;
    buffer.range(4..6).copy_from(&[50, 60]);

    assert_eq!(buffer.range(..).as_slice(), [10, 2, 3, 4, 50, 60, 7, 8]);
    assert_eq!(buffer[5], 60);
    assert_eq!(buffer.byte_range(4..8).cast::<u32>().as_slice(), [2]);

    let halves = buffer.cast::<[u32; 2]>();
    assert_eq!(halves.len(), 4);
    assert_eq!(halves[2], [50, 60]);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn indexing_past_the_buffer_panics() {
    let _gpu = gpu();
    let buffer = Buffer::<u32, Storage>::new(4, true).unwrap();
    let _ = buffer[4];
}

// ---- queues and synchronisation --------------------------------------------------------------

#[test]
fn fence_signals_when_the_submission_finished() {
    let gpu = gpu();
    let buffer = zeroed_buffer::<u32, Storage>(4);
    let pool = gpu.queue.create_pool().unwrap();
    let memory = pool.create_command_buffer().unwrap();
    let fence = Fence::new().unwrap();

    gpu.queue
        .execute_command(
            PendingAccesses::default(),
            &memory,
            Some(&fence),
            &[],
            &[],
            |cmd| {
                cmd.fill_buffer(buffer.range(..), 7);
            },
        )
        .unwrap();
    fence.wait().unwrap();
    assert_eq!(buffer.range(..).as_slice(), [7; 4]);

    // A reset fence can be reused for the next submission.
    fence.reset().unwrap();
    pool.reset().unwrap();
    gpu.queue
        .execute_command(
            PendingAccesses::default(),
            &memory,
            Some(&fence),
            &[],
            &[],
            |cmd| {
                cmd.fill_buffer(buffer.range(..), 8);
            },
        )
        .unwrap();
    fence.wait().unwrap();
    assert_eq!(buffer.range(..).as_slice(), [8; 4]);
}

#[test]
fn timeline_semaphore_reaches_the_signalled_value() {
    let gpu = gpu();
    let semaphore = Semaphore::<Timeline>::new().unwrap();
    let buffer = zeroed_buffer::<u32, Storage>(4);
    gpu.submit_with(&[], &[semaphore.info(3)], |cmd| {
        cmd.fill_buffer(buffer.range(..), 1)
    });
    semaphore.block_until_value(3).unwrap();
    // Earlier values count as reached too.
    semaphore.block_until_value(1).unwrap();
}

#[test]
fn binary_semaphore_orders_two_submissions() {
    let gpu = gpu();
    let semaphore = Semaphore::<Binary>::new().unwrap();
    let buffer = zeroed_buffer::<u32, Storage>(4);
    gpu.submit_with(&[], &[semaphore.info()], |cmd| {
        cmd.fill_buffer(buffer.range(..), 1)
    });
    gpu.submit_with(&[semaphore.info()], &[], |cmd| {
        cmd.fill_buffer(buffer.range(2..), 2)
    });
    assert_eq!(buffer.range(..).as_slice(), [1, 1, 2, 2]);
}

#[test]
fn event_wait_returns_once_set() {
    let _gpu = gpu();
    let event = Event::new().unwrap();
    event.set().unwrap();
    event.wait().unwrap();
}

#[test]
fn queues_are_handed_out_until_the_family_is_exhausted() {
    let _gpu = gpu();
    // The harness holds one queue; the rest of the family is still free.
    let free = Ctx::num_gfx_queues() as usize - 1;
    let queues: Vec<Queue<Gfx>> = (0..free).map(|_| Queue::new().unwrap()).collect();
    assert!(Queue::<Gfx>::new().is_err(), "all queues are in use");

    drop(queues);
    if free > 0 {
        // Dropping a queue frees its slot, and the queue is usable.
        let queue = Queue::<Gfx>::new().unwrap();
        let buffer = zeroed_buffer::<u32, Storage>(4);
        let mut slot = FrameSlot::new(&queue).unwrap();
        slot.begin()
            .unwrap()
            .execute(&queue, PendingAccesses::default(), &[], &[], |cmd| {
                cmd.fill_buffer(buffer.range(..), 9)
            })
            .unwrap();
        drop(slot);
        assert_eq!(buffer.range(..).as_slice(), [9; 4]);
    }
}

#[test]
fn frame_slot_can_be_reused_and_keeps_retired_values_until_the_frame_is_done() {
    struct SetOnDrop(std::sync::Arc<std::sync::atomic::AtomicBool>);
    impl Drop for SetOnDrop {
        fn drop(&mut self) {
            self.0.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }

    let gpu = gpu();
    let buffer = zeroed_buffer::<u32, Storage>(4);
    let dropped = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut slot = FrameSlot::new(&gpu.queue).unwrap();
    let mut pending = PendingAccesses::default();

    for value in 1..=3 {
        let mut frame = slot.begin().unwrap();
        if value == 1 {
            frame.retire(SetOnDrop(dropped.clone()));
        } else {
            // Beginning the next frame waited for the first one and released what it retired.
            assert!(dropped.load(std::sync::atomic::Ordering::SeqCst));
        }
        // Pending accesses are carried from frame to frame like the engine does.
        pending = frame
            .execute(&gpu.queue, pending, &[], &[], |cmd| {
                cmd.fill_buffer(buffer.range(..), value)
            })
            .unwrap();
        if value == 1 {
            assert!(!dropped.load(std::sync::atomic::Ordering::SeqCst));
        }
    }
    drop(slot);
    assert_eq!(buffer.range(..).as_slice(), [3; 4]);
}

// ---- buffer commands -------------------------------------------------------------------------

#[test]
fn fill_buffer_fills_exactly_the_slice() {
    let gpu = gpu();
    let buffer = zeroed_buffer::<u32, Storage>(8);
    gpu.submit(|cmd| cmd.fill_buffer(buffer.range(2..6), 0xDEAD_BEEF));
    assert_eq!(
        buffer.range(..).as_slice(),
        [
            0,
            0,
            0xDEAD_BEEF,
            0xDEAD_BEEF,
            0xDEAD_BEEF,
            0xDEAD_BEEF,
            0,
            0
        ]
    );
}

#[test]
fn update_buffer_writes_one_element() {
    let gpu = gpu();
    let buffer = zeroed_buffer::<[u32; 2], Storage>(4);
    gpu.submit(|cmd| cmd.update_buffer(buffer.range(2..3), &[11, 22]));
    assert_eq!(
        buffer.range(..).as_slice(),
        [[0, 0], [0, 0], [11, 22], [0, 0]]
    );
}

#[test]
fn copy_buffer_copies_the_source_slice_to_the_destination_offset() {
    let gpu = gpu();
    let src = buffer_with::<u32, Storage>(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let dst = zeroed_buffer::<u32, Storage>(8);
    gpu.submit(|cmd| cmd.copy_buffer(src.range(1..4), dst.range(5..)));
    assert_eq!(dst.range(..).as_slice(), [0, 0, 0, 0, 0, 2, 3, 4]);
}

#[test]
fn copy_buffer_regions_copies_each_region() {
    let gpu = gpu();
    let src = buffer_with::<u32, Storage>(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let dst = zeroed_buffer::<u32, Storage>(8);
    let regions = [
        src.range(0..2).region(dst.range(6..)),
        src.range(7..8).region(dst.range(0..1)),
    ];
    gpu.submit(|cmd| cmd.copy_buffer_regions(src.range(..), dst.range(..), &regions));
    assert_eq!(dst.range(..).as_slice(), [8, 0, 0, 0, 0, 0, 1, 2]);
}

/// Regression for the barrier tracking: a write that partially overlaps an earlier write,
/// followed by reads of both parts, all in one command buffer. Synchronisation validation
/// reports the missing barrier if the first write is forgotten.
#[test]
fn partially_overlapping_writes_are_synchronised_with_later_reads() {
    let gpu = gpu();
    let buffer = zeroed_buffer::<u32, Storage>(16);
    let copy = zeroed_buffer::<u32, Storage>(16);
    gpu.submit(|cmd| {
        cmd.fill_buffer(buffer.range(..), 1);
        cmd.fill_buffer(buffer.range(4..8), 2);
        cmd.copy_buffer(buffer.range(0..2), copy.range(0..2));
        cmd.copy_buffer(buffer.range(2..), copy.range(2..));
    });
    let expected: Vec<u32> = (0..16)
        .map(|i| if (4..8).contains(&i) { 2 } else { 1 })
        .collect();
    assert_eq!(copy.range(..).as_slice(), expected);
}

// ---- images ----------------------------------------------------------------------------------

#[test]
fn image_reports_extent_and_mip_chain() {
    let _gpu = gpu();
    let image = Image::<R8G8B8A8Unorm, Sampled>::with_mip_levels(16, 8, 4).unwrap();
    assert_eq!(image.extent, UVec2::new(16, 8));
    assert_eq!(image.mip_levels, 4);
    assert_eq!(image.mip_extent(0), UVec2::new(16, 8));
    assert_eq!(image.mip_extent(2), UVec2::new(4, 2));
    assert_eq!(image.mip_extent(3), UVec2::new(2, 1));

    let whole = image.whole_view();
    assert_eq!(whole.subresource_range().level_count, 4);
    assert_eq!(image.whole().extend, UVec2::new(16, 8));

    let tail = image
        .create_new_view((1..3).into(), Default::default())
        .unwrap();
    let range = tail.subresource_range();
    assert_eq!((range.base_mip_level, range.level_count), (1, 2));
    assert_eq!(tail.handle, image.handle);
}

#[test]
fn bindless_handles_follow_the_image_usage() {
    let _gpu = gpu();
    let sampled = Image::<R8G8B8A8Unorm, Sampled>::new(4, 4).unwrap();
    let storage = Image::<R8G8B8A8Unorm, StorageImage>::new(4, 4).unwrap();
    let both = Image::<R8G8B8A8Unorm, SampledStorage>::new(4, 4).unwrap();
    let attachment = Image::<R8G8B8A8Unorm, ColorAttachment>::new(4, 4).unwrap();

    assert_ne!(sampled.handle.descriptor_index_set0, NULL_HANDLE);
    assert_eq!(sampled.handle.descriptor_index_set1, NULL_HANDLE);
    assert_eq!(storage.handle.descriptor_index_set0, NULL_HANDLE);
    assert_ne!(storage.handle.descriptor_index_set1, NULL_HANDLE);
    assert_ne!(both.handle.descriptor_index_set0, NULL_HANDLE);
    assert_ne!(both.handle.descriptor_index_set1, NULL_HANDLE);
    assert_eq!(attachment.handle, lava::bindless::BindlessHandle::none());

    // Every registered image gets its own slot.
    assert_ne!(
        sampled.handle.descriptor_index_set0,
        both.handle.descriptor_index_set0
    );
    assert_ne!(
        storage.handle.descriptor_index_set1,
        both.handle.descriptor_index_set1
    );
}

#[test]
fn clear_image_sets_every_texel() {
    let gpu = gpu();
    let image = Image::<R8G8B8A8Unorm, StorageImage>::new(8, 8).unwrap();
    let readback = zeroed_buffer::<u8, Storage>(8 * 8 * 4);
    gpu.submit(|cmd| {
        cmd.clear_image(image.whole_view(), [1.0, 0.0, 0.5, 1.0]);
        cmd.copy_image_to_buffer(image.whole(), readback.range(..));
    });
    for texel in readback.range(..).as_slice().chunks_exact(4) {
        assert!(
            close(texel.try_into().unwrap(), [255, 0, 128, 255]),
            "{texel:?}"
        );
    }
}

#[test]
fn buffer_to_image_and_back_round_trips() {
    let gpu = gpu();
    let data = pattern(8, 8);
    let upload = buffer_with::<u8, Storage>(&data);
    let image = Image::<R8G8B8A8Unorm, Sampled>::new(8, 8).unwrap();
    let readback = zeroed_buffer::<u8, Storage>(data.len());
    gpu.submit(|cmd| {
        cmd.copy_buffer_to_image(upload.range(..), image.whole());
        cmd.copy_image_to_buffer(image.whole(), readback.range(..));
    });
    assert_eq!(readback.range(..).as_slice(), data);
}

#[test]
fn image_region_copies_address_the_sub_rectangle() {
    let gpu = gpu();
    let data = pattern(8, 8);
    let upload = buffer_with::<u8, Storage>(&data);
    let image = Image::<R8G8B8A8Unorm, Sampled>::new(8, 8).unwrap();
    let readback = zeroed_buffer::<u8, Storage>(3 * 2 * 4);
    let region = image
        .whole_view()
        .region(UVec2::new(3, 2))
        .offset(IVec2::new(4, 5));
    gpu.submit(|cmd| {
        cmd.copy_buffer_to_image(upload.range(..), image.whole());
        cmd.copy_image_to_buffer(region, readback.range(..));
    });
    let readback = readback.range(..).as_slice();
    for (y, x) in (0..2).flat_map(|y| (0..3).map(move |x| (y, x))) {
        assert_eq!(
            texel(readback, 3, x, y),
            texel(&data, 8, x + 4, y + 5),
            "({x}, {y})"
        );
    }
}

/// Regression: blits used the extent as the second corner, so regions with an offset were
/// scaled and misplaced.
#[test]
fn blit_copies_between_offset_regions() {
    let gpu = gpu();
    let data = pattern(8, 8);
    let upload = buffer_with::<u8, Storage>(&data);
    let src = Image::<R8G8B8A8Unorm, Sampled>::new(8, 8).unwrap();
    let dst = Image::<R8G8B8A8Unorm, Sampled>::new(8, 8).unwrap();
    let readback = zeroed_buffer::<u8, Storage>(data.len());
    gpu.submit(|cmd| {
        cmd.copy_buffer_to_image(upload.range(..), src.whole());
        cmd.clear_image(dst.whole_view(), [0.0, 0.0, 0.0, 1.0]);
        cmd.blit_image(
            src.whole_view()
                .region(UVec2::splat(4))
                .offset(IVec2::new(4, 3)),
            dst.whole_view()
                .region(UVec2::splat(4))
                .offset(IVec2::new(1, 2)),
            Filter::Nearest,
        );
        cmd.copy_image_to_buffer(dst.whole(), readback.range(..));
    });
    let readback = readback.range(..).as_slice();
    for (y, x) in (0..8).flat_map(|y| (0..8).map(move |x| (y, x))) {
        let inside = (1..5).contains(&x) && (2..6).contains(&y);
        let expected = if inside {
            texel(&data, 8, x + 3, y + 1)
        } else {
            &BLACK[..]
        };
        assert_eq!(texel(readback, 8, x, y), expected, "({x}, {y})");
    }
}

#[test]
fn blit_scales_between_differently_sized_regions() {
    let gpu = gpu();
    // 2x2 source, each texel becomes a 4x4 block.
    let data: Vec<u8> = [
        [255, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
        [255, 255, 0, 255],
    ]
    .concat();
    let upload = buffer_with::<u8, Storage>(&data);
    let src = Image::<R8G8B8A8Unorm, Sampled>::new(2, 2).unwrap();
    let dst = Image::<R8G8B8A8Unorm, Sampled>::new(8, 8).unwrap();
    let readback = zeroed_buffer::<u8, Storage>(8 * 8 * 4);
    gpu.submit(|cmd| {
        cmd.copy_buffer_to_image(upload.range(..), src.whole());
        cmd.blit_image(src.whole(), dst.whole(), Filter::Nearest);
        cmd.copy_image_to_buffer(dst.whole(), readback.range(..));
    });
    let readback = readback.range(..).as_slice();
    for (y, x) in (0..8).flat_map(|y| (0..8).map(move |x| (y, x))) {
        assert_eq!(
            texel(readback, 8, x, y),
            texel(&data, 2, x / 4, y / 4),
            "({x}, {y})"
        );
    }
}

#[test]
fn host_copy_uploads_a_mip_level() {
    let gpu = gpu();
    let mut image = Image::<R8G8B8A8Unorm, Sampled>::with_mip_levels(8, 8, 2).unwrap();
    let data = pattern(8, 8);

    if !Ctx::features().rebar {
        // Host image copies need host-visible device memory.
        assert!(image.copy_from(&data, 0).is_err());
        return;
    }

    image.copy_from(&data, 0).unwrap();
    let readback = zeroed_buffer::<u8, Storage>(data.len());
    gpu.submit(|cmd| cmd.copy_image_to_buffer(image.whole(), readback.range(..)));
    assert_eq!(readback.range(..).as_slice(), data);

    image.copy_from(&pattern(4, 4), 1).unwrap();
    assert!(
        image.copy_from(&data, 2).is_err(),
        "the image has two mip levels"
    );
    assert!(image.copy_from(&[], 0).is_err());

    // The data has to be exactly one mip level of RGBA8 texels: too little would read past
    // the slice, too much means the caller mixed up levels or formats.
    assert!(image.copy_from(&data[..data.len() - 4], 0).is_err());
    assert!(image.copy_from(&data, 1).is_err());
    assert!(image.copy_from(&pattern(4, 4), 0).is_err());
    // Rejected copies leave the image as it was.
    gpu.submit(|cmd| cmd.copy_image_to_buffer(image.whole(), readback.range(..)));
    assert_eq!(readback.range(..).as_slice(), data);
}

#[test]
fn host_copy_writes_only_the_region() {
    let gpu = gpu();
    let mut image = Image::<R8G8B8A8Unorm, Sampled>::new(8, 8).unwrap();
    let (offset, extent) = (UVec2::new(4, 5), UVec2::new(3, 2));
    let patch = vec![200u8; 3 * 2 * 4];

    if !Ctx::features().rebar {
        assert!(image.copy_region_from(&patch, 0, offset, extent).is_err());
        return;
    }

    let data = pattern(8, 8);
    image.copy_from(&data, 0).unwrap();
    image.copy_region_from(&patch, 0, offset, extent).unwrap();

    // Regions have to lie inside the mip and match the data length.
    assert!(
        image
            .copy_region_from(&patch, 0, UVec2::new(6, 5), extent)
            .is_err()
    );
    assert!(
        image
            .copy_region_from(&patch[4..], 0, offset, extent)
            .is_err()
    );

    let readback = zeroed_buffer::<u8, Storage>(data.len());
    gpu.submit(|cmd| cmd.copy_image_to_buffer(image.whole(), readback.range(..)));
    let readback = readback.range(..).as_slice();
    for (y, x) in (0..8).flat_map(|y| (0..8).map(move |x| (y, x))) {
        let inside = (4..7).contains(&x) && (5..7).contains(&y);
        let expected = if inside {
            &[200u8; 4][..]
        } else {
            texel(&data, 8, x, y)
        };
        assert_eq!(texel(readback, 8, x, y), expected, "({x}, {y})");
    }
}

// ---- compute passes --------------------------------------------------------------------------

fn compute_image_bindings<'a>(
    target: &'a Target,
    offset: UVec2,
    size: UVec2,
) -> impl FnOnce(&mut CommandBuffer) + 'a {
    move |cmd| {
        cmd.compute(
            TestComputeImage::new(
                target.whole_view(),
                Vec4::new(1.0, 1.0, 1.0, 1.0),
                offset,
                size,
            ),
            [SIZE / 8, SIZE / 8, 1],
        )
    }
}

#[test]
fn compute_pass_writes_a_storage_image() {
    let gpu = gpu();
    let pixels = render(&gpu, |cmd, target| {
        compute_image_bindings(target, UVec2::ZERO, EXTENT)(cmd)
    });

    // Red and green are the UV gradient, blue is an 8x8 checkerboard.
    assert_pixel(&pixels, 0, 0, [2, 2, 0, 255]);
    assert_pixel(&pixels, 63, 0, [253, 2, 255, 255]);
    assert_pixel(&pixels, 0, 63, [2, 253, 255, 255]);
    assert_pixel(&pixels, 63, 63, [253, 253, 0, 255]);
    assert_golden("compute_image", &pixels, [SIZE, SIZE]);
}

#[test]
fn compute_pass_push_constants_reach_the_shader() {
    let gpu = gpu();
    let pixels = render(&gpu, |cmd, target| {
        cmd.compute(
            TestComputeImage::new(
                target.whole_view(),
                Vec4::new(0.0, 1.0, 0.0, 1.0),
                UVec2::new(16, 24),
                UVec2::new(32, 16),
            ),
            [SIZE / 8, SIZE / 8, 1],
        )
    });

    // Only the 32x16 rectangle at (16, 24) is written, and the tint removes red and blue.
    assert_pixel(&pixels, 15, 24, BLACK);
    assert_pixel(&pixels, 16, 23, BLACK);
    assert_pixel(&pixels, 48, 39, BLACK);
    assert_pixel(&pixels, 47, 40, BLACK);
    let inside = pixel(&pixels, 47, 39);
    assert_eq!((inside[0], inside[2]), (0, 0));
    assert!(inside[1] > 240);
    assert_golden("compute_image_rect", &pixels, [SIZE, SIZE]);
}

#[test]
fn indirect_compute_dispatch_matches_the_direct_one() {
    let gpu = gpu();
    let direct = render(&gpu, |cmd, target| {
        compute_image_bindings(target, UVec2::ZERO, EXTENT)(cmd)
    });

    let dispatch = buffer_with::<DispatchIndirectCommand, Indirect>(&[DispatchIndirectCommand {
        x: SIZE / 8,
        y: SIZE / 8,
        z: 1,
    }]);
    let indirect = render(&gpu, |cmd, target| {
        cmd.compute_indirect(
            TestComputeImage::new(target.whole_view(), Vec4::ONE, UVec2::ZERO, EXTENT),
            dispatch.range(..),
        )
    });
    assert_eq!(indirect, direct);
}

#[test]
fn compute_pass_reads_and_writes_buffers() {
    let gpu = gpu();
    let input: Vec<u32> = (1..=100).collect();
    let src = buffer_with::<u32, Storage>(&input);
    let dst = buffer_with::<u32, Storage>(&[u32::MAX; 100]);
    gpu.submit(|cmd| {
        cmd.compute(
            TestComputeBuffer::new(src.range(..), dst.range(..), 90, 3),
            [2, 1, 1],
        )
    });
    let expected: Vec<u32> = (0..100u32)
        .map(|i| {
            if i < 90 {
                input[i as usize] * 3 + i
            } else {
                u32::MAX
            }
        })
        .collect();
    assert_eq!(dst.range(..).as_slice(), expected);
}

/// A compute pass between transfer commands on the same buffers, in one command buffer: the
/// accesses declared by the generated bindings must produce the barriers in between.
#[test]
fn compute_pass_is_synchronised_with_surrounding_transfers() {
    let gpu = gpu();
    let src = zeroed_buffer::<u32, Storage>(64);
    let dst = zeroed_buffer::<u32, Storage>(64);
    let copy = zeroed_buffer::<u32, Storage>(64);
    gpu.submit(|cmd| {
        cmd.fill_buffer(src.range(..), 5);
        cmd.compute(
            TestComputeBuffer::new(src.range(..), dst.range(..), 64, 2),
            [1, 1, 1],
        );
        cmd.copy_buffer(dst.range(..), copy.range(..));
    });
    let expected: Vec<u32> = (0..64).map(|i| 10 + i).collect();
    assert_eq!(copy.range(..).as_slice(), expected);
}

// ---- raster passes ---------------------------------------------------------------------------

fn bindings(
    vertices: &Buffer<TestVertex>,
) -> lava::command_buffer::BindingOutput<lava::bindings::CTestRasterBindings, TestRaster, 0, 1> {
    TestRaster::new(vertices.range(..), Vec2::ZERO)
}

/// Draws the first `vertex_count` vertices of [`two_triangles`] with backface culling on/off.
fn draw_triangles(gpu: &common::Gpu, vertex_count: u32, culling: bool) -> Vec<u8> {
    let vertices = two_triangles();
    render(gpu, |cmd, target| {
        cmd.raster()
            .color_attachment(target.whole_view(), None)
            .backface_culling(culling)
            .draw(bindings(&vertices), EXTENT, vertex_count, 1)
    })
}

#[test]
fn raster_pass_draws_front_facing_triangles() {
    let gpu = gpu();
    let pixels = draw_triangles(&gpu, 6, true);

    // Corner colours are interpolated; near the red top corner red dominates.
    let top = pixel(&pixels, 16, 12);
    assert!(top[0] > 180 && top[1] < 60 && top[2] < 60, "{top:?}");
    assert_ne!(pixel(&pixels, IN_LEFT.0, IN_LEFT.1), BLACK);
    // The back-facing triangle is culled, and nothing is drawn outside the triangles.
    assert_pixel(&pixels, IN_RIGHT.0, IN_RIGHT.1, BLACK);
    assert_pixel(&pixels, 32, 32, BLACK);
    assert_golden("raster_culled", &pixels, [SIZE, SIZE]);
}

#[test]
fn disabling_backface_culling_draws_both_windings() {
    let gpu = gpu();
    let pixels = draw_triangles(&gpu, 6, false);
    assert_ne!(pixel(&pixels, IN_LEFT.0, IN_LEFT.1), BLACK);
    assert_pixel(&pixels, IN_RIGHT.0, IN_RIGHT.1, [255, 255, 0, 255]);
    assert_golden("raster_unculled", &pixels, [SIZE, SIZE]);
}

#[test]
fn wireframe_draws_only_the_edges() {
    let gpu = gpu();
    let vertices = two_triangles();
    let pixels = render(&gpu, |cmd, target| {
        cmd.raster()
            .color_attachment(target.whole_view(), None)
            .backface_culling(false)
            .wire_frame(true)
            .draw(bindings(&vertices), EXTENT, 6, 1)
    });
    // The interior stays empty, the bottom edge (y = 0.8 -> row 57) is drawn.
    assert_pixel(&pixels, IN_RIGHT.0, IN_RIGHT.1, BLACK);
    assert_pixel(&pixels, 48, 57, [255, 255, 0, 255]);
    assert_golden("raster_wireframe", &pixels, [SIZE, SIZE]);
}

#[test]
fn color_attachment_clear_replaces_the_previous_content() {
    let gpu = gpu();
    let vertices = two_triangles();
    let pixels = render(&gpu, |cmd, target| {
        cmd.raster()
            .color_attachment(target.whole_view(), Some([0.0, 0.0, 1.0, 1.0]))
            .draw(bindings(&vertices), EXTENT, 3, 1)
    });
    assert_pixel(&pixels, 32, 32, [0, 0, 255, 255]);
    assert_ne!(pixel(&pixels, IN_LEFT.0, IN_LEFT.1), [0, 0, 255, 255]);
}

#[test]
fn instances_are_drawn_with_their_instance_index() {
    let gpu = gpu();
    let vertices = two_triangles();
    let pixels = render(&gpu, |cmd, target| {
        cmd.raster()
            .color_attachment(target.whole_view(), None)
            // The second instance of the left triangle lands where the right one would be.
            .draw(
                TestRaster::new(vertices.range(..), Vec2::new(1.0, 0.0)),
                EXTENT,
                3,
                2,
            )
    });
    assert_ne!(pixel(&pixels, IN_LEFT.0, IN_LEFT.1), BLACK);
    assert_eq!(
        pixel(&pixels, IN_RIGHT.0, IN_RIGHT.1),
        pixel(&pixels, IN_LEFT.0, IN_LEFT.1)
    );
    assert_golden("raster_instanced", &pixels, [SIZE, SIZE]);
}

#[test]
fn indirect_draw_matches_the_direct_one() {
    let gpu = gpu();
    let direct = draw_triangles(&gpu, 3, true);

    let vertices = two_triangles();
    let commands = buffer_with::<DrawIndirectCommand, Indirect>(&[DrawIndirectCommand {
        vertex_count: 3,
        instance_count: 1,
        first_vertex: 0,
        first_instance: 0,
    }]);
    let indirect = render(&gpu, |cmd, target| {
        cmd.raster()
            .color_attachment(target.whole_view(), None)
            .draw_indirect(bindings(&vertices), EXTENT, commands.range(..))
    });
    assert_eq!(indirect, direct);
}

#[test]
fn indirect_count_draws_only_the_counted_commands() {
    let gpu = gpu();
    let vertices = two_triangles();
    // The shaders see D3D-style vertex ids (without `first_vertex`), so the commands differ
    // in how many vertices they draw: the left triangle, then both.
    let command = |vertex_count| DrawIndirectCommand {
        vertex_count,
        instance_count: 1,
        first_vertex: 0,
        first_instance: 0,
    };
    let commands = buffer_with::<DrawIndirectCommand, Indirect>(&[command(3), command(6)]);

    let draw_counted = |count: u32| {
        let count = buffer_with::<u32, Indirect>(&[count]);
        render(&gpu, |cmd, target| {
            cmd.raster()
                .color_attachment(target.whole_view(), None)
                .backface_culling(false)
                .draw_indirect_count(
                    bindings(&vertices),
                    EXTENT,
                    commands.range(..),
                    count.range(..),
                )
        })
    };
    assert_eq!(draw_counted(2), draw_triangles(&gpu, 6, false));
    assert_eq!(draw_counted(1), draw_triangles(&gpu, 3, false));
    assert!(draw_counted(0).chunks_exact(4).all(|p| p == BLACK));
}

#[test]
fn scissor_clips_and_viewport_scales_the_draw() {
    let gpu = gpu();
    let vertices = two_triangles();
    let full = draw_triangles(&gpu, 6, false);

    // Scissor: the full-size picture, cut to the left half.
    let scissored = render(&gpu, |cmd, target| {
        cmd.raster()
            .color_attachment(target.whole_view(), None)
            .backface_culling(false)
            .draw_with_dynstates(
                bindings(&vertices),
                EXTENT,
                6,
                1,
                &[Scissor {
                    offset: IVec2::ZERO,
                    extent: UVec2::new(SIZE / 2, SIZE),
                }],
                Viewport {
                    offset: IVec2::ZERO,
                    extent: EXTENT,
                },
            )
    });
    for (y, x) in (0..SIZE).flat_map(|y| (0..SIZE).map(move |x| (y, x))) {
        let expected = if x < SIZE / 2 {
            pixel(&full, x, y)
        } else {
            BLACK
        };
        assert_eq!(pixel(&scissored, x, y), expected, "({x}, {y})");
    }

    // Viewport: the whole picture squeezed into the bottom-right quarter.
    let quarter = render(&gpu, |cmd, target| {
        cmd.raster()
            .color_attachment(target.whole_view(), None)
            .backface_culling(false)
            .draw_with_dynstates(
                bindings(&vertices),
                EXTENT,
                6,
                1,
                &[Scissor {
                    offset: IVec2::ZERO,
                    extent: EXTENT,
                }],
                Viewport {
                    offset: IVec2::splat(SIZE as i32 / 2),
                    extent: EXTENT / 2,
                },
            )
    });
    assert_pixel(&quarter, IN_RIGHT.0, IN_RIGHT.1, BLACK);
    assert_pixel(
        &quarter,
        32 + IN_RIGHT.0 / 2,
        32 + IN_RIGHT.1 / 2,
        [255, 255, 0, 255],
    );
    for (y, x) in (0..SIZE).flat_map(|y| (0..SIZE).map(move |x| (y, x))) {
        if x < SIZE / 2 || y < SIZE / 2 {
            assert_eq!(pixel(&quarter, x, y), BLACK, "({x}, {y})");
        }
    }
    assert_golden("raster_viewport", &quarter, [SIZE, SIZE]);
}

/// A near green triangle drawn before a far red one covering the same area.
fn overlapping_triangles() -> Buffer<TestVertex> {
    buffer_with(&[
        vertex(0.0, -0.8, 0.8, GREEN),
        vertex(0.8, 0.8, 0.8, GREEN),
        vertex(-0.8, 0.8, 0.8, GREEN),
        vertex(0.0, -0.8, 0.2, RED),
        vertex(0.8, 0.8, 0.2, RED),
        vertex(-0.8, 0.8, 0.2, RED),
    ])
}

#[test]
fn depth_attachment_keeps_the_nearer_fragment() {
    let gpu = gpu();
    let vertices = overlapping_triangles();

    let without_depth = render(&gpu, |cmd, target| {
        cmd.raster()
            .color_attachment(target.whole_view(), None)
            .draw(bindings(&vertices), EXTENT, 6, 1)
    });
    // Without a depth test the later (far, red) triangle wins.
    assert_pixel(&without_depth, 32, 40, [255, 0, 0, 255]);

    let depth = Image::<D32Sfloat, DepthAttachment>::new(SIZE, SIZE).unwrap();
    let with_depth = render(&gpu, |cmd, target| {
        cmd.raster()
            .color_attachment(target.whole_view(), None)
            .depth_attachment(depth.whole_view(), Some([0.0]), true)
            .draw(bindings(&vertices), EXTENT, 6, 1)
    });
    assert_pixel(&with_depth, 32, 40, [0, 255, 0, 255]);
    assert_golden("raster_depth", &with_depth, [SIZE, SIZE]);
}

#[test]
fn depth_persists_between_draws_unless_cleared() {
    let gpu = gpu();
    let vertices = overlapping_triangles();
    let depth = Image::<D32Sfloat, DepthAttachment>::new(SIZE, SIZE).unwrap();

    let draw_far_after_near = |clear_between: Option<[f32; 1]>| {
        render(&gpu, |cmd, target| {
            cmd.raster()
                .color_attachment(target.whole_view(), None)
                .depth_attachment(depth.whole_view(), Some([0.0]), true)
                .draw(
                    TestRaster::new(vertices.range(0..3), Vec2::ZERO),
                    EXTENT,
                    3,
                    1,
                );
            cmd.raster()
                .color_attachment(target.whole_view(), None)
                .depth_attachment(depth.whole_view(), clear_between, true)
                .draw(
                    TestRaster::new(vertices.range(3..), Vec2::ZERO),
                    EXTENT,
                    3,
                    1,
                );
        })
    };

    // The near triangle's depth is still there for the second draw.
    assert_pixel(&draw_far_after_near(None), 32, 40, [0, 255, 0, 255]);
    // Clearing depth in between lets the far triangle through.
    assert_pixel(&draw_far_after_near(Some([0.0])), 32, 40, [255, 0, 0, 255]);
}

/// Regression: `write: false` only skipped storing the attachment while the pipeline kept
/// writing depth, which left the depth buffer in a driver-dependent state.
#[test]
fn read_only_depth_attachment_tests_but_does_not_write() {
    let gpu = gpu();
    let near = [
        vertex(0.0, -0.8, 0.8, GREEN),
        vertex(0.8, 0.8, 0.8, GREEN),
        vertex(-0.8, 0.8, 0.8, GREEN),
    ];
    let at_depth = |z: f32, color| near.map(|v| vertex(v.position.x, v.position.y, z, color));
    let far = buffer_with::<TestVertex, Storage>(&at_depth(0.2, RED));
    let middle = buffer_with::<TestVertex, Storage>(&at_depth(0.5, BLUE));
    let near = buffer_with::<TestVertex, Storage>(&near);
    let depth = Image::<D32Sfloat, DepthAttachment>::new(SIZE, SIZE).unwrap();

    // Far (written), then near with the given write flag, then a triangle in between.
    let draw = |write_near: bool| {
        render(&gpu, |cmd, target| {
            for (vertices, clear, write) in [
                (&far, Some([0.0]), true),
                (&near, None, write_near),
                (&middle, None, true),
            ] {
                cmd.raster()
                    .color_attachment(target.whole_view(), None)
                    .depth_attachment(depth.whole_view(), clear, write)
                    .draw(bindings(vertices), EXTENT, 3, 1);
            }
        })
    };

    // Written: the near triangle occludes the middle one.
    assert_pixel(&draw(true), 32, 40, [0, 255, 0, 255]);
    // Read-only: the near triangle passed the test against the far one, but left no depth
    // behind, so the middle triangle is only compared with the far one and wins.
    assert_pixel(&draw(false), 32, 40, [0, 0, 255, 255]);

    // A read-only draw is still occluded by what is already in the depth buffer.
    let occluded = render(&gpu, |cmd, target| {
        cmd.raster()
            .color_attachment(target.whole_view(), None)
            .depth_attachment(depth.whole_view(), Some([0.0]), true)
            .draw(bindings(&near), EXTENT, 3, 1);
        cmd.raster()
            .color_attachment(target.whole_view(), None)
            .depth_attachment(depth.whole_view(), None, false)
            .draw(bindings(&far), EXTENT, 3, 1);
    });
    assert_pixel(&occluded, 32, 40, [0, 255, 0, 255]);
}

#[test]
fn depth_clear_takes_effect_on_a_read_only_attachment() {
    let gpu = gpu();
    let vertices = overlapping_triangles();
    let depth = Image::<D32Sfloat, DepthAttachment>::new(SIZE, SIZE).unwrap();
    let pixels = render(&gpu, |cmd, target| {
        // Near triangle written, then depth cleared by a read-only pass, then the far one.
        cmd.raster()
            .color_attachment(target.whole_view(), None)
            .depth_attachment(depth.whole_view(), Some([0.0]), true)
            .draw(
                TestRaster::new(vertices.range(0..3), Vec2::ZERO),
                EXTENT,
                3,
                1,
            );
        cmd.raster()
            .color_attachment(target.whole_view(), None)
            .depth_attachment(depth.whole_view(), Some([0.0]), false)
            .draw(
                TestRaster::new(vertices.range(0..3), Vec2::ZERO),
                EXTENT,
                3,
                1,
            );
        cmd.raster()
            .color_attachment(target.whole_view(), None)
            .depth_attachment(depth.whole_view(), None, true)
            .draw(
                TestRaster::new(vertices.range(3..), Vec2::ZERO),
                EXTENT,
                3,
                1,
            );
    });
    assert_pixel(&pixels, 32, 40, [255, 0, 0, 255]);
}

/// Regression: pipelines for depth-stencil formats were created without a stencil format.
#[test]
fn depth_stencil_formats_work_as_depth_attachment() {
    let gpu = gpu();
    let vertices = overlapping_triangles();
    let depth = Image::<D32SfloatS8Uint, DepthAttachment>::new(SIZE, SIZE).unwrap();
    let pixels = render(&gpu, |cmd, target| {
        cmd.raster()
            .color_attachment(target.whole_view(), None)
            .depth_attachment(depth.whole_view(), Some((0.0, 0)), true)
            .draw(bindings(&vertices), EXTENT, 6, 1)
    });
    assert_pixel(&pixels, 32, 40, [0, 255, 0, 255]);
}

#[test]
fn indexed_draw_samples_a_bindless_texture() {
    let gpu = gpu();
    // 2x2 texture: red, green / blue, yellow.
    let texels: Vec<u8> = [
        [255, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
        [255, 255, 0, 255],
    ]
    .concat();
    let upload = buffer_with::<u8, Storage>(&texels);
    let texture = Image::<R8G8B8A8Unorm, Sampled>::new(2, 2).unwrap();

    let corner = |x: f32, y: f32, u: f32, v: f32| TestTexturedVertex {
        position: Vec2::new(x, y),
        uv: Vec2::new(u, v),
    };
    // A quad over the central half of the target, as four vertices and two triangles.
    let vertices = buffer_with::<TestTexturedVertex, Storage>(&[
        corner(-0.5, -0.5, 0.0, 0.0),
        corner(0.5, -0.5, 1.0, 0.0),
        corner(0.5, 0.5, 1.0, 1.0),
        corner(-0.5, 0.5, 0.0, 1.0),
    ]);
    let indices = buffer_with::<u32, Index>(&[0, 1, 2, 0, 2, 3]);

    let pixels = render(&gpu, |cmd, target| {
        cmd.copy_buffer_to_image(upload.range(..), texture.whole());
        cmd.raster()
            .color_attachment(target.whole_view(), None)
            .draw_indexed(
                // Sampler 0 is nearest-neighbour.
                TestRasterTextured::new(vertices.range(..), texture.whole_view(), 0),
                EXTENT,
                indices.range(..),
                1,
            )
    });

    assert_pixel(&pixels, 24, 24, [255, 0, 0, 255]);
    assert_pixel(&pixels, 40, 24, [0, 255, 0, 255]);
    assert_pixel(&pixels, 24, 40, [0, 0, 255, 255]);
    assert_pixel(&pixels, 40, 40, [255, 255, 0, 255]);
    assert_pixel(&pixels, 8, 8, BLACK);
    assert_golden("raster_textured", &pixels, [SIZE, SIZE]);

    // Only the first three indices: one triangle, the lower-left half stays empty.
    let half = render(&gpu, |cmd, target| {
        cmd.raster()
            .color_attachment(target.whole_view(), None)
            .draw_indexed(
                TestRasterTextured::new(vertices.range(..), texture.whole_view(), 0),
                EXTENT,
                indices.range(..3),
                1,
            )
    });
    assert_pixel(&half, 40, 24, [0, 255, 0, 255]);
    assert_pixel(&half, 24, 40, BLACK);
}

/// A BC7 block (mode 6) of one colour: each channel is `(channel << 1) | low_bit`.
fn bc7_block(channels: [u8; 4], low_bit: u8) -> [u8; 16] {
    let mut bits = 1u128 << 6;
    for (index, channel) in channels.into_iter().enumerate() {
        assert!(channel < 128);
        // Both endpoints of the channel.
        bits |= (channel as u128) << (7 + index * 14);
        bits |= (channel as u128) << (14 + index * 14);
    }
    bits |= (low_bit as u128) << 63 | (low_bit as u128) << 64;
    bits.to_le_bytes()
}

#[test]
fn host_copy_uploads_block_compressed_mips() {
    let gpu = gpu();
    if !Ctx::features().rebar {
        return;
    }
    // 8x8 texels are 2x2 blocks: red, green / blue, yellow.
    let blocks: Vec<u8> = [
        bc7_block([127, 0, 0, 127], 1),
        bc7_block([0, 127, 0, 127], 1),
        bc7_block([0, 0, 127, 127], 1),
        bc7_block([127, 127, 0, 127], 1),
    ]
    .concat();
    let mut texture = Image::<BC7UnormBlock, Sampled>::with_mip_levels(8, 8, 4).unwrap();
    texture.copy_from(&blocks, 0).unwrap();
    // The smaller levels are one block each, even the ones smaller than a block.
    for level in 1..4 {
        texture.copy_from(&blocks[..16], level).unwrap();
        assert!(texture.copy_from(&blocks[..8], level).is_err());
        assert!(texture.copy_from(&blocks[..32], level).is_err());
    }
    assert!(texture.copy_from(&blocks[..48], 0).is_err());
    // A region of one block: the yellow block becomes red.
    texture
        .copy_region_from(&blocks[..16], 0, UVec2::splat(4), UVec2::splat(4))
        .unwrap();

    let corner = |x: f32, y: f32, u: f32, v: f32| TestTexturedVertex {
        position: Vec2::new(x, y),
        uv: Vec2::new(u, v),
    };
    let vertices = buffer_with::<TestTexturedVertex, Storage>(&[
        corner(-0.5, -0.5, 0.0, 0.0),
        corner(0.5, -0.5, 1.0, 0.0),
        corner(0.5, 0.5, 1.0, 1.0),
        corner(-0.5, 0.5, 0.0, 1.0),
    ]);
    let indices = buffer_with::<u32, Index>(&[0, 1, 2, 0, 2, 3]);
    let pixels = render(&gpu, |cmd, target| {
        cmd.raster()
            .color_attachment(target.whole_view(), None)
            .draw_indexed(
                TestRasterTextured::new(vertices.range(..), texture.whole_view(), 0),
                EXTENT,
                indices.range(..),
                1,
            )
    });
    assert_pixel(&pixels, 24, 24, [255, 1, 1, 255]);
    assert_pixel(&pixels, 40, 24, [1, 255, 1, 255]);
    assert_pixel(&pixels, 24, 40, [1, 1, 255, 255]);
    assert_pixel(&pixels, 40, 40, [255, 1, 1, 255]);
}

#[test]
fn mesh_pass_emits_triangles_per_workgroup() {
    let gpu = gpu();
    if !Ctx::features().mesh {
        eprintln!("SKIPPED mesh_pass_emits_triangles_per_workgroup: no mesh shader support");
        return;
    }
    let pixels = render(&gpu, |cmd, target| {
        cmd.raster()
            .color_attachment(target.whole_view(), None)
            .backface_culling(false)
            .launch(
                TestMesh::new(Vec4::new(0.8, 0.8, 0.8, 1.0), Vec2::new(1.0, 0.0), 0.4),
                2,
                1,
                1,
                EXTENT,
            )
    });
    // Workgroup 0 draws around the centre, workgroup 1 shifted to the right edge.
    assert_ne!(pixel(&pixels, 32, 36), BLACK);
    assert_ne!(pixel(&pixels, 60, 36), BLACK);
    assert_pixel(&pixels, 4, 4, BLACK);
    assert_golden("mesh", &pixels, [SIZE, SIZE]);
}

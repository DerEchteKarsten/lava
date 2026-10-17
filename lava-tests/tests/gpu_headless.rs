//! Headless GPU tests: buffers, images, sync, command recording, queries, and golden images per pass type
//!
//! These run on the real GPU with the validation layer on; every test fails if the layer
//! reports an error. The passes come from `lava-tests/shaders`, not from the engine.
// Needed to call `RasterBuilder`'s attachment and draw methods, which return it one larger.
mod common;

use common::{Cmd, buffer_with, golden::assert_golden, gpu, zeroed_buffer};
use glam::{IVec2, UVec2, UVec3, Vec2, Vec4};
use lava::{
    bindless::{BindlessHandle, NULL_HANDLE, max_sampled_images, max_storage_images},
    buffer::{
        Buffer,
        usage::{Index, Indirect, Storage},
    },
    command_buffer::{
        Accesses, Attachment, BindingOutput, Blend, BlendEquation, BlendFactor, BlendOp,
        BufferAccess, ColorComponentFlags, CompareOp, CullModeFlags,
        DispatchIndirectCommand, Draw, DrawIndirectCommand, FrontFace, ImageUse, LogicOp,
        PolygonMode, PrimitiveTopology, RasterState, Scissor, Stencil, StencilFace, StencilOp,
        Viewport, kind,
    },
    image::{
        D2, D3, DepthSplit, Image, MipChain,
        format::{BC7UnormBlock, D32Sfloat, D32SfloatS8Uint, R8G8B8A8Unorm},
        slice::Filter,
        usage::{
            ColorAttachmentSampledTransferSrcTransferDst,
            ColorAttachmentStorageTransferSrcTransferDst, DepthAttachment, Sampled,
            SampledHostCopy, SampledHostCopyTransferSrc, SampledHostCopyTransferSrcTransferDst,
            DepthAttachmentSampled, SampledStorage, SampledTransferDst,
            SampledTransferSrcTransferDst,
            Storage as StorageImage, StorageTransferSrcTransferDst,
        },
    },
    query::{
        Performance, PipelineStatistics, PipelineStats, QueryPipelineStatisticFlags as Statistics,
        QueryPool, QueryRecorder, ScopeInfo, ScopeObserver, Timestamp, counters,
    },
    queue::{Compute, Graphics, Kind, MAX_TIMELINES, Stage, Timeline, Transfer},
    state::Ctx,
};
use lava_tests::bindings::{
    PASS_MAP, TestClock, TestClocks, TestComputeBuffer, TestComputeImage, TestComputeVolume,
    TestConstants, TestMesh, TestRaster, TestRasterTextured, TestSampleDepth, TestTexturedVertex,
    TestVertex,
};

// ---- helpers ---------------------------------------------------------------------------------

/// Side length of the square render targets.
const SIZE: u32 = 64;
const EXTENT: UVec2 = UVec2::splat(SIZE);
const BLACK: [u8; 4] = [0, 0, 0, 255];

type Target = Image<R8G8B8A8Unorm, ColorAttachmentStorageTransferSrcTransferDst>;

/// Clears a fresh target to opaque black, lets `record` draw into it, and reads it back.
fn render(gpu: &common::Gpu, record: impl FnOnce(&mut Cmd, &Target)) -> Vec<u8> {
    let target = Target::new(UVec2::new(SIZE, SIZE)).unwrap().bind_storage().unwrap();
    let readback = zeroed_buffer::<u8, Storage>((SIZE * SIZE * 4) as usize);
    gpu.submit(|cmd| {
        cmd.clear_image(target.whole_view(), [0.0, 0.0, 0.0, 1.0]);
        record(cmd, &target);
        cmd.copy_image_to_buffer(target.whole(), readback.range(..));
    });
    readback.host().to_vec()
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
    assert!(lava::init(&lava::Config::new(&PASS_MAP)).is_err());
}

#[test]
fn context_exposes_the_graphics_queue_family() {
    let _gpu = gpu();
    assert!((1..=MAX_TIMELINES as u32).contains(&Ctx::queue_count(Kind::Graphics)));
    let graphics = Ctx::queue_family(Kind::Graphics).expect("every device has one");
    // The dedicated families are other ones, with queues where they exist.
    for kind in [Kind::Compute, Kind::Transfer] {
        match Ctx::queue_family(kind) {
            Some(family) => {
                assert_ne!(family, graphics, "{kind:?}");
                assert!((1..=MAX_TIMELINES as u32).contains(&Ctx::queue_count(kind)));
            }
            None => assert_eq!(Ctx::queue_count(kind), 0),
        }
    }
    assert_eq!(
        Ctx::queue_family(Kind::Compute).is_some(),
        Ctx::features().async_compute
    );
    assert_eq!(
        Ctx::queue_family(Kind::Transfer).is_some(),
        Ctx::features().async_transfer
    );
}

// ---- buffers ---------------------------------------------------------------------------------

#[test]
fn buffer_has_the_requested_size() {
    let _gpu = gpu();
    let buffer = Buffer::<[u32; 3], Storage>::new(5, true).unwrap();
    assert_eq!(buffer.len(), 5);
    assert_eq!(buffer.size(), 60);
    assert_eq!(buffer.range(..).len(), 5);
    assert_ne!(buffer.address(), 0, "buffers are addressable from shaders");
    assert_eq!(buffer.range(2..).address(), buffer.address() + 24);
}

#[test]
fn host_writes_are_visible_through_slices_and_indexing() {
    let _gpu = gpu();
    let mut buffer = Buffer::<u32, Storage>::new(8, true).unwrap();
    buffer.host_mut().copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
    buffer[0] = 10;
    buffer.host_mut()[4..6].copy_from_slice(&[50, 60]);

    assert_eq!(buffer.host(), [10, 2, 3, 4, 50, 60, 7, 8]);
    assert_eq!(buffer[5], 60);
    assert_eq!(buffer.byte_range(4..8).cast::<u32>().len(), 1);

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

// ---- timelines and lifetimes -----------------------------------------------------------------

#[test]
fn more_threads_than_queues_submit_in_parallel() {
    let _gpu = gpu();
    // One timeline per thread, more than there are queues where the timelines allow it.
    let threads = (Ctx::queue_count(Kind::Graphics) as usize + 1).min(MAX_TIMELINES - 1);
    let barrier = std::sync::Barrier::new(threads);
    std::thread::scope(|scope| {
        for value in 0..threads as u32 {
            let barrier = &barrier;
            scope.spawn(move || {
                let buffer = zeroed_buffer::<u32, Storage>(4);
                let timeline = Timeline::<Graphics>::builder().build().unwrap();
                barrier.wait();
                let submission = timeline.reserve().begin().unwrap();
                let done = submission
                    .submit(|cmd| cmd.fill_buffer(buffer.range(..), value))
                    .unwrap();
                done.block().unwrap();
                assert_eq!(buffer.host(), [value; 4]);
            });
        }
    });
}

/// A timeline's frame slots take turns, and what a submission used lives until the GPU
/// finished it, also when it is dropped while the submission records or right after it was
/// submitted. Destroying it earlier is a validation error, which fails the test.
#[test]
fn frame_slots_cycle_and_what_a_submission_used_lives_until_it_is_done() {
    let gpu = gpu();
    let buffer = zeroed_buffer::<u32, Storage>(4);
    let mut slots = Vec::new();
    let mut last = None;
    for value in 1..=4 {
        let scratch = zeroed_buffer::<u32, Storage>(1024);
        let image =
            Image::<R8G8B8A8Unorm, StorageTransferSrcTransferDst>::new(UVec2::new(64, 64)).unwrap();
        let recording = zeroed_buffer::<u32, Storage>(4);
        let buffer = &buffer;
        let submission = gpu.timeline.reserve().begin().unwrap();
        slots.push(submission.slot());
        let done = submission
            .submit(move |cmd| {
                cmd.fill_buffer(recording.range(..), value);
                drop(recording);
                cmd.clear_image(image.whole_view(), [1.0, 0.0, 0.0, 1.0]);
                cmd.fill_buffer(scratch.range(..), value);
                cmd.copy_buffer(scratch.range(..4), buffer.range(..));
                drop((scratch, image));
            })
            .unwrap();
        last = Some(done);
    }
    last.unwrap().block().unwrap();
    assert_eq!(slots[0], slots[2]);
    assert_eq!(slots[1], slots[3]);
    assert_ne!(slots[0], slots[1]);
    assert_eq!(buffer.host(), [4; 4]);
}

/// A recording that panics already moved the image's tracked layout to `GENERAL`: its barriers
/// are submitted instead, so the next submission finds the image in the layout lava tracks (a
/// mismatch is a validation error), and the timeline's pending accesses aren't poisoned.
#[test]
fn a_panicking_recording_still_makes_its_layout_transitions() {
    let gpu = gpu();
    let image =
        Image::<R8G8B8A8Unorm, StorageTransferSrcTransferDst>::new(UVec2::new(8, 8)).unwrap();
    let readback = zeroed_buffer::<u8, Storage>(8 * 8 * 4);
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        gpu.submit(|cmd| {
            cmd.clear_image(image.whole_view(), [1.0, 0.0, 0.5, 1.0]);
            panic!("recording failed on purpose");
        })
    }));
    assert!(panicked.is_err());
    gpu.submit(|cmd| cmd.copy_image_to_buffer(image.whole(), readback.range(..)));
    // Destroying it locks every timeline's pending accesses.
    drop(image);
    gpu.submit(|cmd| cmd.fill_buffer(readback.range(..), 0));
}

/// A stamp completes once its submissions are done, without any later submission.
#[test]
fn stamps_complete_without_further_submissions() {
    let gpu = gpu();
    let buffer = zeroed_buffer::<u32, Storage>(4);
    let submission = gpu.timeline.reserve().begin().unwrap();
    let stamp = lava::latest_stamp();
    let done = submission
        .submit(|cmd| cmd.fill_buffer(buffer.range(..), 7))
        .unwrap();
    done.block().unwrap();
    assert!(done.is_done());
    assert!(stamp.is_done());
}

/// Dropping a reservation while recording would run destructors there, which lock the
/// pending accesses the recording holds: they wait for the recording instead of deadlocking.
#[test]
fn a_reservation_dropped_while_recording_runs_no_destructors_there() {
    let gpu = gpu();
    let other = Timeline::<Graphics>::builder().build().unwrap();
    let earlier = other.reserve();
    let image =
        Image::<R8G8B8A8Unorm, StorageTransferSrcTransferDst>::new(UVec2::new(8, 8)).unwrap();
    // Destroyed once `earlier` is signalled.
    drop(image);
    let buffer = zeroed_buffer::<u32, Storage>(4);
    gpu.submit(|cmd| {
        drop(earlier);
        cmd.fill_buffer(buffer.range(..), 1);
    });
    assert_eq!(buffer.host(), [1; 4]);
}

/// One timeline's submission waits on the GPU for another's.
#[test]
fn a_submission_waits_for_another_timelines() {
    let gpu = gpu();
    let other = Timeline::<Graphics>::builder().build().unwrap();
    let (src, dst) = (zeroed_buffer::<u32, Storage>(4), zeroed_buffer::<u32, Storage>(4));
    let own = gpu.submit(|_| {});
    // Neither submission is waited for on the host before the second is made.
    let filled = (other.reserve().begin().unwrap())
        .submit(|cmd| cmd.fill_buffer(src.range(..), 9))
        .unwrap();
    let mut copy = gpu.timeline.reserve().begin().unwrap();
    copy.wait_for(filled.at(Stage::Transfer));
    // A wait on its own timeline is dropped: its submissions run in order anyway.
    copy.wait_for(own);
    let copied = copy
        .submit(|cmd| cmd.copy_buffer(src.range(..), dst.range(..)))
        .unwrap();
    copied.block().unwrap();
    assert!(filled.is_done());
    assert_eq!(dst.host(), [9; 4]);
}

/// Values are submitted in the order they were reserved: a later one first panics, and is
/// then signalled with the one before it.
#[test]
fn submitting_out_of_order_panics() {
    let _gpu = gpu();
    let timeline = Timeline::<Graphics>::builder().build().unwrap();
    let (first, second) = (timeline.reserve(), timeline.reserve());
    let second = second.begin().unwrap();
    let value = second.value();
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        second.submit(|_| {})
    }));
    assert!(panicked.is_err());
    let done = first.begin().unwrap().submit(|_| {}).unwrap();
    done.block().unwrap();
    // The panicking value was dropped, so the first one's submission signalled it too.
    let next = timeline.reserve();
    assert_eq!(next.value(), value + 1);
    next.begin().unwrap().submit(|_| {}).unwrap().block().unwrap();
}

/// A dropped reservation is signalled in order, so the values after it begin and complete.
#[test]
fn dropped_reservations_are_signalled() {
    let _gpu = gpu();
    let timeline = Timeline::<Graphics>::builder().in_flight(1).build().unwrap();
    let (dropped, kept) = (timeline.reserve(), timeline.reserve());
    // With one frame in flight, `kept` begins once `dropped` is done, which isn't submitted.
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        drop(timeline.reserve().begin())
    }));
    assert!(refused.is_err(), "begin waits for a value nobody submitted");
    drop(dropped);
    let done = kept.begin().unwrap().submit(|_| {}).unwrap();
    done.block().unwrap();
    // The reservation the refused `begin` dropped is signalled with the next submission.
    let after = timeline.reserve().begin().unwrap().submit(|_| {}).unwrap();
    after.block().unwrap();
}

/// What is dropped while another timeline holds a reservation lives until that one is done
/// too: its slot isn't free before.
#[test]
fn deletions_wait_for_every_timeline() {
    let gpu = gpu();
    let other = Timeline::<Graphics>::builder().build().unwrap();
    lava::wait_idle().unwrap();
    let bind = || {
        Image::<R8G8B8A8Unorm, Sampled>::new(UVec2::new(4, 4))
            .unwrap()
            .bind_sampled()
            .unwrap()
    };
    let held = other.reserve();
    let image = bind();
    let slot = image.handle();
    drop(image);
    gpu.submit(|_| {});
    let unrelated = bind();
    assert_ne!(unrelated.handle(), slot, "the other timeline may still read it");
    let done = held.begin().unwrap().submit(|_| {}).unwrap();
    done.block().unwrap();
    gpu.submit(|_| {});
    assert_eq!(bind().handle(), slot);
}

/// `MAX_TIMELINES` exist at most; a dropped one's semaphore goes to the next, whose values go
/// on, so a `Wait` of the old one stays done.
#[test]
fn timelines_are_limited_and_reused() {
    let _gpu = gpu();
    let mut timelines = Vec::new();
    let error = loop {
        match Timeline::<Graphics>::builder().in_flight(1).build() {
            Ok(timeline) => timelines.push(timeline),
            Err(error) => break error,
        }
    };
    assert!(matches!(error, lava::Error::OutOfTimelines), "{error}");
    assert!(timelines.len() < MAX_TIMELINES, "the harness holds one");
    let last = timelines.pop().unwrap();
    let done = last.reserve().begin().unwrap().submit(|_| {}).unwrap();
    drop(last);
    let reused = Timeline::<Graphics>::builder().build().unwrap();
    assert!(done.is_done());
    let next = reused.reserve();
    next.begin().unwrap().submit(|_| {}).unwrap().block().unwrap();
}

// ---- queue family ownership ------------------------------------------------------------------

/// Whether the device has `feature`'s dedicated family; prints why a test is skipped.
fn has_family(test: &str, kind: Kind) -> bool {
    let has = Ctx::queue_family(kind).is_some();
    if !has {
        eprintln!("SKIPPED {test}: the device has no dedicated {kind:?} family");
    }
    has
}

type Owned = Image<R8G8B8A8Unorm, StorageTransferSrcTransferDst>;

/// The pixels `TestComputeImage` with a red tint writes into an 8x8 image: red is the
/// horizontal UV.
fn expect_red_gradient(pixels: &[u8]) {
    for (i, texel) in pixels.chunks_exact(4).enumerate() {
        let x = (i % 8) as f32;
        let red = ((x + 0.5) / 8.0 * 255.0).round() as u8;
        assert!(close(texel.try_into().unwrap(), [red, 0, 0, 255]), "texel {i}: {texel:?}");
    }
}

/// An image written on the compute family and released to graphics is read there.
#[test]
fn a_compute_image_is_released_to_graphics() {
    let gpu = gpu();
    if !has_family("a_compute_image_is_released_to_graphics", Kind::Compute) {
        return;
    }
    let compute = Timeline::<Compute>::builder().build().unwrap();
    let image = Owned::new(UVec2::splat(8)).unwrap().bind_storage().unwrap();
    let red = Vec4::new(1.0, 0.0, 0.0, 1.0);
    let written = (compute.reserve().begin().unwrap())
        .submit(|cmd| {
            let bindings = TestComputeImage::push_bindings(image.whole_view(), red, UVec2::ZERO, UVec2::splat(8));
            cmd.compute(&bindings, [1, 1, 1], None);
            cmd.release::<Graphics>(image.whole_view());
        })
        .unwrap();
    let readback = zeroed_buffer::<u8, Storage>(8 * 8 * 4);
    gpu.submit_after(&[written], |cmd| {
        cmd.copy_image_to_buffer(image.whole(), readback.range(..))
    });
    expect_red_gradient(readback.host());
}

/// The copy engine uploads an image for graphics, which hands it back for a readback.
#[test]
fn images_go_to_the_transfer_family_and_back() {
    let gpu = gpu();
    if !has_family("images_go_to_the_transfer_family_and_back", Kind::Transfer) {
        return;
    }
    let transfer = Timeline::<Transfer>::builder().build().unwrap();
    let image = Owned::new(UVec2::splat(8)).unwrap();
    let upload = buffer_with::<u8, Storage>(&pattern(8, 8));
    let uploaded = (transfer.reserve().begin().unwrap())
        .submit(|cmd| {
            cmd.copy_buffer_to_image(upload.range(..), image.whole());
            cmd.release::<Graphics>(image.whole_view());
        })
        .unwrap();
    let readback = zeroed_buffer::<u8, Storage>(8 * 8 * 4);
    let drawn = gpu.submit_after(&[uploaded], |cmd| {
        cmd.copy_image_to_buffer(image.whole(), readback.range(..));
        cmd.clear_image(image.whole_view(), [0.0, 1.0, 0.0, 1.0]);
        cmd.release::<Transfer>(image.whole_view());
    });
    assert_eq!(readback.host(), pattern(8, 8));
    let mut back = transfer.reserve().begin().unwrap();
    back.wait_for(drawn);
    let back = back.submit(|cmd| cmd.copy_image_to_buffer(image.whole(), readback.range(..)));
    back.unwrap().block().unwrap();
    assert!(readback.host().chunks_exact(4).all(|t| t == [0, 255, 0, 255]));
}

/// An image another family owns can't be accessed before it released it.
#[test]
fn an_unreleased_image_panics() {
    let gpu = gpu();
    if !has_family("an_unreleased_image_panics", Kind::Compute) {
        return;
    }
    let compute = Timeline::<Compute>::builder().build().unwrap();
    let image = Owned::new(UVec2::splat(8)).unwrap();
    let cleared = (compute.reserve().begin().unwrap())
        .submit(|cmd| cmd.clear_image(image.whole_view(), [1.0; 4]))
        .unwrap();
    cleared.block().unwrap();
    let readback = zeroed_buffer::<u8, Storage>(8 * 8 * 4);
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        gpu.submit(|cmd| cmd.copy_image_to_buffer(image.whole(), readback.range(..)))
    }));
    let message = *panicked.unwrap_err().downcast::<String>().unwrap();
    assert!(message.contains("`cmd.release::<Graphics>(view)`"), "{message}");
    // A query of graphics statistics can't be recorded on compute either.
    if Ctx::features().pipeline_statistics {
        let pool = QueryPool::statistics(1, Statistics::FRAGMENT_SHADER_INVOCATIONS).unwrap();
        let recorded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            (compute.reserve().begin().unwrap()).submit(|cmd| cmd.query(&pool, 0, |_| {}))
        }));
        assert!(recorded.is_err());
    }
}

/// Releasing to the family that owns the image does nothing.
#[test]
fn a_release_to_the_same_family_does_nothing() {
    let gpu = gpu();
    let image = Owned::new(UVec2::splat(8)).unwrap();
    let readback = zeroed_buffer::<u8, Storage>(8 * 8 * 4);
    gpu.submit(|cmd| {
        cmd.clear_image(image.whole_view(), [0.0, 0.0, 1.0, 1.0]);
        cmd.release::<Graphics>(image.whole_view());
        cmd.copy_image_to_buffer(image.whole(), readback.range(..));
    });
    assert!(readback.host().chunks_exact(4).all(|t| t == [0, 0, 255, 255]));
}

/// A group of two compute passes, then a rendering if the device has mesh shaders.
fn record_profiled(cmd: &mut Cmd, target: &Target) {
    cmd.scope("group", |cmd| {
        for color in [Vec4::ONE, Vec4::ZERO] {
            cmd.compute(
                &TestComputeImage::push_bindings(target.whole_view(), color, UVec2::ZERO, EXTENT),
                [SIZE / 8, SIZE / 8, 1],
                None,
            );
        }
    });
    if Ctx::features().mesh_shader {
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::launch(
                &TestMesh::push_bindings(Vec4::ONE, Vec2::new(1.0, 0.0), 0.4).unwrap(),
                [2, 1, 1],
                RasterState::default().cull_mode(CullModeFlags::NONE),
                None,
            ))
            .record("TestMesh");
    }
}

/// Submits `record_profiled` measured by `observer` under the root "frame", and waits for it.
fn submit_profiled(gpu: &common::Gpu, observer: &mut TestObserver, target: &Target) {
    gpu.submit(|cmd| cmd.measure(Some(observer), "frame", |cmd| record_profiled(cmd, target)));
}

/// Logs every scope it sees and times it with two timestamps, and gives every pass a statistics
/// and a counter query if it has the pools and no query of that kind is active.
struct TestObserver {
    timestamps: QueryPool<Timestamp>,
    statistics: Option<QueryPool<PipelineStatistics>>,
    counters: Option<QueryPool<Performance>>,
    /// Every hook call: name, depth, pass, and whether it entered.
    events: Vec<(&'static str, u16, bool, bool)>,
    /// Per scope in entry order, and per pass in entry order with its queries.
    scopes: usize,
    passes: Vec<(&'static str, Option<u32>, Option<u32>)>,
    /// Per open scope: its index and, for a pass, its index in `passes`.
    open: Vec<(usize, Option<usize>)>,
}

impl TestObserver {
    fn new(statistics: Statistics, counters: &[&'static counters::Counter]) -> Self {
        Self {
            timestamps: QueryPool::timestamps(32).unwrap(),
            statistics: (!statistics.is_empty())
                .then(|| QueryPool::statistics(8, statistics).unwrap()),
            counters: (!counters.is_empty()).then(|| QueryPool::performance(8, counters).unwrap()),
            events: Vec::new(),
            scopes: 0,
            passes: Vec::new(),
            open: Vec::new(),
        }
    }

    /// The start and end of every scope in entry order, in ns since the first started.
    fn times(&self) -> Vec<(u64, u64)> {
        let mut ticks = vec![0; 2 * self.scopes];
        assert!(
            self.timestamps.read(0, &mut ticks).unwrap(),
            "the recording finished"
        );
        let ns = |tick| self.timestamps.elapsed_ns(ticks[0], tick);
        ticks.chunks(2).map(|t| (ns(t[0]), ns(t[1]))).collect()
    }
}

impl ScopeObserver for TestObserver {
    fn on_enter(&mut self, queries: &mut QueryRecorder, scope: &ScopeInfo) {
        self.events
            .push((scope.name, scope.depth, scope.pass, true));
        let index = self.scopes;
        self.scopes += 1;
        let start = 2 * index as u32;
        queries.write_timestamp(&self.timestamps, start, lava::PipelineStageFlags2::NONE);
        let mut pass = None;
        if scope.pass {
            let used = self.passes.len() as u32;
            let statistics = (self.statistics.as_ref())
                .filter(|_| !queries.is_active::<PipelineStatistics>())
                .map(|pool| {
                    queries.begin_query(pool, used);
                    used
                });
            let counters = (self.counters.as_ref())
                .filter(|_| !queries.is_active::<Performance>())
                .map(|pool| {
                    queries.begin_query(pool, used);
                    used
                });
            pass = Some(self.passes.len());
            self.passes.push((scope.name, statistics, counters));
        }
        self.open.push((index, pass));
    }

    fn on_exit(&mut self, queries: &mut QueryRecorder, scope: &ScopeInfo) {
        self.events
            .push((scope.name, scope.depth, scope.pass, false));
        let (index, pass) = self.open.pop().unwrap();
        if let Some(pass) = pass {
            let (_, statistics, counters) = self.passes[pass];
            if let (Some(pool), Some(query)) = (&self.counters, counters) {
                queries.end_query(pool, query);
            }
            if let (Some(pool), Some(query)) = (&self.statistics, statistics) {
                queries.end_query(pool, query);
            }
        }
        let end = 2 * index as u32 + 1;
        queries.write_timestamp(
            &self.timestamps,
            end,
            lava::PipelineStageFlags2::ALL_COMMANDS,
        );
    }
}

#[test]
fn observers_see_every_scope_and_pass_once_in_nesting_order() {
    let gpu = gpu();
    let target = Target::new(UVec2::new(SIZE, SIZE)).unwrap().bind_storage().unwrap();
    let mut observer = TestObserver::new(Statistics::empty(), &[]);
    submit_profiled(&gpu, &mut observer, &target);
    let mut expected = vec![
        ("frame", 0, false, true),
        ("group", 1, false, true),
        ("TestComputeImage", 2, true, true),
        ("TestComputeImage", 2, true, false),
        ("TestComputeImage", 2, true, true),
        ("TestComputeImage", 2, true, false),
        ("group", 1, false, false),
    ];
    if Ctx::features().mesh_shader {
        expected.extend([("TestMesh", 1, true, true), ("TestMesh", 1, true, false)]);
    }
    expected.push(("frame", 0, false, false));
    assert_eq!(observer.events, expected);

    // Scopes in entry order: frame, group, the two passes.
    let times = observer.times();
    for (start, end) in &times {
        assert!(start <= end, "{times:?}");
    }
    assert!(times[0].1 > 0);
    let inside = |child: usize, parent: usize| {
        times[parent].0 <= times[child].0 && times[child].1 <= times[parent].1
    };
    assert!(inside(1, 0) && inside(2, 1) && inside(3, 1));
    assert!(times[3].0 >= times[2].1, "passes run in order");

    // Without an observer, a scope is only a label.
    gpu.submit(|cmd| cmd.measure(None, "frame", |cmd| record_profiled(cmd, &target)));
    assert_eq!(observer.events.len(), expected.len());
}

#[test]
fn observers_count_passes_and_renderings_with_queries() {
    let gpu = gpu();
    let target = Target::new(UVec2::new(SIZE, SIZE)).unwrap().bind_storage().unwrap();
    let features = Ctx::features();
    let mesh = features.mesh_shader;
    // One counter, which always fits a single pass. A time counter, if the device has one,
    // can't be 0 for a pass.
    let all = counters::counters().unwrap_or_default();
    let counter = (all.iter())
        .find(|c| c.unit == counters::CounterUnit::Nanoseconds)
        .or(all.first());
    let selection: Vec<&'static counters::Counter> = counter.into_iter().collect();
    let mut statistics = Statistics::empty();
    if features.pipeline_statistics {
        statistics =
            Statistics::FRAGMENT_SHADER_INVOCATIONS | Statistics::COMPUTE_SHADER_INVOCATIONS;
    }
    if features.pipeline_statistics && features.mesh_shader_queries {
        statistics |= Statistics::MESH_SHADER_INVOCATIONS_EXT;
    }
    let mut observer = TestObserver::new(statistics, &selection);
    submit_profiled(&gpu, &mut observer, &target);

    let names: Vec<_> = observer.passes.iter().map(|p| p.0).collect();
    let mut expected = vec!["TestComputeImage", "TestComputeImage"];
    if mesh {
        expected.push("TestMesh");
    }
    assert_eq!(names, expected);
    let passes = observer.passes.len();
    if let Some(pool) = &observer.statistics {
        let mut stats = vec![PipelineStats::default(); passes];
        assert!(pool.read(0, &mut stats).unwrap());
        let count = |pass: usize, statistic| stats[pass].get(statistic);
        for pass in 0..2 {
            let compute = Statistics::COMPUTE_SHADER_INVOCATIONS;
            assert_eq!(count(pass, compute), Some((SIZE * SIZE) as u64));
            assert_eq!(
                count(pass, Statistics::FRAGMENT_SHADER_INVOCATIONS),
                Some(0)
            );
        }
        if mesh {
            assert!(count(2, Statistics::FRAGMENT_SHADER_INVOCATIONS).unwrap() > 0);
            assert_eq!(count(2, Statistics::COMPUTE_SHADER_INVOCATIONS), Some(0));
            if features.mesh_shader_queries {
                assert!(count(2, Statistics::MESH_SHADER_INVOCATIONS_EXT).unwrap() > 0);
            }
        }
    }
    if let (Some(pool), Some(counter)) = (&observer.counters, counter) {
        let mut values = vec![0.0; passes];
        assert!(pool.read(0, &mut values).unwrap());
        for (value, (name, ..)) in values.iter().zip(&observer.passes) {
            assert!(value.is_finite() && *value >= 0.0, "{name}: {value}");
            if counter.unit == counters::CounterUnit::Nanoseconds {
                assert!(*value > 0.0, "{name}: {value}");
            }
        }
    }
}

#[test]
fn a_panic_inside_measure_detaches_the_observer() {
    let gpu = gpu();
    let target = Target::new(UVec2::new(SIZE, SIZE)).unwrap().bind_storage().unwrap();
    let mut observer = TestObserver::new(Statistics::empty(), &[]);
    gpu.submit(|cmd| {
        let measured = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cmd.measure(Some(&mut observer), "frame", |_| {
                panic!("the recording failed")
            })
        }));
        assert!(measured.is_err());
        // The observer's borrow ended, so this pass must not reach it.
        cmd.compute(
            &TestComputeImage::push_bindings(target.whole_view(), Vec4::ONE, UVec2::ZERO, EXTENT),
            [SIZE / 8, SIZE / 8, 1],
            None,
        );
    });
    assert_eq!(observer.events, [("frame", 0, false, true)]);
}

#[test]
fn statistics_queries_count_their_commands_and_keep_observers_from_nesting_one() {
    let gpu = gpu();
    if !Ctx::features().pipeline_statistics {
        return;
    }
    let target = Target::new(UVec2::new(SIZE, SIZE)).unwrap().bind_storage().unwrap();
    let compute = Statistics::COMPUTE_SHADER_INVOCATIONS;
    let pool = QueryPool::statistics(1, compute).unwrap();
    let mut observer = TestObserver::new(compute, &[]);
    gpu.submit(|cmd| {
        cmd.measure(Some(&mut observer), "frame", |cmd| {
            cmd.query(&pool, 0, |cmd| {
                for color in [Vec4::ONE, Vec4::ZERO] {
                    let bindings = TestComputeImage::push_bindings(
                        target.whole_view(),
                        color,
                        UVec2::ZERO,
                        EXTENT,
                    );
                    cmd.compute(&bindings, [SIZE / 8, SIZE / 8, 1], None);
                }
            })
        })
    });
    let mut stats = [PipelineStats::default()];
    assert!(pool.read(0, &mut stats).unwrap());
    assert_eq!(stats[0].get(compute), Some(2 * (SIZE * SIZE) as u64));
    assert_eq!(stats[0].get(Statistics::FRAGMENT_SHADER_INVOCATIONS), None);
    // One statistics query can be active at a time: the observer saw it and began none.
    assert_eq!(observer.passes.len(), 2);
    assert!(
        observer
            .passes
            .iter()
            .all(|(_, statistics, _)| statistics.is_none())
    );
}

#[test]
fn occlusion_queries_count_the_samples_of_a_draw() {
    let gpu = gpu();
    let vertices = overlapping_triangles();
    let depth = Image::<D32Sfloat, DepthAttachment>::new(UVec2::new(SIZE, SIZE)).unwrap();
    let precise = Ctx::features().precise_occlusion;
    let pool = match precise {
        true => QueryPool::precise_occlusion(3),
        false => QueryPool::occlusion(3),
    }
    .unwrap();
    render(&gpu, |cmd, target| {
        let near = TestRaster::push_bindings(vertices.range(0..3), Vec2::ZERO);
        let far = TestRaster::push_bindings(vertices.range(3..), Vec2::ZERO);
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .depth_attachment(depth.whole_view(), Some([0.0]))
            .draw(&Draw::new(&near, 3, 1, RasterState::default(), None).count_samples(&pool, 0))
            .draw(&Draw::new(&far, 3, 1, RasterState::default(), None).count_samples(&pool, 1))
            .draw(&Draw::new(&far, 3, 0, RasterState::default(), None).count_samples(&pool, 2))
            .record("occluded")
    });
    let mut samples = [0; 3];
    assert!(pool.read(0, &mut samples).unwrap());
    assert!(samples[0] > 0, "the near triangle is visible");
    if precise {
        assert!(samples[0] < (SIZE * SIZE) as u64, "{}", samples[0]);
    }
    assert_eq!(samples[1], 0, "the far triangle is hidden behind it");
    assert_eq!(samples[2], 0, "a skipped draw counts nothing");
}

// ---- buffer commands -------------------------------------------------------------------------

#[test]
fn fill_buffer_fills_exactly_the_slice() {
    let gpu = gpu();
    let buffer = zeroed_buffer::<u32, Storage>(8);
    gpu.submit(|cmd| cmd.fill_buffer(buffer.range(2..6), 0xDEAD_BEEF));
    assert_eq!(
        buffer.host(),
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
    assert_eq!(buffer.host(), [[0, 0], [0, 0], [11, 22], [0, 0]]);
}

#[test]
fn copy_buffer_copies_the_source_slice_to_the_destination_offset() {
    let gpu = gpu();
    let src = buffer_with::<u32, Storage>(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let dst = zeroed_buffer::<u32, Storage>(8);
    gpu.submit(|cmd| cmd.copy_buffer(src.range(1..4), dst.range(5..)));
    assert_eq!(dst.host(), [0, 0, 0, 0, 0, 2, 3, 4]);
}

#[test]
fn copy_buffer_regions_copies_each_region() {
    let gpu = gpu();
    let src = buffer_with::<u32, Storage>(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let dst = zeroed_buffer::<u32, Storage>(8);
    let regions = [
        (src.range(0..2), dst.range(6..)),
        (src.range(7..8), dst.range(0..1)),
    ];
    gpu.submit(|cmd| cmd.copy_buffer_regions(&regions));
    assert_eq!(dst.host(), [8, 0, 0, 0, 0, 0, 1, 2]);
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
    assert_eq!(copy.host(), expected);
}

// ---- images ----------------------------------------------------------------------------------

#[test]
fn image_reports_extent_and_mip_chain() {
    let _gpu = gpu();
    let image = Image::<R8G8B8A8Unorm, Sampled>::with_mip_levels(UVec2::new(16, 8), 4).unwrap();
    assert_eq!(image.extent(), UVec2::new(16, 8));
    assert_eq!(image.mip_levels(), 4);
    assert_eq!(image.mip_extent(0), UVec2::new(16, 8));
    assert_eq!(image.mip_extent(2), UVec2::new(4, 2));
    assert_eq!(image.mip_extent(3), UVec2::new(2, 1));

    let whole = image.whole_view();
    assert_eq!(whole.subresource_range().level_count, 4);
    assert_eq!(whole.extent(), UVec2::new(16, 8));
    assert_eq!(image.whole().extend, UVec2::new(16, 8));
}

/// A mip chain has a view per mip, sized like its mip and bound at its own slot, the mips'
/// slots contiguous.
#[test]
fn mip_chains_have_a_view_and_handle_per_mip() {
    let _gpu = gpu();
    let chain = MipChain::<R8G8B8A8Unorm, SampledStorage>::with_mip_levels(UVec2::new(16, 8), 4)
        .unwrap()
        .bind_sampled()
        .unwrap()
        .bind_mips_storage()
        .unwrap();
    assert_ne!(chain.handle().descriptor_index_set0, NULL_HANDLE);
    assert_eq!(chain.handle().descriptor_index_set1, NULL_HANDLE, "only the mips are storage");
    let whole = chain.whole_view();
    assert_eq!(whole.subresource_range().level_count, 4);
    let first = chain.mip_view(0).handle().descriptor_index_set1;
    for level in 0..4 {
        let view = chain.mip_view(level);
        let range = view.subresource_range();
        assert_eq!((range.base_mip_level, range.level_count), (level, 1));
        assert_eq!(view.extent(), chain.mip_extent(level));
        assert_eq!(view.handle(), BindlessHandle::storage(first + level));
        assert_ne!(view.view(), whole.view());
    }
    assert_eq!(chain.mip_view(2).whole().extend, UVec2::new(4, 2));
}

#[test]
fn binds_take_free_slots_of_their_sets() {
    let _gpu = gpu();
    let sampled = Image::<R8G8B8A8Unorm, Sampled>::new(UVec2::new(4, 4))
        .unwrap()
        .bind_sampled()
        .unwrap();
    let storage = Image::<R8G8B8A8Unorm, StorageImage>::new(UVec2::new(4, 4))
        .unwrap()
        .bind_storage()
        .unwrap();
    let both = Image::<R8G8B8A8Unorm, SampledStorage>::new(UVec2::new(4, 4))
        .unwrap()
        .bind_sampled()
        .unwrap()
        .bind_storage()
        .unwrap();
    let unbound = Image::<R8G8B8A8Unorm, Sampled>::new(UVec2::new(4, 4)).unwrap();

    let (s, st, b) = (sampled.handle(), storage.handle(), both.handle());
    assert!(s.descriptor_index_set0 < max_sampled_images());
    assert_eq!(s.descriptor_index_set1, NULL_HANDLE);
    assert_eq!(st.descriptor_index_set0, NULL_HANDLE);
    assert!(st.descriptor_index_set1 < max_storage_images());
    assert_ne!(b.descriptor_index_set0, s.descriptor_index_set0, "one image per slot");
    assert_ne!(b.descriptor_index_set1, st.descriptor_index_set1);
    assert_eq!(unbound.handle(), BindlessHandle::none());

    // `bind` writes reserved slots only, and checks the usage at runtime, static types
    // included.
    let reserved = lava::bindless::reserve_sampled(4).unwrap();
    assert_eq!(reserved.len(), 4);
    assert!(!reserved.indices().contains(&s.descriptor_index_set0));
    lava::bindless::bind([unbound.whole_view(); 4], (0..4).map(|i| reserved.handle(i)));
    let unreserved = std::panic::catch_unwind(|| {
        lava::bindless::bind([unbound.whole_view()], [s])
    });
    assert!(unreserved.is_err(), "the sampled image's slot isn't reserved");
    let storage_slots = lava::bindless::reserve_storage(1).unwrap();
    let refused = std::panic::catch_unwind(|| {
        lava::bindless::bind([unbound.whole_view()], [storage_slots.handle(0)])
    });
    assert!(refused.is_err(), "a sampled image can't be bound as storage");
    assert!(std::panic::catch_unwind(|| reserved.handle(4)).is_err());
}

/// A dropped image's slots, and a dropped reservation's, are free again once no frame reaches
/// them; `bind` refuses a dropped reservation right away.
#[test]
fn slots_are_free_again_once_their_image_is_destroyed() {
    let _gpu = gpu();
    // Destroys what earlier tests dropped, so the lowest free slots are the first ones.
    lava::wait_idle().unwrap();
    let bind = || {
        Image::<R8G8B8A8Unorm, Sampled>::new(UVec2::new(4, 4))
            .unwrap()
            .bind_sampled()
            .unwrap()
    };
    let image = bind();
    let slot = image.handle();
    drop(image);
    assert_ne!(bind().handle(), slot, "taken until no frame reaches it");
    lava::wait_idle().unwrap();
    assert_eq!(bind().handle(), slot);

    // A mip chain frees its whole view's slot and each mip's, erased or not.
    lava::wait_idle().unwrap();
    let chain = MipChain::<R8G8B8A8Unorm, SampledStorage>::with_mip_levels(UVec2::new(4, 4), 3)
        .unwrap()
        .bind_sampled()
        .unwrap()
        .bind_mips_storage()
        .unwrap()
        .erase();
    assert_eq!(chain.format().0, lava::VkFormat::R8G8B8A8_UNORM);
    let (whole, mips) = (chain.handle(), chain.mip_view(0).handle());
    drop(chain);
    lava::wait_idle().unwrap();
    assert_eq!(bind().handle(), whole);
    let storage = lava::bindless::reserve_storage(3).unwrap();
    assert_eq!(storage.start(), mips.descriptor_index_set1, "the mips' slots came back whole");

    // A dropped reservation can't be written, and its slots come back once its frames are
    // done.
    let texture = Image::<R8G8B8A8Unorm, Sampled>::new(UVec2::new(4, 4)).unwrap();
    let handle = storage.handle(0);
    let start = storage.start();
    drop(storage);
    let written = std::panic::catch_unwind(|| {
        lava::bindless::bind([texture.whole_view()], [handle])
    });
    assert!(written.is_err());
    lava::wait_idle().unwrap();
    assert_eq!(lava::bindless::reserve_storage(3).unwrap().start(), start);
}

#[test]
fn clear_image_sets_every_texel() {
    let gpu = gpu();
    let image =
        Image::<R8G8B8A8Unorm, StorageTransferSrcTransferDst>::new(UVec2::new(8, 8)).unwrap();
    let readback = zeroed_buffer::<u8, Storage>(8 * 8 * 4);
    gpu.submit(|cmd| {
        cmd.clear_image(image.whole_view(), [1.0, 0.0, 0.5, 1.0]);
        cmd.copy_image_to_buffer(image.whole(), readback.range(..));
    });
    for texel in readback.host().chunks_exact(4) {
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
    let image =
        Image::<R8G8B8A8Unorm, SampledTransferSrcTransferDst>::new(UVec2::new(8, 8)).unwrap();
    let readback = zeroed_buffer::<u8, Storage>(data.len());
    gpu.submit(|cmd| {
        cmd.copy_buffer_to_image(upload.range(..), image.whole());
        cmd.copy_image_to_buffer(image.whole(), readback.range(..));
    });
    assert_eq!(readback.host(), data);
}

#[test]
fn image_region_copies_address_the_sub_rectangle() {
    let gpu = gpu();
    let data = pattern(8, 8);
    let upload = buffer_with::<u8, Storage>(&data);
    let image =
        Image::<R8G8B8A8Unorm, SampledTransferSrcTransferDst>::new(UVec2::new(8, 8)).unwrap();
    let readback = zeroed_buffer::<u8, Storage>(3 * 2 * 4);
    let region = image
        .whole_view()
        .region(UVec2::new(3, 2))
        .offset(IVec2::new(4, 5));
    gpu.submit(|cmd| {
        cmd.copy_buffer_to_image(upload.range(..), image.whole());
        cmd.copy_image_to_buffer(region, readback.range(..));
    });
    let readback = readback.host();
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
    let src = Image::<R8G8B8A8Unorm, SampledTransferSrcTransferDst>::new(UVec2::new(8, 8)).unwrap();
    let dst = Image::<R8G8B8A8Unorm, SampledTransferSrcTransferDst>::new(UVec2::new(8, 8)).unwrap();
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
    let readback = readback.host();
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
    let src = Image::<R8G8B8A8Unorm, SampledTransferSrcTransferDst>::new(UVec2::new(2, 2)).unwrap();
    let dst = Image::<R8G8B8A8Unorm, SampledTransferSrcTransferDst>::new(UVec2::new(8, 8)).unwrap();
    let readback = zeroed_buffer::<u8, Storage>(8 * 8 * 4);
    gpu.submit(|cmd| {
        cmd.copy_buffer_to_image(upload.range(..), src.whole());
        cmd.blit_image(src.whole(), dst.whole(), Filter::Nearest);
        cmd.copy_image_to_buffer(dst.whole(), readback.range(..));
    });
    let readback = readback.host();
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
    let mut image =
        Image::<R8G8B8A8Unorm, SampledHostCopyTransferSrc>::with_mip_levels(UVec2::new(8, 8), 2)
            .unwrap();
    let data = pattern(8, 8);

    if !Ctx::features().host_image_copy {
        // Host image copies need host-visible device memory.
        assert!(image.copy_from(&data, 0).is_err());
        return;
    }

    image.copy_from(&data, 0).unwrap();
    let readback = zeroed_buffer::<u8, Storage>(data.len());
    gpu.submit(|cmd| cmd.copy_image_to_buffer(image.whole(), readback.range(..)));
    assert_eq!(readback.host(), data);

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
    assert_eq!(readback.host(), data);
}

#[test]
fn host_copy_writes_only_the_region() {
    let gpu = gpu();
    let mut image =
        Image::<R8G8B8A8Unorm, SampledHostCopyTransferSrc>::new(UVec2::new(8, 8)).unwrap();
    let (offset, extent) = (UVec2::new(4, 5), UVec2::new(3, 2));
    let patch = vec![200u8; 3 * 2 * 4];

    if !Ctx::features().host_image_copy {
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
    let readback = readback.host();
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
) -> impl FnOnce(&mut Cmd) + 'a {
    move |cmd| {
        cmd.compute(
            &TestComputeImage::push_bindings(
                target.whole_view(),
                Vec4::new(1.0, 1.0, 1.0, 1.0),
                offset,
                size,
            ),
            [SIZE / 8, SIZE / 8, 1],
            None,
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
            &TestComputeImage::push_bindings(
                target.whole_view(),
                Vec4::new(0.0, 1.0, 0.0, 1.0),
                UVec2::new(16, 24),
                UVec2::new(32, 16),
            ),
            [SIZE / 8, SIZE / 8, 1],
            None,
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
            &TestComputeImage::push_bindings(target.whole_view(), Vec4::ONE, UVec2::ZERO, EXTENT),
            dispatch.range(..),
            None,
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
            &TestComputeBuffer::push_bindings(src.range(..), dst.range(..), 90, 3),
            [2, 1, 1],
            None,
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
    assert_eq!(dst.host(), expected);
}

/// The generated clock readout: `regions` lists the fields in order, `take` splits each
/// counter into ticks and subgroups and zeroes it.
#[test]
fn clock_regions_count_every_subgroup_once() {
    let gpu = gpu();
    if !Ctx::features().shader_clock {
        eprintln!("skipping clock_regions_count_every_subgroup_once: no subgroup clocks");
        return;
    }
    const INVOCATIONS: usize = 256;
    let mut clocks = zeroed_buffer::<TestClocks, Storage>(1);
    let out = zeroed_buffer::<u32, Storage>(INVOCATIONS);
    gpu.submit(|cmd| {
        cmd.compute(
            &TestClock::push_bindings(clocks.address(), out.range(..)).unwrap(),
            [INVOCATIONS as u32 / 64, 1, 1],
            None,
        )
    });
    let expected: Vec<u32> = (1..=INVOCATIONS as u32).collect();
    assert_eq!(out.host(), expected);
    let regions = clocks.host_mut()[0].regions();
    let names: Vec<&str> = regions.iter().map(|(name, _)| *name).collect();
    assert_eq!(names, ["first", "second"]);
    for (name, clock) in regions {
        let (_, subgroups) = clock.take();
        // Subgroups have 4 to 128 lanes.
        let lanes = INVOCATIONS as u64 / subgroups.max(1);
        assert!((4..=128).contains(&lanes), "{name}: {subgroups} subgroups");
        assert_eq!(clock.take(), (0, 0), "{name} is zeroed");
    }
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
            &TestComputeBuffer::push_bindings(src.range(..), dst.range(..), 64, 2),
            [1, 1, 1],
            None,
        );
        cmd.copy_buffer(dst.range(..), copy.range(..));
    });
    let expected: Vec<u32> = (0..64).map(|i| 10 + i).collect();
    assert_eq!(copy.host(), expected);
}

/// Two sets of specialization constants make two pipelines; each dispatch writes through the
/// buffer and image handles of its own set, with its own scalars.
#[test]
fn specialization_constants_pick_the_pipeline() {
    let gpu = gpu();
    let (a, b) = (
        zeroed_buffer::<u32, Storage>(8),
        zeroed_buffer::<u32, Storage>(8),
    );
    let (image_a, image_b) = (
        Target::new(UVec2::new(SIZE, SIZE)).unwrap().bind_storage().unwrap(),
        Target::new(UVec2::new(SIZE, SIZE)).unwrap().bind_storage().unwrap(),
    );
    let pixels = zeroed_buffer::<u8, Storage>((SIZE * SIZE * 4 * 2) as usize);
    let half = (SIZE * SIZE * 4) as usize;
    gpu.submit(|cmd| {
        let groups = [SIZE / 8, SIZE / 8, 1];
        let first = |offset| {
            TestConstants::push_bindings(offset, 8).constant_bindings(
                a.range(..),
                image_a.whole_view(),
                3,
                1.0,
            )
        };
        cmd.compute(&first(100), groups, None);
        cmd.compute(
            &TestConstants::push_bindings(0, 4).constant_bindings(
                b.range(..),
                image_b.whole_view(),
                5,
                0.5,
            ),
            groups,
            None,
        );
        // The first set again, now cached, with other push constants.
        cmd.compute(&first(200), groups, None);
        cmd.copy_image_to_buffer(image_a.whole(), pixels.range(..half));
        cmd.copy_image_to_buffer(image_b.whole(), pixels.range(half..));
    });
    let expected_a: Vec<u32> = (0..8).map(|i| 200 + i * 3).collect();
    assert_eq!(a.host(), expected_a);
    assert_eq!(b.host(), [0, 5, 10, 15, 0, 0, 0, 0]);
    let pixels = pixels.host();
    assert_pixel(&pixels[..half], 7, 9, [255, 0, 0, 255]);
    assert_pixel(&pixels[half..], 7, 9, [128, 0, 0, 255]);
}

// ---- raster passes ---------------------------------------------------------------------------

fn bindings(
    vertices: &Buffer<TestVertex>,
) -> impl std::ops::Deref<Target = BindingOutput<'_, kind::RasterVertex>> {
    TestRaster::push_bindings(vertices.range(..), Vec2::ZERO)
}

fn unculled() -> RasterState<1> {
    RasterState::default().cull_mode(CullModeFlags::NONE)
}

/// Draws the first `vertex_count` vertices of [`two_triangles`] with backface culling on/off.
fn draw_triangles(gpu: &common::Gpu, vertex_count: u32, culling: bool) -> Vec<u8> {
    let vertices = two_triangles();
    render(gpu, |cmd, target| {
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::new(
                &bindings(&vertices),
                vertex_count,
                1,
                RasterState::default().cull_mode(if culling {
                    CullModeFlags::BACK
                } else {
                    CullModeFlags::NONE
                }),
                None,
            ))
            .record("triangles")
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
    if !Ctx::features().fill_mode_non_solid {
        eprintln!("SKIPPED wireframe_draws_only_the_edges: the device lacks `fill-mode-non-solid`");
        return;
    }
    let vertices = two_triangles();
    let pixels = render(&gpu, |cmd, target| {
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::new(
                &bindings(&vertices),
                6,
                1,
                unculled()
                    .polygon_mode(PolygonMode::LINE)
                    .expect("the device has `fill-mode-non-solid`"),
                None,
            ))
            .record("wireframe")
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
        cmd.raster(
            EXTENT,
            [Attachment::color(
                target.whole_view(),
                Some([0.0, 0.0, 1.0, 1.0]),
            )],
        )
        .draw(&Draw::new(
            &bindings(&vertices),
            3,
            1,
            RasterState::default(),
            None,
        ))
        .record("cleared")
    });
    assert_pixel(&pixels, 32, 32, [0, 0, 255, 255]);
    assert_ne!(pixel(&pixels, IN_LEFT.0, IN_LEFT.1), [0, 0, 255, 255]);
}

#[test]
fn instances_are_drawn_with_their_instance_index() {
    let gpu = gpu();
    let vertices = two_triangles();
    let pixels = render(&gpu, |cmd, target| {
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            // The second instance of the left triangle lands where the right one would be.
            .draw(&Draw::new(
                &TestRaster::push_bindings(vertices.range(..), Vec2::new(1.0, 0.0)),
                3,
                2,
                RasterState::default(),
                None,
            ))
            .record("instanced")
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
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::indirect(
                &bindings(&vertices),
                commands.range(..),
                RasterState::default(),
                None,
            ))
            .record("indirect")
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
            cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
                .draw(&Draw::indirect_count(
                    &bindings(&vertices),
                    commands.range(..),
                    count.range(..),
                    unculled(),
                    None,
                ))
                .record("indirect count")
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
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .scissors(&[Scissor {
                offset: IVec2::ZERO,
                extent: UVec2::new(SIZE / 2, SIZE),
            }])
            .draw(&Draw::new(&bindings(&vertices), 6, 1, unculled(), None))
            .record("scissored")
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
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .viewport(Viewport {
                offset: IVec2::splat(SIZE as i32 / 2),
                extent: EXTENT / 2,
            })
            .draw(&Draw::new(&bindings(&vertices), 6, 1, unculled(), None))
            .record("viewport")
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
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::new(
                &bindings(&vertices),
                6,
                1,
                RasterState::default(),
                None,
            ))
            .record("without depth")
    });
    // Without a depth test the later (far, red) triangle wins.
    assert_pixel(&without_depth, 32, 40, [255, 0, 0, 255]);

    let depth = Image::<D32Sfloat, DepthAttachment>::new(UVec2::new(SIZE, SIZE)).unwrap();
    let with_depth = render(&gpu, |cmd, target| {
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .depth_attachment(depth.whole_view(), Some([0.0]))
            .draw(&Draw::new(
                &bindings(&vertices),
                6,
                1,
                RasterState::default(),
                None,
            ))
            .record("with depth")
    });
    assert_pixel(&with_depth, 32, 40, [0, 255, 0, 255]);
    assert_golden("raster_depth", &with_depth, [SIZE, SIZE]);
}

#[test]
fn blend_modes_combine_overlapping_fragments() {
    let gpu = gpu();
    let vertices = overlapping_triangles();
    let draw = |clear: [f32; 4], blend| {
        render(&gpu, |cmd, target| {
            cmd.raster(
                EXTENT,
                [Attachment::color(target.whole_view(), Some(clear))],
            )
            .draw(&Draw::new(
                &bindings(&vertices),
                6,
                1,
                RasterState::default().blend(blend),
                None,
            ))
            .record("blended")
        })
    };
    // Green, then red on top of it, both opaque.
    assert_pixel(&draw([0.0; 4], Blend::Alpha), 32, 40, [255, 0, 0, 255]);
    assert_pixel(&draw([0.0; 4], Blend::Add), 32, 40, [255, 255, 0, 255]);
    // White loses green, then red; alpha goes with the first draw.
    assert_pixel(&draw([1.0; 4], Blend::Attenuate), 32, 40, [0, 0, 255, 0]);
    // Outside the triangles the clear stays.
    assert_pixel(&draw([1.0; 4], Blend::Attenuate), 2, 2, [255; 4]);
}

#[test]
fn skipped_attachment_keeps_its_content() {
    let gpu = gpu();
    if !Ctx::features().independent_blend {
        eprintln!("SKIPPED skipped_attachment_keeps_its_content: the device lacks `independent-blend`");
        return;
    }
    let vertices = overlapping_triangles();
    let second = Target::new(UVec2::new(SIZE, SIZE)).unwrap().bind_storage().unwrap();
    let readback = zeroed_buffer::<u8, Storage>((SIZE * SIZE * 4) as usize);
    let pixels = render(&gpu, |cmd, target| {
        cmd.raster(
            EXTENT,
            [
                Attachment::color(target.whole_view(), None),
                Attachment::color(second.whole_view(), Some([0.0, 0.0, 1.0, 1.0])),
            ],
        )
        .draw(&Draw::new(
            &bindings(&vertices),
            6,
            1,
            RasterState::default()
                .blends([Blend::Replace, Blend::Skip])
                .expect("the device has `independent-blend`"),
            None,
        ))
        .record("skip");
        cmd.copy_image_to_buffer(second.whole(), readback.range(..));
    });
    assert_pixel(&pixels, 32, 40, [255, 0, 0, 255]);
    let second = readback.host();
    assert!(second.chunks_exact(4).all(|p| p == [0, 0, 255, 255]));
}

/// With counter-clockwise front faces the other triangle is culled.
#[test]
fn front_face_decides_which_winding_is_culled() {
    let gpu = gpu();
    let vertices = two_triangles();
    let pixels = render(&gpu, |cmd, target| {
        let state = RasterState::default().front_face(FrontFace::COUNTER_CLOCKWISE);
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::new(&bindings(&vertices), 6, 1, state, None))
            .record("counter-clockwise");
    });
    assert_pixel(&pixels, IN_LEFT.0, IN_LEFT.1, BLACK);
    assert_pixel(&pixels, IN_RIGHT.0, IN_RIGHT.1, [255, 255, 0, 255]);
}

#[test]
fn line_lists_draw_only_edges() {
    let gpu = gpu();
    let vertices = two_triangles();
    let pixels = render(&gpu, |cmd, target| {
        let state = unculled().topology(PrimitiveTopology::LINE_LIST);
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::new(&bindings(&vertices), 6, 1, state, None))
            .record("lines");
    });
    assert_pixel(&pixels, IN_LEFT.0, IN_LEFT.1, BLACK);
    assert!(pixels.chunks_exact(4).any(|p| p != BLACK));
}

#[test]
fn write_masks_keep_the_masked_components() {
    let gpu = gpu();
    let vertices = overlapping_triangles();
    let pixels = render(&gpu, |cmd, target| {
        let blue = Attachment::color(target.whole_view(), Some([0.0, 0.0, 1.0, 1.0]));
        let state = RasterState::default().write_masks([ColorComponentFlags::R]);
        cmd.raster(EXTENT, [blue])
            .draw(&Draw::new(&bindings(&vertices), 6, 1, state, None))
            .record("red only");
    });
    // The red triangle's red over the cleared blue; the green one wrote nothing.
    assert_pixel(&pixels, 32, 40, [255, 0, 255, 255]);
}

#[test]
fn blend_equations_take_any_factors_and_ops() {
    let gpu = gpu();
    let vertices = overlapping_triangles();
    let min = BlendEquation {
        src_color: BlendFactor::ONE,
        dst_color: BlendFactor::ONE,
        color_op: BlendOp::MIN,
        src_alpha: BlendFactor::ONE,
        dst_alpha: BlendFactor::ONE,
        alpha_op: BlendOp::MIN,
    };
    let pixels = render(&gpu, |cmd, target| {
        let white = Attachment::color(target.whole_view(), Some([1.0; 4]));
        let state = RasterState::default().blend(Blend::Equation(min));
        cmd.raster(EXTENT, [white])
            .draw(&Draw::new(&bindings(&vertices), 6, 1, state, None))
            .record("min");
    });
    // White, then the minimum with green, then with red.
    assert_pixel(&pixels, 32, 40, [0, 0, 0, 255]);
    assert_pixel(&pixels, 2, 2, [255; 4]);
}

#[test]
fn logic_ops_replace_blending() {
    let gpu = gpu();
    if !Ctx::features().logic_op {
        eprintln!("SKIPPED logic_ops_replace_blending: the device lacks `logic-op`");
        return;
    }
    let vertices = overlapping_triangles();
    let pixels = render(&gpu, |cmd, target| {
        let white = Attachment::color(target.whole_view(), Some([1.0; 4]));
        let state = RasterState::default().logic_op(Some(LogicOp::XOR)).unwrap();
        cmd.raster(EXTENT, [white])
            .draw(&Draw::new(&bindings(&vertices), 6, 1, state, None))
            .record("xor");
    });
    // White XOR green XOR red.
    assert_pixel(&pixels, 32, 40, [0, 0, 255, 255]);
}

/// The left triangle only marks the stencil; then both triangles are drawn where it is set.
#[test]
fn stencil_limits_a_draw_to_what_an_earlier_one_marked() {
    let gpu = gpu();
    let vertices = two_triangles();
    let depth = Image::<D32SfloatS8Uint, DepthAttachment>::new(UVec2::new(SIZE, SIZE)).unwrap();
    let mark = StencilFace {
        pass: StencilOp::REPLACE,
        reference: 1,
        ..Default::default()
    };
    let marked = StencilFace {
        compare: CompareOp::EQUAL,
        reference: 1,
        ..Default::default()
    };
    let pixels = render(&gpu, |cmd, target| {
        let marking = unculled()
            .depth_test(None)
            .stencil(Some(Stencil {
                front: mark,
                back: mark,
            }))
            .write_masks([ColorComponentFlags::empty()]);
        let drawing = unculled().depth_test(None).stencil(Some(Stencil {
            front: marked,
            back: marked,
        }));
        let left = TestRaster::push_bindings(vertices.range(0..3), Vec2::ZERO);
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .depth_attachment(depth.whole_view(), Some((0.0, 0)))
            .draw(&Draw::new(&left, 3, 1, marking, None))
            .draw(&Draw::new(&bindings(&vertices), 6, 1, drawing, None))
            .record("stencil");
    });
    assert_ne!(pixel(&pixels, IN_LEFT.0, IN_LEFT.1), BLACK);
    assert_pixel(&pixels, IN_RIGHT.0, IN_RIGHT.1, BLACK);
}

/// A depth-stencil image with `DepthSplit` views is drawn to through its view of both aspects,
/// and a pass reads the depth through its depth-only view, which `bind_sampled` binds.
#[test]
fn split_depth_stencil_views_are_drawn_to_and_sampled() {
    let gpu = gpu();
    let vertices = two_triangles();
    let depth = Image::<D32SfloatS8Uint, DepthAttachmentSampled, D2, DepthSplit>::new(EXTENT)
        .unwrap()
        .bind_sampled().unwrap();
    let out = zeroed_buffer::<f32, Storage>((SIZE * SIZE) as usize);
    render(&gpu, |cmd, target| {
        let left = TestRaster::push_bindings(vertices.range(0..3), Vec2::ZERO);
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .depth_attachment(depth.whole_view(), Some((0.25, 0)))
            .draw(&Draw::new(&left, 3, 1, unculled(), None))
            .record("depth");
        cmd.compute(
            &TestSampleDepth::push_bindings(depth.whole_view(), out.range(..)),
            [SIZE / 8, SIZE / 8, 1],
            None,
        );
    });
    let depth_at = |(x, y): (u32, u32)| out.host()[(y * SIZE + x) as usize];
    assert_eq!(depth_at(IN_LEFT), 0.5);
    assert_eq!(depth_at(IN_RIGHT), 0.25);

    // Bound directly, the view of both aspects is refused.
    let slot = lava::bindless::reserve_sampled(1).unwrap();
    let refused = std::panic::catch_unwind(|| {
        lava::bindless::bind([depth.whole_view()], [slot.handle(0)])
    });
    assert!(refused.is_err(), "shaders read one aspect");
}

#[test]
fn depth_persists_between_draws_unless_cleared() {
    let gpu = gpu();
    let vertices = overlapping_triangles();
    let depth = Image::<D32Sfloat, DepthAttachment>::new(UVec2::new(SIZE, SIZE)).unwrap();

    let draw_far_after_near = |clear_between: Option<[f32; 1]>| {
        render(&gpu, |cmd, target| {
            cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
                .depth_attachment(depth.whole_view(), Some([0.0]))
                .draw(&Draw::new(
                    &TestRaster::push_bindings(vertices.range(0..3), Vec2::ZERO),
                    3,
                    1,
                    RasterState::default(),
                    None,
                ))
                .record("near");
            cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
                .depth_attachment(depth.whole_view(), clear_between)
                .draw(&Draw::new(
                    &TestRaster::push_bindings(vertices.range(3..), Vec2::ZERO),
                    3,
                    1,
                    RasterState::default(),
                    None,
                ))
                .record("far");
        })
    };

    // The near triangle's depth is still there for the second draw.
    assert_pixel(&draw_far_after_near(None), 32, 40, [0, 255, 0, 255]);
    // Clearing depth in between lets the far triangle through.
    assert_pixel(&draw_far_after_near(Some([0.0])), 32, 40, [255, 0, 0, 255]);
}

/// A draw is depth-tested against the depth an earlier draw of the same rendering wrote, with
/// no barrier in between.
#[test]
fn draws_of_one_rendering_see_the_earlier_depth() {
    let gpu = gpu();
    let near = overlapping_triangles();
    // Larger than the near triangle, so its rim stays visible.
    let far = buffer_with::<TestVertex, Storage>(&[
        vertex(0.0, -0.95, 0.2, RED),
        vertex(0.95, 0.95, 0.2, RED),
        vertex(-0.95, 0.95, 0.2, RED),
    ]);
    let depth = Image::<D32Sfloat, DepthAttachment>::new(UVec2::new(SIZE, SIZE)).unwrap();
    let pixels = render(&gpu, |cmd, target| {
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .depth_attachment(depth.whole_view(), Some([0.0]))
            .draw(&Draw::new(
                &TestRaster::push_bindings(near.range(0..3), Vec2::ZERO),
                3,
                1,
                RasterState::default(),
                None,
            ))
            .draw(&Draw::new(
                &bindings(&far),
                3,
                1,
                RasterState::default(),
                None,
            ))
            .record("near then far")
    });
    assert_pixel(&pixels, 32, 40, [0, 255, 0, 255]);
    // Below the near triangle's base.
    assert_pixel(&pixels, 32, 61, [255, 0, 0, 255]);
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
    let depth = Image::<D32Sfloat, DepthAttachment>::new(UVec2::new(SIZE, SIZE)).unwrap();

    // Far (written), then near with the given write flag, then a triangle in between.
    let draw = |write_near: bool| {
        render(&gpu, |cmd, target| {
            for (vertices, clear, write) in [
                (&far, Some([0.0]), true),
                (&near, None, write_near),
                (&middle, None, true),
            ] {
                cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
                    .depth_attachment(depth.whole_view(), clear)
                    .draw(&Draw::new(
                        &bindings(vertices),
                        3,
                        1,
                        RasterState::default().depth_write(write),
                        None,
                    ))
                    .record("triangle");
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
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .depth_attachment(depth.whole_view(), Some([0.0]))
            .draw(&Draw::new(
                &bindings(&near),
                3,
                1,
                RasterState::default(),
                None,
            ))
            .record("near");
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .depth_attachment(depth.whole_view(), None)
            .draw(&Draw::new(
                &bindings(&far),
                3,
                1,
                RasterState::default().depth_write(false),
                None,
            ))
            .record("far");
    });
    assert_pixel(&occluded, 32, 40, [0, 255, 0, 255]);
}

#[test]
fn depth_clear_takes_effect_on_a_read_only_attachment() {
    let gpu = gpu();
    let vertices = overlapping_triangles();
    let depth = Image::<D32Sfloat, DepthAttachment>::new(UVec2::new(SIZE, SIZE)).unwrap();
    let pixels = render(&gpu, |cmd, target| {
        // Near triangle written, then depth cleared by a read-only pass, then the far one.
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .depth_attachment(depth.whole_view(), Some([0.0]))
            .draw(&Draw::new(
                &TestRaster::push_bindings(vertices.range(0..3), Vec2::ZERO),
                3,
                1,
                RasterState::default(),
                None,
            ))
            .record("near");
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .depth_attachment(depth.whole_view(), Some([0.0]))
            .draw(&Draw::new(
                &TestRaster::push_bindings(vertices.range(0..3), Vec2::ZERO),
                3,
                1,
                RasterState::default().depth_write(false),
                None,
            ))
            .record("read-only");
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .depth_attachment(depth.whole_view(), None)
            .draw(&Draw::new(
                &TestRaster::push_bindings(vertices.range(3..), Vec2::ZERO),
                3,
                1,
                RasterState::default(),
                None,
            ))
            .record("far");
    });
    assert_pixel(&pixels, 32, 40, [255, 0, 0, 255]);
}

/// Regression: pipelines for depth-stencil formats were created without a stencil format.
#[test]
fn depth_stencil_formats_work_as_depth_attachment() {
    let gpu = gpu();
    let vertices = overlapping_triangles();
    let depth = Image::<D32SfloatS8Uint, DepthAttachment>::new(UVec2::new(SIZE, SIZE)).unwrap();
    let pixels = render(&gpu, |cmd, target| {
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .depth_attachment(depth.whole_view(), Some((0.0, 0)))
            .draw(&Draw::new(
                &bindings(&vertices),
                6,
                1,
                RasterState::default(),
                None,
            ))
            .record("depth stencil")
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
    let texture = Image::<R8G8B8A8Unorm, SampledTransferDst>::new(UVec2::new(2, 2))
        .unwrap()
        .bind_sampled().unwrap();

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
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::indexed(
                &// Sampler 0 is nearest-neighbour.
                TestRasterTextured::push_bindings(vertices.range(..), texture.whole_view(), 0),
                indices.range(..),
                1,
                RasterState::default(),
                None,
            ))
            .record("textured")
    });

    assert_pixel(&pixels, 24, 24, [255, 0, 0, 255]);
    assert_pixel(&pixels, 40, 24, [0, 255, 0, 255]);
    assert_pixel(&pixels, 24, 40, [0, 0, 255, 255]);
    assert_pixel(&pixels, 40, 40, [255, 255, 0, 255]);
    assert_pixel(&pixels, 8, 8, BLACK);
    assert_golden("raster_textured", &pixels, [SIZE, SIZE]);

    // Only the first three indices: one triangle, the lower-left half stays empty.
    let half = render(&gpu, |cmd, target| {
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::indexed(
                &TestRasterTextured::push_bindings(vertices.range(..), texture.whole_view(), 0),
                indices.range(..3),
                1,
                RasterState::default(),
                None,
            ))
            .record("half")
    });
    assert_pixel(&half, 40, 24, [0, 255, 0, 255]);
    assert_pixel(&half, 24, 40, BLACK);
}

/// Images created unbound and bound later with one batch are sampled from their slots.
#[test]
fn bound_textures_are_sampled_from_their_slots() {
    let gpu = gpu();
    let colors = [
        [255, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
        [255, 255, 0, 255],
    ];
    let textures: Vec<_> = (0..colors.len())
        .map(|_| {
            Image::<R8G8B8A8Unorm, SampledTransferDst>::new(UVec2::new(1, 1))
                .unwrap()
                .bind_sampled().unwrap()
        })
        .collect();
    let uploads: Vec<_> = colors
        .iter()
        .map(|color| buffer_with::<u8, Storage>(color))
        .collect();

    // One quad per texture, in the four quadrants.
    let quads: Vec<_> = (0..4)
        .map(|i| {
            let (x, y) = ((i % 2) as f32 - 1.0, (i / 2) as f32 - 1.0);
            let corner = |dx: f32, dy: f32| TestTexturedVertex {
                position: Vec2::new(x + dx, y + dy),
                uv: Vec2::splat(0.5),
            };
            buffer_with::<TestTexturedVertex, Storage>(&[
                corner(0.0, 0.0),
                corner(1.0, 0.0),
                corner(1.0, 1.0),
                corner(0.0, 1.0),
            ])
        })
        .collect();
    let indices = buffer_with::<u32, Index>(&[0, 1, 2, 0, 2, 3]);

    let pixels = render(&gpu, |cmd, target| {
        for (upload, texture) in uploads.iter().zip(&textures) {
            cmd.copy_buffer_to_image(upload.range(..), texture.whole());
        }
        for (quad, texture) in quads.iter().zip(&textures) {
            cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
                .draw(&Draw::indexed(
                    &TestRasterTextured::push_bindings(quad.range(..), texture.whole_view(), 0),
                    indices.range(..),
                    1,
                    RasterState::default(),
                    None,
                ))
                .record("quad")
        }
    });

    assert_pixel(&pixels, 16, 16, colors[0]);
    assert_pixel(&pixels, 48, 16, colors[1]);
    assert_pixel(&pixels, 16, 48, colors[2]);
    assert_pixel(&pixels, 48, 48, colors[3]);
    assert_golden("bindless_batch", &pixels, [SIZE, SIZE]);
}

/// One mip of a chain is drawn to while another is sampled: their layouts are tracked apart,
/// which the validation layer checks.
#[test]
fn a_mip_is_drawn_while_another_is_sampled() {
    let gpu = gpu();
    let chain =
        MipChain::<R8G8B8A8Unorm, ColorAttachmentSampledTransferSrcTransferDst>::with_mip_levels(
            UVec2::new(16, 16),
            2,
        )
        .unwrap()
        .bind_mips_sampled().unwrap();
    let corner = |x: f32, y: f32| TestTexturedVertex {
        position: Vec2::new(x, y),
        uv: Vec2::splat(0.5),
    };
    let quad = buffer_with::<TestTexturedVertex, Storage>(&[
        corner(-1.0, -1.0),
        corner(1.0, -1.0),
        corner(1.0, 1.0),
        corner(-1.0, 1.0),
    ]);
    let indices = buffer_with::<u32, Index>(&[0, 1, 2, 0, 2, 3]);
    let readback = zeroed_buffer::<u8, Storage>(8 * 8 * 4);

    gpu.submit(|cmd| {
        cmd.clear_image(chain.mip_view(0), [0.0, 1.0, 0.0, 1.0]);
        cmd.raster(
            chain.mip_extent(1),
            [Attachment::color(chain.mip_view(1), Some([0.0; 4]))],
        )
        .draw(&Draw::indexed(
            &TestRasterTextured::push_bindings(quad.range(..), chain.mip_view(0), 0),
            indices.range(..),
            1,
            RasterState::default(),
            None,
        ))
        .record("mip");
        cmd.copy_image_to_buffer(chain.mip_view(1).whole(), readback.range(..));
    });
    assert_eq!(readback.host(), [0, 255, 0, 255].repeat(8 * 8));
}

/// A 3D image is uploaded, read by a shader's `Tex3D`, cleared per mip and copied back.
#[test]
fn volumes_are_uploaded_sampled_and_copied() {
    let gpu = gpu();
    let size = UVec3::new(4, 4, 2);
    let texel = |x: u32, y: u32, z: u32| [x as u8 * 60, y as u8 * 60, z as u8 * 200, 255];
    let texels: Vec<u8> = (0..size.z)
        .flat_map(|z| (0..size.y).flat_map(move |y| (0..size.x).map(move |x| (x, y, z))))
        .flat_map(|(x, y, z)| texel(x, y, z))
        .collect();
    let upload = buffer_with::<u8, Storage>(&texels);
    let volume =
        MipChain::<R8G8B8A8Unorm, SampledHostCopyTransferSrcTransferDst, D3>::with_mip_levels(
            size, 2,
        )
        .unwrap()
        .bind_sampled().unwrap();
    assert_eq!(volume.mip_extent(1), UVec3::new(2, 2, 1));
    let target = Image::<R8G8B8A8Unorm, StorageTransferSrcTransferDst>::new(size.truncate())
        .unwrap()
        .bind_storage().unwrap();
    let whole = zeroed_buffer::<u8, Storage>(texels.len());
    let slice = zeroed_buffer::<u8, Storage>(4 * 4 * 4);
    let mip = zeroed_buffer::<u8, Storage>(2 * 2 * 4);

    gpu.submit(|cmd| {
        cmd.copy_buffer_to_image(upload.range(..), volume.mip_view(0).whole());
        cmd.compute(
            &TestComputeVolume::push_bindings(volume.whole_view(), target.whole_view(), 1),
            [1, 1, 1],
            None,
        );
        cmd.copy_image_to_buffer(target.whole(), slice.range(..));
        cmd.clear_image(volume.mip_view(1), [1.0, 0.0, 1.0, 1.0]);
        cmd.copy_image_to_buffer(volume.whole(), whole.range(..));
        cmd.copy_image_to_buffer(volume.mip_view(1).whole(), mip.range(..));
    });
    assert_eq!(whole.host(), texels);
    assert_eq!(slice.host(), &texels[texels.len() / 2..]);
    assert_eq!(mip.host(), [255, 0, 255, 255].repeat(4));

    // A host copy into the second slice: its offset is three-dimensional.
    let mut volume = volume;
    if Ctx::features().host_image_copy {
        let patch = [7u8; 4 * 4 * 4];
        let offset = UVec3::new(0, 0, 1);
        volume
            .copy_region_from(&patch, 0, offset, UVec3::new(4, 4, 1))
            .unwrap();
        assert!(
            volume
                .copy_region_from(&patch, 0, offset, UVec3::new(4, 4, 2))
                .is_err()
        );
        gpu.submit(|cmd| cmd.copy_image_to_buffer(volume.whole(), whole.range(..)));
        let half = texels.len() / 2;
        assert_eq!(&whole.host()[..half], &texels[..half]);
        assert_eq!(&whole.host()[half..], patch);
    }
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
    if !Ctx::features().host_image_copy {
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
    let mut texture = Image::<BC7UnormBlock, SampledHostCopy>::with_mip_levels(UVec2::new(8, 8), 4)
        .unwrap()
        .bind_sampled().unwrap();
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
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::indexed(
                &TestRasterTextured::push_bindings(vertices.range(..), texture.whole_view(), 0),
                indices.range(..),
                1,
                RasterState::default(),
                None,
            ))
            .record("compressed")
    });
    assert_pixel(&pixels, 24, 24, [255, 1, 1, 255]);
    assert_pixel(&pixels, 40, 24, [1, 255, 1, 255]);
    assert_pixel(&pixels, 24, 40, [1, 1, 255, 255]);
    assert_pixel(&pixels, 40, 40, [255, 1, 1, 255]);
}

#[test]
fn mesh_pass_emits_triangles_per_workgroup() {
    let gpu = gpu();
    if !Ctx::features().mesh_shader {
        eprintln!("SKIPPED mesh_pass_emits_triangles_per_workgroup: no mesh shader support");
        return;
    }
    let pixels = render(&gpu, |cmd, target| {
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::launch(
                &TestMesh::push_bindings(Vec4::new(0.8, 0.8, 0.8, 1.0), Vec2::new(1.0, 0.0), 0.4).unwrap(),
                [2, 1, 1],
                unculled(),
                None,
            ))
            .record("mesh")
    });
    // Workgroup 0 draws around the centre, workgroup 1 shifted to the right edge.
    assert_ne!(pixel(&pixels, 32, 36), BLACK);
    assert_ne!(pixel(&pixels, 60, 36), BLACK);
    assert_pixel(&pixels, 4, 4, BLACK);
    assert_golden("mesh", &pixels, [SIZE, SIZE]);
}

#[test]
fn indirect_mesh_launch_matches_the_direct_one() {
    let gpu = gpu();
    if !Ctx::features().mesh_shader {
        eprintln!("SKIPPED indirect_mesh_launch_matches_the_direct_one: no mesh shader support");
        return;
    }
    let bindings =
        || TestMesh::push_bindings(Vec4::new(0.8, 0.8, 0.8, 1.0), Vec2::new(1.0, 0.0), 0.4).unwrap();
    let direct = render(&gpu, |cmd, target| {
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::launch(&bindings(), [2, 1, 1], unculled(), None))
            .record("direct")
    });
    let launch = buffer_with::<DispatchIndirectCommand, Indirect>(&[DispatchIndirectCommand {
        x: 2,
        y: 1,
        z: 1,
    }]);
    let indirect = render(&gpu, |cmd, target| {
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::launch_indirect(
                &bindings(),
                launch.range(..),
                unculled(),
                None,
            ))
            .record("indirect")
    });
    assert_eq!(indirect, direct);
}

// ---- generated bindings ----------------------------------------------------------------------

mod generated {
    use super::*;
    use bytemuck::Pod;
    use lava::{AccessFlags2 as A, ImageLayout as L, PipelineStageFlags2 as S, pipeline::PassKind};
    use lava_tests::bindings::{
        CTestComputeBufferBindings, CTestComputeImageBindings, CTestRasterBindings,
    };

    fn push_constants<G: Pod, K>(bindings: &BindingOutput<'_, K>) -> G {
        bytemuck::pod_read_unaligned(&bindings.sections().push_constants[..size_of::<G>()])
    }

    #[test]
    fn test_passes_are_registered_with_their_kind() {
        let _gpu = gpu();
        let buffer = zeroed_buffer::<u32, Storage>(1);
        let vertices = zeroed_buffer::<TestVertex, Storage>(1);
        let compute = TestComputeBuffer::push_bindings(buffer.range(..), buffer.range(..), 1, 1)
            .sections()
            .pass;
        assert!(matches!(
            PASS_MAP.passes[compute].kind,
            PassKind::Compute {
                entry: "test_compute_buffer\0"
            }
        ));
        let raster = TestRaster::push_bindings(vertices.range(..), Vec2::ZERO)
            .sections()
            .pass;
        assert!(matches!(
            PASS_MAP.passes[raster].kind,
            PassKind::RasterVertex {
                vertex: "test_raster_vertex\0",
                fragment: "test_raster_fragment\0"
            }
        ));
        assert_eq!(PASS_MAP.passes[raster].name, "TestRaster");
    }

    #[test]
    fn push_constants_are_the_struct_at_its_size() {
        let _gpu = gpu();
        let (src, dst) = (
            zeroed_buffer::<u32, Storage>(16),
            zeroed_buffer::<u32, Storage>(16),
        );
        let bindings = TestComputeBuffer::push_bindings(src.range(..), dst.range(..), 16, 3);
        let size = size_of::<CTestComputeBufferBindings>();
        assert_eq!(bindings.sections().push_constants.len(), size);
    }

    #[test]
    fn buffer_fields_become_device_addresses_and_accesses() {
        let _gpu = gpu();
        let (src, dst) = (
            zeroed_buffer::<u32, Storage>(16),
            zeroed_buffer::<u32, Storage>(16),
        );
        let bindings = TestComputeBuffer::push_bindings(src.range(..), dst.range(..), 16, 3);
        let constants: CTestComputeBufferBindings = push_constants(&bindings);
        assert_eq!(constants.src, src.range(..).address());
        assert_eq!(constants.dst, dst.range(..).address());
        assert_eq!((constants.count, constants.multiplier), (16, 3));

        let [src_access, dst_access]: [_; 2] = bindings
            .sections()
            .buffers()
            .collect::<Vec<_>>()
            .try_into()
            .expect("two buffers");
        // `Buf` is read-only, `MutBuf` is read-write.
        assert_eq!(src_access.access, A::SHADER_STORAGE_READ);
        assert_eq!(
            dst_access.access,
            A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE
        );
        assert_eq!(src_access.stage, S::COMPUTE_SHADER);
        assert_eq!(src_access.range, src.range(..).get_range());
        assert_eq!(dst_access.range, dst.range(..).get_range());
    }

    #[test]
    fn raster_passes_access_their_buffers_in_both_shader_stages() {
        let _gpu = gpu();
        let vertices = zeroed_buffer::<TestVertex, Storage>(3);
        let bindings = TestRaster::push_bindings(vertices.range(..), Vec2::new(0.5, 0.0));
        let constants: CTestRasterBindings = push_constants(&bindings);
        assert_eq!(constants.vertices, vertices.range(..).address());
        let [vertices]: [_; 1] = bindings
            .sections()
            .buffers()
            .collect::<Vec<_>>()
            .try_into()
            .expect("one buffer");
        assert_eq!(vertices.stage, S::VERTEX_SHADER | S::FRAGMENT_SHADER);
        assert_eq!(vertices.access, A::SHADER_STORAGE_READ);
    }

    /// The accesses a command takes besides its bindings get barriers like the bindings'
    /// (the second dispatch's read waits for the first one's write); the validation layer
    /// checks them.
    #[test]
    fn extra_accesses_get_barriers_like_the_bindings() {
        let gpu = gpu();
        let image = Image::<R8G8B8A8Unorm, StorageImage>::new(UVec2::new(8, 8)).unwrap();
        let buffer = zeroed_buffer::<u32, Storage>(4);
        gpu.submit(|cmd| {
            let bindings = TestComputeImage::push_bindings(
                image.whole_view(),
                Vec4::ONE,
                UVec2::ZERO,
                UVec2::new(8, 8),
            );
            let images = [ImageUse::storage_read(image.whole_view())];
            let written = [BufferAccess::write(buffer.range(..))];
            cmd.compute(&bindings, [1, 1, 1], Some(Accesses::new(&images, &written)));
            let read = [BufferAccess::read(buffer.range(..))];
            cmd.compute(&bindings, [1, 1, 1], Some(Accesses::buffers(&read)));
        });
    }

    #[test]
    fn image_fields_become_bindless_handles_and_general_layout_accesses() {
        let _gpu = gpu();
        let image = Image::<R8G8B8A8Unorm, StorageImage>::new(UVec2::new(8, 8))
            .unwrap()
            .bind_storage().unwrap();
        let target = image.whole_view();
        let bindings =
            TestComputeImage::push_bindings(target, Vec4::ONE, UVec2::ZERO, UVec2::new(8, 8));
        let constants: CTestComputeImageBindings = push_constants(&bindings);
        assert_eq!(constants.target, image.handle());
        assert_ne!(constants.target.descriptor_index_set1, NULL_HANDLE);
        let [image]: [_; 1] = bindings
            .sections()
            .images()
            .collect::<Vec<_>>()
            .try_into()
            .expect("one image");
        let access = &image.access;
        assert_eq!(
            access.access,
            A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE
        );
        assert_eq!(access.stage, S::COMPUTE_SHADER);
        assert_eq!(access.layout, L::GENERAL);
    }
}

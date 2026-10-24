//! Headless GPU tests: buffers (memory locations, host access, uploads), images, sync, command recording, queries, and golden images per pass type
mod common;

use common::{Cmd, Gpu, buffer_with, golden::assert_golden, gpu, nearest, panic_message, zeroed_buffer};
use glam::{IVec2, UVec2, UVec3, UVec4, Vec2, Vec4};
use lava::{
    bindless::{self, Combined, ImageHandle, Sampler, SamplerDesc, SamplerHandle, max_image_slots},
    buffer::{Buffer, Dynamic, Location, Memory, Readback, Shared, Upload, UploadStats},
    command_buffer::{
        Accesses, Attachment, BindingOutput, Blend, BlendEquation, BlendFactor, BlendOp,
        BufferAccess, ColorComponentFlags, CompareOp, CullModeFlags,
        DispatchIndirectCommand, Draw, DrawIndirectCommand, FrontFace, ImageUse, LogicOp,
        PolygonMode, PrimitiveTopology, RasterState, Rect, Stencil, StencilFace, StencilOp, kind,
    },
    image::{
        D3, Image, MipChain,
        format::{BC7UnormBlock, D32Sfloat, D32SfloatS8Uint, R8G8B8A8Unorm, R32Sfloat},
        slice::Filter,
        usage::usage,
    },
    query::{
        CounterScope, Occlusion, Performance, PipelineStatistics, PipelineStats,
        QueryPipelineStatisticFlags as Statistics, QueryPool, QueryRecorder, ScopeInfo,
        ScopeObserver, Timestamp, counters,
    },
    queue::{Compute, Graphics, Kind, MAX_TIMELINES, GraphicsStage, Timeline, Transfer},
    state::Ctx,
};
use lava_tests::bindings::{
    PASS_MAP, TestClock, TestClocks, TestCombined, TestComputeBuffer, TestComputeImage, TestComputeVolume,
    TestConstants, TestMesh, TestPushData, TestRaster, TestRasterTextured, TestSample,
    TestSampleDepth, TestTexturedVertex, TestVertex,
};

// ---- helpers ---------------------------------------------------------------------------------

const SIZE: u32 = 64;
const EXTENT: UVec2 = UVec2::splat(SIZE);
const BLACK: [u8; 4] = [0, 0, 0, 255];

type Target = Image<R8G8B8A8Unorm, usage![ColorAttachment, Storage, TransferSrc, TransferDst]>;

/// Clears a fresh target to opaque black, lets `record` draw into it, and reads it back.
fn render(gpu: &mut common::Gpu, record: impl FnOnce(&mut Cmd, &Target)) -> Vec<u8> {
    let target = Target::new(UVec2::new(SIZE, SIZE)).unwrap().bind_storage().unwrap();
    let mut readback = zeroed_buffer::<u8>((SIZE * SIZE * 4) as usize);
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
fn two_triangles(gpu: &mut Gpu) -> Buffer<TestVertex> {
    gpu.buffer(&[
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
    let buffer = Buffer::<[u32; 3]>::new(5).unwrap();
    assert_eq!(buffer.len(), 5);
    assert_eq!(buffer.size(), 60);
    assert_eq!(buffer.range(..).len(), 5);
    assert_ne!(buffer.address(), 0, "buffers are addressable from shaders");
    assert_eq!(buffer.range(2..).address(), buffer.address() + 24);
}

#[test]
fn host_writes_are_visible_through_slices_and_copies() {
    let _gpu = gpu();
    let mut buffer = Buffer::<u32, Upload>::upload(8).unwrap();
    buffer.host_mut().copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
    buffer.host_mut()[0] = 10;
    buffer.host_mut()[4..6].copy_from_slice(&[50, 60]);

    assert_eq!(buffer.host(), [10, 2, 3, 4, 50, 60, 7, 8]);
    assert_eq!(buffer.get(5), 60);
    let mut out = [0; 3];
    buffer.read_into(&mut out);
    assert_eq!(out, [10, 2, 3]);
    assert_eq!(buffer.byte_range(4..8).cast::<u32>().len(), 1);

    let halves = buffer.cast::<[u32; 2]>();
    assert_eq!(halves.len(), 4);
    assert_eq!(halves.get(2), [50, 60]);
    assert_eq!(halves.erase().location(), Location::Upload);
}

#[test]
#[should_panic(expected = "element 4 of a buffer of 4")]
fn reading_past_the_buffer_panics() {
    let _gpu = gpu();
    let buffer = Buffer::<u32, Readback>::readback(4).unwrap();
    let _ = buffer.get(4);
}

// ---- memory locations, host access and uploads ------------------------------------------------

/// `init`, `from_slice` and `write` land in their buffers: through the timeline's ring where
/// it has one, else (a `rebar` device, where every `Gpu` buffer is mapped) in place, with
/// nothing to drain.
#[test]
fn uploaded_buffers_round_trip_through_a_readback_copy() {
    let mut gpu = gpu();
    let uploader = gpu.uploader();
    let data: Vec<u32> = (0..1000).collect();
    let buffer = uploader.from_slice(&data).unwrap();
    assert_eq!(buffer.location(), Location::Gpu);
    let addresses = uploader.init(3, |address, out: &mut [u64]| out.fill(address)).unwrap();
    let written = Buffer::<u32>::new(4).unwrap();
    uploader.write(&written, 1, &[7, 8]).unwrap();
    let rebar = Ctx::features().rebar;
    if rebar {
        assert!(!uploader.stages() && uploader.capacity() == 0, "no ring on a rebar device");
        assert_eq!(uploader.pending(), 0, "written in place");
    }
    let (mut readback, mut address_out) = (zeroed_buffer::<u32>(data.len()), zeroed_buffer::<u64>(3));
    let mut written_out = zeroed_buffer::<u32>(4);
    let mut stats = UploadStats::default();
    gpu.submit(|cmd| {
        stats = cmd.drain_uploads();
        cmd.copy_buffer(buffer.range(..), readback.range(..));
        cmd.copy_buffer(addresses.range(..), address_out.range(..));
        cmd.copy_buffer(written.range(..), written_out.range(..));
    });
    match rebar {
        true => assert_eq!(stats, UploadStats::default(), "a drain without a ring is a no-op"),
        false => assert!(stats.copies >= 1 && stats.capacity == uploader.capacity(), "{stats:?}"),
    }
    assert_eq!(readback.host(), data);
    assert_eq!(address_out.host(), [addresses.address(); 3]);
    assert_eq!(written_out.host()[1..3], [7, 8]);
    assert_eq!(uploader.pending(), 0);
    assert!(panic_message(|| uploader.write(&written, 3, &[0, 0])).contains("writing 2 elements at 3"));
}

/// Staged image uploads need a ring: a `rebar` device has none (`Image::copy_from` uploads
/// there); with one, the texture is drawn once the ring is drained.
#[test]
fn image_uploads_land_where_the_ring_is_drained() {
    let mut gpu = gpu();
    let uploader = gpu.uploader();
    // The texture of `indexed_draw_samples_a_bindless_texture`.
    let texels: Vec<u8> = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [255, 255, 0, 255]].concat();
    let texture = Image::<R8G8B8A8Unorm, usage![Sampled, TransferDst]>::new(UVec2::new(2, 2))
        .unwrap()
        .bind_sampled()
        .unwrap();
    if !uploader.stages() {
        let message = uploader.image(&texture, 0, &texels).unwrap_err().to_string();
        assert!(message.contains("Image::copy_from"), "{message}");
        return;
    }
    let message = panic_message(|| uploader.image(&texture, 0, &texels[..12]));
    assert!(message.contains("holds"), "one whole mip: {message}");
    let message = panic_message(|| uploader.image(&texture, 1, &texels));
    assert!(message.contains("out of range"), "it has one mip: {message}");
    uploader.image(&texture, 0, &texels).unwrap();
    // Dropped with an upload staged, which goes with it.
    let dropped = Image::<R8G8B8A8Unorm, usage![TransferDst]>::new(UVec2::new(2, 2)).unwrap();
    uploader.image(&dropped, 0, &texels).unwrap();
    drop(dropped);
    assert_eq!(uploader.pending(), 1);

    let corner = |x: f32, y: f32, u: f32, v: f32| TestTexturedVertex {
        position: Vec2::new(x, y),
        uv: Vec2::new(u, v),
    };
    let vertices = uploader
        .from_slice(&[
            corner(-0.5, -0.5, 0.0, 0.0),
            corner(0.5, -0.5, 1.0, 0.0),
            corner(0.5, 0.5, 1.0, 1.0),
            corner(-0.5, 0.5, 0.0, 1.0),
        ])
        .unwrap();
    let indices = uploader.from_slice(&[0u32, 1, 2, 0, 2, 3]).unwrap();
    let pixels = render(&mut gpu, |cmd, target| {
        cmd.drain_uploads();
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::indexed(
                &TestRasterTextured::push_bindings(vertices.range(..), texture.whole_view(), nearest()),
                indices.range(..),
                1,
                RasterState::default(),
                None,
            ))
            .record("textured")
    });
    assert_golden("raster_textured", &pixels, [SIZE, SIZE]);
}

#[test]
fn host_access_waits_for_gpu_writes_in_flight() {
    let mut gpu = gpu();
    // Long enough to still run when the host looks: 512 MiB of fills, then a copy after them.
    let large = Buffer::<u32>::new(16 << 20).unwrap();
    let mut readback = zeroed_buffer::<u32>(4);
    let submission = gpu.timeline.begin().unwrap();
    let done = submission
        .submit(|cmd| {
            for value in 0..8 {
                cmd.fill_buffer(large.range(..), value);
            }
            cmd.copy_buffer(large.range(..4), readback.range(..));
            // Not submitted yet: the host may look, and the borrow keeps it from being
            // recorded again.
            assert_eq!(readback.host(), [0; 4]);
        })
        .unwrap();
    assert!(readback.try_host().is_none());
    let message = panic_message(|| readback.host_mut()[0] = 1);
    assert!(message.contains("the GPU may still write this buffer"), "{message}");
    // Reads of other buffers don't wait.
    let mut other = zeroed_buffer::<u32>(4);
    assert_eq!(other.host(), [0; 4]);
    done.block().unwrap();
    assert_eq!(readback.try_host(), Some(&[7; 4][..]));
}

#[test]
fn dynamic_locations_check_host_access_at_runtime() {
    let _gpu = gpu();
    let mut gpu_only = Buffer::<u32>::new(4).unwrap().erase();
    assert_eq!(gpu_only.memory(), Dynamic(Location::Gpu));
    let message = panic_message(|| gpu_only.host().len());
    assert!(message.contains("can't access a buffer of Gpu memory"), "{message}");
    let mut either = Buffer::<u32, Dynamic>::with_memory(Dynamic(Location::shared_or_upload()), 4).unwrap();
    either.host_mut().copy_from_slice(&[1, 2, 3, 4]);
    assert_eq!(either.get(3), 4);
}

#[test]
fn shared_memory_is_device_local_and_mapped() {
    let mut gpu = gpu();
    if !Ctx::features().rebar {
        let shared = Buffer::<u32, Shared>::shared(4);
        assert!(matches!(shared, Err(lava::Error::MissingFeature(m)) if m.feature == "rebar"));
        eprintln!("SKIPPED shared_memory_is_device_local_and_mapped: no rebar");
        return;
    }
    let mut shared = Buffer::<u32, Shared>::shared(4).unwrap();
    shared.host_mut().copy_from_slice(&[1, 2, 3, 4]);
    gpu.submit(|cmd| cmd.fill_buffer(shared.range(2..), 9));
    assert_eq!(shared.host(), [1, 2, 9, 9]);
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
                let mut buffer = zeroed_buffer::<u32>(4);
                let mut timeline = Timeline::<Graphics>::new(2, 0).unwrap();
                barrier.wait();
                let submission = timeline.begin().unwrap();
                let done = submission
                    .submit(|cmd| cmd.fill_buffer(buffer.range(..), value))
                    .unwrap();
                done.block().unwrap();
                assert_eq!(buffer.host(), [value; 4]);
            });
        }
    });
}

/// Also when dropped while recording or right after submitting, else validation fails.
#[test]
fn frame_slots_cycle_and_what_a_submission_used_lives_until_it_is_done() {
    let mut gpu = gpu();
    let mut buffer = zeroed_buffer::<u32>(4);
    let mut slots = Vec::new();
    let mut last = None;
    for value in 1..=4 {
        let scratch = zeroed_buffer::<u32>(1024);
        let image =
            Image::<R8G8B8A8Unorm, usage![Storage, TransferSrc, TransferDst]>::new(UVec2::new(64, 64)).unwrap();
        let recording = zeroed_buffer::<u32>(4);
        let buffer = &buffer;
        let submission = gpu.timeline.begin().unwrap();
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

/// Its barriers are submitted instead, so the next submission finds the tracked layout.
#[test]
fn a_panicking_recording_still_makes_its_layout_transitions() {
    let mut gpu = gpu();
    let image =
        Image::<R8G8B8A8Unorm, usage![Storage, TransferSrc, TransferDst]>::new(UVec2::new(8, 8)).unwrap();
    let readback = zeroed_buffer::<u8>(8 * 8 * 4);
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

/// Destructors lock the pending accesses a recording holds.
#[test]
fn a_recording_thread_runs_no_destructors() {
    let mut gpu = gpu();
    let image =
        Image::<R8G8B8A8Unorm, usage![Storage, TransferSrc, TransferDst]>::new(UVec2::new(8, 8)).unwrap();
    drop(image);
    let mut buffer = zeroed_buffer::<u32>(4);
    gpu.submit(|cmd| {
        lava::reclaim();
        cmd.fill_buffer(buffer.range(..), 1);
    });
    assert_eq!(buffer.host(), [1; 4]);
}

#[test]
fn a_submission_waits_for_another_timelines() {
    let mut gpu = gpu();
    let mut other = Timeline::<Graphics>::new(2, 0).unwrap();
    let (src, mut dst) = (zeroed_buffer::<u32>(4), zeroed_buffer::<u32>(4));
    let own = gpu.submit(|_| {});
    // Neither submission is waited for on the host before the second is made.
    let filled = (other.begin().unwrap())
        .submit(|cmd| cmd.fill_buffer(src.range(..), 9))
        .unwrap();
    let mut copy = gpu.timeline.begin().unwrap();
    copy.wait_for(filled.at(GraphicsStage::Transfer));
    // A wait on its own timeline is dropped: its submissions run in order anyway.
    copy.wait_for(own);
    let copied = copy
        .submit(|cmd| cmd.copy_buffer(src.range(..), dst.range(..)))
        .unwrap();
    copied.block().unwrap();
    assert!(filled.is_done());
    assert_eq!(dst.host(), [9; 4]);
}

/// What is dropped after another timeline reserved lives until that one submitted and finished.
#[test]
fn deletions_wait_for_every_timeline() {
    let mut gpu = gpu();
    let mut other = Timeline::<Graphics>::new(2, 0).unwrap();
    lava::wait_idle().unwrap();
    let bind = || {
        Image::<R8G8B8A8Unorm, usage![Sampled]>::new(UVec2::new(4, 4))
            .unwrap()
            .bind_sampled()
            .unwrap()
    };
    other.reserve();
    let image = bind();
    let slot = image.handle::<bindless::Sampled>();
    drop(image);
    gpu.submit(|_| {});
    let unrelated = bind();
    assert_ne!(unrelated.handle::<bindless::Sampled>(), slot, "the other timeline may still read it");
    other.begin().unwrap().submit(|_| {}).unwrap().block().unwrap();
    assert_eq!(bind().handle::<bindless::Sampled>(), slot);
}

/// A dropped timeline's semaphore goes on in the next, so an old `Wait` stays done.
#[test]
fn timelines_are_limited_and_reused() {
    let _gpu = gpu();
    let mut timelines = Vec::new();
    let error = loop {
        match Timeline::<Graphics>::new(1, 0) {
            Ok(timeline) => timelines.push(timeline),
            Err(error) => break error,
        }
    };
    assert!(matches!(error, lava::Error::OutOfTimelines), "{error}");
    assert!(timelines.len() < MAX_TIMELINES, "the harness holds one");
    let mut last = timelines.pop().unwrap();
    let done = last.begin().unwrap().submit(|_| {}).unwrap();
    drop(last);
    let mut reused = Timeline::<Graphics>::new(2, 0).unwrap();
    assert!(done.is_done());
    reused.begin().unwrap().submit(|_| {}).unwrap().block().unwrap();
}

// ---- queue family ownership ------------------------------------------------------------------

fn has_family(test: &str, kind: Kind) -> bool {
    let has = Ctx::queue_family(kind).is_some();
    if !has {
        eprintln!("SKIPPED {test}: the device has no dedicated {kind:?} family");
    }
    has
}

type Owned = Image<R8G8B8A8Unorm, usage![Storage, TransferSrc, TransferDst]>;

#[test]
fn a_compute_image_is_released_to_graphics() {
    let mut gpu = gpu();
    if !has_family("a_compute_image_is_released_to_graphics", Kind::Compute) {
        return;
    }
    let mut compute = Timeline::<Compute>::new(2, 0).unwrap();
    let image = Owned::new(UVec2::splat(8)).unwrap().bind_storage().unwrap();
    let red = Vec4::new(1.0, 0.0, 0.0, 1.0);
    let written = (compute.begin().unwrap())
        .submit(|cmd| {
            let bindings = TestComputeImage::push_bindings(image.whole_view(), red, UVec2::ZERO, UVec2::splat(8));
            cmd.compute(&bindings, [1, 1, 1], None);
            cmd.release::<Graphics>(image.whole_view());
        })
        .unwrap();
    let mut readback = zeroed_buffer::<u8>(8 * 8 * 4);
    gpu.submit_after(&[written], |cmd| {
        cmd.copy_image_to_buffer(image.whole(), readback.range(..))
    });
    // Red is the horizontal UV.
    for (i, texel) in readback.host().chunks_exact(4).enumerate() {
        let x = (i % 8) as f32;
        let red = ((x + 0.5) / 8.0 * 255.0).round() as u8;
        assert!(close(texel.try_into().unwrap(), [red, 0, 0, 255]), "texel {i}: {texel:?}");
    }
}

/// The copy engine uploads an image for graphics, which hands it back for a readback.
#[test]
fn images_go_to_the_transfer_family_and_back() {
    let mut gpu = gpu();
    if !has_family("images_go_to_the_transfer_family_and_back", Kind::Transfer) {
        return;
    }
    let mut transfer = Timeline::<Transfer>::new(2, 0).unwrap();
    let image = Owned::new(UVec2::splat(8)).unwrap();
    let upload = buffer_with::<u8>(&pattern(8, 8));
    let uploaded = (transfer.begin().unwrap())
        .submit(|cmd| {
            cmd.copy_buffer_to_image(upload.range(..), image.whole());
            cmd.release::<Graphics>(image.whole_view());
        })
        .unwrap();
    let mut readback = zeroed_buffer::<u8>(8 * 8 * 4);
    let drawn = gpu.submit_after(&[uploaded], |cmd| {
        cmd.copy_image_to_buffer(image.whole(), readback.range(..));
        cmd.clear_image(image.whole_view(), [0.0, 1.0, 0.0, 1.0]);
        cmd.release::<Transfer>(image.whole_view());
    });
    assert_eq!(readback.host(), pattern(8, 8));
    let mut back = transfer.begin().unwrap();
    back.wait_for(drawn);
    let back = back.submit(|cmd| cmd.copy_image_to_buffer(image.whole(), readback.range(..)));
    back.unwrap().block().unwrap();
    assert!(readback.host().chunks_exact(4).all(|t| t == [0, 255, 0, 255]));
}

#[test]
fn an_unreleased_image_panics() {
    let mut gpu = gpu();
    if !has_family("an_unreleased_image_panics", Kind::Compute) {
        return;
    }
    let mut compute = Timeline::<Compute>::new(2, 0).unwrap();
    let image = Owned::new(UVec2::splat(8)).unwrap();
    let cleared = (compute.begin().unwrap())
        .submit(|cmd| cmd.clear_image(image.whole_view(), [1.0; 4]))
        .unwrap();
    cleared.block().unwrap();
    let readback = zeroed_buffer::<u8>(8 * 8 * 4);
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        gpu.submit(|cmd| cmd.copy_image_to_buffer(image.whole(), readback.range(..)))
    }));
    let message = *panicked.unwrap_err().downcast::<String>().unwrap();
    assert!(message.contains("`cmd.release::<Graphics>(view)`"), "{message}");
}

/// Compute statistics count on a compute family (graphics statistics don't compile there, see
/// `lava::query::RecordedOn`).
#[test]
fn compute_statistics_count_on_a_compute_family() {
    let _gpu = gpu();
    if !has_family("compute_statistics_count_on_a_compute_family", Kind::Compute) {
        return;
    }
    if !Ctx::features().pipeline_statistics {
        eprintln!("SKIPPED compute_statistics_count_on_a_compute_family: no pipeline statistics");
        return;
    }
    let mut compute = Timeline::<Compute>::new(2, 0).unwrap();
    let target = Target::new(EXTENT).unwrap().bind_storage().unwrap();
    let pool = QueryPool::compute_statistics(1).unwrap();
    let counted = (compute.begin().unwrap()).submit(|cmd| {
        cmd.query(&pool, 0, |cmd| {
            let bindings = TestComputeImage::push_bindings(target.whole_view(), Vec4::ONE, UVec2::ZERO, EXTENT);
            cmd.compute(&bindings, [SIZE / 8, SIZE / 8, 1], None);
        })
    });
    counted.unwrap().block().unwrap();
    let mut invocations = [0];
    assert!(pool.read(0, &mut invocations).unwrap());
    assert_eq!(invocations[0], (SIZE * SIZE) as u64);
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

fn submit_profiled(gpu: &mut common::Gpu, observer: &mut TestObserver, target: &Target) {
    gpu.submit(|cmd| cmd.measure(Some(observer), "frame", |cmd| record_profiled(cmd, target)));
}

/// Logs and times every scope, and gives every pass a statistics and a counter query where it can.
struct TestObserver {
    timestamps: QueryPool<Timestamp>,
    statistics: Option<QueryPool<PipelineStatistics>>,
    counters: Option<QueryPool<Performance>>,
    /// Name, depth, pass, and whether it entered.
    events: Vec<(&'static str, u16, bool, bool)>,
    scopes: usize,
    passes: Vec<(&'static str, Option<u32>, Option<u32>)>,
    /// Scope index and, for a pass, its index in `passes`.
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
    let mut gpu = gpu();
    let target = Target::new(UVec2::new(SIZE, SIZE)).unwrap().bind_storage().unwrap();
    let mut observer = TestObserver::new(Statistics::empty(), &[]);
    submit_profiled(&mut gpu, &mut observer, &target);
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
    let mut ticks = vec![0; 2 * observer.scopes];
    assert!(
        observer.timestamps.read(0, &mut ticks).unwrap(),
        "the recording finished"
    );
    let ns = |tick| observer.timestamps.elapsed_ns(ticks[0], tick);
    let times: Vec<(u64, u64)> = ticks.chunks(2).map(|t| (ns(t[0]), ns(t[1]))).collect();
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
    let mut gpu = gpu();
    let target = Target::new(UVec2::new(SIZE, SIZE)).unwrap().bind_storage().unwrap();
    let features = Ctx::features();
    let mesh = features.mesh_shader;
    // One counter always fits a single pass; a time counter can't be 0 for a pass. Observers
    // count passes, so only counters of command scope.
    let all = counters::counters().unwrap_or_default();
    let command = || all.iter().filter(|c| c.scope == CounterScope::Command);
    let counter = command()
        .find(|c| c.unit == counters::CounterUnit::Nanoseconds)
        .or(command().next());
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
    submit_profiled(&mut gpu, &mut observer, &target);

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
    let mut gpu = gpu();
    let target = Target::new(UVec2::new(SIZE, SIZE)).unwrap().bind_storage().unwrap();
    let mut observer = TestObserver::new(Statistics::empty(), &[]);
    gpu.submit(|cmd| {
        let measured = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cmd.measure(Some(&mut observer), "frame", |_| {
                panic!("the recording failed")
            })
        }));
        assert!(measured.is_err());
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
    let mut gpu = gpu();
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
    let mut gpu = gpu();
    let vertices = overlapping_triangles(&mut gpu);
    let depth = Image::<D32Sfloat, usage![DepthAttachment]>::new(UVec2::new(SIZE, SIZE)).unwrap();
    let precise = Ctx::features().precise_occlusion;
    let pool = match precise {
        true => QueryPool::precise_occlusion(3),
        false => QueryPool::occlusion(3),
    }
    .unwrap();
    render(&mut gpu, |cmd, target| {
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

/// Begins query `begin` of `pool` when the root scope is entered and ends `end` (if any) when it
/// is exited.
struct OcclusionObserver<'a> {
    pool: &'a QueryPool<Occlusion>,
    begin: u32,
    end: Option<u32>,
}

impl ScopeObserver for OcclusionObserver<'_> {
    fn on_enter(&mut self, queries: &mut QueryRecorder, scope: &ScopeInfo) {
        if scope.depth == 0 {
            queries.begin_query(self.pool, self.begin);
        }
    }

    fn on_exit(&mut self, queries: &mut QueryRecorder, scope: &ScopeInfo) {
        if let (0, Some(end)) = (scope.depth, self.end) {
            queries.end_query(self.pool, end);
        }
    }
}

#[test]
fn a_query_is_used_once_per_recording() {
    let mut gpu = gpu();
    let pool = QueryPool::timestamps(2).unwrap();
    let message = panic_message(|| {
        gpu.submit(|cmd| {
            cmd.write_timestamp(&pool, 0, lava::PipelineStageFlags2::NONE);
            cmd.write_timestamp(&pool, 0, lava::PipelineStageFlags2::ALL_COMMANDS);
        })
    });
    assert!(message.contains("used twice"), "{message}");
    lava::wait_idle().unwrap();
    let message = panic_message(|| {
        gpu.submit(|cmd| cmd.write_timestamp(&pool, 2, lava::PipelineStageFlags2::NONE))
    });
    assert!(message.contains("of a pool of 2"), "{message}");
    lava::wait_idle().unwrap();
    // Each recording resets the pool, so the next one takes the queries again.
    for _ in 0..2 {
        gpu.submit(|cmd| {
            cmd.write_timestamp(&pool, 0, lava::PipelineStageFlags2::NONE);
            cmd.write_timestamp(&pool, 1, lava::PipelineStageFlags2::ALL_COMMANDS);
        });
    }
    let mut ticks = [0; 2];
    assert!(pool.read(0, &mut ticks).unwrap());
    // A query counts samples for one draw.
    let occlusion = QueryPool::occlusion(1).unwrap();
    let message = panic_message(|| {
        gpu.submit(|cmd| {
            cmd.query(&occlusion, 0, |_| {});
            cmd.query(&occlusion, 0, |_| {});
        })
    });
    assert!(message.contains("used twice"), "{message}");
}

#[test]
fn queries_end_where_they_began_and_before_the_recording_ends() {
    let mut gpu = gpu();
    let pool = QueryPool::occlusion(2).unwrap();
    let other = QueryPool::occlusion(1).unwrap();
    let mut wrong_index = OcclusionObserver { pool: &pool, begin: 0, end: Some(1) };
    let message = panic_message(|| {
        gpu.submit(|cmd| cmd.measure(Some(&mut wrong_index), "frame", |_| {}))
    });
    assert!(message.contains("query 1 of the pool is ended while query 0"), "{message}");
    // The failed recording's replay still holds the pool until it finished.
    lava::wait_idle().unwrap();
    let message = panic_message(|| {
        gpu.submit(|cmd| {
            cmd.query(&pool, 0, |cmd| {
                let mut wrong_pool = OcclusionObserver { pool: &other, begin: 0, end: None };
                cmd.measure(Some(&mut wrong_pool), "frame", |_| {})
            })
        })
    });
    assert!(message.contains("is active"), "{message}");
    lava::wait_idle().unwrap();
    let mut open = OcclusionObserver { pool: &pool, begin: 0, end: None };
    let message = panic_message(|| gpu.submit(|cmd| cmd.measure(Some(&mut open), "frame", |_| {})));
    assert!(message.contains("query 0 still active"), "{message}");
    lava::wait_idle().unwrap();
    // The failed recordings were replaced by their barriers; the next one records as usual.
    let mut balanced = OcclusionObserver { pool: &pool, begin: 1, end: Some(1) };
    gpu.submit(|cmd| cmd.measure(Some(&mut balanced), "frame", |_| {}));
    let mut samples = [7];
    assert!(pool.read(1, &mut samples).unwrap());
    assert_eq!(samples[0], 0);
}

#[test]
fn reads_stay_inside_their_pool() {
    let _gpu = gpu();
    let pool = QueryPool::timestamps(4).unwrap();
    let message = panic_message(|| pool.read(u32::MAX, &mut [0; 2]));
    assert!(message.contains("of a pool of 4"), "{message}");
    let message = panic_message(|| pool.read(1, &mut [0; 4]));
    assert!(message.contains("of a pool of 4"), "{message}");
    assert!(pool.read(4, &mut []).unwrap(), "no queries are always ready");
}

/// Counters of one scope, at most `MAX_COUNTERS`.
fn counters_of(scope: CounterScope) -> Vec<&'static counters::Counter> {
    let all = counters::counters().unwrap_or_default();
    all.iter().filter(|c| c.scope == scope).take(counters::MAX_COUNTERS).collect()
}

#[test]
fn counter_selections_fit_one_pass_or_need_a_capture() {
    let mut gpu = gpu();
    let all = counters::counters().unwrap_or_default();
    if all.is_empty() {
        eprintln!("SKIPPED counter_selections_fit_one_pass_or_need_a_capture: no performance queries");
        return;
    }
    assert!(!counters::fits(&[]), "an empty selection makes no pool");
    assert!(counters::fits(&[&all[0]]));
    let target = Target::new(EXTENT).unwrap().bind_storage().unwrap();
    let dispatch = |cmd: &mut Cmd| {
        let bindings = TestComputeImage::push_bindings(target.whole_view(), Vec4::ONE, UVec2::ZERO, EXTENT);
        cmd.compute(&bindings, [SIZE / 8, SIZE / 8, 1], None);
    };

    // Counters of command-buffer scope count whole recordings, which any counter can.
    let whole = counters_of(CounterScope::CommandBuffer);
    let counter = whole.first().copied().unwrap_or(&all[0]);
    let pool = QueryPool::performance(1, &[counter]).unwrap();
    if whole.is_empty() {
        eprintln!("SKIPPED counters of command-buffer scope in a query: the device has none");
    } else {
        let message = panic_message(|| gpu.submit(|cmd| cmd.query(&pool, 0, |cmd| dispatch(cmd))));
        assert!(message.contains("submit_profiled"), "{message}");
        lava::wait_idle().unwrap();
    }
    let submission = gpu.timeline.begin().unwrap();
    let wait = submission.submit_profiled(&pool, 0, |cmd| dispatch(cmd)).unwrap();
    wait.block().unwrap();
    let mut value = [f64::NAN];
    assert!(pool.read(0, &mut value).unwrap());
    assert!(value[0].is_finite(), "{}: {}", counter.name, value[0]);
    drop(pool);

    // Two counters that need more than one pass, if the device has such.
    let pair = (all.iter().skip(1))
        .map(|other| vec![&all[0], other])
        .find(|pair| counters::passes(pair) > 1);
    let Some(selection) = pair else {
        eprintln!("SKIPPED multi-pass capture: every counter fits one pass with the first");
        return;
    };
    let passes = counters::passes(&selection);
    assert!(!counters::fits(&selection));
    let pool = QueryPool::performance(1, &selection).unwrap();
    assert_eq!(pool.passes(), passes);
    let message = panic_message(|| gpu.submit(|cmd| cmd.query(&pool, 0, |cmd| dispatch(cmd))));
    assert!(message.contains("capture"), "{message}");
    lava::wait_idle().unwrap();
    let mut recorded = 0;
    let wait = gpu.timeline.capture(&pool, 0, |cmd| {
        recorded += 1;
        dispatch(cmd)
    });
    wait.unwrap().block().unwrap();
    assert_eq!(recorded, passes, "one recording per pass");
    let mut values = vec![f64::NAN; selection.len()];
    assert!(pool.read(0, &mut values).unwrap());
    for (value, counter) in values.iter().zip(&selection) {
        assert!(value.is_finite(), "{}: {value}", counter.name);
    }
}

// ---- buffer commands -------------------------------------------------------------------------

#[test]
fn fill_buffer_fills_exactly_the_slice() {
    let mut gpu = gpu();
    let mut buffer = zeroed_buffer::<u32>(8);
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
fn update_buffer_writes_its_elements() {
    let mut gpu = gpu();
    let mut buffer = zeroed_buffer::<[u32; 2]>(4);
    gpu.submit(|cmd| cmd.update_buffer(buffer.range(1..), &[[11, 22], [33, 44]]));
    assert_eq!(buffer.host(), [[0, 0], [11, 22], [33, 44], [0, 0]]);
    // Empty data records nothing; more than the slice holds or 64 KiB panics.
    gpu.submit(|cmd| cmd.update_buffer(buffer.range(..), &[]));
    let message = panic_message(|| gpu.submit(|cmd| cmd.update_buffer(buffer.range(3..), &[[1, 2]; 2])));
    assert!(message.contains("update of 2 elements into a slice of 1"), "{message}");
    let large = zeroed_buffer::<u32>(32768);
    let message = panic_message(|| gpu.submit(|cmd| cmd.update_buffer(large.range(..), &[0; 16385])));
    assert!(message.contains("at most 65536"), "{message}");
    gpu.submit(|cmd| cmd.update_buffer(large.range(..), &[7; 16384]));
    assert_eq!(buffer.host()[1], [11, 22]);
}

// ---- argument checks -------------------------------------------------------------------------

#[test]
fn empty_and_oversized_buffers_and_images_are_errors() {
    let _gpu = gpu();
    assert!(Buffer::<u32>::new(0).is_err());
    assert!(Buffer::<u32>::new(usize::MAX).is_err());
    type Texture = Image<R8G8B8A8Unorm, usage![Sampled]>;
    assert!(Texture::new(UVec2::new(0, 4)).is_err());
    assert!(Texture::with_mip_levels(UVec2::new(4, 4), 0).is_err());
    // 4x4 halves into 4x4, 2x2 and 1x1.
    assert!(Texture::with_mip_levels(UVec2::new(4, 4), 3).is_ok());
    assert!(Texture::with_mip_levels(UVec2::new(4, 4), 4).is_err());
    let max = Ctx::device_properties().limits.max_image_dimension2_d;
    assert!(Texture::new(UVec2::new(max + 1, 1)).is_err());
}

#[test]
fn casts_keep_whole_aligned_elements() {
    let _gpu = gpu();
    let buffer = zeroed_buffer::<u8>(16);
    let message = panic_message(|| buffer.byte_range(0..6).cast::<u32>());
    assert!(message.contains("whole number"), "{message}");
    let message = panic_message(|| buffer.byte_range(1..5).cast::<u32>());
    assert!(message.contains("aligned"), "{message}");
    assert_eq!(buffer.byte_range(4..12).cast::<u32>().len(), 2);
}

#[test]
fn fills_and_updates_write_aligned_words() {
    let mut gpu = gpu();
    let bytes = zeroed_buffer::<u8>(16);
    let message = panic_message(|| gpu.submit(|cmd| cmd.fill_buffer(bytes.byte_range(1..5), 0)));
    assert!(message.contains("4-byte"), "{message}");
    let message = panic_message(|| gpu.submit(|cmd| cmd.fill_buffer(bytes.byte_range(0..6), 0)));
    assert!(message.contains("4-byte"), "{message}");
    let triples = zeroed_buffer::<[u8; 3]>(4);
    let message = panic_message(|| gpu.submit(|cmd| cmd.update_buffer(triples.range(..), &[[1; 3]])));
    assert!(message.contains("4-byte"), "{message}");
    // Empty fills and updates record nothing (Vulkan refuses a size of 0).
    gpu.submit(|cmd| {
        cmd.fill_buffer(bytes.byte_range(4..4), 0);
        cmd.update_buffer(bytes.range(..0), &[]);
    });
}

#[test]
fn copies_and_blits_stay_inside_their_mip() {
    let mut gpu = gpu();
    let image = Image::<R8G8B8A8Unorm, usage![Sampled, TransferSrc, TransferDst]>::new(UVec2::new(8, 8)).unwrap();
    let buffer = zeroed_buffer::<u8>(8 * 8 * 4);
    let view = image.whole_view();
    let outside = view.region(UVec2::new(4, 4)).offset(IVec2::new(6, 0));
    let message = panic_message(|| {
        gpu.submit(|cmd| cmd.copy_buffer_to_image(buffer.range(..), outside))
    });
    assert!(message.contains("outside its mip"), "{message}");
    let message = panic_message(|| {
        gpu.submit(|cmd| cmd.copy_image_to_buffer(view.region(UVec2::new(0, 4)), buffer.range(..)))
    });
    assert!(message.contains("empty"), "{message}");
    let message = panic_message(|| {
        gpu.submit(|cmd| cmd.blit_image(image.whole(), outside, Filter::Nearest))
    });
    assert!(message.contains("outside its mip"), "{message}");
    let negative = view.region(UVec2::new(4, 4)).offset(IVec2::new(-1, 0));
    let message = panic_message(|| {
        gpu.submit(|cmd| cmd.blit_image(negative, image.whole(), Filter::Nearest))
    });
    assert!(message.contains("outside its mip"), "{message}");
}

#[test]
fn block_compressed_copies_cover_whole_blocks() {
    let mut gpu = gpu();
    if !Ctx::features().bc_compression {
        eprintln!("SKIPPED block_compressed_copies_cover_whole_blocks: no BC compression");
        return;
    }
    let image = Image::<BC7UnormBlock, usage![Sampled, TransferDst]>::new(UVec2::new(8, 8)).unwrap();
    let buffer = zeroed_buffer::<u8>(64);
    let view = image.whole_view();
    let message = panic_message(|| {
        let half_block = view.region(UVec2::new(2, 4));
        gpu.submit(|cmd| cmd.copy_buffer_to_image(buffer.range(..), half_block))
    });
    assert!(message.contains("whole 4x4 blocks"), "{message}");
    gpu.submit(|cmd| cmd.copy_buffer_to_image(buffer.range(..), view.region(UVec2::new(4, 4))));
}

#[test]
fn dispatches_stay_within_the_device_limits() {
    let mut gpu = gpu();
    let target = Target::new(EXTENT).unwrap().bind_storage().unwrap();
    let max = Ctx::device_properties().limits.max_compute_work_group_count;
    let Some(over) = max[1].checked_add(1) else {
        eprintln!("SKIPPED dispatches_stay_within_the_device_limits: no group count is too many");
        return;
    };
    let message = panic_message(|| {
        gpu.submit(|cmd| {
            let bindings =
                TestComputeImage::push_bindings(target.whole_view(), Vec4::ONE, UVec2::ZERO, EXTENT);
            cmd.compute(&bindings, [1, over, 1], None)
        })
    });
    assert!(message.contains("exceed"), "{message}");
}

#[test]
fn shader_images_are_bound_in_the_sets_they_are_read_through() {
    let _gpu = gpu();
    let image = Image::<R8G8B8A8Unorm, usage![Sampled, Storage]>::new(UVec2::new(8, 8))
        .unwrap()
        .bind_sampled()
        .unwrap();
    let message = panic_message(|| {
        TestComputeImage::push_bindings(image.whole_view(), Vec4::ONE, UVec2::ZERO, UVec2::ONE)
    });
    assert!(message.contains("bind_storage"), "{message}");
}

#[test]
fn attachments_are_one_mip_at_least_as_large_as_the_rendering() {
    let mut gpu = gpu();
    let mips = Target::with_mip_levels(EXTENT, 2).unwrap();
    let message = panic_message(|| Attachment::color(mips.whole_view(), None));
    assert!(message.contains("one mip"), "{message}");
    let target = Target::new(EXTENT).unwrap();
    let message = panic_message(|| {
        gpu.submit(|cmd| drop(cmd.raster(EXTENT * 2, [Attachment::color(target.whole_view(), None)])))
    });
    assert!(message.contains("doesn't fit its color attachment"), "{message}");
    let depth = Image::<D32Sfloat, usage![DepthAttachment]>::new(EXTENT / 2).unwrap();
    let message = panic_message(|| {
        gpu.submit(|cmd| {
            cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
                .depth_attachment(depth.whole_view(), None);
        })
    });
    assert!(message.contains("doesn't fit its depth attachment"), "{message}");
    let message = panic_message(|| {
        gpu.submit(|cmd| {
            cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
                .scissor(Rect::new(IVec2::new(-1, 0), EXTENT));
        })
    });
    assert!(message.contains("non-negative"), "{message}");
}

#[test]
fn draws_of_a_rendering_access_nothing_another_writes() {
    let mut gpu = gpu();
    let vertices = two_triangles(&mut gpu);
    let bindings = bindings(&vertices);
    let target = Target::new(EXTENT).unwrap();
    let written = zeroed_buffer::<u32>(4);
    let writes = [BufferAccess::write(written.range(..))];
    let reads = [BufferAccess::read(written.range(2..3))];
    // A draw may read what it writes itself, and other draws what nobody writes.
    gpu.submit(|cmd| {
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::new(&bindings, 6, 1, unculled(), Some(Accesses::buffers(&writes))))
            .draw(&Draw::new(&bindings, 6, 1, unculled(), None))
            .record("one writer");
    });
    let message = panic_message(|| {
        gpu.submit(|cmd| {
            cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
                .draw(&Draw::new(&bindings, 6, 1, unculled(), Some(Accesses::buffers(&writes))))
                .draw(&Draw::new(&bindings, 6, 1, unculled(), Some(Accesses::buffers(&reads))))
                .record("hazard");
        })
    });
    assert!(message.contains("draws 0 and 1"), "{message}");
    let storage = Image::<R8G8B8A8Unorm, usage![Storage]>::new(UVec2::new(8, 8)).unwrap();
    let images = [ImageUse::storage_write(storage.whole_view())];
    let message = panic_message(|| {
        gpu.submit(|cmd| {
            let draw = Draw::new(&bindings, 6, 1, unculled(), Some(Accesses::images(&images)));
            cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
                .draw(&draw)
                .draw(&draw)
                .record("twice");
        })
    });
    assert!(message.contains("of the image"), "{message}");
}

#[test]
fn copy_buffer_copies_the_source_slice_to_the_destination_offset() {
    let mut gpu = gpu();
    let src = buffer_with::<u32>(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let mut dst = zeroed_buffer::<u32>(8);
    gpu.submit(|cmd| cmd.copy_buffer(src.range(1..4), dst.range(5..)));
    assert_eq!(dst.host(), [0, 0, 0, 0, 0, 2, 3, 4]);
}

#[test]
fn copy_buffer_regions_copies_each_region() {
    let mut gpu = gpu();
    let src = buffer_with::<u32>(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let mut dst = zeroed_buffer::<u32>(8);
    let regions = [
        (src.range(0..2), dst.range(6..)),
        (src.range(7..8), dst.range(0..1)),
    ];
    gpu.submit(|cmd| cmd.copy_buffer_regions(&regions));
    assert_eq!(dst.host(), [8, 0, 0, 0, 0, 0, 1, 2]);
    // Empty regions are skipped, and a list of only those records nothing.
    gpu.submit(|cmd| cmd.copy_buffer_regions(&[(src.range(..0), dst.range(..))]));
    gpu.submit(|cmd| cmd.copy_buffer_regions(&[(src.range(..0), dst.range(..)), (src.range(..1), dst.range(1..2))]));
    assert_eq!(dst.host(), [8, 1, 0, 0, 0, 0, 1, 2]);
    // No destination may overlap another or a source.
    let message = panic_message(|| {
        gpu.submit(|cmd| cmd.copy_buffer_regions(&[(src.range(0..2), dst.range(0..2)), (src.range(4..6), dst.range(1..3))]))
    });
    assert!(message.contains("regions 0 and 1 overlap"), "{message}");
    let message = panic_message(|| gpu.submit(|cmd| cmd.copy_buffer(src.range(0..4), src.range(2..6))));
    assert!(message.contains("region 0 reads and writes"), "{message}");
}

#[test]
fn reserve_grows_a_buffer_and_keeps_its_contents() {
    let mut gpu = gpu();
    let mut buffer = gpu.buffer(&[1u32, 2, 3]);
    let mut readback = zeroed_buffer::<u32>(8);
    let mut grew = (false, false);
    gpu.submit(|cmd| {
        grew = (buffer.reserve(3, cmd).unwrap(), buffer.reserve(5, cmd).unwrap());
        cmd.fill_buffer(buffer.range(3..), 9);
        cmd.copy_buffer(buffer.range(..), readback.range(..buffer.len()));
    });
    assert_eq!(grew, (false, true));
    // At least twice the old length.
    assert_eq!(buffer.len(), 6);
    assert_eq!(&readback.host()[..6], [1, 2, 3, 9, 9, 9]);
}

#[test]
fn per_frame_values_belong_to_the_slot_a_submission_waited_for() {
    let mut gpu = gpu();
    let timeline = &mut gpu.timeline;
    let mut rings = lava::queue::PerFrame::new(timeline, |_| Buffer::<u32, Upload>::upload(4)).unwrap();
    assert_eq!(rings.len(), timeline.in_flight() as usize);
    let mut slots = Vec::new();
    for frame in 0..4u32 {
        let submission = timeline.begin().unwrap();
        slots.push(submission.slot());
        rings.host_mut(&submission).fill(frame);
        let ring = rings.get(&submission).range(..);
        let mut readback = zeroed_buffer::<u32>(4);
        let readback_slice = readback.range(..);
        submission.submit(|cmd| cmd.copy_buffer(ring, readback_slice)).unwrap().block().unwrap();
        assert_eq!(readback.host(), [frame; 4]);
    }
    // The harness timeline has two slots, which earlier tests' submissions left at either.
    assert!(slots.iter().enumerate().all(|(i, slot)| *slot == (slots[0] + i) % 2), "{slots:?}");
    let mut other = Timeline::<Graphics>::new(2, 0).unwrap();
    let submission = other.begin().unwrap();
    let message = panic_message(|| rings.get(&submission).len());
    assert!(message.contains("timeline it was made for"), "{message}");
    submission.submit(|_| {}).unwrap().block().unwrap();
}

#[test]
fn raw_buffers_hold_whole_elements_at_power_of_two_alignments() {
    let _gpu = gpu();
    let message = panic_message(|| Buffer::<u32, Upload>::raw(Upload, 6, None));
    assert!(message.contains("whole number"), "{message}");
    let message = panic_message(|| Buffer::<u8, Upload>::raw(Upload, 6, Some(48)));
    assert!(message.contains("power of two"), "{message}");
    let buffer = Buffer::<u8, Upload>::raw(Upload, 7, Some(1024)).unwrap();
    assert_eq!((buffer.len(), buffer.address() % 1024), (7, 0));
}

#[test]
fn structs_with_image_handles_are_written_but_not_zeroed() {
    let _gpu = gpu();
    let image = Image::<R8G8B8A8Unorm, usage![Storage]>::new(EXTENT).unwrap().bind_storage().unwrap();
    // A generated struct with a handle derives `NoUninit`, not `Pod`, so it has no `zeroed()`;
    // this one comes from the pass's bindings.
    let made = TestComputeImage::push_bindings(image.whole_view(), Vec4::ONE, UVec2::ZERO, EXTENT);
    let push_constants = made.sections().push_constants;
    assert_eq!(push_constants.len(), size_of::<lava_tests::bindings::CTestComputeImageBindings>());
    // SAFETY: the bytes are a `CTestComputeImageBindings`, which every field holds any bits of.
    let bindings: lava_tests::bindings::CTestComputeImageBindings =
        unsafe { std::ptr::read_unaligned(push_constants.as_ptr().cast()) };
    let mut buffer: Buffer<lava_tests::bindings::CTestComputeImageBindings, Upload> = Buffer::upload(2).unwrap();
    buffer.host_write(1, &[bindings]);
    let mut first = [0u8; 4];
    // SAFETY: element 1 was written, and nothing else reaches the mapping.
    unsafe { std::ptr::copy_nonoverlapping(buffer.host_ptr().add(1).cast::<u8>(), first.as_mut_ptr(), 4) };
    let handle = image.handle::<lava::bindless::Storage>();
    assert_eq!(first, bytemuck::bytes_of(&handle));
    assert_ne!(ImageHandle::<lava::bindless::Storage>::default(), handle);
    let message = panic_message(|| buffer.host_write(2, &[bindings]));
    assert!(message.contains("into a buffer of 2"), "{message}");
}

/// Synchronisation validation reports the missing barrier if the first write is forgotten.
#[test]
fn partially_overlapping_writes_are_synchronised_with_later_reads() {
    let mut gpu = gpu();
    let buffer = zeroed_buffer::<u32>(16);
    let mut copy = zeroed_buffer::<u32>(16);
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
    let image = Image::<R8G8B8A8Unorm, usage![Sampled]>::with_mip_levels(UVec2::new(16, 8), 4).unwrap();
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

/// Each sized like its mip and bound at its own slot, the mips' slots contiguous.
#[test]
fn mip_chains_have_a_view_and_handle_per_mip() {
    let _gpu = gpu();
    let chain = MipChain::<R8G8B8A8Unorm, usage![Sampled, Storage]>::with_mip_levels(UVec2::new(16, 8), 4)
        .unwrap()
        .bind_sampled()
        .unwrap()
        .bind_mips_storage()
        .unwrap();
    assert!(chain.handle::<bindless::Sampled>().sampled_slot().is_some());
    let unbound = panic_message(|| chain.handle::<bindless::Storage>());
    assert!(unbound.contains("bind_storage"), "only the mips are storage: {unbound}");
    let whole = chain.whole_view();
    assert_eq!(whole.subresource_range().level_count, 4);
    let first = chain.mip_view(0).handle::<bindless::Storage>();
    let slot = first.storage_slot().unwrap();
    for level in 0..4 {
        let view = chain.mip_view(level);
        let range = view.subresource_range();
        assert_eq!((range.base_mip_level, range.level_count), (level, 1));
        assert_eq!(view.extent(), chain.mip_extent(level));
        assert_eq!(view.handle::<bindless::Storage>(), first.mip(level));
        assert_eq!(view.handle::<bindless::Storage>().storage_slot(), Some(slot + level));
        // Descriptors are written without views: only attachments have them.
        assert_eq!(view.view(), lava::vk::ImageView::null());
    }
    assert_eq!(chain.mip_view(2).whole().extend, UVec2::new(4, 2));
    let targets = MipChain::<R8G8B8A8Unorm, usage![ColorAttachment, Sampled, TransferSrc, TransferDst]>::with_mip_levels(UVec2::new(16, 8), 2)
        .unwrap();
    let views = [targets.whole_view().view(), targets.mip_view(0).view(), targets.mip_view(1).view()];
    assert!(!views.contains(&lava::vk::ImageView::null()));
    assert!(views[0] != views[1] && views[1] != views[2]);
}

#[test]
fn binds_take_free_slots() {
    let _gpu = gpu();
    let sampled = Image::<R8G8B8A8Unorm, usage![Sampled]>::new(UVec2::new(4, 4))
        .unwrap()
        .bind_sampled()
        .unwrap();
    let storage = Image::<R8G8B8A8Unorm, usage![Storage]>::new(UVec2::new(4, 4))
        .unwrap()
        .bind_storage()
        .unwrap();
    let both = Image::<R8G8B8A8Unorm, usage![Sampled, Storage]>::new(UVec2::new(4, 4))
        .unwrap()
        .bind_combined()
        .unwrap();
    let unbound = Image::<R8G8B8A8Unorm, usage![Sampled]>::new(UVec2::new(4, 4)).unwrap();

    let s = sampled.handle::<bindless::Sampled>();
    let st = storage.handle::<bindless::Storage>();
    let b = both.handle::<Combined>();
    let s_slot = s.sampled_slot().unwrap();
    assert!(s_slot < max_image_slots());
    assert_eq!(s.storage_slot(), None);
    assert_eq!(st.sampled_slot(), None);
    let st_slot = st.storage_slot().unwrap();
    let pair = b.sampled_slot().unwrap();
    assert_eq!(b.storage_slot(), Some(pair + 1), "the storage descriptor follows the sampled one");
    let mut slots = vec![s_slot, st_slot, pair, pair + 1];
    slots.sort();
    slots.dedup();
    assert_eq!(slots.len(), 4, "one descriptor per slot");
    assert_eq!(both.handle::<bindless::Sampled>(), b.as_sampled());
    assert_eq!(both.handle::<bindless::Storage>(), b.as_storage());
    assert!(panic_message(|| unbound.handle::<bindless::Sampled>()).contains("bind_sampled"));
    assert!(panic_message(|| sampled.handle::<bindless::Storage>()).contains("bind_storage"));
    assert!(panic_message(|| sampled.handle::<Combined>()).contains("bind_combined"));
    let twice = panic_message(|| {
        Image::<R8G8B8A8Unorm, usage![Sampled, Storage]>::new(UVec2::new(4, 4))
            .unwrap()
            .bind_sampled()
            .unwrap()
            .bind_storage()
    });
    assert!(twice.contains("bind_combined"), "a view is bound once: {twice}");

    // `bind` writes reserved slots only and checks the usage at runtime, static types included.
    let reserved = bindless::reserve::<bindless::Sampled>(4).unwrap();
    assert_eq!((reserved.len(), reserved.slots().len()), (4, 4));
    assert!(!reserved.slots().contains(&s_slot));
    bindless::bind([unbound.whole_view(); 4], (0..4).map(|i| reserved.handle(i)));
    let unreserved = std::panic::catch_unwind(|| bindless::bind([unbound.whole_view()], [s]));
    assert!(unreserved.is_err(), "the sampled image's slot isn't reserved");
    let storage_slots = bindless::reserve::<bindless::Storage>(1).unwrap();
    let refused = std::panic::catch_unwind(|| {
        bindless::bind([unbound.whole_view()], [storage_slots.handle(0)])
    });
    assert!(refused.is_err(), "a sampled image can't be bound as storage");
    assert!(std::panic::catch_unwind(|| reserved.handle(4)).is_err());
    let pairs = bindless::reserve::<Combined>(2).unwrap();
    assert_eq!((pairs.len(), pairs.slots().len()), (2, 4));
    assert_eq!(pairs.handle(1).sampled_slot(), Some(pairs.slots().start + 2));
}

/// A pair takes the first two neighbouring free slots, past the holes single slots leave.
#[test]
fn pairs_skip_single_holes() {
    let _gpu = gpu();
    lava::wait_idle().unwrap();
    // Single slots until seven in a row: the holes other tests left are filled then.
    let mut taken = Vec::new();
    let in_a_row = |taken: &[bindless::SlotRange<bindless::Sampled>]| {
        taken.len() >= 7 && taken[taken.len() - 7..].windows(2).all(|w| w[1].slots().start == w[0].slots().end)
    };
    while !in_a_row(&taken) {
        taken.push(bindless::reserve::<bindless::Sampled>(1).unwrap());
    }
    let start = taken[taken.len() - 7].slots().start;
    // Every other one of the seven freed again: the holes are one slot each.
    let (kept, dropped): (Vec<_>, Vec<_>) = taken.into_iter().partition(|slot| {
        let slot = slot.slots().start;
        slot < start || (slot - start) % 2 == 0
    });
    drop(dropped);
    lava::wait_idle().unwrap();
    let pair = bindless::reserve::<Combined>(1).unwrap();
    assert!(pair.slots().start >= start + 7, "no hole fits a pair: {:?}", pair.slots());
    let single = bindless::reserve::<bindless::Storage>(1).unwrap();
    assert_eq!(single.slots().start, start + 1, "a single fills the first hole");
    drop(kept);
}

/// Also a dropped reservation's slots, which `bind` refuses right away.
#[test]
fn slots_are_free_again_once_their_image_is_destroyed() {
    let _gpu = gpu();
    // Destroys what earlier tests dropped, so the lowest free slots are the first ones.
    lava::wait_idle().unwrap();
    let bind = || {
        Image::<R8G8B8A8Unorm, usage![Sampled]>::new(UVec2::new(4, 4))
            .unwrap()
            .bind_sampled()
            .unwrap()
    };
    let image = bind();
    let slot = image.handle::<bindless::Sampled>();
    drop(image);
    assert_ne!(bind().handle::<bindless::Sampled>(), slot, "taken until no frame reaches it");
    lava::wait_idle().unwrap();
    assert_eq!(bind().handle::<bindless::Sampled>(), slot);

    // A mip chain frees its whole view's slot and each mip's, erased or not.
    lava::wait_idle().unwrap();
    let chain = MipChain::<R8G8B8A8Unorm, usage![Sampled, Storage]>::with_mip_levels(UVec2::new(4, 4), 3)
        .unwrap()
        .bind_sampled()
        .unwrap()
        .bind_mips_storage()
        .unwrap()
        .erase();
    assert_eq!(chain.format().0, lava::VkFormat::R8G8B8A8_UNORM);
    let (whole, mips) = (chain.handle::<bindless::Sampled>(), chain.mip_view(0).handle::<bindless::Storage>());
    drop(chain);
    lava::wait_idle().unwrap();
    assert_eq!(bind().handle::<bindless::Sampled>(), whole);
    let storage = bindless::reserve::<bindless::Storage>(3).unwrap();
    assert_eq!(storage.handle(0), mips, "the mips' slots came back whole");
    drop(storage);
    lava::wait_idle().unwrap();

    let texture = Image::<R8G8B8A8Unorm, usage![Sampled]>::new(UVec2::new(4, 4)).unwrap();
    let sampled = bindless::reserve::<bindless::Sampled>(3).unwrap();
    let (handle, slots) = (sampled.handle(0), sampled.slots());
    drop(sampled);
    let written = std::panic::catch_unwind(|| bindless::bind([texture.whole_view()], [handle]));
    assert!(written.is_err());
    lava::wait_idle().unwrap();
    assert_eq!(bindless::reserve::<bindless::Sampled>(3).unwrap().slots(), slots);
}

const NO_SAMPLER: SamplerHandle = SamplerHandle { index: !0 };

/// What `TestSample` reads at `uv` of the sampled image `slot` (a handle's bits) through `samplers`, the last a
/// comparison against `reference` unless it is `NO_SAMPLER`.
fn sample(
    gpu: &mut common::Gpu,
    slot: u32,
    [first, second, compare]: [SamplerHandle; 3],
    uv: Vec2,
    reference: f32,
    reads: Accesses,
) -> [Vec4; 3] {
    let mut out = zeroed_buffer::<Vec4>(3);
    gpu.submit(|cmd| {
        let bindings =
            TestSample::push_bindings(out.range(..), first, second, compare, slot, uv, reference);
        cmd.compute(&bindings, [1, 1, 1], Some(reads));
    });
    out.host().try_into().unwrap()
}

/// A 1x1 sampled image of `color`.
fn solid(gpu: &mut common::Gpu, color: [f32; 4]) -> Image<R8G8B8A8Unorm, usage![Sampled, TransferDst]> {
    let image = Image::new(UVec2::ONE).unwrap();
    gpu.submit(|cmd| cmd.clear_image(image.whole_view(), color));
    image
}

/// `bind` into a reserved slot reaches the heap in submission order: a frame still in flight
/// reads the old image, the next one the new image.
#[test]
fn reserved_slots_are_rewritten_between_frames_in_flight() {
    let mut gpu = gpu();
    let red = solid(&mut gpu, [1.0, 0.0, 0.0, 1.0]);
    let green = solid(&mut gpu, [0.0, 1.0, 0.0, 1.0]);
    let slot = bindless::reserve::<bindless::Sampled>(1).unwrap();
    let mut outs = [zeroed_buffer::<Vec4>(3), zeroed_buffer::<Vec4>(3)];
    let frame = |gpu: &mut common::Gpu, image: &Image<R8G8B8A8Unorm, usage![Sampled, TransferDst]>, out: &Buffer<Vec4, Readback>| {
        bindless::bind([image.whole_view()], [slot.handle(0)]);
        let reads = [ImageUse::sampled_read(image.whole_view())];
        let submission = gpu.timeline.begin().unwrap();
        submission
            .submit(|cmd| {
                let bindings = TestSample::push_bindings(
                    out.range(..),
                    nearest(),
                    nearest(),
                    NO_SAMPLER,
                    slot.handle(0).bits(),
                    Vec2::splat(0.5),
                    0.0,
                );
                cmd.compute(&bindings, [1, 1, 1], Some(Accesses::images(&reads)));
            })
            .unwrap()
    };
    let first = frame(&mut gpu, &red, &outs[0]);
    let second = frame(&mut gpu, &green, &outs[1]);
    second.block().unwrap();
    assert!(first.is_done());
    assert_eq!(outs[0].host()[0], Vec4::new(1.0, 0.0, 0.0, 1.0));
    assert_eq!(outs[1].host()[0], Vec4::new(0.0, 1.0, 0.0, 1.0));
}

/// Samplers are runtime objects: two in one pass, each with its filter, and a dropped one's
/// slot is taken until no frame reaches it.
#[test]
fn samplers_filter_by_their_description_and_free_their_slots() {
    let mut gpu = gpu();
    // Black, then white.
    let texels = buffer_with::<[u8; 4]>(&[[0, 0, 0, 255], [255, 255, 255, 255]]);
    let image = Image::<R8G8B8A8Unorm, usage![Sampled, TransferDst]>::new(UVec2::new(2, 1))
        .unwrap()
        .bind_sampled()
        .unwrap();
    gpu.submit(|cmd| cmd.copy_buffer_to_image(texels.range(..), image.whole_view().whole()));
    let linear = Sampler::new(
        &SamplerDesc::new().address_mode(lava::vk::SamplerAddressMode::CLAMP_TO_EDGE),
    )
    .unwrap();
    let reads = [ImageUse::sampled_read(image.whole_view())];
    // A quarter of the way from the first texel's centre to the second's.
    let uv = Vec2::new(0.375, 0.5);
    let slot = image.handle::<bindless::Sampled>().bits();
    let samplers = [nearest(), linear.handle(), NO_SAMPLER];
    let [near, filtered, _] = sample(&mut gpu, slot, samplers, uv, 0.0, Accesses::images(&reads));
    assert_eq!(near.x, 0.0);
    assert!((filtered.x - 0.25).abs() < 0.01, "linear filtering gave {}", filtered.x);

    if Ctx::features().sampler_anisotropy {
        let max = Ctx::device_properties().limits.max_sampler_anisotropy;
        for ratio in [0.5, max + 1.0] {
            panic_message(|| Sampler::new(&SamplerDesc::new().max_anisotropy(ratio)));
        }
        Sampler::new(&SamplerDesc::new().max_anisotropy(max)).unwrap();
    }

    lava::wait_idle().unwrap();
    let desc = SamplerDesc::new();
    let dropped = Sampler::new(&desc).unwrap();
    let index = dropped.handle();
    drop(dropped);
    let next = Sampler::new(&desc).unwrap();
    assert_ne!(next.handle(), index, "taken until no frame reaches it");
    lava::wait_idle().unwrap();
    assert_eq!(Sampler::new(&desc).unwrap().handle(), index);
}

/// A min or max reduction sampler returns the extreme of the texels it filters.
#[test]
fn reduction_samplers_return_the_extremes() {
    let mut gpu = gpu();
    if !Ctx::features().sampler_minmax {
        eprintln!("SKIPPED reduction_samplers_return_the_extremes: no sampler-minmax");
        return;
    }
    let texels = buffer_with::<f32>(&[0.1, 0.2, 0.3, 0.4]);
    let image = Image::<R32Sfloat, usage![Sampled, TransferDst]>::new(UVec2::new(2, 2))
        .unwrap()
        .bind_sampled()
        .unwrap();
    gpu.submit(|cmd| cmd.copy_buffer_to_image(texels.range(..), image.whole_view().whole()));
    let reduce = |mode| {
        let desc = SamplerDesc::new()
            .address_mode(lava::vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .reduction(mode);
        Sampler::new(&desc).unwrap()
    };
    use lava::vk::SamplerReductionMode as Mode;
    let (min, max) = (reduce(Mode::MIN), reduce(Mode::MAX));
    let reads = [ImageUse::sampled_read(image.whole_view())];
    let slot = image.handle::<bindless::Sampled>().bits();
    let samplers = [min.handle(), max.handle(), NO_SAMPLER];
    // The centre, where all four texels weigh the same.
    let [low, high, _] =
        sample(&mut gpu, slot, samplers, Vec2::splat(0.5), 0.0, Accesses::images(&reads));
    assert_eq!((low.x, high.x), (0.1, 0.4));
}

type Combinable = Image<R8G8B8A8Unorm, usage![Sampled, Storage, TransferSrc, TransferDst]>;
type CombinedChain = MipChain<R8G8B8A8Unorm, usage![Sampled, Storage, TransferDst]>;

/// A 4x4 combined image of 0.25 everywhere.
fn quarter(gpu: &mut common::Gpu) -> Combinable {
    let image = Image::new(UVec2::splat(4)).unwrap();
    gpu.submit(|cmd| cmd.clear_image(image.whole_view(), [0.25; 4]));
    image
}

/// A combined mip chain of `levels` mips, mip `i` cleared to `(i + 1) / 4`.
fn combined_chain(gpu: &mut common::Gpu, levels: u32) -> CombinedChain {
    let chain = CombinedChain::with_mip_levels(UVec2::splat(4), levels).unwrap().bind_mips_combined().unwrap();
    gpu.submit(|cmd| {
        for level in 0..levels {
            cmd.clear_image(chain.mip_view(level), [(level + 1) as f32 / 4.0; 4]);
        }
    });
    chain
}

/// What `TestCombined` writes: the samples of a sampled (red), a storage (green) and a null
/// `DynImg`, then texel (0, 0) of each mip of `chain`. It doubles `target` and triples `pair`'s
/// image through the handle given.
fn run_combined(
    gpu: &mut common::Gpu,
    target: &Combinable,
    pair: Option<(ImageHandle<Combined>, &Combinable)>,
    chain: &CombinedChain,
) -> Vec<Vec4> {
    let sampled = Image::<R8G8B8A8Unorm, usage![Sampled, TransferDst]>::new(UVec2::ONE).unwrap().bind_sampled().unwrap();
    let storage = Image::<R8G8B8A8Unorm, usage![Storage, TransferDst]>::new(UVec2::ONE).unwrap().bind_storage().unwrap();
    gpu.submit(|cmd| {
        cmd.clear_image(sampled.whole_view(), [1.0, 0.0, 0.0, 1.0]);
        cmd.clear_image(storage.whole_view(), [0.0, 1.0, 0.0, 1.0]);
    });
    let levels = chain.mip_levels();
    let mut out = zeroed_buffer::<Vec4>(3 + levels as usize);
    let mut reads = vec![
        ImageUse::sampled_read(sampled.whole_view()),
        ImageUse::storage_read(storage.whole_view()),
    ];
    reads.extend((1..levels).map(|level| ImageUse::sampled_read(chain.mip_view(level))));
    let both = lava::AccessFlags2::SHADER_SAMPLED_READ | lava::AccessFlags2::SHADER_STORAGE_WRITE;
    reads.extend(pair.map(|(_, image)| image.whole_view().access(lava::PipelineStageFlags2::NONE, both)));
    gpu.submit(|cmd| {
        let bindings = TestCombined::push_bindings(
            target.whole_view(),
            pair.map_or(!0, |(handle, _)| handle.bits()),
            chain.mip_view(0),
            sampled.handle::<bindless::Sampled>().erase(),
            storage.handle::<bindless::Storage>().erase(),
            ImageHandle::NULL,
            nearest(),
            out.range(..),
            levels,
        );
        cmd.compute(&bindings, [1, 1, 1], Some(Accesses::images(&reads)));
    });
    out.host().to_vec()
}

/// Every texel of a 4x4 image.
fn texels(gpu: &mut common::Gpu, image: &Combinable) -> Vec<[u8; 4]> {
    let mut readback = zeroed_buffer::<[u8; 4]>(16);
    gpu.submit(|cmd| cmd.copy_image_to_buffer(image.whole(), readback.range(..)));
    readback.host().to_vec()
}

/// A combined image reads through its sampled descriptor and writes through the storage one
/// in the next slot; a `DynImg` reads whichever descriptor its handle has, and zeroes for none.
#[test]
fn combined_images_read_sampled_and_write_storage() {
    let mut gpu = gpu();
    let target = quarter(&mut gpu).bind_combined().unwrap();
    let chain = combined_chain(&mut gpu, 1);
    let out = run_combined(&mut gpu, &target, None, &chain);
    for texel in texels(&mut gpu, &target) {
        assert!(close(texel, [128; 4]), "doubled: {texel:?}");
    }
    assert_eq!(out[0], Vec4::new(1.0, 0.0, 0.0, 1.0), "the sampled image");
    assert_eq!(out[1], Vec4::new(0.0, 1.0, 0.0, 1.0), "the storage image, nearest");
    assert_eq!(out[2], Vec4::ZERO, "the null handle");
}

/// `bind` writes both descriptors of a reserved pair.
#[test]
fn reserved_pairs_hold_both_descriptors() {
    let mut gpu = gpu();
    let target = quarter(&mut gpu).bind_combined().unwrap();
    let image = quarter(&mut gpu);
    let pairs = bindless::reserve::<Combined>(2).unwrap();
    bindless::bind([image.whole_view()], [pairs.handle(1)]);
    let chain = combined_chain(&mut gpu, 1);
    run_combined(&mut gpu, &target, Some((pairs.handle(1), &image)), &chain);
    for texel in texels(&mut gpu, &image) {
        assert!(close(texel, [192; 4]), "tripled: {texel:?}");
    }
}

/// `bind_mips_combined` puts each mip's pair after the last, and `Img.Mip` reads each mip.
#[test]
fn combined_mips_lie_a_pair_apart() {
    let mut gpu = gpu();
    let chain = combined_chain(&mut gpu, 3);
    let first = chain.mip_view(0).handle::<Combined>();
    let slot = first.sampled_slot().unwrap();
    for level in 0..3 {
        let handle = chain.mip_view(level).handle::<Combined>();
        assert_eq!(handle, first.mip(level));
        assert_eq!((handle.sampled_slot(), handle.storage_slot()), (Some(slot + 2 * level), Some(slot + 2 * level + 1)));
    }
    let target = quarter(&mut gpu).bind_combined().unwrap();
    let out = run_combined(&mut gpu, &target, None, &chain);
    for level in 0..3 {
        let expected = (level + 1) as f32 / 4.0;
        assert!((out[3 + level] - Vec4::splat(expected)).abs().max_element() < 0.01, "mip {level}: {}", out[3 + level]);
    }
}

/// A pass can push all of `MAX_PUSH_CONSTANTS_SIZE`.
#[test]
fn push_data_fills_all_256_bytes() {
    let mut gpu = gpu();
    const _: () = assert!(size_of::<lava_tests::bindings::CTestPushDataBindings>() == 256);
    let values: [UVec4; 15] = std::array::from_fn(|i| UVec4::splat(i as u32) + UVec4::new(0, 100, 200, 300));
    let last = UVec2::new(7, 11);
    let mut out = zeroed_buffer::<UVec4>(16);
    gpu.submit(|cmd| {
        cmd.compute(&TestPushData::push_bindings(values, out.range(..), last), [1, 1, 1], None)
    });
    assert_eq!(&out.host()[..15], &values);
    assert_eq!(out.host()[15], UVec4::new(7, 11, 0, 0));
}

#[test]
fn clear_image_sets_every_texel() {
    let mut gpu = gpu();
    let image =
        Image::<R8G8B8A8Unorm, usage![Storage, TransferSrc, TransferDst]>::new(UVec2::new(8, 8)).unwrap();
    let mut readback = zeroed_buffer::<u8>(8 * 8 * 4);
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
    let mut gpu = gpu();
    let data = pattern(8, 8);
    let upload = buffer_with::<u8>(&data);
    let image =
        Image::<R8G8B8A8Unorm, usage![Sampled, TransferSrc, TransferDst]>::new(UVec2::new(8, 8)).unwrap();
    let mut readback = zeroed_buffer::<u8>(data.len());
    gpu.submit(|cmd| {
        cmd.copy_buffer_to_image(upload.range(..), image.whole());
        cmd.copy_image_to_buffer(image.whole(), readback.range(..));
    });
    assert_eq!(readback.host(), data);
}

#[test]
fn image_region_copies_address_the_sub_rectangle() {
    let mut gpu = gpu();
    let data = pattern(8, 8);
    let upload = buffer_with::<u8>(&data);
    let image =
        Image::<R8G8B8A8Unorm, usage![Sampled, TransferSrc, TransferDst]>::new(UVec2::new(8, 8)).unwrap();
    let mut readback = zeroed_buffer::<u8>(3 * 2 * 4);
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

/// Regression: blits used the extent as the second corner, so offset regions were misplaced.
#[test]
fn blit_copies_between_offset_regions() {
    let mut gpu = gpu();
    let data = pattern(8, 8);
    let upload = buffer_with::<u8>(&data);
    let src = Image::<R8G8B8A8Unorm, usage![Sampled, TransferSrc, TransferDst]>::new(UVec2::new(8, 8)).unwrap();
    let dst = Image::<R8G8B8A8Unorm, usage![Sampled, TransferSrc, TransferDst]>::new(UVec2::new(8, 8)).unwrap();
    let mut readback = zeroed_buffer::<u8>(data.len());
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
    let mut gpu = gpu();
    // 2x2 source, each texel becomes a 4x4 block.
    let data: Vec<u8> = [
        [255, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
        [255, 255, 0, 255],
    ]
    .concat();
    let upload = buffer_with::<u8>(&data);
    let src = Image::<R8G8B8A8Unorm, usage![Sampled, TransferSrc, TransferDst]>::new(UVec2::new(2, 2)).unwrap();
    let dst = Image::<R8G8B8A8Unorm, usage![Sampled, TransferSrc, TransferDst]>::new(UVec2::new(8, 8)).unwrap();
    let mut readback = zeroed_buffer::<u8>(8 * 8 * 4);
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
    let mut gpu = gpu();
    let mut image =
        Image::<R8G8B8A8Unorm, usage![Sampled, HostCopy, TransferSrc]>::with_mip_levels(UVec2::new(8, 8), 2)
            .unwrap();
    let data = pattern(8, 8);

    if !Ctx::features().host_image_copy {
        // The device lacks `VK_EXT_host_image_copy`.
        assert!(image.copy_from(&data, 0).is_err());
        return;
    }

    image.copy_from(&data, 0).unwrap();
    let mut readback = zeroed_buffer::<u8>(data.len());
    gpu.submit(|cmd| cmd.copy_image_to_buffer(image.whole(), readback.range(..)));
    assert_eq!(readback.host(), data);

    image.copy_from(&pattern(4, 4), 1).unwrap();
    // Programmer errors panic.
    let message = panic_message(|| image.copy_from(&data, 2));
    assert!(message.contains("out of range"), "the image has two mip levels: {message}");
    panic_message(|| image.copy_from(&[], 0));

    // The data has to be exactly one mip level.
    panic_message(|| image.copy_from(&data[..data.len() - 4], 0));
    panic_message(|| image.copy_from(&data, 1));
    panic_message(|| image.copy_from(&pattern(4, 4), 0));
    // Rejected copies leave the image as it was.
    gpu.submit(|cmd| cmd.copy_image_to_buffer(image.whole(), readback.range(..)));
    assert_eq!(readback.host(), data);
}

#[test]
fn host_copy_writes_only_the_region() {
    let mut gpu = gpu();
    let mut image =
        Image::<R8G8B8A8Unorm, usage![Sampled, HostCopy, TransferSrc]>::new(UVec2::new(8, 8)).unwrap();
    let (offset, extent) = (UVec2::new(4, 5), UVec2::new(3, 2));
    let patch = vec![200u8; 3 * 2 * 4];

    if !Ctx::features().host_image_copy {
        assert!(image.copy_region_from(&patch, 0, offset, extent).is_err());
        return;
    }

    let data = pattern(8, 8);
    image.copy_from(&data, 0).unwrap();
    image.copy_region_from(&patch, 0, offset, extent).unwrap();

    panic_message(|| image.copy_region_from(&patch, 0, UVec2::new(6, 5), extent));
    panic_message(|| image.copy_region_from(&patch[4..], 0, offset, extent));

    let mut readback = zeroed_buffer::<u8>(data.len());
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
    let mut gpu = gpu();
    let pixels = render(&mut gpu, |cmd, target| {
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
    let mut gpu = gpu();
    let pixels = render(&mut gpu, |cmd, target| {
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
    let mut gpu = gpu();
    let direct = render(&mut gpu, |cmd, target| {
        compute_image_bindings(target, UVec2::ZERO, EXTENT)(cmd)
    });

    let dispatch = buffer_with::<DispatchIndirectCommand>(&[DispatchIndirectCommand {
        x: SIZE / 8,
        y: SIZE / 8,
        z: 1,
    }]);
    let indirect = render(&mut gpu, |cmd, target| {
        cmd.compute_indirect(
            &TestComputeImage::push_bindings(target.whole_view(), Vec4::ONE, UVec2::ZERO, EXTENT),
            dispatch.range(..),
            None,
        )
    });
    assert_eq!(indirect, direct);
    let image = Image::<R8G8B8A8Unorm, usage![Storage]>::new(EXTENT).unwrap().bind_storage().unwrap();
    let message = panic_message(|| {
        gpu.submit(|cmd| {
            let bindings = TestComputeImage::push_bindings(image.whole_view(), Vec4::ONE, UVec2::ZERO, EXTENT);
            cmd.compute_indirect(&bindings, dispatch.range(..0), None)
        })
    });
    assert!(message.contains("empty slice"), "{message}");
}

#[test]
fn compute_pass_reads_and_writes_buffers() {
    let mut gpu = gpu();
    let input: Vec<u32> = (1..=100).collect();
    let src = buffer_with::<u32>(&input);
    let mut dst = buffer_with::<u32>(&[u32::MAX; 100]);
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

/// `regions` lists the fields in order, `take` splits a counter into ticks and subgroups.
#[test]
fn clock_regions_count_every_subgroup_once() {
    let mut gpu = gpu();
    if !Ctx::features().shader_clock {
        eprintln!("skipping clock_regions_count_every_subgroup_once: no subgroup clocks");
        return;
    }
    const INVOCATIONS: usize = 256;
    let mut clocks = zeroed_buffer::<TestClocks>(1);
    let mut out = zeroed_buffer::<u32>(INVOCATIONS);
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

/// The generated bindings' accesses produce the barriers between the transfers and the pass.
#[test]
fn compute_pass_is_synchronised_with_surrounding_transfers() {
    let mut gpu = gpu();
    let src = zeroed_buffer::<u32>(64);
    let dst = zeroed_buffer::<u32>(64);
    let mut copy = zeroed_buffer::<u32>(64);
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

/// Each dispatch writes through the handles of its own constant set, with its own scalars.
#[test]
fn specialization_constants_pick_the_pipeline() {
    let mut gpu = gpu();
    let (mut a, mut b) = (
        zeroed_buffer::<u32>(8),
        zeroed_buffer::<u32>(8),
    );
    let (image_a, image_b) = (
        Target::new(UVec2::new(SIZE, SIZE)).unwrap().bind_storage().unwrap(),
        Target::new(UVec2::new(SIZE, SIZE)).unwrap().bind_storage().unwrap(),
    );
    let mut pixels = zeroed_buffer::<u8>((SIZE * SIZE * 4 * 2) as usize);
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

fn bindings<M: Memory>(
    vertices: &Buffer<TestVertex, M>,
) -> impl std::ops::Deref<Target = BindingOutput<'_, kind::RasterVertex>> {
    TestRaster::push_bindings(vertices.range(..), Vec2::ZERO)
}

fn unculled() -> RasterState<1> {
    RasterState::default().cull_mode(CullModeFlags::NONE)
}

/// Draws the first `vertex_count` vertices of [`two_triangles`] with backface culling on/off.
fn draw_triangles(gpu: &mut common::Gpu, vertex_count: u32, culling: bool) -> Vec<u8> {
    let vertices = two_triangles(gpu);
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
    let mut gpu = gpu();
    let pixels = draw_triangles(&mut gpu, 6, true);

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
    let mut gpu = gpu();
    let pixels = draw_triangles(&mut gpu, 6, false);
    assert_ne!(pixel(&pixels, IN_LEFT.0, IN_LEFT.1), BLACK);
    assert_pixel(&pixels, IN_RIGHT.0, IN_RIGHT.1, [255, 255, 0, 255]);
    assert_golden("raster_unculled", &pixels, [SIZE, SIZE]);
}

#[test]
fn wireframe_draws_only_the_edges() {
    let mut gpu = gpu();
    if !Ctx::features().fill_mode_non_solid {
        eprintln!("SKIPPED wireframe_draws_only_the_edges: the device lacks `fill-mode-non-solid`");
        return;
    }
    let vertices = two_triangles(&mut gpu);
    let pixels = render(&mut gpu, |cmd, target| {
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
    let mut gpu = gpu();
    let vertices = two_triangles(&mut gpu);
    let pixels = render(&mut gpu, |cmd, target| {
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
    let mut gpu = gpu();
    let vertices = two_triangles(&mut gpu);
    let pixels = render(&mut gpu, |cmd, target| {
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
    let mut gpu = gpu();
    let direct = draw_triangles(&mut gpu, 3, true);

    let vertices = two_triangles(&mut gpu);
    let commands = buffer_with::<DrawIndirectCommand>(&[DrawIndirectCommand {
        vertex_count: 3,
        instance_count: 1,
        first_vertex: 0,
        first_instance: 0,
    }]);
    let indirect = render(&mut gpu, |cmd, target| {
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
    let mut gpu = gpu();
    let vertices = two_triangles(&mut gpu);
    // Shaders see D3D-style vertex ids (no `first_vertex`), so the commands differ in count.
    let command = |vertex_count| DrawIndirectCommand {
        vertex_count,
        instance_count: 1,
        first_vertex: 0,
        first_instance: 0,
    };
    let commands = buffer_with::<DrawIndirectCommand>(&[command(3), command(6)]);

    let draw_counted = |gpu: &mut common::Gpu, count: u32| {
        let count = buffer_with::<u32>(&[count]);
        render(gpu, |cmd, target| {
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
    assert_eq!(draw_counted(&mut gpu, 2), draw_triangles(&mut gpu, 6, false));
    assert_eq!(draw_counted(&mut gpu, 1), draw_triangles(&mut gpu, 3, false));
    assert!(draw_counted(&mut gpu, 0).chunks_exact(4).all(|p| p == BLACK));
}

#[test]
fn scissor_clips_and_viewport_scales_the_draw() {
    let mut gpu = gpu();
    let vertices = two_triangles(&mut gpu);
    let full = draw_triangles(&mut gpu, 6, false);

    // Scissor: the full-size picture, cut to the left half.
    let scissored = render(&mut gpu, |cmd, target| {
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .scissor(Rect::new(IVec2::ZERO, UVec2::new(SIZE / 2, SIZE)))
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
    let quarter = render(&mut gpu, |cmd, target| {
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .viewport(Rect::new(IVec2::splat(SIZE as i32 / 2), EXTENT / 2))
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
fn overlapping_triangles(gpu: &mut Gpu) -> Buffer<TestVertex> {
    gpu.buffer(&[
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
    let mut gpu = gpu();
    let vertices = overlapping_triangles(&mut gpu);

    let without_depth = render(&mut gpu, |cmd, target| {
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

    let depth = Image::<D32Sfloat, usage![DepthAttachment]>::new(UVec2::new(SIZE, SIZE)).unwrap();
    let with_depth = render(&mut gpu, |cmd, target| {
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

    // A depth buffer only the rendering tests against needs no store, and a later depth
    // attachment replaces an earlier one, stencil aspect included (validation checks it).
    let stencil = Image::<D32SfloatS8Uint, usage![DepthAttachment]>::new(EXTENT).unwrap();
    let discarded = render(&mut gpu, |cmd, target| {
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .depth_attachment(stencil.whole_view(), Some((0.0, 0)))
            .depth_attachment(depth.whole_view(), Some([0.0]))
            .discard_depth_after()
            .draw(&Draw::new(&bindings(&vertices), 6, 1, RasterState::default(), None))
            .record("discarded depth")
    });
    assert_eq!(discarded, with_depth);
}

/// `DONT_CARE` loads and stores: a target the draw covers needn't be loaded, and a scratch
/// attachment needn't be stored.
#[test]
fn attachments_discard_their_contents() {
    let mut gpu = gpu();
    let chain =
        MipChain::<R8G8B8A8Unorm, usage![ColorAttachment, Sampled, TransferSrc, TransferDst]>::with_mip_levels(
            UVec2::new(16, 16),
            2,
        )
        .unwrap()
        .bind_mips_sampled()
        .unwrap();
    let scratch = Target::new(UVec2::new(8, 8)).unwrap();
    let corner = |x: f32, y: f32| TestTexturedVertex { position: Vec2::new(x, y), uv: Vec2::splat(0.5) };
    let quad = buffer_with::<TestTexturedVertex>(&[corner(-1.0, -1.0), corner(1.0, -1.0), corner(1.0, 1.0), corner(-1.0, 1.0)]);
    let indices = buffer_with::<u32>(&[0, 1, 2, 0, 2, 3]);
    let mut readback = zeroed_buffer::<u8>(8 * 8 * 4);

    let cleared = panic_message(|| Attachment::color(scratch.whole_view(), Some([0.0; 4])).discard_before());
    assert!(cleared.contains("loads nothing anyway"), "{cleared}");
    gpu.submit(|cmd| {
        let message = panic_message(|| {
            cmd.raster(UVec2::new(8, 8), [Attachment::color(scratch.whole_view(), None)])
                .discard_depth_after();
        });
        assert!(message.contains("needs a `depth_attachment`"), "{message}");
        cmd.clear_image(chain.mip_view(0), [0.0, 1.0, 0.0, 1.0]);
        let sample = TestRasterTextured::push_bindings(quad.range(..), chain.mip_view(0), nearest());
        cmd.raster(
            chain.mip_extent(1),
            [Attachment::color(chain.mip_view(1), None).discard_before()],
        )
        .draw(&Draw::indexed(&sample, indices.range(..), 1, RasterState::default(), None))
        .record("discard before");
        cmd.raster(
            UVec2::new(8, 8),
            [Attachment::color(scratch.whole_view(), Some([1.0; 4])).discard_after()],
        )
        .draw(&Draw::indexed(&sample, indices.range(..), 1, RasterState::default(), None))
        .record("discard after");
        cmd.copy_image_to_buffer(chain.mip_view(1).whole(), readback.range(..));
    });
    assert_eq!(readback.host(), [0, 255, 0, 255].repeat(8 * 8));
}

#[test]
fn blend_modes_combine_overlapping_fragments() {
    let mut gpu = gpu();
    let vertices = overlapping_triangles(&mut gpu);
    let mut draw = |clear: [f32; 4], blend| {
        render(&mut gpu, |cmd, target| {
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
    let mut gpu = gpu();
    if !Ctx::features().independent_blend {
        eprintln!("SKIPPED skipped_attachment_keeps_its_content: the device lacks `independent-blend`");
        return;
    }
    let vertices = overlapping_triangles(&mut gpu);
    let second = Target::new(UVec2::new(SIZE, SIZE)).unwrap().bind_storage().unwrap();
    let mut readback = zeroed_buffer::<u8>((SIZE * SIZE * 4) as usize);
    let pixels = render(&mut gpu, |cmd, target| {
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
    let mut gpu = gpu();
    let vertices = two_triangles(&mut gpu);
    let pixels = render(&mut gpu, |cmd, target| {
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
    let mut gpu = gpu();
    let vertices = two_triangles(&mut gpu);
    let pixels = render(&mut gpu, |cmd, target| {
        let state = unculled().topology(PrimitiveTopology::LINE_LIST);
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::new(&bindings(&vertices), 6, 1, state, None))
            .record("lines");
    });
    assert_pixel(&pixels, IN_LEFT.0, IN_LEFT.1, BLACK);
    assert!(pixels.chunks_exact(4).any(|p| p != BLACK));
    // `gpu_missing_features` draws it with the topology class baked.
    assert_golden("raster_lines", &pixels, [SIZE, SIZE]);
}

#[test]
fn write_masks_keep_the_masked_components() {
    let mut gpu = gpu();
    let vertices = overlapping_triangles(&mut gpu);
    let pixels = render(&mut gpu, |cmd, target| {
        let blue = Attachment::color(target.whole_view(), Some([0.0, 0.0, 1.0, 1.0]));
        let state = RasterState::default().write_mask(ColorComponentFlags::R);
        cmd.raster(EXTENT, [blue])
            .draw(&Draw::new(&bindings(&vertices), 6, 1, state, None))
            .record("red only");
    });
    // The red triangle's red over the cleared blue; the green one wrote nothing.
    assert_pixel(&pixels, 32, 40, [255, 0, 255, 255]);
}

#[test]
fn blend_equations_take_any_factors_and_ops() {
    let mut gpu = gpu();
    let vertices = overlapping_triangles(&mut gpu);
    let min = BlendEquation {
        src_color: BlendFactor::ONE,
        dst_color: BlendFactor::ONE,
        color_op: BlendOp::MIN,
        src_alpha: BlendFactor::ONE,
        dst_alpha: BlendFactor::ONE,
        alpha_op: BlendOp::MIN,
    };
    let pixels = render(&mut gpu, |cmd, target| {
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
    let mut gpu = gpu();
    if !Ctx::features().logic_op {
        eprintln!("SKIPPED logic_ops_replace_blending: the device lacks `logic-op`");
        return;
    }
    let vertices = overlapping_triangles(&mut gpu);
    let pixels = render(&mut gpu, |cmd, target| {
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
    let mut gpu = gpu();
    let vertices = two_triangles(&mut gpu);
    let depth = Image::<D32SfloatS8Uint, usage![DepthAttachment]>::new(UVec2::new(SIZE, SIZE)).unwrap();
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
    let pixels = render(&mut gpu, |cmd, target| {
        let marking = unculled()
            .depth_test(None)
            .stencil(Some(Stencil {
                front: mark,
                back: mark,
            }))
            .write_mask(ColorComponentFlags::empty());
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

/// Drawn to over both aspects, sampled as depth, also through a reserved slot and a comparison
/// sampler.
#[test]
fn combined_depth_stencil_images_are_sampled_as_depth() {
    let mut gpu = gpu();
    let vertices = two_triangles(&mut gpu);
    let depth = Image::<D32SfloatS8Uint, usage![DepthAttachment, Sampled]>::new(EXTENT)
        .unwrap()
        .bind_sampled()
        .unwrap();
    let mut out = zeroed_buffer::<f32>((SIZE * SIZE) as usize);
    render(&mut gpu, |cmd, target| {
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
    let mut depth_at = |(x, y): (u32, u32)| out.host()[(y * SIZE + x) as usize];
    assert_eq!(depth_at(IN_LEFT), 0.5);
    assert_eq!(depth_at(IN_RIGHT), 0.25);

    let slot = bindless::reserve::<bindless::Sampled>(1).unwrap();
    bindless::bind([depth.whole_view()], [slot.handle(0)]);
    let less = Sampler::new(
        &SamplerDesc::new()
            .filter(lava::vk::Filter::NEAREST)
            .compare(CompareOp::LESS),
    )
    .unwrap();
    let samplers = [nearest(), nearest(), less.handle()];
    let uv = |(x, y): (u32, u32)| (Vec2::new(x as f32, y as f32) + 0.5) / SIZE as f32;
    let reads = [ImageUse::sampled_read(depth.whole_view())];
    let at = |gpu: &mut common::Gpu, texel, reference| {
        sample(gpu, slot.handle(0).bits(), samplers, uv(texel), reference, Accesses::images(&reads))
    };
    let [read, _, compared] = at(&mut gpu, IN_LEFT, 0.4);
    assert_eq!((read.x, compared.x), (0.5, 1.0), "0.4 < 0.5");
    let [read, _, compared] = at(&mut gpu, IN_RIGHT, 0.4);
    assert_eq!((read.x, compared.x), (0.25, 0.0), "0.4 >= 0.25");
}

#[test]
fn depth_persists_between_draws_unless_cleared() {
    let mut gpu = gpu();
    let vertices = overlapping_triangles(&mut gpu);
    let depth = Image::<D32Sfloat, usage![DepthAttachment]>::new(UVec2::new(SIZE, SIZE)).unwrap();

    let mut draw_far_after_near = |clear_between: Option<[f32; 1]>| {
        render(&mut gpu, |cmd, target| {
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

/// Without a barrier in between.
#[test]
fn draws_of_one_rendering_see_the_earlier_depth() {
    let mut gpu = gpu();
    let near = overlapping_triangles(&mut gpu);
    // Larger than the near triangle, so its rim stays visible.
    let far = buffer_with::<TestVertex>(&[
        vertex(0.0, -0.95, 0.2, RED),
        vertex(0.95, 0.95, 0.2, RED),
        vertex(-0.95, 0.95, 0.2, RED),
    ]);
    let depth = Image::<D32Sfloat, usage![DepthAttachment]>::new(UVec2::new(SIZE, SIZE)).unwrap();
    let pixels = render(&mut gpu, |cmd, target| {
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
    let mut gpu = gpu();
    let near = [
        vertex(0.0, -0.8, 0.8, GREEN),
        vertex(0.8, 0.8, 0.8, GREEN),
        vertex(-0.8, 0.8, 0.8, GREEN),
    ];
    let at_depth = |z: f32, color| near.map(|v| vertex(v.position.x, v.position.y, z, color));
    let far = buffer_with::<TestVertex>(&at_depth(0.2, RED));
    let middle = buffer_with::<TestVertex>(&at_depth(0.5, BLUE));
    let near = buffer_with::<TestVertex>(&near);
    let depth = Image::<D32Sfloat, usage![DepthAttachment]>::new(UVec2::new(SIZE, SIZE)).unwrap();

    // Far (written), then near with the given write flag, then a triangle in between.
    let mut draw = |write_near: bool| {
        render(&mut gpu, |cmd, target| {
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
    let occluded = render(&mut gpu, |cmd, target| {
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
    let mut gpu = gpu();
    let vertices = overlapping_triangles(&mut gpu);
    let depth = Image::<D32Sfloat, usage![DepthAttachment]>::new(UVec2::new(SIZE, SIZE)).unwrap();
    let pixels = render(&mut gpu, |cmd, target| {
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
    let mut gpu = gpu();
    let vertices = overlapping_triangles(&mut gpu);
    let depth = Image::<D32SfloatS8Uint, usage![DepthAttachment]>::new(UVec2::new(SIZE, SIZE)).unwrap();
    let pixels = render(&mut gpu, |cmd, target| {
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
    let mut gpu = gpu();
    // 2x2 texture: red, green / blue, yellow.
    let texels: Vec<u8> = [
        [255, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
        [255, 255, 0, 255],
    ]
    .concat();
    let upload = buffer_with::<u8>(&texels);
    let texture = Image::<R8G8B8A8Unorm, usage![Sampled, TransferDst]>::new(UVec2::new(2, 2))
        .unwrap()
        .bind_sampled().unwrap();

    let corner = |x: f32, y: f32, u: f32, v: f32| TestTexturedVertex {
        position: Vec2::new(x, y),
        uv: Vec2::new(u, v),
    };
    // A quad over the central half of the target, as four vertices and two triangles.
    let vertices = buffer_with::<TestTexturedVertex>(&[
        corner(-0.5, -0.5, 0.0, 0.0),
        corner(0.5, -0.5, 1.0, 0.0),
        corner(0.5, 0.5, 1.0, 1.0),
        corner(-0.5, 0.5, 0.0, 1.0),
    ]);
    let indices = buffer_with::<u32>(&[0, 1, 2, 0, 2, 3]);

    let pixels = render(&mut gpu, |cmd, target| {
        cmd.copy_buffer_to_image(upload.range(..), texture.whole());
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::indexed(
                &// Sampler 0 is nearest-neighbour.
                TestRasterTextured::push_bindings(vertices.range(..), texture.whole_view(), nearest()),
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
    let half = render(&mut gpu, |cmd, target| {
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::indexed(
                &TestRasterTextured::push_bindings(vertices.range(..), texture.whole_view(), nearest()),
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

#[test]
fn bound_textures_are_sampled_from_their_slots() {
    let mut gpu = gpu();
    let colors = [
        [255, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
        [255, 255, 0, 255],
    ];
    let textures: Vec<_> = (0..colors.len())
        .map(|_| {
            Image::<R8G8B8A8Unorm, usage![Sampled, TransferDst]>::new(UVec2::new(1, 1))
                .unwrap()
                .bind_sampled().unwrap()
        })
        .collect();
    let uploads: Vec<_> = colors
        .iter()
        .map(|color| buffer_with::<u8>(color))
        .collect();

    // One quad per texture, in the four quadrants.
    let quads: Vec<_> = (0..4)
        .map(|i| {
            let (x, y) = ((i % 2) as f32 - 1.0, (i / 2) as f32 - 1.0);
            let corner = |dx: f32, dy: f32| TestTexturedVertex {
                position: Vec2::new(x + dx, y + dy),
                uv: Vec2::splat(0.5),
            };
            buffer_with::<TestTexturedVertex>(&[
                corner(0.0, 0.0),
                corner(1.0, 0.0),
                corner(1.0, 1.0),
                corner(0.0, 1.0),
            ])
        })
        .collect();
    let indices = buffer_with::<u32>(&[0, 1, 2, 0, 2, 3]);

    let pixels = render(&mut gpu, |cmd, target| {
        for (upload, texture) in uploads.iter().zip(&textures) {
            cmd.copy_buffer_to_image(upload.range(..), texture.whole());
        }
        for (quad, texture) in quads.iter().zip(&textures) {
            cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
                .draw(&Draw::indexed(
                    &TestRasterTextured::push_bindings(quad.range(..), texture.whole_view(), nearest()),
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

/// Their layouts are tracked apart, which the validation layer checks.
#[test]
fn a_mip_is_drawn_while_another_is_sampled() {
    let mut gpu = gpu();
    let chain =
        MipChain::<R8G8B8A8Unorm, usage![ColorAttachment, Sampled, TransferSrc, TransferDst]>::with_mip_levels(
            UVec2::new(16, 16),
            2,
        )
        .unwrap()
        .bind_mips_sampled().unwrap();
    let corner = |x: f32, y: f32| TestTexturedVertex {
        position: Vec2::new(x, y),
        uv: Vec2::splat(0.5),
    };
    let quad = buffer_with::<TestTexturedVertex>(&[
        corner(-1.0, -1.0),
        corner(1.0, -1.0),
        corner(1.0, 1.0),
        corner(-1.0, 1.0),
    ]);
    let indices = buffer_with::<u32>(&[0, 1, 2, 0, 2, 3]);
    let mut readback = zeroed_buffer::<u8>(8 * 8 * 4);

    gpu.submit(|cmd| {
        cmd.clear_image(chain.mip_view(0), [0.0, 1.0, 0.0, 1.0]);
        cmd.raster(
            chain.mip_extent(1),
            [Attachment::color(chain.mip_view(1), Some([0.0; 4]))],
        )
        .draw(&Draw::indexed(
            &TestRasterTextured::push_bindings(quad.range(..), chain.mip_view(0), nearest()),
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
    let mut gpu = gpu();
    let size = UVec3::new(4, 4, 2);
    let texel = |x: u32, y: u32, z: u32| [x as u8 * 60, y as u8 * 60, z as u8 * 200, 255];
    let texels: Vec<u8> = (0..size.z)
        .flat_map(|z| (0..size.y).flat_map(move |y| (0..size.x).map(move |x| (x, y, z))))
        .flat_map(|(x, y, z)| texel(x, y, z))
        .collect();
    let upload = buffer_with::<u8>(&texels);
    let volume =
        MipChain::<R8G8B8A8Unorm, usage![Sampled, HostCopy, TransferSrc, TransferDst], D3>::with_mip_levels(
            size, 2,
        )
        .unwrap()
        .bind_sampled().unwrap();
    assert_eq!(volume.mip_extent(1), UVec3::new(2, 2, 1));
    let target = Image::<R8G8B8A8Unorm, usage![Storage, TransferSrc, TransferDst]>::new(size.truncate())
        .unwrap()
        .bind_storage().unwrap();
    let mut whole = zeroed_buffer::<u8>(texels.len());
    let mut slice = zeroed_buffer::<u8>(4 * 4 * 4);
    let mut mip = zeroed_buffer::<u8>(2 * 2 * 4);

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
        panic_message(|| volume.copy_region_from(&patch, 0, offset, UVec3::new(4, 4, 2)));
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
    let mut gpu = gpu();
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
    let mut texture = Image::<BC7UnormBlock, usage![Sampled, HostCopy]>::with_mip_levels(UVec2::new(8, 8), 4)
        .unwrap()
        .bind_sampled().unwrap();
    texture.copy_from(&blocks, 0).unwrap();
    // The smaller levels are one block each, even the ones smaller than a block.
    for level in 1..4 {
        texture.copy_from(&blocks[..16], level).unwrap();
        panic_message(|| texture.copy_from(&blocks[..8], level));
        panic_message(|| texture.copy_from(&blocks[..32], level));
    }
    panic_message(|| texture.copy_from(&blocks[..48], 0));
    // A region of one block: the yellow block becomes red.
    texture
        .copy_region_from(&blocks[..16], 0, UVec2::splat(4), UVec2::splat(4))
        .unwrap();

    let corner = |x: f32, y: f32, u: f32, v: f32| TestTexturedVertex {
        position: Vec2::new(x, y),
        uv: Vec2::new(u, v),
    };
    let vertices = buffer_with::<TestTexturedVertex>(&[
        corner(-0.5, -0.5, 0.0, 0.0),
        corner(0.5, -0.5, 1.0, 0.0),
        corner(0.5, 0.5, 1.0, 1.0),
        corner(-0.5, 0.5, 0.0, 1.0),
    ]);
    let indices = buffer_with::<u32>(&[0, 1, 2, 0, 2, 3]);
    let pixels = render(&mut gpu, |cmd, target| {
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::indexed(
                &TestRasterTextured::push_bindings(vertices.range(..), texture.whole_view(), nearest()),
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
    let mut gpu = gpu();
    if !Ctx::features().mesh_shader {
        eprintln!("SKIPPED mesh_pass_emits_triangles_per_workgroup: no mesh shader support");
        return;
    }
    let pixels = render(&mut gpu, |cmd, target| {
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
fn indirect_mesh_launches_match_the_direct_one() {
    let mut gpu = gpu();
    if !Ctx::features().mesh_shader {
        eprintln!("SKIPPED indirect_mesh_launches_match_the_direct_one: no mesh shader support");
        return;
    }
    let bindings =
        || TestMesh::push_bindings(Vec4::new(0.8, 0.8, 0.8, 1.0), Vec2::new(1.0, 0.0), 0.4).unwrap();
    let direct = render(&mut gpu, |cmd, target| {
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::launch(&bindings(), [2, 1, 1], unculled(), None))
            .record("direct")
    });
    let launch = buffer_with::<DispatchIndirectCommand>(&[DispatchIndirectCommand {
        x: 2,
        y: 1,
        z: 1,
    }]);
    let indirect = render(&mut gpu, |cmd, target| {
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

    // The first of two commands, counted on the GPU.
    let launches = buffer_with::<DispatchIndirectCommand>(&[
        DispatchIndirectCommand { x: 2, y: 1, z: 1 },
        DispatchIndirectCommand { x: 1, y: 1, z: 1 },
    ]);
    let count = buffer_with::<u32>(&[1]);
    let counted = render(&mut gpu, |cmd, target| {
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::launch_indirect_count(
                &bindings(),
                launches.range(..),
                count.range(..),
                unculled(),
                None,
            ))
            .record("indirect count")
    });
    assert_eq!(counted, direct);
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
        let buffer = zeroed_buffer::<u32>(1);
        let vertices = zeroed_buffer::<TestVertex>(1);
        let compute = TestComputeBuffer::push_bindings(buffer.range(..), buffer.range(..), 1, 1)
            .sections()
            .pass;
        assert_eq!(
            PASS_MAP.passes[compute].kind,
            PassKind::Compute {
                entry: c"test_compute_buffer"
            }
        );
        let raster = TestRaster::push_bindings(vertices.range(..), Vec2::ZERO)
            .sections()
            .pass;
        assert_eq!(
            PASS_MAP.passes[raster].kind,
            PassKind::RasterVertex {
                vertex: c"test_raster_vertex",
                fragment: Some(c"test_raster_fragment")
            }
        );
        assert_eq!(PASS_MAP.passes[raster].name, "TestRaster");
    }

    #[test]
    fn push_constants_are_the_struct_at_its_size() {
        let _gpu = gpu();
        let (src, dst) = (
            zeroed_buffer::<u32>(16),
            zeroed_buffer::<u32>(16),
        );
        let bindings = TestComputeBuffer::push_bindings(src.range(..), dst.range(..), 16, 3);
        let size = size_of::<CTestComputeBufferBindings>();
        assert_eq!(bindings.sections().push_constants.len(), size);
    }

    #[test]
    fn buffer_fields_become_device_addresses_and_accesses() {
        let _gpu = gpu();
        let (src, dst) = (
            zeroed_buffer::<u32>(16),
            zeroed_buffer::<u32>(16),
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
        let vertices = zeroed_buffer::<TestVertex>(3);
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

    /// The second dispatch's read waits for the first one's write; validation checks it.
    #[test]
    fn extra_accesses_get_barriers_like_the_bindings() {
        let mut gpu = gpu();
        let image = Image::<R8G8B8A8Unorm, usage![Storage]>::new(UVec2::new(8, 8))
            .unwrap()
            .bind_storage()
            .unwrap();
        let buffer = zeroed_buffer::<u32>(4);
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
        let image = Image::<R8G8B8A8Unorm, usage![Storage]>::new(UVec2::new(8, 8))
            .unwrap()
            .bind_storage().unwrap();
        let target = image.whole_view();
        let bindings =
            TestComputeImage::push_bindings(target, Vec4::ONE, UVec2::ZERO, UVec2::new(8, 8));
        // A struct with handles is `NoUninit`, not `Pod`: compare the handle's bytes.
        let offset = core::mem::offset_of!(CTestComputeImageBindings, target);
        let handle = image.handle::<bindless::Storage>();
        assert_eq!(&bindings.sections().push_constants[offset..offset + 4], bytemuck::bytes_of(&handle));
        assert!(handle.storage_slot().is_some());
        let [image]: [_; 1] = bindings
            .sections()
            .images()
            .collect::<Vec<_>>()
            .try_into()
            .expect("one image");
        assert_eq!(image.access(), A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE);
        assert_eq!(image.stage(), S::COMPUTE_SHADER);
        assert_eq!(image.stage(), lava::pipeline::PassKind::Compute { entry: c"" }.stages());
        assert_eq!(image.layout(), L::GENERAL);
    }
}

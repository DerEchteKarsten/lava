//! Benchmark of recording lava-tests' passes into real command buffers, headless with the validation layer off: per scenario, interleaved frame by frame, the CPU time of building the bindings and recording them, and of the whole submission from `begin` until `submit` returned (`LAVA_BENCH_FRAMES` sets the frame count)

use std::{hint::black_box, time::Instant};

use glam::{UVec2, Vec2, Vec4};
use lava::{
    Ctx,
    bindless::{Sampler, SamplerDesc},
    buffer::Buffer,
    command_buffer::{Accesses, Attachment, CommandBuffer, Draw, ImageUse, RasterState},
    image::{Image, format::R8G8B8A8Unorm, usage::usage},
    queue::{Graphics, Timeline, Wait},
};
use lava_tests::bindings::*;

const SIZE: u32 = 64;
const EXTENT: UVec2 = UVec2::splat(SIZE);
const TEXTURES: usize = 64;

type Target = Image<R8G8B8A8Unorm, usage![ColorAttachment, Storage, TransferSrc, TransferDst]>;
type Texture = Image<R8G8B8A8Unorm, usage![Sampled, TransferDst]>;

struct Resources {
    target: Target,
    textures: Vec<Texture>,
    sampler: Sampler,
    src: Buffer<u32>,
    dst: Vec<Buffer<u32>>,
    out: Buffer<u32>,
    vertices: Buffer<TestVertex>,
    textured: Vec<Buffer<TestTexturedVertex>>,
    indices: Buffer<u32>,
}

/// 64 compute passes: 32 with two buffers, 16 with specialization constants that name a buffer
/// and an image, 16 writing an image.
fn compute(cmd: &mut CommandBuffer<Graphics>, r: &Resources) {
    for i in 0..32 {
        cmd.compute(
            &TestComputeBuffer::push_bindings(
                TestComputeBufferArgs {
                    src: r.src.range(..),
                    dst: r.dst[i % 16].range(..),
                    count: 64,
                    multiplier: i as u32,
                },
                None,
            ),
            [1, 1, 1],
        );
    }
    for i in 0..16 {
        cmd.compute(
            &TestConstants::push_bindings(
                TestConstantsArgs {
                    offset: i,
                    count: 8,
                },
                TestConstantsConstants {
                    out: r.out.range(..),
                    target: r.target.whole_view(),
                    multiplier: 3,
                    brightness: 1.0,
                },
                None,
            ),
            [1, 1, 1],
        );
    }
    for i in 0..16 {
        cmd.compute(
            &TestComputeImage::push_bindings(
                TestComputeImageArgs {
                    target: r.target.whole_view(),
                    tint: Vec4::splat(i as f32 / 16.0),
                    offset: UVec2::ZERO,
                    size: UVec2::splat(8),
                },
                None,
            ),
            [1, 1, 1],
        );
    }
}

/// 8 renderings of 8 indexed draws, each with a vertex buffer, a texture and two declared ones.
fn many_draws(cmd: &mut CommandBuffer<Graphics>, r: &Resources) {
    let textures = |d: usize| {
        [(d + 1) % TEXTURES, (d + 32) % TEXTURES]
            .map(|t| ImageUse::sampled_read(r.textures[t].whole_view()))
    };
    let state = RasterState::default;
    for j in 0..8 {
        let i = j * 8;
        let indices = r.indices.range(..);
        let declared: [[ImageUse; 2]; 8] = std::array::from_fn(|k| textures(i + k));
        let draws: [_; 8] = std::array::from_fn(|k| {
            let args = TestRasterTexturedArgs {
                vertices: r.textured[i + k].range(..),
                tex: r.textures[i + k].whole_view(),
                sampler: r.sampler.handle(),
            };
            TestRasterTextured::push_bindings(args, Some(Accesses::images(&declared[k])))
        });
        cmd.raster(EXTENT, [Attachment::color(r.target.whole_view(), None)])
            .draw(&Draw::indexed(&draws[0], indices, 1, state()))
            .draw(&Draw::indexed(&draws[1], indices, 1, state()))
            .draw(&Draw::indexed(&draws[2], indices, 1, state()))
            .draw(&Draw::indexed(&draws[3], indices, 1, state()))
            .draw(&Draw::indexed(&draws[4], indices, 1, state()))
            .draw(&Draw::indexed(&draws[5], indices, 1, state()))
            .draw(&Draw::indexed(&draws[6], indices, 1, state()))
            .draw(&Draw::indexed(&draws[7], indices, 1, state()))
            .record("many_draws");
    }
}

/// 4 renderings of one draw that reaches 16 declared textures.
fn declared(cmd: &mut CommandBuffer<Graphics>, r: &Resources) {
    for j in 0..4 {
        let textures: [ImageUse; 16] =
            std::array::from_fn(|k| ImageUse::sampled_read(r.textures[j * 16 + k].whole_view()));
        let args = TestRasterArgs {
            vertices: r.vertices.range(..),
            instance_offset: Vec2::ZERO,
        };
        cmd.raster(EXTENT, [Attachment::color(r.target.whole_view(), None)])
            .draw(&Draw::new(
                &TestRaster::push_bindings(args, Some(Accesses::images(&textures))),
                3,
                1,
                RasterState::default(),
            ))
            .record("declared");
    }
}

/// 8 renderings of 4 mesh-shader launches without accesses: a binding of push constants only.
fn mesh(cmd: &mut CommandBuffer<Graphics>, r: &Resources) {
    let launch = |i: u32| {
        TestMesh::push_bindings(
            TestMeshArgs {
                color: Vec4::ONE,
                group_offset: Vec2::new(i as f32, 0.0),
                scale: 0.1,
            },
            None,
        )
        .expect("mesh shaders")
    };
    let state = RasterState::default;
    for _ in 0..8 {
        cmd.raster(EXTENT, [Attachment::color(r.target.whole_view(), None)])
            .draw(&Draw::launch(&launch(0), [1, 1, 1], state()))
            .draw(&Draw::launch(&launch(1), [1, 1, 1], state()))
            .draw(&Draw::launch(&launch(2), [1, 1, 1], state()))
            .draw(&Draw::launch(&launch(3), [1, 1, 1], state()))
            .record("mesh");
    }
}

const WARMUP: usize = 200;
const FRAMES: usize = 3000;

/// Records one scenario's commands.
type Scenario = fn(&mut CommandBuffer<Graphics>, &Resources);

/// The `(median, 10th, 90th percentile)` of `times`.
fn percentiles(times: &mut [u64]) -> [u64; 3] {
    times.sort_unstable();
    let at = |q: f64| times[((times.len() - 1) as f64 * q) as usize];
    [at(0.5), at(0.1), at(0.9)]
}

fn main() {
    lava::init(&lava::Config::new(&PASS_MAP)).expect("headless lava init failed");
    let frames = std::env::var("LAVA_BENCH_FRAMES")
        .ok()
        .map(|n| n.parse().expect("LAVA_BENCH_FRAMES is a frame count"))
        .unwrap_or(FRAMES);
    let mut timeline = Timeline::<Graphics>::new(2, 0).unwrap();
    let buffer = |len| Buffer::new(len).unwrap();
    let resources = Resources {
        sampler: Sampler::new(&SamplerDesc::new()).unwrap(),
        target: Target::new(EXTENT).unwrap().bind_storage().unwrap(),
        textures: (0..TEXTURES)
            .map(|_| {
                Texture::new(UVec2::splat(4))
                    .unwrap()
                    .bind_sampled()
                    .unwrap()
            })
            .collect(),
        src: buffer(256),
        dst: (0..16).map(|_| buffer(256)).collect(),
        out: buffer(256),
        vertices: Buffer::new(6).unwrap(),
        textured: (0..TEXTURES).map(|_| Buffer::new(4).unwrap()).collect(),
        indices: Buffer::new(6).unwrap(),
    };
    // Defined contents: zeroed vertices (degenerate triangles) and indices, numbers to copy.
    let r = &resources;
    (timeline.begin().unwrap())
        .submit(|cmd| {
            cmd.fill_buffer(r.src.range(..), 1);
            for buffer in r.dst.iter().chain([&r.out]) {
                cmd.fill_buffer(buffer.range(..), 0);
            }
            cmd.fill_buffer(r.vertices.range(..).cast::<u32>(), 0);
            for buffer in &r.textured {
                cmd.fill_buffer(buffer.range(..).cast::<u32>(), 0);
            }
            cmd.fill_buffer(r.indices.range(..), 0);
        })
        .unwrap()
        .block()
        .unwrap();

    let mut scenarios: Vec<(&str, Scenario)> = vec![
        ("compute", compute),
        ("many_draws", many_draws),
        ("declared", declared),
    ];
    if Ctx::features().mesh_shader {
        scenarios.push(("mesh", mesh));
    }

    // Per scenario, the recording's and the submission's times of each measured frame.
    let mut times = vec![(Vec::with_capacity(frames), Vec::with_capacity(frames)); scenarios.len()];
    let mut last: Option<Wait> = None;
    for frame in 0..WARMUP + frames {
        // One frame of each scenario after another, so clock changes hit them alike.
        for (&(_, scenario), (recording, submission)) in scenarios.iter().zip(&mut times) {
            // The GPU finished the last frame, so `begin` doesn't wait for it: the submission
            // column is CPU time (the pool reset, recording, the final barrier, merging the
            // pending accesses, ending the command buffer and the submit).
            if let Some(wait) = last.take() {
                wait.block().unwrap();
            }
            let mut elapsed = 0;
            let start = Instant::now();
            let done = (timeline.begin().unwrap())
                .submit(|cmd| {
                    let start = Instant::now();
                    scenario(cmd, black_box(&resources));
                    elapsed = start.elapsed().as_nanos() as u64;
                })
                .unwrap();
            let submitted = start.elapsed().as_nanos() as u64;
            last = Some(done);
            if frame >= WARMUP {
                recording.push(elapsed);
                submission.push(submitted);
            }
        }
    }

    println!(
        "Time per frame in ns (median, 10th and 90th percentile of {frames} frames, scenarios interleaved):"
    );
    println!("  recording: building the bindings and recording them (the submit closure);");
    println!(
        "  submission: begin() until submit() returned, after the GPU finished the last frame."
    );
    println!("{:<14} {:>29}   {:>29}", "", "recording", "submission");
    for ((name, _), (recording, submission)) in scenarios.iter().zip(&mut times) {
        let [r50, r10, r90] = percentiles(recording);
        let [s50, s10, s90] = percentiles(submission);
        println!("{name:<14} {r50:>9} {r10:>9} {r90:>9}   {s50:>9} {s10:>9} {s90:>9}");
    }
    drop(timeline);
    lava::wait_idle().unwrap();
}

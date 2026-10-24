//! Benchmark of recording lava-tests' passes into real command buffers, headless with the validation layer off: the CPU time of building the bindings and recording them, per scenario

use std::{hint::black_box, time::Instant};

use glam::{UVec2, Vec2, Vec4};
use lava::{
    bindless::{Sampler, SamplerDesc},
    buffer::Buffer,
    command_buffer::{Accesses, Attachment, CommandBuffer, Draw, ImageUse, RasterState},
    image::{
        Image,
        format::R8G8B8A8Unorm,
        usage::usage,
    },
    queue::{Graphics, Timeline},
    state::Ctx,
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
            &TestComputeBuffer::push_bindings(TestComputeBufferArgs { src: r.src.range(..), dst: r.dst[i % 16].range(..), count: 64, multiplier: i as u32 }, None),
            [1, 1, 1]);
    }
    for i in 0..16 {
        cmd.compute(
            &TestConstants::push_bindings(TestConstantsArgs { offset: i, count: 8 }, TestConstantsConstants { out: r.out.range(..), target: r.target.whole_view(), multiplier: 3, brightness: 1.0 }, None),
            [1, 1, 1]);
    }
    for i in 0..16 {
        cmd.compute(
            &TestComputeImage::push_bindings(TestComputeImageArgs { target: r.target.whole_view(), tint: Vec4::splat(i as f32 / 16.0), offset: UVec2::ZERO, size: UVec2::splat(8) }, None),
            [1, 1, 1]);
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
        TestMesh::push_bindings(TestMeshArgs { color: Vec4::ONE, group_offset: Vec2::new(i as f32, 0.0), scale: 0.1 }, None).expect("mesh shaders")
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

fn main() {
    lava::init(&lava::Config::new(&PASS_MAP)).expect("headless lava init failed");
    let mut timeline = Timeline::<Graphics>::new(2, 0).unwrap();
    let buffer = |len| Buffer::new(len).unwrap();
    let resources = Resources {
        sampler: Sampler::new(&SamplerDesc::new()).unwrap(),
        target: Target::new(EXTENT).unwrap().bind_storage().unwrap(),
        textures: (0..TEXTURES)
            .map(|_| {
                Texture::new(UVec2::splat(4))
                    .unwrap()
                    .bind_sampled().unwrap()
            })
            .collect(),
        src: buffer(256),
        dst: (0..16).map(|_| buffer(256)).collect(),
        out: buffer(256),
        vertices: Buffer::new(6).unwrap(),
        textured: (0..TEXTURES)
            .map(|_| Buffer::new(4).unwrap())
            .collect(),
        indices: Buffer::new(6).unwrap(),
    };

    let mut scenarios: Vec<(&str, fn(&mut CommandBuffer<Graphics>, &Resources))> = vec![
        ("compute", compute),
        ("many_draws", many_draws),
        ("declared", declared),
    ];
    if Ctx::features().mesh_shader {
        scenarios.push(("mesh", mesh));
    }

    println!(
        "Recording time per frame in ns (median, 10th and 90th percentile of {FRAMES} frames):"
    );
    for (name, scenario) in scenarios {
        let mut times = Vec::with_capacity(FRAMES);
        for frame in 0..WARMUP + FRAMES {
            let mut elapsed = 0;
            (timeline.begin().unwrap())
                .submit(|cmd| {
                    let start = Instant::now();
                    scenario(cmd, black_box(&resources));
                    elapsed = start.elapsed().as_nanos() as u64;
                })
                .unwrap();
            if frame >= WARMUP {
                times.push(elapsed);
            }
        }
        times.sort_unstable();
        let at = |q: f64| times[((times.len() - 1) as f64 * q) as usize];
        println!("{name:<14} {:>9} {:>9} {:>9}", at(0.5), at(0.1), at(0.9));
    }
    drop(timeline);
    lava::wait_idle().unwrap();
}

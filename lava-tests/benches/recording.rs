//! Benchmark of recording lava-tests' passes into real command buffers, headless with the validation layer off: the CPU time of building the bindings and recording them, per scenario
//!
//! Run with `cargo +nightly bench -p lava-tests --bench recording`.
//!
//! Every frame records one scenario into a command buffer of one of two frame slots and
//! submits it; only the recording closure is timed. The GPU work is tiny (one workgroup per
//! dispatch, a 64x64 target), so a slot's previous frame is done by the time it begins again.
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::{hint::black_box, time::Instant};

use glam::{UVec2, Vec2, Vec4};
use lava::{
    buffer::{
        Buffer,
        usage::{Index, Storage},
    },
    command_buffer::{CommandBuffer, RasterState},
    epoch,
    image::{
        Image,
        format::R8G8B8A8Unorm,
        usage::{ColorAttachmentStorageTransferSrcTransferDst, SampledTransferDst},
    },
    state::Ctx,
    vkobjects::queue::{FrameSlot, Gfx, Queue},
};
use lava_tests::bindings::{
    PASS_MAP, TestComputeBuffer, TestComputeImage, TestConstants, TestMesh, TestRaster,
    TestRasterTextured, TestTexturedVertex, TestVertex,
};

const SIZE: u32 = 64;
const EXTENT: UVec2 = UVec2::splat(SIZE);
const TEXTURES: usize = 64;

type Target = Image<R8G8B8A8Unorm, ColorAttachmentStorageTransferSrcTransferDst>;
type Texture = Image<R8G8B8A8Unorm, SampledTransferDst>;

struct Resources {
    target: Target,
    textures: Vec<Texture>,
    src: Buffer<u32, Storage>,
    dst: Vec<Buffer<u32, Storage>>,
    out: Buffer<u32, Storage>,
    vertices: Buffer<TestVertex, Storage>,
    textured: Vec<Buffer<TestTexturedVertex, Storage>>,
    indices: Buffer<u32, Index>,
}

impl Resources {
    fn new() -> Self {
        let buffer = |len| Buffer::new(len, false).unwrap();
        Self {
            target: Target::new(EXTENT).unwrap().bind_storage(0),
            textures: (0..TEXTURES)
                .map(|i| {
                    Texture::new(UVec2::splat(4))
                        .unwrap()
                        .bind_sampled(i as u32)
                })
                .collect(),
            src: buffer(256),
            dst: (0..16).map(|_| buffer(256)).collect(),
            out: buffer(256),
            vertices: Buffer::new(6, false).unwrap(),
            textured: (0..TEXTURES)
                .map(|_| Buffer::new(4, false).unwrap())
                .collect(),
            indices: Buffer::new(6, false).unwrap(),
        }
    }
}

/// 64 compute passes: 32 with two buffers, 16 with specialization constants that name a buffer
/// and an image, 16 writing an image.
fn compute(cmd: &mut CommandBuffer, r: &Resources) {
    for i in 0..32 {
        cmd.compute(
            &TestComputeBuffer::push_bindings(
                r.src.range(..),
                r.dst[i % 16].range(..),
                64,
                i as u32,
            ),
            [1, 1, 1],
        );
    }
    for i in 0..16 {
        cmd.compute(
            &TestConstants::push_bindings(i, 8).constant_bindings(
                r.out.range(..),
                r.target.whole_view(),
                3,
                1.0,
            ),
            [1, 1, 1],
        );
    }
    for i in 0..16 {
        cmd.compute(
            &TestComputeImage::push_bindings(
                r.target.whole_view(),
                Vec4::splat(i as f32 / 16.0),
                UVec2::ZERO,
                UVec2::splat(8),
            ),
            [1, 1, 1],
        );
    }
}

/// 8 renderings of 8 indexed draws, each with a vertex buffer, a texture and two registered
/// textures.
fn many_draws(cmd: &mut CommandBuffer, r: &Resources) {
    let draw = |i: usize| {
        TestRasterTextured::push_bindings(r.textured[i].range(..), r.textures[i].whole_view(), 0)
            .sampled_read(r.textures[(i + 1) % TEXTURES].whole_view())
            .sampled_read(r.textures[(i + 32) % TEXTURES].whole_view())
    };
    let state = RasterState::default;
    for j in 0..8 {
        let i = j * 8;
        let indices = r.indices.range(..);
        cmd.raster(EXTENT)
            .color_attachment(r.target.whole_view(), None)
            .draw_indexed(&draw(i), indices, 1, state())
            .draw_indexed(&draw(i + 1), indices, 1, state())
            .draw_indexed(&draw(i + 2), indices, 1, state())
            .draw_indexed(&draw(i + 3), indices, 1, state())
            .draw_indexed(&draw(i + 4), indices, 1, state())
            .draw_indexed(&draw(i + 5), indices, 1, state())
            .draw_indexed(&draw(i + 6), indices, 1, state())
            .draw_indexed(&draw(i + 7), indices, 1, state())
            .record("many_draws");
    }
}

macro_rules! registrations {
    ($bindings:expr, $r:expr, $base:expr; $($i:literal)*) => {
        $bindings $(.sampled_read($r.textures[$base + $i].whole_view()))*
    };
}

/// 4 renderings of one draw that registers 16 textures.
fn registrations(cmd: &mut CommandBuffer, r: &Resources) {
    for j in 0..4 {
        let t = j * 16;
        cmd.raster(EXTENT)
            .color_attachment(r.target.whole_view(), None)
            .draw(
                &registrations!(TestRaster::push_bindings(r.vertices.range(..), Vec2::ZERO), r, t;
                    0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15),
                3,
                1,
                RasterState::default(),
            )
            .record("registrations");
    }
}

/// 8 renderings of 4 mesh-shader launches without accesses: a binding of push constants only.
fn mesh(cmd: &mut CommandBuffer, r: &Resources) {
    let launch = |i: u32| TestMesh::push_bindings(Vec4::ONE, Vec2::new(i as f32, 0.0), 0.1);
    let state = RasterState::default;
    for _ in 0..8 {
        cmd.raster(EXTENT)
            .color_attachment(r.target.whole_view(), None)
            .launch(&launch(0), [1, 1, 1], state())
            .launch(&launch(1), [1, 1, 1], state())
            .launch(&launch(2), [1, 1, 1], state())
            .launch(&launch(3), [1, 1, 1], state())
            .record("mesh");
    }
}

const WARMUP: usize = 200;
const FRAMES: usize = 3000;

fn main() {
    lava::init(None, false, false, &PASS_MAP).expect("headless lava init failed");
    let queue = Queue::<Gfx>::new();
    let resources = Resources::new();
    let mut slots = [
        FrameSlot::new(&queue).unwrap(),
        FrameSlot::new(&queue).unwrap(),
    ];

    let mut scenarios: Vec<(&str, fn(&mut CommandBuffer, &Resources))> = vec![
        ("compute", compute),
        ("many_draws", many_draws),
        ("registrations", registrations),
    ];
    if Ctx::features().mesh {
        scenarios.push(("mesh", mesh));
    }

    println!(
        "Recording time per frame in ns (median, 10th and 90th percentile of {FRAMES} frames):"
    );
    for (name, scenario) in scenarios {
        let mut times = Vec::with_capacity(FRAMES);
        for frame in 0..WARMUP + FRAMES {
            let slot = &mut slots[frame % 2];
            let mut elapsed = 0;
            slot.begin(epoch::reserve())
                .unwrap()
                .execute(&queue, &[], &[], |cmd| {
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
    drop(slots);
    lava::wait_idle().unwrap();
}

//! GPU tests of pass pipelines: the pipeline cache file (a foreign one is ignored, the saved one is this device's and loads in a new process), threads that make the same new specialization at once, and `reload_pass` (swap, refusal, races with new specializations)
mod common;

use std::{
    path::{Path, PathBuf},
    sync::{Barrier, LazyLock, Once},
};

use common::{Cmd, GpuGuard, zeroed_buffer};
use glam::{UVec2, Vec2, Vec4};
use lava::{
    buffer::Buffer,
    command_buffer::{Attachment, CullModeFlags, Draw, RasterState},
    image::{Image, format::R8G8B8A8Unorm, usage::usage},
    pipeline::{reload_pass, save_cache},
    queue::{Graphics, Timeline},
    state::Ctx,
};
use lava_tests::bindings::{PASS_MAP, TestComputeBuffer, TestRaster, TestSpecialized, TestVertex};

/// Set in the process `the_pipeline_cache_round_trips` starts, which loads the saved cache.
const CHILD: &str = "LAVA_TESTS_PIPELINE_CACHE_CHILD";

static CACHE: LazyLock<PathBuf> =
    LazyLock::new(|| Path::new(env!("CARGO_TARGET_TMPDIR")).join("gpu_pipelines-cache.bin"));

/// A cache header of a device that doesn't exist (vendor `0xFFFF`), which init has to ignore.
fn foreign_cache() -> Vec<u8> {
    let mut data = Vec::new();
    for word in [32u32, 1, 0xFFFF, 0xFFFF] {
        data.extend(word.to_le_bytes());
    }
    data.extend([0xAB; 16 + 64]);
    data
}

/// The harness context with its pipeline cache at [`CACHE`]: a foreign one, except in the
/// child process, which loads what its parent saved.
fn gpu() -> GpuGuard {
    static SETUP: Once = Once::new();
    SETUP.call_once(|| {
        let path: &'static Path = &CACHE;
        if std::env::var_os(CHILD).is_none() {
            std::fs::write(path, foreign_cache()).unwrap();
        }
        common::pipeline_cache(path);
    });
    common::gpu()
}

fn index(name: &str) -> usize {
    (PASS_MAP.passes.iter().position(|pass| pass.name == name)).unwrap()
}

/// `out[i] = i * factor` for 64 elements, through `TestSpecialized`.
fn specialized(cmd: &mut Cmd, out: &Buffer<u32, lava::buffer::Readback>, factor: u32) {
    let bindings = TestSpecialized::push_bindings(out.range(..), 64).constant_bindings(factor);
    cmd.compute(&bindings, [1, 1, 1], None);
}

fn expected(factor: u32) -> Vec<u32> {
    (0..64).map(|i| i * factor).collect()
}

/// Runs `TestSpecialized` with `factor` on a timeline of its own and checks what it wrote.
fn run_specialized(timeline: &mut Timeline<Graphics>, factor: u32) {
    let mut out = zeroed_buffer::<u32>(64);
    let submission = timeline.begin().unwrap();
    let done = submission
        .submit(|cmd| specialized(cmd, &out, factor))
        .unwrap();
    done.block().unwrap();
    assert_eq!(out.host(), expected(factor), "factor {factor}");
}

#[test]
fn the_pipeline_cache_round_trips() {
    let mut gpu = gpu();
    if std::env::var_os(CHILD).is_some() {
        return;
    }
    // Init ignored the foreign cache; pipelines made since go into the new one.
    let mut out = zeroed_buffer::<u32>(64);
    gpu.submit(|cmd| specialized(cmd, &out, 7));
    assert_eq!(out.host(), expected(7));
    save_cache().unwrap();
    let data = std::fs::read(&*CACHE).unwrap();
    let properties = Ctx::device_properties();
    let word = |i: usize| u32::from_le_bytes(data[i * 4..i * 4 + 4].try_into().unwrap());
    assert!(
        word(0) >= 32 && word(1) == 1,
        "a `VkPipelineCacheHeaderVersionOne`"
    );
    assert_eq!(
        (word(2), word(3)),
        (properties.vendor_id, properties.device_id)
    );
    assert_eq!(data[16..32], properties.pipeline_cache_uuid);

    // A new process loads it at init and makes its pipelines through it.
    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "the_saved_cache_loads",
            "--exact",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD, "1")
        .output()
        .unwrap();
    let output = String::from_utf8_lossy(&child.stdout).into_owned()
        + &String::from_utf8_lossy(&child.stderr);
    assert!(
        child.status.success(),
        "the child process failed:\n{output}"
    );
    assert!(
        output.contains("1 passed"),
        "the child process ran its test:\n{output}"
    );
}

/// The child process of [`the_pipeline_cache_round_trips`]; passes trivially elsewhere.
#[test]
fn the_saved_cache_loads() {
    if std::env::var_os(CHILD).is_none() {
        return;
    }
    let mut gpu = gpu();
    let mut out = zeroed_buffer::<u32>(64);
    gpu.submit(|cmd| specialized(cmd, &out, 7));
    assert_eq!(out.host(), expected(7));
}

/// Threads recording the same new specialization at once each get a working pipeline: one makes
/// it, or both do and the first insert wins.
#[test]
fn threads_make_one_new_specialization_together() {
    let _gpu = gpu();
    const THREADS: usize = 4;
    let barrier = Barrier::new(THREADS);
    std::thread::scope(|scope| {
        for _ in 0..THREADS {
            scope.spawn(|| {
                let mut timeline = Timeline::<Graphics>::new(1, 0).unwrap();
                for factor in 100..116 {
                    barrier.wait();
                    run_specialized(&mut timeline, factor);
                }
            });
        }
    });
}

type Target = Image<R8G8B8A8Unorm, usage![ColorAttachment, TransferSrc, TransferDst]>;

/// A red triangle over the middle of a black 16×16 target, read back.
fn triangle(gpu: &mut common::Gpu, vertices: &Buffer<TestVertex>) -> Vec<u8> {
    const SIZE: u32 = 16;
    let target = Target::new(UVec2::splat(SIZE)).unwrap();
    let mut readback = zeroed_buffer::<u8>((SIZE * SIZE * 4) as usize);
    gpu.submit(|cmd| {
        let bindings = TestRaster::push_bindings(vertices.range(..), Vec2::ZERO);
        let state = RasterState::default().cull_mode(CullModeFlags::NONE);
        cmd.raster(
            UVec2::splat(SIZE),
            [Attachment::color(
                target.whole_view(),
                Some([0.0, 0.0, 0.0, 1.0]),
            )],
        )
        .draw(&Draw::new(&bindings, 3, 1, state, None))
        .record("triangle");
        cmd.copy_image_to_buffer(target.whole(), readback.range(..));
    });
    readback.host().to_vec()
}

/// A reload replaces the pipelines of every kind of pass, including the specializations and
/// raster keys made so far; one the driver can't take changes nothing.
#[test]
fn reload_pass_replaces_the_pipelines() {
    let mut gpu = gpu();
    let red = Vec4::new(1.0, 0.0, 0.0, 1.0);
    let vertex = |x: f32, y: f32| TestVertex {
        position: Vec4::new(x, y, 0.5, 1.0),
        color: red,
    };
    let vertices = gpu.buffer(&[vertex(-0.8, -0.8), vertex(0.8, -0.8), vertex(0.0, 0.8)]);
    let src = gpu.buffer(&[1u32, 2, 3, 4]);
    let compute = |gpu: &mut common::Gpu| {
        let mut dst = zeroed_buffer::<u32>(4);
        gpu.submit(|cmd| {
            let bindings = TestComputeBuffer::push_bindings(src.range(..), dst.range(..), 4, 2);
            cmd.compute(&bindings, [1, 1, 1], None);
        });
        dst.host().to_vec()
    };
    let check = |gpu: &mut common::Gpu, image: &[u8]| {
        assert_eq!(compute(gpu), [2, 5, 8, 11]);
        let mut out = zeroed_buffer::<u32>(64);
        gpu.submit(|cmd| specialized(cmd, &out, 3));
        assert_eq!(out.host(), expected(3));
        assert_eq!(triangle(gpu, &vertices), image);
    };
    let image = triangle(&mut gpu, &vertices);
    assert_eq!(
        image[(8 * 16 + 8) * 4..][..4],
        [255, 0, 0, 255],
        "the triangle covers the middle"
    );
    check(&mut gpu, &image);

    for name in ["TestComputeBuffer", "TestSpecialized", "TestRaster"] {
        let pass = &PASS_MAP.passes[index(name)];
        reload_pass(index(name), pass.spirv).unwrap();
        check(&mut gpu, &image);
        assert!(reload_pass(index(name), &[3, 2, 1]).is_err(), "not SPIR-V");
        check(&mut gpu, &image);
    }
    // The replaced pipelines are retired, and destroyed once no submission can use them.
    lava::wait_idle().unwrap();
    check(&mut gpu, &image);
}

/// Reloads while other threads make new specializations of the same pass: each creation holds
/// the module a reload replaces, so every pipeline is made from a live module and rebuilt by the
/// next reload.
#[test]
fn reloads_race_with_new_specializations() {
    let _gpu = gpu();
    let pass = index("TestSpecialized");
    std::thread::scope(|scope| {
        for thread in 0..2u32 {
            scope.spawn(move || {
                let mut timeline = Timeline::<Graphics>::new(1, 0).unwrap();
                for factor in 0..12 {
                    run_specialized(&mut timeline, 1000 + thread * 100 + factor);
                }
            });
        }
        scope.spawn(|| {
            for _ in 0..12 {
                reload_pass(pass, PASS_MAP.passes[pass].spirv).unwrap();
            }
        });
    });
    lava::wait_idle().unwrap();
}

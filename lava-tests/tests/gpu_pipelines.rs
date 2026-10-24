//! GPU tests of pass pipelines: the pipeline cache file (a foreign one is ignored, the saved one is this device's and loads in a new process), threads that make the same new specialization at once, and `reload_pass` (swap, refusal, races with new specializations, ray tracing passes with their binding tables and specializations)
mod common;

use std::{
    path::{Path, PathBuf},
    sync::{Barrier, LazyLock, Once},
};

use common::{Cmd, GpuGuard, zeroed_buffer};
use glam::{UVec2, Vec2, Vec4};
use lava::{
    Ctx,
    buffer::Buffer,
    command_buffer::{Attachment, CullModeFlags, Draw, RasterState},
    image::{Image, format::R8G8B8A8Unorm, usage::usage},
    pipeline::{Reload, reload_pass, save_cache},
    queue::{Graphics, Timeline},
};
use lava_tests::bindings::*;

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
    let bindings = TestSpecialized::push_bindings(
        TestSpecializedArgs {
            out: out.range(..),
            count: 64,
        },
        TestSpecializedConstants { factor },
        None,
    );
    cmd.compute(&bindings, [1, 1, 1]);
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
        eprintln!("SKIPPED the_pipeline_cache_round_trips: this is its child process");
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
        eprintln!(
            "SKIPPED the_saved_cache_loads: runs only in the child process of the_pipeline_cache_round_trips"
        );
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
        let bindings = TestRaster::push_bindings(
            TestRasterArgs {
                vertices: vertices.range(..),
                instance_offset: Vec2::ZERO,
            },
            None,
        );
        let state = RasterState::default().cull_mode(CullModeFlags::NONE);
        cmd.raster(
            UVec2::splat(SIZE),
            [Attachment::color(
                target.whole_view(),
                Some([0.0, 0.0, 0.0, 1.0]),
            )],
        )
        .draw(&Draw::new(&bindings, 3, 1, state))
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
            let bindings = TestComputeBuffer::push_bindings(
                TestComputeBufferArgs {
                    src: src.range(..),
                    dst: dst.range(..),
                    count: 4,
                    multiplier: 2,
                },
                None,
            );
            cmd.compute(&bindings, [1, 1, 1]);
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
        assert_eq!(
            reload_pass(index(name), pass.spirv).unwrap(),
            Reload::Replaced
        );
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

type TraceTarget = Image<R8G8B8A8Unorm, usage![Storage, TransferSrc]>;

/// Reloading a ray tracing pass rebuilds its pipeline and binding table, and every
/// specialization made so far; traces before and after see the same image.
#[test]
fn reload_pass_replaces_ray_tracing_pipelines() {
    use glam::{Affine3A, Vec3};
    use lava::acceleration_structure::{
        BlasGeometry, BuildFlags, Instance, Instances, Scratch, Triangles,
    };
    let mut gpu = gpu();
    if !Ctx::features().ray_tracing_pipeline {
        eprintln!("SKIPPED reload_pass_replaces_ray_tracing_pipelines: no ray tracing pipelines");
        return;
    }
    const SIZE: u32 = 16;
    let vertices = gpu.buffer(&[
        Vec3::new(-0.9, -0.9, 0.0),
        Vec3::new(0.9, -0.9, 0.0),
        Vec3::new(0.0, 0.9, 0.0),
    ]);
    let geometry = [Triangles::new(vertices.range(..))];
    let blas = lava::Blas::new(&BlasGeometry::Triangles(&geometry), BuildFlags::default()).unwrap();
    let instances = gpu.buffer(&[Instance::new(&blas, Affine3A::IDENTITY)]);
    let tlas = lava::Tlas::new(1, BuildFlags::default()).unwrap();
    let (scratch, tlas_scratch) = (
        Scratch::for_any(&[&blas]).unwrap(),
        Scratch::for_any(&[&tlas]).unwrap(),
    );
    gpu.submit(|cmd| {
        cmd.build_all(&[(&blas, BlasGeometry::Triangles(&geometry))], &scratch);
        // SAFETY: the BLAS is built before it and outlives the traces.
        cmd.build_all(
            &[(&tlas, unsafe { Instances::new(instances.range(..)) })],
            &tlas_scratch,
        );
    });
    let target = TraceTarget::new(UVec2::splat(SIZE))
        .unwrap()
        .bind_storage()
        .unwrap();
    let trace = |gpu: &mut common::Gpu, sky: Option<u32>| {
        let mut readback = zeroed_buffer::<u8>((SIZE * SIZE * 4) as usize);
        gpu.submit(|cmd| {
            let size = UVec2::splat(SIZE);
            match sky {
                Some(sky) => cmd.raytrace(
                    &TestRtConstants::push_bindings(
                        TestRtConstantsArgs {
                            target: target.whole_view(),
                            size,
                        },
                        TestRtConstantsConstants { tlas: &tlas, sky },
                        None,
                    )
                    .unwrap(),
                    [SIZE, SIZE, 1],
                ),
                None => cmd.raytrace(
                    &TestRayTracing::push_bindings(
                        TestRayTracingArgs {
                            target: target.whole_view(),
                            tlas: &tlas,
                            size,
                        },
                        None,
                    )
                    .unwrap(),
                    [SIZE, SIZE, 1],
                ),
            }
            cmd.copy_image_to_buffer(target.whole(), readback.range(..));
        });
        readback.host().to_vec()
    };
    let traces = |gpu: &mut common::Gpu| {
        [
            trace(gpu, None),
            trace(gpu, Some(100)),
            trace(gpu, Some(200)),
        ]
    };
    let before = traces(&mut gpu);
    // A corner misses (the sky), the middle hits the triangle.
    let texel =
        |pixels: &[u8], x: u32, y: u32| pixels[((y * SIZE + x) * 4) as usize..][..4].to_vec();
    assert_eq!(texel(&before[1], 0, 0), [0, 100, 0, 255]);
    assert_eq!(texel(&before[2], 0, 0), [0, 200, 0, 255]);
    assert_eq!(texel(&before[1], 8, 8), [204, 204, 204, 255]);
    for name in ["TestRayTracing", "TestRtConstants"] {
        let pass = index(name);
        assert_eq!(
            reload_pass(pass, PASS_MAP.passes[pass].spirv).unwrap(),
            Reload::Replaced,
            "{name}"
        );
        assert_eq!(traces(&mut gpu), before, "{name}");
        assert!(reload_pass(pass, &[3, 2, 1]).is_err(), "not SPIR-V");
        assert_eq!(traces(&mut gpu), before, "{name}");
    }
    lava::wait_idle().unwrap();
    assert_eq!(traces(&mut gpu), before);
}

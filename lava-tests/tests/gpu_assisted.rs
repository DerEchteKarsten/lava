//! GPU tests under GPU-assisted validation (`Config::gpu_assisted_validation`, its own context, so its own binary): representative passes (buffers through device addresses, a storage image, a bindless texture in a draw, a ray query) whose bindless indices, layouts and address bounds the layer checks on the GPU
mod common;

use common::render::*;
use common::{buffer_with, nearest};
use glam::{Affine3A, UVec2, Vec2, Vec3, Vec4};
use lava::{
    Ctx,
    command_buffer::{Attachment, Draw, RasterState},
    image::{Image, format::R8G8B8A8Unorm, usage::usage},
};
use lava_tests::bindings::*;

/// The harness context with GPU-assisted validation on.
fn gpu() -> common::GpuGuard {
    common::gpu_assisted_validation();
    common::gpu()
}

#[test]
fn buffers_are_read_and_written_through_their_addresses() {
    let mut gpu = gpu();
    let input: Vec<u32> = (1..=100).collect();
    let src = buffer_with::<u32>(&input);
    let mut dst = buffer_with::<u32>(&[u32::MAX; 100]);
    gpu.submit(|cmd| {
        cmd.compute(
            &TestComputeBuffer::push_bindings(
                TestComputeBufferArgs {
                    src: src.range(..),
                    dst: dst.range(..),
                    count: 90,
                    multiplier: 3,
                },
                None,
            ),
            [2, 1, 1],
        )
    });
    let expected: Vec<u32> = (0..100u32)
        .map(|i| match i < 90 {
            true => input[i as usize] * 3 + i,
            false => u32::MAX,
        })
        .collect();
    assert_eq!(dst.host(), expected);
}

#[test]
fn a_storage_image_is_written_through_its_handle() {
    let mut gpu = gpu();
    let pixels = render(&mut gpu, |cmd, target| {
        cmd.compute(
            &TestComputeImage::push_bindings(
                TestComputeImageArgs {
                    target: target.whole_view(),
                    tint: Vec4::ONE,
                    offset: UVec2::ZERO,
                    size: EXTENT,
                },
                None,
            ),
            [SIZE / 8, SIZE / 8, 1],
        )
    });
    // The UV gradient in red and green, an 8x8 checkerboard in blue.
    assert_pixel(&pixels, 0, 0, [2, 2, 0, 255]);
    assert_pixel(&pixels, 63, 0, [253, 2, 255, 255]);
    assert_pixel(&pixels, 63, 63, [253, 253, 0, 255]);
}

#[test]
fn a_draw_samples_a_bindless_texture() {
    let mut gpu = gpu();
    let texture = Image::<R8G8B8A8Unorm, usage![Sampled, TransferDst]>::new(UVec2::new(2, 2))
        .unwrap()
        .bind_sampled()
        .unwrap();
    let texels = buffer_with::<[u8; 4]>(&[
        [255, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
        [255, 255, 0, 255],
    ]);
    let corner = |x: f32, y: f32, u: f32, v: f32| TestTexturedVertex {
        position: Vec2::new(x, y),
        uv: Vec2::new(u, v),
    };
    let vertices = gpu.buffer(&[
        corner(-0.5, -0.5, 0.0, 0.0),
        corner(0.5, -0.5, 1.0, 0.0),
        corner(0.5, 0.5, 1.0, 1.0),
        corner(-0.5, 0.5, 0.0, 1.0),
    ]);
    let indices = gpu.buffer(&[0u32, 1, 2, 0, 2, 3]);
    let pixels = render(&mut gpu, |cmd, target| {
        cmd.copy_buffer_to_image(texels.range(..), texture.whole());
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::indexed(
                &TestRasterTextured::push_bindings(
                    TestRasterTexturedArgs {
                        vertices: vertices.range(..),
                        tex: texture.whole_view(),
                        sampler: nearest(),
                    },
                    None,
                ),
                indices.range(..),
                1,
                RasterState::default(),
            ))
            .record("textured");
    });
    assert_pixel(&pixels, 20, 20, [255, 0, 0, 255]);
    assert_pixel(&pixels, 44, 20, [0, 255, 0, 255]);
    assert_pixel(&pixels, 20, 44, [0, 0, 255, 255]);
    assert_pixel(&pixels, 44, 44, [255, 255, 0, 255]);
    assert_pixel(&pixels, 4, 4, BLACK);
}

#[test]
fn a_ray_query_traces_a_tlas() {
    use lava::acceleration_structure::{
        BlasGeometry, BuildFlags, Instance, Instances, Scratch, Triangles,
    };
    let mut gpu = gpu();
    if !Ctx::features().ray_query {
        eprintln!("SKIPPED a_ray_query_traces_a_tlas: no ray queries");
        return;
    }
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
    let pixels = render(&mut gpu, |cmd, target| {
        cmd.build_all(&[(&blas, BlasGeometry::Triangles(&geometry))], &scratch);
        // SAFETY: the BLAS is built before it and outlives the trace.
        cmd.build_all(
            &[(&tlas, unsafe { Instances::new(instances.range(..)) })],
            &tlas_scratch,
        );
        cmd.compute(
            &TestRayQuery::push_bindings(
                TestRayQueryArgs {
                    target: target.whole_view(),
                    tlas: &tlas,
                    size: EXTENT,
                },
                None,
            )
            .unwrap(),
            [SIZE / 8, SIZE / 8, 1],
        );
    });
    // Instance 0's grey in the middle, nothing in a corner.
    assert_pixel(&pixels, 32, 32, [204, 204, 204, 255]);
    assert_pixel(&pixels, 1, 1, BLACK);
}

/// The layer instruments the shaders: reads past a buffer's end (16 bytes, read 16 KiB far) are
/// reported, so the passes above ran checked.
#[test]
fn reads_past_a_buffer_are_reported() {
    let mut gpu = gpu();
    let src = buffer_with::<u32>(&[1, 2, 3, 4]);
    let dst = buffer_with::<u32>(&[0; 4096]);
    gpu.submit(|cmd| {
        cmd.compute(
            &TestComputeBuffer::push_bindings(
                TestComputeBufferArgs {
                    src: src.range(..),
                    dst: dst.range(..),
                    count: 4096,
                    multiplier: 1,
                },
                None,
            ),
            [64, 1, 1],
        )
    });
    // Taken here, so the guard doesn't fail the test on them.
    let errors = common::take_validation_errors();
    assert!(
        !errors.is_empty() && errors.iter().all(|e| e.contains("Out of bounds")),
        "GPU-AV reports the reads past `src`: {errors:#?}"
    );
}

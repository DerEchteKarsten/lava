//! Rendering helpers the GPU tests share: a cleared 64x64 RGBA8 target drawn into and read back, pixel checks, and the test triangles
use glam::{UVec2, Vec4};
use lava::{
    buffer::Buffer,
    image::{Image, format::R8G8B8A8Unorm, usage::usage},
};
use lava_tests::bindings::TestVertex;

use super::{Cmd, Gpu, zeroed_buffer};

pub const SIZE: u32 = 64;
pub const EXTENT: UVec2 = UVec2::splat(SIZE);
pub const BLACK: [u8; 4] = [0, 0, 0, 255];

pub type Target = Image<R8G8B8A8Unorm, usage![ColorAttachment, Storage, TransferSrc, TransferDst]>;

/// Clears a fresh target to opaque black, lets `record` draw into it, and reads it back.
pub fn render(gpu: &mut Gpu, record: impl FnOnce(&mut Cmd, &Target)) -> Vec<u8> {
    render_cleared(gpu, [0.0, 0.0, 0.0, 1.0], record)
}

/// Like [`render`], cleared to `clear`.
pub fn render_cleared(
    gpu: &mut Gpu,
    clear: [f32; 4],
    record: impl FnOnce(&mut Cmd, &Target),
) -> Vec<u8> {
    let target = Target::new(EXTENT).unwrap().bind_storage().unwrap();
    let mut readback = zeroed_buffer::<u8>((SIZE * SIZE * 4) as usize);
    gpu.submit(|cmd| {
        cmd.clear_image(target.whole_view(), clear);
        record(cmd, &target);
        cmd.copy_image_to_buffer(target.whole(), readback.range(..));
    });
    readback.host().to_vec()
}

pub fn pixel(pixels: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * SIZE + x) * 4) as usize;
    pixels[i..i + 4].try_into().unwrap()
}

/// Equal up to rounding: at most 2 apart per channel.
pub fn close(a: [u8; 4], b: [u8; 4]) -> bool {
    a.iter().zip(b).all(|(a, b)| a.abs_diff(b) <= 2)
}

#[track_caller]
pub fn assert_pixel(pixels: &[u8], x: u32, y: u32, expected: [u8; 4]) {
    let actual = pixel(pixels, x, y);
    assert!(
        close(actual, expected),
        "pixel ({x}, {y}) is {actual:?}, expected {expected:?}"
    );
}

pub fn vertex(x: f32, y: f32, z: f32, color: [f32; 4]) -> TestVertex {
    TestVertex {
        position: Vec4::new(x, y, z, 1.0),
        color: Vec4::from_array(color),
    }
}

pub const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
pub const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
pub const BLUE: [f32; 4] = [0.0, 0.0, 1.0, 1.0];
pub const YELLOW: [f32; 4] = [1.0, 1.0, 0.0, 1.0];

/// Two triangles: a front-facing one (clockwise on screen) in the left half with one colour
/// per corner, and a back-facing yellow one in the right half.
pub fn two_triangles(gpu: &mut Gpu) -> Buffer<TestVertex> {
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
pub const IN_LEFT: (u32, u32) = (16, 44);
pub const IN_RIGHT: (u32, u32) = (48, 44);

/// A near green triangle (depth 0.8: lava's default depth test is `GREATER`, reversed depth)
/// drawn before a far red one (0.2) covering the same area.
pub fn overlapping_triangles(gpu: &mut Gpu) -> Buffer<TestVertex> {
    gpu.buffer(&[
        vertex(0.0, -0.8, 0.8, GREEN),
        vertex(0.8, 0.8, 0.8, GREEN),
        vertex(-0.8, 0.8, 0.8, GREEN),
        vertex(0.0, -0.8, 0.2, RED),
        vertex(0.8, 0.8, 0.2, RED),
        vertex(-0.8, 0.8, 0.2, RED),
    ])
}

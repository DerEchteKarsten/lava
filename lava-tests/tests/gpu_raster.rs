//! GPU tests of renderings and draws: raster state on the GPU (provoking vertices, conservative rasterization, wide, rectangular, Bresenham, smooth and stippled lines, depth bounds, blend constants, strips, fans, points and line strips), attachments a draw also accesses or samples (a read-only depth attachment), raster specialization constants, task shaders, indirect draws of several commands and records that start with one, indexed indirect and counted draws, `u16` indices, first vertices and instances, multi-draws, depth bias clamps, per-draw scissors and viewports, render area offsets, vertex-only passes, stencil-only attachments, unordered accesses, array layers (sampled, copied, attached one at a time), layered and multiview renderings, dual-source blending, the depth clip default, and the checks of draw state values
mod common;

use common::render::*;
use common::{Cmd, Gpu, buffer_with, gpu, nearest, panic_message, zeroed_buffer};
use glam::{IVec2, UVec2, Vec2, Vec4};
use lava::{
    Ctx,
    buffer::Buffer,
    command_buffer::{
        Accesses, Attachment, BindingOutput, Blend, BlendEquation, BlendFactor, BlendOp,
        BufferAccess, CompareOp, CullModeFlags, DepthBias, Draw, DrawIndexedIndirectCommand,
        DrawIndirectCommand, ImageUse, LineStipple, PrimitiveTopology, RasterState, Rect,
        StartsWith, Stencil, StencilFace, StencilOp, kind,
    },
    image::{
        D2Array, Image,
        format::{D32Sfloat, R8G8B8A8Unorm, R32Uint, S8Uint},
        usage::usage,
    },
    vk,
};
use lava_tests::bindings::*;

// ---- helpers ---------------------------------------------------------------------------------

type ArrayTarget = Image<R8G8B8A8Unorm, usage![ColorAttachment, TransferSrc, TransferDst], D2Array>;

/// Like [`render`], into an array of `layers` layers, read back one layer after another (a
/// copy of the whole array view).
fn render_layers(
    gpu: &mut Gpu,
    layers: u32,
    record: impl FnOnce(&mut Cmd, &ArrayTarget),
) -> Vec<Vec<u8>> {
    let target = ArrayTarget::with_layers(EXTENT, 1, layers).unwrap();
    let layer_bytes = (SIZE * SIZE * 4) as usize;
    let mut readback = zeroed_buffer::<u8>(layer_bytes * layers as usize);
    gpu.submit(|cmd| {
        cmd.clear_image(target.whole_view(), [0.0, 0.0, 0.0, 1.0]);
        record(cmd, &target);
        cmd.copy_image_to_buffer(target.whole(), readback.range(..));
    });
    readback
        .host()
        .chunks_exact(layer_bytes)
        .map(<[u8]>::to_vec)
        .collect()
}

fn raster(
    vertices: &Buffer<TestVertex>,
) -> impl std::ops::Deref<Target = BindingOutput<'_, kind::RasterVertex>> {
    TestRaster::push_bindings(
        TestRasterArgs {
            vertices: vertices.range(..),
            instance_offset: Vec2::ZERO,
        },
        None,
    )
}

fn unculled<const N: usize>() -> RasterState<N> {
    RasterState::default().cull_mode(CullModeFlags::NONE)
}

/// Both triangles, unculled, by a plain draw of `TestRaster`.
fn both(gpu: &mut Gpu, vertices: &Buffer<TestVertex>) -> Vec<u8> {
    render(gpu, |cmd, target| {
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::new(&raster(vertices), 6, 1, unculled()))
            .record("both");
    })
}

fn skip(test: &str, why: &str) {
    eprintln!("SKIPPED {test}: {why}");
}

// ---- draws -----------------------------------------------------------------------------------

#[test]
fn indirect_draws_take_several_commands_and_a_first_instance() {
    let mut gpu = gpu();
    let vertices = two_triangles(&mut gpu);
    // The left triangle, then again as instance 1, which the shader shifts right.
    let commands = buffer_with(&[
        DrawIndirectCommand {
            vertex_count: 3,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
        },
        DrawIndirectCommand {
            vertex_count: 3,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 1,
        },
    ]);
    let shifted = TestDrawParams::push_bindings(
        TestDrawParamsArgs {
            vertices: vertices.range(..),
            instance_offset: Vec2::new(1.0, 0.0),
        },
        None,
    );
    let indirect = render(&mut gpu, |cmd, target| {
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::indirect(
                &shifted,
                commands.range(..),
                RasterState::default(),
            ))
            .record("indirect");
    });
    let instanced = render(&mut gpu, |cmd, target| {
        let bindings = TestRaster::push_bindings(
            TestRasterArgs {
                vertices: vertices.range(..),
                instance_offset: Vec2::new(1.0, 0.0),
            },
            None,
        );
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::new(&bindings, 3, 2, RasterState::default()))
            .record("instanced");
    });
    assert_ne!(pixel(&indirect, IN_RIGHT.0, IN_RIGHT.1), BLACK);
    assert_eq!(indirect, instanced);
}

/// A culling pass's record: the command first, then what else the pass keeps per draw.
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Record {
    command: DrawIndirectCommand,
    id: u32,
}

// SAFETY: `#[repr(C)]` with the command at offset 0.
unsafe impl StartsWith<DrawIndirectCommand> for Record {}

#[test]
fn indirect_draws_take_records_that_start_with_their_command() {
    let mut gpu = gpu();
    let vertices = two_triangles(&mut gpu);
    let command = |first_vertex| DrawIndirectCommand {
        vertex_count: 3,
        instance_count: 1,
        first_vertex,
        first_instance: 0,
    };
    let records = buffer_with(&[
        Record {
            command: command(0),
            id: 7,
        },
        Record {
            command: command(3),
            id: 9,
        },
    ]);
    let bindings = TestDrawParams::push_bindings(
        TestDrawParamsArgs {
            vertices: vertices.range(..),
            instance_offset: Vec2::ZERO,
        },
        None,
    );
    let pixels = render(&mut gpu, |cmd, target| {
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::indirect(&bindings, records.range(..), unculled()))
            .record("records");
    });
    assert_eq!(pixels, both(&mut gpu, &vertices));
}

#[test]
fn direct_draws_start_at_their_first_vertex_and_instance() {
    let mut gpu = gpu();
    let vertices = two_triangles(&mut gpu);
    let shifted = TestDrawParams::push_bindings(
        TestDrawParamsArgs {
            vertices: vertices.range(..),
            instance_offset: Vec2::new(-1.0, 0.0),
        },
        None,
    );
    // The right triangle (vertices 3..6) as instance 1: shifted into the left half.
    let pixels = render(&mut gpu, |cmd, target| {
        let draw = Draw::new(&shifted, 3, 1, unculled())
            .first_vertex(3)
            .first_instance(1);
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&draw)
            .record("first vertex");
    });
    assert_pixel(&pixels, IN_LEFT.0, IN_LEFT.1, [255, 255, 0, 255]);
    assert_pixel(&pixels, IN_RIGHT.0, IN_RIGHT.1, BLACK);
    let message = panic_message(|| {
        let commands = buffer_with(&[DrawIndirectCommand::default()]);
        let _ = Draw::indirect(&shifted, commands.range(..), unculled::<1>()).first_vertex(1);
    });
    assert!(message.contains("is for `Draw::new`"), "{message}");
}

#[test]
fn indexed_draws_take_u16_indices_and_a_vertex_offset() {
    let mut gpu = gpu();
    let vertices = two_triangles(&mut gpu);
    let bindings = TestDrawParams::push_bindings(
        TestDrawParamsArgs {
            vertices: vertices.range(..),
            instance_offset: Vec2::ZERO,
        },
        None,
    );
    let wide = buffer_with::<u32>(&[0, 1, 2, 3, 4, 5]);
    let narrow = buffer_with::<u16>(&[0, 1, 2]);
    let u32_indices = render(&mut gpu, |cmd, target| {
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::indexed(&bindings, wide.range(3..), 1, unculled()))
            .record("u32");
    });
    // The same triangle through `u16` indices of the left one, offset by 3 vertices.
    let u16_indices = render(&mut gpu, |cmd, target| {
        let draw = Draw::indexed(&bindings, narrow.range(..), 1, unculled()).vertex_offset(3);
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&draw)
            .record("u16");
    });
    assert_pixel(&u16_indices, IN_RIGHT.0, IN_RIGHT.1, [255, 255, 0, 255]);
    assert_pixel(&u16_indices, IN_LEFT.0, IN_LEFT.1, BLACK);
    assert_eq!(u16_indices, u32_indices);
}

#[test]
fn indexed_indirect_draws_match_the_direct_ones() {
    let mut gpu = gpu();
    let vertices = two_triangles(&mut gpu);
    let bindings = TestDrawParams::push_bindings(
        TestDrawParamsArgs {
            vertices: vertices.range(..),
            instance_offset: Vec2::ZERO,
        },
        None,
    );
    let indices = buffer_with::<u16>(&[0, 1, 2, 0, 1, 2]);
    let command = |first_index, vertex_offset| DrawIndexedIndirectCommand {
        index_count: 3,
        instance_count: 1,
        first_index,
        vertex_offset,
        first_instance: 0,
    };
    // Both triangles: the second command's indices name the left one, offset to the right.
    let commands = buffer_with(&[command(0, 0), command(3, 3)]);
    let indirect = render(&mut gpu, |cmd, target| {
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::indexed_indirect(
                &bindings,
                indices.range(..),
                commands.range(..),
                unculled(),
            ))
            .record("indexed indirect");
    });
    assert_eq!(indirect, both(&mut gpu, &vertices));

    let counted = |gpu: &mut Gpu, count: u32| {
        let count = buffer_with::<u32>(&[count]);
        render(gpu, |cmd, target| {
            cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
                .draw(&Draw::indexed_indirect_count(
                    &bindings,
                    indices.range(..),
                    commands.range(..),
                    count.range(..),
                    unculled(),
                ))
                .record("indexed indirect count");
        })
    };
    assert_eq!(counted(&mut gpu, 2), indirect);
    let one = counted(&mut gpu, 1);
    assert_ne!(pixel(&one, IN_LEFT.0, IN_LEFT.1), BLACK);
    assert_pixel(&one, IN_RIGHT.0, IN_RIGHT.1, BLACK);
}

#[test]
fn multi_draws_draw_every_range() {
    let mut gpu = gpu();
    if !Ctx::features().multi_draw {
        return skip(
            "multi_draws_draw_every_range",
            "the device lacks `multi-draw`",
        );
    }
    let vertices = two_triangles(&mut gpu);
    let bindings = TestDrawParams::push_bindings(
        TestDrawParamsArgs {
            vertices: vertices.range(..),
            instance_offset: Vec2::ZERO,
        },
        None,
    );
    let ranges = [
        lava::command_buffer::MultiDraw {
            first_vertex: 0,
            vertex_count: 3,
        },
        lava::command_buffer::MultiDraw {
            first_vertex: 3,
            vertex_count: 3,
        },
    ];
    let multi = render(&mut gpu, |cmd, target| {
        let draw = Draw::multi(&bindings, &ranges, 1, unculled()).unwrap();
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&draw)
            .record("multi");
    });
    let expected = both(&mut gpu, &vertices);
    assert_eq!(multi, expected);

    let indices = buffer_with::<u32>(&[0, 1, 2]);
    let indexed = [
        lava::command_buffer::MultiDrawIndexed {
            first_index: 0,
            index_count: 3,
            vertex_offset: 0,
        },
        lava::command_buffer::MultiDrawIndexed {
            first_index: 0,
            index_count: 3,
            vertex_offset: 3,
        },
    ];
    let multi_indexed = render(&mut gpu, |cmd, target| {
        let draw =
            Draw::multi_indexed(&bindings, indices.range(..), &indexed, 1, unculled()).unwrap();
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&draw)
            .record("multi indexed");
    });
    assert_eq!(multi_indexed, expected);
    let outside = [lava::command_buffer::MultiDrawIndexed {
        first_index: 1,
        index_count: 3,
        vertex_offset: 0,
    }];
    let message = panic_message(|| {
        Draw::multi_indexed(&bindings, indices.range(..), &outside, 1, unculled::<1>()).map(drop)
    });
    assert!(message.contains("of an index buffer of 3"), "{message}");
}

// ---- rasterization state ---------------------------------------------------------------------

#[test]
fn depth_bias_is_clamped() {
    let mut gpu = gpu();
    // A green triangle a little nearer than a red one at the same place, drawn first.
    let vertices = gpu.buffer(&[
        vertex(0.0, -0.8, 0.501, GREEN),
        vertex(0.8, 0.8, 0.501, GREEN),
        vertex(-0.8, 0.8, 0.501, GREEN),
        vertex(0.0, -0.8, 0.5, RED),
        vertex(0.8, 0.8, 0.5, RED),
        vertex(-0.8, 0.8, 0.5, RED),
    ]);
    let depth = Image::<D32Sfloat, usage![DepthAttachment]>::new(EXTENT).unwrap();
    let biased = |gpu: &mut Gpu, clamp: f32| {
        // About 0.006 at a depth of 0.5 (2^-24 per unit), more than the triangles' distance.
        let bias = DepthBias {
            constant: 1e5,
            slope: 0.0,
            clamp,
        };
        let green = TestRaster::push_bindings(
            TestRasterArgs {
                vertices: vertices.range(0..3),
                instance_offset: Vec2::ZERO,
            },
            None,
        );
        let red = TestRaster::push_bindings(
            TestRasterArgs {
                vertices: vertices.range(3..6),
                instance_offset: Vec2::ZERO,
            },
            None,
        );
        render(gpu, |cmd, target| {
            cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
                .depth_attachment(depth.whole_view(), Some([0.0]))
                .draw(&Draw::new(&green, 3, 1, RasterState::default()))
                .draw(&Draw::new(
                    &red,
                    3,
                    1,
                    RasterState::default().depth_bias(Some(bias)),
                ))
                .record("depth bias");
        })
    };
    // Unclamped, the bias lifts the red triangle in front; clamped below the distance, it stays
    // behind.
    assert_pixel(&biased(&mut gpu, 0.0), 32, 40, [255, 0, 0, 255]);
    assert_pixel(&biased(&mut gpu, 1e-4), 32, 40, [0, 255, 0, 255]);
}

#[test]
fn draws_set_their_own_scissors_and_viewports() {
    let mut gpu = gpu();
    let vertices = two_triangles(&mut gpu);
    let full = both(&mut gpu, &vertices);
    let half = SIZE / 2;
    let top = Rect::new(IVec2::ZERO, UVec2::new(SIZE, half));
    let bottom = Rect::new(IVec2::new(0, half as i32), UVec2::new(SIZE, half));
    // The left triangle only in the top half, the right one only in the bottom half, and a
    // third draw back at the rendering's scissor.
    let left = TestRaster::push_bindings(
        TestRasterArgs {
            vertices: vertices.range(0..3),
            instance_offset: Vec2::ZERO,
        },
        None,
    );
    let right = TestRaster::push_bindings(
        TestRasterArgs {
            vertices: vertices.range(3..6),
            instance_offset: Vec2::ZERO,
        },
        None,
    );
    let pixels = render(&mut gpu, |cmd, target| {
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::new(&left, 3, 1, unculled().scissor(Some(top))))
            .draw(&Draw::new(&right, 3, 1, unculled().scissor(Some(bottom))))
            .record("scissors");
    });
    for (y, x) in (0..SIZE).flat_map(|y| (0..SIZE).map(move |x| (y, x))) {
        let expected = match (x < half, y < half) {
            (true, true) | (false, false) => pixel(&full, x, y),
            _ => BLACK,
        };
        assert_eq!(pixel(&pixels, x, y), expected, "({x}, {y})");
    }

    // One draw squeezed into the top-left quarter, the next at the rendering's viewport.
    let quarter = Rect::new(IVec2::ZERO, EXTENT / 2);
    let viewports = render(&mut gpu, |cmd, target| {
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::new(&right, 3, 1, unculled().viewport(Some(quarter))))
            .draw(&Draw::new(&left, 3, 1, unculled()))
            .record("viewports");
    });
    assert_pixel(
        &viewports,
        IN_RIGHT.0 / 2,
        IN_RIGHT.1 / 2,
        [255, 255, 0, 255],
    );
    assert_pixel(&viewports, IN_RIGHT.0, IN_RIGHT.1, BLACK);
    assert_eq!(
        pixel(&viewports, IN_LEFT.0, IN_LEFT.1),
        pixel(&full, IN_LEFT.0, IN_LEFT.1)
    );
}

#[test]
fn a_render_area_offset_leaves_the_rest_untouched() {
    let mut gpu = gpu();
    let vertices = two_triangles(&mut gpu);
    let target = Target::new(EXTENT).unwrap();
    let mut readback = zeroed_buffer::<u8>((SIZE * SIZE * 4) as usize);
    let area = Rect::new(IVec2::new(SIZE as i32 / 2, 0), UVec2::new(SIZE / 2, SIZE));
    gpu.submit(|cmd| {
        cmd.clear_image(target.whole_view(), BLUE);
        // Cleared to black inside the area only; the viewport is the area, so both triangles
        // land in the right half.
        cmd.raster(
            area,
            [Attachment::color(
                target.whole_view(),
                Some([0.0, 0.0, 0.0, 1.0]),
            )],
        )
        .draw(&Draw::new(&raster(&vertices), 6, 1, unculled()))
        .record("offset area");
        cmd.copy_image_to_buffer(target.whole(), readback.range(..));
    });
    let pixels = readback.host().to_vec();
    for y in 0..SIZE {
        for x in 0..SIZE / 2 {
            assert_eq!(pixel(&pixels, x, y), [0, 0, 255, 255], "({x}, {y})");
        }
    }
    assert_pixel(&pixels, 32 + IN_RIGHT.0 / 2, IN_RIGHT.1, [255, 255, 0, 255]);
    assert_ne!(pixel(&pixels, 32 + IN_LEFT.0 / 2, IN_LEFT.1), BLACK);
    let message = panic_message(|| {
        let wide = Rect::new(IVec2::new(1, 0), EXTENT);
        gpu.submit(|cmd| drop(cmd.raster(wide, [Attachment::color(target.whole_view(), None)])));
    });
    assert!(
        message.contains("doesn't fit its color attachment"),
        "{message}"
    );
}

#[test]
fn depth_clip_follows_the_clamp_unless_set() {
    let mut gpu = gpu();
    if !Ctx::features().depth_clamp {
        return skip(
            "depth_clip_follows_the_clamp_unless_set",
            "the device lacks `depth-clamp`",
        );
    }
    // Beyond the far plane: clipped, unless depths are clamped.
    let far = gpu.buffer(&[
        vertex(0.0, -0.8, 1.5, GREEN),
        vertex(0.8, 0.8, 1.5, GREEN),
        vertex(-0.8, 0.8, 1.5, GREEN),
    ]);
    let draw = |gpu: &mut Gpu, state: RasterState<1>| {
        render(gpu, |cmd, target| {
            cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
                .draw(&Draw::new(&raster(&far), 3, 1, state))
                .record("far");
        })
    };
    assert_pixel(&draw(&mut gpu, RasterState::default()), 32, 40, BLACK);
    let clamped = RasterState::default().depth_clamp(true).unwrap();
    assert_pixel(&draw(&mut gpu, clamped), 32, 40, [0, 255, 0, 255]);
    if Ctx::features().depth_clip_enable {
        let clipped = clamped.depth_clip(true).unwrap();
        assert_pixel(&draw(&mut gpu, clipped), 32, 40, BLACK);
    }
}

#[test]
fn dual_source_blends_read_the_second_output() {
    let mut gpu = gpu();
    if !Ctx::features().dual_src_blend {
        return skip(
            "dual_source_blends_read_the_second_output",
            "the device lacks `dual-src-blend`",
        );
    }
    let vertices = gpu.buffer(&[
        vertex(0.0, -0.8, 0.5, RED),
        vertex(0.8, 0.8, 0.5, RED),
        vertex(-0.8, 0.8, 0.5, RED),
    ]);
    // `src * factor + dst * (1 - factor)`, the factor the second output.
    let mix = BlendEquation {
        src_color: BlendFactor::SRC1_COLOR,
        dst_color: BlendFactor::ONE_MINUS_SRC1_COLOR,
        color_op: BlendOp::ADD,
        src_alpha: BlendFactor::ONE,
        dst_alpha: BlendFactor::ZERO,
        alpha_op: BlendOp::ADD,
    };
    let bindings = TestDualSource::push_bindings(
        TestDualSourceArgs {
            vertices: vertices.range(..),
            factor: Vec4::new(0.25, 0.5, 0.5, 1.0),
        },
        None,
    );
    let pixels = render(&mut gpu, |cmd, target| {
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), Some(BLUE))])
            .draw(&Draw::new(
                &bindings,
                3,
                1,
                unculled().blend(Blend::Equation(mix)),
            ))
            .record("dual source");
    });
    // Red at a quarter over blue at three quarters, green 0 either way, blue at a half.
    assert_pixel(&pixels, 32, 40, [64, 0, 128, 255]);

    // A second attachment can't read a second output on this device (one, the limit).
    if Ctx::device_properties()
        .limits
        .max_fragment_dual_src_attachments
        < 2
    {
        let message = panic_message(|| {
            let _ = Draw::new(
                &bindings,
                3,
                1,
                RasterState::<2>::default().blend(Blend::Equation(mix)),
            );
        });
        assert!(
            message.contains("maxFragmentDualSrcAttachments"),
            "{message}"
        );
    }
}

#[test]
fn draw_state_values_are_checked() {
    let mut gpu = gpu();
    let vertices = two_triangles(&mut gpu);
    let bindings = raster(&vertices);
    let check = |state: RasterState<1>| {
        panic_message(|| {
            let _ = Draw::new(&bindings, 3, 1, state);
        })
    };

    let restart = RasterState::default().primitive_restart(true);
    let message = check(restart);
    assert!(
        message.contains("primitive restart cuts strips and fans"),
        "{message}"
    );
    let _ = Draw::new(
        &bindings,
        3,
        1,
        restart.topology(PrimitiveTopology::TRIANGLE_STRIP),
    );

    if Ctx::features().line_stipple {
        let stipple = Some(LineStipple {
            factor: 0,
            pattern: !0,
        });
        let message = check(RasterState::default().line_stipple(stipple).unwrap());
        assert!(message.contains("1 to 256 times"), "{message}");
    }
    if Ctx::features().wide_lines {
        let [_, max] = Ctx::device_properties().limits.line_width_range;
        let message = check(RasterState::default().line_width(max * 2.0).unwrap());
        assert!(message.contains("lineWidthRange"), "{message}");
    }
    if Ctx::features().depth_bounds {
        let message = check(
            RasterState::default()
                .depth_bounds(Some([0.0, 2.0]))
                .unwrap(),
        );
        assert!(
            message.contains("depth bounds lie between 0 and 1"),
            "{message}"
        );
    }
    if Ctx::features().depth_clamp_control {
        let message = check(RasterState::default().depth_clamp_range(0.8, 0.2).unwrap());
        assert!(message.contains("depth clamp range"), "{message}");
    }
    if Ctx::features().conservative_rasterization {
        // Refused unless the device has `primitiveUnderestimation` (the Arc B390 hasn't).
        let mode = vk::ConservativeRasterizationModeEXT::UNDERESTIMATE;
        let state = RasterState::<1>::default().conservative(mode).unwrap();
        let drawn = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = Draw::new(&bindings, 3, 1, state);
        }));
        if let Err(payload) = drawn {
            let message = common::payload_message(&*payload);
            assert!(message.contains("primitiveUnderestimation"), "{message}");
        }
    }
    let message = check(RasterState::default().viewport(Some(Rect::new(IVec2::ZERO, UVec2::ZERO))));
    assert!(message.contains("has no pixels"), "{message}");
    let message = check(RasterState::default().scissor(Some(Rect::new(IVec2::new(-1, 0), EXTENT))));
    assert!(message.contains("non-negative"), "{message}");
}

#[test]
fn blending_into_a_format_that_cant_blend_panics() {
    let mut gpu = gpu();
    let vertices = two_triangles(&mut gpu);
    let bindings = raster(&vertices);
    let target = Image::<R32Uint, usage![ColorAttachment]>::new(EXTENT).unwrap();
    let message = panic_message(|| {
        gpu.submit(|cmd| {
            let draw = Draw::new(&bindings, 3, 1, RasterState::default().blend(Blend::Alpha));
            cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
                .draw(&draw)
                .record("integer blend");
        });
    });
    assert!(
        message.contains("doesn't blend on this device"),
        "{message}"
    );
}

// ---- raster state ----------------------------------------------------------------------------

fn flat(
    vertices: &Buffer<TestVertex>,
) -> impl std::ops::Deref<Target = BindingOutput<'_, kind::RasterVertex>> {
    TestFlat::push_bindings(
        TestFlatArgs {
            vertices: vertices.range(..),
        },
        None,
    )
}

/// Draws `count` vertices of `vertices` with `state` through the flat-shaded pass.
fn draw_flat(
    gpu: &mut Gpu,
    vertices: &Buffer<TestVertex>,
    count: u32,
    state: RasterState<1>,
) -> Vec<u8> {
    render(gpu, |cmd, target| {
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::new(&flat(vertices), count, 1, state))
            .record("flat");
    })
}

/// The clip-space coordinate of the centre of pixel `p` (of 64).
fn centre(p: u32) -> f32 {
    -1.0 + (p as f32 + 0.5) / (SIZE as f32 / 2.0)
}

const WHITE: [u8; 4] = [255; 4];

/// A horizontal line through the centres of row `row`, from `x0` to `x1` in clip space.
fn row_line(gpu: &mut Gpu, row: u32, x0: f32, x1: f32) -> Buffer<TestVertex> {
    let y = centre(row);
    gpu.buffer(&[vertex(x0, y, 0.5, [1.0; 4]), vertex(x1, y, 0.5, [1.0; 4])])
}

#[test]
fn flat_shading_takes_the_provoking_vertex() {
    let mut gpu = gpu();
    let vertices = two_triangles(&mut gpu);
    // The left triangle's vertices are red, green and blue.
    let first = draw_flat(&mut gpu, &vertices, 3, unculled());
    assert_pixel(&first, IN_LEFT.0, IN_LEFT.1, [255, 0, 0, 255]);
    let Ok(last) = unculled().provoking_vertex(vk::ProvokingVertexModeEXT::LAST_VERTEX) else {
        return skip(
            "flat_shading_takes_the_provoking_vertex",
            "the device lacks `provoking-vertex`",
        );
    };
    let last = draw_flat(&mut gpu, &vertices, 3, last);
    assert_pixel(&last, IN_LEFT.0, IN_LEFT.1, [0, 0, 255, 255]);
    // Back to the first in the same rendering: the change is per draw.
    let both = render(&mut gpu, |cmd, target| {
        let last = unculled()
            .provoking_vertex(vk::ProvokingVertexModeEXT::LAST_VERTEX)
            .unwrap();
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::new(&flat(&vertices), 3, 1, last))
            .draw(&Draw::new(&flat(&vertices), 3, 1, unculled()))
            .record("last then first");
    });
    assert_eq!(both, first);
}

#[test]
fn conservative_rasterization_covers_touched_pixels() {
    let mut gpu = gpu();
    // A sliver in the corner of pixel (10, 10) that misses its centre.
    let corner = -1.0 + 10.0 / 32.0;
    let sliver = gpu.buffer(&[
        vertex(corner + 0.001, corner + 0.001, 0.5, [1.0; 4]),
        vertex(corner + 0.006, corner + 0.001, 0.5, [1.0; 4]),
        vertex(corner + 0.001, corner + 0.006, 0.5, [1.0; 4]),
    ]);
    let plain = draw_flat(&mut gpu, &sliver, 3, unculled());
    assert!(plain.as_chunks::<4>().0.iter().all(|&p| p == BLACK));
    let mode = vk::ConservativeRasterizationModeEXT::OVERESTIMATE;
    let Ok(over) = unculled().conservative(mode) else {
        return skip(
            "conservative_rasterization_covers_touched_pixels",
            "the device lacks `conservative-rasterization`",
        );
    };
    let over = draw_flat(&mut gpu, &sliver, 3, over);
    assert_pixel(&over, 10, 10, WHITE);
    assert_pixel(&over, 12, 12, BLACK);
}

#[test]
fn wide_lines_cover_their_width() {
    let mut gpu = gpu();
    let line = row_line(&mut gpu, 20, -0.9, 0.9);
    let lines = || unculled().topology(PrimitiveTopology::LINE_LIST);
    let thin = draw_flat(&mut gpu, &line, 2, lines());
    assert_pixel(&thin, 32, 20, WHITE);
    assert_pixel(&thin, 32, 18, BLACK);
    assert_pixel(&thin, 32, 22, BLACK);
    let [_, max] = Ctx::device_properties().limits.line_width_range;
    let Ok(wide) = lines().line_width(5.0) else {
        return skip(
            "wide_lines_cover_their_width",
            "the device lacks `wide-lines`",
        );
    };
    if max < 5.0 {
        return skip("wide_lines_cover_their_width", "lines can't be 5 wide");
    }
    let wide = draw_flat(&mut gpu, &line, 2, wide);
    for row in 18..=22 {
        assert_pixel(&wide, 32, row, WHITE);
    }
    assert_pixel(&wide, 32, 16, BLACK);
    assert_pixel(&wide, 32, 24, BLACK);
}

#[test]
fn line_modes_and_stipples_draw_their_lines() {
    let mut gpu = gpu();
    let line = row_line(&mut gpu, 20, -1.0, 1.0);
    let lines = || unculled().topology(PrimitiveTopology::LINE_LIST);
    let test = "line_modes_and_stipples_draw_their_lines";
    let modes = [
        ("rectangular", lines().rectangular_lines().ok()),
        ("bresenham", lines().bresenham_lines().ok()),
        ("smooth", lines().smooth_lines().ok()),
    ];
    for (name, state) in modes {
        let Some(state) = state else {
            skip(test, &format!("the device lacks `{name}-lines`"));
            continue;
        };
        let pixels = draw_flat(&mut gpu, &line, 2, state);
        // Smooth lines fade at their edges; their centre row is lit.
        assert_ne!(pixel(&pixels, 32, 20), BLACK, "{name} lines");
        assert_pixel(&pixels, 32, 23, BLACK);
    }
    // Eight pixels on, eight off, from the line's start.
    let stipple = Some(LineStipple {
        factor: 1,
        pattern: 0x00ff,
    });
    let Ok(stippled) = lines().line_stipple(stipple) else {
        return skip(test, "the device lacks `line-stipple`");
    };
    let pixels = draw_flat(&mut gpu, &line, 2, stippled);
    for x in [2, 5, 18, 21] {
        assert_pixel(&pixels, x, 20, WHITE);
    }
    for x in [10, 13, 26, 29] {
        assert_pixel(&pixels, x, 20, BLACK);
    }
    // Twice as long with a factor of 2.
    let doubled = lines()
        .line_stipple(Some(LineStipple {
            factor: 2,
            pattern: 0x00ff,
        }))
        .unwrap();
    let pixels = draw_flat(&mut gpu, &line, 2, doubled);
    assert_pixel(&pixels, 10, 20, WHITE);
    assert_pixel(&pixels, 21, 20, BLACK);
}

#[test]
fn depth_bounds_keep_fragments_over_stored_depths_inside() {
    let mut gpu = gpu();
    let vertices = two_triangles(&mut gpu);
    let Ok(inside) = unculled().depth_bounds(Some([0.0, 0.5])) else {
        return skip(
            "depth_bounds_keep_fragments_over_stored_depths_inside",
            "the device lacks `depth-bounds`",
        );
    };
    let outside = unculled().depth_bounds(Some([0.5, 1.0])).unwrap();
    let depth = Image::<D32Sfloat, usage![DepthAttachment, TransferDst]>::new(EXTENT).unwrap();
    let pixels = render(&mut gpu, |cmd, target| {
        // The bounds test the stored depth (0.25), not the fragment's (0.5).
        cmd.clear_depth_stencil(depth.whole_view(), [0.25]);
        let left = TestFlat::push_bindings(
            TestFlatArgs {
                vertices: vertices.range(0..3),
            },
            None,
        );
        let right = TestFlat::push_bindings(
            TestFlatArgs {
                vertices: vertices.range(3..6),
            },
            None,
        );
        let test = |state: RasterState<1>| state.depth_test(None).depth_write(false);
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .depth_attachment(depth.whole_view(), None)
            .draw(&Draw::new(&left, 3, 1, test(inside)))
            .draw(&Draw::new(&right, 3, 1, test(outside)))
            .record("depth bounds");
    });
    assert_pixel(&pixels, IN_LEFT.0, IN_LEFT.1, [255, 0, 0, 255]);
    assert_pixel(&pixels, IN_RIGHT.0, IN_RIGHT.1, BLACK);
}

#[test]
fn blend_constants_scale_constant_factors() {
    let mut gpu = gpu();
    let vertices = two_triangles(&mut gpu);
    let constant = BlendEquation {
        src_color: BlendFactor::CONSTANT_COLOR,
        dst_color: BlendFactor::ZERO,
        color_op: BlendOp::ADD,
        src_alpha: BlendFactor::CONSTANT_ALPHA,
        dst_alpha: BlendFactor::ZERO,
        alpha_op: BlendOp::ADD,
    };
    let state = |constants| {
        unculled()
            .blend(Blend::Equation(constant))
            .blend_constants(constants)
    };
    let pixels = render(&mut gpu, |cmd, target| {
        let left = TestFlat::push_bindings(
            TestFlatArgs {
                vertices: vertices.range(0..3),
            },
            None,
        );
        let right = TestFlat::push_bindings(
            TestFlatArgs {
                vertices: vertices.range(3..6),
            },
            None,
        );
        // Two draws with their own constants in one rendering.
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::new(&left, 3, 1, state([0.5, 1.0, 1.0, 1.0])))
            .draw(&Draw::new(&right, 3, 1, state([1.0, 0.25, 0.0, 1.0])))
            .record("blend constants");
    });
    // Red (the first vertex's flat colour) halved; yellow with a quarter of its green.
    assert_pixel(&pixels, IN_LEFT.0, IN_LEFT.1, [128, 0, 0, 255]);
    assert_pixel(&pixels, IN_RIGHT.0, IN_RIGHT.1, [255, 64, 0, 255]);
}

#[test]
fn strips_fans_and_points_assemble_their_primitives() {
    let mut gpu = gpu();
    let quad = |order: [(f32, f32); 4]| order.map(|(x, y)| vertex(x, y, 0.5, [1.0; 4]));
    // The left half as a strip and as a fan.
    let strip = gpu.buffer(&quad([(-1.0, -1.0), (0.0, -1.0), (-1.0, 1.0), (0.0, 1.0)]));
    let fan = gpu.buffer(&quad([(-1.0, -1.0), (0.0, -1.0), (0.0, 1.0), (-1.0, 1.0)]));
    for (name, vertices, topology) in [
        ("strip", &strip, PrimitiveTopology::TRIANGLE_STRIP),
        ("fan", &fan, PrimitiveTopology::TRIANGLE_FAN),
    ] {
        let pixels = draw_flat(&mut gpu, vertices, 4, unculled().topology(topology));
        for (x, y) in [(2, 2), (30, 2), (2, 61), (30, 61), (16, 32)] {
            assert_eq!(pixel(&pixels, x, y), WHITE, "{name} at ({x}, {y})");
        }
        assert_pixel(&pixels, 40, 32, BLACK);
    }
    // Two single-pixel points.
    let points = gpu.buffer(&[
        vertex(centre(10), centre(12), 0.5, [1.0; 4]),
        vertex(centre(40), centre(50), 0.5, [1.0; 4]),
    ]);
    let pixels = draw_flat(
        &mut gpu,
        &points,
        2,
        unculled().topology(PrimitiveTopology::POINT_LIST),
    );
    let lit: Vec<_> = (0..SIZE * SIZE)
        .filter(|i| pixel(&pixels, i % SIZE, i / SIZE) != BLACK)
        .map(|i| (i % SIZE, i / SIZE))
        .collect();
    assert_eq!(lit, [(10, 12), (40, 50)]);
    // A strip of two lines: across row 20, then down column 50.
    let (x0, x1, y0, y1) = (centre(5), centre(50), centre(20), centre(60));
    let corner = gpu.buffer(&[
        vertex(x0, y0, 0.5, [1.0; 4]),
        vertex(x1, y0, 0.5, [1.0; 4]),
        vertex(x1, y1, 0.5, [1.0; 4]),
    ]);
    let pixels = draw_flat(
        &mut gpu,
        &corner,
        3,
        unculled().topology(PrimitiveTopology::LINE_STRIP),
    );
    assert_pixel(&pixels, 30, 20, WHITE);
    assert_pixel(&pixels, 50, 40, WHITE);
    assert_pixel(&pixels, 30, 40, BLACK);
}

// ---- attachments read by their own rendering ---------------------------------------------------

/// An attachment that a draw also names in its accesses is rendered in `GENERAL` (its
/// descriptor's layout) and draws as usual; validation checks the layouts the rendering and the
/// copy after it use.
#[test]
fn attachments_reached_through_accesses_draw_as_usual() {
    let mut gpu = gpu();
    let vertices = two_triangles(&mut gpu);
    let expected = both(&mut gpu, &vertices);
    let pixels = render(&mut gpu, |cmd, target| {
        let reached = [ImageUse::storage_read(target.whole_view())];
        let bindings = TestRaster::push_bindings(
            TestRasterArgs {
                vertices: vertices.range(..),
                instance_offset: Vec2::ZERO,
            },
            Some(Accesses::images(&reached)),
        );
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::new(&bindings, 6, 1, unculled()))
            .record("reached");
        // And again, from the `GENERAL` layout the rendering left.
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::new(&raster(&vertices), 6, 1, unculled()))
            .record("plain");
    });
    assert_eq!(pixels, expected);
}

/// A draw samples the read-only depth attachment it is depth-tested against (a `DynImg`, named
/// in its accesses): both in `GENERAL`, neither writing, so no barrier lies between them.
#[test]
fn a_draw_samples_its_own_read_only_depth_attachment() {
    let mut gpu = gpu();
    let depth = Image::<D32Sfloat, usage![DepthAttachment, Sampled, TransferDst]>::new(EXTENT)
        .unwrap()
        .bind_sampled()
        .unwrap();
    let screen =
        |z| [(-1.0, -1.0), (3.0, -1.0), (-1.0, 3.0)].map(|(x, y)| vertex(x, y, z, [1.0; 4]));
    let (near, far) = (gpu.buffer(&screen(0.5)), gpu.buffer(&screen(0.1)));
    let draw = |gpu: &mut Gpu, vertices: &Buffer<TestVertex>| {
        render(gpu, |cmd, target| {
            cmd.clear_depth_stencil(depth.whole_view(), [0.25]);
            let reached = [ImageUse::sampled_read(depth.whole_view())];
            let bindings = TestDepthRead::push_bindings(
                TestDepthReadArgs {
                    vertices: vertices.range(..),
                    depth: depth.handle::<lava::bindless::Sampled>().erase(),
                    sampler: nearest(),
                    size: Vec2::splat(SIZE as f32),
                },
                Some(Accesses::images(&reached)),
            );
            cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
                .depth_attachment(depth.whole_view(), None)
                .draw(&Draw::new(&bindings, 3, 1, unculled().depth_write(false)))
                .record("sampling its depth");
        })
    };
    // In front of the stored 0.25: coloured with it.
    let pixels = draw(&mut gpu, &near);
    for (x, y) in [(1, 1), (32, 32), (62, 62)] {
        assert_pixel(&pixels, x, y, [64, 64, 64, 255]);
    }
    // Behind it: depth-tested away.
    let pixels = draw(&mut gpu, &far);
    assert!(pixels.as_chunks::<4>().0.iter().all(|&p| p == BLACK));
}

// ---- pipelines -------------------------------------------------------------------------------

#[test]
fn raster_specialization_constants_pick_the_pipeline() {
    let mut gpu = gpu();
    let vertices = two_triangles(&mut gpu);
    let reference = both(&mut gpu, &vertices);
    let constants = |brightness, dropped| {
        TestRasterSpecialized::push_bindings(
            TestRasterSpecializedArgs {
                vertices: vertices.range(..),
            },
            TestRasterSpecializedConstants {
                brightness,
                dropped,
            },
            None,
        )
    };
    let (plain, dimmed, no_green) = (constants(1.0, 3), constants(0.5, 3), constants(1.0, 1));
    let draw = |gpu: &mut Gpu, bindings: &BindingOutput<'_, kind::RasterVertex>| {
        render(gpu, |cmd, target| {
            cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
                .draw(&Draw::new(bindings, 6, 1, unculled()))
                .record("constants");
        })
    };
    assert_eq!(draw(&mut gpu, &plain), reference);
    let half = draw(&mut gpu, &dimmed);
    let dropped = draw(&mut gpu, &no_green);
    for (x, y) in [IN_LEFT, IN_RIGHT] {
        let [r, g, b, a] = pixel(&reference, x, y);
        let halve = |c: u8| ((c as f32 / 255.0 * 0.5) * 255.0).round() as u8;
        assert_pixel(&half, x, y, [halve(r), halve(g), halve(b), a]);
        assert_pixel(&dropped, x, y, [r, 0, b, a]);
    }
    // Draws of one rendering with different constants bind their own pipelines.
    let mixed = render(&mut gpu, |cmd, target| {
        let left = TestRasterSpecialized::push_bindings(
            TestRasterSpecializedArgs {
                vertices: vertices.range(0..3),
            },
            TestRasterSpecializedConstants {
                brightness: 0.5,
                dropped: 3,
            },
            None,
        );
        let right = TestRasterSpecialized::push_bindings(
            TestRasterSpecializedArgs {
                vertices: vertices.range(3..6),
            },
            TestRasterSpecializedConstants {
                brightness: 1.0,
                dropped: 1,
            },
            None,
        );
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::new(&left, 3, 1, unculled()))
            .draw(&Draw::new(&right, 3, 1, unculled()))
            .record("mixed constants");
    });
    assert_eq!(
        pixel(&mixed, IN_LEFT.0, IN_LEFT.1),
        pixel(&half, IN_LEFT.0, IN_LEFT.1)
    );
    assert_pixel(&mixed, IN_RIGHT.0, IN_RIGHT.1, [255, 0, 0, 255]);
}

#[test]
fn task_shaders_launch_the_mesh_workgroups_of_their_payload() {
    let mut gpu = gpu();
    let launch = |columns| {
        TestTask::push_bindings(
            TestTaskArgs {
                color: Vec4::new(0.0, 1.0, 0.0, 1.0),
                columns,
                width: 0.5,
            },
            None,
        )
        .ok()
    };
    let Some(two) = launch(2) else {
        return skip(
            "task_shaders_launch_the_mesh_workgroups_of_their_payload",
            "the device lacks `mesh-shader`",
        );
    };
    let four = launch(4).unwrap();
    let draw = |gpu: &mut Gpu, bindings: &BindingOutput<'_, kind::RasterMesh>| {
        render(gpu, |cmd, target| {
            cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
                .draw(&Draw::launch(bindings, [1, 1, 1], unculled()))
                .record("task");
        })
    };
    // Columns 16 pixels wide, from the left.
    let pixels = draw(&mut gpu, &two);
    for x in [2, 14, 18, 30] {
        assert_pixel(&pixels, x, 32, [0, 255, 0, 255]);
    }
    for x in [34, 50, 62] {
        assert_pixel(&pixels, x, 32, BLACK);
    }
    let pixels = draw(&mut gpu, &four);
    assert!(
        pixels
            .as_chunks::<4>()
            .0
            .iter()
            .all(|&p| p == [0, 255, 0, 255]),
        "four columns cover the target"
    );
}

// ---- passes and attachments ------------------------------------------------------------------

#[test]
fn vertex_only_passes_draw_depth() {
    let mut gpu = gpu();
    let vertices = two_triangles(&mut gpu);
    // The left triangle nearer than everything the colour pass draws.
    let near = gpu.buffer(&[
        vertex(-0.5, -0.8, 0.8, RED),
        vertex(-0.1, 0.8, 0.8, RED),
        vertex(-0.9, 0.8, 0.8, RED),
    ]);
    let depth = Image::<D32Sfloat, usage![DepthAttachment]>::new(EXTENT).unwrap();
    let depth_only = TestDepthOnly::push_bindings(
        TestDepthOnlyArgs {
            vertices: near.range(..),
        },
        None,
    );
    let pixels = render(&mut gpu, |cmd, target| {
        cmd.raster(EXTENT, [])
            .depth_attachment(depth.whole_view(), Some([0.0]))
            .draw(&Draw::new(&depth_only, 3, 1, unculled()))
            .record("depth only");
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .depth_attachment(depth.whole_view(), None)
            .draw(&Draw::new(
                &raster(&vertices),
                6,
                1,
                unculled().depth_write(false),
            ))
            .record("tested");
    });
    assert_pixel(&pixels, IN_LEFT.0, IN_LEFT.1, BLACK);
    assert_pixel(&pixels, IN_RIGHT.0, IN_RIGHT.1, [255, 255, 0, 255]);
}

#[test]
fn stencil_only_attachments_mark_and_test() {
    let mut gpu = gpu();
    let features = Ctx::format_features(vk::Format::S8_UINT);
    if !features.contains(vk::FormatFeatureFlags2::DEPTH_STENCIL_ATTACHMENT) {
        return skip(
            "stencil_only_attachments_mark_and_test",
            "S8Uint isn't an attachment here",
        );
    }
    let vertices = two_triangles(&mut gpu);
    let stencil = Image::<S8Uint, usage![DepthAttachment]>::new(EXTENT).unwrap();
    let face = |pass, compare| StencilFace {
        pass,
        compare,
        reference: 1,
        ..Default::default()
    };
    let mark = face(StencilOp::REPLACE, CompareOp::ALWAYS);
    let marked = face(StencilOp::KEEP, CompareOp::EQUAL);
    let left = TestRaster::push_bindings(
        TestRasterArgs {
            vertices: vertices.range(0..3),
            instance_offset: Vec2::ZERO,
        },
        None,
    );
    let pixels = render(&mut gpu, |cmd, target| {
        let marking = unculled()
            .stencil(Some(Stencil {
                front: mark,
                back: mark,
            }))
            .write_mask(vk::ColorComponentFlags::empty());
        let drawing = unculled().stencil(Some(Stencil {
            front: marked,
            back: marked,
        }));
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .stencil_attachment(stencil.whole_view(), Some([0]))
            .draw(&Draw::new(&left, 3, 1, marking))
            .draw(&Draw::new(&raster(&vertices), 6, 1, drawing))
            .record("stencil only");
    });
    assert_ne!(pixel(&pixels, IN_LEFT.0, IN_LEFT.1), BLACK);
    assert_pixel(&pixels, IN_RIGHT.0, IN_RIGHT.1, BLACK);
}

#[test]
fn unordered_accesses_of_one_rendering_may_overlap() {
    let mut gpu = gpu();
    let vertices = two_triangles(&mut gpu);
    let target = Target::new(EXTENT).unwrap();
    let counters = zeroed_buffer::<u32>(4);
    let unordered = [BufferAccess::write(counters.range(..)).unordered()];
    let ordered = [BufferAccess::write(counters.range(..))];
    assert!(unordered[0].is_unordered() && !ordered[0].is_unordered());
    let args = || TestRasterArgs {
        vertices: vertices.range(..),
        instance_offset: Vec2::ZERO,
    };
    let unordered = TestRaster::push_bindings(args(), Some(Accesses::buffers(&unordered)));
    let ordered = TestRaster::push_bindings(args(), Some(Accesses::buffers(&ordered)));
    gpu.submit(|cmd| {
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::new(&unordered, 3, 1, unculled()))
            .draw(&Draw::new(&unordered, 3, 1, unculled()))
            .record("unordered");
    });
    let message = panic_message(|| {
        gpu.submit(|cmd| {
            cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
                .draw(&Draw::new(&unordered, 3, 1, unculled()))
                .draw(&Draw::new(&ordered, 3, 1, unculled()))
                .record("one ordered");
        })
    });
    assert!(message.contains("both access"), "{message}");
}

// ---- array layers ----------------------------------------------------------------------------

#[test]
fn array_layers_are_sampled_and_copied_one_by_one() {
    let mut gpu = gpu();
    type Array = Image<R8G8B8A8Unorm, usage![Sampled, TransferSrc, TransferDst], D2Array>;
    let array = Array::with_layers(UVec2::splat(4), 1, 3)
        .unwrap()
        .bind_sampled()
        .unwrap();
    assert_eq!((array.layers(), array.whole_view().layer_count()), (3, 3));
    let colors = [RED, GREEN, BLUE];
    let mut texels = zeroed_buffer::<Vec4>(4);
    let mut layer = zeroed_buffer::<u8>(4 * 4 * 4);
    let mut all = zeroed_buffer::<u8>(2 * 4 * 4 * 4);
    gpu.submit(|cmd| {
        for (i, color) in colors.iter().enumerate() {
            cmd.clear_image(array.whole_view().layer(i as u32), *color);
        }
        cmd.compute(
            &TestSampleArray::push_bindings(
                TestSampleArrayArgs {
                    image: array.whole_view(),
                    sampler: nearest(),
                    texels: texels.range(..),
                    layers: 3,
                },
                None,
            ),
            [1, 1, 1],
        );
        cmd.copy_image_to_buffer(array.whole_view().layer(1).whole(), layer.range(..));
        cmd.copy_image_to_buffer(array.whole_view().layers(1..3).whole(), all.range(..));
    });
    let texels = texels.host().to_vec();
    for (i, color) in colors.iter().enumerate() {
        assert_eq!(texels[i], Vec4::from_array(*color), "layer {i}");
    }
    assert_eq!(texels[3].x, 3.0, "the descriptor's layer count");
    assert!(
        layer
            .host()
            .as_chunks::<4>()
            .0
            .iter()
            .all(|&t| t == [0, 255, 0, 255])
    );
    let all = all.host().to_vec();
    assert!(
        all[..64]
            .as_chunks::<4>()
            .0
            .iter()
            .all(|&t| t == [0, 255, 0, 255])
    );
    assert!(
        all[64..128]
            .as_chunks::<4>()
            .0
            .iter()
            .all(|&t| t == [0, 0, 255, 255])
    );
    let message = panic_message(|| array.whole_view().layer(3));
    assert!(message.contains("aren't some of the view's"), "{message}");
}

#[test]
fn a_layer_view_is_an_attachment_of_its_layer() {
    let mut gpu = gpu();
    let vertices = two_triangles(&mut gpu);
    let layers = render_layers(&mut gpu, 2, |cmd, target| {
        cmd.raster(
            EXTENT,
            [Attachment::color(target.whole_view().layer(1), None)],
        )
        .draw(&Draw::new(&raster(&vertices), 6, 1, unculled()))
        .record("layer 1");
    });
    assert!(layers[0].as_chunks::<4>().0.iter().all(|&p| p == BLACK));
    assert_eq!(layers[1], both(&mut gpu, &vertices));
}

#[test]
fn layered_renderings_draw_each_instance_into_its_layer() {
    let mut gpu = gpu();
    let vertices = two_triangles(&mut gpu);
    let Ok(bindings) = TestLayered::push_bindings(
        TestLayeredArgs {
            vertices: vertices.range(0..3),
            second: Vec4::from_array(GREEN),
        },
        None,
    ) else {
        return skip(
            "layered_renderings_draw_each_instance_into_its_layer",
            "the device lacks `shader-output-layer`",
        );
    };
    let layers = render_layers(&mut gpu, 3, |cmd, target| {
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .layers(2)
            .draw(&Draw::new(&bindings, 3, 2, unculled()))
            .record("layered");
    });
    let left = render(&mut gpu, |cmd, target| {
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::new(&raster(&vertices), 3, 1, unculled()))
            .record("left");
    });
    assert_eq!(layers[0], left);
    assert_pixel(&layers[1], IN_LEFT.0, IN_LEFT.1, [0, 255, 0, 255]);
    assert_pixel(&layers[1], IN_RIGHT.0, IN_RIGHT.1, BLACK);
    assert!(
        layers[2].as_chunks::<4>().0.iter().all(|&p| p == BLACK),
        "layer 2 isn't drawn"
    );
    let message = panic_message(|| {
        gpu.submit(|cmd| {
            let target = ArrayTarget::with_layers(EXTENT, 1, 2).unwrap();
            cmd.raster(
                EXTENT,
                [Attachment::color(target.whole_view().layers(0..1), None)],
            )
            .layers(2)
            .draw(&Draw::new(&bindings, 3, 2, unculled()))
            .record("too few layers");
        })
    });
    assert!(message.contains("an attachment of 1 layers"), "{message}");
}

#[test]
fn multiview_renderings_draw_every_view() {
    let mut gpu = gpu();
    let vertices = two_triangles(&mut gpu);
    let bindings = TestMultiview::push_bindings(
        TestMultiviewArgs {
            vertices: vertices.range(0..3),
            other: Vec4::from_array(BLUE),
        },
        None,
    );
    let views = render_layers(&mut gpu, 3, |cmd, target| {
        // Views 0 and 2.
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .view_mask(0b101)
            .draw(&Draw::new(&bindings, 3, 1, unculled()))
            .record("multiview");
    });
    let left = render(&mut gpu, |cmd, target| {
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::new(&raster(&vertices), 3, 1, unculled()))
            .record("left");
    });
    assert_eq!(views[0], left);
    assert!(
        views[1].as_chunks::<4>().0.iter().all(|&p| p == BLACK),
        "view 1 isn't drawn"
    );
    assert_pixel(&views[2], IN_LEFT.0, IN_LEFT.1, [0, 0, 255, 255]);
    assert_pixel(&views[2], IN_RIGHT.0, IN_RIGHT.1, BLACK);
}

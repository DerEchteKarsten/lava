//! GPU tests of a device without lava's optional features: init disables them (`Config::disabled`, `rebar` too), the functions that need them return `MissingFeature` (or panic, for draw state), uploads go through staging buffers, and draw state the device can't set dynamically is baked into pipelines and renders the same, the topology class too (without extended dynamic state 3 the device can't leave it)
mod common;

use common::{Cmd, Gpu, buffer_with, golden::assert_golden, gpu, zeroed_buffer};
use glam::{UVec2, Vec2, Vec4};
use lava::{
    Error, Features,
    buffer::{Buffer, Location, Shared},
    command_buffer::{
        Attachment, BindingOutput, Blend, BlendEquation, BlendFactor, BlendOp, CullModeFlags,
        Draw, LogicOp, MultiDraw, PolygonMode, PrimitiveTopology, RasterState, kind,
    },
    features::MissingFeature,
    image::{
        Image,
        format::{BC7UnormBlock, R8G8B8A8Unorm},
        usage::usage,
    },
    memory::memory_report,
    query::{QueryPipelineStatisticFlags as Statistics, QueryPool, Timestamp, counters},
    queue::{Compute, Kind, Timeline, Transfer},
    state::Ctx,
};
use lava_tests::bindings::{TestLayered, TestMesh, TestRaster, TestVertex};

const SIZE: u32 = 64;
const EXTENT: UVec2 = UVec2::splat(SIZE);

type Target = Image<R8G8B8A8Unorm, usage![ColorAttachment, Storage, TransferSrc, TransferDst]>;

fn disabled() -> Features {
    Features {
        rebar: true,
        mesh_shader: true,
        mesh_shader_queries: true,
        acceleration_structure: true,
        ray_query: true,
        ray_tracing_pipeline: true,
        pipeline_statistics: true,
        precise_occlusion: true,
        memory_budget: true,
        shader_clock: true,
        performance_query: true,
        calibrated_timestamps: true,
        async_compute: true,
        async_transfer: true,
        host_image_copy: true,
        bc_compression: true,
        dual_src_blend: true,
        multi_draw: true,
        shader_output_layer: true,
        wide_lines: true,
        depth_bounds: true,
        dynamic_blend: true,
        dynamic_polygon_mode: true,
        dynamic_depth_clamp: true,
        dynamic_depth_clip: true,
        dynamic_depth_clip_negative_one_to_one: true,
        dynamic_logic_op: true,
        dynamic_conservative_rasterization: true,
        dynamic_line_mode: true,
        dynamic_line_stipple: true,
        dynamic_provoking_vertex: true,
        ..Features::DEFAULT
    }
}

fn gpu_without() -> common::GpuGuard {
    common::disable(disabled());
    gpu()
}

fn missing(feature: &'static str) -> MissingFeature {
    MissingFeature { feature }
}

/// Clears a fresh target to `clear`, lets `record` draw into it, and reads it back.
fn render(
    gpu: &mut common::Gpu,
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

#[track_caller]
fn assert_pixel(pixels: &[u8], x: u32, y: u32, expected: [u8; 4]) {
    let i = ((y * SIZE + x) * 4) as usize;
    let actual: [u8; 4] = pixels[i..i + 4].try_into().unwrap();
    let close = actual.iter().zip(expected).all(|(a, b)| a.abs_diff(b) <= 2);
    assert!(
        close,
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

/// The same two triangles as `gpu_headless`: one front-facing in the left half, one
/// back-facing yellow one in the right half.
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

/// A far green triangle and a near red one in front of it, drawn in that order.
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

fn bindings(
    vertices: &Buffer<TestVertex>,
    range: std::ops::Range<usize>,
) -> impl std::ops::Deref<Target = BindingOutput<'_, kind::RasterVertex>> {
    TestRaster::push_bindings(vertices.range(range), Vec2::ZERO)
}

fn unculled() -> RasterState<1> {
    RasterState::default().cull_mode(CullModeFlags::NONE)
}

#[test]
fn disabled_features_are_off() {
    let _gpu = gpu_without();
    let features = Ctx::features();
    assert_eq!(features.without(disabled()), features);
    assert!(!features.dynamic_blend && !features.mesh_shader);
    // Without async compute and transfer, only the graphics family gets queues.
    assert_eq!(Ctx::queue_family(Kind::Compute), None);
    assert_eq!(Ctx::queue_family(Kind::Transfer), None);
    let compute = Timeline::<Compute>::new(2, 0);
    assert!(matches!(compute, Err(Error::MissingFeature(m)) if m == missing("async-compute")));
    let transfer = Timeline::<Transfer>::new(2, 0);
    assert!(matches!(transfer, Err(Error::MissingFeature(m)) if m == missing("async-transfer")));
}

#[test]
fn functions_that_need_a_disabled_feature_return_missing_feature() {
    let _gpu = gpu_without();
    let launch = TestMesh::push_bindings(Vec4::ONE, Vec2::ZERO, 0.1);
    assert_eq!(launch.err(), Some(missing("mesh-shader")));
    let statistics = QueryPool::statistics(1, Statistics::COMPUTE_SHADER_INVOCATIONS);
    assert!(
        matches!(statistics, Err(Error::MissingFeature(m)) if m == missing("pipeline-statistics"))
    );
    assert!(matches!(
        QueryPool::precise_occlusion(1),
        Err(Error::MissingFeature(m)) if m == missing("precise-occlusion")
    ));
    assert!(matches!(
        QueryPool::compacted_sizes(1),
        Err(Error::MissingFeature(m)) if m == missing("acceleration-structure")
    ));
    assert_eq!(
        counters::counters().err(),
        Some(missing("performance-query"))
    );
    let timestamps = QueryPool::<Timestamp>::timestamps(1).unwrap();
    assert_eq!(
        timestamps.instant(0).err(),
        Some(missing("calibrated-timestamps"))
    );
    let heap = &memory_report().heaps[0];
    assert_eq!(heap.usage, Err(missing("memory-budget")));
    assert!(matches!(
        Image::<R8G8B8A8Unorm, usage![Sampled, HostCopy]>::new(EXTENT),
        Err(Error::MissingFeature(m)) if m == missing("host-image-copy")
    ));
    assert!(matches!(
        Image::<BC7UnormBlock, usage![Sampled]>::new(EXTENT),
        Err(Error::MissingFeature(m)) if m == missing("bc-compression")
    ));
    assert!(matches!(
        Buffer::<u32, Shared>::shared(4),
        Err(Error::MissingFeature(m)) if m == missing("rebar")
    ));
    assert_eq!(Location::shared_or_upload(), Location::Upload);
    let state = RasterState::<1>::default();
    assert_eq!(
        state.clone().line_width(2.0).err(),
        Some(missing("wide-lines"))
    );
    assert_eq!(
        state.depth_bounds(Some([0.0, 0.5])).err(),
        Some(missing("depth-bounds"))
    );
    let vertices = buffer_with::<TestVertex>(&[bytemuck::Zeroable::zeroed(); 3]);
    let draws = [MultiDraw {
        first_vertex: 0,
        vertex_count: 3,
    }];
    let bindings = TestRaster::push_bindings(vertices.range(..), Vec2::ZERO);
    let multi = Draw::multi(&bindings, &draws, 1, RasterState::<1>::default(), None);
    assert_eq!(multi.err(), Some(missing("multi-draw")));
    let layered = TestLayered::push_bindings(vertices.range(..), Vec4::ONE);
    assert_eq!(layered.err(), Some(missing("shader-output-layer")));
    // Draw state that needs a feature without a setter of its own panics.
    let src1 = BlendEquation {
        src_color: BlendFactor::SRC1_COLOR,
        dst_color: BlendFactor::ZERO,
        color_op: BlendOp::ADD,
        src_alpha: BlendFactor::ONE,
        dst_alpha: BlendFactor::ZERO,
        alpha_op: BlendOp::ADD,
    };
    let message = common::panic_message(|| {
        let state = RasterState::<1>::default().blend(Blend::Equation(src1));
        drop(Draw::new(&bindings, 3, 1, state, None));
    });
    assert!(message.contains("`dual-src-blend`"), "{message}");
}

#[test]
fn baked_topology_classes_draw_like_dynamic_ones() {
    let mut gpu = gpu_without();
    let vertices = two_triangles(&mut gpu);
    // Pipelines bake the line class, which the draw's dynamic topology stays in (validation
    // checks it), and a triangle draw after it binds its own.
    let pixels = render(&mut gpu, [0.0, 0.0, 0.0, 1.0], |cmd, target| {
        let lines = unculled().topology(PrimitiveTopology::LINE_LIST);
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::new(&bindings(&vertices, 0..6), 6, 1, lines, None))
            .record("lines")
    });
    // The same image as `gpu_headless`'s line list on a device without the restriction.
    assert_golden("raster_lines", &pixels, [SIZE, SIZE]);
    let mixed = render(&mut gpu, [0.0, 0.0, 0.0, 1.0], |cmd, target| {
        let lines = unculled().topology(PrimitiveTopology::LINE_STRIP);
        let (left, right) = (bindings(&vertices, 0..3), bindings(&vertices, 3..6));
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::new(&left, 3, 1, lines, None))
            .draw(&Draw::new(&right, 3, 1, unculled(), None))
            .record("lines then triangles")
    });
    assert_pixel(&mixed, 16, 44, [0, 0, 0, 255]);
    assert_pixel(&mixed, 48, 44, [255, 255, 0, 255]);
}

/// Without `rebar` or host image copies, buffers and images are filled through the timeline's
/// staging ring.
#[test]
fn uploads_need_neither_rebar_nor_host_image_copies() {
    let mut gpu = gpu_without();
    let uploader = gpu.uploader();
    assert!(uploader.stages(), "a timeline without `rebar` has the ring it was given");
    let data: Vec<u32> = (0..256).collect();
    let buffer = uploader.from_slice(&data).unwrap();
    // Staged without `rebar`, unless device-local memory is mapped anyway (integrated GPUs).
    if uploader.pending() == 0 {
        eprintln!("NOTE: device-local memory is mapped here, so `init` wrote in place");
    }
    assert!(uploader.pending() <= 1);
    let texels: Vec<u8> = (0..16 * 4).map(|i| i as u8).collect();
    let image = Image::<R8G8B8A8Unorm, usage![TransferSrc, TransferDst]>::new(UVec2::new(4, 4)).unwrap();
    uploader.image(&image, 0, &texels).unwrap();
    let (mut buffer_out, mut image_out) = (zeroed_buffer::<u32>(data.len()), zeroed_buffer::<u8>(texels.len()));
    gpu.submit(|cmd| {
        let stats = cmd.drain_uploads();
        assert!(stats.copies >= 1 && stats.bytes >= texels.len() as u64, "{stats:?}");
        cmd.copy_buffer(buffer.range(..), buffer_out.range(..));
        cmd.copy_image_to_buffer(image.whole_view().whole(), image_out.range(..));
    });
    assert_eq!(buffer_out.host(), data);
    assert_eq!(image_out.host(), texels);
    assert_eq!(uploader.pending(), 0);
}

#[test]
fn baked_polygon_modes_draw_like_dynamic_ones() {
    let mut gpu = gpu_without();
    let vertices = two_triangles(&mut gpu);
    let pixels = render(&mut gpu, [0.0, 0.0, 0.0, 1.0], |cmd, target| {
        let lines = unculled().polygon_mode(PolygonMode::LINE).unwrap();
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::new(&bindings(&vertices, 0..6), 6, 1, lines, None))
            .record("wireframe")
    });
    // The same image as `gpu_headless`'s dynamic polygon mode.
    assert_golden("raster_wireframe", &pixels, [SIZE, SIZE]);
}

#[test]
fn draws_that_differ_in_baked_state_bind_their_own_pipelines() {
    let mut gpu = gpu_without();
    let vertices = two_triangles(&mut gpu);
    let pixels = render(&mut gpu, [0.0, 0.0, 0.0, 1.0], |cmd, target| {
        let lines = unculled().polygon_mode(PolygonMode::LINE).unwrap();
        let (left, right) = (bindings(&vertices, 0..3), bindings(&vertices, 3..6));
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::new(&left, 3, 1, lines, None))
            .draw(&Draw::new(&right, 3, 1, unculled(), None))
            .record("lines then filled")
    });
    // The left triangle is only its edges, the right one filled.
    assert_pixel(&pixels, 16, 44, [0, 0, 0, 255]);
    assert_pixel(&pixels, 48, 44, [255, 255, 0, 255]);
}

#[test]
fn baked_blends_combine_and_skip() {
    let mut gpu = gpu_without();
    let vertices = overlapping_triangles(&mut gpu);
    // Green, then red added on top of it.
    let added = render(&mut gpu, [0.0, 0.0, 0.0, 1.0], |cmd, target| {
        let add = RasterState::default().blend(Blend::Add);
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::new(&bindings(&vertices, 0..6), 6, 1, add, None))
            .record("added")
    });
    assert_pixel(&added, 32, 40, [255, 255, 0, 255]);
    // A skipped attachment's baked write mask is empty.
    let skipped = render(&mut gpu, BLUE, |cmd, target| {
        let skip = RasterState::default().blend(Blend::Skip);
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::new(&bindings(&vertices, 0..6), 6, 1, skip, None))
            .record("skipped")
    });
    assert!(skipped.chunks_exact(4).all(|p| p == [0, 0, 255, 255]));
}

#[test]
fn baked_logic_ops_replace_blending() {
    let mut gpu = gpu_without();
    if !Ctx::features().logic_op {
        eprintln!("SKIPPED baked_logic_ops_replace_blending: the device lacks `logic-op`");
        return;
    }
    let vertices = overlapping_triangles(&mut gpu);
    let pixels = render(&mut gpu, [1.0; 4], |cmd, target| {
        let xor = RasterState::default().logic_op(Some(LogicOp::XOR)).unwrap();
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .draw(&Draw::new(&bindings(&vertices, 0..6), 6, 1, xor, None))
            .record("xor")
    });
    // White XOR green XOR red.
    assert_pixel(&pixels, 32, 40, [0, 0, 255, 255]);
}

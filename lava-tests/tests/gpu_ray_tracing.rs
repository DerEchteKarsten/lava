//! GPU ray tracing tests: acceleration structure builds, refits and compaction, ray queries and ray tracing pipelines on a small scene
mod common;

use common::{Cmd, buffer_with, golden::assert_golden, gpu, panic_message, zeroed_buffer};
use glam::{Affine3A, Quat, UVec2, Vec3};
use lava::{
    buffer::Buffer,
    command_buffer::TraceRaysIndirectCommand,
    image::{Image, format::R8G8B8A8Unorm, usage::usage},
    query::QueryPool,
    state::Ctx,
    vk,
    vkobjects::acceleration_structure::{
        Aabb, Aabbs, Blas, BlasGeometry, BuildFlags, Instance, Instances, Tlas,
        Triangles, VertexPosition, scratch_alignment,
    },
};
use lava_tests::bindings::{TestRayQuery, TestRayTracing, TestWriteInstances};

const SIZE: u32 = 64;
const EXTENT: UVec2 = UVec2::splat(SIZE);

type Target = Image<R8G8B8A8Unorm, usage![Storage, TransferSrc, TransferDst]>;

fn supported(test: &str, feature: bool) -> bool {
    if !feature {
        eprintln!("SKIPPED {test}: the device lacks the ray tracing features it needs");
    }
    feature
}

/// Lets `record` write `targets` in one submission and reads them back.
fn render_many<const N: usize>(
    gpu: &mut common::Gpu,
    record: impl FnOnce(&mut Cmd, &[Target; N]),
) -> [Vec<u8>; N] {
    let targets: [Target; N] =
        std::array::from_fn(|_| Target::new(EXTENT).unwrap().bind_storage().unwrap());
    let readbacks: [Buffer<u8>; N] =
        std::array::from_fn(|_| zeroed_buffer((SIZE * SIZE * 4) as usize));
    gpu.submit(|cmd| {
        for target in &targets {
            cmd.clear_image(target.whole_view(), [0.0, 0.0, 0.0, 1.0]);
        }
        record(cmd, &targets);
        for (target, readback) in targets.iter().zip(&readbacks) {
            cmd.copy_image_to_buffer(target.whole(), readback.range(..));
        }
    });
    readbacks.map(|readback| readback.host().to_vec())
}

fn render(gpu: &mut common::Gpu, record: impl FnOnce(&mut Cmd, &Target)) -> Vec<u8> {
    let [pixels] = render_many(gpu, |cmd, [target]| record(cmd, target));
    pixels
}

#[track_caller]
fn assert_pixel(pixels: &[u8], x: u32, y: u32, expected: [u8; 3]) {
    let i = ((y * SIZE + x) * 4) as usize;
    let actual: [u8; 4] = pixels[i..i + 4].try_into().unwrap();
    let close = actual[..3].iter().zip(expected).all(|(a, e)| a.abs_diff(e) <= 2);
    assert!(
        close && actual[3] == 255,
        "pixel ({x}, {y}) is {actual:?}, expected {expected:?}"
    );
}

fn trace_query(cmd: &mut Cmd, target: &Target, tlas: &Tlas) {
    cmd.compute(
        &TestRayQuery::push_bindings(target.whole_view(), tlas, EXTENT).unwrap(),
        [SIZE / 8, SIZE / 8, 1],
        None,
    );
}

fn trace_pipeline(cmd: &mut Cmd, target: &Target, tlas: &Tlas) {
    cmd.raytrace(
        &TestRayTracing::push_bindings(target.whole_view(), tlas, EXTENT).unwrap(),
        [SIZE, SIZE, 1],
        None,
    );
}

// Pixels of the scene (x = -1 at column 0, y = -1 at row 0).
const BACKGROUND: (u32, u32) = (1, 32);
/// In front of the wall's first triangle, behind the masked triangle, lit.
const WALL: (u32, u32) = (40, 12);
/// In front of the wall's second triangle.
const WALL_SECOND_TRIANGLE: (u32, u32) = (16, 48);
const TRIANGLE: (u32, u32) = (19, 20);
/// Off the wall's diagonal, so the wall behind it is its first triangle.
const SPHERE: (u32, u32) = (44, 42);
/// On the wall in the triangle's shadow.
const SHADOWED: (u32, u32) = (25, 27);
/// In one of the holes the any-hit shader cuts into the wall.
const HOLE: (u32, u32) = (33, 33);

struct Scene {
    wall: Blas,
    triangle: Blas,
    sphere: Blas,
    tlas: Tlas,
    instances: Buffer<Instance>,
    _inputs: (Buffer<Vec3>, Buffer<u16>, Buffer<Vec3>, Buffer<Aabb>),
}

fn place(scale: f32, position: Vec3) -> Affine3A {
    Affine3A::from_scale_rotation_translation(Vec3::splat(scale), Quat::IDENTITY, position)
}

/// The scene's instances, as `test_write_instances.slang` writes them.
fn instances(wall: u64, triangle: u64, sphere: u64) -> [Instance; 4] {
    [
        Instance::from_address(wall, place(1.0, Vec3::ZERO)),
        Instance::from_address(triangle, place(1.0, Vec3::new(-0.4, -0.3, -0.5))).custom_index(1),
        Instance::from_address(sphere, place(0.25, Vec3::new(0.35, 0.35, -0.5)))
            .custom_index(2)
            .sbt_offset(1),
        Instance::from_address(triangle, place(1.0, Vec3::new(0.45, -0.45, -0.5)))
            .custom_index(3)
            .mask(0x02),
    ]
}

fn scratch(size: u64) -> Buffer<u8> {
    Buffer::raw(false, size, Some(scratch_alignment() as u32)).unwrap()
}

fn scene(gpu: &mut common::Gpu, blas_flags: BuildFlags, tlas_flags: BuildFlags) -> Scene {
    let wall_vertices = buffer_with::<Vec3>(&[
        Vec3::new(-0.9, -0.9, 0.0),
        Vec3::new(0.9, -0.9, 0.0),
        Vec3::new(0.9, 0.9, 0.0),
        Vec3::new(-0.9, 0.9, 0.0),
    ]);
    let wall_indices = buffer_with::<u16>(&[0, 1, 2, 0, 2, 3]);
    let triangle_vertices = buffer_with::<Vec3>(&[
        Vec3::new(-0.3, -0.3, 0.0),
        Vec3::new(0.3, -0.3, 0.0),
        Vec3::new(0.0, 0.3, 0.0),
    ]);
    let boxes = buffer_with::<Aabb>(&[Aabb {
        min: Vec3::splat(-1.0),
        max: Vec3::splat(1.0),
    }]);
    let wall_geometry =
        [Triangles::indexed(wall_vertices.range(..), wall_indices.range(..)).opaque(false)];
    let triangle_geometry = [Triangles::new(triangle_vertices.range(..))];
    let sphere_geometry = [Aabbs::new(boxes.range(..))];
    let wall = BlasGeometry::Triangles(&wall_geometry);
    let triangle = BlasGeometry::Triangles(&triangle_geometry);
    let sphere = BlasGeometry::Aabbs(&sphere_geometry);
    let (wall_blas, triangle_blas, sphere_blas) = (
        Blas::new(&wall, blas_flags).unwrap(),
        Blas::new(&triangle, blas_flags).unwrap(),
        Blas::new(&sphere, blas_flags).unwrap(),
    );
    let instances = buffer_with::<Instance>(&instances(
        wall_blas.address(),
        triangle_blas.address(),
        sphere_blas.address(),
    ));
    let tlas = Tlas::new(4, tlas_flags).unwrap();

    let blas_scratch = [&wall_blas, &triangle_blas, &sphere_blas]
        .iter()
        .map(|b| b.build_scratch_size())
        .sum::<u64>();
    let scratch = scratch(blas_scratch.max(tlas.build_scratch_size()));
    gpu.submit(|cmd| {
        cmd.build(
            &[(&wall_blas, wall), (&triangle_blas, triangle), (&sphere_blas, sphere)],
            scratch.range(..),
        );
        // SAFETY: the BLASes it names outlive the traces of this test.
        cmd.build(&[(&tlas, unsafe { Instances::new(instances.range(..)) })], scratch.range(..));
    });
    Scene {
        wall: wall_blas,
        triangle: triangle_blas,
        sphere: sphere_blas,
        tlas,
        instances,
        _inputs: (wall_vertices, wall_indices, triangle_vertices, boxes),
    }
}

fn assert_query_image(pixels: &[u8]) {
    assert_pixel(pixels, BACKGROUND.0, BACKGROUND.1, [0, 0, 0]);
    // The masked triangle in front of it is invisible.
    assert_pixel(pixels, WALL.0, WALL.1, [204, 204, 204]);
    assert_pixel(pixels, WALL_SECOND_TRIANGLE.0, WALL_SECOND_TRIANGLE.1, [122, 122, 122]);
    assert_pixel(pixels, TRIANGLE.0, TRIANGLE.1, [255, 51, 26]);
    assert_pixel(pixels, SPHERE.0, SPHERE.1, [26, 102, 255]);
}

#[test]
fn ray_queries_hit_triangles_and_aabbs_through_the_instance_mask() {
    let mut gpu = gpu();
    if !supported("ray_queries_hit_triangles_and_aabbs", Ctx::features().ray_query) {
        return;
    }
    let scene = scene(&mut gpu, BuildFlags::default(), BuildFlags::default());
    let pixels = render(&mut gpu, |cmd, target| trace_query(cmd, target, &scene.tlas));
    assert_query_image(&pixels);
    assert_golden("ray_query", &pixels, [SIZE, SIZE]);
}

#[test]
fn ray_tracing_pipelines_run_every_shader_kind() {
    let mut gpu = gpu();
    if !supported("ray_tracing_pipelines_run_every_shader_kind", Ctx::features().ray_tracing_pipeline) {
        return;
    }
    let scene = scene(&mut gpu, BuildFlags::default(), BuildFlags::default());
    let pixels = render(&mut gpu, |cmd, target| trace_pipeline(cmd, target, &scene.tlas));
    let sky = [51, 153, 77];
    assert_pixel(&pixels, BACKGROUND.0, BACKGROUND.1, sky);
    assert_pixel(&pixels, WALL.0, WALL.1, [204, 204, 204]);
    assert_pixel(&pixels, SHADOWED.0, SHADOWED.1, [82, 82, 82]);
    assert_pixel(&pixels, HOLE.0, HOLE.1, sky);
    assert_pixel(&pixels, TRIANGLE.0, TRIANGLE.1, [255, 51, 26]);
    // Shaded by the normal's z, 0.98 here.
    assert_pixel(&pixels, SPHERE.0, SPHERE.1, [25, 101, 253]);
    assert_golden("ray_tracing", &pixels, [SIZE, SIZE]);
}

#[test]
fn indirect_traces_read_their_size_on_the_gpu() {
    let mut gpu = gpu();
    if !supported("indirect_traces_read_their_size_on_the_gpu", Ctx::features().ray_tracing_pipeline) {
        return;
    }
    let scene = scene(&mut gpu, BuildFlags::default(), BuildFlags::default());
    let direct = render(&mut gpu, |cmd, target| trace_pipeline(cmd, target, &scene.tlas));
    let command = zeroed_buffer::<TraceRaysIndirectCommand>(1);
    let indirect = render(&mut gpu, |cmd, target| {
        // Written in the same submission: the trace waits for the transfer.
        cmd.update_buffer(
            command.range(..),
            &TraceRaysIndirectCommand {
                width: SIZE,
                height: SIZE,
                depth: 1,
            },
        );
        cmd.raytrace_indirect(
            &TestRayTracing::push_bindings(target.whole_view(), &scene.tlas, EXTENT).unwrap(),
            command.range(..),
            None,
        );
    });
    assert_eq!(indirect, direct);
}

#[test]
fn instances_written_by_a_pass_build_a_tlas_in_the_same_submission() {
    let mut gpu = gpu();
    if !supported("instances_written_by_a_pass", Ctx::features().ray_query) {
        return;
    }
    let scene = scene(&mut gpu, BuildFlags::default(), BuildFlags::default());
    let written = zeroed_buffer::<Instance>(4);
    let tlas = Tlas::new(4, BuildFlags::default()).unwrap();
    let scratch = scratch(tlas.build_scratch_size());
    let pixels = render(&mut gpu, |cmd, target| {
        cmd.compute(
            &TestWriteInstances::push_bindings(
                written.range(..),
                scene.wall.address(),
                scene.triangle.address(),
                scene.sphere.address(),
            ),
            [1, 1, 1],
            None,
        );
        // SAFETY: the BLASes it names outlive the traces of this test.
        cmd.build(&[(&tlas, unsafe { Instances::new(written.range(..)) })], scratch.range(..));
        trace_query(cmd, target, &tlas);
    });
    assert_eq!(written.host(), scene.instances.host());
    assert_query_image(&pixels);
}

#[test]
fn updates_follow_moved_vertices() {
    let mut gpu = gpu();
    if !supported("updates_follow_moved_vertices", Ctx::features().ray_query) {
        return;
    }
    let updatable = BuildFlags {
        allow_update: true,
        ..Default::default()
    };
    let mut vertices = buffer_with::<Vec3>(&[
        Vec3::new(-0.3, -0.3, 0.0),
        Vec3::new(0.3, -0.3, 0.0),
        Vec3::new(0.0, 0.3, 0.0),
    ]);
    fn geometry(vertices: &Buffer<Vec3>) -> [Triangles<'_>; 1] {
        [Triangles::new(vertices.range(..))]
    }
    let blas = Blas::new(&BlasGeometry::Triangles(&geometry(&vertices)), updatable).unwrap();
    assert!(blas.update_scratch_size() > 0);
    let instances = buffer_with::<Instance>(&[
        Instance::new(&blas, Affine3A::IDENTITY).custom_index(1),
    ]);
    let tlas = Tlas::new(1, updatable).unwrap();
    let scratch = scratch(
        [blas.build_scratch_size(), blas.update_scratch_size(), tlas.build_scratch_size(), tlas.update_scratch_size()]
            .into_iter()
            .max()
            .unwrap(),
    );
    let (old, new) = ((32, 30), (48, 30));
    let pixels = render(&mut gpu, |cmd, target| {
        cmd.build(&[(&blas, BlasGeometry::Triangles(&geometry(&vertices)))], scratch.range(..));
        // SAFETY: the BLASes it names outlive the traces of this test.
        cmd.build(&[(&tlas, unsafe { Instances::new(instances.range(..)) })], scratch.range(..));
        trace_query(cmd, target, &tlas);
    });
    assert_pixel(&pixels, old.0, old.1, [255, 51, 26]);
    assert_pixel(&pixels, new.0, new.1, [0, 0, 0]);

    for vertex in vertices.host_mut() {
        vertex.x += 0.5;
    }
    let pixels = render(&mut gpu, |cmd, target| {
        cmd.update(&[(&blas, BlasGeometry::Triangles(&geometry(&vertices)))], scratch.range(..));
        // SAFETY: the BLASes it names outlive the traces of this test.
        cmd.update(&[(&tlas, unsafe { Instances::new(instances.range(..)) })], scratch.range(..));
        trace_query(cmd, target, &tlas);
    });
    assert_pixel(&pixels, old.0, old.1, [0, 0, 0]);
    assert_pixel(&pixels, new.0, new.1, [255, 51, 26]);
}

#[test]
fn a_tlas_rebuilt_between_two_traces_waits_for_the_first() {
    let mut gpu = gpu();
    if !supported("a_tlas_rebuilt_between_two_traces", Ctx::features().ray_query) {
        return;
    }
    let scene = scene(&mut gpu, BuildFlags::default(), BuildFlags::default());
    let wall_only = buffer_with::<Instance>(&instances(
        scene.wall.address(),
        scene.triangle.address(),
        scene.sphere.address(),
    )[..1]);
    let scratch = scratch(scene.tlas.build_scratch_size());
    let [before, after] = render_many(&mut gpu, |cmd, [first, second]| {
        trace_query(cmd, first, &scene.tlas);
        // SAFETY: the BLASes it names outlive the traces of this test.
        cmd.build(&[(&scene.tlas, unsafe { Instances::new(wall_only.range(..)) })], scratch.range(..));
        trace_query(cmd, second, &scene.tlas);
    });
    assert_query_image(&before);
    // Only the wall is left; the triangle stood in front of its second triangle.
    assert_pixel(&after, TRIANGLE.0, TRIANGLE.1, [122, 122, 122]);
    assert_pixel(&after, SPHERE.0, SPHERE.1, [204, 204, 204]);
}

#[test]
fn compacted_structures_trace_the_same_image() {
    let mut gpu = gpu();
    if !supported("compacted_structures_trace_the_same_image", Ctx::features().ray_query) {
        return;
    }
    let compactable = BuildFlags {
        allow_compaction: true,
        ..Default::default()
    };
    let Scene {
        wall,
        triangle,
        sphere,
        tlas,
        instances: _,
        _inputs,
    } = scene(&mut gpu, compactable, BuildFlags::default());

    let queried = QueryPool::compacted_sizes(3).unwrap();
    let mut sizes = [0; 3];
    // Polling never waits: nothing was queried yet.
    assert!(!queried.read(0, &mut sizes).unwrap());
    gpu.submit(|cmd| cmd.query_compacted_sizes(&[&wall, &triangle, &sphere], &queried, 0));
    assert!(queried.read(0, &mut sizes).unwrap());
    for (size, blas) in sizes.iter().zip([&wall, &triangle, &sphere]) {
        assert!(0 < *size && *size <= blas.size(), "compacted {size} of {}", blas.size());
    }

    let compacted = [
        wall.compacted(sizes[0]).unwrap(),
        triangle.compacted(sizes[1]).unwrap(),
        sphere.compacted(sizes[2]).unwrap(),
    ];
    let [c_wall, c_triangle, c_sphere] = &compacted;
    let moved = buffer_with::<Instance>(&instances(
        c_wall.address(),
        c_triangle.address(),
        c_sphere.address(),
    ));
    let scratch = scratch(tlas.build_scratch_size());
    let pixels = render(&mut gpu, |cmd, target| {
        for (src, dst) in [&wall, &triangle, &sphere].into_iter().zip(&compacted) {
            cmd.compact(src, dst);
        }
        // SAFETY: the BLASes it names outlive the traces of this test.
        cmd.build(&[(&tlas, unsafe { Instances::new(moved.range(..)) })], scratch.range(..));
        // Nothing reaches the originals any more; they go once this frame is done.
        drop((wall, triangle, sphere));
        trace_query(cmd, target, &tlas);
    });
    assert_query_image(&pixels);
    assert_golden("ray_query", &pixels, [SIZE, SIZE]);

    // A compacted structure only has its compacted size, which a build could write past.
    assert!(compacted.iter().all(|c| c.is_compacted()));
    let boxes = buffer_with::<Aabb>(&[Aabb {
        min: Vec3::splat(-1.0),
        max: Vec3::splat(1.0),
    }]);
    let geometry = [Aabbs::new(boxes.range(..))];
    let rebuild = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        gpu.submit(|cmd| {
            cmd.build(&[(c_sphere, BlasGeometry::Aabbs(&geometry))], scratch.range(..))
        })
    }));
    assert!(rebuild.is_err(), "building a compacted structure panics");
}


/// A vertex whose position isn't aligned to its float components.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Unaligned([u8; 12]);
impl VertexPosition for Unaligned {
    const FORMAT: vk::Format = vk::Format::R32G32B32_SFLOAT;
}

#[test]
fn geometry_addresses_are_aligned_to_their_components() {
    let _gpu = gpu();
    let bytes = zeroed_buffer::<u8>(64);
    let vertices = bytes.byte_range(1..37).cast::<Unaligned>();
    let message = panic_message(|| Triangles::new(vertices));
    assert!(message.contains("multiples of the 4 bytes"), "{message}");
}

#[test]
fn batches_build_each_structure_once_and_updates_keep_the_last_build() {
    let mut gpu = gpu();
    if !supported("batches_build_each_structure_once", Ctx::features().acceleration_structure) {
        return;
    }
    let flags = BuildFlags {
        allow_update: true,
        allow_compaction: true,
        ..Default::default()
    };
    let vertices = buffer_with::<Vec3>(&[Vec3::X, Vec3::Y, Vec3::Z, -Vec3::X, -Vec3::Y, -Vec3::Z]);
    let two = [Triangles::new(vertices.range(..))];
    let one = [Triangles::new(vertices.range(..3))];
    let see_through = [Triangles::new(vertices.range(..)).opaque(false)];
    let blas = Blas::new(&BlasGeometry::Triangles(&two), flags).unwrap();
    let scratch = scratch(2 * blas.build_scratch_size().max(blas.update_scratch_size()));
    let build = |cmd: &mut Cmd, geometry: &[Triangles]| {
        cmd.build(&[(&blas, BlasGeometry::Triangles(geometry))], scratch.range(..))
    };
    let update = |cmd: &mut Cmd, geometry: &[Triangles]| {
        cmd.update(&[(&blas, BlasGeometry::Triangles(geometry))], scratch.range(..))
    };
    let message = panic_message(|| gpu.submit(|cmd| update(cmd, &two)));
    assert!(message.contains("built before"), "{message}");
    gpu.submit(|cmd| build(cmd, &two));
    for geometry in [&one[..], &see_through] {
        let message = panic_message(|| gpu.submit(|cmd| update(cmd, geometry)));
        assert!(message.contains("last build"), "{message}");
    }
    let message = panic_message(|| {
        gpu.submit(|cmd| {
            let geometry = BlasGeometry::Triangles(&two);
            cmd.build(&[(&blas, geometry), (&blas, geometry)], scratch.range(..))
        })
    });
    assert!(message.contains("same structure"), "{message}");
    gpu.submit(|cmd| {
        cmd.build(&[(&blas, BlasGeometry::Triangles(&two))], scratch.range(..));
        cmd.update(&[(&blas, BlasGeometry::Triangles(&two))], scratch.range(..));
    });

    let other = Blas::new(&BlasGeometry::Triangles(&two), flags).unwrap();
    let compacted = other.compacted(other.size()).unwrap();
    let message = panic_message(|| gpu.submit(|cmd| cmd.compact(&blas, &compacted)));
    assert!(message.contains("of its source"), "{message}");
    let message = panic_message(|| gpu.submit(|cmd| cmd.compact(&other, &compacted)));
    assert!(message.contains("built before"), "{message}");
}

#[test]
fn traces_stay_within_the_device_limits() {
    let mut gpu = gpu();
    if !supported("traces_stay_within_the_device_limits", Ctx::features().ray_tracing_pipeline) {
        return;
    }
    let limits = &Ctx::device_properties().limits;
    let most = limits.max_compute_work_group_count[0] as u64 * limits.max_compute_work_group_size[0] as u64;
    let Ok(over) = u32::try_from(most + 1) else {
        eprintln!("SKIPPED traces_stay_within_the_device_limits: no width is too wide");
        return;
    };
    let tlas = Tlas::new(1, BuildFlags::default()).unwrap();
    let target = Target::new(EXTENT).unwrap().bind_storage().unwrap();
    let message = panic_message(|| {
        gpu.submit(|cmd| {
            let bindings = TestRayTracing::push_bindings(target.whole_view(), &tlas, EXTENT).unwrap();
            cmd.raytrace(&bindings, [over, 1, 1], None)
        })
    });
    assert!(message.contains("rays"), "{message}");
}

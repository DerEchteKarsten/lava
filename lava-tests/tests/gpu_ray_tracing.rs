//! GPU ray tracing tests: acceleration structure builds (checked against their creation's shapes, batched to fit a scratch, empty ones skipped), refits, compaction by hand and with a `Compactor`, copies, serialization, arenas, `with_capacity`, half, snorm and `Vec3A` positions, instance flags, the built state traces check (applied only once a recording reaches the GPU), the device limits, ray queries and ray tracing pipelines on a small scene, position fetch and shader execution reordering
mod common;

use common::{Cmd, buffer_with, golden::assert_golden, gpu, panic_message, zeroed_buffer};
use glam::{Affine3A, Quat, UVec2, Vec3, Vec3A};
use lava::{
    Error,
    buffer::{Buffer, Gpu, Readback, Upload},
    command_buffer::TraceRaysIndirectCommand,
    image::{Image, format::R8G8B8A8Unorm, usage::usage},
    query::{QueryPool, QueryRecorder, ScopeInfo, ScopeObserver},
    state::Ctx,
    vkobjects::acceleration_structure::{
        Aabb, AabbsDesc, Aabbs, Blas, BlasDesc, BlasGeometry, Bottom, BuildFlags, BuildPreference,
        Compactor, Half4, Instance, InstanceFlags, Instances, Scratch, Snorm16x4, StructureArena,
        Tlas, Triangles, TrianglesDesc, VertexPosition, scratch_alignment,
    },
};
use lava_tests::bindings::*;

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
    let readbacks: [Buffer<u8, Readback>; N] =
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
    readbacks.map(|mut readback| readback.host().to_vec())
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
        &TestRayQuery::push_bindings(TestRayQueryArgs { target: target.whole_view(), tlas, size: EXTENT }, None).unwrap(),
        [SIZE / 8, SIZE / 8, 1]);
}

fn trace_pipeline(cmd: &mut Cmd, target: &Target, tlas: &Tlas) {
    cmd.raytrace(
        &TestRayTracing::push_bindings(TestRayTracingArgs { target: target.whole_view(), tlas, size: EXTENT }, None).unwrap(),
        [SIZE, SIZE, 1]);
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
    instances: Buffer<Instance, Upload>,
    _inputs: (Buffer<Vec3, Upload>, Buffer<u16, Upload>, Buffer<Vec3, Upload>, Buffer<Aabb, Upload>),
}

fn place(scale: f32, position: Vec3) -> Affine3A {
    Affine3A::from_scale_rotation_translation(Vec3::splat(scale), Quat::IDENTITY, position)
}

/// The scene's instances, as `test_write_instances.slang` writes them.
fn instances(wall: u64, triangle: u64, sphere: u64) -> [Instance; 4] {
    [
        Instance::from_address(wall, place(1.0, Vec3::ZERO)),
        Instance::from_address(triangle, place(1.0, Vec3::new(-0.4, -0.3, -0.5))).with_custom_index(1),
        Instance::from_address(sphere, place(0.25, Vec3::new(0.35, 0.35, -0.5)))
            .with_custom_index(2)
            .with_sbt_offset(1),
        Instance::from_address(triangle, place(1.0, Vec3::new(0.45, -0.45, -0.5)))
            .with_custom_index(3)
            .with_mask(0x02),
    ]
}

fn scratch(size: u64) -> Buffer<u8> {
    Buffer::raw(Gpu, size, Some(scratch_alignment())).unwrap()
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

    // Room for the largest build alone, so `build_all` builds one BLAS per batch.
    let scratch = Scratch::for_any(&[&wall_blas, &triangle_blas, &sphere_blas]).unwrap();
    let tlas_scratch = Scratch::for_any(&[&tlas]).unwrap();
    gpu.submit(|cmd| {
        cmd.build_all(
            &[(&wall_blas, wall), (&triangle_blas, triangle), (&sphere_blas, sphere)],
            &scratch,
        );
        // SAFETY: the BLASes it names are built before it and outlive the traces of this test.
        cmd.build_all(&[(&tlas, unsafe { Instances::new(instances.range(..)) })], &tlas_scratch);
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
            &[TraceRaysIndirectCommand {
                width: SIZE,
                height: SIZE,
                depth: 1,
            }],
        );
        cmd.raytrace_indirect(
            &TestRayTracing::push_bindings(TestRayTracingArgs { target: target.whole_view(), tlas: &scene.tlas, size: EXTENT }, None).unwrap(),
            command.range(..),);
    });
    assert_eq!(indirect, direct);
}

#[test]
fn instances_written_by_a_pass_build_a_tlas_in_the_same_submission() {
    let mut gpu = gpu();
    if !supported("instances_written_by_a_pass", Ctx::features().ray_query) {
        return;
    }
    let mut scene = scene(&mut gpu, BuildFlags::default(), BuildFlags::default());
    let mut written = zeroed_buffer::<Instance>(4);
    let tlas = Tlas::new(4, BuildFlags::default()).unwrap();
    let scratch = scratch(tlas.build_scratch_size());
    let pixels = render(&mut gpu, |cmd, target| {
        cmd.compute(
            &TestWriteInstances::push_bindings(TestWriteInstancesArgs { instances: written.range(..), wall: scene.wall.address(), triangle: scene.triangle.address(), sphere: scene.sphere.address() }, None),
            [1, 1, 1]);
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
    fn geometry(vertices: &Buffer<Vec3, Upload>) -> [Triangles<'_>; 1] {
        [Triangles::new(vertices.range(..))]
    }
    let blas = Blas::new(&BlasGeometry::Triangles(&geometry(&vertices)), updatable).unwrap();
    assert!(blas.update_scratch_size() > 0);
    let instances = buffer_with::<Instance>(&[
        Instance::new(&blas, Affine3A::IDENTITY).with_custom_index(1),
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
    let message = panic_message(|| other.compacted(other.size()));
    assert!(message.contains("built before"), "{message}");
    gpu.submit(|cmd| cmd.build(&[(&other, BlasGeometry::Triangles(&two))], scratch.range(..)));
    let compacted = other.compacted(other.size()).unwrap();
    let message = panic_message(|| gpu.submit(|cmd| cmd.compact(&blas, &compacted)));
    assert!(message.contains("of its source"), "{message}");
    let plain = Blas::new(&BlasGeometry::Triangles(&two), BuildFlags::default()).unwrap();
    gpu.submit(|cmd| cmd.build(&[(&plain, BlasGeometry::Triangles(&two))], scratch.range(..)));
    let message = panic_message(|| plain.compacted(plain.size()));
    assert!(message.contains("allow_compaction"), "{message}");
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
    let scene = scene(&mut gpu, BuildFlags::default(), BuildFlags::default());
    let target = Target::new(EXTENT).unwrap().bind_storage().unwrap();
    let message = panic_message(|| {
        gpu.submit(|cmd| {
            let bindings = TestRayTracing::push_bindings(TestRayTracingArgs { target: target.whole_view(), tlas: &scene.tlas, size: EXTENT }, None).unwrap();
            cmd.raytrace(&bindings, [over, 1, 1])
        })
    });
    assert!(message.contains("rays"), "{message}");
}

/// One triangle BLAS of `V` positions with an instance of custom index 1 in a TLAS, traced by
/// a ray query: the triangle of `updates_follow_moved_vertices`, red at (32, 30).
fn trace_triangle_of<V: VertexPosition + bytemuck::Pod>(gpu: &mut common::Gpu, vertices: &[V]) -> Vec<u8> {
    let vertices = buffer_with::<V>(vertices);
    let geometry = [Triangles::new(vertices.range(..))];
    let blas = Blas::new(&BlasGeometry::Triangles(&geometry), BuildFlags::default()).unwrap();
    let instances = buffer_with::<Instance>(&[Instance::new(&blas, Affine3A::IDENTITY).with_custom_index(1)]);
    let tlas = Tlas::new(1, BuildFlags::default()).unwrap();
    let scratch = Scratch::for_any(&[&blas]).unwrap();
    let tlas_scratch = Scratch::for_any(&[&tlas]).unwrap();
    render(gpu, |cmd, target| {
        cmd.build_all(&[(&blas, BlasGeometry::Triangles(&geometry))], &scratch);
        // SAFETY: the BLAS is built before it and outlives the trace.
        cmd.build_all(&[(&tlas, unsafe { Instances::new(instances.range(..)) })], &tlas_scratch);
        trace_query(cmd, target, &tlas);
    })
}

const TRIANGLE_VERTICES: [Vec3; 3] = [Vec3::new(-0.3, -0.3, 0.0), Vec3::new(0.3, -0.3, 0.0), Vec3::new(0.0, 0.3, 0.0)];

#[test]
fn half_snorm_and_padded_positions_build_the_same_triangle() {
    let mut gpu = gpu();
    if !supported("half_snorm_and_padded_positions", Ctx::features().ray_query) {
        return;
    }
    let reference = trace_triangle_of(&mut gpu, &TRIANGLE_VERTICES);
    assert_pixel(&reference, 32, 30, [255, 51, 26]);
    assert_pixel(&reference, 1, 1, [0, 0, 0]);
    // 0.3 rounds to a nearby half or snorm, which moves the edges by less than a pixel's center.
    let half = trace_triangle_of(&mut gpu, &TRIANGLE_VERTICES.map(Half4::new));
    let snorm = trace_triangle_of(&mut gpu, &TRIANGLE_VERTICES.map(Snorm16x4::new));
    assert_eq!(half, reference);
    assert_eq!(snorm, reference);
    // `Vec3A` isn't `Pod` (its padding), so the host writes it through an `Upload` mapping.
    let mut padded: Buffer<Vec3A, Upload> = Buffer::upload(3).unwrap();
    padded.host_mut().copy_from_slice(&TRIANGLE_VERTICES.map(Vec3A::from));
    let geometry = [Triangles::new(padded.range(..))];
    let blas = Blas::new(&BlasGeometry::Triangles(&geometry), BuildFlags::default()).unwrap();
    let instances = buffer_with::<Instance>(&[Instance::new(&blas, Affine3A::IDENTITY).with_custom_index(1)]);
    let tlas = Tlas::new(1, BuildFlags::default()).unwrap();
    let scratch = Scratch::new(blas.build_scratch_size().max(tlas.build_scratch_size())).unwrap();
    let pixels = render(&mut gpu, |cmd, target| {
        cmd.build(&[(&blas, BlasGeometry::Triangles(&geometry))], scratch.slice());
        // SAFETY: the BLAS is built before it and outlives the trace.
        cmd.build(&[(&tlas, unsafe { Instances::new(instances.range(..)) })], scratch.slice());
        trace_query(cmd, target, &tlas);
    });
    assert_eq!(pixels, reference);
}

#[test]
fn builds_have_the_shape_their_structure_was_created_for() {
    let mut gpu = gpu();
    if !supported("builds_have_the_shape", Ctx::features().acceleration_structure) {
        return;
    }
    let vertices = buffer_with::<Vec3>(&[Vec3::X, Vec3::Y, Vec3::Z, -Vec3::X, -Vec3::Y, -Vec3::Z]);
    let half = buffer_with::<Half4>(&[Half4::new(Vec3::X), Half4::new(Vec3::Y), Half4::new(Vec3::Z)]);
    let boxes = buffer_with::<Aabb>(&[Aabb { min: Vec3::ZERO, max: Vec3::ONE }]);
    let indices = buffer_with::<u16>(&[0, 1, 2]);
    let one = [Triangles::new(vertices.range(..3))];
    let blas = Blas::new(&BlasGeometry::Triangles(&one), BuildFlags::default()).unwrap();
    let scratch = Scratch::for_any(&[&blas]).unwrap();
    let build = |cmd: &mut Cmd, geometry: BlasGeometry| cmd.build_all(&[(&blas, geometry)], &scratch);
    let refused = [
        (BlasGeometry::Aabbs(&[Aabbs::new(boxes.range(..))]), "AABBS"),
        (BlasGeometry::Triangles(&[Triangles::new(vertices.range(..))]), "doesn't fit"),
        (BlasGeometry::Triangles(&[Triangles::new(half.range(..))]), "R16G16B16A16_SFLOAT"),
        (BlasGeometry::Triangles(&[Triangles::indexed(vertices.range(..3), indices.range(..))]), "UINT16"),
        (BlasGeometry::Triangles(&[Triangles::new(vertices.range(..3)).opaque(false)]), "doesn't fit"),
        (BlasGeometry::Triangles(&[one[0], one[0]]), "as many geometries"),
    ];
    for (geometry, expected) in refused {
        let message = panic_message(|| gpu.submit(|cmd| build(cmd, geometry)));
        assert!(message.contains(expected), "{expected}: {message}");
    }
    gpu.submit(|cmd| build(cmd, BlasGeometry::Triangles(&one)));

    // Sized from counts: up to 6 vertices and 2 triangles of u16 indices, built with fewer.
    let desc = [TrianglesDesc::indexed::<Vec3, u16>(6, 2)];
    let sized = Blas::with_capacity(&BlasDesc::Triangles(&desc), BuildFlags::default()).unwrap();
    let real = Blas::new(
        &BlasGeometry::Triangles(&[Triangles::indexed(vertices.range(..), buffer_with::<u16>(&[0, 1, 2, 3, 4, 5]).range(..))]),
        BuildFlags::default(),
    )
    .unwrap();
    assert_eq!((sized.size(), sized.build_scratch_size()), (real.size(), real.build_scratch_size()));
    let fewer = [Triangles::indexed(vertices.range(..3), indices.range(..))];
    let scratch = Scratch::for_any(&[&sized]).unwrap();
    gpu.submit(|cmd| cmd.build_all(&[(&sized, BlasGeometry::Triangles(&fewer))], &scratch));
    let boxes_desc = [AabbsDesc::new(4).opaque(false)];
    let procedural = Blas::with_capacity(&BlasDesc::Aabbs(&boxes_desc), BuildFlags::default()).unwrap();
    let see_through = [Aabbs::new(boxes.range(..)).opaque(false)];
    let scratch = Scratch::for_any(&[&procedural]).unwrap();
    gpu.submit(|cmd| cmd.build_all(&[(&procedural, BlasGeometry::Aabbs(&see_through))], &scratch));
}

#[test]
fn a_structure_counts_as_built_only_once_its_build_reaches_the_gpu() {
    let mut gpu = gpu();
    if !supported("a_structure_counts_as_built", Ctx::features().ray_query) {
        return;
    }
    let scene = scene(&mut gpu, BuildFlags::default(), BuildFlags::default());
    let instances = buffer_with::<Instance>(&instances(scene.wall.address(), scene.triangle.address(), scene.sphere.address()));
    let tlas = Tlas::new(4, BuildFlags::default()).unwrap();
    let target = Target::new(EXTENT).unwrap().bind_storage().unwrap();
    // Never built: its bindings refuse it (a trace would read uninitialised memory).
    let message = panic_message(|| TestRayQuery::push_bindings(TestRayQueryArgs { target: target.whole_view(), tlas: &tlas, size: EXTENT }, None));
    assert!(message.contains("never built"), "{message}");
    let scratch = Scratch::for_any(&[&tlas]).unwrap();
    // A recording that fails takes its builds back.
    let message = panic_message(|| {
        gpu.submit(|cmd| {
            // SAFETY: the BLASes it names are built and outlive this test's traces.
            cmd.build_all(&[(&tlas, unsafe { Instances::new(instances.range(..)) })], &scratch);
            assert!(tlas.is_built(), "built by this recording");
            panic!("the recording fails");
        })
    });
    assert!(message.contains("the recording fails"), "{message}");
    assert!(!tlas.is_built());
    let message = panic_message(|| gpu.submit(|cmd| cmd.update(&[(&tlas, unsafe { Instances::new(instances.range(..)) })], scratch.slice())));
    assert!(message.contains("allow_update"), "{message}");
    let pixels = render(&mut gpu, |cmd, target| {
        // SAFETY: as above.
        cmd.build_all(&[(&tlas, unsafe { Instances::new(instances.range(..)) })], &scratch);
        trace_query(cmd, target, &tlas);
    });
    assert!(tlas.is_built());
    assert_query_image(&pixels);
}

#[test]
fn empty_builds_queries_and_copies_record_nothing() {
    let mut gpu = gpu();
    if !supported("empty_builds_record_nothing", Ctx::features().acceleration_structure) {
        return;
    }
    let scratch = Scratch::new(1).unwrap();
    let sizes = QueryPool::compacted_sizes(1).unwrap();
    let serialized = QueryPool::serialized_sizes(1).unwrap();
    gpu.submit(|cmd| {
        cmd.build::<Bottom>(&[], scratch.slice());
        cmd.update::<Bottom>(&[], scratch.slice());
        cmd.build_all::<Bottom>(&[], &scratch);
        cmd.query_compacted_sizes::<Bottom>(&[], &sizes, 0);
        cmd.query_serialized_sizes::<Bottom>(&[], &serialized, 0);
    });
}

#[test]
fn structures_beyond_the_device_limits_are_errors() {
    let _gpu = gpu();
    if !supported("structures_beyond_the_device_limits", Ctx::features().acceleration_structure) {
        return;
    }
    let properties = Ctx::acceleration_structure_properties().unwrap();
    let max_instances = properties.max_instance_count;
    if let Ok(over) = u32::try_from(max_instances + 1) {
        let Err(Error::OverLimit { limit, requested, max }) = Tlas::new(over, BuildFlags::default()) else {
            panic!("a TLAS of more than maxInstanceCount instances is an error");
        };
        assert_eq!((limit, requested, max), ("maxInstanceCount", max_instances + 1, max_instances));
    }
    let max_primitives = properties.max_primitive_count;
    if let Ok(over) = u32::try_from(max_primitives + 1) {
        let desc = [AabbsDesc::new(over)];
        let result = Blas::with_capacity(&BlasDesc::Aabbs(&desc), BuildFlags::default());
        assert!(matches!(result, Err(Error::OverLimit { limit: "maxPrimitiveCount", .. })));
    }
    let max_geometries = properties.max_geometry_count;
    if max_geometries < 1 << 16 {
        let desc = vec![AabbsDesc::new(1); max_geometries as usize + 1];
        let result = Blas::with_capacity(&BlasDesc::Aabbs(&desc), BuildFlags::default());
        assert!(matches!(result, Err(Error::OverLimit { limit: "maxGeometryCount", .. })));
    } else {
        eprintln!("SKIPPED the maxGeometryCount check: {max_geometries} geometries are too many to list");
    }
}

/// Counts the scopes it sees by name.
#[derive(Default)]
struct Scopes(Vec<&'static str>);

impl ScopeObserver for Scopes {
    fn on_enter(&mut self, _: &mut QueryRecorder, scope: &ScopeInfo) {
        self.0.push(scope.name);
    }
    fn on_exit(&mut self, _: &mut QueryRecorder, _: &ScopeInfo) {}
}

#[test]
fn compactors_compact_over_a_few_frames_and_hand_back_traceable_structures() {
    let mut gpu = gpu();
    if !supported("compactors_compact_over_a_few_frames", Ctx::features().ray_query) {
        return;
    }
    let compactable = BuildFlags {
        allow_compaction: true,
        prefer: BuildPreference::None,
        ..Default::default()
    };
    let Scene { wall, triangle, sphere, tlas, instances: _, _inputs } = scene(&mut gpu, compactable, BuildFlags::default());
    let sizes = [wall.size(), triangle.size(), sphere.size()];
    let mut compactor = Compactor::new();
    for (key, blas) in [wall, triangle, sphere].into_iter().enumerate() {
        compactor.add(key, blas);
    }
    let mut compacted: Vec<Option<Blas>> = vec![None, None, None];
    let mut scopes = Scopes::default();
    for _frame in 0..8 {
        gpu.submit(|cmd| {
            cmd.measure(Some(&mut scopes), "compaction", |cmd| compactor.record(cmd).unwrap())
        });
        for (key, blas) in compactor.take_finished() {
            compacted[key] = Some(blas);
        }
        if compactor.is_empty() {
            break;
        }
    }
    assert!(compactor.is_empty(), "every structure is compacted within a few frames");
    assert!(scopes.0.contains(&"query compacted sizes") && scopes.0.contains(&"compact acceleration structure"), "{:?}", scopes.0);
    let compacted: Vec<Blas> = compacted.into_iter().map(Option::unwrap).collect();
    for (blas, size) in compacted.iter().zip(sizes) {
        assert!(blas.is_compacted() && blas.is_built() && blas.size() <= size);
    }
    let moved = buffer_with::<Instance>(&instances(compacted[0].address(), compacted[1].address(), compacted[2].address()));
    let scratch = Scratch::for_any(&[&tlas]).unwrap();
    let pixels = render(&mut gpu, |cmd, target| {
        // SAFETY: the compacted BLASes outlive the trace.
        cmd.build_all(&[(&tlas, unsafe { Instances::new(moved.range(..)) })], &scratch);
        trace_query(cmd, target, &tlas);
    });
    assert_query_image(&pixels);
}

#[test]
fn arenas_hold_structures_and_take_their_ranges_back() {
    let mut gpu = gpu();
    if !supported("arenas_hold_structures", Ctx::features().ray_query) {
        return;
    }
    let reference = trace_triangle_of(&mut gpu, &TRIANGLE_VERTICES);
    let vertices = buffer_with::<Vec3>(&TRIANGLE_VERTICES);
    let geometry = [Triangles::new(vertices.range(..))];
    let own = Blas::new(&BlasGeometry::Triangles(&geometry), BuildFlags::default()).unwrap();
    let tlas_size = Tlas::new(1, BuildFlags::default()).unwrap().size();
    // Room for one BLAS and one TLAS exactly.
    let arena = StructureArena::new(own.size() + tlas_size).unwrap().named("test arena");
    let blas = arena.blas(&BlasGeometry::Triangles(&geometry), BuildFlags::default()).unwrap();
    let tlas = arena.tlas(1, BuildFlags::default()).unwrap();
    assert!(matches!(arena.tlas(1, BuildFlags::default()), Err(Error::ArenaFull { .. })));
    let instances = buffer_with::<Instance>(&[Instance::new(&blas, Affine3A::IDENTITY).with_custom_index(1)]);
    let scratch = Scratch::for_any(&[&blas]).unwrap();
    let tlas_scratch = Scratch::for_any(&[&tlas]).unwrap();
    // The arena can go before its structures, which keep its memory.
    drop(arena);
    let pixels = render(&mut gpu, |cmd, target| {
        cmd.build_all(&[(&blas, BlasGeometry::Triangles(&geometry))], &scratch);
        // SAFETY: the BLAS is built before it and outlives the trace.
        cmd.build_all(&[(&tlas, unsafe { Instances::new(instances.range(..)) })], &tlas_scratch);
        trace_query(cmd, target, &tlas);
    });
    assert_eq!(pixels, reference);

    let arena = StructureArena::new(own.size()).unwrap();
    let first = arena.blas(&BlasGeometry::Triangles(&geometry), BuildFlags::default()).unwrap();
    assert!(arena.blas(&BlasGeometry::Triangles(&geometry), BuildFlags::default()).is_err());
    drop(first);
    lava::wait_idle().unwrap();
    // Destroyed once no frame reaches it: its range is free again.
    arena.blas(&BlasGeometry::Triangles(&geometry), BuildFlags::default()).unwrap();
}

#[test]
fn copies_and_serialized_structures_trace_like_their_sources() {
    let mut gpu = gpu();
    if !supported("copies_and_serialized_structures", Ctx::features().ray_query) {
        return;
    }
    let scene = scene(&mut gpu, BuildFlags::default(), BuildFlags::default());
    let sources = [&scene.wall, &scene.triangle, &scene.sphere];
    let clones: Vec<Blas> = sources.iter().map(|s| s.clone_target().unwrap()).collect();
    let sizes = QueryPool::serialized_sizes(3).unwrap();
    gpu.submit(|cmd| {
        for (source, clone) in sources.iter().zip(&clones) {
            cmd.copy_structure(source, clone);
        }
        cmd.query_serialized_sizes(&sources, &sizes, 0);
    });
    assert!(clones.iter().all(|c| c.is_built() && c.is_buildable()));
    let mut serialized_sizes = [0; 3];
    assert!(sizes.read(0, &mut serialized_sizes).unwrap());

    // Serialized into one readback buffer, each at a multiple of 256.
    let offsets: Vec<u64> = serialized_sizes
        .iter()
        .scan(0, |offset, size| {
            let at = *offset;
            *offset += size.div_ceil(256) * 256;
            Some(at)
        })
        .collect();
    let total = offsets[2] + serialized_sizes[2];
    let mut readback: Buffer<u8, Readback> = Buffer::raw(Readback, total, Some(256)).unwrap();
    gpu.submit(|cmd| {
        for ((source, offset), size) in sources.iter().zip(&offsets).zip(serialized_sizes) {
            cmd.serialize(source, readback.byte_range(*offset as usize..(offset + size) as usize));
        }
    });
    let bytes = readback.host().to_vec();
    let blobs: Vec<&[u8]> = (offsets.iter().zip(serialized_sizes))
        .map(|(&offset, size)| &bytes[offset as usize..(offset + size) as usize])
        .collect();
    // Another driver's data (here: changed driver and compatibility UUIDs) is refused.
    let mut foreign = blobs[0].to_vec();
    foreign[0] ^= 0xFF;
    foreign[16] ^= 0xFF;
    assert!(matches!(Blas::deserialized(&foreign), Err(Error::Incompatible)));

    let mut upload: Buffer<u8, Upload> = Buffer::raw(Upload, total, Some(256)).unwrap();
    upload.host_write(0, &bytes);
    let deserialized: Vec<Blas> = blobs.iter().map(|blob| Blas::deserialized(blob).unwrap()).collect();
    gpu.submit(|cmd| {
        for ((blas, offset), size) in deserialized.iter().zip(&offsets).zip(serialized_sizes) {
            cmd.deserialize(upload.byte_range(*offset as usize..(offset + size) as usize), blas);
        }
    });
    assert!(deserialized.iter().all(|d| d.is_built() && !d.is_buildable()));

    let scratch = Scratch::for_any(&[&scene.tlas]).unwrap();
    for blases in [&clones, &deserialized] {
        let moved = buffer_with::<Instance>(&instances(blases[0].address(), blases[1].address(), blases[2].address()));
        let pixels = render(&mut gpu, |cmd, target| {
            // SAFETY: the BLASes outlive the trace.
            cmd.build_all(&[(&scene.tlas, unsafe { Instances::new(moved.range(..)) })], &scratch);
            trace_query(cmd, target, &scene.tlas);
        });
        assert_query_image(&pixels);
    }
}

#[test]
fn instance_flags_force_opaque_geometry() {
    let mut gpu = gpu();
    if !supported("instance_flags_force_opaque_geometry", Ctx::features().ray_tracing_pipeline) {
        return;
    }
    let scene = scene(&mut gpu, BuildFlags::default(), BuildFlags::default());
    let mut forced = instances(scene.wall.address(), scene.triangle.address(), scene.sphere.address());
    forced[0] = forced[0].with_flags(InstanceFlags::FORCE_OPAQUE);
    assert_eq!(forced[0].flags(), InstanceFlags::FORCE_OPAQUE);
    let forced = buffer_with::<Instance>(&forced);
    let scratch = Scratch::for_any(&[&scene.tlas]).unwrap();
    let pixels = render(&mut gpu, |cmd, target| {
        // SAFETY: the scene's BLASes outlive the trace.
        cmd.build_all(&[(&scene.tlas, unsafe { Instances::new(forced.range(..)) })], &scratch);
        trace_pipeline(cmd, target, &scene.tlas);
    });
    // No any-hit shader runs on the wall, so it has no holes.
    assert_pixel(&pixels, HOLE.0, HOLE.1, [204, 204, 204]);
}

/// A TLAS of the wall alone, its BLAS built with `flags`.
fn wall_scene(gpu: &mut common::Gpu, flags: BuildFlags) -> (Tlas, Blas, Buffer<Vec3, Upload>, Buffer<u16, Upload>) {
    let vertices = buffer_with::<Vec3>(&[
        Vec3::new(-0.9, -0.9, 0.0),
        Vec3::new(0.9, -0.9, 0.0),
        Vec3::new(0.9, 0.9, 0.0),
        Vec3::new(-0.9, 0.9, 0.0),
    ]);
    let indices = buffer_with::<u16>(&[0, 1, 2, 0, 2, 3]);
    let geometry = [Triangles::indexed(vertices.range(..), indices.range(..))];
    let blas = Blas::new(&BlasGeometry::Triangles(&geometry), flags).unwrap();
    let instances = buffer_with::<Instance>(&[Instance::new(&blas, Affine3A::IDENTITY)]);
    let tlas = Tlas::new(1, BuildFlags::default()).unwrap();
    let scratch = Scratch::for_any(&[&blas]).unwrap();
    let tlas_scratch = Scratch::for_any(&[&tlas]).unwrap();
    gpu.submit(|cmd| {
        cmd.build_all(&[(&blas, BlasGeometry::Triangles(&geometry))], &scratch);
        // SAFETY: the BLAS is built before it and outlives the traces.
        cmd.build_all(&[(&tlas, unsafe { Instances::new(instances.range(..)) })], &tlas_scratch);
    });
    (tlas, blas, vertices, indices)
}

#[test]
fn hit_shaders_fetch_the_positions_of_their_triangles() {
    let mut gpu = gpu();
    let features = Ctx::features();
    if !supported("hit_shaders_fetch_positions", features.ray_tracing_pipeline && features.ray_tracing_position_fetch) {
        return;
    }
    let flags = BuildFlags {
        allow_data_access: true,
        ..Default::default()
    };
    let (tlas, _blas, _vertices, _indices) = wall_scene(&mut gpu, flags);
    let pixels = render(&mut gpu, |cmd, target| {
        cmd.raytrace(&TestPositionFetch::push_bindings(TestPositionFetchArgs { target: target.whole_view(), tlas: &tlas, size: EXTENT }, None).unwrap(), [SIZE, SIZE, 1]);
    });
    // The second vertex of the first triangle is (0.9, -0.9, 0), of the second (0.9, 0.9, 0).
    assert_pixel(&pixels, WALL.0, WALL.1, [242, 13, 128]);
    assert_pixel(&pixels, WALL_SECOND_TRIANGLE.0, WALL_SECOND_TRIANGLE.1, [242, 242, 128]);
    assert_pixel(&pixels, BACKGROUND.0, BACKGROUND.1, [0, 0, 0]);
}

#[test]
fn reordered_traces_hit_like_plain_ones() {
    let mut gpu = gpu();
    let features = Ctx::features();
    if !features.ray_tracing_invocation_reorder {
        eprintln!("SKIPPED reordered_traces_hit_like_plain_ones: the device has no shader execution reordering");
        assert!(TestReorder::push_bindings(TestReorderArgs { target: Target::new(EXTENT).unwrap().bind_storage().unwrap().whole_view(), tlas: &wall_scene(&mut gpu, BuildFlags::default()).0, size: EXTENT }, None).is_err());
        return;
    }
    let (tlas, _blas, _vertices, _indices) = wall_scene(&mut gpu, BuildFlags::default());
    let pixels = render(&mut gpu, |cmd, target| {
        cmd.raytrace(&TestReorder::push_bindings(TestReorderArgs { target: target.whole_view(), tlas: &tlas, size: EXTENT }, None).unwrap(), [SIZE, SIZE, 1]);
    });
    assert_pixel(&pixels, WALL.0, WALL.1, [204, 204, 204]);
    assert_pixel(&pixels, WALL_SECOND_TRIANGLE.0, WALL_SECOND_TRIANGLE.1, [122, 122, 122]);
    assert_pixel(&pixels, BACKGROUND.0, BACKGROUND.1, [0, 0, 0]);
}

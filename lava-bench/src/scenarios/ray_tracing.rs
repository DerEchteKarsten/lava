//! `ray_tracing` (needs `ray-query`): compute skinning of 64 meshes, a refit of each one's BLAS, a TLAS rebuilt each frame from 256 instances that reference those BLASes, and a ray-query ambient occlusion pass

use glam::{Affine3A, Mat4, Quat, UVec2, Vec3};
use lava::acceleration_structure::{
    BlasGeometry, BuildFlags, Instance, InstanceInfo, Instances, RawInstance, Scratch, Triangles,
};
use lava::bindless::{Sampled, Storage};
use lava::buffer::Upload;
use lava::image::format::R8G8B8A8Unorm;
use lava::vk::{self, ImageLayout as L};
use lava::{Blas, Buffer, Image, Tlas, usage};

use crate::IN_FLIGHT;
use crate::bindings::*;
use crate::check::Checksums;
use crate::pipelines;
use crate::raw::{self, RawBuffer, RawCmd, img};
use crate::scenario::{LavaCtx, RawCtx, Scenario, camera, sphere};
use crate::scenarios::Setup;

const MESHES: usize = 64;
const INSTANCES: usize = 256;
const EXTENT: UVec2 = UVec2::new(480, 270);
const DT: f32 = 1.0 / 60.0;
/// Refittable BLASes.
fn update() -> BuildFlags {
    BuildFlags {
        allow_update: true,
        ..Default::default()
    }
}

/// Instance `i`'s transform at `time`: a 16 x 16 grid of turning meshes.
fn transform(i: usize, time: f32) -> Affine3A {
    let position = Vec3::new((i % 16) as f32 * 2.5 - 19.0, 0.0, (i / 16) as f32 * 2.5 - 19.0);
    Affine3A::from_rotation_translation(Quat::from_rotation_y(time + i as f32), position)
}

struct LavaSet {
    positions: Buffer<Vec3>,
    blases: Vec<Blas>,
    tlas: Tlas,
    instances: Vec<Buffer<Instance, Upload>>,
    blas_scratch: Scratch,
    tlas_scratch: Scratch,
    target: Image<R8G8B8A8Unorm, usage![Storage, Sampled]>,
    time: f32,
    frame: usize,
}

/// The baseline's structures in one buffer of its own, with their sizes and addresses.
struct RawSet {
    positions: Buffer<Vec3>,
    /// Holds the structures.
    _storage: RawBuffer,
    blases: Vec<(vk::AccelerationStructureKHR, u64)>,
    tlas: (vk::AccelerationStructureKHR, u64),
    instances: Vec<Buffer<RawInstance, Upload>>,
    /// One update scratch range per BLAS, then the TLAS's build scratch.
    scratch: Buffer<u8>,
    update_scratch: u64,
    target: Image<R8G8B8A8Unorm, usage![Storage, Sampled]>,
    time: f32,
    frame: usize,
}

pub struct RayTracing {
    lava: LavaSet,
    raw: RawSet,
    rest: Buffer<MeshVertex>,
    indices: Buffer<u32>,
    vertices: u32,
    triangles: u32,
    view_proj: Mat4,
    eye: Vec3,
    functions: ash::khr::acceleration_structure::Device,
    checksums: Checksums,
    pipelines: [vk::Pipeline; 2],
}

/// The baseline's triangle geometry of `vertices` positions at `positions`.
fn triangle_geometry<'a>(positions: u64, vertices: u32, indices: u64) -> vk::AccelerationStructureGeometryKHR<'a> {
    let data = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
        .vertex_format(vk::Format::R32G32B32_SFLOAT)
        .vertex_data(vk::DeviceOrHostAddressConstKHR { device_address: positions })
        .vertex_stride(12)
        .max_vertex(vertices - 1)
        .index_type(vk::IndexType::UINT32)
        .index_data(vk::DeviceOrHostAddressConstKHR { device_address: indices });
    vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .geometry(vk::AccelerationStructureGeometryDataKHR { triangles: data })
        .flags(vk::GeometryFlagsKHR::OPAQUE)
}

fn instances_geometry<'a>(address: u64) -> vk::AccelerationStructureGeometryKHR<'a> {
    let data = vk::AccelerationStructureGeometryInstancesDataKHR::default()
        .data(vk::DeviceOrHostAddressConstKHR { device_address: address });
    vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .geometry(vk::AccelerationStructureGeometryDataKHR { instances: data })
}

const BLAS_FLAGS: vk::BuildAccelerationStructureFlagsKHR = vk::BuildAccelerationStructureFlagsKHR::from_raw(
    vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE.as_raw()
        | vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE.as_raw(),
);

impl RayTracing {
    pub fn new(setup: &mut Setup) -> Option<Self> {
        if !lava::Ctx::features().ray_query {
            return None;
        }
        let (rest, indices) = sphere(8, 12);
        let (vertices, triangles) = (rest.len() as u32, indices.len() as u32 / 3);
        let uploader = setup.uploader();
        let rest = uploader.from_slice(&rest).unwrap();
        let indices = uploader.from_slice(&indices).unwrap();
        let positions = || Buffer::new(MESHES * vertices as usize).unwrap();
        let target = || Image::new(EXTENT).unwrap().bind_combined().unwrap();
        // Lava: structures sized for their geometry, built once below and refitted per frame.
        let lava_positions = positions();
        let blases: Vec<_> = (0..MESHES)
            .map(|m| {
                let range = lava_positions.range(m * vertices as usize..(m + 1) * vertices as usize);
                let geometry = [Triangles::indexed(range, indices.range(..))];
                Blas::new(&BlasGeometry::Triangles(&geometry), update()).unwrap()
            })
            .collect();
        let blas_refs: Vec<_> = blases.iter().collect();
        let tlas = Tlas::new(INSTANCES as u32, BuildFlags::default()).unwrap();
        let lava = LavaSet {
            // Room for every refit at once: `Scratch::for_any` holds one, and `update_all` then
            // records 64 batches with a barrier between each.
            blas_scratch: Scratch::new(blas_refs.iter().map(|b| b.update_scratch_size()).sum())
                .unwrap(),
            tlas_scratch: Scratch::for_any(&[&tlas]).unwrap(),
            positions: lava_positions,
            blases,
            tlas,
            instances: (0..IN_FLIGHT).map(|_| Buffer::upload(INSTANCES).unwrap()).collect(),
            target: target(),
            time: 0.0,
            frame: 0,
        };

        // The baseline: the same structures by hand, in one buffer.
        let device = lava::raw::device();
        let functions = ash::khr::acceleration_structure::Device::new(lava::raw::instance(), device);
        let raw_positions = positions();
        let sizes = |geometry: &vk::AccelerationStructureGeometryKHR, ty, flags, count: u32| {
            let geometries = [*geometry];
            let info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .ty(ty)
                .flags(flags)
                .geometries(&geometries);
            let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
            unsafe {
                functions.get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &info,
                    &[count],
                    &mut sizes,
                )
            };
            sizes
        };
        let blas_sizes = sizes(
            &triangle_geometry(raw_positions.address(), vertices, indices.address()),
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            BLAS_FLAGS,
            triangles,
        );
        let tlas_sizes = sizes(
            &instances_geometry(0),
            vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
            INSTANCES as u32,
        );
        let align = |n: u64, to: u64| n.next_multiple_of(to);
        let blas_size = align(blas_sizes.acceleration_structure_size, 256);
        let storage = RawBuffer::new(
            blas_size * MESHES as u64 + tlas_sizes.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
        );
        let create = |offset: u64, size: u64, ty| {
            let info = vk::AccelerationStructureCreateInfo2KHR::default()
                .address_range(raw::range(storage.address + offset, size))
                .address_flags(
                    vk::AddressCommandFlagsKHR::UNKNOWN_STORAGE_BUFFER_USAGE
                        | vk::AddressCommandFlagsKHR::FULLY_BOUND,
                )
                .ty(ty);
            let structure = unsafe {
                lava::raw::device_address_commands()
                    .create_acceleration_structure2(&info, None)
                    .unwrap()
            };
            let address = unsafe {
                functions.get_acceleration_structure_device_address(
                    &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                        .acceleration_structure(structure),
                )
            };
            (structure, address)
        };
        let raw_blases = (0..MESHES as u64)
            .map(|m| create(m * blas_size, blas_sizes.acceleration_structure_size, vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL))
            .collect();
        let raw_tlas = create(
            blas_size * MESHES as u64,
            tlas_sizes.acceleration_structure_size,
            vk::AccelerationStructureTypeKHR::TOP_LEVEL,
        );
        let scratch_alignment = lava::acceleration_structure::scratch_alignment();
        let update_scratch = align(
            blas_sizes.update_scratch_size.max(blas_sizes.build_scratch_size),
            scratch_alignment,
        );
        let raw = RawSet {
            scratch: Buffer::raw(
                lava::buffer::Gpu,
                update_scratch * MESHES as u64 + tlas_sizes.build_scratch_size,
                Some(scratch_alignment),
            )
            .unwrap(),
            positions: raw_positions,
            _storage: storage,
            blases: raw_blases,
            tlas: raw_tlas,
            instances: (0..IN_FLIGHT).map(|_| Buffer::upload(INSTANCES).unwrap()).collect(),
            update_scratch,
            target: target(),
            time: 0.0,
            frame: 0,
        };
        let view_proj = camera(Vec3::new(0.0, 14.0, 30.0), EXTENT.x as f32 / EXTENT.y as f32);
        let mut rt = Self {
            lava,
            raw,
            rest,
            indices,
            vertices,
            triangles,
            view_proj,
            eye: Vec3::new(0.0, 14.0, 30.0),
            functions,
            checksums: Checksums::new(2),
            pipelines: [pipelines::compute("Skin"), pipelines::compute("RtAo")],
        };
        // The first frame refits: build every BLAS once from the rest pose.
        let s = &rt.lava;
        setup.lava(|cmd| {
            rt.skin_lava(cmd, s);
            let geometry = rt.lava_geometry(s);
            let builds: Vec<_> = s.blases.iter().zip(&geometry).map(|(b, g)| (b, BlasGeometry::Triangles(g))).collect();
            cmd.build_all(&builds, &s.blas_scratch);
        });
        let s = &rt.raw;
        setup.raw(|cmd| {
            rt.skin_raw(cmd, s);
            cmd.mem(raw::COMPUTE_WRITE, raw::BUILD_READ);
            rt.blas_builds(cmd, s, vk::BuildAccelerationStructureModeKHR::BUILD);
        });
        unsafe {
            lava::raw::write_descriptors(rt.raw.target.whole_view(), L::SHADER_READ_ONLY_OPTIMAL)
                .unwrap()
        };
        rt.lava.time = 0.0;
        rt.raw.time = 0.0;
        Some(rt)
    }

    fn lava_geometry<'a>(&'a self, s: &'a LavaSet) -> Vec<[Triangles<'a>; 1]> {
        let v = self.vertices as usize;
        (0..MESHES)
            .map(|m| [Triangles::indexed(s.positions.range(m * v..(m + 1) * v), self.indices.range(..))])
            .collect()
    }

    fn skin_lava(&self, cmd: &mut lava::CommandBuffer<lava::queue::Graphics>, s: &LavaSet) {
        let v = self.vertices as usize;
        for mesh in 0..MESHES {
            let args = SkinArgs {
                rest: self.rest.range(..),
                positions: s.positions.range(mesh * v..(mesh + 1) * v),
                count: self.vertices,
                mesh: mesh as u32,
                time: s.time,
            };
            cmd.compute(&Skin::push_bindings(args, None), Skin::groups([self.vertices, 1, 1]));
        }
    }

    /// The baseline's skinning: no barriers between the meshes, whose ranges are disjoint.
    fn skin_raw(&self, cmd: &mut RawCmd, s: &RawSet) {
        let mut push: CSkinBindings = bytemuck::Zeroable::zeroed();
        push.rest = self.rest.address();
        push.count = self.vertices;
        push.time = s.time;
        cmd.bind_compute(self.pipelines[0]);
        for mesh in 0..MESHES {
            push.positions = s.positions.address() + (mesh * self.vertices as usize * 12) as u64;
            push.mesh = mesh as u32;
            cmd.push(&push);
            cmd.dispatch(Skin::groups([self.vertices, 1, 1]));
        }
    }

    /// One build call of every BLAS, each with its own scratch range.
    fn blas_builds(&self, cmd: &mut RawCmd, s: &RawSet, mode: vk::BuildAccelerationStructureModeKHR) {
        let geometries: Vec<_> = (0..MESHES)
            .map(|m| {
                let positions = s.positions.address() + (m * self.vertices as usize * 12) as u64;
                [triangle_geometry(positions, self.vertices, self.indices.address())]
            })
            .collect();
        let infos: Vec<_> = (s.blases.iter().zip(&geometries).enumerate())
            .map(|(m, (&(structure, _), geometry))| {
                let mut info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                    .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                    .flags(BLAS_FLAGS)
                    .mode(mode)
                    .dst_acceleration_structure(structure)
                    .geometries(geometry)
                    .scratch_data(vk::DeviceOrHostAddressKHR {
                        device_address: s.scratch.address() + m as u64 * s.update_scratch,
                    });
                if mode == vk::BuildAccelerationStructureModeKHR::UPDATE {
                    info = info.src_acceleration_structure(structure);
                }
                info
            })
            .collect();
        let range = [vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(self.triangles)];
        let ranges: Vec<&[_]> = (0..MESHES).map(|_| &range[..]).collect();
        unsafe { self.functions.cmd_build_acceleration_structures(cmd.cmd, &infos, &ranges) };
    }
}

impl Scenario for RayTracing {
    fn lava(&mut self, ctx: &mut LavaCtx) {
        self.lava.time += DT;
        self.lava.frame += 1;
        let slot = self.lava.frame % IN_FLIGHT as usize;
        let s = &mut self.lava;
        s.instances[slot].host_write_referencing(
            0,
            (0..INSTANCES).map(|i| InstanceInfo::new(&s.blases[i % MESHES], transform(i, s.time))),
        );
        let (this, s) = (&*self, &self.lava);
        let check = ctx.check;
        ctx.submit(|cmd| {
            this.skin_lava(cmd, s);
            let geometry = this.lava_geometry(s);
            let updates: Vec<_> = s.blases.iter().zip(&geometry).map(|(b, g)| (b, BlasGeometry::Triangles(g))).collect();
            cmd.update_all(&updates, &s.blas_scratch);
            cmd.build_all(&[(&s.tlas, Instances::new(s.instances[slot].range(..)))], &s.tlas_scratch);
            let args = RtAoArgs {
                inv_view_proj: this.view_proj.inverse(),
                eye: this.eye.extend(1.0),
                target: s.target.whole_view(),
                tlas: &s.tlas,
                size: EXTENT,
            };
            let bindings = RtAo::push_bindings(args, None).expect("ray queries");
            cmd.compute(&bindings, RtAo::groups([EXTENT.x, EXTENT.y, 1]));
            if check {
                this.checksums.lava_clear(cmd);
                this.checksums.lava_image(cmd, s.target.whole_view(), 0);
                this.checksums.lava_buffer(cmd, s.positions.range(..).cast(), 1);
            }
        });
    }

    fn raw(&mut self, ctx: &mut RawCtx) {
        self.raw.time += DT;
        self.raw.frame += 1;
        let slot = self.raw.frame % IN_FLIGHT as usize;
        let s = &mut self.raw;
        let instances: Vec<_> = (0..INSTANCES)
            .map(|i| RawInstance::new(s.blases[i % MESHES].1, transform(i, s.time)))
            .collect();
        s.instances[slot].host_write(0, &instances);
        let (this, s) = (&*self, &self.raw);
        let check = ctx.check;
        ctx.submit(&[], |cmd| {
            // Last frame's builds read the positions and its trace read the TLAS.
            let trace = (vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR);
            cmd.mem(raw::or(raw::BUILD_READ, trace), raw::or(raw::COMPUTE_WRITE, raw::BUILD_WRITE));
            this.skin_raw(cmd, s);
            cmd.mem(raw::COMPUTE_WRITE, raw::BUILD_READ);
            this.blas_builds(cmd, s, vk::BuildAccelerationStructureModeKHR::UPDATE);
            cmd.mem(raw::BUILD_WRITE, raw::or(raw::BUILD_READ, raw::BUILD_WRITE));
            let geometry = [instances_geometry(s.instances[slot].address())];
            let info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
                .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .dst_acceleration_structure(s.tlas.0)
                .geometries(&geometry)
                .scratch_data(vk::DeviceOrHostAddressKHR {
                    device_address: s.scratch.address() + MESHES as u64 * s.update_scratch,
                });
            let range = [vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(INSTANCES as u32)];
            unsafe { this.functions.cmd_build_acceleration_structures(cmd.cmd, &[info], &[&range]) };
            // Mesa's ANV needs MEMORY_READ for a trace to see a rebuilt TLAS (as lava does there).
            let traced = (
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR | vk::AccessFlags2::MEMORY_READ,
            );
            let target = s.target.whole_view().image();
            cmd.barrier(
                &[raw::mem(raw::BUILD_WRITE, traced)],
                &[img(target, 0..1, L::UNDEFINED, L::GENERAL, (vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::NONE), raw::COMPUTE_WRITE)],
            );
            let mut push: CRtAoBindings = bytemuck::Zeroable::zeroed();
            push.inv_view_proj = this.view_proj.inverse();
            push.eye = this.eye.extend(1.0);
            push.target = s.target.handle::<Storage>();
            push.tlas = s.tlas.1;
            push.size = EXTENT;
            cmd.bind_compute(this.pipelines[1]);
            cmd.push(&push);
            cmd.dispatch(RtAo::groups([EXTENT.x, EXTENT.y, 1]));
            if check {
                cmd.images(&[img(target, 0..1, L::GENERAL, L::SHADER_READ_ONLY_OPTIMAL, raw::COMPUTE_WRITE, raw::COMPUTE_READ)]);
                this.checksums.raw_clear(cmd);
                this.checksums.raw_image(cmd, s.target.handle::<Sampled>(), EXTENT, 0);
                this.checksums.raw_buffer(cmd, s.positions.address(), MESHES as u32 * this.vertices * 3, 1);
                this.checksums.raw_finish(cmd);
            }
        });
    }

    fn checksums(&mut self) -> (Vec<u32>, Vec<u32>) {
        self.checksums.read()
    }
}

impl Drop for RayTracing {
    fn drop(&mut self) {
        lava::wait_idle().unwrap();
        unsafe {
            for &(structure, _) in self.raw.blases.iter().chain([&self.raw.tlas]) {
                self.functions.destroy_acceleration_structure(structure, None);
            }
        }
    }
}

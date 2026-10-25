//! `fluid`: a 64³ smoke grid in 3D images: advection, divergence, 40 Jacobi pressure iterations ping-ponged between two images, and the gradient subtraction

use glam::UVec3;
use lava::bindless::{Sampled, Sampler, SamplerDesc, Storage};
use lava::image::D3;
use lava::image::format::{R16G16B16A16Sfloat, R32Sfloat};
use lava::vk::{self, ImageLayout as L};
use lava::{Image, usage};

use crate::bindings::*;
use crate::check::Checksums;
use crate::pipelines;
use crate::raw::{self, img};
use crate::scenario::{LavaCtx, RawCtx, Scenario};
use crate::scenarios::Setup;

const SIZE: UVec3 = UVec3::splat(64);
const ITERATIONS: usize = 40;
const DT: f32 = 0.1;

type Velocity = Image<R16G16B16A16Sfloat, usage![Sampled, Storage, TransferDst], D3>;
type Scalar = Image<R32Sfloat, usage![Sampled, Storage, TransferDst], D3>;

struct Set {
    velocity: [Velocity; 2],
    pressure: [Scalar; 2],
    divergence: Scalar,
    time: f32,
}

impl Set {
    fn new() -> Self {
        let velocity = || Velocity::new(SIZE).unwrap().bind_combined().unwrap();
        let scalar = || Scalar::new(SIZE).unwrap().bind_combined().unwrap();
        Self {
            velocity: [velocity(), velocity()],
            pressure: [scalar(), scalar()],
            divergence: scalar(),
            time: 0.0,
        }
    }
}

pub struct Fluid {
    lava: Set,
    raw: Set,
    sampler: Sampler,
    checksums: Checksums,
    /// advect, divergence, jacobi, project.
    pipelines: [vk::Pipeline; 4],
}

impl Fluid {
    pub fn new(setup: &mut Setup) -> Self {
        let fluid = Self {
            lava: Set::new(),
            raw: Set::new(),
            sampler: Sampler::new(
                &SamplerDesc::new().address_mode(vk::SamplerAddressMode::CLAMP_TO_EDGE),
            )
            .unwrap(),
            checksums: Checksums::new(1),
            pipelines: ["FluidAdvect", "FluidDivergence", "FluidJacobi", "FluidProject"]
                .map(pipelines::compute),
        };
        // The fluid starts at rest, its pressure zero.
        let s = &fluid.lava;
        setup.lava(|cmd| {
            cmd.clear_image(s.velocity[0].whole_view(), [0.0; 4]);
            cmd.clear_image(s.pressure[0].whole_view(), [0.0]);
        });
        let s = &fluid.raw;
        unsafe {
            for image in &s.velocity {
                lava::raw::write_descriptors(image.whole_view(), L::SHADER_READ_ONLY_OPTIMAL).unwrap();
            }
            for image in s.pressure.iter().chain([&s.divergence]) {
                lava::raw::write_descriptors(image.whole_view(), L::SHADER_READ_ONLY_OPTIMAL).unwrap();
            }
        }
        // Between frames the velocity is in GENERAL (just written), both pressures in
        // SHADER_READ_ONLY_OPTIMAL (last read).
        setup.raw(|cmd| {
            let velocity = s.velocity[0].whole_view().image();
            let pressure = s.pressure.each_ref().map(|p| p.whole_view().image());
            cmd.images(&[
                img(velocity, 0..1, L::UNDEFINED, L::TRANSFER_DST_OPTIMAL, raw::NONE, raw::TRANSFER_WRITE),
                img(pressure[0], 0..1, L::UNDEFINED, L::TRANSFER_DST_OPTIMAL, raw::NONE, raw::TRANSFER_WRITE),
            ]);
            cmd.clear(velocity, [0.0; 4]);
            cmd.clear(pressure[0], [0.0; 4]);
            cmd.images(&[
                img(velocity, 0..1, L::TRANSFER_DST_OPTIMAL, L::GENERAL, raw::TRANSFER_WRITE, raw::COMPUTE_RW),
                img(pressure[0], 0..1, L::TRANSFER_DST_OPTIMAL, L::SHADER_READ_ONLY_OPTIMAL, raw::TRANSFER_WRITE, raw::COMPUTE_READ),
                img(pressure[1], 0..1, L::UNDEFINED, L::SHADER_READ_ONLY_OPTIMAL, raw::NONE, raw::COMPUTE_READ),
            ]);
        });
        fluid
    }
}

impl Scenario for Fluid {
    fn lava(&mut self, ctx: &mut LavaCtx) {
        self.lava.time += DT;
        let s = &self.lava;
        let sampler = self.sampler.handle();
        let check = ctx.check;
        ctx.submit(|cmd| {
            let groups = FluidAdvect::groups(SIZE.into());
            let [velocity, advected] = &s.velocity;
            let args = FluidAdvectArgs {
                velocity: velocity.whole_view(),
                velocity_out: advected.whole_view(),
                sampler,
                size: SIZE,
                dt: DT,
                time: s.time,
            };
            cmd.compute(&FluidAdvect::push_bindings(args, None), groups);
            let args = FluidDivergenceArgs {
                velocity: advected.whole_view(),
                divergence: s.divergence.whole_view(),
                size: SIZE,
            };
            cmd.compute(&FluidDivergence::push_bindings(args, None), groups);
            for i in 0..ITERATIONS {
                let args = FluidJacobiArgs {
                    pressure: s.pressure[i % 2].whole_view(),
                    divergence: s.divergence.whole_view(),
                    pressure_out: s.pressure[1 - i % 2].whole_view(),
                    size: SIZE,
                };
                cmd.compute(&FluidJacobi::push_bindings(args, None), groups);
            }
            let args = FluidProjectArgs {
                velocity: advected.whole_view(),
                pressure: s.pressure[ITERATIONS % 2].whole_view(),
                velocity_out: velocity.whole_view(),
                size: SIZE,
            };
            cmd.compute(&FluidProject::push_bindings(args, None), groups);
            if check {
                self.checksums.lava_clear(cmd);
                self.checksums.lava_volume(cmd, velocity.whole_view(), 0);
            }
        });
    }

    fn raw(&mut self, ctx: &mut RawCtx) {
        self.raw.time += DT;
        let s = &self.raw;
        let sampler = self.sampler.handle();
        let [advect, divergence, jacobi, project] = self.pipelines;
        let check = ctx.check;
        ctx.submit(&[], |cmd| {
            let groups = FluidAdvect::groups(SIZE.into());
            let [velocity, advected] = s.velocity.each_ref().map(|v| v.whole_view().image());
            let pressure = s.pressure.each_ref().map(|p| p.whole_view().image());
            let div = s.divergence.whole_view().image();
            let compute_done = (vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::NONE);
            let read = |image| img(image, 0..1, L::GENERAL, L::SHADER_READ_ONLY_OPTIMAL, raw::COMPUTE_WRITE, raw::COMPUTE_READ);
            let write = |image| img(image, 0..1, L::UNDEFINED, L::GENERAL, compute_done, raw::COMPUTE_WRITE);

            cmd.images(&[read(velocity), write(advected)]);
            let mut push: CFluidAdvectBindings = bytemuck::Zeroable::zeroed();
            push.velocity = s.velocity[0].handle::<Sampled>();
            push.velocity_out = s.velocity[1].handle::<Storage>();
            push.sampler = sampler;
            push.size = SIZE;
            push.dt = DT;
            push.time = s.time;
            cmd.bind_compute(advect);
            cmd.push(&push);
            cmd.dispatch(groups);

            cmd.images(&[read(advected), write(div)]);
            let mut push: CFluidDivergenceBindings = bytemuck::Zeroable::zeroed();
            push.velocity = s.velocity[1].handle::<Sampled>();
            push.divergence = s.divergence.handle::<Storage>();
            push.size = SIZE;
            cmd.bind_compute(divergence);
            cmd.push(&push);
            cmd.dispatch(groups);

            let mut push: CFluidJacobiBindings = bytemuck::Zeroable::zeroed();
            push.divergence = s.divergence.handle::<Sampled>();
            push.size = SIZE;
            cmd.bind_compute(jacobi);
            for i in 0..ITERATIONS {
                let (from, to) = (i % 2, 1 - i % 2);
                // The last iteration's output becomes this one's input; the divergence is read
                // from the first.
                let last = match i {
                    0 => read(div),
                    _ => read(pressure[from]),
                };
                cmd.images(&[last, write(pressure[to])]);
                push.pressure = s.pressure[from].handle::<Sampled>();
                push.pressure_out = s.pressure[to].handle::<Storage>();
                cmd.push(&push);
                cmd.dispatch(groups);
            }

            let solved = ITERATIONS % 2;
            cmd.images(&[read(pressure[solved]), write(velocity)]);
            let mut push: CFluidProjectBindings = bytemuck::Zeroable::zeroed();
            push.velocity = s.velocity[1].handle::<Sampled>();
            push.pressure = s.pressure[solved].handle::<Sampled>();
            push.velocity_out = s.velocity[0].handle::<Storage>();
            push.size = SIZE;
            cmd.bind_compute(project);
            cmd.push(&push);
            cmd.dispatch(groups);

            if check {
                cmd.images(&[read(velocity)]);
                self.checksums.raw_clear(cmd);
                self.checksums.raw_volume(cmd, s.velocity[0].handle::<Sampled>(), SIZE, 0);
                self.checksums.raw_finish(cmd);
                // Back to GENERAL, where the next frame expects it.
                cmd.images(&[img(velocity, 0..1, L::SHADER_READ_ONLY_OPTIMAL, L::GENERAL, raw::COMPUTE_READ, raw::COMPUTE_RW)]);
            }
        });
    }

    fn checksums(&mut self) -> (Vec<u32>, Vec<u32>) {
        self.checksums.read()
    }
}

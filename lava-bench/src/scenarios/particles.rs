//! `particles`: emit and simulate 128k particles, compact the live ones (count, scan, scatter), sort them by depth with a 4-pass radix sort of 3 dispatches each, then draw them with one indirect draw: a long chain of dependent dispatches over buffers

use glam::{Mat4, UVec2, Vec3};
use lava::bindless::Sampled;
use lava::command_buffer::{Attachment, Blend, Draw, DrawIndirectCommand, RasterState};
use lava::image::format::R16G16B16A16Sfloat;
use lava::vk::{self, ImageLayout as L};
use lava::{Buffer, Image, usage};

use crate::bindings::*;
use crate::check::Checksums;
use crate::pipelines::{self, Raster};
use crate::raw::{self, Color, img};
use crate::scenario::{LavaCtx, RawCtx, Scenario, camera};
use crate::scenarios::Setup;

const COUNT: u32 = 1 << 17;
const TILES: u32 = COUNT / 256;
const EXTENT: UVec2 = UVec2::new(1280, 720);
const DT: f32 = 1.0 / 60.0;

struct Set {
    particles: Buffer<Particle>,
    tile_counts: Buffer<u32>,
    keys: [Buffer<u32>; 2],
    values: [Buffer<u32>; 2],
    histogram: Buffer<u32>,
    draw: Buffer<DrawIndirectCommand>,
    target: Image<R16G16B16A16Sfloat, usage![ColorAttachment, Sampled]>,
    frame: u32,
}

impl Set {
    fn new() -> Self {
        let words = || Buffer::new(COUNT as usize).unwrap();
        Self {
            particles: Buffer::new(COUNT as usize).unwrap(),
            tile_counts: Buffer::new(TILES as usize).unwrap(),
            keys: [words(), words()],
            values: [words(), words()],
            histogram: Buffer::new(256 * TILES as usize).unwrap(),
            draw: Buffer::new(1).unwrap(),
            target: Image::new(EXTENT).unwrap().bind_sampled().unwrap(),
            frame: 0,
        }
    }
}

pub struct Particles {
    lava: Set,
    raw: Set,
    view_proj: Mat4,
    checksums: Checksums,
    /// emit, simulate, compact count, scan and scatter, radix count, scan and scatter, draw.
    pipelines: [vk::Pipeline; 9],
}

impl Particles {
    pub fn new(setup: &mut Setup) -> Self {
        let particles = Self {
            lava: Set::new(),
            raw: Set::new(),
            view_proj: camera(Vec3::new(0.0, 4.0, 12.0), EXTENT.x as f32 / EXTENT.y as f32),
            checksums: Checksums::new(3),
            pipelines: [
                pipelines::compute("ParticlesEmit"),
                pipelines::compute("ParticlesSimulate"),
                pipelines::compute("ParticlesCompactCount"),
                pipelines::compute("ParticlesCompactScan"),
                pipelines::compute("ParticlesCompactScatter"),
                pipelines::compute("RadixCount"),
                pipelines::compute("RadixScan"),
                pipelines::compute("RadixScatter"),
                pipelines::raster(
                    "ParticleDraw",
                    Raster {
                        colors: &[vk::Format::R16G16B16A16_SFLOAT],
                        blend: pipelines::Blend::Add,
                        ..Default::default()
                    },
                ),
            ],
        };
        // Every particle starts dead.
        let set = &particles.lava;
        setup.lava(|cmd| cmd.fill_buffer(set.particles.range(..), 0));
        let set = &particles.raw;
        setup.raw(|cmd| {
            cmd.fill(set.particles.address(), set.particles.size(), 0);
            cmd.mem(raw::TRANSFER_WRITE, raw::COMPUTE_RW);
        });
        unsafe {
            lava::raw::write_descriptors(set.target.whole_view(), L::SHADER_READ_ONLY_OPTIMAL)
                .unwrap()
        };
        particles
    }
}

impl Scenario for Particles {
    fn lava(&mut self, ctx: &mut LavaCtx) {
        self.lava.frame += 1;
        let s = &self.lava;
        let check = ctx.check;
        ctx.submit(|cmd| {
            // The five particle passes share their push constants, but each has its own
            // argument type.
            macro_rules! particles_args {
                ($args:ident) => {
                    $args {
                        view_proj: self.view_proj,
                        particles: s.particles.range(..),
                        tile_counts: s.tile_counts.range(..),
                        keys: s.keys[0].range(..),
                        values: s.values[0].range(..),
                        draw: s.draw.range(..).cast(),
                        count: COUNT,
                        frame: s.frame,
                        dt: DT,
                    }
                };
            }
            let threads = [COUNT, 1, 1];
            cmd.compute(
                &ParticlesEmit::push_bindings(particles_args!(ParticlesEmitArgs), None),
                ParticlesEmit::groups(threads),
            );
            cmd.compute(
                &ParticlesSimulate::push_bindings(particles_args!(ParticlesSimulateArgs), None),
                ParticlesSimulate::groups(threads),
            );
            cmd.compute(
                &ParticlesCompactCount::push_bindings(
                    particles_args!(ParticlesCompactCountArgs),
                    None,
                ),
                ParticlesCompactCount::groups(threads),
            );
            cmd.compute(
                &ParticlesCompactScan::push_bindings(particles_args!(ParticlesCompactScanArgs), None),
                [1, 1, 1],
            );
            cmd.compute(
                &ParticlesCompactScatter::push_bindings(
                    particles_args!(ParticlesCompactScatterArgs),
                    None,
                ),
                ParticlesCompactScatter::groups(threads),
            );

            for pass in 0..4 {
                let (from, to) = (pass % 2, 1 - pass % 2);
                macro_rules! radix_args {
                    ($args:ident) => {
                        $args {
                            keys_in: s.keys[from].range(..),
                            values_in: s.values[from].range(..),
                            keys_out: s.keys[to].range(..),
                            values_out: s.values[to].range(..),
                            histogram: s.histogram.range(..),
                            count: COUNT,
                            shift: pass as u32 * 8,
                        }
                    };
                }
                cmd.compute(
                    &RadixCount::push_bindings(radix_args!(RadixCountArgs), None),
                    RadixCount::groups(threads),
                );
                cmd.compute(&RadixScan::push_bindings(radix_args!(RadixScanArgs), None), [1, 1, 1]);
                cmd.compute(
                    &RadixScatter::push_bindings(radix_args!(RadixScatterArgs), None),
                    RadixScatter::groups(threads),
                );
            }

            let args = ParticleDrawArgs {
                view_proj: self.view_proj,
                particles: s.particles.range(..),
                order: s.values[0].range(..),
                size: 0.04,
            };
            let bindings = ParticleDraw::push_bindings(args, None);
            let draw = Draw::indirect(
                &bindings,
                s.draw.range(..),
                RasterState::default().blend(Blend::Add),
            );
            cmd.raster(EXTENT, [Attachment::color(s.target.whole_view(), Some([0.0; 4]))])
                .draw(&draw)
                .record("particles");

            if check {
                self.checksums.lava_clear(cmd);
                self.checksums.lava_buffer(cmd, s.keys[0].range(..), 0);
                self.checksums.lava_buffer(cmd, s.values[0].range(..), 1);
                self.checksums.lava_image(cmd, s.target.whole_view(), 2);
            }
        });
    }

    fn raw(&mut self, ctx: &mut RawCtx) {
        self.raw.frame += 1;
        let s = &self.raw;
        let [emit, simulate, compact_count, compact_scan, compact_scatter, radix_count, radix_scan, radix_scatter, draw] =
            self.pipelines;
        let check = ctx.check;
        ctx.submit(&[], |cmd| {
            // The particle passes' push constants are alike, so one struct serves all five.
            let mut push: CParticlesEmitBindings = bytemuck::Zeroable::zeroed();
            push.view_proj = self.view_proj;
            push.particles = s.particles.address();
            push.tile_counts = s.tile_counts.address();
            push.keys = s.keys[0].address();
            push.values = s.values[0].address();
            push.draw = s.draw.address();
            push.count = COUNT;
            push.frame = s.frame;
            push.dt = DT;
            let groups = ParticlesEmit::groups([COUNT, 1, 1]);
            let target = s.target.whole_view().image();
            // Last frame's draw read the particles, the order and the command.
            cmd.mem(raw::or(raw::INDIRECT_READ, raw::VERTEX_READ), raw::COMPUTE_RW);
            for (pipeline, groups) in [
                (emit, groups),
                (simulate, groups),
                (compact_count, groups),
                (compact_scan, [1, 1, 1]),
                (compact_scatter, groups),
            ] {
                cmd.bind_compute(pipeline);
                cmd.push(&push);
                cmd.dispatch(groups);
                cmd.mem(raw::COMPUTE_RW, raw::COMPUTE_RW);
            }
            let mut push: CRadixCountBindings = bytemuck::Zeroable::zeroed();
            push.histogram = s.histogram.address();
            push.count = COUNT;
            for pass in 0..4 {
                let (from, to) = (pass % 2, 1 - pass % 2);
                push.keys_in = s.keys[from].address();
                push.values_in = s.values[from].address();
                push.keys_out = s.keys[to].address();
                push.values_out = s.values[to].address();
                push.shift = pass as u32 * 8;
                for (pipeline, groups) in [(radix_count, groups), (radix_scan, [1, 1, 1]), (radix_scatter, groups)] {
                    cmd.bind_compute(pipeline);
                    cmd.push(&push);
                    cmd.dispatch(groups);
                    // The draw reads the result after the last scatter.
                    let last = pass == 3 && pipeline == radix_scatter;
                    if !last {
                        cmd.mem(raw::COMPUTE_RW, raw::COMPUTE_RW);
                    }
                }
            }

            cmd.barrier(
                &[raw::mem(raw::COMPUTE_WRITE, raw::or(raw::INDIRECT_READ, raw::VERTEX_READ))],
                &[img(target, 0..1, L::UNDEFINED, L::COLOR_ATTACHMENT_OPTIMAL, (vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::NONE), raw::COLOR_WRITE)],
            );
            cmd.begin_rendering(EXTENT.into(), &[Color { view: s.target.whole_view().vk_view(), clear: Some([0.0; 4]) }], None);
            cmd.bind_graphics(draw);
            let mut push: CParticleDrawBindings = bytemuck::Zeroable::zeroed();
            push.view_proj = self.view_proj;
            push.particles = s.particles.address();
            push.order = s.values[0].address();
            push.size = 0.04;
            cmd.push(&push);
            cmd.draw_indirect(s.draw.address());
            cmd.end_rendering();

            if check {
                cmd.barrier(
                    &[raw::mem(raw::COMPUTE_WRITE, raw::COMPUTE_READ)],
                    &[img(target, 0..1, L::COLOR_ATTACHMENT_OPTIMAL, L::SHADER_READ_ONLY_OPTIMAL, raw::COLOR_WRITE, raw::COMPUTE_READ)],
                );
                self.checksums.raw_clear(cmd);
                self.checksums.raw_buffer(cmd, s.keys[0].address(), COUNT, 0);
                self.checksums.raw_buffer(cmd, s.values[0].address(), COUNT, 1);
                self.checksums.raw_image(cmd, s.target.handle::<Sampled>(), EXTENT, 2);
                self.checksums.raw_finish(cmd);
            }
        });
    }

    fn checksums(&mut self) -> (Vec<u32>, Vec<u32>) {
        self.checksums.read()
    }
}

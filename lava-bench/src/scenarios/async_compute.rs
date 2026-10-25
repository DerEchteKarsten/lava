//! `async_compute` (needs `async-compute`): a `Timeline<Compute>` steps an 8k-body simulation while graphics draws its own background; graphics waits for the step at the raster stage and draws the bodies; the baseline does the same with two queues and their timeline semaphores

use glam::{Mat4, UVec2, Vec3, Vec4};
use lava::bindless::{Sampled, Storage};
use lava::command_buffer::{Attachment, Blend, Draw, RasterState};
use lava::image::format::R8G8B8A8Unorm;
use lava::queue::{Compute, GraphicsStage, Kind};
use lava::vk::{self, ImageLayout as L};
use lava::{Buffer, Image, Timeline, Wait, usage};

use crate::IN_FLIGHT;
use crate::bindings::*;
use crate::check::Checksums;
use crate::pipelines::{self, Raster};
use crate::raw::{self, Color, RawQueue, img};
use crate::scenario::{LavaCtx, RawCtx, Scenario, Stamped, camera, rand01};
use crate::scenarios::Setup;

const BODIES: u32 = 8192;
const EXTENT: UVec2 = UVec2::new(1280, 720);
const DT: f32 = 0.002;

type Target = Image<R8G8B8A8Unorm, usage![ColorAttachment, Storage, Sampled]>;

struct Set {
    positions: [Buffer<Vec4>; 2],
    velocities: [Buffer<Vec4>; 2],
    target: Target,
    /// Never read: the background pass's unused source.
    unused: Image<R8G8B8A8Unorm, usage![Sampled]>,
    frame: usize,
}

pub struct AsyncCompute {
    lava: Set,
    raw: Set,
    compute: Stamped<Compute>,
    last_graphics: Option<Wait>,
    last_compute: Option<Wait>,
    raw_compute: RawQueue,
    view_proj: Mat4,
    checksums: Checksums,
    /// nbody, background, draw.
    pipelines: [vk::Pipeline; 3],
}

impl AsyncCompute {
    pub fn new(setup: &mut Setup) -> Option<Self> {
        let timeline = Timeline::<Compute>::new(IN_FLIGHT, 0).ok()?;
        let uploader = setup.uploader();
        // A disk of bodies on circular orbits around the centre.
        let (positions, velocities): (Vec<_>, Vec<_>) = (0..BODIES)
            .map(|i| {
                let angle = rand01(i * 2) * std::f32::consts::TAU;
                let radius = 0.5 + rand01(i * 2 + 1) * 4.0;
                let position = Vec3::new(angle.cos() * radius, (rand01(i * 3) - 0.5) * 0.2, angle.sin() * radius);
                let speed = (BODIES as f32 * 0.0005 / radius).sqrt();
                let velocity = Vec3::new(-angle.sin(), 0.0, angle.cos()) * speed;
                (position.extend(0.001), velocity.extend(0.0))
            })
            .unzip();
        let set = || Set {
            positions: [uploader.from_slice(&positions).unwrap(), Buffer::new(BODIES as usize).unwrap()],
            velocities: [uploader.from_slice(&velocities).unwrap(), Buffer::new(BODIES as usize).unwrap()],
            target: Target::new(EXTENT).unwrap().bind_combined().unwrap(),
            unused: Image::new(UVec2::ONE).unwrap().bind_sampled().unwrap(),
            frame: 0,
        };
        let scenario = Self {
            lava: set(),
            raw: set(),
            compute: Stamped::new(timeline),
            last_graphics: None,
            last_compute: None,
            raw_compute: RawQueue::new(Kind::Compute, IN_FLIGHT as usize),
            view_proj: camera(Vec3::new(0.0, 6.0, 8.0), EXTENT.x as f32 / EXTENT.y as f32),
            checksums: Checksums::new(2),
            pipelines: [
                pipelines::compute("Nbody"),
                pipelines::compute("FilterGenerate"),
                pipelines::raster(
                    "NbodyDraw",
                    Raster {
                        colors: &[vk::Format::R8G8B8A8_UNORM],
                        blend: pipelines::Blend::Add,
                        ..Default::default()
                    },
                ),
            ],
        };
        setup.lava(|_| {});
        unsafe {
            lava::raw::write_descriptors(scenario.raw.target.whole_view(), L::SHADER_READ_ONLY_OPTIMAL)
                .unwrap()
        };
        Some(scenario)
    }
}

impl Scenario for AsyncCompute {
    fn lava(&mut self, ctx: &mut LavaCtx) {
        self.lava.frame += 1;
        let s = &self.lava;
        let check = ctx.check;
        // The initial state is in buffer 0.
        let (from, to) = ((s.frame + 1) % 2, s.frame % 2);
        // The step overwrites what the graphics frame before last drew.
        let last_graphics = self.last_graphics;
        let (compute, ns) = self.compute.submit(
            |submission| {
                if let Some(wait) = last_graphics {
                    submission.wait_for(wait);
                }
            },
            |cmd| {
                let args = NbodyArgs {
                    positions: s.positions[from].range(..),
                    velocities: s.velocities[from].range(..),
                    positions_out: s.positions[to].range(..),
                    velocities_out: s.velocities[to].range(..),
                    count: BODIES,
                    dt: DT,
                };
                cmd.compute(&Nbody::push_bindings(args, None), Nbody::groups([BODIES, 1, 1]));
            },
        );
        ctx.record_ns += ns;
        self.last_compute = Some(compute);
        // A check reads the positions in a compute pass too.
        let stage = if check { GraphicsStage::All } else { GraphicsStage::Raster };
        ctx.submit_with(
            |submission| {
                submission.wait_for(compute.at(stage));
            },
            |cmd| {
                // Graphics work of its own while the step runs.
                let args = FilterGenerateArgs {
                    src: s.unused.whole_view(),
                    dst: s.target.whole_view(),
                    sampler: lava::SamplerHandle::DEFAULT,
                    size: EXTENT,
                    params: Vec4::ZERO,
                };
                cmd.compute(
                    &FilterGenerate::push_bindings(args, None),
                    FilterGenerate::groups([EXTENT.x, EXTENT.y, 1]),
                );
                let args = NbodyDrawArgs {
                    view_proj: self.view_proj,
                    positions: s.positions[to].range(..),
                    size: 0.01,
                };
                let bindings = NbodyDraw::push_bindings(args, None);
                let draw = Draw::new(&bindings, 6, BODIES, RasterState::default().blend(Blend::Add));
                cmd.raster(EXTENT, [Attachment::color(s.target.whole_view(), None)])
                    .draw(&draw)
                    .record("bodies");
                if check {
                    self.checksums.lava_clear(cmd);
                    self.checksums.lava_image(cmd, s.target.whole_view(), 0);
                    self.checksums.lava_buffer(cmd, s.positions[to].range(..).cast(), 1);
                }
            },
        );
        self.last_graphics = ctx.waits.last().copied();
    }

    fn raw(&mut self, ctx: &mut RawCtx) {
        self.raw.frame += 1;
        let s = &self.raw;
        // The initial state is in buffer 0.
        let (from, to) = ((s.frame + 1) % 2, s.frame % 2);
        let [nbody, background, draw] = self.pipelines;
        // The step overwrites what the graphics frame before last drew.
        let graphics = [(ctx.queue.semaphore, ctx.queue.value, vk::PipelineStageFlags2::COMPUTE_SHADER)];
        let (_, ns) = self.raw_compute.submit(&graphics, |cmd| {
            // The last step on this queue wrote what this one reads.
            cmd.mem(raw::COMPUTE_WRITE, raw::COMPUTE_READ);
            let mut push: CNbodyBindings = bytemuck::Zeroable::zeroed();
            push.positions = s.positions[from].address();
            push.velocities = s.velocities[from].address();
            push.positions_out = s.positions[to].address();
            push.velocities_out = s.velocities[to].address();
            push.count = BODIES;
            push.dt = DT;
            cmd.bind_compute(nbody);
            cmd.push(&push);
            cmd.dispatch(Nbody::groups([BODIES, 1, 1]));
        });
        ctx.record_ns += ns;
        let check = ctx.check;
        // A check reads the positions in a compute pass too.
        let stage = match check {
            true => vk::PipelineStageFlags2::VERTEX_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER,
            false => vk::PipelineStageFlags2::VERTEX_SHADER,
        };
        let compute = [(self.raw_compute.semaphore, self.raw_compute.value, stage)];
        ctx.submit(&compute, |cmd| {
            let target = s.target.whole_view().image();
            let mut push: CFilterGenerateBindings = bytemuck::Zeroable::zeroed();
            push.src = s.unused.handle::<Sampled>();
            push.dst = s.target.handle::<Storage>();
            push.size = EXTENT;
            // Last frame's draw (and a check) used the target.
            let before = raw::or(raw::COLOR_WRITE, raw::COMPUTE_READ);
            cmd.images(&[img(target, 0..1, L::UNDEFINED, L::GENERAL, (before.0, vk::AccessFlags2::NONE), raw::COMPUTE_WRITE)]);
            cmd.bind_compute(background);
            cmd.push(&push);
            cmd.dispatch(FilterGenerate::groups([EXTENT.x, EXTENT.y, 1]));
            cmd.images(&[img(target, 0..1, L::GENERAL, L::COLOR_ATTACHMENT_OPTIMAL, raw::COMPUTE_WRITE, raw::COLOR_WRITE)]);
            cmd.begin_rendering(EXTENT.into(), &[Color { view: s.target.whole_view().vk_view(), clear: None }], None);
            cmd.bind_graphics(draw);
            let mut push: CNbodyDrawBindings = bytemuck::Zeroable::zeroed();
            push.view_proj = self.view_proj;
            push.positions = s.positions[to].address();
            push.size = 0.01;
            cmd.push(&push);
            cmd.draw(6, BODIES);
            cmd.end_rendering();
            if check {
                cmd.images(&[img(target, 0..1, L::COLOR_ATTACHMENT_OPTIMAL, L::SHADER_READ_ONLY_OPTIMAL, raw::COLOR_WRITE, raw::COMPUTE_READ)]);
                self.checksums.raw_clear(cmd);
                self.checksums.raw_image(cmd, s.target.handle::<Sampled>(), EXTENT, 0);
                self.checksums.raw_buffer(cmd, s.positions[to].address(), BODIES * 4, 1);
                self.checksums.raw_finish(cmd);
            }
        });
    }

    fn extra_spans(&mut self, lava: bool) -> Vec<[u64; 2]> {
        match lava {
            true => vec![self.compute.span()],
            false => {
                let slot = (self.raw_compute.value as usize - 1) % self.raw_compute.slots();
                vec![self.raw_compute.stamps(slot)]
            }
        }
    }

    fn block_extra(&mut self, lava: bool) {
        match lava {
            true => self.last_compute.unwrap().block().unwrap(),
            false => self.raw_compute.block(),
        }
    }

    fn checksums(&mut self) -> (Vec<u32>, Vec<u32>) {
        self.checksums.read()
    }
}

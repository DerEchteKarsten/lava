//! `deferred`: a G-buffer of three targets and depth (1000 draws), compute lighting, a bloom of 7 downsample and 6 upsample dispatches over a `MipChain` (per-mip barriers) and a TAA resolve with a ping-pong history

use glam::{Mat4, UVec2, Vec3, Vec4};
use lava::bindless::{Sampled, Sampler, SamplerDesc, Storage};
use lava::command_buffer::{Attachment, Draw, RasterState};
use lava::image::format::{D32Sfloat, R8G8B8A8Unorm, R16G16B16A16Sfloat};
use lava::vk::{self, ImageLayout as L};
use lava::{Buffer, Image, MipChain, usage};

use crate::bindings::*;
use crate::check::Checksums;
use crate::pipelines::{self, Raster};
use crate::raw::{self, Color, img, img_aspect};
use crate::scenario::{LavaCtx, RawCtx, Scenario, camera, rand01, scatter, sphere};
use crate::scenarios::Setup;

const EXTENT: UVec2 = UVec2::new(1280, 720);
const DRAWS: u32 = 1000;
const LIGHTS: u32 = 64;
const BLOOM_MIPS: u32 = 7;

type Target<F> = Image<F, usage![ColorAttachment, Sampled]>;
type Hdr = Image<R16G16B16A16Sfloat, usage![Sampled, Storage]>;
type History = Image<R16G16B16A16Sfloat, usage![Sampled, Storage, TransferDst]>;

struct Set {
    albedo: Target<R8G8B8A8Unorm>,
    normal: Target<R16G16B16A16Sfloat>,
    material: Target<R8G8B8A8Unorm>,
    depth: Image<D32Sfloat, usage![DepthAttachment, Sampled]>,
    hdr: Hdr,
    bloom: MipChain<R16G16B16A16Sfloat, usage![Sampled, Storage]>,
    history: [History; 2],
    output: Image<R8G8B8A8Unorm, usage![Sampled, Storage]>,
    frame: usize,
}

impl Set {
    fn new() -> Self {
        let history = || History::new(EXTENT).unwrap().bind_combined().unwrap();
        Self {
            albedo: Image::new(EXTENT).unwrap().bind_sampled().unwrap(),
            normal: Image::new(EXTENT).unwrap().bind_sampled().unwrap(),
            material: Image::new(EXTENT).unwrap().bind_sampled().unwrap(),
            depth: Image::new(EXTENT).unwrap().bind_sampled().unwrap(),
            hdr: Hdr::new(EXTENT).unwrap().bind_combined().unwrap(),
            bloom: MipChain::with_mip_levels(EXTENT / 2, BLOOM_MIPS)
                .unwrap()
                .bind_mips_combined()
                .unwrap(),
            history: [history(), history()],
            output: Image::new(EXTENT).unwrap().bind_combined().unwrap(),
            frame: 0,
        }
    }
}

pub struct Deferred {
    lava: Set,
    raw: Set,
    vertices: Buffer<MeshVertex>,
    indices: Buffer<u32>,
    instances: Buffer<DrawInstance>,
    lights: Buffer<PointLight>,
    sampler: Sampler,
    view_proj: Mat4,
    checksums: Checksums,
    pipelines: [vk::Pipeline; 5],
}

impl Deferred {
    pub fn new(setup: &mut Setup) -> Self {
        let (vertices, indices) = sphere(12, 16);
        let uploader = setup.uploader();
        let lights: Vec<_> = (0..LIGHTS)
            .map(|i| PointLight {
                position_radius: Vec4::new(
                    (rand01(i * 5) - 0.5) * 30.0,
                    1.0 + rand01(i * 5 + 1) * 3.0,
                    (rand01(i * 5 + 2) - 0.5) * 30.0,
                    6.0,
                ),
                color: Vec4::new(rand01(i * 5 + 3), rand01(i * 5 + 4), 1.0, 1.0) * 2.0,
            })
            .collect();
        let deferred = Self {
            lava: Set::new(),
            raw: Set::new(),
            vertices: uploader.from_slice(&vertices).unwrap(),
            indices: uploader.from_slice(&indices).unwrap(),
            instances: uploader.from_slice(&scatter(DRAWS, 30.0)).unwrap(),
            lights: uploader.from_slice(&lights).unwrap(),
            sampler: Sampler::new(
                &SamplerDesc::new().address_mode(vk::SamplerAddressMode::CLAMP_TO_EDGE),
            )
            .unwrap(),
            view_proj: camera(Vec3::new(0.0, 10.0, 22.0), EXTENT.x as f32 / EXTENT.y as f32),
            checksums: Checksums::new(2),
            pipelines: [
                pipelines::raster(
                    "Gbuffer",
                    Raster {
                        colors: &[
                            vk::Format::R8G8B8A8_UNORM,
                            vk::Format::R16G16B16A16_SFLOAT,
                            vk::Format::R8G8B8A8_UNORM,
                        ],
                        depth: vk::Format::D32_SFLOAT,
                        ..Default::default()
                    },
                ),
                pipelines::compute("Lighting"),
                pipelines::compute("BloomDown"),
                pipelines::compute("BloomUp"),
                pipelines::compute("Taa"),
            ],
        };
        // Both histories start black.
        let set = &deferred.lava;
        setup.lava(|cmd| {
            for history in &set.history {
                cmd.clear_image(history.whole_view(), [0.0; 4]);
            }
        });
        let set = &deferred.raw;
        unsafe {
            let read_only = L::SHADER_READ_ONLY_OPTIMAL;
            lava::raw::write_descriptors(set.albedo.whole_view(), read_only).unwrap();
            lava::raw::write_descriptors(set.normal.whole_view(), read_only).unwrap();
            lava::raw::write_descriptors(set.material.whole_view(), read_only).unwrap();
            lava::raw::write_descriptors(set.depth.whole_view(), read_only).unwrap();
            lava::raw::write_descriptors(set.hdr.whole_view(), read_only).unwrap();
            for mip in 0..BLOOM_MIPS {
                lava::raw::write_descriptors(set.bloom.mip_view(mip), read_only).unwrap();
            }
            for history in &set.history {
                lava::raw::write_descriptors(history.whole_view(), read_only).unwrap();
            }
            lava::raw::write_descriptors(set.output.whole_view(), read_only).unwrap();
        }
        // The baseline's histories are in GENERAL between frames.
        setup.raw(|cmd| {
            let images = set.history.each_ref().map(|h| h.whole_view().image());
            cmd.images(&images.map(|image| {
                img(image, 0..1, L::UNDEFINED, L::TRANSFER_DST_OPTIMAL, raw::NONE, raw::TRANSFER_WRITE)
            }));
            for image in images {
                cmd.clear(image, [0.0; 4]);
            }
            cmd.images(&images.map(|image| {
                img(image, 0..1, L::TRANSFER_DST_OPTIMAL, L::GENERAL, raw::TRANSFER_WRITE, raw::COMPUTE_RW)
            }));
        });
        deferred
    }
}

impl Scenario for Deferred {
    fn lava(&mut self, ctx: &mut LavaCtx) {
        let s = &mut self.lava;
        s.frame += 1;
        let s = &self.lava;
        let check = ctx.check;
        ctx.submit(|cmd| {
            let gbuffer: Vec<_> = (0..DRAWS)
                .map(|instance| {
                    let args = GbufferArgs {
                        view_proj: self.view_proj,
                        vertices: self.vertices.range(..),
                        instances: self.instances.range(..),
                        instance,
                    };
                    Gbuffer::push_bindings(args, None)
                })
                .collect();
            let indices = self.indices.range(..);
            let draws: Vec<_> = (gbuffer.iter())
                .map(|b| Draw::indexed(b, indices, 1, RasterState::default()))
                .collect();
            let mut rendering = cmd.raster(
                EXTENT,
                [
                    Attachment::color(s.albedo.whole_view(), Some([0.0; 4])),
                    Attachment::color(s.normal.whole_view(), Some([0.0; 4])),
                    Attachment::color(s.material.whole_view(), Some([0.0; 4])),
                ],
            );
            rendering.depth_attachment(s.depth.whole_view(), Some([0.0]));
            for draw in &draws {
                rendering.draw(draw);
            }
            rendering.record("gbuffer");

            let args = LightingArgs {
                inv_view_proj: self.view_proj.inverse(),
                albedo: s.albedo.whole_view(),
                normal: s.normal.whole_view(),
                material: s.material.whole_view(),
                depth: s.depth.whole_view(),
                hdr: s.hdr.whole_view(),
                size: EXTENT,
                lights: self.lights.range(..),
                light_count: LIGHTS,
            };
            cmd.compute(
                &Lighting::push_bindings(args, None),
                Lighting::groups([EXTENT.x, EXTENT.y, 1]),
            );

            let sampler = self.sampler.handle();
            for mip in 0..BLOOM_MIPS {
                let src = match mip {
                    0 => s.hdr.whole_view(),
                    _ => s.bloom.mip_view(mip - 1),
                };
                let size = s.bloom.mip_extent(mip);
                let args = BloomDownArgs {
                    src,
                    dst: s.bloom.mip_view(mip),
                    sampler,
                    size,
                };
                cmd.compute(
                    &BloomDown::push_bindings(args, None),
                    BloomDown::groups([size.x, size.y, 1]),
                );
            }
            for mip in (0..BLOOM_MIPS - 1).rev() {
                let size = s.bloom.mip_extent(mip);
                let args = BloomUpArgs {
                    src: s.bloom.mip_view(mip + 1),
                    dst: s.bloom.mip_view(mip),
                    sampler,
                    size,
                };
                cmd.compute(
                    &BloomUp::push_bindings(args, None),
                    BloomUp::groups([size.x, size.y, 1]),
                );
            }

            let (history, history_out) = (&s.history[s.frame % 2], &s.history[(s.frame + 1) % 2]);
            let args = TaaArgs {
                current: s.hdr.whole_view(),
                bloom: s.bloom.mip_view(0),
                history: history.whole_view(),
                history_out: history_out.whole_view(),
                output: s.output.whole_view(),
                sampler,
                size: EXTENT,
            };
            cmd.compute(&Taa::push_bindings(args, None), Taa::groups([EXTENT.x, EXTENT.y, 1]));

            if check {
                self.checksums.lava_clear(cmd);
                self.checksums.lava_image(cmd, s.output.whole_view(), 0);
                self.checksums.lava_image(cmd, s.hdr.whole_view(), 1);
            }
        });
    }

    fn raw(&mut self, ctx: &mut RawCtx) {
        let s = &mut self.raw;
        s.frame += 1;
        let s = &self.raw;
        let [gbuffer, lighting, bloom_down, bloom_up, taa] = self.pipelines;
        let check = ctx.check;
        ctx.submit(&[], |cmd| {
            let albedo = s.albedo.whole_view().image();
            let material = s.material.whole_view().image();
            let normal = s.normal.whole_view().image();
            let depth = s.depth.whole_view().image();
            let hdr = s.hdr.whole_view().image();
            let bloom = s.bloom.whole_view().image();
            let output = s.output.whole_view().image();
            let (history, history_out) = (&s.history[s.frame % 2], &s.history[(s.frame + 1) % 2]);
            let compute_done = (vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::NONE);
            let aspect = vk::ImageAspectFlags::DEPTH;
            // Last frame's compute passes read the targets; the history written last frame
            // becomes this frame's input.
            cmd.images(&[
                img(albedo, 0..1, L::UNDEFINED, L::COLOR_ATTACHMENT_OPTIMAL, compute_done, raw::COLOR_WRITE),
                img(normal, 0..1, L::UNDEFINED, L::COLOR_ATTACHMENT_OPTIMAL, compute_done, raw::COLOR_WRITE),
                img(material, 0..1, L::UNDEFINED, L::COLOR_ATTACHMENT_OPTIMAL, compute_done, raw::COLOR_WRITE),
                img_aspect(depth, aspect, 0..1, L::UNDEFINED, L::DEPTH_ATTACHMENT_OPTIMAL, raw::or(compute_done, raw::DEPTH_WRITE), raw::DEPTH_WRITE),
                img(history.whole_view().image(), 0..1, L::GENERAL, L::SHADER_READ_ONLY_OPTIMAL, raw::COMPUTE_WRITE, raw::COMPUTE_READ),
                img(history_out.whole_view().image(), 0..1, L::UNDEFINED, L::GENERAL, compute_done, raw::COMPUTE_WRITE),
                img(output, 0..1, L::UNDEFINED, L::GENERAL, compute_done, raw::COMPUTE_WRITE),
            ]);
            cmd.begin_rendering(
                EXTENT.into(),
                &[
                    Color { view: s.albedo.whole_view().vk_view(), clear: Some([0.0; 4]) },
                    Color { view: s.normal.whole_view().vk_view(), clear: Some([0.0; 4]) },
                    Color { view: s.material.whole_view().vk_view(), clear: Some([0.0; 4]) },
                ],
                Some((s.depth.whole_view().vk_view(), Some(0.0))),
            );
            cmd.bind_graphics(gbuffer);
            cmd.bind_index_u32(self.indices.address(), self.indices.size());
            let index_count = self.indices.len() as u32;
            let mut push: CGbufferBindings = bytemuck::Zeroable::zeroed();
            push.view_proj = self.view_proj;
            push.vertices = self.vertices.address();
            push.instances = self.instances.address();
            for instance in 0..DRAWS {
                push.instance = instance;
                cmd.push(&push);
                cmd.draw_indexed(index_count, 1);
            }
            cmd.end_rendering();

            cmd.images(&[
                img(albedo, 0..1, L::COLOR_ATTACHMENT_OPTIMAL, L::SHADER_READ_ONLY_OPTIMAL, raw::COLOR_WRITE, raw::COMPUTE_READ),
                img(normal, 0..1, L::COLOR_ATTACHMENT_OPTIMAL, L::SHADER_READ_ONLY_OPTIMAL, raw::COLOR_WRITE, raw::COMPUTE_READ),
                img(material, 0..1, L::COLOR_ATTACHMENT_OPTIMAL, L::SHADER_READ_ONLY_OPTIMAL, raw::COLOR_WRITE, raw::COMPUTE_READ),
                img_aspect(depth, aspect, 0..1, L::DEPTH_ATTACHMENT_OPTIMAL, L::SHADER_READ_ONLY_OPTIMAL, raw::DEPTH_WRITE, raw::COMPUTE_READ),
                img(hdr, 0..1, L::UNDEFINED, L::GENERAL, compute_done, raw::COMPUTE_WRITE),
            ]);
            let mut push: CLightingBindings = bytemuck::Zeroable::zeroed();
            push.inv_view_proj = self.view_proj.inverse();
            push.albedo = s.albedo.handle::<Sampled>();
            push.normal = s.normal.handle::<Sampled>();
            push.material = s.material.handle::<Sampled>();
            push.depth = s.depth.handle::<Sampled>();
            push.hdr = s.hdr.handle::<Storage>();
            push.size = EXTENT;
            push.lights = self.lights.address();
            push.light_count = LIGHTS;
            cmd.bind_compute(lighting);
            cmd.push(&push);
            cmd.dispatch(Lighting::groups([EXTENT.x, EXTENT.y, 1]));

            // Every bloom mip is written in GENERAL, then read in SHADER_READ_ONLY_OPTIMAL.
            cmd.images(&[
                img(hdr, 0..1, L::GENERAL, L::SHADER_READ_ONLY_OPTIMAL, raw::COMPUTE_WRITE, raw::COMPUTE_READ),
                img(bloom, 0..BLOOM_MIPS, L::UNDEFINED, L::GENERAL, compute_done, raw::COMPUTE_WRITE),
            ]);
            let sampler = self.sampler.handle();
            cmd.bind_compute(bloom_down);
            for mip in 0..BLOOM_MIPS {
                let mut push: CBloomDownBindings = bytemuck::Zeroable::zeroed();
                push.src = match mip {
                    0 => s.hdr.handle::<Sampled>(),
                    _ => s.bloom.mip_view(mip - 1).handle::<Sampled>(),
                };
                push.dst = s.bloom.mip_view(mip).handle::<Storage>();
                push.sampler = sampler;
                push.size = s.bloom.mip_extent(mip);
                cmd.push(&push);
                cmd.dispatch(BloomDown::groups([push.size.x, push.size.y, 1]));
                let mut barriers = vec![img(bloom, mip..mip + 1, L::GENERAL, L::SHADER_READ_ONLY_OPTIMAL, raw::COMPUTE_WRITE, raw::COMPUTE_READ)];
                if mip == BLOOM_MIPS - 1 {
                    // The first upsample adds onto the mip above.
                    barriers.push(img(bloom, mip - 1..mip, L::SHADER_READ_ONLY_OPTIMAL, L::GENERAL, raw::COMPUTE_READ, raw::COMPUTE_RW));
                }
                cmd.images(&barriers);
            }
            cmd.bind_compute(bloom_up);
            for mip in (0..BLOOM_MIPS - 1).rev() {
                let mut push: CBloomUpBindings = bytemuck::Zeroable::zeroed();
                push.src = s.bloom.mip_view(mip + 1).handle::<Sampled>();
                push.dst = s.bloom.mip_view(mip).handle::<Storage>();
                push.sampler = sampler;
                push.size = s.bloom.mip_extent(mip);
                cmd.push(&push);
                cmd.dispatch(BloomUp::groups([push.size.x, push.size.y, 1]));
                let mut barriers = vec![img(bloom, mip..mip + 1, L::GENERAL, L::SHADER_READ_ONLY_OPTIMAL, raw::COMPUTE_RW, raw::COMPUTE_READ)];
                if mip > 0 {
                    barriers.push(img(bloom, mip - 1..mip, L::SHADER_READ_ONLY_OPTIMAL, L::GENERAL, raw::COMPUTE_READ, raw::COMPUTE_RW));
                }
                cmd.images(&barriers);
            }

            let mut push: CTaaBindings = bytemuck::Zeroable::zeroed();
            push.current = s.hdr.handle::<Sampled>();
            push.bloom = s.bloom.mip_view(0).handle::<Sampled>();
            push.history = history.handle::<Sampled>();
            push.history_out = history_out.handle::<Storage>();
            push.output = s.output.handle::<Storage>();
            push.sampler = sampler;
            push.size = EXTENT;
            cmd.bind_compute(taa);
            cmd.push(&push);
            cmd.dispatch(Taa::groups([EXTENT.x, EXTENT.y, 1]));

            if check {
                cmd.images(&[img(output, 0..1, L::GENERAL, L::SHADER_READ_ONLY_OPTIMAL, raw::COMPUTE_WRITE, raw::COMPUTE_READ)]);
                self.checksums.raw_clear(cmd);
                self.checksums.raw_image(cmd, s.output.handle::<Sampled>(), EXTENT, 0);
                self.checksums.raw_image(cmd, s.hdr.handle::<Sampled>(), EXTENT, 1);
                self.checksums.raw_finish(cmd);
            }
        });
    }

    fn checksums(&mut self) -> (Vec<u32>, Vec<u32>) {
        self.checksums.read()
    }
}

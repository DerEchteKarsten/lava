//! `gpu_driven`: a compute pass frustum-culls 64k instances into indexed indirect draws and a count, one `draw_indexed_indirect_count` draws them, and a 10-mip Hi-Z reduction of the depth follows

use glam::{Mat4, UVec2, Vec3, Vec4};
use lava::bindless::{Sampled, Storage};
use lava::command_buffer::{Attachment, Draw, DrawIndexedIndirectCommand, RasterState};
use lava::image::format::{D32Sfloat, R8G8B8A8Unorm, R32Sfloat};
use lava::image::{D2, SampledView};
use lava::queue::Graphics;
use lava::vk::{self, ImageLayout as L};
use lava::{Buffer, CommandBuffer, Image, MipChain, usage};

use crate::bindings::*;
use crate::check::Checksums;
use crate::pipelines::{self, Raster};
use crate::raw::{self, Color, img, img_aspect};
use crate::scenario::{LavaCtx, RawCtx, Scenario, camera, rand01, sphere};
use crate::scenarios::Setup;

const EXTENT: UVec2 = UVec2::new(1280, 720);
const INSTANCES: u32 = 65536;
const HIZ_MIPS: u32 = 10;

type HizChain = MipChain<R32Sfloat, usage![Sampled, Storage]>;

struct Set {
    color: Image<R8G8B8A8Unorm, usage![ColorAttachment, Sampled]>,
    depth: Image<D32Sfloat, usage![DepthAttachment, Sampled]>,
    hiz: HizChain,
    draws: Buffer<DrawIndexedIndirectCommand>,
    count: Buffer<u32>,
}

impl Set {
    fn new() -> Self {
        Self {
            color: Image::new(EXTENT).unwrap().bind_sampled().unwrap(),
            depth: Image::new(EXTENT).unwrap().bind_sampled().unwrap(),
            hiz: HizChain::with_mip_levels(EXTENT / 2, HIZ_MIPS)
                .unwrap()
                .bind_mips_combined()
                .unwrap(),
            draws: Buffer::new(INSTANCES as usize).unwrap(),
            count: Buffer::new(1).unwrap(),
        }
    }
}

pub struct GpuDriven {
    lava: Set,
    raw: Set,
    vertices: Buffer<MeshVertex>,
    indices: Buffer<u32>,
    instances: Buffer<CullInstance>,
    view_proj: Mat4,
    planes: [Vec4; 6],
    checksums: Checksums,
    pipelines: [vk::Pipeline; 3],
}

/// The frustum planes of `m` (normals pointing inside), reverse depth.
fn planes(m: Mat4) -> [Vec4; 6] {
    let [r0, r1, r2, r3] = [0, 1, 2, 3].map(|i| m.row(i));
    [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2].map(|p| p / p.truncate().length())
}

/// Records one Hi-Z step from `src` into `dst`.
fn hiz_step<'a>(
    cmd: &mut CommandBuffer<Graphics>,
    src: impl SampledView<Dim = D2, Format: lava::image::format::HasTexels<[f32; 1]>> + 'a,
    dst: lava::ImageView<'a, R32Sfloat, usage![Sampled, Storage]>,
) {
    let src_size = src.view().extent();
    let size = dst.extent();
    let args = HizArgs {
        src,
        dst,
        src_size,
        size,
    };
    cmd.compute(&Hiz::push_bindings(args, None), Hiz::groups([size.x, size.y, 1]));
}

impl GpuDriven {
    pub fn new(setup: &mut Setup) -> Self {
        let (vertices, indices) = sphere(4, 6);
        let uploader = setup.uploader();
        let side = (INSTANCES as f32).sqrt() as u32;
        let instances: Vec<_> = (0..INSTANCES)
            .map(|i| CullInstance {
                sphere: Vec4::new(
                    (i % side) as f32 - side as f32 / 2.0 + rand01(i * 3) * 0.5,
                    rand01(i * 3 + 1) * 2.0,
                    (i / side) as f32 - side as f32 / 2.0 + rand01(i * 3 + 2) * 0.5,
                    0.3,
                ),
            })
            .collect();
        let view_proj = camera(Vec3::new(10.0, 25.0, 60.0), EXTENT.x as f32 / EXTENT.y as f32);
        let driven = Self {
            lava: Set::new(),
            raw: Set::new(),
            vertices: uploader.from_slice(&vertices).unwrap(),
            indices: uploader.from_slice(&indices).unwrap(),
            instances: uploader.from_slice(&instances).unwrap(),
            view_proj,
            planes: planes(view_proj),
            checksums: Checksums::new(2),
            pipelines: [
                pipelines::compute("Cull"),
                pipelines::raster(
                    "DrivenDraw",
                    Raster {
                        colors: &[vk::Format::R8G8B8A8_UNORM],
                        depth: vk::Format::D32_SFLOAT,
                        ..Default::default()
                    },
                ),
                pipelines::compute("Hiz"),
            ],
        };
        setup.lava(|_| {});
        let set = &driven.raw;
        unsafe {
            let read_only = L::SHADER_READ_ONLY_OPTIMAL;
            lava::raw::write_descriptors(set.color.whole_view(), read_only).unwrap();
            lava::raw::write_descriptors(set.depth.whole_view(), read_only).unwrap();
            for mip in 0..HIZ_MIPS {
                lava::raw::write_descriptors(set.hiz.mip_view(mip), read_only).unwrap();
            }
        }
        driven
    }
}

impl Scenario for GpuDriven {
    fn lava(&mut self, ctx: &mut LavaCtx) {
        let s = &self.lava;
        let check = ctx.check;
        ctx.submit(|cmd| {
            cmd.fill_buffer(s.count.range(..), 0);
            let args = CullArgs {
                planes: self.planes,
                instances: self.instances.range(..),
                draws: s.draws.range(..).cast(),
                count: s.count.range(..),
                instance_count: INSTANCES,
                index_count: self.indices.len() as u32,
            };
            cmd.compute(&Cull::push_bindings(args, None), Cull::groups([INSTANCES, 1, 1]));

            let args = DrivenDrawArgs {
                view_proj: self.view_proj,
                vertices: self.vertices.range(..),
                instances: self.instances.range(..),
            };
            let bindings = DrivenDraw::push_bindings(args, None);
            let draw = Draw::indexed_indirect_count(
                &bindings,
                self.indices.range(..),
                s.draws.range(..),
                s.count.range(..),
                RasterState::default(),
            );
            cmd.raster(EXTENT, [Attachment::color(s.color.whole_view(), Some([0.0; 4]))])
                .depth_attachment(s.depth.whole_view(), Some([0.0]))
                .draw(&draw)
                .record("instances");

            hiz_step(cmd, s.depth.whole_view(), s.hiz.mip_view(0));
            for mip in 1..HIZ_MIPS {
                hiz_step(cmd, s.hiz.mip_view(mip - 1), s.hiz.mip_view(mip));
            }

            if check {
                self.checksums.lava_clear(cmd);
                self.checksums.lava_image(cmd, s.color.whole_view(), 0);
                self.checksums.lava_buffer(cmd, s.count.range(..), 1);
            }
        });
    }

    fn raw(&mut self, ctx: &mut RawCtx) {
        let s = &self.raw;
        let [cull, draw, hiz] = self.pipelines;
        let check = ctx.check;
        ctx.submit(&[], |cmd| {
            let color = s.color.whole_view().image();
            let depth = s.depth.whole_view().image();
            let hiz_image = s.hiz.whole_view().image();
            let aspect = vk::ImageAspectFlags::DEPTH;
            let compute_done = (vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::NONE);
            // Last frame's draw read the count and the commands.
            cmd.mem(raw::INDIRECT_READ, raw::TRANSFER_WRITE);
            cmd.fill(s.count.address(), s.count.size(), 0);
            cmd.mem(raw::TRANSFER_WRITE, raw::COMPUTE_RW);
            let mut push: CCullBindings = bytemuck::Zeroable::zeroed();
            push.planes = self.planes;
            push.instances = self.instances.address();
            push.draws = s.draws.address();
            push.count = s.count.address();
            push.instance_count = INSTANCES;
            push.index_count = self.indices.len() as u32;
            cmd.bind_compute(cull);
            cmd.push(&push);
            cmd.dispatch(Cull::groups([INSTANCES, 1, 1]));

            cmd.barrier(
                &[raw::mem(raw::COMPUTE_WRITE, raw::INDIRECT_READ)],
                &[
                    img(color, 0..1, L::UNDEFINED, L::COLOR_ATTACHMENT_OPTIMAL, compute_done, raw::COLOR_WRITE),
                    img_aspect(depth, aspect, 0..1, L::UNDEFINED, L::DEPTH_ATTACHMENT_OPTIMAL, raw::or(compute_done, raw::DEPTH_WRITE), raw::DEPTH_WRITE),
                ],
            );
            cmd.begin_rendering(
                EXTENT.into(),
                &[Color { view: s.color.whole_view().vk_view(), clear: Some([0.0; 4]) }],
                Some((s.depth.whole_view().vk_view(), Some(0.0))),
            );
            cmd.bind_graphics(draw);
            let mut push: CDrivenDrawBindings = bytemuck::Zeroable::zeroed();
            push.view_proj = self.view_proj;
            push.vertices = self.vertices.address();
            push.instances = self.instances.address();
            cmd.push(&push);
            cmd.bind_index_u32(self.indices.address(), self.indices.size());
            cmd.draw_indexed_indirect_count(
                s.draws.address(),
                size_of::<DrawIndexedIndirectCommand>() as u64,
                INSTANCES,
                s.count.address(),
            );
            cmd.end_rendering();

            cmd.images(&[
                img_aspect(depth, aspect, 0..1, L::DEPTH_ATTACHMENT_OPTIMAL, L::SHADER_READ_ONLY_OPTIMAL, raw::DEPTH_WRITE, raw::COMPUTE_READ),
                img(hiz_image, 0..HIZ_MIPS, L::UNDEFINED, L::GENERAL, compute_done, raw::COMPUTE_WRITE),
            ]);
            cmd.bind_compute(hiz);
            for mip in 0..HIZ_MIPS {
                let mut push: CHizBindings = bytemuck::Zeroable::zeroed();
                (push.src, push.src_size) = match mip {
                    0 => (s.depth.handle::<Sampled>(), EXTENT),
                    _ => (s.hiz.mip_view(mip - 1).handle::<Sampled>(), s.hiz.mip_extent(mip - 1)),
                };
                push.dst = s.hiz.mip_view(mip).handle::<Storage>();
                push.size = s.hiz.mip_extent(mip);
                cmd.push(&push);
                cmd.dispatch(Hiz::groups([push.size.x, push.size.y, 1]));
                cmd.images(&[img(hiz_image, mip..mip + 1, L::GENERAL, L::SHADER_READ_ONLY_OPTIMAL, raw::COMPUTE_WRITE, raw::COMPUTE_READ)]);
            }

            if check {
                cmd.barrier(
                    &[raw::mem(raw::COMPUTE_WRITE, raw::COMPUTE_READ)],
                    &[img(color, 0..1, L::COLOR_ATTACHMENT_OPTIMAL, L::SHADER_READ_ONLY_OPTIMAL, raw::COLOR_WRITE, raw::COMPUTE_READ)],
                );
                self.checksums.raw_clear(cmd);
                self.checksums.raw_image(cmd, s.color.handle::<Sampled>(), EXTENT, 0);
                self.checksums.raw_buffer(cmd, s.count.address(), 1, 1);
                self.checksums.raw_finish(cmd);
            }
        });
    }

    fn checksums(&mut self) -> (Vec<u32>, Vec<u32>) {
        self.checksums.read()
    }
}

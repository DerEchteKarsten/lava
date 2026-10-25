//! `forward`, an everyday game frame: a 256-draw shadow map, 2000 textured and shadowed draws whose materials (512 textures and their samplers) come from a referencing buffer, a tonemap dispatch and 64 alpha-blended UI draws

use glam::{Mat4, UVec2, Vec2, Vec3, Vec4};
use lava::bindless::{Sampled, Sampler, SamplerDesc, Storage};
use lava::command_buffer::{Attachment, Blend, Draw, RasterState};
use lava::image::format::{D32Sfloat, R8G8B8A8Unorm, R16G16B16A16Sfloat};
use lava::vk::{self, ImageLayout as L};
use lava::{Buffer, Image, buffer::Upload, usage};

use crate::bindings::*;
use crate::check::Checksums;
use crate::pipelines::{self, Raster};
use crate::raw::{self, Color, RawCmd, img, img_aspect};
use crate::scenario::{LavaCtx, RawCtx, Scenario, camera, rand01, scatter, sphere};
use crate::scenarios::Setup;

const EXTENT: UVec2 = UVec2::new(1280, 720);
const SHADOW: UVec2 = UVec2::splat(2048);
const SHADOW_DRAWS: u32 = 256;
const DRAWS: u32 = 2000;
const TEXTURES: usize = 512;
const UI_DRAWS: u32 = 64;
const UI_QUADS_PER_DRAW: u32 = 16;

type Depth = Image<D32Sfloat, usage![DepthAttachment, Sampled]>;
type Hdr = Image<R16G16B16A16Sfloat, usage![ColorAttachment, Sampled]>;
type Ldr = Image<R8G8B8A8Unorm, usage![ColorAttachment, Storage, Sampled]>;
type Texture = Image<R8G8B8A8Unorm, usage![Sampled, TransferDst]>;

/// The images of one mode, and its materials (which reference its textures).
struct Set {
    shadow: Depth,
    depth: Image<D32Sfloat, usage![DepthAttachment]>,
    hdr: Hdr,
    ldr: Ldr,
    textures: Vec<Texture>,
    atlas: Texture,
    materials: Buffer<Material, Upload>,
}

impl Set {
    fn new() -> Self {
        let texture = || Texture::new(UVec2::splat(64)).unwrap().bind_sampled().unwrap();
        Self {
            shadow: Depth::new(SHADOW).unwrap().bind_sampled().unwrap(),
            depth: Image::new(EXTENT).unwrap(),
            hdr: Hdr::new(EXTENT).unwrap().bind_sampled().unwrap(),
            ldr: Ldr::new(EXTENT).unwrap().bind_combined().unwrap(),
            textures: (0..TEXTURES).map(|_| texture()).collect(),
            atlas: texture(),
            materials: Buffer::upload(TEXTURES).unwrap(),
        }
    }

    fn textures(&self) -> impl Iterator<Item = &Texture> {
        self.textures.iter().chain([&self.atlas])
    }
}

pub struct Forward {
    lava: Set,
    raw: Set,
    vertices: Buffer<MeshVertex>,
    indices: Buffer<u32>,
    instances: Buffer<DrawInstance>,
    quads: Buffer<UiQuad>,
    sampler: Sampler,
    shadow_sampler: Sampler,
    view_proj: Mat4,
    light_view_proj: Mat4,
    checksums: Checksums,
    pipelines: [vk::Pipeline; 4],
}

fn texture_color(i: usize) -> [f32; 4] {
    let i = i as u32;
    [rand01(i * 3), rand01(i * 3 + 1), rand01(i * 3 + 2), 1.0]
}

impl Forward {
    pub fn new(setup: &mut Setup) -> Self {
        let (vertices, indices) = sphere(12, 16);
        let uploader = setup.uploader();
        let instances = scatter(DRAWS, 40.0);
        let quads: Vec<_> = (0..UI_DRAWS * UI_QUADS_PER_DRAW)
            .map(|i| UiQuad {
                position: Vec2::new(rand01(i * 2) * 0.9, rand01(i * 2 + 1) * 0.9),
                size: Vec2::splat(0.05),
                color: Vec4::new(1.0, 1.0, 1.0, 0.6),
                uv_rect: Vec4::new(0.0, 0.0, 1.0, 1.0),
            })
            .collect();
        let sampler = Sampler::new(&SamplerDesc::new()).unwrap();
        let shadow_sampler = Sampler::new(
            &SamplerDesc::new()
                .filter(vk::Filter::NEAREST)
                .address_mode(vk::SamplerAddressMode::CLAMP_TO_EDGE),
        )
        .unwrap();
        let mut lava = Set::new();
        let raw = Set::new();
        // Materials: lava's through their reference infos, which keep the textures; the
        // baseline writes the handles itself.
        lava.materials.host_write_referencing(
            0,
            lava.textures.iter().map(|texture| MaterialReferenceInfo {
                albedo: Some(texture.whole_view()),
                sampler: &sampler,
                tint: [1.0, 0.9, 0.8, 1.0],
            }),
        );
        // `host_mut` takes only `NoUninit` elements, which referencing structs aren't.
        for (i, texture) in raw.textures.iter().enumerate() {
            let material = Material {
                albedo: texture.handle::<Sampled>(),
                sampler: sampler.handle(),
                tint: [1.0, 0.9, 0.8, 1.0],
                ..Default::default()
            };
            unsafe { raw.materials.host_ptr().add(i).write(material) };
        }
        let forward = Self {
            vertices: uploader.from_slice(&vertices).unwrap(),
            indices: uploader.from_slice(&indices).unwrap(),
            instances: uploader.from_slice(&instances).unwrap(),
            quads: uploader.from_slice(&quads).unwrap(),
            view_proj: camera(Vec3::new(0.0, 12.0, 28.0), EXTENT.x as f32 / EXTENT.y as f32),
            light_view_proj: Mat4::orthographic_rh(-25.0, 25.0, -25.0, 25.0, 60.0, 0.1)
                * Mat4::look_at_rh(Vec3::new(10.0, 30.0, 10.0), Vec3::ZERO, Vec3::Y),
            sampler,
            shadow_sampler,
            checksums: Checksums::new(1),
            pipelines: [
                pipelines::raster(
                    "ForwardShadow",
                    Raster {
                        depth: vk::Format::D32_SFLOAT,
                        ..Default::default()
                    },
                ),
                pipelines::raster(
                    "ForwardMain",
                    Raster {
                        colors: &[vk::Format::R16G16B16A16_SFLOAT],
                        depth: vk::Format::D32_SFLOAT,
                        ..Default::default()
                    },
                ),
                pipelines::compute("Tonemap"),
                pipelines::raster(
                    "Ui",
                    Raster {
                        colors: &[vk::Format::R8G8B8A8_UNORM],
                        blend: pipelines::Blend::Alpha,
                        ..Default::default()
                    },
                ),
            ],
            lava,
            raw,
        };
        let set = &forward.lava;
        setup.lava(|cmd| {
            // Static after their upload: their reads go untracked.
            cmd.freeze(&forward.vertices);
            cmd.freeze(&forward.indices);
            cmd.freeze(&forward.instances);
            cmd.freeze(&forward.quads);
            for (i, texture) in set.textures().enumerate() {
                cmd.clear_image(texture.whole_view(), texture_color(i));
            }
        });
        // The baseline's sampled images stay in SHADER_READ_ONLY_OPTIMAL while shaders sample
        // them, and their descriptors say so.
        let set = &forward.raw;
        unsafe {
            for texture in set.textures() {
                lava::raw::write_descriptors(texture.whole_view(), L::SHADER_READ_ONLY_OPTIMAL)
                    .unwrap();
            }
            lava::raw::write_descriptors(set.shadow.whole_view(), L::SHADER_READ_ONLY_OPTIMAL)
                .unwrap();
            lava::raw::write_descriptors(set.hdr.whole_view(), L::SHADER_READ_ONLY_OPTIMAL)
                .unwrap();
            lava::raw::write_descriptors(set.ldr.whole_view(), L::SHADER_READ_ONLY_OPTIMAL)
                .unwrap();
        }
        setup.raw(|cmd| {
            let to_dst: Vec<_> = (set.textures())
                .map(|t| {
                    let image = t.whole_view().image();
                    img(image, 0..1, L::UNDEFINED, L::TRANSFER_DST_OPTIMAL, raw::NONE, raw::TRANSFER_WRITE)
                })
                .collect();
            cmd.images(&to_dst);
            for (i, texture) in set.textures().enumerate() {
                cmd.clear(texture.whole_view().image(), texture_color(i));
            }
            let to_read: Vec<_> = (set.textures())
                .map(|t| {
                    let image = t.whole_view().image();
                    img(image, 0..1, L::TRANSFER_DST_OPTIMAL, L::SHADER_READ_ONLY_OPTIMAL, raw::TRANSFER_WRITE, raw::GRAPHICS_READ)
                })
                .collect();
            cmd.images(&to_read);
        });
        forward
    }
}

impl Scenario for Forward {
    fn lava(&mut self, ctx: &mut LavaCtx) {
        let s = &self.lava;
        let check = ctx.check;
        ctx.submit(|cmd| {
            let vertices = self.vertices.range(..);
            let indices = self.indices.range(..);
            let instances = self.instances.range(..);

            // The shadow map: depth only, one draw per instance.
            let shadow: Vec<_> = (0..SHADOW_DRAWS)
                .map(|instance| {
                    let args = ForwardShadowArgs {
                        light_view_proj: self.light_view_proj,
                        vertices,
                        instances,
                        instance,
                    };
                    ForwardShadow::push_bindings(args, None)
                })
                .collect();
            let draws: Vec<_> = (shadow.iter())
                .map(|b| Draw::indexed(b, indices, 1, RasterState::default()))
                .collect();
            let mut rendering = cmd.raster(SHADOW, []);
            rendering.depth_attachment(s.shadow.whole_view(), Some([0.0]));
            for draw in &draws {
                rendering.draw(draw);
            }
            rendering.record("shadow");

            // The main pass.
            let main: Vec<_> = (0..DRAWS)
                .map(|instance| {
                    let args = ForwardMainArgs {
                        view_proj: self.view_proj,
                        light_view_proj: self.light_view_proj,
                        vertices,
                        instances,
                        materials: s.materials.range(..),
                        shadow: s.shadow.whole_view(),
                        shadow_sampler: self.shadow_sampler.handle(),
                        instance,
                        material: instance % TEXTURES as u32,
                    };
                    ForwardMain::push_bindings(args, None)
                })
                .collect();
            let draws: Vec<_> = (main.iter())
                .map(|b| Draw::indexed(b, indices, 1, RasterState::default()))
                .collect();
            let mut rendering =
                cmd.raster(EXTENT, [Attachment::color(s.hdr.whole_view(), Some([0.0; 4]))]);
            rendering.depth_attachment(s.depth.whole_view(), Some([0.0]));
            for draw in &draws {
                rendering.draw(draw);
            }
            rendering.record("main");

            let args = TonemapArgs {
                hdr: s.hdr.whole_view(),
                ldr: s.ldr.whole_view(),
                size: EXTENT,
            };
            cmd.compute(
                &Tonemap::push_bindings(args, None),
                Tonemap::groups([EXTENT.x, EXTENT.y, 1]),
            );

            let ui: Vec<_> = (0..UI_DRAWS)
                .map(|i| {
                    let args = UiArgs {
                        quads: self.quads.range(..),
                        atlas: s.atlas.whole_view(),
                        sampler: self.sampler.handle(),
                        first: i * UI_QUADS_PER_DRAW,
                    };
                    Ui::push_bindings(args, None)
                })
                .collect();
            let state = RasterState::default().blend(Blend::Alpha);
            let draws: Vec<_> = (ui.iter())
                .map(|b| Draw::new(b, 6, UI_QUADS_PER_DRAW, state))
                .collect();
            let mut rendering = cmd.raster(EXTENT, [Attachment::color(s.ldr.whole_view(), None)]);
            for draw in &draws {
                rendering.draw(draw);
            }
            rendering.record("ui");

            if check {
                self.checksums.lava_clear(cmd);
                self.checksums.lava_image(cmd, s.ldr.whole_view(), 0);
            }
        });
    }

    fn raw(&mut self, ctx: &mut RawCtx) {
        let s = &self.raw;
        let [shadow_pipeline, main_pipeline, tonemap_pipeline, ui_pipeline] = self.pipelines;
        let check = ctx.check;
        ctx.submit(&[], |cmd: &mut RawCmd| {
            let shadow = s.shadow.whole_view().image();
            let depth = vk::ImageAspectFlags::DEPTH;
            // Last frame's main pass sampled the shadow map: its contents are dropped.
            let fragment_done = (vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::NONE);
            cmd.images(&[img_aspect(shadow, depth, 0..1, L::UNDEFINED, L::DEPTH_ATTACHMENT_OPTIMAL, fragment_done, raw::DEPTH_WRITE)]);
            cmd.begin_rendering(SHADOW.into(), &[], Some((s.shadow.whole_view().vk_view(), Some(0.0))));
            cmd.bind_graphics(shadow_pipeline);
            cmd.bind_index_u32(self.indices.address(), self.indices.size());
            let index_count = self.indices.len() as u32;
            let mut push: CForwardShadowBindings = bytemuck::Zeroable::zeroed();
            push.light_view_proj = self.light_view_proj;
            push.vertices = self.vertices.address();
            push.instances = self.instances.address();
            for instance in 0..SHADOW_DRAWS {
                push.instance = instance;
                cmd.push(&push);
                cmd.draw_indexed(index_count, 1);
            }
            cmd.end_rendering();

            let hdr = s.hdr.whole_view().image();
            let ldr = s.ldr.whole_view().image();
            let compute_done = (vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::NONE);
            cmd.images(&[
                img_aspect(shadow, depth, 0..1, L::DEPTH_ATTACHMENT_OPTIMAL, L::SHADER_READ_ONLY_OPTIMAL, raw::DEPTH_WRITE, raw::FRAGMENT_READ),
                // Last frame's tonemap read it.
                img(hdr, 0..1, L::UNDEFINED, L::COLOR_ATTACHMENT_OPTIMAL, compute_done, raw::COLOR_WRITE),
                img_aspect(s.depth.whole_view().image(), depth, 0..1, L::UNDEFINED, L::DEPTH_ATTACHMENT_OPTIMAL, raw::DEPTH_WRITE, raw::DEPTH_WRITE),
            ]);
            cmd.begin_rendering(
                EXTENT.into(),
                &[Color { view: s.hdr.whole_view().vk_view(), clear: Some([0.0; 4]) }],
                Some((s.depth.whole_view().vk_view(), Some(0.0))),
            );
            cmd.bind_graphics(main_pipeline);
            let mut push: CForwardMainBindings = bytemuck::Zeroable::zeroed();
            push.view_proj = self.view_proj;
            push.light_view_proj = self.light_view_proj;
            push.vertices = self.vertices.address();
            push.instances = self.instances.address();
            push.materials = s.materials.address();
            push.shadow = s.shadow.handle::<Sampled>();
            push.shadow_sampler = self.shadow_sampler.handle();
            for instance in 0..DRAWS {
                push.instance = instance;
                push.material = instance % TEXTURES as u32;
                cmd.push(&push);
                cmd.draw_indexed(index_count, 1);
            }
            cmd.end_rendering();

            cmd.images(&[
                img(hdr, 0..1, L::COLOR_ATTACHMENT_OPTIMAL, L::SHADER_READ_ONLY_OPTIMAL, raw::COLOR_WRITE, raw::COMPUTE_READ),
                // Last frame's UI drew into it, and a check read it.
                img(ldr, 0..1, L::UNDEFINED, L::GENERAL, raw::or(raw::COLOR_WRITE, raw::COMPUTE_READ), raw::COMPUTE_WRITE),
            ]);
            cmd.bind_compute(tonemap_pipeline);
            let mut push: CTonemapBindings = bytemuck::Zeroable::zeroed();
            push.hdr = s.hdr.handle::<Sampled>();
            push.ldr = s.ldr.handle::<Storage>();
            push.size = EXTENT;
            cmd.push(&push);
            cmd.dispatch(Tonemap::groups([EXTENT.x, EXTENT.y, 1]));

            cmd.images(&[img(ldr, 0..1, L::GENERAL, L::COLOR_ATTACHMENT_OPTIMAL, raw::COMPUTE_WRITE, raw::COLOR_WRITE)]);
            cmd.begin_rendering(EXTENT.into(), &[Color { view: s.ldr.whole_view().vk_view(), clear: None }], None);
            cmd.bind_graphics(ui_pipeline);
            let mut push: CUiBindings = bytemuck::Zeroable::zeroed();
            push.quads = self.quads.address();
            push.atlas = s.atlas.handle::<Sampled>();
            push.sampler = self.sampler.handle();
            for i in 0..UI_DRAWS {
                push.first = i * UI_QUADS_PER_DRAW;
                cmd.push(&push);
                cmd.draw(6, UI_QUADS_PER_DRAW);
            }
            cmd.end_rendering();

            if check {
                cmd.images(&[img(ldr, 0..1, L::COLOR_ATTACHMENT_OPTIMAL, L::SHADER_READ_ONLY_OPTIMAL, raw::COLOR_WRITE, raw::COMPUTE_READ)]);
                self.checksums.raw_clear(cmd);
                self.checksums.raw_image(cmd, s.ldr.handle::<Sampled>(), EXTENT, 0);
                self.checksums.raw_finish(cmd);
            }
        });
    }

    fn checksums(&mut self) -> (Vec<u32>, Vec<u32>) {
        self.checksums.read()
    }
}

//! `streaming`: per frame the `Uploader` writes 32 buffers and the top mips of 8 textures, `drain_uploads` copies them, blits fill the textures' mip chains, 16 transient buffers are created, filled and dropped, and a consumer reads all of it through a referencing buffer the frame rewrites; the baseline rings its own staging buffer and defers its destroys per frame slot

use glam::UVec2;
use lava::bindless::{Sampled, Sampler, SamplerDesc};
use lava::buffer::Upload;
use lava::image::Filter;
use lava::image::format::R8G8B8A8Unorm;
use lava::vk::{self, ImageLayout as L};
use lava::{Buffer, Image, Uploader, usage};

use crate::IN_FLIGHT;
use crate::bindings::*;
use crate::check::Checksums;
use crate::pipelines;
use crate::raw::{self, img};
use crate::scenario::{LavaCtx, RawCtx, Scenario};
use crate::scenarios::Setup;

const BUFFERS: usize = 32;
const WORDS: usize = 4096;
const TEXTURES: usize = 8;
const TEXTURE: UVec2 = UVec2::splat(256);
const MIPS: u32 = 9;
const TRANSIENTS: usize = 16;
const TRANSIENT_WORDS: usize = 4 * WORDS;
const TEXTURE_BYTES: usize = (TEXTURE.x * TEXTURE.y * 4) as usize;
/// One frame's uploads in the baseline's staging ring.
const FRAME_BYTES: usize = BUFFERS * WORDS * 4 + TEXTURES * TEXTURE_BYTES;

type Texture = Image<R8G8B8A8Unorm, usage![Sampled, TransferSrc, TransferDst]>;

struct Set {
    buffers: Vec<Buffer<u32>>,
    textures: Vec<Texture>,
    /// One referencing buffer of inputs per frame slot.
    inputs: Vec<Buffer<StreamInputs, Upload>>,
    out: Buffer<u32>,
    frame: usize,
}

impl Set {
    fn new() -> Self {
        Self {
            buffers: (0..BUFFERS).map(|_| Buffer::new(WORDS).unwrap()).collect(),
            textures: (0..TEXTURES)
                .map(|_| {
                    Texture::with_mip_levels(TEXTURE, MIPS)
                        .unwrap()
                        .bind_sampled()
                        .unwrap()
                })
                .collect(),
            inputs: (0..IN_FLIGHT).map(|_| Buffer::upload(1).unwrap()).collect(),
            out: Buffer::new(WORDS).unwrap(),
            frame: 0,
        }
    }
}

pub struct Streaming {
    lava: Set,
    raw: Set,
    uploader: Uploader,
    /// The baseline's staging ring: one frame's uploads per frame slot.
    staging: Buffer<u8, Upload>,
    /// The baseline's transient buffers, dropped when their frame slot comes round again.
    raw_transients: Vec<Vec<Buffer<u32>>>,
    data: Vec<Vec<u32>>,
    texels: Vec<Vec<u8>>,
    sampler: Sampler,
    checksums: Checksums,
    pipelines: [vk::Pipeline; 2],
}

impl Streaming {
    pub fn new(setup: &mut Setup) -> Self {
        let data = (0..BUFFERS)
            .map(|b| (0..WORDS as u32).map(|i| i * 31 + b as u32).collect())
            .collect();
        let texels = (0..TEXTURES)
            .map(|t| (0..TEXTURE_BYTES).map(|i| (i * 7 + t * 13) as u8).collect())
            .collect();
        let streaming = Self {
            lava: Set::new(),
            raw: Set::new(),
            uploader: setup.uploader(),
            staging: Buffer::upload(IN_FLIGHT as usize * FRAME_BYTES).unwrap(),
            raw_transients: (0..IN_FLIGHT).map(|_| Vec::new()).collect(),
            data,
            texels,
            sampler: Sampler::new(&SamplerDesc::new()).unwrap(),
            checksums: Checksums::new(1),
            pipelines: [
                pipelines::compute("StreamFill"),
                pipelines::compute("StreamConsume"),
            ],
        };
        unsafe {
            for texture in &streaming.raw.textures {
                lava::raw::write_descriptors(texture.whole_view(), L::SHADER_READ_ONLY_OPTIMAL)
                    .unwrap();
            }
        }
        streaming
    }
}

impl Scenario for Streaming {
    fn lava(&mut self, ctx: &mut LavaCtx) {
        let s = &mut self.lava;
        s.frame += 1;
        let slot = s.frame % IN_FLIGHT as usize;
        let check = ctx.check;
        let (uploader, data, texels, sampler) = (&self.uploader, &self.data, &self.texels, &self.sampler);
        let checksums = &self.checksums;
        ctx.submit(|cmd| {
            for (buffer, data) in s.buffers.iter().zip(data) {
                uploader.write(buffer, 0, data).unwrap();
            }
            for (texture, texels) in s.textures.iter().zip(texels) {
                uploader.image(texture, 0, texels).unwrap();
            }
            cmd.drain_uploads();
            for mip in 1..MIPS {
                for texture in &s.textures {
                    cmd.blit_image(
                        texture.whole().mip(mip - 1),
                        texture.whole().mip(mip),
                        Filter::Linear,
                    );
                }
            }
            // Transient buffers: made, filled and read this frame, dropped at its end (the
            // submission keeps them until it completed).
            let transients: Vec<Buffer<u32>> =
                (0..TRANSIENTS).map(|_| Buffer::new(TRANSIENT_WORDS).unwrap()).collect();
            for (i, transient) in transients.iter().enumerate() {
                let args = StreamFillArgs {
                    dst: transient.range(..),
                    count: TRANSIENT_WORDS as u32,
                    seed: i as u32,
                };
                cmd.compute(
                    &StreamFill::push_bindings(args, None),
                    StreamFill::groups([TRANSIENT_WORDS as u32, 1, 1]),
                );
            }
            let inputs = &mut s.inputs[slot];
            inputs.host_write_referencing(
                0,
                [StreamInputsReferenceInfo {
                    uploaded: std::array::from_fn(|i| s.buffers[i].range(..)),
                    transients: std::array::from_fn(|i| transients[i].range(..)),
                    textures: std::array::from_fn(|i| Some(s.textures[i].whole_view())),
                    sampler,
                    pad: 0,
                }],
            );
            let args = StreamConsumeArgs {
                inputs: s.inputs[slot].range(..),
                out: s.out.range(..),
                count: WORDS as u32,
            };
            cmd.compute(
                &StreamConsume::push_bindings(args, None),
                StreamConsume::groups([WORDS as u32, 1, 1]),
            );
            if check {
                checksums.lava_clear(cmd);
                checksums.lava_buffer(cmd, s.out.range(..), 0);
            }
        });
    }

    fn raw(&mut self, ctx: &mut RawCtx) {
        let s = &mut self.raw;
        s.frame += 1;
        let slot = s.frame % IN_FLIGHT as usize;
        let [fill, consume] = self.pipelines;
        let check = ctx.check;
        let (staging, data, texels) = (&self.staging, &self.data, &self.texels);
        let (transients, sampler, checksums) = (&mut self.raw_transients[slot], &self.sampler, &self.checksums);
        ctx.submit(&[], |cmd| {
            // This slot's last frame completed: its transients go, its staging range is free.
            transients.clear();
            let base = slot * FRAME_BYTES;
            let ring = staging.host_ptr();
            let mut offset = base;
            let mut copies = Vec::with_capacity(BUFFERS);
            for (buffer, data) in s.buffers.iter().zip(data) {
                let bytes: &[u8] = bytemuck::cast_slice(data);
                unsafe { ring.add(offset).copy_from_nonoverlapping(bytes.as_ptr(), bytes.len()) };
                copies.push((staging.address() + offset as u64, buffer.address(), bytes.len() as u64));
                offset += bytes.len();
            }
            let mut texture_offsets = Vec::with_capacity(TEXTURES);
            for texels in texels {
                unsafe { ring.add(offset).copy_from_nonoverlapping(texels.as_ptr(), texels.len()) };
                texture_offsets.push(staging.address() + offset as u64);
                offset += texels.len();
            }

            // Last frame's consumer read the buffers and textures.
            let images: Vec<_> = (s.textures.iter())
                .map(|t| img(t.whole_view().image(), 0..MIPS, L::UNDEFINED, L::TRANSFER_DST_OPTIMAL, raw::COMPUTE_READ, raw::TRANSFER_WRITE))
                .collect();
            cmd.barrier(&[raw::mem(raw::COMPUTE_READ, raw::TRANSFER_WRITE)], &images);
            cmd.copy(&copies);
            for (texture, &address) in s.textures.iter().zip(&texture_offsets) {
                let image = texture.whole_view().image();
                cmd.copy_to_image(address, TEXTURE_BYTES as u64, image, 0, TEXTURE.into());
            }
            // Mip chains: each mip a blit of the one above, all of one level at a time.
            for mip in 1..MIPS {
                let images: Vec<_> = (s.textures.iter())
                    .map(|t| img(t.whole_view().image(), mip - 1..mip, L::TRANSFER_DST_OPTIMAL, L::TRANSFER_SRC_OPTIMAL, raw::TRANSFER_WRITE, raw::TRANSFER_READ))
                    .collect();
                cmd.images(&images);
                for texture in &s.textures {
                    cmd.blit_mip(texture.whole_view().image(), mip - 1, mip, TEXTURE.into());
                }
            }
            let mut images: Vec<_> = (s.textures.iter())
                .map(|t| img(t.whole_view().image(), 0..MIPS - 1, L::TRANSFER_SRC_OPTIMAL, L::SHADER_READ_ONLY_OPTIMAL, raw::TRANSFER_READ, raw::COMPUTE_READ))
                .collect();
            images.extend((s.textures.iter()).map(|t| {
                img(t.whole_view().image(), MIPS - 1..MIPS, L::TRANSFER_DST_OPTIMAL, L::SHADER_READ_ONLY_OPTIMAL, raw::TRANSFER_WRITE, raw::COMPUTE_READ)
            }));
            cmd.barrier(&[raw::mem(raw::TRANSFER_WRITE, raw::COMPUTE_READ)], &images);

            transients.extend((0..TRANSIENTS).map(|_| Buffer::new(TRANSIENT_WORDS).unwrap()));
            cmd.bind_compute(fill);
            for (i, transient) in transients.iter().enumerate() {
                let mut push: CStreamFillBindings = bytemuck::Zeroable::zeroed();
                push.dst = transient.address();
                push.count = TRANSIENT_WORDS as u32;
                push.seed = i as u32;
                cmd.push(&push);
                cmd.dispatch(StreamFill::groups([TRANSIENT_WORDS as u32, 1, 1]));
            }
            let mut inputs = StreamInputs {
                sampler: sampler.handle(),
                ..Default::default()
            };
            for (address, buffer) in inputs.uploaded.iter_mut().zip(&s.buffers) {
                *address = buffer.address();
            }
            for (address, buffer) in inputs.transients.iter_mut().zip(transients.iter()) {
                *address = buffer.address();
            }
            for (handle, texture) in inputs.textures.iter_mut().zip(&s.textures) {
                *handle = texture.handle::<Sampled>();
            }
            unsafe { s.inputs[slot].host_ptr().write(inputs) };
            // The transients' fills and last frame's consumer's write of `out`.
            cmd.mem(raw::COMPUTE_RW, raw::COMPUTE_RW);
            let mut push: CStreamConsumeBindings = bytemuck::Zeroable::zeroed();
            push.inputs = s.inputs[slot].address();
            push.out = s.out.address();
            push.count = WORDS as u32;
            cmd.bind_compute(consume);
            cmd.push(&push);
            cmd.dispatch(StreamConsume::groups([WORDS as u32, 1, 1]));
            if check {
                cmd.mem(raw::COMPUTE_WRITE, raw::COMPUTE_READ);
                checksums.raw_clear(cmd);
                checksums.raw_buffer(cmd, s.out.address(), WORDS as u32, 0);
                checksums.raw_finish(cmd);
            }
        });
    }

    fn checksums(&mut self) -> (Vec<u32>, Vec<u32>) {
        self.checksums.read()
    }
}

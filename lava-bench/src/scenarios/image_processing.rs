//! `image_processing`: a photo or video filter chain on 1080p storage images ping-ponged between two targets (separable blur, grading, sharpening, vignette, output), a luminance histogram of four commuting atomic dispatches, and its readback, which the host turns into the next frame's exposure

use glam::{UVec2, Vec4};
use lava::bindless::{ImageHandle, Sampled, Sampler, SamplerDesc, Storage};
use lava::buffer::Readback;
use lava::command_buffer::{Accesses, BufferAccess};
use lava::image::format::{R8G8B8A8Unorm, R16G16B16A16Sfloat};
use lava::vk::{self, ImageLayout as L};
use lava::{Buffer, Image, usage};

use crate::IN_FLIGHT;
use crate::bindings::*;
use crate::check::Checksums;
use crate::pipelines;
use crate::raw::{self, RawCmd, img};
use crate::scenario::{LavaCtx, RawCtx, Scenario};
use crate::scenarios::Setup;

const EXTENT: UVec2 = UVec2::new(1920, 1080);
const BANDS: u32 = 4;

type Work = Image<R16G16B16A16Sfloat, usage![Sampled, Storage]>;
type Picture = Image<R8G8B8A8Unorm, usage![Sampled, Storage]>;

struct Set {
    input: Picture,
    a: Work,
    b: Work,
    out: Picture,
    bins: Buffer<u32>,
    readback: Vec<Buffer<u32, Readback>>,
    frame: usize,
}

impl Set {
    fn new() -> Self {
        Self {
            input: Picture::new(EXTENT).unwrap().bind_combined().unwrap(),
            a: Work::new(EXTENT).unwrap().bind_combined().unwrap(),
            b: Work::new(EXTENT).unwrap().bind_combined().unwrap(),
            out: Picture::new(EXTENT).unwrap().bind_combined().unwrap(),
            bins: Buffer::new(256).unwrap(),
            // Zero until a frame wrote them: the first frame's exposure is 1.
            readback: (0..IN_FLIGHT)
                .map(|_| {
                    let mut readback = Buffer::readback(256).unwrap();
                    readback.host_mut().fill(0);
                    readback
                })
                .collect(),
            frame: 0,
        }
    }

    /// The exposure from the last frame's histogram: mid-grey at the mean luminance.
    fn exposure(&mut self) -> f32 {
        let last = (self.frame + IN_FLIGHT as usize - 1) % IN_FLIGHT as usize;
        let bins = self.readback[last].host();
        let count: u32 = bins.iter().sum();
        if count == 0 {
            return 1.0;
        }
        let sum: f32 = (bins.iter().enumerate()).map(|(i, &n)| (i as f32 + 0.5) / 256.0 * n as f32).sum();
        (0.18 / (sum / count as f32).max(0.01)).clamp(0.5, 4.0)
    }
}

/// The chain's filters after the first: pass index into the pipelines, its parameters.
const CHAIN: [(usize, Vec4); 4] = [
    (1, Vec4::new(0.0, 1.0 / EXTENT.y as f32, 8.0, 0.0)),
    (2, Vec4::new(0.02, 0.9, 1.1, 1.2)),
    (3, Vec4::new(0.4, 0.0, 0.0, 0.0)),
    (4, Vec4::new(0.6, 3.0, 0.0, 0.0)),
];

pub struct ImageProcessing {
    lava: Set,
    raw: Set,
    sampler: Sampler,
    checksums: Checksums,
    /// blur_h, blur_v, grade, sharpen, vignette, output, histogram, generate.
    pipelines: [vk::Pipeline; 8],
}

impl ImageProcessing {
    pub fn new(setup: &mut Setup) -> Self {
        let processing = Self {
            lava: Set::new(),
            raw: Set::new(),
            sampler: Sampler::new(
                &SamplerDesc::new().address_mode(vk::SamplerAddressMode::CLAMP_TO_EDGE),
            )
            .unwrap(),
            checksums: Checksums::new(2),
            pipelines: [
                "FilterBlurH",
                "FilterBlurV",
                "FilterGrade",
                "FilterSharpen",
                "FilterVignette",
                "FilterOutput",
                "Histogram",
                "FilterGenerate",
            ]
            .map(pipelines::compute),
        };
        let sampler = processing.sampler.handle();
        let s = &processing.lava;
        setup.lava(|cmd| {
            let args = FilterGenerateArgs {
                src: s.a.whole_view(),
                dst: s.input.whole_view(),
                sampler,
                size: EXTENT,
                params: Vec4::ZERO,
            };
            cmd.compute(&FilterGenerate::push_bindings(args, None), FilterGenerate::groups([EXTENT.x, EXTENT.y, 1]));
        });
        let s = &processing.raw;
        unsafe {
            for image in [&s.input, &s.out] {
                lava::raw::write_descriptors(image.whole_view(), L::SHADER_READ_ONLY_OPTIMAL).unwrap();
            }
            for image in [&s.a, &s.b] {
                lava::raw::write_descriptors(image.whole_view(), L::SHADER_READ_ONLY_OPTIMAL).unwrap();
            }
        }
        let generate = processing.pipelines[7];
        setup.raw(|cmd| {
            let input = s.input.whole_view().image();
            cmd.images(&[img(input, 0..1, L::UNDEFINED, L::GENERAL, raw::NONE, raw::COMPUTE_WRITE)]);
            let mut push: CFilterGenerateBindings = bytemuck::Zeroable::zeroed();
            push.src = s.a.handle::<Sampled>();
            push.dst = s.input.handle::<Storage>();
            push.sampler = sampler;
            push.size = EXTENT;
            cmd.bind_compute(generate);
            cmd.push(&push);
            cmd.dispatch(FilterGenerate::groups([EXTENT.x, EXTENT.y, 1]));
            cmd.images(&[img(input, 0..1, L::GENERAL, L::SHADER_READ_ONLY_OPTIMAL, raw::COMPUTE_WRITE, raw::COMPUTE_READ)]);
        });
        processing
    }
}

impl Scenario for ImageProcessing {
    fn lava(&mut self, ctx: &mut LavaCtx) {
        let exposure = self.lava.exposure();
        let s = &mut self.lava;
        s.frame += 1;
        let s = &self.lava;
        let slot = s.frame % IN_FLIGHT as usize;
        let sampler = self.sampler.handle();
        let check = ctx.check;
        ctx.submit(|cmd| {
            cmd.fill_buffer(s.bins.range(..), 0);
            let threads = [EXTENT.x, EXTENT.y, 1];
            let args = FilterBlurHArgs {
                src: s.input.whole_view(),
                dst: s.a.whole_view(),
                sampler,
                size: EXTENT,
                params: Vec4::new(1.0 / EXTENT.x as f32, 0.0, 8.0, 0.0),
            };
            cmd.compute(&FilterBlurH::push_bindings(args, None), FilterBlurH::groups(threads));
            // Ping-pong: each filter reads the target the last one wrote.
            let (mut src, mut dst) = (&s.a, &s.b);
            for (pass, params) in CHAIN {
                macro_rules! args {
                    ($args:ident) => {
                        $args { src: src.whole_view(), dst: dst.whole_view(), sampler, size: EXTENT, params }
                    };
                }
                match pass {
                    1 => cmd.compute(&FilterBlurV::push_bindings(args!(FilterBlurVArgs), None), FilterBlurV::groups(threads)),
                    2 => cmd.compute(&FilterGrade::push_bindings(args!(FilterGradeArgs), None), FilterGrade::groups(threads)),
                    3 => cmd.compute(&FilterSharpen::push_bindings(args!(FilterSharpenArgs), None), FilterSharpen::groups(threads)),
                    _ => cmd.compute(&FilterVignette::push_bindings(args!(FilterVignetteArgs), None), FilterVignette::groups(threads)),
                }
                if pass == 3 {
                    // The histogram of the sharpened picture: bands of rows whose atomics
                    // commute, through a pointer whose access says so.
                    let bins = [BufferAccess::write(s.bins.range(..)).unordered()];
                    for band in 0..BANDS {
                        let args = HistogramArgs {
                            src: dst.whole_view(),
                            bins: s.bins.address(),
                            size: EXTENT,
                            first_row: band * EXTENT.y / BANDS,
                        };
                        cmd.compute(
                            &Histogram::push_bindings(args, Some(Accesses::buffers(&bins))),
                            Histogram::groups([EXTENT.x, EXTENT.y / BANDS, 1]),
                        );
                    }
                }
                (src, dst) = (dst, src);
            }
            let args = FilterOutputArgs {
                src: src.whole_view(),
                dst: s.out.whole_view(),
                sampler,
                size: EXTENT,
                params: Vec4::new(exposure, 0.0, 0.0, 0.0),
            };
            cmd.compute(&FilterOutput::push_bindings(args, None), FilterOutput::groups(threads));
            cmd.copy_buffer(s.bins.range(..), s.readback[slot].range(..));
            if check {
                self.checksums.lava_clear(cmd);
                self.checksums.lava_image(cmd, s.out.whole_view(), 0);
                self.checksums.lava_buffer(cmd, s.bins.range(..), 1);
            }
        });
    }

    fn raw(&mut self, ctx: &mut RawCtx) {
        let exposure = self.raw.exposure();
        let s = &mut self.raw;
        s.frame += 1;
        let s = &self.raw;
        let slot = s.frame % IN_FLIGHT as usize;
        let sampler = self.sampler.handle();
        let pipelines = self.pipelines;
        let check = ctx.check;
        ctx.submit(&[], |cmd| {
            let [a, b] = [&s.a, &s.b].map(|i| i.whole_view().image());
            let out = s.out.whole_view().image();
            let compute_done = (vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::NONE);
            // Last frame's histogram copy read the bins; its passes read the targets.
            cmd.mem(raw::or(raw::TRANSFER_READ, raw::COMPUTE_RW), raw::TRANSFER_WRITE);
            cmd.fill(s.bins.address(), s.bins.size(), 0);
            cmd.barrier(
                &[raw::mem(raw::TRANSFER_WRITE, raw::COMPUTE_RW)],
                &[img(a, 0..1, L::UNDEFINED, L::GENERAL, compute_done, raw::COMPUTE_WRITE)],
            );
            let threads = [EXTENT.x, EXTENT.y, 1];
            // Every filter has the same push constants.
            let mut push: CFilterBlurHBindings = bytemuck::Zeroable::zeroed();
            push.sampler = sampler;
            push.size = EXTENT;
            let mut filter = |cmd: &mut RawCmd, pass: usize, src: ImageHandle<Sampled>, dst: ImageHandle<Storage>, params: Vec4| {
                push.src = src;
                push.dst = dst;
                push.params = params;
                cmd.bind_compute(pipelines[pass]);
                cmd.push(&push);
                cmd.dispatch(FilterBlurH::groups(threads));
            };
            let read = |image| img(image, 0..1, L::GENERAL, L::SHADER_READ_ONLY_OPTIMAL, raw::COMPUTE_WRITE, raw::COMPUTE_READ);
            let write = |image| img(image, 0..1, L::UNDEFINED, L::GENERAL, compute_done, raw::COMPUTE_WRITE);
            let blur_h = Vec4::new(1.0 / EXTENT.x as f32, 0.0, 8.0, 0.0);
            filter(cmd, 0, s.input.handle(), s.a.handle(), blur_h);
            cmd.images(&[read(a), write(b)]);
            filter(cmd, 1, s.a.handle(), s.b.handle(), CHAIN[0].1);
            cmd.images(&[read(b), write(a)]);
            filter(cmd, 2, s.b.handle(), s.a.handle(), CHAIN[1].1);
            cmd.images(&[read(a), write(b)]);
            filter(cmd, 3, s.a.handle(), s.b.handle(), CHAIN[2].1);
            cmd.images(&[read(b), write(a)]);
            // The histogram's bands need no barriers between them (their atomics commute), and
            // the vignette reads what they read.
            let mut histogram: CHistogramBindings = bytemuck::Zeroable::zeroed();
            histogram.src = s.b.handle::<Sampled>();
            histogram.bins = s.bins.address();
            histogram.size = EXTENT;
            cmd.bind_compute(pipelines[6]);
            for band in 0..BANDS {
                histogram.first_row = band * EXTENT.y / BANDS;
                cmd.push(&histogram);
                cmd.dispatch(Histogram::groups([EXTENT.x, EXTENT.y / BANDS, 1]));
            }
            filter(cmd, 4, s.b.handle(), s.a.handle(), CHAIN[3].1);
            cmd.barrier(&[raw::mem(raw::COMPUTE_WRITE, raw::TRANSFER_READ)], &[read(a), write(out)]);
            filter(cmd, 5, s.a.handle(), s.out.handle(), Vec4::new(exposure, 0.0, 0.0, 0.0));
            let readback = &s.readback[slot];
            cmd.copy(&[(s.bins.address(), readback.address(), s.bins.size())]);
            cmd.mem(raw::TRANSFER_WRITE, raw::HOST_READ);
            if check {
                cmd.images(&[img(out, 0..1, L::GENERAL, L::SHADER_READ_ONLY_OPTIMAL, raw::COMPUTE_WRITE, raw::COMPUTE_READ)]);
                self.checksums.raw_clear(cmd);
                self.checksums.raw_image(cmd, s.out.handle::<Sampled>(), EXTENT, 0);
                self.checksums.raw_buffer(cmd, s.bins.address(), 256, 1);
                self.checksums.raw_finish(cmd);
            }
        });
    }

    fn checksums(&mut self) -> (Vec<u32>, Vec<u32>) {
        self.checksums.read()
    }
}

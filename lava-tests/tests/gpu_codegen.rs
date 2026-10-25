//! GPU tests of what lava-build-slang generates and `bindless.slang` offers: struct layouts with gaps, packing, half and 16-bit fields, matrices and handles behind pointers (written through generated reference infos), read-only storage images and image handles read through a pointer, image operations (gathers, gradient and bias samples, mip loads, combined samples, atomics), workgroup sizes and `groups`, dispatch bases, required subgroup sizes, several compute entry points per file, and exported constants
mod common;

use common::{buffer_with, gpu, nearest, panic_message, zeroed_buffer};
use glam::{I64Vec3, Mat3, U64Vec2, UVec2, UVec3, Vec3, Vec4};
use lava::{
    Ctx, ImageView,
    bindless::{self, ImageHandle, ImageReference},
    buffer::{Buffer, Readback, Upload},
    command_buffer::{Attachment, CullModeFlags, Draw, RasterState},
    half::F16,
    image::{
        D3, Image,
        format::{R8G8B8A8Unorm, R32Sint, R32Uint},
        usage::usage,
    },
    pipeline::{PassKind, pass_map},
    reference::DynImage,
};
use lava_tests::bindings::*;

fn skipped(test: &str, why: &str) {
    eprintln!("SKIPPED {test}: {why}");
}

type Chain = Image<R8G8B8A8Unorm, usage![Sampled, TransferDst]>;

/// A 4x4 sampled image of 3 mips: mip 0's texel (i, j) is `[40 i, 40 j, 100 + i + 4 j, 200]`,
/// mip 1 red, mip 2 blue.
fn chain(gpu: &mut common::Gpu) -> Chain {
    let chain = Chain::with_mip_levels(UVec2::splat(4), 3)
        .unwrap()
        .bind_sampled()
        .unwrap();
    let mip0: Vec<[u8; 4]> = (0..16u8)
        .map(|t| {
            let (i, j) = (t % 4, t / 4);
            [40 * i, 40 * j, 100 + i + 4 * j, 200]
        })
        .collect();
    let (mip0, mip1, mip2) = (
        buffer_with(&mip0),
        buffer_with(&[[255u8, 0, 0, 255]; 4]),
        buffer_with(&[[0u8, 0, 255, 255]]),
    );
    gpu.submit(|cmd| {
        let view = chain.whole_view();
        cmd.copy_buffer_to_image(mip0.range(..), view.whole());
        cmd.copy_buffer_to_image(mip1.range(..), view.whole().mip(1));
        cmd.copy_buffer_to_image(mip2.range(..), view.whole().mip(2));
    });
    chain
}

fn bytes(v: Vec4) -> [u8; 4] {
    (v * 255.0).round().to_array().map(|c| c as u8)
}

#[test]
fn images_gather_sample_with_gradients_load_mips_and_run_atomics() {
    let mut gpu = gpu();
    let chain = chain(&mut gpu);
    let combined = Image::<R8G8B8A8Unorm, usage![Sampled, Storage, TransferDst]>::new(UVec2::ONE)
        .unwrap()
        .bind_combined()
        .unwrap();
    let counters = Image::<R32Uint, usage![Storage, TransferDst]>::new(UVec2::ONE)
        .unwrap()
        .bind_storage()
        .unwrap();
    let signed = Image::<R32Sint, usage![Storage, TransferDst]>::new(UVec2::ONE)
        .unwrap()
        .bind_storage()
        .unwrap();
    let volume =
        Image::<R8G8B8A8Unorm, usage![Sampled, Storage, TransferSrc, TransferDst], D3>::new(
            UVec3::splat(2),
        )
        .unwrap()
        .bind_combined()
        .unwrap();
    let mut first = vec![[0u8; 4]; 8];
    first[0] = [10, 20, 30, 40];
    let first = buffer_with(&first);
    let mut voxels = zeroed_buffer::<[u8; 4]>(8);
    let mut out = zeroed_buffer::<Vec4>(8);
    let mut atomics = zeroed_buffer::<i32>(14);
    gpu.submit(|cmd| {
        cmd.copy_buffer_to_image(first.range(..), volume.whole_view().whole());
        cmd.clear_image(combined.whole_view(), [0.0, 1.0, 0.0, 1.0]);
        cmd.clear_image(counters.whole_view(), [0]);
        cmd.clear_image(signed.whole_view(), [0]);
        let args = TestImageOpsArgs {
            chain: chain.whole_view(),
            combined: combined.whole_view(),
            counters: counters.whole_view(),
            volume: volume.whole_view(),
            signed: signed.whole_view(),
            nearest: nearest(),
            out: out.range(..),
            atomics: atomics.range(..),
        };
        cmd.compute(&TestImageOps::push_bindings(args, None), [1, 1, 1]);
        cmd.copy_image_to_buffer(volume.whole_view().whole(), voxels.range(..));
    });
    assert_eq!(
        voxels.host()[7],
        [10, 20, 30, 40],
        "a `MutTex3D` samples and stores"
    );
    let out = out.host();
    // Gathers name the texels (0, 1), (1, 1), (1, 0), (0, 0).
    let gathered = |v: Vec4| v.to_array().map(|c| (c * 255.0).round() as u8);
    assert_eq!(gathered(out[0]), [0, 40, 40, 0], "Gather is red");
    assert_eq!(gathered(out[1]), [0, 40, 40, 0], "GatherRed");
    assert_eq!(gathered(out[2]), [40, 40, 0, 0], "GatherGreen");
    assert_eq!(gathered(out[3]), [104, 105, 101, 100], "GatherBlue");
    assert_eq!(gathered(out[4]), [200; 4], "GatherAlpha");
    assert_eq!(
        bytes(out[5]),
        [0, 0, 255, 255],
        "SampleGrad of the whole image picks mip 2"
    );
    assert_eq!(bytes(out[6]), [255, 0, 0, 255], "Load at mip 1");
    assert_eq!(
        bytes(out[7]),
        [0, 255, 0, 255],
        "a combined image's SampleLevel"
    );
    assert_eq!(
        atomics.host(),
        [0, 5, 9, 7, 15, 6, 5, 42, 100, 100, 0, -4, -4, -5],
        "add, max, min, or, and, xor, exchange, compare-exchange twice, load; signed min, max, add, load"
    );
}

#[test]
fn fragment_shaders_sample_with_a_bias_and_combined_images_sample() {
    let mut gpu = gpu();
    let chain = chain(&mut gpu);
    let combined = Image::<R8G8B8A8Unorm, usage![Sampled, Storage, TransferDst]>::new(UVec2::ONE)
        .unwrap()
        .bind_combined()
        .unwrap();
    let target =
        Image::<R8G8B8A8Unorm, usage![ColorAttachment, TransferSrc]>::new(UVec2::splat(4)).unwrap();
    let mut pixels = zeroed_buffer::<[u8; 4]>(16);
    // A pixel per texel is mip 0; a bias of 1 picks mip 1, one of 10 the last, mip 2.
    for (bias, expected) in [
        (0.0, None),
        (1.0, Some([255, 0, 0, 255])),
        (10.0, Some([0, 0, 255, 255])),
        (-1.0, Some([0, 255, 0, 255])),
    ] {
        gpu.submit(|cmd| {
            cmd.clear_image(combined.whole_view(), [0.0, 1.0, 0.0, 1.0]);
            let args = TestFragmentSamplesArgs {
                chain: chain.whole_view(),
                combined: combined.whole_view(),
                sampler: nearest(),
                bias,
            };
            let bindings = TestFragmentSamples::push_bindings(args, None);
            let state = RasterState::default().cull_mode(CullModeFlags::NONE);
            cmd.raster(
                UVec2::splat(4),
                [Attachment::color(target.whole_view(), None)],
            )
            .draw(&Draw::new(&bindings, 3, 1, state))
            .record("samples");
            cmd.copy_image_to_buffer(target.whole_view().whole(), pixels.range(..));
        });
        let host = pixels.host();
        match expected {
            Some(color) => assert!(host.iter().all(|p| *p == color), "bias {bias}: {host:?}"),
            None => assert_eq!(host[5], [40, 40, 105, 200], "bias 0 reads mip 0"),
        }
    }
}

#[test]
fn several_compute_entry_points_are_passes_with_their_workgroups_and_bases() {
    let mut gpu = gpu();
    assert_eq!(TestMultiFill::WORKGROUP_SIZE, [8, 1, 1]);
    assert_eq!(TestMultiDouble::WORKGROUP_SIZE, [4, 2, 1]);
    assert_eq!(TestMultiFill::groups([20, 1, 1]), [3, 1, 1]);
    assert_eq!(TestMultiDouble::groups([8, 3, 1]), [2, 2, 1]);
    assert_eq!(TestComputeBuffer::groups([0, 1, 1]), [0, 1, 1]);
    let entries: Vec<_> = (pass_map().passes.iter())
        .filter(|p| ["TestMultiFill", "TestMultiDouble"].contains(&p.name))
        .map(|p| (p.name, p.kind))
        .collect();
    assert_eq!(
        entries,
        [
            (
                "TestMultiFill",
                PassKind::Compute {
                    entry: c"fill",
                    subgroup_size: None
                }
            ),
            (
                "TestMultiDouble",
                PassKind::Compute {
                    entry: c"double",
                    subgroup_size: None
                }
            ),
        ],
        "one pass per entry point, in declaration order"
    );
    let mut out = zeroed_buffer::<u32>(32);
    gpu.submit(|cmd| {
        let fill = TestMultiFill::push_bindings(
            TestMultiFillArgs {
                out: out.range(..),
                count: 32,
            },
            None,
        );
        // Workgroup 2 alone: threads 16 to 23.
        cmd.compute_base(&fill, [2, 0, 0], [1, 1, 1]);
        let double = TestMultiDouble::push_bindings(
            TestMultiDoubleArgs {
                out: out.range(..),
                count: 32,
            },
            None,
        );
        cmd.compute(&double, TestMultiDouble::groups([8, 4, 1]));
    });
    let expected: Vec<u32> = (0..32)
        .map(|i| {
            if (16..24).contains(&i) {
                2 * (i + MULTI_FILL_ADD)
            } else {
                0
            }
        })
        .collect();
    assert_eq!(out.host(), expected);
    let max = Ctx::device_properties().limits.max_compute_work_group_count;
    let message = panic_message(|| {
        gpu.submit(|cmd| {
            let fill = TestMultiFill::push_bindings(
                TestMultiFillArgs {
                    out: out.range(..),
                    count: 32,
                },
                None,
            );
            cmd.compute_base(&fill, [max[0], 0, 0], [1, 1, 1]);
        })
    });
    assert!(message.contains("exceed the device's"), "{message}");
}

#[test]
fn rust_consts_are_exported() {
    assert_eq!(MULTI_FILL_ADD, 3u32);
    assert_eq!(MULTI_NEGATIVE, -2i32);
    assert_eq!(MULTI_HALF, 0.5f32);
    const { assert!(MULTI_YES) };
    assert_eq!(MULTI_BIG, 1u64 << 32);
}

#[test]
fn wave_size_requires_the_subgroup_size() {
    let mut gpu = gpu();
    let supported = Ctx::subgroup_sizes().is_some_and(|s| {
        s.stages.contains(lava::vk::ShaderStageFlags::COMPUTE) && (s.min..=s.max).contains(&32)
    });
    let pass = pass_map()
        .passes
        .iter()
        .find(|p| p.name == "TestWaveSize")
        .unwrap();
    assert_eq!(
        pass.kind,
        PassKind::Compute {
            entry: c"test_wave_size",
            subgroup_size: Some(32)
        }
    );
    let mut out = zeroed_buffer::<u32>(64);
    if !supported {
        let message = panic_message(|| {
            gpu.submit(|cmd| {
                let bindings =
                    TestWaveSize::push_bindings(TestWaveSizeArgs { out: out.range(..) }, None);
                cmd.compute(&bindings, [1, 1, 1]);
            })
        });
        assert!(message.contains("subgroup size"), "{message}");
        return skipped(
            "wave_size_requires_the_subgroup_size",
            "the device can't require 32",
        );
    }
    gpu.submit(|cmd| {
        let bindings = TestWaveSize::push_bindings(TestWaveSizeArgs { out: out.range(..) }, None);
        cmd.compute(&bindings, TestWaveSize::groups([64, 1, 1]));
    });
    assert_eq!(out.host(), [32; 64]);
}

type Texture = Image<R8G8B8A8Unorm, usage![Sampled]>;

/// The input `Layouts`, reading `tex`.
fn layouts(
    tex: &Texture,
) -> LayoutsReferenceInfo<'_, ImageView<'_, R8G8B8A8Unorm, usage![Sampled]>> {
    let mut arr = [0.0; 40];
    arr.iter_mut()
        .enumerate()
        .for_each(|(i, a)| *a = 100.0 + i as f32);
    let mut m34 = [[0.0; 3]; 4];
    m34.iter_mut()
        .flatten()
        .enumerate()
        .for_each(|(i, m)| *m = i as f32);
    LayoutsReferenceInfo {
        a: 1,
        b: 1 << 40,
        h: F16::new(1.5),
        after_half: 2.0,
        s: -3,
        u: 250,
        l3: I64Vec3::new(-1, 1 << 33, 7),
        u2: U64Vec2::new(3, 1 << 34),
        flags: UVec3::new(1, 0, 1),
        m33: Mat3::from_cols_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
        m34,
        arr,
        tex: tex.whole_view(),
        r#dyn: DynImage::NULL,
        r#type: 5,
        inner: Inner {
            wide: u64::MAX - 1,
            narrow: 9,
        },
        tail_vec: [1.0, 2.0, 3.0, 4.0],
        tail: 77,
    }
}

#[test]
fn generated_structs_keep_slangs_layout() {
    let mut gpu = gpu();
    // Gaps are zeroed pads, `Default` is all zeroes (null handles), and a struct Slang doesn't
    // round up is packed.
    assert_eq!(size_of::<Layouts>(), 376);
    assert_eq!(size_of::<LayoutsOut>(), 376);
    assert_eq!(size_of::<Inner>(), 12);
    let default = Layouts::default();
    assert_eq!(
        (default.tex, default.r#dyn),
        (ImageHandle::NULL, ImageHandle::NULL)
    );
    let out = LayoutsOut::default();
    assert!(bytemuck::bytes_of(&out).iter().all(|b| *b == 0));

    let first = Texture::new(UVec2::ONE).unwrap().bind_sampled().unwrap();
    let second = Texture::new(UVec2::ONE).unwrap().bind_sampled().unwrap();
    let handles = [
        first.handle::<bindless::Sampled>(),
        second.handle::<bindless::Sampled>(),
    ];
    // Referencing structs are written with their references: views, slices, `DynImage`s.
    let mut input: Buffer<Layouts, Upload> = Buffer::upload(1).unwrap();
    input.host_write_referencing(0, [layouts(&first)]);
    let mut handle_buffer: Buffer<ImageHandle<bindless::Sampled>, Upload> =
        Buffer::upload(2).unwrap();
    handle_buffer.host_write_referencing(
        0,
        [&first, &second].map(|t| ImageReference::new(t.whole_view())),
    );
    let points = buffer_with(&[
        Vec3::new(1.0, 10.0, 100.0),
        Vec3::new(2.0, 20.0, 200.0),
        Vec3::new(3.0, 30.0, 300.0),
    ]);
    let mut output: Buffer<LayoutsOut, Readback> = Buffer::readback(1).unwrap();
    output.host_write(0, &[LayoutsOut::default()]);
    let mut sums = zeroed_buffer::<f32>(1);
    let rotation = Mat3::from_cols(
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(5.0, 3.0, 0.0),
        Vec3::new(0.0, 0.0, 4.0),
    );
    let column = |c: Vec3| [c.x, c.y, c.z, 0.0];
    let args = TestLayoutsArgs {
        input: input.range(..),
        output: output.range(..),
        handles: handle_buffer.range(..),
        points: points.range(..),
        sums: sums.range(..),
        rotation: [
            column(rotation.x_axis),
            column(rotation.y_axis),
            column(rotation.z_axis),
        ],
        scale: F16::new(2.0),
    };
    let Ok(bindings) = TestLayouts::push_bindings(args, None) else {
        return skipped(
            "generated_structs_keep_slangs_layout",
            "no 8- or 16-bit storage or arithmetic",
        );
    };
    gpu.submit(|cmd| cmd.compute(&bindings, [1, 1, 1]));
    let before = layouts(&first);
    let after = output.host()[0];
    assert_eq!((after.a, after.b), (2, (1 << 40) + 1));
    assert_eq!((after.h.to_f32(), after.after_half), (3.0, 3.0));
    assert_eq!((after.s, after.u), (-4, 251));
    assert_eq!(after.l3, before.l3 + I64Vec3::new(1, 2, 3));
    assert_eq!(after.u2, before.u2 * 2);
    assert_eq!(after.flags, UVec3::new(0, 1, 0));
    // The push constants' std430 `float3x3` (columns of four) times the buffer's scalar one.
    assert_eq!(after.m33, rotation * before.m33);
    let mut m34 = before.m34;
    m34[3][2] += 1.0;
    assert_eq!(after.m34, m34, "`m[2][3]` is row 2 of column 3");
    assert!(
        after
            .arr
            .iter()
            .enumerate()
            .all(|(i, a)| *a == 100.0 + 2.0 * i as f32)
    );
    assert_eq!(
        (after.tex, after.r#dyn),
        (handles[1].bits(), 0),
        "a handle read through a `Buf<Tex2D>`, and a null `DynImg`"
    );
    assert_eq!(after.r#type, 15);
    let inner = after.inner;
    assert_eq!((inner.wide, inner.narrow), (u64::MAX, 10));
    assert_eq!((after.tail_vec, after.tail), ([1.0, 2.0, 3.0, 5.0], 78));
    assert_eq!(
        sums.host()[0],
        1.0 + 20.0 + 300.0,
        "`float3`s are 12 bytes apart behind a pointer"
    );
}

/// `Image2D` reads through the storage descriptor; `Image2D` and `Tex2D` handles inside a
/// struct behind a `Buf` read through theirs, reached through the buffer.
#[test]
fn storage_images_and_handles_behind_pointers_are_read() {
    let mut gpu = gpu();
    type Texture = Image<R8G8B8A8Unorm, usage![Sampled, Storage, TransferDst]>;
    let texels = |base: u8| -> Vec<[u8; 4]> {
        (0..4u8)
            .map(|i| [base + i, base + 10 * i, 255 - i, 255])
            .collect()
    };
    let make = |gpu: &mut common::Gpu, base: u8| {
        let image = Texture::new(UVec2::splat(2))
            .unwrap()
            .bind_combined()
            .unwrap();
        let data = buffer_with(&texels(base));
        gpu.submit(|cmd| cmd.copy_buffer_to_image(data.range(..), image.whole()));
        image
    };
    let (top, storage, sampled) = (make(&mut gpu, 10), make(&mut gpu, 50), make(&mut gpu, 90));
    let mut images: Buffer<TestImagePair, Upload> = Buffer::upload(1).unwrap();
    images.host_write_referencing(
        0,
        [TestImagePairReferenceInfo {
            storage: storage.whole_view(),
            sampled: sampled.whole_view(),
        }],
    );
    let mut out = zeroed_buffer::<Vec4>(12);
    gpu.submit(|cmd| {
        let bindings = TestImage2d::push_bindings(
            TestImage2dArgs {
                image: top.whole_view(),
                images: images.range(..),
                sampler: nearest(),
                out: out.range(..),
            },
            None,
        );
        cmd.compute(&bindings, [1, 1, 1]);
    });
    let read: Vec<[u8; 4]> = out.host().iter().map(|v| bytes(*v)).collect();
    assert_eq!(read[0..4], texels(10), "the push constant's `Image2D`");
    assert_eq!(read[4..8], texels(50), "an `Image2D` behind a pointer");
    assert_eq!(read[8..12], texels(90), "a `Tex2D` behind a pointer");
}

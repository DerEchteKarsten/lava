//! GPU tests of images and formats: what the device can't make is refused, texel blocks of compressed formats, 64-bit clears, clears of mips, layer ranges and integer formats, the checks of blits (linear filtering among them), clears and copies, runtime-typed images (`format::Dynamic`, `usage::Dynamic`: their checks where commands and bindings use them, `erase` and `cast`), the layouts host copies transition, depth and stencil clears and copies, image-to-image copies, row pitches, mips of single-view images, reinterpreting views, `Image::fit`, sparse residency and external memory
mod common;

use common::{Cmd, buffer_with, gpu, nearest, panic_message, zeroed_buffer};
use glam::{IVec2, UVec2, UVec3, Vec2, Vec4};
use lava::{
    Ctx, Error,
    bindless::{self, SamplerHandle},
    command_buffer::{Accesses, Attachment, ImageUse},
    image::{
        D3, Filter, Image, MipChain,
        format::{
            self, D32SfloatS8Uint, Format, R8G8B8A8Srgb, R8G8B8A8Uint, R8G8B8A8Unorm,
            R16G16B16A16Unorm, R32G32B32A32Sfloat, R32Sfloat, R32Sint, R32Uint, R64Uint,
        },
        usage::{self, usage},
    },
    vk,
};
use lava_tests::bindings::*;

const NO_SAMPLER: SamplerHandle = SamplerHandle { index: !0 };

/// What `TestSample` reads at the centre of the sampled image whose handle has `bits`.
fn sample(gpu: &mut common::Gpu, bits: u32, reads: &[ImageUse<'_>]) -> Vec4 {
    let mut out = zeroed_buffer::<Vec4>(3);
    gpu.submit(|cmd| {
        let bindings = TestSample::push_bindings(
            TestSampleArgs {
                out: out.range(..),
                first: nearest(),
                second: nearest(),
                compare: NO_SAMPLER,
                slot: bits,
                uv: Vec2::splat(0.5),
                reference: 0.0,
            },
            Some(Accesses::images(reads)),
        );
        cmd.compute(&bindings, [1, 1, 1]);
    });
    out.host()[0]
}

fn has(format: vk::Format, feature: vk::FormatFeatureFlags2) -> bool {
    Ctx::format_features(format).contains(feature)
}

fn skipped(test: &str, why: &str) {
    eprintln!("SKIPPED {test}: {why}");
}

// ---- what the device can't make ----------------------------------------------------------

#[test]
fn descriptions_the_device_lacks_are_unsupported() {
    let _gpu = gpu();
    // Intel has no sRGB storage images, and no colour format is a depth attachment.
    if !has(
        vk::Format::R8G8B8A8_SRGB,
        vk::FormatFeatureFlags2::STORAGE_IMAGE,
    ) {
        let srgb = Image::<R8G8B8A8Srgb, usage![Storage]>::new(UVec2::splat(4));
        assert!(matches!(srgb, Err(Error::Unsupported(_))), "{srgb:?}");
    }
    let depth = Image::<R8G8B8A8Unorm, usage![DepthAttachment]>::new(UVec2::splat(4));
    assert!(matches!(depth, Err(Error::Unsupported(_))), "{depth:?}");
    // A host copy needs the format's HOST_IMAGE_TRANSFER.
    if Ctx::features().host_image_copy {
        let host = Image::<R8G8B8A8Unorm, usage![Sampled, HostCopy]>::new(UVec2::splat(4));
        let feature = has(
            vk::Format::R8G8B8A8_UNORM,
            vk::FormatFeatureFlags2::HOST_IMAGE_TRANSFER_EXT,
        );
        assert_eq!(host.is_ok(), feature, "{host:?}");
    }
    // Attachments with host copies don't compile; a dynamic usage panics.
    let both = usage::Dynamic(
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::HOST_TRANSFER_EXT,
    );
    let message = panic_message(|| {
        Image::<R8G8B8A8Unorm, _>::with_params(R8G8B8A8Unorm, both, UVec2::ONE, 1)
    });
    assert!(message.contains("attachment and host copy"), "{message}");
}

#[test]
fn compressed_formats_need_their_features_and_whole_blocks() {
    let mut gpu = gpu();
    #[allow(unused)]
    use format::{ASTC5X4UnormBlock, EACR11G11UnormBlock, ETC2R8G8B8A8UnormBlock};
    // ASTC LDR: the Arc B390 has none.
    let astc = Image::<ASTC5X4UnormBlock, usage![Sampled, TransferDst]>::new(UVec2::new(10, 8));
    if !Ctx::features().astc_compression {
        assert!(matches!(astc, Err(Error::MissingFeature(_))), "{astc:?}");
    }
    if !Ctx::features().etc2_compression {
        skipped(
            "compressed_formats_need_their_features_and_whole_blocks",
            "no ETC2",
        );
        return;
    }
    let etc = Image::<ETC2R8G8B8A8UnormBlock, usage![Sampled, TransferDst]>::new(UVec2::new(8, 8))
        .unwrap();
    // 8x8 texels are 2x2 blocks of 16 bytes.
    let buffer = zeroed_buffer::<u8>(64);
    let message = panic_message(|| {
        gpu.submit(|cmd| {
            cmd.copy_buffer_to_image(buffer.range(..), etc.whole_view().region(UVec2::new(2, 4)))
        })
    });
    assert!(message.contains("whole 4x4 blocks"), "{message}");
    let message = panic_message(|| {
        gpu.submit(|cmd| cmd.copy_buffer_to_image(buffer.range(..32), etc.whole()))
    });
    assert!(message.contains("takes 64"), "{message}");
    // A box of a larger image: rows of whole blocks.
    let message = panic_message(|| {
        gpu.submit(|cmd| cmd.copy_buffer_to_image(buffer.range(..).rows(10, 0), etc.whole()))
    });
    assert!(message.contains("whole 4x4 blocks"), "{message}");
    gpu.submit(|cmd| cmd.copy_buffer_to_image(buffer.range(..), etc.whole()));
    // EAC RG is 16 bytes a block too.
    let eac =
        Image::<EACR11G11UnormBlock, usage![Sampled, TransferDst]>::new(UVec2::new(4, 4)).unwrap();
    gpu.submit(|cmd| cmd.copy_buffer_to_image(buffer.range(..16), eac.whole()));
    let message = panic_message(|| gpu.submit(|cmd| cmd.clear_image(eac.whole_view(), [0.0; 2])));
    assert!(message.contains("block-compressed"), "{message}");
}

#[test]
fn host_copies_check_blocks_and_rows() {
    let _gpu = gpu();
    if !Ctx::features().host_image_copy || !Ctx::features().etc2_compression {
        skipped(
            "host_copies_check_blocks_and_rows",
            "no host image copies or ETC2",
        );
        return;
    }
    use format::ETC2R8G8B8UnormBlock;
    let format = vk::Format::ETC2_R8G8B8_UNORM_BLOCK;
    if !has(format, vk::FormatFeatureFlags2::HOST_IMAGE_TRANSFER_EXT) {
        skipped(
            "host_copies_check_blocks_and_rows",
            "no host copies of ETC2",
        );
        return;
    }
    let mut etc =
        Image::<ETC2R8G8B8UnormBlock, usage![Sampled, HostCopy]>::new(UVec2::new(8, 8)).unwrap();
    // 2x2 blocks of 8 bytes; a byte short or over panics instead of reading past the data.
    etc.copy_from(&[0; 32], 0).unwrap();
    let message = panic_message(|| etc.copy_from(&[0; 1], 0));
    assert!(message.contains("holds 32 bytes"), "{message}");
    let message =
        panic_message(|| etc.copy_region_from(&[0; 8], 0, IVec2::new(2, 0), UVec2::new(4, 4)));
    assert!(message.contains("whole 4x4 blocks"), "{message}");
    etc.copy_region_from(&[0; 8], 0, IVec2::new(4, 4), UVec2::new(4, 4))
        .unwrap();
    // One block of a row of two.
    etc.copy_rows_from(&[0; 16], 0, IVec2::new(4, 0), UVec2::new(4, 4), 8, 0)
        .unwrap();
    let message =
        panic_message(|| etc.copy_rows_from(&[0; 8], 0, IVec2::ZERO, UVec2::new(4, 8), 0, 0));
    assert!(message.contains("16 bytes"), "{message}");
}

// ---- clears, blits and copies -------------------------------------------------------------

#[test]
fn sixty_four_bit_clears_take_two_words_per_component() {
    let mut gpu = gpu();
    let format = vk::Format::R64_UINT;
    if !has(
        format,
        vk::FormatFeatureFlags2::TRANSFER_DST | vk::FormatFeatureFlags2::TRANSFER_SRC,
    ) {
        skipped(
            "sixty_four_bit_clears_take_two_words_per_component",
            "no R64Uint transfers",
        );
        return;
    }
    let Ok(image) = Image::<R64Uint, usage![TransferSrc, TransferDst]>::new(UVec2::splat(2)) else {
        skipped(
            "sixty_four_bit_clears_take_two_words_per_component",
            "no R64Uint images",
        );
        return;
    };
    let mut out = zeroed_buffer::<u64>(4);
    let value = 0x1234_5678_9abc_def0;
    gpu.submit(|cmd| {
        cmd.clear_image(image.whole_view(), [value]);
        cmd.copy_image_to_buffer(image.whole(), out.range(..));
    });
    assert_eq!(out.host(), [value; 4]);
}

#[test]
fn blits_check_their_formats_and_boxes() {
    let mut gpu = gpu();
    let size = UVec2::splat(8);
    type Floats = Image<R8G8B8A8Unorm, usage![TransferSrc, TransferDst]>;
    let a = Floats::new(size).unwrap();
    let uint = Image::<R32Uint, usage![TransferSrc, TransferDst]>::new(size).unwrap();
    let sint = Image::<R32Sint, usage![TransferSrc, TransferDst]>::new(size).unwrap();
    let message = panic_message(|| {
        gpu.submit(|cmd| cmd.blit_image(uint.whole(), uint.whole(), Filter::Linear))
    });
    assert!(message.contains("integer"), "{message}");
    let message = panic_message(|| {
        gpu.submit(|cmd| cmd.blit_image(uint.whole(), sint.whole(), Filter::Nearest))
    });
    assert!(message.contains("numeric types"), "{message}");
    let message = panic_message(|| {
        gpu.submit(|cmd| cmd.blit_image(uint.whole(), a.whole(), Filter::Nearest))
    });
    assert!(message.contains("numeric types"), "{message}");
    let overlapping = a
        .whole_view()
        .region(UVec2::splat(4))
        .offset(IVec2::splat(2));
    let message = panic_message(|| {
        gpu.submit(|cmd| {
            cmd.blit_image(
                a.whole_view().region(UVec2::splat(4)),
                overlapping,
                Filter::Nearest,
            )
        })
    });
    assert!(message.contains("overlap"), "{message}");
    // Apart, one image blits into itself, and integers blit into integers of their sign.
    let apart = a
        .whole_view()
        .region(UVec2::splat(4))
        .offset(IVec2::splat(4));
    gpu.submit(|cmd| {
        cmd.clear_image(a.whole_view(), [0.0; 4]);
        cmd.clear_image(uint.whole_view(), [0]);
        cmd.blit_image(
            a.whole_view().region(UVec2::splat(4)),
            apart,
            Filter::Linear,
        );
        cmd.blit_image(
            uint.whole_view().region(UVec2::splat(4)),
            uint.whole_view()
                .region(UVec2::splat(4))
                .offset(IVec2::splat(4)),
            Filter::Nearest,
        );
    });
}

#[test]
fn copies_take_addresses_of_whole_texel_blocks() {
    let mut gpu = gpu();
    let image = Image::<R32G32B32A32Sfloat, usage![TransferSrc, TransferDst]>::new(UVec2::splat(2))
        .unwrap();
    let buffer = zeroed_buffer::<u8>(5 * 16);
    let message = panic_message(|| {
        gpu.submit(|cmd| cmd.copy_image_to_buffer(image.whole(), buffer.byte_range(4..68)))
    });
    assert!(message.contains("multiples of 16"), "{message}");
    gpu.submit(|cmd| cmd.copy_image_to_buffer(image.whole(), buffer.byte_range(16..)));
}

#[test]
fn images_copy_into_images_of_their_block() {
    let mut gpu = gpu();
    let size = UVec2::splat(8);
    let texels: Vec<u8> = (0..8 * 8 * 4).map(|i| i as u8).collect();
    let upload = buffer_with::<u8>(&texels);
    type Bytes = Image<R8G8B8A8Unorm, usage![TransferSrc, TransferDst]>;
    let (a, b) = (Bytes::new(size).unwrap(), Bytes::new(size).unwrap());
    // R32Uint has 4-byte texels too: the bits move unchanged.
    let words = Image::<R32Uint, usage![TransferSrc, TransferDst]>::new(size).unwrap();
    let wide = Image::<R16G16B16A16Unorm, usage![TransferSrc, TransferDst]>::new(size).unwrap();
    let mut out = zeroed_buffer::<u8>(texels.len());
    let mut word_out = zeroed_buffer::<u8>(texels.len());
    gpu.submit(|cmd| {
        cmd.copy_buffer_to_image(upload.range(..), a.whole());
        cmd.clear_image(b.whole_view(), [0.0; 4]);
        // The top left quarter of `a` into the bottom right of `b`.
        cmd.copy_image(
            a.whole_view().region(UVec2::splat(4)),
            b.whole_view()
                .region(UVec2::splat(4))
                .offset(IVec2::splat(4)),
        );
        cmd.copy_image(a.whole(), words.whole());
        cmd.copy_image_to_buffer(b.whole(), out.range(..));
        cmd.copy_image_to_buffer(words.whole(), word_out.range(..));
    });
    for (y, x) in (0..8).flat_map(|y| (0..8).map(move |x| (y, x))) {
        let texel = |data: &[u8], x: usize, y: usize| data[(y * 8 + x) * 4..][..4].to_vec();
        let expected = match x >= 4 && y >= 4 {
            true => texel(&texels, x - 4, y - 4),
            false => vec![0; 4],
        };
        assert_eq!(texel(out.host(), x, y), expected, "({x}, {y})");
    }
    assert_eq!(word_out.host(), texels);
    let message = panic_message(|| gpu.submit(|cmd| cmd.copy_image(a.whole(), wide.whole())));
    assert!(message.contains("different texel blocks"), "{message}");
    let message = panic_message(|| {
        gpu.submit(|cmd| cmd.copy_image(a.whole(), b.whole_view().region(UVec2::splat(4))))
    });
    assert!(message.contains("one extent"), "{message}");
    let message = panic_message(|| {
        gpu.submit(|cmd| {
            cmd.copy_image(
                a.whole_view().region(UVec2::splat(4)),
                a.whole_view()
                    .region(UVec2::splat(4))
                    .offset(IVec2::splat(2)),
            )
        })
    });
    assert!(message.contains("overlap"), "{message}");
    // Volumes copy too.
    type Volume = Image<R8G8B8A8Unorm, usage![TransferSrc, TransferDst], D3>;
    let (c, d) = (
        Volume::new(UVec3::splat(2)).unwrap(),
        Volume::new(UVec3::splat(2)).unwrap(),
    );
    let mut volume_out = zeroed_buffer::<u8>(32);
    gpu.submit(|cmd| {
        cmd.copy_buffer_to_image(upload.range(..32), c.whole());
        cmd.copy_image(c.whole(), d.whole());
        cmd.copy_image_to_buffer(d.whole(), volume_out.range(..));
    });
    assert_eq!(volume_out.host(), &texels[..32]);
}

#[test]
fn depth_and_stencil_clear_and_copy_one_aspect_at_a_time() {
    let mut gpu = gpu();
    let format = vk::Format::D32_SFLOAT_S8_UINT;
    let both = vk::FormatFeatureFlags2::TRANSFER_SRC | vk::FormatFeatureFlags2::TRANSFER_DST;
    if !has(format, both) {
        skipped(
            "depth_and_stencil_clear_and_copy_one_aspect_at_a_time",
            "no D32S8 transfers",
        );
        return;
    }
    let image = Image::<D32SfloatS8Uint, usage![DepthAttachment, TransferSrc, TransferDst]>::new(
        UVec2::splat(4),
    )
    .unwrap();
    let mut depth = zeroed_buffer::<f32>(16);
    let mut stencil = zeroed_buffer::<u8>(16);
    gpu.submit(|cmd: &mut Cmd| {
        cmd.clear_depth_stencil(image.whole_view(), (0.25, 7));
        cmd.copy_image_to_buffer(image.whole(), depth.range(..));
        cmd.copy_image_to_buffer(image.whole().stencil(), stencil.range(..));
    });
    assert_eq!(depth.host(), [0.25; 16]);
    assert_eq!(stencil.host(), [7; 16]);
    // Into the stencil aspect alone: the depth stays.
    let marks = buffer_with::<u8>(&[3; 16]);
    gpu.submit(|cmd| {
        cmd.copy_buffer_to_image(marks.range(..), image.whole().stencil());
        cmd.copy_image_to_buffer(image.whole(), depth.range(..));
        cmd.copy_image_to_buffer(image.whole().stencil(), stencil.range(..));
    });
    assert_eq!(depth.host(), [0.25; 16]);
    assert_eq!(stencil.host(), [3; 16]);
    let message =
        panic_message(|| gpu.submit(|cmd| cmd.clear_depth_stencil(image.whole_view(), (2.0, 0))));
    assert!(message.contains("0 to 1"), "{message}");
    // A depth image copies into a depth image of its format, every aspect.
    let copy = Image::<D32SfloatS8Uint, usage![DepthAttachment, TransferSrc, TransferDst]>::new(
        UVec2::splat(4),
    )
    .unwrap();
    gpu.submit(|cmd| {
        cmd.copy_image(image.whole(), copy.whole());
        cmd.copy_image_to_buffer(copy.whole().stencil(), stencil.range(..));
    });
    assert_eq!(stencil.host(), [3; 16]);
}

#[test]
fn rows_of_a_larger_image_copy_a_box_of_it() {
    let mut gpu = gpu();
    // An 8x4 image of texels numbered in the buffer; the 2x2 box at (3, 1) of it.
    let texels: Vec<u32> = (0..32).collect();
    let upload = buffer_with::<u32>(&texels);
    let image = Image::<R32Uint, usage![TransferSrc, TransferDst]>::new(UVec2::splat(2)).unwrap();
    let mut out = zeroed_buffer::<u32>(2 * 8);
    gpu.submit(|cmd| {
        cmd.copy_buffer_to_image(upload.range(8 + 3..).rows(8, 0), image.whole());
        // Back into rows 8 texels apart.
        cmd.fill_buffer(out.byte_range(..), 0);
        cmd.copy_image_to_buffer(image.whole(), out.range(..).rows(8, 0));
    });
    assert_eq!(&out.host()[..2], [11, 12]);
    assert_eq!(&out.host()[8..10], [19, 20]);
    let message = panic_message(|| {
        gpu.submit(|cmd| cmd.copy_image_to_buffer(image.whole(), out.range(..9).rows(8, 0)))
    });
    assert!(message.contains("takes 40"), "{message}");
    let message = panic_message(|| {
        gpu.submit(|cmd| cmd.copy_image_to_buffer(image.whole(), out.range(..).rows(1, 0)))
    });
    assert!(message.contains("narrower"), "{message}");
    if Ctx::features().host_image_copy
        && has(
            vk::Format::R32_UINT,
            vk::FormatFeatureFlags2::HOST_IMAGE_TRANSFER_EXT,
        )
    {
        let mut host =
            Image::<R32Uint, usage![TransferSrc, HostCopy]>::new(UVec2::splat(2)).unwrap();
        let bytes: &[u8] = bytemuck::cast_slice(&texels[8 + 3..]);
        host.copy_rows_from(bytes, 0, IVec2::ZERO, UVec2::splat(2), 8, 0)
            .unwrap();
        let mut back = zeroed_buffer::<u32>(4);
        gpu.submit(|cmd| cmd.copy_image_to_buffer(host.whole(), back.range(..)));
        assert_eq!(back.host(), [11, 12, 19, 20]);
    }
}

#[test]
fn mips_of_single_view_images_are_transferred_alone() {
    let mut gpu = gpu();
    let image = Image::<R8G8B8A8Unorm, usage![TransferSrc, TransferDst]>::with_mip_levels(
        UVec2::splat(4),
        3,
    )
    .unwrap();
    let mip1 = buffer_with::<u8>(&[9; 2 * 2 * 4]);
    let mut out = zeroed_buffer::<u8>(2 * 2 * 4);
    let mut whole = zeroed_buffer::<u8>(4 * 4 * 4);
    gpu.submit(|cmd| {
        cmd.clear_image(image.whole_view(), [0.0; 4]);
        cmd.copy_buffer_to_image(mip1.range(..), image.whole().mip(1));
        cmd.copy_image_to_buffer(image.whole().mip(1), out.range(..));
        cmd.copy_image_to_buffer(image.whole(), whole.range(..));
    });
    assert_eq!(out.host(), [9; 16]);
    assert_eq!(whole.host(), [0; 64], "mip 0 stays");
    let message = panic_message(|| image.whole().mip(3));
    assert!(
        message.contains("isn't one of the view's mips"),
        "{message}"
    );
    // A mip view's slice addresses that mip only.
    let chain = MipChain::<R8G8B8A8Unorm, usage![TransferSrc, TransferDst]>::with_mip_levels(
        UVec2::splat(4),
        3,
    )
    .unwrap();
    let message = panic_message(|| chain.mip_view(1).whole().mip(2));
    assert!(
        message.contains("isn't one of the view's mips"),
        "{message}"
    );
    gpu.submit(|cmd| {
        cmd.copy_buffer_to_image(mip1.range(..), chain.whole().mip(1));
        cmd.copy_image_to_buffer(chain.mip_view(1).whole(), out.range(..));
    });
    assert_eq!(out.host(), [9; 16]);
}

#[test]
fn clears_cover_their_views_mips_and_layers() {
    let mut gpu = gpu();
    // A mip of a chain: the others keep their colour.
    let chain = MipChain::<R8G8B8A8Unorm, usage![TransferSrc, TransferDst]>::with_mip_levels(
        UVec2::splat(4),
        3,
    )
    .unwrap();
    let mut mips = [4, 2, 1].map(|size| zeroed_buffer::<u8>(size * size * 4));
    gpu.submit(|cmd| {
        cmd.clear_image(chain.whole_view(), [1.0, 0.0, 0.0, 1.0]);
        cmd.clear_image(chain.mip_view(1), [0.0, 1.0, 0.0, 1.0]);
        for (level, out) in mips.iter().enumerate() {
            cmd.copy_image_to_buffer(chain.whole().mip(level as u32), out.range(..));
        }
    });
    let all = |out: &mut lava::Buffer<u8, lava::buffer::Readback>, texel: [u8; 4]| {
        out.host().as_chunks::<4>().0.iter().all(|&t| t == texel)
    };
    let [mip0, mip1, mip2] = &mut mips;
    assert!(all(mip0, [255, 0, 0, 255]) && all(mip2, [255, 0, 0, 255]));
    assert!(all(mip1, [0, 255, 0, 255]));
    // Layers of an array: a range of them, then one layer's own view.
    type Array = Image<R8G8B8A8Unorm, usage![TransferSrc, TransferDst], lava::image::D2Array>;
    let array = Array::with_layers(UVec2::splat(2), 1, 4).unwrap();
    let mut layers = zeroed_buffer::<u8>(4 * 2 * 2 * 4);
    gpu.submit(|cmd| {
        cmd.clear_image(array.whole_view(), [0.0, 0.0, 0.0, 1.0]);
        cmd.clear_image(array.whole_view().layers(1..3), [0.0, 0.0, 1.0, 1.0]);
        cmd.clear_image(array.whole_view().layer(3), [1.0, 1.0, 1.0, 1.0]);
        cmd.copy_image_to_buffer(array.whole(), layers.range(..));
    });
    let expected: [[u8; 4]; 4] = [
        [0, 0, 0, 255],
        [0, 0, 255, 255],
        [0, 0, 255, 255],
        [255, 255, 255, 255],
    ];
    for (layer, texels) in layers.host().as_chunks::<16>().0.iter().enumerate() {
        assert!(
            texels.chunks(4).all(|t| t == expected[layer]),
            "layer {layer}: {texels:?}"
        );
    }
    // Signed integers keep their sign; 16-bit unsigned ones clear exactly.
    let sint = Image::<R32Sint, usage![TransferSrc, TransferDst]>::new(UVec2::splat(2)).unwrap();
    let mut out = zeroed_buffer::<i32>(4);
    gpu.submit(|cmd| {
        cmd.clear_image(sint.whole_view(), [-5]);
        cmd.copy_image_to_buffer(sint.whole(), out.range(..));
    });
    assert_eq!(out.host(), [-5; 4]);
    let unorm16 =
        Image::<R16G16B16A16Unorm, usage![TransferSrc, TransferDst]>::new(UVec2::ONE).unwrap();
    let mut out = zeroed_buffer::<u16>(4);
    gpu.submit(|cmd| {
        cmd.clear_image(unorm16.whole_view(), [0.0, 1.0, 0.5, 1.0]);
        cmd.copy_image_to_buffer(unorm16.whole(), out.range(..));
    });
    assert_eq!(out.host(), [0, 65535, 32768, 65535]);
}

#[test]
fn linear_blits_need_linear_filtering() {
    let mut gpu = gpu();
    use vk::{Format as F, FormatFeatureFlags2 as Feature};
    let blits = Feature::BLIT_SRC | Feature::BLIT_DST;
    let candidates = [
        F::R32_SFLOAT,
        F::R32G32_SFLOAT,
        F::R32G32B32A32_SFLOAT,
        F::R64_SFLOAT,
        F::R16G16B16A16_SFLOAT,
        F::R16G16B16A16_UNORM,
        F::B10G11R11_UFLOAT_PACK32,
    ];
    let Some(&unfiltered) = candidates
        .iter()
        .find(|&&f| has(f, blits) && !has(f, Feature::SAMPLED_IMAGE_FILTER_LINEAR))
    else {
        return skipped(
            "linear_blits_need_linear_filtering",
            "every candidate float format that blits filters linearly here",
        );
    };
    let transfers = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
    let image = |extent| {
        DynamicImage::with_params(
            format::Dynamic(unfiltered),
            usage::Dynamic(transfers),
            extent,
            1,
        )
        .unwrap()
    };
    let (small, large) = (image(UVec2::splat(2)), image(UVec2::splat(4)));
    let message = panic_message(|| {
        gpu.submit(|cmd| cmd.blit_image(small.whole(), large.whole(), Filter::Linear))
    });
    assert!(
        message.contains("SAMPLED_IMAGE_FILTER_LINEAR") && message.contains("Filter::Nearest"),
        "{message}"
    );
    // The nearest texel works.
    gpu.submit(|cmd| {
        cmd.clear_image(small.whole_view(), format::ClearValue::Float([0.0; 4]));
        cmd.blit_image(small.whole(), large.whole(), Filter::Nearest)
    });
}

// ---- runtime-typed images ------------------------------------------------------------------

type DynamicImage = Image<format::Dynamic, usage::Dynamic>;

fn dynamic(format: vk::Format, usage: vk::ImageUsageFlags) -> DynamicImage {
    DynamicImage::with_params(
        format::Dynamic(format),
        usage::Dynamic(usage),
        UVec2::splat(4),
        1,
    )
    .unwrap()
}

/// Runtime formats and usages meet every bound, so each command and generated binding checks
/// what its bounds would have: the aspects, the texel type, the usage and the clear variant.
#[test]
fn runtime_typed_images_are_checked_where_they_are_used() {
    let mut gpu = gpu();
    use format::ClearValue;
    use vk::ImageUsageFlags as Usage;
    let transfers = Usage::TRANSFER_SRC | Usage::TRANSFER_DST;
    let words = dynamic(vk::Format::R32_UINT, transfers);
    assert_eq!(words.format(), format::Dynamic(vk::Format::R32_UINT));
    assert_eq!(words.usage(), usage::Dynamic(transfers));
    let mut out = zeroed_buffer::<u32>(16);
    gpu.submit(|cmd| {
        cmd.clear_image(words.whole_view(), ClearValue::Uint([7, 0, 0, 0]));
        cmd.copy_image_to_buffer(words.whole(), out.range(..));
    });
    assert_eq!(out.host(), [7; 16]);

    let message = panic_message(|| {
        gpu.submit(|cmd| cmd.clear_image(words.whole_view(), ClearValue::Float([1.0; 4])))
    });
    assert!(message.contains("can't be cleared to Float"), "{message}");
    let source_only = dynamic(vk::Format::R32_UINT, Usage::TRANSFER_SRC);
    let message = panic_message(|| {
        gpu.submit(|cmd| cmd.clear_image(source_only.whole_view(), ClearValue::Uint([0; 4])))
    });
    assert!(
        message.contains("lacks") && message.contains("TRANSFER_DST"),
        "{message}"
    );
    let depth = dynamic(vk::Format::D32_SFLOAT, transfers);
    let message = panic_message(|| {
        gpu.submit(|cmd| cmd.clear_image(depth.whole_view(), ClearValue::Float([0.0; 4])))
    });
    assert!(
        message.contains("lacks the aspects") && message.contains("COLOR"),
        "{message}"
    );
    let message = panic_message(|| {
        gpu.submit(|cmd| {
            let value = ClearValue::DepthStencil {
                depth: 0.5,
                stencil: 0,
            };
            cmd.clear_depth_stencil(words.whole_view(), value)
        })
    });
    assert!(message.contains("no depth or stencil aspect"), "{message}");
    gpu.submit(|cmd| {
        let value = ClearValue::DepthStencil {
            depth: 0.5,
            stencil: 0,
        };
        cmd.clear_depth_stencil(depth.whole_view(), value)
    });

    // `TestComputeImage` writes float texels: a `uint` image's are refused, a float one's taken.
    let storage = |format| {
        dynamic(format, Usage::STORAGE | Usage::TRANSFER_SRC)
            .bind_storage()
            .unwrap()
    };
    fn args(
        image: &DynamicImage,
    ) -> TestComputeImageArgs<lava::ImageView<'_, format::Dynamic, usage::Dynamic>> {
        TestComputeImageArgs {
            target: image.whole_view(),
            tint: Vec4::ONE,
            offset: UVec2::ZERO,
            size: UVec2::splat(4),
        }
    }
    let uints = storage(vk::Format::R32_UINT);
    let message = panic_message(|| TestComputeImage::push_bindings(args(&uints), None));
    assert!(message.contains("doesn't have texels of type"), "{message}");
    let floats = storage(vk::Format::R8G8B8A8_UNORM);
    let mut texels = zeroed_buffer::<u8>(64);
    gpu.submit(|cmd| {
        cmd.compute(
            &TestComputeImage::push_bindings(args(&floats), None),
            [1, 1, 1],
        );
        cmd.copy_image_to_buffer(floats.whole(), texels.range(..));
    });
    assert!(texels.host().iter().any(|&b| b != 0), "the pass wrote it");
    // A binding that samples needs the sampled usage.
    let unsampled = dynamic(vk::Format::R8G8B8A8_UNORM, Usage::STORAGE);
    let message = panic_message(|| {
        TestRasterTextured::push_bindings(
            TestRasterTexturedArgs {
                vertices: out.range(..).bytes().cast(),
                tex: unsampled.whole_view(),
                sampler: nearest(),
            },
            None,
        )
    });
    assert!(
        message.contains("lacks") && message.contains("SAMPLED"),
        "{message}"
    );
}

/// `erase` keeps an image's format and usage as values, `cast` takes them back as types, and
/// a cast to a format or usage the image doesn't have panics.
#[test]
fn erased_images_cast_back_to_what_they_have() {
    let mut gpu = gpu();
    type Texture = Image<R8G8B8A8Unorm, usage![Sampled, TransferDst]>;
    let erased = Texture::new(UVec2::splat(4))
        .unwrap()
        .bind_sampled()
        .unwrap()
        .erase();
    assert_eq!(erased.format().0, vk::Format::R8G8B8A8_UNORM);
    assert_eq!(
        erased.usage().0,
        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST
    );
    // Still bound, and sampled like the static image.
    let handle = erased.handle::<bindless::Sampled>();
    gpu.submit(|cmd| {
        cmd.clear_image(
            erased.whole_view(),
            format::ClearValue::Float([0.0, 1.0, 0.0, 1.0]),
        )
    });
    let reads = [ImageUse::sampled_read(erased.whole_view())];
    assert_eq!(
        sample(&mut gpu, handle.bits(), &reads),
        Vec4::new(0.0, 1.0, 0.0, 1.0)
    );
    // Fewer usage flags are fine; the handle stays.
    let sampled = erased.cast(R8G8B8A8Unorm, <usage![Sampled]>::default());
    assert_eq!(sampled.handle::<bindless::Sampled>(), handle);
    let message = panic_message(|| {
        sampled
            .erase()
            .cast(R8G8B8A8Srgb, <usage![Sampled]>::default())
    });
    assert!(message.contains("can't cast"), "{message}");
    let other = Texture::new(UVec2::ONE).unwrap().erase();
    let message =
        panic_message(|| other.cast(R8G8B8A8Unorm, <usage![Sampled, Storage]>::default()));
    assert!(message.contains("can't cast"), "{message}");
}

/// A host copy moves the layouts of the mips it finds in another one (here: never accessed,
/// so `UNDEFINED`) to `GENERAL`, leaves those already there, and GPU accesses after it find
/// them there; validation checks each transition's old layout.
#[test]
fn host_copies_transition_only_the_mips_not_yet_general() {
    let mut gpu = gpu();
    let test = "host_copies_transition_only_the_mips_not_yet_general";
    if !has(
        vk::Format::R8G8B8A8_UNORM,
        vk::FormatFeatureFlags2::HOST_IMAGE_TRANSFER_EXT,
    ) {
        return skipped(test, "no host copies of R8G8B8A8Unorm");
    }
    type Chain = MipChain<R8G8B8A8Unorm, usage![HostCopy, TransferSrc, TransferDst]>;
    let mut chain = match Chain::with_mip_levels(UVec2::splat(4), 3) {
        Ok(chain) => chain,
        Err(Error::MissingFeature(_)) => return skipped(test, "no host image copies"),
        Err(error) => panic!("{error}"),
    };
    // Mip 1 is cleared on the GPU (GENERAL); mips 0 and 2 were never accessed.
    gpu.submit(|cmd| cmd.clear_image(chain.mip_view(1), [0.0, 1.0, 0.0, 1.0]));
    let texels: Vec<u8> = (0..4 * 4 * 4).map(|i| i as u8).collect();
    chain.copy_from(&texels, 0).unwrap();
    // The cleared mip kept its contents: it wasn't transitioned from `UNDEFINED`.
    let (mut mip0, mut mip1) = (zeroed_buffer::<u8>(64), zeroed_buffer::<u8>(16));
    gpu.submit(|cmd| {
        cmd.copy_image_to_buffer(chain.mip_view(0).whole(), mip0.range(..));
        cmd.copy_image_to_buffer(chain.mip_view(1).whole(), mip1.range(..));
    });
    assert_eq!(mip0.host(), texels);
    assert!(mip1.host().chunks(4).all(|t| t == [0, 255, 0, 255]));
    // Again after GPU use: nothing to transition.
    chain.copy_from(&[9; 4], 2).unwrap();
    let mut mip2 = zeroed_buffer::<u8>(4);
    gpu.submit(|cmd| cmd.copy_image_to_buffer(chain.mip_view(2).whole(), mip2.range(..)));
    assert_eq!(mip2.host(), [9; 4]);
}

// ---- reinterpreting views ------------------------------------------------------------------

#[test]
fn mutable_format_images_have_views_of_their_class() {
    let mut gpu = gpu();
    // sRGB texels, also stored through a UNORM storage view (storage needs `MutableFormat`'s
    // extended usage on Intel, which has no sRGB storage).
    type Srgb = Image<
        R8G8B8A8Srgb,
        usage![
            Sampled,
            Storage,
            ColorAttachment,
            TransferSrc,
            MutableFormat
        ],
    >;
    let image = Srgb::new(UVec2::splat(4)).unwrap().bind_sampled().unwrap();
    let unorm = image.whole_view().reinterpret::<R8G8B8A8Unorm>();
    assert_eq!(unorm.format().vk_format(), vk::Format::R8G8B8A8_UNORM);
    let slots = bindless::reserve::<bindless::Combined>(1).unwrap();
    bindless::bind([unorm], [slots.handle(0)]);
    // Rendered through the UNORM view, 0.5 is stored as 128, which sRGB reads as 0.216.
    gpu.submit(|cmd| {
        cmd.raster(
            UVec2::splat(4),
            [Attachment::color(unorm, Some([0.5, 0.5, 0.5, 1.0]))],
        )
        .record("unorm");
    });
    let reads = [ImageUse::sampled_read(image.whole_view())];
    let srgb = sample(&mut gpu, image.handle::<bindless::Sampled>().bits(), &reads);
    let linear = sample(
        &mut gpu,
        slots.handle(0).as_sampled().bits(),
        &[ImageUse::sampled_read(unorm)],
    );
    assert!((linear.x - 128.0 / 255.0).abs() < 1e-3, "{linear}");
    assert!((srgb.x - 0.2158).abs() < 2e-3, "{srgb}");
    // Other classes, and images without `MutableFormat`, have none.
    let message = panic_message(|| image.whole_view().reinterpret::<R16G16B16A16Unorm>());
    assert!(message.contains("compatibility classes"), "{message}");
    let fixed = Image::<R8G8B8A8Unorm, usage![Sampled]>::new(UVec2::ONE)
        .unwrap()
        .erase();
    let message = panic_message(|| fixed.whole_view().reinterpret::<R8G8B8A8Uint>());
    assert!(message.contains("MutableFormat"), "{message}");
    // Floats as words: R32Uint shares R32Sfloat's class.
    let floats = Image::<R32Sfloat, usage![Storage, MutableFormat]>::new(UVec2::ONE).unwrap();
    let _words = floats.whole_view().reinterpret::<R32Uint>();
}

// ---- fit -------------------------------------------------------------------------------------

#[test]
fn fitted_images_keep_their_slots() {
    let mut gpu = gpu();
    type Texture = Image<R8G8B8A8Unorm, usage![Sampled, TransferDst]>;
    let mut image = Texture::new(UVec2::splat(2))
        .unwrap()
        .bind_sampled()
        .unwrap();
    let handle = image.handle::<bindless::Sampled>();
    gpu.submit(|cmd| cmd.clear_image(image.whole_view(), [1.0, 0.0, 0.0, 1.0]));
    assert!(
        !image.fit(UVec2::splat(2)).unwrap(),
        "the same extent keeps the image"
    );
    assert!(image.fit(UVec2::splat(8)).unwrap());
    assert_eq!(image.extent(), UVec2::splat(8));
    assert_eq!(image.handle::<bindless::Sampled>(), handle, "the same slot");
    gpu.submit(|cmd| cmd.clear_image(image.whole_view(), [0.0, 1.0, 0.0, 1.0]));
    let reads = [ImageUse::sampled_read(image.whole_view())];
    assert_eq!(
        sample(&mut gpu, handle.bits(), &reads),
        Vec4::new(0.0, 1.0, 0.0, 1.0)
    );
    // A mip chain's mips keep theirs.
    let mut chain = MipChain::<R8G8B8A8Unorm, usage![Sampled, TransferDst]>::with_mip_levels(
        UVec2::splat(4),
        2,
    )
    .unwrap()
    .bind_mips_sampled()
    .unwrap();
    let mip1 = chain.mip_view(1).handle::<bindless::Sampled>();
    assert!(chain.fit(UVec2::splat(8)).unwrap());
    assert_eq!(chain.mip_view(1).handle::<bindless::Sampled>(), mip1);
    gpu.submit(|cmd| cmd.clear_image(chain.mip_view(1), [0.0, 0.0, 1.0, 1.0]));
    let reads = [ImageUse::sampled_read(chain.mip_view(1))];
    assert_eq!(
        sample(&mut gpu, mip1.bits(), &reads),
        Vec4::new(0.0, 0.0, 1.0, 1.0)
    );
    // Too small for its mips: an error, and the image stays.
    assert!(chain.fit(UVec2::ONE).is_err());
    assert_eq!(chain.extent(), UVec2::splat(8));
    // Fitted, then dropped before a submission: the slot's queued rewrite goes with it.
    let mut dropped = Texture::new(UVec2::ONE).unwrap().bind_sampled().unwrap();
    dropped.fit(UVec2::splat(2)).unwrap();
    drop(dropped);
    gpu.submit(|_| {});
}

// ---- sparse residency ------------------------------------------------------------------------

#[test]
fn sparse_buffers_bind_pages() {
    let mut gpu = gpu();
    if !Ctx::features().sparse_residency_buffer {
        skipped("sparse_buffers_bind_pages", "no sparse residency buffers");
        return;
    }
    use lava::sparse::{SparseBind, SparseBuffer};
    let page = 64 * 1024 / 4;
    let buffer = SparseBuffer::<u32>::new(4 * page).unwrap();
    let words = (buffer.page_size() / 4) as usize;
    assert!(buffer.page_count() >= 1);
    let second = 1.min(buffer.page_count() - 1);
    gpu.timeline
        .bind_sparse(&[SparseBind::buffer(&buffer, second..second + 1, true)])
        .unwrap();
    assert!(buffer.is_resident(second));
    let range = buffer.range(second as usize * words..(second as usize + 1) * words);
    let mut out = zeroed_buffer::<u32>(words);
    gpu.submit(|cmd| {
        cmd.fill_buffer(range.bytes(), 0x0102_0304);
        cmd.copy_buffer(range, out.range(..));
    });
    assert!(out.host().iter().all(|&w| w == 0x0102_0304));
    // Binding it again does nothing; unbinding frees it once no submission reaches it.
    gpu.timeline
        .bind_sparse(&[SparseBind::buffer(&buffer, second..second + 1, true)])
        .unwrap();
    gpu.timeline
        .bind_sparse(&[SparseBind::buffer(&buffer, second..second + 1, false)])
        .unwrap();
    assert!(!buffer.is_resident(second));
    gpu.submit(|_| {});
}

#[test]
fn sparse_images_bind_tiles_and_mip_tails() {
    let mut gpu = gpu();
    if !Ctx::features().sparse_residency_image_2d {
        skipped(
            "sparse_images_bind_tiles_and_mip_tails",
            "no sparse residency 2D images",
        );
        return;
    }
    use lava::sparse::{SparseBind, SparseImage};
    let image = match SparseImage::<R8G8B8A8Unorm, usage![Sampled, TransferSrc, TransferDst]>::new(
        UVec2::splat(256),
        9,
    ) {
        Ok(image) => image,
        Err(Error::Unsupported(why)) => {
            return skipped("sparse_images_bind_tiles_and_mip_tails", &why);
        }
        Err(err) => panic!("{err}"),
    };
    let tile = image.granularity().truncate();
    assert!(tile.x > 0 && tile.y > 0);
    let message = panic_message(|| SparseBind::image(&image, 0, IVec2::ONE, tile, true));
    assert!(message.contains("whole tiles"), "{message}");
    let mut binds = vec![SparseBind::image(&image, 0, IVec2::ZERO, tile, true)];
    if image.mip_tail().is_some() {
        binds.push(SparseBind::mip_tail(&image, true));
    }
    gpu.timeline.bind_sparse(&binds).unwrap();
    assert!(image.is_resident(0, UVec3::ZERO));
    let texels = (tile.x * tile.y * 4) as usize;
    let mut out = zeroed_buffer::<u8>(texels);
    gpu.submit(|cmd| {
        cmd.clear_image(image.whole_view(), [1.0, 0.0, 1.0, 1.0]);
        cmd.copy_image_to_buffer(image.whole_view().region(tile), out.range(..));
    });
    assert!(out.host().chunks(4).all(|t| t == [255, 0, 255, 255]));
    if let Some(tail) = image.mip_tail() {
        let last = image.mip_levels() - 1;
        assert!(tail.first_mip <= last);
        let mut texel = zeroed_buffer::<u8>(4);
        gpu.submit(|cmd| cmd.copy_image_to_buffer(image.whole().mip(last), texel.range(..)));
        assert_eq!(texel.host(), [255, 0, 255, 255]);
    }
    gpu.timeline
        .bind_sparse(&[SparseBind::image(&image, 0, IVec2::ZERO, tile, false)])
        .unwrap();
    assert!(!image.is_resident(0, UVec3::ZERO));
    gpu.submit(|_| {});
}

// ---- external memory -------------------------------------------------------------------------

#[test]
fn exported_memory_is_imported_in_the_same_process() {
    let mut gpu = gpu();
    if !Ctx::features().external_memory {
        skipped(
            "exported_memory_is_imported_in_the_same_process",
            "no external memory",
        );
        return;
    }
    use lava::external::{ExternalBuffer, ExternalImage, HandleType};
    for handle_type in [HandleType::OpaqueFd, HandleType::DmaBuf] {
        if !handle_type.supported() {
            skipped(
                "exported_memory_is_imported_in_the_same_process",
                "no dma-bufs",
            );
            continue;
        }
        let exported = ExternalBuffer::<u32>::new(64, handle_type).unwrap();
        let imported =
            ExternalBuffer::<u32>::import(exported.export().unwrap(), 64, handle_type).unwrap();
        assert_eq!(imported.handle_type(), handle_type);
        let mut out = zeroed_buffer::<u32>(64);
        gpu.submit(|cmd| {
            cmd.fill_buffer(exported.byte_range(..), 42);
            cmd.copy_buffer(imported.range(..), out.range(..));
        });
        assert_eq!(out.host(), [42; 64]);
    }
    // An imported image is an image of its own over the same memory: it starts undefined (lava
    // records no layout handoff), so this checks only that it is made and usable.
    type Shared = ExternalImage<R8G8B8A8Unorm, usage![TransferSrc, TransferDst]>;
    let exported = Shared::new(UVec2::splat(4), 1).unwrap();
    let imported = Shared::import(exported.export().unwrap(), UVec2::splat(4), 1).unwrap();
    let mut out = zeroed_buffer::<u8>(64);
    gpu.submit(|cmd| {
        cmd.clear_image(imported.whole_view(), [0.0, 1.0, 0.0, 1.0]);
        cmd.copy_image_to_buffer(imported.whole(), out.range(..));
    });
    assert!(
        out.host().chunks(4).all(|t| t == [0, 255, 0, 255]),
        "{:?}",
        &out.host()[..8]
    );
}

//! GPU tests of barrier tracking across submissions and families: two graphics timelines sharing an image through waits, writes of many buffers tracked in each, new resources at destroyed ones' handles and addresses starting without their accesses, attachments a draw of their rendering samples rendered in `GENERAL`, transitions and acquires of several mips merged into one write, `flush_all` on every family, and the transfer family's copy rules
mod common;

use common::render::*;
use common::{Cmd, buffer_with, gpu, panic_message, zeroed_buffer};
use glam::{UVec2, Vec4};
use lava::{
    Ctx,
    buffer::Buffer,
    command_buffer::{Attachment, CullModeFlags, Draw, RasterState},
    image::{
        Image, MipChain,
        format::{D32Sfloat, R8G8B8A8Unorm, R8Unorm},
        usage::usage,
    },
    queue::{Compute, Graphics, GraphicsStage, Kind, Timeline, Transfer, Wait},
};
use lava_tests::bindings::*;

fn has_family(test: &str, kind: Kind) -> bool {
    let has = Ctx::queue_family(kind).is_some();
    if !has {
        eprintln!("SKIPPED {test}: the device has no dedicated {kind:?} family");
    }
    has
}

fn unculled<const N: usize>() -> RasterState<N> {
    RasterState::default().cull_mode(CullModeFlags::NONE)
}

type Shared = Image<R8G8B8A8Unorm, usage![Storage, TransferSrc, TransferDst]>;

/// Writes `tint` times the shader's gradient into all of `image`.
fn tinted(cmd: &mut Cmd, image: &Shared, tint: Vec4) {
    let size = image.extent();
    cmd.compute(
        &TestComputeImage::push_bindings(
            TestComputeImageArgs {
                target: image.whole_view(),
                tint,
                offset: UVec2::ZERO,
                size,
            },
            None,
        ),
        [size.x.div_ceil(8), size.y.div_ceil(8), 1],
    );
}

fn all_texels(bytes: &[u8], expected: [u8; 4]) -> bool {
    bytes.as_chunks::<4>().0.iter().all(|&t| close(t, expected))
}

// ---- timelines of one family ------------------------------------------------------------------

/// Two graphics timelines take turns writing one image, each waiting for the other's last
/// submission and nothing waited for on the host in between: the layout lava tracks in the
/// image carries over, and the semaphore waits order the writes.
#[test]
fn two_graphics_timelines_share_an_image_through_waits() {
    let mut gpu = gpu();
    let mut other = Timeline::<Graphics>::new(2, 0).unwrap();
    let image = Shared::new(UVec2::splat(16))
        .unwrap()
        .bind_storage()
        .unwrap();
    let bytes = (16 * 16 * 4) as usize;
    let (mut red, mut black, mut green) = (
        zeroed_buffer::<u8>(bytes),
        zeroed_buffer::<u8>(bytes),
        zeroed_buffer::<u8>(bytes),
    );

    // Other: clear red. Ours: read it back, then write black with a shader.
    let cleared = (other.begin().unwrap())
        .submit(|cmd| cmd.clear_image(image.whole_view(), [1.0, 0.0, 0.0, 1.0]))
        .unwrap();
    let mut ours = gpu.timeline.begin().unwrap();
    ours.wait_for(cleared.at(GraphicsStage::Transfer));
    let shaded = ours
        .submit(|cmd| {
            cmd.copy_image_to_buffer(image.whole(), red.range(..));
            tinted(cmd, &image, Vec4::new(0.0, 0.0, 0.0, 1.0));
        })
        .unwrap();
    // Other again: read the black back at the compute stage's wait, then clear green, which
    // ours reads last.
    let mut theirs = other.begin().unwrap();
    theirs.wait_for(shaded.at(GraphicsStage::Compute));
    theirs.wait_for(shaded.at(GraphicsStage::Transfer));
    let greened = theirs
        .submit(|cmd| {
            cmd.copy_image_to_buffer(image.whole(), black.range(..));
            cmd.clear_image(image.whole_view(), [0.0, 1.0, 0.0, 1.0]);
        })
        .unwrap();
    let mut last = gpu.timeline.begin().unwrap();
    last.wait_for(greened.at(GraphicsStage::Transfer));
    let done = last
        .submit(|cmd| cmd.copy_image_to_buffer(image.whole(), green.range(..)))
        .unwrap();
    Wait::all(&[done, greened]).unwrap();

    assert!(all_texels(red.host(), [255, 0, 0, 255]));
    assert!(all_texels(black.host(), [0, 0, 0, 255]));
    assert!(all_texels(green.host(), [0, 255, 0, 255]));
}

// ---- pending buffer writes ---------------------------------------------------------------------

/// Writes of many buffers, each tracked in its buffer: later commands still wait for completed
/// ones of an earlier submission, a host access of a buffer whose writes completed doesn't wait,
/// and a write still in flight keeps blocking host access.
#[test]
fn many_buffers_writes_keep_their_order() {
    let mut gpu = gpu();
    const BUFFERS: usize = 80;
    const LEN: usize = 64;
    let sources: Vec<Buffer<u32>> = (0..BUFFERS).map(|_| Buffer::new(LEN).unwrap()).collect();
    // Writes of every source, completed by the next submission.
    gpu.submit(|cmd| {
        for (i, source) in sources.iter().enumerate() {
            cmd.fill_buffer(source.range(..), i as u32);
        }
    });
    // Refilled sources (written in this submission) and the others (written in the last one)
    // are both read by a shader, which writes as many readbacks again.
    let mut outs: Vec<_> = (0..BUFFERS).map(|_| zeroed_buffer::<u32>(LEN)).collect();
    gpu.submit(|cmd| {
        for (i, source) in sources.iter().enumerate().step_by(2) {
            cmd.fill_buffer(source.range(..), 1000 + i as u32);
        }
        for (source, out) in sources.iter().zip(&outs) {
            cmd.compute(
                &TestComputeBuffer::push_bindings(
                    TestComputeBufferArgs {
                        src: source.range(..),
                        dst: out.range(..),
                        count: LEN as u32,
                        multiplier: 2,
                    },
                    None,
                ),
                [1, 1, 1],
            );
        }
    });
    for (i, out) in outs.iter_mut().enumerate() {
        let source = if i % 2 == 0 {
            1000 + i as u32
        } else {
            i as u32
        };
        let expected: Vec<u32> = (0..LEN as u32).map(|x| source * 2 + x).collect();
        assert_eq!(out.host(), expected, "buffer {i}");
    }

    // The readbacks' writes completed; the next submission has one in flight.
    let large = Buffer::<u32>::new(16 << 20).unwrap();
    let mut late = zeroed_buffer::<u32>(4);
    let in_flight = (gpu.timeline.begin().unwrap())
        .submit(|cmd| {
            for value in 0..8 {
                cmd.fill_buffer(large.range(..), value);
            }
            cmd.copy_buffer(large.range(..4), late.range(..));
            // And a copy from an earlier submission's write, which must wait for it.
            cmd.copy_buffer(outs[1].range(..4), outs[0].range(..4));
        })
        .unwrap();
    if !in_flight.is_done() {
        assert!(late.try_host().is_none(), "a write in flight blocks");
    }
    for (i, out) in outs.iter_mut().enumerate().skip(2) {
        assert!(out.try_host().is_some(), "buffer {i}'s writes completed");
    }
    in_flight.block().unwrap();
    assert_eq!(late.try_host(), Some(&[7; 4][..]));
    assert_eq!(outs[0].host()[..4], [2, 3, 4, 5]);
}

// ---- destruction -------------------------------------------------------------------------------

/// Images and buffers two timelines accessed are destroyed once both finished, and forgotten
/// on both: new ones (often at the same handles and addresses) start without their accesses
/// or layouts, again and again.
#[test]
fn destroyed_resources_are_forgotten_on_every_timeline() {
    let mut gpu = gpu();
    let mut other = Timeline::<Graphics>::new(2, 0).unwrap();
    let (mut handles, mut reused) = (Vec::new(), 0);
    for round in 0..16u32 {
        let image = Shared::new(UVec2::splat(8))
            .unwrap()
            .bind_storage()
            .unwrap();
        if handles.contains(&image.whole_view().image()) {
            reused += 1;
        }
        handles.push(image.whole_view().image());
        let scratch = Buffer::<u32>::new(64).unwrap();
        let mut readback = zeroed_buffer::<u8>(8 * 8 * 4);
        let mut numbers = zeroed_buffer::<u32>(64);
        let written = (other.begin().unwrap())
            .submit(|cmd| {
                cmd.clear_image(image.whole_view(), [0.0, 0.0, 1.0, 1.0]);
                cmd.fill_buffer(scratch.range(..), round);
            })
            .unwrap();
        let mut read = gpu.timeline.begin().unwrap();
        read.wait_for(written);
        let read = read
            .submit(|cmd| {
                cmd.copy_image_to_buffer(image.whole(), readback.range(..));
                cmd.copy_buffer(scratch.range(..), numbers.range(..));
                tinted(cmd, &image, Vec4::ONE);
                cmd.fill_buffer(scratch.range(..), 0);
            })
            .unwrap();
        // Dropped with accesses pending on both timelines.
        drop((image, scratch));
        read.block().unwrap();
        assert!(
            all_texels(readback.host(), [0, 0, 255, 255]),
            "round {round}"
        );
        assert_eq!(numbers.host(), [round; 64], "round {round}");
        if round % 4 == 3 {
            lava::wait_idle().unwrap();
        }
    }
    eprintln!("{reused} of 16 images reused an earlier image's handle");
}

// ---- attachment layouts -------------------------------------------------------------------------

/// A depth attachment that a draw of its own rendering samples is rendered in `GENERAL`, the
/// layout of its descriptor, and tested against while it is read; one no draw samples is in
/// `ATTACHMENT_OPTIMAL`, which the next access transitions from.
#[test]
fn attachments_their_draws_sample_are_rendered_in_general() {
    let mut gpu = gpu();
    let depth = Image::<D32Sfloat, usage![DepthAttachment, Sampled]>::new(EXTENT)
        .unwrap()
        .bind_sampled()
        .unwrap();
    // The left triangle at depth 0.5, and a quad over everything at 0.4.
    let triangles = two_triangles(&mut gpu);
    let quad = gpu.buffer(&[
        vertex(-1.0, -1.0, 0.4, RED),
        vertex(1.0, -1.0, 0.4, RED),
        vertex(1.0, 1.0, 0.4, RED),
        vertex(-1.0, -1.0, 0.4, RED),
        vertex(1.0, 1.0, 0.4, RED),
        vertex(-1.0, 1.0, 0.4, RED),
    ]);
    let left = TestDepthOnly::push_bindings(
        TestDepthOnlyArgs {
            vertices: triangles.range(0..3),
        },
        None,
    );
    let feedback = TestDepthFeedback::push_bindings(
        TestDepthFeedbackArgs {
            vertices: quad.range(..),
            depth: depth.whole_view(),
        },
        None,
    );
    let mut sampled = zeroed_buffer::<f32>((SIZE * SIZE) as usize);
    let pixels = render(&mut gpu, |cmd, target| {
        // Written in `ATTACHMENT_OPTIMAL`: no draw reads it.
        cmd.raster(EXTENT, [])
            .depth_attachment(depth.whole_view(), Some([0.25]))
            .draw(&Draw::new(&left, 3, 1, unculled()))
            .record("depth");
        // Read-only and sampled by the draw: `GENERAL`.
        cmd.raster(EXTENT, [Attachment::color(target.whole_view(), None)])
            .depth_attachment(depth.whole_view(), None)
            .draw(&Draw::new(&feedback, 6, 1, unculled().depth_write(false)))
            .record("feedback");
        // Again written, then sampled by a compute pass after the rendering.
        cmd.raster(EXTENT, [])
            .depth_attachment(depth.whole_view(), None)
            .draw(&Draw::new(
                &TestDepthOnly::push_bindings(
                    TestDepthOnlyArgs {
                        vertices: triangles.range(3..6),
                    },
                    None,
                ),
                3,
                1,
                unculled(),
            ))
            .record("right");
        cmd.compute(
            &TestSampleDepth::push_bindings(
                TestSampleDepthArgs {
                    depth: depth.whole_view(),
                    out: sampled.range(..),
                },
                None,
            ),
            [SIZE / 8, SIZE / 8, 1],
        );
    });
    // Outside the triangle the quad passed the test against 0.25 and read 0.25 there.
    let quarter = (0.25f32 * 255.0).round() as u8;
    assert_pixel(&pixels, 2, 2, [quarter, 0, 0, 255]);
    assert_pixel(&pixels, IN_RIGHT.0, IN_RIGHT.1, [quarter, 0, 0, 255]);
    // Inside it 0.4 doesn't pass 0.5: the target keeps its clear.
    assert_pixel(&pixels, IN_LEFT.0, IN_LEFT.1, BLACK);
    let mut at = |(x, y): (u32, u32)| sampled.host()[(y * SIZE + x) as usize];
    assert_eq!(at(IN_LEFT), 0.5);
    assert_eq!(at(IN_RIGHT), 0.5, "the right triangle's depth");
    assert_eq!(at((2, 2)), 0.25);
}

// ---- transitions of several mips ------------------------------------------------------------------

type Chain = MipChain<R8G8B8A8Unorm, usage![TransferSrc, TransferDst]>;

/// Reads every mip of `chain` back, one copy per mip.
fn read_mips<K: lava::queue::QueueKind>(
    cmd: &mut lava::CommandBuffer<K>,
    chain: &Chain,
    out: &[Buffer<u8, lava::buffer::Readback>],
) {
    for (level, out) in out.iter().enumerate() {
        cmd.copy_image_to_buffer(chain.mip_view(level as u32).whole(), out.range(..));
    }
}

fn mip_buffers(chain: &Chain) -> Vec<Buffer<u8, lava::buffer::Readback>> {
    (0..chain.mip_levels())
        .map(|level| {
            let extent = chain.mip_extent(level);
            zeroed_buffer::<u8>((extent.x * extent.y * 4) as usize)
        })
        .collect()
}

/// A clear of a whole chain whose mips are in different states transitions runs of them in
/// one barrier, which becomes one pending write of all of them, visible to the copies that
/// read each mip after it; a later clear of one mip is waited for by only its own reads.
#[test]
fn transitions_of_several_mips_merge_into_one_write() {
    let mut gpu = gpu();
    let chain = Chain::with_mip_levels(UVec2::splat(16), 4).unwrap();
    let mut outs = mip_buffers(&chain);
    let upload = buffer_with::<u8>(&[7; 16 * 16 * 4]);
    gpu.submit(|cmd| {
        // Mip 0 written first, so it isn't `UNDEFINED` like the others.
        cmd.copy_buffer_to_image(upload.range(..), chain.mip_view(0).whole());
        cmd.clear_image(chain.whole_view(), [0.0, 1.0, 0.0, 1.0]);
        read_mips(cmd, &chain, &outs);
    });
    for (level, out) in outs.iter_mut().enumerate() {
        assert!(all_texels(out.host(), [0, 255, 0, 255]), "mip {level}");
    }
    gpu.submit(|cmd| {
        cmd.clear_image(chain.mip_view(2), [1.0, 0.0, 0.0, 1.0]);
        read_mips(cmd, &chain, &outs);
    });
    for (level, out) in outs.iter_mut().enumerate() {
        let expected = if level == 2 {
            [255, 0, 0, 255]
        } else {
            [0, 255, 0, 255]
        };
        assert!(all_texels(out.host(), expected), "mip {level}");
    }
}

/// A chain cleared on the compute family and released to graphics is acquired in one barrier
/// call for all its mips, whose acquire is the pending write the graphics reads wait for.
#[test]
fn acquires_of_several_mips_merge_into_one_write() {
    let mut gpu = gpu();
    if !has_family(
        "acquires_of_several_mips_merge_into_one_write",
        Kind::Compute,
    ) {
        return;
    }
    let mut compute = Timeline::<Compute>::new(2, 0).unwrap();
    let chain = Chain::with_mip_levels(UVec2::splat(16), 4).unwrap();
    let mut outs = mip_buffers(&chain);
    let released = (compute.begin().unwrap())
        .submit(|cmd| {
            cmd.clear_image(chain.whole_view(), [0.0, 0.0, 1.0, 1.0]);
            cmd.clear_image(chain.mip_view(1), [1.0, 1.0, 0.0, 1.0]);
            cmd.release::<Graphics>(chain.whole_view());
        })
        .unwrap();
    gpu.submit_after(&[released], |cmd| {
        read_mips(cmd, &chain, &outs);
        // And written after the acquire, which the read before must have finished.
        cmd.clear_image(chain.mip_view(3), [1.0; 4]);
    });
    for (level, out) in outs.iter_mut().enumerate() {
        let expected = if level == 1 {
            [255, 255, 0, 255]
        } else {
            [0, 0, 255, 255]
        };
        assert!(all_texels(out.host(), expected), "mip {level}");
    }
    // Back, released mip by mip and acquired by a clear of all of them at once.
    let released = gpu.submit(|cmd| {
        for level in 0..chain.mip_levels() {
            cmd.release::<Compute>(chain.mip_view(level));
        }
    });
    let mut cleared = compute.begin().unwrap();
    cleared.wait_for(released);
    cleared
        .submit(|cmd| {
            cmd.clear_image(chain.whole_view(), [1.0, 0.0, 1.0, 1.0]);
            read_mips(cmd, &chain, &outs);
        })
        .unwrap()
        .block()
        .unwrap();
    for (level, out) in outs.iter_mut().enumerate() {
        assert!(all_texels(out.host(), [255, 0, 255, 255]), "mip {level}");
    }
}

// ---- flush_all ---------------------------------------------------------------------------------------

/// `flush_all` records a full barrier the tracking doesn't know of, on every family, between
/// commands that are tracked as usual.
#[test]
fn flush_all_is_a_full_barrier_on_every_family() {
    let mut gpu = gpu();
    let source = Buffer::<u32>::new(16).unwrap();
    let mut out = zeroed_buffer::<u32>(16);
    gpu.submit(|cmd| {
        cmd.flush_all();
        cmd.fill_buffer(source.range(..), 3);
        cmd.flush_all();
        cmd.copy_buffer(source.range(..), out.range(..));
        cmd.flush_all();
    });
    assert_eq!(out.host(), [3; 16]);

    if Ctx::queue_family(Kind::Compute).is_some() {
        let mut compute = Timeline::<Compute>::new(2, 0).unwrap();
        let done = (compute.begin().unwrap())
            .submit(|cmd| {
                cmd.fill_buffer(source.range(..), 4);
                cmd.flush_all();
                cmd.copy_buffer(source.range(..), out.range(..));
            })
            .unwrap();
        done.block().unwrap();
        assert_eq!(out.host(), [4; 16]);
    } else {
        eprintln!("SKIPPED flush_all on a compute family: the device has none");
    }
    if Ctx::queue_family(Kind::Transfer).is_some() {
        let mut transfer = Timeline::<Transfer>::new(2, 0).unwrap();
        let done = (transfer.begin().unwrap())
            .submit(|cmd| {
                cmd.fill_buffer(source.range(..), 5);
                cmd.flush_all();
                cmd.copy_buffer(source.range(..), out.range(..));
            })
            .unwrap();
        done.block().unwrap();
        assert_eq!(out.host(), [5; 16]);
    } else {
        eprintln!("SKIPPED flush_all on a transfer family: the device has none");
    }
}

// ---- the transfer family --------------------------------------------------------------------------

/// A transfer family copies between images and 4-byte aligned buffer addresses only (other
/// families take any multiple of the texel block), copies boxes of its image granularity, and
/// leaves copies of depth images, either way, to graphics.
#[test]
fn transfer_family_copies_follow_its_rules() {
    let mut gpu = gpu();
    if !has_family("transfer_family_copies_follow_its_rules", Kind::Transfer) {
        return;
    }
    let mut transfer = Timeline::<Transfer>::new(2, 0).unwrap();
    let image = Image::<R8Unorm, usage![TransferSrc, TransferDst]>::new(UVec2::splat(4)).unwrap();
    let bytes: Vec<u8> = (0..32).collect();
    let source = buffer_with::<u8>(&bytes);
    let mut out = zeroed_buffer::<u8>(16);

    // One byte past an aligned address: fine for an R8 copy on graphics.
    gpu.submit(|cmd| {
        cmd.copy_buffer_to_image(source.range(1..17), image.whole());
        cmd.copy_image_to_buffer(image.whole(), out.range(..));
        cmd.release::<Transfer>(image.whole_view());
    });
    assert_eq!(out.host(), &bytes[1..17]);
    let message = panic_message(|| {
        (transfer.begin().unwrap())
            .submit(|cmd| cmd.copy_buffer_to_image(source.range(1..17), image.whole()))
    });
    assert!(
        message.contains("a transfer family copies from and to 4-byte aligned buffer addresses"),
        "{message}"
    );
    let message = panic_message(|| {
        (transfer.begin().unwrap())
            .submit(|cmd| cmd.copy_image_to_buffer(image.whole(), source.range(1..17)))
    });
    assert!(message.contains("4-byte aligned"), "{message}");

    // An aligned copy of an odd box: either the granularity allows it, or it panics saying
    // which granularity it keeps.
    let odd = || {
        (transfer.begin().unwrap())
            .submit(|cmd| {
                cmd.copy_buffer_to_image(
                    source.range(4..13),
                    image
                        .whole()
                        .offset(glam::IVec2::ONE)
                        .with_extent(UVec2::splat(3)),
                );
                cmd.copy_image_to_buffer(image.whole(), out.range(..));
            })
            .unwrap()
            .block()
            .unwrap()
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(odd)) {
        Ok(()) => {
            let texels = out.host();
            assert_eq!(texels[5..8], [4, 5, 6]);
            assert_eq!(texels[13..16], [10, 11, 12]);
            assert_eq!(texels[0], 1, "outside the box");
        }
        Err(payload) => {
            let message = common::payload_message(&*payload);
            assert!(message.contains("image granularity"), "{message}");
        }
    }

    // Depth and stencil copies are graphics' only, also those out of an image.
    let depth =
        Image::<D32Sfloat, usage![DepthAttachment, TransferSrc, TransferDst]>::new(UVec2::splat(4))
            .unwrap();
    let mut depths = zeroed_buffer::<f32>(16);
    gpu.submit(|cmd| {
        cmd.clear_depth_stencil(depth.whole_view(), [0.75]);
        cmd.copy_image_to_buffer(depth.whole(), depths.range(..));
    });
    assert_eq!(depths.host(), [0.75; 16]);
    let message = panic_message(|| {
        (transfer.begin().unwrap())
            .submit(|cmd| cmd.copy_image_to_buffer(depth.whole(), depths.range(..)))
    });
    assert!(
        message.contains("copies of depth or stencil images are recorded on the graphics family"),
        "{message}"
    );
    let floats = buffer_with::<f32>(&[0.5; 16]);
    let message = panic_message(|| {
        (transfer.begin().unwrap())
            .submit(|cmd| cmd.copy_buffer_to_image(floats.range(..), depth.whole()))
    });
    assert!(
        message.contains("copies of depth or stencil images are recorded on the graphics family"),
        "{message}"
    );
}

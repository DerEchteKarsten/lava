//! GPU tests of referencing buffers, through generated reference infos (`TestImagePair`'s, nested arrays of `RefNode`'s with samplers, buffers, a `DynImg` and write references, and the write references of `TestReferenceDraw`'s draws): what a buffer references gets its barriers and layout transitions from commands that name the buffer, without `Accesses`, stays alive after its handles are dropped, survives `Buffer::reserve`, and isn't written by commands
mod common;

use common::{Cmd, gpu, nearest, panic_message, zeroed_buffer};
use glam::{UVec2, Vec4};
use lava::{
    Buffer,
    bindless::{Sampler, SamplerDesc},
    buffer::Upload,
    command_buffer::{Attachment, CullModeFlags, Draw, RasterState},
    image::{Image, format::R8G8B8A8Unorm, usage::usage},
    reference::DynImage,
};
use lava_tests::bindings::*;

type Target = Image<R8G8B8A8Unorm, usage![ColorAttachment, Sampled, Storage]>;

/// A 2x2 image bound with both descriptors.
fn target() -> Target {
    Target::new(UVec2::splat(2))
        .unwrap()
        .bind_combined()
        .unwrap()
}

/// Clears `image` to `color` as a rendering's attachment, which leaves it outside `GENERAL`.
fn render(cmd: &mut Cmd, image: &Target, color: [f32; 4]) {
    cmd.raster(
        UVec2::splat(2),
        [Attachment::color(image.whole_view(), Some(color))],
    )
    .record("clear");
}

/// `TestImage2d` over `top` and the pair `pairs` holds; the texels it read: `top`'s, then the
/// storage image's, then the sampled image's, four each.
fn read_pair(
    cmd: &mut Cmd,
    top: &Target,
    pairs: &Buffer<TestImagePair, impl lava::buffer::Memory>,
    out: &Buffer<Vec4, lava::buffer::Readback>,
) {
    let bindings = TestImage2d::push_bindings(
        TestImage2dArgs {
            image: top.whole_view(),
            images: pairs.range(..),
            sampler: nearest(),
            out: out.range(..),
        },
        None,
    );
    cmd.compute(&bindings, [1, 1, 1]);
}

const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
const BLUE: [f32; 4] = [0.0, 0.0, 1.0, 1.0];

fn check(out: &mut Buffer<Vec4, lava::buffer::Readback>, colors: [[f32; 4]; 3]) {
    let read = out.host();
    for (i, color) in colors.into_iter().enumerate() {
        for texel in &read[i * 4..i * 4 + 4] {
            assert_eq!(*texel, Vec4::from(color), "image {i}");
        }
    }
}

/// Images rendered right before a pass reads them through a buffer get the barrier and the
/// transition to `GENERAL` from the pass's naming of the buffer, without `Accesses`.
#[test]
fn what_a_buffer_references_gets_its_barriers() {
    let mut gpu = gpu();
    let (top, storage, sampled) = (target(), target(), target());
    let mut pairs: Buffer<TestImagePair, Upload> = Buffer::upload(1).unwrap();
    pairs.host_write_referencing(
        0,
        [TestImagePairReferenceInfo {
            storage: Some(storage.whole_view()),
            sampled: Some(sampled.whole_view()),
        }],
    );
    let mut out = zeroed_buffer::<Vec4>(12);
    gpu.submit(|cmd| {
        render(cmd, &top, RED);
        render(cmd, &storage, GREEN);
        render(cmd, &sampled, BLUE);
        read_pair(cmd, &top, &pairs, &out);
    });
    check(&mut out, [RED, GREEN, BLUE]);

    // Rendered again: the buffer is dirty again, and the next read waits for it.
    gpu.submit(|cmd| {
        render(cmd, &storage, BLUE);
        render(cmd, &sampled, RED);
        read_pair(cmd, &top, &pairs, &out);
    });
    check(&mut out, [RED, BLUE, RED]);
}

/// A buffer keeps what it references alive: dropped handles, a grown buffer, and an uploader's
/// staged write all read the images.
#[test]
fn what_a_buffer_references_lives_with_it() {
    let mut gpu = gpu();
    let top = target();
    let (storage, sampled) = (target(), target());
    gpu.submit(|cmd| {
        render(cmd, &top, RED);
        render(cmd, &storage, GREEN);
        render(cmd, &sampled, BLUE);
    });
    let mut pairs = Buffer::<TestImagePair>::new(1).unwrap();
    gpu.uploader()
        .write_referencing(
            &pairs,
            0,
            [TestImagePairReferenceInfo {
                storage: Some(storage.whole_view()),
                sampled: Some(sampled.whole_view()),
            }],
        )
        .unwrap();
    drop((storage, sampled));
    let mut out = zeroed_buffer::<Vec4>(12);
    gpu.submit(|cmd| {
        cmd.drain_uploads();
        read_pair(cmd, &top, &pairs, &out);
    });
    check(&mut out, [RED, GREEN, BLUE]);

    // Grown, the buffer references what the old one did.
    gpu.submit(|cmd| {
        assert!(pairs.reserve(4, cmd).unwrap());
    });
    lava::wait_idle().unwrap();
    let mut out = zeroed_buffer::<Vec4>(12);
    gpu.submit(|cmd| read_pair(cmd, &top, &pairs, &out));
    check(&mut out, [RED, GREEN, BLUE]);
}

/// Commands and passes don't write a referencing buffer: lava couldn't follow what they write.
#[test]
fn referencing_buffers_refuse_command_writes() {
    let mut gpu = gpu();
    let (storage, sampled) = (target(), target());
    let mut pairs: Buffer<TestImagePair, Upload> = Buffer::upload(1).unwrap();
    pairs.host_write_referencing(
        0,
        [TestImagePairReferenceInfo {
            storage: Some(storage.whole_view()),
            sampled: Some(sampled.whole_view()),
        }],
    );
    let message = panic_message(|| gpu.submit(|cmd| cmd.fill_buffer(pairs.range(..), 0)));
    assert!(
        message.contains("is written with `host_write_referencing`"),
        "{message}"
    );
}

/// Nested arrays of referencing structs: what each element's handles and addresses name is
/// sampled, read and written through them, alive after its handles are dropped, and the
/// writes through write references are waited for by a copy of the same submission.
#[test]
fn nested_references_are_reached_and_kept() {
    let mut gpu = gpu();
    let (red, green, blue) = (target(), target(), target());
    let samplers = [
        Sampler::new(&SamplerDesc::new()).unwrap(),
        Sampler::new(&SamplerDesc::new()).unwrap(),
    ];
    let weights = common::buffer_with(&[0.25f32, 0.5]);
    let hits = Buffer::<u32>::new(1).unwrap();
    let slots = Buffer::<u32>::new(1).unwrap();
    gpu.submit(|cmd| {
        render(cmd, &red, RED);
        render(cmd, &green, GREEN);
        render(cmd, &blue, BLUE);
        cmd.fill_buffer(hits.range(..), 0);
    });
    let mut nodes: Buffer<RefNode, Upload> = Buffer::upload(1).unwrap();
    nodes.host_write_referencing(
        0,
        [RefNodeReferenceInfo {
            materials: [
                RefMaterialReferenceInfo {
                    color: Some(red.whole_view()),
                    sampler: &samplers[0],
                },
                RefMaterialReferenceInfo {
                    color: Some(green.whole_view()),
                    sampler: &samplers[1],
                },
            ],
            weights: weights.range(..),
            counter: RefCounterReferenceInfo {
                hits: hits.range(..),
                slots: slots.range(..),
            },
            r#dyn: DynImage::new(blue.whole_view()),
            scale: 3.0,
        }],
    );
    let node = nodes.host_ptr();
    // SAFETY: written above, and nothing else reaches the mapping.
    let written = unsafe { node.read() };
    assert_eq!(written.materials[1].sampler, samplers[1].handle());
    assert_eq!(
        written.r#dyn,
        blue.handle::<lava::bindless::Combined>().erase()
    );
    drop((red, green, blue, samplers, weights));
    let read = |gpu: &mut common::Gpu, nodes: &Buffer<RefNode, Upload>| {
        let out = zeroed_buffer::<Vec4>(4);
        let counts = zeroed_buffer::<u32>(2);
        gpu.submit(|cmd| {
            let bindings = TestReferences::push_bindings(
                TestReferencesArgs {
                    nodes: nodes.range(..),
                    out: out.range(..),
                },
                None,
            );
            cmd.compute(&bindings, [1, 1, 1]);
            // Written through the node's write references, which the dispatch reached.
            cmd.copy_buffer(hits.range(..), counts.range(0..1));
            cmd.copy_buffer(slots.range(..), counts.range(1..2));
        });
        (out, counts)
    };
    let (mut out, mut counts) = read(&mut gpu, &nodes);
    assert_eq!(
        out.host(),
        [
            Vec4::from(RED),
            Vec4::from(GREEN),
            Vec4::new(0.25, 0.5, 3.0, 0.0),
            Vec4::from(BLUE)
        ]
    );
    assert_eq!(counts.host(), [1, 10]);
    let (_, mut counts) = read(&mut gpu, &nodes);
    assert_eq!(counts.host(), [2, 11]);

    // A null `DynImg` names nothing.
    let (red, green) = (target(), target());
    let samplers = [Sampler::new(&SamplerDesc::new()).unwrap()];
    let weights = common::buffer_with(&[1.0f32, 2.0]);
    let material = |view| RefMaterialReferenceInfo {
        color: Some(view),
        sampler: &samplers[0],
    };
    nodes.host_write_referencing(
        0,
        [RefNodeReferenceInfo {
            materials: [material(red.whole_view()), material(green.whole_view())],
            weights: weights.range(..),
            counter: RefCounterReferenceInfo {
                hits: hits.range(..),
                slots: slots.range(..),
            },
            r#dyn: DynImage::NULL,
            scale: 0.0,
        }],
    );
    (out, counts) = read(&mut gpu, &nodes);
    assert_eq!(out.host()[3], Vec4::splat(-1.0));
    assert_eq!(counts.host(), [3, 12]);
}

/// Draws of one rendering may share an `[Unordered]` write reference but not an ordered one:
/// what they write through references is checked like what they name.
#[test]
fn draws_writing_through_references_follow_their_ordering() {
    let mut gpu = gpu();
    let image = target();
    let hits = Buffer::<u32>::new(1).unwrap();
    let slots = [
        Buffer::<u32>::new(1).unwrap(),
        Buffer::<u32>::new(1).unwrap(),
    ];
    let mut shared: Buffer<RefHits, Upload> = Buffer::upload(1).unwrap();
    shared.host_write_referencing(
        0,
        [RefHitsReferenceInfo {
            hits: hits.range(..),
        }],
    );
    // Two buffers that name the first slot, one the second.
    let mut slot_refs: [Buffer<RefSlot, Upload>; 3] =
        std::array::from_fn(|_| Buffer::upload(1).unwrap());
    for (refs, slot) in slot_refs.iter_mut().zip([&slots[0], &slots[1], &slots[0]]) {
        refs.host_write_referencing(
            0,
            [RefSlotReferenceInfo {
                slot: slot.range(..),
            }],
        );
    }
    let [first, second, first_again] = slot_refs.each_ref().map(|slot| {
        TestReferenceDraw::push_bindings(
            TestReferenceDrawArgs {
                hits: shared.range(..),
                slot: slot.range(..),
            },
            None,
        )
    });
    let state = || RasterState::default().cull_mode(CullModeFlags::NONE);
    let mut counts = zeroed_buffer::<u32>(1);
    gpu.submit(|cmd| {
        cmd.fill_buffer(hits.range(..), 0);
        cmd.raster(
            UVec2::splat(2),
            [Attachment::color(image.whole_view(), None)],
        )
        .draw(&Draw::new(&first, 3, 1, state()))
        .draw(&Draw::new(&second, 3, 1, state()))
        .record("unordered hits");
        cmd.copy_buffer(hits.range(..), counts.range(..));
    });
    assert_eq!(counts.host()[0], 8, "4 fragments each");
    let message = panic_message(|| {
        gpu.submit(|cmd| {
            cmd.raster(
                UVec2::splat(2),
                [Attachment::color(image.whole_view(), None)],
            )
            .draw(&Draw::new(&first, 3, 1, state()))
            .draw(&Draw::new(&first_again, 3, 1, state()))
            .record("one slot");
        })
    });
    assert!(message.contains("both access"), "{message}");
}

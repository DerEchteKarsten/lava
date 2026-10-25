//! GPU tests of referencing buffers, with a hand-written `ReferencingData` type over a generated struct of image handles: what a buffer references gets its barriers and layout transitions from commands that name the buffer, without `Accesses`, stays alive after its handles are dropped, survives `Buffer::reserve`, and isn't written by commands
mod common;

use common::{Cmd, gpu, nearest, panic_message, zeroed_buffer};
use glam::{UVec2, Vec4};
use lava::{
    Buffer,
    buffer::Upload,
    command_buffer::Attachment,
    image::{Image, SampledView, StorageView, format::R8G8B8A8Unorm, usage::usage},
    reference::{Reference, ReferenceInfo, ReferencingData},
};
use lava_tests::bindings::*;

/// `TestImagePair` as a referencing type: its storage and sampled handles are references.
#[derive(Clone, Copy, bytemuck::AnyBitPattern)]
#[repr(transparent)]
struct Pair(TestImagePair);

// SAFETY: its two handles, in field order, come with their references from `PairInfo`, and
// the pass only reads through them.
unsafe impl ReferencingData for Pair {
    const REFERENCES: usize = 2;
    const WRITES: u64 = 0;
    const UNORDERED: u64 = 0;
}

struct PairInfo<'a, S, T>
where
    S: StorageView + 'a,
    T: SampledView + 'a,
{
    storage: S,
    sampled: T,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a, S, T> ReferenceInfo<'a> for PairInfo<'a, S, T>
where
    S: StorageView + 'a,
    T: SampledView + 'a,
{
    type Data = Pair;
    fn with_output<R>(self, f: impl FnOnce(&Pair, &[Reference<'a>]) -> R) -> R {
        let (storage, sampled) = (self.storage.view(), self.sampled.view());
        let pair = Pair(TestImagePair {
            storage: storage.handle(),
            sampled: sampled.handle(),
        });
        f(
            &pair,
            &[Reference::image(storage), Reference::image(sampled)],
        )
    }
}

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
    pairs: &Buffer<Pair, impl lava::buffer::Memory>,
    out: &Buffer<Vec4, lava::buffer::Readback>,
) {
    let bindings = TestImage2d::push_bindings(
        TestImage2dArgs {
            image: top.whole_view(),
            images: pairs.range(..).cast::<TestImagePair>(),
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
    let mut pairs: Buffer<Pair, Upload> = Buffer::upload(1).unwrap();
    pairs.host_write_referencing(
        0,
        [PairInfo {
            storage: storage.whole_view(),
            sampled: sampled.whole_view(),
            _marker: Default::default(),
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
    let mut pairs = Buffer::<Pair>::new(1).unwrap();
    gpu.uploader()
        .write_referencing(
            &pairs,
            0,
            [PairInfo {
                storage: storage.whole_view(),
                sampled: sampled.whole_view(),
                _marker: Default::default(),
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
    let mut pairs: Buffer<Pair, Upload> = Buffer::upload(1).unwrap();
    pairs.host_write_referencing(
        0,
        [PairInfo {
            storage: storage.whole_view(),
            sampled: sampled.whole_view(),
            _marker: Default::default(),
        }],
    );
    let message = panic_message(|| gpu.submit(|cmd| cmd.fill_buffer(pairs.range(..), 0)));
    assert!(
        message.contains("is written with `host_write_referencing`"),
        "{message}"
    );
}

//! GPU tests of lava's lifecycle in one process: a failed init destroys what it made and can be retried, `shutdown` refuses while objects live and then destroys everything, every call panics after it, and `init` works again; within each context, how submissions keep what they use until `reclaim` or a dropped timeline lets it go, host waits with timeouts and on several submissions, and typed errors
mod common;

use std::time::Duration;

use common::panic_message;
use glam::UVec2;
use lava::{
    Config, Ctx, Error,
    bindless::Sampled,
    buffer::{Buffer, Readback},
    image::{Image, format::R8G8B8A8Unorm, usage::usage},
    pipeline::{PassEntry, PassKind, PassMap},
    queue::{Graphics, Timeline, Wait},
};
use lava_tests::bindings::*;

fn config(passes: &'static PassMap) -> Config {
    Config {
        validation: true,
        app_name: Some(c"lava-tests"),
        app_version: [0, 1, 0],
        ..Config::new(passes)
    }
}

#[track_caller]
fn no_validation_errors() {
    let errors = common::take_validation_errors();
    assert!(
        errors.is_empty(),
        "Vulkan validation errors:\n{}",
        errors.join("\n\n")
    );
}

type Texture = Image<R8G8B8A8Unorm, usage![Sampled]>;

/// What one context does: typed errors, timelines and destruction, waits, and a refused
/// shutdown.
fn exercise() {
    let mut timeline = Timeline::<Graphics>::new(1, 0).unwrap();
    // Named for debuggers and allocator reports.
    let mut readback = Buffer::<u32, Readback>::readback(4)
        .unwrap()
        .named("readback");
    drop(
        Texture::new(UVec2::splat(4))
            .unwrap()
            .named("named texture"),
    );

    // An image a submission uses lives until that submission completed and something let it
    // go: `reclaim` polls and does, and its slot is the first free one again.
    type Cleared = Image<R8G8B8A8Unorm, usage![Sampled, TransferDst]>;
    let cleared = || {
        Cleared::new(UVec2::splat(4))
            .unwrap()
            .bind_sampled()
            .unwrap()
    };
    let image = cleared();
    let slot = image.handle::<Sampled>().sampled_slot().unwrap();
    let filled = (timeline.begin().unwrap())
        .submit(|cmd| {
            cmd.clear_image(image.whole_view(), [0.0; 4]);
            cmd.fill_buffer(readback.range(..), 7);
        })
        .unwrap();
    drop(image);
    let other_slot = |image: &Cleared| image.handle::<Sampled>().sampled_slot().unwrap();
    let kept = cleared();
    assert_ne!(other_slot(&kept), slot, "the submission keeps the image");
    drop(kept);
    std::thread::sleep(Duration::from_millis(200));
    lava::reclaim();
    assert_eq!(other_slot(&cleared()), slot, "reclaim freed the slot");
    filled.block_timeout(Duration::ZERO).unwrap();
    assert!(filled.block_timeout(Duration::from_secs(5)).unwrap());
    assert_eq!(readback.host(), [7; 4]);

    // A dropped timeline lets go of what its submissions kept; one never submitted holds none.
    let mut dropped = Timeline::<Graphics>::new(1, 0).unwrap();
    let image = cleared();
    let slot = other_slot(&image);
    (dropped.begin().unwrap())
        .submit(|cmd| cmd.clear_image(image.whole_view(), [0.0; 4]))
        .unwrap();
    drop(image);
    drop(dropped);
    assert_eq!(
        other_slot(&cleared()),
        slot,
        "the timeline's drop freed the slot"
    );
    let mut begun = Timeline::<Graphics>::new(2, 0).unwrap();
    drop(begun.begin().unwrap());
    drop(begun);

    // Several submissions at once, and none.
    let mut other = Timeline::<Graphics>::new(1, 0).unwrap();
    let a = (timeline.begin().unwrap()).submit(|_| {}).unwrap();
    let b = (other.begin().unwrap()).submit(|_| {}).unwrap();
    Wait::all(&[a, b, a]).unwrap();
    assert!(a.is_done() && b.is_done());
    assert!(Wait::all_timeout(&[] as &[Wait], Duration::ZERO).unwrap());

    // With objects alive shutdown destroys nothing, and lava keeps working.
    assert!(
        matches!(lava::shutdown(), Err(Error::ObjectsAlive { count }) if count >= 3),
        "a timeline, a buffer and an image are alive"
    );
    let filled = (timeline.begin().unwrap())
        .submit(|cmd| cmd.fill_buffer(readback.range(..), 9))
        .unwrap();
    filled.block().unwrap();
    assert_eq!(readback.host(), [9; 4]);
    lava::wait_idle().unwrap();
}

/// A pass whose "SPIR-V" lava refuses after it made the device.
static BROKEN: PassMap = PassMap {
    passes: &[PassEntry {
        name: "Broken",
        spirv: &[3, 2, 1],
        kind: PassKind::Compute {
            entry: c"main",
            subgroup_size: None,
        },
        constants: &[],
        constants_size: 0,
    }],
};

#[test]
fn shutdown_destroys_everything_and_init_works_again() {
    common::capture_validation_errors();
    let failed = lava::init(&config(&BROKEN));
    assert!(matches!(failed, Err(Error::Other { .. })), "{failed:?}");
    let message = panic_message(Ctx::features);
    assert!(message.contains("lava is not initialised"), "{message}");
    no_validation_errors();

    for _ in 0..2 {
        lava::init(&config(&PASS_MAP)).unwrap();
        assert!(
            lava::init(&config(&PASS_MAP)).is_err(),
            "it is initialised already"
        );
        exercise();
        lava::shutdown().unwrap();
        no_validation_errors();

        let message = panic_message(|| Buffer::<u32>::new(1));
        assert!(message.contains("lava is not initialised"), "{message}");
        panic_message(lava::wait_idle);
        assert!(lava::shutdown().is_ok(), "shutting down twice does nothing");
    }
}

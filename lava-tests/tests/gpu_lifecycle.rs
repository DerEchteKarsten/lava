//! GPU tests of lava's lifecycle in one process: a failed init destroys what it made and can be retried, `shutdown` refuses while objects live and then destroys everything, every call panics after it, and `init` works again; within each context, what dropped timelines and `reclaim` do to destruction, host waits with timeouts and on several submissions, and typed errors
mod common;

use std::time::Duration;

use common::panic_message;
use glam::UVec2;
use lava::{
    Config, Error,
    bindless::{self, Heap, Sampled},
    buffer::{Buffer, Readback},
    image::{Image, format::R8G8B8A8Unorm, usage::usage},
    pipeline::{PassEntry, PassKind, PassMap},
    queue::{Graphics, Timeline, Wait},
    state::Ctx,
};
use lava_tests::bindings::PASS_MAP;

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

fn sampled_slot(image: &Texture) -> u32 {
    image.handle::<Sampled>().sampled_slot().expect("bound")
}

/// What one context does: typed errors, timelines and destruction, waits, and a refused
/// shutdown.
fn exercise() {
    let empty = bindless::reserve::<Sampled>(0).unwrap();
    assert!(empty.is_empty() && empty.slots().is_empty());
    let full = bindless::reserve::<Sampled>(1 << 24);
    assert!(
        matches!(
            full,
            Err(Error::OutOfBindlessSlots {
                heap: Heap::Image,
                ..
            })
        ),
        "{full:?}"
    );

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

    // `reclaim` polls: an image dropped while a submission runs is destroyed by it alone once
    // the GPU finished, and its slot is the first free one again.
    let image = Texture::new(UVec2::splat(4))
        .unwrap()
        .bind_sampled()
        .unwrap();
    let slot = sampled_slot(&image);
    let filled = (timeline.begin().unwrap())
        .submit(|cmd| cmd.fill_buffer(readback.range(..), 7))
        .unwrap();
    drop(image);
    std::thread::sleep(Duration::from_millis(200));
    lava::reclaim();
    let again = Texture::new(UVec2::splat(4))
        .unwrap()
        .bind_sampled()
        .unwrap();
    assert_eq!(sampled_slot(&again), slot, "reclaim freed the slot");
    filled.block_timeout(Duration::ZERO).unwrap();
    assert!(filled.block_timeout(Duration::from_secs(5)).unwrap());
    assert_eq!(readback.host(), [7; 4]);

    // A timeline dropped with its stamp ahead of its last submission (`reserve`, or a `begin`
    // never submitted) signals it, so what was dropped meanwhile doesn't wait forever.
    let reserved = Timeline::<Graphics>::new(1, 0).unwrap();
    reserved.reserve();
    drop(Texture::new(UVec2::splat(4)).unwrap());
    drop(reserved);
    assert!(
        lava::latest_stamp().is_done(),
        "the dropped timeline's stamp is signalled"
    );
    let mut begun = Timeline::<Graphics>::new(2, 0).unwrap();
    drop(begun.begin().unwrap());
    drop(begun);
    assert!(lava::latest_stamp().is_done());

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
        kind: PassKind::Compute { entry: c"main" },
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

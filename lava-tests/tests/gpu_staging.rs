//! GPU tests of the staging rings: a context with `rebar` disabled, so the harness timeline has a 1 MiB ring that uploaders fill, `drain_uploads` records (ordered like any command, wrapping around, requeued after a failed recording, skipping dropped targets), a full ring fails or waits for a drain, several threads share one ring, and a transfer timeline's uploader releases its images to graphics
mod common;

use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

use common::{Cmd, gpu, panic_message, zeroed_buffer};
use glam::UVec2;
use lava::{
    Ctx, Error, Features,
    buffer::{Buffer, Readback, UploadStats},
    image::{Image, MipChain, format::R8G8B8A8Unorm, usage::usage},
    queue::{Graphics, Kind, Timeline, Transfer},
};
use lava_tests::bindings::*;

/// The harness timeline's ring.
const RING: u64 = 1 << 20;

fn gpu_with_ring() -> common::GpuGuard {
    common::disable(Features {
        rebar: true,
        ..Features::NONE
    });
    common::staging(RING);
    let gpu = gpu();
    assert!(!Ctx::features().rebar && gpu.uploader().stages());
    gpu
}

/// `dst[i] = src[i] + i` for the 4 elements of `src`.
fn plus_index(cmd: &mut Cmd, src: &Buffer<u32>, dst: &Buffer<u32, Readback>) {
    cmd.compute(
        &TestComputeBuffer::push_bindings(
            TestComputeBufferArgs {
                src: src.range(..),
                dst: dst.range(..),
                count: 4,
                multiplier: 1,
            },
            None,
        ),
        [1, 1, 1],
    );
}

fn pattern(width: u32, height: u32) -> Vec<u8> {
    (0..height)
        .flat_map(|y| {
            (0..width)
                .flat_map(move |x| [(x * 31) as u8, (y * 29) as u8, (x + y * width) as u8, 255])
        })
        .collect()
}

/// A write is copied where the ring is drained: a dispatch before the drain reads the old
/// data, one after it the new, a submission without a drain writes nothing, and a dropped
/// target's copy is skipped.
#[test]
fn writes_land_where_the_ring_is_drained() {
    let mut gpu = gpu_with_ring();
    let uploader = gpu.uploader();
    let buffer = Buffer::<u32>::new(4).unwrap();
    let outs: [_; 4] = std::array::from_fn(|_| zeroed_buffer::<u32>(4));
    let [mut before, mut after, mut skipped, mut again] = outs;
    gpu.submit(|cmd| cmd.fill_buffer(buffer.range(..), 0));
    uploader.write(&buffer, 0, &[1, 2, 3, 4]).unwrap();
    assert_eq!(uploader.pending(), 1);
    // A submission that doesn't drain writes nothing.
    gpu.submit(|cmd| plus_index(cmd, &buffer, &skipped));
    assert_eq!(skipped.host(), [0, 1, 2, 3]);
    assert_eq!(uploader.pending(), 1);
    // The dispatch before the drain reads the old data, the one after it the new.
    let mut stats = UploadStats::default();
    gpu.submit(|cmd| {
        plus_index(cmd, &buffer, &before);
        stats = cmd.drain_uploads();
        plus_index(cmd, &buffer, &after);
    });
    assert_eq!(before.host(), [0, 1, 2, 3]);
    assert_eq!(after.host(), [1, 3, 5, 7]);
    assert_eq!((stats.copies, stats.bytes, stats.capacity), (1, 16, RING));
    assert_eq!(stats.used, 16, "in flight until the next reclaim");
    assert_eq!(uploader.pending(), 0);

    // A dropped target's copy stays published, but the drain skips it: nothing else holds
    // the buffer.
    uploader.write(&buffer, 2, &[30, 40]).unwrap();
    let dropped = Buffer::<u32>::new(4).unwrap();
    uploader.write(&dropped, 0, &[5; 4]).unwrap();
    drop(dropped);
    assert_eq!(uploader.pending(), 2);
    gpu.submit(|cmd| {
        cmd.fill_buffer(buffer.range(..), 9);
        assert_eq!(cmd.drain_uploads().copies, 1);
        plus_index(cmd, &buffer, &again);
    });
    assert_eq!(again.host(), [9, 10, 32, 43]);
    // Nothing in flight once the GPU is done: the next drain finds the ring empty.
    gpu.submit(|cmd| assert_eq!(cmd.drain_uploads().used, 0));
}

/// A recording that panics after draining publishes the copies again, and the next drain
/// records them.
#[test]
fn a_failed_recording_publishes_its_copies_again() {
    let mut gpu = gpu_with_ring();
    let uploader = gpu.uploader();
    let buffer = Buffer::<u32>::new(4).unwrap();
    let mut out = zeroed_buffer::<u32>(4);
    uploader.write(&buffer, 0, &[1, 2, 3, 4]).unwrap();
    let message = panic_message(|| {
        gpu.submit(|cmd| {
            assert_eq!(cmd.drain_uploads().copies, 1);
            panic!("after the drain");
        })
    });
    assert!(message.contains("after the drain"), "{message}");
    assert_eq!(uploader.pending(), 1, "published again");
    gpu.submit(|cmd| {
        assert_eq!(cmd.drain_uploads().copies, 1);
        cmd.copy_buffer(buffer.range(..), out.range(..));
    });
    assert_eq!(out.host(), [1, 2, 3, 4]);
}

/// Writes of 300 KiB wrap around the 1 MiB ring over several submissions, each freeing its
/// space once it completed.
#[test]
fn the_ring_wraps_around_over_submissions() {
    let mut gpu = gpu_with_ring();
    let uploader = gpu.uploader();
    let len = 300 * 1024 / 4;
    let buffer = Buffer::<u32>::new(len).unwrap();
    let mut readback = zeroed_buffer::<u32>(len);
    for frame in 0..10u32 {
        let data: Vec<u32> = (0..len as u32).map(|i| i * 3 + frame).collect();
        uploader.write(&buffer, 0, &data).unwrap();
        gpu.submit(|cmd| {
            let stats = cmd.drain_uploads();
            assert_eq!((stats.copies, stats.bytes), (1, data.len() as u64 * 4));
            assert!(stats.used <= RING, "{stats:?}");
            cmd.copy_buffer(buffer.range(..), readback.range(..));
        });
        assert_eq!(readback.host(), data, "frame {frame}");
    }
}

/// A full ring fails a write right away with a zero timeout or one larger than the ring, and
/// a blocking write goes through once another thread drains. Writes fail once the timeline
/// is gone.
#[test]
fn a_full_ring_fails_or_waits_for_a_drain() {
    let mut gpu = gpu_with_ring();
    let uploader = gpu.uploader().with_timeout(Duration::ZERO);
    let half = vec![1u8; 600 * 1024];
    let buffer = Buffer::<u8>::new(half.len()).unwrap();
    let other = Buffer::<u8>::new(half.len()).unwrap();
    uploader.write(&buffer, 0, &half).unwrap();
    let full = uploader.write(&other, 0, &half).unwrap_err();
    assert!(
        matches!(full, Error::StagingFull { requested, capacity } if requested == half.len() as u64 && capacity == RING),
        "{full}"
    );
    let patient = gpu.uploader().with_timeout(Duration::from_secs(5));
    let huge = Buffer::<u8>::new(2 * RING as usize).unwrap();
    let started = Instant::now();
    let too_large = patient
        .write(&huge, 0, &vec![0; 2 * RING as usize])
        .unwrap_err();
    assert!(
        matches!(too_large, Error::StagingFull { .. })
            && started.elapsed() < Duration::from_secs(1),
        "{too_large}"
    );

    // Another thread's blocking write waits for the drain here.
    let written = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let (patient, other, written) = (&patient, &other, &written);
        scope.spawn(move || {
            patient.write(other, 0, &vec![2u8; 600 * 1024]).unwrap();
            written.store(true, Ordering::SeqCst);
        });
        std::thread::sleep(Duration::from_millis(100));
        assert!(!written.load(Ordering::SeqCst), "it waits for space");
        gpu.submit(|cmd| assert_eq!(cmd.drain_uploads().copies, 1));
    });
    assert!(written.load(Ordering::SeqCst));
    let (mut out, mut other_out) = (
        zeroed_buffer::<u8>(half.len()),
        zeroed_buffer::<u8>(half.len()),
    );
    gpu.submit(|cmd| {
        assert_eq!(cmd.drain_uploads().copies, 1);
        cmd.copy_buffer(buffer.range(..), out.range(..));
        cmd.copy_buffer(other.range(..), other_out.range(..));
    });
    assert_eq!(out.host(), half);
    assert_eq!(other_out.host(), vec![2u8; half.len()]);

    let timeline = Timeline::<Graphics>::new(1, RING).unwrap();
    let orphan = timeline.uploader();
    drop(timeline);
    assert!(
        orphan
            .write(&buffer, 0, &[1])
            .unwrap_err()
            .to_string()
            .contains("dropped")
    );
}

/// Writes from four threads through cloned uploaders all land, buffers made there included.
#[test]
fn uploaders_write_from_several_threads() {
    let mut gpu = gpu_with_ring();
    let uploader = gpu.uploader();
    const THREADS: usize = 4;
    const WRITES: usize = 16;
    const CHUNK: usize = 256;
    let buffer = Buffer::<u32>::new(THREADS * WRITES * CHUNK).unwrap();
    let made: Vec<Buffer<u32>> = std::thread::scope(|scope| {
        let threads: Vec<_> = (0..THREADS)
            .map(|t| {
                let (uploader, buffer) = (uploader.clone(), &buffer);
                scope.spawn(move || {
                    for w in 0..WRITES {
                        let offset = (t * WRITES + w) * CHUNK;
                        let data: Vec<u32> = (0..CHUNK as u32).map(|i| offset as u32 + i).collect();
                        uploader.write(buffer, offset, &data).unwrap();
                    }
                    uploader.from_slice(&[t as u32; 64]).unwrap()
                })
            })
            .collect();
        threads.into_iter().map(|t| t.join().unwrap()).collect()
    });
    let mut out = zeroed_buffer::<u32>(buffer.len());
    let mut made_out: Vec<_> = (0..THREADS).map(|_| zeroed_buffer::<u32>(64)).collect();
    gpu.submit(|cmd| {
        let stats = cmd.drain_uploads();
        assert!(stats.copies >= THREADS * WRITES, "{stats:?}");
        cmd.copy_buffer(buffer.range(..), out.range(..));
        for (made, out) in made.iter().zip(&made_out) {
            cmd.copy_buffer(made.range(..), out.range(..));
        }
    });
    let expected: Vec<u32> = (0..buffer.len() as u32).collect();
    assert_eq!(out.host(), expected);
    for (t, out) in made_out.iter_mut().enumerate() {
        assert_eq!(out.host(), [t as u32; 64]);
    }
}

/// A transfer timeline's uploader for graphics carries a buffer, a whole image and two of
/// three mips of a chain, which the drain releases to graphics without a `release` call; an
/// image graphics owns panics on the transfer family before any state moves.
#[test]
fn transfer_uploads_go_to_graphics() {
    let mut gpu = gpu_with_ring();
    if Ctx::queue_family(Kind::Transfer).is_none() {
        eprintln!(
            "SKIPPED transfer_uploads_go_to_graphics: the device has no dedicated Transfer family"
        );
        return;
    }
    let mut transfer = Timeline::<Transfer>::new(2, RING).unwrap();
    let uploader = transfer.uploader_for::<Graphics>().unwrap();
    assert_eq!(uploader.destination(), Kind::Graphics);
    let single =
        Image::<R8G8B8A8Unorm, usage![Storage, TransferSrc, TransferDst]>::new(UVec2::splat(8))
            .unwrap();
    let chain = MipChain::<R8G8B8A8Unorm, usage![TransferSrc, TransferDst]>::with_mip_levels(
        UVec2::splat(8),
        3,
    )
    .unwrap();
    let data: Vec<u32> = (0..64).collect();
    let buffer = Buffer::<u32>::new(data.len()).unwrap();
    uploader.write(&buffer, 0, &data).unwrap();
    uploader.image(&single, 0, &pattern(8, 8)).unwrap();
    uploader.image(&chain, 0, &pattern(8, 8)).unwrap();
    uploader.image(&chain, 2, &pattern(2, 2)).unwrap();
    let uploaded = (transfer.begin().unwrap())
        .submit(|cmd| assert_eq!(cmd.drain_uploads().copies, 4))
        .unwrap();
    let mut buffer_out = zeroed_buffer::<u32>(data.len());
    let mut single_out = zeroed_buffer::<u8>(8 * 8 * 4);
    let mut mip0_out = zeroed_buffer::<u8>(8 * 8 * 4);
    let mut mip2_out = zeroed_buffer::<u8>(2 * 2 * 4);
    gpu.submit_after(&[uploaded], |cmd| {
        cmd.copy_buffer(buffer.range(..), buffer_out.range(..));
        cmd.copy_image_to_buffer(single.whole(), single_out.range(..));
        cmd.copy_image_to_buffer(chain.mip_view(0).whole(), mip0_out.range(..));
        cmd.copy_image_to_buffer(chain.mip_view(2).whole(), mip2_out.range(..));
    });
    assert_eq!(buffer_out.host(), data);
    assert_eq!(single_out.host(), pattern(8, 8));
    assert_eq!(mip0_out.host(), pattern(8, 8));
    assert_eq!(mip2_out.host(), pattern(2, 2));

    // Graphics owns `single` now: a drain of a copy into it panics on the transfer family
    // before any state moves, and the graphics timeline's uploader carries it instead.
    uploader.image(&single, 0, &vec![7; 8 * 8 * 4]).unwrap();
    let message = panic_message(|| {
        transfer.begin().unwrap().submit(|cmd| {
            cmd.drain_uploads();
        })
    });
    assert!(
        message.contains("upload it through a Graphics timeline"),
        "{message}"
    );
    assert_eq!(uploader.pending(), 1, "still published");
    gpu.uploader()
        .image(&single, 0, &vec![7; 8 * 8 * 4])
        .unwrap();
    gpu.submit(|cmd| {
        assert_eq!(cmd.drain_uploads().copies, 1);
        cmd.copy_image_to_buffer(single.whole(), single_out.range(..));
    });
    assert_eq!(single_out.host(), vec![7; 8 * 8 * 4]);
    // The transfer ring's copy into `single` goes with the timeline.
    drop(transfer);
}

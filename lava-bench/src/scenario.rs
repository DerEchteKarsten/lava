//! The `Scenario` trait each workload implements twice (idiomatic lava, hand-written Vulkan), the per-mode frame contexts that time what they submit, and helpers both modes share (meshes, the camera)

use std::time::Instant;

use glam::{Mat4, Vec2, Vec3};
use lava::query::Timestamp;
use lava::queue::QueueKind;
use lava::{CommandBuffer, QueryPool, Submission, Timeline, Wait};

use crate::raw::{RawCmd, RawQueue};

/// A workload, recorded once per frame through lava and once by hand, on separate resources.
pub trait Scenario {
    /// One lava frame: every submission goes through `ctx`'s timeline (or one the scenario
    /// owns, with `Stamped::submit`).
    fn lava(&mut self, ctx: &mut LavaCtx);
    /// The same frame by hand, through `ctx.queue` (or one the scenario owns).
    fn raw(&mut self, ctx: &mut RawCtx);
    /// The start and end timestamps of the last frame's submissions on queues the scenario
    /// owns, once they completed (the harness reads its own).
    fn extra_spans(&mut self, _lava: bool) -> Vec<[u64; 2]> {
        Vec::new()
    }
    /// Blocks until the scenario's own queues finished the last frame.
    fn block_extra(&mut self, _lava: bool) {}
    /// After a `--check` frame of both modes completed: lava's and the baseline's checksums.
    fn checksums(&mut self) -> (Vec<u32>, Vec<u32>);
}

/// What a lava frame records with: the harness's graphics timeline, and the frame's timings.
pub struct LavaCtx<'a> {
    pub graphics: &'a mut Stamped<lava::queue::Graphics>,
    /// Record `--check` checksum passes at the end of the frame.
    pub check: bool,
    pub record_ns: u64,
    pub waits: Vec<Wait>,
}

impl LavaCtx<'_> {
    /// One graphics submission, timed and timestamped.
    pub fn submit(&mut self, record: impl FnOnce(&mut CommandBuffer<lava::queue::Graphics>)) {
        self.submit_with(|_| {}, record);
    }

    /// One graphics submission whose `before` adds waits.
    pub fn submit_with(
        &mut self,
        before: impl FnOnce(&mut Submission<'_, lava::queue::Graphics>),
        record: impl FnOnce(&mut CommandBuffer<lava::queue::Graphics>),
    ) {
        let (wait, ns) = self.graphics.submit(before, record);
        self.record_ns += ns;
        self.waits.push(wait);
    }
}

/// What a raw frame records with.
pub struct RawCtx<'a> {
    pub queue: &'a mut RawQueue,
    pub check: bool,
    pub record_ns: u64,
}

impl RawCtx<'_> {
    /// One submission on the harness's graphics queue, waiting for `waits`.
    pub fn submit(
        &mut self,
        waits: &[(lava::vk::Semaphore, u64, lava::vk::PipelineStageFlags2)],
        record: impl FnOnce(&mut RawCmd),
    ) {
        let (_, ns) = self.queue.submit(waits, record);
        self.record_ns += ns;
    }
}

/// A lava timeline whose submissions write a timestamp before and after what they record, in a
/// pool per frame slot.
pub struct Stamped<K: QueueKind> {
    pub timeline: Timeline<K>,
    pools: Vec<QueryPool<Timestamp>>,
    last_slot: Option<usize>,
}

impl<K: QueueKind> Stamped<K> {
    pub fn new(timeline: Timeline<K>) -> Self {
        let pools = (0..timeline.in_flight())
            .map(|_| QueryPool::timestamps(2).unwrap())
            .collect();
        Self {
            timeline,
            pools,
            last_slot: None,
        }
    }

    /// Submits `record` after `before` (waits), with its timestamps; returns the submission's
    /// wait and the time `record` took.
    pub fn submit(
        &mut self,
        before: impl FnOnce(&mut Submission<'_, K>),
        record: impl FnOnce(&mut CommandBuffer<K>),
    ) -> (Wait, u64) {
        let mut submission = self.timeline.begin().unwrap();
        before(&mut submission);
        let slot = submission.slot();
        let pool = &self.pools[slot];
        let mut elapsed = 0;
        let wait = submission
            .submit(|cmd| {
                cmd.write_timestamp(pool, 0, lava::vk::PipelineStageFlags2::TOP_OF_PIPE);
                let start = Instant::now();
                record(cmd);
                elapsed = start.elapsed().as_nanos() as u64;
                cmd.write_timestamp(pool, 1, lava::vk::PipelineStageFlags2::BOTTOM_OF_PIPE);
            })
            .unwrap();
        self.last_slot = Some(slot);
        (wait, elapsed)
    }

    /// The timestamps of the last submission, which completed.
    pub fn span(&self) -> [u64; 2] {
        let slot = self.last_slot.expect("a submission");
        let mut out = [0u64; 2];
        assert!(self.pools[slot].read(0, &mut out).unwrap(), "the submission completed");
        out
    }
}

/// The nanoseconds between the earliest start and the latest end of `spans`.
pub fn gpu_ns(spans: &[[u64; 2]]) -> u64 {
    let start = spans.iter().map(|s| s[0]).min().unwrap();
    let end = spans.iter().map(|s| s[1]).max().unwrap();
    let period = lava::Ctx::device_properties().limits.timestamp_period as f64;
    (end.wrapping_sub(start) as f64 * period) as u64
}

/// The vertices and triangle indices of a UV sphere of radius 1.
pub fn sphere(rings: u32, segments: u32) -> (Vec<crate::bindings::MeshVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    for ring in 0..=rings {
        let v = ring as f32 / rings as f32;
        let theta = v * std::f32::consts::PI;
        for segment in 0..=segments {
            let u = segment as f32 / segments as f32;
            let phi = u * std::f32::consts::TAU;
            let normal = Vec3::new(theta.sin() * phi.cos(), theta.cos(), theta.sin() * phi.sin());
            vertices.push(crate::bindings::MeshVertex {
                position: normal,
                normal,
                uv: Vec2::new(u, v),
            });
        }
    }
    let mut indices = Vec::new();
    let row = segments + 1;
    for ring in 0..rings {
        for segment in 0..segments {
            let a = ring * row + segment;
            let b = a + row;
            // Counter-clockwise seen from outside (see `camera`).
            indices.extend([a, a + 1, b, a + 1, b + 1, b]);
        }
    }
    (vertices, indices)
}

/// A reverse-depth perspective camera at `eye` looking at the origin. Without a y flip, faces
/// counter-clockwise in the world are clockwise in Vulkan's framebuffer: lava's front faces.
pub fn camera(eye: Vec3, aspect: f32) -> Mat4 {
    let projection = Mat4::perspective_infinite_reverse_rh(1.0, aspect, 0.1);
    projection * Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y)
}

/// A deterministic float in [0, 1) from `seed`.
pub fn rand01(seed: u32) -> f32 {
    let mut x = seed.wrapping_mul(0x9E37_79B9) ^ 0x85EB_CA6B;
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846C_A68B);
    x ^= x >> 16;
    (x >> 8) as f32 / 16_777_216.0
}

/// `n` sphere instances of radius 0.3 to 0.8 scattered over a square of `size` around the
/// origin, up to 4 above the ground.
pub fn scatter(n: u32, size: f32) -> Vec<crate::bindings::DrawInstance> {
    (0..n)
        .map(|i| {
            let position = Vec3::new(
                (rand01(i * 4) - 0.5) * size,
                rand01(i * 4 + 1) * 4.0,
                (rand01(i * 4 + 2) - 0.5) * size,
            );
            let scale = Vec3::splat(0.3 + rand01(i * 4 + 3) * 0.5);
            crate::bindings::DrawInstance {
                model: Mat4::from_scale_rotation_translation(scale, Default::default(), position),
                color: glam::Vec4::new(rand01(i * 7), rand01(i * 7 + 1), rand01(i * 7 + 2), 1.0),
            }
        })
        .collect()
}

//! `ml` and `ml_train`: a 12-layer transformer step (layer norm, QKV, causal attention, projection, MLP, residuals: 12 dispatches a layer), its activations sub-ranges of one arena buffer; `ml_train` adds the backward pass (18 dispatches a layer, weight gradients accumulated) and an Adam step that writes the weights the next frame reads

use std::ops::Range;

use lava::vk;
use lava::{Buffer, BufferSlice};

use crate::bindings::*;
use crate::check::Checksums;
use crate::pipelines;
use crate::raw::{self, RawCmd};
use crate::scenario::{LavaCtx, RawCtx, Scenario, rand01};
use crate::scenarios::Setup;

const T: u32 = 128;
const D: u32 = 128;
const H: u32 = 4;
const DH: u32 = D / H;
const F: u32 = 4 * D;
const LAYERS: u32 = 12;

const TD: u32 = T * D;
const TF: u32 = T * F;
const TT: u32 = T * T;

/// The activations of one layer, in floats from its block's start.
mod act {
    use super::*;
    pub const X: u32 = 0;
    pub const XN1: u32 = X + TD;
    pub const QKV: u32 = XN1 + TD;
    pub const SCORES: u32 = QKV + 3 * TD;
    pub const ATT: u32 = SCORES + super::H * TT;
    pub const PROJ: u32 = ATT + TD;
    pub const X_MID: u32 = PROJ + TD;
    pub const XN2: u32 = X_MID + TD;
    pub const H_PRE: u32 = XN2 + TD;
    pub const HIDDEN: u32 = H_PRE + TF;
    pub const MLP: u32 = HIDDEN + TF;
    pub const SIZE: u32 = MLP + TD;
}

/// The weights of one layer.
mod weight {
    use super::*;
    pub const QKV: u32 = 0;
    pub const O: u32 = QKV + 3 * D * D;
    pub const UP: u32 = O + D * D;
    pub const DOWN: u32 = UP + D * F;
    pub const G1: u32 = DOWN + F * D;
    pub const B1: u32 = G1 + D;
    pub const G2: u32 = B1 + D;
    pub const B2: u32 = G2 + D;
    pub const SIZE: u32 = B2 + D;
}

/// The backward pass's scratch gradients, reused by every layer.
mod grad {
    use super::*;
    pub const G: [u32; 2] = [0, TD];
    pub const DHIDDEN: u32 = 2 * TD;
    pub const DXN2: u32 = DHIDDEN + TF;
    pub const DX_MID: u32 = DXN2 + TD;
    pub const DATT: u32 = DX_MID + TD;
    pub const DSCORES: u32 = DATT + TD;
    pub const DQKV: u32 = DSCORES + H * TT;
    pub const DXN1: u32 = DQKV + 3 * TD;
    pub const SIZE: u32 = DXN1 + TD;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Buf {
    Acts,
    Weights,
    Grads,
    Scratch,
    Zeros,
}

/// `len` floats of one buffer from `start`.
#[derive(Clone, Copy, Debug)]
struct Span {
    buffer: Buf,
    start: u32,
    len: u32,
}

impl Span {
    fn range(&self) -> Range<u32> {
        self.start..self.start + self.len
    }

    fn overlaps(&self, other: &Span) -> bool {
        self.buffer == other.buffer
            && self.start < other.start + other.len
            && other.start < self.start + self.len
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Matmul,
    Softmax,
    Layernorm,
    Gelu,
    Add,
    GeluBack,
    SoftmaxBack,
    LayernormBack,
}

/// One dispatch of an `ml.slang` pass: its operands, its parameters and its workgroups.
#[derive(Clone, Copy, Debug)]
struct Op {
    kind: Kind,
    a: Span,
    b: Span,
    c: Span,
    bias: Span,
    params: [u32; 10],
    scale: f32,
    groups: [u32; 3],
    /// Whether the baseline records a barrier before it (`plan_barriers`).
    barrier: bool,
}

const TRANS_A: u32 = 1;
const TRANS_B: u32 = 2;
const ACCUMULATE: u32 = 4;
const CAUSAL: u32 = 8;

fn zeros() -> Span {
    Span {
        buffer: Buf::Zeros,
        start: 0,
        len: 256,
    }
}

/// `C[m x n] (+)= op(A)[m x k] * op(B)[k x n]`, `batch` times.
struct Gemm {
    a: Span,
    b: Span,
    c: Span,
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldb: u32,
    ldc: u32,
    batch: u32,
    strides: [u32; 3],
    flags: u32,
    scale: f32,
}

fn matmul(g: Gemm) -> Op {
    let [a_batch, b_batch, c_batch] = g.strides;
    Op {
        kind: Kind::Matmul,
        a: g.a,
        b: g.b,
        c: g.c,
        bias: zeros(),
        params: [g.m, g.n, g.k, g.lda, g.ldb, g.ldc, a_batch, b_batch, c_batch, g.flags],
        scale: g.scale,
        groups: [g.n.div_ceil(16), g.m.div_ceil(16), g.batch],
        barrier: false,
    }
}

/// A row-wise pass over `rows` rows of `n` (one workgroup each).
fn rows(kind: Kind, a: Span, b: Span, c: Span, bias: Span, rows: u32, n: u32, flags: u32) -> Op {
    Op {
        kind,
        a,
        b,
        c,
        bias,
        params: [rows, n, 0, 0, 0, n, 0, 0, 0, flags],
        scale: 1.0,
        groups: [rows, 1, 1],
        barrier: false,
    }
}

/// An element-wise pass over `count` elements.
fn elementwise(kind: Kind, a: Span, c: Span, bias: Span, count: u32) -> Op {
    Op {
        kind,
        a,
        b: zeros(),
        c,
        bias,
        params: [count, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        scale: 1.0,
        groups: [count.div_ceil(256), 1, 1],
        barrier: false,
    }
}

fn acts(layer: u32, offset: u32, len: u32) -> Span {
    Span {
        buffer: Buf::Acts,
        start: layer * act::SIZE + offset,
        len,
    }
}

fn weights(layer: u32, offset: u32, len: u32) -> Span {
    Span {
        buffer: Buf::Weights,
        start: layer * weight::SIZE + offset,
        len,
    }
}

fn grads(layer: u32, offset: u32, len: u32) -> Span {
    Span {
        buffer: Buf::Grads,
        ..weights(layer, offset, len)
    }
}

fn scratch(offset: u32, len: u32) -> Span {
    Span {
        buffer: Buf::Scratch,
        start: offset,
        len,
    }
}

/// The forward pass's dispatches.
fn forward(ops: &mut Vec<Op>) {
    let scale = 1.0 / (DH as f32).sqrt();
    for l in 0..LAYERS {
        let qkv = |column: u32| acts(l, act::QKV + column, 3 * TD - column);
        ops.push(rows(Kind::Layernorm, acts(l, act::X, TD), weights(l, weight::G1, D), acts(l, act::XN1, TD), weights(l, weight::B1, D), T, D, 0));
        ops.push(matmul(Gemm { a: acts(l, act::XN1, TD), b: weights(l, weight::QKV, 3 * D * D), c: qkv(0), m: T, n: 3 * D, k: D, lda: D, ldb: 3 * D, ldc: 3 * D, batch: 1, strides: [0; 3], flags: 0, scale: 1.0 }));
        ops.push(matmul(Gemm { a: qkv(0), b: qkv(D), c: acts(l, act::SCORES, H * TT), m: T, n: T, k: DH, lda: 3 * D, ldb: 3 * D, ldc: T, batch: H, strides: [DH, DH, TT], flags: TRANS_B, scale }));
        ops.push(rows(Kind::Softmax, zeros(), zeros(), acts(l, act::SCORES, H * TT), zeros(), H * T, T, CAUSAL));
        ops.push(matmul(Gemm { a: acts(l, act::SCORES, H * TT), b: qkv(2 * D), c: acts(l, act::ATT, TD), m: T, n: DH, k: T, lda: T, ldb: 3 * D, ldc: D, batch: H, strides: [TT, DH, DH], flags: 0, scale: 1.0 }));
        ops.push(matmul(Gemm { a: acts(l, act::ATT, TD), b: weights(l, weight::O, D * D), c: acts(l, act::PROJ, TD), m: T, n: D, k: D, lda: D, ldb: D, ldc: D, batch: 1, strides: [0; 3], flags: 0, scale: 1.0 }));
        ops.push(elementwise(Kind::Add, acts(l, act::X, TD), acts(l, act::X_MID, TD), acts(l, act::PROJ, TD), TD));
        ops.push(rows(Kind::Layernorm, acts(l, act::X_MID, TD), weights(l, weight::G2, D), acts(l, act::XN2, TD), weights(l, weight::B2, D), T, D, 0));
        ops.push(matmul(Gemm { a: acts(l, act::XN2, TD), b: weights(l, weight::UP, D * F), c: acts(l, act::H_PRE, TF), m: T, n: F, k: D, lda: D, ldb: F, ldc: F, batch: 1, strides: [0; 3], flags: 0, scale: 1.0 }));
        ops.push(elementwise(Kind::Gelu, acts(l, act::H_PRE, TF), acts(l, act::HIDDEN, TF), zeros(), TF));
        ops.push(matmul(Gemm { a: acts(l, act::HIDDEN, TF), b: weights(l, weight::DOWN, F * D), c: acts(l, act::MLP, TD), m: T, n: D, k: F, lda: F, ldb: D, ldc: D, batch: 1, strides: [0; 3], flags: 0, scale: 1.0 }));
        ops.push(elementwise(Kind::Add, acts(l, act::X_MID, TD), acts(l + 1, act::X, TD), acts(l, act::MLP, TD), TD));
    }
}

/// The backward pass's dispatches, the loss gradient taken to be the output itself.
fn backward(ops: &mut Vec<Op>) {
    let scale = 1.0 / (DH as f32).sqrt();
    let mut g = acts(LAYERS, act::X, TD);
    for (i, l) in (0..LAYERS).rev().enumerate() {
        let qkv = |column: u32| acts(l, act::QKV + column, 3 * TD - column);
        let dqkv = |column: u32| scratch(grad::DQKV + column, 3 * TD - column);
        let next = scratch(grad::G[i % 2], TD);
        ops.push(matmul(Gemm { a: acts(l, act::HIDDEN, TF), b: g, c: grads(l, weight::DOWN, F * D), m: F, n: D, k: T, lda: F, ldb: D, ldc: D, batch: 1, strides: [0; 3], flags: TRANS_A | ACCUMULATE, scale: 1.0 }));
        ops.push(matmul(Gemm { a: g, b: weights(l, weight::DOWN, F * D), c: scratch(grad::DHIDDEN, TF), m: T, n: F, k: D, lda: D, ldb: D, ldc: F, batch: 1, strides: [0; 3], flags: TRANS_B, scale: 1.0 }));
        ops.push(elementwise(Kind::GeluBack, acts(l, act::H_PRE, TF), scratch(grad::DHIDDEN, TF), zeros(), TF));
        ops.push(matmul(Gemm { a: acts(l, act::XN2, TD), b: scratch(grad::DHIDDEN, TF), c: grads(l, weight::UP, D * F), m: D, n: F, k: T, lda: D, ldb: F, ldc: F, batch: 1, strides: [0; 3], flags: TRANS_A | ACCUMULATE, scale: 1.0 }));
        ops.push(matmul(Gemm { a: scratch(grad::DHIDDEN, TF), b: weights(l, weight::UP, D * F), c: scratch(grad::DXN2, TD), m: T, n: D, k: F, lda: F, ldb: F, ldc: D, batch: 1, strides: [0; 3], flags: TRANS_B, scale: 1.0 }));
        ops.push(rows(Kind::LayernormBack, acts(l, act::X_MID, TD), weights(l, weight::G2, D), scratch(grad::DXN2, TD), zeros(), T, D, 0));
        ops.push(elementwise(Kind::Add, g, scratch(grad::DX_MID, TD), scratch(grad::DXN2, TD), TD));
        ops.push(matmul(Gemm { a: acts(l, act::ATT, TD), b: scratch(grad::DX_MID, TD), c: grads(l, weight::O, D * D), m: D, n: D, k: T, lda: D, ldb: D, ldc: D, batch: 1, strides: [0; 3], flags: TRANS_A | ACCUMULATE, scale: 1.0 }));
        ops.push(matmul(Gemm { a: scratch(grad::DX_MID, TD), b: weights(l, weight::O, D * D), c: scratch(grad::DATT, TD), m: T, n: D, k: D, lda: D, ldb: D, ldc: D, batch: 1, strides: [0; 3], flags: TRANS_B, scale: 1.0 }));
        ops.push(matmul(Gemm { a: scratch(grad::DATT, TD), b: qkv(2 * D), c: scratch(grad::DSCORES, H * TT), m: T, n: T, k: DH, lda: D, ldb: 3 * D, ldc: T, batch: H, strides: [DH, DH, TT], flags: TRANS_B, scale: 1.0 }));
        ops.push(rows(Kind::SoftmaxBack, acts(l, act::SCORES, H * TT), zeros(), scratch(grad::DSCORES, H * TT), zeros(), H * T, T, 0));
        ops.push(matmul(Gemm { a: scratch(grad::DSCORES, H * TT), b: qkv(D), c: dqkv(0), m: T, n: DH, k: T, lda: T, ldb: 3 * D, ldc: 3 * D, batch: H, strides: [TT, DH, DH], flags: 0, scale }));
        ops.push(matmul(Gemm { a: scratch(grad::DSCORES, H * TT), b: qkv(0), c: dqkv(D), m: T, n: DH, k: T, lda: T, ldb: 3 * D, ldc: 3 * D, batch: H, strides: [TT, DH, DH], flags: TRANS_A, scale }));
        ops.push(matmul(Gemm { a: acts(l, act::SCORES, H * TT), b: scratch(grad::DATT, TD), c: dqkv(2 * D), m: T, n: DH, k: T, lda: T, ldb: D, ldc: 3 * D, batch: H, strides: [TT, DH, DH], flags: TRANS_A, scale: 1.0 }));
        ops.push(matmul(Gemm { a: acts(l, act::XN1, TD), b: dqkv(0), c: grads(l, weight::QKV, 3 * D * D), m: D, n: 3 * D, k: T, lda: D, ldb: 3 * D, ldc: 3 * D, batch: 1, strides: [0; 3], flags: TRANS_A | ACCUMULATE, scale: 1.0 }));
        ops.push(matmul(Gemm { a: dqkv(0), b: weights(l, weight::QKV, 3 * D * D), c: scratch(grad::DXN1, TD), m: T, n: D, k: 3 * D, lda: 3 * D, ldb: 3 * D, ldc: D, batch: 1, strides: [0; 3], flags: TRANS_B, scale: 1.0 }));
        ops.push(rows(Kind::LayernormBack, acts(l, act::X, TD), weights(l, weight::G1, D), scratch(grad::DXN1, TD), zeros(), T, D, 0));
        ops.push(elementwise(Kind::Add, scratch(grad::DX_MID, TD), next, scratch(grad::DXN1, TD), TD));
        g = next;
    }
}

/// Marks each op that reads or writes what an op since the last barrier wrote, or writes what
/// one read: the barriers a hand-written recording places, known ahead of time.
fn plan_barriers(ops: &mut [Op]) {
    let mut written: Vec<Span> = Vec::new();
    let mut read: Vec<Span> = Vec::new();
    for op in ops {
        let reads = [op.a, op.b, op.bias, op.c];
        let hazard = reads.iter().any(|r| written.iter().any(|w| w.overlaps(r)))
            || read.iter().any(|r| r.overlaps(&op.c));
        if hazard {
            op.barrier = true;
            written.clear();
            read.clear();
        }
        written.push(op.c);
        read.extend([op.a, op.b, op.bias]);
    }
}

struct Set {
    acts: Buffer<f32>,
    weights: Buffer<f32>,
    grads: Buffer<f32>,
    scratch: Buffer<f32>,
    zeros: Buffer<f32>,
    moments: Buffer<glam::Vec2>,
    step: u32,
}

impl Set {
    fn slice(&self, span: Span) -> BufferSlice<'_, f32> {
        let buffer = match span.buffer {
            Buf::Acts => &self.acts,
            Buf::Weights => &self.weights,
            Buf::Grads => &self.grads,
            Buf::Scratch => &self.scratch,
            Buf::Zeros => &self.zeros,
        };
        let range = span.range();
        buffer.range(range.start as usize..range.end as usize)
    }

    fn address(&self, span: Span) -> u64 {
        self.slice(span).address()
    }
}

pub struct Ml {
    train: bool,
    lava: Set,
    raw: Set,
    ops: Vec<Op>,
    checksums: Checksums,
    pipelines: Vec<(Kind, vk::Pipeline)>,
    adam: vk::Pipeline,
}

const WEIGHTS: u32 = LAYERS * weight::SIZE;

impl Ml {
    pub fn new(setup: &mut Setup, train: bool) -> Self {
        let mut ops = Vec::new();
        forward(&mut ops);
        if train {
            backward(&mut ops);
        }
        plan_barriers(&mut ops);
        let uploader = setup.uploader();
        let input: Vec<f32> = (0..TD).map(|i| rand01(i) * 2.0 - 1.0).collect();
        let initial: Vec<f32> = (0..WEIGHTS)
            .map(|i| {
                let offset = i % weight::SIZE;
                match offset >= weight::G1 && offset < weight::B1 || offset >= weight::G2 && offset < weight::B2 {
                    true => 1.0,
                    false if offset >= weight::G1 => 0.0,
                    false => (rand01(i + 77) - 0.5) * 0.1,
                }
            })
            .collect();
        let set = || Set {
            // Suballocated into one span per activation, every op depending on the one
            // before: tracked as whole buffers.
            acts: Buffer::new(((LAYERS + 1) * act::SIZE) as usize).unwrap().coarse(),
            weights: uploader.from_slice(&initial).unwrap(),
            grads: Buffer::new(WEIGHTS as usize).unwrap(),
            scratch: Buffer::new(grad::SIZE as usize).unwrap().coarse(),
            zeros: uploader.from_slice(&[0.0f32; 256]).unwrap(),
            moments: Buffer::new(WEIGHTS as usize).unwrap(),
            step: 0,
        };
        let ml = Self {
            train,
            lava: set(),
            raw: set(),
            ops,
            checksums: Checksums::new(2),
            pipelines: [
                (Kind::Matmul, "MlMatmul"),
                (Kind::Softmax, "MlSoftmax"),
                (Kind::Layernorm, "MlLayernorm"),
                (Kind::Gelu, "MlGelu"),
                (Kind::Add, "MlAdd"),
                (Kind::GeluBack, "MlGeluBack"),
                (Kind::SoftmaxBack, "MlSoftmaxBack"),
                (Kind::LayernormBack, "MlLayernormBack"),
            ]
            .map(|(kind, name)| (kind, pipelines::compute(name)))
            .to_vec(),
            adam: pipelines::compute("Adam"),
        };
        // The input (layer 0's x) and zeroed moments, in both modes.
        let s = &ml.lava;
        setup.lava(|cmd| {
            cmd.fill_buffer(s.moments.range(..), 0);
            cmd.update_buffer(s.acts.range(..TD as usize), &input);
        });
        let s = &ml.raw;
        let staging = uploader.from_slice(&input).unwrap();
        setup.lava(|_| {});
        setup.raw(|cmd| {
            cmd.fill(s.moments.address(), s.moments.size(), 0);
            cmd.copy(&[(staging.address(), s.acts.address(), staging.size())]);
            cmd.mem(raw::TRANSFER_WRITE, raw::COMPUTE_RW);
        });
        ml
    }
}

impl Scenario for Ml {
    fn lava(&mut self, ctx: &mut LavaCtx) {
        self.lava.step += 1;
        let (s, ops, train) = (&self.lava, &self.ops, self.train);
        let check = ctx.check;
        ctx.submit(|cmd| {
            if train {
                cmd.fill_buffer(s.grads.range(..), 0);
            }
            for op in ops {
                let [m, n, k, lda, ldb, ldc, a_batch, b_batch, c_batch, flags] = op.params;
                macro_rules! args {
                    ($args:ident) => {
                        $args {
                            a: s.slice(op.a),
                            b: s.slice(op.b),
                            c: s.slice(op.c),
                            bias: s.slice(op.bias),
                            m,
                            n,
                            k,
                            lda,
                            ldb,
                            ldc,
                            a_batch,
                            b_batch,
                            c_batch,
                            flags,
                            scale: op.scale,
                        }
                    };
                }
                match op.kind {
                    Kind::Matmul => cmd.compute(&MlMatmul::push_bindings(args!(MlMatmulArgs), None), op.groups),
                    Kind::Softmax => cmd.compute(&MlSoftmax::push_bindings(args!(MlSoftmaxArgs), None), op.groups),
                    Kind::Layernorm => cmd.compute(&MlLayernorm::push_bindings(args!(MlLayernormArgs), None), op.groups),
                    Kind::Gelu => cmd.compute(&MlGelu::push_bindings(args!(MlGeluArgs), None), op.groups),
                    Kind::Add => cmd.compute(&MlAdd::push_bindings(args!(MlAddArgs), None), op.groups),
                    Kind::GeluBack => cmd.compute(&MlGeluBack::push_bindings(args!(MlGeluBackArgs), None), op.groups),
                    Kind::SoftmaxBack => cmd.compute(&MlSoftmaxBack::push_bindings(args!(MlSoftmaxBackArgs), None), op.groups),
                    Kind::LayernormBack => cmd.compute(&MlLayernormBack::push_bindings(args!(MlLayernormBackArgs), None), op.groups),
                }
            }
            if train {
                let args = AdamArgs {
                    params: s.weights.range(..),
                    grads: s.grads.range(..),
                    moments: s.moments.range(..),
                    count: WEIGHTS,
                    lr: 1e-4,
                    correction1: 1.0 - 0.9f32.powi(s.step as i32),
                    correction2: 1.0 - 0.999f32.powi(s.step as i32),
                };
                cmd.compute(&Adam::push_bindings(args, None), Adam::groups([WEIGHTS, 1, 1]));
            }
            if check {
                self.checksums.lava_clear(cmd);
                let output = s.slice(acts(LAYERS, act::X, TD));
                self.checksums.lava_buffer(cmd, output.cast(), 0);
                self.checksums.lava_buffer(cmd, s.weights.range(..).cast(), 1);
            }
        });
    }

    fn raw(&mut self, ctx: &mut RawCtx) {
        self.raw.step += 1;
        let (s, ops, train) = (&self.raw, &self.ops, self.train);
        let pipelines = &self.pipelines;
        let adam = self.adam;
        let check = ctx.check;
        ctx.submit(&[], |cmd: &mut RawCmd| {
            // Last frame's passes (the Adam step's weights among them) come first.
            cmd.mem(raw::COMPUTE_RW, raw::or(raw::COMPUTE_RW, raw::TRANSFER_WRITE));
            if train {
                cmd.fill(s.grads.address(), s.grads.size(), 0);
                cmd.mem(raw::TRANSFER_WRITE, raw::COMPUTE_RW);
            }
            // Every `ml.slang` pass has the same push constants.
            let mut push: CMlMatmulBindings = bytemuck::Zeroable::zeroed();
            let mut bound = None;
            for op in ops {
                if op.barrier {
                    cmd.mem(raw::COMPUTE_RW, raw::COMPUTE_RW);
                }
                if bound != Some(op.kind) {
                    let pipeline = pipelines.iter().find(|(kind, _)| *kind == op.kind).unwrap().1;
                    cmd.bind_compute(pipeline);
                    bound = Some(op.kind);
                }
                let [m, n, k, lda, ldb, ldc, a_batch, b_batch, c_batch, flags] = op.params;
                push.a = s.address(op.a);
                push.b = s.address(op.b);
                push.c = s.address(op.c);
                push.bias = s.address(op.bias);
                (push.m, push.n, push.k) = (m, n, k);
                (push.lda, push.ldb, push.ldc) = (lda, ldb, ldc);
                (push.a_batch, push.b_batch, push.c_batch) = (a_batch, b_batch, c_batch);
                push.flags = flags;
                push.scale = op.scale;
                cmd.push(&push);
                cmd.dispatch(op.groups);
            }
            if train {
                cmd.mem(raw::COMPUTE_RW, raw::COMPUTE_RW);
                let mut push: CAdamBindings = bytemuck::Zeroable::zeroed();
                push.params = s.weights.address();
                push.grads = s.grads.address();
                push.moments = s.moments.address();
                push.count = WEIGHTS;
                push.lr = 1e-4;
                push.correction1 = 1.0 - 0.9f32.powi(s.step as i32);
                push.correction2 = 1.0 - 0.999f32.powi(s.step as i32);
                cmd.bind_compute(adam);
                cmd.push(&push);
                cmd.dispatch(Adam::groups([WEIGHTS, 1, 1]));
            }
            if check {
                cmd.mem(raw::COMPUTE_WRITE, raw::COMPUTE_READ);
                self.checksums.raw_clear(cmd);
                self.checksums.raw_buffer(cmd, s.address(acts(LAYERS, act::X, TD)), TD, 0);
                self.checksums.raw_buffer(cmd, s.weights.address(), WEIGHTS, 1);
                self.checksums.raw_finish(cmd);
            }
        });
    }

    fn checksums(&mut self) -> (Vec<u32>, Vec<u32>) {
        self.checksums.read()
    }
}

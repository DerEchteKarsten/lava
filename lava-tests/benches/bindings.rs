//! Microbenchmark of five `BindingOutput` layouts (today's SmallVec one, unsized with lava's access types, unsized with 20-byte compressed accesses, and unsized byte streams with compressed or lava's access types) on core's frame and three scenarios that stress binding creation; no GPU
//!
//! Run with `cargo +nightly bench -p lava-tests --bench bindings`.
//!
//! Every variant models what lava and a generated `push_bindings` do, in one crate: the
//! generated functions build the push-constant struct and the accesses, registrations append
//! accesses, the commands are `#[inline(never)]` like lava's non-generic ones behind the crate
//! boundary, and the raster builder rebuilds its draw array per draw as lava's does. Recording
//! is the same code for every variant: in "full" mode a copy of lava's barrier tracking
//! (`compute_barriers`, `take_layouts`, `PendingAccesses::record`), in "decode" mode just a
//! read of every expanded access, so the representation's own cost shows. Push constants are
//! copied into a byte stream and the pipeline is looked up like lava does it (a plain index, or
//! a hash map keyed by the specialization constants under a read lock).
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::{
    alloc::{GlobalAlloc, Layout, System},
    borrow::Borrow,
    collections::HashMap,
    hint::black_box,
    marker::PhantomData,
    ops::Deref,
    range::Range,
    slice,
    sync::{
        RwLock,
        atomic::{AtomicU32, AtomicU64, Ordering},
    },
    time::Instant,
};

use bytemuck::{Pod, Zeroable, bytes_of};
use glam::{Mat4, UVec2, Vec4};
use lava::command_buffer::{BufferAccess, ImageAccess, kind};
use lava::state::raw_vulkan::{self as vk, Handle};
use smallvec::SmallVec;

type S = vk::PipelineStageFlags2;
type A = vk::AccessFlags2;
type L = vk::ImageLayout;

const MAX_PUSH: usize = 256;
const MAX_CONSTANTS: usize = 128;

const COMPUTE: S = S::COMPUTE_SHADER;
const MESH: S = S::from_raw(
    S::TASK_SHADER_EXT.as_raw() | S::MESH_SHADER_EXT.as_raw() | S::FRAGMENT_SHADER.as_raw(),
);
const VERTEX: S = S::from_raw(S::VERTEX_SHADER.as_raw() | S::FRAGMENT_SHADER.as_raw());

const INSTANCE_CULL: usize = 0;
const BVH_CULL: usize = 1;
const RASTER: usize = 2;
const RASTER_BLENDED: usize = 3;
const COMPOSITE: usize = 4;
const RASTER_OUTLINE: usize = 5;
const DRAW_OUTLINE: usize = 6;
const GIZMOS: usize = 7;
const UI: usize = 8;
const FORWARD: usize = 9;
const DOWNSAMPLE: usize = 10;
const MIP_READ: usize = 11;
/// What lava reads from the pass map: the stages of each pass's shaders.
const PASS_STAGES: [S; 12] = [
    COMPUTE, COMPUTE, MESH, MESH, COMPUTE, MESH, COMPUTE, VERTEX, VERTEX, VERTEX, COMPUTE, COMPUTE,
];

// ---- Counting allocator: heap allocations per frame ---------------------------------------

struct Counting;
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static GLOBAL: Counting = Counting;

// ---- Inputs: images, views and buffer slices as lava hands them to generated code ---------

struct Img {
    image: vk::Image,
    /// One for a `Single` image, one per mip for a `MipChain`.
    layouts: Box<[AtomicU32]>,
    mips: u32,
    aspect: vk::ImageAspectFlags,
    handle: u64,
}

impl Img {
    fn new(id: u64, mips: u32, aspect: vk::ImageAspectFlags, per_mip: bool) -> Self {
        let count = if per_mip { mips } else { 1 };
        Self {
            image: vk::Image::from_raw(0x1000 + id),
            layouts: (0..count).map(|_| AtomicU32::new(0)).collect(),
            mips,
            aspect,
            handle: id << 8,
        }
    }

    fn whole(&self) -> View<'_> {
        View {
            image: self.image,
            layouts: &self.layouts,
            mip_range: (0..self.mips).into(),
            aspect: self.aspect,
        }
    }

    /// A `MipChain`'s `mip_view`.
    fn mip(&self, level: u32) -> View<'_> {
        View {
            layouts: slice::from_ref(&self.layouts[level as usize]),
            mip_range: (level..level + 1).into(),
            ..self.whole()
        }
    }
}

#[derive(Clone, Copy)]
struct View<'a> {
    image: vk::Image,
    layouts: &'a [AtomicU32],
    mip_range: Range<u32>,
    aspect: vk::ImageAspectFlags,
}

#[derive(Clone, Copy)]
struct Slice {
    address: u64,
    size: u64,
}

impl Slice {
    fn range(self) -> Range<u64> {
        (self.address..self.address + self.size).into()
    }
}

struct World {
    gbuffer: Img,
    depth: Img,
    accum: Img,
    revealage: Img,
    target: Img,
    swapchain: Img,
    outline_mask: Img,
    font_atlas: Img,
    chain: Img,
    textures: Vec<Img>,
    roots: Slice,
    aabbs: Slice,
    transforms: Slice,
    stack: Slice,
    variables: Slice,
    meshlets: Slice,
    blended: Slice,
    outlined: Slice,
    candidates: Slice,
    batches: Slice,
    headers: Slice,
    flags: Slice,
    materials: Slice,
    opaque_draw: Slice,
    blended_draw: Slice,
    outlined_draw: Slice,
    gizmos: Slice,
    ui_vertices: Slice,
    ui_indices: Slice,
    vertex_buffers: Vec<Slice>,
    index_buffers: Vec<Slice>,
}

impl World {
    fn new() -> Self {
        let mut next = 0x1_0000_0000u64;
        let mut slice = |size: u64| {
            let s = Slice {
                address: next,
                size,
            };
            next += size.next_multiple_of(256);
            s
        };
        let mut id = 0;
        let mut img = |mips: u32, aspect: vk::ImageAspectFlags, per_mip: bool| {
            id += 1;
            Img::new(id, mips, aspect, per_mip)
        };
        let color = vk::ImageAspectFlags::COLOR;
        Self {
            gbuffer: img(1, color, false),
            depth: img(1, vk::ImageAspectFlags::DEPTH, false),
            accum: img(1, color, false),
            revealage: img(1, color, false),
            target: img(1, color, false),
            swapchain: img(1, color, false),
            outline_mask: img(1, color, false),
            font_atlas: img(1, color, false),
            chain: img(12, color, true),
            textures: (0..64).map(|_| img(10, color, false)).collect(),
            roots: slice(1 << 16),
            aabbs: slice(1 << 16),
            transforms: slice(1 << 18),
            stack: slice(1 << 20),
            variables: slice(256),
            meshlets: slice(1 << 20),
            blended: slice(1 << 20),
            outlined: slice(1 << 20),
            candidates: slice(1 << 20),
            batches: slice(1 << 16),
            headers: slice(1 << 16),
            flags: slice(1 << 12),
            materials: slice(1 << 16),
            opaque_draw: slice(12),
            blended_draw: slice(12),
            outlined_draw: slice(12),
            gizmos: slice(1 << 18),
            ui_vertices: slice(1 << 20),
            ui_indices: slice(1 << 18),
            vertex_buffers: (0..64).map(|_| slice(1 << 16)).collect(),
            index_buffers: (0..64).map(|_| slice(1 << 14)).collect(),
        }
    }
}

/// The per-frame values, through `black_box` every frame.
#[derive(Clone, Copy)]
struct Frame {
    view: Mat4,
    proj: Mat4,
    eye: Vec4,
    clip: Mat4,
    inv_proj: Mat4,
    inv_view: Mat4,
    size: UVec2,
    exposure: f32,
    outline: Vec4,
}

impl Frame {
    fn new() -> Self {
        let view = Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 1.0]);
        let proj = Mat4::perspective_lh(1.1, 16.0 / 9.0, 0.1, 1000.0);
        Self {
            view,
            proj,
            eye: Vec4::new(1.0, 2.0, 3.0, 0.0),
            clip: proj * view,
            inv_proj: proj.inverse(),
            inv_view: view.inverse(),
            size: UVec2::new(2560, 1440),
            exposure: 1.0,
            outline: Vec4::new(1.0, 0.5, 0.0, 3.0),
        }
    }
}

fn groups(size: UVec2) -> [u32; 3] {
    [size.x.div_ceil(8), size.y.div_ceil(8), 1]
}

// ---- Push-constant and specialization-constant structs, as the generator emits them -------

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InstanceCullGpu {
    num_instances: u64,
    roots: u64,
    aabbs: u64,
    transforms: u64,
    stack: u64,
    variables: u64,
    clip_from_world: Mat4,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BvhCullGpu {
    buffers: [u64; 10],
    camera_pos: Vec4,
    proj: Mat4,
    clip_from_world: Mat4,
    window_height: f32,
    lod_bias: f32,
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BvhCullConsts {
    sizes: [u32; 5],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RasterGpu {
    view: Mat4,
    proj: Mat4,
    eye: Vec4,
    transforms: u64,
    meshlets: u64,
    draw: u64,
    materials: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CompositeGpu {
    images: [u64; 5],
    _pad: u64,
    inverse_proj: Mat4,
    inverse_view: Mat4,
    size: UVec2,
    exposure: f32,
    _pad2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DrawOutlineGpu {
    mask: u64,
    out: u64,
    color: Vec4,
    size: UVec2,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GizmosGpu {
    world_to_clip: Mat4,
    gizmos: u64,
    first: u32,
    shape: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct UiGpu {
    vertices: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ForwardGpu {
    world_to_clip: Mat4,
    vertices: u64,
    transforms: u64,
    materials: u64,
    index: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DownsampleGpu {
    src: u64,
    dst: u64,
    size: UVec2,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MipReadGpu {
    chain: u64,
    out: u64,
    mips: u32,
    _pad: u32,
}

// The byte stream variant's generated types spell these sizes as literals.
const _: () = {
    assert!(size_of::<InstanceCullGpu>() == 112);
    assert!(size_of::<BvhCullGpu>() == 240);
    assert!(size_of::<BvhCullConsts>() == 20);
    assert!(size_of::<RasterGpu>() == 176);
    assert!(size_of::<CompositeGpu>() == 192);
    assert!(size_of::<DrawOutlineGpu>() == 48);
    assert!(size_of::<GizmosGpu>() == 80);
    assert!(size_of::<UiGpu>() == 8);
    assert!(size_of::<ForwardGpu>() == 96);
    assert!(size_of::<DownsampleGpu>() == 24);
    assert!(size_of::<MipReadGpu>() == 24);
};

fn instance_cull_gpu(w: &World, f: &Frame) -> InstanceCullGpu {
    InstanceCullGpu {
        num_instances: 400,
        roots: w.roots.address,
        aabbs: w.aabbs.address,
        transforms: w.transforms.address,
        stack: w.stack.address,
        variables: w.variables.address,
        clip_from_world: f.clip,
    }
}

fn bvh_cull_gpu(w: &World, f: &Frame) -> BvhCullGpu {
    BvhCullGpu {
        buffers: [
            w.stack.address,
            w.variables.address,
            w.meshlets.address,
            w.blended.address,
            w.outlined.address,
            w.candidates.address,
            w.batches.address,
            w.transforms.address,
            w.headers.address,
            w.flags.address,
        ],
        camera_pos: f.eye,
        proj: f.proj,
        clip_from_world: f.clip,
        window_height: f.size.y as f32,
        lod_bias: 1.0,
        _pad: [0.0; 2],
    }
}

fn bvh_cull_consts(w: &World) -> BvhCullConsts {
    BvhCullConsts {
        sizes: [
            w.stack.size as u32 / 16,
            w.candidates.size as u32 / 8,
            w.meshlets.size as u32 / 8,
            w.blended.size as u32 / 8,
            w.outlined.size as u32 / 8,
        ],
    }
}

fn raster_gpu(w: &World, f: &Frame, meshlets: Slice, draw: Slice) -> RasterGpu {
    RasterGpu {
        view: f.view,
        proj: f.proj,
        eye: f.eye,
        transforms: w.transforms.address,
        meshlets: meshlets.address,
        draw: draw.address,
        materials: w.materials.address,
    }
}

fn composite_gpu(w: &World, f: &Frame) -> CompositeGpu {
    CompositeGpu {
        images: [
            w.gbuffer.handle,
            w.depth.handle,
            w.accum.handle,
            w.revealage.handle,
            w.target.handle,
        ],
        _pad: 0,
        inverse_proj: f.inv_proj,
        inverse_view: f.inv_view,
        size: f.size,
        exposure: f.exposure,
        _pad2: 0.0,
    }
}

fn draw_outline_gpu(w: &World, f: &Frame) -> DrawOutlineGpu {
    DrawOutlineGpu {
        mask: w.outline_mask.handle,
        out: w.target.handle,
        color: f.outline,
        size: f.size,
        _pad: [0; 2],
    }
}

fn gizmos_gpu(w: &World, f: &Frame, shape: u32) -> GizmosGpu {
    GizmosGpu {
        world_to_clip: f.clip,
        gizmos: w.gizmos.address,
        first: shape * 100,
        shape,
    }
}

fn forward_gpu(w: &World, f: &Frame, i: usize) -> ForwardGpu {
    ForwardGpu {
        world_to_clip: f.clip,
        vertices: w.vertex_buffers[i].address,
        transforms: w.transforms.address,
        materials: w.materials.address,
        index: i as u32,
        _pad: 0,
    }
}

fn downsample_gpu(w: &World, f: &Frame, i: u32) -> DownsampleGpu {
    DownsampleGpu {
        src: w.chain.handle + i as u64 + 1,
        dst: w.chain.handle + i as u64 + 2,
        size: f.size >> (i + 1),
    }
}

fn mip_read_gpu(w: &World) -> MipReadGpu {
    MipReadGpu {
        chain: w.chain.handle,
        out: w.target.handle,
        mips: w.chain.mips,
        _pad: 0,
    }
}

// ---- Shared: lava's access types, raster state and draw commands --------------------------

/// lava's `ImageUse` (its `layouts` field is crate-private).
#[derive(Clone, Debug)]
struct ImageUse<'a> {
    access: ImageAccess,
    layouts: &'a [AtomicU32],
}

fn image_use<'a>(v: View<'a>, stage: S, access: A, layout: L) -> ImageUse<'a> {
    ImageUse {
        access: ImageAccess {
            stage,
            access,
            image: v.image,
            layout,
            aspect: v.aspect,
            mip_range: v.mip_range,
        },
        layouts: v.layouts,
    }
}

fn buffer_access(s: Slice, stage: S, access: A) -> BufferAccess {
    BufferAccess {
        stage,
        access,
        range: s.range(),
    }
}

const SAMPLED_READ: A = A::SHADER_SAMPLED_READ;
const STORAGE_READ: A = A::SHADER_STORAGE_READ;
const STORAGE_WRITE: A = A::from_raw(A::SHADER_STORAGE_READ.as_raw() | A::SHADER_STORAGE_WRITE.as_raw());
const UNIFIED_WRITE: A = A::from_raw(A::SHADER_SAMPLED_READ.as_raw() | A::SHADER_STORAGE_WRITE.as_raw());

fn index_access(s: Slice) -> BufferAccess {
    buffer_access(s, S::INDEX_INPUT, A::INDEX_READ)
}

fn indirect_access(s: Slice) -> BufferAccess {
    buffer_access(s, S::DRAW_INDIRECT, A::INDIRECT_COMMAND_READ)
}

#[derive(Clone, Copy)]
struct State {
    blends: [u8; 4],
    depth_write: bool,
}

impl State {
    const OPAQUE: Self = Self {
        blends: [0, 4, 4, 0],
        depth_write: true,
    };
    const BLENDED: Self = Self {
        blends: [4, 2, 3, 0],
        depth_write: false,
    };
    const ALPHA: Self = Self {
        blends: [1, 0, 0, 0],
        depth_write: false,
    };
}

#[derive(Clone, Copy)]
enum Command {
    Draw { vertices: u32, instances: u32 },
    Indexed { address: u64, size: u64, count: u32 },
    LaunchIndirect { address: u64, size: u64, stride: u64, count: u32 },
}

/// A rendering's attachments, the same in every variant.
struct Attachments<'a> {
    extent: UVec2,
    colors: [Option<ImageUse<'a>>; 4],
    count: usize,
    depth: Option<ImageUse<'a>>,
}

impl<'a> Attachments<'a> {
    fn new(extent: UVec2) -> Self {
        Self {
            extent,
            colors: [None, None, None, None],
            count: 0,
            depth: None,
        }
    }

    fn color(&mut self, v: View<'a>) {
        self.colors[self.count] = Some(image_use(
            v,
            S::COLOR_ATTACHMENT_OUTPUT,
            A::COLOR_ATTACHMENT_WRITE,
            L::ATTACHMENT_OPTIMAL,
        ));
        self.count += 1;
    }

    fn depth(&mut self, v: View<'a>) {
        self.depth = Some(image_use(
            v,
            S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS,
            A::DEPTH_STENCIL_ATTACHMENT_READ | A::DEPTH_STENCIL_ATTACHMENT_WRITE,
            L::ATTACHMENT_OPTIMAL,
        ));
    }

    fn uses(&self) -> impl Iterator<Item = &ImageUse<'a>> + Clone {
        self.colors.iter().flatten().chain(self.depth.iter())
    }
}

fn append<T, const N: usize>(array: [T; N], item: T) -> [T; N + 1] {
    let mut items = array.into_iter().chain(std::iter::once(item));
    std::array::from_fn(|_| items.next().unwrap())
}

fn concat<T, const N: usize, const M: usize>(a: [T; N], b: [T; M]) -> [T; N + M] {
    let mut items = a.into_iter().chain(b);
    std::array::from_fn(|_| items.next().unwrap())
}

#[derive(Clone, Copy)]
struct PushConstants([u8; MAX_PUSH]);

fn padded<G: Pod>(g: &G) -> [u8; MAX_PUSH] {
    let mut bytes = [0; MAX_PUSH];
    bytes[..size_of::<G>()].copy_from_slice(bytes_of(g));
    bytes
}

/// lava's `Constants`, used by the first three variants.
#[derive(Clone, Copy)]
struct Constants {
    bytes: [u8; MAX_CONSTANTS],
    len: usize,
}

impl Constants {
    const NONE: Self = Self {
        bytes: [0; MAX_CONSTANTS],
        len: 0,
    };

    fn new(bytes: &[u8]) -> Self {
        let mut constants = Self::NONE;
        constants.bytes[..bytes.len()].copy_from_slice(bytes);
        constants.len = bytes.len();
        constants
    }

    fn bytes(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
}

// ---- Compressed accesses (variants 3 and 4) ------------------------------------------------
// 4-byte header, then the image handle or buffer address, then the pointer to the first
// tracked layout or the buffer size.

const IMAGE: u32 = 1 << 0;
const WRITE: u32 = 1 << 1;
const ONE_LAYOUT: u32 = 1 << 2;
const DEPTH: u32 = 1 << 3;
const STENCIL: u32 = 1 << 4;
const ACCELERATION_STRUCTURE: u32 = 1 << 5;
const FIRST_MIP_SHIFT: u32 = 16;
const MIP_COUNT_SHIFT: u32 = 21;
const MIP_MASK: u32 = 0b1_1111;

fn compress_image(v: &View, write: bool) -> (u32, u64, u64) {
    let mips = v.mip_range;
    debug_assert!(mips.start < 32 && mips.end > mips.start && mips.end - mips.start <= 32);
    debug_assert!(v.layouts.len() == 1 || v.layouts.len() == (mips.end - mips.start) as usize);
    let mut header = IMAGE
        | (mips.start << FIRST_MIP_SHIFT)
        | ((mips.end - mips.start - 1) << MIP_COUNT_SHIFT);
    if write {
        header |= WRITE;
    }
    if v.layouts.len() == 1 {
        header |= ONE_LAYOUT;
    }
    if v.aspect.contains(vk::ImageAspectFlags::DEPTH) {
        header |= DEPTH;
    }
    if v.aspect.contains(vk::ImageAspectFlags::STENCIL) {
        header |= STENCIL;
    }
    (header, v.image.as_raw(), v.layouts.as_ptr() as u64)
}

fn compress_buffer(s: Slice, write: bool) -> (u32, u64, u64) {
    (if write { WRITE } else { 0 }, s.address, s.size)
}

fn compressed_flags(header: u32) -> A {
    if header & ACCELERATION_STRUCTURE != 0 {
        A::ACCELERATION_STRUCTURE_READ_KHR | A::MEMORY_READ
    } else if header & WRITE != 0 {
        A::SHADER_READ | A::SHADER_WRITE
    } else {
        A::SHADER_READ
    }
}

fn expand_image<'a>(header: u32, image: u64, layouts: u64, stage: S) -> ImageUse<'a> {
    let first = (header >> FIRST_MIP_SHIFT) & MIP_MASK;
    let count = ((header >> MIP_COUNT_SHIFT) & MIP_MASK) + 1;
    let len = if header & ONE_LAYOUT != 0 { 1 } else { count as usize };
    // SAFETY: made from a `&'a [AtomicU32]` of this length by `compress_image`.
    let layouts = unsafe { slice::from_raw_parts(layouts as *const AtomicU32, len) };
    let aspect = if header & (DEPTH | STENCIL) == 0 {
        vk::ImageAspectFlags::COLOR
    } else {
        let mut aspect = vk::ImageAspectFlags::empty();
        if header & DEPTH != 0 {
            aspect |= vk::ImageAspectFlags::DEPTH;
        }
        if header & STENCIL != 0 {
            aspect |= vk::ImageAspectFlags::STENCIL;
        }
        aspect
    };
    ImageUse {
        access: ImageAccess {
            stage,
            access: compressed_flags(header),
            image: vk::Image::from_raw(image),
            layout: L::GENERAL,
            aspect,
            mip_range: (first..first + count).into(),
        },
        layouts,
    }
}

fn expand_buffer(header: u32, address: u64, size: u64, stage: S) -> BufferAccess {
    BufferAccess {
        stage,
        access: compressed_flags(header),
        range: (address..address + size).into(),
    }
}

// ---- Recording: the same for every variant ------------------------------------------------

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
struct Stats {
    barriers: u64,
    transitions: u64,
    images: u64,
    buffers: u64,
    draws: u64,
    push_bytes: u64,
}

struct PendingWrite<T> {
    write: T,
    visible_stages: S,
    visible_access: A,
}

impl<T> PendingWrite<T> {
    fn new(write: T) -> Self {
        Self {
            write,
            visible_stages: S::NONE,
            visible_access: A::NONE,
        }
    }

    fn visible_to(&self, stage: S, access: A) -> bool {
        self.visible_stages.contains(stage) && self.visible_access.contains(access)
    }
}

#[derive(Default)]
struct Pending {
    buffer_reads: Vec<BufferAccess>,
    buffer_writes: Vec<PendingWrite<BufferAccess>>,
    image_reads: Vec<ImageAccess>,
    image_writes: Vec<PendingWrite<ImageAccess>>,
}

fn write_mask() -> A {
    A::SHADER_WRITE
        | A::COLOR_ATTACHMENT_WRITE
        | A::DEPTH_STENCIL_ATTACHMENT_WRITE
        | A::TRANSFER_WRITE
        | A::HOST_WRITE
        | A::MEMORY_WRITE
        | A::SHADER_STORAGE_WRITE
        | A::ACCELERATION_STRUCTURE_WRITE_KHR
}

fn read_mask() -> A {
    A::INDIRECT_COMMAND_READ
        | A::INDEX_READ
        | A::VERTEX_ATTRIBUTE_READ
        | A::UNIFORM_READ
        | A::INPUT_ATTACHMENT_READ
        | A::SHADER_READ
        | A::COLOR_ATTACHMENT_READ
        | A::DEPTH_STENCIL_ATTACHMENT_READ
        | A::TRANSFER_READ
        | A::HOST_READ
        | A::MEMORY_READ
        | A::SHADER_SAMPLED_READ
        | A::SHADER_STORAGE_READ
        | A::ACCELERATION_STRUCTURE_READ_KHR
}

fn overlaps(a: Range<u64>, b: Range<u64>) -> bool {
    a.start < b.end && b.start < a.end
}

impl Pending {
    fn clear(&mut self) {
        self.buffer_reads.clear();
        self.buffer_writes.clear();
        self.image_reads.clear();
        self.image_writes.clear();
    }

    /// lava's `PendingAccesses::record`.
    fn record(
        &mut self,
        images: impl Iterator<Item = ImageAccess>,
        buffers: impl Iterator<Item = BufferAccess>,
    ) {
        for access in buffers {
            let covers = |a: &BufferAccess| {
                access.range.start <= a.range.start && a.range.end <= access.range.end
            };
            let writes = access.access.intersects(write_mask());
            let reads = access.access.intersects(read_mask());
            if writes {
                self.buffer_writes.retain(|a| !covers(&a.write));
                self.buffer_writes.push(PendingWrite::new(access.clone()));
                self.buffer_reads.retain(|a| !covers(a));
            }
            if reads {
                let mut read = access.clone();
                self.buffer_reads.retain(|a| {
                    let covered = covers(a);
                    if covered {
                        read.stage |= a.stage;
                        read.access |= a.access;
                    }
                    !covered
                });
                self.buffer_reads.push(read);
            }
        }
        for access in images {
            let same_image = |a: &ImageAccess| a.image == access.image;
            let writes = access.access.intersects(write_mask());
            let reads = access.access.intersects(read_mask());
            if writes {
                self.image_writes.retain(|a| !same_image(&a.write));
                self.image_writes.push(PendingWrite::new(access.clone()));
                self.image_reads.retain(|a| !same_image(a));
            }
            if reads {
                let mut read = access.clone();
                self.image_reads.retain(|a| {
                    let same = same_image(a);
                    if same {
                        read.stage |= a.stage;
                        read.access |= a.access;
                    }
                    !same
                });
                self.image_reads.push(read);
            }
        }
    }
}

/// lava's `ImageUse::take_layouts`.
fn take_layouts(image: &ImageUse, mut transition: impl FnMut(Range<u32>, L)) {
    let new = image.access.layout;
    let swap = |layout: &AtomicU32| {
        L::from_raw(layout.swap(new.as_raw() as u32, Ordering::Relaxed) as i32)
    };
    let mips = image.access.mip_range;
    if let [layout] = image.layouts {
        let old = swap(layout);
        if old != new {
            transition(mips, old);
        }
        return;
    }
    let mut run: Option<(u32, L)> = None;
    for (mip, layout) in (mips.start..).zip(image.layouts) {
        let old = swap(layout);
        match run {
            Some((_, run_old)) if run_old == old => {}
            Some((start, run_old)) => {
                if run_old != new {
                    transition((start..mip).into(), run_old);
                }
                run = Some((mip, old));
            }
            None => run = Some((mip, old)),
        }
    }
    if let Some((start, old)) = run
        && old != new
    {
        transition((start..mips.end).into(), old);
    }
}

struct Recorder {
    /// "full": lava's barrier tracking; "decode": only read every access.
    model_barriers: bool,
    pending: Pending,
    stream: Vec<u8>,
    plain: Vec<u64>,
    specialized: RwLock<Vec<HashMap<Box<[u8]>, u64>>>,
    stats: Stats,
    check: u64,
}

impl Recorder {
    fn new(model_barriers: bool) -> Self {
        Self {
            model_barriers,
            pending: Pending::default(),
            stream: Vec::with_capacity(1 << 20),
            plain: (0..PASS_STAGES.len() as u64).collect(),
            specialized: RwLock::new(vec![HashMap::new(); PASS_STAGES.len()]),
            stats: Stats::default(),
            check: 0,
        }
    }

    fn reset(&mut self) {
        self.stream.clear();
        self.pending.clear();
    }

    fn push(&mut self, bytes: &[u8]) {
        self.stream.extend_from_slice(bytes);
        self.stats.push_bytes += bytes.len() as u64;
    }

    fn bind_pipeline(&mut self, pass: usize, constants: &[u8]) {
        let pipeline = if constants.is_empty() {
            self.plain[pass]
        } else {
            self.specialized_pipeline(pass, constants)
        };
        self.stream.extend_from_slice(&pipeline.to_le_bytes());
    }

    fn specialized_pipeline(&self, pass: usize, constants: &[u8]) -> u64 {
        if let Some(&pipeline) = self.specialized.read().unwrap()[pass].get(constants) {
            return pipeline;
        }
        let mut maps = self.specialized.write().unwrap();
        let next = 1000 + maps[pass].len() as u64;
        *maps[pass].entry(constants.into()).or_insert(next)
    }

    fn dispatch(&mut self, groups: [u32; 3]) {
        self.stream.extend_from_slice(bytes_of(&groups));
    }

    fn begin(&mut self, extent: UVec2) {
        self.stream.extend_from_slice(bytes_of(&extent));
    }

    fn end(&mut self) {
        self.stream.push(0xff);
    }

    fn draw(&mut self, command: &Command) {
        self.stats.draws += 1;
        match *command {
            Command::Draw {
                vertices,
                instances,
            } => self.stream.extend_from_slice(bytes_of(&[vertices, instances])),
            Command::Indexed {
                address,
                size,
                count,
            } => self
                .stream
                .extend_from_slice(bytes_of(&[address, size, count as u64])),
            Command::LaunchIndirect {
                address,
                size,
                stride,
                count,
            } => self
                .stream
                .extend_from_slice(bytes_of(&[address, size, stride, count as u64])),
        }
    }

    /// lava's `flush_pending`: barriers against the pending accesses, layout transitions,
    /// then the accesses become pending.
    fn flush<'a, I, B>(&mut self, images: I, buffers: B)
    where
        I: Iterator + Clone,
        I::Item: Borrow<ImageUse<'a>>,
        B: Iterator + Clone,
        B::Item: Borrow<BufferAccess>,
    {
        if !self.model_barriers {
            for image in images {
                let image = image.borrow();
                let a = &image.access;
                self.stats.images += 1;
                self.check = self.check.wrapping_add(
                    a.image.as_raw()
                        ^ a.stage.as_raw()
                        ^ a.access.as_raw()
                        ^ a.layout.as_raw() as u64
                        ^ a.aspect.as_raw() as u64
                        ^ ((a.mip_range.start as u64) << 32 | a.mip_range.end as u64)
                        ^ image.layouts.as_ptr() as u64
                        ^ image.layouts.len() as u64,
                );
            }
            for buffer in buffers {
                let b = buffer.borrow();
                self.stats.buffers += 1;
                self.check = self.check.wrapping_add(
                    b.range.start ^ b.range.end ^ b.stage.as_raw() ^ b.access.as_raw(),
                );
            }
            return;
        }

        let (mut src_stage, mut dst_stage) = (S::NONE, S::NONE);
        let (mut src_access, mut dst_access) = (A::NONE, A::NONE);
        for buffer in buffers.clone() {
            let b = buffer.borrow();
            self.stats.buffers += 1;
            let writes = b.access.intersects(write_mask());
            let reads = b.access.intersects(read_mask());
            if writes {
                for p in self.pending.buffer_reads.iter().filter(|p| overlaps(p.range, b.range)) {
                    src_stage |= p.stage;
                    src_access |= p.access;
                    dst_stage |= b.stage;
                    dst_access |= b.access;
                }
            }
            for p in self.pending.buffer_writes.iter().filter(|p| {
                overlaps(p.write.range, b.range) && !p.visible_to(b.stage, b.access)
            }) {
                if reads || writes {
                    src_stage |= p.write.stage;
                    src_access |= p.write.access;
                    dst_stage |= b.stage;
                    dst_access |= b.access;
                }
            }
        }
        for image in images.clone() {
            let image = image.borrow();
            let a = &image.access;
            self.stats.images += 1;
            let writes = a.access.intersects(write_mask());
            let reads = a.access.intersects(read_mask());
            if writes {
                for p in self.pending.image_reads.iter().filter(|p| p.image == a.image) {
                    src_stage |= p.stage;
                    src_access |= p.access;
                    dst_stage |= a.stage;
                    dst_access |= a.access;
                }
            }
            for p in self
                .pending
                .image_writes
                .iter()
                .filter(|p| p.write.image == a.image && !p.visible_to(a.stage, a.access))
            {
                if reads || writes {
                    src_stage |= p.write.stage;
                    src_access |= p.write.access;
                    dst_stage |= a.stage;
                    dst_access |= a.access;
                }
            }
        }
        let mut transitions = 0;
        for image in images.clone() {
            take_layouts(image.borrow(), |_, _| transitions += 1);
        }
        for _ in 0..transitions {
            self.stream.extend_from_slice(&[0; 72]);
        }
        self.stats.transitions += transitions;
        if !src_stage.is_empty() || !dst_stage.is_empty() {
            self.stats.barriers += 1;
            self.stream.extend_from_slice(&[1; 48]);
            for w in &mut self.pending.buffer_writes {
                if src_stage.contains(w.write.stage) && src_access.contains(w.write.access) {
                    w.visible_stages |= dst_stage;
                    w.visible_access |= dst_access;
                }
            }
            for w in &mut self.pending.image_writes {
                if src_stage.contains(w.write.stage) && src_access.contains(w.write.access) {
                    w.visible_stages |= dst_stage;
                    w.visible_access |= dst_access;
                }
            }
        }
        self.pending.record(
            images.map(|i| i.borrow().access.clone()),
            buffers.map(|b| b.borrow().clone()),
        );
    }
}

// ---- The workloads, written once and expanded in each variant's module --------------------
// `b!(expr)` is how the variant hands bindings to a command: by value (today) or borrowed.

macro_rules! registrations {
    ($bindings:expr, $w:expr, $base:expr; $($i:literal)*) => {
        $bindings $(.sampled_read($w.textures[$base + $i].whole()))*
    };
}

macro_rules! workloads {
    () => {
        /// core's frame: culling, the scene, composite, outline, gizmos and UI.
        #[inline(never)]
        pub fn frame(rec: &mut Recorder, w: &World, f: &Frame) {
            compute(rec, b!(instance_cull(w, f)), [7, 1, 1]);
            compute(rec, b!(bvh_cull(w, f)), [64, 1, 1]);
            raster(rec, f.size)
                .color(w.gbuffer.whole())
                .color(w.accum.whole())
                .color(w.revealage.whole())
                .depth(w.depth.whole())
                .launch_indirect(
                    b!(raster_pass(w, f, RASTER, w.meshlets, w.opaque_draw)),
                    w.opaque_draw,
                    State::OPAQUE,
                )
                .launch_indirect(
                    b!(raster_pass(w, f, RASTER_BLENDED, w.blended, w.blended_draw)),
                    w.blended_draw,
                    State::BLENDED,
                )
                .record();
            compute(rec, b!(composite(w, f)), groups(f.size));
            raster(rec, f.size)
                .color(w.outline_mask.whole())
                .launch_indirect(
                    b!(raster_pass(w, f, RASTER_OUTLINE, w.outlined, w.outlined_draw)),
                    w.outlined_draw,
                    State::OPAQUE,
                )
                .record();
            compute(rec, b!(draw_outline(w, f)), groups(f.size));
            raster(rec, f.size)
                .color(w.target.whole())
                .draw(b!(gizmos(w, f, 0)), 36, 10, State::ALPHA)
                .draw(b!(gizmos(w, f, 1)), 576, 4, State::ALPHA)
                .draw(b!(gizmos(w, f, 2)), 216, 3, State::ALPHA)
                .record();
            raster(rec, f.size)
                .color(w.swapchain.whole())
                .draw_indexed(
                    b!(ui(w).storage_read(w.target.whole())),
                    w.ui_indices,
                    State::ALPHA,
                )
                .record();
        }

        /// 8 renderings of 8 indexed draws, each binding with 3 buffers and 2 registered
        /// textures: the raster builder and registrations.
        #[inline(never)]
        pub fn many_draws(rec: &mut Recorder, w: &World, f: &Frame) {
            let fwd = |i: usize| {
                forward(w, f, i)
                    .sampled_read(w.textures[i].whole())
                    .sampled_read(w.textures[(i + 32) % 64].whole())
            };
            for r in 0..8 {
                let i = r * 8;
                raster(rec, f.size)
                    .color(w.target.whole())
                    .depth(w.depth.whole())
                    .draw_indexed(b!(fwd(i)), w.index_buffers[i], State::OPAQUE)
                    .draw_indexed(b!(fwd(i + 1)), w.index_buffers[i + 1], State::OPAQUE)
                    .draw_indexed(b!(fwd(i + 2)), w.index_buffers[i + 2], State::OPAQUE)
                    .draw_indexed(b!(fwd(i + 3)), w.index_buffers[i + 3], State::OPAQUE)
                    .draw_indexed(b!(fwd(i + 4)), w.index_buffers[i + 4], State::OPAQUE)
                    .draw_indexed(b!(fwd(i + 5)), w.index_buffers[i + 5], State::OPAQUE)
                    .draw_indexed(b!(fwd(i + 6)), w.index_buffers[i + 6], State::OPAQUE)
                    .draw_indexed(b!(fwd(i + 7)), w.index_buffers[i + 7], State::OPAQUE)
                    .record();
            }
        }

        /// 11 downsample passes over a mip chain's per-mip views, then one pass reading the
        /// whole chain (one tracked layout per mip): small compute bindings.
        #[inline(never)]
        pub fn mip_chain(rec: &mut Recorder, w: &World, f: &Frame) {
            for i in 0..11 {
                let size = f.size >> (i + 1);
                compute(rec, b!(downsample(w, f, i)), [size.x.div_ceil(8), size.y.div_ceil(8), 1]);
            }
            compute(rec, b!(mip_read(w)), groups(f.size));
        }

        /// 4 UI renderings that each register 16 textures (`DynImg`s) on top of the pass's
        /// own 2 accesses: bindings past today's 8 inline images.
        #[inline(never)]
        pub fn ui_textures(rec: &mut Recorder, w: &World, f: &Frame) {
            for r in 0..4 {
                let t = r * 16;
                raster(rec, f.size)
                    .color(w.swapchain.whole())
                    .draw_indexed(
                        b!(registrations!(ui(w), w, t; 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)),
                        w.ui_indices,
                        State::ALPHA,
                    )
                    .record();
            }
        }
    };
}

// ---- Variant 1: today's `BindingOutput` (SmallVecs, moved by value) -----------------------

mod v1 {
    use super::*;

    pub struct Out<'a, K> {
        pass: usize,
        push_constants: [u8; MAX_PUSH],
        constants: Constants,
        images: SmallVec<[ImageUse<'a>; 8]>,
        buffers: SmallVec<[BufferAccess; 16]>,
        stage: S,
        kind: PhantomData<K>,
    }

    impl<'a, K> Out<'a, K> {
        fn new<G: Pod, const I: usize, const B: usize>(
            pass: usize,
            bindings: &G,
            stage: S,
            images: [ImageUse<'a>; I],
            buffers: [BufferAccess; B],
        ) -> Self {
            Self {
                pass,
                push_constants: padded(bindings),
                constants: Constants::NONE,
                images: images.into_iter().collect(),
                buffers: buffers.into_iter().collect(),
                stage,
                kind: PhantomData,
            }
        }

        fn with_constants<G: Pod, const I: usize, const B: usize>(
            mut self,
            constants: &G,
            images: [ImageUse<'a>; I],
            buffers: [BufferAccess; B],
        ) -> Self {
            self.constants = Constants::new(bytes_of(constants));
            self.images.extend(images);
            self.buffers.extend(buffers);
            self
        }

        pub fn sampled_read(mut self, v: View<'a>) -> Self {
            self.images.push(image_use(v, self.stage, SAMPLED_READ, L::GENERAL));
            self
        }

        pub fn storage_read(mut self, v: View<'a>) -> Self {
            self.images.push(image_use(v, self.stage, STORAGE_READ, L::GENERAL));
            self
        }

        fn erase<'s>(self) -> Out<'s, ()>
        where
            'a: 's,
        {
            Out {
                pass: self.pass,
                push_constants: self.push_constants,
                constants: self.constants,
                // SAFETY: only the lifetime shortens, as in lava.
                images: unsafe {
                    std::mem::transmute::<
                        SmallVec<[ImageUse<'a>; 8]>,
                        SmallVec<[ImageUse<'s>; 8]>,
                    >(self.images)
                },
                buffers: self.buffers,
                stage: self.stage,
                kind: PhantomData,
            }
        }
    }

    pub fn size() -> usize {
        size_of::<Out<'static, kind::Compute>>()
    }

    fn img<'a>(v: View<'a>, s: S, access: A) -> ImageUse<'a> {
        image_use(v, s, access, L::GENERAL)
    }

    fn buf(b: Slice, s: S, access: A) -> BufferAccess {
        buffer_access(b, s, access)
    }

    // Generated `push_bindings(..)(.constant_bindings(..))`.

    pub fn instance_cull<'a>(w: &'a World, f: &Frame) -> Out<'a, kind::Compute> {
        let s = COMPUTE;
        Out::new(INSTANCE_CULL, &instance_cull_gpu(w, f), s, [], [
            buf(w.roots, s, STORAGE_READ),
            buf(w.aabbs, s, STORAGE_READ),
            buf(w.transforms, s, STORAGE_READ),
            buf(w.stack, s, STORAGE_WRITE),
            buf(w.variables, s, STORAGE_WRITE),
        ])
    }

    pub fn bvh_cull<'a>(w: &'a World, f: &Frame) -> Out<'a, kind::Compute> {
        let s = COMPUTE;
        Out::new(BVH_CULL, &bvh_cull_gpu(w, f), s, [], [
            buf(w.stack, s, STORAGE_WRITE),
            buf(w.variables, s, STORAGE_WRITE),
            buf(w.meshlets, s, STORAGE_WRITE),
            buf(w.blended, s, STORAGE_WRITE),
            buf(w.outlined, s, STORAGE_WRITE),
            buf(w.candidates, s, STORAGE_WRITE),
            buf(w.batches, s, STORAGE_WRITE),
            buf(w.transforms, s, STORAGE_READ),
            buf(w.headers, s, STORAGE_READ),
            buf(w.flags, s, STORAGE_READ),
        ])
        .with_constants(&bvh_cull_consts(w), [], [])
    }

    pub fn raster_pass<'a>(
        w: &'a World,
        f: &Frame,
        pass: usize,
        meshlets: Slice,
        draw: Slice,
    ) -> Out<'a, kind::RasterMesh> {
        let s = MESH;
        Out::new(pass, &raster_gpu(w, f, meshlets, draw), s, [], [
            buf(w.transforms, s, STORAGE_READ),
            buf(meshlets, s, STORAGE_READ),
            buf(draw, s, STORAGE_READ),
            buf(w.materials, s, STORAGE_READ),
        ])
        .with_constants(&128u32, [], [])
    }

    pub fn composite<'a>(w: &'a World, f: &Frame) -> Out<'a, kind::Compute> {
        let s = COMPUTE;
        Out::new(COMPOSITE, &composite_gpu(w, f), s, [
            img(w.gbuffer.whole(), s, SAMPLED_READ),
            img(w.depth.whole(), s, SAMPLED_READ),
            img(w.accum.whole(), s, STORAGE_WRITE),
            img(w.revealage.whole(), s, UNIFIED_WRITE),
            img(w.target.whole(), s, STORAGE_WRITE),
        ], [])
        .with_constants(&7u32, [], [])
    }

    pub fn draw_outline<'a>(w: &'a World, f: &Frame) -> Out<'a, kind::Compute> {
        let s = COMPUTE;
        Out::new(DRAW_OUTLINE, &draw_outline_gpu(w, f), s, [
            img(w.outline_mask.whole(), s, SAMPLED_READ),
            img(w.target.whole(), s, STORAGE_WRITE),
        ], [])
    }

    pub fn gizmos<'a>(w: &'a World, f: &Frame, shape: u32) -> Out<'a, kind::RasterVertex> {
        let s = VERTEX;
        Out::new(GIZMOS, &gizmos_gpu(w, f, shape), s, [], [buf(w.gizmos, s, STORAGE_READ)])
    }

    pub fn ui<'a>(w: &'a World) -> Out<'a, kind::RasterVertex> {
        let s = VERTEX;
        Out::new(UI, &UiGpu { vertices: w.ui_vertices.address }, s, [], [
            buf(w.ui_vertices, s, STORAGE_READ),
        ])
        .with_constants(&w.font_atlas.handle, [img(w.font_atlas.whole(), s, SAMPLED_READ)], [])
    }

    pub fn forward<'a>(w: &'a World, f: &Frame, i: usize) -> Out<'a, kind::RasterVertex> {
        let s = VERTEX;
        Out::new(FORWARD, &forward_gpu(w, f, i), s, [], [
            buf(w.vertex_buffers[i], s, STORAGE_READ),
            buf(w.transforms, s, STORAGE_READ),
            buf(w.materials, s, STORAGE_READ),
        ])
    }

    pub fn downsample<'a>(w: &'a World, f: &Frame, i: u32) -> Out<'a, kind::Compute> {
        let s = COMPUTE;
        Out::new(DOWNSAMPLE, &downsample_gpu(w, f, i), s, [
            img(w.chain.mip(i), s, SAMPLED_READ),
            img(w.chain.mip(i + 1), s, STORAGE_WRITE),
        ], [])
    }

    pub fn mip_read<'a>(w: &'a World) -> Out<'a, kind::Compute> {
        let s = COMPUTE;
        Out::new(MIP_READ, &mip_read_gpu(w), s, [
            img(w.chain.whole(), s, SAMPLED_READ),
            img(w.target.whole(), s, STORAGE_WRITE),
        ], [])
    }

    // Commands.

    #[inline(never)]
    pub fn compute(rec: &mut Recorder, b: Out<'_, kind::Compute>, groups: [u32; 3]) {
        rec.flush(b.images.iter(), b.buffers.iter());
        rec.push(&b.push_constants);
        rec.bind_pipeline(b.pass, b.constants.bytes());
        rec.dispatch(groups);
    }

    struct Draw<'a> {
        b: Out<'a, ()>,
        command: Command,
        state: State,
    }

    pub struct RasterBuilder<'a, const D: usize> {
        rec: &'a mut Recorder,
        att: Attachments<'a>,
        draws: [Draw<'a>; D],
    }

    pub fn raster(rec: &mut Recorder, extent: UVec2) -> RasterBuilder<'_, 0> {
        RasterBuilder {
            rec,
            att: Attachments::new(extent),
            draws: [],
        }
    }

    impl<'a, const D: usize> RasterBuilder<'a, D> {
        pub fn color(mut self, v: View<'a>) -> Self {
            self.att.color(v);
            self
        }

        pub fn depth(mut self, v: View<'a>) -> Self {
            self.att.depth(v);
            self
        }

        fn push(self, b: Out<'a, ()>, command: Command, state: State) -> RasterBuilder<'a, { D + 1 }> {
            let draw = Draw { b, command, state };
            RasterBuilder {
                rec: self.rec,
                att: self.att,
                draws: append(self.draws, draw),
            }
        }

        pub fn draw<'b: 'a>(
            self,
            b: Out<'b, kind::RasterVertex>,
            vertices: u32,
            instances: u32,
            state: State,
        ) -> RasterBuilder<'a, { D + 1 }> {
            self.push(b.erase(), Command::Draw { vertices, instances }, state)
        }

        pub fn draw_indexed<'b: 'a>(
            self,
            b: Out<'b, kind::RasterVertex>,
            indices: Slice,
            state: State,
        ) -> RasterBuilder<'a, { D + 1 }> {
            let mut b = b.erase();
            b.buffers.push(index_access(indices));
            let command = Command::Indexed {
                address: indices.address,
                size: indices.size,
                count: (indices.size / 4) as u32,
            };
            self.push(b, command, state)
        }

        pub fn launch_indirect<'b: 'a>(
            self,
            b: Out<'b, kind::RasterMesh>,
            draws: Slice,
            state: State,
        ) -> RasterBuilder<'a, { D + 1 }> {
            let mut b = b.erase();
            b.buffers.push(indirect_access(draws));
            let command = Command::LaunchIndirect {
                address: draws.address,
                size: draws.size,
                stride: 12,
                count: 1,
            };
            self.push(b, command, state)
        }

        pub fn record(self) {
            let Self { rec, att, draws } = self;
            rec.flush(
                att.uses().chain(draws.iter().flat_map(|d| d.b.images.iter())),
                draws.iter().flat_map(|d| d.b.buffers.iter()),
            );
            rec.begin(att.extent);
            for d in &draws {
                rec.bind_pipeline(d.b.pass, d.b.constants.bytes());
                rec.push(&d.b.push_constants);
                rec.stream.push(d.state.blends[0] | d.state.depth_write as u8);
                rec.draw(&d.command);
            }
            rec.end();
        }
    }

    macro_rules! b {
        ($e:expr) => {
            $e
        };
    }

    workloads!();
}

// ---- Variants 2 and 3: unsized, borrowed by the commands ----------------------------------
// One trailing array of accesses: lava's `ImageUse`/`BufferAccess` in an enum (variant 2), or
// 20-byte compressed accesses (variant 3). The generated functions are the same text for
// both; `Acc` and the access constructors differ.

macro_rules! unsized_variant {
    () => {
        pub struct Out<'a, K, T: ?Sized = [Acc<'a>]> {
            pass: usize,
            stage: S,
            constants: Constants,
            /// A non-generic newtype: under `generic_const_exprs` a `[u8; MAX_PUSH]` field
            /// counts as depending on `T`, which rules out unsizing coercion.
            push_constants: PushConstants,
            /// Uses `'a` outside the unsized field.
            kind: PhantomData<(K, &'a ())>,
            accesses: T,
        }

        impl<'a, K, const N: usize> Out<'a, K, [Acc<'a>; N]> {
            fn new<G: Pod>(pass: usize, bindings: &G, stage: S, accesses: [Acc<'a>; N]) -> Self {
                Self {
                    pass,
                    stage,
                    constants: Constants::NONE,
                    push_constants: PushConstants(padded(bindings)),
                    kind: PhantomData,
                    accesses,
                }
            }

            fn with_constants<G: Pod, const M: usize>(
                self,
                constants: &G,
                accesses: [Acc<'a>; M],
            ) -> Out<'a, K, [Acc<'a>; N + M]>
            where
                [(); N + M]:,
            {
                Out {
                    pass: self.pass,
                    stage: self.stage,
                    constants: Constants::new(bytes_of(constants)),
                    push_constants: self.push_constants,
                    kind: PhantomData,
                    accesses: concat(self.accesses, accesses),
                }
            }

            fn with(self, access: Acc<'a>) -> Out<'a, K, [Acc<'a>; N + 1]>
            where
                [(); N + 1]:,
            {
                Out {
                    pass: self.pass,
                    stage: self.stage,
                    constants: self.constants,
                    push_constants: self.push_constants,
                    kind: PhantomData,
                    accesses: append(self.accesses, access),
                }
            }

            pub fn sampled_read(self, v: View<'a>) -> Out<'a, K, [Acc<'a>; N + 1]>
            where
                [(); N + 1]:,
            {
                let s = self.stage;
                self.with(sampled(v, s))
            }

            pub fn storage_read(self, v: View<'a>) -> Out<'a, K, [Acc<'a>; N + 1]>
            where
                [(); N + 1]:,
            {
                let s = self.stage;
                self.with(storage_read(v, s))
            }
        }

        fn erase<'a, K>(b: &'a Out<'a, K>) -> &'a Out<'a, ()> {
            // SAFETY: `K` is a zero-sized marker; the layout and metadata are the same.
            unsafe { &*(b as *const Out<'a, K> as *const Out<'a, ()>) }
        }

        pub fn instance_cull<'a>(w: &'a World, f: &Frame) -> Out<'a, kind::Compute, [Acc<'a>; 5]> {
            let s = COMPUTE;
            Out::new(INSTANCE_CULL, &instance_cull_gpu(w, f), s, [
                read(w.roots, s),
                read(w.aabbs, s),
                read(w.transforms, s),
                write(w.stack, s),
                write(w.variables, s),
            ])
        }

        pub fn bvh_cull<'a>(w: &'a World, f: &Frame) -> Out<'a, kind::Compute, [Acc<'a>; 10]> {
            let s = COMPUTE;
            Out::new(BVH_CULL, &bvh_cull_gpu(w, f), s, [
                write(w.stack, s),
                write(w.variables, s),
                write(w.meshlets, s),
                write(w.blended, s),
                write(w.outlined, s),
                write(w.candidates, s),
                write(w.batches, s),
                read(w.transforms, s),
                read(w.headers, s),
                read(w.flags, s),
            ])
            .with_constants(&bvh_cull_consts(w), [])
        }

        pub fn raster_pass<'a>(
            w: &'a World,
            f: &Frame,
            pass: usize,
            meshlets: Slice,
            draw: Slice,
        ) -> Out<'a, kind::RasterMesh, [Acc<'a>; 4]> {
            let s = MESH;
            Out::new(pass, &raster_gpu(w, f, meshlets, draw), s, [
                read(w.transforms, s),
                read(meshlets, s),
                read(draw, s),
                read(w.materials, s),
            ])
            .with_constants(&128u32, [])
        }

        pub fn composite<'a>(w: &'a World, f: &Frame) -> Out<'a, kind::Compute, [Acc<'a>; 5]> {
            let s = COMPUTE;
            Out::new(COMPOSITE, &composite_gpu(w, f), s, [
                sampled(w.gbuffer.whole(), s),
                sampled(w.depth.whole(), s),
                storage_write(w.accum.whole(), s),
                unified_write(w.revealage.whole(), s),
                storage_write(w.target.whole(), s),
            ])
            .with_constants(&7u32, [])
        }

        pub fn draw_outline<'a>(w: &'a World, f: &Frame) -> Out<'a, kind::Compute, [Acc<'a>; 2]> {
            let s = COMPUTE;
            Out::new(DRAW_OUTLINE, &draw_outline_gpu(w, f), s, [
                sampled(w.outline_mask.whole(), s),
                storage_write(w.target.whole(), s),
            ])
        }

        pub fn gizmos<'a>(
            w: &'a World,
            f: &Frame,
            shape: u32,
        ) -> Out<'a, kind::RasterVertex, [Acc<'a>; 1]> {
            let s = VERTEX;
            Out::new(GIZMOS, &gizmos_gpu(w, f, shape), s, [read(w.gizmos, s)])
        }

        pub fn ui<'a>(w: &'a World) -> Out<'a, kind::RasterVertex, [Acc<'a>; 2]> {
            let s = VERTEX;
            Out::new(UI, &UiGpu { vertices: w.ui_vertices.address }, s, [read(w.ui_vertices, s)])
                .with_constants(&w.font_atlas.handle, [sampled(w.font_atlas.whole(), s)])
        }

        pub fn forward<'a>(
            w: &'a World,
            f: &Frame,
            i: usize,
        ) -> Out<'a, kind::RasterVertex, [Acc<'a>; 3]> {
            let s = VERTEX;
            Out::new(FORWARD, &forward_gpu(w, f, i), s, [
                read(w.vertex_buffers[i], s),
                read(w.transforms, s),
                read(w.materials, s),
            ])
        }

        pub fn downsample<'a>(w: &'a World, f: &Frame, i: u32) -> Out<'a, kind::Compute, [Acc<'a>; 2]> {
            let s = COMPUTE;
            Out::new(DOWNSAMPLE, &downsample_gpu(w, f, i), s, [
                sampled(w.chain.mip(i), s),
                storage_write(w.chain.mip(i + 1), s),
            ])
        }

        pub fn mip_read<'a>(w: &'a World) -> Out<'a, kind::Compute, [Acc<'a>; 2]> {
            let s = COMPUTE;
            Out::new(MIP_READ, &mip_read_gpu(w), s, [
                sampled(w.chain.whole(), s),
                storage_write(w.target.whole(), s),
            ])
        }

        #[inline(never)]
        pub fn compute(rec: &mut Recorder, b: &Out<'_, kind::Compute>, groups: [u32; 3]) {
            rec.flush(images(b), buffers(b));
            rec.push(&b.push_constants.0);
            rec.bind_pipeline(b.pass, b.constants.bytes());
            rec.dispatch(groups);
        }

        struct Draw<'a> {
            b: &'a Out<'a, ()>,
            /// The index or indirect buffer: not at the pass's stage, so not in its list.
            extra: Option<BufferAccess>,
            command: Command,
            state: State,
        }

        pub struct RasterBuilder<'a, const D: usize> {
            rec: &'a mut Recorder,
            att: Attachments<'a>,
            draws: [Draw<'a>; D],
        }

        pub fn raster(rec: &mut Recorder, extent: UVec2) -> RasterBuilder<'_, 0> {
            RasterBuilder {
                rec,
                att: Attachments::new(extent),
                draws: [],
            }
        }

        impl<'a, const D: usize> RasterBuilder<'a, D> {
            pub fn color(mut self, v: View<'a>) -> Self {
                self.att.color(v);
                self
            }

            pub fn depth(mut self, v: View<'a>) -> Self {
                self.att.depth(v);
                self
            }

            fn push(
                self,
                b: &'a Out<'a, ()>,
                extra: Option<BufferAccess>,
                command: Command,
                state: State,
            ) -> RasterBuilder<'a, { D + 1 }> {
                let draw = Draw {
                    b,
                    extra,
                    command,
                    state,
                };
                RasterBuilder {
                    rec: self.rec,
                    att: self.att,
                    draws: append(self.draws, draw),
                }
            }

            pub fn draw(
                self,
                b: &'a Out<'a, kind::RasterVertex>,
                vertices: u32,
                instances: u32,
                state: State,
            ) -> RasterBuilder<'a, { D + 1 }> {
                self.push(erase(b), None, Command::Draw { vertices, instances }, state)
            }

            pub fn draw_indexed(
                self,
                b: &'a Out<'a, kind::RasterVertex>,
                indices: Slice,
                state: State,
            ) -> RasterBuilder<'a, { D + 1 }> {
                let command = Command::Indexed {
                    address: indices.address,
                    size: indices.size,
                    count: (indices.size / 4) as u32,
                };
                self.push(erase(b), Some(index_access(indices)), command, state)
            }

            pub fn launch_indirect(
                self,
                b: &'a Out<'a, kind::RasterMesh>,
                draws: Slice,
                state: State,
            ) -> RasterBuilder<'a, { D + 1 }> {
                let command = Command::LaunchIndirect {
                    address: draws.address,
                    size: draws.size,
                    stride: 12,
                    count: 1,
                };
                self.push(erase(b), Some(indirect_access(draws)), command, state)
            }

            pub fn record(self) {
                let Self { rec, att, draws } = self;
                rec.flush(
                    attachment_images(&att).chain(draws.iter().flat_map(|d| images(d.b))),
                    draws
                        .iter()
                        .flat_map(|d| buffers(d.b).chain(extra_buffers(&d.extra))),
                );
                rec.begin(att.extent);
                for d in &draws {
                    rec.bind_pipeline(d.b.pass, d.b.constants.bytes());
                    rec.push(&d.b.push_constants.0);
                    rec.stream.push(d.state.blends[0] | d.state.depth_write as u8);
                    rec.draw(&d.command);
                }
                rec.end();
            }
        }

        macro_rules! b {
            ($e:expr) => {
                &$e
            };
        }

        workloads!();
    };
}

mod v2 {
    use super::*;

    #[derive(Clone)]
    pub enum Acc<'a> {
        Image(ImageUse<'a>),
        Buffer(BufferAccess),
    }

    fn sampled<'a>(v: View<'a>, s: S) -> Acc<'a> {
        Acc::Image(image_use(v, s, SAMPLED_READ, L::GENERAL))
    }
    fn storage_read<'a>(v: View<'a>, s: S) -> Acc<'a> {
        Acc::Image(image_use(v, s, STORAGE_READ, L::GENERAL))
    }
    fn storage_write<'a>(v: View<'a>, s: S) -> Acc<'a> {
        Acc::Image(image_use(v, s, STORAGE_WRITE, L::GENERAL))
    }
    fn unified_write<'a>(v: View<'a>, s: S) -> Acc<'a> {
        Acc::Image(image_use(v, s, UNIFIED_WRITE, L::GENERAL))
    }
    fn read<'a>(b: Slice, s: S) -> Acc<'a> {
        Acc::Buffer(buffer_access(b, s, STORAGE_READ))
    }
    fn write<'a>(b: Slice, s: S) -> Acc<'a> {
        Acc::Buffer(buffer_access(b, s, STORAGE_WRITE))
    }

    // Lava's types already hold the stage: nothing to expand, the iterators yield references.
    fn images<'s, 'a, K>(b: &'s Out<'a, K>) -> impl Iterator<Item = &'s ImageUse<'a>> + Clone {
        b.accesses.iter().filter_map(|a| match a {
            Acc::Image(i) => Some(i),
            Acc::Buffer(_) => None,
        })
    }
    fn buffers<'s, 'a, K>(b: &'s Out<'a, K>) -> impl Iterator<Item = &'s BufferAccess> + Clone {
        b.accesses.iter().filter_map(|a| match a {
            Acc::Buffer(b) => Some(b),
            Acc::Image(_) => None,
        })
    }

    fn attachment_images<'s, 'a>(att: &'s Attachments<'a>) -> impl Iterator<Item = &'s ImageUse<'a>> + Clone {
        att.uses()
    }
    fn extra_buffers(extra: &Option<BufferAccess>) -> impl Iterator<Item = &BufferAccess> + Clone {
        extra.iter()
    }

    unsized_variant!();

    pub fn access_size() -> usize {
        size_of::<Acc<'static>>()
    }
}

mod v3 {
    use super::*;

    /// 20 bytes: header, image handle or buffer address, layouts pointer or buffer size.
    #[derive(Clone, Copy)]
    #[repr(C, packed(4))]
    pub struct Acc<'a> {
        header: u32,
        handle_or_address: u64,
        layouts_or_size: u64,
        _borrow: PhantomData<&'a AtomicU32>,
    }
    const _: () = assert!(size_of::<Acc<'static>>() == 20);

    impl<'a> Acc<'a> {
        fn new((header, handle_or_address, layouts_or_size): (u32, u64, u64)) -> Self {
            Self {
                header,
                handle_or_address,
                layouts_or_size,
                _borrow: PhantomData,
            }
        }

        fn is_image(self) -> bool {
            self.header & IMAGE != 0
        }

        fn image_use(self, stage: S) -> ImageUse<'a> {
            expand_image(self.header, self.handle_or_address, self.layouts_or_size, stage)
        }

        fn buffer_access(self, stage: S) -> BufferAccess {
            expand_buffer(self.header, self.handle_or_address, self.layouts_or_size, stage)
        }
    }

    fn sampled<'a>(v: View<'a>, _: S) -> Acc<'a> {
        Acc::new(compress_image(&v, false))
    }
    fn storage_read<'a>(v: View<'a>, _: S) -> Acc<'a> {
        Acc::new(compress_image(&v, false))
    }
    fn storage_write<'a>(v: View<'a>, _: S) -> Acc<'a> {
        Acc::new(compress_image(&v, true))
    }
    fn unified_write<'a>(v: View<'a>, _: S) -> Acc<'a> {
        Acc::new(compress_image(&v, true))
    }
    fn read<'a>(b: Slice, _: S) -> Acc<'a> {
        Acc::new(compress_buffer(b, false))
    }
    fn write<'a>(b: Slice, _: S) -> Acc<'a> {
        Acc::new(compress_buffer(b, true))
    }

    fn images<'s, 'a, K>(b: &'s Out<'a, K>) -> impl Iterator<Item = ImageUse<'a>> + Clone + 's {
        let stage = b.stage;
        b.accesses
            .iter()
            .filter(|a| a.is_image())
            .map(move |a| a.image_use(stage))
    }
    fn buffers<'s, 'a, K>(b: &'s Out<'a, K>) -> impl Iterator<Item = BufferAccess> + Clone + 's {
        let stage = b.stage;
        b.accesses
            .iter()
            .filter(|a| !a.is_image())
            .map(move |a| a.buffer_access(stage))
    }

    fn attachment_images<'s, 'a>(att: &'s Attachments<'a>) -> impl Iterator<Item = ImageUse<'a>> + Clone {
        att.uses().cloned()
    }
    fn extra_buffers(extra: &Option<BufferAccess>) -> impl Iterator<Item = BufferAccess> + Clone {
        extra.iter().cloned()
    }

    unsized_variant!();
}

// ---- Variant 4: one unsized byte stream -----------------------------------------------------
// Header (pass, push-constant size, accesses size, constants size: 4 x u32), then the push
// constants at their size, the 20-byte accesses, the specialization constants. The generator
// knows every size, so a binding is a sized `Bytes<P, M, C>` (all u8 arrays, contiguous) that
// derefs to the unsized `Out`; recording reads only the bytes. The stage comes from the pass.

mod v4 {
    use super::*;

    #[repr(C)]
    pub struct Bytes<'a, K, const P: usize, const M: usize, const C: usize> {
        header: [u8; 16],
        push: [u8; P],
        accesses: [[u8; 20]; M],
        constants: [u8; C],
        _p: PhantomData<(&'a AtomicU32, K)>,
    }

    #[repr(C)]
    pub struct Out<'a, K> {
        _p: PhantomData<(&'a AtomicU32, K)>,
        bytes: [u8],
    }

    fn header(pass: usize, p: usize, m: usize, c: usize) -> [u8; 16] {
        let mut header = [0; 16];
        for (i, v) in [pass, p, m, c].into_iter().enumerate() {
            header[i * 4..i * 4 + 4].copy_from_slice(&(v as u32).to_le_bytes());
        }
        header
    }

    fn encode((header, a, b): (u32, u64, u64)) -> [u8; 20] {
        let mut bytes = [0; 20];
        bytes[0..4].copy_from_slice(&header.to_le_bytes());
        bytes[4..12].copy_from_slice(&a.to_le_bytes());
        bytes[12..20].copy_from_slice(&b.to_le_bytes());
        bytes
    }

    fn decode(bytes: &[u8; 20]) -> (u32, u64, u64) {
        (
            u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            u64::from_le_bytes(bytes[4..12].try_into().unwrap()),
            u64::from_le_bytes(bytes[12..20].try_into().unwrap()),
        )
    }

    impl<'a, K, const P: usize, const M: usize> Bytes<'a, K, P, M, 0> {
        fn new<G: Pod>(pass: usize, bindings: &G, accesses: [[u8; 20]; M]) -> Self {
            Self {
                header: header(pass, P, M * 20, 0),
                push: bytes_of(bindings).try_into().unwrap(),
                accesses,
                constants: [],
                _p: PhantomData,
            }
        }

        fn with_constants<G: Pod, const C: usize, const M2: usize>(
            self,
            constants: &G,
            accesses: [[u8; 20]; M2],
        ) -> Bytes<'a, K, P, { M + M2 }, C>
        where
            [(); M + M2]:,
        {
            let mut header = self.header;
            header[8..12].copy_from_slice(&(((M + M2) * 20) as u32).to_le_bytes());
            header[12..16].copy_from_slice(&(C as u32).to_le_bytes());
            Bytes {
                header,
                push: self.push,
                accesses: concat(self.accesses, accesses),
                constants: bytes_of(constants).try_into().unwrap(),
                _p: PhantomData,
            }
        }
    }

    impl<'a, K, const P: usize, const M: usize, const C: usize> Bytes<'a, K, P, M, C> {
        fn with(self, access: [u8; 20]) -> Bytes<'a, K, P, { M + 1 }, C>
        where
            [(); M + 1]:,
        {
            let mut header = self.header;
            header[8..12].copy_from_slice(&(((M + 1) * 20) as u32).to_le_bytes());
            Bytes {
                header,
                push: self.push,
                accesses: append(self.accesses, access),
                constants: self.constants,
                _p: PhantomData,
            }
        }

        pub fn sampled_read(self, v: View<'a>) -> Bytes<'a, K, P, { M + 1 }, C>
        where
            [(); M + 1]:,
        {
            self.with(encode(compress_image(&v, false)))
        }

        pub fn storage_read(self, v: View<'a>) -> Bytes<'a, K, P, { M + 1 }, C>
        where
            [(); M + 1]:,
        {
            self.with(encode(compress_image(&v, false)))
        }
    }

    impl<'a, K, const P: usize, const M: usize, const C: usize> Deref for Bytes<'a, K, P, M, C> {
        type Target = Out<'a, K>;

        fn deref(&self) -> &Out<'a, K> {
            let bytes = std::ptr::slice_from_raw_parts(self as *const Self as *const u8, size_of::<Self>());
            // SAFETY: `Bytes` is `size_of::<Self>()` contiguous bytes (only u8 arrays), and
            // `Out` is those bytes behind a zero-sized marker.
            unsafe { &*(bytes as *const Out<'a, K>) }
        }
    }

    fn erase<'a, K>(b: &'a Out<'a, K>) -> &'a Out<'a, ()> {
        // SAFETY: `K` is a zero-sized marker; the layout and metadata are the same.
        unsafe { &*(b as *const Out<'a, K> as *const Out<'a, ()>) }
    }

    /// The sections of a binding, read from its header.
    #[derive(Clone, Copy)]
    struct Parts<'a> {
        pass: usize,
        stage: S,
        push: &'a [u8],
        accesses: &'a [[u8; 20]],
        constants: &'a [u8],
    }

    impl<'a> Parts<'a> {
        fn of<K>(b: &'a Out<'a, K>) -> Self {
            let bytes = &b.bytes;
            let field = |i: usize| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()) as usize;
            let (pass, p, m, c) = (field(0), field(1), field(2), field(3));
            let (push, rest) = bytes[16..].split_at(p);
            let (accesses, rest) = rest.split_at(m);
            Self {
                pass,
                stage: PASS_STAGES[pass],
                push,
                accesses: accesses.as_chunks::<20>().0,
                constants: &rest[..c],
            }
        }

        fn images(self) -> impl Iterator<Item = ImageUse<'a>> + Clone {
            let stage = self.stage;
            self.accesses.iter().filter_map(move |a| {
                let (header, image, layouts) = decode(a);
                (header & IMAGE != 0).then(|| expand_image(header, image, layouts, stage))
            })
        }

        fn buffers(self) -> impl Iterator<Item = BufferAccess> + Clone {
            let stage = self.stage;
            self.accesses.iter().filter_map(move |a| {
                let (header, address, size) = decode(a);
                (header & IMAGE == 0).then(|| expand_buffer(header, address, size, stage))
            })
        }
    }

    fn sampled<'a>(v: View<'a>) -> [u8; 20] {
        encode(compress_image(&v, false))
    }
    fn storage_write<'a>(v: View<'a>) -> [u8; 20] {
        encode(compress_image(&v, true))
    }
    fn unified_write<'a>(v: View<'a>) -> [u8; 20] {
        encode(compress_image(&v, true))
    }
    fn read(b: Slice) -> [u8; 20] {
        encode(compress_buffer(b, false))
    }
    fn write(b: Slice) -> [u8; 20] {
        encode(compress_buffer(b, true))
    }

    pub fn instance_cull<'a>(w: &'a World, f: &Frame) -> Bytes<'a, kind::Compute, 112, 5, 0> {
        Bytes::new(INSTANCE_CULL, &instance_cull_gpu(w, f), [
            read(w.roots),
            read(w.aabbs),
            read(w.transforms),
            write(w.stack),
            write(w.variables),
        ])
    }

    pub fn bvh_cull<'a>(w: &'a World, f: &Frame) -> Bytes<'a, kind::Compute, 240, 10, 20> {
        Bytes::new(BVH_CULL, &bvh_cull_gpu(w, f), [
            write(w.stack),
            write(w.variables),
            write(w.meshlets),
            write(w.blended),
            write(w.outlined),
            write(w.candidates),
            write(w.batches),
            read(w.transforms),
            read(w.headers),
            read(w.flags),
        ])
        .with_constants(&bvh_cull_consts(w), [])
    }

    pub fn raster_pass<'a>(
        w: &'a World,
        f: &Frame,
        pass: usize,
        meshlets: Slice,
        draw: Slice,
    ) -> Bytes<'a, kind::RasterMesh, 176, 4, 4> {
        Bytes::new(pass, &raster_gpu(w, f, meshlets, draw), [
            read(w.transforms),
            read(meshlets),
            read(draw),
            read(w.materials),
        ])
        .with_constants(&128u32, [])
    }

    pub fn composite<'a>(w: &'a World, f: &Frame) -> Bytes<'a, kind::Compute, 192, 5, 4> {
        Bytes::new(COMPOSITE, &composite_gpu(w, f), [
            sampled(w.gbuffer.whole()),
            sampled(w.depth.whole()),
            storage_write(w.accum.whole()),
            unified_write(w.revealage.whole()),
            storage_write(w.target.whole()),
        ])
        .with_constants(&7u32, [])
    }

    pub fn draw_outline<'a>(w: &'a World, f: &Frame) -> Bytes<'a, kind::Compute, 48, 2, 0> {
        Bytes::new(DRAW_OUTLINE, &draw_outline_gpu(w, f), [
            sampled(w.outline_mask.whole()),
            storage_write(w.target.whole()),
        ])
    }

    pub fn gizmos<'a>(w: &'a World, f: &Frame, shape: u32) -> Bytes<'a, kind::RasterVertex, 80, 1, 0> {
        Bytes::new(GIZMOS, &gizmos_gpu(w, f, shape), [read(w.gizmos)])
    }

    pub fn ui<'a>(w: &'a World) -> Bytes<'a, kind::RasterVertex, 8, 2, 8> {
        Bytes::new(UI, &UiGpu { vertices: w.ui_vertices.address }, [read(w.ui_vertices)])
            .with_constants(&w.font_atlas.handle, [sampled(w.font_atlas.whole())])
    }

    pub fn forward<'a>(w: &'a World, f: &Frame, i: usize) -> Bytes<'a, kind::RasterVertex, 96, 3, 0> {
        Bytes::new(FORWARD, &forward_gpu(w, f, i), [
            read(w.vertex_buffers[i]),
            read(w.transforms),
            read(w.materials),
        ])
    }

    pub fn downsample<'a>(w: &'a World, f: &Frame, i: u32) -> Bytes<'a, kind::Compute, 24, 2, 0> {
        Bytes::new(DOWNSAMPLE, &downsample_gpu(w, f, i), [
            sampled(w.chain.mip(i)),
            storage_write(w.chain.mip(i + 1)),
        ])
    }

    pub fn mip_read<'a>(w: &'a World) -> Bytes<'a, kind::Compute, 24, 2, 0> {
        Bytes::new(MIP_READ, &mip_read_gpu(w), [
            sampled(w.chain.whole()),
            storage_write(w.target.whole()),
        ])
    }

    #[inline(never)]
    pub fn compute(rec: &mut Recorder, b: &Out<'_, kind::Compute>, groups: [u32; 3]) {
        let parts = Parts::of(b);
        rec.flush(parts.images(), parts.buffers());
        rec.push(parts.push);
        rec.bind_pipeline(parts.pass, parts.constants);
        rec.dispatch(groups);
    }

    struct Draw<'a> {
        b: &'a Out<'a, ()>,
        extra: Option<BufferAccess>,
        command: Command,
        state: State,
    }

    pub struct RasterBuilder<'a, const D: usize> {
        rec: &'a mut Recorder,
        att: Attachments<'a>,
        draws: [Draw<'a>; D],
    }

    pub fn raster(rec: &mut Recorder, extent: UVec2) -> RasterBuilder<'_, 0> {
        RasterBuilder {
            rec,
            att: Attachments::new(extent),
            draws: [],
        }
    }

    impl<'a, const D: usize> RasterBuilder<'a, D> {
        pub fn color(mut self, v: View<'a>) -> Self {
            self.att.color(v);
            self
        }

        pub fn depth(mut self, v: View<'a>) -> Self {
            self.att.depth(v);
            self
        }

        fn push(
            self,
            b: &'a Out<'a, ()>,
            extra: Option<BufferAccess>,
            command: Command,
            state: State,
        ) -> RasterBuilder<'a, { D + 1 }> {
            let draw = Draw {
                b,
                extra,
                command,
                state,
            };
            RasterBuilder {
                rec: self.rec,
                att: self.att,
                draws: append(self.draws, draw),
            }
        }

        pub fn draw(
            self,
            b: &'a Out<'a, kind::RasterVertex>,
            vertices: u32,
            instances: u32,
            state: State,
        ) -> RasterBuilder<'a, { D + 1 }> {
            self.push(erase(b), None, Command::Draw { vertices, instances }, state)
        }

        pub fn draw_indexed(
            self,
            b: &'a Out<'a, kind::RasterVertex>,
            indices: Slice,
            state: State,
        ) -> RasterBuilder<'a, { D + 1 }> {
            let command = Command::Indexed {
                address: indices.address,
                size: indices.size,
                count: (indices.size / 4) as u32,
            };
            self.push(erase(b), Some(index_access(indices)), command, state)
        }

        pub fn launch_indirect(
            self,
            b: &'a Out<'a, kind::RasterMesh>,
            draws: Slice,
            state: State,
        ) -> RasterBuilder<'a, { D + 1 }> {
            let command = Command::LaunchIndirect {
                address: draws.address,
                size: draws.size,
                stride: 12,
                count: 1,
            };
            self.push(erase(b), Some(indirect_access(draws)), command, state)
        }

        pub fn record(self) {
            let Self { rec, att, draws } = self;
            let parts = draws.each_ref().map(|d| Parts::of(d.b));
            rec.flush(
                att.uses().cloned().chain(parts.iter().flat_map(|p| p.images())),
                parts
                    .iter()
                    .zip(&draws)
                    .flat_map(|(p, d)| p.buffers().chain(d.extra.iter().cloned())),
            );
            rec.begin(att.extent);
            for (p, d) in parts.iter().zip(&draws) {
                rec.bind_pipeline(p.pass, p.constants);
                rec.push(p.push);
                rec.stream.push(d.state.blends[0] | d.state.depth_write as u8);
                rec.draw(&d.command);
            }
            rec.end();
        }
    }

    macro_rules! b {
        ($e:expr) => {
            &$e
        };
    }

    workloads!();
}

// ---- Variant 5: one unsized byte stream with uncompressed accesses -------------------------
// Header (pass, push-constant size, image accesses size, buffer accesses size, constants size:
// 5 x u32), then the push constants at their size, lava's `ImageUse`s (56 bytes each) and
// `BufferAccess`es (32 bytes each) as raw bytes, the specialization constants. Like variant 4,
// a binding is a sized `Bytes<P, I, B, C>` of u8 arrays that derefs to the unsized `Out`.

mod v5 {
    use super::*;

    const IMAGE_SIZE: usize = size_of::<ImageUse<'static>>();
    const BUFFER_SIZE: usize = size_of::<BufferAccess>();
    // Neither has padding, so their bytes are fully initialised.
    const _: () = assert!(IMAGE_SIZE == 56 && BUFFER_SIZE == 32);

    #[repr(C)]
    pub struct Bytes<'a, K, const P: usize, const I: usize, const B: usize, const C: usize> {
        header: [u8; 20],
        push: [u8; P],
        images: [[u8; IMAGE_SIZE]; I],
        buffers: [[u8; BUFFER_SIZE]; B],
        constants: [u8; C],
        _p: PhantomData<(&'a AtomicU32, K)>,
    }

    #[repr(C)]
    pub struct Out<'a, K> {
        _p: PhantomData<(&'a AtomicU32, K)>,
        bytes: [u8],
    }

    fn header(pass: usize, p: usize, i: usize, b: usize, c: usize) -> [u8; 20] {
        let mut header = [0; 20];
        for (n, v) in [pass, p, i, b, c].into_iter().enumerate() {
            header[n * 4..n * 4 + 4].copy_from_slice(&(v as u32).to_le_bytes());
        }
        header
    }

    fn image_bytes(image: ImageUse<'_>) -> [u8; IMAGE_SIZE] {
        // SAFETY: `ImageUse` has no padding; the borrow it holds is read back with the same
        // lifetime by `Parts::images`.
        unsafe { std::mem::transmute::<ImageUse<'_>, [u8; IMAGE_SIZE]>(image) }
    }

    fn buffer_bytes(buffer: BufferAccess) -> [u8; BUFFER_SIZE] {
        // SAFETY: `BufferAccess` has no padding.
        unsafe { std::mem::transmute::<BufferAccess, [u8; BUFFER_SIZE]>(buffer) }
    }

    impl<'a, K, const P: usize, const I: usize, const B: usize> Bytes<'a, K, P, I, B, 0> {
        fn new<G: Pod>(
            pass: usize,
            bindings: &G,
            images: [ImageUse<'a>; I],
            buffers: [BufferAccess; B],
        ) -> Self {
            Self {
                header: header(pass, P, I * IMAGE_SIZE, B * BUFFER_SIZE, 0),
                push: bytes_of(bindings).try_into().unwrap(),
                images: images.map(image_bytes),
                buffers: buffers.map(buffer_bytes),
                constants: [],
                _p: PhantomData,
            }
        }

        fn with_constants<G: Pod, const C: usize, const I2: usize>(
            self,
            constants: &G,
            images: [ImageUse<'a>; I2],
        ) -> Bytes<'a, K, P, { I + I2 }, B, C>
        where
            [(); I + I2]:,
        {
            let pass = u32::from_le_bytes(self.header[0..4].try_into().unwrap()) as usize;
            Bytes {
                header: header(pass, P, (I + I2) * IMAGE_SIZE, B * BUFFER_SIZE, C),
                push: self.push,
                images: concat(self.images, images.map(image_bytes)),
                buffers: self.buffers,
                constants: bytes_of(constants).try_into().unwrap(),
                _p: PhantomData,
            }
        }
    }

    impl<'a, K, const P: usize, const I: usize, const B: usize, const C: usize> Bytes<'a, K, P, I, B, C> {
        fn stage(&self) -> S {
            PASS_STAGES[u32::from_le_bytes(self.header[0..4].try_into().unwrap()) as usize]
        }

        fn with(self, image: ImageUse<'a>) -> Bytes<'a, K, P, { I + 1 }, B, C>
        where
            [(); I + 1]:,
        {
            let mut header = self.header;
            header[8..12].copy_from_slice(&(((I + 1) * IMAGE_SIZE) as u32).to_le_bytes());
            Bytes {
                header,
                push: self.push,
                images: append(self.images, image_bytes(image)),
                buffers: self.buffers,
                constants: self.constants,
                _p: PhantomData,
            }
        }

        pub fn sampled_read(self, v: View<'a>) -> Bytes<'a, K, P, { I + 1 }, B, C>
        where
            [(); I + 1]:,
        {
            let s = self.stage();
            self.with(image_use(v, s, SAMPLED_READ, L::GENERAL))
        }

        pub fn storage_read(self, v: View<'a>) -> Bytes<'a, K, P, { I + 1 }, B, C>
        where
            [(); I + 1]:,
        {
            let s = self.stage();
            self.with(image_use(v, s, STORAGE_READ, L::GENERAL))
        }
    }

    impl<'a, K, const P: usize, const I: usize, const B: usize, const C: usize> Deref
        for Bytes<'a, K, P, I, B, C>
    {
        type Target = Out<'a, K>;

        fn deref(&self) -> &Out<'a, K> {
            let bytes = std::ptr::slice_from_raw_parts(self as *const Self as *const u8, size_of::<Self>());
            // SAFETY: `Bytes` is `size_of::<Self>()` contiguous bytes (only u8 arrays), and
            // `Out` is those bytes behind a zero-sized marker.
            unsafe { &*(bytes as *const Out<'a, K>) }
        }
    }

    fn erase<'a, K>(b: &'a Out<'a, K>) -> &'a Out<'a, ()> {
        // SAFETY: `K` is a zero-sized marker; the layout and metadata are the same.
        unsafe { &*(b as *const Out<'a, K> as *const Out<'a, ()>) }
    }

    #[derive(Clone, Copy)]
    struct Parts<'a> {
        pass: usize,
        push: &'a [u8],
        images: &'a [[u8; IMAGE_SIZE]],
        buffers: &'a [[u8; BUFFER_SIZE]],
        constants: &'a [u8],
    }

    impl<'a> Parts<'a> {
        fn of<K>(b: &'a Out<'a, K>) -> Self {
            let bytes = &b.bytes;
            let field = |n: usize| u32::from_le_bytes(bytes[n * 4..n * 4 + 4].try_into().unwrap()) as usize;
            let (pass, p, i, bu, c) = (field(0), field(1), field(2), field(3), field(4));
            let (push, rest) = bytes[20..].split_at(p);
            let (images, rest) = rest.split_at(i);
            let (buffers, rest) = rest.split_at(bu);
            Self {
                pass,
                push,
                images: images.as_chunks::<IMAGE_SIZE>().0,
                buffers: buffers.as_chunks::<BUFFER_SIZE>().0,
                constants: &rest[..c],
            }
        }

        fn images(self) -> impl Iterator<Item = ImageUse<'a>> + Clone {
            // SAFETY: written by `image_bytes` from an `ImageUse<'a>`.
            self.images
                .iter()
                .map(|bytes| unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const ImageUse<'a>) })
        }

        fn buffers(self) -> impl Iterator<Item = BufferAccess> + Clone {
            // SAFETY: written by `buffer_bytes` from a `BufferAccess`.
            self.buffers
                .iter()
                .map(|bytes| unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const BufferAccess) })
        }
    }

    fn img<'a>(v: View<'a>, s: S, access: A) -> ImageUse<'a> {
        image_use(v, s, access, L::GENERAL)
    }

    fn buf(b: Slice, s: S, access: A) -> BufferAccess {
        buffer_access(b, s, access)
    }

    pub fn instance_cull<'a>(w: &'a World, f: &Frame) -> Bytes<'a, kind::Compute, 112, 0, 5, 0> {
        let s = COMPUTE;
        Bytes::new(INSTANCE_CULL, &instance_cull_gpu(w, f), [], [
            buf(w.roots, s, STORAGE_READ),
            buf(w.aabbs, s, STORAGE_READ),
            buf(w.transforms, s, STORAGE_READ),
            buf(w.stack, s, STORAGE_WRITE),
            buf(w.variables, s, STORAGE_WRITE),
        ])
    }

    pub fn bvh_cull<'a>(w: &'a World, f: &Frame) -> Bytes<'a, kind::Compute, 240, 0, 10, 20> {
        let s = COMPUTE;
        Bytes::new(BVH_CULL, &bvh_cull_gpu(w, f), [], [
            buf(w.stack, s, STORAGE_WRITE),
            buf(w.variables, s, STORAGE_WRITE),
            buf(w.meshlets, s, STORAGE_WRITE),
            buf(w.blended, s, STORAGE_WRITE),
            buf(w.outlined, s, STORAGE_WRITE),
            buf(w.candidates, s, STORAGE_WRITE),
            buf(w.batches, s, STORAGE_WRITE),
            buf(w.transforms, s, STORAGE_READ),
            buf(w.headers, s, STORAGE_READ),
            buf(w.flags, s, STORAGE_READ),
        ])
        .with_constants(&bvh_cull_consts(w), [])
    }

    pub fn raster_pass<'a>(
        w: &'a World,
        f: &Frame,
        pass: usize,
        meshlets: Slice,
        draw: Slice,
    ) -> Bytes<'a, kind::RasterMesh, 176, 0, 4, 4> {
        let s = MESH;
        Bytes::new(pass, &raster_gpu(w, f, meshlets, draw), [], [
            buf(w.transforms, s, STORAGE_READ),
            buf(meshlets, s, STORAGE_READ),
            buf(draw, s, STORAGE_READ),
            buf(w.materials, s, STORAGE_READ),
        ])
        .with_constants(&128u32, [])
    }

    pub fn composite<'a>(w: &'a World, f: &Frame) -> Bytes<'a, kind::Compute, 192, 5, 0, 4> {
        let s = COMPUTE;
        Bytes::new(COMPOSITE, &composite_gpu(w, f), [
            img(w.gbuffer.whole(), s, SAMPLED_READ),
            img(w.depth.whole(), s, SAMPLED_READ),
            img(w.accum.whole(), s, STORAGE_WRITE),
            img(w.revealage.whole(), s, UNIFIED_WRITE),
            img(w.target.whole(), s, STORAGE_WRITE),
        ], [])
        .with_constants(&7u32, [])
    }

    pub fn draw_outline<'a>(w: &'a World, f: &Frame) -> Bytes<'a, kind::Compute, 48, 2, 0, 0> {
        let s = COMPUTE;
        Bytes::new(DRAW_OUTLINE, &draw_outline_gpu(w, f), [
            img(w.outline_mask.whole(), s, SAMPLED_READ),
            img(w.target.whole(), s, STORAGE_WRITE),
        ], [])
    }

    pub fn gizmos<'a>(w: &'a World, f: &Frame, shape: u32) -> Bytes<'a, kind::RasterVertex, 80, 0, 1, 0> {
        let s = VERTEX;
        Bytes::new(GIZMOS, &gizmos_gpu(w, f, shape), [], [buf(w.gizmos, s, STORAGE_READ)])
    }

    pub fn ui<'a>(w: &'a World) -> Bytes<'a, kind::RasterVertex, 8, 1, 1, 8> {
        let s = VERTEX;
        Bytes::new(UI, &UiGpu { vertices: w.ui_vertices.address }, [], [
            buf(w.ui_vertices, s, STORAGE_READ),
        ])
        .with_constants(&w.font_atlas.handle, [img(w.font_atlas.whole(), s, SAMPLED_READ)])
    }

    pub fn forward<'a>(w: &'a World, f: &Frame, i: usize) -> Bytes<'a, kind::RasterVertex, 96, 0, 3, 0> {
        let s = VERTEX;
        Bytes::new(FORWARD, &forward_gpu(w, f, i), [], [
            buf(w.vertex_buffers[i], s, STORAGE_READ),
            buf(w.transforms, s, STORAGE_READ),
            buf(w.materials, s, STORAGE_READ),
        ])
    }

    pub fn downsample<'a>(w: &'a World, f: &Frame, i: u32) -> Bytes<'a, kind::Compute, 24, 2, 0, 0> {
        let s = COMPUTE;
        Bytes::new(DOWNSAMPLE, &downsample_gpu(w, f, i), [
            img(w.chain.mip(i), s, SAMPLED_READ),
            img(w.chain.mip(i + 1), s, STORAGE_WRITE),
        ], [])
    }

    pub fn mip_read<'a>(w: &'a World) -> Bytes<'a, kind::Compute, 24, 2, 0, 0> {
        let s = COMPUTE;
        Bytes::new(MIP_READ, &mip_read_gpu(w), [
            img(w.chain.whole(), s, SAMPLED_READ),
            img(w.target.whole(), s, STORAGE_WRITE),
        ], [])
    }

    #[inline(never)]
    pub fn compute(rec: &mut Recorder, b: &Out<'_, kind::Compute>, groups: [u32; 3]) {
        let parts = Parts::of(b);
        rec.flush(parts.images(), parts.buffers());
        rec.push(parts.push);
        rec.bind_pipeline(parts.pass, parts.constants);
        rec.dispatch(groups);
    }

    struct Draw<'a> {
        b: &'a Out<'a, ()>,
        extra: Option<BufferAccess>,
        command: Command,
        state: State,
    }

    pub struct RasterBuilder<'a, const D: usize> {
        rec: &'a mut Recorder,
        att: Attachments<'a>,
        draws: [Draw<'a>; D],
    }

    pub fn raster(rec: &mut Recorder, extent: UVec2) -> RasterBuilder<'_, 0> {
        RasterBuilder {
            rec,
            att: Attachments::new(extent),
            draws: [],
        }
    }

    impl<'a, const D: usize> RasterBuilder<'a, D> {
        pub fn color(mut self, v: View<'a>) -> Self {
            self.att.color(v);
            self
        }

        pub fn depth(mut self, v: View<'a>) -> Self {
            self.att.depth(v);
            self
        }

        fn push(
            self,
            b: &'a Out<'a, ()>,
            extra: Option<BufferAccess>,
            command: Command,
            state: State,
        ) -> RasterBuilder<'a, { D + 1 }> {
            let draw = Draw {
                b,
                extra,
                command,
                state,
            };
            RasterBuilder {
                rec: self.rec,
                att: self.att,
                draws: append(self.draws, draw),
            }
        }

        pub fn draw(
            self,
            b: &'a Out<'a, kind::RasterVertex>,
            vertices: u32,
            instances: u32,
            state: State,
        ) -> RasterBuilder<'a, { D + 1 }> {
            self.push(erase(b), None, Command::Draw { vertices, instances }, state)
        }

        pub fn draw_indexed(
            self,
            b: &'a Out<'a, kind::RasterVertex>,
            indices: Slice,
            state: State,
        ) -> RasterBuilder<'a, { D + 1 }> {
            let command = Command::Indexed {
                address: indices.address,
                size: indices.size,
                count: (indices.size / 4) as u32,
            };
            self.push(erase(b), Some(index_access(indices)), command, state)
        }

        pub fn launch_indirect(
            self,
            b: &'a Out<'a, kind::RasterMesh>,
            draws: Slice,
            state: State,
        ) -> RasterBuilder<'a, { D + 1 }> {
            let command = Command::LaunchIndirect {
                address: draws.address,
                size: draws.size,
                stride: 12,
                count: 1,
            };
            self.push(erase(b), Some(indirect_access(draws)), command, state)
        }

        pub fn record(self) {
            let Self { rec, att, draws } = self;
            let parts = draws.each_ref().map(|d| Parts::of(d.b));
            rec.flush(
                att.uses().cloned().chain(parts.iter().flat_map(|p| p.images())),
                parts
                    .iter()
                    .zip(&draws)
                    .flat_map(|(p, d)| p.buffers().chain(d.extra.iter().cloned())),
            );
            rec.begin(att.extent);
            for (p, d) in parts.iter().zip(&draws) {
                rec.bind_pipeline(p.pass, p.constants);
                rec.push(p.push);
                rec.stream.push(d.state.blends[0] | d.state.depth_write as u8);
                rec.draw(&d.command);
            }
            rec.end();
        }
    }

    macro_rules! b {
        ($e:expr) => {
            &$e
        };
    }

    workloads!();
}

// ---- Harness --------------------------------------------------------------------------------

type Workload = fn(&mut Recorder, &World, &Frame);

const N: usize = 5;
const VARIANTS: [&str; N] = ["smallvec", "unsized", "unsized+compressed", "byte stream", "bytes+uncompressed"];
const ROUNDS: usize = 101;

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn print_sizes(w: &World, f: &Frame) {
    println!("Binding sizes in bytes (v1 is one size for every pass; v2's access is {} bytes):", v2::access_size());
    println!("{:<14} {:>9} {:>9} {:>9} {:>9} {:>9}", "pass", VARIANTS[0], "unsized", "compr.", "bytes", "bytes+unc");
    macro_rules! row {
        ($name:literal, $($call:tt)*) => {
            println!(
                "{:<14} {:>9} {:>9} {:>9} {:>9} {:>9}",
                $name,
                v1::size(),
                size_of_val(&v2::$($call)*),
                size_of_val(&v3::$($call)*),
                size_of_val(&v4::$($call)*),
                size_of_val(&v5::$($call)*),
            );
        };
    }
    row!("InstanceCull", instance_cull(w, f));
    row!("BvhCull", bvh_cull(w, f));
    row!("Raster", raster_pass(w, f, RASTER, w.meshlets, w.opaque_draw));
    row!("Composite", composite(w, f));
    row!("DrawOutline", draw_outline(w, f));
    row!("Gizmos", gizmos(w, f, 0));
    row!("Ui", ui(w));
    row!("Forward+2", forward(w, f, 0).sampled_read(w.textures[0].whole()).sampled_read(w.textures[1].whole()));
    row!("Downsample", downsample(w, f, 0));
    println!();
}

fn main() {
    let world = World::new();
    let frame = Frame::new();
    print_sizes(&world, &frame);

    let scenarios: [(&str, [Workload; N]); 4] = [
        ("frame", [v1::frame, v2::frame, v3::frame, v4::frame, v5::frame]),
        ("many_draws", [v1::many_draws, v2::many_draws, v3::many_draws, v4::many_draws, v5::many_draws]),
        ("mip_chain", [v1::mip_chain, v2::mip_chain, v3::mip_chain, v4::mip_chain, v5::mip_chain]),
        ("ui_textures", [v1::ui_textures, v2::ui_textures, v3::ui_textures, v4::ui_textures, v5::ui_textures]),
    ];

    println!(
        "Median ns per scenario run over {ROUNDS} rounds (variants interleaved); in parentheses relative to smallvec, then heap allocations per run."
    );
    println!(
        "{:<12} {:<7} {:>22} {:>22} {:>22} {:>22} {:>22}",
        "scenario", "mode", VARIANTS[0], VARIANTS[1], VARIANTS[2], VARIANTS[3], VARIANTS[4]
    );
    let mut check = 0u64;
    for model_barriers in [true, false] {
        let mode = if model_barriers { "full" } else { "decode" };
        for (name, workloads) in &scenarios {
            let mut recs: [Recorder; N] = std::array::from_fn(|_| Recorder::new(model_barriers));
            // Warm up (pipeline maps, stream capacity, tracked layouts), then check that every
            // variant recorded the same work.
            let mut stats = [Stats::default(); N];
            let mut allocations = [0u64; N];
            for v in 0..N {
                for _ in 0..200 {
                    recs[v].reset();
                    workloads[v](&mut recs[v], &world, &frame);
                }
                recs[v].stats = Stats::default();
                recs[v].reset();
                let before = ALLOCATIONS.load(Ordering::Relaxed);
                workloads[v](&mut recs[v], &world, &frame);
                allocations[v] = ALLOCATIONS.load(Ordering::Relaxed) - before;
                stats[v] = recs[v].stats;
            }
            for v in 1..N {
                let (a, b) = (stats[0], stats[v]);
                assert_eq!(
                    (a.images, a.buffers, a.draws, a.transitions),
                    (b.images, b.buffers, b.draws, b.transitions),
                    "{name}: {} recorded different work than smallvec",
                    VARIANTS[v]
                );
                if a.barriers != b.barriers {
                    println!("note: {name} {mode}: {} records {} barriers, smallvec {}", VARIANTS[v], b.barriers, a.barriers);
                }
            }

            // Batches of about 1 ms.
            let start = Instant::now();
            let mut calibration = 0;
            while start.elapsed().as_micros() < 2000 {
                recs[0].reset();
                workloads[0](&mut recs[0], &world, &black_box(frame));
                calibration += 1;
            }
            let iters = (calibration / 2).max(1);

            let mut samples = [[0f64; ROUNDS]; N];
            for round in 0..ROUNDS {
                for k in 0..N {
                    let v = (round + k) % N;
                    let (rec, workload) = (&mut recs[v], workloads[v]);
                    let start = Instant::now();
                    for _ in 0..iters {
                        let f = black_box(frame);
                        rec.reset();
                        workload(rec, &world, &f);
                    }
                    samples[v][round] = start.elapsed().as_nanos() as f64 / iters as f64;
                }
            }
            let medians: [f64; N] = std::array::from_fn(|v| median(&mut samples[v]));
            let cells: [String; N] = std::array::from_fn(|v| {
                format!(
                    "{:.0} ({:.2}x) {}a",
                    medians[v],
                    medians[v] / medians[0],
                    allocations[v]
                )
            });
            println!(
                "{:<12} {:<7} {:>22} {:>22} {:>22} {:>22} {:>22}",
                name, mode, cells[0], cells[1], cells[2], cells[3], cells[4]
            );
            for rec in &recs {
                check = check.wrapping_add(rec.check).wrapping_add(rec.stream.len() as u64);
            }
            if model_barriers {
                let s = stats[0];
                println!(
                    "{:<12} {:<7} per run: {} images, {} buffers, {} draws, {} barriers, {} transitions, push bytes {} vs byte streams {}",
                    "", "", s.images, s.buffers, s.draws, s.barriers, s.transitions, s.push_bytes, stats[3].push_bytes
                );
            }
        }
    }
    black_box(check);
}

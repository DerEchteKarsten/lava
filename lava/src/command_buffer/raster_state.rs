//! The state of a draw (`RasterState`): every rasterization, depth, stencil and blend setting, its setters gated by the features they need, and how it reaches the GPU: as dynamic state a rendering sets once and then per draw only where it changed, or, where the device can't set it dynamically (`dynamic-*` features), baked into the pipeline through its key (`Baked`)
use std::ops::Deref;

use crate::vk;

pub use crate::vk::{
    BlendFactor, BlendOp, ColorComponentFlags, CompareOp, ConservativeRasterizationModeEXT,
    CullModeFlags, FrontFace, LineRasterizationModeKHR, LogicOp, PolygonMode,
    PrimitiveTopology, ProvokingVertexModeEXT, StencilOp,
};

use smallvec::SmallVec;
#[allow(unused_imports)]
use crate::features::{Features, Gated, gate};
use crate::state::{Ctx, Functions};

/// How a color attachment combines a fragment's output `src` with the stored value `dst`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Blend {
    /// `src`, unblended.
    Replace,
    /// `src.rgb * src.a + dst.rgb * (1 - src.a)`; alpha becomes `src.a`.
    Alpha,
    /// `src + dst`.
    Add,
    /// `dst * (1 - src)`: `src` is how much of `dst` the fragment hides.
    Attenuate,
    /// `dst`: the draw leaves the attachment untouched.
    Skip,
    /// Any blend equation (premultiplied alpha, min/max, the blend constants).
    Equation(BlendEquation),
}

/// `color = src_color_factor * src.rgb (color_op) dst_color_factor * dst.rgb`, alpha alike.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlendEquation {
    pub src_color: BlendFactor,
    pub dst_color: BlendFactor,
    pub color_op: BlendOp,
    pub src_alpha: BlendFactor,
    pub dst_alpha: BlendFactor,
    pub alpha_op: BlendOp,
}

impl Blend {
    /// The equation, or `None` for an unblended (or skipped) attachment.
    pub(crate) fn equation(self) -> Option<BlendEquation> {
        use BlendFactor as F;
        let both = |src, dst, src_alpha, dst_alpha| BlendEquation {
            src_color: src,
            dst_color: dst,
            color_op: BlendOp::ADD,
            src_alpha,
            dst_alpha,
            alpha_op: BlendOp::ADD,
        };
        match self {
            Blend::Replace | Blend::Skip => None,
            Blend::Alpha => Some(both(F::SRC_ALPHA, F::ONE_MINUS_SRC_ALPHA, F::ONE, F::ZERO)),
            Blend::Add => Some(both(F::ONE, F::ONE, F::ONE, F::ONE)),
            Blend::Attenuate => Some(both(
                F::ZERO,
                F::ONE_MINUS_SRC_COLOR,
                F::ZERO,
                F::ONE_MINUS_SRC_ALPHA,
            )),
            Blend::Equation(equation) => Some(equation),
        }
    }

    pub(crate) fn vk_equation(self) -> vk::ColorBlendEquationEXT {
        let e = self.equation().unwrap_or(BlendEquation {
            src_color: BlendFactor::ONE,
            dst_color: BlendFactor::ZERO,
            color_op: BlendOp::ADD,
            src_alpha: BlendFactor::ONE,
            dst_alpha: BlendFactor::ZERO,
            alpha_op: BlendOp::ADD,
        });
        vk::ColorBlendEquationEXT {
            src_color_blend_factor: e.src_color,
            dst_color_blend_factor: e.dst_color,
            color_blend_op: e.color_op,
            src_alpha_blend_factor: e.src_alpha,
            dst_alpha_blend_factor: e.dst_alpha,
            alpha_blend_op: e.alpha_op,
        }
    }
}

/// `constant + slope * max depth slope`, clamped to `clamp` (0 = unclamped).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthBias {
    pub constant: f32,
    pub slope: f32,
    pub clamp: f32,
}

/// Where depths are clamped instead of clipped.
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(dead_code)]
pub(crate) enum DepthClamp {
    /// To the viewport's depth range.
    Viewport,
    /// To `[min, max]`.
    #[cfg_attr(not(feature = "depth-clamp-control"), allow(dead_code))]
    Range { min: f32, max: f32 },
}

/// The stencil test of one face: `compare(reference & compare_mask, stored & compare_mask)`,
/// then one of the ops writes `write_mask` bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StencilFace {
    pub fail: StencilOp,
    pub pass: StencilOp,
    pub depth_fail: StencilOp,
    pub compare: CompareOp,
    pub compare_mask: u32,
    pub write_mask: u32,
    pub reference: u32,
}

impl Default for StencilFace {
    /// Always passes and keeps the stored value.
    fn default() -> Self {
        Self {
            fail: StencilOp::KEEP,
            pass: StencilOp::KEEP,
            depth_fail: StencilOp::KEEP,
            compare: CompareOp::ALWAYS,
            compare_mask: !0,
            write_mask: !0,
            reference: 0,
        }
    }
}

/// The stencil test of front and back faces.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stencil {
    pub front: StencilFace,
    pub back: StencilFace,
}

/// A line stipple: `pattern`'s bits, each repeated `factor` (1 to 256) times.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LineStipple {
    pub factor: u32,
    pub pattern: u16,
}

/// The state of one draw into `COLOR_ATTACHMENTS` color attachments. Draws that differ only
/// here share a pipeline wherever the device has the state as dynamic state; elsewhere the
/// state is part of the pipeline's key (the `dynamic-*` features). The defaults draw filled,
/// back-face-culled triangles with clockwise front faces, depth-tested with `GREATER`
/// (reversed depth) and depth-writing when the rendering has a depth attachment, and
/// replace every color attachment. Setters of state that needs an optional feature return a
/// [`Gated`] state.
#[derive(Clone, Debug, PartialEq)]
pub struct RasterState<const COLOR_ATTACHMENTS: usize> {
    pub(super) topology: PrimitiveTopology,
    pub(super) primitive_restart: bool,
    pub(super) rasterizer_discard: bool,
    pub(super) cull_mode: CullModeFlags,
    pub(super) front_face: FrontFace,
    pub(super) polygon_mode: PolygonMode,
    pub(super) line_width: f32,
    pub(super) line_mode: LineRasterizationModeKHR,
    pub(super) line_stipple: Option<LineStipple>,
    pub(super) provoking_vertex: ProvokingVertexModeEXT,
    pub(super) conservative: ConservativeRasterizationModeEXT,
    pub(super) depth_test: Option<CompareOp>,
    pub(super) depth_write: bool,
    pub(super) depth_bounds: Option<[f32; 2]>,
    pub(super) depth_bias: Option<DepthBias>,
    pub(super) depth_clamp: Option<DepthClamp>,
    pub(super) depth_clip: bool,
    pub(super) depth_clip_negative_one_to_one: bool,
    pub(super) stencil: Option<Stencil>,
    pub(super) logic_op: Option<LogicOp>,
    pub(super) blend_constants: [f32; 4],
    pub(super) blends: [Blend; COLOR_ATTACHMENTS],
    pub(super) write_masks: [ColorComponentFlags; COLOR_ATTACHMENTS],
}

impl<const COLOR_ATTACHMENTS: usize> Default for RasterState<COLOR_ATTACHMENTS> {
    fn default() -> Self {
        Self {
            topology: PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart: false,
            rasterizer_discard: false,
            cull_mode: CullModeFlags::BACK,
            front_face: FrontFace::CLOCKWISE,
            polygon_mode: PolygonMode::FILL,
            line_width: 1.0,
            line_mode: LineRasterizationModeKHR::DEFAULT,
            line_stipple: None,
            provoking_vertex: ProvokingVertexModeEXT::FIRST_VERTEX,
            conservative: ConservativeRasterizationModeEXT::DISABLED,
            depth_test: Some(CompareOp::GREATER),
            depth_write: true,
            depth_bounds: None,
            depth_bias: None,
            depth_clamp: None,
            depth_clip: true,
            depth_clip_negative_one_to_one: false,
            stencil: None,
            logic_op: None,
            blend_constants: [0.0; 4],
            blends: [Blend::Replace; COLOR_ATTACHMENTS],
            write_masks: [ColorComponentFlags::RGBA; COLOR_ATTACHMENTS],
        }
    }
}

macro_rules! setters {
    ($($(#[doc = $doc:literal])* $name:ident: $ty:ty,)*) => {
        impl<const COLOR_ATTACHMENTS: usize> RasterState<COLOR_ATTACHMENTS> {
            $($(#[doc = $doc])* pub fn $name(mut self, $name: $ty) -> Self {
                self.$name = $name;
                self
            })*
        }
    };
}

setters! {
    /// The primitives vertex draws assemble (mesh shaders output their own).
    topology: PrimitiveTopology,
    /// Whether the index `!0` restarts strips and fans.
    primitive_restart: bool,
    /// Discards every primitive before rasterization.
    rasterizer_discard: bool,
    cull_mode: CullModeFlags,
    front_face: FrontFace,
    /// The depth test against the depth attachment; `None` draws without one.
    depth_test: Option<CompareOp>,
    /// Whether fragments that pass the depth test write their depth.
    depth_write: bool,
    depth_bias: Option<DepthBias>,
    /// The stencil test against the depth attachment's stencil; `None` draws without one.
    stencil: Option<Stencil>,
    /// The constants of `CONSTANT_*` blend factors.
    blend_constants: [f32; 4],
    /// The components each color attachment keeps of what blending makes.
    write_masks: [ColorComponentFlags; COLOR_ATTACHMENTS],
}

/// Setters of state that needs feature `$marker` (cargo feature `$cargo`), which exist only
/// with it.
macro_rules! gated_setters {
    ($($(#[doc = $doc:literal])* $cargo:literal => $marker:ident: $name:ident($state:ident $(, $arg:ident: $ty:ty)*) $body:block)*) => {
        impl<const COLOR_ATTACHMENTS: usize> RasterState<COLOR_ATTACHMENTS> {
            $($(#[doc = $doc])*
            #[cfg(feature = $cargo)]
            pub fn $name(mut self, $($arg: $ty),*) -> Gated<gate::$marker, Self> {
                crate::features::gate::<gate::$marker, _>(move || {
                    {
                        let $state = &mut self;
                        $body
                    }
                    self
                })
            })*
        }
    };
}

gated_setters! {
    /// Line or point polygons (`FILL` is the default).
    "fill-mode-non-solid" => FillModeNonSolid: polygon_mode(state, mode: PolygonMode) {
        state.polygon_mode = mode
    }
    /// Lines other than 1 wide.
    "wide-lines" => WideLines: line_width(state, width: f32) { state.line_width = width }
    /// Rectangular lines instead of the device's default.
    "rectangular-lines" => RectangularLines: rectangular_lines(state) {
        state.line_mode = LineRasterizationModeKHR::RECTANGULAR
    }
    /// Bresenham lines instead of the device's default.
    "bresenham-lines" => BresenhamLines: bresenham_lines(state) {
        state.line_mode = LineRasterizationModeKHR::BRESENHAM
    }
    /// Antialiased rectangular lines instead of the device's default.
    "smooth-lines" => SmoothLines: smooth_lines(state) {
        state.line_mode = LineRasterizationModeKHR::RECTANGULAR_SMOOTH
    }
    /// Stippled lines, in any line mode.
    "line-stipple" => LineStipple: line_stipple(state, stipple: Option<LineStipple>) {
        state.line_stipple = stipple
    }
    /// The vertex whose outputs flat-shaded attributes take (`FIRST_VERTEX` is the default).
    "provoking-vertex" => ProvokingVertex: provoking_vertex(state, mode: ProvokingVertexModeEXT) {
        state.provoking_vertex = mode
    }
    /// Over- or underestimating rasterization.
    "conservative-rasterization" => ConservativeRasterization: conservative(state, 
        mode: ConservativeRasterizationModeEXT
    ) {
        state.conservative = mode
    }
    /// Discards fragments whose stored depth is outside `[min, max]`.
    "depth-bounds" => DepthBounds: depth_bounds(state, bounds: Option<[f32; 2]>) {
        state.depth_bounds = bounds
    }
    /// Clamps depths to the viewport's depth range instead of clipping. Without
    /// `depth-clip-enable` on the device, clamped primitives are also not clipped.
    "depth-clamp" => DepthClamp: depth_clamp(state, on: bool) {
        state.depth_clamp = on.then_some(DepthClamp::Viewport)
    }
    /// Clamps depths to `[min, max]` instead of clipping.
    "depth-clamp-control" => DepthClampControl: depth_clamp_range(state, min: f32, max: f32) {
        state.depth_clamp = Some(DepthClamp::Range { min, max })
    }
    /// Whether primitives are clipped against the near and far planes (the default).
    "depth-clip-enable" => DepthClipEnable: depth_clip(state, on: bool) { state.depth_clip = on }
    /// Clip space depth from -1 to 1 (OpenGL's) instead of 0 to 1.
    "depth-clip-control" => DepthClipControl: depth_clip_negative_one_to_one(state, on: bool) {
        state.depth_clip_negative_one_to_one = on
    }
    /// Replaces blending with a logic op on integer attachments.
    "logic-op" => LogicOp: logic_op(state, op: Option<LogicOp>) { state.logic_op = op }
    /// One blend per color attachment, in attachment order ([`Self::blend`] gives all the
    /// same one, which needs no feature).
    "independent-blend" => IndependentBlend: blends(state, blends: [Blend; COLOR_ATTACHMENTS]) {
        state.blends = blends
    }
}

impl<const COLOR_ATTACHMENTS: usize> RasterState<COLOR_ATTACHMENTS> {
    /// The same blend for every color attachment.
    pub fn blend(mut self, blend: Blend) -> Self {
        self.blends = [blend; COLOR_ATTACHMENTS];
        self
    }

    /// The state without the per-attachment parts, and those: a draw keeps them in its own
    /// sections.
    pub(super) fn split(self) -> (RasterState<0>, [Blend; COLOR_ATTACHMENTS], [ColorComponentFlags; COLOR_ATTACHMENTS]) {
        let state = RasterState {
            topology: self.topology,
            primitive_restart: self.primitive_restart,
            rasterizer_discard: self.rasterizer_discard,
            cull_mode: self.cull_mode,
            front_face: self.front_face,
            polygon_mode: self.polygon_mode,
            line_width: self.line_width,
            line_mode: self.line_mode,
            line_stipple: self.line_stipple,
            provoking_vertex: self.provoking_vertex,
            conservative: self.conservative,
            depth_test: self.depth_test,
            depth_write: self.depth_write,
            depth_bounds: self.depth_bounds,
            depth_bias: self.depth_bias,
            depth_clamp: self.depth_clamp,
            depth_clip: self.depth_clip,
            depth_clip_negative_one_to_one: self.depth_clip_negative_one_to_one,
            stencil: self.stencil,
            logic_op: self.logic_op,
            blend_constants: self.blend_constants,
            blends: [],
            write_masks: [],
        };
        (state, self.blends, self.write_masks)
    }

    /// Whether the draw writes depth into a depth attachment.
    pub(super) fn writes_depth(&self) -> bool {
        self.depth_test.is_some() && self.depth_write
    }
}

/// Field `$field` of `$features` (the device's), or `false` if lava was compiled without cargo
/// feature `$cargo`.
macro_rules! on {
    ($features:expr, $cargo:literal, $field:ident) => {{
        #[allow(unused_variables)]
        let features: &Features = $features;
        #[allow(unused_mut, unused_assignments)]
        let mut on = false;
        #[cfg(feature = $cargo)]
        {
            on = features.$field;
        }
        on
    }};
}
pub(crate) use on;

/// Which of a draw's states the device sets dynamically, by the `dynamic-*` features (and the
/// capabilities whose state only exists with them).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Dynamic {
    pub(crate) blend: bool,
    pub(crate) polygon_mode: bool,
    pub(crate) depth_clamp: bool,
    pub(crate) depth_clamp_range: bool,
    pub(crate) depth_clip: bool,
    pub(crate) depth_clip_negative_one_to_one: bool,
    pub(crate) logic_op: bool,
    pub(crate) conservative: bool,
    pub(crate) line_mode: bool,
    pub(crate) line_stipple_enable: bool,
    pub(crate) line_stipple: bool,
    pub(crate) provoking_vertex: bool,
}

impl Dynamic {
    pub(crate) fn of(f: &Features) -> Self {
        Self {
            blend: on!(f, "dynamic-blend", dynamic_blend),
            polygon_mode: on!(f, "dynamic-polygon-mode", dynamic_polygon_mode),
            depth_clamp: on!(f, "dynamic-depth-clamp", dynamic_depth_clamp),
            depth_clamp_range: on!(f, "depth-clamp-control", depth_clamp_control),
            depth_clip: on!(f, "dynamic-depth-clip", dynamic_depth_clip),
            depth_clip_negative_one_to_one: on!(
                f,
                "dynamic-depth-clip-negative-one-to-one",
                dynamic_depth_clip_negative_one_to_one
            ),
            logic_op: on!(f, "dynamic-logic-op", dynamic_logic_op),
            conservative: on!(
                f,
                "dynamic-conservative-rasterization",
                dynamic_conservative_rasterization
            ),
            line_mode: on!(f, "dynamic-line-mode", dynamic_line_mode),
            line_stipple_enable: on!(f, "dynamic-line-stipple", dynamic_line_stipple),
            // The stipple's factor and pattern are dynamic wherever stippling exists.
            line_stipple: on!(f, "line-stipple", line_stipple),
            provoking_vertex: on!(f, "dynamic-provoking-vertex", dynamic_provoking_vertex),
        }
    }
}

/// The state a raster pipeline bakes in because the device can't set it dynamically: part of
/// its key. A field exists only where its capability is compiled and its `dynamic-*` feature
/// isn't asserted (`bake_*` cfgs, from `build.rs`), and holds the draw's value only where the
/// device lacks the dynamic state; elsewhere it keeps its default, so draws share pipelines.
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
pub(crate) struct Baked {
    #[cfg(bake_polygon_mode)]
    polygon_mode: PolygonMode,
    #[cfg(bake_depth_clamp)]
    depth_clamp: bool,
    #[cfg(bake_depth_clip)]
    depth_clip: bool,
    #[cfg(bake_depth_clip_negative_one_to_one)]
    depth_clip_negative_one_to_one: bool,
    #[cfg(bake_logic_op)]
    logic_op: Option<LogicOp>,
    #[cfg(bake_conservative_rasterization)]
    conservative: ConservativeRasterizationModeEXT,
    #[cfg(bake_line_mode)]
    line_mode: LineRasterizationModeKHR,
    #[cfg(bake_line_stipple)]
    line_stipple: bool,
    #[cfg(bake_provoking_vertex)]
    provoking_vertex: ProvokingVertexModeEXT,
}

#[allow(unused_variables, unused_mut)]
impl Baked {
    /// What a pipeline bakes in for a draw of `state` on a device with `dynamic`.
    pub(crate) fn of(state: &RasterState<0>, dynamic: &Dynamic) -> Self {
        let mut baked = Self::default();
        #[cfg(bake_polygon_mode)]
        if !dynamic.polygon_mode {
            baked.polygon_mode = state.polygon_mode;
        }
        #[cfg(bake_depth_clamp)]
        if !dynamic.depth_clamp {
            baked.depth_clamp = state.depth_clamp.is_some();
        }
        #[cfg(bake_depth_clip)]
        if !dynamic.depth_clip {
            baked.depth_clip = state.depth_clip;
        }
        #[cfg(bake_depth_clip_negative_one_to_one)]
        if !dynamic.depth_clip_negative_one_to_one {
            baked.depth_clip_negative_one_to_one = state.depth_clip_negative_one_to_one;
        }
        #[cfg(bake_logic_op)]
        if !dynamic.logic_op {
            baked.logic_op = state.logic_op;
        }
        #[cfg(bake_conservative_rasterization)]
        if !dynamic.conservative {
            baked.conservative = state.conservative;
        }
        #[cfg(bake_line_mode)]
        if !dynamic.line_mode {
            baked.line_mode = state.line_mode;
        }
        #[cfg(bake_line_stipple)]
        if !dynamic.line_stipple_enable {
            baked.line_stipple = state.line_stipple.is_some();
        }
        #[cfg(bake_provoking_vertex)]
        if !dynamic.provoking_vertex {
            baked.provoking_vertex = state.provoking_vertex;
        }
        baked
    }

    // What a pipeline is made with; where the state is dynamic (or has no field) it is ignored
    // or the default.
    pub(crate) fn polygon_mode(&self) -> PolygonMode {
        #[cfg(bake_polygon_mode)]
        return self.polygon_mode;
        #[cfg(not(bake_polygon_mode))]
        PolygonMode::FILL
    }
    pub(crate) fn depth_clamp(&self) -> bool {
        #[cfg(bake_depth_clamp)]
        return self.depth_clamp;
        #[cfg(not(bake_depth_clamp))]
        false
    }
    pub(crate) fn depth_clip(&self) -> bool {
        #[cfg(bake_depth_clip)]
        return self.depth_clip;
        #[cfg(not(bake_depth_clip))]
        true
    }
    pub(crate) fn depth_clip_negative_one_to_one(&self) -> bool {
        #[cfg(bake_depth_clip_negative_one_to_one)]
        return self.depth_clip_negative_one_to_one;
        #[cfg(not(bake_depth_clip_negative_one_to_one))]
        false
    }
    pub(crate) fn logic_op(&self) -> Option<LogicOp> {
        #[cfg(bake_logic_op)]
        return self.logic_op;
        #[cfg(not(bake_logic_op))]
        None
    }
    pub(crate) fn conservative(&self) -> ConservativeRasterizationModeEXT {
        #[cfg(bake_conservative_rasterization)]
        return self.conservative;
        #[cfg(not(bake_conservative_rasterization))]
        ConservativeRasterizationModeEXT::DISABLED
    }
    pub(crate) fn line_mode(&self) -> LineRasterizationModeKHR {
        #[cfg(bake_line_mode)]
        return self.line_mode;
        #[cfg(not(bake_line_mode))]
        LineRasterizationModeKHR::DEFAULT
    }
    pub(crate) fn line_stipple(&self) -> bool {
        #[cfg(bake_line_stipple)]
        return self.line_stipple;
        #[cfg(not(bake_line_stipple))]
        false
    }
    pub(crate) fn provoking_vertex(&self) -> ProvokingVertexModeEXT {
        #[cfg(bake_provoking_vertex)]
        return self.provoking_vertex;
        #[cfg(not(bake_provoking_vertex))]
        ProvokingVertexModeEXT::FIRST_VERTEX
    }
}

/// What a color attachment's write mask is with its blend: nothing for `Skip`.
pub(crate) fn effective_write_mask(blend: Blend, mask: ColorComponentFlags) -> ColorComponentFlags {
    match blend {
        Blend::Skip => ColorComponentFlags::empty(),
        _ => mask,
    }
}

/// Every pipeline's dynamic state: all of a draw's state besides the shaders, the attachment
/// formats and what the device can't set dynamically (`dynamic`). Mesh pipelines leave out
/// the input assembly ones, which they don't have.
pub(crate) fn dynamic_states(mesh: bool, dynamic: &Dynamic) -> Vec<vk::DynamicState> {
    use vk::DynamicState as D;
    let mut states = vec![
        D::VIEWPORT_WITH_COUNT,
        D::SCISSOR_WITH_COUNT,
        D::LINE_WIDTH,
        D::DEPTH_BIAS,
        D::BLEND_CONSTANTS,
        D::DEPTH_BOUNDS,
        D::STENCIL_COMPARE_MASK,
        D::STENCIL_WRITE_MASK,
        D::STENCIL_REFERENCE,
        D::CULL_MODE,
        D::FRONT_FACE,
        D::DEPTH_TEST_ENABLE,
        D::DEPTH_WRITE_ENABLE,
        D::DEPTH_COMPARE_OP,
        D::DEPTH_BOUNDS_TEST_ENABLE,
        D::STENCIL_TEST_ENABLE,
        D::STENCIL_OP,
        D::RASTERIZER_DISCARD_ENABLE,
        D::DEPTH_BIAS_ENABLE,
    ];
    let optional: [(bool, &[vk::DynamicState]); 12] = [
        (
            dynamic.blend,
            &[D::COLOR_BLEND_ENABLE_EXT, D::COLOR_BLEND_EQUATION_EXT, D::COLOR_WRITE_MASK_EXT],
        ),
        (dynamic.polygon_mode, &[D::POLYGON_MODE_EXT]),
        (dynamic.depth_clamp, &[D::DEPTH_CLAMP_ENABLE_EXT]),
        (dynamic.depth_clamp_range, &[vk::DYNAMIC_STATE_DEPTH_CLAMP_RANGE_EXT]),
        (dynamic.depth_clip, &[D::DEPTH_CLIP_ENABLE_EXT]),
        (dynamic.depth_clip_negative_one_to_one, &[D::DEPTH_CLIP_NEGATIVE_ONE_TO_ONE_EXT]),
        (dynamic.logic_op, &[D::LOGIC_OP_EXT, D::LOGIC_OP_ENABLE_EXT]),
        (dynamic.conservative, &[D::CONSERVATIVE_RASTERIZATION_MODE_EXT]),
        (dynamic.line_mode, &[D::LINE_RASTERIZATION_MODE_EXT]),
        (dynamic.line_stipple_enable, &[D::LINE_STIPPLE_ENABLE_EXT]),
        (dynamic.line_stipple, &[D::LINE_STIPPLE_KHR]),
        (dynamic.provoking_vertex, &[D::PROVOKING_VERTEX_MODE_EXT]),
    ];
    for (on, dynamic) in optional {
        if on {
            states.extend_from_slice(dynamic);
        }
    }
    if !mesh {
        states.extend([D::PRIMITIVE_TOPOLOGY, D::PRIMITIVE_RESTART_ENABLE]);
    }
    states
}

/// What the attachments of a rendering decide about its draws' state.
#[derive(Clone, Copy)]
pub(super) struct Rendering {
    pub(super) color_attachments: usize,
    pub(super) depth: bool,
    pub(super) stencil: bool,
}

/// The input assembly state last set in a rendering, which only vertex draws set.
pub(super) type Assembly = Option<(PrimitiveTopology, bool)>;

/// A draw's state as the draw keeps it: the [`RasterState`] without the per-attachment parts,
/// which it derefs to, and a blend and write mask per color attachment.
#[derive(Clone, Copy)]
pub(super) struct DrawState<'s> {
    pub(super) state: &'s RasterState<0>,
    pub(super) blends: &'s [Blend],
    pub(super) write_masks: &'s [ColorComponentFlags],
}

impl Deref for DrawState<'_> {
    type Target = RasterState<0>;

    fn deref(&self) -> &RasterState<0> {
        self.state
    }
}

impl DrawState<'_> {
    /// Records the dynamic state (`dynamic`) that differs from `last`, the state recorded
    /// before in this rendering (`None`: everything), for a draw of a rendering with
    /// `rendering`'s attachments. Vertex draws also set the input assembly that differs from
    /// `assembly`. What isn't dynamic is in the bound pipeline.
    pub(super) fn record(
        &self,
        last: Option<&Self>,
        rendering: Rendering,
        vertex: bool,
        assembly: &mut Assembly,
        dynamic: &Dynamic,
        cmd: vk::CommandBuffer,
    ) {
        let device = Ctx::device();
        let (d2, d3) = (Functions::dynamic_state2(), Functions::dynamic_state3());
        let n = rendering.color_attachments;
        debug_assert_eq!((self.blends.len(), self.write_masks.len()), (n, n));
        macro_rules! changed {
            ($field:ident) => {
                last.is_none_or(|last| last.$field != self.$field)
            };
            ($field:ident[..$n:ident]) => {
                last.is_none_or(|last| last.$field[..$n] != self.$field[..$n])
            };
        }
        unsafe {
            if vertex && *assembly != Some((self.topology, self.primitive_restart)) {
                device.cmd_set_primitive_topology(cmd, self.topology);
                device.cmd_set_primitive_restart_enable(cmd, self.primitive_restart);
                *assembly = Some((self.topology, self.primitive_restart));
            }
            if changed!(rasterizer_discard) {
                device.cmd_set_rasterizer_discard_enable(cmd, self.rasterizer_discard);
            }
            if changed!(cull_mode) {
                device.cmd_set_cull_mode(cmd, self.cull_mode);
            }
            if changed!(front_face) {
                device.cmd_set_front_face(cmd, self.front_face);
            }
            if dynamic.polygon_mode && changed!(polygon_mode) {
                d3.cmd_set_polygon_mode(cmd, self.polygon_mode);
            }
            if changed!(line_width) {
                device.cmd_set_line_width(cmd, self.line_width);
            }
            if dynamic.line_mode && changed!(line_mode) {
                d3.cmd_set_line_rasterization_mode(cmd, self.line_mode);
            }
            if changed!(line_stipple) {
                if dynamic.line_stipple_enable {
                    d3.cmd_set_line_stipple_enable(cmd, self.line_stipple.is_some());
                }
                if dynamic.line_stipple {
                    // Set whether used or not: every dynamic state of the pipeline must be.
                    let stipple = self.line_stipple.unwrap_or(LineStipple {
                        factor: 1,
                        pattern: !0,
                    });
                    Functions::line_rasterization().cmd_set_line_stipple(
                        cmd,
                        stipple.factor,
                        stipple.pattern,
                    );
                }
            }
            if dynamic.provoking_vertex && changed!(provoking_vertex) {
                d3.cmd_set_provoking_vertex_mode(cmd, self.provoking_vertex);
            }
            if dynamic.conservative && changed!(conservative) {
                d3.cmd_set_conservative_rasterization_mode(cmd, self.conservative);
            }

            if changed!(depth_test) || changed!(depth_write) {
                let test = rendering.depth && self.depth_test.is_some();
                device.cmd_set_depth_test_enable(cmd, test);
                device.cmd_set_depth_write_enable(cmd, test && self.depth_write);
                device.cmd_set_depth_compare_op(cmd, self.depth_test.unwrap_or(CompareOp::ALWAYS));
            }
            if changed!(depth_bounds) {
                let [min, max] = self.depth_bounds.unwrap_or([0.0, 1.0]);
                device.cmd_set_depth_bounds_test_enable(
                    cmd,
                    rendering.depth && self.depth_bounds.is_some(),
                );
                device.cmd_set_depth_bounds(cmd, min, max);
            }
            if changed!(depth_bias) {
                let bias = self.depth_bias.unwrap_or(DepthBias {
                    constant: 0.0,
                    slope: 0.0,
                    clamp: 0.0,
                });
                device.cmd_set_depth_bias_enable(cmd, self.depth_bias.is_some());
                device.cmd_set_depth_bias(cmd, bias.constant, bias.clamp, bias.slope);
            }
            if changed!(depth_clamp) {
                if dynamic.depth_clamp {
                    d3.cmd_set_depth_clamp_enable(cmd, self.depth_clamp.is_some());
                }
                if dynamic.depth_clamp_range {
                    let (mode, [min, max]) = match self.depth_clamp {
                        Some(DepthClamp::Range { min, max }) => {
                            (vk::DepthClampModeEXT::USER_DEFINED_RANGE, [min, max])
                        }
                        // The range is ignored.
                        _ => (vk::DepthClampModeEXT::VIEWPORT_RANGE, [0.0, 1.0]),
                    };
                    let range = vk::DepthClampRangeEXT {
                        min_depth_clamp: min,
                        max_depth_clamp: max,
                    };
                    Functions::depth_clamp_control().cmd_set_depth_clamp_range(cmd, mode, &range);
                }
            }
            if dynamic.depth_clip && changed!(depth_clip) {
                d3.cmd_set_depth_clip_enable(cmd, self.depth_clip);
            }
            if dynamic.depth_clip_negative_one_to_one && changed!(depth_clip_negative_one_to_one) {
                d3.cmd_set_depth_clip_negative_one_to_one(cmd, self.depth_clip_negative_one_to_one);
            }
            if changed!(stencil) {
                let stencil = self.stencil.unwrap_or_default();
                device.cmd_set_stencil_test_enable(
                    cmd,
                    rendering.stencil && self.stencil.is_some(),
                );
                for (face, s) in [
                    (vk::StencilFaceFlags::FRONT, stencil.front),
                    (vk::StencilFaceFlags::BACK, stencil.back),
                ] {
                    device.cmd_set_stencil_op(cmd, face, s.fail, s.pass, s.depth_fail, s.compare);
                    device.cmd_set_stencil_compare_mask(cmd, face, s.compare_mask);
                    device.cmd_set_stencil_write_mask(cmd, face, s.write_mask);
                    device.cmd_set_stencil_reference(cmd, face, s.reference);
                }
            }

            if dynamic.logic_op && changed!(logic_op) {
                d3.cmd_set_logic_op_enable(cmd, self.logic_op.is_some());
                d2.cmd_set_logic_op(cmd, self.logic_op.unwrap_or(LogicOp::COPY));
            }
            if changed!(blend_constants) {
                device.cmd_set_blend_constants(cmd, &self.blend_constants);
            }
            let blends_changed = n > 0 && changed!(blends[..n]);
            if dynamic.blend && blends_changed {
                let blends = &self.blends[..n];
                let enables: SmallVec<[vk::Bool32; 8]> =
                    (blends.iter()).map(|b| b.equation().is_some() as vk::Bool32).collect();
                let equations: SmallVec<[vk::ColorBlendEquationEXT; 8]> =
                    blends.iter().map(|b| b.vk_equation()).collect();
                d3.cmd_set_color_blend_enable(cmd, 0, &enables);
                d3.cmd_set_color_blend_equation(cmd, 0, &equations);
            }
            // A skipped attachment writes nothing.
            if dynamic.blend && (blends_changed || (n > 0 && changed!(write_masks[..n]))) {
                let masks: SmallVec<[ColorComponentFlags; 8]> = (self.blends[..n].iter())
                    .zip(&self.write_masks[..n])
                    .map(|(blend, mask)| effective_write_mask(*blend, *mask))
                    .collect();
                d3.cmd_set_color_write_mask(cmd, 0, &masks);
            }
        }
    }
}

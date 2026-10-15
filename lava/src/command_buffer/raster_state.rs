//! The state of a draw (`RasterState`): every rasterization, depth, stencil and blend setting, all of it dynamic state that a rendering sets once and then per draw only where it changed, so raster pipelines are keyed by their attachment formats and constants alone
use ash::vk;

pub use ash::vk::{
    BlendFactor, BlendOp, ColorComponentFlags, CompareOp, ConservativeRasterizationModeEXT,
    CullModeFlags, FrontFace, LineRasterizationModeKHR, LogicOp, PolygonMode,
    PrimitiveTopology, ProvokingVertexModeEXT, StencilOp,
};

use super::MAX_COLOR_ATTACHMENTS;
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
    fn equation(self) -> Option<BlendEquation> {
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

    fn vk_equation(self) -> vk::ColorBlendEquationEXT {
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
pub enum DepthClamp {
    /// To the viewport's depth range.
    Viewport,
    /// To `[min, max]`.
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

/// The state of one draw into `COLOR_ATTACHMENTS` color attachments. All of it is dynamic
/// state, so draws that differ only here share a pipeline. The defaults draw filled,
/// back-face-culled triangles with clockwise front faces, depth-tested with `GREATER`
/// (reversed depth) and depth-writing when the rendering has a depth attachment, and
/// replace every color attachment.
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
    /// `LINE` and `POINT` need `Features::fill_mode_non_solid`.
    polygon_mode: PolygonMode,
    /// Other than 1 needs `Features::wide_lines`.
    line_width: f32,
    /// `RECTANGULAR`, `BRESENHAM` and `RECTANGULAR_SMOOTH` need the matching `Features`.
    line_mode: LineRasterizationModeKHR,
    /// Needs the `Features::stippled_*_lines` of the line mode.
    line_stipple: Option<LineStipple>,
    provoking_vertex: ProvokingVertexModeEXT,
    conservative: ConservativeRasterizationModeEXT,
    /// The depth test against the depth attachment; `None` draws without one.
    depth_test: Option<CompareOp>,
    /// Whether fragments that pass the depth test write their depth.
    depth_write: bool,
    /// Discards fragments whose stored depth is outside `[min, max]`; needs
    /// `Features::depth_bounds`.
    depth_bounds: Option<[f32; 2]>,
    depth_bias: Option<DepthBias>,
    /// Needs `Features::depth_clamp`.
    depth_clamp: Option<DepthClamp>,
    /// Whether primitives are clipped against the near and far planes.
    depth_clip: bool,
    /// Clip space depth from -1 to 1 (OpenGL's) instead of 0 to 1.
    depth_clip_negative_one_to_one: bool,
    /// The stencil test against the depth attachment's stencil; `None` draws without one.
    stencil: Option<Stencil>,
    /// Replaces blending with a logic op on integer attachments; needs `Features::logic_op`.
    logic_op: Option<LogicOp>,
    /// The constants of `CONSTANT_*` blend factors.
    blend_constants: [f32; 4],
    /// One blend per color attachment, in attachment order; different ones need
    /// `Features::independent_blend`.
    blends: [Blend; COLOR_ATTACHMENTS],
    /// The components each color attachment keeps of what blending makes.
    write_masks: [ColorComponentFlags; COLOR_ATTACHMENTS],
}

impl<const COLOR_ATTACHMENTS: usize> RasterState<COLOR_ATTACHMENTS> {
    /// The same state in a list of any rendering's draws: the blends and masks past the first
    /// `COLOR_ATTACHMENTS` are never read.
    pub(super) fn widen(self) -> RasterState<MAX_COLOR_ATTACHMENTS> {
        let mut blends = [Blend::Replace; MAX_COLOR_ATTACHMENTS];
        blends[..COLOR_ATTACHMENTS].copy_from_slice(&self.blends);
        let mut write_masks = [ColorComponentFlags::RGBA; MAX_COLOR_ATTACHMENTS];
        write_masks[..COLOR_ATTACHMENTS].copy_from_slice(&self.write_masks);
        RasterState {
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
            blends,
            write_masks,
        }
    }

    /// Whether the draw writes depth into a depth attachment.
    pub(super) fn writes_depth(&self) -> bool {
        self.depth_test.is_some() && self.depth_write
    }
}

/// Every pipeline's dynamic state: all of a draw's state besides the shaders and attachment
/// formats. Mesh pipelines leave out the input assembly ones, which they don't have.
pub(crate) fn dynamic_states(mesh: bool) -> Vec<vk::DynamicState> {
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
        D::LOGIC_OP_EXT,
        D::POLYGON_MODE_EXT,
        D::DEPTH_CLAMP_ENABLE_EXT,
        D::DEPTH_CLAMP_RANGE_EXT,
        D::DEPTH_CLIP_ENABLE_EXT,
        D::DEPTH_CLIP_NEGATIVE_ONE_TO_ONE_EXT,
        D::RASTERIZATION_SAMPLES_EXT,
        D::SAMPLE_MASK_EXT,
        D::ALPHA_TO_COVERAGE_ENABLE_EXT,
        D::ALPHA_TO_ONE_ENABLE_EXT,
        D::SAMPLE_LOCATIONS_ENABLE_EXT,
        D::LOGIC_OP_ENABLE_EXT,
        D::COLOR_BLEND_ENABLE_EXT,
        D::COLOR_BLEND_EQUATION_EXT,
        D::COLOR_WRITE_MASK_EXT,
        D::COLOR_WRITE_ENABLE_EXT,
        D::CONSERVATIVE_RASTERIZATION_MODE_EXT,
        D::LINE_RASTERIZATION_MODE_EXT,
        D::LINE_STIPPLE_ENABLE_EXT,
        D::LINE_STIPPLE_KHR,
        D::PROVOKING_VERTEX_MODE_EXT,
    ];
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

impl<const MAX: usize> RasterState<MAX> {
    /// Records the state that differs from `last`, the state recorded before in this
    /// rendering (`None`: everything), for a draw of a rendering with `rendering`'s
    /// attachments. Vertex draws also set the input assembly that differs from `assembly`.
    pub(super) fn record(
        &self,
        last: Option<&Self>,
        rendering: Rendering,
        vertex: bool,
        assembly: &mut Assembly,
        cmd: vk::CommandBuffer,
    ) {
        let device = Ctx::device();
        let (d2, d3) = (Functions::dynamic_state2(), Functions::dynamic_state3());
        let features = Ctx::features();
        let n = rendering.color_attachments;
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
            if changed!(polygon_mode) {
                assert!(
                    self.polygon_mode == PolygonMode::FILL || features.fill_mode_non_solid,
                    "{:?} polygons need `Features::fill_mode_non_solid`",
                    self.polygon_mode
                );
                d3.cmd_set_polygon_mode(cmd, self.polygon_mode);
            }
            if changed!(line_width) {
                assert!(
                    self.line_width == 1.0 || features.wide_lines,
                    "lines {} wide need `Features::wide_lines`",
                    self.line_width
                );
                device.cmd_set_line_width(cmd, self.line_width);
            }
            if changed!(line_mode) || changed!(line_stipple) {
                use LineRasterizationModeKHR as L;
                let (supported, stippled) = match self.line_mode {
                    L::DEFAULT => (true, false),
                    L::RECTANGULAR => (features.rectangular_lines, features.stippled_rectangular_lines),
                    L::BRESENHAM => (features.bresenham_lines, features.stippled_bresenham_lines),
                    L::RECTANGULAR_SMOOTH => (features.smooth_lines, features.stippled_smooth_lines),
                    mode => panic!("unknown line mode {mode:?}"),
                };
                assert!(supported, "{:?} lines aren't supported (`Features`)", self.line_mode);
                assert!(
                    self.line_stipple.is_none() || stippled,
                    "stippled {:?} lines aren't supported (`Features::stippled_*_lines`)",
                    self.line_mode
                );
                d3.cmd_set_line_rasterization_mode(cmd, self.line_mode);
                d3.cmd_set_line_stipple_enable(cmd, self.line_stipple.is_some());
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
            if changed!(provoking_vertex) {
                d3.cmd_set_provoking_vertex_mode(cmd, self.provoking_vertex);
            }
            if changed!(conservative) {
                d3.cmd_set_conservative_rasterization_mode(cmd, self.conservative);
            }

            if changed!(depth_test) || changed!(depth_write) {
                let test = rendering.depth && self.depth_test.is_some();
                device.cmd_set_depth_test_enable(cmd, test);
                device.cmd_set_depth_write_enable(cmd, test && self.depth_write);
                device.cmd_set_depth_compare_op(cmd, self.depth_test.unwrap_or(CompareOp::ALWAYS));
            }
            if changed!(depth_bounds) {
                if self.depth_bounds.is_some() {
                    assert!(features.depth_bounds, "depth bounds need `Features::depth_bounds`");
                }
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
                if self.depth_clamp.is_some() {
                    assert!(features.depth_clamp, "depth clamping needs `Features::depth_clamp`");
                }
                d3.cmd_set_depth_clamp_enable(cmd, self.depth_clamp.is_some());
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
            if changed!(depth_clip) {
                d3.cmd_set_depth_clip_enable(cmd, self.depth_clip);
            }
            if changed!(depth_clip_negative_one_to_one) {
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

            if changed!(logic_op) {
                if self.logic_op.is_some() {
                    assert!(features.logic_op, "logic ops need `Features::logic_op`");
                }
                d3.cmd_set_logic_op_enable(cmd, self.logic_op.is_some());
                d2.cmd_set_logic_op(cmd, self.logic_op.unwrap_or(LogicOp::COPY));
            }
            if changed!(blend_constants) {
                device.cmd_set_blend_constants(cmd, &self.blend_constants);
            }
            if n > 0 && changed!(blends[..n]) {
                let blends = &self.blends[..n];
                assert!(
                    features.independent_blend || blends.iter().all(|b| *b == blends[0]),
                    "different blends per attachment need `Features::independent_blend`"
                );
                let enables: [vk::Bool32; MAX] = std::array::from_fn(|i| {
                    (i < n && self.blends[i].equation().is_some()) as vk::Bool32
                });
                let equations: [vk::ColorBlendEquationEXT; MAX] =
                    std::array::from_fn(|i| self.blends[i].vk_equation());
                let writes: [vk::Bool32; MAX] =
                    std::array::from_fn(|i| (self.blends[i] != Blend::Skip) as vk::Bool32);
                d3.cmd_set_color_blend_enable(cmd, 0, &enables[..n]);
                d3.cmd_set_color_blend_equation(cmd, 0, &equations[..n]);
                (Functions::color_write().fp().cmd_set_color_write_enable_ext)(
                    cmd,
                    n as u32,
                    writes.as_ptr(),
                );
            }
            if n > 0 && changed!(write_masks[..n]) {
                d3.cmd_set_color_write_mask(cmd, 0, &self.write_masks[..n]);
            }

            if last.is_none() {
                // One sample per pixel until lava has multisampled attachments.
                d3.cmd_set_rasterization_samples(cmd, vk::SampleCountFlags::TYPE_1);
                d3.cmd_set_sample_mask(cmd, vk::SampleCountFlags::TYPE_1, &[!0]);
                d3.cmd_set_alpha_to_coverage_enable(cmd, false);
                d3.cmd_set_alpha_to_one_enable(cmd, false);
                d3.cmd_set_sample_locations_enable(cmd, false);
            }
        }
    }
}

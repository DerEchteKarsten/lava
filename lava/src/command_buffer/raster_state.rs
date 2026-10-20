//! The state of a draw (`RasterState`): every rasterization, depth, stencil and blend setting plus its viewport and scissor (`Rect`), its setters gated by the features they need, the checks of its values against the device, and how it reaches the GPU: as dynamic state a rendering sets once and then per draw only where it changed, or, where the device can't set it dynamically (`dynamic-*` features, a topology class without `dynamicPrimitiveTopologyUnrestricted`), baked into the pipeline through its key (`Baked`)
use std::ops::Deref;

use crate::vk;
use glam::{I64Vec2, IVec2, UVec2};

pub use crate::vk::{
    BlendFactor, BlendOp, ColorComponentFlags, CompareOp, ConservativeRasterizationModeEXT,
    CullModeFlags, FrontFace, LineRasterizationModeKHR, LogicOp, PolygonMode,
    PrimitiveTopology, ProvokingVertexModeEXT, StencilOp,
};

use smallvec::SmallVec;
#[allow(unused_imports)]
use crate::features::{Features, Gated, On, gate};
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

#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(dead_code)]
pub(crate) enum DepthClamp {
    Viewport,
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

/// A rectangle of texels: a rendering's area, a viewport or a scissor. A `UVec2` is the
/// rectangle of that extent at the origin.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Rect {
    pub offset: IVec2,
    pub extent: UVec2,
}

impl Rect {
    pub const fn new(offset: IVec2, extent: UVec2) -> Self {
        Self { offset, extent }
    }

    pub(crate) fn vk(self) -> vk::Rect2D {
        vk::Rect2D {
            offset: vk::Offset2D {
                x: self.offset.x,
                y: self.offset.y,
            },
            extent: vk::Extent2D {
                width: self.extent.x,
                height: self.extent.y,
            },
        }
    }

    /// Panics unless it has texels.
    #[track_caller]
    pub(crate) fn check_viewport(self) {
        assert!(
            self.extent.cmpgt(UVec2::ZERO).all(),
            "a viewport of extent {} has no pixels",
            self.extent
        );
    }

    /// Panics unless it starts at a non-negative offset and ends below 2^31, as scissors and
    /// render areas do.
    #[track_caller]
    pub(crate) fn check_scissor(self, what: &str) {
        let end = self.offset.as_i64vec2() + self.extent.as_i64vec2();
        assert!(
            self.offset.cmpge(IVec2::ZERO).all() && end.cmple(I64Vec2::splat(i32::MAX as i64)).all(),
            "a {what} starts at a non-negative offset and ends below 2^31, not {} texels at {}",
            self.extent,
            self.offset
        );
    }
}

impl From<UVec2> for Rect {
    fn from(extent: UVec2) -> Self {
        Self {
            offset: IVec2::ZERO,
            extent,
        }
    }
}

/// The state of one draw into `COLOR_ATTACHMENTS` color attachments. The defaults draw
/// back-face-culled triangles with clockwise front faces, depth-tested with `GREATER` (reversed
/// depth) and depth-writing, replace every color attachment, and use the rendering's viewport
/// and scissor.
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
    /// `None`: clipped unless clamped.
    pub(super) depth_clip: Option<bool>,
    pub(super) depth_clip_negative_one_to_one: bool,
    pub(super) stencil: Option<Stencil>,
    pub(super) logic_op: Option<LogicOp>,
    pub(super) blend_constants: [f32; 4],
    pub(super) viewport: Option<Rect>,
    pub(super) scissor: Option<Rect>,
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
            depth_clip: None,
            depth_clip_negative_one_to_one: false,
            stencil: None,
            logic_op: None,
            blend_constants: [0.0; 4],
            viewport: None,
            scissor: None,
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
    /// Whether the index `!0` (of its index type) restarts strips and fans; lists can't restart.
    primitive_restart: bool,
    rasterizer_discard: bool,
    cull_mode: CullModeFlags,
    front_face: FrontFace,
    /// The depth test against the depth attachment; `None` draws without one.
    depth_test: Option<CompareOp>,
    depth_write: bool,
    depth_bias: Option<DepthBias>,
    /// The stencil test against the depth attachment's stencil; `None` draws without one.
    stencil: Option<Stencil>,
    /// The constants of `CONSTANT_*` blend factors.
    blend_constants: [f32; 4],
    /// The draw's viewport (depths 0 to 1); `None` is the rendering's
    /// (`RasterBuilder::viewport`), by default its area.
    viewport: Option<Rect>,
    /// The draw's scissor; `None` is the rendering's (`RasterBuilder::scissor`), by default its
    /// area. Draws of one rendering with different scissors clip it per draw (UI clipping).
    scissor: Option<Rect>,
}

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
    /// Whether primitives are clipped against the near and far planes; by default they are
    /// unless depths are clamped (`depth_clamp`), on every device.
    "depth-clip-enable" => DepthClipEnable: depth_clip(state, on: bool) {
        state.depth_clip = Some(on)
    }
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
    /// The components each color attachment keeps of what blending makes, in attachment order
    /// ([`Self::write_mask`] gives all the same one, which needs no feature).
    "independent-blend" => IndependentBlend: write_masks(state,
        masks: [ColorComponentFlags; COLOR_ATTACHMENTS]
    ) {
        state.write_masks = masks
    }
}

impl<const COLOR_ATTACHMENTS: usize> RasterState<COLOR_ATTACHMENTS> {
    /// The same blend for every color attachment.
    pub fn blend(mut self, blend: Blend) -> Self {
        self.blends = [blend; COLOR_ATTACHMENTS];
        self
    }

    /// The components every color attachment keeps of what blending makes (all by default).
    pub fn write_mask(mut self, mask: ColorComponentFlags) -> Self {
        self.write_masks = [mask; COLOR_ATTACHMENTS];
        self
    }

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
            viewport: self.viewport,
            scissor: self.scissor,
            blends: [],
            write_masks: [],
        };
        (state, self.blends, self.write_masks)
    }

    /// Whether primitives are clipped against the near and far planes: as set, else unless
    /// depths are clamped.
    pub(crate) fn effective_depth_clip(&self) -> bool {
        self.depth_clip.unwrap_or(self.depth_clamp.is_none())
    }

    /// Panics for values the device can't draw with: blend factors of a second output without
    /// `dual-src-blend` or beyond `maxFragmentDualSrcAttachments`, a stipple factor outside 1 to
    /// 256, a line width outside `lineWidthRange`, depth bounds or clamp ranges outside 0 to 1,
    /// underestimation without `primitiveUnderestimation`, primitive restart of lists, and
    /// viewports and scissors without texels or outside 2^31.
    #[track_caller]
    pub(crate) fn check(&self, blends: &[Blend], vertex: bool) {
        let physical = Ctx::physical_device();
        let limits = &physical.properties.limits;
        for (i, blend) in blends.iter().enumerate() {
            let Some(e) = blend.equation() else { continue };
            let src1 = [e.src_color, e.dst_color, e.src_alpha, e.dst_alpha].iter().any(|f| {
                matches!(
                    *f,
                    BlendFactor::SRC1_COLOR
                        | BlendFactor::ONE_MINUS_SRC1_COLOR
                        | BlendFactor::SRC1_ALPHA
                        | BlendFactor::ONE_MINUS_SRC1_ALPHA
                )
            });
            if !src1 {
                continue;
            }
            assert!(
                <gate::DualSrcBlend as On>::on(),
                "the blend of color attachment {i} reads a second fragment output (`SRC1_*`), \
                 which needs lava's `dual-src-blend` feature on the device (`dualSrcBlend`)"
            );
            assert!(
                (i as u32) < limits.max_fragment_dual_src_attachments,
                "the blend of color attachment {i} reads a second fragment output, which the \
                 device has for {} attachments (`maxFragmentDualSrcAttachments`)",
                limits.max_fragment_dual_src_attachments
            );
        }
        if let Some(stipple) = self.line_stipple {
            assert!(
                (1..=256).contains(&stipple.factor),
                "a line stipple repeats each bit 1 to 256 times, not {}",
                stipple.factor
            );
        }
        let [min, max] = limits.line_width_range;
        assert!(
            self.line_width == 1.0 || (min <= self.line_width && self.line_width <= max),
            "a line width of {} is outside the device's {min} to {max} (`lineWidthRange`)",
            self.line_width
        );
        let unit = |v: f32| (0.0..=1.0).contains(&v);
        if let Some([min, max]) = self.depth_bounds {
            assert!(
                unit(min) && unit(max),
                "depth bounds lie between 0 and 1, not {min} to {max}"
            );
        }
        if let Some(DepthClamp::Range { min, max }) = self.depth_clamp {
            assert!(
                unit(min) && unit(max) && min <= max,
                "a depth clamp range lies between 0 and 1 and starts at its minimum, not {min} to \
                 {max}"
            );
        }
        assert!(
            self.conservative != ConservativeRasterizationModeEXT::UNDERESTIMATE
                || physical.primitive_underestimation,
            "the device can't underestimate primitives (`primitiveUnderestimation`)"
        );
        if vertex && self.primitive_restart {
            use PrimitiveTopology as T;
            assert!(
                !matches!(
                    self.topology,
                    T::POINT_LIST
                        | T::LINE_LIST
                        | T::TRIANGLE_LIST
                        | T::LINE_LIST_WITH_ADJACENCY
                        | T::TRIANGLE_LIST_WITH_ADJACENCY
                        | T::PATCH_LIST
                ),
                "primitive restart cuts strips and fans, not {:?} (lava doesn't enable \
                 `primitiveTopologyListRestart`)",
                self.topology
            );
        }
        if let Some(viewport) = self.viewport {
            viewport.check_viewport();
        }
        if let Some(scissor) = self.scissor {
            scissor.check_scissor("scissor");
        }
    }
}

/// The class of a topology, which a pipeline bakes where the device can't leave it dynamically
/// (`dynamicPrimitiveTopologyUnrestricted`): its first topology.
pub(crate) fn topology_class(topology: PrimitiveTopology) -> PrimitiveTopology {
    use PrimitiveTopology as T;
    match topology {
        T::POINT_LIST => T::POINT_LIST,
        T::LINE_LIST | T::LINE_STRIP | T::LINE_LIST_WITH_ADJACENCY | T::LINE_STRIP_WITH_ADJACENCY => {
            T::LINE_LIST
        }
        T::PATCH_LIST => T::PATCH_LIST,
        _ => T::TRIANGLE_LIST,
    }
}

/// Field `$field` of `$features`, or `false` if lava was compiled without cargo feature `$cargo`.
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
    /// A dynamic topology may leave the class the pipeline was made with
    /// (`dynamicPrimitiveTopologyUnrestricted`, a property of the enabled
    /// `VK_EXT_extended_dynamic_state3`); else the class is baked.
    pub(crate) topology_unrestricted: bool,
}

impl Dynamic {
    /// `unrestricted`: the device's `dynamicPrimitiveTopologyUnrestricted`.
    pub(crate) fn of(f: &Features, unrestricted: bool) -> Self {
        let eds3 = ash::ext::extended_dynamic_state3::NAME;
        Self {
            topology_unrestricted: unrestricted && f.extensions().contains(&eds3),
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

/// A field exists only where its capability is compiled and its `dynamic-*` feature isn't
/// asserted (`bake_*` cfgs), and holds the draw's value only where the device lacks the dynamic
/// state, so draws share pipelines elsewhere.
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
pub(crate) struct Baked {
    /// The topology class of vertex pipelines (`topology_class`).
    pub(crate) topology: PrimitiveTopology,
    #[cfg(bake_polygon_mode)]
    pub(crate) polygon_mode: PolygonMode,
    #[cfg(bake_depth_clamp)]
    pub(crate) depth_clamp: bool,
    #[cfg(bake_depth_clip)]
    pub(crate) depth_clip: bool,
    #[cfg(bake_depth_clip_negative_one_to_one)]
    pub(crate) depth_clip_negative_one_to_one: bool,
    #[cfg(bake_logic_op)]
    pub(crate) logic_op: Option<LogicOp>,
    #[cfg(bake_conservative_rasterization)]
    pub(crate) conservative: ConservativeRasterizationModeEXT,
    #[cfg(bake_line_mode)]
    pub(crate) line_mode: LineRasterizationModeKHR,
    #[cfg(bake_line_stipple)]
    pub(crate) line_stipple: bool,
    #[cfg(bake_provoking_vertex)]
    pub(crate) provoking_vertex: ProvokingVertexModeEXT,
}

#[allow(unused_variables, unused_mut)]
impl Baked {
    /// `vertex`: a vertex pipeline, which has a topology.
    pub(crate) fn of(state: &RasterState<0>, dynamic: &Dynamic, vertex: bool) -> Self {
        let mut baked = Self::default();
        if vertex && !dynamic.topology_unrestricted {
            baked.topology = topology_class(state.topology);
        }
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
            baked.depth_clip = state.effective_depth_clip();
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
}

pub(crate) fn effective_write_mask(blend: Blend, mask: ColorComponentFlags) -> ColorComponentFlags {
    match blend {
        Blend::Skip => ColorComponentFlags::empty(),
        _ => mask,
    }
}

#[derive(Clone, Copy)]
pub(super) struct Rendering {
    pub(super) color_attachments: usize,
    pub(super) depth: bool,
    pub(super) stencil: bool,
    /// What draws without a viewport or scissor of their own use.
    pub(super) viewport: Rect,
    pub(super) scissor: Rect,
}

pub(super) type Assembly = Option<(PrimitiveTopology, bool)>;

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
    /// Records the dynamic state that differs from `last` (`None`: everything); what isn't
    /// dynamic is in the bound pipeline.
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
            let viewport = self.viewport.unwrap_or(rendering.viewport);
            if last.is_none_or(|last| last.viewport.unwrap_or(rendering.viewport) != viewport) {
                device.cmd_set_viewport_with_count(
                    cmd,
                    &[vk::Viewport {
                        x: viewport.offset.x as f32,
                        y: viewport.offset.y as f32,
                        width: viewport.extent.x as f32,
                        height: viewport.extent.y as f32,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    }],
                );
            }
            let scissor = self.scissor.unwrap_or(rendering.scissor);
            if last.is_none_or(|last| last.scissor.unwrap_or(rendering.scissor) != scissor) {
                device.cmd_set_scissor_with_count(cmd, &[scissor.vk()]);
            }
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
            let depth_clip = self.effective_depth_clip();
            if dynamic.depth_clip
                && last.is_none_or(|last| last.effective_depth_clip() != depth_clip)
            {
                d3.cmd_set_depth_clip_enable(cmd, depth_clip);
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
                // One set of calls where both faces are alike.
                let faces: &[_] = if stencil.front == stencil.back {
                    &[(vk::StencilFaceFlags::FRONT_AND_BACK, stencil.front)]
                } else {
                    &[
                        (vk::StencilFaceFlags::FRONT, stencil.front),
                        (vk::StencilFaceFlags::BACK, stencil.back),
                    ]
                };
                for &(face, s) in faces {
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

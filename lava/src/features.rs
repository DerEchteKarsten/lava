//! Device features: what lava needs of every device, the optional `Features` behind its cargo features (`X` checked at runtime, `assert-X` required at init and assumed, neither compiled out), the `Gated` values that check them, the features a pass's SPIR-V declares, and the chain of Vulkan feature structs all of them are queried and enabled through
use std::ffi::CStr;

#[allow(unused_imports)]
use crate::{sealed::Sealed, state::Ctx, vk};

/// The error of a function that needs an optional feature the device lacks, by its cargo name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MissingFeature {
    pub feature: &'static str,
}

impl std::fmt::Display for MissingFeature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the device lacks `{}` (lava's cargo feature of that name; see `Ctx::features()`)",
            self.feature
        )
    }
}

impl std::error::Error for MissingFeature {}

/// How lava compiled a cargo feature: [`Asserted`] (`assert-X`) or [`Checked`] (`X` alone).
pub trait Mode: Sealed {
    /// `T` when asserted, `Result<T, MissingFeature>` when checked.
    type Out<T>;
    #[doc(hidden)]
    type Store<T>;
    #[doc(hidden)]
    fn wrap<T>(result: Result<T, MissingFeature>) -> Self::Out<T>;
    #[doc(hidden)]
    fn store<T>(on: bool, load: impl FnOnce() -> T) -> Self::Store<T>;
    #[doc(hidden)]
    fn get<T>(stored: &Self::Store<T>) -> &T;
}

/// `init` refused devices without the feature, so nothing checks it.
pub enum Asserted {}
/// Functions that need the feature check the device.
pub enum Checked {}

impl Sealed for Asserted {}
impl Sealed for Checked {}

impl Mode for Asserted {
    type Out<T> = T;
    type Store<T> = T;
    fn wrap<T>(result: Result<T, MissingFeature>) -> T {
        result.unwrap_or_else(|missing| {
            unreachable!("`{}` is asserted, so `init` checked it", missing.feature)
        })
    }
    fn store<T>(_: bool, load: impl FnOnce() -> T) -> T {
        load()
    }
    fn get<T>(stored: &T) -> &T {
        stored
    }
}

impl Mode for Checked {
    type Out<T> = Result<T, MissingFeature>;
    type Store<T> = Option<T>;
    fn wrap<T>(result: Result<T, MissingFeature>) -> Result<T, MissingFeature> {
        result
    }
    fn store<T>(on: bool, load: impl FnOnce() -> T) -> Option<T> {
        on.then(load)
    }
    fn get<T>(stored: &Option<T>) -> &T {
        stored
            .as_ref()
            .expect("only code past the feature's check reaches its functions")
    }
}

/// The mode of two features together: checked if either is.
pub trait Join<B: Mode>: Mode {
    type Joined: Mode;
}
impl Join<Asserted> for Asserted {
    type Joined = Asserted;
}
impl Join<Checked> for Asserted {
    type Joined = Checked;
}
impl<B: Mode> Join<B> for Checked {
    type Joined = Checked;
}

/// A cargo feature lava was compiled with (a type of [`gate`]), a pair of them, or `()`.
#[diagnostic::on_unimplemented(
    message = "lava was compiled without the cargo feature of `{Self}`",
    note = "each type in `lava::features::gate` names its cargo feature: enable it (or its `assert-` form) on the `lava` dependency"
)]
pub trait Feature: Sealed {
    type Mode: Mode;
    fn check() -> Result<(), MissingFeature>;
}

impl Sealed for () {}
impl Feature for () {
    type Mode = Asserted;
    fn check() -> Result<(), MissingFeature> {
        Ok(())
    }
}

impl<A: Feature, B: Feature> Sealed for (A, B) {}
impl<A: Feature, B: Feature> Feature for (A, B)
where
    A::Mode: Join<B::Mode>,
{
    type Mode = <A::Mode as Join<B::Mode>>::Joined;
    fn check() -> Result<(), MissingFeature> {
        A::check()?;
        B::check()
    }
}

/// `T` when `F` is asserted, `Result<T, MissingFeature>` when lava checks it.
pub type Gated<F, T> = <<F as Feature>::Mode as Mode>::Out<T>;

/// `value()` as a [`Gated`] value: called only if the device has `F`.
pub fn gate<F: Feature, T>(value: impl FnOnce() -> T) -> Gated<F, T> {
    <F::Mode as Mode>::wrap(F::check().map(|()| value()))
}

/// A Vulkan loader that exists only where the device has `F`, read with [`loaded`].
#[allow(dead_code)]
pub(crate) type Loader<F, T> = <<F as Feature>::Mode as Mode>::Store<T>;

#[allow(dead_code)]
pub(crate) fn load<F: Feature, T>(on: bool, load: impl FnOnce() -> T) -> Loader<F, T> {
    <F::Mode as Mode>::store(on, load)
}

#[allow(dead_code)]
pub(crate) fn loaded<F: Feature, T>(stored: &Loader<F, T>) -> &T {
    <F::Mode as Mode>::get(stored)
}

/// Whether the device has a feature, `false` when compiled out.
pub(crate) trait On {
    fn on() -> bool;
}

macro_rules! features {
    ($($(#[doc = $doc:literal])* $name:ident: $cargo:literal / $assert:literal => $marker:ident,)*) => {
        /// The optional device features lava was compiled with, one per cargo feature, plus
        /// `rebar`.
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct Features {
            /// A large device-local heap the host can write (resizable BAR or an integrated GPU).
            pub rebar: bool,
            $($(#[doc = $doc])* #[cfg(feature = $cargo)] pub $name: bool,)*
        }

        impl Features {
            pub const DEFAULT: Features = Features {
                rebar: false,
                $(#[cfg(feature = $cargo)] $name: false,)*
            };

            const NAMES: &[&str] = &["rebar", $(#[cfg(feature = $cargo)] $cargo,)*];

            fn bits(&self) -> impl Iterator<Item = bool> {
                [self.rebar, $(#[cfg(feature = $cargo)] self.$name,)*].into_iter()
            }

            pub(crate) fn from_bits(mut bits: impl Iterator<Item = bool>) -> Self {
                Self {
                    rebar: bits.next().unwrap(),
                    $(#[cfg(feature = $cargo)] $name: bits.next().unwrap(),)*
                }
            }

            pub fn asserted() -> Features {
                Features {
                    rebar: false,
                    $(#[cfg(feature = $cargo)] $name: cfg!(feature = $assert),)*
                }
            }

            /// What a SPIR-V module's `OpCapability`s need; capabilities lava doesn't know are
            /// left to the driver.
            pub(crate) fn of_spirv(spirv: &[u8]) -> Needs {
                let mut needs = Needs {
                    features: Features::default(),
                    compiled_out: Vec::new(),
                };
                let words: Vec<u32> = spirv
                    .chunks_exact(4)
                    .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
                    .collect();
                const OP_CAPABILITY: u32 = 17;
                // After the five header words, each instruction starts with its word count and opcode.
                let mut i = 5;
                while i < words.len() {
                    let (count, opcode) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
                    if opcode == OP_CAPABILITY
                        && let Some(&capability) = words.get(i + 1)
                        && let Some(&(_, cargo)) =
                            SPIRV_CAPABILITIES.iter().find(|(id, _)| *id == capability)
                    {
                        let compiled = match cargo {
                            $(#[cfg(feature = $cargo)] $cargo => {
                                needs.features.$name = true;
                                true
                            })*
                            _ => false,
                        };
                        if !compiled && !needs.compiled_out.contains(&cargo) {
                            needs.compiled_out.push(cargo);
                        }
                    }
                    if count == 0 {
                        break;
                    }
                    i += count;
                }
                needs
            }
        }

        /// One type per optional feature, for [`Gated`] values and the bindings of passes that
        /// need it; it implements [`Feature`] only when lava was compiled with the feature.
        pub mod gate {
            use super::*;
            $(
                #[doc = concat!("lava's `", $cargo, "` cargo feature (`assert-", $cargo, "` asserts it).")]
                pub enum $marker {}
                impl Sealed for $marker {}
                impl On for $marker {
                    #[allow(unreachable_code)]
                    fn on() -> bool {
                        #[cfg(feature = $assert)]
                        return true;
                        #[cfg(all(feature = $cargo, not(feature = $assert)))]
                        return Ctx::features_ref().$name;
                        false
                    }
                }
                #[cfg(feature = $assert)]
                impl Feature for $marker {
                    type Mode = Asserted;
                    fn check() -> Result<(), MissingFeature> {
                        Ok(())
                    }
                }
                #[cfg(all(feature = $cargo, not(feature = $assert)))]
                impl Feature for $marker {
                    type Mode = Checked;
                    fn check() -> Result<(), MissingFeature> {
                        match Ctx::features_ref().$name {
                            true => Ok(()),
                            false => Err(MissingFeature { feature: $cargo }),
                        }
                    }
                }
            )*
        }
    };
}

features! {
    /// `VK_EXT_host_image_copy` on `rebar` memory: `Image::copy_from` and the `*HostCopy` usages.
    host_image_copy: "host-image-copy" / "assert-host-image-copy" => HostImageCopy,
    /// `VK_EXT_mesh_shader` with task shaders: mesh passes.
    mesh_shader: "mesh-shader" / "assert-mesh-shader" => MeshShader,
    /// Task and mesh shader invocations in pipeline statistics.
    mesh_shader_queries: "mesh-shader-queries" / "assert-mesh-shader-queries" => MeshShaderQueries,
    /// `VK_KHR_acceleration_structure`: `Blas`/`Tlas` and their builds.
    acceleration_structure: "acceleration-structure" / "assert-acceleration-structure" => AccelerationStructure,
    /// `VK_KHR_ray_query`: tracing from compute, mesh and fragment passes.
    ray_query: "ray-query" / "assert-ray-query" => RayQuery,
    /// `VK_KHR_ray_tracing_pipeline` with indirect tracing: ray tracing passes.
    ray_tracing_pipeline: "ray-tracing-pipeline" / "assert-ray-tracing-pipeline" => RayTracingPipeline,
    /// Pipeline statistics queries (invocation counts per pass).
    pipeline_statistics: "pipeline-statistics" / "assert-pipeline-statistics" => PipelineStatistics,
    /// Occlusion queries that count samples exactly (`QueryPool::precise_occlusion`).
    precise_occlusion: "precise-occlusion" / "assert-precise-occlusion" => PreciseOcclusion,
    /// `VK_EXT_memory_budget`: per-heap usage and budget in `memory_report`.
    memory_budget: "memory-budget" / "assert-memory-budget" => MemoryBudget,
    /// `VK_KHR_shader_clock` with subgroup clocks (`OpReadClockKHR Subgroup`).
    shader_clock: "shader-clock" / "assert-shader-clock" => ShaderClock,
    /// `VK_KHR_performance_query` with query pools: `query::counters`, `QueryPool::performance`.
    performance_query: "performance-query" / "assert-performance-query" => PerformanceQuery,
    /// `VK_KHR_calibrated_timestamps`: timestamps on the host's clock (`QueryPool::instant`).
    calibrated_timestamps: "calibrated-timestamps" / "assert-calibrated-timestamps" => CalibratedTimestamps,
    /// `VK_EXT_swapchain_maintenance1` with a display: presents signal fences, so what they use
    /// is destroyed once they are done. Asserting it requires it only with a display.
    present_fences: "present-fences" / "assert-present-fences" => PresentFences,
    /// A queue family with compute but no graphics: `Timeline<Compute>`.
    async_compute: "async-compute" / "assert-async-compute" => AsyncCompute,
    /// A queue family with transfer but neither graphics nor compute: `Timeline<Transfer>`.
    async_transfer: "async-transfer" / "assert-async-transfer" => AsyncTransfer,
    /// BC1-7 compressed formats.
    bc_compression: "bc-compression" / "assert-bc-compression" => BcCompression,
    /// Different blends for the color attachments of one draw (`RasterState::blends`).
    independent_blend: "independent-blend" / "assert-independent-blend" => IndependentBlend,
    /// Line and point polygons (`RasterState::polygon_mode`).
    fill_mode_non_solid: "fill-mode-non-solid" / "assert-fill-mode-non-solid" => FillModeNonSolid,
    /// `RasterState::line_width`.
    wide_lines: "wide-lines" / "assert-wide-lines" => WideLines,
    /// `RasterState::depth_bounds`.
    depth_bounds: "depth-bounds" / "assert-depth-bounds" => DepthBounds,
    /// `RasterState::depth_clamp`.
    depth_clamp: "depth-clamp" / "assert-depth-clamp" => DepthClamp,
    /// `VK_EXT_depth_clamp_control`: `RasterState::depth_clamp_range`.
    depth_clamp_control: "depth-clamp-control" / "assert-depth-clamp-control" => DepthClampControl,
    /// `VK_EXT_depth_clip_enable`: `RasterState::depth_clip`.
    depth_clip_enable: "depth-clip-enable" / "assert-depth-clip-enable" => DepthClipEnable,
    /// `VK_EXT_depth_clip_control`: `RasterState::depth_clip_negative_one_to_one`.
    depth_clip_control: "depth-clip-control" / "assert-depth-clip-control" => DepthClipControl,
    /// `RasterState::logic_op`.
    logic_op: "logic-op" / "assert-logic-op" => LogicOp,
    /// `SamplerDesc::max_anisotropy`.
    sampler_anisotropy: "sampler-anisotropy" / "assert-sampler-anisotropy" => SamplerAnisotropy,
    /// Samplers that return the minimum or maximum of the texels they filter
    /// (`SamplerDesc::reduction`).
    sampler_minmax: "sampler-minmax" / "assert-sampler-minmax" => SamplerMinmax,
    /// `VK_EXT_conservative_rasterization`: `RasterState::conservative`.
    conservative_rasterization: "conservative-rasterization" / "assert-conservative-rasterization" => ConservativeRasterization,
    /// `VK_EXT_provoking_vertex` with last-vertex provoking: `RasterState::provoking_vertex`.
    provoking_vertex: "provoking-vertex" / "assert-provoking-vertex" => ProvokingVertex,
    /// `VK_KHR_line_rasterization`'s line modes: `RasterState::rectangular_lines`,
    /// `bresenham_lines` and `smooth_lines`.
    rectangular_lines: "rectangular-lines" / "assert-rectangular-lines" => RectangularLines,
    bresenham_lines: "bresenham-lines" / "assert-bresenham-lines" => BresenhamLines,
    smooth_lines: "smooth-lines" / "assert-smooth-lines" => SmoothLines,
    /// Stippled lines (`RasterState::line_stipple`) in every line mode the device has.
    line_stipple: "line-stipple" / "assert-line-stipple" => LineStipple,
    /// Shader capabilities: 8-, 16- and 64-bit types and 64-bit atomics.
    shader_int8: "shader-int8" / "assert-shader-int8" => ShaderInt8,
    shader_int16: "shader-int16" / "assert-shader-int16" => ShaderInt16,
    shader_float16: "shader-float16" / "assert-shader-float16" => ShaderFloat16,
    shader_float64: "shader-float64" / "assert-shader-float64" => ShaderFloat64,
    shader_int64_atomics: "shader-int64-atomics" / "assert-shader-int64-atomics" => ShaderInt64Atomics,
    /// 8- and 16-bit values in storage and uniform buffers and push constants.
    storage_8bit: "storage-8bit" / "assert-storage-8bit" => Storage8Bit,
    storage_16bit: "storage-16bit" / "assert-storage-16bit" => Storage16Bit,
    /// Extended dynamic state: draws that differ in it share a pipeline; without it the state
    /// is part of the pipeline's key.
    dynamic_blend: "dynamic-blend" / "assert-dynamic-blend" => DynamicBlend,
    dynamic_polygon_mode: "dynamic-polygon-mode" / "assert-dynamic-polygon-mode" => DynamicPolygonMode,
    dynamic_depth_clamp: "dynamic-depth-clamp" / "assert-dynamic-depth-clamp" => DynamicDepthClamp,
    dynamic_depth_clip: "dynamic-depth-clip" / "assert-dynamic-depth-clip" => DynamicDepthClip,
    dynamic_depth_clip_negative_one_to_one: "dynamic-depth-clip-negative-one-to-one" / "assert-dynamic-depth-clip-negative-one-to-one" => DynamicDepthClipNegativeOneToOne,
    dynamic_logic_op: "dynamic-logic-op" / "assert-dynamic-logic-op" => DynamicLogicOp,
    dynamic_conservative_rasterization: "dynamic-conservative-rasterization" / "assert-dynamic-conservative-rasterization" => DynamicConservativeRasterization,
    dynamic_line_mode: "dynamic-line-mode" / "assert-dynamic-line-mode" => DynamicLineMode,
    dynamic_line_stipple: "dynamic-line-stipple" / "assert-dynamic-line-stipple" => DynamicLineStipple,
    dynamic_provoking_vertex: "dynamic-provoking-vertex" / "assert-dynamic-provoking-vertex" => DynamicProvokingVertex,
}

impl Default for Features {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Sets `$features.$field = $value` if lava was compiled with cargo feature `$cargo`.
macro_rules! set {
    ($features:ident: $($cargo:literal => $($field:ident).+ = $value:expr),* $(,)?) => {
        $(#[cfg(feature = $cargo)]
        {
            $features.$($field).+ = $value;
        })*
    };
}

/// The SPIR-V capabilities lava knows, by the cargo feature they need (lava-build-slang has a copy).
#[doc(hidden)]
pub const SPIRV_CAPABILITIES: &[(u32, &str)] = &[
    (9, "shader-float16"),
    (10, "shader-float64"),
    (12, "shader-int64-atomics"),
    (22, "shader-int16"),
    (39, "shader-int8"),
    (4433, "storage-16bit"),
    (4434, "storage-16bit"),
    (4435, "storage-16bit"),
    (4448, "storage-8bit"),
    (4449, "storage-8bit"),
    (4450, "storage-8bit"),
    (4472, "ray-query"),
    (4479, "ray-tracing-pipeline"),
    (5055, "shader-clock"),
    (5283, "mesh-shader"),
];

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Needs {
    pub(crate) features: Features,
    pub(crate) compiled_out: Vec<&'static str>,
}

impl Features {
    pub fn missing_from(&self, available: &Features) -> Vec<&'static str> {
        (self.bits().zip(available.bits()).zip(Self::NAMES))
            .filter(|((wanted, has), _)| *wanted && !*has)
            .map(|(_, name)| *name)
            .collect()
    }

    pub fn union(self, other: Features) -> Features {
        Self::from_bits(self.bits().zip(other.bits()).map(|(a, b)| a || b))
    }

    pub fn without(self, other: Features) -> Features {
        Self::from_bits(self.bits().zip(other.bits()).map(|(a, b)| a && !b))
    }

    /// `self` without the features whose prerequisites it lacks.
    #[allow(unused_mut)]
    pub(crate) fn consistent(mut self) -> Features {
        let rebar = self.rebar;
        let _ = rebar;
        set! { self:
            "host-image-copy" => host_image_copy = self.host_image_copy && rebar,
            "ray-query" => ray_query = self.ray_query && self.acceleration_structure,
            "ray-tracing-pipeline" => ray_tracing_pipeline =
                self.ray_tracing_pipeline && self.acceleration_structure,
            "mesh-shader-queries" => mesh_shader_queries =
                self.mesh_shader_queries && self.mesh_shader && self.pipeline_statistics,
            "depth-clamp-control" => depth_clamp_control =
                self.depth_clamp_control && self.depth_clamp,
            "dynamic-depth-clip" => dynamic_depth_clip =
                self.dynamic_depth_clip && self.depth_clip_enable,
            "dynamic-depth-clip-negative-one-to-one" => dynamic_depth_clip_negative_one_to_one =
                self.dynamic_depth_clip_negative_one_to_one && self.depth_clip_control,
            "dynamic-conservative-rasterization" => dynamic_conservative_rasterization =
                self.dynamic_conservative_rasterization && self.conservative_rasterization,
            "dynamic-provoking-vertex" => dynamic_provoking_vertex =
                self.dynamic_provoking_vertex && self.provoking_vertex,
        }
        self
    }

    /// Whether anything that needs `VK_KHR_line_rasterization` is on.
    pub(crate) fn lines(&self) -> bool {
        let lines: &[bool] = &[
            #[cfg(feature = "rectangular-lines")]
            self.rectangular_lines,
            #[cfg(feature = "bresenham-lines")]
            self.bresenham_lines,
            #[cfg(feature = "smooth-lines")]
            self.smooth_lines,
            #[cfg(feature = "line-stipple")]
            self.line_stipple,
            #[cfg(feature = "dynamic-line-mode")]
            self.dynamic_line_mode,
            #[cfg(feature = "dynamic-line-stipple")]
            self.dynamic_line_stipple,
        ];
        lines.iter().any(|on| *on)
    }

    #[allow(unused_mut)]
    pub(crate) fn extensions(&self) -> Vec<&'static CStr> {
        let mut extensions: Vec<&'static CStr> = Vec::new();
        let mut on = |on: bool, names: &[&'static CStr]| {
            for name in names {
                if on && !extensions.contains(name) {
                    extensions.push(name);
                }
            }
        };
        let eds3 = ash::ext::extended_dynamic_state3::NAME;
        macro_rules! needs {
            ($($cargo:literal => $field:ident: [$($name:expr),*],)*) => {
                $(#[cfg(feature = $cargo)]
                on(self.$field, &[$($name),*]);)*
            };
        }
        needs! {
            "host-image-copy" => host_image_copy: [ash::ext::host_image_copy::NAME],
            "mesh-shader" => mesh_shader: [ash::ext::mesh_shader::NAME],
            "memory-budget" => memory_budget: [ash::ext::memory_budget::NAME],
            "shader-clock" => shader_clock: [ash::khr::shader_clock::NAME],
            "performance-query" => performance_query: [ash::khr::performance_query::NAME],
            "calibrated-timestamps" => calibrated_timestamps: [ash::khr::calibrated_timestamps::NAME],
            "present-fences" => present_fences: [ash::ext::swapchain_maintenance1::NAME],
            "acceleration-structure" => acceleration_structure: [
                ash::khr::deferred_host_operations::NAME,
                ash::khr::acceleration_structure::NAME
            ],
            "ray-query" => ray_query: [ash::khr::ray_query::NAME],
            "ray-tracing-pipeline" => ray_tracing_pipeline: [ash::khr::ray_tracing_pipeline::NAME],
            "depth-clamp-control" => depth_clamp_control: [crate::vk::depth_clamp_control::NAME],
            "depth-clip-enable" => depth_clip_enable: [ash::ext::depth_clip_enable::NAME],
            "depth-clip-control" => depth_clip_control: [ash::ext::depth_clip_control::NAME],
            "conservative-rasterization" => conservative_rasterization: [
                ash::ext::conservative_rasterization::NAME
            ],
            "provoking-vertex" => provoking_vertex: [ash::ext::provoking_vertex::NAME],
            "dynamic-blend" => dynamic_blend: [eds3],
            "dynamic-polygon-mode" => dynamic_polygon_mode: [eds3],
            "dynamic-depth-clamp" => dynamic_depth_clamp: [eds3],
            "dynamic-depth-clip" => dynamic_depth_clip: [eds3],
            "dynamic-depth-clip-negative-one-to-one" => dynamic_depth_clip_negative_one_to_one: [eds3],
            "dynamic-logic-op" => dynamic_logic_op: [
                ash::ext::extended_dynamic_state2::NAME, eds3
            ],
            "dynamic-conservative-rasterization" => dynamic_conservative_rasterization: [eds3],
            "dynamic-line-mode" => dynamic_line_mode: [eds3],
            "dynamic-line-stipple" => dynamic_line_stipple: [eds3],
            "dynamic-provoking-vertex" => dynamic_provoking_vertex: [eds3],
        }
        on(self.lines(), &[ash::khr::line_rasterization::NAME]);
        let _ = eds3;
        extensions
    }
}

pub(crate) const LAVA_EXTENSIONS: [&CStr; 3] = [
    crate::vk::device_address_commands::NAME,
    crate::vk::descriptor_heap::NAME,
    crate::vk::shader_untyped_pointers::NAME,
];

/// What [`FeatureChain::optional`] needs to know of the device besides its feature structs.
#[derive(Clone, Copy)]
pub(crate) struct DeviceFacts {
    pub(crate) rebar: bool,
    #[allow(dead_code)]
    pub(crate) strict_lines: bool,
    #[allow(dead_code)]
    pub(crate) dedicated_compute: bool,
    #[allow(dead_code)]
    pub(crate) dedicated_transfer: bool,
}

/// Every Vulkan feature struct lava queries or enables.
#[derive(Default)]
pub(crate) struct FeatureChain {
    core: vk::PhysicalDeviceFeatures,
    vk11: vk::PhysicalDeviceVulkan11Features<'static>,
    vk12: vk::PhysicalDeviceVulkan12Features<'static>,
    vk13: vk::PhysicalDeviceVulkan13Features<'static>,
    address_commands: vk::PhysicalDeviceDeviceAddressCommandsFeaturesKHR<'static>,
    descriptor_heap: vk::PhysicalDeviceDescriptorHeapFeaturesEXT<'static>,
    untyped_pointers: vk::PhysicalDeviceShaderUntypedPointersFeaturesKHR<'static>,
    dynamic_state2: vk::PhysicalDeviceExtendedDynamicState2FeaturesEXT<'static>,
    dynamic_state3: vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT<'static>,
    lines: vk::PhysicalDeviceLineRasterizationFeaturesKHR<'static>,
    provoking_vertex: vk::PhysicalDeviceProvokingVertexFeaturesEXT<'static>,
    depth_clip_enable: vk::PhysicalDeviceDepthClipEnableFeaturesEXT<'static>,
    depth_clip_control: vk::PhysicalDeviceDepthClipControlFeaturesEXT<'static>,
    depth_clamp_control: vk::PhysicalDeviceDepthClampControlFeaturesEXT<'static>,
    mesh: vk::PhysicalDeviceMeshShaderFeaturesEXT<'static>,
    acceleration_structure: vk::PhysicalDeviceAccelerationStructureFeaturesKHR<'static>,
    ray_query: vk::PhysicalDeviceRayQueryFeaturesKHR<'static>,
    ray_tracing_pipeline: vk::PhysicalDeviceRayTracingPipelineFeaturesKHR<'static>,
    host_image_copy: vk::PhysicalDeviceHostImageCopyFeaturesEXT<'static>,
    clock: vk::PhysicalDeviceShaderClockFeaturesKHR<'static>,
    performance: vk::PhysicalDevicePerformanceQueryFeaturesKHR<'static>,
    swapchain_maintenance1: vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT<'static>,
}

impl FeatureChain {
    pub(crate) fn query(
        instance: &ash::Instance,
        device: vk::PhysicalDevice,
        has: &dyn Fn(&CStr) -> bool,
    ) -> Self {
        let mut chain = Self::default();
        let mut features2 = chain.chain(has);
        unsafe { instance.get_physical_device_features2(device, &mut features2) };
        chain.core = features2.features;
        chain
    }

    /// The structs whose extension `has` names, linked behind one `PhysicalDeviceFeatures2`.
    pub(crate) fn chain<'a>(&'a mut self, has: &dyn Fn(&CStr) -> bool) -> vk::PhysicalDeviceFeatures2<'a> {
        let Self {
            core,
            vk11,
            vk12,
            vk13,
            address_commands,
            descriptor_heap,
            untyped_pointers,
            dynamic_state2,
            dynamic_state3,
            lines,
            provoking_vertex,
            depth_clip_enable,
            depth_clip_control,
            depth_clamp_control,
            mesh,
            acceleration_structure,
            ray_query,
            ray_tracing_pipeline,
            host_image_copy,
            clock,
            performance,
            swapchain_maintenance1,
        } = self;
        // ash 0.38's `push_next` links the pushed struct's whole chain, so unlink a queried one.
        macro_rules! unlink {
            ($($field:ident),*) => {
                $($field.p_next = core::ptr::null_mut();)*
            };
        }
        unlink!(
            vk11,
            vk12,
            vk13,
            address_commands,
            descriptor_heap,
            untyped_pointers,
            dynamic_state2,
            dynamic_state3,
            lines,
            provoking_vertex,
            depth_clip_enable,
            depth_clip_control,
            depth_clamp_control,
            mesh,
            acceleration_structure,
            ray_query,
            ray_tracing_pipeline,
            host_image_copy,
            clock,
            performance,
            swapchain_maintenance1
        );
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .features(*core)
            .push_next(vk11)
            .push_next(vk12)
            .push_next(vk13);
        macro_rules! push_if {
            ($($extension:expr => $field:ident),* $(,)?) => {
                $(if has($extension) {
                    features = features.push_next($field);
                })*
            };
        }
        push_if! {
            crate::vk::device_address_commands::NAME => address_commands,
            crate::vk::descriptor_heap::NAME => descriptor_heap,
            crate::vk::shader_untyped_pointers::NAME => untyped_pointers,
            ash::ext::extended_dynamic_state2::NAME => dynamic_state2,
            ash::ext::extended_dynamic_state3::NAME => dynamic_state3,
            ash::khr::line_rasterization::NAME => lines,
            ash::ext::provoking_vertex::NAME => provoking_vertex,
            ash::ext::depth_clip_enable::NAME => depth_clip_enable,
            ash::ext::depth_clip_control::NAME => depth_clip_control,
            crate::vk::depth_clamp_control::NAME => depth_clamp_control,
            ash::ext::mesh_shader::NAME => mesh,
            ash::khr::acceleration_structure::NAME => acceleration_structure,
            ash::khr::ray_query::NAME => ray_query,
            ash::khr::ray_tracing_pipeline::NAME => ray_tracing_pipeline,
            ash::ext::host_image_copy::NAME => host_image_copy,
            ash::khr::shader_clock::NAME => clock,
            ash::khr::performance_query::NAME => performance,
            ash::ext::swapchain_maintenance1::NAME => swapchain_maintenance1,
        }
        features
    }

    /// What lava needs that the device lacks, by name.
    pub(crate) fn lava_missing(&self, has: &dyn Fn(&CStr) -> bool) -> Vec<String> {
        let extensions = LAVA_EXTENSIONS.iter().filter(|name| !has(name));
        let extensions = extensions.map(|name| name.to_string_lossy().into_owned());
        let (c, v11, v12, v13) = (&self.core, &self.vk11, &self.vk12, &self.vk13);
        let bits = [
            ("shaderInt64", c.shader_int64),
            ("fragmentStoresAndAtomics", c.fragment_stores_and_atomics),
            ("vertexPipelineStoresAndAtomics", c.vertex_pipeline_stores_and_atomics),
            ("shaderDrawParameters", v11.shader_draw_parameters),
            ("bufferDeviceAddress", v12.buffer_device_address),
            ("timelineSemaphore", v12.timeline_semaphore),
            ("drawIndirectCount", v12.draw_indirect_count),
            ("scalarBlockLayout", v12.scalar_block_layout),
            ("vulkanMemoryModel", v12.vulkan_memory_model),
            ("vulkanMemoryModelDeviceScope", v12.vulkan_memory_model_device_scope),
            ("hostQueryReset", v12.host_query_reset),
            ("dynamicRendering", v13.dynamic_rendering),
            ("synchronization2", v13.synchronization2),
            ("maintenance4", v13.maintenance4),
            ("shaderDemoteToHelperInvocation", v13.shader_demote_to_helper_invocation),
            ("deviceAddressCommands", self.address_commands.device_address_commands),
            ("descriptorHeap", self.descriptor_heap.descriptor_heap),
            ("shaderUntypedPointers", self.untyped_pointers.shader_untyped_pointers),
        ];
        let bits = bits.into_iter().filter(|(_, bit)| *bit != vk::TRUE);
        extensions.chain(bits.map(|(name, _)| name.to_owned())).collect()
    }

    #[allow(unused_variables)]
    pub(crate) fn optional(&self, has: &dyn Fn(&CStr) -> bool, device: DeviceFacts) -> Features {
        let t = |bit: vk::Bool32| bit == vk::TRUE;
        let (c, v11, v12) = (&self.core, &self.vk11, &self.vk12);
        let (d2, d3, lines) = (&self.dynamic_state2, &self.dynamic_state3, &self.lines);
        let line_ext = has(ash::khr::line_rasterization::NAME);
        let acceleration_structure = has(ash::khr::acceleration_structure::NAME)
            && has(ash::khr::deferred_host_operations::NAME)
            && t(self.acceleration_structure.acceleration_structure);
        let eds3 = |bit| has(ash::ext::extended_dynamic_state3::NAME) && t(bit);
        #[allow(unused_mut)]
        let mut f = Features {
            rebar: device.rebar,
            ..Features::DEFAULT
        };
        set! { f:
            "host-image-copy" => host_image_copy = t(self.host_image_copy.host_image_copy),
            "mesh-shader" => mesh_shader = t(self.mesh.mesh_shader) && t(self.mesh.task_shader),
            "mesh-shader-queries" => mesh_shader_queries = t(self.mesh.mesh_shader_queries),
            "acceleration-structure" => acceleration_structure = acceleration_structure,
            "ray-query" => ray_query = t(self.ray_query.ray_query),
            "ray-tracing-pipeline" => ray_tracing_pipeline =
                t(self.ray_tracing_pipeline.ray_tracing_pipeline)
                    && t(self.ray_tracing_pipeline.ray_tracing_pipeline_trace_rays_indirect),
            "pipeline-statistics" => pipeline_statistics = t(c.pipeline_statistics_query),
            "precise-occlusion" => precise_occlusion = t(c.occlusion_query_precise),
            "memory-budget" => memory_budget = has(ash::ext::memory_budget::NAME),
            "shader-clock" => shader_clock = t(self.clock.shader_subgroup_clock),
            "performance-query" => performance_query =
                t(self.performance.performance_counter_query_pools),
            "calibrated-timestamps" => calibrated_timestamps =
                has(ash::khr::calibrated_timestamps::NAME),
            "present-fences" => present_fences =
                t(self.swapchain_maintenance1.swapchain_maintenance1),
            "async-compute" => async_compute = device.dedicated_compute,
            "async-transfer" => async_transfer = device.dedicated_transfer,
            "bc-compression" => bc_compression = t(c.texture_compression_bc),
            "independent-blend" => independent_blend = t(c.independent_blend),
            "fill-mode-non-solid" => fill_mode_non_solid = t(c.fill_mode_non_solid),
            "wide-lines" => wide_lines = t(c.wide_lines),
            "depth-bounds" => depth_bounds = t(c.depth_bounds),
            "depth-clamp" => depth_clamp = t(c.depth_clamp),
            "depth-clamp-control" => depth_clamp_control =
                t(self.depth_clamp_control.depth_clamp_control),
            "depth-clip-enable" => depth_clip_enable = t(self.depth_clip_enable.depth_clip_enable),
            "depth-clip-control" => depth_clip_control =
                t(self.depth_clip_control.depth_clip_control),
            "logic-op" => logic_op = t(c.logic_op),
            "sampler-anisotropy" => sampler_anisotropy = t(c.sampler_anisotropy),
            "sampler-minmax" => sampler_minmax = t(v12.sampler_filter_minmax),
            "conservative-rasterization" => conservative_rasterization =
                has(ash::ext::conservative_rasterization::NAME),
            "provoking-vertex" => provoking_vertex = t(self.provoking_vertex.provoking_vertex_last),
            "rectangular-lines" => rectangular_lines = line_ext && t(lines.rectangular_lines),
            "bresenham-lines" => bresenham_lines = line_ext && t(lines.bresenham_lines),
            "smooth-lines" => smooth_lines = line_ext && t(lines.smooth_lines),
            "shader-int8" => shader_int8 = t(v12.shader_int8),
            "shader-int16" => shader_int16 = t(c.shader_int16),
            "shader-float16" => shader_float16 = t(v12.shader_float16),
            "shader-float64" => shader_float64 = t(c.shader_float64),
            "shader-int64-atomics" => shader_int64_atomics = t(v12.shader_buffer_int64_atomics),
            "storage-8bit" => storage_8bit = t(v12.storage_buffer8_bit_access)
                && t(v12.uniform_and_storage_buffer8_bit_access)
                && t(v12.storage_push_constant8),
            "storage-16bit" => storage_16bit = t(v11.storage_buffer16_bit_access)
                && t(v11.uniform_and_storage_buffer16_bit_access)
                && t(v11.storage_push_constant16),
            "dynamic-blend" => dynamic_blend = eds3(d3.extended_dynamic_state3_color_blend_enable)
                && t(d3.extended_dynamic_state3_color_blend_equation)
                && t(d3.extended_dynamic_state3_color_write_mask),
            "dynamic-polygon-mode" => dynamic_polygon_mode =
                eds3(d3.extended_dynamic_state3_polygon_mode),
            "dynamic-depth-clamp" => dynamic_depth_clamp =
                eds3(d3.extended_dynamic_state3_depth_clamp_enable),
            "dynamic-depth-clip" => dynamic_depth_clip =
                eds3(d3.extended_dynamic_state3_depth_clip_enable),
            "dynamic-depth-clip-negative-one-to-one" => dynamic_depth_clip_negative_one_to_one =
                eds3(d3.extended_dynamic_state3_depth_clip_negative_one_to_one),
            "dynamic-logic-op" => dynamic_logic_op =
                eds3(d3.extended_dynamic_state3_logic_op_enable)
                    && has(ash::ext::extended_dynamic_state2::NAME)
                    && t(d2.extended_dynamic_state2_logic_op),
            "dynamic-conservative-rasterization" => dynamic_conservative_rasterization =
                eds3(d3.extended_dynamic_state3_conservative_rasterization_mode),
            "dynamic-line-mode" => dynamic_line_mode =
                line_ext && eds3(d3.extended_dynamic_state3_line_rasterization_mode),
            "dynamic-line-stipple" => dynamic_line_stipple =
                line_ext && eds3(d3.extended_dynamic_state3_line_stipple_enable),
            "dynamic-provoking-vertex" => dynamic_provoking_vertex =
                eds3(d3.extended_dynamic_state3_provoking_vertex_mode),
        }
        let stippled = |mode: vk::Bool32, stippled: vk::Bool32| !t(mode) || t(stippled);
        set! { f:
            "line-stipple" => line_stipple = line_ext
                && device.strict_lines
                && t(lines.stippled_rectangular_lines)
                && stippled(lines.bresenham_lines, lines.stippled_bresenham_lines)
                && stippled(lines.smooth_lines, lines.stippled_smooth_lines),
        }
        f.consistent()
    }

    #[allow(unused_variables)]
    pub(crate) fn enabling(features: &Features) -> Self {
        let on = |b: bool| b as vk::Bool32;
        let f = features;
        let mut chain = Self::default();
        chain.core = vk::PhysicalDeviceFeatures::default()
            .shader_int64(true)
            .fragment_stores_and_atomics(true)
            .vertex_pipeline_stores_and_atomics(true);
        chain.vk11 = vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(true);
        chain.vk12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .timeline_semaphore(true)
            .draw_indirect_count(true)
            .scalar_block_layout(true)
            .vulkan_memory_model(true)
            .vulkan_memory_model_device_scope(true)
            .host_query_reset(true);
        chain.vk13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true)
            .maintenance4(true)
            .shader_demote_to_helper_invocation(true);
        chain.address_commands.device_address_commands = vk::TRUE;
        chain.descriptor_heap.descriptor_heap = vk::TRUE;
        chain.untyped_pointers.shader_untyped_pointers = vk::TRUE;
        let FeatureChain {
            core: c,
            vk11: v11,
            vk12: v12,
            dynamic_state2: d2,
            dynamic_state3: d3,
            lines,
            ..
        } = &mut chain;
        set! { c:
            "bc-compression" => texture_compression_bc = on(f.bc_compression),
            "independent-blend" => independent_blend = on(f.independent_blend),
            "fill-mode-non-solid" => fill_mode_non_solid = on(f.fill_mode_non_solid),
            "wide-lines" => wide_lines = on(f.wide_lines),
            "depth-bounds" => depth_bounds = on(f.depth_bounds),
            "depth-clamp" => depth_clamp = on(f.depth_clamp),
            "logic-op" => logic_op = on(f.logic_op),
            "sampler-anisotropy" => sampler_anisotropy = on(f.sampler_anisotropy),
            "shader-int16" => shader_int16 = on(f.shader_int16),
            "shader-float64" => shader_float64 = on(f.shader_float64),
            "pipeline-statistics" => pipeline_statistics_query = on(f.pipeline_statistics),
            "precise-occlusion" => occlusion_query_precise = on(f.precise_occlusion),
        }
        set! { v11:
            "storage-16bit" => storage_buffer16_bit_access = on(f.storage_16bit),
            "storage-16bit" => uniform_and_storage_buffer16_bit_access = on(f.storage_16bit),
            "storage-16bit" => storage_push_constant16 = on(f.storage_16bit),
        }
        set! { v12:
            "sampler-minmax" => sampler_filter_minmax = on(f.sampler_minmax),
            "shader-int8" => shader_int8 = on(f.shader_int8),
            "shader-float16" => shader_float16 = on(f.shader_float16),
            "shader-int64-atomics" => shader_buffer_int64_atomics = on(f.shader_int64_atomics),
            "storage-8bit" => storage_buffer8_bit_access = on(f.storage_8bit),
            "storage-8bit" => uniform_and_storage_buffer8_bit_access = on(f.storage_8bit),
            "storage-8bit" => storage_push_constant8 = on(f.storage_8bit),
        }
        set! { d3:
            "dynamic-blend" => extended_dynamic_state3_color_blend_enable = on(f.dynamic_blend),
            "dynamic-blend" => extended_dynamic_state3_color_blend_equation = on(f.dynamic_blend),
            "dynamic-blend" => extended_dynamic_state3_color_write_mask = on(f.dynamic_blend),
            "dynamic-polygon-mode" => extended_dynamic_state3_polygon_mode =
                on(f.dynamic_polygon_mode),
            "dynamic-depth-clamp" => extended_dynamic_state3_depth_clamp_enable =
                on(f.dynamic_depth_clamp),
            "dynamic-depth-clip" => extended_dynamic_state3_depth_clip_enable =
                on(f.dynamic_depth_clip),
            "dynamic-depth-clip-negative-one-to-one" =>
                extended_dynamic_state3_depth_clip_negative_one_to_one =
                    on(f.dynamic_depth_clip_negative_one_to_one),
            "dynamic-logic-op" => extended_dynamic_state3_logic_op_enable = on(f.dynamic_logic_op),
            "dynamic-conservative-rasterization" =>
                extended_dynamic_state3_conservative_rasterization_mode =
                    on(f.dynamic_conservative_rasterization),
            "dynamic-line-mode" => extended_dynamic_state3_line_rasterization_mode =
                on(f.dynamic_line_mode),
            "dynamic-line-stipple" => extended_dynamic_state3_line_stipple_enable =
                on(f.dynamic_line_stipple),
            "dynamic-provoking-vertex" => extended_dynamic_state3_provoking_vertex_mode =
                on(f.dynamic_provoking_vertex),
        }
        set! { d2:
            "dynamic-logic-op" => extended_dynamic_state2_logic_op = on(f.dynamic_logic_op),
        }
        set! { lines:
            "rectangular-lines" => rectangular_lines = on(f.rectangular_lines),
            "bresenham-lines" => bresenham_lines = on(f.bresenham_lines),
            "smooth-lines" => smooth_lines = on(f.smooth_lines),
            "line-stipple" => stippled_rectangular_lines = on(f.line_stipple),
        }
        #[cfg(feature = "line-stipple")]
        {
            lines.stippled_bresenham_lines = on(f.line_stipple && lines.bresenham_lines == vk::TRUE);
            lines.stippled_smooth_lines = on(f.line_stipple && lines.smooth_lines == vk::TRUE);
        }
        let _ = (c, v11, v12, d2, d3, lines);
        set! { chain:
            "provoking-vertex" => provoking_vertex.provoking_vertex_last = on(f.provoking_vertex),
            "depth-clip-enable" => depth_clip_enable.depth_clip_enable = on(f.depth_clip_enable),
            "depth-clip-control" => depth_clip_control.depth_clip_control =
                on(f.depth_clip_control),
            "depth-clamp-control" => depth_clamp_control.depth_clamp_control =
                on(f.depth_clamp_control),
            "mesh-shader" => mesh.task_shader = on(f.mesh_shader),
            "mesh-shader" => mesh.mesh_shader = on(f.mesh_shader),
            "mesh-shader-queries" => mesh.mesh_shader_queries = on(f.mesh_shader_queries),
            "acceleration-structure" => acceleration_structure.acceleration_structure =
                on(f.acceleration_structure),
            "ray-query" => ray_query.ray_query = on(f.ray_query),
            "ray-tracing-pipeline" => ray_tracing_pipeline.ray_tracing_pipeline =
                on(f.ray_tracing_pipeline),
            "ray-tracing-pipeline" =>
                ray_tracing_pipeline.ray_tracing_pipeline_trace_rays_indirect =
                    on(f.ray_tracing_pipeline),
            "host-image-copy" => host_image_copy.host_image_copy = on(f.host_image_copy),
            "shader-clock" => clock.shader_subgroup_clock = on(f.shader_clock),
            "performance-query" => performance.performance_counter_query_pools =
                on(f.performance_query),
            "present-fences" => swapchain_maintenance1.swapchain_maintenance1 =
                on(f.present_fences),
        }
        chain
    }
}

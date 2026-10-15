//! Device features: what lava itself needs of every device, the optional `Features` it enables wherever the device has them, the features a pass's SPIR-V declares, and the chain of Vulkan feature structs all of them are queried and enabled through
use std::ffi::CStr;

use ash::vk::{self, TaggedStructure};

macro_rules! features {
    ($($(#[doc = $doc:literal])* $name:ident,)*) => {
        /// Optional device features. lava enables each one the device has
        /// (`Ctx::features()`); `Config::required` makes a missing one an init error, and a pass
        /// whose SPIR-V needs a missing one gets no pipeline and panics when recorded.
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct Features {
            $($(#[doc = $doc])* pub $name: bool,)*
        }

        impl Features {
            /// No feature, for `..Features::DEFAULT` in constants.
            pub const DEFAULT: Features = Features { $($name: false,)* };

            const NAMES: &[&str] = &[$(stringify!($name)),*];

            fn bits(&self) -> impl Iterator<Item = bool> {
                [$(self.$name),*].into_iter()
            }

            fn from_bits(mut bits: impl Iterator<Item = bool>) -> Self {
                Self { $($name: bits.next().unwrap(),)* }
            }
        }
    };
}

features! {
    /// A large device-local heap the host can write (resizable BAR or an integrated GPU); not
    /// a Vulkan feature. Buffers live there, mapped.
    rebar,
    /// `VK_EXT_host_image_copy` on `rebar` memory: `Image::copy_from`.
    host_image_copy,
    /// `VK_EXT_mesh_shader` with task shaders: mesh passes.
    mesh,
    /// Task and mesh shader invocations in pipeline statistics.
    mesh_queries,
    /// `VK_KHR_acceleration_structure`: `Blas`/`Tlas` and their builds. `ray_query` and
    /// `ray_tracing_pipeline` need it.
    acceleration_structure,
    /// `VK_KHR_ray_query`: tracing from compute, mesh and fragment passes.
    ray_query,
    /// `VK_KHR_ray_tracing_pipeline` with indirect tracing: ray tracing passes.
    ray_tracing_pipeline,
    /// Pipeline statistics queries (invocation counts per pass).
    pipeline_statistics,
    /// Occlusion queries that count samples exactly (`QueryPool::occlusion`).
    precise_occlusion,
    /// `VK_EXT_memory_budget`: per-heap usage and budget in `memory_report`.
    memory_budget,
    /// `VK_KHR_shader_clock` with subgroup clocks (`OpReadClockKHR Subgroup`).
    shader_clock,
    /// `VK_KHR_performance_query` with query pools: `QueryPool::performance`.
    performance_query,
    /// `VK_KHR_calibrated_timestamps`, which puts timestamps on the host's clock
    /// (`QueryPool::instant`).
    calibrated_timestamps,
    /// BC1-7 compressed formats.
    bc_compression,
    /// Different `Blend`s for the color attachments of one draw.
    independent_blend,
    /// Line and point `PolygonMode`s.
    fill_mode_non_solid,
    /// `RasterState::line_width` other than 1.
    wide_lines,
    /// `RasterState::depth_bounds`.
    depth_bounds,
    /// `RasterState::depth_clamp`.
    depth_clamp,
    /// `RasterState::logic_op`.
    logic_op,
    /// `SamplerDesc::max_anisotropy`.
    sampler_anisotropy,
    /// The line rasterization modes of `RasterState::line`, and their stippled forms.
    rectangular_lines,
    bresenham_lines,
    smooth_lines,
    stippled_rectangular_lines,
    stippled_bresenham_lines,
    stippled_smooth_lines,
    /// Shader capabilities: 8-, 16- and 64-bit types, 64-bit atomics, and 8- and 16-bit
    /// values in storage buffers and push constants.
    int8,
    int16,
    float16,
    float64,
    int64_atomics,
    storage_8bit,
    storage_16bit,
    push_constant_8bit,
    push_constant_16bit,
}

impl Default for Features {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl Features {
    /// The names of the features `self` has that `available` lacks.
    pub fn missing_from(&self, available: &Features) -> Vec<&'static str> {
        (self.bits().zip(available.bits()).zip(Self::NAMES))
            .filter(|((wanted, has), _)| *wanted && !*has)
            .map(|(_, name)| *name)
            .collect()
    }

    /// The features either has.
    pub fn union(self, other: Features) -> Features {
        Self::from_bits(self.bits().zip(other.bits()).map(|(a, b)| a || b))
    }

    /// The device extensions these features need.
    pub(crate) fn extensions(&self) -> Vec<&'static CStr> {
        [
            (self.host_image_copy, ash::ext::host_image_copy::NAME),
            (self.mesh, ash::ext::mesh_shader::NAME),
            (self.memory_budget, ash::ext::memory_budget::NAME),
            (self.shader_clock, ash::khr::shader_clock::NAME),
            (self.performance_query, ash::khr::performance_query::NAME),
            (self.calibrated_timestamps, ash::khr::calibrated_timestamps::NAME),
            (self.acceleration_structure, ash::khr::deferred_host_operations::NAME),
            (self.acceleration_structure, ash::khr::acceleration_structure::NAME),
            (self.ray_query, ash::khr::ray_query::NAME),
            (self.ray_tracing_pipeline, ash::khr::ray_tracing_pipeline::NAME),
        ]
        .into_iter()
        .filter_map(|(on, name)| on.then_some(name))
        .collect()
    }

    /// The features a SPIR-V module's `OpCapability`s need. Capabilities lava doesn't know
    /// are left to the driver.
    pub(crate) fn of_spirv(spirv: &[u8]) -> Features {
        let mut features = Features::default();
        for capability in capabilities(spirv) {
            match capability {
                9 => features.float16 = true,
                10 => features.float64 = true,
                12 => features.int64_atomics = true,
                22 => features.int16 = true,
                39 => features.int8 = true,
                4433 | 4434 => features.storage_16bit = true,
                4435 => features.push_constant_16bit = true,
                4448 | 4449 => features.storage_8bit = true,
                4450 => features.push_constant_8bit = true,
                4472 => features.ray_query = true,
                4479 => features.ray_tracing_pipeline = true,
                5055 => features.shader_clock = true,
                5283 => features.mesh = true,
                _ => {}
            }
        }
        features
    }
}

/// The operands of a SPIR-V module's `OpCapability` instructions.
fn capabilities(spirv: &[u8]) -> impl Iterator<Item = u32> {
    const OP_CAPABILITY: u32 = 17;
    let words: Vec<u32> = spirv
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
        .collect();
    let mut found = Vec::new();
    // After the five header words, each instruction starts with its word count and opcode.
    let mut i = 5;
    while i < words.len() {
        let (count, opcode) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        if opcode == OP_CAPABILITY
            && let Some(&capability) = words.get(i + 1)
        {
            found.push(capability);
        }
        if count == 0 {
            break;
        }
        i += count;
    }
    found.into_iter()
}

/// The device extensions lava needs on every device: buffer address commands and the dynamic
/// state of renderings.
pub(crate) const LAVA_EXTENSIONS: [&CStr; 11] = [
    ash::khr::device_address_commands::NAME,
    ash::ext::extended_dynamic_state2::NAME,
    ash::ext::extended_dynamic_state3::NAME,
    ash::ext::color_write_enable::NAME,
    ash::khr::line_rasterization::NAME,
    ash::ext::provoking_vertex::NAME,
    ash::ext::conservative_rasterization::NAME,
    ash::ext::depth_clip_enable::NAME,
    ash::ext::depth_clip_control::NAME,
    ash::ext::depth_clamp_control::NAME,
    ash::ext::sample_locations::NAME,
];

/// Every Vulkan feature struct lava queries or enables. [`Self::chain`] links the ones whose
/// extension is in use.
#[derive(Default)]
pub(crate) struct FeatureChain {
    core: vk::PhysicalDeviceFeatures,
    vk11: vk::PhysicalDeviceVulkan11Features<'static>,
    vk12: vk::PhysicalDeviceVulkan12Features<'static>,
    vk13: vk::PhysicalDeviceVulkan13Features<'static>,
    address_commands: vk::PhysicalDeviceDeviceAddressCommandsFeaturesKHR<'static>,
    dynamic_state2: vk::PhysicalDeviceExtendedDynamicState2FeaturesEXT<'static>,
    dynamic_state3: vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT<'static>,
    color_write: vk::PhysicalDeviceColorWriteEnableFeaturesEXT<'static>,
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
}

impl FeatureChain {
    /// The features of `device` whose extensions it has.
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
            dynamic_state2,
            dynamic_state3,
            color_write,
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
        } = self;
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .features(*core)
            .push(vk11)
            .push(vk12)
            .push(vk13);
        macro_rules! push_if {
            ($($extension:expr => $field:ident),* $(,)?) => {
                $(if has($extension) {
                    features = features.push($field);
                })*
            };
        }
        push_if! {
            ash::khr::device_address_commands::NAME => address_commands,
            ash::ext::extended_dynamic_state2::NAME => dynamic_state2,
            ash::ext::extended_dynamic_state3::NAME => dynamic_state3,
            ash::ext::color_write_enable::NAME => color_write,
            ash::khr::line_rasterization::NAME => lines,
            ash::ext::provoking_vertex::NAME => provoking_vertex,
            ash::ext::depth_clip_enable::NAME => depth_clip_enable,
            ash::ext::depth_clip_control::NAME => depth_clip_control,
            ash::ext::depth_clamp_control::NAME => depth_clamp_control,
            ash::ext::mesh_shader::NAME => mesh,
            ash::khr::acceleration_structure::NAME => acceleration_structure,
            ash::khr::ray_query::NAME => ray_query,
            ash::khr::ray_tracing_pipeline::NAME => ray_tracing_pipeline,
            ash::ext::host_image_copy::NAME => host_image_copy,
            ash::khr::shader_clock::NAME => clock,
            ash::khr::performance_query::NAME => performance,
        }
        features
    }

    /// What lava needs of every device, by name: the bindless descriptor indexing set, buffer
    /// addresses and address commands, timeline semaphores, host query resets, the memory
    /// model, sync2, dynamic rendering and the dynamic state of renderings.
    fn lava_bits(&self) -> [(&'static str, vk::Bool32); 49] {
        let (c, v11, v12, v13) = (&self.core, &self.vk11, &self.vk12, &self.vk13);
        let (d2, d3) = (&self.dynamic_state2, &self.dynamic_state3);
        [
            ("shaderInt64", c.shader_int64),
            ("fragmentStoresAndAtomics", c.fragment_stores_and_atomics),
            ("vertexPipelineStoresAndAtomics", c.vertex_pipeline_stores_and_atomics),
            ("shaderDrawParameters", v11.shader_draw_parameters),
            ("descriptorIndexing", v12.descriptor_indexing),
            ("runtimeDescriptorArray", v12.runtime_descriptor_array),
            ("shaderSampledImageArrayNonUniformIndexing", v12.shader_sampled_image_array_non_uniform_indexing),
            ("shaderStorageImageArrayNonUniformIndexing", v12.shader_storage_image_array_non_uniform_indexing),
            ("descriptorBindingPartiallyBound", v12.descriptor_binding_partially_bound),
            ("descriptorBindingVariableDescriptorCount", v12.descriptor_binding_variable_descriptor_count),
            ("descriptorBindingSampledImageUpdateAfterBind", v12.descriptor_binding_sampled_image_update_after_bind),
            ("descriptorBindingStorageImageUpdateAfterBind", v12.descriptor_binding_storage_image_update_after_bind),
            ("descriptorBindingUpdateUnusedWhilePending", v12.descriptor_binding_update_unused_while_pending),
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
            ("extendedDynamicState2LogicOp", d2.extended_dynamic_state2_logic_op),
            ("extendedDynamicState3PolygonMode", d3.extended_dynamic_state3_polygon_mode),
            ("extendedDynamicState3DepthClampEnable", d3.extended_dynamic_state3_depth_clamp_enable),
            ("extendedDynamicState3DepthClipEnable", d3.extended_dynamic_state3_depth_clip_enable),
            ("extendedDynamicState3DepthClipNegativeOneToOne", d3.extended_dynamic_state3_depth_clip_negative_one_to_one),
            ("extendedDynamicState3RasterizationSamples", d3.extended_dynamic_state3_rasterization_samples),
            ("extendedDynamicState3SampleMask", d3.extended_dynamic_state3_sample_mask),
            ("extendedDynamicState3AlphaToCoverageEnable", d3.extended_dynamic_state3_alpha_to_coverage_enable),
            ("extendedDynamicState3AlphaToOneEnable", d3.extended_dynamic_state3_alpha_to_one_enable),
            ("extendedDynamicState3LogicOpEnable", d3.extended_dynamic_state3_logic_op_enable),
            ("extendedDynamicState3ColorBlendEnable", d3.extended_dynamic_state3_color_blend_enable),
            ("extendedDynamicState3ColorBlendEquation", d3.extended_dynamic_state3_color_blend_equation),
            ("extendedDynamicState3ColorWriteMask", d3.extended_dynamic_state3_color_write_mask),
            ("extendedDynamicState3ConservativeRasterizationMode", d3.extended_dynamic_state3_conservative_rasterization_mode),
            ("extendedDynamicState3LineRasterizationMode", d3.extended_dynamic_state3_line_rasterization_mode),
            ("extendedDynamicState3LineStippleEnable", d3.extended_dynamic_state3_line_stipple_enable),
            ("extendedDynamicState3ProvokingVertexMode", d3.extended_dynamic_state3_provoking_vertex_mode),
            ("extendedDynamicState3SampleLocationsEnable", d3.extended_dynamic_state3_sample_locations_enable),
            ("colorWriteEnable", self.color_write.color_write_enable),
            ("provokingVertexLast", self.provoking_vertex.provoking_vertex_last),
            ("depthClipEnable", self.depth_clip_enable.depth_clip_enable),
            ("depthClipControl", self.depth_clip_control.depth_clip_control),
            ("depthClampControl", self.depth_clamp_control.depth_clamp_control),
            // The one line mode every line rasterization device has.
            ("rectangularLines", self.lines.rectangular_lines),
        ]
    }

    /// What lava needs that the device lacks, by name.
    pub(crate) fn lava_missing(&self, has: &dyn Fn(&CStr) -> bool) -> Vec<String> {
        let extensions = LAVA_EXTENSIONS.iter().filter(|name| !has(name));
        let extensions = extensions.map(|name| name.to_string_lossy().into_owned());
        let bits = self.lava_bits().into_iter().filter(|(_, bit)| *bit != vk::TRUE);
        extensions.chain(bits.map(|(name, _)| name.to_owned())).collect()
    }

    /// The optional features the device has (`rebar` is the caller's).
    pub(crate) fn optional(&self, has: &dyn Fn(&CStr) -> bool, rebar: bool) -> Features {
        let t = |bit: vk::Bool32| bit == vk::TRUE;
        let (c, v11, v12) = (&self.core, &self.vk11, &self.vk12);
        let acceleration_structure = has(ash::khr::acceleration_structure::NAME)
            && has(ash::khr::deferred_host_operations::NAME)
            && t(self.acceleration_structure.acceleration_structure);
        Features {
            rebar,
            host_image_copy: rebar && t(self.host_image_copy.host_image_copy),
            mesh: t(self.mesh.mesh_shader) && t(self.mesh.task_shader),
            mesh_queries: t(self.mesh.mesh_shader_queries),
            acceleration_structure,
            ray_query: acceleration_structure && t(self.ray_query.ray_query),
            ray_tracing_pipeline: acceleration_structure
                && t(self.ray_tracing_pipeline.ray_tracing_pipeline)
                && t(self.ray_tracing_pipeline.ray_tracing_pipeline_trace_rays_indirect),
            pipeline_statistics: t(c.pipeline_statistics_query),
            precise_occlusion: t(c.occlusion_query_precise),
            memory_budget: has(ash::ext::memory_budget::NAME),
            shader_clock: t(self.clock.shader_subgroup_clock),
            performance_query: t(self.performance.performance_counter_query_pools),
            calibrated_timestamps: has(ash::khr::calibrated_timestamps::NAME),
            bc_compression: t(c.texture_compression_bc),
            independent_blend: t(c.independent_blend),
            fill_mode_non_solid: t(c.fill_mode_non_solid),
            wide_lines: t(c.wide_lines),
            depth_bounds: t(c.depth_bounds),
            depth_clamp: t(c.depth_clamp),
            logic_op: t(c.logic_op),
            sampler_anisotropy: t(c.sampler_anisotropy),
            rectangular_lines: t(self.lines.rectangular_lines),
            bresenham_lines: t(self.lines.bresenham_lines),
            smooth_lines: t(self.lines.smooth_lines),
            stippled_rectangular_lines: t(self.lines.stippled_rectangular_lines),
            stippled_bresenham_lines: t(self.lines.stippled_bresenham_lines),
            stippled_smooth_lines: t(self.lines.stippled_smooth_lines),
            int8: t(v12.shader_int8),
            int16: t(c.shader_int16),
            float16: t(v12.shader_float16),
            float64: t(c.shader_float64),
            int64_atomics: t(v12.shader_buffer_int64_atomics),
            storage_8bit: t(v12.storage_buffer8_bit_access)
                && t(v12.uniform_and_storage_buffer8_bit_access),
            storage_16bit: t(v11.storage_buffer16_bit_access)
                && t(v11.uniform_and_storage_buffer16_bit_access),
            push_constant_8bit: t(v12.storage_push_constant8),
            push_constant_16bit: t(v11.storage_push_constant16),
        }
    }

    /// The chain that enables lava's needs and `features`.
    pub(crate) fn enabling(features: &Features) -> Self {
        let on = |b: bool| b as vk::Bool32;
        let f = features;
        let mut chain = Self::default();
        chain.core = vk::PhysicalDeviceFeatures::default()
            .shader_int64(true)
            .fragment_stores_and_atomics(true)
            .vertex_pipeline_stores_and_atomics(true)
            .texture_compression_bc(f.bc_compression)
            .independent_blend(f.independent_blend)
            .fill_mode_non_solid(f.fill_mode_non_solid)
            .wide_lines(f.wide_lines)
            .depth_bounds(f.depth_bounds)
            .depth_clamp(f.depth_clamp)
            .logic_op(f.logic_op)
            .sampler_anisotropy(f.sampler_anisotropy)
            .shader_int16(f.int16)
            .shader_float64(f.float64)
            .pipeline_statistics_query(f.pipeline_statistics)
            .occlusion_query_precise(f.precise_occlusion);
        chain.vk11 = vk::PhysicalDeviceVulkan11Features::default()
            .shader_draw_parameters(true)
            .storage_buffer16_bit_access(f.storage_16bit)
            .uniform_and_storage_buffer16_bit_access(f.storage_16bit)
            .storage_push_constant16(f.push_constant_16bit);
        chain.vk12 = vk::PhysicalDeviceVulkan12Features::default()
            .descriptor_indexing(true)
            .runtime_descriptor_array(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .shader_storage_image_array_non_uniform_indexing(true)
            .descriptor_binding_partially_bound(true)
            .descriptor_binding_variable_descriptor_count(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .descriptor_binding_storage_image_update_after_bind(true)
            .descriptor_binding_update_unused_while_pending(true)
            .buffer_device_address(true)
            .timeline_semaphore(true)
            .draw_indirect_count(true)
            .scalar_block_layout(true)
            .vulkan_memory_model(true)
            .vulkan_memory_model_device_scope(true)
            .host_query_reset(true)
            .shader_int8(f.int8)
            .shader_float16(f.float16)
            .shader_buffer_int64_atomics(f.int64_atomics)
            .storage_buffer8_bit_access(f.storage_8bit)
            .uniform_and_storage_buffer8_bit_access(f.storage_8bit)
            .storage_push_constant8(f.push_constant_8bit);
        chain.vk13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true)
            .maintenance4(true)
            // What `discard` compiles to.
            .shader_demote_to_helper_invocation(true);
        chain.address_commands.device_address_commands = vk::TRUE;
        chain.dynamic_state2.extended_dynamic_state2_logic_op = vk::TRUE;
        let d3 = &mut chain.dynamic_state3;
        for bit in [
            &mut d3.extended_dynamic_state3_polygon_mode,
            &mut d3.extended_dynamic_state3_depth_clamp_enable,
            &mut d3.extended_dynamic_state3_depth_clip_enable,
            &mut d3.extended_dynamic_state3_depth_clip_negative_one_to_one,
            &mut d3.extended_dynamic_state3_rasterization_samples,
            &mut d3.extended_dynamic_state3_sample_mask,
            &mut d3.extended_dynamic_state3_alpha_to_coverage_enable,
            &mut d3.extended_dynamic_state3_alpha_to_one_enable,
            &mut d3.extended_dynamic_state3_logic_op_enable,
            &mut d3.extended_dynamic_state3_color_blend_enable,
            &mut d3.extended_dynamic_state3_color_blend_equation,
            &mut d3.extended_dynamic_state3_color_write_mask,
            &mut d3.extended_dynamic_state3_conservative_rasterization_mode,
            &mut d3.extended_dynamic_state3_line_rasterization_mode,
            &mut d3.extended_dynamic_state3_line_stipple_enable,
            &mut d3.extended_dynamic_state3_provoking_vertex_mode,
            &mut d3.extended_dynamic_state3_sample_locations_enable,
        ] {
            *bit = vk::TRUE;
        }
        chain.color_write.color_write_enable = vk::TRUE;
        chain.provoking_vertex.provoking_vertex_last = vk::TRUE;
        chain.depth_clip_enable.depth_clip_enable = vk::TRUE;
        chain.depth_clip_control.depth_clip_control = vk::TRUE;
        chain.depth_clamp_control.depth_clamp_control = vk::TRUE;
        let lines = &mut chain.lines;
        lines.rectangular_lines = vk::TRUE;
        lines.bresenham_lines = on(f.bresenham_lines);
        lines.smooth_lines = on(f.smooth_lines);
        lines.stippled_rectangular_lines = on(f.stippled_rectangular_lines);
        lines.stippled_bresenham_lines = on(f.stippled_bresenham_lines);
        lines.stippled_smooth_lines = on(f.stippled_smooth_lines);
        chain.mesh.task_shader = on(f.mesh);
        chain.mesh.mesh_shader = on(f.mesh);
        chain.mesh.mesh_shader_queries = on(f.mesh_queries);
        chain.acceleration_structure.acceleration_structure = on(f.acceleration_structure);
        chain.ray_query.ray_query = on(f.ray_query);
        chain.ray_tracing_pipeline.ray_tracing_pipeline = on(f.ray_tracing_pipeline);
        chain.ray_tracing_pipeline.ray_tracing_pipeline_trace_rays_indirect =
            on(f.ray_tracing_pipeline);
        chain.host_image_copy.host_image_copy = on(f.host_image_copy);
        chain.clock.shader_subgroup_clock = on(f.shader_clock);
        chain.performance.performance_counter_query_pools = on(f.performance_query);
        chain
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn module(instructions: &[&[u32]]) -> Vec<u8> {
        let header = [0x0723_0203u32, 0x0001_0600, 0, 1, 0];
        let words = header.iter().chain(instructions.iter().flat_map(|i| i.iter()));
        words.flat_map(|w| w.to_le_bytes()).collect()
    }

    fn capability(id: u32) -> [u32; 2] {
        [2 << 16 | 17, id]
    }

    #[test]
    fn capabilities_map_to_the_features_they_need() {
        let spirv = module(&[&capability(1), &capability(39), &capability(4472), &capability(5283)]);
        let features = Features::of_spirv(&spirv);
        assert!(features.int8 && features.ray_query && features.mesh);
        assert!(!features.float16 && !features.shader_clock);
        // `Shader` (1) needs nothing optional.
        assert_eq!(Features::of_spirv(&module(&[&capability(1)])), Features::default());
    }

    #[test]
    fn capability_operands_elsewhere_are_not_capabilities() {
        // `OpExtension "SPV_KHR_ray_query"` (opcode 10) holds the text, not a capability.
        let extension = [5 << 16 | 10, 0x5f565053, 0x5f52484b, 0x5f796172, 0x72657571];
        let spirv = module(&[&extension, &capability(1)]);
        assert_eq!(Features::of_spirv(&spirv), Features::default());
    }

    #[test]
    fn missing_features_are_named() {
        let wanted = Features {
            mesh: true,
            bc_compression: true,
            int8: true,
            ..Default::default()
        };
        let available = Features {
            mesh: true,
            ..Default::default()
        };
        assert_eq!(wanted.missing_from(&available), ["bc_compression", "int8"]);
        assert!(wanted.missing_from(&wanted).is_empty());
        assert_eq!(wanted.union(Features::default()), wanted);
    }

    /// Every bit lava checks on a device it also enables, without any optional feature.
    #[test]
    fn lava_enables_everything_it_requires() {
        let chain = FeatureChain::enabling(&Features::default());
        for (name, bit) in chain.lava_bits() {
            assert_eq!(bit, vk::TRUE, "{name}");
        }
        assert_eq!(chain.mesh.mesh_shader, vk::FALSE);
        assert_eq!(chain.core.texture_compression_bc, vk::FALSE);
    }

    /// What enabling a feature sets is what detecting it reads.
    #[test]
    fn enabled_optional_features_read_back_as_supported() {
        let all = Features::from_bits(std::iter::repeat(true));
        let chain = FeatureChain::enabling(&all);
        assert_eq!(chain.optional(&|_| true, true), all);
    }

    #[test]
    fn extensions_follow_the_features() {
        assert!(Features::default().extensions().is_empty());
        let mesh = Features {
            mesh: true,
            ..Default::default()
        };
        assert_eq!(mesh.extensions(), [ash::ext::mesh_shader::NAME]);
    }
}

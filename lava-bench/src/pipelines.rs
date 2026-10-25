//! The baseline's pipelines, made from the pass map's SPIR-V before timing starts with the flags lava's pipelines have (descriptor heaps, no layout), and their state baked in but the viewport and scissor

use std::sync::Mutex;

use lava::pipeline::{PassEntry, PassKind};
use lava::vk;

use crate::bindings::PASS_MAP;

/// Every pipeline and module made, destroyed by `destroy_all`.
static MADE: Mutex<(Vec<vk::Pipeline>, Vec<vk::ShaderModule>)> =
    Mutex::new((Vec::new(), Vec::new()));

/// The pass named `name` in the pass map.
pub fn pass(name: &str) -> &'static PassEntry {
    (PASS_MAP.passes.iter())
        .find(|pass| pass.name == name)
        .unwrap_or_else(|| panic!("no pass {name}"))
}

fn module(pass: &PassEntry) -> vk::ShaderModule {
    let code = ash::util::read_spv(&mut std::io::Cursor::new(pass.spirv)).unwrap();
    let module = unsafe {
        lava::raw::device()
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
            .unwrap()
    };
    MADE.lock().unwrap().1.push(module);
    module
}

fn stage<'a>(
    stage: vk::ShaderStageFlags,
    module: vk::ShaderModule,
    entry: &'static std::ffi::CStr,
    specialization: &'a vk::SpecializationInfo<'a>,
) -> vk::PipelineShaderStageCreateInfo<'a> {
    vk::PipelineShaderStageCreateInfo::default()
        .stage(stage)
        .module(module)
        .name(entry)
        .specialization_info(specialization)
}

/// The compute pipeline of the pass named `name`.
pub fn compute(name: &str) -> vk::Pipeline {
    let pass = pass(name);
    let PassKind::Compute { entry, .. } = pass.kind else {
        panic!("{name} is no compute pass")
    };
    let module = module(pass);
    let specialization = vk::SpecializationInfo::default();
    let mut flags = vk::PipelineCreateFlags2CreateInfoKHR::default().flags(
        vk::PIPELINE_CREATE_2_DESCRIPTOR_HEAP_EXT | vk::PipelineCreateFlags2KHR::DISPATCH_BASE,
    );
    let info = vk::ComputePipelineCreateInfo::default()
        .stage(stage(vk::ShaderStageFlags::COMPUTE, module, entry, &specialization))
        .push_next(&mut flags);
    let pipeline = unsafe {
        lava::raw::device()
            .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
            .unwrap()[0]
    };
    MADE.lock().unwrap().0.push(pipeline);
    pipeline
}

/// Lava's blends that the scenarios use.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Blend {
    Replace,
    Alpha,
    Add,
}

/// The state a raster pipeline bakes in. The defaults are lava's `RasterState` defaults:
/// clockwise front faces, back faces culled, a `GREATER` depth test with writes.
#[derive(Clone, Copy)]
pub struct Raster<'a> {
    pub colors: &'a [vk::Format],
    pub depth: vk::Format,
    pub blend: Blend,
    pub depth_test: bool,
    pub depth_write: bool,
    pub cull: vk::CullModeFlags,
}

impl Default for Raster<'_> {
    fn default() -> Self {
        Self {
            colors: &[],
            depth: vk::Format::UNDEFINED,
            blend: Blend::Replace,
            depth_test: true,
            depth_write: true,
            cull: vk::CullModeFlags::BACK,
        }
    }
}

/// The raster pipeline of the pass named `name` with `state`.
pub fn raster(name: &str, state: Raster) -> vk::Pipeline {
    let pass = pass(name);
    let module = module(pass);
    let specialization = vk::SpecializationInfo::default();
    let mut stages = Vec::new();
    match pass.kind {
        PassKind::RasterVertex { vertex, fragment } => {
            stages.push(stage(vk::ShaderStageFlags::VERTEX, module, vertex, &specialization));
            if let Some(fragment) = fragment {
                stages.push(stage(vk::ShaderStageFlags::FRAGMENT, module, fragment, &specialization));
            }
        }
        _ => panic!("{name} is no vertex raster pass"),
    }
    let mut rendering = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(state.colors)
        .depth_attachment_format(state.depth);
    let mut flags =
        vk::PipelineCreateFlags2CreateInfoKHR::default().flags(vk::PIPELINE_CREATE_2_DESCRIPTOR_HEAP_EXT);
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport = vk::PipelineViewportStateCreateInfo::default();
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(state.cull)
        .front_face(vk::FrontFace::CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(state.depth_test && state.depth != vk::Format::UNDEFINED)
        .depth_write_enable(state.depth_write && state.depth != vk::Format::UNDEFINED)
        .depth_compare_op(vk::CompareOp::GREATER);
    use vk::BlendFactor as F;
    let (blend_enable, src, dst, src_alpha, dst_alpha) = match state.blend {
        Blend::Replace => (false, F::ONE, F::ZERO, F::ONE, F::ZERO),
        Blend::Alpha => (true, F::SRC_ALPHA, F::ONE_MINUS_SRC_ALPHA, F::ONE, F::ZERO),
        Blend::Add => (true, F::ONE, F::ONE, F::ONE, F::ONE),
    };
    let attachments: Vec<_> = (state.colors.iter())
        .map(|_| vk::PipelineColorBlendAttachmentState {
            blend_enable: blend_enable as vk::Bool32,
            src_color_blend_factor: src,
            dst_color_blend_factor: dst,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: src_alpha,
            dst_alpha_blend_factor: dst_alpha,
            alpha_blend_op: vk::BlendOp::ADD,
            color_write_mask: vk::ColorComponentFlags::RGBA,
        })
        .collect();
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
    let dynamic_states = [
        vk::DynamicState::VIEWPORT_WITH_COUNT,
        vk::DynamicState::SCISSOR_WITH_COUNT,
    ];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&assembly)
        .viewport_state(&viewport)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic)
        .push_next(&mut rendering)
        .push_next(&mut flags);
    let pipeline = unsafe {
        lava::raw::device()
            .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
            .unwrap()[0]
    };
    MADE.lock().unwrap().0.push(pipeline);
    pipeline
}

/// Destroys every pipeline and module made, once nothing uses them.
pub fn destroy_all() {
    let device = lava::raw::device();
    let mut made = MADE.lock().unwrap();
    for pipeline in made.0.drain(..) {
        unsafe { device.destroy_pipeline(pipeline, None) };
    }
    for module in made.1.drain(..) {
        unsafe { device.destroy_shader_module(module, None) };
    }
}

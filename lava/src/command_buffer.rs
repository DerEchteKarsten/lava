//! Command recording for generated passes, with pipeline caching and shader hot-reload
use std::{
    collections::HashMap,
    ffi::{CStr, CString},
    fmt::Debug,
    iter,
    marker::PhantomData,
    ops::{IntoBounds, RangeBounds},
    path::{Path, PathBuf},
    range::Range,
    sync::{Mutex, OnceLock, RwLock, mpsc},
    time::Duration,
};

use crate::{
    bindings::{NUM_COMPUTE_PIPELINES, NUM_RASTER_PIPELINES, NUM_RAY_TRACING_PIPELINES, PASS_MAP},
    bindless::Bindless,
    buffer::{
        self,
        slice::BufferSlice,
        usage::{BufferUsage, IsIndex, IsIndirect},
    },
    error::Result,
    image::{
        format::{ColorAspect, DepthAspect, Format},
        slice::{ImageSlice, ImageView},
        usage::{ImageUsage, IsColorAttachment, IsDepthAttachment},
    },
    state::{Ctx, Functions},
    vkobjects::{
        queue::PendingAccesses,
        rt_pipeline::{
            RayTracingShaderCreateInfo, RayTracingShaderGroup, RaytracingPipeline,
            ShaderBindingTable,
        },
    },
};
use ash::vk::{self, BufferCopy, IndexType};
use bytemuck::{NoUninit, Pod, Zeroable, bytes_of};
use glam::{IVec2, UVec2};
use lava_macros::validation_trace;
use notify::EventHandler;
use shader_slang as slang;
use smallvec::{SmallVec, smallvec};

#[derive(Debug, Clone)]
pub(crate) struct BufferAccess {
    pub(crate) stage: vk::PipelineStageFlags2,
    pub(crate) access: vk::AccessFlags2,
    pub(crate) range: Range<u64>,
}

#[derive(Debug, Clone)]
pub(crate) struct ImageAccess {
    pub(crate) stage: vk::PipelineStageFlags2,
    pub(crate) access: vk::AccessFlags2,
    pub(crate) image: vk::Image,
    pub(crate) layout: vk::ImageLayout,
    pub(crate) aspect: vk::ImageAspectFlags,
    pub(crate) old_layout: vk::ImageLayout,
}

#[derive(Debug)]
pub struct CommandBuffer {
    pub(crate) handle: vk::CommandBuffer,
    pub(crate) pending_accesses: PendingAccesses,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Default)]
pub struct ShaderHash {
    pub entry: &'static str,
    pub file: &'static str,
}

pub const PASS_DIR: &str = concat!(env!("OUT_DIR"), "/passes");
pub const SHADER_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../shaders");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassKind {
    Compute {
        entry: &'static str,
    },
    RayTracing {
        ray_gen: &'static str,
        ray_any: &'static str,
        ray_closest: &'static str,
    },
    RasterVertex {
        vertex: &'static str,
        fragment: &'static str,
    },
    RasterMesh {
        amp: Option<&'static str>,
        mesh: &'static str,
        fragment: &'static str,
    },
}

pub struct PassEntry {
    pub path: &'static str,
    pub source: &'static str,
    pub kind: PassKind,
    pub index: usize,
}

static PIPELINES: OnceLock<PipelineManager> = OnceLock::new();

pub(crate) fn pipelines() -> &'static PipelineManager {
    PIPELINES.get().expect(
        "pipelines were not initialized; call `command_buffer::init()` after `lava::init()`",
    )
}

pub struct PipelineManager {
    pass_modules: RwLock<Vec<vk::ShaderModule>>,
    pub compute_pipelines: RwLock<[vk::Pipeline; NUM_COMPUTE_PIPELINES]>,
    pub raytracing_pipelines: RwLock<[Option<RaytracingPipeline>; NUM_RAY_TRACING_PIPELINES]>,
    pub raster_pipelines: RwLock<[HashMap<RasterHash, vk::Pipeline>; NUM_RASTER_PIPELINES]>,
    pending_reloads: Mutex<Vec<(usize, Vec<u8>)>>,
}

pub(crate) struct RetiredPipeline {
    pipeline: vk::Pipeline,
    _sbt: Option<ShaderBindingTable>,
}

impl Drop for RetiredPipeline {
    fn drop(&mut self) {
        unsafe { Ctx::device().destroy_pipeline(self.pipeline, None) };
    }
}

pub(crate) fn apply_pending_reloads() -> Vec<RetiredPipeline> {
    let Some(manager) = PIPELINES.get() else {
        return Vec::new();
    };
    let pending = match manager.pending_reloads.lock() {
        Ok(mut pending) => std::mem::take(&mut *pending),
        Err(_) => {
            tracing::error!("failed to acquire lock on pending shader reloads");
            return Vec::new();
        }
    };
    pending
        .into_iter()
        .flat_map(|(pass_index, spirv)| manager.reload_pass(pass_index, &spirv))
        .collect()
}

impl PipelineManager {
    fn new() -> Self {
        let manager = Self {
            pass_modules: RwLock::new(vec![vk::ShaderModule::null(); PASS_MAP.len()]),
            compute_pipelines: RwLock::new([vk::Pipeline::null(); NUM_COMPUTE_PIPELINES]),
            raytracing_pipelines: RwLock::new([None; NUM_RAY_TRACING_PIPELINES]),
            raster_pipelines: RwLock::new(std::array::from_fn::<_, NUM_RASTER_PIPELINES, _>(
                |_| HashMap::new(),
            )),
            pending_reloads: Mutex::new(Vec::new()),
        };

        for (i, pass) in PASS_MAP.iter().enumerate() {
            manager.reload_pass(i, &read_pass_spirv(pass.path));
        }

        manager
    }

    fn reload_pass(&self, pass_index: usize, spirv: &[u8]) -> Vec<RetiredPipeline> {
        let pass = &PASS_MAP[pass_index];
        let module = create_module(spirv);
        let mut retired = Vec::new();

        let Ok(mut modules) = self.pass_modules.write() else {
            tracing::error!("failed to acquire lock on pass modules");
            return retired;
        };
        let old = std::mem::replace(&mut modules[pass_index], module);
        unsafe { Ctx::device().destroy_shader_module(old, None) };

        match pass.kind {
            PassKind::Compute { entry } => {
                let pipeline = create_compute_pipeline(module, entry);
                let Ok(mut pipelines) = self.compute_pipelines.write() else {
                    tracing::error!("failed to acquire lock on compute pipelines");
                    return retired;
                };
                retired.push(RetiredPipeline {
                    pipeline: std::mem::replace(&mut pipelines[pass.index], pipeline),
                    _sbt: None,
                });
            }
            PassKind::RayTracing {
                ray_any,
                ray_closest,
                ray_gen,
            } => {
                let pipeline = create_raytracing_pipeline(module, ray_any, ray_closest, ray_gen);
                let Ok(mut pipelines) = self.raytracing_pipelines.write() else {
                    tracing::error!("failed to acquire lock on ray tracing pipelines");
                    return retired;
                };
                if let Some(old) = pipelines[pass.index].replace(pipeline) {
                    retired.push(RetiredPipeline {
                        pipeline: old.pipeline,
                        _sbt: Some(old.sbt),
                    });
                }
            }
            PassKind::RasterVertex { .. } | PassKind::RasterMesh { .. } => {
                let Ok(mut pipelines) = self.raster_pipelines.write() else {
                    tracing::error!("failed to acquire lock on raster pipelines");
                    return retired;
                };
                for (k, v) in pipelines[pass.index].iter_mut() {
                    let new = create_raster_pipeline(module, k, pass_index);
                    retired.push(RetiredPipeline {
                        pipeline: std::mem::replace(v, new),
                        _sbt: None,
                    });
                }
            }
        }
        retired
    }
}

fn read_pass_spirv(file: &str) -> Vec<u8> {
    let path = std::path::Path::new(PASS_DIR).join(file);
    std::fs::read(&path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

fn create_compute_pipeline(module: vk::ShaderModule, entry: &str) -> vk::Pipeline {
    let stage = make_shader_stage(entry, vk::ShaderStageFlags::COMPUTE, module);
    let create_info = vk::ComputePipelineCreateInfo::default()
        .layout(Bindless::layout())
        .stage(stage);
    let pipeline = unsafe {
        Ctx::device()
            .create_compute_pipelines(vk::PipelineCache::null(), &[create_info], None)
            .unwrap()
    }[0];
    Functions::set_debug_name(entry, pipeline);
    pipeline
}

fn create_raytracing_pipeline(
    module: vk::ShaderModule,
    raygen_entry: &str,
    hit_entry: &str,
    miss_entry: &str,
) -> RaytracingPipeline {
    let raygen = make_shader_stage(raygen_entry, vk::ShaderStageFlags::RAYGEN_KHR, module);
    let hit = make_shader_stage(hit_entry, vk::ShaderStageFlags::CLOSEST_HIT_KHR, module);
    let miss = make_shader_stage(miss_entry, vk::ShaderStageFlags::MISS_KHR, module);

    RaytracingPipeline::new(
        Bindless::layout(),
        &[
            RayTracingShaderCreateInfo {
                stages: std::slice::from_ref(&raygen),
                group: RayTracingShaderGroup::RayGen,
            },
            RayTracingShaderCreateInfo {
                stages: std::slice::from_ref(&hit),
                group: RayTracingShaderGroup::Hit,
            },
            RayTracingShaderCreateInfo {
                stages: std::slice::from_ref(&miss),
                group: RayTracingShaderGroup::Miss,
            },
        ],
    )
    .unwrap()
}

pub fn init() {
    use notify::Watcher;

    let manager = PIPELINES.get_or_init(PipelineManager::new);

    let manager: &'static PipelineManager = manager;

    static WATCHER: OnceLock<notify::RecommendedWatcher> = OnceLock::new();
    let _ = WATCHER.get_or_init(|| {
        let (changed, rx) = mpsc::channel();
        std::thread::Builder::new()
            .name("shader_hot_reload".into())
            .spawn(move || hot_reload(manager, rx))
            .unwrap();

        let mut watcher = notify::recommended_watcher(FileWatcher { changed }).unwrap();
        for dir in ["passes", "include"] {
            watcher
                .watch(
                    &Path::new(SHADER_DIR).join(dir),
                    notify::RecursiveMode::Recursive,
                )
                .unwrap();
        }
        watcher
    });
}

struct FileWatcher {
    changed: mpsc::Sender<PathBuf>,
}

impl EventHandler for FileWatcher {
    fn handle_event(&mut self, e: notify::Result<notify::Event>) {
        let event = match e {
            Ok(event) => event,
            Err(e) => {
                tracing::error!(target: "shader_file_watcher", "{e}");
                return;
            }
        };

        if !(event.kind.is_modify() || event.kind.is_create()) {
            return;
        }

        for path in event.paths {
            if path.extension().is_some_and(|ext| ext == "slang") {
                let _ = self.changed.send(path);
            }
        }
    }
}

fn hot_reload(manager: &'static PipelineManager, changed: mpsc::Receiver<PathBuf>) {
    let Some(global) = slang::GlobalSession::new() else {
        tracing::error!(target: "shader_file_watcher", "failed to create slang session, hot reloading is disabled");
        return;
    };

    while let Ok(first) = changed.recv() {
        std::thread::sleep(Duration::from_millis(50));
        let paths: Vec<PathBuf> = iter::once(first).chain(changed.try_iter()).collect();

        for pass_index in passes_to_reload(&paths, &PASS_MAP) {
            let pass = &PASS_MAP[pass_index];

            match compile_pass(&global, pass.source) {
                Ok(spirv) => {
                    tracing::info!(target: "shader_file_watcher", "Reloading pass `{}`", pass.source);
                    match manager.pending_reloads.lock() {
                        Ok(mut pending) => pending.push((pass_index, spirv.as_slice().to_vec())),
                        Err(_) => {
                            tracing::error!(target: "shader_file_watcher", "failed to acquire lock on pending shader reloads")
                        }
                    }
                }
                Err(e) => {
                    tracing::error!(target: "shader_file_watcher", "{}:\n{e}", pass.source);
                }
            }
        }
    }
}

/// Indices of the passes that have to be recompiled after `changed` files were modified:
/// the passes whose own source changed, or every pass as soon as a non-pass file (an include)
/// is among them.
fn passes_to_reload(changed: &[PathBuf], passes: &[PassEntry]) -> Vec<usize> {
    let is_source_of = |path: &PathBuf, pass: &PassEntry| path.ends_with(pass.source);
    let all = changed
        .iter()
        .any(|path| !passes.iter().any(|pass| is_source_of(path, pass)));
    passes
        .iter()
        .enumerate()
        .filter(|(_, pass)| all || changed.iter().any(|path| is_source_of(path, pass)))
        .map(|(index, _)| index)
        .collect()
}

fn compile_pass(global: &slang::GlobalSession, source: &str) -> slang::Result<slang::Blob> {
    let root = CString::new(SHADER_DIR).unwrap();
    let include = CString::new(format!("{SHADER_DIR}/include")).unwrap();
    // Lava's own test passes live outside `shaders/`; see `build.rs`.
    #[cfg(feature = "test-passes")]
    let tests = CString::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests")).unwrap();
    let search_paths = [
        root.as_ptr(),
        include.as_ptr(),
        #[cfg(feature = "test-passes")]
        tests.as_ptr(),
    ];
    let targets = [slang::TargetDesc::default()
        .format(slang::CompileTarget::Spirv)
        .profile(global.find_profile("spirv_1_6"))];
    let options = slang::CompilerOptions::default()
        .vulkan_use_entry_point_name(true)
        .matrix_layout_column(true);

    let session = global
        .create_session(
            &slang::SessionDesc::default()
                .targets(&targets)
                .search_paths(&search_paths)
                .options(&options),
        )
        .expect("failed to create slang session");

    let module = session.load_module(source)?;
    let mut components: Vec<slang::ComponentType> = vec![module.clone().into()];
    components.extend(module.entry_points().map(Into::into));

    session
        .create_composite_component_type(&components)?
        .link()?
        .target_code(0)
}

fn create_module(bytes: &[u8]) -> vk::ShaderModule {
    let decoded_code = ash::util::read_spv(&mut std::io::Cursor::new(bytes)).unwrap();
    let create_info = vk::ShaderModuleCreateInfo::default().code(&decoded_code);

    unsafe {
        Ctx::device()
            .create_shader_module(&create_info, None)
            .unwrap()
    }
}

fn make_shader_stage<'a>(
    entry: &'a str,
    stage: vk::ShaderStageFlags,
    module: vk::ShaderModule,
) -> vk::PipelineShaderStageCreateInfo<'a> {
    vk::PipelineShaderStageCreateInfo::default()
        .stage(stage)
        .module(module)
        .name(CStr::from_bytes_with_nul(entry.as_bytes()).unwrap())
}

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub struct RasterHash {
    backface_culling: bool,
    wire_frame: bool,
    color_formats: SmallVec<[vk::Format; 4]>,
    depth_format: vk::Format,
    stencil_format: vk::Format,
    /// Whether fragments that pass the depth test write their depth.
    depth_write: bool,
}

fn get_raster_pipeline(hash: &RasterHash, pass_index: usize) -> vk::Pipeline {
    let manager = pipelines();
    let map = &mut manager.raster_pipelines.write().unwrap()[PASS_MAP[pass_index].index];
    if let Some(pipeline) = map.get(hash) {
        return *pipeline;
    }
    let module = manager.pass_modules.read().unwrap()[pass_index];
    let pipeline = create_raster_pipeline(module, hash, pass_index);
    map.insert(hash.clone(), pipeline);
    pipeline
}

fn create_raster_pipeline(
    module: vk::ShaderModule,
    hash: &RasterHash,
    pass_index: usize,
) -> vk::Pipeline {
    let _span = tracing::info_span!("create raster pipeline");

    let pass = &PASS_MAP[pass_index];
    let stages = match pass.kind {
        PassKind::RasterVertex { vertex, fragment } => {
            smallvec![
                make_shader_stage(vertex, vk::ShaderStageFlags::VERTEX, module),
                make_shader_stage(fragment, vk::ShaderStageFlags::FRAGMENT, module),
            ]
        }
        PassKind::RasterMesh {
            amp,
            mesh,
            fragment,
        } => {
            let mut stages = SmallVec::<[vk::PipelineShaderStageCreateInfo<'_>; 3]>::new();
            if let Some(task) = amp {
                stages.push(make_shader_stage(
                    task,
                    vk::ShaderStageFlags::TASK_EXT,
                    module,
                ));
            }
            stages.push(make_shader_stage(
                mesh,
                vk::ShaderStageFlags::MESH_EXT,
                module,
            ));
            stages.push(make_shader_stage(
                fragment,
                vk::ShaderStageFlags::FRAGMENT,
                module,
            ));
            stages
        }
        other => panic!("pass {other:?} is not a raster pass"),
    };

    let mut create_info = vk::GraphicsPipelineCreateInfo::default();
    let ia = vk::PipelineInputAssemblyStateCreateInfo::default()
        .primitive_restart_enable(false)
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_attribute_descriptions(&[])
        .vertex_binding_descriptions(&[]);
    create_info = create_info
        .input_assembly_state(&ia)
        .vertex_input_state(&vertex_input_state);

    let mut rendering = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&hash.color_formats)
        .depth_attachment_format(hash.depth_format)
        .stencil_attachment_format(hash.stencil_format)
        .view_mask(0);
    let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
        .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

    let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1)
        .sample_shading_enable(false)
        .min_sample_shading(1.0)
        .alpha_to_coverage_enable(false)
        .alpha_to_one_enable(false)
        .sample_mask(&[]);

    let color_blend_attachments = hash
        .color_formats
        .iter()
        .map(|_| {
            vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(true)
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
                .alpha_blend_op(vk::BlendOp::ADD)
        })
        .collect::<Vec<_>>();
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(color_blend_attachments.as_slice())
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .blend_constants([1.0, 1.0, 1.0, 1.0]);

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .scissor_count(1)
        .viewport_count(1);
    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .line_width(1.0)
        .polygon_mode(if hash.wire_frame {
            vk::PolygonMode::LINE
        } else {
            vk::PolygonMode::FILL
        })
        .cull_mode(if hash.backface_culling {
            vk::CullModeFlags::BACK
        } else {
            vk::CullModeFlags::NONE
        })
        .front_face(vk::FrontFace::CLOCKWISE)
        .depth_bias_enable(true);
    let depth_stencil_state = if hash.depth_format != vk::Format::UNDEFINED {
        vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_bounds_test_enable(false)
            .depth_compare_op(vk::CompareOp::GREATER)
            .depth_test_enable(true)
            .depth_write_enable(hash.depth_write)
            .min_depth_bounds(1.0)
            .max_depth_bounds(0.0)
            .stencil_test_enable(false)
    } else {
        vk::PipelineDepthStencilStateCreateInfo::default()
    };

    create_info = create_info
        .stages(&stages)
        .layout(Bindless::layout())
        .dynamic_state(&dynamic_state)
        .multisample_state(&multisampling)
        .color_blend_state(&color_blend_state)
        .rasterization_state(&rasterization_state)
        .viewport_state(&viewport_state)
        .depth_stencil_state(&depth_stencil_state)
        .base_pipeline_handle(vk::Pipeline::null())
        .base_pipeline_index(-1)
        .push_next(&mut rendering);

    let pipeline = unsafe {
        Ctx::device()
            .create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
            .unwrap()
    }[0];

    pipeline
}

pub struct BindingOutput<GpuBinding: Pod, T: PassType, const Images: usize, const Buffers: usize> {
    pub(crate) images: [ImageAccess; Images],
    pub(crate) buffers: [BufferAccess; Buffers],
    pub(crate) gpu_bindings: GpuBinding,
    /// Shader stages of the pass; accesses registered later use the same stages.
    pub(crate) stage: vk::PipelineStageFlags2,
    pub(crate) _marker: PhantomData<T>,
}

/// Registering resources the pass reaches without naming them in its push constants, e.g.
/// through a bindless handle or pointer stored in a buffer. Each call adds the access to the
/// barriers recorded before the pass and returns a `BindingOutput` that is one entry larger.
impl<GpuBinding: Pod, T: PassType, const Images: usize, const Buffers: usize>
    BindingOutput<GpuBinding, T, Images, Buffers>
{
    /// The pass samples `image` (`Tex2D`, or a `DynImg` with a sampled index).
    pub fn sampled_read<F: Format, U: crate::image::usage::IsSampled>(
        self,
        image: ImageView<'_, F, U>,
    ) -> BindingOutput<GpuBinding, T, { Images + 1 }, Buffers> {
        self.with_image(image, vk::AccessFlags2::SHADER_SAMPLED_READ)
    }

    /// The pass loads from `image` as a storage image (`Image2D`, or a `DynImg` without a
    /// sampled index).
    pub fn storage_read<F: Format, U: crate::image::usage::IsStorage>(
        self,
        image: ImageView<'_, F, U>,
    ) -> BindingOutput<GpuBinding, T, { Images + 1 }, Buffers> {
        self.with_image(image, vk::AccessFlags2::SHADER_STORAGE_READ)
    }

    /// The pass loads from and stores to `image` (`MutImage2D`).
    pub fn storage_write<F: Format, U: crate::image::usage::IsStorage>(
        self,
        image: ImageView<'_, F, U>,
    ) -> BindingOutput<GpuBinding, T, { Images + 1 }, Buffers> {
        self.with_image(
            image,
            vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
        )
    }

    /// The pass reads `buffer` through a pointer (`Buf`).
    pub fn buffer_read<V: Copy + Pod, U: crate::buffer::usage::IsStorage>(
        self,
        buffer: BufferSlice<'_, V, U>,
    ) -> BindingOutput<GpuBinding, T, Images, { Buffers + 1 }> {
        self.with_buffer(buffer, vk::AccessFlags2::SHADER_STORAGE_READ)
    }

    /// The pass reads and writes `buffer` through a pointer (`MutBuf`).
    pub fn buffer_write<V: Copy + Pod, U: crate::buffer::usage::IsStorage>(
        self,
        buffer: BufferSlice<'_, V, U>,
    ) -> BindingOutput<GpuBinding, T, Images, { Buffers + 1 }> {
        self.with_buffer(
            buffer,
            vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
        )
    }

    fn with_image<F: Format, U: ImageUsage>(
        self,
        image: ImageView<'_, F, U>,
        access: vk::AccessFlags2,
    ) -> BindingOutput<GpuBinding, T, { Images + 1 }, Buffers> {
        let added = image.access(self.stage, access, vk::ImageLayout::GENERAL);
        let mut images = self.images.into_iter().chain(std::iter::once(added));
        BindingOutput {
            images: std::array::from_fn(|_| images.next().expect("one access per slot")),
            buffers: self.buffers,
            gpu_bindings: self.gpu_bindings,
            stage: self.stage,
            _marker: PhantomData,
        }
    }

    fn with_buffer<V: Copy + Pod, U: BufferUsage>(
        self,
        buffer: BufferSlice<'_, V, U>,
        access: vk::AccessFlags2,
    ) -> BindingOutput<GpuBinding, T, Images, { Buffers + 1 }> {
        let added = buffer.access(self.stage, access);
        let mut buffers = self.buffers.into_iter().chain(std::iter::once(added));
        BindingOutput {
            images: self.images,
            buffers: std::array::from_fn(|_| buffers.next().expect("one access per slot")),
            gpu_bindings: self.gpu_bindings,
            stage: self.stage,
            _marker: PhantomData,
        }
    }
}

pub trait PassType {
    const PASS_INDEX: usize;
}

pub trait ComputePass: PassType {}
pub trait RaytracingPass: PassType {}
pub trait RasterPass: PassType {}
pub trait RasterVertexPass: RasterPass {}
pub trait RasterMeshPass: RasterPass {}

#[repr(i32)]
pub enum Filter {
    Nearest = 0,
    Liniear = 1,
}

pub struct RasterBuilder<'command_buffer_ref> {
    hash: RasterHash,
    color_attachments: SmallVec<[(vk::ImageView, Option<vk::ClearValue>); 2]>,
    color_accesses: SmallVec<[ImageAccess; 2]>,
    depth_attachment: vk::ImageView,
    depth_access: Option<ImageAccess>,
    clear_depth: Option<vk::ClearValue>,
    write_depth: bool,
    cmd_buf: &'command_buffer_ref mut CommandBuffer,
}

#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct DrawIndirectCommand {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct DispatchIndirectCommand {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Scissor {
    pub offset: IVec2,
    pub extent: UVec2,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Viewport {
    pub offset: IVec2,
    pub extent: UVec2,
}

impl<'a> RasterBuilder<'a> {
    fn render<GpuBinding: Pod, T: RasterPass, const Images: usize, const Buffers: usize>(
        self,
        bindings: BindingOutput<GpuBinding, T, Images, Buffers>,
        access1: Option<BufferAccess>,
        access2: Option<BufferAccess>,
        total_extent: UVec2,
        scissors: &[vk::Rect2D],
        viewport: Viewport,
        draw: impl FnOnce(vk::CommandBuffer),
    ) {
        let _span = tracing::info_span!("draw");

        let pipeline = get_raster_pipeline(&self.hash, T::PASS_INDEX);

        self.cmd_buf.flush_pending(
            bindings
                .images
                .iter()
                .chain(self.color_accesses.iter())
                .chain(self.depth_access.iter()),
            bindings
                .buffers
                .iter()
                .chain(access1.iter())
                .chain(access2.iter()),
        );

        self.cmd_buf.push_constants(&bindings.gpu_bindings);

        let color_attachments = self
            .color_attachments
            .iter()
            .map(|e| {
                let ret = vk::RenderingAttachmentInfo::default()
                    .image_layout(ATTACHMENT_LAYOUT)
                    .image_view(e.0)
                    .store_op(vk::AttachmentStoreOp::STORE);
                if let Some(clear_value) = e.1 {
                    ret.clear_value(clear_value)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                } else {
                    ret.load_op(vk::AttachmentLoadOp::LOAD)
                }
            })
            .collect::<Vec<_>>();
        let mut rendering_info = vk::RenderingInfo::default()
            .color_attachments(color_attachments.as_slice())
            .layer_count(1)
            .render_area(vk::Rect2D {
                extent: vk::Extent2D {
                    height: total_extent.y,
                    width: total_extent.x,
                },
                offset: vk::Offset2D { x: 0, y: 0 },
            })
            .view_mask(0);

        let mut render_info1;

        if self.depth_attachment != vk::ImageView::null() {
            render_info1 = vk::RenderingAttachmentInfo::default()
                .image_layout(ATTACHMENT_LAYOUT)
                .image_view(self.depth_attachment)
                .store_op(vk::AttachmentStoreOp::NONE)
                .load_op(vk::AttachmentLoadOp::LOAD);
            if let Some(clear_value) = self.clear_depth {
                render_info1 = render_info1
                    .clear_value(clear_value)
                    .load_op(vk::AttachmentLoadOp::CLEAR);
            }
            // A clear has to be stored even when the draw itself only tests against depth.
            if self.write_depth || self.clear_depth.is_some() {
                render_info1 = render_info1.store_op(vk::AttachmentStoreOp::STORE);
            }
            rendering_info = rendering_info.depth_attachment(&render_info1);
            if self.hash.stencil_format != vk::Format::UNDEFINED {
                rendering_info = rendering_info.stencil_attachment(&render_info1);
            }
        }

        unsafe {
            Ctx::device().cmd_begin_rendering(self.cmd_buf.handle, &rendering_info);
            Ctx::device().cmd_bind_pipeline(
                self.cmd_buf.handle,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline,
            );

            Ctx::device().cmd_set_viewport(
                self.cmd_buf.handle,
                0,
                &[vk::Viewport {
                    x: viewport.offset.x as f32,
                    y: viewport.offset.y as f32,
                    width: viewport.extent.x as f32,
                    height: viewport.extent.y as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            Ctx::device().cmd_set_scissor(self.cmd_buf.handle, 0, scissors);
            draw(self.cmd_buf.handle);
            Ctx::device().cmd_end_rendering(self.cmd_buf.handle);
        };
    }

    fn full_extent_dynstates(total_extent: UVec2) -> ([Scissor; 1], Viewport) {
        (
            [Scissor {
                extent: total_extent,
                offset: IVec2::ZERO,
            }],
            Viewport {
                extent: total_extent,
                offset: IVec2::ZERO,
            },
        )
    }

    #[validation_trace]
    pub fn draw_with_dynstates<
        GpuBinding: Pod,
        T: RasterVertexPass,
        const Images: usize,
        const Buffers: usize,
    >(
        self,
        bindings: BindingOutput<GpuBinding, T, Images, Buffers>,
        total_extent: UVec2,
        vertex_count: u32,
        instance_count: u32,
        scissors: &[Scissor],
        viewport: Viewport,
    ) {
        self.render(
            bindings,
            None,
            None,
            total_extent,
            unsafe { std::mem::transmute(scissors) },
            viewport,
            |cmd| unsafe {
                Ctx::device().cmd_draw(cmd, vertex_count, instance_count, 0, 0);
            },
        );
    }

    #[validation_trace]
    pub fn draw<GpuBinding: Pod, T: RasterVertexPass, const Images: usize, const Buffers: usize>(
        self,
        bindings: BindingOutput<GpuBinding, T, Images, Buffers>,
        total_extent: UVec2,
        vertex_count: u32,
        instance_count: u32,
    ) {
        let (scissors, viewport) = Self::full_extent_dynstates(total_extent);
        self.draw_with_dynstates(
            bindings,
            total_extent,
            vertex_count,
            instance_count,
            &scissors,
            viewport,
        );
    }

    #[validation_trace]
    pub fn draw_indexed_with_dynstates<
        GpuBinding: Pod,
        T: RasterVertexPass,
        IDX: IsIndex,
        const Images: usize,
        const Buffers: usize,
    >(
        self,
        bindings: BindingOutput<GpuBinding, T, Images, Buffers>,
        total_extent: UVec2,
        index_buffer: BufferSlice<'a, u32, IDX>,
        instance_count: u32,
        scissors: &[Scissor],
        viewport: Viewport,
    ) {
        let index_access = BufferAccess {
            access: vk::AccessFlags2::INDEX_READ,
            stage: vk::PipelineStageFlags2::INDEX_INPUT,
            range: index_buffer.get_range(),
        };
        let id = index_buffer.handle;
        let offset = index_buffer.offset();
        let count = index_buffer.len() as u32;
        self.render(
            bindings,
            Some(index_access),
            None,
            total_extent,
            unsafe { std::mem::transmute(scissors) },
            viewport,
            move |cmd| unsafe {
                Ctx::device().cmd_bind_index_buffer(cmd, id, offset, IndexType::UINT32);
                Ctx::device().cmd_draw_indexed(cmd, count, instance_count, 0, 0, 0);
            },
        );
    }

    #[validation_trace]
    pub fn draw_indexed<
        GpuBinding: Pod,
        T: RasterVertexPass,
        IDX: IsIndex,
        const Images: usize,
        const Buffers: usize,
    >(
        self,
        bindings: BindingOutput<GpuBinding, T, Images, Buffers>,
        total_extent: UVec2,
        index_buffer: BufferSlice<'a, u32, IDX>,
        instance_count: u32,
    ) {
        let (scissors, viewport) = Self::full_extent_dynstates(total_extent);
        self.draw_indexed_with_dynstates(
            bindings,
            total_extent,
            index_buffer,
            instance_count,
            &scissors,
            viewport,
        );
    }

    #[validation_trace]
    pub fn draw_indirect_with_dynstates<
        GpuBinding: Pod,
        T: RasterVertexPass,
        IND: IsIndirect,
        const Images: usize,
        const Buffers: usize,
    >(
        self,
        bindings: BindingOutput<GpuBinding, T, Images, Buffers>,
        total_extent: UVec2,
        buffer: BufferSlice<'a, DrawIndirectCommand, IND>,
        scissors: &[Scissor],
        viewport: Viewport,
    ) {
        let indirect_access = BufferAccess {
            access: vk::AccessFlags2::INDIRECT_COMMAND_READ,
            stage: vk::PipelineStageFlags2::DRAW_INDIRECT,
            range: buffer.get_range(),
        };
        let id = buffer.handle;
        let offset = buffer.offset();
        let draw_count = buffer.len() as u32;
        self.render(
            bindings,
            Some(indirect_access),
            None,
            total_extent,
            unsafe { std::mem::transmute(scissors) },
            viewport,
            move |cmd| unsafe {
                Ctx::device().cmd_draw_indirect(
                    cmd,
                    id,
                    offset,
                    draw_count,
                    size_of::<vk::DrawIndirectCommand>() as u32,
                );
            },
        );
    }

    #[validation_trace]
    pub fn draw_indirect<
        GpuBinding: Pod,
        T: RasterVertexPass,
        IND: IsIndirect,
        const Images: usize,
        const Buffers: usize,
    >(
        self,
        bindings: BindingOutput<GpuBinding, T, Images, Buffers>,
        total_extent: UVec2,
        buffer: BufferSlice<'a, DrawIndirectCommand, IND>,
    ) {
        let (scissors, viewport) = Self::full_extent_dynstates(total_extent);
        self.draw_indirect_with_dynstates(bindings, total_extent, buffer, &scissors, viewport);
    }

    #[validation_trace]
    pub fn draw_indirect_count_with_dynstates<
        GpuBinding: Pod,
        T: RasterVertexPass,
        IND: IsIndirect,
        CNT: IsIndirect,
        const Images: usize,
        const Buffers: usize,
    >(
        self,
        bindings: BindingOutput<GpuBinding, T, Images, Buffers>,
        total_extent: UVec2,
        buffer: BufferSlice<'a, DrawIndirectCommand, IND>,
        count_buffer: BufferSlice<'a, u32, CNT>,
        scissors: &[Scissor],
        viewport: Viewport,
    ) {
        let indirect_access = BufferAccess {
            access: vk::AccessFlags2::INDIRECT_COMMAND_READ,
            stage: vk::PipelineStageFlags2::DRAW_INDIRECT,
            range: buffer.get_range(),
        };
        let count_access = BufferAccess {
            access: vk::AccessFlags2::INDIRECT_COMMAND_READ,
            stage: vk::PipelineStageFlags2::DRAW_INDIRECT,
            range: count_buffer.get_range(),
        };
        let id = buffer.handle;
        let offset = buffer.offset();
        let count_id = count_buffer.handle;
        let count_offset = count_buffer.offset();
        let max_draw_count = buffer.len() as u32;
        self.render(
            bindings,
            Some(indirect_access),
            Some(count_access),
            total_extent,
            unsafe { std::mem::transmute(scissors) },
            viewport,
            move |cmd| unsafe {
                Ctx::device().cmd_draw_indirect_count(
                    cmd,
                    id,
                    offset,
                    count_id,
                    count_offset,
                    max_draw_count,
                    size_of::<vk::DrawIndirectCommand>() as u32,
                );
            },
        );
    }

    #[validation_trace]
    pub fn draw_indirect_count<
        GpuBinding: Pod,
        T: RasterVertexPass,
        IND: IsIndirect,
        CNT: IsIndirect,
        const Images: usize,
        const Buffers: usize,
    >(
        self,
        bindings: BindingOutput<GpuBinding, T, Images, Buffers>,
        total_extent: UVec2,
        buffer: BufferSlice<'a, DrawIndirectCommand, IND>,
        count_buffer: BufferSlice<'a, u32, CNT>,
    ) {
        let (scissors, viewport) = Self::full_extent_dynstates(total_extent);
        self.draw_indirect_count_with_dynstates(
            bindings,
            total_extent,
            buffer,
            count_buffer,
            &scissors,
            viewport,
        );
    }

    #[validation_trace]
    pub fn launch_with_dynstates<
        GpuBinding: Pod,
        T: RasterMeshPass,
        const Images: usize,
        const Buffers: usize,
    >(
        self,
        bindings: BindingOutput<GpuBinding, T, Images, Buffers>,
        x: u32,
        y: u32,
        z: u32,
        extend: UVec2,
        scissors: &[Scissor],
        viewport: Viewport,
    ) {
        self.render(
            bindings,
            None,
            None,
            extend,
            unsafe { std::mem::transmute(scissors) },
            viewport,
            |cmd| unsafe {
                Functions::mesh().unwrap().cmd_draw_mesh_tasks(cmd, x, y, z);
            },
        );
    }

    #[validation_trace]
    pub fn launch<GpuBinding: Pod, T: RasterMeshPass, const Images: usize, const Buffers: usize>(
        self,
        bindings: BindingOutput<GpuBinding, T, Images, Buffers>,
        x: u32,
        y: u32,
        z: u32,
        extent: UVec2,
    ) {
        let (scissors, viewport) = Self::full_extent_dynstates(extent);
        self.launch_with_dynstates(bindings, x, y, z, extent, &scissors, viewport);
    }

    pub fn backface_culling(mut self, backface_culling: bool) -> Self {
        self.hash.backface_culling = backface_culling;
        self
    }
    pub fn wire_frame(mut self, wire_frame: bool) -> Self {
        self.hash.wire_frame = wire_frame;
        self
    }
    pub fn color_attachment<F: Format, U>(
        mut self,
        image: ImageView<'a, F, U>,
        clear: Option<F::Texels>,
    ) -> Self
    where
        U: IsColorAttachment,
        F: ColorAspect,
    {
        assert!(F::ASPECTS.contains(vk::ImageAspectFlags::COLOR));
        self.hash.color_formats.push(F::format());
        let no_clear = clear.is_none();
        self.color_attachments
            .push((image.view, clear.map(|e| F::clear_value(e))));
        self.color_accesses.push(image.access(
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                | if no_clear {
                    vk::AccessFlags2::COLOR_ATTACHMENT_READ
                } else {
                    vk::AccessFlags2::empty()
                },
            ATTACHMENT_LAYOUT,
        ));
        self
    }

    /// Depth-tests the draw against `image` (nearer = greater). `clear` resets the depth first;
    /// with `write` off, fragments are tested but leave the stored depth untouched.
    pub fn depth_attachment<'b, F: Format, U>(
        mut self,
        image: ImageView<'b, F, U>,
        clear: Option<F::Texels>,
        write: bool,
    ) -> Self
    where
        U: IsDepthAttachment,
        F: DepthAspect,
    {
        assert!(F::ASPECTS.contains(vk::ImageAspectFlags::DEPTH));
        self.hash.depth_format = F::format();
        if F::ASPECTS.contains(vk::ImageAspectFlags::STENCIL) {
            self.hash.stencil_format = F::format();
        }
        self.depth_attachment = image.view;
        self.clear_depth = clear.map(|e| F::clear_value(e));
        self.write_depth = write;
        self.hash.depth_write = write;
        self.depth_access = Some(image.access(
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                | if self.write_depth || self.clear_depth.is_some() {
                    vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
                } else {
                    vk::AccessFlags2::empty()
                },
            ATTACHMENT_LAYOUT,
        ));
        self
    }
}

impl CommandBuffer {
    #[validation_trace]
    pub fn fill_buffer<'a, T: Copy + Pod>(&mut self, buffer: BufferSlice<'a, T>, data: u32) {
        let _span = tracing::info_span!("fill_buffer");
        self.flush_pending(
            std::iter::empty(),
            std::iter::once(&BufferAccess {
                access: vk::AccessFlags2::TRANSFER_WRITE,
                stage: vk::PipelineStageFlags2::TRANSFER,
                range: buffer.get_range(),
            }),
        );
        unsafe {
            Ctx::device().cmd_fill_buffer(
                self.handle,
                buffer.handle,
                buffer.offset(),
                buffer.size,
                data,
            )
        };
    }

    #[validation_trace]
    pub fn clear_image<'a, F: Format, U: ImageUsage>(
        &'a mut self,
        image: ImageView<'a, F, U>,
        clear_color: F::Texels,
    ) where
        F: ColorAspect,
    {
        let _span = tracing::info_span!("clear_image");
        self.flush_pending(
            std::iter::once(&image.access(
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::ImageLayout::GENERAL,
            )),
            std::iter::empty(),
        );
        unsafe {
            Ctx::device().cmd_clear_color_image(
                self.handle,
                image.image,
                vk::ImageLayout::GENERAL,
                &F::clear_value(clear_color).color,
                &[image.subresource_range()],
            )
        };
    }
    #[validation_trace]
    pub fn update_buffer<'a, T: Copy + Pod, U: BufferUsage>(
        &mut self,
        buffer: BufferSlice<'a, T, U>,
        data: &T,
    ) {
        let _span = tracing::info_span!("update_buffer_element");
        self.flush_pending(
            iter::empty(),
            iter::once(&BufferAccess {
                stage: vk::PipelineStageFlags2::TRANSFER,
                access: vk::AccessFlags2::TRANSFER_WRITE,
                range: buffer.get_range(),
            }),
        );
        unsafe {
            Ctx::device().cmd_update_buffer(
                self.handle,
                buffer.handle,
                buffer.offset(),
                bytemuck::bytes_of(data),
            )
        };
    }
    #[validation_trace]
    pub fn blit_image<'a, F: Format, U: ImageUsage, F2: Format, U2: ImageUsage>(
        &mut self,
        src: ImageSlice<'a, F, U>,
        dst: ImageSlice<'a, F2, U2>,
        filter: Filter,
    ) where
        F: ColorAspect,
        F2: ColorAspect,
    {
        let _span = tracing::info_span!("blit_image");
        self.flush_pending(
            [
                src.view.access(
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::TRANSFER_READ,
                    vk::ImageLayout::GENERAL,
                ),
                dst.view.access(
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::TRANSFER_WRITE,
                    vk::ImageLayout::GENERAL,
                ),
            ]
            .iter(),
            iter::empty(),
        );

        let regions = [blit_region(&src, &dst)];
        unsafe {
            Ctx::device().cmd_blit_image(
                self.handle,
                src.view.image,
                vk::ImageLayout::GENERAL,
                dst.view.image,
                vk::ImageLayout::GENERAL,
                &regions,
                vk::Filter::from_raw(filter as i32),
            );
        }
    }
    #[validation_trace]

    pub fn copy_buffer<'a, T: Copy + Pod>(
        &mut self,
        src: BufferSlice<'a, T>,
        dst: BufferSlice<'a, T>,
    ) {
        self.copy_buffer_regions(src, dst, &[src.region(dst)]);
    }
    #[validation_trace]

    pub fn copy_buffer_regions<'a, T: Copy + Pod>(
        &mut self,
        src: BufferSlice<'a, T>,
        dst: BufferSlice<'a, T>,
        regions: &[BufferCopy],
    ) {
        let _span = tracing::info_span!("copy_buffer");
        self.flush_pending(
            iter::empty(),
            [
                BufferAccess {
                    access: vk::AccessFlags2::TRANSFER_READ,
                    stage: vk::PipelineStageFlags2::TRANSFER,
                    range: src.get_range(),
                },
                BufferAccess {
                    access: vk::AccessFlags2::TRANSFER_WRITE,
                    stage: vk::PipelineStageFlags2::TRANSFER,
                    range: dst.get_range(),
                },
            ]
            .iter(),
        );
        unsafe { Ctx::device().cmd_copy_buffer(self.handle, src.handle, dst.handle, regions) };
    }

    #[validation_trace]
    pub fn copy_buffer_to_image<'a, T: Copy + Pod, F: Format, U: ImageUsage>(
        &mut self,
        src: BufferSlice<'a, T>,
        dst: ImageSlice<'a, F, U>,
    ) where
        F: ColorAspect,
    {
        let _span = tracing::info_span!("copy_buffer_to_image");
        self.flush_pending(
            iter::once(&dst.view.access(
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::ImageLayout::GENERAL,
            )),
            iter::once(&BufferAccess {
                access: vk::AccessFlags2::TRANSFER_READ,
                stage: vk::PipelineStageFlags2::TRANSFER,
                range: src.get_range(),
            }),
        );
        let regions = [vk::BufferImageCopy {
            image_extent: vk::Extent3D {
                width: dst.extend.x,
                height: dst.extend.y,
                depth: 1,
            },
            image_subresource: dst.view.subresource_layers(dst.view.mip_range.start),
            buffer_image_height: 0,
            buffer_offset: src.offset(),
            buffer_row_length: 0,
            image_offset: vk::Offset3D {
                x: dst.offset.x,
                y: dst.offset.y,
                z: 0,
            },
        }];
        unsafe {
            Ctx::device().cmd_copy_buffer_to_image(
                self.handle,
                src.handle,
                dst.view.image,
                vk::ImageLayout::GENERAL,
                &regions,
            )
        };
    }

    /// Copies the texels of `src` (its first mip level) into `dst`, tightly packed row by row.
    #[validation_trace]
    pub fn copy_image_to_buffer<'a, T: Copy + Pod, F: Format, U: ImageUsage, BU: BufferUsage>(
        &mut self,
        src: ImageSlice<'a, F, U>,
        dst: BufferSlice<'a, T, BU>,
    ) where
        F: ColorAspect,
    {
        let _span = tracing::info_span!("copy_image_to_buffer");
        self.flush_pending(
            iter::once(&src.view.access(
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_READ,
                vk::ImageLayout::GENERAL,
            )),
            iter::once(&BufferAccess {
                access: vk::AccessFlags2::TRANSFER_WRITE,
                stage: vk::PipelineStageFlags2::TRANSFER,
                range: dst.get_range(),
            }),
        );
        let regions = [vk::BufferImageCopy {
            image_extent: vk::Extent3D {
                width: src.extend.x,
                height: src.extend.y,
                depth: 1,
            },
            image_subresource: src.view.subresource_layers(src.view.mip_range.start),
            buffer_image_height: 0,
            buffer_offset: dst.offset(),
            buffer_row_length: 0,
            image_offset: vk::Offset3D {
                x: src.offset.x,
                y: src.offset.y,
                z: 0,
            },
        }];
        unsafe {
            Ctx::device().cmd_copy_image_to_buffer(
                self.handle,
                src.view.image,
                vk::ImageLayout::GENERAL,
                dst.handle,
                &regions,
            )
        };
    }

    #[validation_trace]
    pub fn raster<'a>(&'a mut self) -> RasterBuilder<'a> {
        RasterBuilder {
            write_depth: true,
            clear_depth: None,
            cmd_buf: self,
            color_accesses: SmallVec::new(),
            color_attachments: SmallVec::new(),
            depth_attachment: vk::ImageView::null(),
            hash: RasterHash {
                backface_culling: true,
                color_formats: SmallVec::new(),
                depth_format: vk::Format::UNDEFINED,
                stencil_format: vk::Format::UNDEFINED,
                depth_write: true,
                wire_frame: false,
            },
            depth_access: None,
        }
    }

    fn compute_private<
        'b,
        GpuBinding: Pod,
        T: ComputePass,
        IND: IsIndirect,
        const Images: usize,
        const Buffers: usize,
    >(
        &mut self,
        bindings: BindingOutput<GpuBinding, T, Images, Buffers>,
        dispatch: [u32; 3],
        indirect_buffer: Option<BufferSlice<'b, DispatchIndirectCommand, IND>>,
    ) {
        let _span = tracing::info_span!("compute");
        let access1 = indirect_buffer.map(|b| BufferAccess {
            access: vk::AccessFlags2::INDIRECT_COMMAND_READ,
            stage: vk::PipelineStageFlags2::DRAW_INDIRECT,
            range: b.get_range(),
        });
        self.flush_pending(
            bindings.images.iter(),
            bindings.buffers.iter().chain(access1.iter()),
        );
        self.push_constants(&bindings.gpu_bindings);

        let pipeline = pipelines().compute_pipelines.read().unwrap()[PASS_MAP[T::PASS_INDEX].index];

        unsafe {
            Ctx::device().cmd_bind_pipeline(self.handle, vk::PipelineBindPoint::COMPUTE, pipeline);
            if let Some(slice) = indirect_buffer {
                Ctx::device().cmd_dispatch_indirect(self.handle, slice.handle, slice.offset());
            } else {
                Ctx::device().cmd_dispatch(self.handle, dispatch[0], dispatch[1], dispatch[2]);
            }
        }
    }

    #[validation_trace]
    pub fn compute_indirect<
        'b,
        GpuBinding: Pod,
        T: ComputePass,
        IND: IsIndirect,
        const Images: usize,
        const Buffers: usize,
    >(
        &mut self,
        bindings: BindingOutput<GpuBinding, T, Images, Buffers>,
        buffer: BufferSlice<'b, DispatchIndirectCommand, IND>,
    ) {
        self.compute_private(bindings, [0, 0, 0], Some(buffer));
    }

    #[validation_trace]
    pub fn compute<
        'b,
        GpuBinding: Pod,
        T: ComputePass,
        const Images: usize,
        const Buffers: usize,
    >(
        &mut self,
        bindings: BindingOutput<GpuBinding, T, Images, Buffers>,
        dispatch: [u32; 3],
    ) {
        self.compute_private::<_, _, buffer::usage::Indirect, _, _>(bindings, dispatch, None);
    }

    pub fn raytrace<
        'b,
        GpuBinding: Pod,
        T: RaytracingPass,
        const Images: usize,
        const Buffers: usize,
    >(
        &mut self,
        bindings: BindingOutput<GpuBinding, T, Images, Buffers>,
        x: u32,
        y: u32,
    ) {
        self.flush_pending(bindings.images.iter(), bindings.buffers.iter());
        self.push_constants(&bindings.gpu_bindings);

        let lock = pipelines().raytracing_pipelines.read().unwrap();
        let pipeline = lock[PASS_MAP[T::PASS_INDEX].index].as_ref().unwrap();
        unsafe {
            Ctx::device().cmd_bind_pipeline(
                self.handle,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                pipeline.pipeline,
            );
            let call_region = vk::StridedDeviceAddressRegionKHR::default();
            Functions::raytracing_pipeline().unwrap().cmd_trace_rays(
                self.handle,
                &pipeline.sbt.raygen_region,
                &pipeline.sbt.hit_region,
                &pipeline.sbt.miss_region,
                &call_region,
                x,
                y,
                1,
            );
        };
    }

    #[validation_trace]
    pub fn present<'a, F: Format, U: ImageUsage>(
        &'a mut self,
        swapchain_image: ImageView<'a, F, U>,
    ) {
        let _span = tracing::info_span!("present_barriers");
        self.flush_pending(
            iter::once(&swapchain_image.access(
                vk::PipelineStageFlags2::empty(),
                vk::AccessFlags2::empty(),
                vk::ImageLayout::PRESENT_SRC_KHR,
            )),
            iter::empty(),
        );
    }

    fn push_constants<'b, T: NoUninit>(&mut self, binding: &T) {
        let constants = bytes_of(binding);
        unsafe {
            Ctx::device().cmd_push_constants(
                self.handle,
                Bindless::layout(),
                vk::ShaderStageFlags::ALL,
                0,
                constants,
            )
        };
    }

    pub(crate) fn flush_pending<'a>(
        &mut self,
        image_acceses: impl Iterator<Item = &'a ImageAccess> + Clone,
        buffer_acceses: impl Iterator<Item = &'a BufferAccess> + Clone,
    ) {
        let barriers = compute_barriers(
            &self.pending_accesses,
            image_acceses.clone(),
            buffer_acceses.clone(),
        );
        if let Some(barriers) = barriers {
            let memory = [barriers.memory];
            let mut info = vk::DependencyInfo::default().memory_barriers(&memory);
            if !barriers.images.is_empty() {
                info = info.image_memory_barriers(&barriers.images);
            }
            unsafe {
                Ctx::device().cmd_pipeline_barrier2(self.handle, &info);
            }
        }

        self.pending_accesses.record(image_acceses, buffer_acceses);
    }

    pub(crate) fn begin(&mut self) -> Result<()> {
        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe { Ctx::device().begin_command_buffer(self.handle, &begin_info)? };
        Bindless::bind(&self.handle);
        Ok(())
    }

    pub(crate) fn end(&mut self) -> Result<()> {
        unsafe { Ctx::device().end_command_buffer(self.handle)? };
        Ok(())
    }
}

/// The layout of colour and depth attachments while they are drawn to. Every other access
/// asks for `GENERAL`, which costs nothing when sampling but can keep a driver from
/// compressing a framebuffer.
pub(crate) const ATTACHMENT_LAYOUT: vk::ImageLayout = vk::ImageLayout::ATTACHMENT_OPTIMAL;

/// Barriers one command needs before it may run, given what is still pending.
pub(crate) struct Barriers {
    pub(crate) memory: vk::MemoryBarrier2<'static>,
    /// Layout transitions; empty when every image already is in the requested layout.
    pub(crate) images: SmallVec<[vk::ImageMemoryBarrier2<'static>; 1]>,
}

/// Works out the barriers needed to run a command with the given accesses after `pending`.
///
/// A hazard exists when the command touches something a pending access wrote (read-after-write,
/// write-after-write) or writes something a pending access read. Buffers conflict when their
/// byte ranges overlap, images when they are the same image. An image whose tracked layout
/// differs from the requested one additionally gets a layout transition. Returns `None` when
/// neither applies.
pub(crate) fn compute_barriers<'a>(
    pending: &PendingAccesses,
    image_acceses: impl Iterator<Item = &'a ImageAccess> + Clone,
    buffer_acceses: impl Iterator<Item = &'a BufferAccess>,
) -> Option<Barriers> {
    let mut src_stage = vk::PipelineStageFlags2::NONE;
    let mut dst_stage = vk::PipelineStageFlags2::NONE;
    let mut src_access = vk::AccessFlags2::empty();
    let mut dst_access = vk::AccessFlags2::empty();

    for buffer_access in buffer_acceses {
        let is_write = buffer_access.access.intersects(all_write_access());
        let is_read = buffer_access.access.intersects(all_read_access());

        if is_write {
            for pending in pending
                .buffer_reads
                .iter()
                .filter(|r| !r.range.intersect(buffer_access.range).is_empty())
            {
                src_stage |= pending.stage;
                src_access |= pending.access;
                dst_stage |= buffer_access.stage;
                dst_access |= buffer_access.access;
            }
        }
        for w in pending
            .buffer_writes
            .iter()
            .filter(|w| !w.range.intersect(buffer_access.range).is_empty())
        {
            if is_read || is_write {
                src_stage |= w.stage;
                src_access |= w.access;
                dst_stage |= buffer_access.stage;
                dst_access |= buffer_access.access;
            }
        }
    }

    for image_access in image_acceses.clone() {
        let is_write = image_access.access.intersects(all_write_access());
        let is_read = image_access.access.intersects(all_read_access());

        if is_write {
            for pending in pending
                .image_reads
                .iter()
                .filter(|r| r.image == image_access.image)
            {
                src_stage |= pending.stage;
                src_access |= pending.access;
                dst_stage |= image_access.stage;
                dst_access |= image_access.access;
            }
        }
        for w in pending
            .image_writes
            .iter()
            .filter(|r| r.image == image_access.image)
        {
            if is_read || is_write {
                src_stage |= w.stage;
                src_access |= w.access;
                dst_stage |= image_access.stage;
                dst_access |= image_access.access;
            }
        }
    }

    let mut image_barriers: SmallVec<[vk::ImageMemoryBarrier2; 1]> = SmallVec::new();
    for image_access in image_acceses {
        let old_layout = image_access.old_layout;
        if old_layout == image_access.layout {
            continue;
        }

        let mut src_stage = vk::PipelineStageFlags2::ALL_COMMANDS;
        let mut src_access = vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE;
        for p in pending
            .image_reads
            .iter()
            .chain(pending.image_writes.iter())
            .filter(|r| r.image == image_access.image)
        {
            src_stage |= p.stage;
            src_access |= p.access;
        }

        let mut dst_stage = image_access.stage;
        let mut dst_access = image_access.access;
        if dst_stage.is_empty() {
            dst_stage = vk::PipelineStageFlags2::ALL_COMMANDS;
        }
        if dst_access.is_empty() {
            dst_access = vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE;
        }

        if old_layout == vk::ImageLayout::PRESENT_SRC_KHR {
            src_access = vk::AccessFlags2::NONE;
        }
        if image_access.layout == vk::ImageLayout::PRESENT_SRC_KHR {
            dst_access = vk::AccessFlags2::NONE;
        }

        image_barriers.push(
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(src_stage)
                .src_access_mask(src_access)
                .dst_stage_mask(dst_stage)
                .dst_access_mask(dst_access)
                .old_layout(old_layout)
                .new_layout(image_access.layout)
                .image(image_access.image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: image_access.aspect,
                    base_array_layer: 0,
                    layer_count: vk::REMAINING_ARRAY_LAYERS,
                    base_mip_level: 0,
                    level_count: vk::REMAINING_MIP_LEVELS,
                }),
        );
    }

    if src_stage.is_empty() && dst_stage.is_empty() && image_barriers.is_empty() {
        return None;
    }

    if src_stage.is_empty() {
        src_stage = vk::PipelineStageFlags2::ALL_COMMANDS;
    }
    if dst_stage.is_empty() {
        dst_stage = vk::PipelineStageFlags2::ALL_COMMANDS;
    }
    if src_access.is_empty() {
        src_access = vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE;
    }

    Some(Barriers {
        memory: vk::MemoryBarrier2::default()
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access),
        images: image_barriers,
    })
}

impl PendingAccesses {
    /// Remembers the accesses of a command that was just recorded, so later commands can be
    /// synchronised against them.
    ///
    /// An older buffer access is only forgotten once a newer access of the same kind covers its
    /// whole range: a partially overlapped write still has bytes nobody has synchronised with.
    pub(crate) fn record<'a>(
        &mut self,
        image_acceses: impl Iterator<Item = &'a ImageAccess>,
        buffer_acceses: impl Iterator<Item = &'a BufferAccess>,
    ) {
        for access in buffer_acceses {
            let covers = |a: &BufferAccess| {
                access.range.start <= a.range.start && a.range.end <= access.range.end
            };
            let writes = access.access.intersects(all_write_access());
            let reads = access.access.intersects(all_read_access());

            if writes {
                self.buffer_writes.retain(|a| !covers(a));
                self.buffer_writes.push(access.clone());
            }
            if reads {
                self.buffer_reads.retain(|a| !covers(a));
                self.buffer_reads.push(access.clone());
            }
        }

        for access in image_acceses {
            let same_image = |a: &ImageAccess| a.image == access.image;
            let writes = access.access.intersects(all_write_access());
            let reads = access.access.intersects(all_read_access());

            if writes {
                self.image_writes.retain(|a| !same_image(a));
                self.image_writes.push(access.clone());
            }
            if reads {
                self.image_reads.retain(|a| !same_image(a));
                self.image_reads.push(access.clone());
            }
        }
    }
}

/// Blit region covering the two slices: from each slice's offset to offset + extent.
fn blit_region<F: Format, U: ImageUsage, F2: Format, U2: ImageUsage>(
    src: &ImageSlice<'_, F, U>,
    dst: &ImageSlice<'_, F2, U2>,
) -> vk::ImageBlit {
    let corners = |offset: IVec2, extend: UVec2| {
        [
            vk::Offset3D {
                x: offset.x,
                y: offset.y,
                z: 0,
            },
            vk::Offset3D {
                x: offset.x + extend.x as i32,
                y: offset.y + extend.y as i32,
                z: 1,
            },
        ]
    };
    vk::ImageBlit {
        src_offsets: corners(src.offset, src.extend),
        dst_offsets: corners(dst.offset, dst.extend),
        src_subresource: src.view.subresource_layers(src.view.mip_range.start),
        dst_subresource: dst.view.subresource_layers(dst.view.mip_range.start),
    }
}

fn all_write_access() -> vk::AccessFlags2 {
    vk::AccessFlags2::SHADER_WRITE
        | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
        | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
        | vk::AccessFlags2::TRANSFER_WRITE
        | vk::AccessFlags2::HOST_WRITE
        | vk::AccessFlags2::MEMORY_WRITE
        | vk::AccessFlags2::SHADER_STORAGE_WRITE
        | vk::AccessFlags2::TRANSFORM_FEEDBACK_WRITE_EXT
        | vk::AccessFlags2::TRANSFORM_FEEDBACK_COUNTER_WRITE_EXT
        | vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR
        | vk::AccessFlags2::COMMAND_PREPROCESS_WRITE_NV
        | vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR
        | vk::AccessFlags2::VIDEO_ENCODE_WRITE_KHR
        | vk::AccessFlags2::MICROMAP_WRITE_EXT
        | vk::AccessFlags2::OPTICAL_FLOW_WRITE_NV
}

fn all_read_access() -> vk::AccessFlags2 {
    vk::AccessFlags2::INDIRECT_COMMAND_READ
        | vk::AccessFlags2::INDEX_READ
        | vk::AccessFlags2::VERTEX_ATTRIBUTE_READ
        | vk::AccessFlags2::UNIFORM_READ
        | vk::AccessFlags2::INPUT_ATTACHMENT_READ
        | vk::AccessFlags2::SHADER_READ
        | vk::AccessFlags2::COLOR_ATTACHMENT_READ
        | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
        | vk::AccessFlags2::TRANSFER_READ
        | vk::AccessFlags2::HOST_READ
        | vk::AccessFlags2::MEMORY_READ
        | vk::AccessFlags2::SHADER_SAMPLED_READ
        | vk::AccessFlags2::SHADER_STORAGE_READ
        | vk::AccessFlags2::SHADER_BINDING_TABLE_READ_KHR
        | vk::AccessFlags2::TRANSFORM_FEEDBACK_COUNTER_READ_EXT
        | vk::AccessFlags2::CONDITIONAL_RENDERING_READ_EXT
        | vk::AccessFlags2::COLOR_ATTACHMENT_READ_NONCOHERENT_EXT
        | vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR
        | vk::AccessFlags2::FRAGMENT_DENSITY_MAP_READ_EXT
        | vk::AccessFlags2::FRAGMENT_SHADING_RATE_ATTACHMENT_READ_KHR
        | vk::AccessFlags2::COMMAND_PREPROCESS_READ_NV
        | vk::AccessFlags2::DESCRIPTOR_BUFFER_READ_EXT
        | vk::AccessFlags2::INVOCATION_MASK_READ_HUAWEI
        | vk::AccessFlags2::VIDEO_DECODE_READ_KHR
        | vk::AccessFlags2::VIDEO_ENCODE_READ_KHR
        | vk::AccessFlags2::MICROMAP_READ_EXT
        | vk::AccessFlags2::OPTICAL_FLOW_READ_NV
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{
        format::{D32Sfloat, D32SfloatS8Uint, R8G8B8A8Unorm},
        usage::{ColorAttachment, DepthAttachment, Sampled},
    };
    use ash::vk::Handle;
    use std::sync::atomic::AtomicU32;
    use vk::{AccessFlags2 as A, ImageLayout as L, PipelineStageFlags2 as S};

    // ---- helpers -------------------------------------------------------------------------

    fn buffer(stage: S, access: A, range: std::ops::Range<u64>) -> BufferAccess {
        BufferAccess {
            stage,
            access,
            range: range.into(),
        }
    }

    fn read(range: std::ops::Range<u64>) -> BufferAccess {
        buffer(S::VERTEX_SHADER, A::SHADER_STORAGE_READ, range)
    }

    fn write(range: std::ops::Range<u64>) -> BufferAccess {
        buffer(S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE, range)
    }

    fn image(id: u64, stage: S, access: A, old_layout: L, layout: L) -> ImageAccess {
        ImageAccess {
            stage,
            access,
            image: vk::Image::from_raw(id),
            layout,
            aspect: vk::ImageAspectFlags::COLOR,
            old_layout,
        }
    }

    fn image_read(id: u64) -> ImageAccess {
        image(
            id,
            S::FRAGMENT_SHADER,
            A::SHADER_SAMPLED_READ,
            L::GENERAL,
            L::GENERAL,
        )
    }

    fn image_write(id: u64) -> ImageAccess {
        image(
            id,
            S::COMPUTE_SHADER,
            A::SHADER_STORAGE_WRITE,
            L::GENERAL,
            L::GENERAL,
        )
    }

    fn pending(buffers: &[BufferAccess], images: &[ImageAccess]) -> PendingAccesses {
        let mut pending = PendingAccesses::default();
        pending.record(images.iter(), buffers.iter());
        pending
    }

    fn buffer_barrier(pending: &PendingAccesses, access: &BufferAccess) -> Option<Barriers> {
        compute_barriers(pending, iter::empty(), iter::once(access))
    }

    fn image_barrier(pending: &PendingAccesses, access: &ImageAccess) -> Option<Barriers> {
        compute_barriers(pending, iter::once(access), iter::empty())
    }

    /// A command buffer that is never handed to Vulkan: enough for state-only code paths.
    fn offline_cmd() -> CommandBuffer {
        CommandBuffer {
            handle: vk::CommandBuffer::null(),
            pending_accesses: PendingAccesses::default(),
        }
    }

    fn view<F: Format, U: ImageUsage>(layout: &AtomicU32) -> ImageView<'_, F, U> {
        ImageView {
            image: vk::Image::from_raw(1),
            view: vk::ImageView::from_raw(2),
            mip_range: (0..1).into(),
            handle: crate::bindless::BindlessHandle::none(),
            layout,
            _marker: PhantomData,
            _marker2: PhantomData,
        }
    }

    fn layout(layout: L) -> AtomicU32 {
        AtomicU32::new(layout.as_raw() as u32)
    }

    // ---- access classification -----------------------------------------------------------

    #[test]
    fn read_and_write_masks_are_disjoint_and_cover_the_accesses_lava_uses() {
        assert!((all_read_access() & all_write_access()).is_empty());
        for access in [
            A::SHADER_STORAGE_READ,
            A::SHADER_SAMPLED_READ,
            A::TRANSFER_READ,
            A::INDEX_READ,
            A::INDIRECT_COMMAND_READ,
            A::COLOR_ATTACHMENT_READ,
            A::DEPTH_STENCIL_ATTACHMENT_READ,
        ] {
            assert!(all_read_access().contains(access), "{access:?}");
        }
        for access in [
            A::SHADER_STORAGE_WRITE,
            A::TRANSFER_WRITE,
            A::COLOR_ATTACHMENT_WRITE,
            A::DEPTH_STENCIL_ATTACHMENT_WRITE,
        ] {
            assert!(all_write_access().contains(access), "{access:?}");
        }
    }

    // ---- buffer hazards ------------------------------------------------------------------

    #[test]
    fn first_access_needs_no_barrier() {
        let nothing = PendingAccesses::default();
        assert!(buffer_barrier(&nothing, &read(0..64)).is_none());
        assert!(buffer_barrier(&nothing, &write(0..64)).is_none());
        assert!(image_barrier(&nothing, &image_write(1)).is_none());
    }

    #[test]
    fn read_after_write_needs_a_barrier_from_the_writer_to_the_reader() {
        let barriers = buffer_barrier(&pending(&[write(0..100)], &[]), &read(10..20)).unwrap();
        assert_eq!(barriers.memory.src_stage_mask, S::COMPUTE_SHADER);
        assert_eq!(barriers.memory.src_access_mask, A::SHADER_STORAGE_WRITE);
        assert_eq!(barriers.memory.dst_stage_mask, S::VERTEX_SHADER);
        assert_eq!(barriers.memory.dst_access_mask, A::SHADER_STORAGE_READ);
        assert!(barriers.images.is_empty());
    }

    #[test]
    fn write_after_write_needs_a_barrier() {
        let transfer = buffer(S::TRANSFER, A::TRANSFER_WRITE, 50..60);
        let barriers = buffer_barrier(&pending(&[write(0..100)], &[]), &transfer).unwrap();
        assert_eq!(barriers.memory.src_stage_mask, S::COMPUTE_SHADER);
        assert_eq!(barriers.memory.dst_stage_mask, S::TRANSFER);
        assert_eq!(barriers.memory.dst_access_mask, A::TRANSFER_WRITE);
    }

    #[test]
    fn write_after_read_needs_a_barrier_from_the_reader() {
        let barriers = buffer_barrier(&pending(&[read(0..100)], &[]), &write(0..8)).unwrap();
        assert_eq!(barriers.memory.src_stage_mask, S::VERTEX_SHADER);
        assert_eq!(barriers.memory.src_access_mask, A::SHADER_STORAGE_READ);
        assert_eq!(barriers.memory.dst_stage_mask, S::COMPUTE_SHADER);
    }

    #[test]
    fn read_after_read_needs_no_barrier() {
        assert!(buffer_barrier(&pending(&[read(0..100)], &[]), &read(0..100)).is_none());
    }

    #[test]
    fn disjoint_and_adjacent_ranges_do_not_conflict() {
        let pending = pending(&[write(100..200)], &[]);
        assert!(buffer_barrier(&pending, &read(0..100)).is_none());
        assert!(buffer_barrier(&pending, &write(200..300)).is_none());
        assert!(buffer_barrier(&pending, &read(1000..2000)).is_none());
        // One shared byte is enough.
        assert!(buffer_barrier(&pending, &read(0..101)).is_some());
        assert!(buffer_barrier(&pending, &read(199..300)).is_some());
    }

    #[test]
    fn barrier_merges_every_conflicting_pending_access() {
        let pending = pending(
            &[
                write(0..10),
                buffer(S::TRANSFER, A::TRANSFER_WRITE, 10..20),
                buffer(S::FRAGMENT_SHADER, A::SHADER_STORAGE_WRITE, 500..600),
            ],
            &[],
        );
        let barriers = buffer_barrier(&pending, &read(0..20)).unwrap();
        assert_eq!(
            barriers.memory.src_stage_mask,
            S::COMPUTE_SHADER | S::TRANSFER
        );
        assert_eq!(
            barriers.memory.src_access_mask,
            A::SHADER_STORAGE_WRITE | A::TRANSFER_WRITE
        );
    }

    #[test]
    fn read_write_access_conflicts_with_pending_reads_and_writes() {
        let read_write = buffer(
            S::COMPUTE_SHADER,
            A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE,
            0..10,
        );
        assert!(buffer_barrier(&pending(&[read(0..10)], &[]), &read_write).is_some());
        assert!(buffer_barrier(&pending(&[write(0..10)], &[]), &read_write).is_some());
    }

    // ---- access recording ----------------------------------------------------------------

    #[test]
    fn accesses_are_recorded_by_kind() {
        let read_write = buffer(
            S::COMPUTE_SHADER,
            A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE,
            0..10,
        );
        let pending = pending(&[read(20..30), write(40..50), read_write], &[]);
        assert_eq!(pending.buffer_reads.len(), 2);
        assert_eq!(pending.buffer_writes.len(), 2);
    }

    #[test]
    fn a_covering_access_replaces_the_older_ones() {
        let pending = pending(&[write(10..20), write(30..40), write(0..100)], &[]);
        assert_eq!(pending.buffer_writes.len(), 1);
        assert_eq!(pending.buffer_writes[0].range, (0u64..100).into());

        // Same range again: still a single entry, lists don't grow frame over frame.
        let mut pending = pending;
        for _ in 0..10 {
            pending.record(iter::empty(), [write(0..100), read(0..100)].iter());
        }
        assert_eq!(pending.buffer_writes.len(), 1);
        assert_eq!(pending.buffer_reads.len(), 1);
    }

    /// Regression: a write that only partially overlapped an older pending write used to
    /// erase it, so a later read of the untouched part got no barrier.
    #[test]
    fn a_partially_overlapping_write_keeps_the_older_write_pending() {
        let pending = pending(
            &[
                write(0..100),
                buffer(S::TRANSFER, A::TRANSFER_WRITE, 50..60),
            ],
            &[],
        );
        assert_eq!(pending.buffer_writes.len(), 2);

        let barriers =
            buffer_barrier(&pending, &read(0..10)).expect("the first write is unsynchronised");
        assert_eq!(barriers.memory.src_stage_mask, S::COMPUTE_SHADER);

        let barriers = buffer_barrier(&pending, &read(52..58)).unwrap();
        assert!(barriers.memory.src_stage_mask.contains(S::TRANSFER));
    }

    #[test]
    fn image_accesses_replace_older_ones_of_the_same_image_only() {
        let other_stage = image(1, S::TRANSFER, A::TRANSFER_WRITE, L::GENERAL, L::GENERAL);
        let pending = pending(&[], &[image_write(1), image_write(2), other_stage]);
        assert_eq!(pending.image_writes.len(), 2);
        let first = pending
            .image_writes
            .iter()
            .find(|a| a.image == vk::Image::from_raw(1))
            .unwrap();
        assert_eq!(first.stage, S::TRANSFER);
    }

    // ---- image hazards and layout transitions --------------------------------------------

    #[test]
    fn image_hazards_are_per_image() {
        let pending = pending(&[], &[image_write(1)]);
        let barriers = image_barrier(&pending, &image_read(1)).unwrap();
        assert_eq!(barriers.memory.src_stage_mask, S::COMPUTE_SHADER);
        assert_eq!(barriers.memory.dst_stage_mask, S::FRAGMENT_SHADER);
        assert!(barriers.images.is_empty(), "same layout: no transition");

        assert!(image_barrier(&pending, &image_read(2)).is_none());
    }

    #[test]
    fn image_write_after_read_needs_a_barrier_but_read_after_read_does_not() {
        let pending = pending(&[], &[image_read(1)]);
        assert!(image_barrier(&pending, &image_write(1)).is_some());
        assert!(image_barrier(&pending, &image_read(1)).is_none());
    }

    #[test]
    fn layout_change_emits_a_transition_for_the_whole_image() {
        let access = image(7, S::TRANSFER, A::TRANSFER_WRITE, L::UNDEFINED, L::GENERAL);
        let barriers = image_barrier(&PendingAccesses::default(), &access).unwrap();
        assert_eq!(barriers.images.len(), 1);
        let transition = &barriers.images[0];
        assert_eq!(transition.image, vk::Image::from_raw(7));
        assert_eq!(
            (transition.old_layout, transition.new_layout),
            (L::UNDEFINED, L::GENERAL)
        );
        assert_eq!(transition.dst_stage_mask, S::TRANSFER);
        assert_eq!(transition.dst_access_mask, A::TRANSFER_WRITE);
        assert_eq!(
            transition.subresource_range.aspect_mask,
            vk::ImageAspectFlags::COLOR
        );
        assert_eq!(
            transition.subresource_range.level_count,
            vk::REMAINING_MIP_LEVELS
        );
        assert_eq!(
            transition.subresource_range.layer_count,
            vk::REMAINING_ARRAY_LAYERS
        );
    }

    #[test]
    fn transition_waits_for_the_pending_accesses_of_that_image() {
        let pending = pending(&[], &[image_write(7), image_write(8)]);
        let access = image(
            7,
            S::TRANSFER,
            A::TRANSFER_READ,
            L::GENERAL,
            L::TRANSFER_SRC_OPTIMAL,
        );
        let barriers = image_barrier(&pending, &access).unwrap();
        let transition = &barriers.images[0];
        assert!(transition.src_stage_mask.contains(S::COMPUTE_SHADER));
        assert!(transition.src_access_mask.contains(A::SHADER_STORAGE_WRITE));
    }

    #[test]
    fn transition_to_present_has_no_destination_access() {
        let access = image(7, S::empty(), A::empty(), L::GENERAL, L::PRESENT_SRC_KHR);
        let barriers = image_barrier(&pending(&[], &[image_write(7)]), &access).unwrap();
        let transition = &barriers.images[0];
        assert_eq!(transition.new_layout, L::PRESENT_SRC_KHR);
        assert_eq!(transition.dst_access_mask, A::NONE);
        assert!(!transition.dst_stage_mask.is_empty());
    }

    #[test]
    fn transition_from_present_has_no_source_access() {
        let access = image(
            7,
            S::TRANSFER,
            A::TRANSFER_WRITE,
            L::PRESENT_SRC_KHR,
            L::GENERAL,
        );
        let barriers = image_barrier(&PendingAccesses::default(), &access).unwrap();
        assert_eq!(barriers.images[0].src_access_mask, A::NONE);
    }

    #[test]
    fn only_images_that_change_layout_are_transitioned() {
        let accesses = [
            image_read(1),
            image(2, S::TRANSFER, A::TRANSFER_WRITE, L::UNDEFINED, L::GENERAL),
        ];
        let barriers =
            compute_barriers(&PendingAccesses::default(), accesses.iter(), iter::empty()).unwrap();
        assert_eq!(barriers.images.len(), 1);
        assert_eq!(barriers.images[0].image, vk::Image::from_raw(2));
    }

    #[test]
    fn flush_without_hazards_only_records_and_needs_no_device() {
        let mut cmd = offline_cmd();
        cmd.flush_pending([image_write(1)].iter(), [write(0..10), read(20..30)].iter());
        assert_eq!(cmd.pending_accesses.buffer_writes.len(), 1);
        assert_eq!(cmd.pending_accesses.buffer_reads.len(), 1);
        assert_eq!(cmd.pending_accesses.image_writes.len(), 1);
    }

    // ---- blits ---------------------------------------------------------------------------

    #[test]
    fn blit_region_spans_offset_to_offset_plus_extent() {
        let (a, b) = (layout(L::GENERAL), layout(L::GENERAL));
        let src = view::<R8G8B8A8Unorm, Sampled>(&a)
            .region(UVec2::new(16, 8))
            .offset(IVec2::new(4, 2));
        let dst = view::<R8G8B8A8Unorm, Sampled>(&b)
            .region(UVec2::new(32, 16))
            .offset(IVec2::new(10, 20));
        let region = blit_region(&src, &dst);

        let corner = |o: vk::Offset3D| (o.x, o.y, o.z);
        assert_eq!(corner(region.src_offsets[0]), (4, 2, 0));
        assert_eq!(corner(region.src_offsets[1]), (20, 10, 1));
        assert_eq!(corner(region.dst_offsets[0]), (10, 20, 0));
        assert_eq!(corner(region.dst_offsets[1]), (42, 36, 1));
        assert_eq!(region.src_subresource.mip_level, 0);
        assert_eq!(
            region.src_subresource.aspect_mask,
            vk::ImageAspectFlags::COLOR
        );
    }

    // ---- raster builder ------------------------------------------------------------------

    #[test]
    fn raster_defaults_to_filled_backface_culled_without_attachments() {
        let mut cmd = offline_cmd();
        let builder = cmd.raster();
        assert!(builder.hash.backface_culling);
        assert!(!builder.hash.wire_frame);
        assert!(builder.hash.color_formats.is_empty());
        assert_eq!(builder.hash.depth_format, vk::Format::UNDEFINED);
        assert_eq!(builder.hash.stencil_format, vk::Format::UNDEFINED);
        assert!(builder.write_depth);
        assert!(builder.hash.depth_write);
    }

    #[test]
    fn raster_state_setters_end_up_in_the_pipeline_key() {
        let mut cmd = offline_cmd();
        let builder = cmd.raster().backface_culling(false).wire_frame(true);
        assert!(!builder.hash.backface_culling);
        assert!(builder.hash.wire_frame);

        let mut cmd2 = offline_cmd();
        let default = cmd2.raster();
        assert_ne!(
            builder.hash, default.hash,
            "different state must not share a pipeline"
        );
    }

    #[test]
    fn color_attachment_records_format_view_and_access() {
        let state = layout(L::UNDEFINED);
        let mut cmd = offline_cmd();
        let builder = cmd.raster().color_attachment(
            view::<R8G8B8A8Unorm, ColorAttachment>(&state),
            Some([0.0, 0.5, 1.0, 1.0]),
        );

        assert_eq!(
            builder.hash.color_formats.as_slice(),
            [vk::Format::R8G8B8A8_UNORM]
        );
        let (attachment, clear) = builder.color_attachments[0];
        assert_eq!(attachment, vk::ImageView::from_raw(2));
        assert_eq!(
            unsafe { clear.unwrap().color.float32 },
            [0.0, 0.5, 1.0, 1.0]
        );

        let access = &builder.color_accesses[0];
        assert_eq!(access.stage, S::COLOR_ATTACHMENT_OUTPUT);
        // Cleared attachments are not read.
        assert_eq!(access.access, A::COLOR_ATTACHMENT_WRITE);
        assert_eq!(
            (access.old_layout, access.layout),
            (L::UNDEFINED, ATTACHMENT_LAYOUT)
        );
    }

    /// An image is only in the attachment layout while it is drawn to: sampling it in
    /// between is an access in `GENERAL`, so the next draw transitions it back.
    #[test]
    fn attachments_are_transitioned_into_the_attachment_layout() {
        let color = layout(L::GENERAL);
        let depth = layout(L::GENERAL);
        let draw = |cmd: &mut CommandBuffer| {
            let builder = cmd
                .raster()
                .color_attachment(view::<R8G8B8A8Unorm, ColorAttachment>(&color), None)
                .depth_attachment(view::<D32Sfloat, DepthAttachment>(&depth), None, true);
            (
                builder.color_accesses[0].old_layout,
                builder.depth_access.as_ref().unwrap().old_layout,
                builder.depth_access.as_ref().unwrap().layout,
            )
        };
        let mut cmd = offline_cmd();
        assert_eq!(draw(&mut cmd), (L::GENERAL, L::GENERAL, ATTACHMENT_LAYOUT));
        // A second draw finds them there.
        assert_eq!(
            draw(&mut cmd),
            (ATTACHMENT_LAYOUT, ATTACHMENT_LAYOUT, ATTACHMENT_LAYOUT)
        );
        let sampled = view::<R8G8B8A8Unorm, ColorAttachment>(&color).access(
            S::FRAGMENT_SHADER,
            A::SHADER_SAMPLED_READ,
            L::GENERAL,
        );
        assert_eq!(sampled.old_layout, ATTACHMENT_LAYOUT);
        assert_eq!(draw(&mut cmd).0, L::GENERAL);
    }

    #[test]
    fn loaded_color_attachment_is_also_read() {
        let state = layout(L::GENERAL);
        let mut cmd = offline_cmd();
        let builder = cmd
            .raster()
            .color_attachment(view::<R8G8B8A8Unorm, ColorAttachment>(&state), None);
        assert!(builder.color_attachments[0].1.is_none());
        assert_eq!(
            builder.color_accesses[0].access,
            A::COLOR_ATTACHMENT_WRITE | A::COLOR_ATTACHMENT_READ
        );
    }

    #[test]
    fn read_only_depth_attachment_tests_without_writing() {
        let state = layout(L::GENERAL);
        let mut cmd = offline_cmd();
        let builder =
            cmd.raster()
                .depth_attachment(view::<D32Sfloat, DepthAttachment>(&state), None, false);

        assert_eq!(builder.hash.depth_format, vk::Format::D32_SFLOAT);
        assert_eq!(builder.hash.stencil_format, vk::Format::UNDEFINED);
        // Depth writes are pipeline state, so read-only draws get their own pipeline.
        assert!(!builder.hash.depth_write);
        assert!(!builder.write_depth);
        assert!(builder.clear_depth.is_none());
        let access = builder.depth_access.as_ref().unwrap();
        assert_eq!(access.access, A::DEPTH_STENCIL_ATTACHMENT_READ);
        assert_eq!(
            access.stage,
            S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS
        );

        let mut cmd2 = offline_cmd();
        let writing =
            cmd2.raster()
                .depth_attachment(view::<D32Sfloat, DepthAttachment>(&state), None, true);
        assert!(writing.hash.depth_write);
        assert_ne!(builder.hash, writing.hash);
    }

    /// Clearing modifies the attachment even if the draw only tests against it.
    #[test]
    fn cleared_read_only_depth_attachment_is_still_a_write_access() {
        let state = layout(L::UNDEFINED);
        let mut cmd = offline_cmd();
        let builder = cmd.raster().depth_attachment(
            view::<D32Sfloat, DepthAttachment>(&state),
            Some([0.25]),
            false,
        );
        assert!(!builder.hash.depth_write);
        assert_eq!(
            unsafe { builder.clear_depth.unwrap().depth_stencil.depth },
            0.25
        );
        assert_eq!(
            builder.depth_access.as_ref().unwrap().access,
            A::DEPTH_STENCIL_ATTACHMENT_READ | A::DEPTH_STENCIL_ATTACHMENT_WRITE
        );
    }

    #[test]
    fn written_depth_attachment_is_a_write_access() {
        let state = layout(L::UNDEFINED);
        let mut cmd = offline_cmd();
        let builder =
            cmd.raster()
                .depth_attachment(view::<D32Sfloat, DepthAttachment>(&state), None, true);
        assert_eq!(
            builder.depth_access.as_ref().unwrap().access,
            A::DEPTH_STENCIL_ATTACHMENT_READ | A::DEPTH_STENCIL_ATTACHMENT_WRITE
        );
    }

    /// Regression: the stencil format was never set, so pipelines for depth-stencil
    /// attachments were created with a mismatching stencil format.
    #[test]
    fn depth_stencil_attachment_sets_the_stencil_format_too() {
        let state = layout(L::UNDEFINED);
        let mut cmd = offline_cmd();
        let builder = cmd.raster().depth_attachment(
            view::<D32SfloatS8Uint, DepthAttachment>(&state),
            Some((0.0, 0)),
            true,
        );
        assert_eq!(builder.hash.depth_format, vk::Format::D32_SFLOAT_S8_UINT);
        assert_eq!(builder.hash.stencil_format, vk::Format::D32_SFLOAT_S8_UINT);
    }

    #[test]
    fn full_extent_dynstates_cover_the_whole_target() {
        let (scissors, viewport) = RasterBuilder::full_extent_dynstates(UVec2::new(640, 480));
        assert_eq!(
            (scissors[0].offset, scissors[0].extent),
            (IVec2::ZERO, UVec2::new(640, 480))
        );
        assert_eq!(
            (viewport.offset, viewport.extent),
            (IVec2::ZERO, UVec2::new(640, 480))
        );
    }

    // ---- layouts that are transmuted or read by the GPU ----------------------------------

    /// `&[Scissor]` is transmuted to `&[vk::Rect2D]` when drawing.
    #[test]
    fn scissor_has_the_layout_of_a_vulkan_rect() {
        assert_eq!(size_of::<Scissor>(), size_of::<vk::Rect2D>());
        assert_eq!(align_of::<Scissor>(), align_of::<vk::Rect2D>());
        let scissor = Scissor {
            offset: IVec2::new(-3, 7),
            extent: UVec2::new(640, 480),
        };
        let rect: vk::Rect2D = unsafe { std::mem::transmute(scissor) };
        assert_eq!((rect.offset.x, rect.offset.y), (-3, 7));
        assert_eq!((rect.extent.width, rect.extent.height), (640, 480));
    }

    #[test]
    fn indirect_commands_match_the_vulkan_structs() {
        assert_eq!(
            size_of::<DrawIndirectCommand>(),
            size_of::<vk::DrawIndirectCommand>()
        );
        assert_eq!(
            size_of::<DispatchIndirectCommand>(),
            size_of::<vk::DispatchIndirectCommand>()
        );

        let draw = DrawIndirectCommand {
            vertex_count: 1,
            instance_count: 2,
            first_vertex: 3,
            first_instance: 4,
        };
        let raw: vk::DrawIndirectCommand = unsafe { std::mem::transmute(draw) };
        assert_eq!(
            (
                raw.vertex_count,
                raw.instance_count,
                raw.first_vertex,
                raw.first_instance
            ),
            (1, 2, 3, 4)
        );
        let dispatch = DispatchIndirectCommand { x: 5, y: 6, z: 7 };
        let raw: vk::DispatchIndirectCommand = unsafe { std::mem::transmute(dispatch) };
        assert_eq!((raw.x, raw.y, raw.z), (5, 6, 7));
    }

    // ---- shader stages and hot reload ----------------------------------------------------

    #[test]
    fn shader_stage_uses_the_nul_terminated_entry_name() {
        let stage = make_shader_stage(
            "main_vs\0",
            vk::ShaderStageFlags::VERTEX,
            vk::ShaderModule::null(),
        );
        assert_eq!(stage.stage, vk::ShaderStageFlags::VERTEX);
        assert_eq!(unsafe { CStr::from_ptr(stage.p_name) }, c"main_vs");
    }

    #[test]
    #[should_panic]
    fn shader_stage_rejects_entry_names_without_terminator() {
        make_shader_stage(
            "main_vs",
            vk::ShaderStageFlags::VERTEX,
            vk::ShaderModule::null(),
        );
    }

    fn pass(source: &'static str) -> PassEntry {
        PassEntry {
            path: "",
            source,
            kind: PassKind::Compute { entry: "main\0" },
            index: 0,
        }
    }

    #[test]
    fn changed_pass_sources_reload_only_those_passes() {
        let passes = [
            pass("passes/a.slang"),
            pass("passes/b.slang"),
            pass("passes/c.slang"),
        ];
        let changed = [
            PathBuf::from("/repo/shaders/passes/c.slang"),
            PathBuf::from("/repo/shaders/passes/a.slang"),
        ];
        assert_eq!(passes_to_reload(&changed, &passes), [0, 2]);
        assert!(passes_to_reload(&[], &passes).is_empty());
    }

    #[test]
    fn a_changed_include_reloads_every_pass() {
        let passes = [pass("passes/a.slang"), pass("passes/b.slang")];
        let changed = [
            PathBuf::from("/repo/shaders/passes/a.slang"),
            PathBuf::from("/repo/shaders/include/math.slang"),
        ];
        assert_eq!(passes_to_reload(&changed, &passes), [0, 1]);
    }

    #[test]
    fn pass_sources_match_on_whole_path_components() {
        // `xa.slang` is not pass `a.slang`; it counts as an unknown file, i.e. an include.
        let passes = [pass("passes/a.slang"), pass("passes/b.slang")];
        let changed = [PathBuf::from("/repo/shaders/passes/xa.slang")];
        assert_eq!(passes_to_reload(&changed, &passes), [0, 1]);
    }

    fn watch(events: Vec<notify::Result<notify::Event>>) -> Vec<PathBuf> {
        let (changed, received) = mpsc::channel();
        let mut watcher = FileWatcher { changed };
        for event in events {
            watcher.handle_event(event);
        }
        drop(watcher);
        received.iter().collect()
    }

    fn event(kind: notify::EventKind, paths: &[&str]) -> notify::Result<notify::Event> {
        let mut event = notify::Event::new(kind);
        for path in paths {
            event = event.add_path(PathBuf::from(path));
        }
        Ok(event)
    }

    #[test]
    fn watcher_forwards_modified_and_created_slang_files() {
        use notify::{
            EventKind,
            event::{CreateKind, ModifyKind},
        };
        let paths = watch(vec![
            event(
                EventKind::Modify(ModifyKind::Any),
                &["/s/passes/a.slang", "/s/notes.txt"],
            ),
            event(
                EventKind::Create(CreateKind::File),
                &["/s/include/new.slang"],
            ),
        ]);
        assert_eq!(
            paths,
            [
                PathBuf::from("/s/passes/a.slang"),
                PathBuf::from("/s/include/new.slang")
            ]
        );
    }

    #[test]
    fn watcher_ignores_other_events_and_errors() {
        use notify::{
            EventKind,
            event::{AccessKind, RemoveKind},
        };
        let paths = watch(vec![
            event(EventKind::Remove(RemoveKind::File), &["/s/passes/a.slang"]),
            event(EventKind::Access(AccessKind::Any), &["/s/passes/a.slang"]),
            Err(notify::Error::generic("watch failed")),
        ]);
        assert!(paths.is_empty());
    }

    #[test]
    fn pending_reloads_are_empty_before_pipelines_exist() {
        assert!(apply_pending_reloads().is_empty());
    }

    // ---- generated bindings --------------------------------------------------------------

    #[test]
    fn pass_map_entries_are_consistent() {
        let (mut compute, mut raster, mut raytracing) = (Vec::new(), Vec::new(), Vec::new());
        for pass in PASS_MAP.iter() {
            let names: Vec<&str> = match pass.kind {
                PassKind::Compute { entry } => {
                    compute.push(pass.index);
                    vec![entry]
                }
                PassKind::RasterVertex { vertex, fragment } => {
                    raster.push(pass.index);
                    vec![vertex, fragment]
                }
                PassKind::RasterMesh {
                    amp,
                    mesh,
                    fragment,
                } => {
                    raster.push(pass.index);
                    amp.into_iter().chain([mesh, fragment]).collect()
                }
                PassKind::RayTracing {
                    ray_gen,
                    ray_any,
                    ray_closest,
                } => {
                    raytracing.push(pass.index);
                    vec![ray_gen, ray_any, ray_closest]
                }
            };
            for name in names {
                // `make_shader_stage` needs exactly one NUL, at the end.
                assert!(
                    CStr::from_bytes_with_nul(name.as_bytes()).is_ok(),
                    "{name:?} in {}",
                    pass.source
                );
            }
        }
        // Each kind indexes its own pipeline array densely: 0..NUM_*.
        assert_eq!(compute, (0..NUM_COMPUTE_PIPELINES).collect::<Vec<_>>());
        assert_eq!(raster, (0..NUM_RASTER_PIPELINES).collect::<Vec<_>>());
        assert_eq!(
            raytracing,
            (0..NUM_RAY_TRACING_PIPELINES).collect::<Vec<_>>()
        );
    }

    #[test]
    fn every_pass_has_compiled_spirv() {
        for pass in PASS_MAP.iter() {
            let bytes = read_pass_spirv(pass.path);
            let words = ash::util::read_spv(&mut std::io::Cursor::new(&bytes))
                .unwrap_or_else(|e| panic!("{} is not valid SPIR-V: {e}", pass.path));
            assert_eq!(
                words[0], 0x0723_0203,
                "{} lacks the SPIR-V magic number",
                pass.path
            );
        }
    }

    #[cfg(feature = "test-passes")]
    mod generated {
        use super::*;
        use crate::bindings::{TestComputeBuffer, TestComputeImage, TestRaster, TestVertex};
        use crate::buffer::usage::Storage as StorageBuffer;
        use crate::image::usage::Storage as StorageImage;

        fn slice<T: Pod>(gpu_ptr: u64, len: u64) -> BufferSlice<'static, T, StorageBuffer> {
            BufferSlice {
                handle: vk::Buffer::null(),
                size: len * size_of::<T>() as u64,
                cpu_ptr: 0,
                gpu_ptr,
                base_address: gpu_ptr,
                _marker: PhantomData,
                _usage: PhantomData,
                _lifetime: PhantomData,
            }
        }

        #[test]
        fn test_passes_are_registered_with_their_kind() {
            assert!(matches!(
                PASS_MAP[TestComputeBuffer::PASS_INDEX].kind,
                PassKind::Compute {
                    entry: "test_compute_buffer\0"
                }
            ));
            assert!(matches!(
                PASS_MAP[TestRaster::PASS_INDEX].kind,
                PassKind::RasterVertex {
                    vertex: "test_raster_vertex\0",
                    fragment: "test_raster_fragment\0"
                }
            ));
            assert_eq!(
                PASS_MAP[TestRaster::PASS_INDEX].source,
                "shaders/test_raster.slang"
            );
        }

        #[test]
        fn buffer_fields_become_device_addresses_and_accesses() {
            let bindings =
                TestComputeBuffer::new(slice::<u32>(0x1000, 16), slice::<u32>(0x2000, 16), 16, 3);
            assert_eq!(bindings.gpu_bindings.src, 0x1000);
            assert_eq!(bindings.gpu_bindings.dst, 0x2000);
            assert_eq!(
                (
                    bindings.gpu_bindings.count,
                    bindings.gpu_bindings.multiplier
                ),
                (16, 3)
            );

            let [src, dst] = &bindings.buffers;
            // `Buf` is read-only, `MutBuf` is read-write.
            assert_eq!(src.access, A::SHADER_STORAGE_READ);
            assert_eq!(dst.access, A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE);
            assert_eq!(src.stage, S::COMPUTE_SHADER);
            assert_eq!(src.range, (0x1000u64..0x1040).into());
            assert_eq!(dst.range, (0x2000u64..0x2040).into());
        }

        #[test]
        fn raster_passes_access_their_buffers_in_both_shader_stages() {
            let bindings =
                TestRaster::new(slice::<TestVertex>(0x4000, 3), glam::Vec2::new(0.5, 0.0));
            assert_eq!(bindings.gpu_bindings.vertices, 0x4000);
            assert_eq!(
                bindings.buffers[0].stage,
                S::VERTEX_SHADER | S::FRAGMENT_SHADER
            );
            assert_eq!(bindings.buffers[0].access, A::SHADER_STORAGE_READ);
        }

        #[test]
        fn registered_accesses_are_appended_with_the_pass_stages() {
            let state = layout(L::UNDEFINED);
            let extra = view::<R8G8B8A8Unorm, StorageImage>(&state);
            let bindings =
                TestRaster::new(slice::<TestVertex>(0x4000, 3), glam::Vec2::new(0.5, 0.0))
                    .storage_read(extra)
                    .buffer_read(slice::<u32>(0x5000, 4))
                    .buffer_write(slice::<u32>(0x6000, 4));

            let stages = S::VERTEX_SHADER | S::FRAGMENT_SHADER;
            let [image] = &bindings.images;
            assert_eq!(image.access, A::SHADER_STORAGE_READ);
            assert_eq!(image.stage, stages);
            assert_eq!((image.old_layout, image.layout), (L::UNDEFINED, L::GENERAL));

            // The pass's own buffer stays first, registered ones follow in call order.
            let [vertices, read, write] = &bindings.buffers;
            assert_eq!(vertices.range.start, 0x4000);
            assert_eq!(read.range, (0x5000u64..0x5010).into());
            assert_eq!(read.access, A::SHADER_STORAGE_READ);
            assert_eq!(write.range.start, 0x6000);
            assert_eq!(
                write.access,
                A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE
            );
            assert_eq!((read.stage, write.stage), (stages, stages));
            assert_eq!(bindings.gpu_bindings.vertices, 0x4000);
        }

        #[test]
        fn image_fields_become_bindless_handles_and_general_layout_accesses() {
            let state = layout(L::UNDEFINED);
            let mut target = view::<R8G8B8A8Unorm, StorageImage>(&state);
            target.handle.descriptor_index_set1 = 5;
            let bindings =
                TestComputeImage::new(target, glam::Vec4::ONE, UVec2::ZERO, UVec2::new(8, 8));
            assert_eq!(bindings.gpu_bindings.target.descriptor_index_set1, 5);
            let [access] = &bindings.images;
            assert_eq!(
                access.access,
                A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE
            );
            assert_eq!(access.stage, S::COMPUTE_SHADER);
            assert_eq!(
                (access.old_layout, access.layout),
                (L::UNDEFINED, L::GENERAL)
            );
        }
    }
}

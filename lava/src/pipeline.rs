//! Pass pipelines: the pass table's entries, the cache of every pass's pipelines per specialization and raster state, and their creation; with `hot-reload`, a second cache whose reloads replace pipelines that command buffers keep alive
#[cfg(feature = "hot-reload")]
use std::sync::Arc;
use std::{
    collections::HashMap,
    ffi::CStr,
    hash::Hash,
    sync::{OnceLock, RwLock},
};

use ash::vk;
use smallvec::{SmallVec, smallvec};

#[cfg(feature = "hot-reload")]
use crate::vkobjects::rt_pipeline::ShaderBindingTable;
use crate::{
    bindings::{
        MAX_SPECIALIZATION_CONSTANTS_SIZE, NUM_COMPUTE_PIPELINES, NUM_RASTER_PIPELINES,
        NUM_RAY_TRACING_PIPELINES, NUM_SPECIALIZED_COMPUTE_PIPELINES,
        NUM_SPECIALIZED_RAY_TRACING_PIPELINES, PASS_MAP,
    },
    bindless::Bindless,
    command_buffer::Blend,
    error::Result,
    state::{Ctx, Functions},
    vkobjects::rt_pipeline::{
        RayTracingShaderCreateInfo, RayTracingShaderGroup, RaytracingPipeline,
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassKind {
    Compute {
        entry: &'static str,
    },
    /// One raygen group, a miss group if there is a miss shader, and a triangle hit group
    /// if there is a closest-hit or any-hit shader.
    RayTracing {
        raygen: &'static str,
        miss: Option<&'static str>,
        closest_hit: Option<&'static str>,
        any_hit: Option<&'static str>,
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
    /// The generated struct's name, e.g. `"BvhCull"`; also the GPU scope of a compute or ray
    /// tracing pass.
    pub name: &'static str,
    /// The SPIR-V `build.rs` compiled, read at start; reloads replace it.
    #[cfg(feature = "hot-reload")]
    pub path: &'static str,
    /// The SPIR-V `build.rs` compiled, built into the binary.
    #[cfg(not(feature = "hot-reload"))]
    pub spirv: &'static [u8],
    pub source: &'static str,
    pub kind: PassKind,
    /// Index into the pipelines of its kind: the specialized ones if it has `constants`.
    pub index: usize,
    /// Where each specialization constant sits in `BindingOutput::constants`.
    pub constants: &'static [vk::SpecializationMapEntry],
    pub constants_size: usize,
    /// The names of its shader clock regions (`CLOCK_REGION` in `profile.slang`) in slot
    /// order, whose hashes its `constant_bindings` sets. New names need a `cargo build`.
    pub regions: &'static [&'static str],
    /// `slang_compile::layout_hash` of the push constants, specialization constants and
    /// entry points the bindings were generated from; a hot reload refuses a pass whose
    /// hash changed, since the Rust side can't follow.
    pub layout: u64,
}

impl PassEntry {
    #[cfg(not(feature = "hot-reload"))]
    fn spirv(&self) -> &'static [u8] {
        self.spirv
    }

    #[cfg(feature = "hot-reload")]
    fn spirv(&self) -> Vec<u8> {
        std::fs::read(self.path).unwrap_or_else(|e| panic!("failed to read {}: {e}", self.path))
    }
}

/// The specialization constants of a pipeline, as `BindingOutput::constants` holds them.
pub(crate) type Constants = [u8; MAX_SPECIALIZATION_CONSTANTS_SIZE];

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub(crate) struct RasterHash {
    pub(crate) color_formats: SmallVec<[vk::Format; 4]>,
    pub(crate) depth_format: vk::Format,
    pub(crate) stencil_format: vk::Format,
    pub(crate) blends: SmallVec<[Blend; 4]>,
    pub(crate) depth_write: bool,
    pub(crate) backface_culling: bool,
    pub(crate) wire_frame: bool,
    /// Zeros for a pass without specialization constants.
    pub(crate) constants: Constants,
}

static PIPELINES: OnceLock<PipelineManager> = OnceLock::new();

pub(crate) fn pipelines() -> &'static PipelineManager {
    PIPELINES
        .get()
        .expect("pipelines were not initialized; call `lava::init()` first")
}

/// Creates every pass's shader module and its pipelines without specialization constants;
/// with `hot-reload`, also starts watching the shaders.
pub(crate) fn init() {
    let manager = PIPELINES.get_or_init(PipelineManager::new);
    #[cfg(feature = "hot-reload")]
    crate::hot_reload::watch(manager);
    #[cfg(not(feature = "hot-reload"))]
    let _ = manager;
}

/// The specialization constants of a pass without them.
const NO_CONSTANTS: Constants = [0; MAX_SPECIALIZATION_CONSTANTS_SIZE];

/// Every pass's pipelines: compute and ray tracing passes without specialization constants
/// have one, made at start; the others one per specialization (and raster state), made at
/// first use. Its pipelines never change.
#[cfg(not(feature = "hot-reload"))]
pub(crate) struct PipelineManager {
    shader_modules: Box<[vk::ShaderModule]>,
    compute_pipelines: RwLock<[vk::Pipeline; NUM_COMPUTE_PIPELINES]>,
    specialized_compute_pipelines:
        RwLock<[HashMap<Constants, vk::Pipeline>; NUM_SPECIALIZED_COMPUTE_PIPELINES]>,
    raytracing_pipelines: RwLock<[Option<RaytracingPipeline>; NUM_RAY_TRACING_PIPELINES]>,
    specialized_raytracing_pipelines:
        RwLock<[HashMap<Constants, RaytracingPipeline>; NUM_SPECIALIZED_RAY_TRACING_PIPELINES]>,
    raster_pipelines: RwLock<[HashMap<RasterHash, vk::Pipeline>; NUM_RASTER_PIPELINES]>,
}

#[cfg(not(feature = "hot-reload"))]
impl PipelineManager {
    fn new() -> Self {
        let modules: Box<[_]> = PASS_MAP
            .iter()
            .map(|pass| must(pass, create_module(pass.spirv())))
            .collect();
        Self {
            compute_pipelines: RwLock::new(create_unspecialized(false, |pass_index, pass| {
                must(
                    pass,
                    create_compute_pipeline(modules[pass_index], pass, &NO_CONSTANTS),
                )
            })),
            specialized_compute_pipelines: RwLock::new(std::array::from_fn(|_| HashMap::new())),
            raytracing_pipelines: RwLock::new(create_unspecialized(true, |pass_index, pass| {
                Some(must(
                    pass,
                    create_raytracing_pipeline(modules[pass_index], pass, &NO_CONSTANTS),
                ))
            })),
            specialized_raytracing_pipelines: RwLock::new(std::array::from_fn(|_| HashMap::new())),
            raster_pipelines: RwLock::new(std::array::from_fn(|_| HashMap::new())),
            shader_modules: modules,
        }
    }

    /// The compute pipeline of `pass_index` with `constants`.
    pub(crate) fn compute(&self, pass_index: usize, constants: &Constants) -> vk::Pipeline {
        let pass = &PASS_MAP[pass_index];
        if pass.constants.is_empty() {
            return self.compute_pipelines.read().unwrap()[pass.index];
        }
        let module = self.shader_modules[pass_index];
        cached(&self.specialized_compute_pipelines, pass, constants, || {
            must(pass, create_compute_pipeline(module, pass, constants))
        })
    }

    /// Calls `record` with the ray tracing pipeline of `pass_index` with `constants`, while
    /// the cache stays locked.
    pub(crate) fn ray_tracing<R>(
        &self,
        pass_index: usize,
        constants: &Constants,
        record: impl FnOnce(&RaytracingPipeline) -> R,
    ) -> R {
        let pass = &PASS_MAP[pass_index];
        if pass.constants.is_empty() {
            return record(
                self.raytracing_pipelines.read().unwrap()[pass.index]
                    .as_ref()
                    .unwrap(),
            );
        }
        let maps = &self.specialized_raytracing_pipelines;
        if !maps.read().unwrap()[pass.index].contains_key(constants) {
            let module = self.shader_modules[pass_index];
            let map = &mut maps.write().unwrap()[pass.index];
            let cached = map.len();
            map.entry(*constants).or_insert_with(|| {
                note_new_pipeline(pass, cached);
                must(pass, create_raytracing_pipeline(module, pass, constants))
            });
        }
        record(&maps.read().unwrap()[pass.index][constants])
    }

    /// The pipeline of raster pass `pass_index` for `hash`.
    pub(crate) fn raster(&self, hash: &RasterHash, pass_index: usize) -> vk::Pipeline {
        let module = self.shader_modules[pass_index];
        let pass = &PASS_MAP[pass_index];
        cached(&self.raster_pipelines, pass, hash, || {
            must(pass, create_raster_pipeline(module, hash, pass_index))
        })
    }
}

/// A pipeline and, for a ray tracing pass, its shader binding table. With `hot-reload`,
/// every command buffer keeps the ones it records (`CommandBuffer::keep`) until its frame
/// slot is begun again, so a pipeline a reload replaced is destroyed once no pending frame
/// uses it.
#[cfg(feature = "hot-reload")]
pub(crate) struct GenericPipeline {
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) sbt: Option<ShaderBindingTable>,
}

#[cfg(feature = "hot-reload")]
impl Drop for GenericPipeline {
    fn drop(&mut self) {
        unsafe { Ctx::device().destroy_pipeline(self.pipeline, None) };
    }
}

#[cfg(feature = "hot-reload")]
fn generic(pipeline: vk::Pipeline) -> Arc<GenericPipeline> {
    Arc::new(GenericPipeline {
        pipeline,
        sbt: None,
    })
}

#[cfg(feature = "hot-reload")]
fn generic_ray_tracing(pipeline: RaytracingPipeline) -> Arc<GenericPipeline> {
    Arc::new(GenericPipeline {
        pipeline: pipeline.pipeline,
        sbt: Some(pipeline.sbt),
    })
}

/// Like the manager without `hot-reload`, but a reload replaces a pass's module and
/// pipelines (`reload_pass`).
#[cfg(feature = "hot-reload")]
pub(crate) struct PipelineManager {
    shader_modules: RwLock<Vec<vk::ShaderModule>>,
    compute_pipelines: RwLock<[Arc<GenericPipeline>; NUM_COMPUTE_PIPELINES]>,
    specialized_compute_pipelines:
        RwLock<[HashMap<Constants, Arc<GenericPipeline>>; NUM_SPECIALIZED_COMPUTE_PIPELINES]>,
    raytracing_pipelines: RwLock<[Option<Arc<GenericPipeline>>; NUM_RAY_TRACING_PIPELINES]>,
    specialized_raytracing_pipelines:
        RwLock<[HashMap<Constants, Arc<GenericPipeline>>; NUM_SPECIALIZED_RAY_TRACING_PIPELINES]>,
    raster_pipelines: RwLock<[HashMap<RasterHash, Arc<GenericPipeline>>; NUM_RASTER_PIPELINES]>,
}

#[cfg(feature = "hot-reload")]
impl PipelineManager {
    fn new() -> Self {
        let modules: Vec<_> = PASS_MAP
            .iter()
            .map(|pass| must(pass, create_module(&pass.spirv())))
            .collect();
        Self {
            compute_pipelines: RwLock::new(create_unspecialized(false, |pass_index, pass| {
                generic(must(
                    pass,
                    create_compute_pipeline(modules[pass_index], pass, &NO_CONSTANTS),
                ))
            })),
            specialized_compute_pipelines: RwLock::new(std::array::from_fn(|_| HashMap::new())),
            raytracing_pipelines: RwLock::new(create_unspecialized(true, |pass_index, pass| {
                let pipeline = create_raytracing_pipeline(modules[pass_index], pass, &NO_CONSTANTS);
                Some(generic_ray_tracing(must(pass, pipeline)))
            })),
            specialized_raytracing_pipelines: RwLock::new(std::array::from_fn(|_| HashMap::new())),
            raster_pipelines: RwLock::new(std::array::from_fn(|_| HashMap::new())),
            shader_modules: RwLock::new(modules),
        }
    }

    /// The compute pipeline of `pass_index` with `constants`.
    pub(crate) fn compute(&self, pass_index: usize, constants: &Constants) -> Arc<GenericPipeline> {
        let pass = &PASS_MAP[pass_index];
        if pass.constants.is_empty() {
            return self.compute_pipelines.read().unwrap()[pass.index].clone();
        }
        // Held until the pipeline exists, so a reload can't destroy the module before.
        let modules = self.shader_modules.read().unwrap();
        let module = modules[pass_index];
        cached(&self.specialized_compute_pipelines, pass, constants, || {
            generic(must(pass, create_compute_pipeline(module, pass, constants)))
        })
    }

    /// The ray tracing pipeline of `pass_index` with `constants`.
    pub(crate) fn ray_tracing(
        &self,
        pass_index: usize,
        constants: &Constants,
    ) -> Arc<GenericPipeline> {
        let pass = &PASS_MAP[pass_index];
        if pass.constants.is_empty() {
            return self.raytracing_pipelines.read().unwrap()[pass.index]
                .clone()
                .unwrap();
        }
        let modules = self.shader_modules.read().unwrap();
        let module = modules[pass_index];
        cached(
            &self.specialized_raytracing_pipelines,
            pass,
            constants,
            || {
                generic_ray_tracing(must(
                    pass,
                    create_raytracing_pipeline(module, pass, constants),
                ))
            },
        )
    }

    /// The pipeline of raster pass `pass_index` for `hash`.
    pub(crate) fn raster(&self, hash: &RasterHash, pass_index: usize) -> Arc<GenericPipeline> {
        let modules = self.shader_modules.read().unwrap();
        let module = modules[pass_index];
        let pass = &PASS_MAP[pass_index];
        cached(&self.raster_pipelines, pass, hash, || {
            generic(must(pass, create_raster_pipeline(module, hash, pass_index)))
        })
    }

    /// Replaces the pass's module and every pipeline made from it. A replaced pipeline is
    /// destroyed once no pending command buffer uses it. If the driver rejects the module or
    /// one of the pipelines, nothing is replaced: every new pipeline is made before any is
    /// swapped in, and no lock is held while one fails.
    pub(crate) fn reload_pass(&self, pass_index: usize, spirv: &[u8]) -> Result<()> {
        let pass = &PASS_MAP[pass_index];
        let module = create_module(spirv)?;
        let destroy_module = |module| unsafe { Ctx::device().destroy_shader_module(module, None) };
        let unspecialized = pass.constants.is_empty();

        // Every pipeline of the pass with the new module, then the swap.
        let result: Result<()> = (|| {
            match pass.kind {
                PassKind::Compute { .. } if unspecialized => {
                    let pipeline = generic(create_compute_pipeline(module, pass, &NO_CONSTANTS)?);
                    self.compute_pipelines.write().unwrap()[pass.index] = pipeline;
                }
                PassKind::Compute { .. } => {
                    let maps = &self.specialized_compute_pipelines;
                    let keys: Vec<Constants> =
                        maps.read().unwrap()[pass.index].keys().copied().collect();
                    let mut new = Vec::with_capacity(keys.len());
                    for constants in &keys {
                        new.push(generic(create_compute_pipeline(module, pass, constants)?));
                    }
                    let map = &mut maps.write().unwrap()[pass.index];
                    map.extend(keys.into_iter().zip(new));
                }
                PassKind::RayTracing { .. } if unspecialized => {
                    let pipeline = create_raytracing_pipeline(module, pass, &NO_CONSTANTS)?;
                    self.raytracing_pipelines.write().unwrap()[pass.index] =
                        Some(generic_ray_tracing(pipeline));
                }
                PassKind::RayTracing { .. } => {
                    let maps = &self.specialized_raytracing_pipelines;
                    let keys: Vec<Constants> =
                        maps.read().unwrap()[pass.index].keys().copied().collect();
                    let mut new = Vec::with_capacity(keys.len());
                    for constants in &keys {
                        let pipeline = create_raytracing_pipeline(module, pass, constants)?;
                        new.push(generic_ray_tracing(pipeline));
                    }
                    let map = &mut maps.write().unwrap()[pass.index];
                    map.extend(keys.into_iter().zip(new));
                }
                PassKind::RasterVertex { .. } | PassKind::RasterMesh { .. } => {
                    let maps = &self.raster_pipelines;
                    let keys: Vec<RasterHash> =
                        maps.read().unwrap()[pass.index].keys().cloned().collect();
                    let mut new = Vec::with_capacity(keys.len());
                    for hash in &keys {
                        new.push(generic(create_raster_pipeline(module, hash, pass_index)?));
                    }
                    let map = &mut maps.write().unwrap()[pass.index];
                    map.extend(keys.into_iter().zip(new));
                }
            }
            Ok(())
        })();
        if let Err(err) = result {
            destroy_module(module);
            return Err(err);
        }
        let old = std::mem::replace(
            &mut self.shader_modules.write().unwrap()[pass_index],
            module,
        );
        // Pipelines don't keep their module alive, so it can go as soon as it is unused.
        destroy_module(old);
        Ok(())
    }
}

/// A pipeline lava can't do without: created at start or at a pass's first use, so a
/// driver that rejects it ends the program with the pass's name.
fn must<T>(pass: &PassEntry, result: Result<T>) -> T {
    result.unwrap_or_else(|err| panic!("pass `{}`: {err}", pass.name))
}

/// `create` called for every compute (or ray tracing) pass without specialization constants,
/// as their array: `build.rs` numbers each array's passes in `PASS_MAP` order.
fn create_unspecialized<T, const N: usize>(
    ray_tracing: bool,
    mut create: impl FnMut(usize, &'static PassEntry) -> T,
) -> [T; N] {
    let pipelines: Vec<T> = PASS_MAP
        .iter()
        .enumerate()
        .filter(|(_, pass)| {
            let kind_matches = match pass.kind {
                PassKind::Compute { .. } => !ray_tracing,
                PassKind::RayTracing { .. } => ray_tracing,
                _ => false,
            };
            kind_matches && pass.constants.is_empty()
        })
        .map(|(pass_index, pass)| create(pass_index, pass))
        .collect();
    match pipelines.try_into() {
        Ok(pipelines) => pipelines,
        Err(_) => unreachable!("the generated pipeline counts match PASS_MAP"),
    }
}

/// A pass whose pipeline cache grows this large gets one warning: its specialization constants
/// (or raster states) change too often, so they should be push constants.
const MANY_PIPELINES: usize = 16;

/// The pipeline of `key` in the pass's map, made by `create` at first use.
fn cached<K: Hash + Eq + Clone, V: Clone, const N: usize>(
    maps: &RwLock<[HashMap<K, V>; N]>,
    pass: &PassEntry,
    key: &K,
    create: impl FnOnce() -> V,
) -> V {
    if let Some(pipeline) = maps.read().unwrap()[pass.index].get(key) {
        return pipeline.clone();
    }
    let map = &mut maps.write().unwrap()[pass.index];
    let cached = map.len();
    let pipeline = map.entry(key.clone()).or_insert_with(|| {
        note_new_pipeline(pass, cached);
        create()
    });
    pipeline.clone()
}

fn note_new_pipeline(pass: &PassEntry, cached: usize) {
    if cached + 1 == MANY_PIPELINES {
        tracing::warn!(
            "pass `{}` now has {MANY_PIPELINES} pipelines: its specialization constants change \
             too often, pass those values as push constants",
            pass.name
        );
    }
}

fn create_module(bytes: &[u8]) -> Result<vk::ShaderModule> {
    let decoded_code = ash::util::read_spv(&mut std::io::Cursor::new(bytes))?;
    let create_info = vk::ShaderModuleCreateInfo::default().code(&decoded_code);
    Ok(unsafe { Ctx::device().create_shader_module(&create_info, None)? })
}

/// `pass.constants` places the pass's specialization constants in `constants`; a pass without
/// them has none.
fn create_compute_pipeline(
    module: vk::ShaderModule,
    pass: &PassEntry,
    constants: &Constants,
) -> Result<vk::Pipeline> {
    let PassKind::Compute { entry } = pass.kind else {
        unreachable!("{} is a compute pass", pass.name)
    };
    let info = vk::SpecializationInfo::default()
        .map_entries(pass.constants)
        .data(&constants[..pass.constants_size]);
    let stage = make_shader_stage(entry, vk::ShaderStageFlags::COMPUTE, module, &info);
    let create_info = vk::ComputePipelineCreateInfo::default()
        .layout(Bindless::layout())
        .stage(stage);
    let pipeline = unsafe {
        Ctx::device()
            .create_compute_pipelines(vk::PipelineCache::null(), &[create_info], None)
            .map_err(|(_, err)| err)?
    }[0];
    Functions::set_debug_name(entry, pipeline);
    Ok(pipeline)
}

fn create_raytracing_pipeline(
    module: vk::ShaderModule,
    pass: &PassEntry,
    constants: &Constants,
) -> Result<RaytracingPipeline> {
    let PassKind::RayTracing {
        raygen,
        miss,
        closest_hit,
        any_hit,
    } = pass.kind
    else {
        unreachable!("{} is a ray tracing pass", pass.name)
    };
    let info = vk::SpecializationInfo::default()
        .map_entries(pass.constants)
        .data(&constants[..pass.constants_size]);
    let stage = |entry, flags| make_shader_stage(entry, flags, module, &info);
    let raygen = [stage(raygen, vk::ShaderStageFlags::RAYGEN_KHR)];
    let miss = miss.map(|miss| [stage(miss, vk::ShaderStageFlags::MISS_KHR)]);
    // A hit group's stages: closest hit first, then any hit (see `RaytracingPipeline::new`).
    let hit: SmallVec<[_; 2]> = closest_hit
        .map(|entry| stage(entry, vk::ShaderStageFlags::CLOSEST_HIT_KHR))
        .into_iter()
        .chain(any_hit.map(|entry| stage(entry, vk::ShaderStageFlags::ANY_HIT_KHR)))
        .collect();
    let mut groups = vec![RayTracingShaderCreateInfo {
        stages: &raygen,
        group: RayTracingShaderGroup::RayGen,
    }];
    if let Some(miss) = &miss {
        groups.push(RayTracingShaderCreateInfo {
            stages: miss,
            group: RayTracingShaderGroup::Miss,
        });
    }
    if !hit.is_empty() {
        groups.push(RayTracingShaderCreateInfo {
            stages: &hit,
            group: RayTracingShaderGroup::Hit,
        });
    }
    RaytracingPipeline::new(Bindless::layout(), &groups)
}

fn make_shader_stage<'a>(
    entry: &'a str,
    stage: vk::ShaderStageFlags,
    module: vk::ShaderModule,
    constants: &'a vk::SpecializationInfo<'a>,
) -> vk::PipelineShaderStageCreateInfo<'a> {
    vk::PipelineShaderStageCreateInfo::default()
        .stage(stage)
        .module(module)
        .name(CStr::from_bytes_with_nul(entry.as_bytes()).unwrap())
        .specialization_info(constants)
}

fn create_raster_pipeline(
    module: vk::ShaderModule,
    hash: &RasterHash,
    pass_index: usize,
) -> Result<vk::Pipeline> {
    let _span = tracing::info_span!("create raster pipeline").entered();

    let pass = &PASS_MAP[pass_index];
    let c = vk::SpecializationInfo::default()
        .map_entries(pass.constants)
        .data(&hash.constants[..pass.constants_size]);
    let stages = match pass.kind {
        PassKind::RasterVertex { vertex, fragment } => {
            smallvec![
                make_shader_stage(vertex, vk::ShaderStageFlags::VERTEX, module, &c),
                make_shader_stage(fragment, vk::ShaderStageFlags::FRAGMENT, module, &c),
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
                    &c,
                ));
            }
            stages.push(make_shader_stage(
                mesh,
                vk::ShaderStageFlags::MESH_EXT,
                module,
                &c,
            ));
            stages.push(make_shader_stage(
                fragment,
                vk::ShaderStageFlags::FRAGMENT,
                module,
                &c,
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
        .blends
        .iter()
        .map(|blend| {
            use vk::BlendFactor as F;
            let unblended = vk::PipelineColorBlendAttachmentState::default();
            let (src_color, dst_color, src_alpha, dst_alpha) = match blend {
                Blend::Replace => return unblended.color_write_mask(vk::ColorComponentFlags::RGBA),
                Blend::Skip => return unblended,
                Blend::Alpha => (F::SRC_ALPHA, F::ONE_MINUS_SRC_ALPHA, F::ONE, F::ZERO),
                Blend::Add => (F::ONE, F::ONE, F::ONE, F::ONE),
                Blend::Attenuate => (
                    F::ZERO,
                    F::ONE_MINUS_SRC_COLOR,
                    F::ZERO,
                    F::ONE_MINUS_SRC_ALPHA,
                ),
            };
            vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(true)
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .src_color_blend_factor(src_color)
                .dst_color_blend_factor(dst_color)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(src_alpha)
                .dst_alpha_blend_factor(dst_alpha)
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
            .map_err(|(_, err)| err)?
    }[0];
    Ok(pipeline)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- shader stages -------------------------------------------------------------------

    #[test]
    fn shader_stage_uses_the_nul_terminated_entry_name() {
        let constants = vk::SpecializationInfo::default();
        let stage = make_shader_stage(
            "main_vs\0",
            vk::ShaderStageFlags::VERTEX,
            vk::ShaderModule::null(),
            &constants,
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
            &vk::SpecializationInfo::default(),
        );
    }

    // ---- generated bindings --------------------------------------------------------------

    #[test]
    fn pass_map_entries_are_consistent() {
        let (mut compute, mut raster, mut raytracing) = (Vec::new(), Vec::new(), Vec::new());
        let (mut specialized_compute, mut specialized_raytracing) = (Vec::new(), Vec::new());
        for pass in PASS_MAP.iter() {
            let specialized = !pass.constants.is_empty();
            // The constants fill their struct exactly, and fit every pipeline key.
            let end = pass.constants.iter().map(|c| c.offset as usize + c.size);
            assert_eq!(end.max().unwrap_or(0), pass.constants_size, "{}", pass.name);
            assert!(pass.constants_size <= MAX_SPECIALIZATION_CONSTANTS_SIZE);
            let names: Vec<&str> = match pass.kind {
                PassKind::Compute { entry } => {
                    if specialized {
                        &mut specialized_compute
                    } else {
                        &mut compute
                    }
                    .push(pass.index);
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
                    raygen,
                    miss,
                    closest_hit,
                    any_hit,
                } => {
                    if specialized {
                        &mut specialized_raytracing
                    } else {
                        &mut raytracing
                    }
                    .push(pass.index);
                    [Some(raygen), miss, closest_hit, any_hit]
                        .into_iter()
                        .flatten()
                        .collect()
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
        assert_eq!(
            specialized_compute,
            (0..NUM_SPECIALIZED_COMPUTE_PIPELINES).collect::<Vec<_>>()
        );
        assert_eq!(
            specialized_raytracing,
            (0..NUM_SPECIALIZED_RAY_TRACING_PIPELINES).collect::<Vec<_>>()
        );
    }

    #[test]
    fn every_pass_has_compiled_spirv() {
        for pass in PASS_MAP.iter() {
            let bytes = pass.spirv();
            let words = ash::util::read_spv(&mut std::io::Cursor::new(&*bytes))
                .unwrap_or_else(|e| panic!("{} is not valid SPIR-V: {e}", pass.name));
            assert_eq!(
                words[0], 0x0723_0203,
                "{} lacks the SPIR-V magic number",
                pass.name
            );
        }
    }
}

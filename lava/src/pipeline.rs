//! Pass pipelines: the pass map lava-build generates and `init` takes, its entries (ray tracing passes with their shader groups), the cache of every pass's pipelines per specialization and raster state, and their creation (none for passes the device lacks the features of); with `hot-reload`, reloads that replace a pass's pipelines and retire the old ones
use ash::vk::TaggedStructure;
use std::{
    collections::HashMap,
    ffi::CStr,
    hash::Hash,
    sync::{OnceLock, RwLock},
};

use ash::vk;
use smallvec::{SmallVec, smallvec};

use crate::{
    bindless::Bindless,
    command_buffer::Blend,
    error::Result,
    state::{Ctx, Functions},
    vkobjects::rt_pipeline::RaytracingPipeline,
};

/// The version of the code lava-build generates; the generated file asserts it matches.
pub const CODEGEN_VERSION: u32 = 1;
/// The push constants every pass gets, pushed whole; Vulkan 1.4 guarantees 256 bytes.
pub const MAX_PUSH_CONSTANTS_SIZE: usize = 256;
/// The most bytes of specialization constants a pass can have.
pub const MAX_SPECIALIZATION_CONSTANTS_SIZE: usize = 128;
/// The images and buffers a [`crate::command_buffer::BindingOutput`] holds inline; more spill
/// to the heap.
pub const MAX_PASS_IMAGES: usize = 8;
pub const MAX_PASS_BUFFERS: usize = 16;

/// Every pass of the application and how lava-build compiled them (the generated `PASS_MAP`),
/// which hot reload compiles them with again.
pub struct PassMap {
    /// A pass's index here is its `BindingOutput::pass()` and its `PROFILE_PASS`.
    pub passes: &'static [PassEntry],
    /// The absolute include directories.
    pub search_paths: &'static [&'static str],
    /// The preprocessor macros (name, value).
    pub defines: &'static [(&'static str, &'static str)],
    /// The absolute pass and include directories, which hot reload watches.
    pub watch_dirs: &'static [&'static str],
    /// lava-build's `Optimization` level.
    pub optimization: u8,
    /// Compiled with `PROFILING`, for lava with its `profiling` feature.
    pub profiling: bool,
}

static PASS_MAP: OnceLock<&'static PassMap> = OnceLock::new();

/// The passes `lava::init` was given.
pub fn pass_map() -> &'static PassMap {
    PASS_MAP
        .get()
        .expect("lava has no passes; call `lava::init()` first")
}

pub(crate) fn set_pass_map(passes: &'static PassMap) {
    let _ = PASS_MAP.set(passes);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassKind {
    Compute {
        entry: &'static str,
    },
    /// The shader groups in binding table order (`[Group(n)]` in `ray_tracing.slang`).
    RayTracing {
        raygen: &'static str,
        miss: &'static [&'static str],
        hit: &'static [HitGroup],
        callable: &'static [&'static str],
        /// `maxPipelineRayRecursionDepth` (`[MaxRecursion(n)]`).
        max_recursion: u32,
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

/// The entry points of a hit group: triangles without an intersection shader, else procedural.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HitGroup {
    pub closest_hit: Option<&'static str>,
    pub any_hit: Option<&'static str>,
    pub intersection: Option<&'static str>,
}

pub struct PassEntry {
    /// The generated struct's name, e.g. `"BvhCull"`; also the GPU scope of a compute or ray
    /// tracing pass.
    pub name: &'static str,
    /// The SPIR-V lava-build compiled, built into the binary; hot reloads replace its module.
    pub spirv: &'static [u8],
    /// The absolute path of its `.slang` file.
    pub source: &'static str,
    pub kind: PassKind,
    /// Index into the pipelines of its kind (the specialized ones if it has `constants`):
    /// lava-build numbers each kind's passes densely in map order.
    pub index: usize,
    /// Where each specialization constant sits in `BindingOutput::constants`.
    pub constants: &'static [vk::SpecializationMapEntry],
    pub constants_size: usize,
    /// The names of its shader clock regions (`CLOCK_REGION` in `profile.slang`) in slot
    /// order, whose hashes its `constant_bindings` sets. New names need a `cargo build`.
    pub regions: &'static [&'static str],
    /// `lava_build::compile::layout_hash` of the push constants, specialization constants and
    /// entry points the bindings were generated from; a hot reload refuses a pass whose
    /// hash changed, since the Rust side can't follow.
    pub layout: u64,
    /// Its SPIR-V uses ray queries.
    pub ray_query: bool,
}

impl PassEntry {
    /// What the pass needs that the device lacks, if anything.
    fn missing_feature(&self) -> Option<&'static str> {
        let features = Ctx::features();
        if self.ray_query && !features.ray_query {
            Some("ray queries (`Features::ray_query`)")
        } else if matches!(self.kind, PassKind::RayTracing { .. }) && !features.ray_tracing_pipeline
        {
            Some("ray tracing pipelines (`Features::ray_tracing_pipeline`)")
        } else {
            None
        }
    }

    /// Panics if the device can't run the pass.
    fn assert_supported(&self) {
        if let Some(feature) = self.missing_feature() {
            panic!("pass `{}` needs {feature}, which this device lacks", self.name);
        }
    }
}

/// The specialization constants of a pipeline, as `BindingOutput` holds them: the pass's
/// first `len` bytes count, the rest are zeros, so a pipeline key hashes no more than the
/// pass has.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Constants {
    bytes: [u8; MAX_SPECIALIZATION_CONSTANTS_SIZE],
    len: usize,
}

impl Constants {
    /// Those of a pass without specialization constants.
    pub(crate) const NONE: Self = Self {
        bytes: [0; MAX_SPECIALIZATION_CONSTANTS_SIZE],
        len: 0,
    };

    pub(crate) fn new(bytes: &[u8]) -> Self {
        let mut constants = Self::NONE;
        constants.bytes[..bytes.len()].copy_from_slice(bytes);
        constants.len = bytes.len();
        constants
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
}

impl PartialEq for Constants {
    fn eq(&self, other: &Self) -> bool {
        self.bytes() == other.bytes()
    }
}

impl Eq for Constants {}

impl Hash for Constants {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.bytes().hash(state);
    }
}

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
/// with `hot-reload`, also starts watching the shaders. Needs the pass map.
pub(crate) fn init() {
    let manager = PIPELINES.get_or_init(PipelineManager::new);
    #[cfg(feature = "hot-reload")]
    crate::hot_reload::watch(manager);
    #[cfg(not(feature = "hot-reload"))]
    let _ = manager;
}

/// Every pass's pipelines: compute and ray tracing passes without specialization constants
/// have one, made at start; the others one per specialization (and raster state), made at
/// first use. A pass the device can't run (`PassEntry::missing_feature`) has no module and no
/// pipeline, and panics when it is recorded. With `hot-reload`, [`Self::reload_pass`] replaces a pass's module and pipelines,
/// and the replaced pipelines are destroyed once no frame can use them (`epoch`).
pub(crate) struct PipelineManager {
    shader_modules: RwLock<Box<[vk::ShaderModule]>>,
    compute_pipelines: RwLock<Box<[vk::Pipeline]>>,
    specialized_compute_pipelines: RwLock<Box<[HashMap<Constants, vk::Pipeline>]>>,
    raytracing_pipelines: RwLock<Box<[Option<RaytracingPipeline>]>>,
    specialized_raytracing_pipelines: RwLock<Box<[HashMap<Constants, RaytracingPipeline>]>>,
    raster_pipelines: RwLock<Box<[HashMap<RasterHash, vk::Pipeline>]>>,
}

impl PipelineManager {
    fn new() -> Self {
        let passes = pass_map().passes;
        // The size of each kind's pipeline array: its passes are numbered densely.
        let count = |matches: fn(&PassEntry) -> bool| passes.iter().filter(|p| matches(p)).count();
        fn maps<K, V>(n: usize) -> RwLock<Box<[HashMap<K, V>]>> {
            RwLock::new((0..n).map(|_| HashMap::new()).collect())
        }
        let modules: Box<[_]> = passes
            .iter()
            .map(|pass| match pass.missing_feature() {
                Some(_) => vk::ShaderModule::null(),
                None => must(pass, create_module(pass.spirv)),
            })
            .collect();
        Self {
            compute_pipelines: RwLock::new(create_unspecialized(false, |pass_index, pass| {
                match pass.missing_feature() {
                    Some(_) => vk::Pipeline::null(),
                    None => must(
                        pass,
                        create_compute_pipeline(modules[pass_index], pass, &Constants::NONE),
                    ),
                }
            })),
            specialized_compute_pipelines: maps(count(|p| {
                matches!(p.kind, PassKind::Compute { .. }) && !p.constants.is_empty()
            })),
            raytracing_pipelines: RwLock::new(create_unspecialized(true, |pass_index, pass| {
                pass.missing_feature().is_none().then(|| {
                    must(
                        pass,
                        create_raytracing_pipeline(modules[pass_index], pass, &Constants::NONE),
                    )
                })
            })),
            specialized_raytracing_pipelines: maps(count(|p| {
                matches!(p.kind, PassKind::RayTracing { .. }) && !p.constants.is_empty()
            })),
            raster_pipelines: maps(count(|p| {
                matches!(p.kind, PassKind::RasterVertex { .. } | PassKind::RasterMesh { .. })
            })),
            shader_modules: RwLock::new(modules),
        }
    }

    /// The compute pipeline of `pass_index` with `constants`.
    pub(crate) fn compute(&self, pass_index: usize, constants: &Constants) -> vk::Pipeline {
        let pass = &pass_map().passes[pass_index];
        pass.assert_supported();
        if pass.constants.is_empty() {
            return self.compute_pipelines.read().unwrap()[pass.index];
        }
        cached(&self.specialized_compute_pipelines, pass, constants, || {
            // Held until the pipeline exists, so a reload can't destroy the module before.
            let module = self.shader_modules.read().unwrap()[pass_index];
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
        let pass = &pass_map().passes[pass_index];
        pass.assert_supported();
        if pass.constants.is_empty() {
            return record(
                self.raytracing_pipelines.read().unwrap()[pass.index]
                    .as_ref()
                    .unwrap(),
            );
        }
        let maps = &self.specialized_raytracing_pipelines;
        if !maps.read().unwrap()[pass.index].contains_key(constants) {
            let map = &mut maps.write().unwrap()[pass.index];
            let cached = map.len();
            map.entry(*constants).or_insert_with(|| {
                note_new_pipeline(pass, cached);
                let modules = self.shader_modules.read().unwrap();
                must(
                    pass,
                    create_raytracing_pipeline(modules[pass_index], pass, constants),
                )
            });
        }
        record(&maps.read().unwrap()[pass.index][constants])
    }

    /// The pipeline of raster pass `pass_index` for `hash`.
    pub(crate) fn raster(&self, hash: &RasterHash, pass_index: usize) -> vk::Pipeline {
        let pass = &pass_map().passes[pass_index];
        pass.assert_supported();
        cached(&self.raster_pipelines, pass, hash, || {
            let module = self.shader_modules.read().unwrap()[pass_index];
            must(pass, create_raster_pipeline(module, hash, pass_index))
        })
    }

    /// Replaces the pass's module and every pipeline made from it. A replaced pipeline is
    /// destroyed once no frame can use it (`epoch`). If the driver rejects the module or one
    /// of the pipelines, nothing is replaced: every new pipeline is made before any is swapped
    /// in, and no lock is held while one fails. A pass the device can't run is skipped.
    #[cfg(feature = "hot-reload")]
    pub(crate) fn reload_pass(&self, pass_index: usize, spirv: &[u8]) -> Result<()> {
        let pass = &pass_map().passes[pass_index];
        if pass.missing_feature().is_some() {
            return Ok(());
        }
        let module = create_module(spirv)?;
        let destroy_module = |module| unsafe { Ctx::device().destroy_shader_module(module, None) };
        let unspecialized = pass.constants.is_empty();

        // Every pipeline of the pass with the new module, then the swap.
        let result: Result<()> = (|| {
            match pass.kind {
                PassKind::Compute { .. } if unspecialized => {
                    let pipeline = create_compute_pipeline(module, pass, &Constants::NONE)?;
                    let pipelines = &mut self.compute_pipelines.write().unwrap()[pass.index];
                    retire([std::mem::replace(pipelines, pipeline)]);
                }
                PassKind::Compute { .. } => {
                    retire(rebuild(
                        &self.specialized_compute_pipelines,
                        pass,
                        |constants| create_compute_pipeline(module, pass, constants),
                    )?);
                }
                PassKind::RayTracing { .. } if unspecialized => {
                    let pipeline = create_raytracing_pipeline(module, pass, &Constants::NONE)?;
                    let pipelines = &mut self.raytracing_pipelines.write().unwrap()[pass.index];
                    retire(pipelines.replace(pipeline));
                }
                PassKind::RayTracing { .. } => {
                    let maps = &self.specialized_raytracing_pipelines;
                    retire(rebuild(maps, pass, |constants| {
                        create_raytracing_pipeline(module, pass, constants)
                    })?);
                }
                PassKind::RasterVertex { .. } | PassKind::RasterMesh { .. } => {
                    retire(rebuild(&self.raster_pipelines, pass, |hash| {
                        create_raster_pipeline(module, hash, pass_index)
                    })?);
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

impl Into<vk::Pipeline> for RaytracingPipeline {
    fn into(self) -> vk::Pipeline {
        self.pipeline
    }
}

/// Destroys `pipelines` once no frame can use them any more.
#[cfg(feature = "hot-reload")]
fn retire<P: Into<vk::Pipeline> + Send + 'static>(pipelines: impl IntoIterator<Item = P>) {
    let pipelines: Vec<P> = pipelines.into_iter().collect();
    crate::epoch::retire(move || {
        pipelines
            .into_iter()
            .for_each(|i| unsafe { Ctx::device().destroy_pipeline(i.into(), None) })
    });
}

/// Makes the pipeline of every key in the pass's map with `create`, then swaps them all in
/// and returns the replaced ones. If one fails, the new ones are destroyed and nothing changes.
#[cfg(feature = "hot-reload")]
fn rebuild<K: Hash + Eq + Clone, P: Into<vk::Pipeline> + Send + 'static>(
    maps: &RwLock<Box<[HashMap<K, P>]>>,
    pass: &PassEntry,
    mut create: impl FnMut(&K) -> Result<P>,
) -> Result<Vec<P>> {
    let keys: Vec<K> = maps.read().unwrap()[pass.index].keys().cloned().collect();
    let mut new = Vec::with_capacity(keys.len());
    for key in &keys {
        match create(key) {
            Ok(pipeline) => new.push(pipeline),
            Err(err) => {
                retire(new);
                return Err(err);
            }
        }
    }
    let map = &mut maps.write().unwrap()[pass.index];
    let replaced = keys.into_iter().zip(new);
    Ok(replaced
        .filter_map(|(key, pipeline)| map.insert(key, pipeline))
        .collect())
}

/// A pipeline lava can't do without: created at start or at a pass's first use, so a
/// driver that rejects it ends the program with the pass's name.
fn must<T>(pass: &PassEntry, result: Result<T>) -> T {
    result.unwrap_or_else(|err| panic!("pass `{}`: {err}", pass.name))
}

/// `create` called for every compute (or ray tracing) pass without specialization constants,
/// as their array: lava-build numbers each array's passes in map order.
fn create_unspecialized<T>(
    ray_tracing: bool,
    mut create: impl FnMut(usize, &'static PassEntry) -> T,
) -> Box<[T]> {
    pass_map()
        .passes
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
        .collect()
}

/// A pass whose pipeline cache grows this large gets one warning: its specialization constants
/// (or raster states) change too often, so they should be push constants.
const MANY_PIPELINES: usize = 16;

/// The pipeline of `key` in the pass's map, made by `create` at first use.
fn cached<K: Hash + Eq + Clone, V: Clone>(
    maps: &RwLock<Box<[HashMap<K, V>]>>,
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
        .data(constants.bytes());
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
        hit,
        callable,
        max_recursion,
    } = pass.kind
    else {
        unreachable!("{} is a ray tracing pass", pass.name)
    };
    let info = vk::SpecializationInfo::default()
        .map_entries(pass.constants)
        .data(constants.bytes());
    let mut stages: SmallVec<[vk::PipelineShaderStageCreateInfo; 8]> = SmallVec::new();
    let mut groups: SmallVec<[vk::RayTracingShaderGroupCreateInfoKHR; 8]> = SmallVec::new();
    // Adds the stage and returns its index, or `SHADER_UNUSED_KHR` without one.
    let mut stage = |entry: Option<&'static str>, flags| match entry {
        Some(entry) => {
            stages.push(make_shader_stage(entry, flags, module, &info));
            stages.len() as u32 - 1
        }
        None => vk::SHADER_UNUSED_KHR,
    };
    let group = |ty| {
        vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(ty)
            .general_shader(vk::SHADER_UNUSED_KHR)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR)
    };
    let general = vk::RayTracingShaderGroupTypeKHR::GENERAL;
    groups.push(group(general).general_shader(stage(Some(raygen), vk::ShaderStageFlags::RAYGEN_KHR)));
    for &entry in miss {
        groups.push(group(general).general_shader(stage(Some(entry), vk::ShaderStageFlags::MISS_KHR)));
    }
    for g in hit {
        let ty = if g.intersection.is_some() {
            vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP
        } else {
            vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP
        };
        groups.push(
            group(ty)
                .closest_hit_shader(stage(g.closest_hit, vk::ShaderStageFlags::CLOSEST_HIT_KHR))
                .any_hit_shader(stage(g.any_hit, vk::ShaderStageFlags::ANY_HIT_KHR))
                .intersection_shader(stage(g.intersection, vk::ShaderStageFlags::INTERSECTION_KHR)),
        );
    }
    for &entry in callable {
        groups.push(
            group(general).general_shader(stage(Some(entry), vk::ShaderStageFlags::CALLABLE_KHR)),
        );
    }
    let counts = [1, miss.len(), hit.len(), callable.len()].map(|n| n as u32);
    RaytracingPipeline::new(Bindless::layout(), &stages, &groups, counts, max_recursion)
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

    let pass = &pass_map().passes[pass_index];
    let c = vk::SpecializationInfo::default()
        .map_entries(pass.constants)
        .data(hash.constants.bytes());
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
        .push(&mut rendering);

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
}

//! Pass pipelines: the pass map a frontend generates and `init` takes (refused if a pass needs a feature lava was compiled without), its entries (ray tracing passes with their shader groups), the cache of every pass's pipelines per specialization and raster key (attachment formats and the draw state the device can't set dynamically), their creation (none for passes the device lacks the features of), and `reload_pass`, which replaces a pass's pipelines and retires the old ones
use std::{
    ffi::CStr,
    hash::Hash,
    sync::{OnceLock, RwLock},
};

use crate::vk;
use hashbrown::{Equivalent, HashMap};
use smallvec::{SmallVec, smallvec};

use crate::{
    command_buffer::{Baked, ColorComponentFlags, Dynamic},
    error::{Error, Result},
    features::Features,
    state::{Ctx, Functions},
};
#[cfg(bake_blend)]
use crate::command_buffer::{Blend, effective_write_mask};
#[cfg(feature = "ray-tracing-pipeline")]
use crate::vkobjects::rt_pipeline::RaytracingPipeline;

/// The most bytes of push data a pass pushes; lava picks only devices with as much
/// (`maxPushDataSize`).
pub const MAX_PUSH_CONSTANTS_SIZE: usize = 256;

/// Every pass of the application (the generated `PASS_MAP`).
pub struct PassMap {
    /// A pass's index here is its `BindingOutput::pass()`.
    pub passes: &'static [PassEntry],
}

pub(crate) static PASS_MAP: OnceLock<&'static PassMap> = OnceLock::new();

/// The passes `lava::init` was given.
pub fn pass_map() -> &'static PassMap {
    PASS_MAP
        .get()
        .expect("lava has no passes; call `lava::init()` first")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassKind {
    Compute {
        entry: &'static str,
    },
    /// The shader groups in binding table order (`[Group(n)]`).
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

impl PassKind {
    /// The stages its resource accesses happen in, its marker's `STAGES` (`command_buffer::kind`).
    pub fn stages(&self) -> vk::PipelineStageFlags2 {
        use crate::command_buffer::kind;
        match self {
            PassKind::Compute { .. } => kind::Compute::STAGES,
            PassKind::RayTracing { .. } => kind::RayTracing::STAGES,
            PassKind::RasterVertex { .. } => kind::RasterVertex::STAGES,
            PassKind::RasterMesh { .. } => kind::RasterMesh::STAGES,
        }
    }
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
    pub spirv: &'static [u8],
    pub kind: PassKind,
    /// Index into the pipelines of its kind (the specialized ones if it has `constants`):
    /// the frontend numbers each kind's passes densely in map order.
    pub index: usize,
    /// Where each specialization constant sits in `BindingOutput::constants`.
    pub constants: &'static [vk::SpecializationMapEntry],
    pub constants_size: usize,
}

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub(crate) struct RasterKey {
    color_formats: SmallVec<[vk::Format; 4]>,
    depth_format: vk::Format,
    stencil_format: vk::Format,
    constants: Box<[u8]>,
    baked: Baked,
    #[cfg(bake_blend)]
    blends: SmallVec<[Blend; 4]>,
    #[cfg(bake_blend)]
    write_masks: SmallVec<[ColorComponentFlags; 4]>,
}

/// Hashes like the [`RasterKey`] it stands for: the same fields in the same order, and slices,
/// `SmallVec`s and boxed slices all hash as slices.
#[derive(Hash, Clone, Copy, Debug, PartialEq)]
pub(crate) struct RasterKeyRef<'a> {
    pub(crate) color_formats: &'a [vk::Format],
    pub(crate) depth_format: vk::Format,
    pub(crate) stencil_format: vk::Format,
    pub(crate) constants: &'a [u8],
    pub(crate) baked: Baked,
    #[cfg(bake_blend)]
    pub(crate) blends: &'a [Blend],
    #[cfg(bake_blend)]
    pub(crate) write_masks: &'a [ColorComponentFlags],
}

impl Equivalent<RasterKey> for RasterKeyRef<'_> {
    fn equivalent(&self, key: &RasterKey) -> bool {
        #[cfg(bake_blend)]
        if self.blends != key.blends.as_slice() || self.write_masks != key.write_masks.as_slice()
        {
            return false;
        }
        self.color_formats == key.color_formats.as_slice()
            && self.depth_format == key.depth_format
            && self.stencil_format == key.stencil_format
            && self.constants == &*key.constants
            && self.baked == key.baked
    }
}

type Specialized<K, V> = HashMap<K, V>;

static PIPELINES: OnceLock<PipelineManager> = OnceLock::new();

pub(crate) fn pipelines() -> &'static PipelineManager {
    PIPELINES
        .get()
        .expect("pipelines were not initialized; call `lava::init()` first")
}

pub(crate) fn init() -> Result<()> {
    let compiled_out: Vec<String> = (pass_map().passes.iter())
        .filter_map(|pass| {
            let needs = Features::of_spirv(pass.spirv);
            (!needs.compiled_out.is_empty())
                .then(|| format!("`{}` needs `{}`", pass.name, needs.compiled_out.join("`, `")))
        })
        .collect();
    if !compiled_out.is_empty() {
        return Err(Error::message(format!(
            "lava was compiled without cargo features passes need: {}",
            compiled_out.join("; ")
        )));
    }
    PIPELINES.get_or_init(|| {
        let passes = pass_map().passes;
        let count = |matches: fn(&PassEntry) -> bool| passes.iter().filter(|p| matches(p)).count();
        fn maps<K, V>(n: usize) -> RwLock<Box<[HashMap<K, V>]>> {
            RwLock::new((0..n).map(|_| HashMap::new()).collect())
        }
        let missing: Box<[_]> = passes
            .iter()
            .map(|pass| Features::of_spirv(pass.spirv).features.missing_from(&Ctx::features()))
            .collect();
        let modules: Box<[_]> = (passes.iter().zip(&missing))
            .map(|(pass, missing)| match missing.is_empty() {
                false => vk::ShaderModule::null(),
                true => must(pass, create_module(pass.spirv)),
            })
            .collect();
        PipelineManager {
            compute_pipelines: RwLock::new(create_unspecialized(
                false,
                |pass_index, pass| match missing[pass_index].is_empty() {
                    false => vk::Pipeline::null(),
                    true => must(
                        pass,
                        create_compute_pipeline(modules[pass_index], pass, &[]),
                    ),
                },
            )),
            specialized_compute_pipelines: maps(count(|p| {
                matches!(p.kind, PassKind::Compute { .. }) && !p.constants.is_empty()
            })),
            #[cfg(feature = "ray-tracing-pipeline")]
            raytracing_pipelines: RwLock::new(create_unspecialized(true, |pass_index, pass| {
                missing[pass_index].is_empty().then(|| {
                    must(
                        pass,
                        create_raytracing_pipeline(modules[pass_index], pass, &[]),
                    )
                })
            })),
            #[cfg(feature = "ray-tracing-pipeline")]
            specialized_raytracing_pipelines: maps(count(|p| {
                matches!(p.kind, PassKind::RayTracing { .. }) && !p.constants.is_empty()
            })),
            raster_pipelines: maps(count(|p| {
                matches!(
                    p.kind,
                    PassKind::RasterVertex { .. } | PassKind::RasterMesh { .. }
                )
            })),
            shader_modules: RwLock::new(modules),
            missing,
        }
    });
    Ok(())
}

/// Replaces the SPIR-V of pass `pass_index` and every pipeline made from it, which must keep
/// the layout the bindings were generated from; if the driver rejects it, nothing is replaced.
pub fn reload_pass(pass_index: usize, spirv: &[u8]) -> Result<()> {
    let manager = pipelines();
    let pass = &pass_map().passes[pass_index];
    if !manager.missing[pass_index].is_empty() {
        return Ok(());
    }
    let needs = Features::of_spirv(spirv);
    let mut missing = needs.features.missing_from(&Ctx::features());
    missing.extend(needs.compiled_out);
    if !missing.is_empty() {
        return Err(Error::message(&format!(
            "the new `{}` needs `{}`, which this device lacks or lava was compiled without",
            pass.name,
            missing.join("`, `")
        )));
    }
    let module = create_module(spirv)?;
    let destroy_module = |module| unsafe { Ctx::device().destroy_shader_module(module, None) };
    let unspecialized = pass.constants.is_empty();

    let result: Result<()> = (|| {
        match pass.kind {
            PassKind::Compute { .. } if unspecialized => {
                let pipeline = create_compute_pipeline(module, pass, &[])?;
                let pipelines = &mut manager.compute_pipelines.write().unwrap()[pass.index];
                retire([std::mem::replace(pipelines, pipeline)]);
            }
            PassKind::Compute { .. } => {
                retire(rebuild(
                    &manager.specialized_compute_pipelines,
                    pass,
                    |constants| create_compute_pipeline(module, pass, constants),
                )?);
            }
            #[cfg(feature = "ray-tracing-pipeline")]
            PassKind::RayTracing { .. } if unspecialized => {
                let pipeline = create_raytracing_pipeline(module, pass, &[])?;
                let pipelines = &mut manager.raytracing_pipelines.write().unwrap()[pass.index];
                retire(pipelines.replace(pipeline));
            }
            #[cfg(feature = "ray-tracing-pipeline")]
            PassKind::RayTracing { .. } => {
                let maps = &manager.specialized_raytracing_pipelines;
                retire(rebuild(maps, pass, |constants| {
                    create_raytracing_pipeline(module, pass, constants)
                })?);
            }
            PassKind::RasterVertex { .. } | PassKind::RasterMesh { .. } => {
                retire(rebuild(&manager.raster_pipelines, pass, |key| {
                    let key = RasterKeyRef {
                        color_formats: &key.color_formats,
                        depth_format: key.depth_format,
                        stencil_format: key.stencil_format,
                        constants: &key.constants,
                        baked: key.baked,
                        #[cfg(bake_blend)]
                        blends: &key.blends,
                        #[cfg(bake_blend)]
                        write_masks: &key.write_masks,
                    };
                    create_raster_pipeline(module, &key, pass_index)
                })?);
            }
            #[cfg(not(feature = "ray-tracing-pipeline"))]
            PassKind::RayTracing { .. } => unreachable!("a ray tracing pass without its feature"),
        }
        Ok(())
    })();
    if let Err(err) = result {
        destroy_module(module);
        return Err(err);
    }
    let old = std::mem::replace(
        &mut manager.shader_modules.write().unwrap()[pass_index],
        module,
    );
    // Pipelines don't need their module once created.
    destroy_module(old);
    Ok(())
}

/// A pass whose SPIR-V needs features the device lacks has no module and no pipeline.
pub(crate) struct PipelineManager {
    missing: Box<[Vec<&'static str>]>,
    shader_modules: RwLock<Box<[vk::ShaderModule]>>,
    compute_pipelines: RwLock<Box<[vk::Pipeline]>>,
    specialized_compute_pipelines: RwLock<Box<[Specialized<Box<[u8]>, vk::Pipeline>]>>,
    #[cfg(feature = "ray-tracing-pipeline")]
    raytracing_pipelines: RwLock<Box<[Option<RaytracingPipeline>]>>,
    #[cfg(feature = "ray-tracing-pipeline")]
    specialized_raytracing_pipelines: RwLock<Box<[Specialized<Box<[u8]>, RaytracingPipeline>]>>,
    raster_pipelines: RwLock<Box<[Specialized<RasterKey, vk::Pipeline>]>>,
}

impl PipelineManager {
    /// Only hand-made bindings reach a pass the device can't run: generated ones can't be made.
    fn assert_supported(&self, pass_index: usize) {
        let missing = &self.missing[pass_index];
        if !missing.is_empty() {
            panic!(
                "pass `{}` needs `{}`, which this device lacks",
                pass_map().passes[pass_index].name,
                missing.join("`, `")
            );
        }
    }

    pub(crate) fn compute(&self, pass_index: usize, constants: &[u8]) -> vk::Pipeline {
        let pass = &pass_map().passes[pass_index];
        self.assert_supported(pass_index);
        if pass.constants.is_empty() {
            return self.compute_pipelines.read().unwrap()[pass.index];
        }
        cached(
            &self.specialized_compute_pipelines,
            pass,
            constants,
            |constants| constants.into(),
            || {
                let module = self.shader_modules.read().unwrap()[pass_index];
                must(pass, create_compute_pipeline(module, pass, constants))
            },
        )
    }

    /// Calls `record` while the cache stays locked.
    #[cfg(feature = "ray-tracing-pipeline")]
    pub(crate) fn ray_tracing<R>(
        &self,
        pass_index: usize,
        constants: &[u8],
        record: impl FnOnce(&RaytracingPipeline) -> R,
    ) -> R {
        let pass = &pass_map().passes[pass_index];
        self.assert_supported(pass_index);
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
            let count = map.len();
            map.entry(constants.into()).or_insert_with(|| {
                note_new_pipeline(pass, count);
                let modules = self.shader_modules.read().unwrap();
                must(
                    pass,
                    create_raytracing_pipeline(modules[pass_index], pass, constants),
                )
            });
        }
        record(&maps.read().unwrap()[pass.index][constants])
    }

    pub(crate) fn raster(&self, key: &RasterKeyRef, pass_index: usize) -> vk::Pipeline {
        let pass = &pass_map().passes[pass_index];
        self.assert_supported(pass_index);
        cached(
            &self.raster_pipelines,
            pass,
            key,
            |key| RasterKey {
                color_formats: SmallVec::from_slice(key.color_formats),
                depth_format: key.depth_format,
                stencil_format: key.stencil_format,
                constants: key.constants.into(),
                baked: key.baked,
                #[cfg(bake_blend)]
                blends: SmallVec::from_slice(key.blends),
                #[cfg(bake_blend)]
                write_masks: SmallVec::from_slice(key.write_masks),
            },
            || {
                let module = self.shader_modules.read().unwrap()[pass_index];
                must(pass, create_raster_pipeline(module, key, pass_index))
            },
        )
    }
}

#[cfg(feature = "ray-tracing-pipeline")]
impl Into<vk::Pipeline> for RaytracingPipeline {
    fn into(self) -> vk::Pipeline {
        self.pipeline
    }
}

fn retire<P: Into<vk::Pipeline> + Send + 'static>(pipelines: impl IntoIterator<Item = P>) {
    let pipelines: Vec<P> = pipelines.into_iter().collect();
    crate::lifetime::retire(move || {
        pipelines
            .into_iter()
            .for_each(|i| unsafe { Ctx::device().destroy_pipeline(i.into(), None) })
    });
}

/// Returns the replaced pipelines; if one fails, the new ones are destroyed and nothing changes.
fn rebuild<K: Hash + Eq + Clone, P: Into<vk::Pipeline> + Send + 'static>(
    maps: &RwLock<Box<[Specialized<K, P>]>>,
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

fn must<T>(pass: &PassEntry, result: Result<T>) -> T {
    result.unwrap_or_else(|err| panic!("pass `{}`: {err}", pass.name))
}

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

const MANY_PIPELINES: usize = 16;

fn cached<K: Hash + Eq, Q: Hash + Equivalent<K> + ?Sized, V: Clone>(
    maps: &RwLock<Box<[Specialized<K, V>]>>,
    pass: &PassEntry,
    key: &Q,
    to_owned: impl FnOnce(&Q) -> K,
    create: impl FnOnce() -> V,
) -> V {
    if let Some(pipeline) = maps.read().unwrap()[pass.index].get(key) {
        return pipeline.clone();
    }
    let map = &mut maps.write().unwrap()[pass.index];
    if let Some(pipeline) = map.get(key) {
        return pipeline.clone();
    }
    note_new_pipeline(pass, map.len());
    let pipeline = create();
    map.insert(to_owned(key), pipeline.clone());
    pipeline
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

fn create_compute_pipeline(
    module: vk::ShaderModule,
    pass: &PassEntry,
    constants: &[u8],
) -> Result<vk::Pipeline> {
    let PassKind::Compute { entry } = pass.kind else {
        unreachable!("{} is a compute pass", pass.name)
    };
    let info = vk::SpecializationInfo::default()
        .map_entries(pass.constants)
        .data(constants);
    let stage = make_shader_stage(entry, vk::ShaderStageFlags::COMPUTE, module, &info);
    let mut flags = heap_flags();
    let create_info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .push_next(&mut flags);
    let pipeline = unsafe {
        Ctx::device()
            .create_compute_pipelines(vk::PipelineCache::null(), &[create_info], None)
            .map_err(|(_, err)| err)?
    }[0];
    Functions::set_debug_name(entry, pipeline);
    Ok(pipeline)
}

#[cfg(feature = "ray-tracing-pipeline")]
fn create_raytracing_pipeline(
    module: vk::ShaderModule,
    pass: &PassEntry,
    constants: &[u8],
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
        .data(constants);
    let mut stages: SmallVec<[vk::PipelineShaderStageCreateInfo; 8]> = SmallVec::new();
    let mut groups: SmallVec<[vk::RayTracingShaderGroupCreateInfoKHR; 8]> = SmallVec::new();
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
    groups
        .push(group(general).general_shader(stage(Some(raygen), vk::ShaderStageFlags::RAYGEN_KHR)));
    for &entry in miss {
        groups.push(
            group(general).general_shader(stage(Some(entry), vk::ShaderStageFlags::MISS_KHR)),
        );
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
                .intersection_shader(stage(
                    g.intersection,
                    vk::ShaderStageFlags::INTERSECTION_KHR,
                )),
        );
    }
    for &entry in callable {
        groups.push(
            group(general).general_shader(stage(Some(entry), vk::ShaderStageFlags::CALLABLE_KHR)),
        );
    }
    let counts = [1, miss.len(), hit.len(), callable.len()].map(|n| n as u32);
    RaytracingPipeline::new(&stages, &groups, counts, max_recursion)
}

/// Every pipeline reads the descriptor heaps and has no layout.
pub(crate) fn heap_flags() -> vk::PipelineCreateFlags2CreateInfoKHR<'static> {
    vk::PipelineCreateFlags2CreateInfoKHR::default().flags(vk::PIPELINE_CREATE_2_DESCRIPTOR_HEAP_EXT)
}

/// A stage of `entry`.
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
    key: &RasterKeyRef,
    pass_index: usize,
) -> Result<vk::Pipeline> {
    let constants = key.constants;
    let _span = tracing::info_span!("create raster pipeline").entered();

    let pass = &pass_map().passes[pass_index];
    let c = vk::SpecializationInfo::default()
        .map_entries(pass.constants)
        .data(constants);
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
        .color_attachment_formats(key.color_formats)
        .depth_attachment_format(key.depth_format)
        .stencil_attachment_format(key.stencil_format)
        .view_mask(0);
    // State the device sets dynamically is ignored here; the rest is the key's (`Baked`).
    let features = Ctx::features();
    let dynamic = Dynamic::of(&features);
    use vk::DynamicState as D;
    let mut dynamic_states = vec![
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
    for (on, states) in optional {
        if on {
            dynamic_states.extend_from_slice(states);
        }
    }
    // Mesh pipelines have no input assembly state.
    if !matches!(pass.kind, PassKind::RasterMesh { .. }) {
        dynamic_states.extend([D::PRIMITIVE_TOPOLOGY, D::PRIMITIVE_RESTART_ENABLE]);
    }
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let color_blend_attachments: SmallVec<[vk::PipelineColorBlendAttachmentState; 8]> = (0
        ..key.color_formats.len())
        .map(|i| {
            #[cfg(bake_blend)]
            if !key.blends.is_empty() {
                let blend = key.blends[i];
                let equation = blend.vk_equation();
                return vk::PipelineColorBlendAttachmentState {
                    blend_enable: blend.equation().is_some() as vk::Bool32,
                    src_color_blend_factor: equation.src_color_blend_factor,
                    dst_color_blend_factor: equation.dst_color_blend_factor,
                    color_blend_op: equation.color_blend_op,
                    src_alpha_blend_factor: equation.src_alpha_blend_factor,
                    dst_alpha_blend_factor: equation.dst_alpha_blend_factor,
                    alpha_blend_op: equation.alpha_blend_op,
                    color_write_mask: effective_write_mask(blend, key.write_masks[i]),
                };
            }
            let _ = i;
            vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(ColorComponentFlags::RGBA)
        })
        .collect();
    #[cfg(bake_logic_op)]
    let logic_op = key.baked.logic_op;
    #[cfg(not(bake_logic_op))]
    let logic_op = None;
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(&color_blend_attachments)
        .logic_op_enable(logic_op.is_some())
        .logic_op(logic_op.unwrap_or(vk::LogicOp::COPY));
    #[cfg(bake_depth_clip_negative_one_to_one)]
    let negative_one_to_one = key.baked.depth_clip_negative_one_to_one;
    #[cfg(not(bake_depth_clip_negative_one_to_one))]
    let negative_one_to_one = false;
    let mut viewport_state = vk::PipelineViewportStateCreateInfo::default();
    let mut clip_control = vk::PipelineViewportDepthClipControlCreateInfoEXT::default()
        .negative_one_to_one(negative_one_to_one);
    if crate::command_buffer::on!(&features, "depth-clip-control", depth_clip_control) {
        viewport_state = viewport_state.push_next(&mut clip_control);
    }
    #[cfg(bake_polygon_mode)]
    let polygon_mode = key.baked.polygon_mode;
    #[cfg(not(bake_polygon_mode))]
    let polygon_mode = vk::PolygonMode::FILL;
    #[cfg(bake_depth_clamp)]
    let depth_clamp = key.baked.depth_clamp;
    #[cfg(not(bake_depth_clamp))]
    let depth_clamp = false;
    let mut rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(polygon_mode)
        .depth_clamp_enable(depth_clamp)
        .line_width(1.0);
    #[cfg(bake_conservative_rasterization)]
    let conservative = key.baked.conservative;
    #[cfg(not(bake_conservative_rasterization))]
    let conservative = vk::ConservativeRasterizationModeEXT::DISABLED;
    let mut conservative = vk::PipelineRasterizationConservativeStateCreateInfoEXT::default()
        .conservative_rasterization_mode(conservative);
    if crate::command_buffer::on!(&features, "conservative-rasterization", conservative_rasterization) {
        rasterization_state = rasterization_state.push_next(&mut conservative);
    }
    #[cfg(bake_provoking_vertex)]
    let provoking_vertex = key.baked.provoking_vertex;
    #[cfg(not(bake_provoking_vertex))]
    let provoking_vertex = vk::ProvokingVertexModeEXT::FIRST_VERTEX;
    let mut provoking = vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT::default()
        .provoking_vertex_mode(provoking_vertex);
    if crate::command_buffer::on!(&features, "provoking-vertex", provoking_vertex) {
        rasterization_state = rasterization_state.push_next(&mut provoking);
    }
    #[cfg(bake_depth_clip)]
    let depth_clip = key.baked.depth_clip;
    #[cfg(not(bake_depth_clip))]
    let depth_clip = true;
    let mut depth_clip = vk::PipelineRasterizationDepthClipStateCreateInfoEXT::default()
        .depth_clip_enable(depth_clip);
    if crate::command_buffer::on!(&features, "depth-clip-enable", depth_clip_enable) {
        rasterization_state = rasterization_state.push_next(&mut depth_clip);
    }
    #[cfg(bake_line_mode)]
    let line_mode = key.baked.line_mode;
    #[cfg(not(bake_line_mode))]
    let line_mode = vk::LineRasterizationModeKHR::DEFAULT;
    #[cfg(bake_line_stipple)]
    let line_stipple = key.baked.line_stipple;
    #[cfg(not(bake_line_stipple))]
    let line_stipple = false;
    let mut line = vk::PipelineRasterizationLineStateCreateInfoKHR::default()
        .line_rasterization_mode(line_mode)
        .stippled_line_enable(line_stipple)
        .line_stipple_factor(1)
        .line_stipple_pattern(!0);
    if features.lines() {
        rasterization_state = rasterization_state.push_next(&mut line);
    }
    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default();

    let mut flags = heap_flags();
    create_info = create_info
        .stages(&stages)
        .push_next(&mut flags)
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

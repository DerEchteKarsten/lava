//! Pass pipelines: the pass map a frontend generates and `init` takes (refused if a pass needs a feature lava was compiled without), its entries (ray tracing passes with their shader groups), each pass's pipelines per specialization and raster key (attachment formats, view mask and the draw state the device can't set dynamically, the topology class among it), raster passes with or without a fragment shader behind a lock of their own and made outside it, their creation through one `VkPipelineCache` (loaded from and saved to `Config::pipeline_cache`; batched at init; none for passes the device can't run), `reload_pass`, which replaces a pass's pipelines and retires the old ones, and their destruction at `shutdown`
use std::{
    ffi::CStr,
    hash::Hash,
    path::Path,
    sync::{
        PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard,
        atomic::{AtomicPtr, Ordering},
    },
};

use crate::vk;
use hashbrown::{Equivalent, HashMap};
use smallvec::SmallVec;

use crate::{
    command_buffer::{Baked, ColorComponentFlags},
    error::{Error, Result},
    features::Features,
    global::{Global, NOT_INITIALISED},
    state::{Ctx, Functions},
};
#[cfg(bake_blend)]
use crate::command_buffer::{Blend, effective_write_mask};
#[cfg(feature = "ray-tracing-pipeline")]
use crate::vkobjects::rt_pipeline::{BoundRayTracing, RayTracingDesc, RaytracingPipeline};

/// The most bytes of push data a pass pushes; lava picks only devices with as much
/// (`maxPushDataSize`).
pub const MAX_PUSH_CONSTANTS_SIZE: usize = 256;

/// Every pass of the application (the generated `PASS_MAP`).
pub struct PassMap {
    /// A pass's index here is the one its bindings name (`Bindings::new`, `Sections::pass`).
    pub passes: &'static [PassEntry],
}

/// The `Config::passes` of `init`; null when lava isn't initialised.
static PASS_MAP: AtomicPtr<PassMap> = AtomicPtr::new(std::ptr::null_mut());

/// The passes `lava::init` was given.
#[track_caller]
pub fn pass_map() -> &'static PassMap {
    try_pass_map().expect(NOT_INITIALISED)
}

/// The passes `lava::init` was given, `None` when lava isn't initialised.
pub(crate) fn try_pass_map() -> Option<&'static PassMap> {
    // SAFETY: only ever a `&'static PassMap` (`set_pass_map`).
    unsafe { PASS_MAP.load(Ordering::Acquire).as_ref() }
}

pub(crate) fn set_pass_map(passes: &'static PassMap) {
    PASS_MAP.store(std::ptr::from_ref(passes).cast_mut(), Ordering::Release);
}

pub(crate) fn unset_pass_map() {
    PASS_MAP.store(std::ptr::null_mut(), Ordering::Release);
}

/// A pass's entry points by stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassKind {
    Compute {
        entry: &'static CStr,
    },
    /// The shader groups in binding table order (`[Group(n)]`).
    RayTracing {
        raygen: &'static CStr,
        miss: &'static [&'static CStr],
        hit: &'static [HitGroup],
        callable: &'static [&'static CStr],
        /// `maxPipelineRayRecursionDepth` (`[MaxRecursion(n)]`, 0 to 31); a pass deeper than the
        /// device's `maxRayRecursionDepth` gets no pipeline, like one needing a missing feature.
        max_recursion: u32,
    },
    /// Without a fragment shader, a pass draws only depth and stencil (shadow maps, depth
    /// prepasses).
    RasterVertex {
        vertex: &'static CStr,
        fragment: Option<&'static CStr>,
    },
    RasterMesh {
        amp: Option<&'static CStr>,
        mesh: &'static CStr,
        fragment: Option<&'static CStr>,
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
    pub closest_hit: Option<&'static CStr>,
    pub any_hit: Option<&'static CStr>,
    pub intersection: Option<&'static CStr>,
}

pub struct PassEntry {
    /// The generated struct's name, e.g. `"BvhCull"`; also the GPU scope of a compute or ray
    /// tracing pass and the debug name of its pipelines.
    pub name: &'static str,
    pub spirv: &'static [u8],
    pub kind: PassKind,
    /// Where each specialization constant sits in `BindingOutput::constants`.
    pub constants: &'static [vk::SpecializationMapEntry],
    /// The size of the constant bytes its bindings carry (`Bindings::with_constants`).
    pub constants_size: usize,
}

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub(crate) struct RasterKey {
    color_formats: SmallVec<[vk::Format; 4]>,
    depth_format: vk::Format,
    stencil_format: vk::Format,
    view_mask: u32,
    constants: Box<[u8]>,
    baked: Baked,
    #[cfg(bake_blend)]
    blends: SmallVec<[Blend; 4]>,
    #[cfg(bake_blend)]
    write_masks: SmallVec<[ColorComponentFlags; 4]>,
}

impl RasterKey {
    fn borrow(&self) -> RasterKeyRef<'_> {
        RasterKeyRef {
            color_formats: &self.color_formats,
            depth_format: self.depth_format,
            stencil_format: self.stencil_format,
            view_mask: self.view_mask,
            constants: &self.constants,
            baked: self.baked,
            #[cfg(bake_blend)]
            blends: &self.blends,
            #[cfg(bake_blend)]
            write_masks: &self.write_masks,
        }
    }
}

/// Hashes like the [`RasterKey`] it stands for: the same fields in the same order, and slices,
/// `SmallVec`s and boxed slices all hash as slices.
#[derive(Hash, Clone, Copy, Debug, PartialEq)]
pub(crate) struct RasterKeyRef<'a> {
    pub(crate) color_formats: &'a [vk::Format],
    pub(crate) depth_format: vk::Format,
    pub(crate) stencil_format: vk::Format,
    pub(crate) view_mask: u32,
    pub(crate) constants: &'a [u8],
    pub(crate) baked: Baked,
    #[cfg(bake_blend)]
    pub(crate) blends: &'a [Blend],
    #[cfg(bake_blend)]
    pub(crate) write_masks: &'a [ColorComponentFlags],
}

impl RasterKeyRef<'_> {
    fn to_key(&self) -> RasterKey {
        RasterKey {
            color_formats: SmallVec::from_slice(self.color_formats),
            depth_format: self.depth_format,
            stencil_format: self.stencil_format,
            view_mask: self.view_mask,
            constants: self.constants.into(),
            baked: self.baked,
            #[cfg(bake_blend)]
            blends: SmallVec::from_slice(self.blends),
            #[cfg(bake_blend)]
            write_masks: SmallVec::from_slice(self.write_masks),
        }
    }
}

impl Equivalent<RasterKey> for RasterKeyRef<'_> {
    fn equivalent(&self, key: &RasterKey) -> bool {
        *self == key.borrow()
    }
}

static PIPELINES: Global<PipelineManager> = Global::new();

#[track_caller]
pub(crate) fn pipelines() -> &'static PipelineManager {
    PIPELINES.get()
}

/// A lock's data even if a thread panicked while writing it: every write leaves the maps whole.
fn read<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(PoisonError::into_inner)
}

fn write<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(PoisonError::into_inner)
}

/// The pipelines of every pass, each pass behind locks of its own, all made through one cache.
pub(crate) struct PipelineManager {
    cache: vk::PipelineCache,
    /// Where `save_cache` writes `cache` (`Config::pipeline_cache`).
    cache_path: Option<&'static Path>,
    passes: Box<[PassPipelines]>,
}

/// A pass whose SPIR-V needs what the device lacks has no module and no pipeline.
struct PassPipelines {
    /// The features (or limits) the device lacks for the pass.
    missing: Vec<String>,
    /// Read-locked from before a pipeline is made until it is in `cache`; `reload_pass` holds it
    /// for writing through the rebuild, so no creation uses a module it destroys, and every
    /// pipeline made from the old module is rebuilt.
    module: RwLock<vk::ShaderModule>,
    cache: Cache,
}

/// A pass's pipelines by what they were made for: constant bytes (empty without constants,
/// made at init) or a raster key.
enum Cache {
    Compute(RwLock<HashMap<Box<[u8]>, vk::Pipeline>>),
    #[cfg(feature = "ray-tracing-pipeline")]
    RayTracing(RwLock<HashMap<Box<[u8]>, RaytracingPipeline>>),
    Raster(RwLock<HashMap<RasterKey, vk::Pipeline>>),
}

/// `Unsupported` when a pass of `map` needs a cargo feature lava was compiled without; reads
/// only the SPIR-V, so `init` checks it before it makes anything.
pub(crate) fn check_compiled(map: &PassMap) -> Result<()> {
    let compiled_out: Vec<String> = (map.passes.iter())
        .map(|pass| (pass, Features::of_spirv(pass.spirv)))
        .filter(|(_, needs)| !needs.compiled_out.is_empty())
        .map(|(pass, needs)| format!("`{}` needs `{}`", pass.name, needs.compiled_out.join("`, `")))
        .collect();
    match compiled_out.is_empty() {
        true => Ok(()),
        false => Err(Error::Unsupported(format!(
            "lava was compiled without cargo features passes need: {}",
            compiled_out.join("; ")
        ))),
    }
}

pub(crate) fn init(cache_path: Option<&'static Path>) -> Result<()> {
    let passes = pass_map().passes;
    let needs: Vec<_> = passes.iter().map(|pass| Features::of_spirv(pass.spirv)).collect();
    let features = *Ctx::features_ref();
    let missing: Vec<Vec<String>> = (passes.iter().zip(&needs))
        .map(|(pass, needs)| {
            let mut missing: Vec<String> = (needs.features.missing_from(&features).into_iter())
                .map(str::to_owned)
                .collect();
            if missing.is_empty() {
                missing.extend(missing_limits(pass));
            }
            missing
        })
        .collect();

    let device = Ctx::device();
    let cache = create_pipeline_cache(cache_path)?;
    let mut modules = Vec::with_capacity(passes.len());
    // Undoes what init made so far when a step fails.
    let undo = |modules: &[vk::ShaderModule], pipelines: &[vk::Pipeline]| unsafe {
        pipelines.iter().for_each(|p| device.destroy_pipeline(*p, None));
        modules.iter().for_each(|m| device.destroy_shader_module(*m, None));
        device.destroy_pipeline_cache(cache, None);
    };
    for (pass, missing) in passes.iter().zip(&missing) {
        let module = match missing.is_empty() {
            false => vk::ShaderModule::null(),
            true => match create_module(pass.spirv) {
                Ok(module) => module,
                Err(err) => {
                    undo(&modules, &[]);
                    return Err(err.context(format!("the shader module of pass `{}`", pass.name)));
                }
            },
        };
        modules.push(module);
    }
    // The pipelines without constants, made now, one call per kind.
    let unspecialized = |ray_tracing: bool| -> Vec<(vk::ShaderModule, &'static PassEntry, &'static [u8])> {
        (passes.iter().enumerate())
            .filter(|(i, pass)| {
                let kind = match pass.kind {
                    PassKind::Compute { .. } => !ray_tracing,
                    PassKind::RayTracing { .. } => ray_tracing,
                    _ => false,
                };
                kind && pass.constants.is_empty() && missing[*i].is_empty()
            })
            .map(|(i, pass)| (modules[i], pass, &[][..]))
            .collect()
    };
    let names = |batch: &[(vk::ShaderModule, &PassEntry, &[u8])]| {
        let names: Vec<&str> = batch.iter().map(|(_, pass, _)| pass.name).collect();
        names.join("`, `")
    };
    let compute_batch = unspecialized(false);
    let compute = create_compute_pipelines(cache, &compute_batch).map_err(|err| {
        undo(&modules, &[]);
        err.context(format!("the pipelines of `{}`", names(&compute_batch)))
    })?;
    #[cfg(feature = "ray-tracing-pipeline")]
    let mut ray_tracing = {
        let batch = unspecialized(true);
        let made = match batch.is_empty() {
            true => Vec::new(),
            false => create_raytracing_pipelines(cache, &batch).map_err(|err| {
                undo(&modules, &compute);
                err.context(format!("the pipelines of `{}`", names(&batch)))
            })?,
        };
        made.into_iter()
    };
    let mut compute = compute.into_iter();

    let passes: Box<[PassPipelines]> = (passes.iter().zip(missing).zip(modules))
        .map(|((pass, missing), module)| {
            let made = missing.is_empty() && pass.constants.is_empty();
            let cache = match pass.kind {
                PassKind::Compute { .. } => {
                    let mut map = HashMap::new();
                    if made {
                        map.insert(Box::default(), compute.next().unwrap());
                    }
                    Cache::Compute(RwLock::new(map))
                }
                #[cfg(feature = "ray-tracing-pipeline")]
                PassKind::RayTracing { .. } => {
                    let mut map = HashMap::new();
                    if made {
                        map.insert(Box::default(), ray_tracing.next().unwrap());
                    }
                    Cache::RayTracing(RwLock::new(map))
                }
                #[cfg(not(feature = "ray-tracing-pipeline"))]
                PassKind::RayTracing { .. } => {
                    unreachable!("a ray tracing pass declares `RayTracingKHR`, which needs `ray-tracing-pipeline`")
                }
                PassKind::RasterVertex { .. } | PassKind::RasterMesh { .. } => {
                    Cache::Raster(RwLock::new(HashMap::new()))
                }
            };
            PassPipelines {
                missing,
                module: RwLock::new(module),
                cache,
            }
        })
        .collect();
    PIPELINES.set(PipelineManager {
        cache,
        cache_path,
        passes,
    });
    Ok(())
}

/// Destroys every pass's pipelines and module and the pipeline cache, at `shutdown` (after
/// `save_cache_now`); ray tracing pipelines retire their binding tables.
///
/// # Safety
/// The device is idle and nothing records (lava is closed).
pub(crate) unsafe fn destroy() {
    let Some(manager) = (unsafe { PIPELINES.take() }) else {
        return;
    };
    let device = Ctx::device();
    let PipelineManager { cache, passes, .. } = *manager;
    for pass in passes {
        match pass.cache {
            Cache::Compute(map) => map.into_inner().unwrap_or_else(PoisonError::into_inner).into_values().for_each(destroy_pipeline),
            #[cfg(feature = "ray-tracing-pipeline")]
            Cache::RayTracing(map) => map.into_inner().unwrap_or_else(PoisonError::into_inner).into_values().for_each(destroy_ray_tracing),
            Cache::Raster(map) => map.into_inner().unwrap_or_else(PoisonError::into_inner).into_values().for_each(destroy_pipeline),
        }
        let module = pass.module.into_inner().unwrap_or_else(PoisonError::into_inner);
        if module != vk::ShaderModule::null() {
            unsafe { device.destroy_shader_module(module, None) };
        }
    }
    unsafe { device.destroy_pipeline_cache(cache, None) };
}

/// The device limits a pass exceeds, as what it lacks.
#[allow(unused_variables)]
fn missing_limits(pass: &PassEntry) -> Option<String> {
    #[cfg(feature = "ray-tracing-pipeline")]
    if let PassKind::RayTracing { max_recursion, .. } = pass.kind
        && let Some(properties) = Ctx::physical_device().ray_tracing_pipeline_properties
        && max_recursion > properties.max_ray_recursion_depth
    {
        return Some(format!(
            "a maxRayRecursionDepth of {max_recursion} (it has {})",
            properties.max_ray_recursion_depth
        ));
    }
    None
}

/// Replaces the SPIR-V of pass `pass_index` and every pipeline made from it, which must keep
/// the layout the bindings were generated from; if the driver rejects it, nothing is replaced.
/// Recordings that looked up a pipeline before keep using it; it is retired like any object.
pub fn reload_pass(pass_index: usize, spirv: &[u8]) -> Result<()> {
    let _call = crate::state::call();
    let manager = pipelines();
    let entry = &pass_map().passes[pass_index];
    let pass = &manager.passes[pass_index];
    if !pass.missing.is_empty() {
        return Ok(());
    }
    let needs = Features::of_spirv(spirv);
    let mut missing = needs.features.missing_from(Ctx::features_ref());
    missing.extend(needs.compiled_out);
    if !missing.is_empty() {
        return Err(Error::Unsupported(format!(
            "the new `{}` needs `{}`, which this device lacks or lava was compiled without",
            entry.name,
            missing.join("`, `")
        )));
    }
    // Held through the rebuild: no creation uses the old module meanwhile or starts from it after.
    let mut module = write(&pass.module);
    let new_module = create_module(spirv)?;
    let cache = manager.cache;
    let result = match &pass.cache {
        Cache::Compute(map) => rebuild(map, destroy_pipeline, |constants| {
            Ok(create_compute_pipelines(cache, &[(new_module, entry, constants)])?[0])
        })
        .map(retire),
        #[cfg(feature = "ray-tracing-pipeline")]
        Cache::RayTracing(map) => rebuild(map, destroy_ray_tracing, |constants| {
            let mut made = create_raytracing_pipelines(cache, &[(new_module, entry, constants)])?;
            Ok(made.remove(0))
        })
        .map(retire),
        Cache::Raster(map) => rebuild(map, destroy_pipeline, |key| {
            create_raster_pipeline(cache, new_module, &key.borrow(), entry)
        })
        .map(retire),
    };
    let device = Ctx::device();
    if let Err(err) = result {
        unsafe { device.destroy_shader_module(new_module, None) };
        return Err(err);
    }
    let old = std::mem::replace(&mut *module, new_module);
    // Pipelines don't need their module once created, and no creation holds this one.
    unsafe { device.destroy_shader_module(old, None) };
    Ok(())
}

/// Writes the pipeline cache to `Config::pipeline_cache` (nothing without one), through a
/// temporary file beside it, so an interrupted write leaves the last one; `lava::shutdown`
/// calls it too.
pub fn save_cache() -> Result<()> {
    let _call = crate::state::call();
    save_cache_now()
}

/// [`save_cache`] without a count, for `shutdown`.
pub(crate) fn save_cache_now() -> Result<()> {
    let Some((manager, path)) = PIPELINES.try_get().and_then(|m| Some((m, m.cache_path?))) else {
        return Ok(());
    };
    let saved = (|| -> Result<()> {
    let data = unsafe { Ctx::device().get_pipeline_cache_data(manager.cache)? };
    if let Some(dir) = path.parent().filter(|dir| !dir.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(".tmp");
    std::fs::write(&temporary, &data)?;
    std::fs::rename(&temporary, path)?;
    Ok(())
    })();
    saved.map_err(|err| err.context(format!("the pipeline cache {}", path.display())))
}

/// Whether `data` starts with the `VkPipelineCacheHeaderVersionOne` of a device with
/// `properties` (its fields are little-endian on every host).
pub(crate) fn cache_header_matches(data: &[u8], properties: &vk::PhysicalDeviceProperties) -> bool {
    let word = |i: usize| {
        let bytes = data.get(i * 4..i * 4 + 4)?;
        Some(u32::from_le_bytes(bytes.try_into().unwrap()))
    };
    word(0).is_some_and(|size| size >= 32 && size as usize <= data.len())
        && word(1) == Some(vk::PipelineCacheHeaderVersion::ONE.as_raw() as u32)
        && word(2) == Some(properties.vendor_id)
        && word(3) == Some(properties.device_id)
        && data.get(16..32) == Some(&properties.pipeline_cache_uuid[..])
}

/// The cache every pipeline is made through, with the data at `path` if it was made by this
/// device and driver (else a warning, and it starts empty).
fn create_pipeline_cache(path: Option<&Path>) -> Result<vk::PipelineCache> {
    let data = path.and_then(|path| match std::fs::read(path) {
        Ok(data) => Some(data),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => None,
        Err(err) => {
            tracing::warn!("ignoring the pipeline cache {}: {err}", path.display());
            None
        }
    });
    let data = data.filter(|data| {
        let matches = cache_header_matches(data, Ctx::device_properties());
        if !matches {
            tracing::warn!(
                "ignoring the pipeline cache {}: another device or driver made it",
                path.unwrap().display()
            );
        }
        matches
    });
    let device = Ctx::device();
    let create = |data: &[u8]| unsafe {
        device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default().initial_data(data), None)
    };
    match data {
        Some(data) => create(&data).or_else(|err| {
            tracing::warn!("ignoring the pipeline cache {}: {err}", path.unwrap().display());
            create(&[])
        }),
        None => create(&[]),
    }
    .map_err(Error::from)
}

impl PipelineManager {
    /// Only hand-made bindings reach a pass the device can't run: generated ones can't be made.
    fn pass(&self, pass_index: usize) -> (&PassPipelines, &'static PassEntry) {
        let (pass, entry) = (&self.passes[pass_index], &pass_map().passes[pass_index]);
        if !pass.missing.is_empty() {
            panic!(
                "pass `{}` needs `{}`, which this device lacks",
                entry.name,
                pass.missing.join("`, `")
            );
        }
        (pass, entry)
    }

    pub(crate) fn compute(&self, pass_index: usize, constants: &[u8]) -> vk::Pipeline {
        let (pass, entry) = self.pass(pass_index);
        let Cache::Compute(map) = &pass.cache else {
            panic!("pass `{}` isn't a compute pass", entry.name)
        };
        debug_assert_eq!(constants.len(), entry.constants_size, "`{}`'s constants", entry.name);
        let cache = self.cache;
        get_or_create(pass, entry, map, constants, |c: &[u8]| c.into(), |p| *p, destroy_pipeline, |module| {
            Ok(create_compute_pipelines(cache, &[(module, entry, constants)])?[0])
        })
    }

    #[cfg(feature = "ray-tracing-pipeline")]
    pub(crate) fn ray_tracing(&self, pass_index: usize, constants: &[u8]) -> BoundRayTracing {
        let (pass, entry) = self.pass(pass_index);
        let Cache::RayTracing(map) = &pass.cache else {
            panic!("pass `{}` isn't a ray tracing pass", entry.name)
        };
        debug_assert_eq!(constants.len(), entry.constants_size, "`{}`'s constants", entry.name);
        let cache = self.cache;
        let bound = RaytracingPipeline::bound;
        get_or_create(pass, entry, map, constants, |c: &[u8]| c.into(), bound, destroy_ray_tracing, |module| {
            Ok(create_raytracing_pipelines(cache, &[(module, entry, constants)])?.remove(0))
        })
    }

    pub(crate) fn raster(&self, key: &RasterKeyRef, pass_index: usize) -> vk::Pipeline {
        let (pass, entry) = self.pass(pass_index);
        let Cache::Raster(map) = &pass.cache else {
            panic!("pass `{}` isn't a raster pass", entry.name)
        };
        debug_assert_eq!(key.constants.len(), entry.constants_size, "`{}`'s constants", entry.name);
        let cache = self.cache;
        get_or_create(pass, entry, map, key, RasterKeyRef::to_key, |p| *p, destroy_pipeline, |module| {
            create_raster_pipeline(cache, module, key, entry)
        })
    }
}

/// The pipeline for `key`, made from the pass's module outside the map's lock on a miss; if
/// another thread made one meanwhile, that one is kept and this one destroyed. A failed creation
/// panics with the pass's name and leaves no lock poisoned.
#[allow(clippy::too_many_arguments)]
fn get_or_create<K: Hash + Eq, Q: Hash + Equivalent<K> + ?Sized, V, Out>(
    pass: &PassPipelines,
    entry: &PassEntry,
    map: &RwLock<HashMap<K, V>>,
    key: &Q,
    to_owned: impl FnOnce(&Q) -> K,
    out: impl Fn(&V) -> Out,
    destroy: impl FnOnce(V),
    create: impl FnOnce(vk::ShaderModule) -> Result<V>,
) -> Out {
    if let Some(pipeline) = read(map).get(key) {
        return out(pipeline);
    }
    // Held until the pipeline is in the map, so a reload waits for it and then rebuilds it.
    let module = read(&pass.module);
    if let Some(pipeline) = read(map).get(key) {
        return out(pipeline);
    }
    let created = create(*module).unwrap_or_else(|err| panic!("pass `{}`: {err}", entry.name));
    let mut map = write(map);
    if let Some(pipeline) = map.get(key) {
        let theirs = out(pipeline);
        drop(map);
        // Never handed out.
        destroy(created);
        return theirs;
    }
    note_new_pipeline(entry, map.len());
    let made = out(&created);
    map.insert(to_owned(key), created);
    made
}

#[cfg(feature = "ray-tracing-pipeline")]
impl From<RaytracingPipeline> for vk::Pipeline {
    fn from(pipeline: RaytracingPipeline) -> vk::Pipeline {
        pipeline.pipeline
    }
}

fn destroy_pipeline(pipeline: vk::Pipeline) {
    unsafe { Ctx::device().destroy_pipeline(pipeline, None) }
}

#[cfg(feature = "ray-tracing-pipeline")]
fn destroy_ray_tracing(pipeline: RaytracingPipeline) {
    destroy_pipeline(pipeline.into())
}

fn retire<P: Into<vk::Pipeline> + Send + 'static>(pipelines: Vec<P>) {
    if pipelines.is_empty() {
        return;
    }
    crate::lifetime::retire(move || pipelines.into_iter().for_each(|p| destroy_pipeline(p.into())));
}

/// Makes a new pipeline for every key of `map` and swaps them in, returning the replaced ones;
/// if one fails, the new ones are destroyed and nothing changes. The caller holds the pass's
/// module for writing, so no key is added meanwhile.
fn rebuild<K: Hash + Eq + Clone, V>(
    map: &RwLock<HashMap<K, V>>,
    destroy: impl Fn(V),
    mut create: impl FnMut(&K) -> Result<V>,
) -> Result<Vec<V>> {
    let keys: Vec<K> = read(map).keys().cloned().collect();
    let mut new = Vec::with_capacity(keys.len());
    for key in &keys {
        match create(key) {
            Ok(pipeline) => new.push(pipeline),
            Err(err) => {
                new.into_iter().for_each(destroy);
                return Err(err);
            }
        }
    }
    let mut map = write(map);
    let replaced = keys.into_iter().zip(new);
    Ok(replaced.filter_map(|(key, pipeline)| map.insert(key, pipeline)).collect())
}

const MANY_PIPELINES: usize = 16;

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

fn specialization<'a>(pass: &'a PassEntry, constants: &'a [u8]) -> vk::SpecializationInfo<'a> {
    vk::SpecializationInfo::default()
        .map_entries(pass.constants)
        .data(constants)
}

/// One compute pipeline per `(module, pass, constants)`, made with one call and named after
/// their passes.
fn create_compute_pipelines(
    cache: vk::PipelineCache,
    passes: &[(vk::ShaderModule, &PassEntry, &[u8])],
) -> Result<Vec<vk::Pipeline>> {
    if passes.is_empty() {
        return Ok(Vec::new());
    }
    let infos: Vec<_> = (passes.iter())
        .map(|(_, pass, constants)| specialization(pass, constants))
        .collect();
    let mut flags: Vec<_> = passes.iter().map(|_| heap_flags()).collect();
    let create_infos: Vec<_> = (passes.iter().zip(&infos).zip(&mut flags))
        .map(|(((module, pass, _), info), flags)| {
            let PassKind::Compute { entry } = pass.kind else {
                unreachable!("{} is a compute pass", pass.name)
            };
            let stage = make_shader_stage(entry, vk::ShaderStageFlags::COMPUTE, *module, info);
            vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .push_next(flags)
        })
        .collect();
    let device = Ctx::device();
    let pipelines = unsafe {
        device
            .create_compute_pipelines(cache, &create_infos, None)
            .map_err(|(partial, err)| {
                (partial.into_iter())
                    .filter(|p| *p != vk::Pipeline::null())
                    .for_each(destroy_pipeline);
                err
            })?
    };
    for (pipeline, (_, pass, _)) in pipelines.iter().zip(passes) {
        Functions::set_debug_name(pass.name, *pipeline);
    }
    Ok(pipelines)
}

/// One ray tracing pipeline per `(module, pass, constants)`, made with one call and named after
/// their passes.
#[cfg(feature = "ray-tracing-pipeline")]
fn create_raytracing_pipelines(
    cache: vk::PipelineCache,
    passes: &[(vk::ShaderModule, &PassEntry, &[u8])],
) -> Result<Vec<RaytracingPipeline>> {
    let infos: Vec<_> = (passes.iter())
        .map(|(_, pass, constants)| specialization(pass, constants))
        .collect();
    let descs: Vec<_> = (passes.iter().zip(&infos))
        .map(|((module, pass, _), info)| ray_tracing_desc(*module, pass, info))
        .collect();
    let pipelines = RaytracingPipeline::create_all(cache, &descs)?;
    for (pipeline, (_, pass, _)) in pipelines.iter().zip(passes) {
        Functions::set_debug_name(pass.name, pipeline.pipeline);
    }
    Ok(pipelines)
}

#[cfg(feature = "ray-tracing-pipeline")]
fn ray_tracing_desc<'a>(
    module: vk::ShaderModule,
    pass: &PassEntry,
    info: &'a vk::SpecializationInfo<'a>,
) -> RayTracingDesc<'a> {
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
    let mut stages = SmallVec::new();
    let mut groups = SmallVec::new();
    let mut stage = |entry: Option<&'static CStr>, flags| match entry {
        Some(entry) => {
            stages.push(make_shader_stage(entry, flags, module, info));
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
    RayTracingDesc {
        stages,
        groups,
        counts: [1, miss.len(), hit.len(), callable.len()].map(|n| n as u32),
        max_recursion,
    }
}

/// Every pipeline reads the descriptor heaps and has no layout.
pub(crate) fn heap_flags() -> vk::PipelineCreateFlags2CreateInfoKHR<'static> {
    vk::PipelineCreateFlags2CreateInfoKHR::default().flags(vk::PIPELINE_CREATE_2_DESCRIPTOR_HEAP_EXT)
}

/// A stage of `entry`.
fn make_shader_stage<'a>(
    entry: &'static CStr,
    stage: vk::ShaderStageFlags,
    module: vk::ShaderModule,
    constants: &'a vk::SpecializationInfo<'a>,
) -> vk::PipelineShaderStageCreateInfo<'a> {
    vk::PipelineShaderStageCreateInfo::default()
        .stage(stage)
        .module(module)
        .name(entry)
        .specialization_info(constants)
}

fn create_raster_pipeline(
    cache: vk::PipelineCache,
    module: vk::ShaderModule,
    key: &RasterKeyRef,
    pass: &PassEntry,
) -> Result<vk::Pipeline> {
    let _span = tracing::info_span!("create raster pipeline", pass = pass.name).entered();
    let c = specialization(pass, key.constants);
    let stages = match pass.kind {
        PassKind::RasterVertex { vertex, fragment } => {
            let mut stages = SmallVec::<[vk::PipelineShaderStageCreateInfo<'_>; 3]>::new();
            stages.push(make_shader_stage(vertex, vk::ShaderStageFlags::VERTEX, module, &c));
            if let Some(fragment) = fragment {
                stages.push(make_shader_stage(
                    fragment,
                    vk::ShaderStageFlags::FRAGMENT,
                    module,
                    &c,
                ));
            }
            stages
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
            if let Some(fragment) = fragment {
                stages.push(make_shader_stage(
                    fragment,
                    vk::ShaderStageFlags::FRAGMENT,
                    module,
                    &c,
                ));
            }
            stages
        }
        other => panic!("pass {other:?} is not a raster pass"),
    };

    let mut create_info = vk::GraphicsPipelineCreateInfo::default();
    // Draws set the topology dynamically, within the baked class where the device can't leave
    // it (`Baked::topology`).
    let topology = match Ctx::dynamic().topology_unrestricted {
        true => vk::PrimitiveTopology::TRIANGLE_LIST,
        false => key.baked.topology,
    };
    let ia = vk::PipelineInputAssemblyStateCreateInfo::default()
        .primitive_restart_enable(false)
        .topology(topology);
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
        .view_mask(key.view_mask);
    // State the device sets dynamically is ignored here; the rest is the key's (`Baked`).
    let features = Ctx::features_ref();
    let dynamic = Ctx::dynamic();
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
    if crate::command_buffer::on!(features, "depth-clip-control", depth_clip_control) {
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
    if crate::command_buffer::on!(features, "conservative-rasterization", conservative_rasterization) {
        rasterization_state = rasterization_state.push_next(&mut conservative);
    }
    #[cfg(bake_provoking_vertex)]
    let provoking_vertex = key.baked.provoking_vertex;
    #[cfg(not(bake_provoking_vertex))]
    let provoking_vertex = vk::ProvokingVertexModeEXT::FIRST_VERTEX;
    let mut provoking = vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT::default()
        .provoking_vertex_mode(provoking_vertex);
    if crate::command_buffer::on!(features, "provoking-vertex", provoking_vertex) {
        rasterization_state = rasterization_state.push_next(&mut provoking);
    }
    #[cfg(bake_depth_clip)]
    let depth_clip = key.baked.depth_clip;
    // Unused: the state is dynamic or the feature compiled out (then clamped primitives aren't
    // clipped, as without the extension).
    #[cfg(not(bake_depth_clip))]
    let depth_clip = true;
    let mut depth_clip = vk::PipelineRasterizationDepthClipStateCreateInfoEXT::default()
        .depth_clip_enable(depth_clip);
    if crate::command_buffer::on!(features, "depth-clip-enable", depth_clip_enable) {
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
            .create_graphics_pipelines(cache, &[create_info], None)
            .map_err(|(_, err)| err)?
    }[0];
    Functions::set_debug_name(pass.name, pipeline);
    Ok(pipeline)
}

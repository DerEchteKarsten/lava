//! Slang sessions and the compilation of one pass file into its passes (SPIR-V, description and warnings), shared by the build script and the hot reload, so both compile alike: pass names and their checks, entry point stages and attributes, ray tracing groups
use std::{collections::BTreeMap, ffi::CString, path::Path};

pub(crate) use shader_slang as slang;

use crate::describe::{self, PassDesc};
use crate::reflect::{self, CheckedAttributes, CheckedEntryPoint, CheckedName, CheckedShader};
use crate::{Optimization, SHADER_DIR};

pub struct Options<'a> {
    /// Absolute include directories; [`SHADER_DIR`] is always searched too.
    pub search_paths: &'a [&'a str],
    pub defines: &'a [(&'a str, &'a str)],
    pub optimization: Optimization,
    /// Emit Slang's debug information (source-level debugging in RenderDoc or Nsight).
    pub debug_info: bool,
}

/// A session that compiles passes, named by their absolute paths, to SPIR-V 1.6.
pub fn session(global: &slang::GlobalSession, options: &Options) -> slang::Session {
    let paths: Vec<CString> = options
        .search_paths
        .iter()
        .copied()
        .chain([SHADER_DIR])
        .map(|path| CString::new(path).unwrap())
        .collect();
    let search_paths: Vec<_> = paths.iter().map(|path| path.as_ptr()).collect();
    let targets = [slang::TargetDesc::default()
        .format(slang::CompileTarget::Spirv)
        .profile(global.find_profile("spirv_1_6"))];
    let optimization = match options.optimization {
        Optimization::None => slang::OptimizationLevel::None,
        Optimization::Default => slang::OptimizationLevel::Default,
        Optimization::Maximal => slang::OptimizationLevel::Maximal,
    };
    let mut compiler = slang::CompilerOptions::default()
        .optimization(optimization)
        .vulkan_use_entry_point_name(true)
        .matrix_layout_column(true)
        // "Profile implicitly upgraded" for every capability a pass uses: lava reads them from
        // the SPIR-V (`features.rs`), so they need no warning.
        .disable_warning("41012")
        // Shaders index lava's descriptor heaps directly (`SPV_EXT_descriptor_heap`).
        .capability(global.find_capability("spvDescriptorHeapEXT"));
    if options.debug_info {
        compiler = compiler.debug_information(slang::DebugInfoLevel::Standard);
    }
    for (name, value) in options.defines {
        compiler = compiler.macro_define(name, value);
    }
    global
        .create_session(
            &slang::SessionDesc::default()
                .targets(&targets)
                .search_paths(&search_paths)
                .options(&compiler),
        )
        .expect("failed to create a slang session")
}

/// A pass of a file: all its entry points, or one of several compute entry points.
pub struct CompiledPass {
    /// The generated struct's name (`bvh_cull.slang` is `BvhCull`; the entry point `main` of a
    /// file of several compute entry points `BvhCullMain`).
    pub name: String,
    /// The entry point it was linked with alone, for a file of several compute entry points.
    pub entry: Option<String>,
    pub desc: PassDesc,
    pub spirv: Vec<u8>,
}

/// What compiling one pass file gave: its passes, or every error, and Slang's warnings.
pub struct CompiledFile {
    pub passes: Result<Vec<CompiledPass>, Vec<String>>,
    pub warnings: Vec<String>,
}

/// The pass name of a file stem: `bvh_cull` is `BvhCull`. Refused unless it makes a Rust
/// identifier: ASCII letters, digits and `_`, not starting with a digit.
pub fn pass_name(stem: &str) -> Result<String, String> {
    let valid = !stem.is_empty()
        && stem.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !stem.starts_with(|c: char| c.is_ascii_digit());
    let name: String = stem.split('_').map(describe::capitalize_first).collect();
    match valid && !name.is_empty() {
        true => Ok(name),
        false => Err(format!(
            "the pass name `{stem}` isn't a Rust identifier: name the file with ASCII letters, \
             digits and `_`, not starting with a digit"
        )),
    }
}

/// Compiles the pass file `source` (an absolute path) into its passes: one with all its entry
/// points, or, for a file of several compute entry points, one per entry point, linked alone
/// and named after the file and the entry point.
pub fn compile_file(session: &slang::Session, source: &str) -> CompiledFile {
    let mut warnings = Vec::new();
    let passes = compile_passes(session, source, &mut warnings);
    CompiledFile { passes, warnings }
}

fn compile_passes(
    session: &slang::Session,
    source: &str,
    warnings: &mut Vec<String>,
) -> Result<Vec<CompiledPass>, Vec<String>> {
    let stem = Path::new(source)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let file_name = pass_name(stem).map_err(|e| vec![e])?;
    let (module, diagnostics) = reflect::load_module(session, source).map_err(|e| vec![e])?;
    warnings.extend(diagnostics);
    // The stages of every entry point, from the module linked whole.
    let (whole, diagnostics) = link(session, &module, None).map_err(|e| vec![e])?;
    warnings.extend(diagnostics);
    let layout = whole.layout(0).map_err(|e| vec![e.to_string()])?;
    let stages: Vec<(String, slang::Stage)> = (layout.checked_entry_points())
        .map(|ep| (ep.checked_name().unwrap_or("?").to_string(), ep.stage()))
        .collect();
    let split = stages.len() > 1 && stages.iter().all(|(_, s)| *s == slang::Stage::Compute);
    let targets: Vec<(String, Option<String>)> = match split {
        false => vec![(file_name, None)],
        true => (stages.iter())
            .map(|(entry, _)| match pass_name(entry) {
                Ok(name) => Ok((file_name.clone() + &name, Some(entry.clone()))),
                Err(_) => Err(vec![format!(
                    "the compute entry point `{entry}` doesn't make a pass name: name it with \
                     ASCII letters, digits and `_`"
                )]),
            })
            .collect::<Result<_, _>>()?,
    };
    let mut passes = Vec::new();
    let mut errors = Vec::new();
    for (name, entry) in targets {
        let program = match &entry {
            None => Ok(whole.clone()),
            Some(entry) => link(session, &module, Some(entry)).map(|(program, diagnostics)| {
                warnings.extend(diagnostics);
                program
            }),
        };
        let compiled = program.and_then(|program| {
            let layout = program.layout(0).map_err(|e| e.to_string())?;
            let desc = describe::describe(layout, &module).map_err(|e| e.join("\n"))?;
            let (spirv, diagnostics) = reflect::target_code(&program, 0)?;
            warnings.extend(diagnostics);
            Ok(CompiledPass {
                name: name.clone(),
                entry: entry.clone(),
                desc,
                spirv,
            })
        });
        match compiled {
            Ok(pass) => passes.push(pass),
            Err(e) => match &entry {
                Some(entry) => errors.push(format!("entry point `{entry}` ({name}):\n{e}")),
                None => errors.push(e),
            },
        }
    }
    match errors.is_empty() {
        true => Ok(passes),
        false => Err(errors),
    }
}

/// `module` with all its entry points, or only `entry`, linked.
fn link(
    session: &slang::Session,
    module: &slang::Module,
    entry: Option<&str>,
) -> Result<(slang::ComponentType, Option<String>), String> {
    use slang::Downcast;
    let mut components: Vec<slang::ComponentType> = vec![module.downcast().clone()];
    match entry {
        None => components.extend(module.entry_points().map(|entry| entry.downcast().clone())),
        Some(name) => {
            let entry = (module.find_entry_point_by_name(name))
                .ok_or(format!("no entry point `{name}`"))?;
            components.push(entry.downcast().clone());
        }
    }
    let composite = session
        .create_composite_component_type(&components)
        .map_err(|e| e.to_string())?;
    reflect::link(&composite)
}

/// The stage of an entry point lava has passes for. Declared in the order of their names, so a
/// sorted list of a pass's stages matches the patterns of `describe`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Stage {
    /// Slang's amplification stage.
    Task,
    AnyHit,
    Callable,
    ClosestHit,
    Compute,
    Fragment,
    Intersection,
    Mesh,
    Miss,
    RayGen,
    Vertex,
}

impl Stage {
    pub fn of(stage: slang::Stage) -> Result<Self, String> {
        use slang::Stage as S;
        Ok(match stage {
            S::Vertex => Self::Vertex,
            S::Fragment => Self::Fragment,
            S::Mesh => Self::Mesh,
            S::Amplification => Self::Task,
            S::Compute => Self::Compute,
            S::RayGeneration => Self::RayGen,
            S::AnyHit => Self::AnyHit,
            S::ClosestHit => Self::ClosestHit,
            S::Miss => Self::Miss,
            S::Intersection => Self::Intersection,
            S::Callable => Self::Callable,
            other => return Err(format!("the stage {other:?} has no lava pass")),
        })
    }

    /// Its name in messages and layout hashes.
    pub fn name(self) -> &'static str {
        match self {
            Self::Vertex => "vertex",
            Self::Fragment => "fragment",
            Self::Mesh => "mesh",
            Self::Task => "amplification",
            Self::Compute => "compute",
            Self::RayGen => "raygen",
            Self::AnyHit => "any_hit",
            Self::ClosestHit => "closest_hit",
            Self::Miss => "miss",
            Self::Intersection => "intersection",
            Self::Callable => "callable",
        }
    }

    /// A stage of ray tracing pipelines.
    pub fn is_ray_tracing(self) -> bool {
        matches!(
            self,
            Self::RayGen
                | Self::Miss
                | Self::ClosestHit
                | Self::AnyHit
                | Self::Intersection
                | Self::Callable
        )
    }
}

/// The integer argument of the user attribute `name` (`Group`, `MaxRecursion`).
pub fn entry_attribute(entry: &slang::reflection::EntryPoint, name: &str) -> Option<i32> {
    entry
        .checked_function()?
        .checked_user_attributes()
        .find(|a| a.checked_name() == Some(name))?
        .argument_value_int(0)
}

pub struct Entry<'a> {
    pub stage: Stage,
    pub name: &'a str,
    pub group: Option<i32>,
    pub max_recursion: Option<i32>,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct HitGroupEntries {
    pub closest_hit: Option<String>,
    pub any_hit: Option<String>,
    pub intersection: Option<String>,
}

/// A ray tracing pass's shader groups in shader binding table order.
#[derive(Debug, PartialEq, Eq)]
pub struct RayTracingGroups {
    pub raygen: String,
    pub miss: Vec<String>,
    pub hit: Vec<HitGroupEntries>,
    pub callable: Vec<String>,
    pub max_recursion: u32,
}

/// Groups the entry points of a ray tracing pass by their `Group` index (0 without it), dense
/// per region; `MaxRecursion` is only allowed on the single raygen.
pub fn ray_tracing_groups(entries: &[Entry]) -> Result<RayTracingGroups, String> {
    let mut raygen = None;
    let mut max_recursion = 1;
    let mut miss = BTreeMap::new();
    let mut hit: BTreeMap<i32, HitGroupEntries> = BTreeMap::new();
    let mut callable = BTreeMap::new();
    for entry in entries {
        let name = entry.name.to_string();
        let index = entry.group.unwrap_or(0);
        if index < 0 {
            return Err(format!("`{}` has the negative group {index}", entry.name));
        }
        if let Some(depth) = entry.max_recursion {
            if entry.stage != Stage::RayGen {
                return Err(format!(
                    "`{}` isn't a raygen but has `MaxRecursion`",
                    entry.name
                ));
            }
            if !(0..=31).contains(&depth) {
                return Err(format!("`MaxRecursion({depth})` isn't between 0 and 31"));
            }
            max_recursion = depth as u32;
        }
        let taken = |region: &str| format!("two {region} entry points have group {index}");
        match entry.stage {
            Stage::RayGen => {
                if entry.group.is_some() {
                    return Err(format!("the raygen `{}` has a `Group`", entry.name));
                }
                if raygen.replace(name).is_some() {
                    return Err("a ray tracing pass has exactly one raygen".into());
                }
            }
            Stage::Miss => {
                if miss.insert(index, name).is_some() {
                    return Err(taken("miss"));
                }
            }
            Stage::Callable => {
                if callable.insert(index, name).is_some() {
                    return Err(taken("callable"));
                }
            }
            stage => {
                let group = hit.entry(index).or_default();
                let slot = match stage {
                    Stage::ClosestHit => &mut group.closest_hit,
                    Stage::AnyHit => &mut group.any_hit,
                    Stage::Intersection => &mut group.intersection,
                    other => {
                        return Err(format!("`{}` has the stage {}", entry.name, other.name()));
                    }
                };
                if slot.replace(name).is_some() {
                    let stage = stage.name();
                    return Err(format!("hit group {index} has two {stage} entry points"));
                }
            }
        }
    }
    fn dense<T>(region: &str, groups: BTreeMap<i32, T>) -> Result<Vec<T>, String> {
        groups
            .into_iter()
            .enumerate()
            .map(|(i, (index, group))| {
                if index as usize == i {
                    Ok(group)
                } else {
                    Err(format!("the {region} groups skip group {i}"))
                }
            })
            .collect()
    }
    Ok(RayTracingGroups {
        raygen: raygen.ok_or("a ray tracing pass needs a raygen")?,
        miss: dense("miss", miss)?,
        hit: dense("hit", hit)?,
        callable: dense("callable", callable)?,
        max_recursion,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(stage: Stage, name: &'static str, group: Option<i32>) -> Entry<'static> {
        Entry {
            stage,
            name,
            group,
            max_recursion: None,
        }
    }

    #[test]
    fn entries_are_grouped_by_index_per_region() {
        let mut raygen = entry(Stage::RayGen, "rg", None);
        raygen.max_recursion = Some(2);
        let groups = ray_tracing_groups(&[
            entry(Stage::Miss, "shadow", Some(1)),
            entry(Stage::ClosestHit, "sphere_hit", Some(1)),
            raygen,
            entry(Stage::Miss, "sky", None),
            entry(Stage::AnyHit, "holes", None),
            entry(Stage::Intersection, "sphere", Some(1)),
            entry(Stage::ClosestHit, "triangle_hit", Some(0)),
            entry(Stage::Callable, "colour", None),
        ])
        .unwrap();
        assert_eq!(groups.miss, ["sky", "shadow"]);
        assert_eq!(
            groups.hit,
            [
                HitGroupEntries {
                    closest_hit: Some("triangle_hit".into()),
                    any_hit: Some("holes".into()),
                    intersection: None,
                },
                HitGroupEntries {
                    closest_hit: Some("sphere_hit".into()),
                    any_hit: None,
                    intersection: Some("sphere".into()),
                },
            ]
        );
        assert_eq!(groups.callable, ["colour"]);
        assert_eq!(groups.max_recursion, 2);
    }

    #[test]
    fn malformed_groups_are_refused() {
        let rg = || entry(Stage::RayGen, "rg", None);
        let refused = |entries: &[Entry]| ray_tracing_groups(entries).unwrap_err();
        assert!(refused(&[entry(Stage::Miss, "m", None)]).contains("needs a raygen"));
        assert!(refused(&[rg(), entry(Stage::RayGen, "rg2", None)]).contains("exactly one raygen"));
        assert!(
            refused(&[
                rg(),
                entry(Stage::Miss, "a", None),
                entry(Stage::Miss, "b", None)
            ])
            .contains("two miss")
        );
        assert!(refused(&[rg(), entry(Stage::Miss, "a", Some(1))]).contains("skip group 0"));
        assert!(
            refused(&[
                rg(),
                entry(Stage::ClosestHit, "a", None),
                entry(Stage::ClosestHit, "b", None)
            ])
            .contains("two closest_hit")
        );
        assert!(refused(&[rg(), entry(Stage::Callable, "c", Some(-1))]).contains("negative"));
        let mut miss = entry(Stage::Miss, "m", None);
        miss.max_recursion = Some(2);
        assert!(refused(&[rg(), miss]).contains("isn't a raygen"));
        let mut deep = rg();
        deep.max_recursion = Some(32);
        assert!(refused(&[deep]).contains("between 0 and 31"));
        let mut flat = rg();
        flat.max_recursion = Some(0);
        assert_eq!(
            ray_tracing_groups(&[flat]).unwrap().max_recursion,
            0,
            "Vulkan allows 0"
        );
    }
}

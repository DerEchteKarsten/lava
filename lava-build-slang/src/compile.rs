//! Slang session and pass linking shared by the build script and the hot reload, plus the reflection both read: entry point stages and attributes, ray tracing groups, layout hashes
use std::{collections::BTreeMap, ffi::CString};

/// The Slang bindings, for the sessions and programs these functions take and return.
pub use shader_slang as slang;

use crate::{Optimization, SLANG_DIR};

/// How a pass compiles: what the build script was given.
pub struct Options<'a> {
    /// Absolute include directories; lava-build-slang's own `slang/` is always searched too.
    pub search_paths: &'a [&'a str],
    pub defines: &'a [(&'a str, &'a str)],
    pub optimization: Optimization,
}

/// A session that compiles a pass, named by its absolute path, to SPIR-V 1.6.
pub fn session(global: &slang::GlobalSession, options: &Options) -> slang::Session {
    let paths: Vec<CString> = options
        .search_paths
        .iter()
        .copied()
        .chain([SLANG_DIR])
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
        .matrix_layout_column(true);
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

/// A hash of what the generated bindings of a pass depend on: its push constant fields (name,
/// offset, size), its specialization constants (id, size) and its entry points (stage, name).
/// A reload whose hash differs from the baked one would run against stale bindings.
pub fn layout_hash(layout: &slang::reflection::Shader) -> u64 {
    use slang::ParameterCategory as Cat;
    use std::hash::{DefaultHasher, Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    for parameter in layout.parameters() {
        match parameter.category() {
            Some(Cat::PushConstantBuffer) => {
                let Some(fields) = parameter
                    .type_layout()
                    .and_then(|t| t.element_type_layout())
                else {
                    continue;
                };
                fields.size(Cat::Uniform).hash(&mut hasher);
                for field in fields.fields() {
                    field.variable().and_then(|v| v.name()).hash(&mut hasher);
                    field.offset(Cat::Uniform).hash(&mut hasher);
                    field
                        .type_layout()
                        .map(|t| t.size(Cat::Uniform))
                        .hash(&mut hasher);
                }
            }
            Some(Cat::SpecializationConstant) => {
                parameter
                    .offset(Cat::SpecializationConstant)
                    .hash(&mut hasher);
                parameter
                    .type_layout()
                    .map(|t| t.size(Cat::Uniform))
                    .hash(&mut hasher);
            }
            _ => {}
        }
    }
    for entry in layout.entry_points() {
        format!("{:?}", entry.stage()).hash(&mut hasher);
        entry.name().hash(&mut hasher);
        // The ray tracing groups, which the pass table bakes.
        entry_attribute(entry, "Group").hash(&mut hasher);
        entry_attribute(entry, "MaxRecursion").hash(&mut hasher);
    }
    hasher.finish()
}

/// The stage of an entry point as the pass table names it.
pub fn stage_name(stage: slang::Stage) -> &'static str {
    use slang::Stage;
    match stage {
        Stage::Vertex => "vertex",
        Stage::Fragment => "fragment",
        Stage::Mesh => "mesh",
        Stage::Amplification => "amplification",
        Stage::Compute => "compute",
        Stage::RayGeneration => "raygen",
        Stage::AnyHit => "any_hit",
        Stage::ClosestHit => "closest_hit",
        Stage::Miss => "miss",
        Stage::Intersection => "intersection",
        Stage::Callable => "callable",
        other => panic!("unsupported stage {other:?}"),
    }
}

/// The stages of a ray tracing pass.
pub const RAY_TRACING_STAGES: [&str; 6] = [
    "raygen",
    "miss",
    "closest_hit",
    "any_hit",
    "intersection",
    "callable",
];

/// The integer argument of the user attribute `name` (`Group`, `MaxRecursion` in
/// `ray_tracing.slang`) on an entry point.
pub fn entry_attribute(entry: &slang::reflection::EntryPoint, name: &str) -> Option<i32> {
    entry
        .function()?
        .user_attributes()
        .find(|a| a.name() == Some(name))?
        .argument_value_int(0)
}

/// An entry point of a pass with its `Group` and `MaxRecursion` attributes (ray tracing only).
pub struct Entry<'a> {
    pub stage: &'static str,
    pub name: &'a str,
    pub group: Option<i32>,
    pub max_recursion: Option<i32>,
}

/// The entry points of one hit group.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct HitGroupEntries {
    pub closest_hit: Option<String>,
    pub any_hit: Option<String>,
    pub intersection: Option<String>,
}

/// A ray tracing pass's shader groups, in shader binding table order: each region's groups by
/// their `Group` index.
#[derive(Debug, PartialEq, Eq)]
pub struct RayTracingGroups {
    pub raygen: String,
    pub miss: Vec<String>,
    pub hit: Vec<HitGroupEntries>,
    pub callable: Vec<String>,
    pub max_recursion: u32,
}

/// Groups the entry points of a ray tracing pass: exactly one raygen; miss, hit and callable
/// groups indexed 0, 1, 2, ... per region by `Group` (0 without it), where hit entry points of
/// the same index form one group with each stage at most once; `MaxRecursion` only on the
/// raygen, between 1 and 31 (1 without it).
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
            if entry.stage != "raygen" {
                return Err(format!("`{}` isn't a raygen but has `MaxRecursion`", entry.name));
            }
            if !(1..=31).contains(&depth) {
                return Err(format!("`MaxRecursion({depth})` isn't between 1 and 31"));
            }
            max_recursion = depth as u32;
        }
        let taken = |region: &str| format!("two {region} entry points have group {index}");
        match entry.stage {
            "raygen" => {
                if entry.group.is_some() {
                    return Err(format!("the raygen `{}` has a `Group`", entry.name));
                }
                if raygen.replace(name).is_some() {
                    return Err("a ray tracing pass has exactly one raygen".into());
                }
            }
            "miss" => {
                if miss.insert(index, name).is_some() {
                    return Err(taken("miss"));
                }
            }
            "callable" => {
                if callable.insert(index, name).is_some() {
                    return Err(taken("callable"));
                }
            }
            stage => {
                let group = hit.entry(index).or_default();
                let slot = match stage {
                    "closest_hit" => &mut group.closest_hit,
                    "any_hit" => &mut group.any_hit,
                    "intersection" => &mut group.intersection,
                    other => return Err(format!("`{}` has the stage {other}", entry.name)),
                };
                if slot.replace(name).is_some() {
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

/// Whether a SPIR-V module declares the `RayQueryKHR` capability, which needs
/// `Features::ray_query`.
pub fn uses_ray_query(spirv: &[u8]) -> bool {
    const OP_CAPABILITY: u32 = 17;
    const RAY_QUERY_KHR: u32 = 4472;
    let words: Vec<u32> = spirv
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
        .collect();
    // After the five header words, each instruction starts with its word count and opcode.
    let mut i = 5;
    while i < words.len() {
        let (count, opcode) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        if opcode == OP_CAPABILITY && words.get(i + 1) == Some(&RAY_QUERY_KHR) {
            return true;
        }
        if count == 0 {
            return false;
        }
        i += count;
    }
    false
}

/// The pass at the absolute path `source` with all its entry points, linked.
pub fn link(session: &slang::Session, source: &str) -> slang::Result<slang::ComponentType> {
    let module = session.load_module(source)?;
    let mut components: Vec<slang::ComponentType> = vec![module.clone().into()];
    components.extend(module.entry_points().map(Into::into));
    session.create_composite_component_type(&components)?.link()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(stage: &'static str, name: &'static str, group: Option<i32>) -> Entry<'static> {
        Entry {
            stage,
            name,
            group,
            max_recursion: None,
        }
    }

    #[test]
    fn a_raygen_alone_makes_one_group() {
        let groups = ray_tracing_groups(&[entry("raygen", "main", None)]).unwrap();
        assert_eq!(groups.raygen, "main");
        assert!(groups.miss.is_empty() && groups.hit.is_empty() && groups.callable.is_empty());
        assert_eq!(groups.max_recursion, 1);
    }

    #[test]
    fn entries_are_grouped_by_index_per_region() {
        let mut raygen = entry("raygen", "rg", None);
        raygen.max_recursion = Some(2);
        let groups = ray_tracing_groups(&[
            entry("miss", "shadow", Some(1)),
            entry("closest_hit", "sphere_hit", Some(1)),
            raygen,
            entry("miss", "sky", None),
            entry("any_hit", "holes", None),
            entry("intersection", "sphere", Some(1)),
            entry("closest_hit", "triangle_hit", Some(0)),
            entry("callable", "colour", None),
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
        let rg = || entry("raygen", "rg", None);
        let refused = |entries: &[Entry]| ray_tracing_groups(entries).unwrap_err();
        assert!(refused(&[entry("miss", "m", None)]).contains("needs a raygen"));
        assert!(refused(&[rg(), entry("raygen", "rg2", None)]).contains("exactly one raygen"));
        assert!(refused(&[rg(), entry("miss", "a", None), entry("miss", "b", None)]).contains("two miss"));
        assert!(refused(&[rg(), entry("miss", "a", Some(1))]).contains("skip group 0"));
        assert!(
            refused(&[rg(), entry("closest_hit", "a", None), entry("closest_hit", "b", None)])
                .contains("two closest_hit")
        );
        assert!(refused(&[rg(), entry("callable", "c", Some(-1))]).contains("negative"));
        let mut miss = entry("miss", "m", None);
        miss.max_recursion = Some(2);
        assert!(refused(&[rg(), miss]).contains("isn't a raygen"));
        let mut deep = rg();
        deep.max_recursion = Some(32);
        assert!(refused(&[deep]).contains("between 1 and 31"));
    }

    fn module(instructions: &[&[u32]]) -> Vec<u8> {
        let header = [0x0723_0203, 0x0001_0600, 0, 100, 0];
        header
            .iter()
            .chain(instructions.iter().flat_map(|i| i.iter()))
            .flat_map(|w| w.to_le_bytes())
            .collect()
    }

    #[test]
    fn ray_query_capabilities_are_found() {
        let shader = [2 << 16 | 17, 1];
        let ray_query = [2 << 16 | 17, 4472];
        // `OpExtension "SPV_KHR_ray_query"`, whose text isn't a capability.
        let extension = [5 << 16 | 10, 4472, 0, 0, 0];
        assert!(uses_ray_query(&module(&[&shader, &ray_query])));
        assert!(!uses_ray_query(&module(&[&extension, &shader])));
        assert!(!uses_ray_query(&module(&[])));
    }
}

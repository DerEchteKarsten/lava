//! Slang session and pass linking shared by the build script and the hot reload, plus the reflection both read: entry point stages and attributes, ray tracing groups, layout hashes
use std::{collections::BTreeMap, ffi::CString};

pub use shader_slang as slang;

use crate::reflect::{
    CheckedAttributes, CheckedEntryPoint, CheckedName, CheckedShader, CheckedTypeLayout,
    CheckedVariableLayout,
};
use crate::{Optimization, SLANG_DIR};

pub struct Options<'a> {
    /// Absolute include directories; [`SLANG_DIR`] is always searched too.
    pub search_paths: &'a [&'a str],
    pub defines: &'a [(&'a str, &'a str)],
    pub optimization: Optimization,
}

/// A session that compiles passes, named by their absolute paths, to SPIR-V 1.6.
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

/// A hash of what the generated bindings of a pass depend on: its push constants,
/// specialization constants and entry points.
pub fn layout_hash(layout: &slang::reflection::Shader) -> u64 {
    use slang::ParameterCategory as Cat;
    use std::hash::{DefaultHasher, Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    for parameter in layout.checked_parameters() {
        match parameter.category() {
            Cat::PushConstantBuffer => {
                let Some(fields) = parameter
                    .checked_type_layout()
                    .and_then(|t| t.checked_element_type_layout())
                else {
                    continue;
                };
                fields.size(Cat::Uniform).hash(&mut hasher);
                for field in fields.checked_fields() {
                    field
                        .variable()
                        .and_then(|v| v.checked_name())
                        .hash(&mut hasher);
                    field.offset(Cat::Uniform).hash(&mut hasher);
                    field
                        .checked_type_layout()
                        .map(|t| t.size(Cat::Uniform))
                        .hash(&mut hasher);
                }
            }
            Cat::SpecializationConstant => {
                parameter
                    .offset(Cat::SpecializationConstant)
                    .hash(&mut hasher);
                parameter
                    .checked_type_layout()
                    .map(|t| t.size(Cat::Uniform))
                    .hash(&mut hasher);
            }
            _ => {}
        }
    }
    for entry in layout.checked_entry_points() {
        format!("{:?}", entry.stage()).hash(&mut hasher);
        entry.checked_name().hash(&mut hasher);
        entry_attribute(entry, "Group").hash(&mut hasher);
        entry_attribute(entry, "MaxRecursion").hash(&mut hasher);
    }
    hasher.finish()
}

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

pub const RAY_TRACING_STAGES: [&str; 6] = [
    "raygen",
    "miss",
    "closest_hit",
    "any_hit",
    "intersection",
    "callable",
];

/// The integer argument of the user attribute `name` (`Group`, `MaxRecursion`).
pub fn entry_attribute(entry: &slang::reflection::EntryPoint, name: &str) -> Option<i32> {
    entry
        .checked_function()?
        .checked_user_attributes()
        .find(|a| a.checked_name() == Some(name))?
        .argument_value_int(0)
}

pub struct Entry<'a> {
    pub stage: &'static str,
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
            if entry.stage != "raygen" {
                return Err(format!(
                    "`{}` isn't a raygen but has `MaxRecursion`",
                    entry.name
                ));
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

/// The pass at the absolute path `source` with all its entry points, linked.
pub fn link(session: &slang::Session, source: &str) -> slang::Result<slang::ComponentType> {
    let module = session.load_module(source)?;
    use slang::Downcast;
    let mut components: Vec<slang::ComponentType> = vec![module.downcast().clone()];
    components.extend(module.entry_points().map(|entry| entry.downcast().clone()));
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
        assert!(
            refused(&[rg(), entry("miss", "a", None), entry("miss", "b", None)])
                .contains("two miss")
        );
        assert!(refused(&[rg(), entry("miss", "a", Some(1))]).contains("skip group 0"));
        assert!(
            refused(&[
                rg(),
                entry("closest_hit", "a", None),
                entry("closest_hit", "b", None)
            ])
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
}

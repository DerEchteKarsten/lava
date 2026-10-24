//! lava's GPU tests as an ordinary lava user: the test passes in `shaders/` compiled by lava-build-slang, and checks of the pass map it generates

pub mod bindings {
    lava::include_passes!();
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;

    use lava::pipeline::PassKind;

    use crate::bindings::PASS_MAP;

    #[test]
    fn pass_map_entries_are_consistent() {
        for pass in PASS_MAP.passes {
            let end = pass.constants.iter().map(|c| c.offset as usize + c.size);
            assert_eq!(end.max().unwrap_or(0), pass.constants_size, "{}", pass.name);
            let names: Vec<&CStr> = match pass.kind {
                PassKind::Compute { entry, .. } => vec![entry],
                PassKind::RasterVertex { vertex, fragment } => {
                    [vertex].into_iter().chain(fragment).collect()
                }
                PassKind::RasterMesh {
                    task,
                    mesh,
                    fragment,
                } => task.into_iter().chain([mesh]).chain(fragment).collect(),
                PassKind::RayTracing {
                    raygen,
                    miss,
                    hit,
                    callable,
                    max_recursion,
                } => {
                    assert!((0..=31).contains(&max_recursion));
                    let hit = hit
                        .iter()
                        .flat_map(|g| [g.closest_hit, g.any_hit, g.intersection])
                        .flatten();
                    [raygen]
                        .into_iter()
                        .chain(miss.iter().copied())
                        .chain(hit)
                        .chain(callable.iter().copied())
                        .collect()
                }
            };
            for name in names {
                assert!(!name.is_empty(), "an empty entry point in {}", pass.name);
            }
        }
    }

    #[test]
    fn every_pass_has_compiled_spirv() {
        for pass in PASS_MAP.passes {
            assert!(
                pass.spirv.len() >= 4 && pass.spirv.len() % 4 == 0,
                "{} is not SPIR-V",
                pass.name
            );
            let magic = u32::from_le_bytes(pass.spirv[..4].try_into().unwrap());
            assert_eq!(
                magic, 0x0723_0203,
                "{} lacks the SPIR-V magic number",
                pass.name
            );
        }
    }
}

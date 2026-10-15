//! lava's GPU tests as an ordinary lava user: the test passes in `shaders/` compiled by lava-build-slang, and checks of the pass map it generates
// Needed to call `RasterBuilder`'s attachment and draw methods, which return it one larger.
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

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
        let (mut compute, mut raster, mut raytracing) = (Vec::new(), Vec::new(), Vec::new());
        let (mut specialized_compute, mut specialized_raytracing) = (Vec::new(), Vec::new());
        for pass in PASS_MAP.passes {
            let specialized = !pass.constants.is_empty();
            // The constants fill their struct exactly.
            let end = pass.constants.iter().map(|c| c.offset as usize + c.size);
            assert_eq!(end.max().unwrap_or(0), pass.constants_size, "{}", pass.name);
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
                    hit,
                    callable,
                    max_recursion,
                } => {
                    if specialized {
                        &mut specialized_raytracing
                    } else {
                        &mut raytracing
                    }
                    .push(pass.index);
                    assert!((1..=31).contains(&max_recursion));
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
                // `make_shader_stage` needs exactly one NUL, at the end.
                assert!(
                    CStr::from_bytes_with_nul(name.as_bytes()).is_ok(),
                    "{name:?} in {}",
                    pass.name
                );
            }
        }
        // Each kind indexes its own pipeline array densely, which lava sizes by counting.
        for indices in [
            compute,
            raster,
            raytracing,
            specialized_compute,
            specialized_raytracing,
        ] {
            assert_eq!(indices, (0..indices.len()).collect::<Vec<_>>());
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
            assert_eq!(magic, 0x0723_0203, "{} lacks the SPIR-V magic number", pass.name);
        }
    }
}

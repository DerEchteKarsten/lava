//! The SPIR-V capabilities that need an optional lava feature, with its cargo feature and `lava::features::gate` type, which the generated `push_bindings` of a pass using them returns its bindings through (`lava::features::Gated`)

/// SPIR-V capability, cargo feature, type in `lava::features::gate`: a copy of lava's
/// `features::SPIRV_CAPABILITIES`.
pub(crate) const CAPABILITY_FEATURES: &[(u32, &str, &str)] = &[
    (9, "shader-float16", "ShaderFloat16"),
    (10, "shader-float64", "ShaderFloat64"),
    (12, "shader-int64-atomics", "ShaderInt64Atomics"),
    (22, "shader-int16", "ShaderInt16"),
    (39, "shader-int8", "ShaderInt8"),
    (4433, "storage-16bit", "Storage16Bit"),
    (4434, "storage-16bit", "Storage16Bit"),
    (4435, "storage-16bit", "Storage16Bit"),
    (4448, "storage-8bit", "Storage8Bit"),
    (4449, "storage-8bit", "Storage8Bit"),
    (4450, "storage-8bit", "Storage8Bit"),
    (4472, "ray-query", "RayQuery"),
    (4479, "ray-tracing-pipeline", "RayTracingPipeline"),
    (5055, "shader-clock", "ShaderClock"),
    (5283, "mesh-shader", "MeshShader"),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_matches_lavas() {
        let ours: Vec<(u32, &str)> = CAPABILITY_FEATURES
            .iter()
            .map(|(c, f, _)| (*c, *f))
            .collect();
        assert_eq!(ours, lava::features::SPIRV_CAPABILITIES);
        for (_, feature, gate) in CAPABILITY_FEATURES {
            let pascal: String = feature
                .split('-')
                .map(|part| {
                    let mut chars = part.chars();
                    let first = chars.next().unwrap().to_ascii_uppercase();
                    first.to_string() + &chars.as_str().replace("bit", "Bit")
                })
                .collect();
            assert_eq!(pascal, *gate);
        }
    }
}

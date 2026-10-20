//! The SPIR-V capabilities that need an optional lava feature, with its cargo feature and `lava::features::gate` type, which the generated `push_bindings` of a pass using them returns its bindings through (`lava::features::Gated`), and the scan of a module's capabilities

/// SPIR-V capability, cargo feature, type in `lava::features::gate`: a copy of lava's
/// `features::SPIRV_CAPABILITIES`.
pub(crate) const CAPABILITY_FEATURES: &[(u32, &str, &str)] = &[
    (9, "shader-float16", "ShaderFloat16"),
    (10, "shader-float64", "ShaderFloat64"),
    (12, "shader-int64-atomics", "ShaderInt64Atomics"),
    (22, "shader-int16", "ShaderInt16"),
    (32, "clip-distance", "ClipDistance"),
    (33, "cull-distance", "CullDistance"),
    (39, "shader-int8", "ShaderInt8"),
    (4433, "storage-16bit", "Storage16Bit"),
    (4434, "storage-16bit", "Storage16Bit"),
    (4435, "storage-16bit", "Storage16Bit"),
    (4436, "storage-input-output-16", "StorageInputOutput16"),
    (4448, "storage-8bit", "Storage8Bit"),
    (4449, "storage-8bit", "Storage8Bit"),
    (4450, "storage-8bit", "Storage8Bit"),
    (4472, "ray-query", "RayQuery"),
    (4478, "ray-traversal-primitive-culling", "RayTraversalPrimitiveCulling"),
    (4479, "ray-tracing-pipeline", "RayTracingPipeline"),
    (5055, "shader-clock", "ShaderClock"),
    (5283, "mesh-shader", "MeshShader"),
];

/// The operands of a SPIR-V module's `OpCapability`s, like lava's scan: they and the
/// `OpExtension`s follow the header, so it stops at the first other instruction.
pub(crate) fn capabilities(spirv: &[u8]) -> Vec<u32> {
    const OP_EXTENSION: u32 = 10;
    const OP_CAPABILITY: u32 = 17;
    let words: Vec<u32> = (spirv.chunks_exact(4))
        .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
        .collect();
    let mut capabilities = Vec::new();
    // After the five header words, each instruction starts with its word count and opcode.
    let mut i = 5;
    while let Some(&first) = words.get(i) {
        let (count, opcode) = ((first >> 16) as usize, first & 0xFFFF);
        match opcode {
            OP_CAPABILITY => capabilities.extend(words.get(i + 1)),
            OP_EXTENSION => {}
            _ => break,
        }
        if count == 0 {
            break;
        }
        i += count;
    }
    capabilities
}

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

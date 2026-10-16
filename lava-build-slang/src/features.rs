//! The optional lava features a pass's SPIR-V needs, by its `OpCapability`s: the gate types the generated `push_bindings` of such a pass returns its bindings through (`lava::features::Gated`)

/// The SPIR-V capabilities that need an optional lava feature: its cargo feature and its type
/// in `lava::features::gate`. The same table as lava's `features::SPIRV_CAPABILITIES`, which
/// lava checks again at `init` and on hot reloads (this crate's test compares them).
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

/// The operands of a SPIR-V module's `OpCapability` instructions.
fn capabilities(spirv: &[u8]) -> Vec<u32> {
    const OP_CAPABILITY: u32 = 17;
    let words: Vec<u32> = spirv
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
        .collect();
    let mut found = Vec::new();
    // After the five header words, each instruction starts with its word count and opcode.
    let mut i = 5;
    while i < words.len() {
        let (count, opcode) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        if opcode == OP_CAPABILITY
            && let Some(&capability) = words.get(i + 1)
        {
            found.push(capability);
        }
        if count == 0 {
            break;
        }
        i += count;
    }
    found
}

/// The `lava::features::gate` types of the features `spirv` needs, each once, in table order.
pub(crate) fn needed(spirv: &[u8]) -> Vec<&'static str> {
    let capabilities = capabilities(spirv);
    let mut gates: Vec<&'static str> = Vec::new();
    for (capability, _, gate) in CAPABILITY_FEATURES {
        if capabilities.contains(capability) && !gates.contains(gate) {
            gates.push(gate);
        }
    }
    gates
}

/// The `Feature` type of `gates` together: one gate type, or nested pairs of them; `None`
/// for none.
pub(crate) fn gate_type(gates: &[&str]) -> Option<String> {
    let (last, rest) = gates.split_last()?;
    let mut ty = format!("lava::features::gate::{last}");
    for gate in rest.iter().rev() {
        ty = format!("(lava::features::gate::{gate}, {ty})");
    }
    Some(ty)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn module(capabilities: &[u32]) -> Vec<u8> {
        let mut words = vec![0x0723_0203u32, 0x0001_0600, 0, 1, 0];
        for capability in capabilities {
            words.extend([2 << 16 | 17, *capability]);
        }
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    #[test]
    fn passes_need_the_features_of_their_capabilities_once() {
        assert!(needed(&module(&[1])).is_empty());
        assert_eq!(
            needed(&module(&[5283, 39, 4448, 4449])),
            ["ShaderInt8", "Storage8Bit", "MeshShader"]
        );
    }

    #[test]
    fn gate_types_nest_pairs() {
        assert_eq!(gate_type(&[]), None);
        assert_eq!(
            gate_type(&["MeshShader"]).unwrap(),
            "lava::features::gate::MeshShader"
        );
        assert_eq!(
            gate_type(&["ShaderInt8", "Storage8Bit", "MeshShader"]).unwrap(),
            "(lava::features::gate::ShaderInt8, (lava::features::gate::Storage8Bit, lava::features::gate::MeshShader))"
        );
    }

    /// lava's copy of the table names the same cargo features.
    #[test]
    fn the_table_matches_lavas() {
        let ours: Vec<(u32, &str)> = CAPABILITY_FEATURES
            .iter()
            .map(|(c, f, _)| (*c, *f))
            .collect();
        assert_eq!(ours, lava::features::SPIRV_CAPABILITIES);
    }

    /// Every gate type is lava's type of its cargo feature, `PascalCase` of its name.
    #[test]
    fn gate_types_are_named_after_their_cargo_feature() {
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

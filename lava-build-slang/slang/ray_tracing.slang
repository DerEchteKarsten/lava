#pragma once
// Ray tracing passes: lava's entry point attributes and the instance layout of its `Tlas`.

/// The index of a miss, hit or callable entry point within its region of the shader binding
/// table: TraceRay's miss index, an instance's hit group offset (`Instance::sbt_offset`), or
/// CallShader's index. Hit entry points (closest hit, any hit, intersection) with the same index
/// form one hit group, procedural if it has an intersection shader. Indices are 0 without it,
/// and each region's have to be 0, 1, 2, ...
[__AttributeUsage(_AttributeTargets.Function)]
public struct GroupAttribute { int index; };

/// On the raygen entry point: how deep TraceRay calls may nest, 0 to 31 (1 without it: only the
/// raygen traces; 2 lets a closest hit trace, e.g. shadow rays; 0 traces nothing, e.g. a pass of
/// callables). A device with a lower `maxRayRecursionDepth` gets no pipeline for the pass.
[__AttributeUsage(_AttributeTargets.Function)]
public struct MaxRecursionAttribute { int depth; };

/// An instance of a TLAS build (`VkAccelerationStructureInstanceKHR`, lava's
/// `acceleration_structure::Instance`), for passes that write them.
public struct AccelerationStructureInstance {
    /// The rows of the object-to-world transform. Not a `float3x4`: lava lays out matrices by
    /// columns.
    public float4 transform[3];
    /// The custom index (`InstanceID()`) in bits 0..24, the mask in 24..32.
    public uint custom_index_and_mask;
    /// The hit group offset in bits 0..24, the flags in 24..32.
    public uint sbt_offset_and_flags;
    /// The BLAS's address (`AccelerationStructure::address`).
    public uint64_t blas;

    public __init(float3x4 object_to_world, uint64_t blas, uint custom_index = 0, uint mask = 0xFF,
                  uint sbt_offset = 0, uint flags = 0) {
        for (int row = 0; row < 3; row++) {
            transform[row] = object_to_world[row];
        }
        custom_index_and_mask = (custom_index & 0xFFFFFF) | (mask << 24);
        sbt_offset_and_flags = (sbt_offset & 0xFFFFFF) | (flags << 24);
        this.blas = blas;
    }
}

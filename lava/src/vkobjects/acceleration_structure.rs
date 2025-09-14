use anyhow::Result;
use ash::vk;

use crate::{
    state::Functions,
    vkobjects::buffer::{Buffer, DynamicBuffer},
};

pub struct AccelerationStructure {
    pub ty: vk::AccelerationStructureTypeKHR,
    pub accel: vk::AccelerationStructureKHR,
    pub size: u64,
}

trait ScratchBuffer {
    fn vk(&self) -> vk::Buffer;
}

impl ScratchBuffer for DynamicBuffer {
    fn vk(&self) -> vk::Buffer {
        self.buffer.buffer
    }
}

impl ScratchBuffer for Buffer {
    fn vk(&self) -> vk::Buffer {
        self.buffer
    }
}

impl AccelerationStructure {
    pub fn new(
        level: vk::AccelerationStructureTypeKHR,
        as_geometry: &[vk::AccelerationStructureGeometryKHR],
        as_ranges: &[vk::AccelerationStructureBuildRangeInfoKHR],
        max_primitive_counts: &[u32],
        buffer: &impl ScratchBuffer,
        offset: u64,
        scratch_buffer: &mut DynamicBuffer,
        cmd: &vk::CommandBuffer,
    ) -> Result<AccelerationStructure> {
        let build_geo_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(level)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(as_geometry);

        let build_size = unsafe {
            let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
            Functions::acceleration_structure()
                .unwrap()
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &build_geo_info,
                    max_primitive_counts,
                    &mut size_info,
                );
            size_info
        };

        let create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(buffer.vk())
            .offset(offset)
            .size(build_size.acceleration_structure_size)
            .ty(level);
        let handle = unsafe {
            Functions::acceleration_structure()
                .unwrap()
                .create_acceleration_structure(&create_info, None)?
        };
        scratch_buffer.grow_to_size(build_size.build_scratch_size);
        let build_geo_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(level)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(as_geometry)
            .dst_acceleration_structure(handle)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch_buffer.buffer.address,
            });

        unsafe {
            Functions::acceleration_structure()
                .unwrap()
                .cmd_build_acceleration_structures(*cmd, &[build_geo_info], &[as_ranges])
        };

        Ok(AccelerationStructure {
            accel: handle,
            ty: level,
            size: build_size.acceleration_structure_size,
        })
    }
}

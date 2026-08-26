use anyhow::Result;
use ash::vk;

use crate::{buffer::Buffer, state::Functions};

pub struct AccelerationStructure {
    pub ty: vk::AccelerationStructureTypeKHR,
    pub accel: vk::AccelerationStructureKHR,
    pub size: u64,
}

impl AccelerationStructure {
    pub fn get_build_size<'a>(
        level: vk::AccelerationStructureTypeKHR,
        as_geometry: &'a [vk::AccelerationStructureGeometryKHR],
        max_primitive_counts: &'a [u32],
    ) -> vk::AccelerationStructureBuildSizesInfoKHR<'a> {
        let build_geo_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(level)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(as_geometry);

        unsafe {
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
        }
    }
    pub fn new(
        level: vk::AccelerationStructureTypeKHR,
        as_geometry: &[vk::AccelerationStructureGeometryKHR],
        as_ranges: &[vk::AccelerationStructureBuildRangeInfoKHR],
        _max_primitive_counts: &[u32],
        offset: u64,
        scratch_buffer: &mut Buffer<u8>,
        buffer: &Buffer<u8>,
        cmd: &vk::CommandBuffer,
    ) -> Result<AccelerationStructure> {
        let create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(buffer.handle)
            .offset(offset)
            .size(buffer.size())
            .ty(level);
        let handle = unsafe {
            Functions::acceleration_structure()
                .unwrap()
                .create_acceleration_structure(&create_info, None)?
        };
        let build_geo_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(level)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(as_geometry)
            .dst_acceleration_structure(handle)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch_buffer.address,
            });

        unsafe {
            Functions::acceleration_structure()
                .unwrap()
                .cmd_build_acceleration_structures(*cmd, &[build_geo_info], &[as_ranges])
        };

        Ok(AccelerationStructure {
            accel: handle,
            ty: level,
            size: buffer.size(),
        })
    }
}

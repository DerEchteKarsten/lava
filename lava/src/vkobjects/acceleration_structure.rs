//! Ray tracing acceleration structures: build-size queries, construction and destruction
use crate::error::{Error, Result};
use ash::vk;

use crate::{
    buffer::Buffer,
    state::{Ctx, Functions},
};

pub struct AccelerationStructure {
    pub ty: vk::AccelerationStructureTypeKHR,
    pub accel: vk::AccelerationStructureKHR,
    /// What the structure takes of its buffer, from `offset`.
    pub size: u64,
}

fn functions() -> Result<&'static ash::khr::acceleration_structure::Device> {
    Functions::acceleration_structure()
        .ok_or_else(|| Error::message("acceleration structure functions are unavailable"))
}

impl AccelerationStructure {
    /// The sizes a build of `as_geometry` with at most `max_primitive_counts` primitives per
    /// geometry needs: `acceleration_structure_size` of the buffer and `build_scratch_size`
    /// of the scratch buffer.
    pub(crate) fn get_build_size<'a>(
        level: vk::AccelerationStructureTypeKHR,
        as_geometry: &'a [vk::AccelerationStructureGeometryKHR],
        max_primitive_counts: &'a [u32],
    ) -> Result<vk::AccelerationStructureBuildSizesInfoKHR<'a>> {
        let build_geo_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(level)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(as_geometry);

        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            functions()?.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_geo_info,
                max_primitive_counts,
                &mut size_info,
            )
        };
        Ok(size_info)
    }

    /// Creates the structure at `offset` of `buffer` (a multiple of 256) and records its build
    /// into `cmd`, reading `as_geometry` and writing `scratch_buffer`, whose address has to be
    /// aligned to the device's scratch alignment. Both buffers have to be big enough for
    /// `get_build_size`.
    pub(crate) fn new(
        level: vk::AccelerationStructureTypeKHR,
        as_geometry: &[vk::AccelerationStructureGeometryKHR],
        as_ranges: &[vk::AccelerationStructureBuildRangeInfoKHR],
        max_primitive_counts: &[u32],
        offset: u64,
        scratch_buffer: &mut Buffer<u8>,
        buffer: &Buffer<u8>,
        cmd: &vk::CommandBuffer,
    ) -> Result<AccelerationStructure> {
        let sizes = Self::get_build_size(level, as_geometry, max_primitive_counts)?;
        let size = sizes.acceleration_structure_size;
        if offset % 256 != 0 || offset + size > buffer.size() {
            return Err(Error::message(format!(
                "an acceleration structure of {size} bytes at offset {offset} doesn't fit a buffer of {} bytes, or the offset isn't a multiple of 256",
                buffer.size()
            )));
        }
        let scratch_alignment = Ctx::physical_device()
            .acceleration_structure_properties
            .map_or(1, |p| {
                p.min_acceleration_structure_scratch_offset_alignment as u64
            });
        if scratch_buffer.address % scratch_alignment != 0
            || scratch_buffer.size() < sizes.build_scratch_size
        {
            return Err(Error::message(format!(
                "the scratch buffer needs {} bytes at an address aligned to {scratch_alignment}",
                sizes.build_scratch_size
            )));
        }

        let functions = functions()?;
        let create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(buffer.handle)
            .offset(offset)
            .size(size)
            .ty(level);
        let handle = unsafe { functions.create_acceleration_structure(&create_info, None)? };
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
            functions.cmd_build_acceleration_structures(*cmd, &[build_geo_info], &[as_ranges])
        };

        Ok(AccelerationStructure {
            accel: handle,
            ty: level,
            size,
        })
    }
}

impl Drop for AccelerationStructure {
    fn drop(&mut self) {
        if let Ok(functions) = functions() {
            unsafe { functions.destroy_acceleration_structure(self.accel, None) };
        }
    }
}

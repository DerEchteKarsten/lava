//! ash's Vulkan bindings (0.38, headers 1.3.281) plus what lava needs from newer headers: `VK_KHR_device_address_commands` and `VK_EXT_depth_clamp_control`, their types and their device functions
//!
//! Everything of `ash::vk` is re-exported, so `lava::vk` is a superset. The added structs mirror
//! the registry's (and ash master's) layout, implement ash 0.38's `TaggedStructure` and
//! `Extends*` traits, so `push_next` takes them, and have the usual builder setters. Their
//! device functions are loaded by [`device_address_commands::Device`] and
//! [`depth_clamp_control::Device`]. Delete this module's additions once an ash release has them.

use core::ffi::{CStr, c_void};
use core::marker::PhantomData;

pub use ash::vk::*;

/// `VkAddressCommandFlagBitsKHR`: how an address range is used and whether it is fully bound.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AddressCommandFlagsKHR(Flags);
ash::vk_bitflags_wrapped!(AddressCommandFlagsKHR, Flags);
impl AddressCommandFlagsKHR {
    pub const PROTECTED: Self = Self(0b1);
    pub const FULLY_BOUND: Self = Self(0b10);
    pub const STORAGE_BUFFER_USAGE: Self = Self(0b100);
    pub const UNKNOWN_STORAGE_BUFFER_USAGE: Self = Self(0b1000);
    pub const TRANSFORM_FEEDBACK_BUFFER_USAGE: Self = Self(0b1_0000);
    pub const UNKNOWN_TRANSFORM_FEEDBACK_BUFFER_USAGE: Self = Self(0b10_0000);
}
impl core::fmt::Debug for AddressCommandFlagsKHR {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "AddressCommandFlagsKHR({:#b})", self.0)
    }
}

/// `VkDepthClampModeEXT`.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Debug)]
pub struct DepthClampModeEXT(i32);
impl DepthClampModeEXT {
    pub const VIEWPORT_RANGE: Self = Self(0);
    pub const USER_DEFINED_RANGE: Self = Self(1);
    #[inline]
    pub const fn from_raw(x: i32) -> Self {
        Self(x)
    }
    #[inline]
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}

/// The `VkStructureType`s and `VkDynamicState` the added structs and commands use.
mod values {
    use super::{DynamicState, StructureType};
    pub const DEVICE_MEMORY_COPY_KHR: StructureType = StructureType::from_raw(1_000_318_000);
    pub const COPY_DEVICE_MEMORY_INFO_KHR: StructureType = StructureType::from_raw(1_000_318_001);
    pub const DEVICE_MEMORY_IMAGE_COPY_KHR: StructureType = StructureType::from_raw(1_000_318_002);
    pub const COPY_DEVICE_MEMORY_IMAGE_INFO_KHR: StructureType =
        StructureType::from_raw(1_000_318_003);
    pub const PHYSICAL_DEVICE_DEVICE_ADDRESS_COMMANDS_FEATURES_KHR: StructureType =
        StructureType::from_raw(1_000_318_006);
    pub const BIND_INDEX_BUFFER_3_INFO_KHR: StructureType = StructureType::from_raw(1_000_318_007);
    pub const DRAW_INDIRECT_2_INFO_KHR: StructureType = StructureType::from_raw(1_000_318_009);
    pub const DRAW_INDIRECT_COUNT_2_INFO_KHR: StructureType =
        StructureType::from_raw(1_000_318_010);
    pub const DISPATCH_INDIRECT_2_INFO_KHR: StructureType = StructureType::from_raw(1_000_318_011);
    pub const ACCELERATION_STRUCTURE_CREATE_INFO_2_KHR: StructureType =
        StructureType::from_raw(1_000_318_015);
    pub const PHYSICAL_DEVICE_DEPTH_CLAMP_CONTROL_FEATURES_EXT: StructureType =
        StructureType::from_raw(1_000_582_000);
    pub const DYNAMIC_STATE_DEPTH_CLAMP_RANGE_EXT: DynamicState =
        DynamicState::from_raw(1_000_582_000);
}
/// `VK_DYNAMIC_STATE_DEPTH_CLAMP_RANGE_EXT` (`VK_EXT_depth_clamp_control`).
pub const DYNAMIC_STATE_DEPTH_CLAMP_RANGE_EXT: DynamicState =
    values::DYNAMIC_STATE_DEPTH_CLAMP_RANGE_EXT;

/// `VkDeviceAddressRangeKHR`.
#[repr(C)]
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq)]
pub struct DeviceAddressRangeKHR {
    pub address: DeviceAddress,
    pub size: DeviceSize,
}

/// `VkStridedDeviceAddressRangeKHR`.
#[repr(C)]
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq)]
pub struct StridedDeviceAddressRangeKHR {
    pub address: DeviceAddress,
    pub size: DeviceSize,
    pub stride: DeviceSize,
}

/// `VkDepthClampRangeEXT`.
#[repr(C)]
#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct DepthClampRangeEXT {
    pub min_depth_clamp: f32,
    pub max_depth_clamp: f32,
}

/// A struct with `sType` and `pNext` (`$next` is `*const` for inputs, `*mut` for outputs),
/// its `Default`, its `TaggedStructure` impl and a setter per field.
macro_rules! tagged {
    (
        $(#[$attr:meta])*
        $name:ident = $s_type:ident, $next:ty {
            $($field:ident: $ty:ty),* $(,)?
        }
    ) => {
        $(#[$attr])*
        #[repr(C)]
        #[derive(Copy, Clone, Debug)]
        #[must_use]
        pub struct $name<'a> {
            pub s_type: StructureType,
            pub p_next: $next,
            $(pub $field: $ty,)*
            pub _marker: PhantomData<&'a ()>,
        }
        unsafe impl Send for $name<'_> {}
        unsafe impl Sync for $name<'_> {}
        impl Default for $name<'_> {
            #[inline]
            fn default() -> Self {
                Self {
                    s_type: values::$s_type,
                    p_next: core::ptr::null_mut(),
                    $($field: Default::default(),)*
                    _marker: PhantomData,
                }
            }
        }
        unsafe impl TaggedStructure for $name<'_> {
            const STRUCTURE_TYPE: StructureType = values::$s_type;
        }
        impl<'a> $name<'a> {
            $(
                #[inline]
                pub fn $field(mut self, $field: $ty) -> Self {
                    self.$field = $field;
                    self
                }
            )*
        }
    };
}

tagged! {
    /// `VkBindIndexBuffer3InfoKHR`.
    BindIndexBuffer3InfoKHR = BIND_INDEX_BUFFER_3_INFO_KHR, *const c_void {
        address_range: DeviceAddressRangeKHR,
        address_flags: AddressCommandFlagsKHR,
        index_type: IndexType,
    }
}
tagged! {
    /// `VkDrawIndirect2InfoKHR`.
    DrawIndirect2InfoKHR = DRAW_INDIRECT_2_INFO_KHR, *const c_void {
        address_range: StridedDeviceAddressRangeKHR,
        address_flags: AddressCommandFlagsKHR,
        draw_count: u32,
    }
}
tagged! {
    /// `VkDrawIndirectCount2InfoKHR`.
    DrawIndirectCount2InfoKHR = DRAW_INDIRECT_COUNT_2_INFO_KHR, *const c_void {
        address_range: StridedDeviceAddressRangeKHR,
        address_flags: AddressCommandFlagsKHR,
        count_address_range: DeviceAddressRangeKHR,
        count_address_flags: AddressCommandFlagsKHR,
        max_draw_count: u32,
    }
}
tagged! {
    /// `VkDispatchIndirect2InfoKHR`.
    DispatchIndirect2InfoKHR = DISPATCH_INDIRECT_2_INFO_KHR, *const c_void {
        address_range: DeviceAddressRangeKHR,
        address_flags: AddressCommandFlagsKHR,
    }
}
tagged! {
    /// `VkDeviceMemoryCopyKHR`.
    DeviceMemoryCopyKHR = DEVICE_MEMORY_COPY_KHR, *const c_void {
        src_range: DeviceAddressRangeKHR,
        src_flags: AddressCommandFlagsKHR,
        dst_range: DeviceAddressRangeKHR,
        dst_flags: AddressCommandFlagsKHR,
    }
}
tagged! {
    /// `VkDeviceMemoryImageCopyKHR`.
    DeviceMemoryImageCopyKHR = DEVICE_MEMORY_IMAGE_COPY_KHR, *const c_void {
        address_range: DeviceAddressRangeKHR,
        address_flags: AddressCommandFlagsKHR,
        address_row_length: u32,
        address_image_height: u32,
        image_subresource: ImageSubresourceLayers,
        image_layout: ImageLayout,
        image_offset: Offset3D,
        image_extent: Extent3D,
    }
}
tagged! {
    /// `VkAccelerationStructureCreateInfo2KHR`.
    AccelerationStructureCreateInfo2KHR = ACCELERATION_STRUCTURE_CREATE_INFO_2_KHR, *const c_void {
        create_flags: AccelerationStructureCreateFlagsKHR,
        address_range: DeviceAddressRangeKHR,
        address_flags: AddressCommandFlagsKHR,
        ty: AccelerationStructureTypeKHR,
    }
}
tagged! {
    /// `VkPhysicalDeviceDeviceAddressCommandsFeaturesKHR`.
    PhysicalDeviceDeviceAddressCommandsFeaturesKHR =
        PHYSICAL_DEVICE_DEVICE_ADDRESS_COMMANDS_FEATURES_KHR, *mut c_void {
        device_address_commands: Bool32,
    }
}
unsafe impl ExtendsPhysicalDeviceFeatures2 for PhysicalDeviceDeviceAddressCommandsFeaturesKHR<'_> {}
unsafe impl ExtendsDeviceCreateInfo for PhysicalDeviceDeviceAddressCommandsFeaturesKHR<'_> {}
tagged! {
    /// `VkPhysicalDeviceDepthClampControlFeaturesEXT`.
    PhysicalDeviceDepthClampControlFeaturesEXT =
        PHYSICAL_DEVICE_DEPTH_CLAMP_CONTROL_FEATURES_EXT, *mut c_void {
        depth_clamp_control: Bool32,
    }
}
unsafe impl ExtendsPhysicalDeviceFeatures2 for PhysicalDeviceDepthClampControlFeaturesEXT<'_> {}
unsafe impl ExtendsDeviceCreateInfo for PhysicalDeviceDepthClampControlFeaturesEXT<'_> {}

/// A struct whose regions are a pointer and a count, set together by `regions`.
macro_rules! with_regions {
    ($(#[$attr:meta])* $name:ident = $s_type:ident, $region:ident { $($field:ident: $ty:ty),* }) => {
        $(#[$attr])*
        #[repr(C)]
        #[derive(Copy, Clone, Debug)]
        #[must_use]
        pub struct $name<'a> {
            pub s_type: StructureType,
            pub p_next: *const c_void,
            $(pub $field: $ty,)*
            pub region_count: u32,
            pub p_regions: *const $region<'a>,
            pub _marker: PhantomData<&'a ()>,
        }
        unsafe impl Send for $name<'_> {}
        unsafe impl Sync for $name<'_> {}
        impl Default for $name<'_> {
            #[inline]
            fn default() -> Self {
                Self {
                    s_type: values::$s_type,
                    p_next: core::ptr::null(),
                    $($field: Default::default(),)*
                    region_count: 0,
                    p_regions: core::ptr::null(),
                    _marker: PhantomData,
                }
            }
        }
        unsafe impl TaggedStructure for $name<'_> {
            const STRUCTURE_TYPE: StructureType = values::$s_type;
        }
        impl<'a> $name<'a> {
            $(
                #[inline]
                pub fn $field(mut self, $field: $ty) -> Self {
                    self.$field = $field;
                    self
                }
            )*
            #[inline]
            pub fn regions(mut self, regions: &'a [$region<'a>]) -> Self {
                self.region_count = regions.len() as _;
                self.p_regions = regions.as_ptr();
                self
            }
        }
    };
}

with_regions! {
    /// `VkCopyDeviceMemoryInfoKHR`.
    CopyDeviceMemoryInfoKHR = COPY_DEVICE_MEMORY_INFO_KHR, DeviceMemoryCopyKHR {}
}
with_regions! {
    /// `VkCopyDeviceMemoryImageInfoKHR`.
    CopyDeviceMemoryImageInfoKHR = COPY_DEVICE_MEMORY_IMAGE_INFO_KHR, DeviceMemoryImageCopyKHR {
        image: Image
    }
}

/// Loads device functions by name; a missing one is `None`.
fn load<T: Copy>(instance: &ash::Instance, device: Device, name: &CStr) -> Option<T> {
    assert_eq!(size_of::<T>(), size_of::<unsafe extern "system" fn()>());
    let f = unsafe { instance.get_device_proc_addr(device, name.as_ptr()) }?;
    // SAFETY: `T` is the function pointer type of the command `name`.
    Some(unsafe { core::mem::transmute_copy(&f) })
}

/// Panics naming a device function the driver didn't return.
#[cold]
fn missing(name: &str) -> ! {
    panic!("the Vulkan driver has no {name}")
}

/// `VK_KHR_device_address_commands`: the commands that take address ranges instead of buffers.
/// Only the ones lava records are loaded.
pub mod device_address_commands {
    use super::*;

    pub const NAME: &CStr = c"VK_KHR_device_address_commands";

    type CmdInfo<T> = unsafe extern "system" fn(CommandBuffer, *const T);
    type CmdRange = unsafe extern "system" fn(
        CommandBuffer,
        *const DeviceAddressRangeKHR,
        AddressCommandFlagsKHR,
        DeviceSize,
        *const c_void,
    );
    type CmdFill = unsafe extern "system" fn(
        CommandBuffer,
        *const DeviceAddressRangeKHR,
        AddressCommandFlagsKHR,
        u32,
    );
    type CreateAccelerationStructure = unsafe extern "system" fn(
        super::Device,
        *const AccelerationStructureCreateInfo2KHR<'_>,
        *const AllocationCallbacks,
        *mut AccelerationStructureKHR,
    ) -> Result;

    /// The extension's device functions.
    #[derive(Clone)]
    pub struct Device {
        handle: super::Device,
        bind_index_buffer3: Option<CmdInfo<BindIndexBuffer3InfoKHR<'static>>>,
        draw_indirect2: Option<CmdInfo<DrawIndirect2InfoKHR<'static>>>,
        draw_indexed_indirect2: Option<CmdInfo<DrawIndirect2InfoKHR<'static>>>,
        draw_indirect_count2: Option<CmdInfo<DrawIndirectCount2InfoKHR<'static>>>,
        draw_indexed_indirect_count2: Option<CmdInfo<DrawIndirectCount2InfoKHR<'static>>>,
        draw_mesh_tasks_indirect2: Option<CmdInfo<DrawIndirect2InfoKHR<'static>>>,
        draw_mesh_tasks_indirect_count2: Option<CmdInfo<DrawIndirectCount2InfoKHR<'static>>>,
        dispatch_indirect2: Option<CmdInfo<DispatchIndirect2InfoKHR<'static>>>,
        copy_memory: Option<CmdInfo<CopyDeviceMemoryInfoKHR<'static>>>,
        copy_memory_to_image: Option<CmdInfo<CopyDeviceMemoryImageInfoKHR<'static>>>,
        copy_image_to_memory: Option<CmdInfo<CopyDeviceMemoryImageInfoKHR<'static>>>,
        update_memory: Option<CmdRange>,
        fill_memory: Option<CmdFill>,
        create_acceleration_structure2: Option<CreateAccelerationStructure>,
    }

    impl Device {
        /// Loads the functions of `device`. The mesh and acceleration structure ones are only
        /// there with those extensions.
        pub fn new(instance: &ash::Instance, device: &ash::Device) -> Self {
            let handle = device.handle();
            macro_rules! load {
                ($name:literal) => {
                    super::load(instance, handle, $name)
                };
            }
            Self {
                handle,
                bind_index_buffer3: load!(c"vkCmdBindIndexBuffer3KHR"),
                draw_indirect2: load!(c"vkCmdDrawIndirect2KHR"),
                draw_indexed_indirect2: load!(c"vkCmdDrawIndexedIndirect2KHR"),
                draw_indirect_count2: load!(c"vkCmdDrawIndirectCount2KHR"),
                draw_indexed_indirect_count2: load!(c"vkCmdDrawIndexedIndirectCount2KHR"),
                draw_mesh_tasks_indirect2: load!(c"vkCmdDrawMeshTasksIndirect2EXT"),
                draw_mesh_tasks_indirect_count2: load!(c"vkCmdDrawMeshTasksIndirectCount2EXT"),
                dispatch_indirect2: load!(c"vkCmdDispatchIndirect2KHR"),
                copy_memory: load!(c"vkCmdCopyMemoryKHR"),
                copy_memory_to_image: load!(c"vkCmdCopyMemoryToImageKHR"),
                copy_image_to_memory: load!(c"vkCmdCopyImageToMemoryKHR"),
                update_memory: load!(c"vkCmdUpdateMemoryKHR"),
                fill_memory: load!(c"vkCmdFillMemoryKHR"),
                create_acceleration_structure2: load!(c"vkCreateAccelerationStructure2KHR"),
            }
        }
    }

    /// One method per command, with ash master's names and arguments.
    macro_rules! info_commands {
        ($($method:ident($field:ident, $info:ident) = $vk_name:literal;)*) => {
            impl Device {
                $(
                    #[doc = concat!("`", $vk_name, "`.")]
                    #[inline]
                    pub unsafe fn $method(&self, command_buffer: CommandBuffer, info: &$info<'_>) {
                        let f = self.$field.unwrap_or_else(|| missing($vk_name));
                        unsafe { f(command_buffer, (info as *const $info<'_>).cast()) }
                    }
                )*
            }
        };
    }
    info_commands! {
        cmd_bind_index_buffer3(bind_index_buffer3, BindIndexBuffer3InfoKHR) = "vkCmdBindIndexBuffer3KHR";
        cmd_draw_indirect2(draw_indirect2, DrawIndirect2InfoKHR) = "vkCmdDrawIndirect2KHR";
        cmd_draw_indexed_indirect2(draw_indexed_indirect2, DrawIndirect2InfoKHR) = "vkCmdDrawIndexedIndirect2KHR";
        cmd_draw_indirect_count2(draw_indirect_count2, DrawIndirectCount2InfoKHR) = "vkCmdDrawIndirectCount2KHR";
        cmd_draw_indexed_indirect_count2(draw_indexed_indirect_count2, DrawIndirectCount2InfoKHR) = "vkCmdDrawIndexedIndirectCount2KHR";
        cmd_draw_mesh_tasks_indirect2(draw_mesh_tasks_indirect2, DrawIndirect2InfoKHR) = "vkCmdDrawMeshTasksIndirect2EXT";
        cmd_draw_mesh_tasks_indirect_count2(draw_mesh_tasks_indirect_count2, DrawIndirectCount2InfoKHR) = "vkCmdDrawMeshTasksIndirectCount2EXT";
        cmd_dispatch_indirect2(dispatch_indirect2, DispatchIndirect2InfoKHR) = "vkCmdDispatchIndirect2KHR";
        cmd_copy_memory(copy_memory, CopyDeviceMemoryInfoKHR) = "vkCmdCopyMemoryKHR";
        cmd_copy_memory_to_image(copy_memory_to_image, CopyDeviceMemoryImageInfoKHR) = "vkCmdCopyMemoryToImageKHR";
        cmd_copy_image_to_memory(copy_image_to_memory, CopyDeviceMemoryImageInfoKHR) = "vkCmdCopyImageToMemoryKHR";
    }

    impl Device {
        /// `vkCmdUpdateMemoryKHR`.
        #[inline]
        pub unsafe fn cmd_update_memory(
            &self,
            command_buffer: CommandBuffer,
            dst_range: &DeviceAddressRangeKHR,
            dst_flags: AddressCommandFlagsKHR,
            data: &[u8],
        ) {
            let f = self
                .update_memory
                .unwrap_or_else(|| missing("vkCmdUpdateMemoryKHR"));
            unsafe {
                f(
                    command_buffer,
                    dst_range,
                    dst_flags,
                    data.len() as DeviceSize,
                    data.as_ptr().cast(),
                )
            }
        }

        /// `vkCmdFillMemoryKHR`.
        #[inline]
        pub unsafe fn cmd_fill_memory(
            &self,
            command_buffer: CommandBuffer,
            dst_range: &DeviceAddressRangeKHR,
            dst_flags: AddressCommandFlagsKHR,
            data: u32,
        ) {
            let f = self
                .fill_memory
                .unwrap_or_else(|| missing("vkCmdFillMemoryKHR"));
            unsafe { f(command_buffer, dst_range, dst_flags, data) }
        }

        /// `vkCreateAccelerationStructure2KHR`.
        #[inline]
        pub unsafe fn create_acceleration_structure2(
            &self,
            create_info: &AccelerationStructureCreateInfo2KHR<'_>,
            allocation_callbacks: Option<&AllocationCallbacks>,
        ) -> ash::prelude::VkResult<AccelerationStructureKHR> {
            let f = self
                .create_acceleration_structure2
                .unwrap_or_else(|| missing("vkCreateAccelerationStructure2KHR"));
            let mut structure = AccelerationStructureKHR::null();
            let callbacks = allocation_callbacks.map_or(core::ptr::null(), |c| c as *const _);
            unsafe { f(self.handle, create_info, callbacks, &mut structure) }
                .result_with_success(structure)
        }
    }
}

/// `VK_EXT_depth_clamp_control`: a depth clamp range other than the viewport's.
pub mod depth_clamp_control {
    use super::*;

    pub const NAME: &CStr = c"VK_EXT_depth_clamp_control";

    type CmdSetRange =
        unsafe extern "system" fn(CommandBuffer, DepthClampModeEXT, *const DepthClampRangeEXT);

    /// The extension's device function.
    #[derive(Clone)]
    pub struct Device {
        set_depth_clamp_range: Option<CmdSetRange>,
    }

    impl Device {
        pub fn new(instance: &ash::Instance, device: &ash::Device) -> Self {
            Self {
                set_depth_clamp_range: super::load(
                    instance,
                    device.handle(),
                    c"vkCmdSetDepthClampRangeEXT",
                ),
            }
        }

        /// `vkCmdSetDepthClampRangeEXT`; `range` is only read with `USER_DEFINED_RANGE`.
        #[inline]
        pub unsafe fn cmd_set_depth_clamp_range(
            &self,
            command_buffer: CommandBuffer,
            mode: DepthClampModeEXT,
            range: &DepthClampRangeEXT,
        ) {
            let f = self
                .set_depth_clamp_range
                .unwrap_or_else(|| missing("vkCmdSetDepthClampRangeEXT"));
            unsafe { f(command_buffer, mode, range) }
        }
    }
}

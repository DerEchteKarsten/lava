//! Global Vulkan context: instance, device, queues, features, and validation logging
use std::{
    cell::Cell,
    ffi::{c_char, c_void},
    fmt::Debug,
    panic::{self, Location},
    sync::{Mutex, MutexGuard, OnceLock, atomic::AtomicBool},
};

use crate::error::{Error, Result};
use ash::{
    Device, Entry,
    ext::debug_utils,
    vk::{self, Handle, make_api_version},
};
use gpu_allocator::{
    AllocationSizes, AllocatorDebugSettings,
    vulkan::{Allocator, AllocatorCreateDesc},
};
use lava_macros::validation_trace;
use std::ffi::{CStr, CString};

use crate::vkobjects::physical_device::{PhysicalDevice, QueueFamily};

pub use ash::vk as raw_vulkan;

impl Debug for Ctx {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

pub struct Ctx {
    features: Features,
    device: Device,
    physical_device: PhysicalDevice,
    allocator: Mutex<Allocator>,

    pub(crate) gfx_queue_familie: QueueFamily,
    pub(crate) gfx_queues_in_use: Box<[AtomicBool]>,

    pub(crate) transfer_queue_familie: Option<QueueFamily>,
    pub(crate) transfer_queues_in_use: Option<Box<[AtomicBool]>>,
}

use raw_window_handle::RawDisplayHandle;
static STATE: OnceLock<Ctx> = OnceLock::new();
impl Ctx {
    #[allow(static_mut_refs)]
    pub(crate) fn get() -> &'static Self {
        STATE.wait()
    }
    pub(crate) fn device() -> &'static Device {
        &Ctx::get().device
    }
    pub(crate) fn physical_device() -> &'static PhysicalDevice {
        &Ctx::get().physical_device
    }
    pub fn gfx_queue_index() -> u32 {
        Ctx::get().gfx_queue_familie.index
    }
    pub fn num_gfx_queues() -> u32 {
        Ctx::get().gfx_queues_in_use.len() as u32
    }
    pub(crate) fn transfer_queue_index() -> u32 {
        Ctx::get()
            .transfer_queue_familie
            .as_ref()
            .map(|e| e.index)
            .unwrap_or(Ctx::get().gfx_queue_familie.index)
    }
    /// Presentation always goes through the graphics family; `Surface::new` verifies that the
    /// family can present to the surface it wraps.
    pub fn present_queue_index() -> u32 {
        Ctx::get().gfx_queue_familie.index
    }
    pub(crate) fn allocator<'a>() -> MutexGuard<'a, Allocator> {
        Ctx::get()
            .allocator
            .lock()
            .expect("allocator mutex was poisoned")
    }

    pub fn features() -> Features {
        Ctx::get().features.clone()
    }

    /// Creates the instance, device, and allocator. With `display` set, the instance gets the
    /// platform surface extensions and the device gets a swapchain; without it lava runs headless.
    pub(super) fn init(
        display: Option<&RawDisplayHandle>,
        enable_validation: bool,
        enable_gpu_assited_validation: bool,
    ) -> Result<()> {
        let entry = unsafe { Entry::load()? };
        let app_info = vk::ApplicationInfo::default().api_version(make_api_version(0, 1, 4, 0));
        let layer_names = unsafe {
            [
                CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0"),
                // CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_timeline_semaphore\0"),
                // CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_synchronization2\0")
            ]
        };
        let layers_names_raw: Vec<*const c_char> = layer_names
            .iter()
            .map(|raw_name| raw_name.as_ptr())
            .collect();

        let mut instance_extensions = match display {
            Some(display) => ash_window::enumerate_required_extensions(*display)?.to_vec(),
            None => Vec::new(),
        };

        let mut features = Features::default();
        features.present = display.is_some();
        #[cfg(debug_assertions)]
        {
            features.debug_utils = enable_validation;
            features.device_debug_utils = enable_validation;
        }
        let mut validation_features = vk::ValidationFeaturesEXT::default();
        let mut validation_f;
        let mut instance_info = vk::InstanceCreateInfo::default();
        if features.debug_utils {
            instance_extensions.push(ash::ext::debug_utils::NAME.as_ptr());
            validation_f = vec![
                vk::ValidationFeatureEnableEXT::DEBUG_PRINTF,
                vk::ValidationFeatureEnableEXT::BEST_PRACTICES,
                vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION,
            ];
            if enable_gpu_assited_validation {
                validation_f.push(vk::ValidationFeatureEnableEXT::GPU_ASSISTED);
            }

            validation_features = validation_features.enabled_validation_features(&validation_f);

            instance_info = instance_info
                .enabled_layer_names(&layers_names_raw)
                .push_next(&mut validation_features);
        }
        instance_info = instance_info
            .application_info(&app_info)
            .enabled_extension_names(&instance_extensions);

        let instance = unsafe { entry.create_instance(&instance_info, None)? };
        let mut instance_debug_utils = None;
        if features.debug_utils {
            instance_debug_utils = Some(debug_utils::Instance::new(&entry, &instance));
            let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                        | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(vulkan_debug_callback));
            unsafe {
                instance_debug_utils
                    .as_ref()
                    .expect("debug utils instance was just created")
                    .create_debug_utils_messenger(&debug_info, None)?
            };
        }

        let surface_fn = display.map(|_| ash::khr::surface::Instance::new(&entry, &instance));

        let physical_devices = PhysicalDevice::enumerate_physical_devices(&instance)?;
        let (physical_device, graphics_queue_family, transfer_queue_familie) =
            PhysicalDevice::select_suitable_physical_device(
                physical_devices.as_slice(),
                &mut features,
            )?;

        let mut queues = vec![(
            graphics_queue_family.index,
            graphics_queue_family.num_queues,
        )];
        if let Some(tqf) = &transfer_queue_familie {
            queues.push((tqf.index, tqf.num_queues));
        }
        let device = create_device(queues, &physical_device, &features, &instance)?;
        let mut debug_utils = None;
        if features.device_debug_utils {
            debug_utils = Some(ash::ext::debug_utils::Device::new(&instance, &device));
        }

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device: physical_device.handel,
            debug_settings: AllocatorDebugSettings {
                log_allocations: true,
                log_frees: true,
                log_leaks_on_shutdown: false,
                log_memory_information: true,
                ..Default::default()
            },
            buffer_device_address: true,
            allocation_sizes: AllocationSizes::new(256, if features.rebar { 256 } else { 16 }),
        })?;

        FUNCTIONS
            .set(Functions {
                mesh: if features.mesh {
                    Some(ash::ext::mesh_shader::Device::new(&instance, &device))
                } else {
                    None
                },
                raytracing_pipeline: if features.raytracing {
                    Some(ash::khr::ray_tracing_pipeline::Device::new(
                        &instance, &device,
                    ))
                } else {
                    None
                },
                acceleration_structure: if features.raytracing {
                    Some(ash::khr::acceleration_structure::Device::new(
                        &instance, &device,
                    ))
                } else {
                    None
                },
                host_image_copy: ash::ext::host_image_copy::Device::new(&instance, &device),
                swapchain: ash::khr::swapchain::Device::new(&instance, &device),
                instance,
                entry,
                surface: surface_fn,
                debug_utils: instance_debug_utils,
                device_debug_utils: debug_utils,
            })
            .map_err(|_| Error::message("Vulkan functions were already initialized"))?;

        STATE
            .set(Ctx {
                device,
                allocator: Mutex::new(allocator),
                gfx_queues_in_use: (0..graphics_queue_family.num_queues)
                    .map(|_| AtomicBool::new(false))
                    .collect(),
                gfx_queue_familie: graphics_queue_family,
                transfer_queues_in_use: if let Some(transfer) = &transfer_queue_familie {
                    Some(
                        (0..transfer.num_queues)
                            .map(|_| AtomicBool::new(false))
                            .collect(),
                    )
                } else {
                    None
                },
                transfer_queue_familie,

                features: features,
                physical_device: physical_device,
            })
            .map_err(|_| Error::message("Vulkan context was already initialized"))?;

        Ok(())
    }
}

thread_local! {
    pub static CALLSITE: Cell<Option<Location<'static>>> = Cell::new(None);
}

/// Holds the caller location in [`CALLSITE`] while a `#[validation_trace]` function runs.
///
/// Only the outermost guard on a thread owns the location, and it clears it on drop, so the
/// location survives nested traced calls and is released on early returns.
pub struct CallsiteGuard {
    owns: bool,
}

impl CallsiteGuard {
    pub fn enter(location: &'static Location<'static>) -> Self {
        let owns = CALLSITE.get().is_none();
        if owns {
            CALLSITE.set(Some(*location));
        }
        Self { owns }
    }
}

impl Drop for CallsiteGuard {
    fn drop(&mut self) {
        if self.owns {
            CALLSITE.set(None);
        }
    }
}

unsafe extern "system" fn vulkan_debug_callback(
    flag: vk::DebugUtilsMessageSeverityFlagsEXT,
    typ: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    unsafe {
        if p_callback_data != std::ptr::null() && (*p_callback_data).p_message != std::ptr::null() {
            let message = CStr::from_ptr((*p_callback_data).p_message).to_string_lossy();
            #[cfg(debug_assertions)]
            {
                let split = message.split("DebugPrintf:\n").collect::<Vec<_>>();
                if split.len() > 1 {
                    let printf_message = split[1..]
                        .iter()
                        .map(|s| s.chars())
                        .flatten()
                        .collect::<String>();
                    tracing::info!("{}", printf_message);
                    return vk::FALSE;
                }
            }
            let typ = format!("{typ:?}");
            let location = CALLSITE.get().map(|loc| loc.to_string());
            let location = location.as_deref();
            match flag {
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
                    tracing::error!(target: "vulkan-validation", typ, validation_location = location, "{}", message)
                }
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
                    tracing::warn!(target: "vulkan-validation", typ, validation_location = location, "{}", message)
                }
                vk::DebugUtilsMessageSeverityFlagsEXT::INFO => {
                    tracing::debug!(target: "vulkan-validation", typ, validation_location = location, "{}", message)
                }
                _ => {
                    tracing::trace!(target: "vulkan-validation", typ, validation_location = location, "{}", message)
                }
            }
        }
    }
    vk::FALSE
}

#[derive(Default, Debug, Clone)]
pub struct Features {
    pub rebar: bool,
    pub present: bool,
    pub device_debug_utils: bool,
    pub debug_utils: bool,
    pub mesh: bool,
    pub raytracing: bool,
    /// Pipeline statistics queries (invocation counts per pass).
    pub pipeline_statistics: bool,
    /// Task and mesh shader invocations in pipeline statistics.
    pub mesh_queries: bool,
    /// `VK_EXT_memory_budget`: per-heap usage and budget in `memory_report`.
    pub memory_budget: bool,
    /// `VK_KHR_shader_clock` with subgroup clocks, which `profile.slang` reads. Shaders that
    /// use `PROFILE` need it.
    pub shader_clock: bool,
}
impl Features {
    pub fn extensions(&self) -> Vec<&CStr> {
        let mut extensions = vec![
            ash::ext::extended_dynamic_state3::NAME,
            // unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_unified_image_layouts\0") }
        ];
        if self.rebar {
            extensions.push(ash::ext::host_image_copy::NAME);
        }
        if self.present {
            extensions.push(ash::khr::swapchain::NAME);
        }
        if self.debug_utils {
            // extensions.push(ash::ext::device_address_binding_report::NAME);
        }
        if self.device_debug_utils {
            extensions.push(ash::ext::debug_utils::NAME);
        }
        if self.mesh {
            extensions.push(ash::ext::mesh_shader::NAME);
        }
        if self.memory_budget {
            extensions.push(ash::ext::memory_budget::NAME);
        }
        if self.shader_clock {
            extensions.push(ash::khr::shader_clock::NAME);
        }
        if self.raytracing {
            extensions.push(ash::khr::ray_tracing_pipeline::NAME);
            extensions.push(ash::khr::deferred_host_operations::NAME);
            extensions.push(ash::khr::acceleration_structure::NAME);
        }
        extensions
    }

    fn features<'a>(
        &self,
        vk11: &'a mut vk::PhysicalDeviceVulkan11Features,
        vk12: &'a mut vk::PhysicalDeviceVulkan12Features,
        vk13: &'a mut vk::PhysicalDeviceVulkan13Features,
        dn3: &'a mut vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT,
        dy2: &'a mut vk::PhysicalDeviceExtendedDynamicState2FeaturesEXT,
        mesh: &'a mut vk::PhysicalDeviceMeshShaderFeaturesEXT,
        ray: &'a mut vk::PhysicalDeviceRayTracingPipelineFeaturesKHR,
        acc: &'a mut vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
        host_image_copy: &'a mut vk::PhysicalDeviceHostImageCopyFeaturesEXT,
        clock: &'a mut vk::PhysicalDeviceShaderClockFeaturesKHR,
    ) -> vk::PhysicalDeviceFeatures2<'a> {
        *vk11 = vk11.shader_draw_parameters(true);
        *vk12 = vk12
            .runtime_descriptor_array(true)
            .buffer_device_address(true)
            .descriptor_indexing(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .shader_storage_image_array_non_uniform_indexing(true)
            .shader_float16(true)
            .descriptor_binding_storage_buffer_update_after_bind(true)
            .descriptor_binding_partially_bound(true)
            .descriptor_binding_variable_descriptor_count(true)
            .descriptor_binding_storage_image_update_after_bind(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .descriptor_binding_update_unused_while_pending(true)
            .timeline_semaphore(true)
            .draw_indirect_count(true)
            .scalar_block_layout(true)
            .storage_push_constant8(true)
            .vulkan_memory_model(true)
            .vulkan_memory_model_device_scope(true)
            .storage_buffer8_bit_access(true)
            .shader_buffer_int64_atomics(true)
            .shader_int8(true);
        *vk13 = vk13
            .dynamic_rendering(true)
            .maintenance4(true)
            // What `discard` compiles to.
            .shader_demote_to_helper_invocation(true)
            .synchronization2(true);
        let phfeatures = vk::PhysicalDeviceFeatures::default()
            .shader_int64(true)
            .texture_compression_bc(true)
            .fill_mode_non_solid(true)
            // Per-attachment `Blend` modes.
            .independent_blend(true)
            .fragment_stores_and_atomics(true)
            .shader_int16(true)
            .pipeline_statistics_query(self.pipeline_statistics)
            .vertex_pipeline_stores_and_atomics(true);

        *dn3 = dn3
            .extended_dynamic_state3_depth_clamp_enable(true)
            .extended_dynamic_state3_polygon_mode(true)
            .extended_dynamic_state3_logic_op_enable(true)
            .extended_dynamic_state3_color_blend_equation(true)
            .extended_dynamic_state3_color_write_mask(true)
            .extended_dynamic_state3_color_blend_enable(true);
        *dy2 = dy2.extended_dynamic_state2_logic_op(true);

        *host_image_copy =
            vk::PhysicalDeviceHostImageCopyFeaturesEXT::default().host_image_copy(true);
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .features(phfeatures)
            .push_next(vk11)
            .push_next(vk12)
            .push_next(vk13)
            .push_next(dy2)
            .push_next(host_image_copy)
            .push_next(dn3);
        if self.mesh {
            *mesh = mesh
                .task_shader(true)
                .mesh_shader(true)
                .mesh_shader_queries(self.mesh_queries);
            features = features.push_next(mesh);
        }
        if self.shader_clock {
            *clock = clock.shader_subgroup_clock(true);
            features = features.push_next(clock);
        }
        if self.raytracing {
            *ray = ray.ray_tracing_pipeline(true);
            *acc = acc
                .acceleration_structure(true)
                .descriptor_binding_acceleration_structure_update_after_bind(true);
            features = features.push_next(ray).push_next(acc);
        }
        features
    }
}

pub struct Functions {
    instance: ash::Instance,
    host_image_copy: ash::ext::host_image_copy::Device,
    entry: ash::Entry,
    swapchain: ash::khr::swapchain::Device,
    surface: Option<ash::khr::surface::Instance>,
    debug_utils: Option<ash::ext::debug_utils::Instance>,
    device_debug_utils: Option<ash::ext::debug_utils::Device>,
    mesh: Option<ash::ext::mesh_shader::Device>,
    raytracing_pipeline: Option<ash::khr::ray_tracing_pipeline::Device>,
    acceleration_structure: Option<ash::khr::acceleration_structure::Device>,
}

impl Debug for Functions {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

static FUNCTIONS: OnceLock<Functions> = OnceLock::new();

impl Functions {
    pub(crate) fn surface() -> Option<&'static ash::khr::surface::Instance> {
        get().surface.as_ref()
    }
    pub(crate) fn host_image_copy() -> &'static ash::ext::host_image_copy::Device {
        &get().host_image_copy
    }
    pub(crate) fn instance() -> &'static ash::Instance {
        &get().instance
    }
    pub(crate) fn entry() -> &'static ash::Entry {
        &get().entry
    }
    pub(crate) fn swapchain() -> &'static ash::khr::swapchain::Device {
        &get().swapchain
    }
    pub(crate) fn debug_utils() -> Option<&'static ash::ext::debug_utils::Device> {
        get().device_debug_utils.as_ref()
    }
    pub(crate) fn instance_debug_utils() -> Option<&'static ash::ext::debug_utils::Instance> {
        get().debug_utils.as_ref()
    }
    pub(crate) fn mesh() -> Option<&'static ash::ext::mesh_shader::Device> {
        get().mesh.as_ref()
    }
    pub(crate) fn raytracing_pipeline() -> Option<&'static ash::khr::ray_tracing_pipeline::Device> {
        get().raytracing_pipeline.as_ref()
    }
    pub(crate) fn acceleration_structure()
    -> Option<&'static ash::khr::acceleration_structure::Device> {
        get().acceleration_structure.as_ref()
    }

    pub(crate) fn set_debug_name<T>(name: &str, object: T)
    where
        T: Handle,
    {
        if let Some(debug_utils) = Self::debug_utils() {
            let name = debug_name(name);
            let name_info = vk::DebugUtilsObjectNameInfoEXT::default()
                .object_handle(object)
                .object_name(&name);
            if let Err(err) = unsafe { debug_utils.set_debug_utils_object_name(&name_info) } {
                tracing::error!(%err, "failed to set Vulkan debug name");
            }
        }
    }

    pub(crate) fn cmd_start_label(cmd: &vk::CommandBuffer, name: &str) {
        if let Some(debug_utils) = Self::debug_utils() {
            let name = debug_name(name);
            let name_info = vk::DebugUtilsLabelEXT::default().label_name(&name);
            unsafe { debug_utils.cmd_begin_debug_utils_label(*cmd, &name_info) };
        }
    }
    pub(crate) fn cmd_insert_label(cmd: &vk::CommandBuffer, name: &str) {
        if let Some(debug_utils) = Self::debug_utils() {
            let name = debug_name(name);
            let name_info = vk::DebugUtilsLabelEXT::default().label_name(&name);
            unsafe { debug_utils.cmd_insert_debug_utils_label(*cmd, &name_info) };
        }
    }
    pub(crate) fn cmd_end_label(cmd: &vk::CommandBuffer) {
        if let Some(debug_utils) = Self::debug_utils() {
            unsafe { debug_utils.cmd_end_debug_utils_label(*cmd) };
        }
    }
}
/// Converts a debug name to a C string. Names may already carry a trailing NUL (pass entry
/// names do); anything from the first NUL on is dropped.
pub(crate) fn debug_name(name: &str) -> CString {
    let end = name.find('\0').unwrap_or(name.len());
    CString::new(&name[..end]).expect("interior NULs were cut off")
}

fn get() -> &'static Functions {
    FUNCTIONS
        .get()
        .expect("Vulkan functions have not been initialized")
}

#[validation_trace]
pub(super) fn create_device(
    mut queue_families: Vec<(u32, u32)>,
    physical_device: &PhysicalDevice,
    features: &Features,
    instance: &ash::Instance,
) -> Result<ash::Device> {
    // One priority per queue: every queue of a selected family is created, so that
    // `Queue::new` can hand out all the slots tracked in `*_queues_in_use`.
    queue_families.sort_unstable();
    queue_families.dedup_by_key(|(index, _)| *index);
    let queue_priorities = queue_families
        .iter()
        .map(|(_, count)| vec![1.0f32; *count as usize])
        .collect::<Vec<_>>();
    let queue_create_infos = queue_families
        .iter()
        .zip(&queue_priorities)
        .map(|((index, _), priorities)| {
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(*index)
                .queue_priorities(priorities)
        })
        .collect::<Vec<_>>();

    let required_extensions = features.extensions();
    let device_extensions_as_ptr = required_extensions
        .into_iter()
        .map(|e| e.as_ptr() as *const i8)
        .collect::<Vec<_>>();

    let (
        mut vk11,
        mut vk12,
        mut vk13,
        mut dy2,
        mut dn3,
        mut mesh,
        mut ray,
        mut acc,
        mut host_image_copy,
        mut clock,
    ) = Default::default();
    let mut features = features.features(
        &mut vk11,
        &mut vk12,
        &mut vk13,
        &mut dn3,
        &mut dy2,
        &mut mesh,
        &mut ray,
        &mut acc,
        &mut host_image_copy,
        &mut clock,
    );
    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_create_infos)
        .enabled_extension_names(device_extensions_as_ptr.as_slice())
        .push_next(&mut features);

    let device =
        unsafe { instance.create_device(physical_device.handel, &device_create_info, None)? };

    Ok(device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use tracing::field::{Field, Visit};
    use tracing_subscriber::{Layer, layer::SubscriberExt};

    fn names(features: &Features) -> Vec<String> {
        features
            .extensions()
            .into_iter()
            .map(|e| e.to_str().unwrap().to_owned())
            .collect()
    }

    #[test]
    fn headless_devices_need_only_the_dynamic_state_extension() {
        assert_eq!(
            names(&Features::default()),
            ["VK_EXT_extended_dynamic_state3"]
        );
    }

    #[test]
    fn each_feature_adds_its_device_extensions() {
        let only = |f: Features| names(&f)[1..].to_vec();
        assert_eq!(
            only(Features {
                present: true,
                ..Default::default()
            }),
            ["VK_KHR_swapchain"]
        );
        assert_eq!(
            only(Features {
                rebar: true,
                ..Default::default()
            }),
            ["VK_EXT_host_image_copy"]
        );
        assert_eq!(
            only(Features {
                mesh: true,
                ..Default::default()
            }),
            ["VK_EXT_mesh_shader"]
        );
        assert_eq!(
            only(Features {
                memory_budget: true,
                ..Default::default()
            }),
            ["VK_EXT_memory_budget"]
        );
        assert_eq!(
            only(Features {
                shader_clock: true,
                ..Default::default()
            }),
            ["VK_KHR_shader_clock"]
        );
        assert_eq!(
            only(Features {
                device_debug_utils: true,
                ..Default::default()
            }),
            ["VK_EXT_debug_utils"]
        );
        assert_eq!(
            only(Features {
                raytracing: true,
                ..Default::default()
            }),
            [
                "VK_KHR_ray_tracing_pipeline",
                "VK_KHR_deferred_host_operations",
                "VK_KHR_acceleration_structure"
            ]
        );
        // Instance-level validation alone needs no device extension.
        assert!(
            only(Features {
                debug_utils: true,
                ..Default::default()
            })
            .is_empty()
        );
    }

    /// Runs `Features::features` and returns the optional feature structs it filled in.
    fn requested(
        features: &Features,
    ) -> (
        vk::PhysicalDeviceVulkan12Features<'static>,
        vk::PhysicalDeviceVulkan13Features<'static>,
        vk::PhysicalDeviceMeshShaderFeaturesEXT<'static>,
        vk::PhysicalDeviceRayTracingPipelineFeaturesKHR<'static>,
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR<'static>,
    ) {
        let (mut vk11, mut vk12, mut vk13, mut dn3, mut dy2, mut mesh, mut ray, mut acc, mut hic) =
            Default::default();
        let mut clock = Default::default();
        features.features(
            &mut vk11, &mut vk12, &mut vk13, &mut dn3, &mut dy2, &mut mesh, &mut ray, &mut acc,
            &mut hic, &mut clock,
        );
        // Detach the copies from the pNext chain that borrowed them.
        vk12.p_next = std::ptr::null_mut();
        vk13.p_next = std::ptr::null_mut();
        mesh.p_next = std::ptr::null_mut();
        ray.p_next = std::ptr::null_mut();
        acc.p_next = std::ptr::null_mut();
        (vk12, vk13, mesh, ray, acc)
    }

    #[test]
    fn core_features_lava_relies_on_are_always_requested() {
        let (vk12, vk13, mesh, ray, acc) = requested(&Features::default());
        // Bindless descriptors, buffer pointers in push constants, sync2 barriers, dynamic rendering.
        assert_eq!(vk12.buffer_device_address, vk::TRUE);
        assert_eq!(vk12.runtime_descriptor_array, vk::TRUE);
        assert_eq!(vk12.descriptor_binding_partially_bound, vk::TRUE);
        assert_eq!(vk12.descriptor_binding_variable_descriptor_count, vk::TRUE);
        assert_eq!(vk12.descriptor_binding_update_unused_while_pending, vk::TRUE);
        assert_eq!(vk12.timeline_semaphore, vk::TRUE);
        assert_eq!(vk12.scalar_block_layout, vk::TRUE);
        assert_eq!(vk13.synchronization2, vk::TRUE);
        assert_eq!(vk13.dynamic_rendering, vk::TRUE);
        assert_eq!(vk13.shader_demote_to_helper_invocation, vk::TRUE);

        assert_eq!(mesh.mesh_shader, vk::FALSE);
        assert_eq!(ray.ray_tracing_pipeline, vk::FALSE);
        assert_eq!(acc.acceleration_structure, vk::FALSE);
    }

    #[test]
    fn optional_features_are_requested_only_when_enabled() {
        let (_, _, mesh, ray, acc) = requested(&Features {
            mesh: true,
            ..Default::default()
        });
        assert_eq!((mesh.mesh_shader, mesh.task_shader), (vk::TRUE, vk::TRUE));
        assert_eq!(ray.ray_tracing_pipeline, vk::FALSE);
        assert_eq!(acc.acceleration_structure, vk::FALSE);

        let (_, _, mesh, ray, acc) = requested(&Features {
            raytracing: true,
            ..Default::default()
        });
        assert_eq!(mesh.mesh_shader, vk::FALSE);
        assert_eq!(ray.ray_tracing_pipeline, vk::TRUE);
        assert_eq!(acc.acceleration_structure, vk::TRUE);
    }

    #[test]
    fn debug_names_end_at_the_first_nul() {
        assert_eq!(debug_name("pipeline").as_bytes(), b"pipeline");
        // Pass entry names already carry their terminator.
        assert_eq!(debug_name("skybox\0").as_bytes(), b"skybox");
        assert_eq!(debug_name("a\0b").as_bytes(), b"a");
        assert_eq!(debug_name("").as_bytes(), b"");
    }

    #[track_caller]
    fn here() -> &'static Location<'static> {
        Location::caller()
    }

    #[test]
    fn callsite_guard_sets_and_clears_the_location() {
        assert!(CALLSITE.get().is_none());
        let location = here();
        {
            let _guard = CallsiteGuard::enter(location);
            assert_eq!(CALLSITE.get().unwrap().line(), location.line());
        }
        assert!(CALLSITE.get().is_none());
    }

    #[test]
    fn nested_callsite_guards_keep_the_outermost_location() {
        let outer = here();
        let inner = here();
        let _outer_guard = CallsiteGuard::enter(outer);
        {
            let _inner_guard = CallsiteGuard::enter(inner);
            assert_eq!(CALLSITE.get().unwrap().line(), outer.line());
        }
        // The inner guard must not clear what the outer one owns.
        assert_eq!(CALLSITE.get().unwrap().line(), outer.line());
    }

    #[derive(Debug, Default, Clone)]
    struct Captured {
        level: String,
        target: String,
        message: String,
        location: Option<String>,
    }

    impl Visit for Captured {
        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            if field.name() == "message" {
                self.message = format!("{value:?}");
            }
        }
        fn record_str(&mut self, field: &Field, value: &str) {
            if field.name() == "validation_location" {
                self.location = Some(value.to_owned());
            }
        }
    }

    struct Capture(Arc<Mutex<Vec<Captured>>>);

    impl<S: tracing::Subscriber> Layer<S> for Capture {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _: tracing_subscriber::layer::Context<'_, S>,
        ) {
            let mut captured = Captured {
                level: event.metadata().level().to_string(),
                target: event.metadata().target().to_owned(),
                ..Default::default()
            };
            event.record(&mut captured);
            self.0.lock().unwrap().push(captured);
        }
    }

    /// Calls the debug callback the way the validation layer does and returns what it logged.
    fn log(
        severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        message: Option<&CStr>,
    ) -> Vec<Captured> {
        let events = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::registry().with(Capture(events.clone()));
        tracing::subscriber::with_default(subscriber, || {
            let mut data = vk::DebugUtilsMessengerCallbackDataEXT::default();
            if let Some(message) = message {
                data.p_message = message.as_ptr();
            }
            let result = unsafe {
                vulkan_debug_callback(
                    severity,
                    vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
                    &data,
                    std::ptr::null_mut(),
                )
            };
            // The callback must never ask the driver to abort the call.
            assert_eq!(result, vk::FALSE);
        });
        events.lock().unwrap().clone()
    }

    #[test]
    fn validation_messages_are_logged_at_the_matching_level() {
        use vk::DebugUtilsMessageSeverityFlagsEXT as S;
        for (severity, level) in [
            (S::ERROR, "ERROR"),
            (S::WARNING, "WARN"),
            (S::INFO, "DEBUG"),
            (S::VERBOSE, "TRACE"),
        ] {
            let events = log(severity, Some(c"something is wrong"));
            assert_eq!(events.len(), 1, "{level}");
            assert_eq!(events[0].level, level);
            assert_eq!(events[0].target, "vulkan-validation");
            assert_eq!(events[0].message, "something is wrong");
        }
    }

    #[test]
    fn validation_messages_carry_the_recorded_callsite() {
        let severity = vk::DebugUtilsMessageSeverityFlagsEXT::ERROR;
        assert_eq!(log(severity, Some(c"m"))[0].location, None);

        let location = here();
        let _guard = CallsiteGuard::enter(location);
        let events = log(severity, Some(c"m"));
        assert_eq!(
            events[0].location.as_deref(),
            Some(location.to_string().as_str())
        );
    }

    #[test]
    fn shader_printf_output_is_logged_as_info_without_the_layer_prefix() {
        let events = log(
            vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
            Some(c"Validation Information: [ WARNING-DEBUG-PRINTF ] DebugPrintf:\nvalue = 42"),
        );
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].level, "INFO");
        assert_eq!(events[0].message, "value = 42");
    }

    #[test]
    fn callbacks_without_a_message_log_nothing() {
        let severity = vk::DebugUtilsMessageSeverityFlagsEXT::ERROR;
        assert!(log(severity, None).is_empty());

        let result = unsafe {
            vulkan_debug_callback(
                severity,
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(result, vk::FALSE);
    }
}

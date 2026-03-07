use std::{
    cell::{Cell, LazyCell, OnceCell, UnsafeCell},
    collections::HashMap,
    ffi::{c_char, c_void},
    fmt::{Debug, write},
    mem::MaybeUninit,
    sync::{
        Mutex, MutexGuard, OnceLock, RwLock, RwLockReadGuard,
        atomic::{AtomicBool, AtomicU16, AtomicU32, AtomicU64, Ordering},
    },
    time::Instant,
};

use anyhow::{Result, anyhow};
use ash::{
    Device, Entry,
    ext::debug_utils,
    vk::{self, Handle},
};
use bytemuck::Pod;
use gpu_allocator::{
    AllocationSizes, AllocatorDebugSettings,
    vulkan::{Allocator, AllocatorCreateDesc},
};
use std::ffi::CStr;
use winit::raw_window_handle::{
    HasDisplayHandle, HasRawDisplayHandle, HasRawWindowHandle, HasWindowHandle,
};

use crate::{
    bindless::{Bindless, BindlessHandle},
    buffer::{Buffer},
    command_buffer::{CommandBuffer, ResourceHandle, ResourceState},
    image::{format, slice::ImageSlice, usage::ColorAttachmentStorage},
    vkobjects::{
        physical_device::{PhysicalDevice, QueueFamily},
        queue::{
            Binary, CommandBufferMemory, CommandPool, Queue, Semaphore, SemaphoreInfo, Timeline,
        },
        surface::Surface,
        swapchain::Swapchain,
    },
};

pub use ash::vk as raw_vulkan;

#[cfg(feature = "trace")]
#[macro_export]
macro_rules! tracy_span {
    ($name:expr) => {
        tracy_client::span!($name)
    };
}

#[cfg(not(feature = "trace"))]
#[macro_export]
macro_rules! tracy_span {
    ($name:expr) => {
        ()
    };
}

impl Debug for Ctx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

pub struct Ctx {
    features: Features,
    device: Device,
    physical_device: PhysicalDevice,
    surface: Surface,
    allocator: Mutex<Allocator>,

    pub(crate) gfx_queue_familie: QueueFamily,
    pub(crate) gfx_queues_in_use: Vec<AtomicBool>,

    pub(crate) transfer_queue_familie: Option<QueueFamily>,
    pub(crate) transfer_queues_in_use: Option<Vec<AtomicBool>>,

    pub(crate) present_queue_familie: Option<QueueFamily>,
    pub(crate) present_queues_in_use: Option<Vec<AtomicBool>>,

    // #[cfg(debug_assertions)]
    // printf: Mutex<HashMap<String, usize>>,
    #[cfg(debug_assertions)]
    last_message: Mutex<(u32, String)>,
}

use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
static STATE: OnceLock<Ctx> = OnceLock::new();
impl Ctx {
    #[allow(static_mut_refs)]
    pub(crate) fn get() -> &'static Self {
        STATE.wait()
    }
    pub fn device() -> &'static Device {
        &Ctx::get().device
    }
    pub fn physical_device() -> &'static PhysicalDevice {
        &Ctx::get().physical_device
    }
    pub fn gfx_queue_index() -> u32 {
        Ctx::get().gfx_queue_familie.index
    }
    pub fn num_gfx_queues() -> u32 {
        Ctx::get().gfx_queues_in_use.len() as u32
    }
    pub fn transfer_queue_index() -> u32 {
        Ctx::get()
            .transfer_queue_familie
            .as_ref()
            .map(|e| e.index)
            .unwrap_or(Ctx::get().gfx_queue_familie.index)
    }
    pub fn present_queue_index() -> u32 {
        Ctx::get()
            .present_queue_familie
            .as_ref()
            .map(|e| e.index)
            .unwrap_or(Ctx::get().gfx_queue_familie.index)
    }
    pub fn allocator<'a>() -> MutexGuard<'a, Allocator> {
        unsafe { Ctx::get().allocator.lock().unwrap() }
    }

    pub(crate) fn surface() -> &'static Surface {
        &Ctx::get().surface
    }

    pub fn features() -> Features {
        Ctx::get().features.clone()
    }

    pub fn start_frame() {
        tracy_span!("Wait for Semaphore");
    }

    #[cfg(debug_assertions)]
    pub fn log_debug_printf_output() {
        // let mut lock = Ctx::get().printf.lock().unwrap();
        // let mut messages = lock.iter().collect::<Vec<_>>();
        // if messages.len() > 0 {
        //     log::info!("Printf output this frame:");
        //     messages.sort_by(|(_, a), (_, b)| b.cmp(a));
        //     for (message, value) in messages {
        //         log::info!("    {}x: {}", *value, *message);
        //     }
        // }
        // lock.clear();

        let mut last_message = Ctx::get().last_message.lock().unwrap();
        if last_message.1.len() != 0 {
            if last_message.0 == 1 {
                log::info!("{}", last_message.1);
            }else {
                log::info!("{}x {}", last_message.0, last_message.1);
            }
            last_message.1 = String::new();
            last_message.0 = 1;
            log::info!("---------");
        }
    }

    pub(super) fn init(
        display: &RawDisplayHandle,
        window: &RawWindowHandle,
        enable_validation: bool,
        enable_gpu_assited_validation: bool,
    ) -> Result<()> {
        let entry = unsafe { Entry::load()? };
        let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
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

        let mut instance_extensions = ash_window::enumerate_required_extensions(*display)
            .unwrap()
            .to_vec();

        let mut features = Features::default();
        features.present = true;
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
                    .unwrap()
                    .create_debug_utils_messenger(&debug_info, None)
                    .unwrap()
            };
        }

        let surface =
            unsafe { ash_window::create_surface(&entry, &instance, *display, *window, None) }
                .unwrap();

        let surface_fn = Some(ash::khr::surface::Instance::new(&entry, &instance));

        let physical_devices =
            PhysicalDevice::enumerate_physical_devices(&surface, &instance, surface_fn.as_ref())?;
        let (physical_device, graphics_queue_family, present_queue_familie, transfer_queue_familie) =
            PhysicalDevice::select_suitable_physical_device(
                physical_devices.as_slice(),
                &mut features,
            )?;

        let mut queues = vec![graphics_queue_family.index];
        if let Some(pqf) = &present_queue_familie {
            queues.push(pqf.index);
        }
        if let Some(tqf) = &transfer_queue_familie {
            queues.push(tqf.index);
        }
        let device = create_device(queues, &physical_device, &features, &instance)?;
        let mut debug_utils = None;
        if features.device_debug_utils {
            debug_utils = Some(ash::ext::debug_utils::Device::new(&instance, &device));
        }

        let mut allocator = Allocator::new(&AllocatorCreateDesc {
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

        let surface = Some(Surface::new(
            surface,
            &physical_device,
            &surface_fn.as_ref().unwrap(),
        ));

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
            .unwrap();

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

                present_queues_in_use: if let Some(present) = &present_queue_familie {
                    Some(
                        (0..present.num_queues)
                            .map(|_| AtomicBool::new(false))
                            .collect(),
                    )
                } else {
                    None
                },
                present_queue_familie,

                features: features,
                physical_device: physical_device,
                // printf: Mutex::new(HashMap::new()),
                #[cfg(debug_assertions)]
                last_message: Mutex::new((0, String::new())),
                surface: surface.unwrap(),
            })
            .expect("Faild to initilize Vulkan Context");

        Ok(())
    }
}

unsafe extern "system" fn vulkan_debug_callback(
    flag: vk::DebugUtilsMessageSeverityFlagsEXT,
    typ: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    unsafe {
        use vk::DebugUtilsMessageSeverityFlagsEXT as Flag;
        if p_callback_data != std::ptr::null() && (*p_callback_data).p_message != std::ptr::null() {
            let message = CStr::from_ptr((*p_callback_data).p_message).to_string_lossy();
            #[cfg(debug_assertions)]
            {
                let split = message.split("DebugPrintf:\n").collect::<Vec<_>>();
                if STATE.get().is_some() && split.len() > 1 {
                    let printf_message = split[1..]
                        .iter()
                        .map(|s| s.chars())
                        .flatten()
                        .collect::<String>();
                    if printf_message.len() != 0 {
                        let mut last_message = Ctx::get().last_message.lock().unwrap();
                        if printf_message == last_message.1 {
                            last_message.0 += 1;
                        }else {
                            if last_message.1.len() != 0 {
                                if last_message.0 == 1 {
                                    log::info!("{}", last_message.1);
                                }else {
                                    log::info!("{}x {}", last_message.0, last_message.1);
                                }
                            }
                            last_message.0 = 1;
                            last_message.1 = printf_message;
                        }
                    }
                    return vk::FALSE;
                }
            }

            match flag {
                Flag::VERBOSE => log::info!("{:?} - {}", typ, message),
                Flag::INFO => {
                    log::info!("{:?} - {}", typ, message)
                }
                Flag::WARNING => log::warn!("{}", message),
                Flag::ERROR => log::error!("{}", message),
                _ => {
                    log::info!("{}", message)
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
    ) -> vk::PhysicalDeviceFeatures2<'a> {
        *vk11 = vk11.shader_draw_parameters(true);
        *vk12 = vk12
            .runtime_descriptor_array(true)
            .buffer_device_address(true)
            .descriptor_indexing(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .shader_float16(true)
            .descriptor_binding_storage_buffer_update_after_bind(true)
            .descriptor_binding_partially_bound(true)
            .descriptor_binding_variable_descriptor_count(true)
            .descriptor_binding_storage_image_update_after_bind(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .timeline_semaphore(true)
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
            .synchronization2(true);
        let phfeatures = vk::PhysicalDeviceFeatures::default()
            .shader_int64(true)
            .fill_mode_non_solid(true)
            .fragment_stores_and_atomics(true)
            .shader_int16(true)
            .vertex_pipeline_stores_and_atomics(true);

        *dn3 = dn3
            .extended_dynamic_state3_depth_clamp_enable(true)
            .extended_dynamic_state3_polygon_mode(true)
            .extended_dynamic_state3_logic_op_enable(true)
            .extended_dynamic_state3_color_blend_equation(true)
            .extended_dynamic_state3_color_write_mask(true)
            .extended_dynamic_state3_color_blend_enable(true);
        *dy2 = dy2.extended_dynamic_state2_logic_op(true);

        let mut features = vk::PhysicalDeviceFeatures2::default()
            .features(phfeatures)
            .push_next(vk11)
            .push_next(vk12)
            .push_next(vk13)
            .push_next(dy2)
            .push_next(dn3);
        if self.mesh {
            *mesh = mesh.task_shader(true).mesh_shader(true);
            features = features.push_next(mesh);
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
    pub fn surface() -> Option<&'static ash::khr::surface::Instance> {
        get().surface.as_ref()
    }
    pub fn host_image_copy() -> &'static ash::ext::host_image_copy::Device {
        &get().host_image_copy
    }
    pub fn instance() -> &'static ash::Instance {
        &get().instance
    }
    pub fn entry() -> &'static ash::Entry {
        &get().entry
    }
    pub fn swapchain() -> &'static ash::khr::swapchain::Device {
        &get().swapchain
    }
    pub fn debug_utils() -> Option<&'static ash::ext::debug_utils::Device> {
        get().device_debug_utils.as_ref()
    }
    pub fn instance_debug_utils() -> Option<&'static ash::ext::debug_utils::Instance> {
        get().debug_utils.as_ref()
    }
    pub fn mesh() -> Option<&'static ash::ext::mesh_shader::Device> {
        get().mesh.as_ref()
    }
    pub fn raytracing_pipeline() -> Option<&'static ash::khr::ray_tracing_pipeline::Device> {
        get().raytracing_pipeline.as_ref()
    }
    pub fn acceleration_structure() -> Option<&'static ash::khr::acceleration_structure::Device> {
        get().acceleration_structure.as_ref()
    }

    pub fn set_debug_name<T>(name: &str, object: T)
    where
        T: Handle,
    {
        if let Some(debug_utils) = Self::debug_utils() {
            let name = format!("{}\0", name);
            let name = CStr::from_bytes_with_nul(name.as_bytes()).unwrap();
            let name_info = vk::DebugUtilsObjectNameInfoEXT::default()
                .object_handle(object)
                .object_name(name);
            unsafe { debug_utils.set_debug_utils_object_name(&name_info) }.unwrap();
        }
    }

    pub fn cmd_start_label(cmd: &vk::CommandBuffer, name: &str) {
        if let Some(debug_utils) = Self::debug_utils() {
            let name = format!("{}\0", name);
            let name = CStr::from_bytes_with_nul(name.as_bytes()).unwrap();
            let name_info = vk::DebugUtilsLabelEXT::default().label_name(name);
            unsafe { debug_utils.cmd_begin_debug_utils_label(*cmd, &name_info) };
        }
    }
    pub fn cmd_insert_label(cmd: &vk::CommandBuffer, name: &str) {
        if let Some(debug_utils) = Self::debug_utils() {
            let name = format!("{}\0", name);
            let name = CStr::from_bytes_with_nul(name.as_bytes()).unwrap();
            let name_info = vk::DebugUtilsLabelEXT::default().label_name(name);
            unsafe { debug_utils.cmd_insert_debug_utils_label(*cmd, &name_info) };
        }
    }
    pub fn cmd_end_label(cmd: &vk::CommandBuffer) {
        if let Some(debug_utils) = Self::debug_utils() {
            unsafe { debug_utils.cmd_end_debug_utils_label(*cmd) };
        }
    }
}
fn get() -> &'static Functions {
    FUNCTIONS.get().unwrap()
}

pub(super) fn create_device(
    mut queue_families: Vec<u32>,
    physical_device: &PhysicalDevice,
    features: &Features,
    instance: &ash::Instance,
) -> Result<ash::Device> {
    let queue_priorities = [1.0f32];
    queue_families.dedup();
    let queue_create_infos = queue_families
        .into_iter()
        .map(|i| {
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(i)
                .queue_priorities(&queue_priorities)
        })
        .collect::<Vec<_>>();

    let required_extensions = features.extensions();
    let device_extensions_as_ptr = required_extensions
        .into_iter()
        .map(|e| e.as_ptr() as *const i8)
        .collect::<Vec<_>>();

    let (mut vk11, mut vk12, mut vk13, mut dy2, mut dn3, mut mesh, mut ray, mut acc) =
        Default::default();
    let mut features = features.features(
        &mut vk11, &mut vk12, &mut vk13, &mut dn3, &mut dy2, &mut mesh, &mut ray, &mut acc,
    );
    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_create_infos)
        .enabled_extension_names(device_extensions_as_ptr.as_slice())
        .push_next(&mut features);

    let device = unsafe {
        instance
            .create_device(physical_device.handel, &device_create_info, None)
            .unwrap()
    };

    Ok(device)
}

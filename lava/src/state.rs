//! Global Vulkan context: instance, device selection and creation, queues, features, and validation logging
use std::{
    cell::Cell,
    ffi::{c_char, c_void},
    fmt::Debug,
    panic::Location,
    sync::{Mutex, MutexGuard, OnceLock},
};

use crate::error::{Error, Result};
use crate::vk::{self, Handle, make_api_version};
use ash::{Device, Entry, ext::debug_utils};
use gpu_allocator::{
    AllocationSizes, AllocatorDebugSettings,
    vulkan::{Allocator, AllocatorCreateDesc},
};
use lava_macros::validation_trace;
use std::ffi::{CStr, CString};

#[allow(unused_imports)]
use crate::features::{Loader, gate, load, loaded};
use crate::{
    Config,
    command_buffer::Dynamic,
    pipeline::MAX_PUSH_CONSTANTS_SIZE,
    features::{DeviceFacts, FeatureChain, Features, LAVA_EXTENSIONS},
    queue::{
        Kind,
        DeviceQueues, FamilyQueues, MAX_TIMELINES,
    },
    vkobjects::physical_device::{PhysicalDevice, QueueFamily, pick_queue_families},
};

pub use crate::vk as raw_vulkan;

impl Debug for Ctx {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

pub struct Ctx {
    features: Features,
    /// The draw state the device sets dynamically, from `features`.
    dynamic: Dynamic,
    device: Device,
    physical_device: PhysicalDevice,
    allocator: Mutex<Allocator>,

    pub(crate) queues: DeviceQueues,
}

static STATE: OnceLock<Ctx> = OnceLock::new();

/// Undoes a step of `Ctx::init` when a later one fails, unless disarmed.
struct Undo<F: FnOnce()>(Option<F>);

impl<F: FnOnce()> Undo<F> {
    fn disarm(mut self) {
        self.0 = None;
    }
}

impl<F: FnOnce()> Drop for Undo<F> {
    fn drop(&mut self) {
        if let Some(undo) = self.0.take() {
            undo();
        }
    }
}

impl Ctx {
    pub(crate) fn get() -> &'static Self {
        STATE
            .get()
            .expect("lava isn't initialised: call `lava::init` first (and check that it succeeded)")
    }
    pub(crate) fn device() -> &'static Device {
        &Ctx::get().device
    }
    pub(crate) fn physical_device() -> &'static PhysicalDevice {
        &Ctx::get().physical_device
    }
    pub fn device_properties() -> &'static vk::PhysicalDeviceProperties {
        &Ctx::get().physical_device.properties
    }
    #[cfg(feature = "mesh-shader")]
    pub fn mesh_shader_properties()
    -> crate::features::Gated<gate::MeshShader, &'static vk::PhysicalDeviceMeshShaderPropertiesEXT<'static>>
    {
        crate::features::gate::<gate::MeshShader, _>(|| {
            &Ctx::get().physical_device.mesh_shader_properties
        })
    }
    /// The family index of `kind`'s queues, `None` where lava has none.
    pub fn queue_family(kind: Kind) -> Option<u32> {
        Ctx::get().queues.family(kind).map(|family| family.index)
    }
    /// How many queues lava created of `kind`'s family (at most `MAX_TIMELINES`).
    pub fn queue_count(kind: Kind) -> u32 {
        let family = Ctx::get().queues.family(kind);
        family.map_or(0, |family| family.queues.len() as u32)
    }
    pub(crate) fn allocator<'a>() -> MutexGuard<'a, Allocator> {
        Ctx::get()
            .allocator
            .lock()
            .expect("allocator mutex was poisoned")
    }

    /// The optional features lava enabled: those the device has, minus `Config::disabled`.
    pub fn features() -> Features {
        Ctx::get().features
    }
    pub(crate) fn features_ref() -> &'static Features {
        &Ctx::get().features
    }
    pub(crate) fn dynamic() -> &'static Dynamic {
        &Ctx::get().dynamic
    }

    #[validation_trace]
    pub(super) fn init(config: &Config) -> Result<()> {
        if STATE.get().is_some() {
            return Err(Error::message("lava is already initialised"));
        }
        let display = config.display.as_ref();
        let entry = unsafe { Entry::load()? };
        let app_info = vk::ApplicationInfo::default().api_version(make_api_version(0, 1, 4, 0));
        let layers_names_raw = [c"VK_LAYER_KHRONOS_validation".as_ptr()];

        let mut instance_extensions = match display {
            Some(display) => ash_window::enumerate_required_extensions(*display)
                .map_err(|err| Error::message(&format!("no surface extensions for this display: {err}")))?
                .to_vec(),
            None => Vec::new(),
        };

        let surface_maintenance = [
            ash::khr::get_surface_capabilities2::NAME,
            ash::ext::surface_maintenance1::NAME,
        ];
        let present_fences = cfg!(feature = "present-fences") && display.is_some() && {
            let available = unsafe { entry.enumerate_instance_extension_properties(None)? };
            surface_maintenance.iter().all(|name| {
                (available.iter()).any(|e| e.extension_name_as_c_str() == Ok(*name))
            })
        };
        if present_fences {
            instance_extensions.extend(surface_maintenance.iter().map(|name| name.as_ptr()));
        }

        let debug = cfg!(debug_assertions) && config.validation;
        let mut validation_features = vk::ValidationFeaturesEXT::default();
        let mut validation_f;
        let mut instance_info = vk::InstanceCreateInfo::default();
        if debug {
            instance_extensions.push(ash::ext::debug_utils::NAME.as_ptr());
            validation_f = vec![
                vk::ValidationFeatureEnableEXT::DEBUG_PRINTF,
                vk::ValidationFeatureEnableEXT::BEST_PRACTICES,
                vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION,
            ];
            if config.gpu_assisted_validation {
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
        let undo_instance = Undo(Some({
            let instance = instance.clone();
            move || unsafe { instance.destroy_instance(None) }
        }));
        let mut instance_debug_utils = None;
        let mut messenger = vk::DebugUtilsMessengerEXT::null();
        if debug {
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
            messenger = unsafe {
                instance_debug_utils
                    .as_ref()
                    .expect("debug utils instance was just created")
                    .create_debug_utils_messenger(&debug_info, None)?
            };
        }
        let undo_messenger = Undo(Some({
            let debug_utils = instance_debug_utils.clone();
            move || {
                if let Some(debug_utils) = debug_utils {
                    unsafe { debug_utils.destroy_debug_utils_messenger(messenger, None) };
                }
            }
        }));

        let surface_fn = display.map(|_| ash::khr::surface::Instance::new(&entry, &instance));

        #[allow(unused_mut)]
        let mut required = Features::asserted();
        #[cfg(feature = "assert-present-fences")]
        {
            required.present_fences = display.is_some();
        }
        let asserted = Features::asserted();
        let asserted_and_disabled = asserted.missing_from(&asserted.without(config.disabled));
        if !asserted_and_disabled.is_empty() {
            return Err(Error::message(format!(
                "`Config::disabled` names asserted features: {}",
                asserted_and_disabled.join(", ")
            )));
        }
        let mut physical_devices = unsafe { instance.enumerate_physical_devices()? }
            .into_iter()
            .map(|handel| {
                let properties = unsafe { instance.get_physical_device_properties(handel) };
                let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
                    .to_str()?
                    .to_owned();
                let queue_families =
                    unsafe { instance.get_physical_device_queue_family_properties(handel) }
                        .into_iter()
                        .enumerate()
                        .map(|(index, p)| QueueFamily {
                            num_queues: p.queue_count,
                            index: index as _,
                            handel: p,
                        })
                        .collect::<Vec<_>>();
                let supported_extensions =
                    unsafe { instance.enumerate_device_extension_properties(handel)? }
                        .into_iter()
                        .map(|p| {
                            let name = unsafe { CStr::from_ptr(p.extension_name.as_ptr()) };
                            Ok(name.to_str()?.to_owned())
                        })
                        .collect::<Result<Vec<String>>>()?;
                let has = |name: &CStr| {
                    (supported_extensions.iter()).any(|e| e.as_bytes() == name.to_bytes())
                };
                let chain = FeatureChain::query(&instance, handel, &has);

                let mut vk12_properties = vk::PhysicalDeviceVulkan12Properties::default();
                let mut rt_pipeline_properties =
                    vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
                let mut acc_properties =
                    vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
                let mut mesh_shader_properties = vk::PhysicalDeviceMeshShaderPropertiesEXT::default();
                let mut heap_properties = vk::PhysicalDeviceDescriptorHeapPropertiesEXT::default();
                // Only the structs of extensions the device has.
                let mut properties2 =
                    vk::PhysicalDeviceProperties2::default().push_next(&mut vk12_properties);
                if has(ash::khr::ray_tracing_pipeline::NAME) {
                    properties2 = properties2.push_next(&mut rt_pipeline_properties);
                }
                if has(ash::khr::acceleration_structure::NAME) {
                    properties2 = properties2.push_next(&mut acc_properties);
                }
                if has(ash::ext::mesh_shader::NAME) {
                    properties2 = properties2.push_next(&mut mesh_shader_properties);
                }
                if has(vk::descriptor_heap::NAME) {
                    properties2 = properties2.push_next(&mut heap_properties);
                }
                unsafe { instance.get_physical_device_properties2(handel, &mut properties2) };
                // Detach them from the chain on this stack.
                mesh_shader_properties.p_next = std::ptr::null_mut();
                vk12_properties.p_next = std::ptr::null_mut();
                rt_pipeline_properties.p_next = std::ptr::null_mut();
                acc_properties.p_next = std::ptr::null_mut();
                heap_properties.p_next = std::ptr::null_mut();
                let _ = (&rt_pipeline_properties, &acc_properties);

                let mem_properties =
                    unsafe { instance.get_physical_device_memory_properties(handel) };
                let rebar = mem_properties.memory_types
                    [..mem_properties.memory_type_count as usize]
                    .iter()
                    .any(|e| {
                        e.property_flags.contains(
                            vk::MemoryPropertyFlags::DEVICE_LOCAL
                                | vk::MemoryPropertyFlags::HOST_VISIBLE
                                | vk::MemoryPropertyFlags::HOST_COHERENT,
                        ) && mem_properties.memory_heaps[e.heap_index as usize].size
                            > 500 * 1024 * 1024
                    });
                let families = pick_queue_families(&queue_families);
                let facts = DeviceFacts {
                    rebar,
                    strict_lines: properties.limits.strict_lines == vk::TRUE,
                    dedicated_compute: families.as_ref().is_some_and(|f| f.compute.is_some()),
                    dedicated_transfer: families.as_ref().is_some_and(|f| f.transfer.is_some()),
                };
                let supported_features = chain.optional(&has, facts);
                let mut lava_missing = chain.lava_missing(&has);
                if properties.api_version < make_api_version(0, 1, 4, 0) {
                    lava_missing.push("Vulkan 1.4".into());
                }
                let push_data = MAX_PUSH_CONSTANTS_SIZE as u64;
                if has(vk::descriptor_heap::NAME) && heap_properties.max_push_data_size < push_data {
                    lava_missing.push(format!("{push_data} bytes of push data (maxPushDataSize)"));
                }
                Ok(PhysicalDevice {
                    properties,
                    mesh_shader_properties,
                    descriptor_heap_properties: heap_properties,
                    vk12_properties,
                    mem_properties,
                    handel,
                    name,
                    queue_families,
                    supported_extensions,
                    #[cfg(feature = "acceleration-structure")]
                    acceleration_structure_properties: supported_features
                        .acceleration_structure
                        .then_some(acc_properties),
                    #[cfg(feature = "ray-tracing-pipeline")]
                    ray_tracing_pipeline_properties: supported_features
                        .ray_tracing_pipeline
                        .then_some(rt_pipeline_properties),
                    supported_features,
                    lava_missing,
                })
            })
            .collect::<Result<Vec<PhysicalDevice>>>()?;
        physical_devices.sort_by_key(|pd| match pd.properties.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 0,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
            _ => 2,
        });

        let mut rejected = Vec::new();
        let mut selected = None;
        for device in physical_devices.iter().filter(|device| {
            device.properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
                || device.properties.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
        }) {
            let mut missing = device.lava_missing.clone();
            missing.extend(
                required
                    .missing_from(&device.supported_features)
                    .into_iter()
                    .map(|name| format!("`{name}` (lava asserts it)")),
            );
            let families = pick_queue_families(&device.queue_families);
            if families.is_none() {
                missing.push("a queue family with graphics, compute and timestamps".into());
            }
            match families {
                Some(families) if missing.is_empty() => {
                    selected = Some((device.clone(), families));
                    break;
                }
                _ => rejected.push(format!("{} lacks {}", device.name, missing.join(", "))),
            }
        }
        let Some((physical_device, families)) = selected else {
            return Err(Error::message(&if rejected.is_empty() {
                "no GPU found".to_owned()
            } else {
                format!("no suitable GPU: {}", rejected.join("; "))
            }));
        };
        #[allow(unused_mut)]
        let mut features = physical_device.supported_features;
        #[cfg(feature = "present-fences")]
        {
            features.present_fences &= present_fences;
            if required.present_fences && !features.present_fences {
                return Err(Error::message(
                    "`assert-present-fences` needs the instance extensions VK_KHR_get_surface_capabilities2 and VK_EXT_surface_maintenance1",
                ));
            }
        }
        let features = features.without(config.disabled).consistent();
        let mut extensions: Vec<&CStr> = LAVA_EXTENSIONS.to_vec();
        extensions.extend(features.extensions());
        if display.is_some() {
            extensions.push(ash::khr::swapchain::NAME);
        }

        #[cfg(feature = "async-compute")]
        let compute = families.compute.as_ref().filter(|_| features.async_compute);
        #[cfg(not(feature = "async-compute"))]
        let compute = None;
        #[cfg(feature = "async-transfer")]
        let transfer = families.transfer.as_ref().filter(|_| features.async_transfer);
        #[cfg(not(feature = "async-transfer"))]
        let transfer = None;
        let used = [Some(&families.graphics), compute, transfer];
        let count = |family: &QueueFamily| family.num_queues.min(MAX_TIMELINES as u32);
        let mut queue_families: Vec<(u32, u32)> =
            used.iter().flatten().map(|f| (f.index, count(f))).collect();
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
        let extension_names: Vec<*const c_char> = extensions.iter().map(|e| e.as_ptr()).collect();
        let queried = FeatureChain::query(&instance, physical_device.handel, &|name: &CStr| {
            (physical_device.supported_extensions.iter()).any(|e| e.as_bytes() == name.to_bytes())
        });
        let mut enabled = FeatureChain::enabling(&features, &queried);
        let mut features2 = enabled.chain(&|name| extensions.contains(&name));
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&extension_names)
            .push_next(&mut features2);
        let device =
            unsafe { instance.create_device(physical_device.handel, &device_create_info, None)? };
        let undo_device = Undo(Some({
            let device = device.clone();
            move || unsafe { device.destroy_device(None) }
        }));
        let mut debug_utils = None;
        if debug {
            debug_utils = Some(ash::ext::debug_utils::Device::new(&instance, &device));
        }

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device: physical_device.handel,
            debug_settings: {
                let mut settings = AllocatorDebugSettings::default();
                settings.log_allocations = false;
                settings.log_frees = false;
                settings.log_leaks_on_shutdown = false;
                settings.log_memory_information = true;
                settings
            },
            buffer_device_address: true,
            // In bytes: blocks start at 64 MiB of device memory and 16 MiB of host-visible
            // memory and double with every block that didn't fit an allocation, to 256 and 64.
            allocation_sizes: AllocationSizes::new(64 << 20, 16 << 20)
                .with_max_device_memblock_size(256 << 20)
                .with_max_host_memblock_size(64 << 20),
        })?;

        FUNCTIONS
            .set(Functions {
                #[cfg(feature = "mesh-shader")]
                mesh: load::<gate::MeshShader, _>(features.mesh_shader, || {
                    ash::ext::mesh_shader::Device::new(&instance, &device)
                }),
                #[cfg(feature = "performance-query")]
                performance_query: load::<gate::PerformanceQuery, _>(features.performance_query, || {
                    (
                        ash::khr::performance_query::Instance::new(&entry, &instance),
                        ash::khr::performance_query::Device::new(&instance, &device),
                    )
                }),
                #[cfg(feature = "ray-tracing-pipeline")]
                raytracing_pipeline: load::<gate::RayTracingPipeline, _>(features.ray_tracing_pipeline, || {
                    ash::khr::ray_tracing_pipeline::Device::new(&instance, &device)
                }),
                #[cfg(feature = "acceleration-structure")]
                acceleration_structure: load::<gate::AccelerationStructure, _>(features.acceleration_structure, || {
                    ash::khr::acceleration_structure::Device::new(&instance, &device)
                }),
                #[cfg(feature = "calibrated-timestamps")]
                calibrated_timestamps: load::<gate::CalibratedTimestamps, _>(features.calibrated_timestamps, || {
                    ash::khr::calibrated_timestamps::Device::new(&instance, &device)
                }),
                #[cfg(feature = "host-image-copy")]
                host_image_copy: load::<gate::HostImageCopy, _>(features.host_image_copy, || {
                    ash::ext::host_image_copy::Device::new(&instance, &device)
                }),
                dynamic_state2: ash::ext::extended_dynamic_state2::Device::new(&instance, &device),
                dynamic_state3: ash::ext::extended_dynamic_state3::Device::new(&instance, &device),
                line_rasterization: ash::khr::line_rasterization::Device::new(&instance, &device),
                depth_clamp_control: crate::vk::depth_clamp_control::Device::new(
                    &instance, &device,
                ),
                device_address_commands: crate::vk::device_address_commands::Device::new(
                    &instance, &device,
                ),
                descriptor_heap: crate::vk::descriptor_heap::Device::new(&instance, &device),
                swapchain: ash::khr::swapchain::Device::new(&instance, &device),
                instance,
                entry,
                surface: surface_fn,
                device_debug_utils: debug_utils,
            })
            .map_err(|_| Error::message("Vulkan functions were already initialized"))?;

        let kinds = [Kind::Graphics, Kind::Compute, Kind::Transfer];
        let queues = DeviceQueues(std::array::from_fn(|i| {
            used[i].map(|family| FamilyQueues {
                kind: kinds[i],
                index: family.index,
                timestamp_valid_bits: family.handel.timestamp_valid_bits,
                image_granularity: family.handel.min_image_transfer_granularity,
                queues: (0..count(family))
                    .map(|q| Mutex::new(unsafe { device.get_device_queue(family.index, q) }))
                    .collect(),
                timelines: Mutex::new(vec![0; count(family) as usize].into()),
            })
        }));
        STATE
            .set(Ctx {
                device,
                allocator: Mutex::new(allocator),
                queues,

                dynamic: Dynamic::of(&features),
                features,
                physical_device,
            })
            .map_err(|_| Error::message("Vulkan context was already initialized"))?;
        undo_device.disarm();
        undo_messenger.disarm();
        undo_instance.disarm();

        Ok(())
    }
}

thread_local! {
    pub static CALLSITE: Cell<Option<Location<'static>>> = Cell::new(None);
}

/// Holds the caller location in [`CALLSITE`] while a `#[validation_trace]` function runs; only
/// the outermost guard on a thread sets and clears it.
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

pub struct Functions {
    instance: ash::Instance,
    #[cfg(feature = "host-image-copy")]
    host_image_copy: Loader<gate::HostImageCopy, ash::ext::host_image_copy::Device>,
    // Loaded on every device; called only for dynamic state lava enabled.
    dynamic_state2: ash::ext::extended_dynamic_state2::Device,
    dynamic_state3: ash::ext::extended_dynamic_state3::Device,
    line_rasterization: ash::khr::line_rasterization::Device,
    depth_clamp_control: crate::vk::depth_clamp_control::Device,
    device_address_commands: crate::vk::device_address_commands::Device,
    descriptor_heap: crate::vk::descriptor_heap::Device,
    entry: ash::Entry,
    swapchain: ash::khr::swapchain::Device,
    surface: Option<ash::khr::surface::Instance>,
    device_debug_utils: Option<ash::ext::debug_utils::Device>,
    #[cfg(feature = "mesh-shader")]
    mesh: Loader<gate::MeshShader, ash::ext::mesh_shader::Device>,
    #[cfg(feature = "performance-query")]
    performance_query: Loader<
        gate::PerformanceQuery,
        (
            ash::khr::performance_query::Instance,
            ash::khr::performance_query::Device,
        ),
    >,
    #[cfg(feature = "ray-tracing-pipeline")]
    raytracing_pipeline: Loader<gate::RayTracingPipeline, ash::khr::ray_tracing_pipeline::Device>,
    #[cfg(feature = "acceleration-structure")]
    acceleration_structure:
        Loader<gate::AccelerationStructure, ash::khr::acceleration_structure::Device>,
    #[cfg(feature = "calibrated-timestamps")]
    calibrated_timestamps:
        Loader<gate::CalibratedTimestamps, ash::khr::calibrated_timestamps::Device>,
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
    #[cfg(feature = "host-image-copy")]
    pub(crate) fn host_image_copy() -> &'static ash::ext::host_image_copy::Device {
        loaded::<gate::HostImageCopy, _>(&get().host_image_copy)
    }
    pub(crate) fn device_address_commands() -> &'static crate::vk::device_address_commands::Device {
        &get().device_address_commands
    }
    pub(crate) fn descriptor_heap() -> &'static crate::vk::descriptor_heap::Device {
        &get().descriptor_heap
    }
    pub(crate) fn dynamic_state2() -> &'static ash::ext::extended_dynamic_state2::Device {
        &get().dynamic_state2
    }
    pub(crate) fn dynamic_state3() -> &'static ash::ext::extended_dynamic_state3::Device {
        &get().dynamic_state3
    }
    pub(crate) fn line_rasterization() -> &'static ash::khr::line_rasterization::Device {
        &get().line_rasterization
    }
    pub(crate) fn depth_clamp_control() -> &'static crate::vk::depth_clamp_control::Device {
        &get().depth_clamp_control
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
    #[cfg(feature = "performance-query")]
    pub(crate) fn performance_query() -> &'static (
        ash::khr::performance_query::Instance,
        ash::khr::performance_query::Device,
    ) {
        loaded::<gate::PerformanceQuery, _>(&get().performance_query)
    }
    #[cfg(feature = "mesh-shader")]
    pub(crate) fn mesh() -> &'static ash::ext::mesh_shader::Device {
        loaded::<gate::MeshShader, _>(&get().mesh)
    }
    #[cfg(feature = "calibrated-timestamps")]
    pub(crate) fn calibrated_timestamps() -> &'static ash::khr::calibrated_timestamps::Device {
        loaded::<gate::CalibratedTimestamps, _>(&get().calibrated_timestamps)
    }
    #[cfg(feature = "ray-tracing-pipeline")]
    pub(crate) fn raytracing_pipeline() -> &'static ash::khr::ray_tracing_pipeline::Device {
        loaded::<gate::RayTracingPipeline, _>(&get().raytracing_pipeline)
    }
    #[cfg(feature = "acceleration-structure")]
    pub(crate) fn acceleration_structure() -> &'static ash::khr::acceleration_structure::Device {
        loaded::<gate::AccelerationStructure, _>(&get().acceleration_structure)
    }

    pub(crate) fn set_debug_name<T>(name: &str, object: T)
    where
        T: Handle,
    {
        if let Some(debug_utils) = Self::debug_utils() {
            with_debug_name(name, |name| {
                let name_info = vk::DebugUtilsObjectNameInfoEXT::default()
                    .object_handle(object)
                    .object_name(name);
                if let Err(err) = unsafe { debug_utils.set_debug_utils_object_name(&name_info) } {
                    tracing::error!(%err, "failed to set Vulkan debug name");
                }
            });
        }
    }
}

/// Calls `f` with a debug name as a C string, cut at its first NUL; one of up to 127 bytes is
/// copied to the stack, so labels don't allocate.
pub(crate) fn with_debug_name<R>(name: &str, f: impl FnOnce(&CStr) -> R) -> R {
    let name = name.split('\0').next().unwrap_or_default().as_bytes();
    let mut buffer = [0u8; 128];
    match buffer.get_mut(..name.len()) {
        Some(head) if name.len() < 128 => {
            head.copy_from_slice(name);
            f(CStr::from_bytes_until_nul(&buffer).expect("the buffer ends with a NUL"))
        }
        _ => f(&CString::new(name).expect("interior NULs were cut off")),
    }
}

fn get() -> &'static Functions {
    FUNCTIONS
        .get()
        .expect("Vulkan functions have not been initialized")
}

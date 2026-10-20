//! Global Vulkan context: instance, device ranking, selection and creation, queues, features, format properties and image format limits (cached per format and description), validation logging, and their destruction at shutdown
use std::{
    cell::Cell,
    collections::HashMap,
    ffi::{c_char, c_void},
    fmt::Debug,
    panic::Location,
    sync::{
        Mutex, MutexGuard,
        atomic::{AtomicBool, Ordering},
    },
};

use crate::error::{Error, Result};
use crate::vk::{self, Handle, make_api_version};
use ash::{Device, Entry, ext::debug_utils};
use gpu_allocator::{
    AllocationSizes, AllocatorDebugSettings,
    vulkan::{Allocator, AllocatorCreateDesc},
};
use std::ffi::{CStr, CString};

#[allow(unused_imports)]
use crate::features::{Loader, gate, load, loaded};
use crate::{
    Config,
    command_buffer::Dynamic,
    features::{DeviceFacts, FeatureChain, Features, LAVA_EXTENSIONS},
    global::{Global, Live},
    pipeline::MAX_PUSH_CONSTANTS_SIZE,
    queue::{DeviceQueues, FamilyQueues, Kind, MAX_TIMELINES},
    vkobjects::physical_device::{PhysicalDevice, QueueFamily, pick_queue_families},
};

pub use crate::vk as raw_vulkan;

impl Debug for Ctx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ctx")
            .field("device", &self.physical_device.name)
            .finish_non_exhaustive()
    }
}

/// lava's device and what it knows of it. Never freed (see `global`), so its getters read valid
/// memory even while `lava::shutdown` runs; they panic once lava isn't initialised.
pub struct Ctx {
    features: Features,
    /// The draw state the device sets dynamically, from `features`.
    dynamic: Dynamic,
    device: Device,
    physical_device: PhysicalDevice,
    /// `None` once `shutdown` freed its memory.
    allocator: Mutex<Option<Allocator>>,
    pub(crate) queues: DeviceQueues,
    /// Set by the first poll that saw the device lost, which logs it.
    pub(crate) device_lost: AtomicBool,
    /// The features of each format asked so far (`Ctx::format_properties`).
    format_properties: Mutex<HashMap<vk::Format, FormatProperties>>,
    /// The limits of each image description asked so far (`Ctx::image_format_limits`).
    image_limits: Mutex<HashMap<ImageFormatKey, Option<vk::ImageFormatProperties>>>,
    /// Whether lava enabled `VK_EXT_external_memory_dma_buf`.
    #[allow(dead_code)] // `external-memory`'s.
    pub(crate) dma_buf: bool,
}

/// What a format supports in each tiling and in buffers (`VkFormatProperties3`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FormatProperties {
    pub optimal: vk::FormatFeatureFlags2,
    pub linear: vk::FormatFeatureFlags2,
    pub buffer: vk::FormatFeatureFlags2,
}

/// An image description `vkGetPhysicalDeviceImageFormatProperties2` is asked about, optimal
/// tiling implied; `external` the handle types it must export or import (none: 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ImageFormatKey {
    pub(crate) format: vk::Format,
    pub(crate) image_type: vk::ImageType,
    pub(crate) usage: vk::ImageUsageFlags,
    pub(crate) flags: vk::ImageCreateFlags,
    pub(crate) external: vk::ExternalMemoryHandleTypeFlags,
}

static STATE: Global<Ctx> = Global::new();

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

/// The allocator, locked.
pub(crate) struct AllocatorGuard<'a>(MutexGuard<'a, Option<Allocator>>);

impl std::ops::Deref for AllocatorGuard<'_> {
    type Target = Allocator;
    fn deref(&self) -> &Allocator {
        self.0.as_ref().expect(crate::global::NOT_INITIALISED)
    }
}

impl std::ops::DerefMut for AllocatorGuard<'_> {
    fn deref_mut(&mut self) -> &mut Allocator {
        self.0.as_mut().expect(crate::global::NOT_INITIALISED)
    }
}

impl AllocatorGuard<'_> {
    /// Allocates, counting the bytes for `memory_report`.
    pub(crate) fn allocate(
        &mut self,
        desc: &gpu_allocator::vulkan::AllocationCreateDesc<'_>,
    ) -> gpu_allocator::Result<gpu_allocator::vulkan::Allocation> {
        let allocation = std::ops::DerefMut::deref_mut(self).allocate(desc)?;
        crate::memory::ALLOCATED.fetch_add(allocation.size(), std::sync::atomic::Ordering::Relaxed);
        Ok(allocation)
    }

    /// Frees, counting the bytes for `memory_report`.
    pub(crate) fn free(
        &mut self,
        allocation: gpu_allocator::vulkan::Allocation,
    ) -> gpu_allocator::Result<()> {
        let size = allocation.size();
        std::ops::DerefMut::deref_mut(self).free(allocation)?;
        crate::memory::ALLOCATED.fetch_sub(size, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
}

/// Where lava tries a device of `device_type`: discrete GPUs first, then integrated ones, then
/// virtual GPUs and CPUs (software renderers such as lavapipe) as a last resort.
pub(crate) fn device_rank(device_type: vk::PhysicalDeviceType) -> u8 {
    match device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => 0,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
        vk::PhysicalDeviceType::CPU => 3,
        _ => 4,
    }
}

/// The indices of `types` in the order lava tries them: by [`device_rank`], keeping the
/// driver's order within a rank.
pub(crate) fn device_order(types: &[vk::PhysicalDeviceType]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..types.len()).collect();
    order.sort_by_key(|&i| device_rank(types[i]));
    order
}

/// The debug messenger's severities: errors always, the others where `tracing` logs them when
/// lava is initialised (validation messages at `vulkan-validation` as warn, debug and trace;
/// shader printf as info).
fn enabled_severities() -> vk::DebugUtilsMessageSeverityFlagsEXT {
    use tracing::Level;
    use vk::DebugUtilsMessageSeverityFlagsEXT as S;
    let mut severities = S::ERROR;
    if tracing::enabled!(target: "vulkan-validation", Level::WARN) {
        severities |= S::WARNING;
    }
    if tracing::enabled!(target: "vulkan-validation", Level::DEBUG)
        || tracing::enabled!(Level::INFO)
    {
        severities |= S::INFO;
    }
    if tracing::enabled!(target: "vulkan-validation", Level::TRACE) {
        severities |= S::VERBOSE;
    }
    severities
}

/// What `Ctx::init` learns of one physical device.
fn describe_device(instance: &ash::Instance, handel: vk::PhysicalDevice) -> Result<PhysicalDevice> {
    let properties = unsafe { instance.get_physical_device_properties(handel) };
    let name = properties.device_name_as_c_str().map_or_else(
        |_| "an unnamed device".into(),
        |name| name.to_string_lossy().into_owned(),
    );
    let queue_families = unsafe { instance.get_physical_device_queue_family_properties(handel) }
        .into_iter()
        .enumerate()
        .map(|(index, p)| QueueFamily {
            num_queues: p.queue_count,
            index: index as _,
            handel: p,
        })
        .collect::<Vec<_>>();
    let supported_extensions = unsafe { instance.enumerate_device_extension_properties(handel)? }
        .iter()
        .filter_map(|p| p.extension_name_as_c_str().ok())
        .map(|name| name.to_string_lossy().into_owned())
        .collect::<Vec<String>>();
    let has = |name: &CStr| (supported_extensions.iter()).any(|e| e.as_bytes() == name.to_bytes());
    let mem_properties = unsafe { instance.get_physical_device_memory_properties(handel) };

    let mut vk11_properties = vk::PhysicalDeviceVulkan11Properties::default();
    let mut vk12_properties = vk::PhysicalDeviceVulkan12Properties::default();
    let mut conservative_properties =
        vk::PhysicalDeviceConservativeRasterizationPropertiesEXT::default();
    let mut eds3_properties = vk::PhysicalDeviceExtendedDynamicState3PropertiesEXT::default();
    let mut multi_draw_properties = vk::PhysicalDeviceMultiDrawPropertiesEXT::default();
    let mut multiview_mesh_shader = false;
    let mut rt_pipeline_properties = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
    let mut acc_properties = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
    let mut mesh_shader_properties = vk::PhysicalDeviceMeshShaderPropertiesEXT::default();
    let mut heap_properties = vk::PhysicalDeviceDescriptorHeapPropertiesEXT::default();
    // A device below Vulkan 1.4 is rejected for that alone: the 1.1 to 1.3 structs aren't
    // chained into its queries, which it may not know.
    let vulkan_1_4 = properties.api_version >= make_api_version(0, 1, 4, 0);
    let (supported_features, lava_missing) = if !vulkan_1_4 {
        (Features::DEFAULT, vec!["Vulkan 1.4".to_owned()])
    } else {
        let chain = FeatureChain::query(instance, handel, &has);
        multiview_mesh_shader = chain.multiview_mesh_shader();
        // Only the structs of extensions the device has.
        let mut properties2 = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut vk11_properties)
            .push_next(&mut vk12_properties);
        if has(ash::ext::conservative_rasterization::NAME) {
            properties2 = properties2.push_next(&mut conservative_properties);
        }
        if has(ash::ext::extended_dynamic_state3::NAME) {
            properties2 = properties2.push_next(&mut eds3_properties);
        }
        if has(ash::ext::multi_draw::NAME) {
            properties2 = properties2.push_next(&mut multi_draw_properties);
        }
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
        let rebar = mem_properties.memory_types[..mem_properties.memory_type_count as usize]
            .iter()
            .any(|e| {
                e.property_flags.contains(
                    vk::MemoryPropertyFlags::DEVICE_LOCAL
                        | vk::MemoryPropertyFlags::HOST_VISIBLE
                        | vk::MemoryPropertyFlags::HOST_COHERENT,
                ) && mem_properties.memory_heaps[e.heap_index as usize].size > 500 * 1024 * 1024
            });
        let families = pick_queue_families(&queue_families);
        let facts = DeviceFacts {
            rebar,
            strict_lines: properties.limits.strict_lines == vk::TRUE,
            dedicated_compute: families.as_ref().is_some_and(|f| f.compute.is_some()),
            dedicated_transfer: families.as_ref().is_some_and(|f| f.transfer.is_some()),
            sparse_queue: families.as_ref().is_some_and(|f| {
                f.graphics
                    .handel
                    .queue_flags
                    .contains(vk::QueueFlags::SPARSE_BINDING)
            }),
        };
        let mut lava_missing = chain.lava_missing(&has);
        let push_data = MAX_PUSH_CONSTANTS_SIZE as u64;
        if has(vk::descriptor_heap::NAME) && heap_properties.max_push_data_size < push_data {
            lava_missing.push(format!("{push_data} bytes of push data (maxPushDataSize)"));
        }
        if families.is_none() {
            lava_missing.push("a queue family with graphics, compute and timestamps".into());
        }
        (chain.optional(&has, facts), lava_missing)
    };
    // Detach them from the chain on this stack.
    mesh_shader_properties.p_next = std::ptr::null_mut();
    vk11_properties.p_next = std::ptr::null_mut();
    vk12_properties.p_next = std::ptr::null_mut();
    rt_pipeline_properties.p_next = std::ptr::null_mut();
    acc_properties.p_next = std::ptr::null_mut();
    heap_properties.p_next = std::ptr::null_mut();
    let _ = (&rt_pipeline_properties, &acc_properties);
    Ok(PhysicalDevice {
        properties,
        mesh_shader_properties,
        descriptor_heap_properties: heap_properties,
        vk11_properties,
        vk12_properties,
        primitive_underestimation: conservative_properties.primitive_underestimation == vk::TRUE,
        topology_unrestricted: eds3_properties.dynamic_primitive_topology_unrestricted == vk::TRUE,
        max_multi_draw_count: multi_draw_properties.max_multi_draw_count,
        multiview_mesh_shader,
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
}

impl Ctx {
    #[track_caller]
    pub(crate) fn get() -> &'static Self {
        STATE.get()
    }
    pub(crate) fn device() -> &'static Device {
        &Ctx::get().device
    }
    pub(crate) fn physical_device() -> &'static PhysicalDevice {
        &Ctx::get().physical_device
    }
    /// The device's properties and limits.
    pub fn device_properties() -> &'static vk::PhysicalDeviceProperties {
        &Ctx::get().physical_device.properties
    }
    #[cfg(feature = "mesh-shader")]
    pub fn mesh_shader_properties() -> crate::features::Gated<
        gate::MeshShader,
        &'static vk::PhysicalDeviceMeshShaderPropertiesEXT<'static>,
    > {
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
    pub(crate) fn allocator<'a>() -> AllocatorGuard<'a> {
        AllocatorGuard(crate::queue::lock(&Ctx::get().allocator))
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

    /// What images of `format` in optimal tiling support: `format_properties(format).optimal`.
    pub fn format_features(format: vk::Format) -> vk::FormatFeatureFlags2 {
        Ctx::format_properties(format).optimal
    }

    /// What `format` supports in optimal and linear tiling and in buffers
    /// (`VkFormatProperties3`), queried once per format and then one lock and lookup.
    pub fn format_properties(format: vk::Format) -> FormatProperties {
        let ctx = Ctx::get();
        let mut cache = crate::queue::lock(&ctx.format_properties);
        *cache.entry(format).or_insert_with(|| {
            let mut properties3 = vk::FormatProperties3::default();
            let mut properties = vk::FormatProperties2::default().push_next(&mut properties3);
            unsafe {
                Functions::instance().get_physical_device_format_properties2(
                    ctx.physical_device.handel,
                    format,
                    &mut properties,
                )
            };
            FormatProperties {
                optimal: properties3.optimal_tiling_features,
                linear: properties3.linear_tiling_features,
                buffer: properties3.buffer_features,
            }
        })
    }

    /// The limits of optimal-tiling images of `key`'s description
    /// (`vkGetPhysicalDeviceImageFormatProperties2`), `None` where the device can't make them;
    /// queried once per description. With external handle types, also `None` unless the
    /// device exports and imports them for it.
    pub(crate) fn image_format_limits(key: ImageFormatKey) -> Option<vk::ImageFormatProperties> {
        let ctx = Ctx::get();
        let mut cache = crate::queue::lock(&ctx.image_limits);
        *cache.entry(key).or_insert_with(|| {
            let mut external_info = vk::PhysicalDeviceExternalImageFormatInfo::default()
                .handle_type(vk::ExternalMemoryHandleTypeFlags::from_raw(
                    key.external.as_raw(),
                ));
            let mut info = vk::PhysicalDeviceImageFormatInfo2::default()
                .format(key.format)
                .ty(key.image_type)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(key.usage)
                .flags(key.flags);
            let mut external_properties = vk::ExternalImageFormatProperties::default();
            let mut properties = vk::ImageFormatProperties2::default();
            if !key.external.is_empty() {
                info = info.push_next(&mut external_info);
                properties = properties.push_next(&mut external_properties);
            }
            let result = unsafe {
                Functions::instance().get_physical_device_image_format_properties2(
                    ctx.physical_device.handel,
                    &info,
                    &mut properties,
                )
            };
            let limits = properties.image_format_properties;
            let features = external_properties
                .external_memory_properties
                .external_memory_features;
            let external = key.external.is_empty()
                || features.contains(
                    vk::ExternalMemoryFeatureFlags::EXPORTABLE
                        | vk::ExternalMemoryFeatureFlags::IMPORTABLE,
                );
            (result.is_ok() && external).then_some(limits)
        })
    }

    /// Creates the instance, picks and creates the device, and sets lava's context and
    /// functions. What it made is destroyed again when it fails.
    pub(super) fn init(config: &Config) -> Result<()> {
        let display = config.display.as_ref();
        let entry = unsafe { Entry::load()? };
        let engine_version = make_api_version(
            0,
            env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap_or(0),
            env!("CARGO_PKG_VERSION_MINOR").parse().unwrap_or(0),
            env!("CARGO_PKG_VERSION_PATCH").parse().unwrap_or(0),
        );
        let [major, minor, patch] = config.app_version;
        let mut app_info = vk::ApplicationInfo::default()
            .api_version(make_api_version(0, 1, 4, 0))
            .engine_name(c"lava")
            .engine_version(engine_version)
            .application_version(make_api_version(0, major, minor, patch));
        if let Some(name) = config.app_name {
            app_info = app_info.application_name(name);
        }
        let validation_layer = c"VK_LAYER_KHRONOS_validation";
        let layers_names_raw = [validation_layer.as_ptr()];

        let mut instance_extensions = match display {
            Some(display) => ash_window::enumerate_required_extensions(*display)
                .map_err(|err| {
                    Error::Unsupported(format!("no surface extensions for this display: {err}"))
                })?
                .to_vec(),
            None => Vec::new(),
        };

        let available = unsafe { entry.enumerate_instance_extension_properties(None)? };
        let instance_has =
            |name: &CStr| (available.iter()).any(|e| e.extension_name_as_c_str() == Ok(name));
        let surface_maintenance = [
            ash::khr::get_surface_capabilities2::NAME,
            ash::ext::surface_maintenance1::NAME,
        ];
        let present_fences = cfg!(feature = "present-fences")
            && display.is_some()
            && surface_maintenance.iter().all(|name| instance_has(name));
        if present_fences {
            instance_extensions.extend(surface_maintenance.iter().map(|name| name.as_ptr()));
        }
        // The colour spaces besides sRGB (HDR among them) that surfaces then list.
        let colorspace = ash::ext::swapchain_colorspace::NAME;
        if display.is_some() && instance_has(colorspace) {
            instance_extensions.push(colorspace.as_ptr());
        }

        let validation = config.validation
            && {
                let layers = unsafe { entry.enumerate_instance_layer_properties()? };
                let installed =
                    (layers.iter()).any(|l| l.layer_name_as_c_str() == Ok(validation_layer));
                if !installed {
                    tracing::warn!(
                        "`Config::validation` is set, but the Khronos validation layer isn't installed: lava runs without validation"
                    );
                }
                installed
            };
        let mut validation_features = vk::ValidationFeaturesEXT::default();
        let mut validation_f;
        let mut instance_info = vk::InstanceCreateInfo::default();
        if validation {
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
        let mut debug_messenger = None;
        if validation {
            let debug_utils = debug_utils::Instance::new(&entry, &instance);
            let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(enabled_severities())
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(vulkan_debug_callback));
            let messenger = unsafe { debug_utils.create_debug_utils_messenger(&debug_info, None)? };
            debug_messenger = Some((debug_utils, messenger));
        }
        let undo_messenger = Undo(Some({
            let debug_messenger = debug_messenger.clone();
            move || {
                if let Some((debug_utils, messenger)) = debug_messenger {
                    unsafe { debug_utils.destroy_debug_utils_messenger(messenger, None) };
                }
            }
        }));

        let surface_fn = display.map(|_| ash::khr::surface::Instance::new(&entry, &instance));

        #[allow(unused_mut)]
        let mut required = Features::asserted();
        // Presenting features are asserted only with a display.
        #[cfg(feature = "assert-present-fences")]
        {
            required.present_fences = display.is_some();
        }
        #[cfg(feature = "assert-hdr-metadata")]
        {
            required.hdr_metadata = display.is_some();
        }
        #[cfg(feature = "assert-fifo-latest-ready")]
        {
            required.fifo_latest_ready = display.is_some();
        }
        let handles = unsafe { instance.enumerate_physical_devices()? };
        let mut physical_devices = handles
            .into_iter()
            .map(|handel| describe_device(&instance, handel))
            .collect::<Result<Vec<PhysicalDevice>>>()?;
        let types: Vec<_> = physical_devices
            .iter()
            .map(|d| d.properties.device_type)
            .collect();
        let order = device_order(&types);

        let mut rejected = Vec::new();
        let mut selected = None;
        for &i in &order {
            let device = &physical_devices[i];
            let mut missing = device.lava_missing.clone();
            missing.extend(
                required
                    .missing_from(&device.supported_features)
                    .into_iter()
                    .map(|name| format!("`{name}` (lava asserts it)")),
            );
            match pick_queue_families(&device.queue_families) {
                Some(families) if missing.is_empty() => {
                    selected = Some((i, families));
                    break;
                }
                _ => rejected.push(format!("{} lacks {}", device.name, missing.join(", "))),
            }
        }
        let Some((selected, families)) = selected else {
            return Err(Error::NoSuitableGpu { rejected });
        };
        let physical_device = physical_devices.swap_remove(selected);
        if device_rank(physical_device.properties.device_type) >= 2 {
            tracing::warn!(
                "lava picked {}, a {:?} device, since no discrete or integrated GPU has what it needs",
                physical_device.name,
                physical_device.properties.device_type
            );
        }
        #[allow(unused_mut)]
        let mut features = physical_device.supported_features;
        #[cfg(feature = "hdr-metadata")]
        {
            features.hdr_metadata &= display.is_some();
        }
        #[cfg(feature = "fifo-latest-ready")]
        {
            features.fifo_latest_ready &= display.is_some();
        }
        #[cfg(feature = "present-fences")]
        {
            features.present_fences &= present_fences;
            if required.present_fences && !features.present_fences {
                return Err(Error::Unsupported(
                    "`assert-present-fences` needs the instance extensions VK_KHR_get_surface_capabilities2 and VK_EXT_surface_maintenance1".into(),
                ));
            }
        }
        let features = features.without(config.disabled).consistent();
        let mut extensions: Vec<&CStr> = LAVA_EXTENSIONS.to_vec();
        extensions.extend(features.extensions());
        let device_has = |name: &CStr| {
            (physical_device.supported_extensions.iter()).any(|e| e.as_bytes() == name.to_bytes())
        };
        // dma-bufs where the device has them, besides opaque descriptors.
        #[cfg(feature = "external-memory")]
        let dma_buf =
            features.external_memory && device_has(ash::ext::external_memory_dma_buf::NAME);
        #[cfg(not(feature = "external-memory"))]
        let dma_buf = false;
        if dma_buf {
            extensions.push(ash::ext::external_memory_dma_buf::NAME);
        }
        if display.is_some() {
            extensions.push(ash::khr::swapchain::NAME);
        }

        #[cfg(feature = "async-compute")]
        let compute = families.compute.as_ref().filter(|_| features.async_compute);
        #[cfg(not(feature = "async-compute"))]
        let compute = None;
        #[cfg(feature = "async-transfer")]
        let transfer = families
            .transfer
            .as_ref()
            .filter(|_| features.async_transfer);
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
        let queried = FeatureChain::query(&instance, physical_device.handel, &device_has);
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
        let debug_utils =
            validation.then(|| ash::ext::debug_utils::Device::new(&instance, &device));

        let blocks = config.memory_blocks;
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
            allocation_sizes: AllocationSizes::new(blocks.device, blocks.host)
                .with_max_device_memblock_size(blocks.max_device)
                .with_max_host_memblock_size(blocks.max_host),
        })?;

        FUNCTIONS.set(Functions {
            #[cfg(feature = "mesh-shader")]
            mesh: load::<gate::MeshShader, _>(features.mesh_shader, || {
                ash::ext::mesh_shader::Device::new(&instance, &device)
            }),
            #[cfg(feature = "performance-query")]
            performance_query: load::<gate::PerformanceQuery, _>(
                features.performance_query,
                || {
                    (
                        ash::khr::performance_query::Instance::new(&entry, &instance),
                        ash::khr::performance_query::Device::new(&instance, &device),
                    )
                },
            ),
            #[cfg(feature = "ray-tracing-pipeline")]
            raytracing_pipeline: load::<gate::RayTracingPipeline, _>(
                features.ray_tracing_pipeline,
                || ash::khr::ray_tracing_pipeline::Device::new(&instance, &device),
            ),
            #[cfg(feature = "acceleration-structure")]
            acceleration_structure: load::<gate::AccelerationStructure, _>(
                features.acceleration_structure,
                || ash::khr::acceleration_structure::Device::new(&instance, &device),
            ),
            #[cfg(feature = "calibrated-timestamps")]
            calibrated_timestamps: load::<gate::CalibratedTimestamps, _>(
                features.calibrated_timestamps,
                || ash::khr::calibrated_timestamps::Device::new(&instance, &device),
            ),
            #[cfg(feature = "hdr-metadata")]
            hdr_metadata: load::<gate::HdrMetadata, _>(features.hdr_metadata, || {
                ash::ext::hdr_metadata::Device::new(&instance, &device)
            }),
            #[cfg(feature = "host-image-copy")]
            host_image_copy: load::<gate::HostImageCopy, _>(features.host_image_copy, || {
                ash::ext::host_image_copy::Device::new(&instance, &device)
            }),
            #[cfg(feature = "multi-draw")]
            multi_draw: load::<gate::MultiDraw, _>(features.multi_draw, || {
                ash::ext::multi_draw::Device::new(&instance, &device)
            }),
            #[cfg(feature = "external-memory")]
            external_memory_fd: load::<gate::ExternalMemory, _>(features.external_memory, || {
                ash::khr::external_memory_fd::Device::new(&instance, &device)
            }),
            dynamic_state2: ash::ext::extended_dynamic_state2::Device::new(&instance, &device),
            dynamic_state3: ash::ext::extended_dynamic_state3::Device::new(&instance, &device),
            line_rasterization: ash::khr::line_rasterization::Device::new(&instance, &device),
            depth_clamp_control: crate::vk::depth_clamp_control::Device::new(&instance, &device),
            device_address_commands: crate::vk::device_address_commands::Device::new(
                &instance, &device,
            ),
            descriptor_heap: crate::vk::descriptor_heap::Device::new(&instance, &device),
            swapchain: ash::khr::swapchain::Device::new(&instance, &device),
            instance,
            entry,
            surface: surface_fn,
            device_debug_utils: debug_utils,
            debug_messenger,
        });

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
        STATE.set(Ctx {
            device,
            allocator: Mutex::new(Some(allocator)),
            queues,
            dynamic: Dynamic::of(&features, physical_device.topology_unrestricted),
            features,
            physical_device,
            device_lost: AtomicBool::new(false),
            format_properties: Mutex::new(HashMap::new()),
            image_limits: Mutex::new(HashMap::new()),
            dma_buf,
        });
        undo_device.disarm();
        undo_messenger.disarm();
        undo_instance.disarm();
        Ok(())
    }

    /// Frees the allocator's memory, destroys the device, the debug messenger and the instance,
    /// and unsets the context, which is leaked (see `global`), and the functions.
    ///
    /// # Safety
    /// Everything made from the device is destroyed, and nothing reads the context any more
    /// (`global::close`, under `global::LIFECYCLE`).
    pub(crate) unsafe fn destroy() {
        let Some(ctx) = (unsafe { STATE.take() }) else {
            return;
        };
        let ctx: &'static Ctx = Box::leak(ctx);
        drop(crate::queue::lock(&ctx.allocator).take());
        unsafe { ctx.device.destroy_device(None) };
        if let Some(functions) = unsafe { FUNCTIONS.take() } {
            if let Some((debug_utils, messenger)) = &functions.debug_messenger {
                unsafe { debug_utils.destroy_debug_utils_messenger(*messenger, None) };
            }
            unsafe { functions.instance.destroy_instance(None) };
        }
    }

    /// Whether lava's context exists (between `Ctx::init` and `destroy`).
    pub(crate) fn exists() -> bool {
        STATE.try_get().is_some()
    }
}

/// `err` as lava's error; the first call that sees the device lost logs it.
pub(crate) fn vk_error(err: vk::Result) -> Error {
    if err == vk::Result::ERROR_DEVICE_LOST
        && let Some(ctx) = STATE.try_get()
        && !ctx.device_lost.swap(true, Ordering::Relaxed)
    {
        tracing::error!(
            "the device was lost: lava's calls fail with `Error::DeviceLost` from now on"
        );
    }
    err.into()
}

/// A count for a public call that isn't a method of a lava object: see `global`.
#[track_caller]
pub(crate) fn call() -> Live {
    Live::new()
}

thread_local! {
    pub static CALLSITE: Cell<Option<Location<'static>>> = const { Cell::new(None) };
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

/// Logs the layers' messages as `tracing` events, formatted only where they are logged.
///
/// Validation messages are events of target `vulkan-validation`, at `error`, `warn`, `debug`
/// or `trace` by severity, with the fields `message` (the layer's text), `typ` (the raw
/// `VkDebugUtilsMessageTypeFlagsEXT` bits, an integer), `flags` (the raw
/// `VkDebugUtilsMessageSeverityFlagsEXT` bit, an integer) and `validation_location` (the
/// caller of the `#[validation_trace]` lava function the message came from, if any). Shader
/// `printf` output is an `info` event of target `lava::state` with only a message.
unsafe extern "system" fn vulkan_debug_callback(
    flag: vk::DebugUtilsMessageSeverityFlagsEXT,
    typ: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    use vk::DebugUtilsMessageSeverityFlagsEXT as S;
    if p_callback_data.is_null() || unsafe { (*p_callback_data).p_message.is_null() } {
        return vk::FALSE;
    }
    let message = unsafe { CStr::from_ptr((*p_callback_data).p_message) }.to_bytes();
    const PRINTF: &[u8] = b"DebugPrintf:\n";
    if let Some(at) = message.windows(PRINTF.len()).position(|w| w == PRINTF) {
        let printed = &message[at + PRINTF.len()..];
        tracing::info!("{}", String::from_utf8_lossy(printed));
        return vk::FALSE;
    }
    let text = || String::from_utf8_lossy(message);
    let location = || CALLSITE.get().map(tracing::field::display);
    let (typ, flags) = (typ.as_raw(), flag.as_raw());
    match flag {
        S::ERROR => tracing::error!(
            target: "vulkan-validation", typ, flags, validation_location = location(), "{}", text()
        ),
        S::WARNING => tracing::warn!(
            target: "vulkan-validation", typ, flags, validation_location = location(), "{}", text()
        ),
        S::INFO => tracing::debug!(
            target: "vulkan-validation", typ, flags, validation_location = location(), "{}", text()
        ),
        _ => tracing::trace!(
            target: "vulkan-validation", typ, flags, validation_location = location(), "{}", text()
        ),
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
    /// The validation layer's messenger, destroyed at shutdown.
    debug_messenger: Option<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
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
    #[cfg(feature = "hdr-metadata")]
    hdr_metadata: Loader<gate::HdrMetadata, ash::ext::hdr_metadata::Device>,
    #[cfg(feature = "multi-draw")]
    multi_draw: Loader<gate::MultiDraw, ash::ext::multi_draw::Device>,
    #[cfg(feature = "external-memory")]
    external_memory_fd: Loader<gate::ExternalMemory, ash::khr::external_memory_fd::Device>,
}

impl Debug for Functions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Functions").finish_non_exhaustive()
    }
}

static FUNCTIONS: Global<Functions> = Global::new();

impl Functions {
    pub(crate) fn surface() -> Option<&'static ash::khr::surface::Instance> {
        get().surface.as_ref()
    }
    #[cfg(feature = "host-image-copy")]
    pub(crate) fn host_image_copy() -> &'static ash::ext::host_image_copy::Device {
        loaded::<gate::HostImageCopy, _>(&get().host_image_copy)
    }
    #[cfg(feature = "external-memory")]
    pub(crate) fn external_memory_fd() -> &'static ash::khr::external_memory_fd::Device {
        loaded::<gate::ExternalMemory, _>(&get().external_memory_fd)
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
    #[cfg(feature = "hdr-metadata")]
    pub(crate) fn hdr_metadata() -> &'static ash::ext::hdr_metadata::Device {
        loaded::<gate::HdrMetadata, _>(&get().hdr_metadata)
    }
    #[cfg(feature = "multi-draw")]
    pub(crate) fn multi_draw() -> &'static ash::ext::multi_draw::Device {
        loaded::<gate::MultiDraw, _>(&get().multi_draw)
    }
    #[cfg(feature = "ray-tracing-pipeline")]
    pub(crate) fn raytracing_pipeline() -> &'static ash::khr::ray_tracing_pipeline::Device {
        loaded::<gate::RayTracingPipeline, _>(&get().raytracing_pipeline)
    }
    #[cfg(feature = "acceleration-structure")]
    pub(crate) fn acceleration_structure() -> &'static ash::khr::acceleration_structure::Device {
        loaded::<gate::AccelerationStructure, _>(&get().acceleration_structure)
    }

    /// Names `object` for debuggers and the validation layer, where debug utils are loaded.
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

#[track_caller]
fn get() -> &'static Functions {
    FUNCTIONS.get()
}

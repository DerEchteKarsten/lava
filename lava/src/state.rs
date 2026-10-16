//! Global Vulkan context: instance, device, queues, features, and validation logging
use std::{
    cell::Cell,
    ffi::{c_char, c_void},
    fmt::Debug,
    panic::Location,
    sync::{Mutex, MutexGuard, OnceLock},
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

use crate::{
    Config,
    features::{FeatureChain, Features, LAVA_EXTENSIONS},
    vkobjects::{physical_device::PhysicalDevice, queue::FamilyQueues},
};

pub use crate::vk as raw_vulkan;

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

    pub(crate) gfx_queues: FamilyQueues,
    pub(crate) transfer_queues: Option<FamilyQueues>,
}

static STATE: OnceLock<Ctx> = OnceLock::new();

/// Runs its closure when dropped, unless disarmed: undoes a step of `Ctx::init` when a later
/// one fails, so a failed init leaks nothing.
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
    /// Panics before `lava::init` succeeded.
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
    pub fn mesh_shader_properties() -> &'static vk::PhysicalDeviceMeshShaderPropertiesEXT<'static> {
        &Ctx::get().physical_device.mesh_shader_properties
    }
    pub fn gfx_queue_index() -> u32 {
        Ctx::get().gfx_queues.family
    }
    pub fn num_gfx_queues() -> u32 {
        Ctx::get().gfx_queues.len() as u32
    }
    /// Presentation always goes through the graphics family; `Surface::new` verifies that the
    /// family can present to the surface it wraps.
    pub fn present_queue_index() -> u32 {
        Ctx::get().gfx_queues.family
    }
    pub(crate) fn allocator<'a>() -> MutexGuard<'a, Allocator> {
        Ctx::get()
            .allocator
            .lock()
            .expect("allocator mutex was poisoned")
    }

    /// The optional features lava enabled: every one the device has.
    pub fn features() -> Features {
        Ctx::get().features
    }

    /// Creates the instance, device, and allocator. With a display, the instance gets the
    /// platform surface extensions and the device gets a swapchain; without it lava runs headless.
    pub(super) fn init(config: &Config) -> Result<()> {
        if STATE.get().is_some() {
            return Err(Error::message("lava is already initialised"));
        }
        let display = config.display.as_ref();
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
            Some(display) => ash_window::enumerate_required_extensions(*display)
                .map_err(|err| Error::message(&format!("no surface extensions for this display: {err}")))?
                .to_vec(),
            None => Vec::new(),
        };

        // Present fences (`Features::present_fences`) need the surface maintenance instance
        // extensions.
        let surface_maintenance = [
            ash::khr::get_surface_capabilities2::NAME,
            ash::ext::surface_maintenance1::NAME,
        ];
        let present_fences = display.is_some() && {
            let available = unsafe { entry.enumerate_instance_extension_properties(None)? };
            surface_maintenance.iter().all(|name| {
                (available.iter()).any(|e| e.extension_name_as_c_str() == Ok(*name))
            })
        };
        if present_fences {
            instance_extensions.extend(surface_maintenance.iter().map(|name| name.as_ptr()));
        }

        // The validation layer and debug names and labels, in debug builds.
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
        // Destroyed before the instance.
        let undo_messenger = Undo(Some({
            let debug_utils = instance_debug_utils.clone();
            move || {
                if let Some(debug_utils) = debug_utils {
                    unsafe { debug_utils.destroy_debug_utils_messenger(messenger, None) };
                }
            }
        }));

        let surface_fn = display.map(|_| ash::khr::surface::Instance::new(&entry, &instance));

        let physical_devices = PhysicalDevice::enumerate_physical_devices(&instance)?;
        let (physical_device, graphics_queue_family, transfer_queue_familie) =
            PhysicalDevice::select_suitable_physical_device(
                physical_devices.as_slice(),
                &config.required,
            )?;
        let mut features = physical_device.supported_features;
        features.present_fences &= present_fences;
        if config.required.present_fences && !features.present_fences {
            return Err(Error::message(
                "`present_fences` needs a display and the instance extensions VK_KHR_get_surface_capabilities2 and VK_EXT_surface_maintenance1",
            ));
        }
        let mut extensions: Vec<&CStr> = LAVA_EXTENSIONS.to_vec();
        extensions.extend(features.extensions());
        if display.is_some() {
            extensions.push(ash::khr::swapchain::NAME);
        }

        let mut queues = vec![(
            graphics_queue_family.index,
            graphics_queue_family.num_queues,
        )];
        if let Some(tqf) = &transfer_queue_familie {
            queues.push((tqf.index, tqf.num_queues));
        }
        let device = create_device(queues, &physical_device, &features, &extensions, &instance)?;
        // Destroyed before the messenger and the instance, after the allocator.
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
                settings.log_allocations = true;
                settings.log_frees = true;
                settings.log_leaks_on_shutdown = false;
                settings.log_memory_information = true;
                settings
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
                performance_query: features.performance_query.then(|| {
                    (
                        ash::khr::performance_query::Instance::new(&entry, &instance),
                        ash::khr::performance_query::Device::new(&instance, &device),
                    )
                }),
                raytracing_pipeline: features.ray_tracing_pipeline.then(|| {
                    ash::khr::ray_tracing_pipeline::Device::new(&instance, &device)
                }),
                acceleration_structure: features.acceleration_structure.then(|| {
                    ash::khr::acceleration_structure::Device::new(&instance, &device)
                }),
                calibrated_timestamps: features.calibrated_timestamps.then(|| {
                    ash::khr::calibrated_timestamps::Device::new(&instance, &device)
                }),
                host_image_copy: ash::ext::host_image_copy::Device::new(&instance, &device),
                dynamic_state2: ash::ext::extended_dynamic_state2::Device::new(&instance, &device),
                dynamic_state3: ash::ext::extended_dynamic_state3::Device::new(&instance, &device),
                color_write: ash::ext::color_write_enable::Device::new(&instance, &device),
                line_rasterization: ash::khr::line_rasterization::Device::new(&instance, &device),
                depth_clamp_control: crate::vk::depth_clamp_control::Device::new(
                    &instance, &device,
                ),
                device_address_commands: crate::vk::device_address_commands::Device::new(
                    &instance, &device,
                ),
                swapchain: ash::khr::swapchain::Device::new(&instance, &device),
                instance,
                entry,
                surface: surface_fn,
                debug_utils: instance_debug_utils,
                device_debug_utils: debug_utils,
            })
            .map_err(|_| Error::message("Vulkan functions were already initialized"))?;

        let gfx_queues = FamilyQueues::new(
            &device,
            graphics_queue_family.index,
            graphics_queue_family.num_queues,
            graphics_queue_family.handel.queue_flags,
            true,
        );
        let transfer_queues = transfer_queue_familie.map(|family| {
            let flags = family.handel.queue_flags;
            FamilyQueues::new(&device, family.index, family.num_queues, flags, false)
        });
        STATE
            .set(Ctx {
                device,
                allocator: Mutex::new(allocator),
                gfx_queues,
                transfer_queues,

                features: features,
                physical_device: physical_device,
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

pub struct Functions {
    instance: ash::Instance,
    host_image_copy: ash::ext::host_image_copy::Device,
    dynamic_state2: ash::ext::extended_dynamic_state2::Device,
    dynamic_state3: ash::ext::extended_dynamic_state3::Device,
    color_write: ash::ext::color_write_enable::Device,
    line_rasterization: ash::khr::line_rasterization::Device,
    depth_clamp_control: crate::vk::depth_clamp_control::Device,
    device_address_commands: crate::vk::device_address_commands::Device,
    entry: ash::Entry,
    swapchain: ash::khr::swapchain::Device,
    surface: Option<ash::khr::surface::Instance>,
    debug_utils: Option<ash::ext::debug_utils::Instance>,
    device_debug_utils: Option<ash::ext::debug_utils::Device>,
    mesh: Option<ash::ext::mesh_shader::Device>,
    performance_query: Option<(
        ash::khr::performance_query::Instance,
        ash::khr::performance_query::Device,
    )>,
    raytracing_pipeline: Option<ash::khr::ray_tracing_pipeline::Device>,
    acceleration_structure: Option<ash::khr::acceleration_structure::Device>,
    calibrated_timestamps: Option<ash::khr::calibrated_timestamps::Device>,
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
    pub(crate) fn device_address_commands() -> &'static crate::vk::device_address_commands::Device {
        &get().device_address_commands
    }
    pub(crate) fn dynamic_state2() -> &'static ash::ext::extended_dynamic_state2::Device {
        &get().dynamic_state2
    }
    pub(crate) fn dynamic_state3() -> &'static ash::ext::extended_dynamic_state3::Device {
        &get().dynamic_state3
    }
    pub(crate) fn color_write() -> &'static ash::ext::color_write_enable::Device {
        &get().color_write
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
    pub(crate) fn instance_debug_utils() -> Option<&'static ash::ext::debug_utils::Instance> {
        get().debug_utils.as_ref()
    }
    pub(crate) fn performance_query() -> Option<&'static (
        ash::khr::performance_query::Instance,
        ash::khr::performance_query::Device,
    )> {
        get().performance_query.as_ref()
    }
    pub(crate) fn mesh() -> Option<&'static ash::ext::mesh_shader::Device> {
        get().mesh.as_ref()
    }
    pub(crate) fn calibrated_timestamps()
    -> Option<&'static ash::khr::calibrated_timestamps::Device> {
        get().calibrated_timestamps.as_ref()
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
    extensions: &[&CStr],
    instance: &ash::Instance,
) -> Result<ash::Device> {
    // One priority per queue: every queue of a selected family is created, so that threads
    // can each get their own (`FamilyQueues`).
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
    let mut enabled = FeatureChain::enabling(features);
    let mut features2 = enabled.chain(&|name| extensions.contains(&name));
    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_create_infos)
        .enabled_extension_names(&extension_names);
    // `push_next` links `features2` with its whole chain; every struct in it lives in `enabled`.
    let device_create_info = device_create_info.push_next(&mut features2);

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

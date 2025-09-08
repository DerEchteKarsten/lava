use std::{
    cell::LazyCell,
    collections::HashMap,
    ffi::{c_char, CStr},
    mem::MaybeUninit,
    os::raw::c_void,
    sync::{LazyLock, Mutex, Once, OnceLock, RwLock},
};

use anyhow::Result;
use ash::{
    ext::{debug_utils, extended_dynamic_state3, mesh_shader},
    khr::*,
    vk::{self, Handle},
    Device, Entry, Instance,
};
use bevy_ecs::{resource::Resource, world::FromWorld};
use device::{create_device, DEVICE_EXTENSIONS};
use gpu_allocator::{
    vulkan::{Allocator, AllocatorCreateDesc},
    AllocationSizes, AllocatorDebugSettings,
};
use physical_device::{PhysicalDevice, QueueFamily};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::{dpi::PhysicalSize, event_loop::EventLoop, window::WindowAttributes};

use crate::WINDOW_SIZE;

pub static CTX: OnceLock<Context> = OnceLock::new();

pub struct Context {
    pub device: Device,
    pub physical_device: PhysicalDevice,
    pub instance: ash::Instance,
    pub graphics_queue: vk::Queue,
    pub graphics_queue_family: QueueFamily,
    pub present_queue: vk::Queue,
    pub present_queue_family: QueueFamily,
    pub surface: Surface,
    pub allocator: Mutex<Allocator>,
    pub command_pool: vk::CommandPool,
    pub debug_utils: ash::ext::debug_utils::Device,
    pub _entry: ash::Entry,
    pub mesh_fn: Option<ash::ext::mesh_shader::Device>,
    pub transfer_queue: vk::Queue,
    pub transfer_queue_family: QueueFamily,
}

impl Context {

    pub fn get() -> &'static Self {
        CTX.get().unwrap()
    }

    pub fn new(
        display_handle: &dyn HasDisplayHandle,
        window_handle: &dyn HasWindowHandle,
    ) -> Result<Self> {
        let required_extensions = DEVICE_EXTENSIONS
            .into_iter()
            .map(|e| e.to_str().unwrap())
            .collect::<Vec<_>>();
        let required_extensions = required_extensions.as_slice();

        let entry = unsafe { Entry::load()? };

        let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);

        let layer_names = unsafe {
            [CStr::from_bytes_with_nul_unchecked(
                b"VK_LAYER_KHRONOS_validation\0",
            )]
        };

        let layers_names_raw: Vec<*const c_char> = layer_names
            .iter()
            .map(|raw_name| raw_name.as_ptr())
            .collect();

        let mut instance_extensions = ash_window::enumerate_required_extensions(
            display_handle.display_handle().unwrap().into(),
        )
        .unwrap()
        .to_vec();

        //#[cfg(debug_assertions)]
        instance_extensions.push(debug_utils::NAME.as_ptr());

        let mut validation_features = vk::ValidationFeaturesEXT::default()
            .enabled_validation_features(&[vk::ValidationFeatureEnableEXT::DEBUG_PRINTF]);

        let mut instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&instance_extensions);

        //#[cfg(debug_assertions)]
        {
            instance_info = instance_info
                .enabled_layer_names(&layers_names_raw)
                .push_next(&mut validation_features);
        }

        let instance = unsafe { entry.create_instance(&instance_info, None)? };

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
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                    | vk::DebugUtilsMessageTypeFlagsEXT::DEVICE_ADDRESS_BINDING,
            )
            .pfn_user_callback(Some(Self::vulkan_debug_callback));
        let debug_utils_loader = debug_utils::Instance::new(&entry, &instance);
        unsafe {
            debug_utils_loader
                .create_debug_utils_messenger(&debug_info, None)
                .unwrap()
        };

        let vk_surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                display_handle.display_handle().unwrap().into(),
                window_handle.window_handle().unwrap().into(),
                None,
            )
        }
        .unwrap();
        let ash_surface = ash::khr::surface::Instance::new(&entry, &instance);

        let physical_devices =
            Self::enumerate_physical_devices(&instance, &ash_surface, &vk_surface)?;
        let (physical_device, graphics_queue_family, present_queue_family, transfer_queue_family) =
            Self::select_suitable_physical_device(physical_devices.as_slice(), required_extensions)?;

        let queue_families = [
            &graphics_queue_family,
            &present_queue_family,
            &transfer_queue_family,
        ];

        let device = create_device(
            &instance,
            &physical_device,
            &queue_families,
            required_extensions,
        )?;

        let graphics_queue = unsafe { device.get_device_queue(graphics_queue_family.index, 0) };
        let present_queue = unsafe { device.get_device_queue(present_queue_family.index, 0) };
        let transfer_queue = unsafe { device.get_device_queue(transfer_queue_family.index, 0) };

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device: physical_device.handel,
            debug_settings: AllocatorDebugSettings {
                log_allocations: false,
                log_frees: false,
                log_leaks_on_shutdown: false,
                log_memory_information: false,
                ..Default::default()
            },
            buffer_device_address: true,
            allocation_sizes: AllocationSizes::new(64, 64),
        })?;

        let command_pool_info =
            vk::CommandPoolCreateInfo::default().queue_family_index(graphics_queue_family.index);
        let command_pool = unsafe { device.create_command_pool(&command_pool_info, None)? };

        let debug_utils = debug_utils::Device::new(&instance, &device);
        

        let mut mesh_fn = None;
        #[cfg(not(feature = "no_mesh_shading"))]
        {
            mesh_fn = Some(mesh_shader::Device::new(&instance, &device));
        }


        Ok(Self {
            transfer_queue,
            transfer_queue_family,
            mesh_fn,
            allocator: Mutex::new(allocator),
            surface: Surface {
                ash: ash_surface,
                vulkan: vk_surface,
            },
            present_queue,
            graphics_queue,
            graphics_queue_family,
            present_queue_family,
            device,
            physical_device,
            instance,
            command_pool,
            _entry: entry,
            debug_utils,
        })
    }

    fn enumerate_physical_devices(
        instance: &Instance,
        ash_surface: &ash::khr::surface::Instance,
        vk_surface: &vk::SurfaceKHR,
    ) -> Result<Vec<PhysicalDevice>> {
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };

        let mut physical_devices = physical_devices
            .into_iter()
            .map(|pd| PhysicalDevice::new(&instance, ash_surface, vk_surface, pd))
            .collect::<Result<Vec<PhysicalDevice>>>()?;

        physical_devices.sort_by_key(|pd| match pd.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 0,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
            _ => 2,
        });
        Ok(physical_devices)
    }

    fn select_suitable_physical_device(
        devices: &[PhysicalDevice],
        required_extensions: &[&str],
    ) -> Result<(PhysicalDevice, QueueFamily, QueueFamily, QueueFamily)> {
        let mut graphics = None;
        let mut present = None;
        let mut transfer_queue = None;

        let device = devices
            .iter()
            .find(|device| {
                for family in device.queue_families.iter().filter(|f| f.has_queues()) {
                    if family.supports_graphics()
                        && family.supports_compute()
                        && family.supports_timestamp_queries()
                        && graphics.is_none()
                    {
                        graphics = Some(family.clone());
                    }

                    if family.supports_present() && present.is_none() {
                        present = Some(family.clone());
                    }
                    if family.supports_transfer() && transfer_queue.is_none() {
                        transfer_queue = Some(family.clone());
                    }

                    if graphics.is_some() && present.is_some() {
                        break;
                    }
                }

                let extention_support = device.supports_extensions(required_extensions);
                graphics.is_some()
                    && present.is_some()
                    && extention_support
                    && !device.supported_surface_formats.is_empty()
                    && !device.supported_present_modes.is_empty()
                    && (device.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
                        || device.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU)
            })
            .ok_or_else(|| anyhow::anyhow!("Could not find a suitable device"))?;
        println!("{}", device.name);
        Ok((
            device.clone(),
            graphics.unwrap(),
            present.unwrap(),
            transfer_queue.unwrap(),
        ))
    }

    pub fn execute_one_time_commands<R, F: FnOnce(&vk::CommandBuffer) -> R>(
        &self,
        executor: F,
    ) -> Result<R> {
        let command_buffer = unsafe {
            self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_buffer_count(1)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_pool(self.command_pool),
            )?
        }[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)?
        };

        let executor_result = executor(&command_buffer);

        unsafe { self.device.end_command_buffer(command_buffer)? };

        let fence = unsafe {
            self.device
                .create_fence(&vk::FenceCreateInfo::default(), None)?
        };

        let cmd_buffer_submit_info =
            vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer);

        let submit_info = vk::SubmitInfo2::default()
            .command_buffer_infos(std::slice::from_ref(&cmd_buffer_submit_info));

        unsafe {
            self.device.queue_submit2(
                self.transfer_queue,
                std::slice::from_ref(&submit_info),
                fence,
            )?
        };

        unsafe { self.device.wait_for_fences(&[fence], true, u64::MAX)? };
        unsafe {
            self.device
                .free_command_buffers(self.command_pool, &[command_buffer])
        };

        Ok(executor_result)
    }

    pub fn set_debug_name<T>(&self, name: &str, object: T)
    where
        T: Handle,
    {
        let name = format!("{}\0", name);
        let name = CStr::from_bytes_with_nul(name.as_bytes()).unwrap();
        let name_info = vk::DebugUtilsObjectNameInfoEXT::default()
            .object_handle(object)
            .object_name(name);
        unsafe { self.debug_utils.set_debug_utils_object_name(&name_info) }.unwrap();
    }

    pub fn cmd_start_label(&self, cmd: &vk::CommandBuffer, name: &str) {
        let name = format!("{}\0", name);
        let name = CStr::from_bytes_with_nul(name.as_bytes()).unwrap();
        let name_info = vk::DebugUtilsLabelEXT::default().label_name(name);
        unsafe {
            self.debug_utils
                .cmd_begin_debug_utils_label(*cmd, &name_info)
        };
    }
    pub fn cmd_insert_label(&self, cmd: &vk::CommandBuffer, name: &str) {
        let name = format!("{}\0", name);
        let name = CStr::from_bytes_with_nul(name.as_bytes()).unwrap();
        let name_info = vk::DebugUtilsLabelEXT::default().label_name(name);
        unsafe {
            self.debug_utils
                .cmd_insert_debug_utils_label(*cmd, &name_info)
        };
    }
    pub fn cmd_end_label(&self, cmd: &vk::CommandBuffer) {
        unsafe { self.debug_utils.cmd_end_debug_utils_label(*cmd) };
    }

    pub fn submit(
        &self,
        cmd: &vk::CommandBuffer,
        wait_semaphores: &[vk::SemaphoreSubmitInfo<'_>],
        signal_semaphores: &[vk::SemaphoreSubmitInfo<'_>],
    ) {
        let command_buffers = [vk::CommandBufferSubmitInfo::default().command_buffer(*cmd)];

        let submit = vk::SubmitInfo2::default()
            .command_buffer_infos(&command_buffers)
            .signal_semaphore_infos(signal_semaphores)
            .wait_semaphore_infos(wait_semaphores);
        unsafe {
            self.device
                .queue_submit2(self.graphics_queue, &[submit], vk::Fence::null())
                .unwrap()
        };
    }
}

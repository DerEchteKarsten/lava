use std::{default, ffi::{c_char, c_void}, mem::MaybeUninit, sync::{atomic::{ AtomicU64}, Arc, Mutex, MutexGuard, OnceLock}};

use anyhow::Result;
use ash::{ext::debug_utils, vk::{self, Handle}, Device, Entry};
use gpu_allocator::{vulkan::{Allocator, AllocatorCreateDesc}, AllocationSizes, AllocatorDebugSettings};
use winit::raw_window_handle::{HasDisplayHandle, HasRawDisplayHandle, HasRawWindowHandle, HasWindowHandle};
use std::ffi::CStr;

use crate::{vkobjects::{physical_device::PhysicalDevice, queue::{CommandBuffer, Queue}, surface::Surface, swapchain::Swapchain}, FRAMES_IN_FLIGHT};


pub struct Frame {
    fence: vk::Fence,                          // fence-per-frame for CPU recycling
    image_available: vk::Semaphore,            // binary, signaled by acquire
    render_finished: vk::Semaphore,            // binary, waited by present
    pool: vk::CommandPool,
    cmd: CommandBuffer,
    ticket: u64,                                // graphics timeline value signaled by this frame
}

#[derive(Default)]
pub struct Ctx {
    device: Device,
    physical_device: PhysicalDevice,
    surface: Surface,
    swapchain: Swapchain,
    timeline: vk::Semaphore,
    frames: [Frame; FRAMES_IN_FLIGHT],
    queue: Queue,
    frame_in_flight: usize,
    frame_counter: u64,
    transfer_queue: Option<Queue>,
    present_queue: Option<Queue>,
    allocator: Mutex<Allocator>,
}

impl Ctx {
    pub fn device() -> &'static Device {
        &STATE.get().unwrap().device
    }
    pub fn physical_device() -> &'static Device {
        &STATE.get().unwrap().physical_device
    }
    pub fn queue() -> &'static Queue {
        STATE.get().unwrap().queue
    }
    pub fn transfer_queue() -> &'static Queue {
        let state = STATE.get().unwrap();
        &state.transfer_queue.unwrap_or(state.queue)
    }
    pub fn present_queue() -> &'static Queue {
        let state = STATE.get().unwrap();
        &state.present_queue.unwrap_or(state.queue)
    }
    pub fn allocator<'a>() -> MutexGuard<'a, Allocator> {
        STATE.get().unwrap().allocator.lock().unwrap()
    }
    pub fn surface() -> &'static Surface {
        &STATE.get().unwrap().surface
    }

    pub fn next_frame<'a, F: FnOnce(&'a vk::CommandBuffer) -> Result<()>>(func: F) -> Result<()> {
        STATE.get_mut().unwrap().frame_in_flight = (STATE.get().unwrap().frame_in_flight + 1) % FRAMES_IN_FLIGHT;
        let s = STATE.get().unwrap();
        let f = &s.frames[s.frame_in_flight as usize];
        unsafe {
            Ctx::device().wait_for_fences(&[f.fence], true, u64::MAX)?;
            Ctx::device().device.reset_fences(&[f.fence])?;
            Ctx::device().device.reset_command_pool(f.pool, vk::CommandPoolResetFlags::empty())?;
        }

        let (image_index, _suboptimal) = unsafe {
            Functions::acquire_next_image(s.swapchain.handle, u64::MAX, f.image_available, vk::Fence::null())
        }?;

        f.cmd.record(func)?;

        STATE.get_mut().unwrap().frame_counter += 1;
        let waits = [
            vk::SemaphoreSubmitInfo {
                semaphore: f.image_available,
                stage_mask: vk::PipelineStageFlags2::NONE,
                ..Default::default()
            }
        ];

        let cb_info = vk::CommandBufferSubmitInfo::default().command_buffer(f.gcmd);
        let sig_render_finished = vk::SemaphoreSubmitInfo {
                semaphore: f.render_finished,
                stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                ..Default::default()
            };

        let sig_graphics_timeline = vk::SemaphoreSubmitInfo {
                semaphore: s.timeline,
                value: s.frame_counter,
                stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                ..Default::default()
            };

        let signals = [sig_render_finished, sig_graphics_timeline];

        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&waits)
            .command_buffer_infos(std::slice::from_ref(&cb_info))
            .signal_semaphore_infos(&signals);

        unsafe { Ctx::device().queue_submit2(Ctx::queue(), std::slice::from_ref(&submit), f.fence)?; }

        let swapchains = [s.swapchain.handle];
        let indices = [image_index];
        let wait_sems = [f.render_finished];
        let present = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_sems)
            .swapchains(&swapchains)
            .image_indices(&indices);
        unsafe { Functions::swapchain().queue_present(Ctx::present_queue(), &present)?; }
    }

    pub(super) fn init(display_handle: &dyn HasDisplayHandle, window_handle: &dyn HasWindowHandle) -> Result<()> {
        FUNCTIONS.set(Functions::default())?;
        FUNCTIONS.get_mut()?.entry = unsafe { Entry::load()? };

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

        let mut instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&instance_extensions);

        let mut features = Features::default();
        #[cfg(debug_assertions)]
        {
            features.debug_utils = true;
        }

        if features.debug_utils {
            instance_extensions.push(ash::ext::debug_utils::NAME.as_ptr());
            let mut validation_features = vk::ValidationFeaturesEXT::default()
                .enabled_validation_features(&[vk::ValidationFeatureEnableEXT::DEBUG_PRINTF]);
            instance_info = instance_info
                .enabled_layer_names(&layers_names_raw)
                .push_next(&mut validation_features);
        }
        
        FUNCTIONS.get_mut()?.instance = unsafe { Functions::entry().create_instance(&instance_info, None)? };
        
        if features.debug_utils {
            FUNCTIONS.get_mut()?.debug_utils_loader = debug_utils::Instance::new(&Functions::entry(), Functions::instance());
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
            unsafe {
                Functions::debug_utils()
                    .create_debug_utils_messenger(&debug_info, None)
                    .unwrap()
            };
        }

        let surface = unsafe {
            ash_window::create_surface(
                Functions::entry(),
                Functions::instance(),
                display_handle.display_handle().unwrap().into(),
                window_handle.window_handle().unwrap().into(),
                None,
            )
        }
        .unwrap();
    
        FUNCTIONS.get_mut()?.surface = ash::khr::surface::Instance::new(&Functions::entry(), &Functions::instance());
        
        let physical_devices = PhysicalDevice::enumerate_physical_devices(&surface)?;
        let (physical_device, graphics_queue_family, present_queue_family, transfer_queue_family) =
            PhysicalDevice::select_suitable_physical_device(physical_devices.as_slice())?;

        let mut queue_families = vec![graphics_queue_family.index];
        if let Some(present_queue_family) = present_queue_family {
            queue_families.push(present_queue_family.index);
        }
        if let Some(transfer_queue_family) = transfer_queue_family {
            queue_families.push(transfer_queue_family.index);
        }

        let device = create_device(
            &queue_families,
            &features,
            &physical_device,
        )?;

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: Functions::instance().clone(),
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

        STATE.set(Self {
            allocator,
            device,
            physical_device,
            ..Default::default()
        })?;
        
        let state = STATE.get_mut().unwrap();
        state.surface = Surface::new(surface);
        state.swapchain = Swapchain::new()?;
        state.queue = Queue::new(0);
        state.present_queue = Queue::new(1);
        state.transfer_queue = Queue::new(2);

        Ok(())
    }
}

static STATE: OnceLock<Ctx> = OnceLock::new();

unsafe extern "system" fn vulkan_debug_callback(
    flag: vk::DebugUtilsMessageSeverityFlagsEXT,
    typ: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    use vk::DebugUtilsMessageSeverityFlagsEXT as Flag;
    if p_callback_data != std::ptr::null() && (*p_callback_data).p_message != std::ptr::null() {
        let message = CStr::from_ptr((*p_callback_data).p_message);
        match flag {
            Flag::VERBOSE => log::info!("{:?} - {:?}", typ, message),
            Flag::INFO => {
                let message = message.to_str().unwrap_or("");
                log::info!("{:?} - {:?}", typ, message.to_owned())
            }
            Flag::WARNING => log::warn!("{:?}", message),
            Flag::ERROR => log::error!("{:?}", message),
            _ => {}
        }
    }
    vk::FALSE
}

#[derive(Default, Debug)]
pub struct Features {
    pub debug_utils: bool,
    pub mesh: bool,
    pub raytracing: bool,
}
impl Features {
    pub fn extensions(&self) -> Vec<&'static str> {
        let mut extensions = vec![
            ash::khr::swapchain::NAME,
            ash::khr::get_memory_requirements2::NAME,
            ash::khr::shader_float_controls::NAME,
            ash::khr::synchronization2::NAME,
            ash::ext::descriptor_indexing::NAME,
            ash::ext::extended_dynamic_state3::NAME,
            ash::ext::scalar_block_layout::NAME,
        ];
        if self.debug_utils {
            extensions.push(ash::ext::debug_utils::NAME);
            extensions.push(ash::khr::shader_non_semantic_info::NAME)
        }
        if self.mesh {
            extensions.push(ash::ext::mesh_shader::NAME);
        }
        if self.raytracing {
            extensions.push(ash::khr::ray_tracing_pipeline::NAME);
            extensions.push(ash::khr::acceleration_structure::NAME);
        }
        extensions
    }

    fn features(&self) -> vk::PhysicalDeviceFeatures {

        let mut acceleration_struct_feature: vk::PhysicalDeviceAccelerationStructureFeaturesKHR;
        let mut ray_tracing_feature: vk::PhysicalDeviceRayTracingPipelineFeaturesKHR;
        let mut mesh_shading: vk::PhysicalDeviceMeshShaderFeaturesEXT;

        let mut vulkan_12_features = vk::PhysicalDeviceVulkan12Features::default()
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
            .shader_int8(true);
        let mut vulkan_13_features = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .maintenance4(true)
            .synchronization2(true);
        let phfeatures = vk::PhysicalDeviceFeatures::default()
            .shader_int64(true)
            .fragment_stores_and_atomics(true)
            .shader_int16(true)
            .vertex_pipeline_stores_and_atomics(true);

        let mut dynamic_state = vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT::default()
            .extended_dynamic_state3_depth_clamp_enable(true)
            .extended_dynamic_state3_polygon_mode(true)
            .extended_dynamic_state3_logic_op_enable(true)
            .extended_dynamic_state3_color_blend_equation(true)
            .extended_dynamic_state3_color_write_mask(true)
            .extended_dynamic_state3_color_blend_enable(true);
        let mut dynamic_state2 = vk::PhysicalDeviceExtendedDynamicState2FeaturesEXT::default()
            .extended_dynamic_state2_logic_op(true);

        let mut features = vk::PhysicalDeviceFeatures2::default()
            .features(phfeatures)
            .push_next(&mut vulkan_12_features)
            .push_next(&mut vulkan_13_features)
            .push_next(&mut dynamic_state)
            .push_next(&mut dynamic_state2);

        if self.mesh {
            mesh_shading = vk::PhysicalDeviceMeshShaderFeaturesEXT::default()
                .task_shader(true)
                .mesh_shader(true);
            features = features.push_next(&mut mesh_shading);
        }
        if self.raytracing {
            ray_tracing_feature = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default()
                    .ray_tracing_pipeline(true);
            acceleration_struct_feature = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
                    .acceleration_structure(true)
                    .descriptor_binding_acceleration_structure_update_after_bind(true);
            features = features.push_next(&mut ray_tracing_feature)
                    .push_next(&mut acceleration_struct_feature);
        }
        features
    }
}


#[derive(Debug, Default)]
pub struct Functions {
    instance: ash::Instance,
    entry: ash::Entry,
    surface: ash::khr::surface::Instance,
    swapchain: ash::khr::swapchain::Device,
    debug_utils: Option<ash::ext::debug_utils::Device>,
    mesh: Option<ash::ext::mesh_shader::Device>,
    raytracing_pipeline: Option<ash::khr::ray_tracing_pipeline::Device>,
    acceleration_structure: Option<ash::khr::acceleration_structure::Device>,
}

static FUNCTIONS: OnceLock<Functions> = OnceLock::new();

impl Functions {
    pub fn surface() -> &'static ash::khr::surface::Instance { &get().surface }
    pub fn instance() -> &'static ash::Instance { &get().instance }
    pub fn entry() -> &'static ash::Entry { &get().entry }
    pub fn swapchain() -> &'static ash::khr::swapchain::Device { get().swapchain.as_ref() }
    pub fn debug_utils() -> Option<&'static ash::ext::debug_utils::Instance> { get().debug_utils.as_ref() }
    pub fn mesh() -> Option<&'static ash::ext::mesh_shader::Device> { get().mesh.as_ref() }
    pub fn raytracing_pipeline() -> Option<&'static ash::khr::ray_tracing_pipeline::Device> { get().raytracing_pipeline.as_ref() }
    pub fn acceleration_structure() -> Option<&'static ash::khr::acceleration_structure::Device> { get().acceleration_structure.as_ref() }
    
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

    pub fn cmd_start_label(&self, cmd: &vk::CommandBuffer, name: &str) {
        if let Some(debug_utils) = Self::debug_utils() {
            let name = format!("{}\0", name);
            let name = CStr::from_bytes_with_nul(name.as_bytes()).unwrap();
            let name_info = vk::DebugUtilsLabelEXT::default().label_name(name);
            unsafe {
                debug_utils
                    .cmd_begin_debug_utils_label(*cmd, &name_info)
            };
        }
    }
    pub fn cmd_insert_label(&self, cmd: &vk::CommandBuffer, name: &str) {
        if let Some(debug_utils) = Self::debug_utils() {
            let name = format!("{}\0", name);
            let name = CStr::from_bytes_with_nul(name.as_bytes()).unwrap();
            let name_info = vk::DebugUtilsLabelEXT::default().label_name(name);
            unsafe {
                debug_utils
                    .cmd_insert_debug_utils_label(*cmd, &name_info)
            };
        }
    }
    pub fn cmd_end_label(&self, cmd: &vk::CommandBuffer) {
        unsafe { self.debug_utils.cmd_end_debug_utils_label(*cmd) };
    }
}
fn get() -> &'static Functions {
    unsafe { FUNCTIONS.get().unwrap() }
}


pub(super) fn create_device(
    mut queue_families: Vec<u32>,
    physical_device: &PhysicalDevice,
    features: &Features,
) -> Result<ash::Device> {
    let queue_priorities = [1.0f32];
    let queue_create_infos = {
        queue_families.dedup();
        queue_families
            .iter()
            .map(|index| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(*index)
                    .queue_priorities(&queue_priorities)
            })
            .collect::<Vec<_>>()
    };
    
    let required_extensions = features.extensions();
    let device_extensions_as_ptr = required_extensions
        .into_iter()
        .map(|e| e.as_ptr() as *const i8)
        .collect::<Vec<_>>();

    let mut features = features.features();
    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_create_infos)
        .enabled_extension_names(device_extensions_as_ptr.as_slice())
        .push_next(&mut features);

    let device = unsafe {
        Functions::instance()
            .create_device(physical_device.handel, &device_create_info, None)
            .unwrap()
    };

    Ok(device)
}

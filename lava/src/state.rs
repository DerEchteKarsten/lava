use std::{
    cell::{Cell, LazyCell, OnceCell, UnsafeCell},
    collections::HashMap,
    ffi::{c_char, c_void},
    fmt::{Debug, write},
    mem::MaybeUninit,
    sync::{
        Mutex, MutexGuard, OnceLock, RwLock, RwLockReadGuard,
        atomic::{AtomicU16, AtomicU32, AtomicU64, Ordering},
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
    FRAMES_IN_FLIGHT,
    bindless::{Bindless, BindlessHandle},
    buffer::{Buffer, GpuBuffer, Location},
    command_buffer::{CommandBuffer, ResourceHandle, ResourceState},
    image::{format, slice::ImageView, usage::ColorAttachmentStorage},
    vkobjects::{
        physical_device::{PhysicalDevice, QueueFamily},
        queue::{Binary, CommandBufferMemory, CommandPool, Queue, Semaphore, Timeline},
        surface::Surface,
        swapchain::Swapchain,
    },
};

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

unsafe impl Sync for Ctx {}
struct TrustMeBro(UnsafeCell<Option<Ctx>>);
unsafe impl Sync for TrustMeBro {}


pub struct Ctx {
    features: Features,
    device: Device,
    physical_device: PhysicalDevice,

    frame_counter: Cell<u64>,

    surface: Surface,
    swapchain: UnsafeCell<Swapchain>,
    pub(crate) swpachain_needs_resizing: Cell<Option<(u32, u32)>>,

    pub(crate) timeline: Semaphore<Timeline>,
    pub(crate) command_buffers: [CommandBufferMemory; FRAMES_IN_FLIGHT],
    pub(crate) pools: [CommandPool; FRAMES_IN_FLIGHT],
    pub(crate) image_available: [Semaphore<Binary>; FRAMES_IN_FLIGHT],
    pub(crate) render_finished: [Semaphore<Binary>; FRAMES_IN_FLIGHT],

    pub(crate) gfx_queue_familie: u32,
    pub(crate) queues: Mutex<Vec<vk::Queue>>,

    allocator: Mutex<Allocator>,
    #[cfg(debug_assertions)]
    printf: Mutex<HashMap<String, usize>>,
    delay_deletion: Mutex<Vec<(Buffer<u8>, u64)>>,
}

thread_local! {
    static QUEUE: Queue = Queue::new().unwrap();
}

impl Debug for Ctx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write(
            f,
            format_args!(
                "device: there, physical_device: {:?}, allocator: {:?}",
                self.physical_device, self.allocator
            ),
        )
    }
}
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
pub(crate) static STATE: TrustMeBro = TrustMeBro(UnsafeCell::new(None));

impl Ctx {
    pub(crate) fn get() -> &'static Self {
        unsafe { STATE
                    .0
                    .get()
                    .as_ref()
                    .unwrap()
                    .as_ref()
                    .ok_or(anyhow!("Vulkan Context was not Initilized"))
                    .unwrap() }
    }
    pub(crate) fn get_mut() -> &'static mut Self {
        unsafe { STATE
                    .0
                    .get()
                    .as_mut()
                    .unwrap()
                    .as_mut()
                    .ok_or(anyhow!("Vulkan Context was not Initilized"))
                    .unwrap() }
    }
    pub(crate) fn num_swapchain_images() -> u32 {
        unsafe { Ctx::get().swapchain.get().as_ref().unwrap() }.images.len() as u32
    }
    pub(crate) fn is_init() -> bool {
        unsafe { STATE.0.get().as_ref().unwrap().is_some() }
    }
    pub(crate) fn resize_swapchain(width: u32, height: u32) {
        Ctx::get()
            .swpachain_needs_resizing
            .set(Some((width, height)));
    }
    pub(crate) fn device() -> &'static Device {
        &Ctx::get().device
    }
    pub(crate) fn physical_device() -> &'static PhysicalDevice {
        &Ctx::get().physical_device
    }
    pub fn queue<R, F: FnOnce(&Queue) -> R>(func: F) -> R {
        QUEUE.with(func)
    }
    pub(crate) fn allocator<'a>() -> MutexGuard<'a, Allocator> {
        Ctx::get().allocator.lock().unwrap()
    }
    pub(crate) fn delay_deletion<T: Copy + Pod, L: Location + 'static>(buff: Buffer<T, L>) {
        Ctx::get()
            .delay_deletion
            .lock()
            .unwrap()
            .push((buff.cast_owned(), Ctx::current_frame()));
    }
    pub(crate) fn surface() -> &'static Surface {
        &Ctx::get()
            .surface
    }
    pub fn window_width() -> u32 {
        unsafe { (*Ctx::get().swapchain.get()).size[0] }
    }
    pub fn window_height() -> u32 {
        unsafe { (*Ctx::get().swapchain.get()).size[1] }
    }

    pub(crate) fn features() -> Features {
        Ctx::get().features.clone()
    }

    pub fn current_frame() -> u64 {
        Ctx::get().frame_counter.get()
    }

    pub fn frame_in_flight() -> usize {
        Ctx::get().frame_counter.get() as usize % FRAMES_IN_FLIGHT
    }

    pub(crate) fn swapchain() -> vk::SwapchainKHR {
        unsafe { Ctx::get().swapchain.get().as_ref().unwrap() }.handle
    }
    pub(crate) fn swapchain_format() -> vk::Format {
        unsafe { Ctx::get().swapchain.get().as_ref().unwrap() }.format
    }

    pub fn start_frame() {
        tracy_span!("Wait for Semaphore");
        let next_frame = Ctx::current_frame() + 1;
        let next_frame_in_flight = next_frame as usize % FRAMES_IN_FLIGHT;

        Ctx::get().timeline.block_until_value(next_frame);

        {
            let mut lock = Ctx::get().delay_deletion.lock().unwrap();
            lock.retain(|(_, last_used)| (last_used + FRAMES_IN_FLIGHT as u64) >= next_frame);
        }
        Ctx::get().pools[next_frame_in_flight].reset();

        Ctx::get().frame_counter.set(next_frame);
    }

    pub fn record_frame<
        F: FnMut(
            &mut CommandBuffer,
            ImageView<format::Swapchain, ColorAttachmentStorage>,
        ) -> Result<()>,
    >(
        func: &mut F,
    ) -> Result<()> {
        tracy_span!("Acquire next Image");
        
        let image_index = Swapchain::aquire_image(&Ctx::get().image_available[Ctx::frame_in_flight()]);

        let img =
            unsafe { Ctx::get().swapchain.get().as_ref().unwrap() }.images[image_index as usize];

        Ctx::queue(|queue| {
            queue.execute_command(
                &Ctx::get().command_buffers[Ctx::frame_in_flight()],
                None,
                &[
                    Ctx::get().timeline.info(Ctx::current_frame() - 1),
                    Ctx::get().image_available[Ctx::frame_in_flight()].info()
                ],
                &[
                    Ctx::get().timeline.info(Ctx::current_frame()),
                    Ctx::get().render_finished[Ctx::frame_in_flight()].info(),
                ],
                |cmd| {
                    func(cmd, img);
                },
            ).unwrap();
            queue.present(image_index, &[&Ctx::get().render_finished[Ctx::frame_in_flight()]]).unwrap();
        });

        #[cfg(debug_assertions)]
        {
            let mut lock = Ctx::get().printf.lock().unwrap();
            let mut messages = lock.iter().collect::<Vec<_>>();
            if messages.len() > 0 {
                log::info!("Printf output this frame:");
                messages.sort_by(|(_, a), (_, b)| b.cmp(a));
                for (message, value) in messages {
                    log::info!("    {}x: {}", *value, *message);
                }
            }
            lock.clear();
        }

        if let Some(size) = Ctx::get().swpachain_needs_resizing.take() {
            unsafe { Ctx::get().swapchain.get().as_mut() }.unwrap().recreate([size.0, size.1]);
        }
        Ok(())
    }

    pub(super) fn init(
        display: &RawDisplayHandle,
        window: &RawWindowHandle,
        enable_validation: bool,
        enable_gpu_assited_validation: bool,
    ) -> Result<()> {
        unsafe { *(STATE.0.get().as_mut().unwrap()) = Some(std::mem::MaybeUninit::uninit().assume_init()) };
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
        let (physical_device, graphics_queue_family, present_queue_family, transfer_queue_family) =
            PhysicalDevice::select_suitable_physical_device(
                physical_devices.as_slice(),
                &mut features,
            )?;

        if let Some(pre) = present_queue_family {
            assert!(pre == graphics_queue_family);
        }
        if let Some(tra) = transfer_queue_family {
            assert!(tra == graphics_queue_family);
        }

        let device = create_device(graphics_queue_family.index, &physical_device, &features, &instance)?;
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
                log_memory_information: false,
                ..Default::default()
            },
            buffer_device_address: true,
            allocation_sizes: AllocationSizes::default(),
        })?;

        let surface = Some(Surface::new(
            surface,
            &physical_device,
            &surface_fn.as_ref().unwrap(),
        ));

        Ctx::get_mut().device = device;

        FUNCTIONS
            .set(Functions {
                mesh: if features.mesh {
                    Some(ash::ext::mesh_shader::Device::new(
                        &instance,
                        &Self::device(),
                    ))
                } else {
                    None
                },
                raytracing_pipeline: if features.raytracing {
                    Some(ash::khr::ray_tracing_pipeline::Device::new(
                        &instance,
                        &Self::device(),
                    ))
                } else {
                    None
                },
                acceleration_structure: if features.raytracing {
                    Some(ash::khr::acceleration_structure::Device::new(
                        &instance,
                        &Self::device(),
                    ))
                } else {
                    None
                },
                swapchain: ash::khr::swapchain::Device::new(&instance, &Self::device()),
                instance,
                entry,
                surface: surface_fn,
                debug_utils: instance_debug_utils,
                device_debug_utils: debug_utils,
            })
            .unwrap();
        Ctx::get_mut().allocator = Mutex::new(allocator);
        Ctx::get_mut().queues = Mutex::new((0..graphics_queue_family.num_queues).map(|i| {
            unsafe { Ctx::device().get_device_queue(graphics_queue_family.index, i as u32) }
        }).collect());
        Ctx::get_mut().pools = Ctx::queue(|queue| {
            (0..FRAMES_IN_FLIGHT).map(|_| queue.create_pool()).collect::<Vec<_>>().try_into().unwrap()
        });
        Ctx::get_mut().command_buffers = (0..FRAMES_IN_FLIGHT).map(|i| Ctx::get().pools[i].create_command_buffer()).collect::<Vec<_>>().try_into().unwrap();
        Ctx::get_mut().features = features;
        Ctx::get_mut().delay_deletion = Mutex::new(Vec::new());
        Ctx::get_mut().frame_counter = Cell::new(0);
        Ctx::get_mut().gfx_queue_familie = graphics_queue_family.index;
        Ctx::get_mut().image_available = Default::default();
        Ctx::get_mut().physical_device = physical_device;
        Ctx::get_mut().printf = Mutex::new(HashMap::new());
        Ctx::get_mut().render_finished = Default::default();
        Ctx::get_mut().surface = surface.unwrap();
        Ctx::get_mut().swapchain = UnsafeCell::new(Swapchain::new(graphics_queue_family.index, graphics_queue_family.index, None, None).unwrap());
        Ctx::get_mut().swpachain_needs_resizing = Cell::new(None);
        Ctx::get_mut().timeline = Default::default();
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
                if Ctx::is_init() && split.len() > 1
                {
                    let printf_message = split[1..]
                        .iter()
                        .map(|s| s.chars())
                        .flatten()
                        .collect::<String>();
                    if printf_message.len() != 0 {
                        *(Ctx::get().printf.lock().unwrap().entry(printf_message).or_insert(0)) += 1;
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
    queue_familie: u32,
    physical_device: &PhysicalDevice,
    features: &Features,
    instance: &ash::Instance,
) -> Result<ash::Device> {
    let queue_priorities = [1.0f32];
    let queue_create_infos = [
        vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_familie)
            .queue_priorities(&queue_priorities)
    ];

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
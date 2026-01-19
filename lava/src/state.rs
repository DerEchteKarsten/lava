use std::{
    collections::HashMap,
    ffi::{c_char, c_void},
    fmt::{Debug, write},
    mem::MaybeUninit,
    sync::{Mutex, MutexGuard, OnceLock, atomic::AtomicU64}, time::Instant,
};

use anyhow::{Result, anyhow};
use ash::{
    Device, Entry,
    ext::debug_utils,
    vk::{self, Handle},
};
use gpu_allocator::{
    AllocationSizes, AllocatorDebugSettings,
    vulkan::{Allocator, AllocatorCreateDesc},
};
use std::ffi::CStr;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::{
    FRAMES_IN_FLIGHT,
    bindless::{Bindless, BindlessHandle},
    command_buffer::{CommandBuffer, ResourceHandle, ResourceState},
    vkobjects::{
        image::Image, physical_device::PhysicalDevice, queue::Queue, surface::Surface,
        swapchain::Swapchain,
    },
};

#[derive(Debug)]
pub struct Frame {
    fence: vk::Fence,               // fence-per-frame for CPU recycling
    image_available: vk::Semaphore, // binary, signaled by acquire
    render_finished: vk::Semaphore, // binary, waited by present
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
}

pub struct Ctx {
    features: Features,
    device: Device,
    physical_device: PhysicalDevice,
    present: Present,
    pub(crate) resource_cache: Mutex<HashMap<ResourceHandle, ResourceState>>,
    queue: Queue,
    transfer_queue: Option<Queue>,
    present_queue: Option<Queue>,
    allocator: Mutex<Allocator>,
    #[cfg(debug_assertions)]
    printf: Mutex<HashMap<String, usize>>,
}

impl Debug for Ctx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write(
            f,
            format_args!(
                "device: there, physical_device: {:?}, present: {:?}, queue: {:?}, transfer_queue: {:?}, present_queue: {:?}, allocator: {:?}",
                self.physical_device,
                self.present,
                self.queue,
                self.transfer_queue,
                self.present_queue,
                self.allocator
            ),
        )
    }
}

#[derive(Debug)]
pub struct Present {
    frame_counter: AtomicU64,
    surface: Surface,
    swapchain: Mutex<Swapchain>,
    swpachain_needs_resizing: Mutex<Option<(u32, u32)>>,
    timeline: vk::Semaphore,
    frames: [Frame; FRAMES_IN_FLIGHT],
}

impl Ctx {
    pub fn resize_swapchain(width: u32, height: u32) {
        *(STATE
            .get()
            .ok_or(anyhow!("Vulkan Context was not Initilized"))
            .unwrap()
            .present
            .swpachain_needs_resizing
            .lock()
            .unwrap()) = Some((width, height));
    }
    pub fn device() -> &'static Device {
        &STATE
            .get()
            .ok_or(anyhow!("Vulkan Context was not Initilized"))
            .unwrap()
            .device
    }
    pub fn physical_device() -> &'static PhysicalDevice {
        &STATE
            .get()
            .ok_or(anyhow!("Vulkan Context was not Initilized"))
            .unwrap()
            .physical_device
    }
    pub fn queue() -> &'static Queue {
        &STATE
            .get()
            .ok_or(anyhow!("Vulkan Context was not Initilized"))
            .unwrap()
            .queue
    }
    pub fn transfer_queue() -> &'static Queue {
        let state = STATE
            .get()
            .ok_or(anyhow!("Vulkan Context was not Initilized"))
            .unwrap();
        state.transfer_queue.as_ref().unwrap_or(&state.queue)
    }
    pub fn present_queue() -> &'static Queue {
        let state = STATE
            .get()
            .ok_or(anyhow!("Vulkan Context was not Initilized"))
            .unwrap();
        state.present_queue.as_ref().unwrap_or(&state.queue)
    }
    pub fn allocator<'a>() -> MutexGuard<'a, Allocator> {
        STATE
            .get()
            .ok_or(anyhow!("Vulkan Context was not Initilized"))
            .unwrap()
            .allocator
            .lock()
            .unwrap()
    }
    pub fn surface() -> &'static Surface{
        &STATE
            .get()
            .ok_or(anyhow!("Vulkan Context was not Initilized"))
            .unwrap()
            .present
            .surface
    }
    pub fn swapchain<'a>() -> MutexGuard<'a, Swapchain> {
        STATE
            .get()
            .ok_or(anyhow!("Vulkan Context was not Initilized"))
            .unwrap()
            .present
            .swapchain.lock().unwrap()
    }
    pub fn window_width() -> u32 {
        STATE
            .get()
            .ok_or(anyhow!("Vulkan Context was not Initilized"))
            .unwrap()
            .present
            .swapchain.lock().unwrap().size[0]
    }
    pub fn window_height() -> u32 {
        STATE
            .get()
            .ok_or(anyhow!("Vulkan Context was not Initilized"))
            .unwrap()
            .present
            .swapchain.lock().unwrap().size[1]
    }

    pub fn features() -> Features {
        STATE
            .get()
            .ok_or(anyhow!("Vulkan Context was not Initilized"))
            .unwrap()
            .features
            .clone()
    }

    pub fn current_frame() -> u64 {
        STATE
            .get()
            .ok_or(anyhow!("Vulkan Context was not Initilized"))
            .unwrap()
            .present
            .frame_counter.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn next_frame<'a, F: FnMut(&mut CommandBuffer, Image) -> Result<()>>(
        func: &mut F,
    ) -> Result<()> {
        let ctx = STATE
            .get()
            .unwrap();

        let s = &ctx.present;
        let frame = s.frame_counter.load(std::sync::atomic::Ordering::Relaxed);
        let frame_in_flight = (frame + 1) % FRAMES_IN_FLIGHT as u64;
        let f = &s.frames[frame_in_flight as usize];
        unsafe {
            Ctx::device().wait_for_fences(&[f.fence], true, u64::MAX)?;
            Ctx::device().reset_fences(&[f.fence])?;
            Ctx::device().reset_command_pool(f.pool, vk::CommandPoolResetFlags::empty())?;
        }

        let (image_index, _suboptimal) = unsafe {
            Functions::swapchain().acquire_next_image(
                s.swapchain.lock().unwrap().handle,
                u64::MAX,
                f.image_available,
                vk::Fence::null(),
            )
        }?;
{}
        let mut resource_cache = ctx.resource_cache.lock().unwrap();

        let mut cmd = CommandBuffer {
            handle: f.cmd,
            resource_hashes: &mut resource_cache,
        };
        
        let img = Ctx::swapchain().images[image_index as usize].clone();
        cmd.begin();
        let result = func(&mut cmd, img);
        cmd.end();

        s.frame_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let frame = s.frame_counter.load(std::sync::atomic::Ordering::Relaxed);
        let waits = [vk::SemaphoreSubmitInfo {
            semaphore: f.image_available,
            stage_mask: vk::PipelineStageFlags2::TOP_OF_PIPE,
            ..Default::default()
        }];

        let cb_info = vk::CommandBufferSubmitInfo::default().command_buffer(f.cmd);
        let sig_render_finished = vk::SemaphoreSubmitInfo {
            semaphore: f.render_finished,
            stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
            ..Default::default()
        };

        let sig_graphics_timeline = vk::SemaphoreSubmitInfo {
            semaphore: s.timeline,
            value: frame as u64,
            stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
            ..Default::default()
        };

        let signals = [sig_render_finished, sig_graphics_timeline];

        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&waits)
            .command_buffer_infos(std::slice::from_ref(&cb_info))
            .signal_semaphore_infos(&signals);

        unsafe {
            Ctx::device().queue_submit2(
                Ctx::queue().handle,
                std::slice::from_ref(&submit),
                f.fence,
            )?;
        }

        {
            let lock = STATE.get().unwrap().printf.lock().unwrap();
            let mut messages = lock.iter().collect::<Vec<_>>();
            if messages.len() > 0 {
                log::info!("Printf output this frame:");
                messages.sort_by(|(_, a), (_, b)| b.cmp(a));
                for (message, value) in messages {
                    log::info!("    {}x: {}", *value, *message);
                }
            }
        }

        STATE.get().unwrap().printf.lock().unwrap().drain();

        let mut needs_recreation = s.swpachain_needs_resizing.lock().unwrap().is_some();
        let swapchains = [s.swapchain.lock().unwrap().handle];
        let indices = [image_index];
        let wait_sems = [f.render_finished];
        let present = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_sems)
            .swapchains(&swapchains)
            .image_indices(&indices);
        Ctx::swapchain().resized = false;
        match unsafe {
            Functions::swapchain()
                .queue_present(Ctx::present_queue().handle, &present)
        } {
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                needs_recreation = true;
            }
            Ok(_) => {}
            Err(e) => return Err(anyhow!("present failed: {e:?}")),
        }

        if needs_recreation {
            log::info!("resized swapchain");
            let size = if let Some(size) = *s.swpachain_needs_resizing.lock().unwrap() {
                [size.0, size.1]
            } else {
                Ctx::swapchain().size
            };
            let mut swapchain = Swapchain::new(
                Ctx::surface(),
                Ctx::device(),
                Ctx::queue().family_index,
                Ctx::present_queue().family_index,
                Functions::swapchain(),
                Functions::debug_utils(),
                Some(s.swapchain.lock().unwrap().handle),
                Some(size),
            )
            .unwrap();
            swapchain.resized = true;

            unsafe {
                Ctx::device().device_wait_idle().unwrap();
                Functions::swapchain()
                    .destroy_swapchain(Ctx::swapchain().handle, None);
            };

            for (i, image) in swapchain.images.iter_mut().enumerate() {
                let handle = BindlessHandle {
                    descriptor_index: i as u32,
                    descriptor_set: 1
                };
                Bindless::write_image(image, handle);
                image.bindless_handle = Some(handle);
            }

            s.swapchain.set(swapchain).unwrap();
            s.swpachain_needs_resizing.set(None).unwrap();
        }

        result
    }

    pub(super) fn init<T: HasWindowHandle + HasDisplayHandle>(
        window: &T,
        enable_validation: bool,
        enable_gpu_assited_validation: bool,
    ) -> Result<()> {
        if STATE.get().is_some() {
            return Ok(());
        }
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

        let mut instance_extensions = 
            ash_window::enumerate_required_extensions(window.display_handle().unwrap().into())
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

            validation_features = validation_features.
                enabled_validation_features(&validation_f);

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
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
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
                unsafe {
                    ash_window::create_surface(
                        &entry,
                        &instance,
                        window.display_handle().unwrap().into(),
                        window.window_handle().unwrap().into(),
                        None,
                    )
                }
                .unwrap();

        let surface_fn = Some(ash::khr::surface::Instance::new(&entry, &instance));

        let physical_devices = PhysicalDevice::enumerate_physical_devices(
            &surface,
            &instance,
            surface_fn.as_ref(),
        )?;
        let (physical_device, graphics_queue_family, present_queue_family, transfer_queue_family) =
            PhysicalDevice::select_suitable_physical_device(
                physical_devices.as_slice(),
                &mut features,
            )?;

        let mut queue_families = vec![graphics_queue_family.index];
        if let Some(present_queue_family) = &present_queue_family {
            queue_families.push(present_queue_family.index);
        }
        if let Some(transfer_queue_family) = &transfer_queue_family {
            queue_families.push(transfer_queue_family.index);
        }

        let device = create_device(queue_families, &physical_device, &features, &instance)?;
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
            allocation_sizes: AllocationSizes::new(64, 64),
        })?;

        let mut frames: [Frame; FRAMES_IN_FLIGHT] = unsafe { MaybeUninit::zeroed().assume_init() };
        for i in 0..FRAMES_IN_FLIGHT {
            let create_info = vk::SemaphoreCreateInfo::default();
            let image_availible_semaphore =
                unsafe { device.create_semaphore(&create_info, None).unwrap() };
            let render_finished_semaphore =
                unsafe { device.create_semaphore(&create_info, None).unwrap() };
            let fence = unsafe {
                device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )?
            };
            let pool = unsafe {
                device.create_command_pool(
                    &vk::CommandPoolCreateInfo {
                        queue_family_index: graphics_queue_family.index,
                        ..Default::default()
                    },
                    None,
                )?
            };
            let allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .command_buffer_count(1)
                .level(vk::CommandBufferLevel::PRIMARY);
            let cmd = unsafe { device.allocate_command_buffers(&allocate_info) }.unwrap()[0];
            frames[i] = Frame {
                image_available: image_availible_semaphore,
                render_finished: render_finished_semaphore,
                fence,
                pool,
                cmd,
            }
        }

        let surface = 
            Some(Surface::new(
                surface,
                &physical_device,
                &surface_fn.as_ref().unwrap(),
            ));

        let mut info = vk::SemaphoreTypeCreateInfo {
            semaphore_type: vk::SemaphoreType::TIMELINE,
            initial_value: 0,
            ..Default::default()
        };
        let swapchain_fn = ash::khr::swapchain::Device::new(&instance, &device);

        let present = 
            Present {
                swpachain_needs_resizing: Mutex::new(None),
                swapchain: Mutex::new(Swapchain::new(
                    &surface.as_ref().unwrap(),
                    &device,
                    graphics_queue_family.index,
                    present_queue_family
                        .as_ref()
                        .map(|e| e.index)
                        .unwrap_or(graphics_queue_family.index),
                    &swapchain_fn,
                    debug_utils.as_ref(),
                    None,
                    None,
                )?),
                frame_counter: AtomicU64::new(0),
                timeline: unsafe {
                    device.create_semaphore(
                        &vk::SemaphoreCreateInfo::default().push_next(&mut info),
                        None,
                    )?
                },
                surface: surface.unwrap(),
                frames,
            };

        let ctx = Self {
            allocator: Mutex::new(allocator),
            queue: Queue::new(&device, graphics_queue_family.index)?,
            present_queue: if let Some(present) = present_queue_family {
                Some(Queue::new(&device, present.index).unwrap())
            } else {
                None
            },
            transfer_queue: if let Some(transfer) = transfer_queue_family {
                Some(Queue::new(&device, transfer.index).unwrap())
            } else {
                None
            },
            physical_device: physical_device,
            device,
            present,
            features: features.clone(),
            #[cfg(debug_assertions)]
            printf: Mutex::new(HashMap::new()),
            resource_cache: Mutex::new(HashMap::new()),
        };
        STATE.set(ctx).unwrap();
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
                instance,
                entry,
                surface: surface_fn,
                swapchain: swapchain_fn,
                debug_utils: instance_debug_utils,
                device_debug_utils: debug_utils,
            })
            .unwrap();
        Ok(())
    }
}

pub(crate)  static STATE: OnceLock<Ctx> = OnceLock::new();

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
            let split = message.split("DebugPrintf:\n").collect::<Vec<_>>();
            if let Some(s) = STATE.get() && split.len() > 1
            {
                let printf_message = split[1..]
                    .iter()
                    .map(|s| s.chars())
                    .flatten()
                    .collect::<String>();
                if printf_message.len() != 0 {
                    *(s.printf.lock().unwrap().entry(printf_message).or_insert(0)) += 1;
                }
                return vk::FALSE;
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
    mut queue_families: Vec<u32>,
    physical_device: &PhysicalDevice,
    features: &Features,
    instance: &ash::Instance,
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

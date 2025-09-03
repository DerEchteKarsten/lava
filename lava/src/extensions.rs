use std::{mem::MaybeUninit, sync::Arc};

pub struct Features {
    pub debug_utils: bool,
    pub mesh: bool,
    pub raytracing: bool,
}

#[derive(Debug)]
struct Functions {
    pub instance: ash::Instance,
    pub entry: ash::Entry,
    pub surface: ash::khr::surface::Instance,
    pub swapchain: Option<ash::khr::swapchain::Device>,
    pub debug_utils: Option<ash::ext::debug_utils::Device>,
    pub mesh: Option<ash::ext::mesh_shader::Device>,
    pub raytracing_pipeline: Option<ash::khr::ray_tracing_pipeline::Device>,
    pub acceleration_structure: Option<ash::khr::acceleration_structure::Device>,
}

static FUNCTIONS: Arc<MaybeUninit<Functions>> = Arc::new(MaybeUninit::uninit());
impl Functions {
    pub fn instance() -> &ash::khr::surface::Instance { &get().instance }
    pub fn entry() -> &ash::Entry { &get().entry }
    pub fn swapchain() -> Option<&ash::khr::swapchain::Device> { get().swapchain.as_ref() }
    pub fn debug_utils() -> Option<&ash::ext::debug_utils::Instance> { get().debug_utils.as_ref() }
    pub fn mesh() -> Option<&ash::ext::mesh_shader::Device> { get().mesh.as_ref() }
    pub fn raytracing_pipeline() -> Option<&ash::khr::ray_tracing_pipeline::Device> { get().raytracing_pipeline.as_ref() }
    pub fn acceleration_structure() -> Option<&ash::khr::acceleration_structure::Device> { get().acceleration_structure.as_ref() }
}
fn get() -> &Functions {
    unsafe { FUNCTIONS.assume_init() }
}


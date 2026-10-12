//Tow diffrent implementations for PipelineManager depending on the feature flag

use std::{collections::HashMap, sync::Arc, sync::RwLock};

use ash::vk::{self, ShaderModule};
use smallvec::SmallVec;

use crate::{
    bindings::{
        NUM_COMPUTE_PIPELINES, NUM_RASTER_PIPELINES, NUM_RAY_TRACING_PIPELINES,
        NUM_SPECIALIZED_COMPUTE_PIPELINES, NUM_SPECIALIZED_RAY_TRACING_PIPELINES,
    },
    pipeline::{Constants, RasterHash},
    state::Ctx,
    vkobjects::rt_pipeline::{RaytracingPipeline, ShaderBindingTable},
};

//Static Pipeline Manager
struct PipelineManager {
    shader_modules: Box<[ShaderModule]>, //initilized once at init, PassEntry include!() 's the spirv instead of storing the path to the module (only if hot reload is disabled)
    pub compute_pipelines: RwLock<[vk::Pipeline; NUM_COMPUTE_PIPELINES]>,
    specialized_compute_pipelines:
        RwLock<[HashMap<Constants, vk::Pipeline>; NUM_SPECIALIZED_COMPUTE_PIPELINES]>,
    pub raytracing_pipelines: RwLock<[Option<RaytracingPipeline>; NUM_RAY_TRACING_PIPELINES]>,
    specialized_raytracing_pipelines:
        RwLock<[HashMap<Constants, RaytracingPipeline>; NUM_SPECIALIZED_RAY_TRACING_PIPELINES]>,
    pub raster_pipelines: RwLock<[HashMap<RasterHash, vk::Pipeline>; NUM_RASTER_PIPELINES]>,
}

//----------------------- ONLY FOR WITH THE HOTRELOAD FEATURE ENABELD
//Hot reload PipelineManager

//is now returned by pipelines.compute()/raster()/raytracing etc.
pub(crate) struct GenericPipeline {
    pipeline: vk::Pipeline,
    _sbt: Option<ShaderBindingTable>,
}

impl Drop for GenericPipeline {
    fn drop(&mut self) {
        unsafe { Ctx::device().destroy_pipeline(self.pipeline, None) };
    }
}

struct PipelineManager2 {
    shader_modules: RwLock<Vec<ShaderModule>>,
    pub compute_pipelines: RwLock<[Arc<GenericPipeline>; NUM_COMPUTE_PIPELINES]>,
    specialized_compute_pipelines:
        RwLock<[HashMap<Constants, Arc<GenericPipeline>>; NUM_SPECIALIZED_COMPUTE_PIPELINES]>,
    pub raytracing_pipelines: RwLock<[Option<Arc<GenericPipeline>>; NUM_RAY_TRACING_PIPELINES]>,
    specialized_raytracing_pipelines:
        RwLock<[HashMap<Constants, Arc<GenericPipeline>>; NUM_SPECIALIZED_RAY_TRACING_PIPELINES]>,
    pub raster_pipelines: RwLock<[HashMap<RasterHash, GenericPipeline>; NUM_RASTER_PIPELINES]>,
}

struct CommandBuffer {
    //...
    pipelines: Vec<Arc<GenericPipeline>>, // gets pushed every time pipelines().compute()/raster/etc. is called
}

//then frameslot.begin clears the this vec after waiting the fence and all the pipelines get automaticly dropped.

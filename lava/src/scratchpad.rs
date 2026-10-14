//! API sketches for the "find solutions" entries of STATE.md that touch lava (pseudo-Rust in a
//! block comment, so it needn't compile). The reasoning is in the chat summary.
/*
// ------------------------------------------------------------------------------------------
// 1. Pipelines are created synchronously on first use, under a write lock, without a cache.
// ------------------------------------------------------------------------------------------
// (a) A VkPipelineCache, saved next to the binary: cheap, no API change, and the driver's own
//     cache already makes repeat starts fast, so this only helps a shader change at start.
pub struct PipelineManager {
    cache: vk::PipelineCache, // created from `$OUT_DIR/pipeline.cache` if it exists
    // ...
}
impl Drop for PipelineManager {
    fn drop(&mut self) {
        // `get_pipeline_cache_data` -> written back to the file.
    }
}

// (b) Warm-up before the first frame: every pass's pipelines with the constants and raster
//     states the engine uses, built in parallel on worker threads. The `RasterHash` is what
//     the command buffer would compute; core lists them once at startup.
pub fn warm_up(requests: &[PipelineRequest]) {
    requests.par_iter().for_each(|r| match r {
        PipelineRequest::Compute { pass, constants } => { pipelines().compute(pass.index, constants); }
        PipelineRequest::Raster { pass, hash } => { pipelines().raster(hash, pass.index); }
    });
}
pub enum PipelineRequest {
    Compute { pass: &'static PassEntry, constants: Constants },
    Raster { pass: &'static PassEntry, hash: RasterHash },
}

// (c) Only the write lock is a problem for *other* passes while one compiles: `cached` holds
//     it during `create()`. A per-pass lock (an array of `Mutex<HashMap<..>>` instead of one
//     `RwLock` over the array) lets passes compile in parallel and readers of other passes
//     through; that is the smallest change and removes the stall for everything but the
//     pass being created.
specialized_compute_pipelines: [Mutex<HashMap<Constants, vk::Pipeline>>; NUM_SPECIALIZED_COMPUTE_PIPELINES],
raster_pipelines: [Mutex<HashMap<RasterHash, vk::Pipeline>>; NUM_RASTER_PIPELINES],
*/

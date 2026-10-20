# State: known bugs, intended limitations and ideas (lava)

The engine's are in `../GameEngine/STATE.md`. Same conventions: `--- fix ---` entries get fixed right away; `--- intended ---` entries stay so they aren't forgotten; `--- find solutions ---` entries get proposals without code changes; `--- future work ---` entries are agreed directions waiting for their time. `## Ideas` holds design proposals we don't want to lose.

## lava: timelines, lifetimes and presenting (2026-10-17)
----find solutions----
- Recording one image on two timelines without a `Wait` between them corrupts its tracked state undetected (both move it); so does accessing a released image without waiting for the release's submission. A debug check could remember the last timeline and value per image.
- Host image copies (`Image::copy_from`) leave the image's owner alone; check that against `VK_EXT_host_image_copy`'s rules for images another family owns.
- Replaced swapchains are destroyed by their surface at its next `acquire`, so a surface that stops acquiring keeps its old swapchains (and their bindless slots) until it is dropped.
- Recording 64 compute dispatches costs about 7% more than before timelines (`lava-tests` recording bench, 2026-10-17, `taskset -c 2`, alternating builds: 22 vs 20.5 µs; draws, declared accesses and mesh launches at par). Not found without a profiler; `compute_barriers` and the barrier log were ruled out.
----intended----
- A `reserve()` without a following submission holds back every deletion until the timeline's next submission: reserve right before reading.
- `submit` signals the timeline's stamp, which may cover a `reserve()` made for a later frame (pipelined extract before the previous frame submitted); core's extract runs after the previous render submitted.
- Swapchain images are owned by graphics: compositing into them on an async compute timeline isn't supported (the present would need an ownership transfer back first).
- Dropping a `PresentSurface` with a current image waits on the host for an empty submission that consumes the semaphore it left signalled.
- A recording that uses an acquired image and fails before `cmd.present` of it (a panic, or the check that it presents what it uses) is replaced by its barriers without that image's, so its tracked layout is wrong until the next `acquire` resets it to `UNDEFINED` (also when it keeps the image current).
- The acquire and the present of an image are in one submission: drawing into it on one timeline and presenting from another isn't supported.
- Without present fences, a replaced swapchain waits for a later value of the timeline that presented it.
- `VkSwapchainPresentModesCreateInfoEXT` names only the swapchain's own mode, so a present-mode change recreates the swapchain.

## lava: optional features (cargo features `X` / `assert-X`, 2026-10-17)
----intended----
- `PassKind` and the `kind::*` markers stay compiled whatever the features: they describe the pass map. A pass map whose SPIR-V needs a compiled-out feature fails at `init`, and its generated bindings don't compile.
- `line-stipple` is one feature for every line mode the device has (default lines need `strictLines` and stippled rectangular lines), so a device with stippled Bresenham but not stippled smooth lines has no stippling at all.
- The ETC2 and EAC format markers have no feature: lava never enables `textureCompressionETC2`, so images of them fail in the driver.
- With `assert-present-fences` but no display, `Ctx::features().present_fences` is false while the code assumes fences; nothing presents headless, so nothing reads it.
- The engine-facing `Gated` setters of `RasterState` check the capability whatever the value (`polygon_mode(FILL)` needs `fill-mode-non-solid` too).
- Baked draw state can make many pipelines per pass on devices without the dynamic state (the 16-pipeline warning then names the pass); `dynamic-*` without `assert-` keeps that possible.

## lava: descriptor heaps (`VK_EXT_descriptor_heap`, required since 2026-10-19)
----find solutions----
- Whether RenderDoc can capture an application on descriptor heaps is unchecked (none is installed here); Nsight Graphics 2026.2 can. Whether either reads shaders that index the heaps directly (`SPV_EXT_descriptor_heap`, untyped pointers, since 2026-10-19) is unchecked too. lava can't fall back to descriptor sets.
- `gpu_window` hung twice on 2026-10-19 in `vkAcquireNextImageKHR` (timeout `u64::MAX`) at its resize step (`a_new_size_recreates_the_swapchain_and_frees_the_old_slots`), once on the commit before the heaps and once after, and a third time at the same step with direct heap access (not traced), then passed: probably while the compositor didn't show the test window. A finite acquire timeout would make it fail instead of hang.
----intended----
- A submission that copies queued `bind` rewrites waits once at its start for every earlier command of its queue, so batch rewrites. Another timeline sees a rewrite only after a `Wait` on that submission, and a submission of another timeline must not read a slot while it is rewritten (the same rule as shared images).
- The heaps are `CpuToGpu` buffers, on a discrete GPU without ReBAR inside the 256 MiB BAR window (a few MiB: 4 MiB of image descriptors and 128 KiB of samplers on Intel). If a driver warns about host-visible heaps, make them device-local and copy every write the way `bind` copies rewrites.
- Samplers aren't deduplicated: equal descriptions take a slot each (32 bytes on Intel).

## lava: ray tracing (tested by `gpu_ray_tracing`, not used by core yet)
----intended----
- Binding table records hold only shader group handles: no shader record data, since shaders read everything through push constants and addresses.
- A TLAS names its BLASes by address, which lava doesn't track: a BLAS dropped while a TLAS still names it is destroyed once the submissions in flight are done, so rebuild the TLAS before tracing it again. `Instances::new` is `unsafe` with that contract (`Instance::from_address` stays safe: `Instance`'s fields are public and passes write instances).
- `compacted(size)` takes the compacted size on the caller's word (decided 2026-10-18): `compact` checks only that the destination was made by `compacted` of its source, which was built.
- No indirect builds (`vkCmdBuildAccelerationStructuresIndirectKHR`): instance and primitive counts come from the host, so a GPU-culled TLAS builds every instance slot (cull with the mask or a zero transform).
- Any acceleration structure write waits for every pending trace and TLAS build (the read of `EVERY_ACCELERATION_STRUCTURE`), also of unrelated structures.
- Trace reads carry `MEMORY_READ` for Mesa's ANV (26.2.2): with `ACCELERATION_STRUCTURE_READ` alone, a trace after a TLAS rebuilt in the same submission (traced before it) saw stale nodes. Worth reporting to Mesa and dropping once fixed.
- Ray tracing's triangle facing against lava's clockwise front faces is unverified (the tests cull nothing).

## lava: passes from lava-build-slang (lava's Slang frontend)
----find solutions----
- A pass that loads 8-bit integers through a buffer pointer needs `storage-8bit` (VUID-RuntimeSpirv-storageBuffer8BitAccess-06328; 16-bit ones probably `storage-16bit`, unchecked), but Slang declares only `Int8` for it, no `StorageBuffer8BitAccess`, so lava-build-slang gates it on `shader-int8` alone and the application has to assert the storage feature itself. The validation layer reports it only since lava enables `shaderUntypedPointers` (2026-10-19; the engine's meshlet triangle indices, now `assert-storage-8bit`). Reading the pass's loads (an `OpLoad` of an 8-bit type from `PhysicalStorageBuffer`) would let it gate them.
----intended----
- `lava::init` takes one `PassMap`, so an application and a library on lava can't both bring passes; `BindingOutput` names its pass by index into that map.
- The generated-binding tests need real buffers and images (`BufferSlice` and `ImageView` have no public constructor), so they run in `gpu_headless` instead of as unit tests.

## lava: buffer memory and uploads (memory locations 2026-10-19, explicit uploads 2026-10-20)
----intended----
- Host writes racing GPU reads aren't checked: `host_mut` on an `Upload` or `Shared` range a frame in flight still reads is a logic race, not undefined behaviour, and per-frame data takes one range per frame slot by convention (the instance, gizmo and UI rings). SURVEY 4.6 (`PerFrame<Buffer<T, Upload>>`) would make it a type.
- The host-access check (`host`, `host_mut`, `try_host`) sees the declared buffer writes of submitted recordings: not writes a pass makes through an address it doesn't declare in `Accesses` (core's shader clocks), and not writes recorded but not submitted yet (the `&mut` borrow keeps new recordings off the buffer, so only another thread's recording begun before the borrow could still reach it). `read_into` and `get` copy without checking. It reads each timeline's snapshot of its submissions' writes (`Index::writes`), never their pending accesses, so it never waits for a recording.
- Uploads are explicit (2026-10-20): a timeline's `Uploader`s publish writes into its staging ring, which only a `cmd.drain_uploads` of that timeline copies, at that point of the recording. Another timeline sees the data only after a `Wait` on that submission, and a drain on a family other than an uploader's destination releases its images, so the recording can't use them after it. An image another family owns has to be uploaded through that family's timeline.
- No staging ring on `rebar` devices (integrated GPUs included): every `Gpu` buffer comes from `Shared`'s memory types, `init` and `write` write it in place, so `Uploader::write` into a buffer a frame in flight reads races it (double-buffer, as with `host_mut`), and images without `HostCopy` can't be uploaded there without `host-image-copy` (`Uploader::image` errors). Elsewhere `write` always stages, since a frame in flight may read the target.
- `init` writes in place wherever the buffer's memory is mapped, also without `rebar` (integrated GPUs map device-local memory), so `gpu_staging` tests the ring through `write`, which stages wherever there is a ring.
- The ring's tail never passes a writer still counted in (reserved, not published): a slow fill (a large mesh) holds the tail for its duration, and concurrent writers that find the ring full wait for it (`StagingFull` after the uploader's timeout).
- `cmd.upload`, the `vkCmdUpdateMemoryKHR` shortcut inside a recording, is gone: an ordered write is `uploader.write` followed by `cmd.drain_uploads`. `cmd.update_buffer` keeps the small-update path.

----future work----
- `Shared` needs `rebar`, which needs a device-local host-visible heap of more than 500 MiB; on a discrete GPU without ReBAR the 256 MiB BAR window would still serve small buffers. `Location::shared_or_upload` falls back to `Upload` there.

## lava: bugs from the publishing survey (2026-10-14)
Found in a read-only review; details and directions are in `lava/SURVEY.md` section 2. The rest were fixed on 2026-10-16.
----intended----
- Dropping a `PresentSurface` waits at most 1 s for its present fences, then stops waiting for the ones that didn't signal (a present that failed before it was queued) and retires them with the swapchain.
- Completed pending buffer writes are only folded (into one write of every address, which every later access is made visible to once) once there are more than 64; until then they stay, since a later submission still needs a barrier to see them.

## lava: bugs from the second survey (2026-10-18)
Found in a read-only review; details, line numbers and the lower-severity rest (bugs 17-98) are in `lava/SURVEY.md` section 2.
----find solutions----
- Layout transitions and queue family acquires aren't tracked as writes, so a later read at another stage isn't ordered after the transition (SURVEY 17).
- Safe code reaches undefined behaviour: `Bindings::new` checks neither the pass kind nor the specialization constants' length (a `VkSpecializationInfo` with `dataSize` 0); `lava::destroy()` can run twice or before further use; `QueryPool::read` overflows; tracing an unbuilt TLAS; building a BLAS from another geometry than its creation's; host copies of ASTC/ETC2 images skip the data length check; empty indirect slices (SURVEY 23-37).
- `multiDrawIndirect`, `drawIndirectFirstInstance` and `depthBiasClamp` are never enabled, though `Draw::indirect` passes `buffer.len()` draws and `DepthBias::clamp` is public (38, 39).
- Without extended dynamic state 3, line and point draws use a pipeline baked with `TRIANGLE_LIST` (`dynamicPrimitiveTopologyUnrestricted` is never queried) (40). `depth_clamp(true)` still clips where `VK_EXT_depth_clip_enable` exists (41). 64-bit clear values are packed as 32-bit (43).
- `wait_for` and queries use stages and query types the waiting family doesn't support (44, 45).
- A `Timeline` dropped after `reserve()` or an unsubmitted `begin()` holds back every later destruction for good; a failed barrier-only replay leaves a value nothing signals (64, 65). The public `reclaim()` never polls (67). `init` isn't atomic: a late failure leaks the device and every retry says "already initialised" (69).
- A failed pipeline creation poisons the pipeline cache of its kind for every later recording (72).
- `cmd.present` marks the image presented while recording, so `surface.present()` inside the submit closure presents on an unsignalled semaphore; a failed swapchain recreation keeps the retired swapchain (77, 78).
- The hot reload's `layout_hash` ignores field types: `Tex2D` to `Buf<uint>` reloads and the shader reads a handle as an address (85). Codegen merges same-named structs with different field types, and pointees and array elements of handles become generated structs instead of `ImageHandle` (87, 88).

## Performance
--- find solutions ---
- Pipelines are created synchronously on first use, under a write lock, with no `VkPipelineCache` (`command_buffer.rs:233-244`, `:405`, `:602-607`, `:768`). (Ideas: per-pass locks, a warm-up list, a `VkPipelineCache` file.)
- Recording a draw costs about twice a dispatch on the CPU (`lava-tests` recording bench, 2026-10-16, borrowed unsized draws: 64 indexed draws in 8 renderings about 85 µs, 64 compute dispatches 40 to 55 µs, noisy). Not profiled yet. Suspects: the per-draw state diff and push constants, and each rendering's span, scope and attachment setup.

--- intended ---
- Hot reload recompiles every pass serially on any include change (`lava-build-slang/src/hot_reload.rs`).
(hot reload is a debug feature anyways)

## Ideas

### Explicit bindless rewrites (2026-10-20)
`bindless::bind` still has an implicit queue (`Bindless::queued`), recorded by the next graphics or compute submission before its closure, like uploads were until 2026-10-20. Move it to an explicit batch the same way: `bind` writes into a `Rewrites` batch the application owns, which `cmd.record_rewrites(&batch)` records where it chooses (`flush_queued`'s barriers and `vkCmdUpdateMemoryKHR` runs), dropping a `SlotRange` forgets its entries, and a failed submission returns them, as `Staging::requeue` does for drained uploads. It would also let a transfer timeline rewrite slots, since it needs no bound heap to write heap memory.

### Transient attachments (2026-10-19)
Image memory is always `GpuOnly`; images have no memory type parameter. Their only other useful memory is lazily allocated memory for attachments that never leave a rendering (depth of a single pass, MSAA targets once lava has them), which tilers never back with memory. It belongs in the usage set: a `Transient` flag of `Usage` (`TRANSIENT_ATTACHMENT`, allowed only with attachment flags and without sampled, storage or transfer ones), whose images allocate from `LAZILY_ALLOCATED` memory where the device has it.

### Multi-pass hardware counter captures (2026-10-15)
`QueryPool::performance` refuses selections that need more than one counter pass, since a frame is submitted once. A tool capture could record one frame's command buffer once and submit it once per pass with `VkPerformanceQuerySubmitInfoKHR.counterPassIndex` (with the per-pass resets the extension asks for), reading each pass's counters from the same pool. It needs a submission path that replays a recorded command buffer, which `Submission` doesn't have.

### lava-build-glsl beside lava-build-slang (2026-10-15)
lava's side is already language-neutral: `PassMap`/`PassEntry` (only what lava reads), `Bindings::new`/`with_constants`, `pipeline::reload_pass` and the shader contract in lava's `lib.rs` docs. Inside lava-build-slang the seam is between reflection and emission: Slang-specific are `compile.rs` (sessions, linking, `layout_hash`) and the walk over `TypeLayout`s in `codegen.rs` (it recognises `Img`/`Buf`/`Accel`/`SCONST` by Slang generics and attributes); language-neutral are the Rust emission (structs with layout asserts, `push_bindings`/`constant_bindings`, the `PassEntry` and `PASS_MAP` literals, the hot reload's `Build`, `write_if_changed`), `ray_tracing_groups`, the pass kind from the stages, the `Shaders` builder and the hot-reload watcher. When a GLSL frontend is written, extract the neutral part into a shared `lava-build` crate whose input is a reflected pass model (entries; push-constant fields as plain/buffer/image/accel with offsets; specialization constants; a layout hash), with a hot reload that takes a `compile(pass_index) -> Result<Vec<u8>, String>`. lava-build-glsl then compiles with shaderc/glslang, reflects the SPIR-V (or GLSL annotations) into that model, and ships its own `bindless.glsl` honouring the contract (and its own clock helpers, if wanted: lava has no part in them).

# State: known bugs, intended limitations and ideas (lava)

The engine's are in `../GameEngine/STATE.md`. Same conventions: `--- fix ---` entries get fixed right away; `--- intended ---` entries stay so they aren't forgotten; `--- find solutions ---` entries get proposals without code changes; `--- future work ---` entries are agreed directions waiting for their time. `## Ideas` holds design proposals we don't want to lose.

## lava: timelines, lifetimes and presenting (2026-10-17)
----find solutions----
- Recording one image on two timelines without a `Wait` between them corrupts its tracked state undetected (both move it); so does accessing a released image without waiting for the release's submission. A debug check could remember the last timeline and value per image.
- Host image copies (`Image::copy_from`) leave the image's owner alone; check that against `VK_EXT_host_image_copy`'s rules for images another family owns.
- Replaced swapchains are destroyed by their surface at its next `acquire`, so a surface that stops acquiring keeps its old swapchains (and their bindless slots) until it is dropped.
- Recording 64 compute dispatches costs about 7% more than before timelines (`lava-tests` recording bench, 2026-10-17, `taskset -c 2`, alternating builds: 22 vs 20.5 µs; draws, declared accesses and mesh launches at par). Not found without a profiler; `compute_barriers` and the barrier log were ruled out.
----intended----
- Swapchain images are owned by graphics: compositing into them on an async compute timeline isn't supported (the present would need an ownership transfer back first).
- Dropping a `PresentSurface` with a current image waits on the host for an empty submission that consumes the semaphore it left signalled.
- A recording that uses an acquired image and fails before `cmd.present` of it (a panic, or the check that it presents what it uses) is replaced by its barriers without that image's, so its tracked layout is wrong until the next `acquire` resets it to `UNDEFINED` (also when it keeps the image current).
- The acquire and the present of an image are in one submission: drawing into it on one timeline and presenting from another isn't supported.
- Windows need `present-fences` (`VK_EXT_swapchain_maintenance1` and the surface maintenance instance extensions): `PresentSurface::new` returns `MissingFeature` without them, since only a present's fence tells when a replaced swapchain is done with (2026-10-20; the coverage of `VK_EXT_swapchain_maintenance1` wasn't checked).
- `FifoLatestReady` comes only from `VK_EXT_present_mode_fifo_latest_ready`, not its KHR promotion (a driver with the KHR one alone has no `FifoLatestReady`). It is untested here: the Arc B390 (Mesa 26.2) doesn't list it.
- `present_all` presents on graphics queue 0 like `present`; one surface's error (other than suboptimal or out of date) is returned after every surface was handled, and the others' images were presented.
- `VkSwapchainPresentModesCreateInfoEXT` names only the swapchain's own mode, so a present-mode change recreates the swapchain.

## lava: references and lifetimes (Arc-owned objects, 2026-10-25)

- Objects live in `Arc`s that submissions keep until they completed; what no recording tracks (replaced pipelines, freed sampler slots, unbound sparse pages, retired swapchains, children `host_write_referencing` replaced) waits for every recording in progress and submission in flight (`keep_until_idle`), so it can outlive its last real use by a frame or two.
- What completed submissions kept is released by `begin`, waits, `reclaim`, `wait_idle` and a timeline's drop: an application that neither begins nor waits keeps it (and its memory) meanwhile.
- A write's visibility across recordings is tracked per object only through the barrier logs of the recordings that touch it (`Write::applied`): a barrier of another recording that happened to cover it isn't seen, which costs a redundant barrier, never a missing one.
- `writes_below` never clears: a buffer that once held a write reference below it is walked on every command that names it, also after the write reference is gone.
- Dirtiness is per object: a write anywhere in a suballocated buffer marks every buffer that references any part of it, and a walk reaches each child with the union of the ranges its parent references.
- A write reference is a "may write": every command that names its buffer counts as writing the child (barriers, and draws of one rendering refused unless `[Unordered]`), whether or not the shader writes.
- A write or layout transition of a referenced object waits for every pending access of every buffer above it (`ancestor_hazards`), also for reads that never reached it through them.
- Push-constant `DynImg`s and `Sampler`s name their objects without keeping them; a sampler's slot waits for what is in flight at its drop, but bindings built before the drop and recorded after it read whatever reuses the slot.
- Recording bench, 2026-10-25 (`taskset -c 2`, two rounds alternating with the commit before Arc objects, `3814a19`): `compute` about 20% and `many_draws` about 17% faster (per-object access state instead of per-timeline lists), `declared` about 6% and `mesh` about 20% slower (15.8 vs 13.5 µs; it was 8% after referencing buffers, before keep lists: each tracked object costs an atomic swap and, once per recording, an `Arc` clone). Not profiled.

## lava: queries and hardware counters (2026-10-20)
----intended----
- A query pool keeps one used bit per query, cleared when a new recording first touches the pool: a query takes one timestamp, query or compacted size per recording, and a pool one recording at a time (one pool per frame slot).
- `Timeline::capture` records its closure once per counter pass and blocks after each pass; the passes are separate recordings of what should be the same frame, so counters of different passes come from different (if alike) work. It doesn't reset the profiled pool after the first pass (the validation layer tracks query state per pass); it presents nothing.
- One counter pool per recording: lava doesn't enable `performanceCounterMultipleQueryPools`. `MAX_COUNTERS` (8) is lava's own cap on a pool's counters, kept so a query's values fit an array; a capture lifts only the one-pass limit.
- The Arc B390's counters (Mesa 26.2) are all of command scope, so the `CommandBuffer`-scope refusal in `cmd.query` is tested only where a device has such counters; `submit_profiled` and a two-pass capture are tested here.

## lava: optional features (cargo features `X` / `assert-X`, 2026-10-17)
----intended----
- `PassKind` and the `kind::*` markers stay compiled whatever the features: they describe the pass map. A pass map whose SPIR-V needs a compiled-out feature fails at `init`, and its generated bindings don't compile.
- `line-stipple` is one feature for every line mode the device has (default lines need `strictLines` and stippled rectangular lines), so a device with stippled Bresenham but not stippled smooth lines has no stippling at all.
- With `assert-present-fences`, `assert-hdr-metadata` or `assert-fifo-latest-ready` but no display, `Ctx::features()` has the feature false while the code assumes it; nothing presents headless, so nothing reads it.
- The engine-facing `Gated` setters of `RasterState` check the capability whatever the value (`polygon_mode(FILL)` needs `fill-mode-non-solid` too).
- Baked draw state can make many pipelines per pass on devices without the dynamic state (the 16-pipeline warning then names the pass); `dynamic-*` without `assert-` keeps that possible.
----fix----
- `ray-traversal-primitive-culling` compiled without `ray-query` and `ray-tracing-pipeline` can never be on (`Features::consistent` drops it), so `assert-ray-traversal-primitive-culling` alone makes `init` refuse every device; its `Cargo.toml` entry implies only `acceleration-structure` (2026-10-24, found by `lava/src/tests.rs`, whose consistency tests skip that combination).
- Release builds of lava warn that `CallsiteGuard` and its `enter` are unused (`state.rs`, the `#[validation_trace]` callsite helper; a release build of lava-bench shows it).

## lava: renderings and draws (2026-10-21)
----find solutions----
- Dynamic rendering local read (a draw reading an attachment of its own rendering) isn't there: input attachments need an image usage flag, heap descriptors of their own type, and an `InputAttachmentIndex` Slang only puts on global parameters (details in SURVEY section 5).
----intended----
- An image's tracked states cover all its layers (one per mip for a `MipChain`), so a transition moves every layer of the mip, layer views share their image's states, and an attachment of one layer whose mip a draw also reads (another layer) is rendered in `GENERAL`. Hazards between draws are per image and mip, not per layer.
- Layer views (`ImageView::layer`, a `layers` range other than the view's) have no bindless handle: `bind_*` bind the whole array, and `view.bind::<K>()` binds a layer alone (at slots the image owns, kept in its `ViewCache`). Their `VkImageView`s (attachments only) are made the first time a rendering or `view()` asks and kept in the image (`ViewCache`) until it is destroyed.
- A rendering has one depth-stencil image: `stencil_attachment` replaces a `depth_attachment` and the other way round (Vulkan wants the same view for both), so a separate depth image and `S8Uint` image can't be attached together; a combined format attaches both through `depth_attachment`.
- `BufferAccess::unordered` is the application's word: lava doesn't check that unordered draws' writes commute.
- Multiview renderings of mesh passes panic on devices without `multiviewMeshShader` (the Arc B390, Mesa 26.2, has none), so they are untested here.
- Blending checks the attachment format's `COLOR_ATTACHMENT_BLEND` through `Ctx::format_features`, a cache of each format's features filled on first use (a mutex lock per check, once per rendering whose draws blend; blits, clears and image copies take one or two per command too).
- Recording bench, 2026-10-21 (`taskset -c 2`, two runs alternating with the commit before): `many_draws` about 2% slower (243 vs 236 µs), `mesh` about 10% (35.6 vs 32.2 µs for 32 launches in 8 renderings, some 100 ns a draw), `compute` and `declared` within noise. Suspects: `RasterState::check` per `Draw` constructor, the larger `DrawCommand`, and the per-draw viewport and scissor comparisons; not profiled.
- The topology class is baked only without `dynamicPrimitiveTopologyUnrestricted` on an enabled `VK_EXT_extended_dynamic_state3` (every `dynamic-*` feature off, as `gpu_missing_features` has it); the Arc B390 has the property.

## lava: images, formats and transfers (2026-10-21)
----intended----
- `MutableFormat` images are created without a `VkImageFormatListCreateInfo` (lava doesn't know their view formats up front), so drivers may give up compression for them (Intel's CCS); give the flag only to images that need other formats. A reinterpreting view has no handle of its own, and a `Dynamic` usage never creates a `MutableFormat` image (an erased one keeps its flags).
- `Uploader::image` refuses formats whose texel block doesn't divide the ring's 16-byte alignment (3, 6, 12, 24 and 32-byte texels): their copies need addresses that are multiples of the block. Upload those through `copy_buffer_to_image` from a buffer of your own, or with host copies.
- `Image::fit` binds the new image at new slots (its handles change; rewrite what references it) and drops its debug name and the views bound with `view.bind`.
- Sparse resources: binds go through graphics timelines only (`Timeline<Graphics>::bind_sparse`), wait only for their own timeline (another timeline's work using the resources waits for the returned `Wait`), and allocate one gpu-allocator allocation per page or tile (64 KiB on Intel). Sparse images are colour, one layer, `Single` views; sparse buffers always have residency (no binding-only buffers). Reads of what isn't resident are undefined unless the device has `residencyNonResidentStrict`; lava doesn't check that accesses stay on resident pages.
- External memory: Unix file descriptors only (no Win32 handles), dedicated allocations from the first device-local type of the resource's requirements (an importer in another process gets the same type only for the same description on the same driver), dma-bufs for buffers only (no DRM format modifiers for images). An imported image is a new `VkImage` that starts undefined: lava records no layout handoff or `QUEUE_FAMILY_EXTERNAL` ownership transfer, so contents shared between images aren't defined; buffers share contents.

## lava: descriptor heaps (`VK_EXT_descriptor_heap`, required since 2026-10-19)
----find solutions----
- Whether RenderDoc can capture an application on descriptor heaps is unchecked (none is installed here); Nsight Graphics 2026.2 can. Whether either reads shaders that index the heaps directly (`SPV_EXT_descriptor_heap`, untyped pointers, since 2026-10-19) is unchecked too. lava can't fall back to descriptor sets.
- `gpu_window` hung twice on 2026-10-19 in `vkAcquireNextImageKHR` (timeout `u64::MAX`) at its resize step (`a_new_size_recreates_the_swapchain_and_frees_the_old_slots`), once on the commit before the heaps and once after, and a third time at the same step with direct heap access (not traced), then passed: probably while the compositor didn't show the test window. A finite acquire timeout would make it fail instead of hang.
----intended----
- The heaps are `CpuToGpu` buffers, on a discrete GPU without ReBAR inside the 256 MiB BAR window (a few MiB: 4 MiB of image descriptors and 128 KiB of samplers on Intel). If a driver warns about host-visible heaps, make them device-local and copy every write the way `bind` copies rewrites.
- Samplers aren't deduplicated: equal descriptions take a slot each (32 bytes on Intel).

## lava: ray tracing (tested by `gpu_ray_tracing`, not used by core yet)
----intended----
- Binding table records hold only shader group handles: no shader record data, since shaders read everything through push constants and addresses.
- A TLAS built from raw instances (`unsafe Instances::from_raw`, what passes write) names BLASes lava doesn't know: the caller keeps them built and alive while it is traced, and its builds and traces wait for every structure write (`EVERY_ACCELERATION_STRUCTURE`). TLASes built from `Instance`s (written with their BLASes) keep their BLASes and wait for their writes precisely; a serialized one counts as raw once deserialized.
- A structure counts as built (`is_built`, what traces, compaction and copies check) once a recording that didn't fail recorded its build; its last build's shapes, which updates keep, apply once the recording reached the GPU. A trace recorded on one timeline while another timeline's recording that builds the TLAS is still being recorded passes the check and would read the old contents (or nothing) if that recording failed; the reading timeline must `wait_for` the building one anyway.
- A deserialized structure has no known shapes: it can be traced, copied and serialized, but not built, updated or compacted. A serialized TLAS names its BLASes by their old addresses (rebuild it instead).
- A `Compactor` places compacted structures in their own memory, not an arena, and waits for each compaction to complete before handing it back (one frame-in-flight cycle more than needed on one timeline).
- `compacted(size)` takes the compacted size on the caller's word (decided 2026-10-18): `compact` checks only that the destination was made by `compacted` of its source, which was built.
- No indirect builds (`vkCmdBuildAccelerationStructuresIndirectKHR`): instance and primitive counts come from the host, so a GPU-culled TLAS builds every instance slot (cull with the mask or a zero transform).
- Any acceleration structure write waits for every pending trace and build of a TLAS built from raw instances (their read of `EVERY_ACCELERATION_STRUCTURE`), also of unrelated structures.
- Trace reads carry `MEMORY_READ` on Mesa's ANV (26.2.2, `DriverId::INTEL_OPEN_SOURCE_MESA`, decided at init: `PhysicalDevice::stale_trace_reads`): with `ACCELERATION_STRUCTURE_READ` alone, a trace after a TLAS rebuilt in the same submission (traced before it) saw stale nodes. Worth reporting to Mesa and dropping once fixed.
- Shader execution reordering (`ray-tracing-invocation-reorder`, `VK_EXT_ray_tracing_invocation_reorder`) is untested on a device that has it: the Arc B390 (Mesa 26.2.5) has neither the EXT nor the NV extension, so `reordered_traces_hit_like_plain_ones` only checks that its bindings are refused. Slang emits `ShaderInvocationReorderEXT` for `HitObject`, so lava doesn't map the NV capability.
- Ray tracing's triangle facing against lava's clockwise front faces is unverified (the tests cull nothing).

## lava: passes from lava-build-slang (lava's Slang frontend)
----find solutions----
- A pass that loads 8-bit integers through a buffer pointer needs `storage-8bit` (VUID-RuntimeSpirv-storageBuffer8BitAccess-06328; 16-bit ones probably `storage-16bit`, unchecked), but Slang declares only `Int8` for it, no `StorageBuffer8BitAccess`, so lava-build-slang gates it on `shader-int8` alone and the application has to assert the storage feature itself. The validation layer reports it only since lava enables `shaderUntypedPointers` (2026-10-19; the engine's meshlet triangle indices, now `assert-storage-8bit`). Reading the pass's loads (an `OpLoad` of an 8-bit type from `PhysicalStorageBuffer`) would let it gate them.
----intended----
- `lava::init` takes one `PassMap`, so an application and a library on lava can't both bring passes; `BindingOutput` names its pass by index into that map.
- The generated-binding tests need real buffers and images (`BufferSlice` and `ImageView` have no public constructor), so they run in `gpu_headless` and `gpu_codegen` instead of as unit tests; lava-build-slang's own tests compile Slang snippets into pass descriptions and check the mapping without a GPU.
- Generated structs with references (handles, addresses, samplers) are `AnyBitPattern`, not `NoUninit`: read back as slices, written only through their reference infos. Push-constant structs with handles implement `NoUninit` by hand; their `DynImg`s and `Sampler`s name objects without keeping them.
- A scalar-layout struct whose size isn't a multiple of its alignment (`{ uint64_t a; uint b; }` is 12 bytes behind a pointer) is generated `repr(C, packed(4))`, so its fields can't be borrowed (`&s.a` doesn't compile; copy them). glam's SIMD-aligned `Vec4`, `Mat2` and `Mat4` appear only where offset and size are 16-byte aligned; elsewhere their fields are `[f32; 4]` and column arrays, as non-square matrices and std430's `float3x3` always are (`[[f32; 4]; 3]`; glam's `Mat3A` isn't `Pod`).
- A file of several compute entry points is split only when every entry point is compute (a file of compute and other stages is refused); its passes share the file's push constants, each linked with its entry point alone. There is no `[Pass("Name")]` override: the name is the stem and the entry point.
- `[RustConst]` exports come from the pass file's top-level declarations and its `#include`s, not from `import`ed modules (their declarations aren't the module's children), and only scalars; `float` values are printed with Rust's `{:?}` of the `f32` Slang reports.
- The build compiles each file once in a session shared by its thread, so a module `import`ed by several passes is loaded once per thread; the hot reload compiles in a fresh session per changed file, since a session keeps the modules it loaded.
- Every compute pipeline is created with `DISPATCH_BASE`, so any compute pass can be recorded with `compute_base`; no driver here showed a cost.

## lava: buffer memory and uploads (memory locations 2026-10-19, explicit uploads 2026-10-20)
----intended----
- Host writes racing GPU reads aren't checked: `host_mut` on an `Upload` or `Shared` range a frame in flight still reads is a logic race, not undefined behaviour. `queue::PerFrame<T>` keeps one value per frame slot, reached only through the `Submission` whose `begin` waited for the slot (`get_mut`, `host_mut`); data written into one range per slot of a single buffer stays a convention.
- Host access takes elements by what it does: writes (`host_write`, `Uploader::write`/`from_slice`, `cmd.update_buffer`) and `&mut` mappings (`host_mut`) `NoUninit` ones, reads (`host`, `Uploader::init`) `AnyBitPattern` ones, so referencing structs are read back but written only with their references; GPU-only uses take any `Copy + 'static` element (`Vec3A` positions).
- The host-access check (`host`, `host_mut`, `try_host`) sees the declared buffer writes of submitted recordings: not writes a pass makes through an address it doesn't declare in `Accesses` (core's shader clocks), and not writes recorded but not submitted yet (the `&mut` borrow keeps new recordings off the buffer, so only another thread's recording begun before the borrow could still reach it). `read_into` and `get` copy without checking. It reads the buffer's own tracked writes against each timeline's last submitted value (`Index::submitted`), so it never waits for a recording.
- Uploads are explicit (2026-10-20): a timeline's `Uploader`s publish writes into its staging ring, which only a `cmd.drain_uploads` of that timeline copies, at that point of the recording. Another timeline sees the data only after a `Wait` on that submission, and a drain on a family other than an uploader's destination releases its images, so the recording can't use them after it. An image another family owns has to be uploaded through that family's timeline.
- No staging ring on `rebar` devices (integrated GPUs included): every `Gpu` buffer comes from `Shared`'s memory types, `init` and `write` write it in place, so `Uploader::write` into a buffer a frame in flight reads races it (double-buffer, as with `host_mut`), and images without `HostCopy` can't be uploaded there without `host-image-copy` (`Uploader::image` errors). Elsewhere `write` always stages, since a frame in flight may read the target.
- `init` writes in place wherever the buffer's memory is mapped, also without `rebar` (integrated GPUs map device-local memory), so `gpu_staging` tests the ring through `write`, which stages wherever there is a ring.
- The ring's tail never passes a writer still counted in (reserved, not published): a slow fill (a large mesh) holds the tail for its duration, and concurrent writers that find the ring full wait for it (`StagingFull` after the uploader's timeout).
- `cmd.upload`, the `vkCmdUpdateMemoryKHR` shortcut inside a recording, is gone: an ordered write is `uploader.write` followed by `cmd.drain_uploads`. `cmd.update_buffer(slice, &data)` keeps the small-update path: up to 64 KiB of whole words, recorded in place.

----future work----
- `Shared` needs `rebar`, which needs a device-local host-visible heap of more than 500 MiB; on a discrete GPU without ReBAR the 256 MiB BAR window would still serve small buffers. `Location::shared_or_upload` falls back to `Upload` there.

## lava: bugs from the publishing survey (2026-10-14)
Found in a read-only review; details and directions are in `lava/SURVEY.md` section 2. The rest were fixed on 2026-10-16.
----intended----
- Dropping a `PresentSurface` waits at most 1 s for its present fences, then stops waiting for the ones that didn't signal (a present that failed before it was queued) and retires them with the swapchain.
- `lava::shutdown` refuses (`Error::ObjectsAlive`) while any lava object lives, also one in a static or leaked on purpose, and then destroys nothing; it never destroys objects the application still owns. Each shutdown leaks lava's `Ctx` (a few KB: the device description and its extension names) and the counter list, so getters racing it and `&'static Counter`s stay valid memory.
- Completed pending buffer writes are only folded (into one write of every address, which every later access is made visible to once) once there are more than 64; until then they stay, since a later submission still needs a barrier to see them.

## lava: bugs from the second survey (2026-10-18)
Found in a read-only review; details, line numbers and the lower-severity rest (bugs 17-98) are in `lava/SURVEY.md` section 2.
----find solutions----
- `compact` recognises the source a compacted structure was made for by its raw handle, so a structure created at a handle value reused after the source's destruction passes (31).

## lava: tests (2026-10-24)
----intended----
- What the tests can't reach on the Arc B390 (Mesa 26.2.5), so it passes there untested:
  - the transfer family's granularity panic (its `minImageTransferGranularity` is 1x1x1; `transfer_family_copies_follow_its_rules` accepts either outcome);
  - linear blits refused for lack of `SAMPLED_IMAGE_FILTER_LINEAR` (every float format that blits filters linearly; `linear_blits_need_linear_filtering` skips);
  - smooth and stippled lines (no `smooth-lines`, no `line-stipple`), shader execution reordering, `FifoLatestReady`, counters of command-buffer scope;
  - a `[MaxRecursion(n)]` above the device's `maxRayRecursionDepth` (31 here, and lava-build-slang refuses more);
  - stage masks of waits that hold back too little (ANV waits for semaphores per batch), wrong visibility kept by the fold of more than 64 buffer writes (Intel flushes its caches at every submission end), and missing forgets on destruction (they only add waits): those tests check data and validation only.
- The harness always asks for validation and fails every test of a binary when the layer is missing. The layer prints its settings (`VALIDATION-SETTINGS`, `CURRENT-VALIDATION-ENABLED`) to stdout at `vkCreateInstance`, before lava's messenger exists, past libtest's capture; chaining a `VkDebugUtilsMessengerCreateInfoEXT` into the instance info would route them through `tracing`.
- GPU-assisted validation (`gpu_assisted`) works with `VK_EXT_descriptor_heap` on this layer and reports out-of-bounds reads through buffer addresses; it warns that it is slow together with the core checks.

## lava: found while the engine followed the survey work (2026-10-24)
----find solutions----
- `[RustConst]`'s generated doc names the pass being compiled, not the file that declares the constant (constants of `datatypes.slang` say "of `bvh_cull.slang`"; `lava-build-slang/src/codegen.rs`, the export emission).
- *(unverified)* Headless sponza reported `memory_report().allocated` = 16.6 MiB against 96 MiB of heap usage; lava's own allocated-bytes counter (`memory.rs`, fed by the allocator guard) may miss some allocations (images, dedicated or sparse ones).

----future work----
- A struct shared by several shaders can't keep a 16-byte size without hand padding, since Slang's scalar layout doesn't round sizes up (the engine's `BvhNode`, `Material`, `UIVertex` keep their pads). A round-up option or an alignment attribute in lava-build-slang would let shaders drop them.
- `PerFrame::get` needs a `&Submission`, so systems that only know the frame slot (the engine's bevy systems) can't use it; an accessor by a proven-free slot would.
- `Buffer::reserve` always copies the old contents; scratch buffers want a grow without contents.
- Generated code names `::core::` paths, which break in doc tests of a crate itself named `core` (the engine's).

## lava-bench: lava against hand-written Vulkan (2026-10-25)
The figures above that name the `lava-tests` recording bench come from the bench lava-bench replaced (2026-10-25; older commits have it). The friction log is `lava-bench/API_NOTES.md`; what of it is worth keeping is here.
----find solutions----
- Results on the Arc B390 (Mesa 26.2.5, no `VK_KHR_unified_image_layouts`; two full runs pinned to core 2, 500 frames each): lava's recording costs 1.5 to 7 times the baseline's (`ml_train` +630%, `forward` +375%, `ml` +300 to 360%, `ray_tracing` +270%, `deferred` +200 to 290%); per command about 0.55 µs against 0.12 µs for an indexed draw (`forward`) and 2.5 µs against 0.35 to 0.6 µs for a dispatch (`ml`, `particles`). The small scenarios' CPU medians are bimodal between runs (`gpu_driven`'s raw recording 34 or 94 µs); their 10th percentiles are steady. GPU time is at par (within ±3.6%) in 9 of 11 scenarios, so lava's barriers and `GENERAL` layouts cost nothing measurable here. `streaming` is +6 to 7% (probably one barrier per blit where the baseline merges a mip level's eight into one: lava places barriers per command and has no way to batch those of independent commands; not traced), `ml_train` +2.5% (not traced), and `forward` is 3 to 7% *faster* with lava than with the optimal-layout baseline (not profiled; its depth images go through `SHADER_READ_ONLY_OPTIMAL` and `UNDEFINED` transitions there).
- Tracking made cheaper on 2026-10-25 (branch `tracking-perf`): recordings work on copies of the access states (`locals.rs`, no lock per access), `Write::apply` runs lazily (at `hazards` and `publish`, not per `record`) over a table of the log's distinct barrier scopes instead of the log, writes fold once at import, covered entries are swapped out instead of shifted, a rendering's accesses are flattened and deduplicated before one flush, and the per-command `tracing` spans are behind the `command-spans` feature (the engine's `profiling` enables it; the bench leaves them out). Overheads went from +375% to +150% (`forward`), +290% to +100% (`deferred`), +630% to +350% (`ml_train`), +330% to +300% (`ml`), +270% to +145% (`ray_tracing`); the small compute scenarios barely moved (`fluid` +60%, `particles` +100%). What remains, by profile: `ml_train`'s hazard scans over the many writes of its suballocated buffers (every dispatch's spans are distinct ranges of one buffer, and writes within a recording never fold), the per-dispatch fixed cost (bindings, span, pipeline lookup, push data), and in `forward` the application's own binding construction (10% of the frame) and Mesa's `vkCmdDrawIndexed`. The next structural steps would be sorted, merging range segments per buffer and the recorded plans below.
- Lava rediscovers every hazard per command per frame; a fixed command sequence (`ml`'s 145 dispatches, the same every frame) costs it 4 to 5 times the baseline's CPU time, which plans its barriers once at setup. See the recorded-plan idea below.
- Lava records in program order and can't reorder: a per-texture mip-generation loop (each blit depends on the one before) cost `streaming` +32% GPU time against the baseline's level-major batching, +6% with the same loop order. Worth a doc line or a batched `cmd.generate_mips(&[..])`.
- `Scratch::for_any` holds one build, so `update_all` of many structures with it records one batch per structure with a barrier between each (`ray_tracing`: +8% GPU on its 64 refits, at par with a scratch for all). A `Scratch::for_all(&structures)` and a doc line on `build_all`/`update_all` would fix the trap.
- A submission's `wait_for(w.at(stage))` covers only `stage`; an access of what the waited-for timeline wrote at another stage of the same submission (a compute pass after the raster stage it waited at) is unsynchronised, and lava, which tracks that write, says nothing. A debug check could panic there.
- `Uploader::image` errors on every `rebar` device (all integrated GPUs), since they have no staging ring: streaming textures needs a second path through `HostCopy`. lava-bench disables `rebar` to upload everywhere. A small image-only ring there, or a fallback to host copies for `HostCopy` images, would make one path work everywhere.
- `BufferAccess::unordered()` is documented for draws of one rendering only; whether compute dispatches with commuting atomics (`image_processing`'s histogram bands) skip their barriers isn't said.
- There is no way to see the barriers a recording made (counts, or the barrier log in debug builds), so a GPU-time gap can't be traced to them.
- Fresh buffers and images hold whatever the memory held; lava-bench's checksums caught two reads of never-written memory that agreed between the modes only while the memory happened to be alike. A debug `Config` option filling new memory with a pattern would show them at once.
- `PerFrame::get_mut(&submission)` isn't reachable inside the `submit` closure, which consumes the submission; a closure that rewrites a per-slot buffer keeps its own slot counter (`cmd.slot()` would do).
- lava-build-slang: a file of several compute entry points shares one push struct but gets an argument type (and `C{Pass}Bindings`) per pass, so one set of arguments can't serve several dispatches; push-constant structs keep private `_pad`s (structs behind pointers have `pub #[doc(hidden)]` ones and a `Default`); a pointee whose size isn't a multiple of its alignment is refused rather than padded to its stride; Slang's indirect command structs aren't mirrored to lava's (`DrawIndexedIndirectCommand` and kin), so their buffers need a `.cast()`.
- Views of differently typed images are different types, so a step whose source is one image or another (Hi-Z mip 0 from depth, later mips from the chain) needs a generic helper spelling out the generated bound; an `ImageView::erase()` would let one call site take either. `SamplerHandle` has `DEFAULT` but no `Default`.

## Performance
--- find solutions ---
- Specialized and raster pipelines are created synchronously on the recording thread at first use (outside the pass's map lock, through the `VkPipelineCache`). (Ideas: a warm-up list, `FAIL_ON_PIPELINE_COMPILE_REQUIRED` for async compiles.)
- `ray-traversal-primitive-culling` on a device with ray queries and ray tracing pipelines but without `rayTraversalPrimitiveCulling` stays on when `Config::disabled` turns ray queries off, so a ray tracing pass with culling flags would be invalid there (no known device lacks the bit).
- Recording a draw costs about twice a dispatch on the CPU (`lava-tests` recording bench, 2026-10-16, borrowed unsized draws: 64 indexed draws in 8 renderings about 85 µs, 64 compute dispatches 40 to 55 µs, noisy). Not profiled yet. Suspects: the per-draw state diff and push constants, and each rendering's span, scope and attachment setup.

--- intended ---
- Hot reload recompiles every pass file serially on any change of a file that is no pass's source (an include, or any other file in the watched directories but hidden ones and editor backups; `lava-build-slang/src/hot_reload.rs`).
(hot reload is a debug feature anyways)

## Ideas

### Recorded plans (2026-10-25, from lava-bench)
Workloads whose command sequence doesn't change between frames (an ML graph, a post-processing chain) pay lava's hazard tracking anew for every command every frame, while a hand-written recording plans its barriers once. A plan could record a sequence once, keep the barriers it computed and the accesses they came from (a hash of every command's ranges, images and layouts), and replay it with new push data, checking the hash instead of walking the accesses. The open questions are what invalidates a plan (a resized image, a rebound handle, a referencing buffer gone dirty) and how replays interleave with tracked commands on the same objects.

### Transient attachments (2026-10-19)
Image memory is always `GpuOnly`; images have no memory type parameter. Their only other useful memory is lazily allocated memory for attachments that never leave a rendering (depth of a single pass, MSAA targets once lava has them), which tilers never back with memory. It belongs in the usage set: a `Transient` flag of `Usage` (`TRANSIENT_ATTACHMENT`, allowed only with attachment flags and without sampled, storage or transfer ones), whose images allocate from `LAZILY_ALLOCATED` memory where the device has it.

### lava-build-glsl beside lava-build-slang (2026-10-15)
lava's side is already language-neutral: `PassMap`/`PassEntry` (only what lava reads), `Bindings::new`/`with_constants`, `pipeline::reload_pass` and the shader contract in lava's `lib.rs` docs. Inside lava-build-slang the seam is `describe::PassDesc` (since 2026-10-24): Slang-specific are `compile.rs` (sessions, linking, pass names) and `describe.rs`'s walk over `TypeLayout`s (it recognises `Img`/`Buf`/`Accel`/`SCONST`/`[RustConst]` by Slang generics and attributes); language-neutral are the `PassDesc` model itself (kind, entries with workgroup sizes, push-constant fields and constants as `Ty` trees, exports, `layout_hash`) and everything in `codegen.rs` (the Rust emission: argument and padded structs with layout asserts, `push_bindings`, the `PassEntry` and `PASS_MAP` literals, the hot reload's `Build`, `write_if_changed`), `ray_tracing_groups`, the `Shaders` builder and the hot-reload watcher. When a GLSL frontend is written, extract the neutral part into a shared `lava-build` crate whose input is `PassDesc`, with a hot reload that takes a `compile(pass_index) -> Result<Vec<u8>, String>`. lava-build-glsl then compiles with shaderc/glslang, reflects the SPIR-V (or GLSL annotations) into that model, and ships its own `bindless.glsl` honouring the contract (and its own clock helpers, if wanted: lava has no part in them).

# lava: publishing survey (2026-10-14)

A read-only review of the whole crate against the publishing goals: meaningful abstractions,
cheap-but-valuable performance trade-offs, broad real-world coverage, a forward-looking device
baseline, a clean API, free safeguards, and raw Vulkan only in extreme cases. Line numbers are
from 2026-10-14. Unmarked items were checked in the code; *(plausible)* ones follow from it but
weren't reproduced. Bugs are also listed in `STATE.md`.

---

## 1. Publishing blockers

1. ~~**Codegen is hard-wired to this workspace.**~~ **Done (2026-10-15):** the `lava-build-slang`
   crate, called from the application's `build.rs`
   (`Shaders::new().passes(..).include(..).define(..).optimization(..).compile()`), compiles
   the application's passes and writes the SPIR-V and the bindings into its `OUT_DIR`, which
   it includes with `lava::include_passes!()`.
2. ~~**The runtime is sized from the generated code.**~~ **Done (2026-10-15)**, differently from
   the direction sketched here: the generated `PASS_MAP` (a `PassMap`: the entries; the frontend keeps its hot reload's compile options itself) goes to `lava::init`, and lava sizes its pipeline arrays from it,
   so there is still one pass set per application (an app and a library on lava can't both
   bring passes yet). The caps are fixed in `pipeline.rs`,
   and the generated code uses public API (`Bindings::new`/`with_constants`,
   `BufferSlice::access`, `ImageView::access`, `Tlas::trace_accesses`), documented for other
   binding generators, instead of a `__private` module. lava-build-slang ships `bindless.slang`,
   `profile.slang` and `ray_tracing.slang` in its `slang/`, searched by every session, and
   lava's GPU tests moved to the `lava-tests` crate, an ordinary consumer, so `test-passes` is
   gone.
3. ~~**Git dependencies.**~~ **Done (2026-10-16):** every dependency is a crates.io release.
   lava uses ash 0.38, ash-window 0.13 and gpu-allocator 0.28 without patches and defines what
   ash 0.38's headers (1.3.281) lack itself (`lava::vk`: `VK_KHR_device_address_commands`,
   `VK_EXT_depth_clamp_control`). lava-build-slang uses slang-rs 0.1.0 and checks the
   reflection results 0.1.0 would turn from null into references (`src/reflect.rs`); the
   generated bindings are byte-identical to those from slang-rs's fixed main. *Tried and
   reverted (2026-10-16):* lava's own forks (`lava-ash`, `lava-gpu-allocator`, `lava-slang`,
   ...); the user chose upstream.
4. **Slang is needed by every user.** lava-build-slang links `slang-compiler` through slang-rs,
   whose build script runs bindgen (clang) on `slang.h` from `SLANG_DIR`/`VULKAN_SDK`. Build
   scripts and hot-reload binaries find the library through an absolute rpath (or
   `LD_LIBRARY_PATH`) the application sets up; lava-build-slang's README documents it. The
   user accepted this (2026-10-16). docs.rs can't build lava-build-slang this way.
5. ~~**The default `hot-reload` feature makes binaries non-relocatable.**~~ **Done
   (2026-10-15):** `PassEntry.spirv` is always `include_bytes!`'d and hot reload compiles from
   the absolute source path (lava-build-slang's `BuiltPass.source`), so a moved binary runs
   with the pipelines it was built with. lava has no hot-reload feature now (it is lava-build-slang's); since 2026-10-17 its cargo features are the optional device features (`X` checked, `assert-X` required, see `lava::features`).
6. ~~**Nightly.**~~ **Done (2026-10-16):** lava, its tests and the generated code build on stable
   (rust-version 1.96). Registrations became the commands' `Accesses` parameter (`ImageUse::sampled_read(view)` and the like), `RasterBuilder`
   takes its attachments up front (see 4.1), and the range intersection is written out.
7. **Packaging.** *Done (2026-10-16) but for the name:* MIT OR Apache-2.0, metadata, READMEs,
   `include` lists, no scratchpad. The name `lava` belongs to an abandoned 2020 crate; its owner
   was asked to hand it over. If they decline, the package becomes `lava-vk` with
   `[lib] name = "lava"`.
8. ~~**Engine conventions are library behaviour.**~~ **Done (2026-10-16):** the raster state is
   all dynamic and overridable (the old values are the documented defaults), the swapchain
   takes its usage as a type and its format from a preference list, samplers come from
   `Config::samplers`, and the core-only features are optional (`Features`, `Config::required`;
   passes are gated by their SPIR-V capabilities).

---

## 2. Correctness and soundness

### Bugs
1. ~~**Generated push-constant structs are `Pod` with padding.**~~ **Done (2026-10-16):** push-constant structs get an explicit zeroed `_pad` field per gap Slang leaves (tail included), `#[repr(C)]` without `align`, and derive `Pod`. `#[repr(C, align(16))]` plus
   `unsafe impl Pod` with gaps. `CBvhCullBindings` has fields up to byte 232 but a size of 240,
   and `CCompositeBindings` has interior padding. `bytes_of` (`command_buffer.rs:182`) then
   reads uninitialised bytes, which is UB. The offsets also only hold while glam uses SIMD
   alignment: glam's `scalar-math` anywhere in the dependency graph breaks the asserts.
   **Direction:** emit explicit `_pad` fields from the Slang offsets and derive `Pod`
   (`build.rs:291-301`).
2. ~~**Write visibility is tracked as two separate unions.**~~ **Done (2026-10-16):** `PendingWrite::visible` keeps each barrier's destination (stages, accesses) as a pair, merged only where that is exact; a regression test covers the sequence below. `PendingWrite::visible_to`
   (`queue.rs:611-617`) checks `visible_stages` and `visible_access` independently, and
   `made_visible` (`command_buffer.rs:1343-1383`) ORs both in. The failing sequence:
   1. Compute writes X.
   2. A fragment shader samples X.
   3. Compute storage-reads X.
   4. Compute samples X, and no barrier is made, although the write was never made visible to
      sampled reads in compute.

   **Direction:** keep (stages, accesses) pairs per barrier and check that one pair contains both.
3. ~~**Debug names and labels never turn on.**~~ **Done (2026-10-16):** debug utils load with the instance extension whenever validation is on (debug builds); labels in release builds are still missing (section 5). `device_debug_utils` looks for
   `VK_EXT_debug_utils` among *device* extensions (`physical_device.rs:164-169`), but it is an
   instance extension. So `set_debug_name`, the scope labels and the swapchain image names do
   nothing, and the test at `state.rs:769-773` asserts the wrong behaviour.
4. ~~**A compacted acceleration structure can be rebuilt into too little memory.**~~ **Done (2026-10-16):** `compacted()` structures are marked; `cmd.build` on one panics (updates still refit in place), and `compact` checks its destination came from `compacted`. `compacted()`
   copies the source's `max_primitives` and build scratch (`acceleration_structure.rs:255-265`),
   and `record_builds` only checks primitive counts. So `cmd.build` on a compacted structure
   writes a full build past its smaller buffer.
5. ~~**Reused `CompactedSizes` read stale sizes.**~~ **Done (2026-10-15):** `CompactedSizes`
   became `QueryPool<CompactedSize>`, and every query pool is reset on the host when a
   recording first touches it, after checking that the frames that used it before completed.
6. ~~**`wait_idle` races other threads' submissions**~~ **Done (2026-10-16):** `wait_idle` locks every queue for the wait and marks the epochs done before unlocking, so only epochs submitted before it complete; destructors run after the unlock. (`lib.rs:74-78`, `epoch.rs:70-72`).
   - `vkDeviceWaitIdle` needs every queue externally synchronised, and lava doesn't lock them.
   - `device_idle()` completes every epoch marked submitted, including one submitted after the
     wait returned. Its destructors then run while the GPU is still using the objects.
   - Core calls it from `Swapchain::drop` while the render thread runs.
7. ~~**One idle `FrameSlot` stops all destruction, process-wide.**~~ **Done (2026-10-16):** `reserve` and `completed` poll the fences of the oldest submitted epochs (`vkGetFenceStatus`), so a slot that isn't begun again holds nothing back; `reserve` also runs what that frees. A slot completes its epoch before resetting its fence, so nothing polls a reset fence. An epoch completes only at
   the slot's next `begin` or its drop (`queue.rs:489-523`), and `completed()` stops at the
   oldest open epoch. A slot used once and kept, such as a tool's slot or a closed window's
   loop, leaks every later drop.
   - Without any frame loop, nothing is freed at all: `reclaim` runs only on `Epoch` drop or
     `wait_idle`.
   - **Direction:** one timeline semaphore per queue, with epochs mapped to its values and
     polled in `reserve`/`reclaim`.
8. ~~**Dropping an `Epoch` inside `Frame::execute` deadlocks**~~ **Done (2026-10-16):** destructors never run on a thread that is recording (`epoch::recording`); they wait for the next `reserve`, epoch drop or `wait_idle`. *(plausible trigger)*. The chain:
   `Epoch::drop` → `reclaim` → a destructor → `forget_image`/`forget_buffer` → `for_each_pending`
   locks every queue's `pending` mutex (`queue.rs:272-290`), which the recording already holds.
9. ~~**A panic while recording poisons the queue for good.**~~ **Done (2026-10-16):** the recording closure runs under `catch_unwind`; its pending accesses are put back, its barriers (logged in `CommandBuffer::barrier_log`) are submitted instead so the tracked layouts stay true, then the panic resumes. Lava's mutexes ignore poisoning. `execute_command` holds `pending`
   while the user's closure runs (`queue.rs:634`), and lava panics during recording on purpose
   (dynamic usage checks, unsupported passes).
   - After one panic, every later `execute` on that queue panics.
   - Every destruction panics in `for_each_pending`.
   - A drop during unwinding aborts the process.
10. ~~**An unbalanced `begin_scope`/`end_scope` wedges the frame slot when timestamps are on.**~~
    **Done (2026-10-15):** scopes are closures (`cmd.scope(name, f)`), and frame slots no
    longer own or read queries: a user's `ScopeObserver` sees every scope and pass of a
    `cmd.measure` and records its own queries there.
11. ~~**`Queue<Transfer>` is unusable as built.**~~ **Done (2026-10-16):** *in part:* a transfer-only family binds no descriptors and pushes no constants, dispatches, renderings and image accesses on it panic, and buffers are `CONCURRENT` between the graphics and transfer families. *Open:* images stay exclusive to graphics (no ownership transfers), and `CommandBuffer` still isn't typed by family.
    - Every command buffer binds descriptor sets at the graphics and compute bind points, which
      is invalid on a transfer-only family.
    - Resources are `EXCLUSIVE` (`image/mod.rs:54`, `buffer/mod.rs:112`), and lava never records
      a queue-family ownership transfer. So an upload on the transfer family leaves its contents
      undefined on graphics.
    - `CommandBuffer` isn't typed by family, so dispatches compile for a transfer queue.
12. ~~**An attachment that is also read by a shader in the same rendering ends in the wrong
    layout.**~~ **Done (2026-10-16):** an attachment a draw of the same rendering reads is
    rendered in `GENERAL` (`RasterBuilder::attachment_accesses`). The draws ask for `GENERAL` and the attachment for `ATTACHMENT_OPTIMAL`, so two
    transitions of one subresource go into one barrier, and the descriptor says `GENERAL`
    (`command_buffer.rs:746-755`). Read-only depth that is also sampled (soft particles, decals)
    hits this.
13. ~~**Depth-stencil formats get views with both aspects**~~ **Done (2026-10-16):** a sampled or storage image of a combined depth-stencil format gets a depth-only view per view, which bindless binds; `subresource_layers` names the depth aspect, and host copies of such formats are refused. (`image/mod.rs:126`). Sampling such a
    view is invalid (VUID-VkDescriptorImageInfo-imageView-01976), and host copies and
    `subresource_layers` pass both aspects.
14. ~~**Pending accesses never retire once the GPU is done.**~~ **Done (2026-10-16):** pending buffer reads are tagged with their frame's epoch and dropped once it completed; past 64 pending buffer writes, the completed ones fold into one write of every address (`EVERY_BUFFER`), which later accesses are made visible to once. Image accesses stay (one of each kind per image). A read is dropped only when a later
    access covers its whole range (`command_buffer.rs:1414-1475`). A ring whose read ranges move
    each frame adds entries forever, which grows every command's O(accesses × pending) scan and
    adds needless barriers. **Direction:** tag entries with the submission's serial and fold
    completed ones into one summary.
15. ~~**Safe code can corrupt lava's state through public mutable fields.**~~ **Done (2026-10-16):** the fields are private, with getters (`Buffer::address/handle`, `ImageView::image/view/mip_range/extent/handle`, `Swapchain::size/format/present_modes/present_mode` and `set_present_mode`, `Surface::handle/formats/present_modes/capabilities`); `BufferSlice::address` too.
    - `Buffer { handle, address, allocation }`: swapping allocations makes `host()` read out of
      bounds.
    - `ImageView { image, view, mip_range, extent, handle }`: retargeting a copy or widening
      `mip_range` misdirects barriers.
    - `Swapchain.size`/`format`/`present_mode` and `Surface.handle`.

    Make the fields private with getters.
16. ~~**Host access depends on the device.**~~ **Done (2026-10-16):** *the first half:* host access works exactly on buffers created `cpu_writable`, on every device (core's `variables` is now `cpu_writable`). *Open:* `host()` still hands out `&[T]` over memory a frame in flight may write. With `rebar`, every buffer is mapped
    (`buffer/mod.rs:141-145`).
    - So core's `variables: Buffer::new(1, false)`, read through `Index` (`render.rs:605,638`),
      panics on a discrete GPU without ReBAR.
    - `host()` returns `&[T]` over memory the GPU may be writing, which is UB under Rust's
      aliasing rules (a polling loop can be hoisted).
17. ~~**Required device features are requested but never checked**~~ **Done (2026-10-16):** `features.rs` checks lava's needs and `Config::required` when picking the GPU, and enables only optional features the device has. (`state.rs:418-515`).
    - A device lacking any of them fails `vkCreateDevice` with `ERROR_FEATURE_NOT_PRESENT`.
    - `task_shader(true)` is requested without checking it.
    - `bindless_supported` is computed and never read.
    - Missing extensions are only logged, and init carries on.
18. ~~**`Ctx::get()` blocks forever before `lava::init`**~~ **Done (2026-10-16):** `Ctx::get()` panics before init; a second `init` fails before creating anything, and a failed init destroys what it created (device, messenger, instance). (`state.rs:47-50`). A second `init`
    builds a whole instance and device before failing, and leaks them; a partly failed init leaks
    too. Core works around the blocking with a lazy `OnceLock` in `material.rs`.
19. ~~**Acquire and present can race from safe code.**~~ **Done (2026-10-16):** a mutex in the swapchain is held by `aquire_image` and `Queue::present`. `aquire_image(&self)` and
    `present(&Swapchain)` don't synchronise the swapchain, and `Swapchain` is `Sync`.
20. ~~**`Surface::new` takes raw window handles but is safe**~~ **Done (2026-10-16):** `Surface::new` is `unsafe` with its contract documented, and every swapchain holds the surface's handle (`Arc`), so the surface is destroyed after its last swapchain whatever the drop order. (a dangling handle is UB), and a
    `Swapchain` doesn't own its surface, so drop order is left to field order in core.
21. ~~**Bindless `Load`/`Store` lack `NonUniformResourceIndex`**~~ **Done (2026-10-16):** every `Load` and `Store` in `bindless.slang` indexes with `NonUniformResourceIndex`. (`bindless.slang:21,67,76,79,92,95`).
    Handles read from buffers are divergent, and AMD requires the qualifier. Only `Sample`,
    `SampleLevel` and `GetDimensions` have it.
22. ~~**Passes with their own descriptor bindings aren't rejected.**~~ **Done (2026-10-16):** lava-build-slang rejects global parameters other than push constants, specialization constants and lava's three bindless heaps (by name, binding and set). A global `Texture2D t;` gets
    a Slang-assigned binding that collides with set 0 (`build.rs:917-925`).
23. ~~**Feature gating covers only ray tracing.**~~ **Done (2026-10-16):** a pass's needs come from its SPIR-V capabilities (`Features::of_spirv`). A mesh pass on a device without mesh shaders
    reaches pipeline creation with a `MESH_EXT` stage before the friendly panic
    (`pipeline.rs:89-99`, `command_buffer.rs:830,907`). The same holds for int64, float16 and
    shader clock. Map every `OpCapability` to `Features` at build time.
24. ~~**Handles nested in push-constant structs are silently untracked.**~~ **Done (2026-10-16):** a handle in a struct or array of push constants is a build error; `Buf<T>`'s pointer is `Ptr<T, Access.Read>`, so writes through it don't compile (core's `instance_flags` helper takes such a pointer now). A `Buf`/`Img`/`Accel`
    inside a struct becomes a bare `u64` with no access or barrier (`build.rs:392-396`). `Buf<T>`
    is also writable in Slang (`bindless.slang:151`) but tracked as a read.
25. ~~**Submit failure leaves stale tracking.**~~ **Done (2026-10-16):** see 9: a failed submit is replaced by one of its barriers, and a failed pool reset is logged instead of hiding the original error. Pending accesses and layouts are committed before
    `queue_submit2` (`queue.rs:660-689`), and `slot.pool.reset()?` can hide the original error.
26. ~~**Present retirement relies on a heuristic.**~~ **Done (2026-10-16):** with `Features::present_fences` (`VK_EXT_swapchain_maintenance1` and the surface maintenance instance extensions), every present signals a fence, and what a present may use waits for the presents made before its drop; `wait_idle` waits for them too (at most 1 s). Devices without it keep the heuristic. "One more epoch" assumes a later fence means
    earlier presents released their semaphores, which the spec doesn't promise. Use
    `VK_KHR_swapchain_maintenance1` present fences where available; this device has it.
27. ~~**Pool sizes and set counts use different limits.**~~ **Done (2026-10-16):** the pool sizes are the variable descriptor counts the sets allocate. The variable descriptor counts use the
    update-after-bind limits, but the pool uses the non-UAB ones (`bindless.rs:218-227,296-314`).
28. ~~**Some texel types are wrong.**~~ **Done (2026-10-16):** 64-bit formats have 64-bit texels (clear values must fit 32 bits), stencil is `[u32; 1]`, and combined depth-stencil formats read as `[f32; 1]` (their depth); lava-build-slang maps 64-bit shader element types too. `R64Sfloat` texels are `[f32; 1]` and R64 ints are 32-bit,
    while stencil is `[u8]` but shaders read `uint` (`format.rs:175,325-328,403-405,424-426`).
29. ~~**`#[validation_trace]` reads the proc-macro crate's `debug_assertions`, not the user's**~~ **Done (2026-10-16):** the expansion is guarded by `cfg(debug_assertions)`/`cfg_attr`, evaluated in the crate that uses it.
    (`lava-macros/src/lib.rs:12-21`).
30. ~~**Timestamps are written on families with `timestamp_valid_bits == 0`**~~ **Done
    (2026-10-15):** writing a timestamp on such a family panics.

### Free checks that are missing (a few integer compares each)
- **Zero and limit checks:**
  - zero-size buffers, zero extents, `mip_levels == 0` or beyond `log2(max) + 1`;
  - compute groups beyond `maxComputeWorkGroupCount`, raytrace dimensions beyond the device limit.
- **Copy bounds:** blit and copy regions outside the mip. ~~`copy_image_to_buffer` doesn't
  check the buffer size~~ *Done (2026-10-16): both buffer-image copies check the buffer holds
  the texels.*
- **Buffer commands:** `update_buffer` writes one `T` but tracks the whole slice, and checks
  neither 4-byte multiples nor 64 KiB. `fill_buffer` doesn't check alignment.
- **Casts:** `BufferSlice::cast` doesn't check size divisibility or alignment, and `mapped()`
  has only a `debug_assert`.
- **Attachments:** the extent isn't checked against the attachment, and a multi-mip `Single`
  view is accepted as an attachment.
- **Rendering hazards:** a rendering's draws are never checked for one reading what another
  writes, though all their accesses are already gathered.
- **Scissors:** `scissors(&[])` sets nothing, and extra scissors can never be used, since the
  pipeline has one.
- **Ray tracing:**
  - the vertex address isn't checked for component alignment;
  - a structure may appear twice in one batch;
  - `compact` doesn't check the destination size;
  - `update` checks nothing against the last build (counts, formats, transform presence).
- **Bindless:** `bindless::bind` is safe, but rewriting a slot that pending work reads is UB
  (either `unsafe`, or impossible once lava owns slots). Generated bindings could
  `debug_assert` that a passed image is bound in the needed set.
- **BLAS lifetime:** `Instance::new(&Blas)` looks borrow-checked but keeps nothing alive (STATE
  says this is intended). For a public crate, make `from_address`/`Instances::new` `unsafe`, or
  let the `Tlas` hold `Arc`s of host-built instances.

---

## 3. Performance

### Recording and pipelines
- **Per draw and dispatch** (`command_buffer.rs:829-833,1055-1065`):
  *Done for draws (2026-10-16):* the pipeline key is formats plus constants, looked up once per pass and constants in a rendering, with redundant binds skipped; a pass pushes only its own bytes.
  - it builds a `RasterHash` (two SmallVecs plus 128 B of constants) and SipHashes it under the
    global `RwLock`;
  - it always calls `cmd_bind_pipeline`;
  - it pushes all 256 bytes of push constants.

  Resolve the pipeline once per (rendering, state), use FxHash, skip redundant binds, and push
  only `size_of::<G>()`.
- **Compiling under locks.** A cache miss compiles while holding the write lock over every
  pass's map (`pipeline.rs:403-419`), inside the recording that holds the queue's pending lock.
  So one compile stalls every thread.
  - There is no `VkPipelineCache`.
  - Init builds every unspecialised pipeline serially.
  - There is no prewarm API.
  - Beyond `scratchpad.rs`: a per-pass `Mutex`, `FAIL_ON_PIPELINE_COMPILE_REQUIRED` for async
    compiles, `VK_KHR_pipeline_binary`, and deferred host operations for ray tracing pipelines.
- **Dynamic state.** Use core 1.3 dynamic state for cull mode, front face, depth test, write and
  *Done (2026-10-16):* every state this device supports is dynamic (`command_buffer/raster_state.rs`).
  compare, depth bias, stencil, topology and primitive restart, plus extended dynamic state 3
  where present for polygon mode, samples, blend and write masks. The pipeline key then shrinks
  to formats plus constants.
- **`RasterBuilder` moves `[Draw; N]` (about 1.2 KB each) on every call.** That is O(n²) memcpy
  *Done (2026-10-16):* the builder works through `&mut self` and borrows its draws (unsized `DrawOutput`s, 16 inline).
  and one monomorphisation per draw count.
- **Raster passes are coarse in barriers.** Every resource of a raster pass is accessed at
  `VERTEX|FRAGMENT` or `TASK|MESH|FRAGMENT` (`build.rs:30-41`).
- **Fresh images transition from `ALL_COMMANDS`** (`command_buffer.rs:1301-1304`), which drains
  the GPU once per streamed image. Only swapchain images need that.
- **Discarding is missing.** There is no `DONT_CARE` load or store, and discarding accesses
  (clears, full copies, the first use after acquire) don't transition from `UNDEFINED`. Core
  loads the G-buffer and the UI target every frame.
- **`VK_KHR_unified_image_layouts`** would remove the per-mip layout atomics, the
  GENERAL↔ATTACHMENT round trips (four per frame for the G-buffer), bug 2.12, and the rule that
  command buffers sharing an image be recorded in order.
- **Command buffer hints:** no `ONE_TIME_SUBMIT` and no `TRANSIENT` pool.
- **Per-frame allocations.** Barrier SmallVecs spill when a command transitions several images,
  as the scene rendering does every frame, and labels allocate a `CString`.

### Memory
- **There is no readback memory.** The only host-visible location is `CpuToGpu` (write-combined).
  That fits core's headless readback taking about 10 ms for 3.7 MB. Add `GpuToCpu` (cached).
- **With ReBAR, every buffer goes into the BAR heap,** scratch included, and spills silently to
  system memory when it fills. The SBT is host-visible too.
- **No staging path.** Without `host_image_copy`, which lava only enables on ReBAR devices,
  textures, the font atlas and previews can't be uploaded, so core's UI can't start on such a
  device.
- **Allocator settings are fixed.** Per-allocation logging is on, blocks are 256 MiB, and every
  allocation is named "buffer" or "ImageMemory" (`state.rs:195-204`).

### Other
- **The ANV `MEMORY_READ` workaround on trace reads applies to every vendor.** Gate it on
  `DriverId::INTEL_OPEN_SOURCE_MESA` (`acceleration_structure.rs:25-32`).
- **Every queue of a chosen family is created.** NVIDIA's graphics family has 16
  (`state.rs:646-662`).
- **The debug callback works for every message.** It enables VERBOSE and INFO and formats each
  message before `tracing` filters it.
- **Descriptor heaps or descriptor buffers** would turn `bind` into a memcpy. They would remove
  the pool, the update-after-bind flags and the per-command-buffer binds of two sets, and make
  user samplers and buffer descriptors cheap.
- **Shader builds.** Each pass has its own Slang session, compiled serially. There is always `Maximal` optimisation and never debug info, so no
  source-level shader debugging.

---

## 4. API friction

1. **`RasterBuilder` counts draws in its type**, so draws can't be added in a loop; core records
     *Done (2026-10-16):* `cmd.raster(extent, [Attachment::color(..); N])`, borrowed draws, addable in loops.
   draws of count 0. Only the attachment count needs checking. **Direction:**
   `cmd.raster(extent, [attachments; N], depth)` with a plain const `N` (stable Rust), and the
   draws in a `Vec` the `CommandBuffer` reuses.
2. **`push_bindings` is positional** (15 arguments for `BvhCull`), so swapping two
   `BufferSlice<u32>`s or two `Mat4`s compiles. Generate a struct with named fields, with
   specialization constants as fields too (that removes `IncompleteX`).
3. **Init takes positional bools** (`init(display, bool, bool)`).
     *Partly done (2026-10-16):* `lava::init(&Config)` with passes, display, validation, samplers and required features. Still missing: app name, device choice, extra extensions and a `DeviceInfo`.
   - **Missing settings:** app name and version, required and optional features, device choice
     (ranked by capability, user override, presentation support), extra extensions and a pNext
     hook, validation settings and queue counts.
   - **Validation:** it is silently ignored in release builds, and init fails without the SDK's
     layer.
   - **Direction:** `lava::init(Config) -> Result<DeviceInfo>`, with `DeviceInfo` reporting
     supported vs enabled features, limits, subgroup, mesh and ray tracing properties.
4. **The swapchain leaves the hard parts to the user.**
   - The user manages acquire semaphores per frame in flight, `render_finished` per image
     (recreated on every resize), the image index, and the `PRESENT_SRC` transition.
   - Out-of-date is an `Err` from acquire but `Ok(true)` from present, suboptimal is dropped,
     and acquire's `wait_on` parameter is actually signalled.
   - **Direction:** a presenter owning surface, swapchain and semaphores.
     `acquire() -> Acquired | OutOfDate | Minimised` recreates as needed, and
     `frame.execute_and_present(..)` wires the waits and signals. Core's `render.rs:100-345` is
     what every user would otherwise rewrite.
5. **Memory is chosen by `cpu_writable: bool` plus the device.** **Direction:** a location type
   (`Device`, `Upload`, `Readback`, `DeviceMapped`) where host access exists only on mapped
   locations, plus an uploader (a staging ring recycled by epoch, with host copy as the fast
   path).
6. **Frames in flight are a convention.** Core builds per-frame rings four ways: instance
   offsets, gizmo `first`, UI buffers per slot, and an unsynchronised readback. `FrameSlot::new`
   and `execute` each take the queue, and can disagree. A blocking one-shot submission needs six
   steps. **Direction:**
   - `Frames<N>` owning the queue lease and slots;
   - `PerFrame<Buffer<T>>` whose `host_mut(&frame)` ties writes to the waited slot;
   - a growable stream buffer and a typed `Readback<T>`;
   - `queue.submit_and_wait(|cmd| ..)`;
   - several submissions per frame.
7. **Bindless slots are the user's job.**
   - The caller picks every index, and freed slots come back through one global `fn` hook, so
     two libraries can't both use it. Core's 90-line allocator and `material.rs`'s epoch free
     list are the same machinery twice.
   - A handle is 8 bytes for two indices.
   - **Direction:** lava allocates slots with epoch-deferred reuse, using the same index in both
     sets so a handle can be one `u32`. Keep `reserve_range(n)` for asset-indexed schemes, and
     add a `SlotBuffer<T>` for host-written slot arrays like the materials.
8. **95 image usage marker types**, with names like
   `ColorAttachmentSampledStorageHostCopyTransferSrcTransferDst`, and each new flag (transient,
   input attachment) doubles them. **Direction:** const-generic flags for images. For buffers
   the parameter only switches `address_flags`, so consider dropping it, after the same
   benchmark as for the transfer flags.
9. **Errors are mostly `Message(String)`.**
   - Out-of-date, minimised, surface lost, device lost, unsupported and invalid argument
     can't be matched.
   - Out-of-memory comes in two variants.
   - Recording panics on runtime failures such as pipeline creation (`must`).
   - **Direction:** typed `#[non_exhaustive]` variants, and the rule "panic for programmer
     errors, `Err` for device and environment".
10. **Transfers only reach a view's first mip,** so uploading mip k of a `Single` image needs a
    `MipChain`, and there is no `generate_mips`. Take a subresource (image, mip, layers,
    region), with several regions per call.
11. **Mixed image types are awkward.**
    - `bind` takes one `F/U/D` for all images.
    - There is no `ImageView::erase()`.
    - The viewport target is typed like the swapchain image, so it lacks SAMPLED, and the UI
      reads it through a storage load, which is several times slower on Intel.
    - Bound combinations repeat (`HasTexels<[f32; 4]> + ColorAspect, IsColorAttachment + IsStorage`).
12. **Shader facts are mirrored by hand.** Workgroup sizes appear as `div_ceil(8)` and
    `div_ceil(64)`, and constants like `MESHLET_STRIDE`, `INSTANCE_FLAG_BITS` and the overflow
    bits are copied. Generate `Pass::WORKGROUP_SIZE` with a `compute_threads(extent)`, and export
    marked Slang constants.
13. **Indirect commands need byte gymnastics.** Core uses
    `byte_range(offset_of!..).cast::<MeshletDraw>()` and a 16→12-byte cast. Add field projection,
    and let indirect draws take any `T` that starts with the command, using its stride.
14. **Scopes are fragile.** *Partly done (2026-10-15):* `cmd.scope(name, |cmd| ..)` replaced
    `begin_scope`/`end_scope`, so scopes can't be unbalanced. Names must still be
    `&'static str`.
15. **Resize and grow is hand-rolled.** `RenderResources::fit`, a `fresh` flag and
    `grow_overflowed` could become `Image::fit(extent)` and `Buffer::reserve(len)` that keep the
    bound slots.
16. **Ray tracing ergonomics.**
    - Scratch: users call `Buffer::raw(.., Some(scratch_alignment() as u32))`, sum the sizes by
      hand, and a batch that doesn't fit panics. Add a scratch type and a `build_all` that splits
      into batches.
    - Compaction is a six-step sequence; add a compactor.
    - Every structure owns a `VkBuffer`. Allow placement in an arena slice.
    - Sizing a BLAS needs real buffers; add `Blas::with_capacity(desc)`.
    - `BuildFlags` can't say "no preference"; flags are raw `vk::GeometryInstanceFlagsKHR`;
      `VertexPosition` is open and exposes `vk::Format`.
    - `Instance` has no getters, and `Describe`/`Described` are public.
17. **The counters API mirrors one engine's UI.** *Partly done (2026-10-15):* a counter
    selection is a `QueryPool::performance`, validated when it is created, and results are
    `&[f64]`.
    - Selections must still fit one pass, and only the graphics family is counted.
    - **Direction:** a multi-pass capture for tools (STATE.md).
18. **Generated types belong to lava,** so users can't implement serde or `Reflect` on them
    (the orphan rule). Generating into the user crate fixes this, with configurable derives.
19. **The surface is messy.**
    - Deep paths (`lava::vkobjects::queue::..`, `lava::image::slice::ImageView`).
    - No prelude, and glam and bytemuck aren't re-exported.
    - Unused re-exports: `VkPipeline`, `VkShaderModule`, `ShaderHash`.
    - Public internals: `PassEntry`, `Functions`, `PhysicalDevice`, `CALLSITE`,
      `RaytracingPipeline`, `ShaderBindingTable`.
    - Typos: `aquire_image`, `QueueFamilie`, `handel`, `Filter::Liniear`, `extend` for extent,
      `enable_gpu_assited_validation`, `unsupports_extensions`.
    - Fix these before 0.1, since renames break users afterwards.
20. **Codegen errors lack context.** Panics give no file or field (`build.rs:111,118,236`), the
    first failing pass hides the rest, and Slang warnings aren't forwarded.

---

## 5. Missing features

- **Raster:**
  - **Fixed-function state:** topology (lines, points, strips), front face, depth compare and
      *Done (2026-10-16) as dynamic state:* topology, front face, depth compare and test without write, stencil, depth bias values, write masks, custom and min/max blending, blend constants, depth clamp and bounds, conservative raster, line modes and stipple. MSAA states stay fixed until lava has MSAA.
    test-without-write, stencil, depth bias values (bias is enabled with zeros today), colour
    write masks, custom and premultiplied blending, min/max, blend constants, depth
    clamp/bounds, conservative raster.
  - **Sampling:** MSAA with resolve attachments, sample shading, alpha-to-coverage.
  - **Rendering:** multiview, layered rendering, a render area offset, per-draw scissor and
    viewport (UI clipping), vertex-only passes (depth and shadow), dynamic rendering local read.
  - **Draws:** `draw_indexed_indirect(_count)`, `launch_indirect_count`, u16 indices,
    first vertex/instance and vertex offset, multi-draw.
- **Compute:** reflected workgroup size, dispatch base, subgroup size control (`[WaveSize]` →
  `requiredSubgroupSize`), several entry points per file.
- **Images:**
  - **Types:** array layers, cube maps, 1D, MSAA.
  - **Views:** format reinterpretation (sRGB storage through a UNORM view, R32 aliases for
    atomics), swizzles, per-aspect depth/stencil views.
  - **Checks:** a format-support query.
  - **Transfers:** image-to-image copy, depth/stencil clears and copies, row pitch, mip
    generation.
  - **Memory:** transient (lazily allocated) attachments, aliasing in shared memory blocks,
    sparse residency, external memory interop.
- **Samplers:** user-created samplers, anisotropy (`samplerAnisotropy` is never enabled),
    *Partly done (2026-10-16):* `Config::samplers` (`SamplerDesc`: filters, mipmap mode, address mode, border, anisotropy). Compare and min/max reduction samplers are still missing.
  clamp-to-edge, compare samplers for PCF, min/max reduction for depth pyramids, mirror.
- **Slang coverage:**
  - **Types:** `half` and 16-bit ints (panic today, though float16 is enabled), non-square
    matrices (no glam type), `float3x3` layout vs `Mat3` *(plausible)*.
  - **Image ops:** `Gather`, `SampleCmp`, `SampleGrad`, `SampleBias`, `Load` at a mip, image
    atomics.
  - **Codegen:** Rust keywords as field names, arrays over 32 with `derive(Default)`, pass and
    struct name clashes.
- **Memory:** staging uploads, readback, per-frame transient rings, a budget query outside
  profiling, object and allocation names.
- **Queues:** an async compute family, transfer with ownership transfers, several submissions
  and command buffers per frame, parallel recording (per-thread trackers merged by a barrier
  prologue at submit).
  - Timeline semaphores lack `value`, `signal`, timeouts and multi-wait.
  - `Fence` is half-exposed, and `FrameSlot` has no non-blocking `is_done()`.
- **Swapchain:** format and colour space choice (HDR10, scRGB, `VK_EXT_hdr_metadata`), extra
  usages, image count, composite alpha, `FIFO_LATEST_READY`, swapchain maintenance1,
  present_id/present_wait for pacing, presenting several swapchains in one call, discard on
  acquire.
- ~~**Queries outside `profiling`:** timestamps (for dynamic resolution), occlusion queries,
  pipeline statistics with chosen flags.~~ **Done (2026-10-15):** user-owned `QueryPool`s of
  every kind. Conditional rendering is still missing.
- **Debug:** object names on buffers, images and structures; labels in release builds whenever
  the extension exists; `insert_label`.
- **Escape hatches:**
  - an `unsafe` `lava::raw` with device, instance and queue;
  - `CommandBuffer::raw(declared_accesses, |vk_cmd| ..)` that flushes barriers for what it
    declares;
  - an access declaration for memory reached only through pointers;
  - runtime SPIR-V passes for vendor libraries, material graphs and mods.
- **Ray tracing:**
  - **Pipelines:** deferred host operations, pipeline libraries,
    `vkCmdSetRayTracingPipelineStackSizeKHR`.
  - **Structures:** serialization and copies (BLAS disk cache, clones), position fetch
    (`ALLOW_DATA_ACCESS`), opacity micromaps.
  - **Shaders:** shader execution reordering, `NO_DUPLICATE_ANY_HIT_INVOCATION`.
  - **Formats and instances:** half and snorm vertex positions, a documented inactive instance
    (BLAS address 0).
- **Lifecycle and testing:**
  - `lava::shutdown()` that waits, drains epochs, destroys everything and allows re-init. Today
    `destroy()` destroys only the bindless objects and leaves lava unusable, and nothing calls
    it.
  - ~~Feature masking at init, so tests can cover the fallback paths.~~ **Done (2026-10-17):** `Config::disabled`, used by `gpu_missing_features`.

---

## Suggested order

1. **Decide the packaging architecture first,** since it reshapes the generated API: `lava-build-slang`,
   per-pass statics, Slang only at build time, stable Rust, and released ash.
2. **Fix the soundness bugs (section 2),** most of them local.
3. **Redesign the APIs users touch first:** `RasterBuilder`, init config, memory locations,
   swapchain presenter, frame and ring helpers, bindless slots, errors, naming.
4. **Fill the raster state through dynamic state,** which is also a performance win.
5. **Add features by real-world demand:** arrays, cubes and MSAA, samplers, staging, queries,
   async compute.

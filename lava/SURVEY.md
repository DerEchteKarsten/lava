# lava: publishing survey (2026-10-14, second pass 2026-10-18)

A read-only review of the whole crate against the publishing goals: meaningful abstractions,
cheap-but-valuable performance trade-offs, broad real-world coverage, a forward-looking device
baseline, a clean API, free safeguards, and raw Vulkan only in extreme cases. The second pass
(2026-10-18) read every line of lava, lava-build-slang, lava-macros and lava-tests, plus the
engine's use of lava, and counts anything as a finding, down to a redundant line. Its line
numbers are from the working tree of 2026-10-18; older ones from 2026-10-14. Unmarked items were
checked in the code; *(plausible)* ones follow from it but weren't reproduced. Bugs are also
listed in `STATE.md`. Done items are removed.

## 1. Decided work (2026-10-20)

Every open item was re-checked on 2026-10-20; stale ones were deleted. The user decided what to
do with each remaining one. This is the work list, in planned commit groups (at most ten lava
commits). Numbers are this survey's items. Delete a line, and its item below, when it is done.
Items that aren't named here stay as they are, on purpose (e.g. bug 31, MSAA, async pipeline
compiles, `VK_KHR_unified_image_layouts`, bind pairs and `ImageView::erase`).

3. **Timelines, lifetimes, init and errors:** 24 (a full `lava::shutdown()` that waits, drains,
   destroys device and instance and allows re-init), 64/65 (host-signal the semaphore when a
   timeline drops with its stamp ahead or a replay fails), 66, 67 (the public `reclaim` polls),
   68 (a reclaim mutex), 69 (a failed init tears down through `shutdown`; pure checks before
   device creation), 71, 74, 75 (CPU and virtual GPUs accepted, ranked last), 76 (device loss
   propagates as a typed error), 44 (a stage set per queue family, a type parameter of `Wait`,
   masked by features), 49, 61 (document the validation layer version that accepts GPU-assisted
   validation with printf), 4.9 (typed `#[non_exhaustive]` errors, one `OutOfMemory`, no
   duplicate `Display`, a heap enum, no `From<String>`, no `Lock`, the panic-vs-`Err` rule
   documented and applied: host copy checks panic, `reserve(0)` is empty, `pipeline::init`
   returns `Err`, statistics return `MissingFeature`, `performance(0)` checks before locking),
   4.3 (app name and version in `Config`; `validation` honoured in release; a missing layer
   warns), `Stamp::is_done` uses cached values first, the debug callback filters before
   formatting, allocator block sizes in `Config`, allocation and object names, 4.29 (integer
   `typ` and `flags`), `Wait::block_timeout` and multi-wait.
4. **Queries, counters and presenting:** 22/26 (used-index bitset per pool; the open pool and
   index per kind; `end_query` checks them; submit fails with an open query), 25, 45 (query
   kinds bounded by queue kind at the type level; empty statistics are an error), 46
   (command-buffer-scope counters begin as the recording's first command; render-pass scope is
   asserted outside renderings), 4.17 (`counters::fits`, and the multi-pass capture of STATE.md),
   77/82 (`present`/`acquire` panic on a recording thread), 78, 79 (present fences required for
   windowed use), 80, 81, 83, 4.25 (`PresentMode` enum, lava surface formats, a reported
   fallback), composite alpha, `FIFO_LATEST_READY`, HDR metadata and colour spaces, presenting
   several swapchains at once, discard on acquire, cached surface capabilities, cached
   calibration in `instant`, a cheap `memory_report`.
5. **Raster and draws:** 38 and 39 (require `multiDrawIndirect`, `drawIndirectFirstInstance`,
   `depthBiasClamp`), 40 (query `dynamicPrimitiveTopologyUnrestricted`; key by topology class
   without it), 41 (`depth_clip` defaults to not clamped), 42 (`write_mask` and gated
   `write_masks`), 55 (asserts, a `dual-src-blend` feature), render area offset, per-draw scissor
   and viewport, vertex-only passes, array layers, layered rendering, multiview, dynamic rendering
   local read, indexed indirect (count) and `launch_indirect_count`, u16 indices, first
   vertex/instance and vertex offset, multi-draw, stencil attachments, `BufferAccess::unordered`,
   4.13 (an unsafe prefix trait for indirect command structs), `Scissor`/`Viewport` merged,
   stencil state set `FRONT_AND_BACK` when equal.
6. **Images and formats:** 27 (block sizes for BC, ETC2, EAC and ASTC; the compression features
   gated and enabled), 43 (64-bit clear components as two words), 53 (a format-property cache;
   blit, clear, address, granularity and host block checks; `with_params` asks the device), 54
   (attachment usages with `HostCopy` don't compile; states commit only on success), 57, 58,
   format reinterpretation views, image-to-image copies, depth and stencil clears and copies, row
   pitch, `ImageSlice::mip(level)` on `Single` images, sparse residency, external memory,
   `Image::fit`, `UnifiedBinding` deleted.
7. **Buffers and acceleration structures:** 28-30 (full shapes, `built` applied when the
   recording reaches the GPU, a cheap built flag for traces), 34, 36, 37, 47, 48, 59, 4.16 (all
   of part 1 and 2: scratch type and `build_all`, compactor, arena placement,
   `Blas::with_capacity`, flags, sealed `VertexPosition` with `Vec3A`, getters, scopes, safety
   docs), copies and serialization, position fetch, half and snorm positions, shader execution
   reordering, 4.27, `Buffer::reserve`, 4.6 (`PerFrame<T>`), 4.23 (`ImageHandle` loses
   `Zeroable`; upload and host-write bounds become `NoUninit`), the ANV workaround only on Intel
   Mesa, stored `unsized_struct!` offsets.
8. **lava-build-slang:** 4.2 (two generated argument structs, push constants and specialization
   constants, plus the extra accesses as a third argument of `push_bindings`; no
   `IncompletePass`), 4.12 (`WORKGROUP_SIZE` and exported constants), 4.20, 4.21 (qualified
   paths, `#[track_caller]`), 4.22, 35, 63, 85-88, 90 (`Accel` allowed in `SCONST`, cost
   documented), 91-97, debug info and parallel compiles, half and 16-bit fields, non-square
   matrices, `r#` field names, `Gather`/`SampleGrad`/`SampleBias`, `Load` at a mip, image
   atomics, `MutTex2D.Sample` and `MutTex3D`, dispatch base, `[WaveSize]`, several compute entry
   points per file, `#[validation_trace]` argument check, `#[must_use]` on `Shaders`.
9. **Surface, cleanup and docs:** 4.14, 4.19 (shallow re-exports, `lava::prelude`, glam and
   bytemuck re-exported, internals hidden or deleted, every rename, ash as a documented public
   dependency with only `lava::vk`, the derives), all of section 6 (bake table stays) and
   section 7 (scope observer docs follow the code: scopes start after their barriers).
10. **Tests:** all of section 8, plus tests for every change above, one GPU-assisted validation
    run, and Miri for `unsized_struct!`; the tools generator fix of section 7.

The engine follows in its own repository, only after the other agent's uncommitted engine work
is committed: 84, the push constants of 3, 4.11's viewport usage, 4.15, 4.17's `fits`, 4.21's
imports, 4.22's padding, release builds without validation, and all of section 9 (drop
`dynamic-blend` once checked).

## 2. Bugs

### Barrier tracking (2026-10-18)
22. **One query index can be used twice in a recording** *(plausible)* (`mod.rs:723-724, 1121,
    1137`): the pool is reset once per recording, and two `count_samples`, queries or timestamps
    on one index aren't refused.

### Undefined behaviour from safe code (2026-10-18)
24. **`lava::destroy()` is safe but leaves `BINDLESS` set** (`lib.rs:151-165`): a later bind or
    descriptor write uses the freed heaps. It also
    sets `CALLSITE` by hand and never clears it, in release builds too, so every later validation
    message on that thread blames `destroy`'s caller (`lib.rs:153`). Make it `unsafe` or guard it.
25. **`QueryPool::read` overflows** (`query/mod.rs:191-197`): `first + queries` wraps in release
    and `out.len() as u32` truncates, so `read(u32::MAX, ..)` passes the assert.
26. **`QueryRecorder::end_query` checks no pool or index** (`observer.rs:50-52`,
    `mod.rs:1343-1364`): `active_queries` knows only the kind. Nothing checks that a recording ends
    with no query open either (`timeline.rs:389-403`), so an observer that only begins one submits
    an active query. Track `(pool, index)` per kind and assert at submit.
27. **Host copies of ASTC and ETC2 images skip every size check** (`format.rs:357-394`). The
    formats have no block info, so `expected_data_len` is `None`, and `copy_from(&[0u8; 1], 0)`
    makes the driver read past the data. lava never enables `textureCompressionASTC_LDR` either
    (STATE has the ETC2 half). Give them block sizes or drop the markers.
28. **Tracing never checks that the TLAS was built** (`acceleration_structure.rs:352-354`,
    `mod.rs:1477`, generated `Accel` bindings): a fresh `Tlas::new` traced reads uninitialised
    memory. Update, compaction and queries check `built`; an `AtomicBool` makes this cheap.
29. **A build is checked against the creation only by counts** (`acceleration_structure.rs:771-798`).
    Vulkan sizes memory and scratch from an identical build info (geometry type, vertex format,
    max vertex, index type, transform, flags). Building AABBs into a BLAS created from triangles,
    or more vertices, passes. Keep the creation `Shape` and compare.
30. **`built` is set when the build is recorded** *(plausible)* (`acceleration_structure.rs:851-853`):
    a recording that never reaches the GPU leaves the structure counted as built.
31. **`compacted_from` stores the source's raw handle** *(plausible)* (`:181, :306, :961`): a
    handle value reused after the source's destruction passes `compact`'s check. Use an id.
34. **Indirect commands read past an empty slice** (`mod.rs:958-965, 1014-1021, 1055-1062`):
    `compute_indirect` and `raytrace_indirect` pass only the address. Assert `len() >= 1`.
35. **lava-build-slang calls an unchecked 0.1.0 accessor** (`compile.rs:61`, `codegen.rs:585, 594,
    633`): `VariableLayout::category()` is `self.type_layout().parameter_category()`, the unchecked
    `type_layout` the crate forbids. Use `checked_type_layout()`.
36. **`unsized_struct!`'s alignment assert sits only in `from_parts`** (`unsized_struct.rs:67`),
    though the accessors' safety depends on it. Hardening: move it into `Deref` or `LAYOUTS`.
37. **`iter_draws<'a>` returns `&'a DrawOutput<'a>` for any caller-chosen `'a`** (`mod.rs:546-550`).
    Internal; make it a method of the builder so the lifetime is its own.

### Vulkan valid usage (2026-10-18)
38. **`multiDrawIndirect` and `drawIndirectFirstInstance` are never enabled** (`features.rs:657-683,
    794-797`). `Draw::indirect` and `launch_indirect` pass `drawCount = buffer.len()`
    (`mod.rs:406, 428`), so any buffer of more than one command is invalid, and so is a non-zero
    `first_instance`. The tests draw one command. Require and enable both.
39. **`depthBiasClamp` is never enabled** (`raster_state.rs:90-96, 600-608`), so any non-zero
    `DepthBias::clamp` is invalid on every device. Depth bias has no test.
40. **Line and point draws break without extended dynamic state 3** (`pipeline.rs:672-674, 736`).
    Vertex pipelines bake `TRIANGLE_LIST` and set the topology dynamically, which crosses topology
    classes only with `dynamicPrimitiveTopologyUnrestricted`, never queried. `gpu_headless` draws
    `LINE_LIST` this way; `Config::disabled` can't show it, since it is a property. Put the
    topology class into `Baked` unless the property is set.
41. **`depth_clamp(true)` still clips where `VK_EXT_depth_clip_enable` exists**
    (`raster_state.rs:293-301`, `pipeline.rs:813-821`): lava always pushes `depth_clip = true`,
    so one state renders differently per device. Default `depth_clip` to `!clamp` unless set.
42. **Per-attachment write masks need `independentBlend`** (`raster_state.rs:234-235`,
    `pipeline.rs:742-764`): `write_masks` has no gate while `blends` does, and on the baked path
    different masks make different `pAttachments` entries.
43. **64-bit clear values are packed as 32-bit** (`format.rs:198-204`). Vulkan reads a 64-bit
    component from two array elements, so `R64Sfloat` clears to garbage, `R64G64Uint [a, b]` puts
    both into the first component, and values above 32 bits panic.
44. **`wait_for` stage masks ignore the waiting queue** (`timeline.rs:289-297`): `Stage::Raster`
    on a compute or transfer timeline, compute stages on a transfer one, and ray tracing stages
    without the feature are all invalid. Mask by `K` and the features.
45. **Queries ignore the family** (`query/mod.rs:29-37, 133-141`): occlusion queries compile on
    compute and transfer timelines, and compute-only statistics on transfer.
    `statistics(count, empty())` makes a pool with no statistics (`:319-335`).
46. **Counter scopes aren't checked** *(plausible)* (`query/mod.rs:113-125`, `counters.rs`):
    command-buffer-scope counters must begin as the first command, but lava binds descriptors
    first; render-pass-scope ones must not begin inside a rendering.
47. **Empty work reaches the driver.** `build(&[])`, `update(&[])` and `query_compacted_sizes(&[], ..)`
    pass a count of 0 (`acceleration_structure.rs:754-945`); an empty `fill_buffer`, copy region or
    region list does too (`buffer/slice.rs:185-295`, *plausible* for the address variants). Return
    early.
48. **Overlapping copy regions pass** *(plausible)* (`buffer/slice.rs:253-288`), and `src.len()`
    truncates a range that isn't a whole number of elements.
49. **Unsupported device extensions are chained.** *Partly fixed 2026-10-19:* `properties2` chains
    only the property structs of extensions the device has, and devices below Vulkan 1.4 are
    rejected. `Vulkan13Features` is still chained into the feature query of every device,
    including those then rejected (`features.rs:653-686`).
53. **Image commands miss format checks.**
    - `blit_image` (`image/slice.rs:308-367`) doesn't check `BLIT_SRC`/`BLIT_DST`, linear
      filtering support, integer formats with linear filtering, mixed signedness, or overlap.
    - `clear_image` accepts block-compressed formats (`:277-302`).
    - Buffer addresses aren't checked against the texel block size outside transfer families
      *(plausible)* (`:373-469`).
    - `minImageTransferGranularity` is in blocks for compressed formats *(plausible)* (`:561-572`).
    - `copy_region_from` doesn't check BC block alignment (`image/mod.rs:665-681`).
    - `with_params` never asks `vkGetPhysicalDeviceImageFormatProperties2`
      (`image/mod.rs:378-411`): `R8G8B8A8Srgb` storage, 3D BC or depth images and host copies of
      formats without `HOST_IMAGE_TRANSFER` compile and are invalid.
54. **A host layout transition starts from the tracked layout** *(plausible)* (`image/mod.rs:740`):
    `ATTACHMENT_OPTIMAL` may not be in `pCopySrcLayouts`, and the states move before
    `transition_image_layout` can fail.
55. **Raster state values aren't checked** *(plausible)* (`raster_state.rs`): `SRC1_*` factors
    without `dualSrcBlend`; blending on attachments without the `BLEND` format bit; logic ops;
    the stipple factor (1..=256); `line_width` against `lineWidthRange`; depth bounds and clamp
    ranges against [0, 1]; `UNDERESTIMATE` without `primitiveUnderestimation`
    (`features.rs:737-738`); primitive restart on list topologies.
57. **`E5B9G9R9UfloatPack32` has 4 components** (`format.rs:356`), so a `Tex2D<float3>` refuses it.
58. **`Dynamic::info` panics for formats outside the table** (`format.rs:77`): A4R4G4B4, A8 and
    A1B5G5R5 are missing, and `PresentSurface::new`'s fallback could pick one *(plausible)*.
59. **Acceleration structure limits aren't checked** (`acceleration_structure.rs:335-349`):
    `maxGeometryCount`, `maxPrimitiveCount`, `maxInstanceCount`.
61. **`GPU_ASSISTED` is enabled together with `DEBUG_PRINTF`** *(plausible)* (`state.rs:153-160`,
    only with `Config::gpu_assisted_validation`), which older validation layers refuse.
63. **`bindless.slang`** *(plausible)*: `DynImg.Sample` returns an uninitialised `zero`.

### Timelines, lifetimes and init (2026-10-18)
64. **A `Timeline` dropped after `reserve()` or an unsubmitted `begin()` stops every destruction**
    (`timeline.rs:127-141`). Its index keeps `stamp > completed`, and `latest_stamp()` covers all
    indices, so nothing dropped afterwards is destroyed until another timeline takes the index and
    submits. Host-signal the stamp in `Drop`.
65. **A failed barrier-only replay leaves a value nothing signals** (`timeline.rs:423-433`): the
    next `begin` with `in_flight == 1`, and the timeline's drop, wait forever.
66. **The "must present it" panic has no message** (`timeline.rs:348-351, 441`): `resume_unwind`
    skips the panic hook, so an uncaught case exits with code 101 and no text.
67. **The public `reclaim()` never polls** (`lifetime.rs:50-67`): it compares against cached
    values, so an application calling it alone frees nothing.
68. **Destructors run in drop order only for one reclaimer** (`lifetime.rs:49-66`): each is popped
    under the lock and run outside it, so two threads can run a view's and its image's destructors
    together or swapped.
69. **`init` isn't atomic** (`lib.rs:126-134`). `STATE` and `FUNCTIONS` are set before bindless,
    counter and pipeline init can fail (a heap allocation, a pass needing a compiled-out
    feature), so the device leaks and every retry says "already initialised".
71. **Failed creations leak.** A failed command pool leaks the earlier slots' pools
    (`timeline.rs:178-202`).
74. **`#[validation_trace]` on `Ctx::init` always records `lib.rs:127`** (`state.rs:118`), since
    `lava::init` isn't `track_caller`. Put it on `lava::init`.
75. **Software and virtual GPUs are dropped silently** (`state.rs:329-332`): lavapipe never shows
    in `rejected`, so a CI machine gets "no GPU found". The `_ => 2` sort arm is dead.
76. **After device loss every poll logs an error** (`timeline.rs:60-68`), several per frame.

### Presenting (2026-10-18)
77. **A surface can present before its submission exists** (`present.rs:49-57, 491-526`).
    `cmd.present` marks the image presented while recording, and the submit closure may hold the
    surface, so `surface.present()` or `acquire()` inside it queues a present on an unsignalled
    semaphore. The struct doc of `Current` still says the value comes from the submission.
78. **A failed swapchain recreation keeps the retired swapchain** (`present.rs:351-370, 427-430`).
    With `oldSwapchain` set, Vulkan retires it even when creation fails, but it stays
    `self.swapchain` and is passed as `oldSwapchain` again.
79. **Dropping a surface without present fences can destroy a swapchain mid-present**
    *(plausible)* (`present.rs:579-593`): the drop stamp covers the last submission, not the
    `vkQueuePresentKHR` after it. Presents go on queue 0 while the timeline may use another
    graphics queue, which weakens the "later value" rule too (`:322`).
80. **`present()` clears `current` before its fence calls can fail** (`present.rs:495-515`): the
    image is never presented and a popped fence leaks.
81. **`presented` is set before `flush_pending`** *(plausible)* (`present.rs:57`): a panic between
    them leaves the surface believing the image was presented.
82. **`acquire` inside a submit closure deadlocks** *(plausible)* (`present.rs:319-327, 97-111`):
    destroying an old swapchain locks every timeline's pending accesses, including the one held.
83. **Accesses after `cmd.present` in the same recording aren't refused** (`present.rs:49`).
84. **The engine's `Presenter::drop` waits before the surface drops** (engine
    `render/render.rs:139-147`): `PresentSurface::drop` only retires its destruction, so surface
    and swapchain outlive the window unless something reclaims later.

### lava-build-slang (2026-10-18)
85. **`layout_hash` ignores field types** (`compile.rs:54-101`): a hot reload from `Tex2D foo` to
    `Buf<uint> foo` is accepted, and the shader dereferences a bindless handle as an address.
    `float` to `uint`, an image's shape, set or element, a pointee's layout and a constant's type
    pass too. Hash each field's generated type and kind.
86. **`layout_hash` uses `DefaultHasher` over `usize`** (`compile.rs:58, 69-89`): a cross-compiled
    application refuses every reload. Hash fixed-width integers.
87. **Struct signatures hold names and offsets, not types** (`codegen.rs:132-157`): `Foo { float x }`
    and `Foo { uint x }` from two passes, or `Pair<float>` and `Pair<uint>`, merge into the first.
88. **Pointees and array elements skip `gpu_type`** (`codegen.rs:125-129, 402`): `Buf<Tex2D>`
    becomes `BufferSlice<Img>` with a generated `Img` struct instead of `ImageHandle`, and so do
    `[Img; N]`, `[Buf<T>; N]` and `[Accel; N]` behind a `Buf`.
90. **`SCONST` rules disagree with the docs** (`codegen.rs:614-622`): `Accel` is refused for
    run-dependent addresses while `Buf`/`MutBuf`, also addresses, pass.
91. **Non-struct pointee strides aren't checked** *(plausible)* (`codegen.rs:401-403`):
    `Buf<float3>` becomes 12-byte `Vec3` elements against a 16-byte std430 stride.
92. **Matrices ignore their scalar type** (`codegen.rs:117-124`): `int4x4` becomes `Mat4`;
    `float3x3` against `Mat3` fails the size assert *(plausible)*, where `Mat3A` would fit.
93. **Texel types without a mapping drop the format bound** (`codegen.rs:348-370`): an image of
    `half` or 8/16-bit integers accepts any format; a usage above 3 falls into storage.
94. **Entry-point `uniform` parameters and a second push constant buffer are ignored**
    *(plausible)* (`codegen.rs:583-648`), and aren't in `layout_hash`.
95. **Pass names come unchecked from file stems** (`codegen.rs:563-568`): `my-pass.slang` or
    `2d_blit.slang` make invalid identifiers, and one stem in two directories overwrites a `.spv`.
96. **Any struct named `Clock` gets the clock readout impl** (`codegen.rs:55, 180`), which doesn't
    compile without its `counters` field.
97. **The hot reload thread dies on any panic** (`hot_reload.rs:79-123`). The build matches lava's map by pass count only
    (`:58`); only `.slang` files trigger (`:157`), not `.slangh` includes; `reload_pass` returns
    `Ok` for a pass the device can't run, and the watcher logs a reload (`pipeline.rs:231-234`).

## 3. Performance

### Recording and pipelines
- **Pipeline creation.** Specialized and raster pipelines are still made synchronously on the
  recording thread at first use (outside the pass's map lock, through the `VkPipelineCache`).
  - There is no prewarm API.
  - Beyond `scratchpad.rs`: `FAIL_ON_PIPELINE_COMPILE_REQUIRED` for async compiles,
    `VK_KHR_pipeline_binary`, and deferred host operations for ray tracing pipelines.
- **Raster passes are coarse in barriers.** Every resource of a raster pass is accessed at
  `VERTEX|FRAGMENT` or `TASK|MESH|FRAGMENT` (`build.rs:30-41`).
- **Discarding accesses don't transition from `UNDEFINED`** (clears, full copies, attachments
  with `discard_before` that cover their whole mip), so drivers keep decompressing contents
  nothing reads. Core loads the G-buffer and the UI target every frame (`discard_before` exists
  now).
- **`VK_KHR_unified_image_layouts`** would remove the per-mip layout atomics, the
  GENERAL↔ATTACHMENT round trips (four per frame for the G-buffer), and the rule that command
  buffers sharing an image be recorded in order.

### Found 2026-10-18
- **Small per-call costs:** a stencil change makes 8 calls where front equals back
  (`:640-648`); `unsized_struct!` accessors walk every earlier section (`unsized_struct.rs:12-28`);
  `Stamp::is_done` polls every timeline even when cached values suffice (`lifetime.rs:14-18`).
- **`acquire` queries surface capabilities every frame** *(plausible)* (`present.rs:331-335`), an
  X11 round trip on Mesa.
- **`QueryPool::instant` calibrates on every call** (`query/mod.rs:252-277`) and ignores
  `max_deviation`.
- **`memory_report` builds the full allocator report** under the allocator lock
  (`memory.rs:32`) to read two totals, and queries the memory properties `PhysicalDevice` caches.
- **The engine's specialization constants change at runtime** (engine `render.rs:723-729, 793`):
  `BvhCull`'s buffer lengths double on overflow and `Composite`'s sky slot changes per sky, so
  each value compiles a pipeline mid-frame.

### Memory
- **Allocator settings are fixed.** Blocks start at 64 MiB of device memory and 16 MiB of
  host-visible memory and double to 256 and 64 MiB, and every allocation is named "buffer" or
  "ImageMemory" (`state.rs`).

### Other
- **The ANV `MEMORY_READ` workaround on trace reads applies to every vendor.** Gate it on
  `DriverId::INTEL_OPEN_SOURCE_MESA` (`acceleration_structure.rs:25-32`).
- **The debug callback works for every message.** It enables VERBOSE and INFO and formats each
  message before `tracing` filters it.
- **Shader builds.** Each pass has its own Slang session, compiled serially, and there is never
  debug info, so no source-level shader debugging. `Shaders::optimization` exists now; core
  doesn't call it.

---

## 4. API friction

2. **`push_bindings` is positional** (15 arguments for `BvhCull`), so swapping two
   `BufferSlice<u32>`s or two `Mat4`s compiles. Generate a struct with named fields, with
   specialization constants as fields too (that removes `IncompleteX`).
3. **`Config` lacks settings.** `lava::init(&Config)` has passes, display, validation,
   disabled features and a pipeline cache file.
   - **Missing settings:** app name and version, device choice (ranked by capability, user
     override, presentation support), extra extensions and a pNext hook, validation settings
     and queue counts.
   - **Validation:** it is silently ignored in release builds, and init fails without the SDK's
     layer.
   - **Direction:** `lava::init(Config) -> Result<DeviceInfo>`, with `DeviceInfo` reporting
     supported vs enabled features, limits, subgroup, mesh and ray tracing properties.
6. **Per-frame resources are a convention.** A `Timeline` owns its frame slots
   (`Submission::slot()`), but core builds per-frame rings four ways: instance offsets, gizmo
   `first`, UI buffers per slot, and the cull `variables` readback per slot. **Direction:**
   - `PerFrame<Buffer<T, Upload>>` whose `host_mut(&submission)` ties writes to the waited slot;
   - a growable stream buffer.
9. **Errors are mostly `Message(String)`.**
   - Out-of-date, minimised, surface lost, device lost, unsupported and invalid argument
     can't be matched.
   - Out-of-memory comes in two variants.
   - Recording panics on runtime failures such as pipeline creation (`must`).
   - **Direction:** typed `#[non_exhaustive]` variants, and the rule "panic for programmer
     errors, `Err` for device and environment".
   - *2026-10-18, against that rule:* host copies return `Err` for a wrong mip, region or data
     length while GPU copies panic (`image/mod.rs:674, 683-702`); `reserve_sampled(0)` panics; `init` returns `Result` but SPIR-V,
     module and pipeline failures panic through `must` (`pipeline.rs:157-225`); statistics
     without `mesh-shader-queries` return `Message`, not `MissingFeature` (`query/mod.rs:319-335`);
     `performance(0, ..)` takes the profiling lock for good before panicking (`:156, 384-402`).
   - `Display` prints the inner error and `source()` returns it, so reporters print it twice
     (`error.rs:30-66`). `OutOfBindlessSlots { set: &'static str }` is stringly typed.
     `Error::message` duplicates `From<String>`/`From<&str>`, which turn any string error into
     `Message` silently. `Error::Lock` and `From<PoisonError>` are dead (`error.rs:15, 117-121`).
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
14. **Scope names must be `&'static str`.**
15. **Resize and grow is hand-rolled.** `RenderResources::fit`, a `fresh` flag and
    `grow_overflowed` could become `Image::fit(extent)` and `Buffer::reserve(len)` that keep the
    bound slots.
16. **Ray tracing ergonomics.**
    - Scratch: users call `Buffer::raw(.., Some(scratch_alignment() as u32))`, sum the sizes by
      hand, and a batch that doesn't fit panics. Add a scratch type and a `build_all` that splits
      into batches.
    - Compaction is a six-step sequence; add a compactor. `compacted()` checks neither
      `allow_compaction` nor that the source was built, and copies its build scratch size
      (`acceleration_structure.rs:299-308`).
    - Every structure owns a `VkBuffer`. Allow placement in an arena slice.
    - Sizing a BLAS needs real buffers; add `Blas::with_capacity(desc)`.
    - `BuildFlags` can't say "no preference"; flags are raw `vk::GeometryInstanceFlagsKHR`;
      `VertexPosition` is open and exposes `vk::Format`, and has no impl for glam's `Vec3A`.
    - `Instance` has no getters, and `Describe`/`Described` are public.
    - Builds, updates, compaction and size queries open no scope or label, so observers and
      captures miss them (`:731-945`).
    - `every_structure_read` picks its access by comparing the stage with `BUILD` (`:60-61`).
    - `Instances::new`'s safety contract covers tracing but not the TLAS build, which reads every
      named BLAS, which must be built (`:701-705`).
17. **Counter selections must fit one pass,** and only the graphics family is counted.
    **Direction:** a multi-pass capture for tools (STATE.md). The engine re-implements the
    acceptance test (engine `profiler/mod.rs:244-247`); add `counters::fits`. `MAX_COUNTERS = 8`
    is lava's own cap and undocumented.
19. **The surface is messy.**
    - Deep paths (`lava::image::slice::ImageView`).
    - No prelude, and glam and bytemuck aren't re-exported.
    - Unused re-exports: `VkPipeline`, `VkShaderModule`, `ShaderHash`.
    - Public internals: `PassEntry`, `Functions`, `PhysicalDevice`, `CALLSITE`,
      `RaytracingPipeline`, `ShaderBindingTable`.
    - Typos: `handel`, `Filter::Liniear`, `extend` for extent.
    - Fix these before 0.1, since renames break users afterwards.
    - *2026-10-18, more public internals:* `pub use ash::vk::*` makes every ash upgrade a breaking
      lava release (`vk.rs:6`); `lava::vk` and `lava::state::raw_vulkan` are one module under two
      paths (`state.rs:35`; the engine and generated code use the second, lava the first);
      `pub mod error`, `lifetime` and `state` duplicate root re-exports; `CallsiteGuard`;
      `BindingOutput::sections()` returns an unnameable pub `Sections`; `RawImage`, `MipViews` and
      the sealed `Views` methods expose `vk::ImageView` and the state atomics; `format::bc_block`
      (internal use only), `Buffer::handle()` (no caller anywhere); `lava_build_slang::compile`
      and its re-export of shader-slang 0.1.0, which ties lava-build-slang's semver to slang-rs.
    - *2026-10-18, unused:* `usage::UnifiedBinding`, `Features::union`, `ImageView::cast`,
      `ImageSlice::cast`, `Image::aspect_check`/`texel_check`, `Dim::texels`,
      `SlotRange::is_empty` (always false), `Config::gpu_assisted_validation` (no user), the
      root re-exports `AccessFlags2`/`ImageLayout`, and three loaded draw commands in `vk.rs`
      (`cmd_draw_indexed_indirect2`, `..._count2`, `cmd_draw_mesh_tasks_indirect_count2`).
    - *2026-10-18, naming traps:* `ImageRef::view()` returns the `ImageView` while
      `ImageView::view()` returns the `vk::ImageView` (`kind.rs:18-22`); `features::gate` is a
      function and a module; `lava_build_slang::SLANG_DIR` (the shader directory) shares the
      name of the `SLANG_DIR` variable (the SDK); `BufferSlice::get_range` returns
      `std::range::Range` where the rest uses `std::ops::Range`; `RasterMesh.amp` where
      lava says task; `Features::DEFAULT` means none; `copy_region_from` takes its offset as a
      `D::Extent`; `image_acceses`, `buffer_acceses`.
    - *2026-10-18, missing derives:* `RasterState` and `BufferAccess` could be `Copy`; `Filter` has
      no `Clone`/`Copy`/`Debug` and no cubic filter; `ImageSlice` no `Debug`; `Scissor` and
      `Viewport` are identical structs; `Config` has no docs or `Debug`.
20. **Codegen errors lack context.** Panics give no file or field (`build.rs:111,118,236`), the
    first failing pass hides the rest, and Slang warnings aren't forwarded.
21. **Generated code leaks into its includer** (2026-10-18, `codegen.rs:511-535`). It glob-imports
    glam, `lava::command_buffer`, `pipeline`, `format` and `slice`, and imports `vk`, all pasted
    into the user's module by `include_passes!`. Core relies on the leak (`Zeroable`, `Vec2`,
    `ImageHandle` in its `bindings.rs`), and a Slang struct named `Bindings`, `Draw` or
    `PassEntry` breaks the build. Emit `::lava::` paths. The generated functions also lack
    `#[track_caller]`, so binding panics point into `$OUT_DIR` (`codegen.rs:932, 982`).
22. **Only push-constant structs get `_pad` fields** (2026-10-18, `codegen.rs:171-178, 1032-1035`).
    Structs behind a `Buf` with a gap fail `derive(Pod)` or the offset assert in the user's crate,
    so the engine's shaders pad by hand (`datatypes.slang`, `raster_ui.slang`) and core sets `pad`.
    A pass without push constants panics instead of getting an empty struct (`:583-588`); `i64`,
    `f64` and `bool` vectors panic; arrays above 32 need bytemuck's `min_const_generics`.
23. **`ImageHandle` is `Zeroable` but its default is null** (2026-10-18): `X::zeroed()` of a
    generated struct names sampled slot 0.
24. **`bind` zips two iterators** (2026-10-18, `bindless.rs:308-345`): a length mismatch binds less
    silently, and core collects a `Vec` per frame to line them up. Take pairs and assert.
25. **Presenting takes raw Vulkan** (2026-10-18, `present.rs:170-174, 269-281`):
    `PresentSurface::new` takes `vk::SurfaceFormatKHR`s and silently falls back to any format,
    and present modes are a `usize` into a per-device list with lava-owned strings, so a saved
    setting means another mode elsewhere. A `PresentMode` enum with `supports(mode)` fits.
27. **Buffer creation edges** (2026-10-18, `buffer/mod.rs:107-146, 205-207`): a zero-sized `T`
    makes `len()` divide by zero; a size that isn't a whole number of elements keeps tail bytes in
    `range(..)` but not in `range(..len())`; `alignment: Option<u32>` isn't checked for a power of
    two and is narrower than Vulkan's `u64`. `update_buffer` writes one element only
    (`slice.rs:217-222`).
28. **Small API gaps** (2026-10-18): `DepthTarget` needs a depth aspect, so `S8Uint` can't be an
    attachment (`kind.rs:51-52`); `check_hazards` forbids two draws of a rendering writing one
    buffer, ruling out atomics across draws (`mod.rs:263-293`); `depth_clamp_range`'s wrapper
    takes a reference where Vulkan's is optional (`vk.rs:466-475`);
    `dynamic-line-mode` implies no capability (`Cargo.toml:179`); `#[validation_trace(anything)]`
    compiles and the non-function error has no span (`lava-macros/src/lib.rs:10`); `Shaders` isn't
    `#[must_use]`; device ranking lives inline in `Ctx::init` and can't be unit-tested.
29. **Validation events have undocumented fields** (2026-10-18, `state.rs:566-580`). The engine's
    console reads `typ` and `flags` as integers, but lava sends `typ` as a `Debug` string and no
    `flags`, so its validation branch is dead (engine `editor/console.rs:194-203`).

---

## 5. Missing features

- **Raster:**
  - **Sampling:** MSAA with resolve attachments, sample shading, alpha-to-coverage.
  - **Rendering:** multiview, layered rendering, a render area offset, per-draw scissor and
    viewport (UI clipping), vertex-only passes (depth and shadow), dynamic rendering local read.
  - **Draws:** `draw_indexed_indirect(_count)` and `launch_indirect_count` (their loaders exist), u16 indices,
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
- **Slang coverage:**
  - **Types:** `half` and 16-bit ints (panic today, though float16 is enabled), 64-bit scalars
    and `bool` vectors, non-square matrices (no glam type), `float3x3` layout vs `Mat3`
    *(plausible)*.
  - **Image ops:** `Gather`, `SampleGrad`, `SampleBias`, `Load` at a mip, image
    atomics. `MutTex2D` has only `Load`, no `Sample`; there is no `MutTex3D`.
  - **Codegen:** Rust keywords as field names, arrays over 32 (bytemuck's `Pod` limit), pass and
    struct name clashes.
- **Memory:** per-frame transient rings (only `cmd.upload`'s internal ring exists), object and
  allocation names.
- **Queues:** several command buffers per submission, parallel recording (per-thread trackers
  merged by a barrier prologue at submit).
  - `Wait` has `block()` and `is_done()`, no timeouts or multi-wait.
- **Swapchain:** HDR (`VK_EXT_hdr_metadata`), image count, composite alpha, `FIFO_LATEST_READY`,
  present_id/present_wait for pacing, presenting several swapchains in one call, discard on
  acquire.
- **Queries:** conditional rendering.
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
  - **Formats:** half and snorm vertex positions.
- **Hot reload:** per-pass include sets from Slang's module dependencies, for reloads and for
  exact `rerun-if-changed` (which would also catch imports outside the given directories).
- **Lifecycle:** `lava::shutdown()` that waits, drains epochs, destroys everything and allows
  re-init. Today `destroy()` destroys the bindless objects and runs the retired destructors,
  leaves lava unusable (bug 24), and nothing calls it.

---

## 6. Cleanup (2026-10-18)

Small things, by file; none changes behaviour.
- **Clippy** (default lints) flags `is_multiple_of` (13 places), `From` over `Into`
  (`buffer/mod.rs:80`, `pipeline.rs:414`), redundant field names, needless borrows, `map().flatten()`,
  `!= ptr::null()`, `clone()` on `Location`, a single-range `vec!` (`bindless.rs:538`, intended),
  a useless `format!` (`codegen.rs:511`), missing `# Safety` docs (`vk.rs`) and non-const
  thread-locals. Only one warning reaches `cargo check`: the deprecated `fetch_update`
  (`image_state.rs:200`), whose `unwrap_or_else` is dead. `cargo fmt --check` reports diffs across lava and lava-tests.
- **Manifests:** lava's `tracing-subscriber` and lava-tests' `smallvec` dev-dependencies are
  unused; glam's `mint` feature is used by nothing in the workspace; lava-build-slang's
  dev-dependency on lava has no version, so its packaged test won't compile; `quote`/`syn` aren't
  workspace dependencies; `lava/build.rs` has no `rerun-if-changed`, so it reruns on any change,
  and its bake table repeats knowledge of `Baked`, `Dynamic::of` and the feature implications.
- **command_buffer:** the `is_read || is_write` test is loop-invariant and
  `let buffer_access = &buffer_access;` is redundant (`mod.rs:1495-1552`); the "access of no stage
  (`present`)" fallback comment is stale and its `src_access.is_empty()` branch never fires
  (`:1641-1650`); the `index < count` assert is repeated in three places instead of
  `Recording::begin_query`, and `end_query` bounds `QueryKind` where `begin_query` bounds
  `Scoped`; `view_mask(0)` is the default. `pending.rs`: the transitions loop in `made_visible`
  is redundant *(plausible)* (`:102-123`) and the access sets could be `const`s (`:235-281`).
  `image_state.rs`: `for_each` ignores `mips.end` without an assert (`:72-79`),
  `run_mips.end == mips.start` is always true (`:119`), a panic prints `Some(Compute)`
  (`:153-158`). `binding_output.rs`: runtime asserts of compile-time sizes (`:77-80, 112-114`).
- **raster_state and pipeline:** `DepthClamp` has two `dead_code` allows (`raster_state.rs:98-104`);
  `split()` copies 21 fields by hand to make a `RasterState<0>` (`:324-351`); the `[..n]` arm of
  `changed!` and `Rendering.color_attachments` repeat known lengths, and `blends_changed` is
  computed when blends aren't dynamic (`:529-535, 658, 668`); `gated_setters!` entries repeat
  `(state, ..)`. In `pipeline.rs`: `create_raster_pipeline` builds `create_info` in two steps,
  sets defaults and passes vertex states to mesh pipelines.
- **features and vk:** `has()` guards on bits of structs only linked when present are redundant
  (`features.rs:693-780`); `#[allow(unused_mut)]`
  looks dead (`:459`); `unlink!` guards a reuse no path makes, while `query` returns `p_next`s into
  its old stack (`:592-618`); missing feature docs on all but the first field of each group
  (`:328-353`). `vk.rs`: `DYNAMIC_STATE_DEPTH_CLAMP_RANGE_EXT` is defined in `values` and
  re-exported (`:61-65`); `with_regions!` repeats `tagged!` (`:91-258`); the function pointer size
  check could be `const` (`:270`).
- **state and init:** `get_physical_device_properties` duplicates `properties2`
  (`state.rs:231`); `let _ = (&rt_pipeline_properties, &acc_properties);` (`:275`);
  `pick_queue_families` runs twice per device (`:290, 340`); `sort_unstable` plus `dedup_by_key`
  of disjoint families removes nothing (`:387-390`); `CStr::from_ptr` where ash has `*_as_c_str()`,
  and one non-UTF-8 device name aborts init for every device (`:231-252`); the empty `Debug`
  impls of `Ctx` and `Functions` (`:37-41, 621-625`); `Ctx::allocator` panics on poison where
  every other lock recovers (`:103-108`), as do `LOCK_USERS` (`counters.rs:148`,
  `query/mod.rs:385`); the `STATE.set` error path is unreachable (`:497-506`); the printf
  detection splits into a `Vec` and drops separators (`:552-564`). `physical_device.rs`: unused
  `PartialEq for QueueFamily`, `num_queues` duplicating `handel.queue_count`, and init-only data
  kept for the program's lifetime.
- **images:** `from_raw(DEPTH.as_raw() | STENCIL.as_raw())` and casts of values already `u32`
  (`format.rs:187-210`); `PerMip` with one mip makes duplicate views (`image/mod.rs:461-464`); imports at the bottom of
  `image/mod.rs` under two cfg combinations (`:638-648`); `Dim::Offset` is signed though blit boxes must now lie inside
  the mip (`dim.rs:11`); `region_fits` duplicates `check_box` (`dim.rs:66-104`); `Copy + Pod`;
  `subresource_layers` takes a parameter that is always `mip_range.start`; `check_copy_size`
  lacks `#[track_caller]`.
- **buffers:** `iter::empty()` and `std::iter::empty()`
  mixed, `BufferAccess` built by hand beside `slice.access(..)`, `.cloned()` for `.copied()`
  (`slice.rs:192-278`); `use crate::vk::{self}`.
- **queues, present, queries:** `begin` polls its index three times (`timeline.rs:245-250`); the
  replay lacks `ONE_TIME_SUBMIT` (`:406-409`); `MAX_TIMELINES` also caps queues per family
  (`queue/mod.rs:10-11`); `Current` is `pub(crate)` but module-local (`present.rs:138-145`); a
  module-wide `#[allow(unused_imports)]` and `let _ = locked;` (`query/mod.rs:8, 459`);
  `CStr::from_ptr` over fixed arrays (`counters.rs:64-68`); `valid_bits` is 0 until a timestamp
  is written, so `elapsed_ns` returns 0, and is last-writer-wins across families
  (`query/mod.rs:51-54, 246-250`).
- **ray tracing:** the scratch offset is computed in two loops and the infos re-collected
  (`acceleration_structure.rs:813-889`); `with_geometries` checks the gate twice and `Drop` uses
  the checked loader, leaking on `Err` (`:214-361`); `layout.size.max(1)` is redundant
  (`rt_pipeline.rs:79`); the generated code calls `trace_accesses` twice per `Accel`
  (`codegen.rs:424-427`).
- **lava-build-slang:** `compile()` is one 570-line function (`codegen.rs:488-1057`);
  `trim_end_matches` where `strip_suffix` and `file_stem` fit (`:565, 624`); the capability scan
  duplicates `Features::of_spirv` (`:886-906`); stages are strings (`compile.rs:103-128`).
- **lava-tests:** `let _ = needs;` is dead (`tests.rs:194`); `gpu_missing_features` copies
  gpu_headless's render helpers verbatim instead of sharing them in `common`; misindented code
  and a comment inside an expression (`gpu_headless.rs:680, 2339`).

## 7. Documentation (2026-10-18)

CLAUDE.md, the `//!` line of `command_buffer/mod.rs` and STATE.md's `wait_idle` entry were fixed
with this survey. Still wrong:
- **README:** "a draw's raster state is dynamic state, so pipelines depend only on the shaders,
  their constants and the attachment formats" (`README.md:10`) is stale since extended dynamic
  state became optional.
- **lava-build-slang README:** it doesn't say the application needs glam and bytemuck with
  `derive`, or that hot reload needs lava-build-slang as a normal dependency too. Pass discovery
  isn't recursive while `rerun-if-changed` is. `build.rs`'s comment implies `-lslang` is avoided,
  but shader-slang-sys still emits it *(plausible)*.
- **rustdoc:** `features::gate` is ambiguous between function and module (`features.rs:93`); the
  link to `watch` is unresolved (`lava-build-slang/src/lib.rs:31`); the links to `Instance`
  (`lib.rs:24-26`) and `Compute` (`queue/mod.rs:35-39`) break without their features.
- **Module docs:** `tests.rs`'s omits the `unsized_struct!` test; `lifetime.rs`'s "in drop order" holds for one
  reclaimer only.
- **Scope observers:** `observer.rs` and CLAUDE.md say what `on_enter` records comes before a
  pass's barriers, but `compute`, `raytrace` and renderings flush the barriers before
  `rec.enter` (`command_buffer/mod.rs:640-696, 929-939, 994-1002`), so scope timestamps and debug
  labels exclude them. Change the code or the docs, whichever is meant.
- **Item docs:** the 16-pipeline warning always blames constants, though raster keys grow too (`:499-507`); the `update` doc of acceleration structures
  says primitive counts where the whole shape must match (`:743-745`); `wait_idle`'s doc omits that
  it reclaims; `Config::validation`'s omits that it enables printf; `ray_tracing.slang`'s
  `__init` truncates out-of-range fields silently, where Rust asserts; `SCONST` splits on
  commas, so generic images need an alias; `Img.Load` reads mip 0 and a null index is undefined;
  `memory_report`, `blocks` and several `Features` items have no docs; `bindless.slang`'s `Accel`
  doc names a missing `BindingOutput::acceleration_structure_read`.
- **Tests:** `ray_scene.slang:2` and `test_write_instances.slang:2` name `gpu_headless.rs`, but the
  scene lives in `gpu_ray_tracing.rs`; `gpu_missing_features.rs:125` swaps near and far.
- **Generated context:** the tools generator lists `pub fn` lines inside codegen's raw-string
  templates as lava-build-slang API, and the test copy of `CallsiteGuard` as lava-macros API. Fix
  in `../GameEngine/tools/src/main.rs`: skip string literals and `tests/`.

## 8. Tests (2026-10-18)

### Tests that can't fail
- `skipped_attachment_keeps_its_content` (`gpu_headless.rs:1907-1940`): `test_raster.slang`
  writes only location 0, so `Blend::Skip` on the second attachment is never exercised.
- `extra_accesses_get_barriers_like_the_bindings` (`:2766-2788`): the shader never touches the
  buffer and nothing is asserted. Sync validation doesn't track buffer device addresses, so the
  "validation reports the missing barrier" premises at `:999-1001` and `:1523-1525` hold only as
  data checks *(plausible)*.
- The slot test's "taken until no frame reaches it" (`:1147`) passes because `bind()` doesn't poll.
- The "submit in parallel" test (`:200-221`) checks results, not concurrency.
- Bare `catch_unwind(..).is_err()` without the message passes on any panic (`:441-453, 1094-1103,
  2095-2098`, `gpu_ray_tracing.rs:428-433`, `gpu_window.rs:159-168`); `panic_message` exists.
- The `instances_have_the_layout_of_the_vulkan_struct` unit test compares sizes, not offsets.
- The capability unit test checks only the bits it sets (`tests.rs:181-195`).

### Tests that fail or skip wrongly
- The `!host_image_copy` branches at `gpu_headless.rs:1305-1316, 1340-1350` can't run: the image
  creation before them unwraps the `MissingFeature`. 
- `volumes_are_uploaded_sampled_and_copied` (`:2489-2494`) fails without host image copy, though
  only its last part needs it; `:2555-2568` creates a BC7 image without checking `bc_compression`
  and returns silently.
- The clock test checks `shader_clock` but not `shader_int64_atomics`, and prints `skipping`,
  not `SKIPPED` (`:1495-1504`); `:692-694` and `:447-453` skip silently.
- `gpu_missing_features.rs:215, 229` use `polygon_mode(LINE)` without checking
  `fill_mode_non_solid`.
- `gpu_ray_tracing.rs:505-515` multiplies `maxComputeWorkGroupCount` by the size, which exceeds
  `u32` on most drivers, so it nearly always skips.
- `gpu_window.rs:124-128` assumes the swapchain can be smaller than the window, which fails on
  X11 *(plausible)*; each step prints `ok` before validation errors are checked.
- The timestamp order `times[3].0 >= times[2].1` (`gpu_headless.rs:599`) isn't guaranteed by the
  spec *(plausible flaky)*.

### Harness
- `cargo test --release` runs every GPU test with validation off (validation needs
  `debug_assertions`), and nothing asserts the layer is active.
- The subscriber has only the capture layer, so lava's own warnings and errors are dropped and
  failing tests show no logs. Only ERROR and the `message` field are kept, dropping the recorded
  Rust callsite.
- `gpu()` discards errors gathered before its lock, including those of `lava::init`, and an init
  panic poisons the `LazyLock`, so every later test fails with "poisoned".
- `LAVA_BLESS=0` blesses too (`golden.rs`); goldens shared by two tests or binaries are rewritten
  by whichever runs last and race between binaries.
- The recording bench times only the closure (`benches/recording.rs:185-191`): the end barrier,
  pending merge, `vkEndCommandBuffer` and the submit are outside, where STATE's unexplained 7%
  may live. Its buffers are uninitialised, `begin()` throttles on GPU time, and scenarios
  run one after another, not interleaved.

### Coverage gaps
Each was checked by grep over lava-tests and `lava/src/tests.rs`.
- **Barriers:** two timelines of one family sharing an image; `forget` on destruction; the
  `ATTACHMENT_OPTIMAL` vs `GENERAL` choice (only through the validation layer); `make_visible` merging; the swapchain's acquire wait stage; `take_host_layouts`.
- **Runtime-typed resources:** `format::Dynamic`, `usage::Dynamic`, `with_params`, the
  `aspect_check`/`texel_check`/`usage_check` panics,
  the clear-value assert and `cast_check`. Pure parts (`expected_data_len`, `mip_extent`,
  `region_fits`, `cast_check`, the format table's sizes) could be unit tests.
- **Images:** `DepthSplitMips`, 64-bit and unusual clears, linear blits on formats without linear
  filtering.
- **Draws:** `Draw::indirect` with several commands (bug 38); attachments reached through accesses;
  a draw reading an attachment of its own rendering.
- **Raster state:** depth bias, clamp and clip, conservative raster, provoking vertex, line
  width/mode/stipple, depth bounds, blend constants, any topology but `LINE_LIST`. `RasterKeyRef`
  hashing like `RasterKey` and `Baked::of` are pure and could be unit tests.
- **Pipelines:** raster or ray tracing passes with specialization constants, mesh passes with a
  task shader, a `[MaxRecursion(n)]` above the device's limit (no device here has less than 31,
  and lava-build-slang refuses more), `reload_pass` of a ray tracing pass.
- **Slang shapes:** `Image2D`, handles or `Accel` inside structs behind a
  `Buf`; struct merging, padding and naming in codegen; the hand-copied constants
  (`MAX_PUSH_CONSTANTS_SIZE`, the clock constants) against lava and `profile.slang`.
- **Bindless:** `OutOfBindlessSlots`, `Ranges::give`'s double-free panic.
- **Features:** `Features::consistent`, `without`/`missing_from`, `extensions()` agreeing with
  `optional()`, the cargo names in `SPIRV_CAPABILITIES`; the disabled ray tracing and
  performance paths in `gpu_missing_features`.
- **Lifetimes and timelines:** `latest_stamp`, `Stamp::is_done`, slot reuse, a timeline dropped
  after `reserve`, two `wait_for`s on one timeline, stages other than `Transfer`.
- **Queries:** the pool-reuse panic, `QueryRecorder` misuse, `instant()` on a device that has
  it, `memory_report` with the budget.
- **Commands and families:** `flush_all`; transfer-family alignment and granularity;
  acceleration structure transforms, `Instance::flags`, building from a different geometry.
- **Presenting:** dropping a surface with a current image, the format fallback.
- **Host access:** `host_ptr`; `Shared` memory where the device lacks `rebar` (only its error).
- **Validation:** GPU-assisted validation is never on, so bindless indices, layouts and buffer
  address bounds go unchecked. `unsized_struct!`'s test needs no GPU and could run under Miri
  once.

## 9. Seen from the engine (2026-10-18)

Engine-side findings that don't belong to lava's code but came up in this survey:
- A minimised frame drops its slot's `ScopeQueries`, and `GpuSlot::prepare` doesn't rebuild them
  until a switch changes (`core/src/render/render.rs:441-463`).
- `[build-dependencies]` `walkdir`, `serde` and `serde_json` are unused, and `bytemuck_derive` is
  redundant (`core/Cargo.toml:66-70`).
- `dynamic-blend` lets no draws share a pipeline, since every pass has one blend set
  *(plausible)*.
- The swapchain and viewport target carry `TransferDst` without use (`render/render.rs:122`).
- `!Ctx::features().shader_clock` can't be true under `profiling` (`profiler/clock.rs:22`).
- Validation is derived from `debug_assertions` in two styles (`render/headless.rs:40`,
  `render/mod.rs:263-268`).
- Unused imports of lava internals (`render/world.rs:21-42`), and a CPU mip size computed through
  `<D2 as Dim>::mip_extent` (`assets/texture.rs:420`).
- Engine docs say `core/build.rs` calls `.optimization(..)`; it doesn't.

---

## Suggested order

1. **Fix the soundness bugs:** host access (16), the undefined behaviour from safe code (24-37),
   and the hot reload's layout hash (85).
2. **Fix the barrier tracking (22)** and the valid-usage bugs every device hits: multi-draw
   indirect (38), the depth bias clamp (39), line topologies without extended dynamic state (40),
   depth clamp (41).
3. **Fix the timeline and presenting hangs and leaks (64-84).**
4. **Redesign the APIs users touch first:** named bindings, init config, memory locations,
   frame and ring helpers, errors, naming, and the public internals of 19 before 0.1.
5. **Make the tests able to fail** (section 8), then add features by real-world demand: arrays,
   cubes and MSAA, samplers, staging.

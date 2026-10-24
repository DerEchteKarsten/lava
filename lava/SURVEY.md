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

5. **Raster and draws:** dynamic rendering local read (skipped 2026-10-21, see section 5:
   input attachments don't fit lava's descriptor heaps and Slang yet).
9. **Surface, cleanup and docs:** 4.14, 4.19 (shallow re-exports, `lava::prelude` (glam and
   bytemuck are re-exported since 2026-10-24), internals hidden or deleted, every rename, ash
   as a documented public dependency with only `lava::vk`, the derives), all of section 6
   (bake table stays) and
   section 7 (scope observer docs follow the code: scopes start after their barriers).
10. **Tests:** all of section 8, plus tests for every change above, one GPU-assisted validation
    run, and Miri for `unsized_struct!`; the tools generator fix of section 7.

The engine follows in its own repository, only after the other agent's uncommitted engine work
is committed: 84, the push constants of 3, 4.11's viewport usage, 4.15, 4.17's `fits`, the
generated API of 2026-10-24 (argument structs, extra accesses in `push_bindings`, no imports
into `bindings.rs`, generated padding and `Default`s instead of the shaders' hand padding,
`WORKGROUP_SIZE`/`groups`, `[RustConst]` instead of copied constants), release builds without
validation, and all of section 9 (drop `dynamic-blend` once checked).

## 2. Bugs

### Undefined behaviour from safe code (2026-10-18)
31. **`compacted_from` stores the source's raw handle** *(plausible)* (`:181, :306, :961`): a
    handle value reused after the source's destruction passes `compact`'s check. Use an id.

### Presenting (2026-10-18)
84. **The engine's `Presenter::drop` waits before the surface drops** (engine
    `render/render.rs:139-147`): `PresentSurface::drop` only retires its destruction, so surface
    and swapchain outlive the window unless something reclaims later.

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
- **The engine's specialization constants change at runtime** (engine `render.rs:723-729, 793`):
  `BvhCull`'s buffer lengths double on overflow and `Composite`'s sky slot changes per sky, so
  each value compiles a pipeline mid-frame.

---

## 4. API friction

3. **`Config` lacks settings.** `lava::init(&Config)` has passes, display, app name and
   version, validation, disabled features, a pipeline cache file and allocator block sizes.
   - **Missing settings:** device choice (ranked by capability, user override, presentation
     support), extra extensions and a pNext hook, validation settings and queue counts.
   - **Direction:** `lava::init(Config) -> Result<DeviceInfo>`, with `DeviceInfo` reporting
     supported vs enabled features, limits, subgroup, mesh and ray tracing properties.
6. **Per-frame resources:** core builds per-frame rings four ways (instance offsets, gizmo
   `first`, UI buffers per slot, the cull `variables` readback per slot); `queue::PerFrame<T>`
   exists since 2026-10-24 for the per-slot ones. Missing: a growable stream buffer.
10. **Transfers take one region per call,** and there is no `generate_mips` (`ImageSlice::mip`
    reaches any mip of a `Single` image since 2026-10-21).
11. **Mixed image types are awkward.**
    - `bind` takes one `F/U/D` for all images.
    - There is no `ImageView::erase()`.
    - The viewport target is typed like the swapchain image, so it lacks SAMPLED, and the UI
      reads it through a storage load, which is several times slower on Intel.
13. **Field projection.** Core slices a struct field out of a buffer with
    `byte_range(offset_of!..).cast::<MeshletDraw>()`. (Indirect draws and dispatches take any
    `T: StartsWith<C>` since 2026-10-21, which removes the 16→12-byte cast.)
14. **Scope names must be `&'static str`.**
15. **Resize and grow is hand-rolled in core.** `grow_overflowed` can use `buffer.reserve(len, cmd)`
    (since 2026-10-24) and `RenderResources::fit` with its `fresh` flag `Image::fit(extent)`,
    which keeps the bound slots (since 2026-10-21).
19. **The surface is messy.**
    - Deep paths (`lava::image::slice::ImageView`).
    - No prelude, and glam and bytemuck aren't re-exported.
    - Unused re-exports: `VkPipeline`, `VkShaderModule`, `ShaderHash`.
    - Public internals: `PassEntry`, `Functions`, `PhysicalDevice`, `CALLSITE`,
      `RaytracingPipeline`, `ShaderBindingTable`.
    - Typos: `handel`, `extend` for extent.
    - Fix these before 0.1, since renames break users afterwards.
    - *2026-10-18, more public internals:* `pub use ash::vk::*` makes every ash upgrade a breaking
      lava release (`vk.rs:6`); `lava::vk` and `lava::state::raw_vulkan` are one module under two
      paths (`state.rs:35`; the engine and generated code use the second, lava the first);
      `pub mod error`, `lifetime` and `state` duplicate root re-exports; `CallsiteGuard`;
      `BindingOutput::sections()` returns an unnameable pub `Sections`; `RawImage`, `MipViews` and
      the sealed `Views` methods expose `vk::ImageView` and the state atomics; `Buffer::handle()` (no caller anywhere); `lava_build_slang::compile`
      and its re-export of shader-slang 0.1.0, which ties lava-build-slang's semver to slang-rs.
    - *2026-10-18, unused:* `Features::union`, `ImageView::cast`,
      `ImageSlice::cast`, `Image::aspect_check`/`texel_check`, `Dim::texels`,
      `SlotRange::is_empty` (always false), `Config::gpu_assisted_validation` (no user), the
      root re-exports `AccessFlags2`/`ImageLayout`.
    - *2026-10-18, naming traps:* `ImageRef::view()` returns the `ImageView` while
      `ImageView::view()` returns the `vk::ImageView` (`kind.rs:18-22`); `features::gate` is a
      function and a module; `lava_build_slang::SLANG_DIR` (the shader directory) shares the
      name of the `SLANG_DIR` variable (the SDK); `BufferSlice::get_range` returns
      `std::range::Range` where the rest uses `std::ops::Range`; `RasterMesh.amp` where
      lava says task; `Features::DEFAULT` means none; `copy_region_from` takes its offset as a
      `D::Extent`; `image_acceses`, `buffer_acceses`.
    - *2026-10-18, missing derives:* `RasterState` and `BufferAccess` could be `Copy`; `Filter` has
      no cubic filter; `Config` has no docs or `Debug`.
24. **`bind` zips two iterators** (2026-10-18, `bindless.rs:308-345`): a length mismatch binds less
    silently, and core collects a `Vec` per frame to line them up. Take pairs and assert.
28. **Small API gaps** (2026-10-18): `depth_clamp_range`'s wrapper
    takes a reference where Vulkan's is optional (`vk.rs:466-475`);
    `dynamic-line-mode` implies no capability (`Cargo.toml:179`).

---

## 5. Missing features

- **Raster:**
  - **Sampling:** MSAA with resolve attachments, sample shading, alpha-to-coverage.
  - **Rendering:** dynamic rendering local read (core in 1.4; the Arc B390 has it). Skipped
    2026-10-21: an input attachment is a heap descriptor of type `INPUT_ATTACHMENT`, which
    needs an `InputAttachment` usage flag (a new `Usage` parameter) and a bind for it; Slang
    decorates `InputAttachmentIndex` only on global parameters, which lava-build-slang rejects,
    and a heap-loaded `SubpassInput` (it compiles, spirv-val accepts it) is undecorated, which
    the default mapping gives the depth-stencil attachment, never a colour one, unless
    `vkCmdSetRenderingInputAttachmentIndices` maps them; draws that read what earlier ones of
    the rendering wrote also need a by-region barrier inside it (`FRAGMENT_SHADER` from
    `COLOR_ATTACHMENT_OUTPUT`, the attachment in `RENDERING_LOCAL_READ`), which `record` would
    have to insert between them.
- **Images:**
  - **Types:** cube maps (arrays of 6 layers exist since 2026-10-21; a cube view and
    `TexCube` don't), 1D, MSAA.
  - **Views:** swizzles, per-aspect depth/stencil views for shaders.
  - **Transfers:** mip generation.
  - **Memory:** transient (lazily allocated) attachments, aliasing in shared memory blocks.
- **Memory:** per-frame transient rings (only `cmd.upload`'s internal ring exists).
- **Queues:** several command buffers per submission, parallel recording (per-thread trackers
  merged by a barrier prologue at submit).
- **Swapchain:** image count, present_id/present_wait for pacing.
- **Queries:** conditional rendering.
- **Debug:** labels in release builds whenever the extension exists; `insert_label`; object
  names on views, query pools and timelines.
- **Escape hatches:**
  - an `unsafe` `lava::raw` with device, instance and queue;
  - `CommandBuffer::raw(declared_accesses, |vk_cmd| ..)` that flushes barriers for what it
    declares;
  - an access declaration for memory reached only through pointers;
  - runtime SPIR-V passes for vendor libraries, material graphs and mods.
- **Ray tracing:**
  - **Pipelines:** deferred host operations, pipeline libraries,
    `vkCmdSetRayTracingPipelineStackSizeKHR`.
  - **Structures:** opacity micromaps; position fetch in ray queries
    (`RayQueryPositionFetchKHR` is mapped, but no test reads positions from a ray query).
  - **Shaders:** `NO_DUPLICATE_ANY_HIT_INVOCATION`.
- **Hot reload:** per-pass include sets from Slang's module dependencies, for reloads and for
  exact `rerun-if-changed` (which would also catch imports outside the given directories).

---

## 6. Cleanup (2026-10-18)

Small things, by file; none changes behaviour.
- **Clippy** (default lints) flags `is_multiple_of` (13 places), `From` over `Into`
  (`buffer/mod.rs:80`, `pipeline.rs:414`), redundant field names, needless borrows, `map().flatten()`,
  `!= ptr::null()`, `clone()` on `Location`, a single-range `vec!` (`bindless.rs:538`, intended), missing `# Safety` docs (`vk.rs`) and non-const
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
  `Scoped`. `pending.rs`: the transitions loop in `made_visible`
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
  (`describe_device`); `let _ = (&rt_pipeline_properties, &acc_properties);`;
  `pick_queue_families` runs twice per device (`describe_device` and the selection loop);
  `sort_unstable` plus `dedup_by_key` of disjoint families removes nothing.
  `physical_device.rs`: unused `PartialEq for QueueFamily`, `num_queues` duplicating
  `handel.queue_count`, and init-only data kept for the program's lifetime.
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
  (`record_builds`); `with_geometries` checks the gate twice (and `create` again); `layout.size.max(1)` is redundant
  (`rt_pipeline.rs:79`).
- **lava-build-slang:** the capability scan duplicates `Features::of_spirv`
  (`features::capabilities`); stages are strings (`compile::stage_name`); `Shaders::compile` is
  still some 200 lines.
- **lava-tests:** `let _ = needs;` is dead (`tests.rs:194`); `gpu_missing_features` copies
  gpu_headless's render helpers verbatim instead of sharing them in `common`; misindented code
  and a comment inside an expression (`gpu_headless.rs:680, 2339`).

## 7. Documentation (2026-10-18)

CLAUDE.md, the `//!` line of `command_buffer/mod.rs` and STATE.md's `wait_idle` entry were fixed
with this survey. Still wrong:
- **README:** "a draw's raster state is dynamic state, so pipelines depend only on the shaders,
  their constants and the attachment formats" (`README.md:10`) is stale since extended dynamic
  state became optional.
- **lava-build-slang:** `build.rs`'s comment implies `-lslang` is avoided, but shader-slang-sys
  still emits it *(plausible)*.
- **rustdoc:** `features::gate` is ambiguous between function and module (`features.rs:93`); the
  link to `watch` is unresolved (`lava-build-slang/src/lib.rs:31`); the links to `Instance`
  (`lib.rs:24-26`) and `Compute` (`queue/mod.rs:35-39`) break without their features.
- **Module docs:** `tests.rs`'s omits the `unsized_struct!` test; `lifetime.rs`'s "in drop order" holds for one
  reclaimer only.
- **Scope observers:** `observer.rs` and CLAUDE.md say what `on_enter` records comes before a
  pass's barriers, but `compute`, `raytrace` and renderings flush the barriers before
  `rec.enter` (`command_buffer/mod.rs:640-696, 929-939, 994-1002`), so scope timestamps and debug
  labels exclude them. Change the code or the docs, whichever is meant.
- **Item docs:** the 16-pipeline warning always blames constants, though raster keys grow too (`:499-507`); `wait_idle`'s doc omits that
  it reclaims; `Config::validation`'s omits that it enables printf; `ray_tracing.slang`'s
  `__init` truncates out-of-range fields silently, where Rust asserts; `Img.Load(p)` reads mip 0
  and a null index is undefined; `memory_report`, `blocks` and several `Features` items have no
  docs.
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
  `ATTACHMENT_OPTIMAL` vs `GENERAL` choice (only through the validation layer); `make_visible` merging; the swapchain's acquire wait stage; `host_layouts`.
- **Runtime-typed resources:** `format::Dynamic`, `usage::Dynamic`, `with_params`, the
  `aspect_check`/`texel_check`/`usage_check` panics,
  the clear-value assert and `cast_check`. Pure parts (`mip_extent`, `region_fits`,
  `cast_check`, the format table's sizes) could be unit tests (`format::block`, `rows_len`,
  `pack_64` and `format::class` are).
- **Images:** `DepthSplitMips`, unusual clears, linear blits on float formats without linear
  filtering.
- **Draws:** `Draw::indirect` with several commands (bug 38); attachments reached through accesses;
  a draw reading an attachment of its own rendering.
- **Raster state:** depth bias, clamp and clip, conservative raster, provoking vertex, line
  width/mode/stipple, depth bounds, blend constants, any topology but `LINE_LIST`. `RasterKeyRef`
  hashing like `RasterKey` and `Baked::of` are pure and could be unit tests.
- **Pipelines:** raster or ray tracing passes with specialization constants, mesh passes with a
  task shader, a `[MaxRecursion(n)]` above the device's limit (no device here has less than 31,
  and lava-build-slang refuses more), `reload_pass` of a ray tracing pass.
- **Slang shapes:** `Image2D`, and `Accel` inside structs behind a `Buf`, on the GPU
  (lava-build-slang's unit tests map them); the hand-copied constants
  (`MAX_PUSH_CONSTANTS_SIZE`, the clock constants) against lava and `profile.slang`.
- **Bindless:** `Ranges::give`'s double-free panic.
- **Features:** `Features::consistent`, `without`/`missing_from`, `extensions()` agreeing with
  `optional()`, the cargo names in `SPIRV_CAPABILITIES`; the disabled ray tracing and
  performance paths in `gpu_missing_features`.
- **Lifetimes and timelines:** slot reuse, two `wait_for`s on one timeline, stages other than
  `Transfer` on a GPU.
- **Queries:** the pool-reuse panic, `QueryRecorder` misuse, `instant()` on a device that has
  it, `memory_report` with the budget.
- **Commands and families:** `flush_all`; transfer-family alignment and granularity;
  acceleration structure transforms.
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

1. **Fix the soundness bugs:** host access (16) and the undefined behaviour from safe code
   (24-37).
2. **Fix the barrier tracking (22)** and the valid-usage bugs every device hits: multi-draw
   indirect (38), the depth bias clamp (39), line topologies without extended dynamic state (40),
   depth clamp (41).
3. **Fix the timeline and presenting hangs and leaks (64-84).**
4. **Redesign the APIs users touch first:** init config, memory locations,
   frame and ring helpers, errors, naming, and the public internals of 19 before 0.1.
5. **Make the tests able to fail** (section 8), then add features by real-world demand: arrays,
   cubes and MSAA, samplers, staging.

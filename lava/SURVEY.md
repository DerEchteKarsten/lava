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
10. **Tests:** done on 2026-10-24 but for the Miri run of section 8 (Miri isn't installed).

The engine followed on 2026-10-24 (GameEngine branch `survey-follow`), with three deliberate
differences: its shaders keep their hand padding (Slang's scalar layout doesn't round struct sizes
up, see STATE), `PerFrame` and `Buffer::reserve` aren't used (no `Submission` in its systems; the
cull buffers are scratch), and `present-fences` stays checked so headless runs anywhere. The tools
generator fix of section 7 is in that branch too.

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
15. **Resize and grow is hand-rolled in core.** `grow_overflowed` can use `buffer.reserve(len, cmd)`
    (since 2026-10-24) and `RenderResources::fit` with its `fresh` flag `Image::fit(extent)`,
    which keeps the bound slots (since 2026-10-21).
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
  - **Samplers:** `Filter` has no cubic filter (`VK_EXT_filter_cubic`).
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

Done on 2026-10-24 but for these, which turned out to be meant as they are:
- **state and init:** `get_physical_device_properties` beside `properties2` (`describe_device`)
  stays: the API version it reads decides whether the 1.1 to 1.3 structs may be chained into
  the second call.
- **features:** `#[allow(unused_mut)]` on `Features::consistent` is needed without features;
  lava/build.rs's bake table repeats knowledge of `Baked`, `Dynamic::of` and the feature
  implications (kept, user decision 2026-10-20).
- **command_buffer:** the transitions loop in `PendingAccesses::made_visible` isn't redundant:
  an image has a barrier per run of mips, which it merges into one write. `image_state.rs`'s
  `for_each` needs no assert on `mips.end`: one state covers every mip, and several are indexed
  (bounds-checked).
- **raster_state:** `gated_setters!` entries name their `state` because macro hygiene keeps a
  body from reaching the setter's `self`.
- **queues:** `MAX_TIMELINES` capping queues per family is intended (documented there): a
  family never uses more queues than there are timelines to pin to them.
- **lava-build-slang:** its capability table copies `features::SPIRV_CAPABILITIES` on purpose:
  the build script's copy never compiles lava, and a test compares the two.

## 8. Tests (2026-10-18)

Done on 2026-10-24 but for this one; what the tests can't reach on the Arc B390 is in `STATE.md`
("lava: tests").
- **Validation:** `unsized_struct!`'s test hasn't run under Miri: cargo-miri isn't installed
  here (the command is in CLAUDE.md's Tests section).

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
   frame and ring helpers and errors before 0.1.
5. **Add features by real-world demand** (the tests can fail since 2026-10-24): arrays,
   cubes and MSAA, samplers, staging.

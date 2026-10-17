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
----find solutions----
- `vkCmdDispatch(): ... 144 bytes were never set with vkCmdPushConstants` (a validation warning) on the engine's headless `render` of sponza, at `core/src/render/render.rs:802`: passes push only their own struct, and the rest of the range is never set.
----intended----
- `PassKind` and the `kind::*` markers stay compiled whatever the features: they describe the pass map. A pass map whose SPIR-V needs a compiled-out feature fails at `init`, and its generated bindings don't compile.
- `line-stipple` is one feature for every line mode the device has (default lines need `strictLines` and stippled rectangular lines), so a device with stippled Bresenham but not stippled smooth lines has no stippling at all.
- The ETC2 and EAC format markers have no feature: lava never enables `textureCompressionETC2`, so images of them fail in the driver.
- With `assert-present-fences` but no display, `Ctx::features().present_fences` is false while the code assumes fences; nothing presents headless, so nothing reads it.
- The engine-facing `Gated` setters of `RasterState` check the capability whatever the value (`polygon_mode(FILL)` needs `fill-mode-non-solid` too).
- Baked draw state can make many pipelines per pass on devices without the dynamic state (the 16-pipeline warning then names the pass); `dynamic-*` without `assert-` keeps that possible.

## lava: ray tracing (tested by `gpu_ray_tracing`, not used by core yet)
----intended----
- Binding table records hold only shader group handles: no shader record data, since shaders read everything through push constants and addresses.
- A TLAS names its BLASes by address, which lava doesn't track: a BLAS dropped while a TLAS still names it is destroyed once the submissions in flight are done, so rebuild the TLAS before tracing it again.
- No indirect builds (`vkCmdBuildAccelerationStructuresIndirectKHR`): instance and primitive counts come from the host, so a GPU-culled TLAS builds every instance slot (cull with the mask or a zero transform).
- Any acceleration structure write waits for every pending trace and TLAS build (the read of `EVERY_ACCELERATION_STRUCTURE`), also of unrelated structures.
- Trace reads carry `MEMORY_READ` for Mesa's ANV (26.2.2): with `ACCELERATION_STRUCTURE_READ` alone, a trace after a TLAS rebuilt in the same submission (traced before it) saw stale nodes. Worth reporting to Mesa and dropping once fixed.
- Ray tracing's triangle facing against lava's clockwise front faces is unverified (the tests cull nothing).

## lava: passes from lava-build-slang (lava's Slang frontend)
----intended----
- `lava::init` takes one `PassMap`, so an application and a library on lava can't both bring passes; `BindingOutput` names its pass by index into that map.
- The generated-binding tests need real buffers and images (`BufferSlice` and `ImageView` have no public constructor), so they run in `gpu_headless` instead of as unit tests.

## lava: bugs from the publishing survey (2026-10-14)
Found in a read-only review; details and directions are in `lava/SURVEY.md` section 2. The rest were fixed on 2026-10-16.
----find solutions----
- Host access depends on the device (SURVEY bug 16): it works on every mapped buffer (`is_mapped()`), and with `rebar` every buffer is mapped, so `host()` on a buffer not created `cpu_writable` works with ReBAR and panics on a discrete GPU without it. Core's `variables` is `cpu_writable`.
- `Buffer::host()` hands out `&[T]` over memory a submission in flight may write, which Rust's aliasing rules don't allow (a polling loop over one slice can be hoisted). Needs an API that reads copies or volatile values.

----intended----
- `wait_idle` waits at most 1 s for the present fences after the device went idle, then gives up on the ones that didn't signal (a present that failed before it was queued) and leaks them.
- Completed pending buffer writes are only folded (into one write of every address, which every later access is made visible to once) once there are more than 64; until then they stay, since a later submission still needs a barrier to see them.

## Performance
--- find solutions ---
- Pipelines are created synchronously on first use, under a write lock, with no `VkPipelineCache` (`command_buffer.rs:233-244`, `:405`, `:602-607`, `:768`). (Ideas: per-pass locks, a warm-up list, a `VkPipelineCache` file.)
- Recording a draw costs about twice a dispatch on the CPU (`lava-tests` recording bench, 2026-10-16, borrowed unsized draws: 64 indexed draws in 8 renderings about 85 µs, 64 compute dispatches 40 to 55 µs, noisy). Not profiled yet. Suspects: the per-draw state diff and push constants, and each rendering's span, scope and attachment setup.

--- intended ---
- Hot reload recompiles every pass serially on any include change (`lava-build-slang/src/hot_reload.rs`).
(hot reload is a debug feature anyways)

## Ideas

### Multi-pass hardware counter captures (2026-10-15)
`QueryPool::performance` refuses selections that need more than one counter pass, since a frame is submitted once. A tool capture could record one frame's command buffer once and submit it once per pass with `VkPerformanceQuerySubmitInfoKHR.counterPassIndex` (with the per-pass resets the extension asks for), reading each pass's counters from the same pool. It needs a submission path that replays a recorded command buffer, which `Submission` doesn't have.

### lava-build-glsl beside lava-build-slang (2026-10-15)
lava's side is already language-neutral: `PassMap`/`PassEntry` (only what lava reads), `Bindings::new`/`with_constants`, `pipeline::reload_pass` and the shader contract in lava's `lib.rs` docs. Inside lava-build-slang the seam is between reflection and emission: Slang-specific are `compile.rs` (sessions, linking, `layout_hash`) and the walk over `TypeLayout`s in `codegen.rs` (it recognises `Img`/`Buf`/`Accel`/`SCONST` by Slang generics and attributes); language-neutral are the Rust emission (structs with layout asserts, `push_bindings`/`constant_bindings`, the `PassEntry` and `PASS_MAP` literals, the hot reload's `Build`, `write_if_changed`), `ray_tracing_groups`, the pass kind from the stages, the `Shaders` builder and the hot-reload watcher. When a GLSL frontend is written, extract the neutral part into a shared `lava-build` crate whose input is a reflected pass model (entries; push-constant fields as plain/buffer/image/accel with offsets; specialization constants; a layout hash), with a hot reload that takes a `compile(pass_index) -> Result<Vec<u8>, String>`. lava-build-glsl then compiles with shaderc/glslang, reflects the SPIR-V (or GLSL annotations) into that model, and ships its own `bindless.glsl` honouring the contract (and its own clock helpers, if wanted: lava has no part in them).

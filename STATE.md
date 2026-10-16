# State: known bugs, intended limitations and ideas (lava)

The engine's are in `../GameEngine/STATE.md`. Same conventions: `--- fix ---` entries get fixed right away; `--- intended ---` entries stay so they aren't forgotten; `--- find solutions ---` entries get proposals without code changes; `--- future work ---` entries are agreed directions waiting for their time. `## Ideas` holds design proposals we don't want to lose.

## lava: optional features (cargo features `X` / `assert-X`, 2026-10-17)
----find solutions----
- `vkCmdDispatch(): ... 144 bytes were never set with vkCmdPushConstants` (a validation warning) on the engine's headless `render` of sponza, at `core/src/render/render.rs:802`, before and after the feature change; `CommandBuffer::begin` is meant to zero the whole range once, so a recording path seems to skip it.
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
- A TLAS names its BLASes by address, which lava doesn't track: a BLAS dropped while a TLAS still names it is destroyed once the frames in flight are done, so rebuild the TLAS before tracing it again.
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
- `Queue<Transfer>`: images are `EXCLUSIVE` to the graphics family and lava records no queue family ownership transfers, so a transfer-only family may touch no image (it panics); buffers are `CONCURRENT`. `CommandBuffer` isn't typed by family either: dispatches and renderings on a transfer-only family panic when recorded instead of failing to compile.
- `Buffer::host()` hands out `&[T]` over memory a frame in flight may write, which Rust's aliasing rules don't allow (a polling loop over one slice can be hoisted). Needs an API that reads copies or volatile values.

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
`QueryPool::performance` refuses selections that need more than one counter pass, since a frame is submitted once. A tool capture could record one frame's command buffer once and submit it once per pass with `VkPerformanceQuerySubmitInfoKHR.counterPassIndex` (with the per-pass resets the extension asks for), reading each pass's counters from the same pool. It needs a submission path that replays a recorded command buffer, which `FrameSlot` doesn't have.

### lava-build-glsl beside lava-build-slang (2026-10-15)
lava's side is already language-neutral: `PassMap`/`PassEntry` (only what lava reads), `Bindings::new`/`with_constants`, `pipeline::reload_pass` and the shader contract in lava's `lib.rs` docs. Inside lava-build-slang the seam is between reflection and emission: Slang-specific are `compile.rs` (sessions, linking, `layout_hash`) and the walk over `TypeLayout`s in `codegen.rs` (it recognises `Img`/`Buf`/`Accel`/`SCONST` by Slang generics and attributes); language-neutral are the Rust emission (structs with layout asserts, `push_bindings`/`constant_bindings`, the `PassEntry` and `PASS_MAP` literals, the hot reload's `Build`, `write_if_changed`), `ray_tracing_groups`, the pass kind from the stages, the `Shaders` builder and the hot-reload watcher. When a GLSL frontend is written, extract the neutral part into a shared `lava-build` crate whose input is a reflected pass model (entries; push-constant fields as plain/buffer/image/accel with offsets; specialization constants; a layout hash), with a hot reload that takes a `compile(pass_index) -> Result<Vec<u8>, String>`. lava-build-glsl then compiles with shaderc/glslang, reflects the SPIR-V (or GLSL annotations) into that model, and ships its own `bindless.glsl` honouring the contract (and its own clock helpers, if wanted: lava has no part in them).

# State: known bugs, intended limitations and ideas (lava)

The engine's are in `../GameEngine/STATE.md`. Same conventions: `--- fix ---` entries get fixed right away; `--- intended ---` entries stay so they aren't forgotten; `--- find solutions ---` entries get proposals without code changes; `--- future work ---` entries are agreed directions waiting for their time. `## Ideas` holds design proposals we don't want to lose.

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
--- fix ---
- A pass whose regions count (`USE_CLOCKS`) uses `ShaderClockKHR`, but lava doesn't know that: without `Features::shader_clock` its module, and the pipeline of a pass without specialization constants, fail at `lava::init`. lava-tests' `test_clock.slang` is such a pass, so on such a device every GPU test fails to start (core's composite has constants, so it only fails when recorded). Fix: detect the capability like `uses_ray_query` and skip it in `PassEntry::missing_feature`.

## lava: bugs from the publishing survey (2026-10-14, not triaged yet)
Found in a read-only review; details, line numbers and directions are in `lava/SURVEY.md` section 2.
- Generated push-constant structs are `unsafe impl Pod` with padding (`CBvhCullBindings`: fields end at 232, size 240), so `bytes_of` reads uninitialised bytes; the offsets also rely on glam's SIMD alignment.
- `PendingWrite::visible_to` checks the stage and access unions separately, so a write made visible to (FRAGMENT, SAMPLED) and (COMPUTE, STORAGE) counts as visible to (COMPUTE, SAMPLED) and gets no barrier.
- `device_debug_utils` looks for `VK_EXT_debug_utils` among device extensions; it is an instance extension, so debug names and labels never turn on.
- A compacted acceleration structure keeps the source's `max_primitives` and build scratch, so `cmd.build` on it writes past its smaller buffer.
- `wait_idle` doesn't lock the queues and completes epochs submitted after the wait returned, so a frame submitted meanwhile can have its objects destroyed (plausible; core calls it from `Swapchain::drop`).
- A `FrameSlot` that isn't begun again keeps its epoch open, which stops every later destruction process-wide; without any frame loop nothing is freed.
- Dropping an `Epoch` inside `Frame::execute` can deadlock: a destructor's `forget_image` locks the `pending` mutex the recording holds (plausible).
- A panic while recording poisons the queue's `pending` mutex: later submits and destructions panic, and a drop while unwinding aborts.
- `Queue<Transfer>`: descriptor sets are bound at graphics/compute bind points on a transfer-only family, and there are no queue family ownership transfers for `EXCLUSIVE` resources.
- An image that is an attachment and also read by a draw of the same rendering gets two transitions in one barrier and ends in `ATTACHMENT_OPTIMAL` while its descriptor says `GENERAL`.
- Combined depth-stencil formats get views with both aspects, which can't be sampled; host copies pass both aspects too.
- Pending reads are only dropped when a later access covers their whole range, never when their submission completed, so moving ring ranges grow the lists forever.
- `Buffer`, `ImageView`, `Swapchain` and `Surface` have mutable pub fields through which safe code can cause out-of-bounds host reads or wrong barriers.
- Buffers are mapped whenever the device has ReBAR, so core's `variables` buffer (`Buffer::new(1, false)`, read on the host) panics on a discrete GPU without it.
- Unchecked device features: everything in `Features::features` is requested without checking support (`ERROR_FEATURE_NOT_PRESENT`), `task_shader` included.
- `Ctx::get()` blocks forever before `lava::init`; a second `init` leaks a whole instance and device.
- Bindless `Load`/`Store` in `bindless.slang` lack `NonUniformResourceIndex`.
- lava-build-slang accepts passes with their own descriptor bindings (they collide with set 0), and nested `Buf`/`Img` in push-constant structs are silently untracked.
- Mesh passes on a device without mesh shaders reach pipeline creation before the feature panic.
- Pending accesses and layouts are committed before `queue_submit2`, so a failed submit leaves the tracking describing work that never ran.
- `copy_image_to_buffer` doesn't check the buffer size; with address commands nothing else does.
- `R64*` formats have 32-bit texel types, stencil `[u8]` instead of `uint`.

## Performance
--- find solutions ---
- Pipelines are created synchronously on first use, under a write lock, with no `VkPipelineCache` (`command_buffer.rs:233-244`, `:405`, `:602-607`, `:768`). (Ideas: per-pass locks, a warm-up list, a `VkPipelineCache` file.)
- Recording a draw costs about four times a dispatch on the CPU (`lava-tests` recording bench, 2026-10-15: 64 indexed draws in 8 renderings about 94 µs, 64 compute dispatches about 22 µs). Not profiled yet. Suspects: every draw builds a `RasterHash` (two `SmallVec::from_slice`, a `Constants` copy) and hashes it under the pipeline cache's read lock, and each rendering adds its span, scope and attachment setup.

--- intended ---
- Hot reload recompiles every pass serially on any include change (`lava-build-slang/src/hot_reload.rs`).
(hot reload is a debug feature anyways)

## Ideas

### Multi-pass hardware counter captures (2026-10-15)
`QueryPool::performance` refuses selections that need more than one counter pass, since a frame is submitted once. A tool capture could record one frame's command buffer once and submit it once per pass with `VkPerformanceQuerySubmitInfoKHR.counterPassIndex` (with the per-pass resets the extension asks for), reading each pass's counters from the same pool. It needs a submission path that replays a recorded command buffer, which `FrameSlot` doesn't have.

### lava-build-glsl beside lava-build-slang (2026-10-15)
lava's side is already language-neutral: `PassMap`/`PassEntry` (only what lava reads), `Bindings::new`/`with_constants`, `pipeline::reload_pass` and the shader contract in lava's `lib.rs` docs. Inside lava-build-slang the seam is between reflection and emission: Slang-specific are `compile.rs` (sessions, linking, `layout_hash`) and the walk over `TypeLayout`s in `codegen.rs` (it recognises `Img`/`Buf`/`Accel`/`SCONST` by Slang generics and attributes); language-neutral are the Rust emission (structs with layout asserts, `push_bindings`/`constant_bindings`, the `PassEntry` and `PASS_MAP` literals, the hot reload's `Build`, `write_if_changed`), `ray_tracing_groups`, `uses_ray_query`, the pass kind from the stages, the `Shaders` builder and the hot-reload watcher. When a GLSL frontend is written, extract the neutral part into a shared `lava-build` crate whose input is a reflected pass model (entries; push-constant fields as plain/buffer/image/accel with offsets; specialization constants; a layout hash), with a hot reload that takes a `compile(pass_index) -> Result<Vec<u8>, String>`. lava-build-glsl then compiles with shaderc/glslang, reflects the SPIR-V (or GLSL annotations) into that model, and ships its own `bindless.glsl` honouring the contract (and its own clock helpers, if wanted: lava has no part in them).

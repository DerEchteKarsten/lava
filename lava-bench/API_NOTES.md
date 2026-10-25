# API notes: lava as seen while writing the benchmark scenarios

A running log kept while writing each scenario's lava side (and, for contrast, its hand-written
baseline). Each entry names the scenario and code it came from, its kind, a severity (blocker,
annoying, minor) and a suggested fix. Problems were worked around inside the bench; the only
lava change is the hidden `raw` module.

Kinds: **friction** (verbose or awkward call sites), **missing** (a feature or use case lava
lacks), **surprise** (behaviour that needed lava's source or docs, or panicked unexpectedly),
**errors** (unclear panics or diagnostics), **raw view** (what the hand-written version says
more directly, which may point to an API gap rather than overhead).

## forward

- **friction, annoying** (`forward.rs`, every rendering): draws made in a loop need two
  collections that outlive the builder: the bindings (`Vec<Bindings>`), then the `Draw`s that
  borrow them (`Vec<Draw>`), then a loop of `rendering.draw(draw)`. Three passes over the
  draws and two allocations for what the baseline does in one loop of push + draw.
  *Fix:* a `RasterBuilder::draw_with(|| ..)` or `draws(iter)` that takes owned
  `(Bindings, DrawKind, RasterState)` and stores them in the builder's own arena, so the
  common loop needs no caller-side `Vec`s.
- **surprise, minor** (baseline push constants): the generated `C{Pass}Bindings` push-constant
  structs have *private* `_pad` fields where Slang leaves gaps (tail padding included), so code
  that fills them itself (the baseline, or another binding generator) can't use a struct
  literal or `..Default::default()`; it zeroes and assigns field by field. Structs behind
  pointers have `pub #[doc(hidden)]` pads and a `Default`. *Fix:* the same `pub` hidden pads and
  a zero `Default` for push-constant structs.
- **surprise, minor** (baseline materials): `host_mut` takes only `NoUninit` elements, so a
  buffer of a referencing struct can't be written in place at all, not even with handles the
  caller keeps alive; the baseline writes through `host_ptr()` with `unsafe`. Intended (lava
  follows references only from reference infos), but there's no documented unchecked escape
  hatch. *Fix:* an `unsafe fn host_mut_unreferenced(&mut self) -> &mut [T]` for callers that
  keep the referenced objects alive themselves.
- **raw view, minor** (shadow map, HDR target): every frame overwrites them completely, and the
  baseline says so with `UNDEFINED` old layouts. Lava keeps their contents (each transition goes
  from the tracked layout). *Fix:* worth measuring `Attachment::discard_before()` (not used
  here); an `Image::discard()` that resets the tracked layout to `UNDEFINED` before a frame
  would cover compute-written images too.
- **surprise, minor** (`--check`): with validation on, the layer's best-practices checks warn at
  device creation about the D3D/GL emulation extensions lava enables because the default
  features compile them (`VK_EXT_depth_clip_enable`, `depth_clip_control`, `provoking_vertex`).
  *Fix:* none needed in lava; a note in the docs on why they're on.

## deferred

- **friction, minor** (`deferred.rs`, bloom): one call site that takes its source from either
  the HDR target or a bloom mip compiles only because both images have the same usage type
  (`usage![Sampled, Storage]`), so their views are one type. Views of differently typed images
  need a generic helper per call site (see `gpu_driven`). *Fix:* an
  `ImageView::erase() -> ImageView<format::Dynamic, usage::Dynamic, D>` for heterogeneous
  sources; the generated bindings would check it at runtime as they do for erased images.
- **positive** (bloom, TAA): a `MipChain` and its per-mip views made a 13-dispatch chain with
  per-mip barriers and a ping-pong history read like the algorithm, while the baseline spells
  out 20 layout transitions and has to know the history's layout from the last frame.

## gpu_driven

- **friction, minor** (`cull.slang`): the shader's indirect command struct is generated as its
  own type (`DrawIndexedCommand`), so the buffer is lava's `DrawIndexedIndirectCommand` and the
  pass argument a `.cast()` of it. *Fix:* add the indirect command structs to
  `describe::MIRRORED_TYPES` (Slang names `DrawIndexedIndirectCommand` etc. in
  `bindless.slang`), as `RawInstance` and `Sampler` are.
- **friction, minor** (Hi-Z): mip 0 reads the depth image, later mips the chain, so the step
  is a generic function whose bound spells out what the generated code wants
  (`impl SampledView<Dim = D2, Format: HasTexels<[f32; 1]>> + 'a`). Same fix as above.
- **positive**: `Draw::indexed_indirect_count` takes the command and count slices; the baseline
  computes the strided range, the stride and the maximum by hand.

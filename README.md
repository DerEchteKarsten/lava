# lava

A Vulkan abstraction for Rust that keeps Vulkan's model but takes its bookkeeping off your hands:

- **Bindless resources.** Every image sits in one of two global descriptor arrays, and shaders reach buffers through device addresses. Passes have no descriptor sets of their own.
- **Automatic barriers.** Commands know what they read and write, so lava records the barriers and layout transitions between them, per image and buffer range.
- **Typed passes generated from shaders.** `lava-build-slang` compiles your Slang passes in your build script and generates a Rust function per pass whose arguments are the pass's push constants: buffer slices, image views, plain values. A wrong buffer or image type doesn't compile.
- **Deferred destruction.** Dropped objects are destroyed once no frame in flight can reach them.
- **Dynamic state.** A draw's raster state (culling, depth, stencil, blending, lines, …) is dynamic state, so pipelines depend only on the shaders, their constants and the attachment formats.

lava is young (0.x) and is developed alongside a game engine. Expect breaking changes.

## Crates

| Crate | What it is |
|---|---|
| `lava` | The library. It takes SPIR-V and knows no shader language. |
| `lava-build-slang` | The Slang frontend, used from `build.rs`; also the optional runtime shader hot reload. |
| `lava-macros` | lava's internal proc macros. |

## Requirements

- Rust 1.96 or newer.
- A Vulkan 1.4 driver with `VK_KHR_device_address_commands`, the extended dynamic state 2 and 3 extensions (including the logic op and the extended dynamic state 3 bits lava sets), `VK_EXT_color_write_enable`, `VK_KHR_line_rasterization`, `VK_EXT_provoking_vertex`, `VK_EXT_conservative_rasterization`, `VK_EXT_depth_clip_enable`, `VK_EXT_depth_clip_control`, `VK_EXT_depth_clamp_control` and `VK_EXT_sample_locations`. `lava::init` names whatever a GPU lacks.
- Everything else is optional (`lava::Features`: mesh shaders, ray tracing, BC textures, …). lava enables what the device has, `Config::required` makes features mandatory, and a pass whose SPIR-V needs a missing feature panics when it is recorded.
- The Slang compiler at build time (see [`lava-build-slang`](lava-build-slang/README.md)).

lava is tested on Linux with Mesa's ANV driver (Intel Arc).

## A first pass

`Cargo.toml`:

```toml
[dependencies]
lava = "0.1"
glam = { version = "0.32", features = ["bytemuck"] }
bytemuck = { version = "1", features = ["derive"] }

[build-dependencies]
lava-build-slang = "0.1"
```

`shaders/double.slang`, one pass per file:

```slang
#include "bindless.slang"

struct DoubleBindings {
    MutBuf<uint> values;
    uint count;
}

[[vk::push_constant]] DoubleBindings b;

[shader("compute")]
[numthreads(64, 1, 1)]
void double_values(uint i: SV_DispatchThreadID) {
    if (i < b.count) {
        b.values.val[i] *= 2;
    }
}
```

`build.rs`:

```rust
fn main() {
    lava_build_slang::Shaders::new().passes("shaders").compile();
}
```

`src/main.rs`:

```rust
use lava::{
    buffer::{Buffer, usage::Storage},
    epoch,
    vkobjects::queue::{FrameSlot, Gfx, Queue},
};

mod bindings {
    lava::include_passes!();
}
use bindings::{Double, PASS_MAP};

fn main() -> lava::Result<()> {
    lava::init(&lava::Config::new(&PASS_MAP))?;

    let mut values = Buffer::<u32, Storage>::new(256, true)?;
    values.host_mut().iter_mut().enumerate().for_each(|(i, v)| *v = i as u32);

    let queue = Queue::<Gfx>::new();
    let mut slot = FrameSlot::new(&queue)?;
    slot.begin(epoch::reserve())?.execute(&queue, &[], &[], |cmd| {
        cmd.compute(&Double::push_bindings(values.range(..), 256), [4, 1, 1]);
    })?;
    // Dropping a submitted slot waits for it.
    drop(slot);

    assert_eq!(values.host()[100], 200);
    Ok(())
}
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option. Unless you explicitly state otherwise, any contribution you intentionally submit for inclusion in lava, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

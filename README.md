# lava

A Vulkan abstraction for Rust that keeps Vulkan's model but takes its bookkeeping off your hands:

- **Bindless resources.** Every image sits in one of two global descriptor arrays, at slots lava allocates, and shaders reach buffers through device addresses. Passes have no descriptor sets of their own.
- **Automatic barriers.** Commands know what they read and write, so lava records the barriers and layout transitions between them, per image and buffer range.
- **Typed passes generated from shaders.** `lava-build-slang` compiles your Slang passes in your build script and generates a Rust function per pass whose arguments are the pass's push constants: buffer slices, image views, plain values. A wrong buffer or image type doesn't compile.
- **Timelines.** Work is submitted on up to eight timelines (graphics, and async compute and transfer where the device has them), ordered across them with waits, with image ownership transfers recorded on request. Presenting needs no semaphores of your own: the swapchain image's acquire and present follow from what your submission does with it.
- **Deferred destruction.** Dropped objects are destroyed once no submission in flight can reach them.
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
- A Vulkan 1.4 driver with `VK_KHR_device_address_commands`. `lava::init` names whatever a GPU lacks.
- Everything else is optional (`lava::Features`: mesh shaders, ray tracing, BC textures, extended dynamic state, …), and each optional feature is two cargo features:
  - `X` (all of them are default) compiles it, and lava checks the device where it is used. Functions that need it return `lava::features::Gated` values: `Result`s with `MissingFeature`. That includes the generated bindings of a pass whose SPIR-V needs it.
  - `assert-X` makes `lava::init` refuse a device without it. The checks go, and those functions return their value directly.
  - Without either, its API isn't compiled.
- The `dynamic-*` features are extended dynamic state. Where the device has a state, draws that differ in it share a pipeline. Elsewhere the state is part of the pipeline's key. Asserting one keeps it out of the key.
- An application that turns off lava's defaults (`default-features = false`) must do so in every crate that depends on lava, since cargo enables the union of all of them.
- The Slang SDK at build time, found through `SLANG_DIR`, and an rpath to it for build scripts and hot-reload binaries. [`lava-build-slang`](lava-build-slang/README.md#setting-up-slang) shows the setup.

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

`.cargo/config.toml`, pointing at your Slang SDK:

```toml
[env]
SLANG_DIR = { value = "slang-sdk", relative = true }

[build]
rustflags = ["-C", "link-arg=-Wl,-rpath,/absolute/path/to/slang-sdk/lib"]
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
    queue::{Graphics, Timeline},
};

mod bindings {
    lava::include_passes!();
}
use bindings::{Double, PASS_MAP};

fn main() -> lava::Result<()> {
    lava::init(&lava::Config::new(&PASS_MAP))?;

    let mut values = Buffer::<u32, Storage>::new(256, true)?;
    values.host_mut().iter_mut().enumerate().for_each(|(i, v)| *v = i as u32);

    let timeline = Timeline::<Graphics>::new(2)?;
    let submission = timeline.begin()?;
    let done = submission.submit(|cmd| {
        cmd.compute(&Double::push_bindings(values.range(..), 256), [4, 1, 1], None);
    })?;
    done.block()?;

    assert_eq!(values.host()[100], 200);
    Ok(())
}
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option. Unless you explicitly state otherwise, any contribution you intentionally submit for inclusion in lava, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

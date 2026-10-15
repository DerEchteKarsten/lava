# lava-build-slang

[lava](https://github.com/DerEchteKarsten/lava)'s Slang frontend. Called from an application's `build.rs`, it compiles every `.slang` file of a directory as one pass, writes the SPIR-V to `OUT_DIR`, and generates the pass's typed bindings and the `PASS_MAP` that `lava::init` takes. The application includes them with `lava::include_passes!()`.

```rust
// build.rs
fn main() {
    lava_build_slang::Shaders::new()
        .passes("shaders/passes")
        .include("shaders/include")
        .define("MY_DEFINE", "1")
        .compile();
}
```

Every compilation can include `bindless.slang` (lava's bindless images, buffer pointers and specialization constants), `ray_tracing.slang` and `profile.slang`, which this crate ships.

With the `hot-reload` feature, `lava_build_slang::watch!()` (called after `lava::init`) recompiles changed passes while the application runs and swaps their pipelines in.

## The Slang compiler

lava-build-slang loads Slang at runtime, so nothing needs an rpath or `LD_LIBRARY_PATH`. At build time it looks for the library in, in order:

1. `SLANG_LIB_DIR`;
2. `SLANG_DIR` (a [Slang release](https://github.com/shader-slang/slang/releases), unpacked);
3. `VULKAN_SDK` (LunarG's Vulkan SDK ships Slang).

Without any of them the system's loader looks for `libslang-compiler` at runtime. A binary that compiles shaders at runtime (hot reload) and runs on another machine needs Slang there too.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.

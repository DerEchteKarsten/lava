# lava-build-slang

[lava](https://github.com/DerEchteKarsten/lava)'s Slang frontend. Called from an application's `build.rs`, it compiles every `.slang` file of a directory (not its subdirectories) as one pass, on up to four threads, writes the SPIR-V to `OUT_DIR`, and generates the pass's typed bindings and the `PASS_MAP` that `lava::init` takes. The application includes them with `lava::include_passes!()`.

```rust
// build.rs
fn main() {
    lava_build_slang::Shaders::new()
        .passes("shaders/passes")
        .include("shaders/include")
        .define("MY_DEFINE", "1")
        .debug_info(cfg!(debug_assertions)) // source-level shader debugging in RenderDoc or Nsight
        .compile();
}
```

A pass is named after its file (`bvh_cull.slang` is `BvhCull`; the name must make a Rust identifier, and two files may not share one); a file of several compute entry points is one pass per entry point (`main` of `cull.slang` is `CullMain`). For each pass the bindings have an argument struct of its push constants (`BvhCullArgs`, buffers as `BufferSlice`s, images as views, TLASes as `&Tlas`) and, if it has specialization constants, one of them (`BvhCullConstants`); `BvhCull::push_bindings(args, [constants,] accesses)` returns the bindings, `accesses` naming what the pass reaches through handles and pointers in buffers. Compute passes also get `WORKGROUP_SIZE` and `groups(threads)`. Structs the passes read through pointers are generated too, padded to Slang's layout, with a `Default` that nulls image handles; `[RustConst] static const` scalars become Rust constants. Compile errors and lava-build-slang's own (with the file and field) are collected and reported together; Slang's warnings become cargo warnings.

```rust
cmd.compute(
    &BvhCull::push_bindings(BvhCullArgs { nodes: nodes.range(..), visible: visible.range(..), count }, None),
    BvhCull::groups([count, 1, 1]),
);
```

The generated code names everything through lava (`::lava::glam`, `::lava::bytemuck`, which lava re-exports with bytemuck's `derive` and `min_const_generics`), so the application needs no other dependency for it.

Every compilation can include `bindless.slang` (lava's bindless images, buffer pointers and specialization constants), `ray_tracing.slang` and `profile.slang`, which this crate ships.

With the `hot-reload` feature, `lava_build_slang::watch!()` (called after `lava::init`) recompiles changed passes while the application runs and swaps their pipelines in. It needs lava-build-slang as a normal dependency with the feature, beside the build dependency.

## Setting up Slang

lava-build-slang links the Slang compiler through [slang-rs](https://github.com/FloatyMonkey/slang-rs), so every application needs a Slang SDK at build time, and every binary that links it needs to find the library when it starts. You set this up once per checkout:

1. Get Slang: unpack a [Slang release](https://github.com/shader-slang/slang/releases), or install LunarG's Vulkan SDK, which ships it. slang-rs generates its bindings from `slang.h` with bindgen, so clang has to be installed too.
2. Tell the build where it is. slang-rs reads `SLANG_DIR` (the unpacked release, with `include/` and `lib/`), else `VULKAN_SDK`. `SLANG_INCLUDE_DIR` and `SLANG_LIB_DIR` override either half. The simplest place is your project's `.cargo/config.toml`:

   ```toml
   [env]
   SLANG_DIR = { value = "slang-sdk", relative = true }
   ```

3. Let binaries find the library. Your build script runs lava-build-slang, so it links `libslang-compiler`, and so does an application built with the `hot-reload` feature. Cargo doesn't put the SDK on the library path, so add an rpath to it, again in `.cargo/config.toml`. It has to be an absolute path:

   ```toml
   [build]
   rustflags = ["-C", "link-arg=-Wl,-rpath,/absolute/path/to/slang-sdk/lib"]
   ```

   Setting `LD_LIBRARY_PATH` to the SDK's `lib/` works as well. On Windows, put the SDK's `bin/` on `PATH`.

A binary built with `hot-reload` needs Slang on every machine it runs on. Without the feature, only the build needs it.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.

//! Links the Slang compiler into every binary that uses lava-build-slang
fn main() {
    // shader-slang-sys still asks for `-lslang`, which the linker may resolve to the unrelated
    // S-Lang library; naming the SDK's `slang-compiler` here makes sure the compiler is linked.
    println!("cargo::rustc-link-lib=dylib=slang-compiler");
}

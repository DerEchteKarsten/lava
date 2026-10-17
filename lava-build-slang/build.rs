//! Links the Slang compiler into every binary that uses lava-build-slang
fn main() {
    // `-lslang` can resolve to the unrelated S-Lang library.
    println!("cargo::rustc-link-lib=dylib=slang-compiler");
}

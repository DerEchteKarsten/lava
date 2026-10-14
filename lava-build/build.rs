//! Links the Slang compiler into every binary that uses lava-build
fn main() {
    // `-lslang` can resolve to the unrelated S-Lang library in /usr/lib, which comes first in
    // the search path of the final binary.
    println!("cargo::rustc-link-lib=dylib=slang-compiler");
}

//! Compiles the benchmark passes in `shaders/`, apart from the tests' pass map
fn main() {
    lava_build_slang::Shaders::new()
        .passes("shaders")
        .include("shaders/include")
        .compile();
}

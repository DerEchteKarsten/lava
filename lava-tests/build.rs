//! Compiles the test passes in `shaders/`
fn main() {
    lava_build_slang::Shaders::new()
        .passes("shaders")
        .include("shaders/include")
        .compile();
}

//! Compiles the test passes in `shaders/`
fn main() {
    lava_build::Shaders::new()
        .passes("shaders")
        .include("shaders/include")
        .compile();
}

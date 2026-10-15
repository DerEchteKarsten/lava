//! Compiles the test passes in `shaders/`
fn main() {
    lava_build_slang::Shaders::new()
        .passes("shaders")
        .include("shaders/include")
        // `test_clock.slang`'s regions count.
        .define("USE_CLOCKS", "1")
        .compile();
}

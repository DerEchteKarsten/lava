//! Tells lava-build-slang in the application's build script whether lava has its `profiling` feature
fn main() {
    if std::env::var_os("CARGO_FEATURE_PROFILING").is_some() {
        println!("cargo::metadata=profiling=1");
    }
}

//! lava-build-slang: compiles an application's Slang passes to SPIR-V and generates their typed bindings and `PASS_MAP` for lava, called from the application's build script; with `hot-reload`, recompiles them while the application runs
use std::{env, fs, path::Path};

mod codegen;
pub mod compile;
#[cfg(feature = "hot-reload")]
pub mod hot_reload;

/// The Slang side of lava's shader contract (lava's crate docs; `bindless.slang`,
/// `ray_tracing.slang`: the descriptor sets, acceleration structure instances), plus the shader
/// clock helpers of `profile.slang`, searched by every session.
pub const SLANG_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/slang");

/// Slang's optimization level for every pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Optimization {
    /// For debugging the SPIR-V.
    None,
    Default,
    #[default]
    Maximal,
}

/// The passes of an application, compiled by its build script:
/// ```ignore
/// lava_build_slang::Shaders::new()
///     .passes("../shaders/passes")
///     .include("../shaders/include")
///     .compile();
/// ```
/// and included with `lava::include_passes!()`, whose `PASS_MAP` goes to `lava::init`. With
/// the `hot-reload` feature, [`watch!`] (called in the same crate, after `lava::init`)
/// recompiles them while the application runs.
/// Relative paths are resolved against the package's directory.
#[derive(Default)]
pub struct Shaders {
    /// Absolute directories whose every `*.slang` is a pass.
    passes: Vec<String>,
    /// Absolute include directories.
    includes: Vec<String>,
    defines: Vec<(String, String)>,
    optimization: Optimization,
}

impl Shaders {
    pub fn new() -> Self {
        Self::default()
    }

    /// Every `*.slang` file in `dir` is a pass, named after its file (`bvh_cull.slang` is
    /// `BvhCull`).
    pub fn passes(mut self, dir: impl AsRef<Path>) -> Self {
        self.passes.push(absolute(dir.as_ref()));
        self
    }

    /// A directory `#include`s are searched in.
    pub fn include(mut self, dir: impl AsRef<Path>) -> Self {
        self.includes.push(absolute(dir.as_ref()));
        self
    }

    /// A preprocessor macro of every pass.
    pub fn define(mut self, name: &str, value: &str) -> Self {
        self.defines.push((name.to_string(), value.to_string()));
        self
    }

    pub fn optimization(mut self, optimization: Optimization) -> Self {
        self.optimization = optimization;
        self
    }

    /// Compiles every pass into `$OUT_DIR/lava/passes/<Pass>.spv` and writes the bindings to
    /// `$OUT_DIR/lava/bindings.rs`. Panics with the file on a compile error.
    pub fn compile(self) {
        for dir in self.passes.iter().chain(&self.includes).map(String::as_str) {
            println!("cargo::rerun-if-changed={dir}");
        }
        println!("cargo::rerun-if-changed={SLANG_DIR}");
        let out_dir =
            Path::new(&env::var("OUT_DIR").expect("called from a build script")).join("lava");
        codegen::generate(&self, &out_dir);
    }
}

/// `dir` resolved against the package's directory, without `..` and symlinks, so hot reload
/// matches the paths it watches against the passes' sources.
fn absolute(dir: &Path) -> String {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("called from a build script");
    let dir = Path::new(&manifest_dir).join(dir);
    fs::canonicalize(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .to_str()
        .expect("shader paths are UTF-8")
        .to_string()
}

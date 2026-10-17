//! lava-build-slang: compiles an application's Slang passes to SPIR-V and generates their typed bindings and `PASS_MAP` for lava, called from the application's build script; with `hot-reload`, recompiles them while the application runs
use std::{env, fs, path::Path};

mod codegen;
pub mod compile;
mod features;
mod reflect;
#[cfg(feature = "hot-reload")]
pub mod hot_reload;

/// The Slang side of lava's shader contract (`bindless.slang`, `ray_tracing.slang`) and the
/// shader clock helpers (`profile.slang`), searched by every session.
pub const SLANG_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/slang");

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Optimization {
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
/// The crate includes them with `lava::include_passes!()` and hands `PASS_MAP` to
/// `lava::init`; with `hot-reload`, [`watch!`] (after `lava::init`, in the same crate)
/// recompiles them while it runs. Relative paths are resolved against the package's directory.
#[derive(Default)]
pub struct Shaders {
    passes: Vec<String>,
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

    pub fn include(mut self, dir: impl AsRef<Path>) -> Self {
        self.includes.push(absolute(dir.as_ref()));
        self
    }

    pub fn define(mut self, name: &str, value: &str) -> Self {
        self.defines.push((name.to_string(), value.to_string()));
        self
    }

    pub fn optimization(mut self, optimization: Optimization) -> Self {
        self.optimization = optimization;
        self
    }
}

/// Canonical, so the hot reload matches the paths it watches against the passes' sources.
fn absolute(dir: &Path) -> String {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("called from a build script");
    let dir = Path::new(&manifest_dir).join(dir);
    fs::canonicalize(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .to_str()
        .expect("shader paths are UTF-8")
        .to_string()
}

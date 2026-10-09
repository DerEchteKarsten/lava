//! Slang session and pass linking shared by build.rs and shader hot reload
use std::ffi::CString;

use shader_slang as slang;

pub const SHADER_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../shaders");
/// Lava's own test passes, in `lava/tests/shaders`.
pub const TEST_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests");

/// How a pass compiles: the cargo features that matter, and which pass it is.
pub struct Options {
    /// Also search `lava/tests`, whose `shaders/` holds the test passes.
    pub test_passes: bool,
    /// Defines `PROFILING`.
    pub profiling: bool,
    /// The pass's index in `PASS_MAP`, defined as `PROFILE_PASS` for the shader clocks of
    /// `profile.slang`.
    pub pass_index: usize,
}

/// A session that compiles a pass to SPIR-V 1.6. Pass sources are named relative to the
/// shader directory (`passes/x.slang`) or, with `test_passes`, to `lava/tests`.
pub fn session(global: &slang::GlobalSession, options: &Options) -> slang::Session {
    let mut paths = vec![
        CString::new(SHADER_DIR).unwrap(),
        CString::new(format!("{SHADER_DIR}/include")).unwrap(),
    ];
    if options.test_passes {
        paths.push(CString::new(TEST_DIR).unwrap());
    }
    let search_paths: Vec<_> = paths.iter().map(|path| path.as_ptr()).collect();
    let targets = [slang::TargetDesc::default()
        .format(slang::CompileTarget::Spirv)
        .profile(global.find_profile("spirv_1_6"))];
    let mut compiler = slang::CompilerOptions::default()
        .optimization(slang::OptimizationLevel::Maximal)
        .vulkan_use_entry_point_name(true)
        .matrix_layout_column(true);
    if options.profiling {
        compiler = compiler.macro_define("PROFILING", "1");
    }
    compiler = compiler.macro_define("PROFILE_PASS", &options.pass_index.to_string());
    global
        .create_session(
            &slang::SessionDesc::default()
                .targets(&targets)
                .search_paths(&search_paths)
                .options(&compiler),
        )
        .expect("failed to create a slang session")
}

/// The pass in `source` with all its entry points, linked.
pub fn link(session: &slang::Session, source: &str) -> slang::Result<slang::ComponentType> {
    let module = session.load_module(source)?;
    let mut components: Vec<slang::ComponentType> = vec![module.clone().into()];
    components.extend(module.entry_points().map(Into::into));
    session.create_composite_component_type(&components)?.link()
}

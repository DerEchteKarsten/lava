//! Slang session and pass linking shared by build.rs and shader hot reload
use std::ffi::CString;

use shader_slang as slang;

pub const SHADER_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../shaders");
/// Lava's own test passes, in `lava/tests/shaders`.
pub const TEST_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests");
/// The specialization constant ids of a pass's shader clock region hashes (`CLOCK_REGION_0..`
/// in `profile.slang`). `build.rs` bakes their values into the pass's `constant_bindings`
/// instead of making them parameters.
pub const CLOCK_REGION_IDS: std::ops::Range<u32> = 901..909;

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

/// The names of the clock regions in a linked pass: the strings it hashes, which only
/// `CLOCK_REGION` does.
#[allow(dead_code)]
pub fn clock_regions(program: &slang::ComponentType) -> Vec<String> {
    let layout = program.layout(0).expect("a linked pass has a layout");
    layout.hashed_strings().map(str::to_string).collect()
}

/// A hash of what the generated bindings of a pass depend on: its push constant fields (name,
/// offset, size), its specialization constants (id, size) and its entry points (stage, name).
/// A reload whose hash differs from the baked one would run against stale bindings.
pub fn layout_hash(layout: &slang::reflection::Shader) -> u64 {
    use slang::ParameterCategory as Cat;
    use std::hash::{DefaultHasher, Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    for parameter in layout.parameters() {
        match parameter.category() {
            Some(Cat::PushConstantBuffer) => {
                let Some(fields) = parameter
                    .type_layout()
                    .and_then(|t| t.element_type_layout())
                else {
                    continue;
                };
                fields.size(Cat::Uniform).hash(&mut hasher);
                for field in fields.fields() {
                    field.variable().and_then(|v| v.name()).hash(&mut hasher);
                    field.offset(Cat::Uniform).hash(&mut hasher);
                    field
                        .type_layout()
                        .map(|t| t.size(Cat::Uniform))
                        .hash(&mut hasher);
                }
            }
            Some(Cat::SpecializationConstant) => {
                parameter
                    .offset(Cat::SpecializationConstant)
                    .hash(&mut hasher);
                parameter
                    .type_layout()
                    .map(|t| t.size(Cat::Uniform))
                    .hash(&mut hasher);
            }
            _ => {}
        }
    }
    for entry in layout.entry_points() {
        format!("{:?}", entry.stage()).hash(&mut hasher);
        entry.name().hash(&mut hasher);
    }
    hasher.finish()
}

/// The pass in `source` with all its entry points, linked.
pub fn link(session: &slang::Session, source: &str) -> slang::Result<slang::ComponentType> {
    let module = session.load_module(source)?;
    let mut components: Vec<slang::ComponentType> = vec![module.clone().into()];
    components.extend(module.entry_points().map(Into::into));
    session.create_composite_component_type(&components)?.link()
}

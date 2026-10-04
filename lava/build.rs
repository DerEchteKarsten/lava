//! Compiles Slang passes to SPIR-V and generates typed pass bindings from reflection
use std::{
    collections::{BTreeMap, HashMap},
    env,
    ffi::CString,
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use shader_slang::{
    self as slang, CompilerOptions, ComponentType, ParameterCategory as Cat, ScalarType,
    SessionDesc, Stage, TypeKind,
    reflection::{TypeLayout, VariableLayout},
};

struct StructDef {
    body: String,
    checks: String,
    signature: Vec<(String, usize)>,
    size: usize,
}

#[derive(Clone, Copy)]
enum PassKind {
    Compute,
    RayTracing,
    RasterVertex,
    RasterMesh,
}

impl PassKind {
    fn stage(self) -> &'static str {
        match self {
            PassKind::Compute => "vk::PipelineStageFlags2::COMPUTE_SHADER",
            PassKind::RayTracing => "vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR",
            PassKind::RasterVertex => {
                "vk::PipelineStageFlags2::VERTEX_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER"
            }
            PassKind::RasterMesh => {
                "vk::PipelineStageFlags2::MESH_SHADER_EXT | vk::PipelineStageFlags2::FRAGMENT_SHADER"
            }
        }
    }
}

enum FieldKind {
    Plain,
    Buffer,
    Image { usage: u64 },
}

type Structs = BTreeMap<String, StructDef>;

const IMAGE_TYPES: [&str; 1] = ["Img"];
const BUFFER_TYPES: [&str; 2] = ["Buf", "MutBuf"];
/// Image types as struct fields: all are a `BindlessHandle` on the Rust side.
const HANDLE_TYPES: [&str; 2] = ["Img", "DynImg"];

fn field_name(f: &VariableLayout) -> &str {
    f.variable().and_then(|v| v.name()).expect("unnamed field")
}

fn type_name(t: &TypeLayout) -> &str {
    t.name().unwrap_or("UnknownStruct")
}

fn is_struct_named(t: &TypeLayout, names: &[&str]) -> bool {
    t.kind() == TypeKind::Struct && names.contains(&type_name(t))
}

fn layout_signature(t: &TypeLayout) -> Vec<(String, usize)> {
    t.fields()
        .map(|f| (field_name(f).to_string(), f.offset(Cat::Uniform)))
        .collect()
}

fn layout_checks(rust_name: &str, t: &TypeLayout) -> String {
    let mut out = String::new();
    for f in t.fields() {
        out.push_str(&format!(
            "const _: () = assert!(core::mem::offset_of!({rust_name}, {}) == {});\n",
            field_name(f),
            f.offset(Cat::Uniform)
        ));
    }
    out.push_str(&format!(
        "const _: () = assert!(core::mem::size_of::<{rust_name}>() == {});\n",
        t.size(Cat::Uniform)
    ));
    out
}

fn rust_type(t: &TypeLayout, structs: &mut Structs) -> String {
    match t.kind() {
        TypeKind::Scalar => match t.scalar_type() {
            Some(ScalarType::Uint32) => "u32".into(),
            Some(ScalarType::Int32) => "i32".into(),
            Some(ScalarType::Float32) => "f32".into(),
            Some(ScalarType::Uint8) => "u8".into(),
            Some(ScalarType::Uint64) => "u64".into(),
            Some(ScalarType::Bool) => "u32".into(),
            other => panic!("unsupported scalar type {other:?}"),
        },
        TypeKind::Vector => {
            let prefix = match t.scalar_type() {
                Some(ScalarType::Float32) => "",
                Some(ScalarType::Uint32) => "U",
                Some(ScalarType::Int32) => "I",
                other => panic!("unsupported vector element type {other:?}"),
            };
            format!("{prefix}Vec{}", t.element_count().unwrap())
        }
        TypeKind::Matrix => {
            let (cc, rc) = (t.column_count().unwrap(), t.row_count().unwrap());
            if cc == rc {
                format!("Mat{cc}")
            } else {
                format!("Mat{cc}x{rc}")
            }
        }
        TypeKind::Array => format!(
            "[{}; {}]",
            rust_type(t.element_type_layout().unwrap(), structs),
            t.element_count().unwrap()
        ),
        TypeKind::Struct => {
            let name = type_name(t).to_string();
            let signature = layout_signature(t);
            let size = t.size(Cat::Uniform);

            if let Some(existing) = structs.get(&name) {
                if existing.signature != signature || existing.size != size {
                    panic!(
                        "struct `{name}` appears with two different layouts:\n  {:?} (size {})\n  {:?} (size {})\n\
                         (same name in different passes, or used both in push constants and behind a pointer)",
                        existing.signature, existing.size, signature, size
                    );
                }
                return name;
            }

            structs.insert(
                name.clone(),
                StructDef {
                    body: String::new(),
                    checks: String::new(),
                    signature,
                    size,
                },
            );
            let mut body = String::new();
            for f in t.fields() {
                body.push_str(&format!(
                    "    pub {}: {},\n",
                    field_name(f),
                    gpu_type(f.type_layout().unwrap(), structs)
                ));
            }
            let def = structs.get_mut(&name).unwrap();
            def.body = body;
            def.checks = layout_checks(&name, t);
            name
        }
        TypeKind::Pointer => {
            if let Some(pointee) = t.element_type_layout() {
                rust_type(pointee, structs);
            }
            "u64".into()
        }
        other => format!("compile_error!(\"Unsupported Type {other:?}\")"),
    }
}

fn gpu_type(t: &TypeLayout, structs: &mut Structs) -> String {
    if is_struct_named(t, &HANDLE_TYPES) {
        "BindlessHandle".into()
    } else if is_struct_named(t, &BUFFER_TYPES) {
        rust_type(t.field_by_index(0).unwrap().type_layout().unwrap(), structs);
        "u64".into()
    } else {
        rust_type(t, structs)
    }
}

fn image_usage(t: &TypeLayout) -> u64 {
    let generics = t.ty().unwrap().generic_container().unwrap();
    generics
        .value_parameters()
        .find(|vp| vp.name() == Some("U"))
        .map(|vp| generics.concrete_int_val(vp) as u64)
        .unwrap_or(0)
}

fn image_access(usage: u64) -> &'static str {
    match usage {
        0 => "vk::AccessFlags2::SHADER_SAMPLED_READ",
        1 => "vk::AccessFlags2::SHADER_STORAGE_READ",
        _ => "vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE",
    }
}

fn image_texel(t: &TypeLayout) -> Option<(&'static str, usize)> {
    let generics = t.ty().unwrap().generic_container().unwrap();
    let element = generics
        .type_parameters()
        .find(|tp| tp.name() == Some("E"))
        .and_then(|tp| generics.concrete_type(tp))?;
    let scalar = match element.scalar_type() {
        ScalarType::Float32 => "f32",
        ScalarType::Uint32 => "u32",
        ScalarType::Int32 => "i32",
        ScalarType::Uint8 => "u8",
        _ => return None,
    };
    let components = match element.kind() {
        TypeKind::Vector => element.element_count(),
        _ => 1,
    };
    Some((scalar, components))
}

fn field_kind(t: &TypeLayout) -> FieldKind {
    if is_struct_named(t, &IMAGE_TYPES) {
        FieldKind::Image {
            usage: image_usage(t),
        }
    } else if is_struct_named(t, &BUFFER_TYPES) {
        FieldKind::Buffer
    } else {
        FieldKind::Plain
    }
}

fn generate_gpu_struct(name: &str, pc: &TypeLayout, structs: &mut Structs) -> String {
    let fields = pc
        .fields()
        .map(|f| {
            format!(
                "    pub {}: {},",
                field_name(f),
                gpu_type(f.type_layout().unwrap(), structs)
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let checks = layout_checks(name, pc);

    format!(
        r#"
#[derive(Clone, Copy)]
#[repr(C)]
pub struct {name} {{
{fields}
}}
{checks}
unsafe impl bytemuck::Pod for {name} {{}}
unsafe impl bytemuck::Zeroable for {name} {{}}
"#
    )
}

struct NewFn {
    lifetime: bool,
    generics: Vec<String>,
    params: Vec<String>,
    gpu_inits: Vec<String>,
    buffer_accesses: Vec<String>,
    image_accesses: Vec<String>,
}

fn build_new_fn(pc: &TypeLayout, structs: &mut Structs, stage: &str) -> NewFn {
    let mut out = NewFn {
        lifetime: false,
        generics: Vec::new(),
        params: Vec::new(),
        gpu_inits: Vec::new(),
        buffer_accesses: Vec::new(),
        image_accesses: Vec::new(),
    };

    for f in pc.fields() {
        let name = field_name(f);
        let t = f.type_layout().unwrap();
        let suffix = camel_case(name);
        match field_kind(t) {
            FieldKind::Image { usage } => {
                out.lifetime = true;
                let format_bound = match image_texel(t) {
                    Some((texel, components)) => {
                        format!("F{suffix}: Format<Texels = [{texel}; {components}]>")
                    }
                    None => format!("F{suffix}: Format"),
                };
                out.generics.push(format_bound);
                out.generics.push(format!(
                    "U{suffix}: {}",
                    match usage {
                        0 => "image::usage::IsSampled",
                        _ => "image::usage::IsStorage",
                    }
                ));
                out.params
                    .push(format!("{name}: ImageView<'a, F{suffix}, U{suffix}>"));
                out.gpu_inits.push(format!("{name}: {name}.handle"));
                out.image_accesses.push(format!(
                    "{name}.access({stage}, {access}, vk::ImageLayout::GENERAL)",
                    access = image_access(usage)
                ));
            }
            FieldKind::Buffer => {
                out.lifetime = true;
                let ptr = t.field_by_index(0).unwrap().type_layout().unwrap();
                let inner = rust_type(ptr.element_type_layout().unwrap(), structs);
                let ty = format!("BufferSlice<'a, {inner}, U{suffix}>");
                out.generics
                    .push(format!("U{suffix}: buffer::usage::IsStorage"));
                out.params.push(format!("{name}: {ty}"));
                out.gpu_inits.push(format!("{name}: {name}.gpu_ptr"));
                let access = if type_name(t) == "MutBuf" {
                    "vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE"
                } else {
                    "vk::AccessFlags2::SHADER_STORAGE_READ"
                };
                out.buffer_accesses
                    .push(format!("{name}.access({stage}, {access})"));
            }
            FieldKind::Plain => {
                let ty = rust_type(t, structs);
                out.params.push(format!("{name}: {ty}"));
                out.gpu_inits.push(name.to_string());
            }
        }
    }

    out
}

fn generate_pass(
    pass_name: &str,
    pc: &TypeLayout,
    entries: &[(&'static str, String)],
    structs: &mut Structs,
    pass_map: &mut Vec<String>,
    num_compute_pipelines: usize,
    num_ray_tracing_pipelines: usize,
    num_raster_pipelines: usize,
    path_dir: &PathBuf,
    source: &str,
) -> (String, PassKind) {
    let gpu_name = format!("C{pass_name}Bindings");
    let index = pass_map.len();

    let entry = |stage: &str| {
        entries
            .iter()
            .find(|(s, _)| *s == stage)
            .map(|(_, n)| n.as_str())
    };
    let mut stages: Vec<&str> = entries.iter().map(|(s, _)| *s).collect();
    stages.sort_unstable();

    let kind = match stages.as_slice() {
        ["fragment", "vertex"] => PassKind::RasterVertex,
        ["fragment", "mesh"] | ["amplification", "fragment", "mesh"] => PassKind::RasterMesh,
        ["compute"] => PassKind::Compute,
        ["closest_hit", "raygen"] | ["any_hit", "closest_hit", "raygen"] => PassKind::RayTracing,
        other => panic!("entry points {other:?} in pass {pass_name} don't match any pass pattern"),
    };
    let stage = kind.stage();

    let pass_entry = match kind {
        PassKind::RasterVertex => {
            let (vertex, fragment) = (entry("vertex").unwrap(), entry("fragment").unwrap());
            format!(
                r#"    PassEntry {{
                    path: {:?},
                    source: {source:?},
                    kind: PassKind::RasterVertex {{
                        fragment: "{fragment}\0",
                        vertex: "{vertex}\0",
                    }},
                    index: {num_raster_pipelines}
                }},"#,
                path_dir
            )
        }
        PassKind::RasterMesh => {
            let (mesh, fragment) = (entry("mesh").unwrap(), entry("fragment").unwrap());
            let task = match entry("amplification") {
                Some(task) => format!("Some(\"{task}\0\")"),
                None => "None".to_string(),
            };
            format!(
                r#"    PassEntry {{
                    path: {:?},
                    source: {source:?},
                    kind: PassKind::RasterMesh {{
                        fragment: "{fragment}\0",
                        mesh: "{mesh}\0",
                        amp: {task},
                    }},
                    index: {num_raster_pipelines}
                }},"#,
                path_dir
            )
        }
        PassKind::Compute => {
            let compute = entry("compute").unwrap();
            format!(
                r#"    PassEntry {{
                    path: {:?},
                    source: {source:?},
                    kind: PassKind::Compute {{
                        entry: "{compute}\0",
                    }},
                    index: {num_compute_pipelines}
                }},"#,
                path_dir
            )
        }
        PassKind::RayTracing => {
            let (raygen, closest) = (entry("raygen").unwrap(), entry("closest_hit").unwrap());
            let any = match entry("any_hit") {
                Some(any) => format!("\"{any}\""),
                None => format!("\"{closest}\""),
            };
            format!(
                r#"    PassEntry {{
                    path: {:?},
                    source: {source:?},
                    kind: PassKind::RayTracing {{
                        ray_gen: "{raygen}\0",
                        ray_hit: "{any}\0",
                        ray_closest: "{closest}\0",
                    }},
                    index: {num_ray_tracing_pipelines}
                }},"#,
                path_dir
            )
        }
    };
    pass_map.push(pass_entry);

    let trait_impl = match kind {
        PassKind::RasterVertex => "RasterVertexPass",
        PassKind::RasterMesh => "RasterMeshPass",
        PassKind::Compute => "ComputePass",
        PassKind::RayTracing => "RaytracingPass",
    };

    let gpu_struct = generate_gpu_struct(&gpu_name, pc, structs);
    let new = build_new_fn(pc, structs, stage);

    let mut generics: Vec<String> = Vec::new();
    if new.lifetime {
        generics.push("'a".to_string());
    }
    generics.extend(new.generics.iter().cloned());
    let generics = if generics.is_empty() {
        String::new()
    } else {
        format!("<{}>", generics.join(", "))
    };

    let params = new.params.join(", ");
    let image_accesses = new.image_accesses.join(",\n");
    let buffer_accesses = new.buffer_accesses.join(",\n");
    let gpu_inits = new.gpu_inits.join(", ");
    let num_images = new.image_accesses.len();
    let num_buffers = new.buffer_accesses.len();

    let mut out = String::new();
    out.push_str(&gpu_struct);

    let base_impl = match kind {
        PassKind::RasterVertex | PassKind::RasterMesh => {
            format!("impl RasterPass for {pass_name} {{}}\n")
        }
        _ => String::new(),
    };

    out.push_str(&format!(
        r#"
pub struct {pass_name};

impl PassType for {pass_name} {{
    const PASS_INDEX: usize = {index};
}}

{base_impl}
impl {trait_impl} for {pass_name} {{}}

impl {pass_name} {{
    pub fn new{generics}({params}) -> BindingOutput<{gpu_name}, Self, {num_images}, {num_buffers}> {{
        BindingOutput {{
            images: [
                {image_accesses}
            ],
            buffers: [
                {buffer_accesses}
            ],
            gpu_bindings: {gpu_name} {{
                {gpu_inits}
            }},
            stage: {stage},
            _marker: PhantomData,
        }}
    }}
}}
"#
    ));

    (out, kind)
}

fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

fn camel_case(s: &str) -> String {
    s.split('_')
        .filter(|part| !part.is_empty())
        .map(|part| capitalize_first(part))
        .collect()
}

fn main() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let shaders = PathBuf::from(manifest_dir).join("../shaders");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    // `-lslang` can resolve to the unrelated S-Lang library in /usr/lib, which
    // comes first in the search path of the final binary.
    println!("cargo::rustc-link-lib=dylib=slang-compiler");
    println!(
        "cargo::rerun-if-changed={}",
        shaders.join("passes").display()
    );
    println!(
        "cargo::rerun-if-changed={}",
        shaders.join("include").display()
    );
    // Lava's own tests render with dedicated passes from `lava/tests/shaders`, so they don't
    // depend on the engine's shaders. Those passes are only compiled with the `test-passes`
    // feature, and the bindings then go to OUT_DIR so `src/bindings.rs` keeps describing
    // exactly the engine passes.
    let test_passes = env::var_os("CARGO_FEATURE_TEST_PASSES").is_some();
    let tests = PathBuf::from(manifest_dir).join("tests");
    if test_passes {
        println!(
            "cargo::rerun-if-changed={}",
            tests.join("shaders").display()
        );
    }

    let global = slang::GlobalSession::new().unwrap();
    let root = CString::new(shaders.to_str().unwrap()).unwrap();
    let include = CString::new(shaders.join("include").to_str().unwrap()).unwrap();
    let tests_path = CString::new(tests.to_str().unwrap()).unwrap();
    let mut search_paths = vec![root.as_ptr(), include.as_ptr()];
    if test_passes {
        search_paths.push(tests_path.as_ptr());
    }
    let targets = [slang::TargetDesc::default()
        .format(slang::CompileTarget::Spirv)
        .profile(global.find_profile("spirv_1_6"))];
    let options = CompilerOptions::default()
        .vulkan_use_entry_point_name(true)
        .matrix_layout_column(true);
    let session = global
        .create_session(
            &SessionDesc::default()
                .targets(&targets)
                .search_paths(&search_paths)
                .options(&options),
        )
        .unwrap();

    // (directory relative to a search path, file name) of every pass, engine passes first.
    let list_passes = |parent: &PathBuf, dir: &'static str| {
        let mut files: Vec<(&'static str, String)> = fs::read_dir(parent.join(dir))
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|f| f.ends_with(".slang"))
            .map(|f| (dir, f))
            .collect();
        files.sort();
        files
    };
    let mut pass_files = list_passes(&shaders, "passes");
    if test_passes {
        pass_files.extend(list_passes(&tests, "shaders"));
    }

    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let (h, m, s) = ((secs / 3600) % 24, (secs / 60) % 60, (secs % 60));
    let mut bindings = format!(
        r#"
//{h}:{m}:{s}
use std::marker::PhantomData;
use std::sync::OnceLock;
use glam::*;
use bytemuck::{{Pod, Zeroable}};
use crate::bindless::BindlessHandle;
use crate::buffer::slice::*;
use crate::image::format::*;
use crate::image::slice::*;
use crate::buffer;
use crate::image;
use crate::command_buffer::*;
use ash::vk;
use std::path::PathBuf;
use std::str::FromStr;

"#,
    );

    let mut structs = Structs::new();
    let mut entry_owner: HashMap<(&'static str, String), String> = HashMap::new();
    let mut pass_map: Vec<String> = Vec::new();
    let mut num_compute_pipelines: usize = 0;
    let mut num_ray_tracing_pipelines: usize = 0;
    let mut num_raster_pipelines: usize = 0;

    let passes_dir = out_dir.join("passes");
    fs::create_dir_all(&passes_dir).unwrap();

    for (dir, file) in pass_files.iter() {
        let source = format!("{dir}/{file}");
        let module = session
            .load_module(&source)
            .unwrap_or_else(|e| panic!("{file}:\n{e}"));

        let mut components: Vec<ComponentType> = vec![module.clone().into()];
        components.extend(module.entry_points().map(Into::into));

        let program = session
            .create_composite_component_type(&components)
            .and_then(|c| c.link())
            .unwrap_or_else(|e| panic!("{file}:\n{e}"));
        let layout = program.layout(0).unwrap();

        let pass_name: String = file
            .trim_end_matches(".slang")
            .split('_')
            .map(capitalize_first)
            .collect();

        let entries: Vec<(&'static str, String)> = layout
            .entry_points()
            .map(|ep| (stage_name(ep.stage()), ep.name().unwrap().to_string()))
            .collect();
        for (stage, name) in &entries {
            if let Some(other) = entry_owner.insert((*stage, name.clone()), file.clone()) {
                panic!(
                    "{stage} entry point `{name}` exists in both {other} and {file}; \
                     names must be unique per stage inside one SPIR-V module"
                );
            }
        }

        let pc = layout
            .parameters()
            .find(|p| p.category() == Some(Cat::PushConstantBuffer))
            .and_then(|p| p.type_layout())
            .and_then(|t| t.element_type_layout())
            .unwrap_or_else(|| panic!("no push constant buffer found in {file}"));

        let out = passes_dir.join(format!("{pass_name}.spv"));
        let (pass_code, kind) = generate_pass(
            &pass_name,
            pc,
            &entries,
            &mut structs,
            &mut pass_map,
            num_compute_pipelines,
            num_ray_tracing_pipelines,
            num_raster_pipelines,
            &out,
            &source,
        );
        match kind {
            PassKind::Compute => num_compute_pipelines += 1,
            PassKind::RayTracing => num_ray_tracing_pipelines += 1,
            PassKind::RasterVertex | PassKind::RasterMesh => num_raster_pipelines += 1,
        }
        bindings.push_str(&pass_code);

        let pass_spirv = program
            .target_code(0)
            .unwrap_or_else(|e| panic!("{file}: {e}"));
        fs::write(out, pass_spirv.as_slice()).unwrap();
    }

    bindings.push_str(&format!(
        "pub const NUM_COMPUTE_PIPELINES: usize = {};\n",
        num_compute_pipelines
    ));
    bindings.push_str(&format!(
        "pub const NUM_RAY_TRACING_PIPELINES: usize = {};\n",
        num_ray_tracing_pipelines
    ));
    bindings.push_str(&format!(
        "pub const NUM_RASTER_PIPELINES: usize = {};\n",
        num_raster_pipelines
    ));

    bindings.push_str(&format!(
        "pub const PASS_MAP: [PassEntry; {}] = [\n",
        pass_map.len()
    ));
    bindings.push_str(&pass_map.join("\n"));
    bindings.push_str("];\n");

    for (name, def) in &structs {
        bindings.push_str(&format!(
            "\n#[derive(Pod, Copy, Clone, Zeroable, Debug, Default)]\n#[repr(C)]\npub struct {name} {{\n{}}}\n{}",
            def.body, def.checks
        ));
    }

    let bindings_path = if test_passes {
        out_dir.join("bindings.rs")
    } else {
        PathBuf::from(manifest_dir).join("src/bindings.rs")
    };
    fs::write(bindings_path, bindings).unwrap();
}

fn stage_name(stage: Stage) -> &'static str {
    match stage {
        Stage::Vertex => "vertex",
        Stage::Fragment => "fragment",
        Stage::Mesh => "mesh",
        Stage::Amplification => "amplification",
        Stage::Compute => "compute",
        Stage::RayGeneration => "raygen",
        Stage::AnyHit => "any_hit",
        Stage::ClosestHit => "closest_hit",
        Stage::Miss => "miss",
        other => panic!("unsupported stage {other:?}"),
    }
}

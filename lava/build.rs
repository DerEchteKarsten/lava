//! Compiles Slang passes to SPIR-V and generates typed pass bindings from reflection
use std::{
    collections::BTreeMap,
    env, fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use shader_slang::{
    self as slang, ParameterCategory as Cat, ScalarType, Stage, TypeKind,
    reflection::{TypeLayout, VariableLayout},
};

#[path = "src/slang_compile.rs"]
mod slang_compile;

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
                "vk::PipelineStageFlags2::TASK_SHADER_EXT | vk::PipelineStageFlags2::MESH_SHADER_EXT | vk::PipelineStageFlags2::FRAGMENT_SHADER"
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

/// The push constants every pass gets, pushed whole; Vulkan 1.4 guarantees 256 bytes.
const MAX_PUSH_CONSTANTS_SIZE: usize = 256;

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
        3 => "vk::AccessFlags2::SHADER_SAMPLED_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE",
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
    // Slang rounds the size up to its alignment (8 for a trailing `uint2` and `float`), which
    // can be more than the Rust fields need.
    let align = pc.alignment(Cat::Uniform).max(1);

    format!(
        r#"
#[derive(Clone, Copy)]
#[repr(C, align({align}))]
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

fn build_new_fn<'a>(
    fields: impl Iterator<Item = (&'a str, &'a TypeLayout)>,
    structs: &mut Structs,
    stage: &str,
) -> NewFn {
    let mut out = NewFn {
        lifetime: false,
        generics: Vec::new(),
        params: Vec::new(),
        gpu_inits: Vec::new(),
        buffer_accesses: Vec::new(),
        image_accesses: Vec::new(),
    };

    for (name, t) in fields {
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
                        3 => "image::usage::UnifiedBinding",
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

/// A specialization constant of a pass, which lava bakes into its pipelines.
struct SConst<'a> {
    name: String,
    id: u32,
    layout: &'a TypeLayout,
}

/// The pass's specialization constants by id. `SCONST(T, name)` (`bindless.slang`) declares a
/// `uint64_t name_sconst` whose `SConst` attribute names `T`, a buffer or image handle; any
/// other one is a plain scalar.
fn sconsts<'a>(layout: &'a slang::reflection::Shader, file: &str) -> Vec<SConst<'a>> {
    let mut consts: Vec<SConst> = layout
        .parameters()
        .filter(|p| p.category() == Some(Cat::SpecializationConstant))
        .map(|p| (p, p.offset(Cat::SpecializationConstant) as u32))
        .map(|(p, id)| {
            let name = field_name(p);
            let attribute = p
                .variable()
                .and_then(|v| v.user_attributes().find(|a| a.name() == Some("SConst")));
            let Some(attribute) = attribute else {
                return SConst {
                    name: name.to_string(),
                    id,
                    layout: p.type_layout().unwrap(),
                };
            };
            let ty = attribute.argument_value_string(0).unwrap();
            let layout = layout
                .find_type_by_name(ty)
                .and_then(|t| layout.type_layout(t, slang::LayoutRules::Default))
                .unwrap_or_else(|| panic!("{file}: unknown type `{ty}` of SCONST `{name}`"));
            assert!(
                matches!(
                    field_kind(layout),
                    FieldKind::Buffer | FieldKind::Image { .. }
                ),
                "{file}: SCONST `{name}` has type `{ty}`, which isn't a buffer or image"
            );
            SConst {
                name: name.trim_end_matches("_sconst").to_string(),
                id,
                layout,
            }
        })
        .collect();
    consts.sort_by_key(|c| c.id);
    consts
}

/// Pipeline counts per kind; a pass's pipelines go to the specialized maps if it has
/// specialization constants.
#[derive(Default)]
struct Pipelines {
    compute: usize,
    specialized_compute: usize,
    ray_tracing: usize,
    specialized_ray_tracing: usize,
    raster: usize,
}

impl Pipelines {
    /// The pass's index into the pipeline array of its kind, counting it.
    fn index(&mut self, kind: PassKind, specialized: bool) -> usize {
        let count = match (kind, specialized) {
            (PassKind::Compute, false) => &mut self.compute,
            (PassKind::Compute, true) => &mut self.specialized_compute,
            (PassKind::RayTracing, false) => &mut self.ray_tracing,
            (PassKind::RayTracing, true) => &mut self.specialized_ray_tracing,
            (PassKind::RasterVertex | PassKind::RasterMesh, _) => &mut self.raster,
        };
        *count += 1;
        *count - 1
    }
}

/// What a pass adds to the generated bindings.
struct GeneratedPass {
    code: String,
    images: usize,
    buffers: usize,
    constants_size: String,
}

fn generate_pass(
    pass_name: &str,
    pc: &TypeLayout,
    consts: &[SConst],
    entries: &[(&'static str, String)],
    structs: &mut Structs,
    pass_map: &mut Vec<String>,
    pipelines: &mut Pipelines,
    path_dir: &PathBuf,
    source: &str,
) -> GeneratedPass {
    let gpu_name = format!("C{pass_name}Bindings");
    let consts_name = format!("C{pass_name}Constants");
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
    let pipeline_index = pipelines.index(kind, !consts.is_empty());

    let kind_fields = match kind {
        PassKind::RasterVertex => {
            let (vertex, fragment) = (entry("vertex").unwrap(), entry("fragment").unwrap());
            format!("RasterVertex {{ fragment: \"{fragment}\\0\", vertex: \"{vertex}\\0\" }}")
        }
        PassKind::RasterMesh => {
            let (mesh, fragment) = (entry("mesh").unwrap(), entry("fragment").unwrap());
            let task = match entry("amplification") {
                Some(task) => format!("Some(\"{task}\\0\")"),
                None => "None".to_string(),
            };
            format!(
                "RasterMesh {{ fragment: \"{fragment}\\0\", mesh: \"{mesh}\\0\", amp: {task} }}"
            )
        }
        PassKind::Compute => {
            let compute = entry("compute").unwrap();
            format!("Compute {{ entry: \"{compute}\\0\" }}")
        }
        PassKind::RayTracing => {
            let (raygen, closest) = (entry("raygen").unwrap(), entry("closest_hit").unwrap());
            let any = entry("any_hit").unwrap_or(closest);
            format!(
                "RayTracing {{ ray_gen: \"{raygen}\\0\", ray_any: \"{any}\\0\", ray_closest: \"{closest}\\0\" }}"
            )
        }
    };
    let (constants, constants_size) = if consts.is_empty() {
        ("&[]".to_string(), "0".to_string())
    } else {
        let entries: Vec<String> = consts
            .iter()
            .map(|c| {
                format!(
                    "vk::SpecializationMapEntry {{ constant_id: {id}, offset: core::mem::offset_of!({consts_name}, {name}) as u32, size: size_of::<{ty}>() }}",
                    id = c.id,
                    name = c.name,
                    ty = gpu_type(c.layout, structs),
                )
            })
            .collect();
        (
            format!("&[{}]", entries.join(", ")),
            format!("size_of::<{consts_name}>()"),
        )
    };
    pass_map.push(format!(
        r#"    PassEntry {{
        name: "{pass_name}",
        path: {path_dir:?},
        source: {source:?},
        kind: PassKind::{kind_fields},
        index: {pipeline_index},
        constants: {constants},
        constants_size: {constants_size},
    }},"#
    ));

    let kind_marker = match kind {
        PassKind::RasterVertex => "RasterVertex",
        PassKind::RasterMesh => "RasterMesh",
        PassKind::Compute => "Compute",
        PassKind::RayTracing => "RayTracing",
    };
    let size = pc.size(Cat::Uniform);
    assert!(
        size <= MAX_PUSH_CONSTANTS_SIZE,
        "the push constants of {source} take {size} bytes, more than the {MAX_PUSH_CONSTANTS_SIZE} every pass gets"
    );

    let gpu_struct = generate_gpu_struct(&gpu_name, pc, structs);
    let push = build_new_fn(
        pc.fields()
            .map(|f| (field_name(f), f.type_layout().unwrap())),
        structs,
        stage,
    );
    let output = format!("BindingOutput<kind::{kind_marker}>");
    let incomplete = format!("Incomplete{pass_name}");

    let binding_output = format!(
        r#"BindingOutput::new(
            {index},
            &{gpu_name} {{
                {gpu_inits}
            }},
            {stage},
            [
                {image_accesses}
            ],
            [
                {buffer_accesses}
            ],
        )"#,
        gpu_inits = push.gpu_inits.join(", "),
        image_accesses = push.image_accesses.join(",\n"),
        buffer_accesses = push.buffer_accesses.join(",\n"),
    );
    let (returns, body) = if consts.is_empty() {
        (output.clone(), binding_output)
    } else {
        (
            incomplete.clone(),
            format!("{incomplete}({binding_output})"),
        )
    };

    let mut out = gpu_struct;
    out.push_str(&format!(
        r#"
pub struct {pass_name};

impl {pass_name} {{
    pub fn push_bindings{generics}({params}) -> {returns} {{
        {body}
    }}
}}
"#,
        generics = fn_generics(&push),
        params = push.params.join(", "),
    ));
    let (mut images, mut buffers) = (push.image_accesses.len(), push.buffer_accesses.len());

    if !consts.is_empty() {
        let new = build_new_fn(
            consts.iter().map(|c| (c.name.as_str(), c.layout)),
            structs,
            stage,
        );
        let fields: Vec<String> = consts
            .iter()
            .map(|c| format!("    pub {}: {},", c.name, gpu_type(c.layout, structs)))
            .collect();
        out.push_str(&format!(
            r#"
/// Packed: the specialization constant data is read at each entry's offset.
#[derive(Clone, Copy)]
#[repr(C, packed)]
pub struct {consts_name} {{
{fields}
}}
unsafe impl bytemuck::Pod for {consts_name} {{}}
unsafe impl bytemuck::Zeroable for {consts_name} {{}}

#[must_use = "`{pass_name}` has specialization constants: call `.constant_bindings(..)`"]
pub struct {incomplete}({output});

impl {incomplete} {{
    pub fn constant_bindings{generics}(self, {params}) -> {output} {{
        self.0.with_constants(
            &{consts_name} {{
                {gpu_inits}
            }},
            [
                {image_accesses}
            ],
            [
                {buffer_accesses}
            ],
        )
    }}
}}
"#,
            fields = fields.join("\n"),
            generics = fn_generics(&new),
            params = new.params.join(", "),
            gpu_inits = new.gpu_inits.join(", "),
            image_accesses = new.image_accesses.join(",\n"),
            buffer_accesses = new.buffer_accesses.join(",\n"),
        ));
        images += new.image_accesses.len();
        buffers += new.buffer_accesses.len();
    }

    GeneratedPass {
        code: out,
        images,
        buffers,
        constants_size,
    }
}

/// `<'a, F…, U…>` of a generated function, empty without parameters that borrow.
fn fn_generics(new: &NewFn) -> String {
    let mut generics: Vec<String> = Vec::new();
    if new.lifetime {
        generics.push("'a".to_string());
    }
    generics.extend(new.generics.iter().cloned());
    if generics.is_empty() {
        String::new()
    } else {
        format!("<{}>", generics.join(", "))
    }
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
    let shaders = PathBuf::from(slang_compile::SHADER_DIR);
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
    let tests = PathBuf::from(slang_compile::TEST_DIR);
    if test_passes {
        println!(
            "cargo::rerun-if-changed={}",
            tests.join("shaders").display()
        );
    }

    let global = slang::GlobalSession::new().unwrap();
    let profiling = env::var_os("CARGO_FEATURE_PROFILING").is_some();

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
    let mut pass_map: Vec<String> = Vec::new();
    let mut pipelines = Pipelines::default();
    let (mut max_images, mut max_buffers) = (0, 0);
    let mut constants_sizes = vec!["0".to_string()];

    let passes_dir = out_dir.join("passes");
    fs::create_dir_all(&passes_dir).unwrap();

    for (dir, file) in pass_files.iter() {
        let source = format!("{dir}/{file}");
        let options = slang_compile::Options {
            test_passes,
            profiling,
            pass_index: pass_map.len(),
        };
        let session = slang_compile::session(&global, &options);
        let program =
            slang_compile::link(&session, &source).unwrap_or_else(|e| panic!("{file}:\n{e}"));
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

        let pc = layout
            .parameters()
            .find(|p| p.category() == Some(Cat::PushConstantBuffer))
            .and_then(|p| p.type_layout())
            .and_then(|t| t.element_type_layout())
            .unwrap_or_else(|| panic!("no push constant buffer found in {file}"));

        let consts = sconsts(layout, file);

        let out = passes_dir.join(format!("{pass_name}.spv"));
        let pass = generate_pass(
            &pass_name,
            pc,
            &consts,
            &entries,
            &mut structs,
            &mut pass_map,
            &mut pipelines,
            &out,
            &source,
        );
        max_images = max_images.max(pass.images);
        max_buffers = max_buffers.max(pass.buffers);
        constants_sizes.push(pass.constants_size);
        bindings.push_str(&pass.code);

        let pass_spirv = program
            .target_code(0)
            .unwrap_or_else(|e| panic!("{file}: {e}"));
        fs::write(out, pass_spirv.as_slice()).unwrap();
    }

    let Pipelines {
        compute,
        specialized_compute,
        ray_tracing,
        specialized_ray_tracing,
        raster,
    } = pipelines;
    bindings.push_str(&format!(
        "pub const NUM_COMPUTE_PIPELINES: usize = {compute};\n\
         pub const NUM_SPECIALIZED_COMPUTE_PIPELINES: usize = {specialized_compute};\n\
         pub const NUM_RAY_TRACING_PIPELINES: usize = {ray_tracing};\n\
         pub const NUM_SPECIALIZED_RAY_TRACING_PIPELINES: usize = {specialized_ray_tracing};\n\
         pub const NUM_RASTER_PIPELINES: usize = {raster};\n"
    ));
    // Every pipeline cache is keyed by this many bytes of specialization constants.
    bindings.push_str(&format!(
        r#"pub const MAX_SPECIALIZATION_CONSTANTS_SIZE: usize = {{
    let sizes = [{}];
    let (mut max, mut i) = (0, 0);
    while i < sizes.len() {{
        if sizes[i] > max {{
            max = sizes[i];
        }}
        i += 1;
    }}
    max
}};
const _: () = assert!(MAX_SPECIALIZATION_CONSTANTS_SIZE < 128, "Keep SConstants small!!");
"#,
        constants_sizes.join(", ")
    ));
    // The inline capacity of a pass's accesses; the margin covers the buffers of a draw and
    // registered accesses, and more only spill to the heap.
    bindings.push_str(&format!(
        "pub const MAX_PUSH_CONSTANTS_SIZE: usize = {MAX_PUSH_CONSTANTS_SIZE};\n\
         pub const MAX_PASS_IMAGES: usize = {};\n\
         pub const MAX_PASS_BUFFERS: usize = {};\n",
        max_images + 2,
        max_buffers + 2
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

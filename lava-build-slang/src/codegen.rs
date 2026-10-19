//! Generates the typed pass bindings and `PASS_MAP` from the reflection of each compiled pass, and what the hot reload compiles them with again
use std::{collections::BTreeMap, env, fs, path::Path};

use shader_slang::{
    self as slang, ParameterCategory as Cat, ScalarType, TypeKind,
    reflection::{TypeLayout, VariableLayout},
};

use crate::reflect::{
    CheckedAttributes, CheckedGeneric, CheckedName, CheckedShader, CheckedTypeLayout,
    CheckedVariableLayout,
};
use crate::{SLANG_DIR, Shaders, compile};

struct StructDef {
    body: String,
    checks: String,
    signature: Vec<(String, usize)>,
    size: usize,
    /// It mirrors a lava type (`MIRRORED_TYPES`), which isn't generated, only checked.
    mirror: bool,
    impls: String,
}

#[derive(Clone, Copy)]
enum PassKind {
    Compute,
    RayTracing,
    RasterVertex,
    RasterMesh,
}

enum FieldKind {
    Plain,
    Buffer,
    Image { usage: u64 },
    Accel,
}

type Structs = BTreeMap<String, StructDef>;

/// A copy of `lava::pipeline::MAX_PUSH_CONSTANTS_SIZE`.
const MAX_PUSH_CONSTANTS_SIZE: usize = 256;

const IMAGE_TYPES: [&str; 1] = ["Img"];
const BUFFER_TYPES: [&str; 2] = ["Buf", "MutBuf"];
const HANDLE_TYPES: [&str; 2] = ["Img", "DynImg"];
const ACCEL_TYPES: [&str; 1] = ["Accel"];
/// (Slang name, Rust path) of the Slang structs that mirror a lava type.
const MIRRORED_TYPES: [(&str, &str); 2] = [
    (
        "AccelerationStructureInstance",
        "lava::vkobjects::acceleration_structure::Instance",
    ),
    ("Sampler", "lava::bindless::SamplerHandle"),
];

const CLOCK_TYPE: &str = "Clock";
/// Copies of `CLOCK_STRIPE_WORDS` and `CLOCK_TICK_BITS` in `slang/profile.slang`.
const CLOCK_STRIPE_WORDS: usize = 8;
const CLOCK_TICK_BITS: u32 = 40;


fn field_name(f: &VariableLayout) -> &str {
    f.variable()
        .and_then(|v| v.checked_name())
        .expect("unnamed field")
}

fn type_name(t: &TypeLayout) -> &str {
    t.checked_name().unwrap_or("UnknownStruct")
}

fn is_struct_named(t: &TypeLayout, names: &[&str]) -> bool {
    t.kind() == TypeKind::Struct && names.contains(&type_name(t))
}

fn layout_checks(rust_name: &str, t: &TypeLayout) -> String {
    let mut out = String::new();
    for f in t.checked_fields() {
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
            rust_type(t.checked_element_type_layout().unwrap(), structs),
            t.element_count().unwrap()
        ),
        TypeKind::Struct => {
            let name = type_name(t).to_string();
            let signature: Vec<(String, usize)> = t
                .checked_fields()
                .map(|f| (field_name(f).to_string(), f.offset(Cat::Uniform)))
                .collect();
            let size = t.size(Cat::Uniform);
            if let Some(&(_, rust)) = MIRRORED_TYPES.iter().find(|(slang, _)| *slang == name) {
                structs.entry(name).or_insert_with(|| StructDef {
                    body: String::new(),
                    checks: layout_checks(rust, t),
                    signature,
                    size,
                    mirror: true,
                    impls: String::new(),
                });
                return rust.to_string();
            }

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
                    mirror: false,
                    impls: String::new(),
                },
            );
            let mut body = String::new();
            for f in t.checked_fields() {
                body.push_str(&format!(
                    "    pub {}: {},\n",
                    field_name(f),
                    gpu_type(f.checked_type_layout().unwrap(), structs)
                ));
            }
            // The clock readout: `take` on `Clock`, `regions` on a struct of only `Clock`s.
            let impls = if name == CLOCK_TYPE {
                format!(
                    r#"
impl {CLOCK_TYPE} {{
    /// The ticks and subgroups `CLOCK_REGION` added since the last take, leaving the clock
    /// zeroed. Only while no submitted work can still add to it.
    pub fn take(&mut self) -> (u64, u64) {{
        let stripes = self.counters.iter_mut().step_by({CLOCK_STRIPE_WORDS});
        let sum: u64 = stripes.map(core::mem::take).sum();
        (sum & ((1 << {CLOCK_TICK_BITS}) - 1), sum >> {CLOCK_TICK_BITS})
    }}
}}
"#
                )
            } else {
                let fields: Vec<&str> = t.checked_fields().map(field_name).collect();
                if fields.is_empty()
                    || !t
                        .checked_fields()
                        .all(|f| is_struct_named(f.checked_type_layout().unwrap(), &[CLOCK_TYPE]))
                {
                    String::new()
                } else {
                    let regions: Vec<String> = fields
                        .iter()
                        .map(|f| format!("({f:?}, &mut self.{f})"))
                        .collect();
                    format!(
                        r#"
impl {name} {{
    /// Every region by its field name, in declaration order.
    pub fn regions(&mut self) -> [(&'static str, &mut {CLOCK_TYPE}); {n}] {{
        [{regions}]
    }}
}}
"#,
                        n = fields.len(),
                        regions = regions.join(", "),
                    )
                }
            };
            let def = structs.get_mut(&name).unwrap();
            def.body = body;
            def.checks = layout_checks(&name, t);
            def.impls = impls;
            name
        }
        TypeKind::Pointer => {
            if let Some(pointee) = t.checked_element_type_layout() {
                rust_type(pointee, structs);
            }
            "u64".into()
        }
        other => format!("compile_error!(\"Unsupported Type {other:?}\")"),
    }
}

/// The `ImageHandle` kind of an `Img` of kind `usage` (its `U`).
fn handle_kind(usage: u64) -> &'static str {
    match usage {
        0 => "Sampled",
        3 => "Combined",
        _ => "Storage",
    }
}

fn gpu_type(t: &TypeLayout, structs: &mut Structs) -> String {
    if let FieldKind::Image { usage } = field_kind(t) {
        format!("ImageHandle<bindless::{}>", handle_kind(usage))
    } else if is_struct_named(t, &HANDLE_TYPES) {
        "ImageHandle".into()
    } else if is_struct_named(t, &ACCEL_TYPES) {
        "u64".into()
    } else if is_struct_named(t, &BUFFER_TYPES) {
        rust_type(
            t.field_by_index(0).unwrap().checked_type_layout().unwrap(),
            structs,
        );
        "u64".into()
    } else {
        rust_type(t, structs)
    }
}

/// A specialization constant's Rust type: an `SCONST` is a `u64` whatever its handle type.
fn constant_type(t: &TypeLayout, structs: &mut Structs) -> String {
    match field_kind(t) {
        FieldKind::Image { .. } => "u64".into(),
        _ => gpu_type(t, structs),
    }
}

fn field_kind(t: &TypeLayout) -> FieldKind {
    if is_struct_named(t, &IMAGE_TYPES) {
        let generics = t.ty().unwrap().generic_container().unwrap();
        let usage = generics
            .checked_value_parameters()
            .find(|vp| vp.checked_name() == Some("U"))
            .map(|vp| generics.concrete_int_val(vp) as u64)
            .unwrap_or(0);
        FieldKind::Image { usage }
    } else if is_struct_named(t, &BUFFER_TYPES) {
        FieldKind::Buffer
    } else if is_struct_named(t, &ACCEL_TYPES) {
        FieldKind::Accel
    } else {
        FieldKind::Plain
    }
}

fn reject_nested_handles(field: &str, t: &TypeLayout, source: &str) {
    let nested = |t: &TypeLayout| {
        is_struct_named(t, &HANDLE_TYPES)
            || is_struct_named(t, &BUFFER_TYPES)
            || is_struct_named(t, &ACCEL_TYPES)
    };
    match t.kind() {
        TypeKind::Array => {
            let element = t.checked_element_type_layout().unwrap();
            if nested(element) {
                panic!(
                    "{source}: push constant `{field}` is an array of `{}` handles, which lava \
                     can't track: give each its own field",
                    type_name(element)
                );
            }
            reject_nested_handles(field, element, source);
        }
        TypeKind::Struct if !MIRRORED_TYPES.iter().any(|(slang, _)| *slang == type_name(t)) => {
            for f in t.checked_fields() {
                let inner = f.checked_type_layout().unwrap();
                let path = format!("{field}.{}", field_name(f));
                if nested(inner) {
                    panic!(
                        "{source}: push constant `{path}` is a `{}` inside a struct, which lava \
                         can't track: make it a field of the push-constant struct itself",
                        type_name(inner)
                    );
                }
                reject_nested_handles(&path, inner, source);
            }
        }
        _ => {}
    }
}

struct NewFn {
    params: Vec<String>,
    gpu_inits: Vec<String>,
    buffer_accesses: Vec<String>,
    image_accesses: Vec<String>,
    /// The image parameters' `.view()`s and the runtime format and usage checks.
    checks: Vec<String>,
}

/// `constants`: the fields are specialization constants, whose image handles are `u64`s.
fn build_new_fn<'a>(
    fields: impl Iterator<Item = (&'a str, &'a TypeLayout)>,
    structs: &mut Structs,
    stage: &str,
    constants: bool,
) -> NewFn {
    let mut out = NewFn {
        params: Vec::new(),
        gpu_inits: Vec::new(),
        buffer_accesses: Vec::new(),
        image_accesses: Vec::new(),
        checks: Vec::new(),
    };

    for (name, t) in fields {
        match field_kind(t) {
            FieldKind::Image { usage } => {
                let generics = t.ty().unwrap().generic_container().unwrap();
                let concrete = |parameter: &str| {
                    generics
                        .checked_type_parameters()
                        .find(|tp| tp.checked_name() == Some(parameter))
                        .and_then(|tp| generics.checked_concrete_type(tp))
                };
                let dim = match concrete("Shape").and_then(|shape| shape.checked_name()) {
                    Some("__Shape2D") => "image::D2",
                    Some("__Shape3D") => "image::D3",
                    other => panic!("image shape {other:?} has no Rust dimension"),
                };
                let mut bounds = format!("Dim = {dim}");
                out.checks.push(format!("let {name} = {name}.view();"));
                let texel = concrete("E").and_then(|element| {
                    let scalar = match element.scalar_type() {
                        ScalarType::Float32 => "f32",
                        ScalarType::Uint32 => "u32",
                        ScalarType::Int32 => "i32",
                        ScalarType::Uint8 => "u8",
                        ScalarType::Float64 => "f64",
                        ScalarType::Uint64 => "u64",
                        ScalarType::Int64 => "i64",
                        _ => return None,
                    };
                    let components = match element.kind() {
                        TypeKind::Vector => element.element_count(),
                        _ => 1,
                    };
                    Some((scalar, components))
                });
                if let Some((texel, components)) = texel {
                    let texels = format!("[{texel}; {components}]");
                    out.checks
                        .push(format!("{name}.texel_check::<{texels}>();"));
                    bounds.push_str(&format!(", Format: HasTexels<{texels}>"));
                }
                let (traits, flags) = match usage {
                    0 => (format!("image::SampledView<{bounds}>"), "SAMPLED"),
                    3 => (
                        format!("image::SampledView<{bounds}> + image::StorageView"),
                        "SAMPLED | vk::ImageUsageFlags::STORAGE",
                    ),
                    _ => (format!("image::StorageView<{bounds}>"), "STORAGE"),
                };
                out.checks
                    .push(format!("{name}.usage_check(vk::ImageUsageFlags::{flags});"));
                out.params.push(format!("{name}: impl {traits} + 'a"));
                // A specialization constant is a `u64`, the handle zero-extended.
                let bits = if constants { ".bits() as u64" } else { "" };
                out.gpu_inits.push(format!(
                    "{name}: {name}.handle::<bindless::{}>(){bits}",
                    handle_kind(usage)
                ));
                let access = match usage {
                    0 => "vk::AccessFlags2::SHADER_SAMPLED_READ",
                    1 => "vk::AccessFlags2::SHADER_STORAGE_READ",
                    3 => "vk::AccessFlags2::SHADER_SAMPLED_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE",
                    _ => "vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE",
                };
                out.image_accesses.push(format!(
                    "{name}.access({stage}, {access}, vk::ImageLayout::GENERAL)"
                ));
            }
            FieldKind::Buffer => {
                let ptr = t.field_by_index(0).unwrap().checked_type_layout().unwrap();
                let inner = rust_type(ptr.checked_element_type_layout().unwrap(), structs);
                let ty = format!("BufferSlice<'a, {inner}>");
                out.params.push(format!("{name}: {ty}"));
                out.gpu_inits.push(format!("{name}: {name}.address()"));
                let access = if type_name(t) == "MutBuf" {
                    "vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE"
                } else {
                    "vk::AccessFlags2::SHADER_STORAGE_READ"
                };
                out.buffer_accesses
                    .push(format!("{name}.access({stage}, {access})"));
            }
            FieldKind::Accel => {
                out.params.push(format!(
                    "{name}: &'a lava::vkobjects::acceleration_structure::Tlas"
                ));
                out.gpu_inits.push(format!("{name}: {name}.address()"));
                for i in 0..2 {
                    out.buffer_accesses
                        .push(format!("{name}.trace_accesses({stage})[{i}].clone()"));
                }
            }
            FieldKind::Plain => {
                // A `DynImg` is a plain `ImageHandle`, whose image the command's accesses name.
                let ty = gpu_type(t, structs);
                out.params.push(format!("{name}: {ty}"));
                out.gpu_inits.push(name.to_string());
            }
        }
    }

    out
}

struct SConst<'a> {
    name: String,
    id: u32,
    layout: &'a TypeLayout,
}

/// Pipeline counts per kind; a pass with specialization constants goes to the specialized maps.
#[derive(Default)]
struct Pipelines {
    compute: usize,
    specialized_compute: usize,
    ray_tracing: usize,
    specialized_ray_tracing: usize,
    raster: usize,
}

/// `<'a>`, the lifetime the views, slices and result share, where `lifetime` asks for it.
fn fn_generics(lifetime: bool) -> &'static str {
    if lifetime { "<'a>" } else { "" }
}

fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

/// Only a changed file makes cargo rebuild the crate that includes it.
fn write_if_changed(path: &Path, bytes: &[u8]) {
    if fs::read(path).ok().as_deref() != Some(bytes) {
        fs::write(path, bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    }
}

impl Shaders {
    /// Compiles every pass into `$OUT_DIR/lava/passes/<Pass>.spv` and writes the bindings to
    /// `$OUT_DIR/lava/bindings.rs`. Panics with the file on a compile error.
    pub fn compile(self) {
        for dir in self.passes.iter().chain(&self.includes).map(String::as_str) {
            println!("cargo::rerun-if-changed={dir}");
        }
        println!("cargo::rerun-if-changed={SLANG_DIR}");
        let out_dir =
            Path::new(&env::var("OUT_DIR").expect("called from a build script")).join("lava");
        let global = slang::GlobalSession::new().unwrap();

        // Each directory's files sorted, directories in call order.
        let mut pass_files: Vec<String> = Vec::new();
        for dir in &self.passes {
            let mut files: Vec<String> = fs::read_dir(dir)
                .unwrap_or_else(|e| panic!("{dir}: {e}"))
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|path| path.extension().is_some_and(|ext| ext == "slang"))
                .map(|path| path.to_str().expect("shader paths are UTF-8").to_string())
                .collect();
            files.sort();
            pass_files.extend(files);
        }

        let mut bindings = format!(
            r#"
#[allow(unused_imports)]
use glam::*;
#[allow(unused_imports)]
use bytemuck::{{Pod, Zeroable}};
#[allow(unused_imports)]
use lava::state::raw_vulkan as vk;
#[allow(unused_imports)]
use lava::bindless::{{self, ImageHandle}};
#[allow(unused_imports)]
use lava::buffer::slice::*;
#[allow(unused_imports)]
use lava::image::format::*;
#[allow(unused_imports)]
use lava::image::slice::*;
#[allow(unused_imports)]
use lava::image;
#[allow(unused_imports)]
use lava::command_buffer::*;
#[allow(unused_imports)]
use lava::pipeline::*;
"#
        );

        let mut structs = Structs::new();
        let mut pass_map: Vec<String> = Vec::new();
        let mut built_passes: Vec<String> = Vec::new();
        let mut pipelines = Pipelines::default();

        let passes_dir = out_dir.join("passes");
        fs::create_dir_all(&passes_dir).unwrap();
        let search_paths: Vec<&str> = self.includes.iter().map(String::as_str).collect();
        let defines: Vec<(&str, &str)> = self
            .defines
            .iter()
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect();

        for source in &pass_files {
            let options = compile::Options {
                search_paths: &search_paths,
                defines: &defines,
                optimization: self.optimization,
            };
            let session = compile::session(&global, &options);
            let program =
                compile::link(&session, source).unwrap_or_else(|e| panic!("{source}:\n{e}"));
            let layout = program.layout(0).unwrap();

            let file = Path::new(source).file_name().unwrap().to_str().unwrap();
            let pass_name: String = file
                .trim_end_matches(".slang")
                .split('_')
                .map(capitalize_first)
                .collect();

            let entries: Vec<compile::Entry> = layout
                .checked_entry_points()
                .map(|ep| compile::Entry {
                    stage: compile::stage_name(ep.stage()),
                    name: ep.checked_name().unwrap(),
                    group: compile::entry_attribute(ep, "Group"),
                    max_recursion: compile::entry_attribute(ep, "MaxRecursion"),
                })
                .collect();
            let pass_spirv = program
                .target_code(0)
                .unwrap_or_else(|e| panic!("{source}: {e}"));

            let pc = layout
                .checked_parameters()
                .find(|p| p.category() == Cat::PushConstantBuffer)
                .and_then(|p| p.checked_type_layout())
                .and_then(|t| t.checked_element_type_layout())
                .unwrap_or_else(|| panic!("no push constant buffer found in {source}"));

            // `SCONST(T, name)` (`bindless.slang`) declares `name_sconst` with an `SConst`
            // attribute naming `T`.
            let mut consts: Vec<SConst> = layout
                .checked_parameters()
                .filter(|p| p.category() == Cat::SpecializationConstant)
                .map(|p| (p, p.offset(Cat::SpecializationConstant) as u32))
                .map(|(p, id)| {
                    let name = field_name(p);
                    let attribute = p.variable().and_then(|v| {
                        v.checked_user_attributes()
                            .find(|a| a.checked_name() == Some("SConst"))
                    });
                    let Some(attribute) = attribute else {
                        return SConst {
                            name: name.to_string(),
                            id,
                            layout: p.checked_type_layout().unwrap(),
                        };
                    };
                    let ty = attribute.argument_value_string(0).unwrap();
                    let layout = layout
                        .find_type_by_name(ty)
                        .and_then(|t| layout.type_layout(t, slang::LayoutRules::Default))
                        .unwrap_or_else(|| panic!("{source}: unknown type `{ty}` of SCONST `{name}`"));
                    assert!(
                        matches!(
                            field_kind(layout),
                            FieldKind::Buffer | FieldKind::Image { .. }
                        ),
                        "{source}: SCONST `{name}` has type `{ty}`, which isn't a buffer or image handle \
                         (acceleration structure addresses change from run to run, so they never go \
                         into specialization constants)"
                    );
                    SConst {
                        name: name.trim_end_matches("_sconst").to_string(),
                        id,
                        layout,
                    }
                })
                .collect();
            consts.sort_by_key(|c| c.id);

            for p in layout.checked_parameters() {
                let category = p.category();
                if matches!(category, Cat::PushConstantBuffer | Cat::SpecializationConstant) {
                    continue;
                }
                // The descriptor heaps shaders index aren't parameters.
                let name = p.variable().and_then(|v| v.checked_name()).unwrap_or("?");
                panic!(
                    "{source}: global parameter `{name}` ({category:?}): a pass has no descriptors or \
                     uniforms of its own; reach resources through push-constant handles (`Img`, \
                     `Buf`, `MutBuf`, `Accel`) instead"
                );
            }

            let spirv = passes_dir.join(format!("{pass_name}.spv"));
            let gpu_name = format!("C{pass_name}Bindings");
            let consts_name = format!("C{pass_name}Constants");
            let index = pass_map.len();

            let entry = |stage: &str| entries.iter().find(|e| e.stage == stage).map(|e| e.name);
            let mut stages: Vec<&str> = entries.iter().map(|e| e.stage).collect();
            stages.sort_unstable();

            let kind = match stages.as_slice() {
                ["fragment", "vertex"] => PassKind::RasterVertex,
                ["fragment", "mesh"] | ["amplification", "fragment", "mesh"] => {
                    PassKind::RasterMesh
                }
                ["compute"] => PassKind::Compute,
                stages
                    if stages.contains(&"raygen")
                        && stages
                            .iter()
                            .all(|s| compile::RAY_TRACING_STAGES.contains(s)) =>
                {
                    PassKind::RayTracing
                }
                other => panic!(
                    "entry points {other:?} in pass {pass_name} don't match any pass pattern"
                ),
            };
            let stage = match kind {
                PassKind::Compute => "vk::PipelineStageFlags2::COMPUTE_SHADER",
                PassKind::RayTracing => "vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR",
                PassKind::RasterVertex => {
                    "vk::PipelineStageFlags2::VERTEX_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER"
                }
                PassKind::RasterMesh => {
                    "vk::PipelineStageFlags2::TASK_SHADER_EXT | vk::PipelineStageFlags2::MESH_SHADER_EXT | vk::PipelineStageFlags2::FRAGMENT_SHADER"
                }
            };
            let count = match (kind, !consts.is_empty()) {
                (PassKind::Compute, false) => &mut pipelines.compute,
                (PassKind::Compute, true) => &mut pipelines.specialized_compute,
                (PassKind::RayTracing, false) => &mut pipelines.ray_tracing,
                (PassKind::RayTracing, true) => &mut pipelines.specialized_ray_tracing,
                (PassKind::RasterVertex | PassKind::RasterMesh, _) => &mut pipelines.raster,
            };
            *count += 1;
            let pipeline_index = *count - 1;

            let kind_fields = match kind {
                PassKind::RasterVertex => {
                    let (vertex, fragment) =
                        (entry("vertex").unwrap(), entry("fragment").unwrap());
                    format!(
                        "RasterVertex {{ fragment: \"{fragment}\\0\", vertex: \"{vertex}\\0\" }}"
                    )
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
                    let groups = compile::ray_tracing_groups(&entries)
                        .unwrap_or_else(|e| panic!("{source}: {e}"));
                    let name = |name: &str| format!("\"{name}\\0\"");
                    let optional = |entry: &Option<String>| match entry {
                        Some(entry) => format!("Some({})", name(entry)),
                        None => "None".to_string(),
                    };
                    let names = |entries: &[String]| {
                        entries
                            .iter()
                            .map(|e| name(e))
                            .collect::<Vec<_>>()
                            .join(", ")
                    };
                    let hit: Vec<String> = groups
                        .hit
                        .iter()
                        .map(|g| {
                            format!(
                                "HitGroup {{ closest_hit: {}, any_hit: {}, intersection: {} }}",
                                optional(&g.closest_hit),
                                optional(&g.any_hit),
                                optional(&g.intersection)
                            )
                        })
                        .collect();
                    format!(
                        "RayTracing {{ raygen: {}, miss: &[{}], hit: &[{}], callable: &[{}], max_recursion: {} }}",
                        name(&groups.raygen),
                        names(&groups.miss),
                        hit.join(", "),
                        names(&groups.callable),
                        groups.max_recursion,
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
                            ty = constant_type(c.layout, &mut structs),
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
        spirv: include_bytes!({spirv:?}),
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
                "the push constants of {source} take {size} bytes, more than the {MAX_PUSH_CONSTANTS_SIZE} a pass can have"
            );

            // Every gap Slang leaves, up to its size, is a zeroed `_pad` field, so the struct
            // has no padding bytes and derives `Pod`.
            let mut gaps = Vec::new();
            let mut end = 0;
            for f in pc.checked_fields() {
                let offset = f.offset(Cat::Uniform);
                if offset > end {
                    gaps.push((end, offset - end));
                }
                end = offset + f.checked_type_layout().unwrap().size(Cat::Uniform);
            }
            if size > end {
                gaps.push((end, size - end));
            }
            let mut fields: Vec<(usize, String)> = pc
                .checked_fields()
                .map(|f| {
                    let line = format!(
                        "    pub {}: {},",
                        field_name(f),
                        gpu_type(f.checked_type_layout().unwrap(), &mut structs)
                    );
                    (f.offset(Cat::Uniform), line)
                })
                .collect();
            fields.extend(
                gaps.iter()
                    .map(|(offset, bytes)| (*offset, format!("    _pad{offset}: [u8; {bytes}],"))),
            );
            // A gap sorts after a zero-sized field at its offset.
            fields.sort_by_key(|(offset, line)| (*offset, line.starts_with("    _pad")));
            let fields: Vec<String> = fields.into_iter().map(|(_, line)| line).collect();
            let fields = fields.join("\n");
            let checks = layout_checks(&gpu_name, pc);
            bindings.push_str(&format!(
                r#"
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct {gpu_name} {{
{fields}
}}
{checks}"#
            ));

            for f in pc.checked_fields() {
                reject_nested_handles(field_name(f), f.checked_type_layout().unwrap(), source);
            }
            let mut push = build_new_fn(
                pc.checked_fields()
                    .map(|f| (field_name(f), f.checked_type_layout().unwrap())),
                &mut structs,
                stage,
                false,
            );
            push.gpu_inits.extend(
                gaps.iter()
                    .map(|(offset, bytes)| format!("_pad{offset}: [0; {bytes}]")),
            );
            let (images, buffers) = (push.image_accesses.len(), push.buffer_accesses.len());
            let output =
                format!("Bindings<'a, kind::{kind_marker}, {size}, {images}, {buffers}, 0>");
            let incomplete = format!("Incomplete{pass_name}");

            let binding_output = format!(
                r#"Bindings::new(
            {index},
            &{gpu_name} {{
                {gpu_inits}
            }},
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
                    format!("{incomplete}<'a>"),
                    format!("{incomplete}({binding_output})"),
                )
            };

            const OP_CAPABILITY: u32 = 17;
            let words: Vec<u32> = pass_spirv
                .as_slice()
                .chunks_exact(4)
                .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
                .collect();
            let mut capabilities = Vec::new();
            // After the five header words, each instruction starts with its word count and opcode.
            let mut i = 5;
            while i < words.len() {
                let (count, opcode) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
                if opcode == OP_CAPABILITY
                    && let Some(&capability) = words.get(i + 1)
                {
                    capabilities.push(capability);
                }
                if count == 0 {
                    break;
                }
                i += count;
            }
            let mut gates: Vec<&str> = Vec::new();
            for (capability, _, gate) in crate::features::CAPABILITY_FEATURES {
                if capabilities.contains(capability) && !gates.contains(gate) {
                    gates.push(gate);
                }
            }
            let (returns, body) = match gates.split_last() {
                None => (returns, body),
                Some((last, rest)) => {
                    let mut gate = format!("lava::features::gate::{last}");
                    for g in rest.iter().rev() {
                        gate = format!("(lava::features::gate::{g}, {gate})");
                    }
                    (
                        format!("lava::features::Gated<{gate}, {returns}>"),
                        format!("lava::features::gate::<{gate}, _>(move || {body})"),
                    )
                }
            };

            bindings.push_str(&format!(
                r#"
pub struct {pass_name};

impl {pass_name} {{
    pub fn push_bindings{generics}({params}) -> {returns} {{
        {checks}
        {body}
    }}
}}
"#,
                generics = fn_generics(true),
                params = push.params.join(", "),
                checks = push.checks.join("\n        "),
            ));
            if !consts.is_empty() {
                let new = build_new_fn(
                    consts.iter().map(|c| (c.name.as_str(), c.layout)),
                    &mut structs,
                    stage,
                    true,
                );
                let fields: Vec<String> = consts
                    .iter()
                    .map(|c| format!("    pub {}: {},", c.name, constant_type(c.layout, &mut structs)))
                    .collect();
                let constants_size: usize = consts
                    .iter()
                    .map(|c| match constant_type(c.layout, &mut structs).as_str() {
                        "u8" => 1,
                        "u32" | "i32" | "f32" => 4,
                        "u64" => 8,
                        other => panic!("a specialization constant of type {other}"),
                    })
                    .sum();
                let complete = format!(
                    "Bindings<'a, kind::{kind_marker}, {size}, {}, {}, {constants_size}>",
                    images + new.image_accesses.len(),
                    buffers + new.buffer_accesses.len(),
                );
                bindings.push_str(&format!(
                    r#"
/// Packed: the specialization constant data is read at each entry's offset.
#[derive(Clone, Copy)]
#[repr(C, packed)]
pub struct {consts_name} {{
{fields}
}}
unsafe impl bytemuck::Pod for {consts_name} {{}}
unsafe impl bytemuck::Zeroable for {consts_name} {{}}
const _: () = assert!(size_of::<{consts_name}>() == {constants_size});

#[must_use = "`{pass_name}` has specialization constants: call `.constant_bindings(..)`"]
pub struct {incomplete}<'a>({output});

impl<'a> {incomplete}<'a> {{
    pub fn constant_bindings{generics}(self, {params}) -> {complete} {{
        {checks}
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
                    generics = fn_generics(false),
                    params = new.params.join(", "),
                    checks = new.checks.join("\n        "),
                    gpu_inits = new.gpu_inits.join(", "),
                    image_accesses = new.image_accesses.join(",\n"),
                    buffer_accesses = new.buffer_accesses.join(",\n"),
                ));
            }

            built_passes.push(format!(
                "        BuiltPass {{ source: {source:?}, layout: {:#x} }},",
                compile::layout_hash(layout)
            ));
            write_if_changed(&spirv, pass_spirv.as_slice());
        }

        bindings.push_str(&format!(
            r#"
/// Every pass; hand it to `lava::init`.
pub static PASS_MAP: PassMap = PassMap {{
    passes: &[
{passes}
    ],
}};
"#,
            passes = pass_map.join("\n"),
        ));

        for (name, def) in &structs {
            if def.mirror {
                bindings.push_str(&def.checks);
                continue;
            }
            bindings.push_str(&format!(
                "\n#[derive(Pod, Copy, Clone, Zeroable, Debug)]\n#[repr(C)]\npub struct {name} {{\n{}}}\n{}{}",
                def.body, def.checks, def.impls
            ));
        }

        write_if_changed(&out_dir.join("bindings.rs"), bindings.as_bytes());

        // Only text: the build script's lava-build-slang has no `hot_reload` module.
        let watch_dirs: Vec<&String> = self.passes.iter().chain(&self.includes).collect();
        let build = format!(
            r#"Build {{
    search_paths: &{search_paths:?},
    defines: &{defines:?},
    watch_dirs: &{watch_dirs:?},
    optimization: Optimization::{optimization:?},
    passes: &[
{passes}
    ],
}}
"#,
            optimization = self.optimization,
            passes = built_passes.join("\n"),
        );
        write_if_changed(&out_dir.join("hot_reload.rs"), build.as_bytes());
    }
}

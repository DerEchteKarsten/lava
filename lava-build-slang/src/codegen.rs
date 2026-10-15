//! Generates the typed pass bindings and `PASS_MAP` from the reflection of each compiled pass, and what the hot reload compiles them with again
use std::{collections::BTreeMap, fs, path::Path};

use shader_slang::{
    self as slang, ParameterCategory as Cat, ScalarType, TypeKind,
    reflection::{TypeLayout, VariableLayout, compute_string_hash},
};

use crate::{Shaders, compile};

struct StructDef {
    body: String,
    checks: String,
    signature: Vec<(String, usize)>,
    size: usize,
    /// It mirrors a lava type (`MIRRORED_TYPES`), which isn't generated, only checked.
    mirror: bool,
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
    Image {
        usage: u64,
    },
    /// A TLAS by its address.
    Accel,
}

type Structs = BTreeMap<String, StructDef>;

/// The most bytes of push constants a pass can have; Vulkan 1.4 guarantees 256 bytes. A copy
/// of `lava::pipeline::MAX_PUSH_CONSTANTS_SIZE`.
const MAX_PUSH_CONSTANTS_SIZE: usize = 256;

const IMAGE_TYPES: [&str; 1] = ["Img"];
const BUFFER_TYPES: [&str; 2] = ["Buf", "MutBuf"];
/// Image types as struct fields: all are a `BindlessHandle` on the Rust side.
const HANDLE_TYPES: [&str; 2] = ["Img", "DynImg"];
const ACCEL_TYPES: [&str; 1] = ["Accel"];
/// Slang structs that mirror a lava type: (Slang name, Rust path).
const MIRRORED_TYPES: [(&str, &str); 1] = [(
    "AccelerationStructureInstance",
    "lava::vkobjects::acceleration_structure::Instance",
)];

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
            if let Some(&(_, rust)) = MIRRORED_TYPES.iter().find(|(slang, _)| *slang == name) {
                structs.entry(name).or_insert_with(|| StructDef {
                    body: String::new(),
                    checks: layout_checks(rust, t),
                    signature,
                    size,
                    mirror: true,
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
    } else if is_struct_named(t, &ACCEL_TYPES) {
        "u64".into()
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

/// The Rust dimension marker of an `Img<E, Shape, U>`'s shape.
fn image_dim(t: &TypeLayout) -> &'static str {
    let generics = t.ty().unwrap().generic_container().unwrap();
    let shape = generics
        .type_parameters()
        .find(|tp| tp.name() == Some("Shape"))
        .and_then(|tp| generics.concrete_type(tp))
        .and_then(|shape| shape.name());
    match shape {
        Some("__Shape2D") => "image::D2",
        Some("__Shape3D") => "image::D3",
        other => panic!("image shape {other:?} has no Rust dimension"),
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
    } else if is_struct_named(t, &ACCEL_TYPES) {
        FieldKind::Accel
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
    generics: Vec<String>,
    params: Vec<String>,
    gpu_inits: Vec<String>,
    buffer_accesses: Vec<String>,
    image_accesses: Vec<String>,
    /// `format_check`/`usage_check` calls: what the generics' bounds check at compile time,
    /// checked at runtime for dynamic formats and usages.
    checks: Vec<String>,
}

fn build_new_fn<'a>(
    fields: impl Iterator<Item = (&'a str, &'a TypeLayout)>,
    structs: &mut Structs,
    stage: &str,
) -> NewFn {
    let mut out = NewFn {
        generics: Vec::new(),
        params: Vec::new(),
        gpu_inits: Vec::new(),
        buffer_accesses: Vec::new(),
        image_accesses: Vec::new(),
        checks: Vec::new(),
    };

    for (name, t) in fields {
        let suffix = camel_case(name);
        match field_kind(t) {
            FieldKind::Image { usage } => {
                let format_bound = match image_texel(t) {
                    Some((texel, components)) => {
                        let texels = format!("[{texel}; {components}]");
                        out.checks
                            .push(format!("{name}.texel_check::<{texels}>();"));
                        format!("F{suffix}: HasTexels<{texels}>")
                    }
                    None => format!("F{suffix}: Format"),
                };
                out.generics.push(format_bound);
                let (usage_bound, flags) = match usage {
                    0 => ("IsSampled", "SAMPLED"),
                    3 => ("UnifiedBinding", "SAMPLED | vk::ImageUsageFlags::STORAGE"),
                    _ => ("IsStorage", "STORAGE"),
                };
                out.generics
                    .push(format!("U{suffix}: image::usage::{usage_bound}"));
                out.checks
                    .push(format!("{name}.usage_check(vk::ImageUsageFlags::{flags});"));
                out.params.push(format!(
                    "{name}: ImageView<'a, F{suffix}, U{suffix}, {}>",
                    image_dim(t)
                ));
                out.gpu_inits.push(format!("{name}: {name}.handle"));
                out.image_accesses.push(format!(
                    "{name}.access({stage}, {access}, vk::ImageLayout::GENERAL)",
                    access = image_access(usage)
                ));
            }
            FieldKind::Buffer => {
                let ptr = t.field_by_index(0).unwrap().type_layout().unwrap();
                let inner = rust_type(ptr.element_type_layout().unwrap(), structs);
                let ty = format!("BufferSlice<'a, {inner}, U{suffix}>");
                out.generics
                    .push(format!("U{suffix}: buffer::usage::IsStorage"));
                out.checks.push(format!(
                    "{name}.usage_check(vk::BufferUsageFlags::STORAGE_BUFFER);"
                ));
                out.params.push(format!("{name}: {ty}"));
                out.gpu_inits.push(format!("{name}: {name}.address"));
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
                let ty = rust_type(t, structs);
                out.params.push(format!("{name}: {ty}"));
                out.gpu_inits.push(name.to_string());
            }
        }
    }

    out
}

/// The bytes of a specialization constant of Rust type `ty`: a scalar or a handle.
fn constant_size(ty: &str) -> usize {
    match ty {
        "u8" => 1,
        "u32" | "i32" | "f32" => 4,
        "u64" | "BindlessHandle" => 8,
        other => panic!("a specialization constant of type {other}"),
    }
}

/// A specialization constant of a pass, which lava bakes into its pipelines.
struct SConst<'a> {
    name: String,
    id: u32,
    layout: &'a TypeLayout,
    /// The value of one of lava's own constants, which `constant_bindings` sets without a
    /// parameter.
    value: Option<u32>,
}

/// The pass's specialization constants by id. `SCONST(T, name)` (`bindless.slang`) declares a
/// `uint64_t name_sconst` whose `SConst` attribute names `T`, a buffer or image handle; any
/// other one is a plain scalar. `CLOCK_REGION_IDS` (`profile.slang`) get the hashes of the
/// pass's clock `regions`, which `constant_bindings` sets while `clock::enabled()`.
fn sconsts<'a>(
    layout: &'a slang::reflection::Shader,
    file: &str,
    regions: &[String],
) -> Vec<SConst<'a>> {
    let clock_ids = compile::CLOCK_REGION_IDS;
    assert!(
        regions.len() <= clock_ids.len(),
        "{file} has {} clock regions, at most {} fit",
        regions.len(),
        clock_ids.len()
    );
    let mut consts: Vec<SConst> = layout
        .parameters()
        .filter(|p| p.category() == Some(Cat::SpecializationConstant))
        .map(|p| (p, p.offset(Cat::SpecializationConstant) as u32))
        .map(|(p, id)| {
            let name = field_name(p);
            if clock_ids.contains(&id) {
                let region = regions.get((id - clock_ids.start) as usize);
                return SConst {
                    name: name.to_lowercase(),
                    id,
                    layout: p.type_layout().unwrap(),
                    value: Some(region.map_or(0, |name| compute_string_hash(name))),
                };
            }
            let attribute = p
                .variable()
                .and_then(|v| v.user_attributes().find(|a| a.name() == Some("SConst")));
            let Some(attribute) = attribute else {
                return SConst {
                    name: name.to_string(),
                    id,
                    layout: p.type_layout().unwrap(),
                    value: None,
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
                "{file}: SCONST `{name}` has type `{ty}`, which isn't a buffer or image handle \
                 (acceleration structure addresses change from run to run, so they never go \
                 into specialization constants)"
            );
            SConst {
                name: name.trim_end_matches("_sconst").to_string(),
                id,
                layout,
                value: None,
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

fn generate_pass(
    pass_name: &str,
    pc: &TypeLayout,
    consts: &[SConst],
    regions: &[String],
    entries: &[compile::Entry],
    ray_query: bool,
    structs: &mut Structs,
    pass_map: &mut Vec<String>,
    pipelines: &mut Pipelines,
    spirv: &Path,
    source: &str,
) -> String {
    let gpu_name = format!("C{pass_name}Bindings");
    let consts_name = format!("C{pass_name}Constants");
    let index = pass_map.len();

    let entry = |stage: &str| entries.iter().find(|e| e.stage == stage).map(|e| e.name);
    let mut stages: Vec<&str> = entries.iter().map(|e| e.stage).collect();
    stages.sort_unstable();

    let kind = match stages.as_slice() {
        ["fragment", "vertex"] => PassKind::RasterVertex,
        ["fragment", "mesh"] | ["amplification", "fragment", "mesh"] => PassKind::RasterMesh,
        ["compute"] => PassKind::Compute,
        stages
            if stages.contains(&"raygen")
                && stages
                    .iter()
                    .all(|s| compile::RAY_TRACING_STAGES.contains(s)) =>
        {
            PassKind::RayTracing
        }
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
            let groups =
                compile::ray_tracing_groups(entries).unwrap_or_else(|e| panic!("{source}: {e}"));
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
        spirv: include_bytes!({spirv:?}),
        kind: PassKind::{kind_fields},
        index: {pipeline_index},
        constants: {constants},
        constants_size: {constants_size},
        regions: &{regions:?},
        ray_query: {ray_query},
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

    let gpu_struct = generate_gpu_struct(&gpu_name, pc, structs);
    let push = build_new_fn(
        pc.fields()
            .map(|f| (field_name(f), f.type_layout().unwrap())),
        structs,
        stage,
    );
    // The generated types spell out each section's size, so a binding is as big as its pass.
    let (images, buffers) = (push.image_accesses.len(), push.buffer_accesses.len());
    let output = format!("Bindings<'a, kind::{kind_marker}, {size}, {images}, {buffers}, 0>");
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

    let mut out = gpu_struct;
    out.push_str(&format!(
        r#"
pub struct {pass_name};

impl {pass_name} {{
    pub fn push_bindings{generics}({params}) -> {returns} {{
        {checks}
        {body}
    }}
}}
"#,
        generics = fn_generics(&push, true),
        params = push.params.join(", "),
        checks = push.checks.join("\n        "),
    ));
    if !consts.is_empty() {
        let parameters = consts.iter().filter(|c| c.value.is_none());
        let new = build_new_fn(
            parameters.map(|c| (c.name.as_str(), c.layout)),
            structs,
            stage,
        );
        // Only the clock regions' hashes are baked, and only while clocks are on.
        let baked = consts.iter().filter_map(|c| match c.value? {
            0 => Some(format!("{}: 0", c.name)),
            hash => Some(format!(
                "{}: if lava::profiling::clock::enabled() {{ {hash:#x} }} else {{ 0 }}",
                c.name
            )),
        });
        let gpu_inits: Vec<String> = new.gpu_inits.iter().cloned().chain(baked).collect();
        let fields: Vec<String> = consts
            .iter()
            .map(|c| format!("    pub {}: {},", c.name, gpu_type(c.layout, structs)))
            .collect();
        let constants_size: usize = consts
            .iter()
            .map(|c| constant_size(&gpu_type(c.layout, structs)))
            .sum();
        let complete = format!(
            "Bindings<'a, kind::{kind_marker}, {size}, {}, {}, {constants_size}>",
            images + new.image_accesses.len(),
            buffers + new.buffer_accesses.len(),
        );
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
            generics = fn_generics(&new, false),
            params = new.params.join(", "),
            checks = new.checks.join("\n        "),
            gpu_inits = gpu_inits.join(", "),
            image_accesses = new.image_accesses.join(",\n"),
            buffer_accesses = new.buffer_accesses.join(",\n"),
        ));
    }

    out
}

/// `<'a, F…, U…>` of a generated function, empty without parameters that borrow.
/// The generics of a generated function; `lifetime` adds the `'a` its views, slices and
/// result share.
fn fn_generics(new: &NewFn, lifetime: bool) -> String {
    let mut generics: Vec<String> = Vec::new();
    if lifetime {
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

/// Compiles every pass of `shaders`, writes its SPIR-V to `out_dir/passes/<Pass>.spv`, the
/// bindings with `PASS_MAP` to `out_dir/bindings.rs`, and the `hot_reload::Build` that
/// `watch!` includes to `out_dir/hot_reload.rs`.
pub(crate) fn generate(shaders: &Shaders, profiling: bool, out_dir: &Path) {
    let global = slang::GlobalSession::new().unwrap();

    // Every pass's absolute path: each directory's files sorted, directories in call order.
    let mut pass_files: Vec<String> = Vec::new();
    for dir in &shaders.passes {
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
use lava::bindless::BindlessHandle;
#[allow(unused_imports)]
use lava::buffer::slice::*;
#[allow(unused_imports)]
use lava::image::format::*;
#[allow(unused_imports)]
use lava::image::slice::*;
#[allow(unused_imports)]
use lava::buffer;
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
    // `hot_reload::BuiltPass`es, in map order.
    let mut built_passes: Vec<String> = Vec::new();
    let mut pipelines = Pipelines::default();

    let passes_dir = out_dir.join("passes");
    fs::create_dir_all(&passes_dir).unwrap();
    let search_paths: Vec<&str> = shaders.includes.iter().map(String::as_str).collect();
    let defines: Vec<(&str, &str)> = shaders
        .defines
        .iter()
        .map(|(name, value)| (name.as_str(), value.as_str()))
        .collect();

    for source in &pass_files {
        let options = compile::Options {
            search_paths: &search_paths,
            defines: &defines,
            optimization: shaders.optimization,
            profiling,
            pass_index: pass_map.len(),
        };
        let session = compile::session(&global, &options);
        let program = compile::link(&session, source).unwrap_or_else(|e| panic!("{source}:\n{e}"));
        let layout = program.layout(0).unwrap();

        let file = Path::new(source).file_name().unwrap().to_str().unwrap();
        let pass_name: String = file
            .trim_end_matches(".slang")
            .split('_')
            .map(capitalize_first)
            .collect();

        let entries: Vec<compile::Entry> = layout
            .entry_points()
            .map(|ep| compile::Entry {
                stage: compile::stage_name(ep.stage()),
                name: ep.name().unwrap(),
                group: compile::entry_attribute(ep, "Group"),
                max_recursion: compile::entry_attribute(ep, "MaxRecursion"),
            })
            .collect();
        let pass_spirv = program
            .target_code(0)
            .unwrap_or_else(|e| panic!("{source}: {e}"));

        let pc = layout
            .parameters()
            .find(|p| p.category() == Some(Cat::PushConstantBuffer))
            .and_then(|p| p.type_layout())
            .and_then(|t| t.element_type_layout())
            .unwrap_or_else(|| panic!("no push constant buffer found in {source}"));

        let regions = compile::clock_regions(&program);
        let consts = sconsts(layout, source, &regions);

        let out = passes_dir.join(format!("{pass_name}.spv"));
        let code = generate_pass(
            &pass_name,
            pc,
            &consts,
            &regions,
            &entries,
            compile::uses_ray_query(pass_spirv.as_slice()),
            &mut structs,
            &mut pass_map,
            &mut pipelines,
            &out,
            source,
        );
        built_passes.push(format!(
            "        BuiltPass {{ source: {source:?}, layout: {:#x} }},",
            compile::layout_hash(layout)
        ));
        bindings.push_str(&code);
        write_if_changed(&out, pass_spirv.as_slice());
    }

    bindings.push_str(&format!(
        r#"
/// Every pass; hand it to `lava::init`.
pub static PASS_MAP: PassMap = PassMap {{
    passes: &[
{passes}
    ],
    profiling: {profiling},
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
            "\n#[derive(Pod, Copy, Clone, Zeroable, Debug, Default)]\n#[repr(C)]\npub struct {name} {{\n{}}}\n{}",
            def.body, def.checks
        ));
    }

    write_if_changed(&out_dir.join("bindings.rs"), bindings.as_bytes());

    // Only text here: the build script's lava-build-slang has no `hot_reload` module.
    let watch_dirs: Vec<&String> = shaders.passes.iter().chain(&shaders.includes).collect();
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
        optimization = shaders.optimization,
        passes = built_passes.join("\n"),
    );
    write_if_changed(&out_dir.join("hot_reload.rs"), build.as_bytes());
}

/// Only a changed file makes cargo rebuild the crate that includes it (`include!`,
/// `include_bytes!`), so an edit to one pass rebuilds the crate only if its SPIR-V changed.
fn write_if_changed(path: &Path, bytes: &[u8]) {
    if fs::read(path).ok().as_deref() != Some(bytes) {
        fs::write(path, bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    }
}

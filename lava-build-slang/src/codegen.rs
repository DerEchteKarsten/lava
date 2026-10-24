//! Generates the typed pass bindings (an argument struct per pass, one for its specialization constants, `push_bindings` with the extra `Accesses`, workgroup sizes), the shared structs (padded, `Default`), the exported constants and `PASS_MAP` from the descriptions of the compiled passes, compiled in parallel; and what the hot reload compiles them with again
use std::{
    collections::BTreeMap,
    env, fs,
    path::Path,
    sync::{Mutex, atomic::AtomicUsize, atomic::Ordering},
};

use crate::compile::{self, CompiledFile, CompiledPass};
use crate::describe::{
    Constant, Export, Image, Kind, Pointee, Scalar, Shape, Struct, Ty, capitalize_first,
};
use crate::{SLANG_DIR, Shaders};

const CLOCK_TYPE: &str = "Clock";
/// Copies of `CLOCK_STRIPES`, `CLOCK_STRIPE_WORDS` and `CLOCK_TICK_BITS` in
/// `slang/profile.slang`.
const CLOCK_STRIPES: u32 = 64;
const CLOCK_STRIPE_WORDS: u32 = 8;
const CLOCK_TICK_BITS: u32 = 40;

/// Rust's strict and reserved keywords (edition 2024), which fields get as raw identifiers.
const KEYWORDS: &[&str] = &[
    "as", "async", "await", "break", "const", "continue", "dyn", "else", "enum", "extern", "false",
    "fn", "for", "gen", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut", "pub",
    "ref", "return", "static", "struct", "trait", "true", "type", "unsafe", "use", "where",
    "while", "abstract", "become", "box", "do", "final", "macro", "override", "priv", "try",
    "typeof", "unsized", "virtual", "yield",
];

/// A field name as a Rust identifier: keywords raw (`r#type`); those that can't be raw refused.
pub(crate) fn ident(name: &str) -> Result<String, String> {
    if ["self", "Self", "super", "crate", "_"].contains(&name) {
        return Err(format!(
            "the field name `{name}` can't be a Rust identifier: rename it"
        ));
    }
    Ok(match KEYWORDS.contains(&name) {
        true => format!("r#{name}"),
        false => name.to_string(),
    })
}

/// The largest power of two, at most 16, that divides `position` (16 for 0): how aligned a
/// field can be that sits there.
fn alignment_at(position: u32) -> u32 {
    match position {
        0 => 16,
        p => (1 << p.trailing_zeros()).min(16),
    }
}

fn scalar(s: Scalar) -> &'static str {
    match s {
        Scalar::Bool | Scalar::U32 => "u32",
        Scalar::I8 => "i8",
        Scalar::U8 => "u8",
        Scalar::I16 => "i16",
        Scalar::U16 => "u16",
        Scalar::F16 => "::lava::half::F16",
        Scalar::I32 => "i32",
        Scalar::F32 => "f32",
        Scalar::I64 => "i64",
        Scalar::U64 => "u64",
        Scalar::F64 => "f64",
    }
}

/// The Rust alignment of a scalar: its size.
fn scalar_align(s: Scalar) -> u32 {
    s.size()
}

/// The bytes a value of `ty` takes in Slang's layout.
fn slang_size(ty: &Ty) -> Option<u32> {
    Some(match ty {
        Ty::Scalar(s) => s.size(),
        Ty::Vector(s, n) => s.size() * n,
        Ty::Matrix {
            columns,
            column_stride,
            ..
        } => columns * column_stride,
        Ty::Array { count, stride, .. } => count * stride,
        Ty::Struct(s) | Ty::Mirror(_, s) => s.size,
        Ty::Pointer(_) | Ty::Buffer { .. } | Ty::Accel => 8,
        Ty::Image(_) | Ty::DynImg => 4,
        Ty::Named(_) => return None,
    })
}

/// A Rust type and its alignment.
struct Rust {
    ty: String,
    align: u32,
}

/// The `ImageHandle` kind of an image's kind `U`.
fn handle_kind(image: &Image) -> &'static str {
    match image.usage {
        0 => "Sampled",
        3 => "Combined",
        _ => "Storage",
    }
}

/// The generated Rust type of `ty` where the GPU reads it (push constants, structs behind a
/// pointer): handles are `ImageHandle`s and addresses `u64`s. `at` is the alignment its
/// positions share (`alignment_at`), which decides whether glam's SIMD-aligned types fit.
/// `structs` is every generated struct's alignment, by name.
fn rust_ty(ty: &Ty, at: u32, structs: &BTreeMap<String, u32>) -> Result<Rust, String> {
    let plain = |ty: String, align: u32| Ok(Rust { ty, align });
    match ty {
        Ty::Scalar(s) => plain(scalar(*s).into(), scalar_align(*s)),
        Ty::Vector(s, n) => {
            let glam = match s {
                Scalar::F32 => "Vec",
                Scalar::F64 => "DVec",
                Scalar::I32 => "IVec",
                Scalar::U32 | Scalar::Bool => "UVec",
                Scalar::I64 => "I64Vec",
                Scalar::U64 => "U64Vec",
                Scalar::I16 => "I16Vec",
                Scalar::U16 => "U16Vec",
                Scalar::I8 => "I8Vec",
                Scalar::U8 => "U8Vec",
                Scalar::F16 => "",
            };
            let array = || format!("[{}; {n}]", scalar(*s));
            // glam's `Vec4` is 16-byte aligned (SIMD); elsewhere an array of four.
            let simd = *s == Scalar::F32 && *n == 4;
            match (glam, n) {
                ("", _) | (_, 1) => plain(array(), scalar_align(*s)),
                _ if simd && at < 16 => plain(array(), 4),
                _ if simd => plain("::lava::glam::Vec4".into(), 16),
                (glam, 2..=4) => plain(format!("::lava::glam::{glam}{n}"), scalar_align(*s)),
                _ => Err(format!("a vector of {n} components")),
            }
        }
        Ty::Matrix {
            scalar: s,
            rows,
            columns,
            column_stride,
        } => {
            if !column_stride.is_multiple_of(s.size()) {
                return Err(format!("matrix columns {column_stride} bytes apart"));
            }
            let tight = *column_stride == rows * s.size();
            let square = rows == columns && (2..=4).contains(rows);
            // glam's `Mat2` and `Mat4` are 16-byte aligned (SIMD), `Mat3` isn't.
            let simd = *s == Scalar::F32 && *rows != 3;
            match s {
                Scalar::F32 | Scalar::F64 if tight && square && !(simd && at < 16) => {
                    let prefix = if *s == Scalar::F32 { "Mat" } else { "DMat" };
                    let align = if simd { 16 } else { s.size() };
                    plain(format!("::lava::glam::{prefix}{rows}"), align)
                }
                // Columns, each padded to its stride.
                _ => plain(
                    format!(
                        "[[{}; {}]; {columns}]",
                        scalar(*s),
                        column_stride / s.size()
                    ),
                    scalar_align(*s),
                ),
            }
        }
        Ty::Array {
            element,
            count,
            stride,
        } => {
            let element_at = at.min(alignment_at(*stride));
            let element = padded(element, *stride, element_at, structs)?;
            plain(format!("[{}; {count}]", element.ty), element.align)
        }
        Ty::Struct(s) => {
            let align = *structs.get(&s.name).unwrap_or(&1);
            if align > at {
                return Err(format!(
                    "the struct `{}` needs {align}-byte alignment (it holds glam's SIMD types), \
                     but sits where only {at} bytes are",
                    s.name
                ));
            }
            plain(s.name.clone(), align)
        }
        Ty::Named(name) => plain(name.clone(), *structs.get(name).unwrap_or(&1)),
        Ty::Mirror(path, s) => plain(path.to_string(), alignment_at(s.size).min(8)),
        Ty::Pointer(_) | Ty::Buffer { .. } | Ty::Accel => plain("u64".into(), 8),
        Ty::Image(image) => plain(
            format!(
                "::lava::bindless::ImageHandle<::lava::bindless::{}>",
                handle_kind(image)
            ),
            4,
        ),
        Ty::DynImg => plain("::lava::bindless::ImageHandle".into(), 4),
    }
}

/// The Rust type of `element`s `stride` bytes apart: the element's own if its size is the
/// stride, an array of its scalars padded to the stride for a scalar or vector, else refused.
fn padded(
    element: &Ty,
    stride: u32,
    at: u32,
    structs: &BTreeMap<String, u32>,
) -> Result<Rust, String> {
    let size = slang_size(element);
    if size.is_none_or(|size| size == stride) {
        return rust_ty(element, at, structs);
    }
    let size = size.unwrap();
    match element {
        Ty::Scalar(s) | Ty::Vector(s, _) if stride.is_multiple_of(s.size()) => Ok(Rust {
            ty: format!("[{}; {}]", scalar(*s), stride / s.size()),
            align: scalar_align(*s),
        }),
        _ => Err(format!(
            "elements {stride} bytes apart that are {size} bytes long: pad the element type to \
             its stride"
        )),
    }
}

/// The struct generation's view of a shared struct.
struct SharedStruct {
    desc: Struct,
    /// The pass file that first used it.
    source: String,
    mirror: Option<&'static str>,
}

/// Whether `s` is `profile.slang`'s `Clock`: by name and layout, so another struct of that name
/// gets no clock readout.
fn is_clock(s: &Struct) -> bool {
    s.name == CLOCK_TYPE
        && matches!(
            s.fields.as_slice(),
            [f] if f.name == "counters" && f.offset == 0 && f.ty == Ty::Array {
                element: Box::new(Ty::Scalar(Scalar::U64)),
                count: CLOCK_STRIPES * CLOCK_STRIPE_WORDS,
                stride: 8,
            }
        )
}

/// What a top-level field of the push constants or specialization constants becomes.
struct Binding {
    /// The field in the argument struct.
    arg_ty: String,
    /// An image's generic parameter: its name and bounds.
    generic: Option<(String, String)>,
    /// The argument borrows for `'a` (a slice or TLAS).
    lifetime: bool,
    /// The field in the GPU struct.
    gpu_ty: String,
    /// Statements before the GPU struct is made.
    prep: Vec<String>,
    /// The GPU field's value.
    init: String,
    images: Vec<String>,
    buffers: Vec<String>,
}

/// One pass's generated code and pass map entry.
struct GeneratedPass {
    code: String,
    entry: String,
}

struct Generator {
    structs: BTreeMap<String, SharedStruct>,
    exports: BTreeMap<String, (Export, String)>,
    /// Every item name generated so far, with the pass file it comes from.
    items: BTreeMap<String, String>,
}

impl Generator {
    /// Records every struct `ty` reaches; a struct of one name has one layout.
    fn collect(&mut self, ty: &Ty, source: &str) -> Result<(), String> {
        match ty {
            Ty::Array { element, .. } => self.collect(element, source),
            Ty::Pointer(Some(pointee)) | Ty::Buffer { pointee, .. } => {
                self.collect(&pointee.ty, source)
            }
            Ty::Struct(s) | Ty::Mirror(_, s) => {
                let mirror = match ty {
                    Ty::Mirror(path, _) => Some(*path),
                    _ => None,
                };
                match self.structs.get(&s.name) {
                    Some(existing) if existing.desc != *s => {
                        return Err(format!(
                            "the struct `{}` has another layout or field types in {} than in \
                             {}: give one of them another name",
                            s.name, source, existing.source
                        ));
                    }
                    Some(_) => return Ok(()),
                    None => {}
                }
                self.structs.insert(
                    s.name.clone(),
                    SharedStruct {
                        desc: s.clone(),
                        source: source.to_string(),
                        mirror,
                    },
                );
                for f in &s.fields {
                    self.collect(&f.ty, source)
                        .map_err(|e| format!("`{}.{}`: {e}", s.name, f.name))?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Claims a generated item name for `source`.
    fn claim(&mut self, name: &str, source: &str) -> Result<(), String> {
        match self.items.get(name) {
            Some(other) => Err(format!(
                "the generated item `{name}` of {source} clashes with one of {other}: rename a \
                 pass or struct"
            )),
            None => {
                self.items.insert(name.to_string(), source.to_string());
                Ok(())
            }
        }
    }

    /// Every generated struct's Rust alignment: its fields' largest.
    fn alignments(&self) -> Result<BTreeMap<String, u32>, String> {
        let mut aligns = BTreeMap::new();
        // Nested structs first: repeat until every struct is known (struct fields aren't
        // cyclic; pointers are `u64`s).
        let mut left: Vec<&SharedStruct> = self.structs.values().collect();
        while !left.is_empty() {
            let before = left.len();
            let mut rest = Vec::new();
            for s in left {
                let pending = s
                    .desc
                    .fields
                    .iter()
                    .any(|f| depends_on_unknown(&f.ty, &aligns));
                if pending {
                    rest.push(s);
                    continue;
                }
                let (_, align) =
                    struct_layout(&s.desc, &aligns).map_err(|e| format!("{}: {e}", s.source))?;
                aligns.insert(s.desc.name.clone(), align.effective);
            }
            if rest.len() == before {
                return Err("structs that contain each other".into());
            }
            left = rest;
        }
        Ok(aligns)
    }
}

/// A generated struct's alignment: its fields' largest (`natural`), and what it can have
/// (`effective`): Slang's scalar layout doesn't round a struct's size up to its alignment
/// (`{ uint64_t a; uint b; }` is 12 bytes), so such a struct is `repr(C, packed(effective))`.
struct Alignment {
    natural: u32,
    effective: u32,
}

/// The Rust types of a struct's fields and its alignment.
fn struct_layout(
    s: &Struct,
    aligns: &BTreeMap<String, u32>,
) -> Result<(Vec<Rust>, Alignment), String> {
    let mut natural = 1;
    let mut fields = Vec::new();
    for f in &s.fields {
        // glam's SIMD types only where the struct's size keeps them aligned too.
        let at = alignment_at(f.offset).min(alignment_at(s.size));
        let rust =
            rust_ty(&f.ty, at, aligns).map_err(|e| format!("`{}.{}`: {e}", s.name, f.name))?;
        natural = natural.max(rust.align);
        fields.push(rust);
    }
    let effective = natural.min(alignment_at(s.size));
    Ok((fields, Alignment { natural, effective }))
}

fn depends_on_unknown(ty: &Ty, known: &BTreeMap<String, u32>) -> bool {
    match ty {
        Ty::Struct(s) => !known.contains_key(&s.name),
        Ty::Array { element, .. } => depends_on_unknown(element, known),
        _ => false,
    }
}

/// The text of a `repr(C)` struct of `fields` (name, offset, size, Rust type) and `size` bytes,
/// with a zeroed `_pad` field for every gap Slang leaves (the tail too), so it derives `Pod`
/// without padding bytes; `public_pads` for structs applications build (`..Default::default()`
/// needs every field visible).
fn struct_fields(
    fields: &[(String, u32, u32, String)],
    size: u32,
    public_pads: bool,
) -> (String, Vec<(u32, u32)>) {
    let mut gaps = Vec::new();
    let mut end = 0;
    let mut sorted: Vec<&(String, u32, u32, String)> = fields.iter().collect();
    sorted.sort_by_key(|f| f.1);
    for (_, offset, field_size, _) in &sorted {
        if *offset > end {
            gaps.push((end, offset - end));
        }
        end = end.max(offset + field_size);
    }
    if size > end {
        gaps.push((end, size - end));
    }
    let mut lines: Vec<(u32, bool, String)> = sorted
        .iter()
        .map(|(name, offset, _, ty)| (*offset, false, format!("    pub {name}: {ty},")))
        .collect();
    let pad_vis = match public_pads {
        true => "    #[doc(hidden)]\n    pub ",
        false => "    ",
    };
    lines.extend((gaps.iter()).map(|(offset, bytes)| {
        (
            *offset,
            true,
            format!("{pad_vis}_pad{offset}: [u8; {bytes}],"),
        )
    }));
    // A gap sorts after a zero-sized field at its offset.
    lines.sort_by_key(|(offset, pad, _)| (*offset, *pad));
    let body = lines
        .into_iter()
        .map(|(_, _, line)| line)
        .collect::<Vec<_>>()
        .join("\n");
    (body, gaps)
}

fn layout_checks(rust_name: &str, fields: &[(String, u32)], size: u32) -> String {
    let mut out = String::new();
    for (name, offset) in fields {
        out.push_str(&format!(
            "const _: () = assert!(::core::mem::offset_of!({rust_name}, {name}) == {offset});\n"
        ));
    }
    out.push_str(&format!(
        "const _: () = assert!(::core::mem::size_of::<{rust_name}>() == {size});\n"
    ));
    out
}

fn derives(handles: bool) -> &'static str {
    match handles {
        true => "::lava::bytemuck::NoUninit",
        false => "::lava::bytemuck::Pod, ::lava::bytemuck::Zeroable",
    }
}

/// The value of a zeroed `ty` in `Default`: null handles, zero elsewhere.
fn zero(ty: &Ty) -> String {
    match ty {
        Ty::Image(_) | Ty::DynImg => "::lava::bindless::ImageHandle::NULL".into(),
        Ty::Array { element, count, .. } if ty.holds_handles() => {
            format!("[{}; {count}]", zero(element))
        }
        Ty::Struct(_) if ty.holds_handles() => "::core::default::Default::default()".into(),
        _ => "::lava::bytemuck::Zeroable::zeroed()".into(),
    }
}

/// The texel type lava's formats have for an image of `texel`s (`HasTexels`): halves read as
/// `f32`, small integers as 32-bit ones.
fn texels(image: &Image) -> Result<String, String> {
    let (s, n) = image.texel;
    let rust = match s {
        Scalar::F32 | Scalar::F16 => "f32",
        Scalar::U32 | Scalar::U16 | Scalar::U8 | Scalar::Bool => "u32",
        Scalar::I32 | Scalar::I16 | Scalar::I8 => "i32",
        Scalar::F64 => "f64",
        Scalar::U64 => "u64",
        Scalar::I64 => "i64",
    };
    match n {
        1..=4 => Ok(format!("[{rust}; {n}]")),
        n => Err(format!("an image of {n}-component texels")),
    }
}

/// What a top-level field `name` of `ty` becomes, read from the parameter `base` (`args`, or
/// `constants` for specialization constants, whose handles are `u64`s). `generic` is the image
/// parameter's name.
fn binding(
    base: &str,
    name: &str,
    ty: &Ty,
    at: u32,
    stage: &str,
    generic: &str,
    structs: &BTreeMap<String, u32>,
) -> Result<Binding, String> {
    let constant = base == "constants";
    let arg = format!("{base}.{name}");
    let local = format!("{base}_{}", name.trim_start_matches("r#"));
    let read = "::lava::vk::AccessFlags2::SHADER_STORAGE_READ";
    let write = "::lava::vk::AccessFlags2::SHADER_STORAGE_WRITE";
    let sampled = "::lava::vk::AccessFlags2::SHADER_SAMPLED_READ";
    Ok(match ty {
        Ty::Image(image) => {
            let dim = match image.shape {
                Shape::D2 => "::lava::image::D2",
                Shape::D3 => "::lava::image::D3",
                Shape::D2Array => "::lava::image::D2Array",
            };
            let texels = texels(image)?;
            let bounds = format!("Dim = {dim}, Format: ::lava::image::format::HasTexels<{texels}>");
            let (traits, flags, access) = match image.usage {
                0 => (
                    format!("::lava::image::SampledView<{bounds}>"),
                    "SAMPLED",
                    sampled.to_string(),
                ),
                1 => (
                    format!("::lava::image::StorageView<{bounds}>"),
                    "STORAGE",
                    read.to_string(),
                ),
                2 => (
                    format!("::lava::image::StorageView<{bounds}>"),
                    "STORAGE",
                    format!("{read} | {write}"),
                ),
                _ => (
                    format!("::lava::image::SampledView<{bounds}> + ::lava::image::StorageView"),
                    "SAMPLED | ::lava::vk::ImageUsageFlags::STORAGE",
                    format!("{sampled} | {write}"),
                ),
            };
            let view = format!("__view_{local}");
            let kind = handle_kind(image);
            let handle = format!("{view}.handle::<::lava::bindless::{kind}>()");
            Binding {
                arg_ty: generic.to_string(),
                generic: Some((generic.to_string(), format!("{traits} + 'a"))),
                lifetime: false,
                gpu_ty: match constant {
                    true => "u64".into(),
                    false => format!("::lava::bindless::ImageHandle<::lava::bindless::{kind}>"),
                },
                prep: vec![
                    format!("let {view} = ::lava::image::ImageRef::view({arg});"),
                    format!("{view}.texel_check::<{texels}>();"),
                    format!("{view}.usage_check(::lava::vk::ImageUsageFlags::{flags});"),
                ],
                // A specialization constant is a `u64`, the handle zero-extended.
                init: match constant {
                    true => format!("{handle}.bits() as u64"),
                    false => handle,
                },
                images: vec![format!("{view}.access({stage}, {access})")],
                buffers: Vec::new(),
            }
        }
        Ty::Buffer { pointee, writable } => {
            let element = pointee_ty(pointee, structs)?;
            let access = match writable {
                true => format!("{read} | {write}"),
                false => read.to_string(),
            };
            Binding {
                arg_ty: format!("::lava::buffer::slice::BufferSlice<'a, {element}>"),
                generic: None,
                lifetime: true,
                gpu_ty: "u64".into(),
                prep: Vec::new(),
                init: format!("{arg}.address()"),
                images: Vec::new(),
                buffers: vec![format!("{arg}.access({stage}, {access})")],
            }
        }
        Ty::Accel => {
            let trace = format!("__trace_{local}");
            Binding {
                arg_ty: "&'a ::lava::vkobjects::acceleration_structure::Tlas".into(),
                generic: None,
                lifetime: true,
                gpu_ty: "u64".into(),
                prep: vec![format!(
                    "let [{trace}0, {trace}1] = {arg}.trace_accesses({stage});"
                )],
                init: format!("{arg}.address()"),
                images: Vec::new(),
                buffers: vec![format!("{trace}0"), format!("{trace}1")],
            }
        }
        // A `DynImg` is a plain `ImageHandle`, whose image the bindings' `Accesses` name.
        other => {
            let rust = rust_ty(other, at, structs)?;
            Binding {
                arg_ty: rust.ty.clone(),
                generic: None,
                lifetime: false,
                gpu_ty: rust.ty,
                prep: Vec::new(),
                init: arg,
                images: Vec::new(),
                buffers: Vec::new(),
            }
        }
    })
}

/// The element type of a `BufferSlice` of `pointee`, checked against its stride.
fn pointee_ty(pointee: &Pointee, structs: &BTreeMap<String, u32>) -> Result<String, String> {
    let at = alignment_at(pointee.stride);
    Ok(padded(&pointee.ty, pointee.stride, at, structs)?.ty)
}

/// The size asserts of every pointee `ty` reaches (`BufferSlice<T>` elements, raw pointers'
/// targets): its Rust type is its stride.
fn pointee_checks(ty: &Ty, structs: &BTreeMap<String, u32>, out: &mut Vec<String>) {
    match ty {
        Ty::Array { element, .. } => pointee_checks(element, structs, out),
        Ty::Struct(s) => (s.fields.iter()).for_each(|f| pointee_checks(&f.ty, structs, out)),
        Ty::Pointer(Some(p)) | Ty::Buffer { pointee: p, .. } => {
            if let (Ok(element), false) = (pointee_ty(p, structs), matches!(p.ty, Ty::Named(_))) {
                out.push(format!(
                    "const _: () = assert!(::core::mem::size_of::<{element}>() == {});\n",
                    p.stride
                ));
            }
        }
        _ => {}
    }
}

/// The items of one pass. `index` is its place in `PASS_MAP`.
fn generate_pass(
    pass: &CompiledPass,
    source: &str,
    index: usize,
    spirv_path: &Path,
    structs: &BTreeMap<String, u32>,
) -> Result<GeneratedPass, Vec<String>> {
    let name = &pass.name;
    let desc = &pass.desc;
    let mut errors = Vec::new();
    let (kind_marker, stage) = match desc.kind {
        Kind::Compute => ("Compute", "::lava::command_buffer::kind::Compute::STAGES"),
        Kind::RayTracing => (
            "RayTracing",
            "::lava::command_buffer::kind::RayTracing::STAGES",
        ),
        Kind::RasterVertex => (
            "RasterVertex",
            "::lava::command_buffer::kind::RasterVertex::STAGES",
        ),
        Kind::RasterMesh => (
            "RasterMesh",
            "::lava::command_buffer::kind::RasterMesh::STAGES",
        ),
    };
    let gpu_name = format!("C{name}Bindings");
    let args_name = format!("{name}Args");
    let consts_name = format!("{name}Constants");
    let packed_name = format!("C{name}Constants");
    let mut code = String::new();

    // The generic parameter names: a field's PascalCase name and `View`, clear of every struct.
    let generic_name = |field: &str, prefix: &str| {
        let pascal: String = field.split('_').map(capitalize_first).collect();
        let mut name = format!("{prefix}{pascal}View");
        while structs.contains_key(&name) {
            name.push('_');
        }
        name
    };

    // The push constants and the specialization constants, each a list of bindings.
    let mut fields = Vec::new();
    let mut bindings = Vec::new();
    if let Some(pc) = &desc.push_constants {
        for f in &pc.fields {
            let field = match ident(&f.name) {
                Ok(field) => field,
                Err(e) => {
                    errors.push(e);
                    continue;
                }
            };
            let at = alignment_at(f.offset).min(alignment_at(pc.size));
            let generic = generic_name(&f.name, "");
            match binding("args", &field, &f.ty, at, stage, &generic, structs) {
                Ok(b) => {
                    fields.push((field, f.offset, f.size));
                    bindings.push(b);
                }
                Err(e) => errors.push(format!("push constant `{}`: {e}", f.name)),
            }
        }
    }
    let mut constant_bindings = Vec::new();
    let mut constant_fields: Vec<(String, &Constant)> = Vec::new();
    for c in &desc.constants {
        let field = match ident(&c.name) {
            Ok(field) => field,
            Err(e) => {
                errors.push(e);
                continue;
            }
        };
        let generic = generic_name(&c.name, "Const");
        match binding("constants", &field, &c.ty, 1, stage, &generic, structs) {
            Ok(b) => {
                constant_fields.push((field, c));
                constant_bindings.push(b);
            }
            Err(e) => errors.push(format!("specialization constant `{}`: {e}", c.name)),
        }
    }
    if !errors.is_empty() {
        return Err(errors);
    }

    // The push constants as the GPU reads them.
    let push_size = desc.push_constants.as_ref().map_or(0, |pc| pc.size);
    let mut gaps = Vec::new();
    if let Some(pc) = &desc.push_constants {
        let gpu_fields: Vec<(String, u32, u32, String)> = (fields.iter().zip(&bindings))
            .map(|((field, offset, size), b)| (field.clone(), *offset, *size, b.gpu_ty.clone()))
            .collect();
        let (body, pads) = struct_fields(&gpu_fields, push_size, false);
        gaps = pads;
        let handles = pc.fields.iter().any(|f| f.ty.holds_handles());
        let checks = layout_checks(
            &gpu_name,
            &(fields.iter())
                .map(|(f, o, _)| (f.clone(), *o))
                .collect::<Vec<_>>(),
            push_size,
        );
        code.push_str(&format!(
            r#"
/// The push constants of `{name}` as its shaders read them.
#[derive(Clone, Copy, {derives})]
#[bytemuck(crate = "::lava::bytemuck")]
#[repr(C)]
pub struct {gpu_name} {{
{body}
}}
{checks}"#,
            derives = derives(handles),
        ));
    }

    // The argument structs: their generics and fields.
    let struct_generics = |bindings: &[Binding]| {
        let lifetime = bindings.iter().any(|b| b.lifetime);
        let mut params: Vec<String> = Vec::new();
        if lifetime {
            params.push("'a".into());
        }
        params.extend(
            bindings
                .iter()
                .filter_map(|b| b.generic.as_ref().map(|g| g.0.clone())),
        );
        match params.is_empty() {
            true => String::new(),
            false => format!("<{}>", params.join(", ")),
        }
    };
    let arg_fields = |names: &[String], bindings: &[Binding]| {
        (names.iter().zip(bindings))
            .map(|(name, b)| format!("    pub {name}: {},\n", b.arg_ty))
            .collect::<String>()
    };
    let field_names: Vec<String> = fields.iter().map(|(f, _, _)| f.clone()).collect();
    let args_generics = struct_generics(&bindings);
    let source_file = Path::new(source)
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or(source);
    code.push_str(&format!(
        r#"
/// The push constants of `{name}` (`{source_file}`), for `{name}::push_bindings`.
///
/// Buffers are slices, images views (of the shader's dimension and texel type, bound with the
/// descriptors it reads), TLASes references, plain values as the shader reads them.
pub struct {args_name}{args_generics} {{
{fields}}}
"#,
        fields = arg_fields(&field_names, &bindings),
    ));

    let constant_names: Vec<String> = constant_fields.iter().map(|(f, _)| f.clone()).collect();
    let consts_generics = struct_generics(&constant_bindings);
    let constants_size: u32 = desc.constants.iter().map(Constant::size).sum();
    if !desc.constants.is_empty() {
        let packed: String = (constant_names.iter().zip(&constant_bindings))
            .map(|(name, b)| format!("    pub {name}: {},\n", b.gpu_ty))
            .collect();
        code.push_str(&format!(
            r#"
/// The specialization constants of `{name}`, baked into its pipelines: every distinct value
/// builds one, so only values that never change at runtime.
pub struct {consts_name}{consts_generics} {{
{fields}}}

/// The specialization constants of `{name}` as its pipelines take them, packed in id order:
/// the data is read at each entry's offset.
#[derive(Clone, Copy)]
#[repr(C, packed)]
pub struct {packed_name} {{
{packed}}}
// SAFETY: packed, of plain integers and floats.
unsafe impl ::lava::bytemuck::Pod for {packed_name} {{}}
unsafe impl ::lava::bytemuck::Zeroable for {packed_name} {{}}
const _: () = assert!(::core::mem::size_of::<{packed_name}>() == {constants_size});
"#,
            fields = arg_fields(&constant_names, &constant_bindings),
        ));
    }

    // `push_bindings`.
    let mut fn_generics = vec!["'a".to_string()];
    let mut bounds = Vec::new();
    for b in bindings.iter().chain(&constant_bindings) {
        if let Some((param, bound)) = &b.generic {
            fn_generics.push(param.clone());
            bounds.push(format!("        {param}: {bound},"));
        }
    }
    let mut seen = std::collections::BTreeSet::new();
    if let Some(twice) = fn_generics.iter().find(|g| !seen.insert(*g)) {
        return Err(vec![format!(
            "two image fields make the generic parameter `{twice}`: rename one"
        )]);
    }
    let mut params = vec![format!("args: {args_name}{args_generics}")];
    if !desc.constants.is_empty() {
        params.push(format!("constants: {consts_name}{consts_generics}"));
    }
    params.push("accesses: ::core::option::Option<::lava::command_buffer::Accesses<'a>>".into());

    let images = bindings.iter().map(|b| b.images.len()).sum::<usize>()
        + constant_bindings
            .iter()
            .map(|b| b.images.len())
            .sum::<usize>();
    let buffers = bindings.iter().map(|b| b.buffers.len()).sum::<usize>()
        + constant_bindings
            .iter()
            .map(|b| b.buffers.len())
            .sum::<usize>();
    let output = format!(
        "::lava::command_buffer::Bindings<'a, ::lava::command_buffer::kind::{kind_marker}, \
         {push_size}, {images}, {buffers}, {constants_size}>"
    );
    let mut body = Vec::new();
    for b in &bindings {
        body.extend(b.prep.iter().cloned());
    }
    for b in &constant_bindings {
        body.extend(b.prep.iter().cloned());
    }
    let push = match desc.push_constants {
        None => "&()".to_string(),
        Some(_) => {
            let mut inits: Vec<String> = (field_names.iter().zip(&bindings))
                .map(|(name, b)| format!("{name}: {}", b.init))
                .collect();
            inits.extend(
                gaps.iter()
                    .map(|(offset, bytes)| format!("_pad{offset}: [0; {bytes}]")),
            );
            body.push(format!(
                "let __push = {gpu_name} {{ {} }};",
                inits.join(", ")
            ));
            "&__push".to_string()
        }
    };
    let list = |items: Vec<String>| format!("[{}]", items.join(", "));
    body.push(format!(
        "let __images = {};",
        list(bindings.iter().flat_map(|b| b.images.clone()).collect())
    ));
    body.push(format!(
        "let __buffers = {};",
        list(bindings.iter().flat_map(|b| b.buffers.clone()).collect())
    ));
    let mut make = format!(
        "::lava::command_buffer::Bindings::new({index}, {push}, __images, __buffers, accesses)"
    );
    if !desc.constants.is_empty() {
        let inits: Vec<String> = (constant_names.iter().zip(&constant_bindings))
            .map(|(name, b)| format!("{name}: {}", b.init))
            .collect();
        body.push(format!(
            "let __constants = {packed_name} {{ {} }};",
            inits.join(", ")
        ));
        body.push(format!(
            "let __constant_images = {};",
            list(
                constant_bindings
                    .iter()
                    .flat_map(|b| b.images.clone())
                    .collect()
            )
        ));
        body.push(format!(
            "let __constant_buffers = {};",
            list(
                constant_bindings
                    .iter()
                    .flat_map(|b| b.buffers.clone())
                    .collect()
            )
        ));
        make = format!(
            "{make}\n                .with_constants(&__constants, __constant_images, __constant_buffers)"
        );
    }
    let safety = format!(
        "// SAFETY: generated from the pass's reflection: pass {index} of `PASS_MAP`, of this \
         kind, with its push constant struct and the accesses of its fields{}.",
        match desc.constants.is_empty() {
            true => "",
            false =>
                ", and its constants laid out as its `PassEntry.constants` say (packed in id order, asserted to be `constants_size` bytes) with their accesses",
        }
    );

    let capabilities = crate::features::capabilities(&pass.spirv);
    let mut gates: Vec<&str> = Vec::new();
    for (capability, _, gate) in crate::features::CAPABILITY_FEATURES {
        if capabilities.contains(capability) && !gates.contains(gate) {
            gates.push(gate);
        }
    }
    let (returns, finish) = match gates.split_last() {
        None => (
            output.clone(),
            format!("{safety}\n        unsafe {{\n            {make}\n        }}"),
        ),
        Some((last, rest)) => {
            let mut gate = format!("::lava::features::gate::{last}");
            for g in rest.iter().rev() {
                gate = format!("(::lava::features::gate::{g}, {gate})");
            }
            (
                format!("::lava::features::Gated<{gate}, {output}>"),
                format!(
                    "::lava::features::gate::<{gate}, _>(move || {{\n            {safety}\n            unsafe {{\n                {make}\n            }}\n        }})"
                ),
            )
        }
    };
    let where_clause = match bounds.is_empty() {
        true => String::new(),
        false => format!("\n    where\n{}", bounds.join("\n")),
    };
    let workgroup = match desc.kind {
        Kind::Compute => {
            let entry = desc.entry("compute").unwrap();
            let [x, y, z] = entry.workgroup_size.map(|n| n.max(1));
            format!(
                r#"
    /// The threads of one workgroup (`[numthreads({x}, {y}, {z})]`).
    pub const WORKGROUP_SIZE: [u32; 3] = [{x}, {y}, {z}];

    /// The workgroups that cover `threads` threads on each axis, for `CommandBuffer::compute`:
    /// each count rounded up to whole workgroups.
    pub const fn groups(threads: [u32; 3]) -> [u32; 3] {{
        [
            threads[0].div_ceil({x}),
            threads[1].div_ceil({y}),
            threads[2].div_ceil({z}),
        ]
    }}
"#
            )
        }
        _ => String::new(),
    };
    let constants_doc = match desc.constants.is_empty() {
        true => "",
        false => " and specialization constants",
    };
    code.push_str(&format!(
        r#"
pub struct {name};

impl {name} {{{workgroup}
    /// The bindings of its push constants{constants_doc}, plus `accesses`: the images and
    /// buffers it reaches without naming them in a field (through a handle or pointer in a
    /// buffer, a `DynImg`), tracked like its fields' accesses.
    #[track_caller]
    #[allow(clippy::type_complexity, unused_variables)]
    pub fn push_bindings<{generics}>({params}) -> {returns}{where_clause}
    {{
        {body}
        {finish}
    }}
}}
"#,
        generics = fn_generics.join(", "),
        params = params.join(", "),
        body = body.join("\n        "),
    ));

    // The pass map entry.
    let optional = |stage: &str| match desc.entry(stage) {
        Some(e) => format!("Some(c{:?})", e.name),
        None => "None".to_string(),
    };
    let required = |stage: &str| format!("c{:?}", desc.entry(stage).unwrap().name);
    let kind_fields = match desc.kind {
        Kind::RasterVertex => format!(
            "RasterVertex {{ fragment: {}, vertex: {} }}",
            optional("fragment"),
            required("vertex")
        ),
        Kind::RasterMesh => format!(
            "RasterMesh {{ fragment: {}, mesh: {}, amp: {} }}",
            optional("fragment"),
            required("mesh"),
            optional("amplification")
        ),
        Kind::Compute => {
            let wave = desc.entry("compute").unwrap().wave_size;
            let subgroup = match wave {
                0 => "None".to_string(),
                n => format!("Some({n})"),
            };
            format!(
                "Compute {{ entry: {}, subgroup_size: {subgroup} }}",
                required("compute")
            )
        }
        Kind::RayTracing => {
            let groups = desc.groups.as_ref().unwrap();
            let c = |name: &str| format!("c{name:?}");
            let opt = |entry: &Option<String>| match entry {
                Some(entry) => format!("Some({})", c(entry)),
                None => "None".to_string(),
            };
            let names =
                |entries: &[String]| entries.iter().map(|e| c(e)).collect::<Vec<_>>().join(", ");
            let hit: Vec<String> = (groups.hit.iter())
                .map(|g| {
                    format!(
                        "::lava::pipeline::HitGroup {{ closest_hit: {}, any_hit: {}, intersection: {} }}",
                        opt(&g.closest_hit),
                        opt(&g.any_hit),
                        opt(&g.intersection)
                    )
                })
                .collect();
            format!(
                "RayTracing {{ raygen: {}, miss: &[{}], hit: &[{}], callable: &[{}], max_recursion: {} }}",
                c(&groups.raygen),
                names(&groups.miss),
                hit.join(", "),
                names(&groups.callable),
                groups.max_recursion,
            )
        }
    };
    let (constants, constants_size_expr) = match desc.constants.is_empty() {
        true => ("&[]".to_string(), "0".to_string()),
        false => {
            let entries: Vec<String> = (constant_fields.iter().zip(&constant_bindings))
                .map(|((field, c), b)| {
                    format!(
                        "::lava::vk::SpecializationMapEntry {{ constant_id: {}, offset: ::core::mem::offset_of!({packed_name}, {field}) as u32, size: ::core::mem::size_of::<{}>() }}",
                        c.id, b.gpu_ty
                    )
                })
                .collect();
            (
                format!("&[{}]", entries.join(", ")),
                format!("::core::mem::size_of::<{packed_name}>()"),
            )
        }
    };
    let entry = format!(
        r#"    ::lava::pipeline::PassEntry {{
        name: "{name}",
        spirv: include_bytes!({spirv_path:?}),
        kind: ::lava::pipeline::PassKind::{kind_fields},
        constants: {constants},
        constants_size: {constants_size_expr},
    }},"#
    );
    Ok(GeneratedPass { code, entry })
}

/// The text of a shared struct (behind a pointer): padded, deriving `Pod` (`NoUninit` with
/// handles), `Default` zeroed with null handles, layout-checked, and the clock readout on
/// `profile.slang`'s `Clock`.
fn shared_struct(
    s: &SharedStruct,
    structs: &BTreeMap<String, u32>,
    clocks: &[String],
) -> Result<String, String> {
    let desc = &s.desc;
    let mut fields = Vec::new();
    let mut checks = Vec::new();
    let (types, align) = struct_layout(desc, structs)?;
    for (f, rust) in desc.fields.iter().zip(types) {
        let name = ident(&f.name)?;
        checks.push((name.clone(), f.offset));
        fields.push((name, f.offset, f.size, rust.ty));
    }
    let repr = match align.effective < align.natural {
        true => format!("C, packed({})", align.effective),
        false => "C".to_string(),
    };
    if let Some(path) = s.mirror {
        return Ok(layout_checks(path, &checks, desc.size));
    }
    let name = &desc.name;
    let (body, gaps) = struct_fields(&fields, desc.size, true);
    let handles = desc.fields.iter().any(|f| f.ty.holds_handles());
    let mut zeros: Vec<String> = (fields.iter().zip(&desc.fields))
        .map(|((field, ..), f)| format!("            {field}: {},", zero(&f.ty)))
        .collect();
    zeros.extend(
        gaps.iter()
            .map(|(offset, bytes)| format!("            _pad{offset}: [0; {bytes}],")),
    );
    let mut text = format!(
        r#"
#[derive(Clone, Copy, Debug, {derives})]
#[bytemuck(crate = "::lava::bytemuck")]
#[repr({repr})]
pub struct {name} {{
{body}
}}
{checks}
impl ::core::default::Default for {name} {{
    /// Zeroed, with null image handles.
    fn default() -> Self {{
        Self {{
{zeros}
        }}
    }}
}}
"#,
        derives = derives(handles),
        checks = layout_checks(name, &checks, desc.size),
        zeros = zeros.join("\n"),
    );
    // The clock readout: `take` on `Clock`, `regions` on a struct of only `Clock`s.
    if is_clock(desc) {
        text.push_str(&format!(
            r#"
impl {name} {{
    /// The ticks and subgroups `CLOCK_REGION` added since the last take, leaving the clock
    /// zeroed. Only while no submitted work can still add to it.
    pub fn take(&mut self) -> (u64, u64) {{
        let stripes = self.counters.iter_mut().step_by({CLOCK_STRIPE_WORDS});
        let sum: u64 = stripes.map(::core::mem::take).sum();
        (sum & ((1 << {CLOCK_TICK_BITS}) - 1), sum >> {CLOCK_TICK_BITS})
    }}
}}
"#
        ));
    } else if !desc.fields.is_empty()
        && desc
            .fields
            .iter()
            .all(|f| matches!(&f.ty, Ty::Struct(s) if clocks.contains(&s.name)))
    {
        let regions: Vec<String> = (fields.iter())
            .map(|(f, ..)| format!("({:?}, &mut self.{f})", f.trim_start_matches("r#")))
            .collect();
        text.push_str(&format!(
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
        ));
    }
    let mut pointees = Vec::new();
    for f in &desc.fields {
        pointee_checks(&f.ty, structs, &mut pointees);
    }
    text.extend(pointees);
    Ok(text)
}

/// Each thread pays for a global session (Slang's core module) and its own imports: lava-tests'
/// 30 passes took 1.25 s on one thread, 0.71 s on 4, 1.0 s on 8 and 1.9 s on 16 (2026-10-24).
const MAX_THREADS: usize = 4;

/// Only a changed file makes cargo rebuild the crate that includes it.
fn write_if_changed(path: &Path, bytes: &[u8]) {
    if fs::read(path).ok().as_deref() != Some(bytes) {
        fs::write(path, bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    }
}

/// Compiles every file on up to `MAX_THREADS` threads (one global session and session each,
/// which loads `import`ed modules once per thread), into results in file order.
fn compile_all(files: &[String], options: &compile::Options) -> Vec<CompiledFile> {
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    let threads = cores.min(MAX_THREADS).min(files.len());
    let next = AtomicUsize::new(0);
    let results: Vec<Mutex<Option<CompiledFile>>> =
        files.iter().map(|_| Mutex::new(None)).collect();
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                let global = compile::slang::GlobalSession::new().expect("no Slang global session");
                let session = compile::session(&global, options);
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(file) = files.get(i) else { break };
                    let compiled = compile::compile_file(&session, file);
                    *results[i].lock().unwrap() = Some(compiled);
                }
            });
        }
    });
    results
        .into_iter()
        .map(|r| r.into_inner().unwrap().unwrap())
        .collect()
}

impl Shaders {
    /// Compiles every pass into `$OUT_DIR/lava/passes/<Pass>.spv` and writes the bindings to
    /// `$OUT_DIR/lava/bindings.rs`, forwarding Slang's warnings as cargo warnings. Panics once,
    /// listing every pass that failed with its file and field.
    pub fn compile(self) {
        for dir in self.passes.iter().chain(&self.includes).map(String::as_str) {
            println!("cargo::rerun-if-changed={dir}");
        }
        println!("cargo::rerun-if-changed={SLANG_DIR}");
        let out_dir =
            Path::new(&env::var("OUT_DIR").expect("called from a build script")).join("lava");
        let passes_dir = out_dir.join("passes");
        fs::create_dir_all(&passes_dir).unwrap();

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

        let search_paths: Vec<&str> = self.includes.iter().map(String::as_str).collect();
        let defines: Vec<(&str, &str)> = (self.defines.iter())
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect();
        let options = compile::Options {
            search_paths: &search_paths,
            defines: &defines,
            optimization: self.optimization,
            debug_info: self.debug_info,
        };
        let compiled = compile_all(&pass_files, &options);

        // Warnings first, then every failure at once.
        let mut failures: Vec<String> = Vec::new();
        let mut passes: Vec<(&CompiledPass, &str)> = Vec::new();
        let mut names: BTreeMap<&str, &str> = BTreeMap::new();
        for (file, source) in compiled.iter().zip(&pass_files) {
            for warning in &file.warnings {
                for line in warning.lines().filter(|l| !l.trim().is_empty()) {
                    println!("cargo::warning={line}");
                }
            }
            match &file.passes {
                Err(errors) => failures.push(format!("{source}:\n  {}", errors.join("\n  "))),
                Ok(file_passes) => {
                    for pass in file_passes {
                        if let Some(other) = names.insert(&pass.name, source) {
                            failures.push(format!(
                                "{source}: the pass `{}` comes from {other} too: rename one",
                                pass.name
                            ));
                        }
                        passes.push((pass, source));
                    }
                }
            }
        }

        let mut generator = Generator {
            structs: BTreeMap::new(),
            exports: BTreeMap::new(),
            items: BTreeMap::new(),
        };
        let mut file_errors: BTreeMap<&str, Vec<String>> = BTreeMap::new();
        for (pass, source) in &passes {
            let desc = &pass.desc;
            let mut errors = Vec::new();
            let fields = (desc.push_constants.iter())
                .flat_map(|pc| pc.fields.iter().map(|f| (&f.name, &f.ty)));
            let constants = desc.constants.iter().map(|c| (&c.name, &c.ty));
            for (name, ty) in fields.chain(constants) {
                if let Err(e) = generator.collect(ty, source) {
                    errors.push(format!("`{name}`: {e}"));
                }
            }
            let name = &pass.name;
            for item in [
                name.clone(),
                format!("C{name}Bindings"),
                format!("{name}Args"),
                format!("{name}Constants"),
                format!("C{name}Constants"),
            ] {
                if let Err(e) = generator.claim(&item, source) {
                    errors.push(e);
                }
            }
            for export in &desc.exports {
                match generator.exports.get(&export.name) {
                    Some((other, other_source)) if other != export => errors.push(format!(
                        "`[RustConst]` `{}` is `{}: {}` here but `{}: {}` in {other_source}",
                        export.name, export.value, export.ty, other.value, other.ty
                    )),
                    Some(_) => {}
                    None => {
                        generator
                            .exports
                            .insert(export.name.clone(), (export.clone(), source.to_string()));
                    }
                }
            }
            file_errors.entry(source).or_default().extend(errors);
        }
        let shared: Vec<(String, String)> = (generator.structs.iter())
            .filter(|(_, s)| s.mirror.is_none())
            .map(|(name, s)| (name.clone(), s.source.clone()))
            .collect();
        for (name, source) in shared {
            if let Err(e) = generator.claim(&name, &source) {
                failures.push(e);
            }
        }
        let aligns = match generator.alignments() {
            Ok(aligns) => aligns,
            Err(e) => {
                failures.push(e);
                BTreeMap::new()
            }
        };

        let mut bindings = String::from(
            "// Generated by lava-build-slang from the application's Slang passes; included by\n\
             // `lava::include_passes!()`. Every path is absolute, so nothing is imported.\n",
        );
        let mut pass_map: Vec<String> = Vec::new();
        let mut built_passes: Vec<String> = Vec::new();
        let mut spirv_files = Vec::new();
        for (pass, source) in &passes {
            let spirv = passes_dir.join(format!("{}.spv", pass.name));
            match generate_pass(pass, source, pass_map.len(), &spirv, &aligns) {
                Ok(generated) => {
                    bindings.push_str(&generated.code);
                    pass_map.push(generated.entry);
                }
                Err(errors) => file_errors.entry(source).or_default().extend(errors),
            }
            built_passes.push(format!(
                "        BuiltPass {{ name: {:?}, source: {source:?}, entry: {:?}, layout: {:#x} }},",
                pass.name,
                pass.entry,
                pass.desc.layout_hash()
            ));
            spirv_files.push((spirv, &pass.spirv));
        }
        let clocks: Vec<String> = (generator.structs.values())
            .filter(|s| is_clock(&s.desc))
            .map(|s| s.desc.name.clone())
            .collect();
        let mut struct_text = String::new();
        for s in generator.structs.values() {
            match shared_struct(s, &aligns, &clocks) {
                Ok(text) => struct_text.push_str(&text),
                Err(e) => file_errors.entry(&s.source).or_default().push(e),
            }
        }
        for (source, errors) in file_errors {
            if !errors.is_empty() {
                failures.push(format!("{source}:\n  {}", errors.join("\n  ")));
            }
        }
        if !failures.is_empty() {
            panic!(
                "lava-build-slang: {} pass file(s) failed:\n\n{}\n",
                failures.len(),
                failures.join("\n\n")
            );
        }

        // Pointee size asserts of the push constants' buffers.
        let mut pointees = Vec::new();
        for (pass, _) in &passes {
            for f in pass.desc.push_constants.iter().flat_map(|pc| &pc.fields) {
                pointee_checks(&f.ty, &aligns, &mut pointees);
            }
        }
        pointees.sort();
        pointees.dedup();

        for (export, source) in generator.exports.values() {
            let file = Path::new(source)
                .file_name()
                .and_then(|f| f.to_str())
                .unwrap_or(source);
            bindings.push_str(&format!(
                "\n/// `[RustConst]` of `{file}`.\npub const {}: {} = {};\n",
                export.name, export.ty, export.value
            ));
        }
        bindings.push_str(&format!(
            r#"
/// Every pass; hand it to `lava::init`.
pub static PASS_MAP: ::lava::pipeline::PassMap = ::lava::pipeline::PassMap {{
    passes: &[
{passes}
    ],
}};
"#,
            passes = pass_map.join("\n"),
        ));
        bindings.push_str(&struct_text);
        bindings.extend(pointees);
        for (path, spirv) in spirv_files {
            write_if_changed(&path, spirv);
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
    debug_info: {debug_info},
    passes: &[
{passes}
    ],
}}
"#,
            optimization = self.optimization,
            debug_info = self.debug_info,
            passes = built_passes.join("\n"),
        );
        write_if_changed(&out_dir.join("hot_reload.rs"), build.as_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::describe::{Field, Pointee};

    fn rust(ty: &Ty, at: u32) -> String {
        rust_ty(ty, at, &BTreeMap::new()).unwrap().ty
    }

    fn matrix(scalar: Scalar, rows: u32, columns: u32, column_stride: u32) -> Ty {
        Ty::Matrix {
            scalar,
            rows,
            columns,
            column_stride,
        }
    }

    #[test]
    fn types_map_to_rust_types_of_their_layout() {
        assert_eq!(rust(&Ty::Scalar(Scalar::Bool), 4), "u32");
        assert_eq!(rust(&Ty::Scalar(Scalar::F16), 2), "::lava::half::F16");
        assert_eq!(
            rust(&Ty::Vector(Scalar::F16, 3), 2),
            "[::lava::half::F16; 3]"
        );
        assert_eq!(rust(&Ty::Vector(Scalar::Bool, 3), 4), "::lava::glam::UVec3");
        assert_eq!(
            rust(&Ty::Vector(Scalar::I64, 3), 8),
            "::lava::glam::I64Vec3"
        );
        assert_eq!(rust(&Ty::Vector(Scalar::F64, 2), 8), "::lava::glam::DVec2");
        assert_eq!(
            rust(&Ty::Vector(Scalar::U16, 2), 4),
            "::lava::glam::U16Vec2"
        );
        // glam's `Vec4` is SIMD-aligned: only where 16 bytes are.
        assert_eq!(rust(&Ty::Vector(Scalar::F32, 4), 16), "::lava::glam::Vec4");
        assert_eq!(rust(&Ty::Vector(Scalar::F32, 4), 4), "[f32; 4]");
        assert_eq!(
            rust(&matrix(Scalar::F32, 3, 3, 12), 4),
            "::lava::glam::Mat3"
        );
        assert_eq!(
            rust(&matrix(Scalar::F32, 4, 4, 16), 16),
            "::lava::glam::Mat4"
        );
        assert_eq!(rust(&matrix(Scalar::F32, 4, 4, 16), 4), "[[f32; 4]; 4]");
        // std430's `float3x3`: columns padded to 16 bytes, like `Mat3A`.
        assert_eq!(rust(&matrix(Scalar::F32, 3, 3, 16), 16), "[[f32; 4]; 3]");
        assert_eq!(
            rust(&matrix(Scalar::F32, 3, 4, 12), 4),
            "[[f32; 3]; 4]",
            "4 columns of 3 rows"
        );
        assert_eq!(rust(&matrix(Scalar::I32, 4, 4, 16), 16), "[[i32; 4]; 4]");
        assert_eq!(
            rust(&matrix(Scalar::F64, 2, 2, 16), 8),
            "::lava::glam::DMat2"
        );
        let image = Image {
            shape: Shape::D2,
            usage: 3,
            texel: (Scalar::F32, 4),
        };
        assert_eq!(
            rust(&Ty::Image(image), 4),
            "::lava::bindless::ImageHandle<::lava::bindless::Combined>"
        );
        assert_eq!(rust(&Ty::DynImg, 4), "::lava::bindless::ImageHandle");
        assert_eq!(rust(&Ty::Accel, 8), "u64");
    }

    #[test]
    fn arrays_and_pointees_follow_their_stride() {
        let array = |element: Ty, count, stride| Ty::Array {
            element: Box::new(element),
            count,
            stride,
        };
        assert_eq!(
            rust(&array(Ty::Vector(Scalar::F32, 3), 4, 12), 16),
            "[::lava::glam::Vec3; 4]"
        );
        // std430 arrays of `float3`: elements padded to 16 bytes.
        assert_eq!(
            rust(&array(Ty::Vector(Scalar::F32, 3), 4, 16), 16),
            "[[f32; 4]; 4]"
        );
        let image = Image {
            shape: Shape::D2,
            usage: 0,
            texel: (Scalar::F32, 4),
        };
        assert_eq!(
            rust(&array(Ty::Image(image), 40, 4), 4),
            "[::lava::bindless::ImageHandle<::lava::bindless::Sampled>; 40]"
        );
        let padded_struct = Ty::Struct(Struct {
            name: "S".into(),
            fields: Vec::new(),
            size: 12,
        });
        let error = rust_ty(&array(padded_struct, 2, 16), 16, &BTreeMap::new())
            .err()
            .unwrap();
        assert!(
            error.contains("16 bytes apart that are 12 bytes long"),
            "{error}"
        );
        let pointee = Pointee {
            ty: Ty::Vector(Scalar::F32, 3),
            stride: 12,
        };
        assert_eq!(
            pointee_ty(&pointee, &BTreeMap::new()).unwrap(),
            "::lava::glam::Vec3"
        );
        let image_pointee = Pointee {
            ty: Ty::Image(image),
            stride: 4,
        };
        assert_eq!(
            pointee_ty(&image_pointee, &BTreeMap::new()).unwrap(),
            "::lava::bindless::ImageHandle<::lava::bindless::Sampled>"
        );
    }

    #[test]
    fn gaps_become_zeroed_pads_and_tails_too() {
        let fields = [
            ("a".to_string(), 0, 4, "u32".to_string()),
            ("b".to_string(), 8, 8, "u64".to_string()),
        ];
        let (body, gaps) = struct_fields(&fields, 24, true);
        assert_eq!(gaps, [(4, 4), (16, 8)]);
        assert!(
            body.contains("pub _pad4: [u8; 4],") && body.contains("pub _pad16: [u8; 8],"),
            "{body}"
        );
        let (body, _) = struct_fields(&fields, 24, false);
        assert!(
            body.contains("    _pad4: [u8; 4],") && !body.contains("pub _pad"),
            "{body}"
        );
    }

    fn field(name: &str, offset: u32, ty: Ty) -> Field {
        let size = slang_size(&ty).unwrap();
        Field {
            name: name.into(),
            offset,
            size,
            ty,
        }
    }

    #[test]
    fn structs_of_one_name_have_one_layout() {
        let mut generator = Generator {
            structs: BTreeMap::new(),
            exports: BTreeMap::new(),
            items: BTreeMap::new(),
        };
        let float = Ty::Struct(Struct {
            name: "Foo".into(),
            fields: vec![field("x", 0, Ty::Scalar(Scalar::F32))],
            size: 4,
        });
        let uint = Ty::Struct(Struct {
            name: "Foo".into(),
            fields: vec![field("x", 0, Ty::Scalar(Scalar::U32))],
            size: 4,
        });
        generator.collect(&float, "a.slang").unwrap();
        generator.collect(&float, "b.slang").unwrap();
        let error = generator.collect(&uint, "c.slang").unwrap_err();
        assert!(
            error.contains("c.slang") && error.contains("a.slang"),
            "{error}"
        );
        generator.claim("Foo", "a.slang").unwrap();
        let clash = generator.claim("Foo", "d.slang").unwrap_err();
        assert!(
            clash.contains("d.slang") && clash.contains("a.slang"),
            "{clash}"
        );
    }

    #[test]
    fn structs_slang_doesnt_round_up_are_packed() {
        let inner = Struct {
            name: "Inner".into(),
            fields: vec![
                field("wide", 0, Ty::Scalar(Scalar::U64)),
                field("narrow", 8, Ty::Scalar(Scalar::U32)),
            ],
            size: 12,
        };
        let (_, align) = struct_layout(&inner, &BTreeMap::new()).unwrap();
        assert_eq!((align.natural, align.effective), (8, 4));
        let shared = SharedStruct {
            desc: inner,
            source: "a.slang".into(),
            mirror: None,
        };
        let text = shared_struct(&shared, &BTreeMap::new(), &[]).unwrap();
        assert!(text.contains("#[repr(C, packed(4))]"), "{text}");
        assert!(
            text.contains("impl ::core::default::Default for Inner"),
            "{text}"
        );
    }

    #[test]
    fn keywords_are_raw_identifiers() {
        assert_eq!(ident("type").unwrap(), "r#type");
        assert_eq!(ident("dyn").unwrap(), "r#dyn");
        assert_eq!(ident("color").unwrap(), "color");
        assert!(ident("self").is_err());
    }

    #[test]
    fn only_profiles_clock_gets_the_readout() {
        let counters = Ty::Array {
            element: Box::new(Ty::Scalar(Scalar::U64)),
            count: CLOCK_STRIPES * CLOCK_STRIPE_WORDS,
            stride: 8,
        };
        let clock = Struct {
            name: "Clock".into(),
            fields: vec![field("counters", 0, counters)],
            size: 4096,
        };
        assert!(is_clock(&clock));
        let other = Struct {
            name: "Clock".into(),
            fields: vec![field("ticks", 0, Ty::Scalar(Scalar::U64))],
            size: 8,
        };
        assert!(!is_clock(&other));
        let shared = SharedStruct {
            desc: other,
            source: "a.slang".into(),
            mirror: None,
        };
        assert!(
            !shared_struct(&shared, &BTreeMap::new(), &[])
                .unwrap()
                .contains("fn take")
        );
    }
}

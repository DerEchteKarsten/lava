//! The plain-data description of a compiled pass that the codegen and the hot reload read from Slang's reflection: its kind and entry points (workgroup and subgroup sizes), push constants and specialization constants as type trees with lava's handle types, its exported constants, and the layout hash of everything its generated bindings depend on
use shader_slang::{
    self as slang, MatrixLayoutMode, ParameterCategory as Cat, ScalarType, TypeKind,
    reflection::{Shader, TypeLayout, VariableLayout},
};

use crate::compile::{self, RayTracingGroups, Stage};
use crate::reflect::{
    CheckedAttributes, CheckedDecl, CheckedGeneric, CheckedName, CheckedShader, CheckedType,
    CheckedTypeLayout, CheckedVariable, CheckedVariableLayout,
};

/// A copy of `lava::pipeline::MAX_PUSH_CONSTANTS_SIZE`.
pub const MAX_PUSH_CONSTANTS_SIZE: u32 = 256;

/// The scalar types of lava's bindings; Slang's others are refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Scalar {
    /// 4 bytes in every layout Slang uses for lava.
    Bool,
    I8,
    U8,
    I16,
    U16,
    F16,
    I32,
    U32,
    F32,
    I64,
    U64,
    F64,
}

impl Scalar {
    fn of(scalar: ScalarType) -> Option<Self> {
        Some(match scalar {
            ScalarType::Bool => Self::Bool,
            ScalarType::Int8 => Self::I8,
            ScalarType::Uint8 => Self::U8,
            ScalarType::Int16 => Self::I16,
            ScalarType::Uint16 => Self::U16,
            ScalarType::Float16 => Self::F16,
            ScalarType::Int32 => Self::I32,
            ScalarType::Uint32 => Self::U32,
            ScalarType::Float32 => Self::F32,
            ScalarType::Int64 => Self::I64,
            ScalarType::Uint64 => Self::U64,
            ScalarType::Float64 => Self::F64,
            _ => return None,
        })
    }

    pub fn size(self) -> u32 {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 | Self::F16 => 2,
            Self::Bool | Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
        }
    }
}

/// The dimension of an `Img`'s views.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Shape {
    D2,
    D3,
    /// `ImgArray`.
    D2Array,
}

/// An `Img` or `ImgArray`: its shape, kind (`U`, 0 sampled, 1 and 2 storage, 3 both) and
/// texels (the shader's element type: scalar and component count).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Image {
    pub shape: Shape,
    pub usage: u8,
    pub texel: (Scalar, u32),
}

/// A field's type as the bindings see it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ty {
    Scalar(Scalar),
    Vector(Scalar, u32),
    /// Column-major: `columns` columns of `rows` scalars, `column_stride` bytes apart.
    Matrix {
        scalar: Scalar,
        rows: u32,
        columns: u32,
        column_stride: u32,
    },
    Array {
        element: Box<Ty>,
        count: u32,
        stride: u32,
    },
    Struct(Struct),
    /// A struct being described further up, reached again through a pointer.
    Named(String),
    /// A raw pointer `T*` (an address without an access); `None` where Slang doesn't lay out
    /// the pointee.
    Pointer(Option<Box<Pointee>>),
    /// `Buf<T>` (read) or `MutBuf<T>` (read and written).
    Buffer {
        pointee: Box<Pointee>,
        writable: bool,
    },
    Image(Image),
    /// `DynImg`: a handle whose image the bindings' `Accesses` name.
    DynImg,
    /// `Accel`: a TLAS's address.
    Accel,
    /// A Slang struct that mirrors a lava type (`MIRRORED_TYPES`): its Rust path and layout.
    Mirror(&'static str, Struct),
}

/// What a pointer points to, `stride` bytes apart.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pointee {
    pub ty: Ty,
    pub stride: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Field {
    pub name: String,
    pub offset: u32,
    pub size: u32,
    pub ty: Ty,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Struct {
    /// Its Rust name: the Slang name, with generic arguments (`Pair<float>` is `PairFloat`).
    pub name: String,
    pub fields: Vec<Field>,
    pub size: u32,
}

impl Ty {
    /// Whether it holds an `ImageHandle` (which isn't `Zeroable`), directly or in a field.
    pub fn holds_handles(&self) -> bool {
        match self {
            Ty::Image(_) | Ty::DynImg => true,
            Ty::Array { element, .. } => element.holds_handles(),
            Ty::Struct(s) => s.fields.iter().any(|f| f.ty.holds_handles()),
            _ => false,
        }
    }
}

/// (Slang name, Rust path) of the Slang structs that mirror a lava type.
pub const MIRRORED_TYPES: [(&str, &str); 2] = [
    (
        "AccelerationStructureInstance",
        "::lava::acceleration_structure::Instance",
    ),
    ("Sampler", "::lava::bindless::SamplerHandle"),
];

/// `ImgArray` is a 2D array (`lava::image::D2Array`).
const IMAGE_TYPES: [&str; 2] = ["Img", "ImgArray"];
const BUFFER_TYPES: [&str; 2] = ["Buf", "MutBuf"];

/// A specialization constant: a scalar, or an `SCONST` of a handle type (a `u64` constant).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Constant {
    pub name: String,
    pub id: u32,
    pub ty: Ty,
}

impl Constant {
    /// Its bytes in the specialization data: 8 for an `SCONST`, 4 for a bool.
    pub fn size(&self) -> u32 {
        match &self.ty {
            Ty::Scalar(scalar) => scalar.size(),
            _ => 8,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub stage: Stage,
    pub name: String,
    pub group: Option<i32>,
    pub max_recursion: Option<i32>,
    /// A compute entry point's `[numthreads]`.
    pub workgroup_size: [u32; 3],
    /// A compute entry point's `[WaveSize(n)]`, 0 without it.
    pub wave_size: u32,
}

/// A `[RustConst] static const` of the pass's module, as a Rust constant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Export {
    pub name: String,
    pub ty: &'static str,
    /// The Rust literal.
    pub value: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Compute,
    RayTracing,
    RasterVertex,
    RasterMesh,
}

#[derive(Debug)]
pub struct PassDesc {
    pub kind: Kind,
    pub entries: Vec<Entry>,
    /// The ray tracing pass's shader groups.
    pub groups: Option<RayTracingGroups>,
    /// The `[[vk::push_constant]]` struct; one without push constants has none.
    pub push_constants: Option<Struct>,
    /// In id order.
    pub constants: Vec<Constant>,
    pub exports: Vec<Export>,
}

impl PassDesc {
    /// The entry point of a stage.
    pub fn entry(&self, stage: Stage) -> Option<&Entry> {
        self.entries.iter().find(|e| e.stage == stage)
    }

    /// A hash of everything the generated bindings depend on (every field's name, offset and
    /// full type, the constants, entry points and their groups and sizes, the exports): a hot
    /// reload that changes it is refused. FNV-1a over fixed-width values, so it is the same on
    /// every host.
    pub fn layout_hash(&self) -> u64 {
        let mut h = Fnv::new();
        h.u8(self.kind as u8);
        h.u32(self.entries.len() as u32);
        for e in &self.entries {
            h.str(e.stage.name());
            h.str(&e.name);
            h.u32(e.group.map_or(u32::MAX, |g| g as u32));
            h.u32(e.max_recursion.map_or(u32::MAX, |g| g as u32));
            e.workgroup_size.iter().for_each(|&n| h.u32(n));
            h.u32(e.wave_size);
        }
        match &self.push_constants {
            None => h.u8(0),
            Some(s) => {
                h.u8(1);
                s.hash_into(&mut h);
            }
        }
        h.u32(self.constants.len() as u32);
        for c in &self.constants {
            h.str(&c.name);
            h.u32(c.id);
            c.ty.hash_into(&mut h);
        }
        h.u32(self.exports.len() as u32);
        for e in &self.exports {
            h.str(&e.name);
            h.str(e.ty);
            h.str(&e.value);
        }
        h.0
    }
}

/// FNV-1a, 64 bits.
pub struct Fnv(pub u64);

impl Fnv {
    pub fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    pub fn u8(&mut self, byte: u8) {
        self.0 = (self.0 ^ byte as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }

    pub fn u32(&mut self, value: u32) {
        value.to_le_bytes().into_iter().for_each(|b| self.u8(b));
    }

    /// Length-prefixed, so `("ab", "c")` and `("a", "bc")` differ.
    pub fn str(&mut self, value: &str) {
        self.u32(value.len() as u32);
        value.bytes().for_each(|b| self.u8(b));
    }
}

impl Default for Fnv {
    fn default() -> Self {
        Self::new()
    }
}

impl Struct {
    fn hash_into(&self, h: &mut Fnv) {
        h.str(&self.name);
        h.u32(self.size);
        h.u32(self.fields.len() as u32);
        for f in &self.fields {
            h.str(&f.name);
            h.u32(f.offset);
            h.u32(f.size);
            f.ty.hash_into(h);
        }
    }
}

impl Ty {
    pub fn hash_into(&self, h: &mut Fnv) {
        let scalar = |h: &mut Fnv, s: &Scalar| h.u8(*s as u8);
        match self {
            Ty::Scalar(s) => {
                h.u8(0);
                scalar(h, s);
            }
            Ty::Vector(s, n) => {
                h.u8(1);
                scalar(h, s);
                h.u32(*n);
            }
            Ty::Matrix {
                scalar: s,
                rows,
                columns,
                column_stride,
            } => {
                h.u8(2);
                scalar(h, s);
                h.u32(*rows);
                h.u32(*columns);
                h.u32(*column_stride);
            }
            Ty::Array {
                element,
                count,
                stride,
            } => {
                h.u8(3);
                h.u32(*count);
                h.u32(*stride);
                element.hash_into(h);
            }
            Ty::Struct(s) => {
                h.u8(4);
                s.hash_into(h);
            }
            Ty::Named(name) => {
                h.u8(5);
                h.str(name);
            }
            Ty::Pointer(pointee) => {
                h.u8(6);
                match pointee {
                    None => h.u8(0),
                    Some(p) => {
                        h.u8(1);
                        h.u32(p.stride);
                        p.ty.hash_into(h);
                    }
                }
            }
            Ty::Buffer { pointee, writable } => {
                h.u8(7);
                h.u8(*writable as u8);
                h.u32(pointee.stride);
                pointee.ty.hash_into(h);
            }
            Ty::Image(image) => {
                h.u8(8);
                h.u8(image.shape as u8);
                h.u8(image.usage);
                scalar(h, &image.texel.0);
                h.u32(image.texel.1);
            }
            Ty::DynImg => h.u8(9),
            Ty::Accel => h.u8(10),
            Ty::Mirror(path, s) => {
                h.u8(11);
                h.str(path);
                s.hash_into(h);
            }
        }
    }
}

/// Describes the type of `t`; `path` is the field's, for the errors, and `outer` the structs
/// being described (a pointer back to one of them is a `Named`).
fn describe_ty(t: &TypeLayout, outer: &mut Vec<String>) -> Result<Ty, String> {
    let scalar = |t: &TypeLayout| {
        let raw = t.scalar_type().unwrap_or(ScalarType::None);
        Scalar::of(raw).ok_or_else(|| format!("the scalar type {raw:?} has no Rust type"))
    };
    Ok(match t.kind() {
        TypeKind::Scalar => Ty::Scalar(scalar(t)?),
        TypeKind::Vector => Ty::Vector(scalar(t)?, t.element_count().unwrap_or(0) as u32),
        TypeKind::Matrix => {
            if t.matrix_layout_mode() == MatrixLayoutMode::RowMajor {
                return Err("a `row_major` matrix: lava lays matrices out by columns".into());
            }
            let (rows, columns) = (t.row_count().unwrap_or(0), t.column_count().unwrap_or(0));
            let size = t.size(Cat::Uniform) as u32;
            if columns == 0 || !size.is_multiple_of(columns) {
                return Err(format!("a {rows}x{columns} matrix of {size} bytes"));
            }
            Ty::Matrix {
                scalar: scalar(t)?,
                rows,
                columns,
                column_stride: size / columns,
            }
        }
        TypeKind::Array => {
            let element = t
                .checked_element_type_layout()
                .ok_or("an array whose elements Slang doesn't lay out")?;
            let count = t.element_count().unwrap_or(0) as u32;
            if count == 0 {
                return Err("an array without a length".into());
            }
            Ty::Array {
                element: Box::new(describe_ty(element, outer)?),
                count,
                stride: t.element_stride(Cat::Uniform) as u32,
            }
        }
        TypeKind::Pointer => match t.checked_element_type_layout() {
            None => Ty::Pointer(None),
            Some(pointee) => Ty::Pointer(Some(Box::new(describe_pointee(pointee, outer)?))),
        },
        TypeKind::Struct => describe_struct(t, outer)?,
        other => return Err(format!("the type kind {other:?} has no Rust type")),
    })
}

fn describe_pointee(pointee: &TypeLayout, outer: &mut Vec<String>) -> Result<Pointee, String> {
    let named = pointee.kind() == TypeKind::Struct
        && rust_struct_name(pointee).is_some_and(|name| outer.contains(&name));
    let ty = match named {
        true => Ty::Named(rust_struct_name(pointee).unwrap()),
        false => describe_ty(pointee, outer)?,
    };
    Ok(Pointee {
        ty,
        stride: pointee.stride(Cat::Uniform) as u32,
    })
}

/// The Rust name of a struct: its Slang name with generic arguments in PascalCase
/// (`Pair<float>` is `PairFloat`).
fn rust_struct_name(t: &TypeLayout) -> Option<String> {
    let full = t.ty().and_then(|ty| ty.checked_full_name());
    let name = full.as_deref().or(t.checked_name())?;
    // A struct in a namespace keeps its own name.
    let name = match name.split_once('<') {
        Some((base, args)) => format!("{}<{args}", base.rsplit('.').next().unwrap_or(base)),
        None => name.rsplit('.').next().unwrap_or(name).to_string(),
    };
    Some(
        name.split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .filter(|part| !part.is_empty())
            .map(capitalize_first)
            .collect(),
    )
}

pub fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

fn describe_struct(t: &TypeLayout, outer: &mut Vec<String>) -> Result<Ty, String> {
    let slang_name = t.checked_name().unwrap_or("");
    if IMAGE_TYPES.contains(&slang_name) {
        return Ok(Ty::Image(describe_image(t)?));
    }
    if slang_name == "DynImg" {
        return Ok(Ty::DynImg);
    }
    if slang_name == "Accel" {
        return Ok(Ty::Accel);
    }
    if BUFFER_TYPES.contains(&slang_name) {
        let pointer = (t.checked_fields().next())
            .and_then(|f| f.checked_type_layout())
            .filter(|p| p.kind() == TypeKind::Pointer)
            .ok_or(format!("a `{slang_name}` without its pointer"))?;
        let pointee = pointer.checked_element_type_layout().ok_or(format!(
            "a `{slang_name}` whose pointee Slang doesn't lay out"
        ))?;
        return Ok(Ty::Buffer {
            pointee: Box::new(describe_pointee(pointee, outer)?),
            writable: slang_name == "MutBuf",
        });
    }
    let name = rust_struct_name(t).ok_or("a struct without a name")?;
    let mirror = MIRRORED_TYPES
        .iter()
        .find(|(slang, _)| *slang == slang_name);
    outer.push(name.clone());
    let fields = describe_fields(t, outer);
    outer.pop();
    let s = Struct {
        name,
        fields: fields?,
        size: t.size(Cat::Uniform) as u32,
    };
    Ok(match mirror {
        Some((_, rust)) => Ty::Mirror(rust, s),
        None => Ty::Struct(s),
    })
}

fn describe_fields(t: &TypeLayout, outer: &mut Vec<String>) -> Result<Vec<Field>, String> {
    t.checked_fields()
        .map(|f| {
            let name = field_name(f)?;
            let layout = f
                .checked_type_layout()
                .ok_or(format!("`{name}`: Slang doesn't lay it out"))?;
            let ty = describe_ty(layout, outer).map_err(|e| prefix(name, e))?;
            Ok(Field {
                name: name.to_string(),
                offset: f.offset(Cat::Uniform) as u32,
                size: layout.size(Cat::Uniform) as u32,
                ty,
            })
        })
        .collect()
}

/// `name: e`, or `name.e` when `e` starts with a nested field's path.
fn prefix(name: &str, e: String) -> String {
    match e.starts_with('`') {
        true => format!("`{name}.{}", &e[1..]),
        false => format!("`{name}`: {e}"),
    }
}

fn field_name(f: &VariableLayout) -> Result<&str, String> {
    (f.variable().and_then(|v| v.checked_name())).ok_or_else(|| "an unnamed field".to_string())
}

fn describe_image(t: &TypeLayout) -> Result<Image, String> {
    let generics = (t.ty().and_then(|ty| ty.generic_container()))
        .ok_or("an image type without its generic arguments")?;
    let usage = generics
        .checked_value_parameters()
        .find(|vp| vp.checked_name() == Some("U"))
        .map(|vp| generics.concrete_int_val(vp))
        .unwrap_or(0);
    if !(0..=3).contains(&usage) {
        return Err(format!("an image of kind {usage}, not 0 to 3"));
    }
    let concrete = |parameter: &str| {
        generics
            .checked_type_parameters()
            .find(|tp| tp.checked_name() == Some(parameter))
            .and_then(|tp| generics.checked_concrete_type(tp))
    };
    let shape = match t.checked_name() {
        Some("ImgArray") => Shape::D2Array,
        _ => match concrete("Shape").and_then(|shape| shape.checked_name()) {
            Some("__Shape2D") => Shape::D2,
            Some("__Shape3D") => Shape::D3,
            other => return Err(format!("the image shape {other:?} has no Rust dimension")),
        },
    };
    let element = concrete("E").ok_or("an image whose texel type Slang doesn't name")?;
    let raw = element.scalar_type();
    let scalar = Scalar::of(raw).ok_or(format!("an image of {raw:?} texels"))?;
    let components = match element.kind() {
        TypeKind::Vector => element.element_count() as u32,
        _ => 1,
    };
    Ok(Image {
        shape,
        usage: usage as u8,
        texel: (scalar, components),
    })
}

/// The parameter categories of uniforms and resources an entry point can't have: lava binds no
/// descriptors and pushes only the push-constant struct.
fn is_resource(category: Cat) -> bool {
    matches!(
        category,
        Cat::Uniform
            | Cat::ConstantBuffer
            | Cat::ShaderResource
            | Cat::UnorderedAccess
            | Cat::SamplerState
            | Cat::DescriptorTableSlot
            | Cat::PushConstantBuffer
            | Cat::SpecializationConstant
            | Cat::Subpass
    )
}

/// Describes a linked pass; `module` is the loaded module (for its exports). Every problem is
/// an error, with the field's path.
pub fn describe(layout: &Shader, module: &slang::Module) -> Result<PassDesc, Vec<String>> {
    let mut errors = Vec::new();
    let mut entries = Vec::new();
    for ep in layout.checked_entry_points() {
        let name = ep.checked_name().unwrap_or("?").to_string();
        let stage = match Stage::of(ep.stage()) {
            Ok(stage) => stage,
            Err(e) => {
                errors.push(format!("entry point `{name}`: {e}"));
                continue;
            }
        };
        for i in 0..ep.parameter_count() {
            let Some(p) = ep.parameter_by_index(i) else {
                continue;
            };
            let category = p.checked_category().unwrap_or(Cat::None);
            if is_resource(category) {
                let param = p.variable().and_then(|v| v.checked_name()).unwrap_or("?");
                errors.push(format!(
                    "entry point `{name}` has the `uniform` parameter `{param}` ({category:?}): \
                     lava pushes only the `[[vk::push_constant]]` struct, so make it a field of \
                     that"
                ));
            }
        }
        let (workgroup_size, wave_size) = match stage {
            Stage::Compute => (
                ep.compute_thread_group_size().map(|n| n as u32),
                ep.compute_wave_size() as u32,
            ),
            _ => ([0; 3], 0),
        };
        entries.push(Entry {
            stage,
            group: compile::entry_attribute(ep, "Group"),
            max_recursion: compile::entry_attribute(ep, "MaxRecursion"),
            name,
            workgroup_size,
            wave_size,
        });
    }

    let mut push_constants = None;
    let mut constants = Vec::new();
    for p in layout.checked_parameters() {
        let name = p.variable().and_then(|v| v.checked_name()).unwrap_or("?");
        match p.checked_category() {
            Some(Cat::PushConstantBuffer) => {
                let Some(t) = p
                    .checked_type_layout()
                    .and_then(|t| t.checked_element_type_layout())
                else {
                    errors.push(format!("the push constant buffer `{name}` has no layout"));
                    continue;
                };
                if push_constants.is_some() {
                    errors.push(format!(
                        "a second push constant buffer `{name}`: a pass pushes one struct"
                    ));
                    continue;
                }
                match describe_ty(t, &mut Vec::new()) {
                    Ok(Ty::Struct(s)) => push_constants = Some(s),
                    Ok(_) => errors.push(format!("the push constants `{name}` aren't a struct")),
                    Err(e) => errors.push(format!("push constant {e}")),
                }
            }
            Some(Cat::SpecializationConstant) => match describe_constant(layout, p, name) {
                Ok(c) => constants.push(c),
                Err(e) => errors.push(e),
            },
            category => errors.push(format!(
                "global parameter `{name}` ({category:?}): a pass has no descriptors or uniforms \
                 of its own; reach resources through push-constant handles (`Img`, `Buf`, \
                 `MutBuf`, `Accel`) instead"
            )),
        }
    }
    constants.sort_by_key(|c| c.id);
    if let Some(pc) = &push_constants {
        if pc.size > MAX_PUSH_CONSTANTS_SIZE {
            errors.push(format!(
                "the push constants take {} bytes, more than the {MAX_PUSH_CONSTANTS_SIZE} a pass \
                 can have",
                pc.size
            ));
        }
        for f in &pc.fields {
            if let Err(e) = reject_nested_handles(&f.name, &f.ty) {
                errors.push(e);
            }
        }
    }

    let exports = match exports(module) {
        Ok(exports) => exports,
        Err(e) => {
            errors.extend(e);
            Vec::new()
        }
    };

    let mut stages: Vec<Stage> = entries.iter().map(|e| e.stage).collect();
    stages.sort_unstable();
    use Stage::{Compute, Fragment, Mesh, RayGen, Task, Vertex};
    let kind = match stages.as_slice() {
        // Without a fragment shader, a raster pass draws depth and stencil alone.
        [Fragment, Vertex] | [Vertex] => Some(Kind::RasterVertex),
        [Fragment, Mesh] | [Task, Fragment, Mesh] | [Mesh] | [Task, Mesh] => Some(Kind::RasterMesh),
        [Compute] => Some(Kind::Compute),
        stages if stages.contains(&RayGen) && stages.iter().all(|s| s.is_ray_tracing()) => {
            Some(Kind::RayTracing)
        }
        other => {
            let names: Vec<_> = other.iter().map(|s| s.name()).collect();
            errors.push(format!(
                "the entry points {names:?} don't match any pass pattern"
            ));
            None
        }
    };
    let groups = match kind {
        Some(Kind::RayTracing) => {
            let entries: Vec<compile::Entry> = (entries.iter())
                .map(|e| compile::Entry {
                    stage: e.stage,
                    name: &e.name,
                    group: e.group,
                    max_recursion: e.max_recursion,
                })
                .collect();
            match compile::ray_tracing_groups(&entries) {
                Ok(groups) => Some(groups),
                Err(e) => {
                    errors.push(e);
                    None
                }
            }
        }
        _ => None,
    };
    match (kind, errors.is_empty()) {
        (Some(kind), true) => Ok(PassDesc {
            kind,
            entries,
            groups,
            push_constants,
            constants,
            exports,
        }),
        _ => Err(errors),
    }
}

/// `SCONST(T, name)` (`bindless.slang`) declares `name_sconst` with an `SConst` attribute naming
/// `T`; a plain `[SpecializationConstant]` is a scalar.
fn describe_constant(layout: &Shader, p: &VariableLayout, name: &str) -> Result<Constant, String> {
    let id = p.offset(Cat::SpecializationConstant) as u32;
    let attribute = p.variable().and_then(|v| {
        v.checked_user_attributes()
            .find(|a| a.checked_name() == Some("SConst"))
    });
    let Some(attribute) = attribute else {
        let t = (p.checked_type_layout()).ok_or(format!("constant `{name}` has no layout"))?;
        return match describe_ty(t, &mut Vec::new()) {
            Ok(ty @ Ty::Scalar(_)) => Ok(Constant {
                name: name.to_string(),
                id,
                ty,
            }),
            Ok(other) => Err(format!("constant `{name}` is a {other:?}, not a scalar")),
            Err(e) => Err(format!("constant `{name}`: {e}")),
        };
    };
    let ty_name = attribute.argument_value_string(0).unwrap_or("?");
    let t = layout
        .find_type_by_name(ty_name)
        .and_then(|t| layout.type_layout(t, slang::LayoutRules::Default))
        .ok_or(format!("unknown type `{ty_name}` of SCONST `{name}`"))?;
    let ty = describe_ty(t, &mut Vec::new()).map_err(|e| format!("SCONST `{name}`: {e}"))?;
    if !matches!(ty, Ty::Buffer { .. } | Ty::Image(_) | Ty::Accel) {
        return Err(format!(
            "SCONST `{name}` has type `{ty_name}`, which isn't a buffer, image or acceleration \
             structure handle"
        ));
    }
    Ok(Constant {
        name: name.strip_suffix("_sconst").unwrap_or(name).to_string(),
        id,
        ty,
    })
}

/// Handles inside push-constant structs or arrays would be untracked.
fn reject_nested_handles(path: &str, ty: &Ty) -> Result<(), String> {
    let nested = |ty: &Ty| {
        matches!(
            ty,
            Ty::Image(_) | Ty::DynImg | Ty::Buffer { .. } | Ty::Accel
        )
    };
    match ty {
        Ty::Array { element, .. } => {
            if nested(element) {
                return Err(format!(
                    "push constant `{path}` is an array of handles, which lava can't track: give \
                     each its own field"
                ));
            }
            reject_nested_handles(path, element)
        }
        Ty::Struct(s) => {
            for f in &s.fields {
                let path = format!("{path}.{}", f.name);
                if nested(&f.ty) {
                    return Err(format!(
                        "push constant `{path}` is a handle inside a struct, which lava can't \
                         track: make it a field of the push-constant struct itself"
                    ));
                }
                reject_nested_handles(&path, &f.ty)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// The module's `[RustConst] static const` scalars, as Rust constants.
fn exports(module: &slang::Module) -> Result<Vec<Export>, Vec<String>> {
    let mut exports = Vec::new();
    let mut errors = Vec::new();
    for decl in module.module_reflection().checked_children() {
        let Some(var) = decl.checked_variable() else {
            continue;
        };
        if !var
            .checked_user_attributes()
            .any(|a| a.checked_name() == Some("RustConst"))
        {
            continue;
        }
        let name = var.checked_name().unwrap_or("?").to_string();
        let scalar = var
            .checked_ty()
            .filter(|t| t.kind() == TypeKind::Scalar)
            .and_then(|t| Scalar::of(t.scalar_type()));
        let int = var.default_value_int();
        let export = |ty: &'static str, value: Option<String>| match value {
            Some(value) => Ok(Export {
                name: name.clone(),
                ty,
                value,
            }),
            None => Err(format!("`[RustConst]` `{name}` has no constant value")),
        };
        let export = match scalar {
            Some(Scalar::Bool) => export("bool", int.map(|v| (v != 0).to_string())),
            Some(Scalar::I32) => export("i32", int.map(|v| (v as i32).to_string())),
            Some(Scalar::U32) => export("u32", int.map(|v| (v as u32).to_string())),
            Some(Scalar::I64) => export("i64", int.map(|v| v.to_string())),
            Some(Scalar::U64) => export("u64", int.map(|v| (v as u64).to_string())),
            Some(Scalar::F32) => {
                let value = var.default_value_float();
                export("f32", value.map(|v| format!("{v:?}")))
            }
            _ => Err(format!(
                "`[RustConst]` `{name}` isn't a `bool`, `int`, `uint`, `int64_t`, `uint64_t` or \
                 `float`"
            )),
        };
        match export {
            Ok(export) => exports.push(export),
            Err(e) => errors.push(e),
        }
    }
    match errors.is_empty() {
        true => Ok(exports),
        false => Err(errors),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::compile::{self, CompiledFile};

    /// Compiles `source` as the pass file `{stem}.slang` in a directory of its own.
    pub(crate) fn compile(stem: &str, source: &str) -> CompiledFile {
        compile_with(stem, source, false)
    }

    pub(crate) fn compile_with(stem: &str, source: &str, debug_info: bool) -> CompiledFile {
        let dir = std::env::temp_dir().join(format!(
            "lava-build-slang-{}-{stem}-{}",
            std::process::id(),
            std::thread::current()
                .name()
                .unwrap_or("")
                .replace("::", "-")
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{stem}.slang"));
        std::fs::write(&path, source).unwrap();
        let global = slang::GlobalSession::new().unwrap();
        let options = compile::Options {
            search_paths: &[],
            defines: &[],
            optimization: crate::Optimization::None,
            debug_info,
        };
        let session = compile::session(&global, &options);
        compile::compile_file(&session, path.to_str().unwrap())
    }

    /// The description of the one pass of a compute pass file with these push constant
    /// fields and declarations.
    pub(crate) fn describe_fields(
        stem: &str,
        decls: &str,
        fields: &str,
    ) -> Result<PassDesc, String> {
        let source = format!(
            "#include \"bindless.slang\"\n{decls}\nstruct Pc {{ {fields} }}\n[[vk::push_constant]] Pc pc;\n\
             [shader(\"compute\")] [numthreads(1, 1, 1)] void main() {{}}\n"
        );
        let compiled = compile(stem, &source);
        let mut passes = compiled.passes.map_err(|e| e.join("\n"))?;
        assert_eq!(passes.len(), 1);
        Ok(passes.remove(0).desc)
    }

    fn hash(stem: &str, fields: &str) -> u64 {
        describe_fields(stem, "", fields).unwrap().layout_hash()
    }

    #[test]
    fn fnv_is_fnv_1a_64() {
        assert_eq!(Fnv::new().0, 0xcbf2_9ce4_8422_2325);
        let mut h = Fnv::new();
        h.u8(b'a');
        assert_eq!(h.0, 0xaf63_dc4c_8601_ec8c, "FNV-1a of \"a\"");
        let (mut ab, mut a) = (Fnv::new(), Fnv::new());
        ab.str("ab");
        ab.str("c");
        a.str("a");
        a.str("bc");
        assert_ne!(ab.0, a.0, "strings are length-prefixed");
    }

    #[test]
    fn layout_hashes_cover_field_types() {
        let base = hash("hash_base", "Tex2D<float4> image; float x; Buf<uint> data;");
        assert_eq!(
            base,
            hash("hash_same", "Tex2D<float4> image; float x; Buf<uint> data;")
        );
        for (stem, fields) in [
            (
                "hash_image_to_buffer",
                "Buf<uint> image; float x; Buf<uint> data;",
            ),
            (
                "hash_float_to_uint",
                "Tex2D<float4> image; uint x; Buf<uint> data;",
            ),
            (
                "hash_shape",
                "Tex3D<float4> image; float x; Buf<uint> data;",
            ),
            (
                "hash_kind",
                "Image2D<float4> image; float x; Buf<uint> data;",
            ),
            (
                "hash_element",
                "Tex2D<uint4> image; float x; Buf<uint> data;",
            ),
            (
                "hash_pointee",
                "Tex2D<float4> image; float x; Buf<int> data;",
            ),
            (
                "hash_written",
                "Tex2D<float4> image; float x; MutBuf<uint> data;",
            ),
            ("hash_name", "Tex2D<float4> image; float y; Buf<uint> data;"),
        ] {
            assert_ne!(base, hash(stem, fields), "{stem}");
        }
    }

    #[test]
    fn handle_types_map_to_lava_types() {
        let desc = describe_fields(
            "handles",
            "struct Node { Tex2D<float4> tex; Accel scene; Buf<float> data; DynImg<float4> any; Node* next; }",
            "Buf<Tex2D<float4>> textures; Buf<Node> nodes; Accel tlas; Sampler s; DynImg<float4> dynamic; uint* raw;",
        )
        .unwrap();
        let fields = desc.push_constants.unwrap().fields;
        let image = Image {
            shape: Shape::D2,
            usage: 0,
            texel: (Scalar::F32, 4),
        };
        let Ty::Buffer {
            pointee,
            writable: false,
        } = &fields[0].ty
        else {
            panic!("{:?}", fields[0].ty)
        };
        assert_eq!((&pointee.ty, pointee.stride), (&Ty::Image(image), 4));
        let Ty::Buffer { pointee, .. } = &fields[1].ty else {
            panic!()
        };
        let Ty::Struct(node) = &pointee.ty else {
            panic!()
        };
        assert_eq!(node.fields[0].ty, Ty::Image(image));
        assert_eq!(node.fields[1].ty, Ty::Accel);
        assert!(matches!(node.fields[2].ty, Ty::Buffer { .. }));
        assert_eq!(node.fields[3].ty, Ty::DynImg);
        let Ty::Pointer(Some(next)) = &node.fields[4].ty else {
            panic!()
        };
        assert_eq!(
            next.ty,
            Ty::Named("Node".into()),
            "a pointer back to its struct"
        );
        assert_eq!(fields[2].ty, Ty::Accel);
        assert!(matches!(
            fields[3].ty,
            Ty::Mirror("::lava::bindless::SamplerHandle", _)
        ));
        assert_eq!(fields[4].ty, Ty::DynImg);
        assert!(matches!(&fields[5].ty, Ty::Pointer(Some(p)) if p.ty == Ty::Scalar(Scalar::U32)));
    }

    #[test]
    fn scalar_vector_and_matrix_layouts_are_described() {
        let desc = describe_fields(
            "scalars",
            "struct S { half h; half3 h3; int16_t s; uint8_t u; bool3 b; int64_t3 l; float3x3 m; float3x4 n; int4x4 i; }",
            "Buf<S> s; float3x3 m; float3 v; float after;",
        )
        .unwrap();
        let pc = desc.push_constants.unwrap();
        let Ty::Buffer { pointee, .. } = &pc.fields[0].ty else {
            panic!()
        };
        let Ty::Struct(s) = &pointee.ty else { panic!() };
        let types: Vec<&Ty> = s.fields.iter().map(|f| &f.ty).collect();
        assert_eq!(types[0], &Ty::Scalar(Scalar::F16));
        assert_eq!(types[1], &Ty::Vector(Scalar::F16, 3));
        assert_eq!(types[2], &Ty::Scalar(Scalar::I16));
        assert_eq!(types[3], &Ty::Scalar(Scalar::U8));
        assert_eq!(types[4], &Ty::Vector(Scalar::Bool, 3));
        assert_eq!(s.fields[4].size, 12, "bools are 4 bytes");
        assert_eq!(types[5], &Ty::Vector(Scalar::I64, 3));
        // Behind a pointer the layout is scalar: columns 12 bytes apart.
        let matrix = |rows, columns, column_stride, scalar| Ty::Matrix {
            scalar,
            rows,
            columns,
            column_stride,
        };
        assert_eq!(types[6], &matrix(3, 3, 12, Scalar::F32));
        assert_eq!(
            types[7],
            &matrix(3, 4, 12, Scalar::F32),
            "`float3x4` is 3 rows of 4 columns"
        );
        assert_eq!(types[8], &matrix(4, 4, 16, Scalar::I32));
        // Push constants are std430: a `float3x3`'s columns are 16 bytes apart.
        assert_eq!(pc.fields[1].ty, matrix(3, 3, 16, Scalar::F32));
        assert_eq!(
            (
                pc.fields[1].offset,
                pc.fields[2].offset,
                pc.fields[3].offset
            ),
            (16, 64, 76)
        );
    }

    #[test]
    fn constants_take_handles_and_accel() {
        let source = "#include \"bindless.slang\"\nSCONST(Accel, scene)\nSCONST(Buf<uint>, data)\n\
                      [SpecializationConstant] const uint count = 1;\n\
                      [shader(\"compute\")] [numthreads(1, 1, 1)] void main() { if (count > 0) { let x = data.val[0] + uint(scene.address); } }\n";
        let compiled = compile("constants", source);
        let desc = &compiled.passes.unwrap()[0].desc;
        let names: Vec<(&str, &Ty)> = desc
            .constants
            .iter()
            .map(|c| (c.name.as_str(), &c.ty))
            .collect();
        assert!(
            names
                .iter()
                .any(|(name, ty)| *name == "scene" && **ty == Ty::Accel),
            "{names:?}"
        );
        assert!(
            names
                .iter()
                .any(|(name, ty)| *name == "data" && matches!(ty, Ty::Buffer { .. }))
        );
        assert!(
            names
                .iter()
                .any(|(name, ty)| *name == "count" && **ty == Ty::Scalar(Scalar::U32))
        );
        assert_eq!(desc.constants.iter().map(Constant::size).sum::<u32>(), 20);
    }

    #[test]
    fn uniforms_and_second_push_constants_are_refused() {
        let uniform = compile(
            "uniform_param",
            "[shader(\"compute\")] [numthreads(1, 1, 1)] void main(uniform uint extra) {}\n",
        );
        let errors = uniform.passes.err().unwrap().join("\n");
        assert!(errors.contains("`uniform` parameter `extra`"), "{errors}");
        let two = compile(
            "two_push",
            "struct A { uint a; }\nstruct B { uint b; }\n[[vk::push_constant]] A a;\n[[vk::push_constant]] B b;\n\
             [shader(\"compute\")] [numthreads(1, 1, 1)] void main() {}\n",
        );
        let errors = two.passes.err().unwrap().join("\n");
        assert!(errors.contains("second push constant buffer"), "{errors}");
        let nested =
            describe_fields("nested", "struct H { Tex2D<float4> t; }", "H inner;").unwrap_err();
        assert!(
            nested.contains("push constant `inner.t` is a handle inside a struct"),
            "{nested}"
        );
    }

    #[test]
    fn errors_name_the_field_path() {
        let unsized_array =
            describe_fields("unsized", "struct U { uint data[]; }", "Buf<U> u;").unwrap_err();
        assert!(
            unsized_array.contains("push constant `u.data`: an array without a length"),
            "{unsized_array}"
        );
    }

    #[test]
    fn exports_are_scalar_constants_with_the_attribute() {
        let desc = describe_fields(
            "exports",
            "[RustConst] static const uint A = 4;\n[RustConst] static const int B = -1;\n\
             [RustConst] static const float C = 0.25;\n[RustConst] static const bool D = false;\n\
             static const uint E = 1;",
            "uint x;",
        )
        .unwrap();
        let exports: Vec<(&str, &str, &str)> = (desc.exports.iter())
            .map(|e| (e.name.as_str(), e.ty, e.value.as_str()))
            .collect();
        assert_eq!(
            exports,
            [
                ("A", "u32", "4"),
                ("B", "i32", "-1"),
                ("C", "f32", "0.25"),
                ("D", "bool", "false")
            ]
        );
    }

    #[test]
    fn several_compute_entry_points_are_one_pass_each() {
        let source = "#include \"bindless.slang\"\nstruct P { uint x; }\n[[vk::push_constant]] P p;\n\
                      [shader(\"compute\")] [numthreads(4, 1, 1)] void first() {}\n\
                      [shader(\"compute\")] [numthreads(2, 2, 1)] [WaveSize(32)] void second_one() {}\n";
        let passes = compile("split", source).passes.unwrap();
        let names: Vec<(&str, Option<&str>)> = (passes.iter())
            .map(|p| (p.name.as_str(), p.entry.as_deref()))
            .collect();
        assert_eq!(
            names,
            [
                ("SplitFirst", Some("first")),
                ("SplitSecondOne", Some("second_one"))
            ]
        );
        assert_eq!(passes[0].desc.entries.len(), 1, "linked alone");
        assert_eq!(passes[0].desc.entries[0].workgroup_size, [4, 1, 1]);
        assert_eq!(
            (
                passes[1].desc.entries[0].workgroup_size,
                passes[1].desc.entries[0].wave_size
            ),
            ([2, 2, 1], 32)
        );
        assert_ne!(passes[0].desc.layout_hash(), passes[1].desc.layout_hash());
    }

    #[test]
    fn pass_names_are_rust_identifiers() {
        assert_eq!(compile::pass_name("bvh_cull").unwrap(), "BvhCull");
        assert_eq!(compile::pass_name("blit2d").unwrap(), "Blit2d");
        for bad in ["my-pass", "2d_blit", "pässe", "", "a b"] {
            assert!(compile::pass_name(bad).is_err(), "{bad}");
        }
        let error = compile("2d_blit", "").passes.err().unwrap().join("\n");
        assert!(error.contains("isn't a Rust identifier"), "{error}");
    }

    #[test]
    fn debug_info_is_emitted_on_request() {
        let source = "[shader(\"compute\")] [numthreads(1, 1, 1)] void main() {}\n";
        let has_debug_info = |debug_info| {
            let passes = compile_with(&format!("debug_{debug_info}"), source, debug_info)
                .passes
                .unwrap();
            let spirv = &passes[0].spirv;
            spirv
                .windows(32)
                .any(|w| w == b"NonSemantic.Shader.DebugInfo.100")
        };
        assert!(has_debug_info(true));
        assert!(!has_debug_info(false));
    }

    #[test]
    fn warnings_are_returned() {
        let compiled = compile(
            "warns",
            "[shader(\"compute\")] [numthreads(1, 1, 1)] void main() { int x = 1.5; }\n",
        );
        assert!(compiled.passes.is_ok());
        assert!(
            !compiled.warnings.is_empty(),
            "an implicit float to int conversion warns"
        );
    }
}

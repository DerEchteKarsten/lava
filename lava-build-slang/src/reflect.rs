//! Null-checked Slang reflection: shader-slang 0.1.0 turns some null results of Slang's C API into references, so these `checked_*` methods return `Option`s, calling the C API directly where 0.1.0's wrapper would dereference null; and the module loads, links and code generation whose diagnostics (warnings included) 0.1.0 drops
use std::ffi::{CStr, CString, c_void};

use shader_slang::{
    self as slang, ParameterCategory,
    reflection::{
        Decl, EntryPoint, Function, Generic, Shader, Type, TypeLayout, UserAttribute, Variable,
        VariableLayout,
    },
};
use shader_slang_sys as sys;

/// The C object behind a 0.1.0 wrapper, which is `repr(transparent)` over it.
fn raw<T, P>(wrapper: &T) -> *mut P {
    (wrapper as *const T).cast_mut().cast()
}

/// # Safety
/// `ptr` is null or points to the C object `T` wraps, which lives for `'a`.
unsafe fn wrap<'a, T, P>(ptr: *mut P) -> Option<&'a T> {
    unsafe { ptr.cast::<T>().as_ref() }
}

pub trait CheckedName {
    fn checked_name(&self) -> Option<&str>;
}

macro_rules! checked_name {
    ($($wrapper:ty => $function:ident),* $(,)?) => {
        $(impl CheckedName for $wrapper {
            fn checked_name(&self) -> Option<&str> {
                // SAFETY: Slang's names live as long as the object they name.
                let name = unsafe { sys::$function(raw(self)) };
                if name.is_null() {
                    return None;
                }
                unsafe { CStr::from_ptr(name) }.to_str().ok()
            }
        })*
    };
}

checked_name! {
    Variable => spReflectionVariable_GetName,
    Type => spReflectionType_GetName,
    EntryPoint => spReflectionEntryPoint_getName,
    UserAttribute => spReflectionUserAttribute_GetName,
}

impl CheckedName for TypeLayout {
    fn checked_name(&self) -> Option<&str> {
        self.ty()?.checked_name()
    }
}

pub trait CheckedTypeLayout {
    /// `None` for a pointer whose pointee Slang doesn't lay out.
    fn checked_element_type_layout(&self) -> Option<&TypeLayout>;
    fn checked_fields(&self) -> impl Iterator<Item = &VariableLayout>;
}

impl CheckedTypeLayout for TypeLayout {
    fn checked_element_type_layout(&self) -> Option<&TypeLayout> {
        // SAFETY: the element layout lives as long as its parent.
        unsafe { wrap(sys::spReflectionTypeLayout_GetElementTypeLayout(raw(self))) }
    }

    fn checked_fields(&self) -> impl Iterator<Item = &VariableLayout> {
        (0..self.field_count()).filter_map(|i| self.field_by_index(i))
    }
}

pub trait CheckedVariableLayout {
    fn checked_type_layout(&self) -> Option<&TypeLayout>;
    /// 0.1.0's `category()` is `type_layout().parameter_category()`, through the unchecked
    /// `type_layout`.
    fn checked_category(&self) -> Option<ParameterCategory>;
}

impl CheckedVariableLayout for VariableLayout {
    fn checked_type_layout(&self) -> Option<&TypeLayout> {
        // SAFETY: the type layout lives as long as the variable's.
        unsafe { wrap(sys::spReflectionVariableLayout_GetTypeLayout(raw(self))) }
    }

    fn checked_category(&self) -> Option<ParameterCategory> {
        Some(self.checked_type_layout()?.parameter_category())
    }
}

pub trait CheckedVariable {
    fn checked_ty(&self) -> Option<&Type>;
    /// The value of a `static const` float (`default_value_int` reads integers and bools).
    fn default_value_float(&self) -> Option<f32>;
}

impl CheckedVariable for Variable {
    fn checked_ty(&self) -> Option<&Type> {
        // SAFETY: the type lives as long as the variable.
        unsafe { wrap(sys::spReflectionVariable_GetType(raw(self))) }
    }

    fn default_value_float(&self) -> Option<f32> {
        let mut value = 0.0;
        // SAFETY: a variable and a place for the value.
        let result =
            unsafe { sys::spReflectionVariable_GetDefaultValueFloat(raw(self), &mut value) };
        (result >= 0).then_some(value)
    }
}

pub trait CheckedType {
    /// The name with its generic arguments (`Pair<float>`), where `checked_name` is `Pair`.
    fn checked_full_name(&self) -> Option<String>;
}

impl CheckedType for Type {
    fn checked_full_name(&self) -> Option<String> {
        let mut blob = std::ptr::null_mut();
        // SAFETY: a type and a place for the blob, which we own on success.
        let result = unsafe { sys::spReflectionType_GetFullName(raw(self), &mut blob) };
        if result < 0 {
            return None;
        }
        // SAFETY: a blob the call returned a reference to.
        unsafe { blob_text(blob.cast()) }
    }
}

pub trait CheckedDecl {
    fn checked_children(&self) -> impl Iterator<Item = &Decl>;
    /// The variable it declares, if it is one.
    fn checked_variable(&self) -> Option<&Variable>;
}

impl CheckedDecl for Decl {
    fn checked_children(&self) -> impl Iterator<Item = &Decl> {
        (0..self.child_count()).filter_map(|i| self.child_by_index(i))
    }

    fn checked_variable(&self) -> Option<&Variable> {
        if self.kind() != slang::DeclKind::Variable {
            return None;
        }
        // SAFETY: a variable declaration's variable lives as long as the declaration.
        unsafe { wrap(sys::spReflectionDecl_castToVariable(raw(self))) }
    }
}

pub trait CheckedEntryPoint {
    fn checked_function(&self) -> Option<&Function>;
}

impl CheckedEntryPoint for EntryPoint {
    fn checked_function(&self) -> Option<&Function> {
        // SAFETY: the function lives as long as its entry point.
        unsafe { wrap(sys::spReflectionEntryPoint_getFunction(raw(self))) }
    }
}

pub trait CheckedShader {
    fn checked_parameters(&self) -> impl Iterator<Item = &VariableLayout>;
    fn checked_entry_points(&self) -> impl Iterator<Item = &EntryPoint>;
}

impl CheckedShader for Shader {
    fn checked_parameters(&self) -> impl Iterator<Item = &VariableLayout> {
        (0..self.parameter_count()).filter_map(|i| self.parameter_by_index(i))
    }

    fn checked_entry_points(&self) -> impl Iterator<Item = &EntryPoint> {
        (0..self.entry_point_count()).filter_map(|i| self.entry_point_by_index(i))
    }
}

pub trait CheckedAttributes {
    fn checked_user_attributes(&self) -> impl Iterator<Item = &UserAttribute>;
}

impl CheckedAttributes for Variable {
    fn checked_user_attributes(&self) -> impl Iterator<Item = &UserAttribute> {
        (0..self.user_attribute_count()).filter_map(|i| self.user_attribute_by_index(i))
    }
}

impl CheckedAttributes for Function {
    fn checked_user_attributes(&self) -> impl Iterator<Item = &UserAttribute> {
        (0..self.user_attribute_count()).filter_map(|i| self.user_attribute_by_index(i))
    }
}

pub trait CheckedGeneric {
    /// The type parameters, as the variables the C API returns.
    fn checked_type_parameters(&self) -> impl Iterator<Item = &Variable>;
    fn checked_value_parameters(&self) -> impl Iterator<Item = &Variable>;
    fn checked_concrete_type(&self, parameter: &Variable) -> Option<&Type>;
}

impl CheckedGeneric for Generic {
    fn checked_type_parameters(&self) -> impl Iterator<Item = &Variable> {
        (0..self.type_parameter_count()).filter_map(|i| {
            // SAFETY: a type parameter lives as long as its generic.
            unsafe { wrap(sys::spReflectionGeneric_GetTypeParameter(raw(self), i)) }
        })
    }

    fn checked_value_parameters(&self) -> impl Iterator<Item = &Variable> {
        (0..self.value_parameter_count()).filter_map(|i| self.value_parameter_by_index(i))
    }

    fn checked_concrete_type(&self, parameter: &Variable) -> Option<&Type> {
        // SAFETY: the concrete type lives as long as the generic.
        unsafe {
            wrap(sys::spReflectionGeneric_GetConcreteType(
                raw(self),
                raw(parameter),
            ))
        }
    }
}

/// The COM object behind a 0.1.0 interface wrapper (`repr(transparent)` over a `NonNull`).
fn com<T>(wrapper: &T) -> *mut c_void {
    // SAFETY: the wrapper is one non-null pointer.
    unsafe { *(wrapper as *const T).cast::<*mut c_void>() }
}

/// # Safety
/// `object` is a live COM object whose vtable starts with `V`.
unsafe fn vtable<'a, V>(object: *mut c_void) -> &'a V {
    unsafe { &**object.cast::<*const V>() }
}

/// # Safety
/// `object` is a live COM object.
unsafe fn unknown<'a>(object: *mut c_void) -> &'a sys::ISlangUnknown__bindgen_vtable {
    unsafe { vtable(object) }
}

/// The text of a diagnostics blob we own a reference to (null for none), released.
///
/// # Safety
/// `blob` is null or a blob the caller owns a reference to.
unsafe fn blob_text(blob: *mut c_void) -> Option<String> {
    let bytes = unsafe { blob_bytes(blob)? };
    let text = String::from_utf8_lossy(&bytes)
        .trim_end_matches('\0')
        .trim()
        .to_string();
    (!text.is_empty()).then_some(text)
}

/// The bytes of a blob we own a reference to (null for none), released.
///
/// # Safety
/// As for [`blob_text`].
unsafe fn blob_bytes(blob: *mut c_void) -> Option<Vec<u8>> {
    if blob.is_null() {
        return None;
    }
    unsafe {
        let functions: &sys::IBlobVtable = vtable(blob);
        let (pointer, size) = (
            (functions.getBufferPointer)(blob),
            (functions.getBufferSize)(blob),
        );
        let bytes = match pointer.is_null() {
            true => Vec::new(),
            false => std::slice::from_raw_parts(pointer.cast::<u8>(), size).to_vec(),
        };
        (unknown(blob).ISlangUnknown_release)(blob.cast());
        Some(bytes)
    }
}

/// A 0.1.0 wrapper owning a reference to `object`.
///
/// # Safety
/// `object` is a non-null COM object of the interface `T` wraps, and the caller owns a
/// reference to it, which the wrapper releases.
unsafe fn owned<T>(object: *mut c_void) -> T {
    const { assert!(size_of::<T>() == size_of::<*mut c_void>()) };
    unsafe { std::mem::transmute_copy(&object) }
}

/// Loads the module at `path` with its diagnostics (warnings on success, errors on failure).
pub fn load_module(
    session: &slang::Session,
    path: &str,
) -> Result<(slang::Module, Option<String>), String> {
    let name = CString::new(path).map_err(|e| e.to_string())?;
    let session = com(session);
    let mut diagnostics = std::ptr::null_mut();
    // SAFETY: a live session, a path and a place for the diagnostics, which we own.
    unsafe {
        let functions: &sys::ISessionVtable = vtable(session);
        let module = (functions.loadModule)(session, name.as_ptr(), &mut diagnostics);
        let diagnostics = blob_text(diagnostics.cast());
        if module.is_null() {
            return Err(diagnostics.unwrap_or_else(|| "the module failed to load".into()));
        }
        // The session owns the module; the wrapper's reference is ours.
        (unknown(module.cast()).ISlangUnknown_addRef)(module.cast());
        Ok((owned(module.cast()), diagnostics))
    }
}

/// Links `component` with its diagnostics.
pub fn link(
    component: &slang::ComponentType,
) -> Result<(slang::ComponentType, Option<String>), String> {
    let component = com(component);
    let (mut linked, mut diagnostics) = (std::ptr::null_mut(), std::ptr::null_mut());
    // SAFETY: a live component and places for the results, which we own.
    unsafe {
        let functions: &sys::IComponentTypeVtable = vtable(component);
        let result = (functions.link)(component, &mut linked, &mut diagnostics);
        let diagnostics = blob_text(diagnostics.cast());
        if result < 0 || linked.is_null() {
            return Err(diagnostics.unwrap_or_else(|| format!("linking failed ({result})")));
        }
        Ok((owned(linked.cast()), diagnostics))
    }
}

/// The code of target `target` of a linked program, with its diagnostics.
pub fn target_code(
    program: &slang::ComponentType,
    target: i64,
) -> Result<(Vec<u8>, Option<String>), String> {
    let program = com(program);
    let (mut code, mut diagnostics) = (std::ptr::null_mut(), std::ptr::null_mut());
    // SAFETY: a live program and places for the results, which we own.
    unsafe {
        let functions: &sys::IComponentTypeVtable = vtable(program);
        let result = (functions.getTargetCode)(program, target as _, &mut code, &mut diagnostics);
        let diagnostics = blob_text(diagnostics.cast());
        let code = blob_bytes(code.cast());
        match code {
            Some(code) if result >= 0 => Ok((code, diagnostics)),
            _ => Err(diagnostics.unwrap_or_else(|| format!("code generation failed ({result})"))),
        }
    }
}

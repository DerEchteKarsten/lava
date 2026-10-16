//! Null-checked Slang reflection: shader-slang 0.1.0 turns some null results of Slang's C API into references, so these `checked_*` methods return `Option`s, calling the C API directly where 0.1.0's wrapper would dereference null
//!
//! The iterators go through 0.1.0's `*_by_index` methods, which already check. Generic type
//! parameters come back as [`Variable`]s, as the C API returns them (0.1.0 types them as
//! `TypeParameter`, which its `concrete_type` doesn't take). Use these instead of 0.1.0's
//! `name`, `type_layout`, `element_type_layout`, `function`, `fields`, `parameters`,
//! `entry_points`, `user_attributes`, `type_parameters`, `value_parameters` and `concrete_type`.
//! Delete this module once slang-rs releases its null-checked API (its git main has it).

use std::ffi::{CStr, c_char};

use shader_slang::reflection::{
    EntryPoint, Function, Generic, Shader, Type, TypeLayout, UserAttribute, Variable,
    VariableLayout,
};
use shader_slang_sys as sys;

/// The C object behind a 0.1.0 wrapper reference (every wrapper is `repr(transparent)` over the
/// opaque C struct, so a reference to it is the C pointer).
fn raw<T, P>(wrapper: &T) -> *mut P {
    (wrapper as *const T).cast_mut().cast()
}

/// `ptr` as a reference to its 0.1.0 wrapper `T`, or `None` for null.
///
/// # Safety
/// `ptr` is null or points to the C object `T` wraps, which lives for `'a`.
unsafe fn wrap<'a, T, P>(ptr: *mut P) -> Option<&'a T> {
    unsafe { ptr.cast::<T>().as_ref() }
}

/// A C string from Slang, or `None` for null or invalid UTF-8.
///
/// # Safety
/// `ptr` is null or a C string that lives for `'a`.
unsafe fn string<'a>(ptr: *const c_char) -> Option<&'a str> {
    if ptr.is_null() {
        None
    } else {
        unsafe { CStr::from_ptr(ptr) }.to_str().ok()
    }
}

/// The name of a reflected object, if it has one.
pub trait CheckedName {
    fn checked_name(&self) -> Option<&str>;
}

macro_rules! checked_name {
    ($($wrapper:ty => $function:ident),* $(,)?) => {
        $(impl CheckedName for $wrapper {
            fn checked_name(&self) -> Option<&str> {
                // SAFETY: Slang's names live as long as the object they name.
                unsafe { string(sys::$function(raw(self))) }
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
    /// The element type of an array, the pointee of a pointer, the contents of a constant
    /// buffer; `None` for a pointer whose pointee Slang doesn't lay out.
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
}

impl CheckedVariableLayout for VariableLayout {
    fn checked_type_layout(&self) -> Option<&TypeLayout> {
        // SAFETY: the type layout lives as long as the variable's.
        unsafe { wrap(sys::spReflectionVariableLayout_GetTypeLayout(raw(self))) }
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
    /// The type a specialization gives the type parameter `parameter`.
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

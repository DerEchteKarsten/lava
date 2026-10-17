//! `unsized_struct!`: a sized struct whose const generics are the lengths of its array sections, and the unsized struct it derefs to, which keeps the fixed fields and reads the sections back as slices (pass bindings, draws)
use std::mem::MaybeUninit;

/// 8-aligned and a multiple of 8 long, so the sections after it start at the same offset in
/// both structs.
#[repr(C, align(8))]
pub(crate) struct Lens<const S: usize>(pub(crate) [u32; S]);

/// # Safety
/// `tail` is an unsized struct's tail and `T` the type of its section `index`.
#[inline]
pub(crate) unsafe fn section<'s, T>(
    tail: &'s [MaybeUninit<u8>],
    lens: &[u32],
    layouts: &[(usize, usize)],
    index: usize,
) -> &'s [T] {
    let align = |offset: usize, align: usize| (offset + align - 1) & !(align - 1);
    let mut offset = 0usize;
    for (&len, &(size, alignment)) in lens.iter().zip(layouts).take(index) {
        offset = align(offset, alignment) + len as usize * size;
    }
    let offset = align(offset, layouts[index].1);
    let len = lens[index] as usize;
    debug_assert!(offset + len * size_of::<T>() <= tail.len());
    // SAFETY: the sized struct wrote `len` `T`s there, aligned since the tail is 8-aligned.
    unsafe { std::slice::from_raw_parts(tail.as_ptr().add(offset).cast::<T>(), len) }
}

/// A sized struct with array sections (made by `from_parts`) that derefs to an unsized one
/// reading them back as slices; section types are aligned to at most 8.
macro_rules! unsized_struct {
    (
        $(#[$sized_attr:meta])*
        $vis:vis struct $sized:ident<$lt:lifetime $(, $param:ident)* $(,)?>[$($len:ident),* $(,)?];
        $(#[$unsized_attr:meta])*
        $uvis:vis struct $unsized:ident {
            $($(#[$field_attr:meta])* $field_vis:vis $field:ident: $field_ty:ty,)*
        } sections {
            $($(#[$section_attr:meta])* $section_vis:vis $section:ident: [$section_ty:ty; $section_len:ident],)*
        }
    ) => {
        $(#[$sized_attr])*
        #[repr(C)]
        $vis struct $sized<$lt, $($param,)* $(const $len: usize),*> {
            $($(#[$field_attr])* $field_vis $field: $field_ty,)*
            marker: ::std::marker::PhantomData<(&$lt (), $($param,)* $([$section_ty; 0],)*)>,
            lens: $crate::unsized_struct::Lens<{ [$(stringify!($section)),*].len() }>,
            $($section: [$section_ty; $section_len],)*
        }

        $(#[$unsized_attr])*
        #[repr(C)]
        $uvis struct $unsized<$lt, $($param),*> {
            $($(#[$field_attr])* $field_vis $field: $field_ty,)*
            marker: ::std::marker::PhantomData<(&$lt (), $($param,)* $([$section_ty; 0],)*)>,
            lens: $crate::unsized_struct::Lens<{ [$(stringify!($section)),*].len() }>,
            tail: [::std::mem::MaybeUninit<u8>],
        }

        impl<$lt, $($param,)* $(const $len: usize),*> $sized<$lt, $($param,)* $($len),*> {
            #[allow(clippy::too_many_arguments)]
            fn from_parts(
                $($field: $field_ty,)*
                $($section: [$section_ty; $section_len],)*
            ) -> Self {
                const { $(assert!(align_of::<$section_ty>() <= 8);)* }
                Self {
                    $($field,)*
                    marker: ::std::marker::PhantomData,
                    lens: $crate::unsized_struct::Lens([$($section_len as u32),*]),
                    $($section,)*
                }
            }
        }

        impl<$lt, $($param,)* $(const $len: usize),*> ::std::ops::Deref
            for $sized<$lt, $($param,)* $($len),*>
        {
            type Target = $unsized<$lt, $($param),*>;

            fn deref(&self) -> &Self::Target {
                let tail = ::std::mem::offset_of!(Self, lens) + size_of_val(&self.lens);
                let bytes = ::std::ptr::slice_from_raw_parts(
                    (self as *const Self).cast::<::std::mem::MaybeUninit<u8>>(),
                    size_of::<Self>() - tail,
                );
                // SAFETY: both are `repr(C)` with the same fields up to `lens`, which ends at
                // a multiple of 8.
                unsafe { &*(bytes as *const Self::Target) }
            }
        }

        impl<$lt, $($param),*> $unsized<$lt, $($param),*> {
            const LAYOUTS: &'static [(usize, usize)] =
                &[$((size_of::<$section_ty>(), align_of::<$section_ty>())),*];

            $crate::unsized_struct::unsized_struct!(@accessors [0]
                $($(#[$section_attr])* ($section_vis) $section: $section_ty,)*);
        }
    };
    (@accessors [$($index:tt)*]) => {};
    (@accessors [$($index:tt)*]
        $(#[$attr:meta])* ($vis:vis) $section:ident: $ty:ty, $($rest:tt)*
    ) => {
        $(#[$attr])*
        #[inline]
        $vis fn $section(&self) -> &[$ty] {
            // SAFETY: section `$index` is the `$ty`s.
            unsafe {
                $crate::unsized_struct::section(&self.tail, &self.lens.0, Self::LAYOUTS, $($index)*)
            }
        }
        $crate::unsized_struct::unsized_struct!(@accessors [$($index)* + 1] $($rest)*);
    };
}
pub(crate) use unsized_struct;

//! `unsized_struct!`: a sized struct whose const generics are the lengths of its array sections, and the unsized struct it derefs to, which keeps the fixed fields and reads the sections back as slices (pass bindings, draws)
use std::mem::MaybeUninit;

/// The section lengths of an [`unsized_struct!`]. It is 8-aligned and a multiple of 8 bytes
/// long, so the sections after it (aligned to at most 8) start at the same offset in the sized
/// and the unsized struct.
#[repr(C, align(8))]
pub(crate) struct Lens<const S: usize>(pub(crate) [u32; S]);

/// Section `index` of `tail`, whose sections have the `lens` and `(size, align)` `layouts`:
/// each starts where the one before ends, rounded up to its alignment, as `repr(C)` lays out
/// the sized struct's arrays.
///
/// # Safety
/// `tail` is an unsized struct's tail and `T` the type of its section `index`.
#[inline]
pub(crate) unsafe fn section<'s, T>(
    tail: &'s [MaybeUninit<u8>],
    lens: &[u32],
    layouts: &[(usize, usize)],
    index: usize,
) -> &'s [T] {
    // Alignments are powers of two: a mask, not a division.
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

/// Declares a sized struct and the unsized struct it derefs to, like a `Vec` and its slice,
/// but with fixed fields in front and several arrays:
///
/// ```ignore
/// unsized_struct! {
///     /// The sized one: `I` images, `B` buffers.
///     pub struct Bindings<'a, K>[I, B];
///     /// The unsized one commands borrow.
///     pub struct BindingOutput {
///         pass: u32,
///     } sections {
///         images: [ImageUse<'a>; I],
///         buffers: [BufferAccess; B],
///     }
/// }
/// ```
///
/// Both have the fixed fields (`pass`), readable through either, with the visibility given.
/// The sized struct (`Bindings<'a, K, I, B>`) holds the sections as arrays, whose lengths are
/// its const parameters, and is made with its private `from_parts(fixed.., sections..)`. It
/// derefs to the unsized one (`BindingOutput<'a, K>`), which has the same layout up to the
/// sections, then reads them as slices of the lengths stored in front of them, through a
/// method per section with its visibility (`fn images(&self) -> &[ImageUse<'a>]`). The
/// structs have one lifetime and any number of type parameters (markers), whose variance and
/// auto traits they take, also those of the section types. Section types are aligned to at
/// most 8.
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
            /// The sections, as the sized struct laid them out.
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
                // a multiple of 8, where the sections start (aligned to at most 8) and the
                // tail with the bytes from there on.
                unsafe { &*(bytes as *const Self::Target) }
            }
        }

        impl<$lt, $($param),*> $unsized<$lt, $($param),*> {
            /// Each section's `(size, align)`.
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

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;

    unsized_struct! {
        /// Sections of every alignment, an empty one among them.
        struct Sized<'a, K>[A, B, C, D];
        struct Unsized {
            first: u8,
            name: &'a str,
        } sections {
            bytes: [u8; A],
            words: [u64; B],
            nothing: [u32; C],
            halves: [u16; D],
        }
    }

    #[test]
    fn the_unsized_struct_reads_back_every_field_and_section() {
        let sized: Sized<'_, PhantomData<()>, 3, 2, 0, 1> =
            Sized::from_parts(7, "lava", [1, 2, 3], [4, u64::MAX], [], [6]);
        let unsized_: &Unsized<'_, PhantomData<()>> = &sized;
        assert_eq!((unsized_.first, unsized_.name), (7, "lava"));
        assert_eq!(unsized_.bytes(), [1, 2, 3]);
        assert_eq!(unsized_.words(), [4, u64::MAX]);
        assert!(unsized_.nothing().is_empty());
        assert_eq!(unsized_.halves(), [6]);
        assert_eq!(size_of_val(unsized_), size_of_val(&sized));
        assert!(std::ptr::eq(unsized_.words(), &sized.words));
    }
}

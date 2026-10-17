//! Pass bindings as generated code builds them (`Bindings`, sized by the pass) and as commands borrow them (`BindingOutput`, unsized, an `unsized_struct!`): the push constants, accesses and specialization constants of one pass

use bytemuck::{Pod, bytes_of};

use super::{BufferAccess, ImageUse};
use crate::pipeline::MAX_PUSH_CONSTANTS_SIZE;
use crate::unsized_struct::unsized_struct;

/// The kind of pass a [`BindingOutput`] was made for, which decides how it can be recorded.
pub mod kind {
    pub struct Compute;
    pub struct RayTracing;
    pub struct RasterVertex;
    pub struct RasterMesh;
}

unsized_struct! {
    /// A pass's bindings as generated code builds them: `P` bytes of push constants, `I` image
    /// and `B` buffer accesses, `C` bytes of specialization constants.
    pub struct Bindings<'a, K>[P, I, B, C];
    /// A pass's bindings as commands borrow them, unsized; can be recorded more than once.
    pub struct BindingOutput {
        pass: u32,
    } sections {
        push_constants: [u8; P],
        images: [ImageUse<'a>; I],
        buffers: [BufferAccess; B],
        constants: [u8; C],
    }
}

#[derive(Clone, Copy)]
pub struct Sections<'s, 'a> {
    pub pass: usize,
    pub push_constants: &'s [u8],
    images: &'s [ImageUse<'a>],
    buffers: &'s [BufferAccess],
    pub constants: &'s [u8],
}

impl<'s, 'a> Sections<'s, 'a> {
    pub fn images(self) -> impl ExactSizeIterator<Item = ImageUse<'a>> + Clone {
        self.images.iter().cloned()
    }

    pub fn buffers(self) -> impl ExactSizeIterator<Item = BufferAccess> + Clone {
        self.buffers.iter().cloned()
    }
}

impl<'a, K> BindingOutput<'a, K> {
    #[inline]
    pub fn sections(&self) -> Sections<'_, 'a> {
        Sections {
            pass: self.pass as usize,
            push_constants: self.push_constants(),
            images: self.images(),
            buffers: self.buffers(),
            constants: self.constants(),
        }
    }

    pub(super) fn erase(&self) -> &BindingOutput<'a, ()> {
        // SAFETY: `K` is a zero-sized marker; the layout and the tail length stay the same.
        unsafe { &*(self as *const Self as *const BindingOutput<'a, ()>) }
    }
}

impl<'a, K, const P: usize, const I: usize, const B: usize> Bindings<'a, K, P, I, B, 0> {
    /// The bindings of pass `pass` (its index in the pass map). `K` must be the pass's kind.
    pub fn new<G: Pod>(
        pass: usize,
        bindings: &G,
        images: [ImageUse<'a>; I],
        buffers: [BufferAccess; B],
    ) -> Self {
        assert!(P <= MAX_PUSH_CONSTANTS_SIZE);
        let push_constants = bytes_of(bindings)
            .try_into()
            .expect("`P` is the size of the push constants");
        Self::from_parts(pass as u32, push_constants, images, buffers, [])
    }

    /// Sets the specialization constants, laid out as `PassEntry.constants` says, and adds the
    /// accesses they make; `IO` and `BO` are `I + I2` and `B + B2`.
    pub fn with_constants<
        G: Pod,
        const C: usize,
        const I2: usize,
        const B2: usize,
        const IO: usize,
        const BO: usize,
    >(
        self,
        constants: &G,
        images: [ImageUse<'a>; I2],
        buffers: [BufferAccess; B2],
    ) -> Bindings<'a, K, P, IO, BO, C> {
        const { assert!(IO == I + I2 && BO == B + B2) };
        let Self {
            pass,
            push_constants,
            images: own_images,
            buffers: own_buffers,
            ..
        } = self;
        Bindings::from_parts(
            pass,
            push_constants,
            concat(own_images, images),
            concat(own_buffers, buffers),
            bytes_of(constants)
                .try_into()
                .expect("`C` is the size of the specialization constants"),
        )
    }
}

fn concat<T, const N: usize, const M: usize, const O: usize>(a: [T; N], b: [T; M]) -> [T; O] {
    const { assert!(O == N + M) };
    let mut items = a.into_iter().chain(b);
    std::array::from_fn(|_| items.next().unwrap())
}

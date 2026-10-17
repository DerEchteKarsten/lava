//! Pass bindings as generated code builds them (`Bindings`, sized by the pass) and as commands borrow them (`BindingOutput`, unsized, an `unsized_struct!`): the push constants, accesses and specialization constants of one pass

use bytemuck::{Pod, bytes_of};

use super::{BufferAccess, ImageUse};
use crate::pipeline::MAX_PUSH_CONSTANTS_SIZE;
use crate::unsized_struct::unsized_struct;

/// Marker types for the kind of pass a [`BindingOutput`] was made for, so that it can only be
/// recorded the way its shaders run.
pub mod kind {
    pub struct Compute;
    pub struct RayTracing;
    pub struct RasterVertex;
    pub struct RasterMesh;
}

unsized_struct! {
    /// A pass's bindings as the generated functions build them: the [`BindingOutput`] it derefs
    /// to, sized by what the pass has: `P` bytes of push constants, `I` image and `B` buffer
    /// accesses, `C` bytes of specialization constants. Binding generators make it with
    /// [`Self::new`] and [`Self::with_constants`]. Resources a pass reaches without naming them
    /// (through a handle or pointer in a buffer) are passed to the command as its `Accesses`.
    pub struct Bindings<'a, K>[P, I, B, C];
    /// A pass's push constants, specialization constants and the resources it accesses, as
    /// commands take them: borrowed and unsized, so a binding is as big as its pass needs and
    /// commands don't depend on that size. The generated `Pass::push_bindings` (and
    /// `constant_bindings` for a pass with specialization constants) returns it as a sized
    /// [`Bindings`], which derefs to this, so `cmd.compute(&Pass::push_bindings(..), ..)` records
    /// it. It borrows its images' tracked states until it is recorded, and can be recorded more
    /// than once.
    pub struct BindingOutput {
        /// The pass's index in the pass map.
        pass: u32,
    } sections {
        push_constants: [u8; P],
        images: [ImageUse<'a>; I],
        buffers: [BufferAccess; B],
        constants: [u8; C],
    }
}

/// The sections of a [`BindingOutput`].
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

    /// The same bindings without their kind, so draws of both raster kinds share a list.
    pub(super) fn erase(&self) -> &BindingOutput<'a, ()> {
        // SAFETY: `K` is a zero-sized marker; the layout and the tail length stay the same.
        unsafe { &*(self as *const Self as *const BindingOutput<'a, ()>) }
    }
}

impl<'a, K, const P: usize, const I: usize, const B: usize> Bindings<'a, K, P, I, B, 0> {
    /// The bindings of pass `pass` (its index in the pass map) with the push constants
    /// `bindings` (`P` bytes), accessing `images` and `buffers`. `K` must be the pass's kind.
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

    /// Sets the specialization constants (`C` bytes), laid out as the pass's
    /// `PassEntry.constants` says, and adds the accesses of the resources they name: `IO` and
    /// `BO` are `I + I2` and `B + B2`, which the generated code's return type spells out.
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

/// `a` followed by `b`; `O` is `N + M`.
fn concat<T, const N: usize, const M: usize, const O: usize>(a: [T; N], b: [T; M]) -> [T; O] {
    const { assert!(O == N + M) };
    let mut items = a.into_iter().chain(b);
    std::array::from_fn(|_| items.next().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command_buffer::ImageAccess;
    use std::sync::atomic::AtomicU64;
    use crate::vk::{self, Handle};
    use vk::{AccessFlags2 as A, ImageLayout as L, PipelineStageFlags2 as S};

    /// What a binding generator puts in comes back out of the unsized bytes unchanged.
    #[test]
    fn bindings_read_back_every_section() {
        let layouts = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
        let image = ImageUse {
            access: ImageAccess {
                stage: S::COMPUTE_SHADER,
                access: A::SHADER_STORAGE_WRITE,
                image: vk::Image::from_raw(7),
                layout: L::GENERAL,
                aspect: vk::ImageAspectFlags::COLOR,
                mip_range: (1..3).into(),
            },
            states: &layouts[1..],
        };
        let buffer = BufferAccess {
            stage: S::COMPUTE_SHADER,
            access: A::SHADER_STORAGE_READ,
            range: (0..64).into(),
        };
        let push = [1u32, 2, 3];
        let bindings: Bindings<'_, kind::Compute, 12, 2, 1, 8> =
            Bindings::new(5, &push, [image.clone()], [buffer.clone()]).with_constants(
                &[9u32, 10],
                [ImageUse {
                    access: ImageAccess {
                        image: vk::Image::from_raw(3),
                        ..image.access.clone()
                    },
                    states: &layouts[..1],
                }],
                [],
            );
        let output: &BindingOutput<'_, kind::Compute> = &bindings;

        // The pass and the lengths, then the sections, each aligned to its type.
        assert_eq!(size_of_val(output), 24 + 16 + 2 * 56 + 32 + 8);
        assert_eq!(output.sections().pass, 5);
        assert_eq!(
            output.sections().push_constants,
            bytemuck::cast_slice::<u32, u8>(&push)
        );
        assert_eq!(
            output.sections().constants,
            bytemuck::cast_slice::<u32, u8>(&[9, 10])
        );
        let images: Vec<_> = output.sections().images().collect();
        assert_eq!(images.len(), 2);
        assert_eq!(images[0].access, image.access);
        assert!(std::ptr::eq(images[0].states, &layouts[1..]));
        assert_eq!(images[1].access.image, vk::Image::from_raw(3));
        let buffers: Vec<_> = output.sections().buffers().collect();
        assert_eq!(buffers, [buffer.clone()]);
    }
}

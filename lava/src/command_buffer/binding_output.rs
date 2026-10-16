//! Pass bindings as generated code builds them (`Bindings`, sized by the pass) and as commands borrow them (`BindingOutput`, unsized bytes): the push constants, accesses and specialization constants of one pass
use std::{
    marker::PhantomData,
    mem::{MaybeUninit, transmute},
    ops::Deref,
    ptr,
    sync::atomic::AtomicU32,
};

use bytemuck::{Pod, bytes_of};

use super::{BufferAccess, ImageUse};
use crate::pipeline::MAX_PUSH_CONSTANTS_SIZE;

/// Marker types for the kind of pass a [`BindingOutput`] was made for, so that it can only be
/// recorded the way its shaders run.
pub mod kind {
    pub struct Compute;
    pub struct RayTracing;
    pub struct RasterVertex;
    pub struct RasterMesh;
}

/// The header of a [`BindingOutput`]: the pass index, then the push-constant bytes, image
/// accesses, buffer accesses and specialization-constant bytes that follow it.
type Header = [u32; 5];
const HEADER_SIZE: usize = size_of::<Header>();
const IMAGE_USE_SIZE: usize = size_of::<ImageUse<'static>>();
const BUFFER_ACCESS_SIZE: usize = size_of::<BufferAccess>();

fn header(
    pass: usize,
    push_constants: usize,
    images: usize,
    buffers: usize,
    constants: usize,
) -> Header {
    [pass, push_constants, images, buffers, constants].map(|n| n as u32)
}

/// A pass's push constants, specialization constants and the resources it accesses, as
/// commands take them: borrowed and unsized, so a binding is as big as its pass needs and
/// commands don't depend on that size. The generated `Pass::push_bindings` (and
/// `constant_bindings` for a pass with specialization constants) returns it as a sized
/// [`Bindings`], which derefs to this, so `cmd.compute(&Pass::push_bindings(..), ..)` records
/// it. It borrows its images' tracked layouts until it is recorded, and can be recorded more
/// than once.
///
/// It is one run of bytes: a header (pass index and the size of each section), the push
/// constants at their size, the image accesses ([`ImageUse`]s), the buffer accesses
/// ([`BufferAccess`]es) and the specialization constants.
#[repr(C)]
pub struct BindingOutput<'a, K> {
    marker: PhantomData<(&'a [AtomicU32], K)>,
    /// Bytes, but also the pointers of the accesses' borrows, so `MaybeUninit` keeps their
    /// provenance.
    bytes: [MaybeUninit<u8>],
}

/// The sections of a [`BindingOutput`], read from its header once.
#[derive(Clone, Copy)]
pub struct Sections<'s, 'a> {
    pub pass: usize,
    pub push_constants: &'s [u8],
    images: &'s [[MaybeUninit<u8>; IMAGE_USE_SIZE]],
    buffers: &'s [[MaybeUninit<u8>; BUFFER_ACCESS_SIZE]],
    pub constants: &'s [u8],
    marker: PhantomData<&'a [AtomicU32]>,
}

impl<'s, 'a> Sections<'s, 'a> {
    pub fn images(self) -> impl ExactSizeIterator<Item = ImageUse<'a>> + Clone {
        // SAFETY: `Bindings` wrote an `ImageUse<'a>` there.
        self.images
            .iter()
            .map(|&bytes| unsafe { transmute::<_, ImageUse<'a>>(bytes) })
    }

    pub fn buffers(self) -> impl ExactSizeIterator<Item = BufferAccess> + Clone {
        // SAFETY: `Bindings` wrote a `BufferAccess` there.
        self.buffers
            .iter()
            .map(|&bytes| unsafe { transmute::<_, BufferAccess>(bytes) })
    }
}

impl<'a, K> BindingOutput<'a, K> {
    pub fn sections(&self) -> Sections<'_, 'a> {
        let (&header, rest) = self.bytes.split_first_chunk::<HEADER_SIZE>().unwrap();
        // SAFETY: `Bindings` wrote a `Header` there.
        let [pass, push_len, images_len, buffers_len, constants_len] =
            unsafe { transmute::<_, Header>(header) }.map(|n| n as usize);
        let (push_constants, rest) = rest.split_at(push_len);
        let (images, rest) = rest.split_at(images_len * IMAGE_USE_SIZE);
        let (buffers, rest) = rest.split_at(buffers_len * BUFFER_ACCESS_SIZE);
        // SAFETY: `Bindings` wrote the push and specialization constants from `Pod` values.
        let init = |bytes: &[MaybeUninit<u8>]| unsafe { transmute::<_, &[u8]>(bytes) };
        Sections {
            pass,
            push_constants: init(push_constants),
            images: images.as_chunks().0,
            buffers: buffers.as_chunks().0,
            constants: init(&rest[..constants_len]),
            marker: PhantomData,
        }
    }

    /// The same bindings without their kind, so draws of both raster kinds share a list.
    pub(super) fn erase(&self) -> &BindingOutput<'a, ()> {
        // SAFETY: `K` is a zero-sized marker; the layout and the slice length stay the same.
        unsafe { &*(self as *const Self as *const BindingOutput<'a, ()>) }
    }
}

/// A pass's bindings as the generated functions build them: the [`BindingOutput`] it derefs
/// to, sized by what the pass has: `P` bytes of push constants, `I` image and `B` buffer
/// accesses, `C` bytes of specialization constants. Binding generators make it with
/// [`Self::new`] and [`Self::with_constants`]. Resources a pass reaches without naming them
/// (through a handle or pointer in a buffer) are declared with `CommandBuffer::record_access`.
#[repr(C, packed)]
pub struct Bindings<'a, K, const P: usize, const I: usize, const B: usize, const C: usize> {
    header: Header,
    push_constants: [u8; P],
    images: [ImageUse<'a>; I],
    buffers: [BufferAccess; B],
    constants: [u8; C],
    marker: PhantomData<(&'a [AtomicU32], K)>,
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
        Self {
            header: header(pass, P, I, B, 0),
            push_constants: bytes_of(bindings)
                .try_into()
                .expect("`P` is the size of the push constants"),
            images,
            buffers,
            constants: [],
            marker: PhantomData,
        }
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
        let pass = self.header[0] as usize;
        let Self {
            push_constants,
            images: own_images,
            buffers: own_buffers,
            ..
        } = self;
        Bindings {
            header: header(pass, P, IO, BO, C),
            push_constants,
            images: concat(own_images, images),
            buffers: concat(own_buffers, buffers),
            constants: bytes_of(constants)
                .try_into()
                .expect("`C` is the size of the specialization constants"),
            marker: PhantomData,
        }
    }
}

impl<'a, K, const P: usize, const I: usize, const B: usize, const C: usize> Deref
    for Bindings<'a, K, P, I, B, C>
{
    type Target = BindingOutput<'a, K>;

    fn deref(&self) -> &BindingOutput<'a, K> {
        let bytes = ptr::slice_from_raw_parts(
            (self as *const Self).cast::<MaybeUninit<u8>>(),
            size_of::<Self>(),
        );
        // SAFETY: `Bindings` is packed, so it is `size_of::<Self>()` bytes laid out as
        // `BindingOutput` reads them; `BindingOutput` is those bytes behind a zero-sized
        // marker.
        unsafe { &*(bytes as *const BindingOutput<'a, K>) }
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
    use crate::vk::{self, Handle};
    use vk::{AccessFlags2 as A, ImageLayout as L, PipelineStageFlags2 as S};

    /// What a binding generator puts in comes back out of the unsized bytes unchanged.
    #[test]
    fn bindings_read_back_every_section() {
        let layouts = [AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0)];
        let image = ImageUse {
            access: ImageAccess {
                stage: S::COMPUTE_SHADER,
                access: A::SHADER_STORAGE_WRITE,
                image: vk::Image::from_raw(7),
                layout: L::GENERAL,
                aspect: vk::ImageAspectFlags::COLOR,
                mip_range: (1..3).into(),
            },
            layouts: &layouts[1..],
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
                    layouts: &layouts[..1],
                }],
                [],
            );
        let output: &BindingOutput<'_, kind::Compute> = &bindings;

        assert_eq!(size_of_val(output), 20 + 12 + 2 * 56 + 32 + 8);
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
        assert!(std::ptr::eq(images[0].layouts, &layouts[1..]));
        assert_eq!(images[1].access.image, vk::Image::from_raw(3));
        let buffers: Vec<_> = output.sections().buffers().collect();
        assert_eq!(buffers, [buffer.clone()]);
    }
}

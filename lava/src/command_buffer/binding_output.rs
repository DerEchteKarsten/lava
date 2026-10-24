//! Pass bindings as generated code builds them (`Bindings`, sized by the pass, through `unsafe` constructors whose contract generators keep) and as commands borrow them (`BindingOutput`, unsized, an `unsized_struct!`): the push constants, accesses, the extra `Accesses` of what the pass reaches through handles and pointers, and specialization constants of one pass; and the pass kind markers (`kind::Kind`) with the stages their accesses happen in

use bytemuck::{NoUninit, bytes_of};

use super::{Accesses, BufferAccess, ImageUse};
use crate::vk;
use crate::pipeline::MAX_PUSH_CONSTANTS_SIZE;
use crate::unsized_struct::unsized_struct;

/// The kind of pass a [`BindingOutput`] was made for, which decides how it can be recorded,
/// and its shader stages (`STAGES`, also `PassKind::stages`), where the accesses of its
/// bindings, their `Accesses` among them, happen.
pub mod kind {
    use crate::{pipeline::PassKind, sealed::Sealed, vk::PipelineStageFlags2 as S};

    pub struct Compute;
    pub struct RayTracing;
    pub struct RasterVertex;
    pub struct RasterMesh;

    /// One of the kind markers, which [`super::Bindings::new`] checks a pass against.
    pub trait Kind: Sealed {
        /// Its marker's `STAGES`.
        const STAGES: S;
        /// Whether a pass of `kind` has this marker.
        fn is(kind: &PassKind) -> bool;
    }

    macro_rules! kinds {
        ($($marker:ident),*) => {$(
            impl Sealed for $marker {}
            impl Kind for $marker {
                const STAGES: S = $marker::STAGES;
                fn is(kind: &PassKind) -> bool {
                    matches!(kind, PassKind::$marker { .. })
                }
            }
        )*};
    }
    kinds!(Compute, RayTracing, RasterVertex, RasterMesh);

    impl Compute {
        pub const STAGES: S = S::COMPUTE_SHADER;
    }
    impl RayTracing {
        pub const STAGES: S = S::RAY_TRACING_SHADER_KHR;
    }
    impl RasterVertex {
        pub const STAGES: S = S::from_raw(S::VERTEX_SHADER.as_raw() | S::FRAGMENT_SHADER.as_raw());
    }
    impl RasterMesh {
        pub const STAGES: S = S::from_raw(
            S::TASK_SHADER_EXT.as_raw() | S::MESH_SHADER_EXT.as_raw() | S::FRAGMENT_SHADER.as_raw(),
        );
    }
}

unsized_struct! {
    /// A pass's bindings as generated code builds them: `P` bytes of push constants, `I` image
    /// and `B` buffer accesses, `C` bytes of specialization constants.
    pub struct Bindings<'a, K>[P, I, B, C];
    /// A pass's bindings as commands borrow them, unsized; can be recorded more than once.
    pub struct BindingOutput {
        pass: u32,
        /// The pass kind's `STAGES`, where `extra`'s accesses happen.
        stages: vk::PipelineStageFlags2,
        /// What the pass reaches without naming it in its push constants or constants.
        extra: Accesses<'a>,
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
    stages: vk::PipelineStageFlags2,
    extra: Accesses<'a>,
    pub constants: &'s [u8],
}

impl<'s, 'a> Sections<'s, 'a> {
    /// The image accesses of its fields, then its extra `Accesses` at the pass's stages.
    pub fn images(self) -> impl Iterator<Item = ImageUse<'a>> + Clone {
        let stages = self.stages;
        let extra = self.extra.images.iter().cloned().map(move |mut image| {
            image.access.stage = stages;
            image
        });
        self.images.iter().cloned().chain(extra)
    }

    /// The buffer accesses of its fields, then its extra `Accesses` at the pass's stages.
    pub fn buffers(self) -> impl Iterator<Item = BufferAccess> + Clone {
        let stages = self.stages;
        let extra = self.extra.buffers.iter().cloned().map(move |mut buffer| {
            buffer.stage = stages;
            buffer
        });
        self.buffers.iter().cloned().chain(extra)
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
            stages: self.stages,
            extra: self.extra,
            constants: self.constants(),
        }
    }

    pub(super) fn erase(&self) -> &BindingOutput<'a, ()> {
        // SAFETY: `K` is a zero-sized marker; the layout and the tail length stay the same.
        unsafe { &*(self as *const Self as *const BindingOutput<'a, ()>) }
    }
}

/// The pass map entry of `pass` once lava is initialised, for the debug checks of the generator
/// contract.
fn entry(pass: usize) -> Option<&'static crate::pipeline::PassEntry> {
    crate::pipeline::try_pass_map()?.passes.get(pass)
}

impl<'a, K: kind::Kind, const P: usize, const I: usize, const B: usize>
    Bindings<'a, K, P, I, B, 0>
{
    /// The bindings of pass `pass` (its index in the pass map) with `bindings` as its push
    /// constants and the accesses they make, plus `accesses`, what the pass reaches without
    /// naming it (through a handle or pointer in a buffer), tracked at `K::STAGES`; a pass with
    /// specialization constants is recorded only after [`Self::with_constants`]. This is the API
    /// of binding generators.
    ///
    /// # Safety
    /// What generated code guarantees, since commands and the driver trust it:
    /// - `pass` is the index of a pass in the `PASS_MAP` given to `lava::init`, of kind `K`;
    /// - `bindings` is the pass's push constant struct, laid out as its shaders read it;
    /// - `images` and `buffers` name every resource the push constants reach, with the accesses
    ///   the pass makes (the application names the rest in `accesses`);
    /// - a pass with specialization constants gets them with [`Self::with_constants`] before it
    ///   is recorded.
    ///
    /// Debug builds check the kind once lava is initialised.
    pub unsafe fn new<G: NoUninit>(
        pass: usize,
        bindings: &G,
        images: [ImageUse<'a>; I],
        buffers: [BufferAccess; B],
        accesses: Option<Accesses<'a>>,
    ) -> Self {
        const { assert!(P <= MAX_PUSH_CONSTANTS_SIZE) };
        debug_assert!(
            entry(pass).is_none_or(|entry| K::is(&entry.kind)),
            "pass {pass} isn't of the bindings' kind"
        );
        let push_constants = bytes_of(bindings)
            .try_into()
            .expect("`P` is the size of the push constants");
        Self::from_parts(
            pass as u32,
            K::STAGES,
            accesses.unwrap_or_default(),
            push_constants,
            images,
            buffers,
            [],
        )
    }

    /// Sets the specialization constants, laid out as `PassEntry.constants` says, and adds the
    /// accesses they make; `IO` and `BO` are `I + I2` and `B + B2`.
    ///
    /// # Safety
    /// `constants` is the pass's specialization constant data: `PassEntry.constants_size`
    /// bytes, each constant at the offset and with the size of its `PassEntry.constants` entry
    /// (debug builds check the size once lava is initialised); `images` and `buffers` name what
    /// the constants reach, as for [`Self::new`].
    pub unsafe fn with_constants<
        G: NoUninit,
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
        debug_assert!(
            entry(self.pass as usize).is_none_or(|entry| entry.constants_size == C),
            "the constants of pass {} aren't its `constants_size`",
            self.pass
        );
        let Self {
            pass,
            stages,
            extra,
            push_constants,
            images: own_images,
            buffers: own_buffers,
            ..
        } = self;
        Bindings::from_parts(
            pass,
            stages,
            extra,
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

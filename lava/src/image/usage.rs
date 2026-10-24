//! Image usages: one const-generic `Usage` with a `bool` per flag (and `MutableFormat`, for views of other formats), which implements the usage traits only for the sets lava allows (host copies only with `host-image-copy` and never with attachments), `usage!` to name one by its flags, and `Dynamic`, whose usage is only known at runtime
use crate::vk;

#[diagnostic::on_unimplemented(
    message = "`{Self}` is no image usage",
    label = "not a usage an image can have",
    note = "an image needs at least one usage, can't be both a colour and a depth attachment, \
            has `HostCopy` only with lava's `host-image-copy` feature and never with an \
            attachment usage, and `MutableFormat` is no usage of its own"
)]
pub trait ImageUsage: crate::sealed::Sealed + 'static + Copy + Send + Sync + std::fmt::Debug {
    fn flags(&self) -> vk::ImageUsageFlags;
    /// The create flags it implies: `MUTABLE_FORMAT` with `EXTENDED_USAGE` for a usage with
    /// `MutableFormat`, else none (a `Dynamic` usage never has them).
    fn create_flags(&self) -> vk::ImageCreateFlags {
        vk::ImageCreateFlags::empty()
    }
    /// Panics unless the usage has every flag of `required`; a static usage's does nothing.
    fn check(&self, required: vk::ImageUsageFlags);
}

/// Every usage but `Dynamic`.
pub trait StaticUsage: ImageUsage + Default {
    const VK: vk::ImageUsageFlags;
}

pub trait IsSampled: ImageUsage {}
pub trait IsStorage: ImageUsage {}
pub trait IsColorAttachment: ImageUsage {}
pub trait IsDepthAttachment: ImageUsage {}
/// Usages the host can copy texels into (`Image::copy_from`). It can cost the driver its
/// compressed layouts, so only images the host writes should have it.
#[cfg(feature = "host-image-copy")]
pub trait IsHostCopy: ImageUsage {}
/// Usages that can be the source of a copy or blit.
pub trait IsTransferSrc: ImageUsage {}
/// Usages that can be the destination of a copy, blit or clear.
pub trait IsTransferDst: ImageUsage {}
/// Usages whose images have views of other formats of their compatibility class
/// (`ImageView::reinterpret`): `MutableFormat`, and `Dynamic`, checked at runtime.
pub trait IsMutableFormat: ImageUsage {}

/// A static image usage, one `bool` per flag. Name one with [`usage!`](crate::usage), which
/// takes its flags in any order: `Image<R8Unorm, usage![Sampled, HostCopy]>`.
///
/// `MutableFormat` isn't a usage but a create flag (`MUTABLE_FORMAT` with `EXTENDED_USAGE`): its
/// images have views of the other formats of their compatibility class
/// ([`ImageView::reinterpret`](crate::image::slice::ImageView::reinterpret)), and may have
/// usages their own format lacks if such a format has them (storage of an sRGB image through a
/// UNORM view). It can cost the driver compression, so give it only to images that need it.
///
/// Only the sets lava allows are usages ([`ImageUsage`]): none is empty, none is both a
/// colour and a depth attachment, host copies need lava's `host-image-copy` feature and never
/// go with an attachment usage (their layouts would be the attachments', which host copies
/// don't take). The others are types that `cargo check` refuses wherever an image needs a
/// usage:
/// ```compile_fail
/// # use lava::image::{Image, format::R8Unorm, usage::Usage};
/// fn empty(_: Image<R8Unorm, Usage<false, false, false, false, false, false, false, false>>) {}
/// ```
/// ```compile_fail
/// # use lava::image::{Image, format::R8Unorm, usage::usage};
/// fn mutable_alone(_: Image<R8Unorm, usage![MutableFormat]>) {}
/// ```
/// ```compile_fail
/// # use lava::image::{Image, format::D32Sfloat, usage::usage};
/// fn both(_: Image<D32Sfloat, usage![ColorAttachment, DepthAttachment]>) {}
/// ```
/// ```compile_fail
/// # use lava::image::{Image, format::R8G8B8A8Unorm, usage::usage};
/// fn host_copied_target(_: Image<R8G8B8A8Unorm, usage![ColorAttachment, HostCopy]>) {}
/// ```
/// ```compile_fail
/// # use lava::image::{Image, format::D32Sfloat, usage::usage};
/// fn host_copied_depth(_: Image<D32Sfloat, usage![DepthAttachment, Sampled, HostCopy]>) {}
/// ```
/// The traits commands bound by follow the flags:
/// ```compile_fail
/// # use lava::{command_buffer::Attachment, image::{format::R8G8B8A8Unorm, slice::ImageView, usage::usage}};
/// fn target(view: ImageView<R8G8B8A8Unorm, usage![Sampled, Storage]>) {
///     Attachment::color(view, None);
/// }
/// ```
/// ```
/// # use lava::{command_buffer::Attachment, image::{format::R8G8B8A8Unorm, slice::ImageView, usage::usage}};
/// fn target(view: ImageView<R8G8B8A8Unorm, usage![Sampled, ColorAttachment]>) {
///     Attachment::color(view, None);
/// }
/// ```
#[derive(Clone, Copy, Default)]
pub struct Usage<
    const COLOR_ATTACHMENT: bool,
    const DEPTH_ATTACHMENT: bool,
    const SAMPLED: bool,
    const STORAGE: bool,
    const HOST_COPY: bool,
    const TRANSFER_SRC: bool,
    const TRANSFER_DST: bool,
    const MUTABLE_FORMAT: bool,
>;

impl<
    const C: bool,
    const D: bool,
    const S: bool,
    const ST: bool,
    const H: bool,
    const TS: bool,
    const TD: bool,
    const M: bool,
> Usage<C, D, S, ST, H, TS, TD, M>
{
    /// The create flags of `MutableFormat`.
    pub const CREATE_FLAGS: vk::ImageCreateFlags = if M {
        vk::ImageCreateFlags::from_raw(
            vk::ImageCreateFlags::MUTABLE_FORMAT.as_raw()
                | vk::ImageCreateFlags::EXTENDED_USAGE.as_raw(),
        )
    } else {
        vk::ImageCreateFlags::empty()
    };

    /// The Vulkan flags of the set `bool`s.
    pub const FLAGS: vk::ImageUsageFlags = {
        const fn flag(set: bool, flag: vk::ImageUsageFlags) -> vk::Flags {
            if set { flag.as_raw() } else { 0 }
        }
        vk::ImageUsageFlags::from_raw(
            flag(C, vk::ImageUsageFlags::COLOR_ATTACHMENT)
                | flag(D, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                | flag(S, vk::ImageUsageFlags::SAMPLED)
                | flag(ST, vk::ImageUsageFlags::STORAGE)
                | flag(H, vk::ImageUsageFlags::HOST_TRANSFER_EXT)
                | flag(TS, vk::ImageUsageFlags::TRANSFER_SRC)
                | flag(TD, vk::ImageUsageFlags::TRANSFER_DST),
        )
    };
}

impl<
    const C: bool,
    const D: bool,
    const S: bool,
    const ST: bool,
    const H: bool,
    const TS: bool,
    const TD: bool,
    const M: bool,
> std::fmt::Debug for Usage<C, D, S, ST, H, TS, TD, M>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match M {
            true => write!(f, "Usage({:?}, mutable format)", Self::FLAGS),
            false => write!(f, "Usage({:?})", Self::FLAGS),
        }
    }
}

impl<
    const C: bool,
    const D: bool,
    const S: bool,
    const ST: bool,
    const H: bool,
    const TS: bool,
    const TD: bool,
    const M: bool,
> crate::sealed::Sealed for Usage<C, D, S, ST, H, TS, TD, M>
{
}

mod private {
    /// A `bool` as a type, for bounds on one.
    pub struct Bool<const B: bool>;
    /// Whether a usage may have `HostCopy`: always without it, with it only with the feature.
    #[diagnostic::on_unimplemented(
        message = "`HostCopy` usages need lava's `host-image-copy` feature",
        label = "a usage with `HostCopy`"
    )]
    pub trait HostCopyAllowed {}
    impl HostCopyAllowed for Bool<false> {}
    #[cfg(feature = "host-image-copy")]
    impl HostCopyAllowed for Bool<true> {}
}
use private::{Bool, HostCopyAllowed};

/// Implements `trait` for each usage type, generic over the `bool`s it names: `ImageUsage` with
/// its flags, or a marker trait for the usages among them (`where Self: ImageUsage`).
macro_rules! usage_impls {
    ($trait:ident $body:tt $($(#[$attr:meta])* <$($param:ident),*> $ty:ty $(where $bounded:ty: $bound:path)?;)*) => {$(
        $(#[$attr])*
        impl<$(const $param: bool),*> $trait for $ty $(where $bounded: $bound)? $body
    )*};
}

// One impl per flag, for the sets in which it is the first: together every non-empty set but
// colour with depth attachment, which no format supports, and attachments with host copies.
usage_impls! {
    ImageUsage {
        fn flags(&self) -> vk::ImageUsageFlags {
            Self::FLAGS
        }
        fn create_flags(&self) -> vk::ImageCreateFlags {
            Self::CREATE_FLAGS
        }
        fn check(&self, _: vk::ImageUsageFlags) {}
    }
    <S, ST, TS, TD, M> Usage<true, false, S, ST, false, TS, TD, M>;
    <S, ST, TS, TD, M> Usage<false, true, S, ST, false, TS, TD, M>;
    <ST, H, TS, TD, M> Usage<false, false, true, ST, H, TS, TD, M> where Bool<H>: HostCopyAllowed;
    <H, TS, TD, M> Usage<false, false, false, true, H, TS, TD, M> where Bool<H>: HostCopyAllowed;
    #[cfg(feature = "host-image-copy")]
    <TS, TD, M> Usage<false, false, false, false, true, TS, TD, M>;
    <TD, M> Usage<false, false, false, false, false, true, TD, M>;
    <M> Usage<false, false, false, false, false, false, true, M>;
}

/// Implements each marker trait for the usages with its flag set.
macro_rules! flag_traits {
    ($($(#[$attr:meta])* $trait:ident: <$($param:ident),*> $ty:ty;)*) => {$(
        usage_impls! { $trait {} $(#[$attr])* <$($param),*> $ty where Self: ImageUsage; }
    )*};
}

flag_traits! {
    IsColorAttachment: <D, S, ST, H, TS, TD, M> Usage<true, D, S, ST, H, TS, TD, M>;
    IsDepthAttachment: <C, S, ST, H, TS, TD, M> Usage<C, true, S, ST, H, TS, TD, M>;
    IsSampled: <C, D, ST, H, TS, TD, M> Usage<C, D, true, ST, H, TS, TD, M>;
    IsStorage: <C, D, S, H, TS, TD, M> Usage<C, D, S, true, H, TS, TD, M>;
    #[cfg(feature = "host-image-copy")]
    IsHostCopy: <C, D, S, ST, TS, TD, M> Usage<C, D, S, ST, true, TS, TD, M>;
    IsTransferSrc: <C, D, S, ST, H, TD, M> Usage<C, D, S, ST, H, true, TD, M>;
    IsTransferDst: <C, D, S, ST, H, TS, M> Usage<C, D, S, ST, H, TS, true, M>;
    IsMutableFormat: <C, D, S, ST, H, TS, TD> Usage<C, D, S, ST, H, TS, TD, true>;
}

impl<
    const C: bool,
    const D: bool,
    const S: bool,
    const ST: bool,
    const H: bool,
    const TS: bool,
    const TD: bool,
    const M: bool,
> StaticUsage for Usage<C, D, S, ST, H, TS, TD, M>
where
    Self: ImageUsage,
{
    const VK: vk::ImageUsageFlags = Self::FLAGS;
}

/// The flags [`usage!`](crate::usage) takes.
#[doc(hidden)]
#[derive(Clone, Copy)]
pub enum Flag {
    ColorAttachment,
    DepthAttachment,
    Sampled,
    Storage,
    HostCopy,
    TransferSrc,
    TransferDst,
    MutableFormat,
}

/// Whether `flags` holds `flag`, for [`usage!`](crate::usage).
#[doc(hidden)]
pub const fn has(flags: &[Flag], flag: Flag) -> bool {
    let mut i = 0;
    while i < flags.len() {
        if flags[i] as u8 == flag as u8 {
            return true;
        }
        i += 1;
    }
    false
}

/// The [`Usage`] with the given flags, in any order: `ColorAttachment`, `DepthAttachment`,
/// `Sampled`, `Storage`, `HostCopy`, `TransferSrc`, `TransferDst`, and `MutableFormat` (no
/// usage, see [`Usage`]).
/// `usage![Sampled, TransferDst]` and `usage![TransferDst, Sampled]` are the same type.
#[macro_export]
macro_rules! usage {
    ($($flag:ident),+ $(,)?) => {
        $crate::image::usage::Usage<
            { $crate::image::usage::has(&[$($crate::image::usage::Flag::$flag),+], $crate::image::usage::Flag::ColorAttachment) },
            { $crate::image::usage::has(&[$($crate::image::usage::Flag::$flag),+], $crate::image::usage::Flag::DepthAttachment) },
            { $crate::image::usage::has(&[$($crate::image::usage::Flag::$flag),+], $crate::image::usage::Flag::Sampled) },
            { $crate::image::usage::has(&[$($crate::image::usage::Flag::$flag),+], $crate::image::usage::Flag::Storage) },
            { $crate::image::usage::has(&[$($crate::image::usage::Flag::$flag),+], $crate::image::usage::Flag::HostCopy) },
            { $crate::image::usage::has(&[$($crate::image::usage::Flag::$flag),+], $crate::image::usage::Flag::TransferSrc) },
            { $crate::image::usage::has(&[$($crate::image::usage::Flag::$flag),+], $crate::image::usage::Flag::TransferDst) },
            { $crate::image::usage::has(&[$($crate::image::usage::Flag::$flag),+], $crate::image::usage::Flag::MutableFormat) },
        >
    };
}
pub use crate::usage;

/// A usage only known at runtime. It meets every usage bound, and `check` panics where the
/// bound wouldn't have compiled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dynamic(pub vk::ImageUsageFlags);

impl crate::sealed::Sealed for Dynamic {}
impl ImageUsage for Dynamic {
    fn flags(&self) -> vk::ImageUsageFlags {
        self.0
    }
    #[track_caller]
    fn check(&self, required: vk::ImageUsageFlags) {
        assert!(
            self.0.contains(required),
            "an image with usage {:?} lacks {:?}",
            self.0,
            required & !self.0
        );
    }
}

impl IsSampled for Dynamic {}
impl IsStorage for Dynamic {}
impl IsColorAttachment for Dynamic {}
impl IsDepthAttachment for Dynamic {}
#[cfg(feature = "host-image-copy")]
impl IsHostCopy for Dynamic {}
impl IsTransferSrc for Dynamic {}
impl IsTransferDst for Dynamic {}
impl IsMutableFormat for Dynamic {}

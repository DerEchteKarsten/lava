//! Image usages: one marker type per combination of usages Vulkan allows, mapping to Vulkan flags and the bindless sets they can be bound in, and `Dynamic`, whose usage is only known at runtime
use crate::vk;

pub trait ImageUsage: crate::sealed::Sealed + 'static + Copy + Send + Sync + std::fmt::Debug {
    fn flags(&self) -> vk::ImageUsageFlags;
    /// Panics unless the usage has every flag of `required`. The command buffer calls it with
    /// what its trait bounds (`IsSampled`, `IsTransferDst`, ...) require. A static usage's type
    /// already guarantees that, so its `check` does nothing and compiles away; `Dynamic` meets
    /// every bound and checks at runtime instead.
    fn check(&self, required: vk::ImageUsageFlags);
}

/// Every usage but `Dynamic`: the type alone says what the usage is.
pub trait StaticUsage: ImageUsage + Default {
    const VK: vk::ImageUsageFlags;
}

pub trait IsSampled: ImageUsage {}
pub trait IsStorage: ImageUsage {}
pub trait IsColorAttachment: ImageUsage {}
pub trait IsDepthAttachment: ImageUsage {}
/// Usages the host can copy texels into (`Image::copy_from`). Host transfer usage can cost
/// the driver its compressed layouts, so only images the host writes should have it.
pub trait IsHostCopy: ImageUsage {}
/// Usages that can be the source of a copy or blit.
pub trait IsTransferSrc: ImageUsage {}
/// Usages that can be the destination of a copy, blit or clear.
pub trait IsTransferDst: ImageUsage {}

/// Usages that can be bound in both sets, which a `MutTex2D` needs.
pub trait UnifiedBinding: IsSampled + IsStorage {}
impl<U: IsSampled + IsStorage> UnifiedBinding for U {}

/// Defines one usage per `Name: Flag...;` line: its Vulkan flags and a marker trait per flag.
macro_rules! usages {
    (@flag ColorAttachment) => { vk::ImageUsageFlags::COLOR_ATTACHMENT };
    (@flag DepthAttachment) => { vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT };
    (@flag Sampled) => { vk::ImageUsageFlags::SAMPLED };
    (@flag Storage) => { vk::ImageUsageFlags::STORAGE };
    (@flag HostCopy) => { vk::ImageUsageFlags::HOST_TRANSFER_EXT };
    (@flag TransferSrc) => { vk::ImageUsageFlags::TRANSFER_SRC };
    (@flag TransferDst) => { vk::ImageUsageFlags::TRANSFER_DST };

    (@trait $name:ident ColorAttachment) => { impl IsColorAttachment for $name {} };
    (@trait $name:ident DepthAttachment) => { impl IsDepthAttachment for $name {} };
    (@trait $name:ident Sampled) => { impl IsSampled for $name {} };
    (@trait $name:ident Storage) => { impl IsStorage for $name {} };
    (@trait $name:ident HostCopy) => { impl IsHostCopy for $name {} };
    (@trait $name:ident TransferSrc) => { impl IsTransferSrc for $name {} };
    (@trait $name:ident TransferDst) => { impl IsTransferDst for $name {} };

    ($($name:ident: $($flag:ident)+;)*) => {
        $(
            #[derive(Clone, Copy, Debug, Default)]
            pub struct $name;
            impl StaticUsage for $name {
                const VK: vk::ImageUsageFlags =
                    vk::ImageUsageFlags::from_raw(0 $(| usages!(@flag $flag).as_raw())+);
            }
            impl crate::sealed::Sealed for $name {}
            impl ImageUsage for $name {
                fn flags(&self) -> vk::ImageUsageFlags {
                    Self::VK
                }
                fn check(&self, _: vk::ImageUsageFlags) {}
            }
            $(usages!(@trait $name $flag);)+
        )*

        #[cfg(test)]
        const ALL: &[(&str, vk::ImageUsageFlags)] = &[$((stringify!($name), $name::VK)),*];
    };
}

// Every non-empty set of these flags except colour together with depth attachment, which no
// format supports. Whether a format supports a usage (storage on depth, host copies) is
// up to the device.
usages! {
    ColorAttachment: ColorAttachment;
    ColorAttachmentSampled: ColorAttachment Sampled;
    ColorAttachmentStorage: ColorAttachment Storage;
    ColorAttachmentSampledStorage: ColorAttachment Sampled Storage;
    ColorAttachmentHostCopy: ColorAttachment HostCopy;
    ColorAttachmentSampledHostCopy: ColorAttachment Sampled HostCopy;
    ColorAttachmentStorageHostCopy: ColorAttachment Storage HostCopy;
    ColorAttachmentSampledStorageHostCopy: ColorAttachment Sampled Storage HostCopy;
    ColorAttachmentTransferSrc: ColorAttachment TransferSrc;
    ColorAttachmentSampledTransferSrc: ColorAttachment Sampled TransferSrc;
    ColorAttachmentStorageTransferSrc: ColorAttachment Storage TransferSrc;
    ColorAttachmentSampledStorageTransferSrc: ColorAttachment Sampled Storage TransferSrc;
    ColorAttachmentHostCopyTransferSrc: ColorAttachment HostCopy TransferSrc;
    ColorAttachmentSampledHostCopyTransferSrc: ColorAttachment Sampled HostCopy TransferSrc;
    ColorAttachmentStorageHostCopyTransferSrc: ColorAttachment Storage HostCopy TransferSrc;
    ColorAttachmentSampledStorageHostCopyTransferSrc: ColorAttachment Sampled Storage HostCopy TransferSrc;
    ColorAttachmentTransferDst: ColorAttachment TransferDst;
    ColorAttachmentSampledTransferDst: ColorAttachment Sampled TransferDst;
    ColorAttachmentStorageTransferDst: ColorAttachment Storage TransferDst;
    ColorAttachmentSampledStorageTransferDst: ColorAttachment Sampled Storage TransferDst;
    ColorAttachmentHostCopyTransferDst: ColorAttachment HostCopy TransferDst;
    ColorAttachmentSampledHostCopyTransferDst: ColorAttachment Sampled HostCopy TransferDst;
    ColorAttachmentStorageHostCopyTransferDst: ColorAttachment Storage HostCopy TransferDst;
    ColorAttachmentSampledStorageHostCopyTransferDst: ColorAttachment Sampled Storage HostCopy TransferDst;
    ColorAttachmentTransferSrcTransferDst: ColorAttachment TransferSrc TransferDst;
    ColorAttachmentSampledTransferSrcTransferDst: ColorAttachment Sampled TransferSrc TransferDst;
    ColorAttachmentStorageTransferSrcTransferDst: ColorAttachment Storage TransferSrc TransferDst;
    ColorAttachmentSampledStorageTransferSrcTransferDst: ColorAttachment Sampled Storage TransferSrc TransferDst;
    ColorAttachmentHostCopyTransferSrcTransferDst: ColorAttachment HostCopy TransferSrc TransferDst;
    ColorAttachmentSampledHostCopyTransferSrcTransferDst: ColorAttachment Sampled HostCopy TransferSrc TransferDst;
    ColorAttachmentStorageHostCopyTransferSrcTransferDst: ColorAttachment Storage HostCopy TransferSrc TransferDst;
    ColorAttachmentSampledStorageHostCopyTransferSrcTransferDst: ColorAttachment Sampled Storage HostCopy TransferSrc TransferDst;
    DepthAttachment: DepthAttachment;
    DepthAttachmentSampled: DepthAttachment Sampled;
    DepthAttachmentStorage: DepthAttachment Storage;
    DepthAttachmentSampledStorage: DepthAttachment Sampled Storage;
    DepthAttachmentHostCopy: DepthAttachment HostCopy;
    DepthAttachmentSampledHostCopy: DepthAttachment Sampled HostCopy;
    DepthAttachmentStorageHostCopy: DepthAttachment Storage HostCopy;
    DepthAttachmentSampledStorageHostCopy: DepthAttachment Sampled Storage HostCopy;
    DepthAttachmentTransferSrc: DepthAttachment TransferSrc;
    DepthAttachmentSampledTransferSrc: DepthAttachment Sampled TransferSrc;
    DepthAttachmentStorageTransferSrc: DepthAttachment Storage TransferSrc;
    DepthAttachmentSampledStorageTransferSrc: DepthAttachment Sampled Storage TransferSrc;
    DepthAttachmentHostCopyTransferSrc: DepthAttachment HostCopy TransferSrc;
    DepthAttachmentSampledHostCopyTransferSrc: DepthAttachment Sampled HostCopy TransferSrc;
    DepthAttachmentStorageHostCopyTransferSrc: DepthAttachment Storage HostCopy TransferSrc;
    DepthAttachmentSampledStorageHostCopyTransferSrc: DepthAttachment Sampled Storage HostCopy TransferSrc;
    DepthAttachmentTransferDst: DepthAttachment TransferDst;
    DepthAttachmentSampledTransferDst: DepthAttachment Sampled TransferDst;
    DepthAttachmentStorageTransferDst: DepthAttachment Storage TransferDst;
    DepthAttachmentSampledStorageTransferDst: DepthAttachment Sampled Storage TransferDst;
    DepthAttachmentHostCopyTransferDst: DepthAttachment HostCopy TransferDst;
    DepthAttachmentSampledHostCopyTransferDst: DepthAttachment Sampled HostCopy TransferDst;
    DepthAttachmentStorageHostCopyTransferDst: DepthAttachment Storage HostCopy TransferDst;
    DepthAttachmentSampledStorageHostCopyTransferDst: DepthAttachment Sampled Storage HostCopy TransferDst;
    DepthAttachmentTransferSrcTransferDst: DepthAttachment TransferSrc TransferDst;
    DepthAttachmentSampledTransferSrcTransferDst: DepthAttachment Sampled TransferSrc TransferDst;
    DepthAttachmentStorageTransferSrcTransferDst: DepthAttachment Storage TransferSrc TransferDst;
    DepthAttachmentSampledStorageTransferSrcTransferDst: DepthAttachment Sampled Storage TransferSrc TransferDst;
    DepthAttachmentHostCopyTransferSrcTransferDst: DepthAttachment HostCopy TransferSrc TransferDst;
    DepthAttachmentSampledHostCopyTransferSrcTransferDst: DepthAttachment Sampled HostCopy TransferSrc TransferDst;
    DepthAttachmentStorageHostCopyTransferSrcTransferDst: DepthAttachment Storage HostCopy TransferSrc TransferDst;
    DepthAttachmentSampledStorageHostCopyTransferSrcTransferDst: DepthAttachment Sampled Storage HostCopy TransferSrc TransferDst;
    Sampled: Sampled;
    Storage: Storage;
    SampledStorage: Sampled Storage;
    HostCopy: HostCopy;
    SampledHostCopy: Sampled HostCopy;
    StorageHostCopy: Storage HostCopy;
    SampledStorageHostCopy: Sampled Storage HostCopy;
    TransferSrc: TransferSrc;
    SampledTransferSrc: Sampled TransferSrc;
    StorageTransferSrc: Storage TransferSrc;
    SampledStorageTransferSrc: Sampled Storage TransferSrc;
    HostCopyTransferSrc: HostCopy TransferSrc;
    SampledHostCopyTransferSrc: Sampled HostCopy TransferSrc;
    StorageHostCopyTransferSrc: Storage HostCopy TransferSrc;
    SampledStorageHostCopyTransferSrc: Sampled Storage HostCopy TransferSrc;
    TransferDst: TransferDst;
    SampledTransferDst: Sampled TransferDst;
    StorageTransferDst: Storage TransferDst;
    SampledStorageTransferDst: Sampled Storage TransferDst;
    HostCopyTransferDst: HostCopy TransferDst;
    SampledHostCopyTransferDst: Sampled HostCopy TransferDst;
    StorageHostCopyTransferDst: Storage HostCopy TransferDst;
    SampledStorageHostCopyTransferDst: Sampled Storage HostCopy TransferDst;
    TransferSrcTransferDst: TransferSrc TransferDst;
    SampledTransferSrcTransferDst: Sampled TransferSrc TransferDst;
    StorageTransferSrcTransferDst: Storage TransferSrc TransferDst;
    SampledStorageTransferSrcTransferDst: Sampled Storage TransferSrc TransferDst;
    HostCopyTransferSrcTransferDst: HostCopy TransferSrc TransferDst;
    SampledHostCopyTransferSrcTransferDst: Sampled HostCopy TransferSrc TransferDst;
    StorageHostCopyTransferSrcTransferDst: Storage HostCopy TransferSrc TransferDst;
    SampledStorageHostCopyTransferSrcTransferDst: Sampled Storage HostCopy TransferSrc TransferDst;
}

/// A usage only known at runtime. It meets every usage bound, and `check` panics where the
/// bound wouldn't have compiled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dynamic(pub vk::ImageUsageFlags);

impl crate::sealed::Sealed for Dynamic {}
impl ImageUsage for Dynamic {
    fn flags(&self) -> vk::ImageUsageFlags {
        self.0
    }
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
impl IsHostCopy for Dynamic {}
impl IsTransferSrc for Dynamic {}
impl IsTransferDst for Dynamic {}

#[cfg(test)]
mod tests {
    use super::*;
    use vk::ImageUsageFlags as F;

    const FLAGS: [(&str, F); 7] = [
        ("ColorAttachment", F::COLOR_ATTACHMENT),
        ("DepthAttachment", F::DEPTH_STENCIL_ATTACHMENT),
        ("Sampled", F::SAMPLED),
        ("Storage", F::STORAGE),
        ("HostCopy", F::HOST_TRANSFER_EXT),
        ("TransferSrc", F::TRANSFER_SRC),
        ("TransferDst", F::TRANSFER_DST),
    ];

    #[test]
    fn names_list_their_flags() {
        for (name, flags) in ALL {
            let expected: String = FLAGS
                .iter()
                .filter(|(_, flag)| flags.contains(*flag))
                .map(|(name, _)| *name)
                .collect();
            assert_eq!(*name, expected);
        }
    }

    #[test]
    fn every_allowed_combination_exists_once() {
        let mut expected: Vec<F> = (1..1u32 << FLAGS.len())
            .map(|mask| {
                FLAGS
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| mask & 1 << i != 0)
                    .fold(F::empty(), |all, (_, (_, flag))| all | *flag)
            })
            .filter(|flags| !flags.contains(F::COLOR_ATTACHMENT | F::DEPTH_STENCIL_ATTACHMENT))
            .collect();
        let mut all: Vec<F> = ALL.iter().map(|(_, flags)| *flags).collect();
        expected.sort();
        all.sort();
        assert_eq!(all, expected);
    }

    #[test]
    fn marker_traits_match_the_flags() {
        fn sampled<U: IsSampled + StaticUsage>() -> F {
            U::VK
        }
        fn storage<U: IsStorage + StaticUsage>() -> F {
            U::VK
        }
        fn host_copy<U: IsHostCopy + StaticUsage>() -> F {
            U::VK
        }
        fn src<U: IsTransferSrc + StaticUsage>() -> F {
            U::VK
        }
        fn dst<U: IsTransferDst + StaticUsage>() -> F {
            U::VK
        }
        fn unified_binding<U: UnifiedBinding + StaticUsage>() -> F {
            U::VK
        }
        assert!(sampled::<DepthAttachmentSampledTransferSrc>().contains(F::SAMPLED));
        assert!(storage::<ColorAttachmentStorageTransferDst>().contains(F::STORAGE));
        assert!(host_copy::<SampledHostCopy>().contains(F::HOST_TRANSFER_EXT));
        assert!(src::<ColorAttachmentTransferSrc>().contains(F::TRANSFER_SRC));
        assert!(dst::<SampledHostCopyTransferDst>().contains(F::TRANSFER_DST));
        assert_eq!(
            unified_binding::<SampledStorageTransferSrc>(),
            F::SAMPLED | F::STORAGE | F::TRANSFER_SRC
        );
    }

    #[test]
    fn dynamic_usage_checks_its_flags() {
        let usage = Dynamic(F::SAMPLED | F::TRANSFER_DST);
        usage.check(F::SAMPLED);
        usage.check(F::SAMPLED | F::TRANSFER_DST);
        assert!(std::panic::catch_unwind(|| usage.check(F::STORAGE)).is_err());
        // Static usages are checked by their type.
        Sampled.check(F::STORAGE);
    }
}

//! Type-level image usage markers, one per combination of usages Vulkan allows, mapping to Vulkan flags and the bindless sets they can be bound in
use ash::vk;

pub trait ImageUsage: 'static + Copy + Clone {
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

/// Usages bound only in the sampled set.
pub trait SampledBinding: IsSampled {}
/// Usages bound only in the storage set.
pub trait StorageBinding: IsStorage {}
/// Usages bound in both sets.
pub trait UnifiedBinding: IsSampled + IsStorage {}

/// Defines one usage per `Name: Flag...;` line: its Vulkan flags, a marker trait per flag,
/// and the binding trait its sampled and storage flags pick.
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

    (@bind $name:ident $sampled:tt $storage:tt; Sampled $($rest:ident)*) => {
        usages!(@bind $name yes $storage; $($rest)*);
    };
    (@bind $name:ident $sampled:tt $storage:tt; Storage $($rest:ident)*) => {
        usages!(@bind $name $sampled yes; $($rest)*);
    };
    (@bind $name:ident $sampled:tt $storage:tt; $other:ident $($rest:ident)*) => {
        usages!(@bind $name $sampled $storage; $($rest)*);
    };
    (@bind $name:ident yes no;) => { impl SampledBinding for $name {} };
    (@bind $name:ident no yes;) => { impl StorageBinding for $name {} };
    (@bind $name:ident yes yes;) => { impl UnifiedBinding for $name {} };
    (@bind $name:ident no no;) => {};

    ($($name:ident: $($flag:ident)+;)*) => {
        $(
            #[derive(Clone, Copy, Debug)]
            pub struct $name;
            impl ImageUsage for $name {
                const VK: vk::ImageUsageFlags =
                    vk::ImageUsageFlags::from_raw(0 $(| usages!(@flag $flag).as_raw())+);
            }
            $(usages!(@trait $name $flag);)+
            usages!(@bind $name no no; $($flag)+);
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
        fn sampled<U: IsSampled>() -> F {
            U::VK
        }
        fn storage<U: IsStorage>() -> F {
            U::VK
        }
        fn host_copy<U: IsHostCopy>() -> F {
            U::VK
        }
        fn src<U: IsTransferSrc>() -> F {
            U::VK
        }
        fn dst<U: IsTransferDst>() -> F {
            U::VK
        }
        fn sampled_binding<U: SampledBinding>() -> F {
            U::VK
        }
        fn storage_binding<U: StorageBinding>() -> F {
            U::VK
        }
        fn unified_binding<U: UnifiedBinding>() -> F {
            U::VK
        }
        assert!(sampled::<DepthAttachmentSampledTransferSrc>().contains(F::SAMPLED));
        assert!(storage::<ColorAttachmentStorageTransferDst>().contains(F::STORAGE));
        assert!(host_copy::<SampledHostCopy>().contains(F::HOST_TRANSFER_EXT));
        assert!(src::<ColorAttachmentTransferSrc>().contains(F::TRANSFER_SRC));
        assert!(dst::<SampledHostCopyTransferDst>().contains(F::TRANSFER_DST));
        assert_eq!(sampled_binding::<SampledTransferDst>(), F::SAMPLED | F::TRANSFER_DST);
        assert_eq!(storage_binding::<DepthAttachmentStorage>(), F::DEPTH_STENCIL_ATTACHMENT | F::STORAGE);
        assert_eq!(unified_binding::<SampledStorageTransferSrc>(), F::SAMPLED | F::STORAGE | F::TRANSFER_SRC);
    }
}

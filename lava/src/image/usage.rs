//! Image usages: one marker type per combination of usages Vulkan allows (host copies only with `host-image-copy`), mapping to Vulkan flags and the bindless descriptors they can be bound with, and `Dynamic`, whose usage is only known at runtime
use crate::vk;

pub trait ImageUsage: crate::sealed::Sealed + 'static + Copy + Send + Sync + std::fmt::Debug {
    fn flags(&self) -> vk::ImageUsageFlags;
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

/// Usages that can be bound with both descriptors (`bind_combined`), which a `MutTex2D` needs.
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
    (@trait $name:ident HostCopy) => {
        #[cfg(feature = "host-image-copy")]
        impl IsHostCopy for $name {}
    };
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
    };
}

// Every non-empty set of these flags but colour with depth attachment, which no format
// supports.
usages! {
    ColorAttachment: ColorAttachment;
    ColorAttachmentSampled: ColorAttachment Sampled;
    ColorAttachmentStorage: ColorAttachment Storage;
    ColorAttachmentSampledStorage: ColorAttachment Sampled Storage;
    ColorAttachmentTransferSrc: ColorAttachment TransferSrc;
    ColorAttachmentSampledTransferSrc: ColorAttachment Sampled TransferSrc;
    ColorAttachmentStorageTransferSrc: ColorAttachment Storage TransferSrc;
    ColorAttachmentSampledStorageTransferSrc: ColorAttachment Sampled Storage TransferSrc;
    ColorAttachmentTransferDst: ColorAttachment TransferDst;
    ColorAttachmentSampledTransferDst: ColorAttachment Sampled TransferDst;
    ColorAttachmentStorageTransferDst: ColorAttachment Storage TransferDst;
    ColorAttachmentSampledStorageTransferDst: ColorAttachment Sampled Storage TransferDst;
    ColorAttachmentTransferSrcTransferDst: ColorAttachment TransferSrc TransferDst;
    ColorAttachmentSampledTransferSrcTransferDst: ColorAttachment Sampled TransferSrc TransferDst;
    ColorAttachmentStorageTransferSrcTransferDst: ColorAttachment Storage TransferSrc TransferDst;
    ColorAttachmentSampledStorageTransferSrcTransferDst: ColorAttachment Sampled Storage TransferSrc TransferDst;
    DepthAttachment: DepthAttachment;
    DepthAttachmentSampled: DepthAttachment Sampled;
    DepthAttachmentStorage: DepthAttachment Storage;
    DepthAttachmentSampledStorage: DepthAttachment Sampled Storage;
    DepthAttachmentTransferSrc: DepthAttachment TransferSrc;
    DepthAttachmentSampledTransferSrc: DepthAttachment Sampled TransferSrc;
    DepthAttachmentStorageTransferSrc: DepthAttachment Storage TransferSrc;
    DepthAttachmentSampledStorageTransferSrc: DepthAttachment Sampled Storage TransferSrc;
    DepthAttachmentTransferDst: DepthAttachment TransferDst;
    DepthAttachmentSampledTransferDst: DepthAttachment Sampled TransferDst;
    DepthAttachmentStorageTransferDst: DepthAttachment Storage TransferDst;
    DepthAttachmentSampledStorageTransferDst: DepthAttachment Sampled Storage TransferDst;
    DepthAttachmentTransferSrcTransferDst: DepthAttachment TransferSrc TransferDst;
    DepthAttachmentSampledTransferSrcTransferDst: DepthAttachment Sampled TransferSrc TransferDst;
    DepthAttachmentStorageTransferSrcTransferDst: DepthAttachment Storage TransferSrc TransferDst;
    DepthAttachmentSampledStorageTransferSrcTransferDst: DepthAttachment Sampled Storage TransferSrc TransferDst;
    Sampled: Sampled;
    Storage: Storage;
    SampledStorage: Sampled Storage;
    TransferSrc: TransferSrc;
    SampledTransferSrc: Sampled TransferSrc;
    StorageTransferSrc: Storage TransferSrc;
    SampledStorageTransferSrc: Sampled Storage TransferSrc;
    TransferDst: TransferDst;
    SampledTransferDst: Sampled TransferDst;
    StorageTransferDst: Storage TransferDst;
    SampledStorageTransferDst: Sampled Storage TransferDst;
    TransferSrcTransferDst: TransferSrc TransferDst;
    SampledTransferSrcTransferDst: Sampled TransferSrc TransferDst;
    StorageTransferSrcTransferDst: Storage TransferSrc TransferDst;
    SampledStorageTransferSrcTransferDst: Sampled Storage TransferSrc TransferDst;
}

#[cfg(feature = "host-image-copy")]
usages! {
    ColorAttachmentHostCopy: ColorAttachment HostCopy;
    ColorAttachmentSampledHostCopy: ColorAttachment Sampled HostCopy;
    ColorAttachmentStorageHostCopy: ColorAttachment Storage HostCopy;
    ColorAttachmentSampledStorageHostCopy: ColorAttachment Sampled Storage HostCopy;
    ColorAttachmentHostCopyTransferSrc: ColorAttachment HostCopy TransferSrc;
    ColorAttachmentSampledHostCopyTransferSrc: ColorAttachment Sampled HostCopy TransferSrc;
    ColorAttachmentStorageHostCopyTransferSrc: ColorAttachment Storage HostCopy TransferSrc;
    ColorAttachmentSampledStorageHostCopyTransferSrc: ColorAttachment Sampled Storage HostCopy TransferSrc;
    ColorAttachmentHostCopyTransferDst: ColorAttachment HostCopy TransferDst;
    ColorAttachmentSampledHostCopyTransferDst: ColorAttachment Sampled HostCopy TransferDst;
    ColorAttachmentStorageHostCopyTransferDst: ColorAttachment Storage HostCopy TransferDst;
    ColorAttachmentSampledStorageHostCopyTransferDst: ColorAttachment Sampled Storage HostCopy TransferDst;
    ColorAttachmentHostCopyTransferSrcTransferDst: ColorAttachment HostCopy TransferSrc TransferDst;
    ColorAttachmentSampledHostCopyTransferSrcTransferDst: ColorAttachment Sampled HostCopy TransferSrc TransferDst;
    ColorAttachmentStorageHostCopyTransferSrcTransferDst: ColorAttachment Storage HostCopy TransferSrc TransferDst;
    ColorAttachmentSampledStorageHostCopyTransferSrcTransferDst: ColorAttachment Sampled Storage HostCopy TransferSrc TransferDst;
    DepthAttachmentHostCopy: DepthAttachment HostCopy;
    DepthAttachmentSampledHostCopy: DepthAttachment Sampled HostCopy;
    DepthAttachmentStorageHostCopy: DepthAttachment Storage HostCopy;
    DepthAttachmentSampledStorageHostCopy: DepthAttachment Sampled Storage HostCopy;
    DepthAttachmentHostCopyTransferSrc: DepthAttachment HostCopy TransferSrc;
    DepthAttachmentSampledHostCopyTransferSrc: DepthAttachment Sampled HostCopy TransferSrc;
    DepthAttachmentStorageHostCopyTransferSrc: DepthAttachment Storage HostCopy TransferSrc;
    DepthAttachmentSampledStorageHostCopyTransferSrc: DepthAttachment Sampled Storage HostCopy TransferSrc;
    DepthAttachmentHostCopyTransferDst: DepthAttachment HostCopy TransferDst;
    DepthAttachmentSampledHostCopyTransferDst: DepthAttachment Sampled HostCopy TransferDst;
    DepthAttachmentStorageHostCopyTransferDst: DepthAttachment Storage HostCopy TransferDst;
    DepthAttachmentSampledStorageHostCopyTransferDst: DepthAttachment Sampled Storage HostCopy TransferDst;
    DepthAttachmentHostCopyTransferSrcTransferDst: DepthAttachment HostCopy TransferSrc TransferDst;
    DepthAttachmentSampledHostCopyTransferSrcTransferDst: DepthAttachment Sampled HostCopy TransferSrc TransferDst;
    DepthAttachmentStorageHostCopyTransferSrcTransferDst: DepthAttachment Storage HostCopy TransferSrc TransferDst;
    DepthAttachmentSampledStorageHostCopyTransferSrcTransferDst: DepthAttachment Sampled Storage HostCopy TransferSrc TransferDst;
    HostCopy: HostCopy;
    SampledHostCopy: Sampled HostCopy;
    StorageHostCopy: Storage HostCopy;
    SampledStorageHostCopy: Sampled Storage HostCopy;
    HostCopyTransferSrc: HostCopy TransferSrc;
    SampledHostCopyTransferSrc: Sampled HostCopy TransferSrc;
    StorageHostCopyTransferSrc: Storage HostCopy TransferSrc;
    SampledStorageHostCopyTransferSrc: Sampled Storage HostCopy TransferSrc;
    HostCopyTransferDst: HostCopy TransferDst;
    SampledHostCopyTransferDst: Sampled HostCopy TransferDst;
    StorageHostCopyTransferDst: Storage HostCopy TransferDst;
    SampledStorageHostCopyTransferDst: Sampled Storage HostCopy TransferDst;
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
#[cfg(feature = "host-image-copy")]
impl IsHostCopy for Dynamic {}
impl IsTransferSrc for Dynamic {}
impl IsTransferDst for Dynamic {}

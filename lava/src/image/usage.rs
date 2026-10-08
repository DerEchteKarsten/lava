//! Type-level image usage markers mapping to Vulkan flags and the bindless sets they can be bound in
use ash::vk;

pub trait ImageUsage: 'static + Copy + Clone {
    const VK: vk::ImageUsageFlags;
}

#[derive(Clone, Copy, Debug)]
pub struct Sampled;
#[derive(Clone, Copy, Debug)]
pub struct Storage;
#[derive(Clone, Copy, Debug)]
pub struct ColorAttachment;
#[derive(Clone, Copy, Debug)]
pub struct DepthAttachment;
#[derive(Clone, Copy, Debug)]
pub struct ColorAttachmentStorage;
#[derive(Clone, Copy, Debug)]
pub struct ColorAttachmentSampled;
#[derive(Clone, Copy, Debug)]
pub struct DepthAttachmentSampled;
#[derive(Clone, Copy, Debug)]
pub struct SampledStorage;

impl ImageUsage for Sampled {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::SAMPLED;
}
impl ImageUsage for Storage {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::STORAGE;
}
impl ImageUsage for ColorAttachment {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::COLOR_ATTACHMENT;
}
impl ImageUsage for DepthAttachment {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
}
impl ImageUsage for ColorAttachmentSampled {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
        vk::ImageUsageFlags::COLOR_ATTACHMENT.as_raw() | vk::ImageUsageFlags::SAMPLED.as_raw(),
    );
}
impl ImageUsage for DepthAttachmentSampled {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT.as_raw()
            | vk::ImageUsageFlags::SAMPLED.as_raw(),
    );
}
impl ImageUsage for ColorAttachmentStorage {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
        vk::ImageUsageFlags::COLOR_ATTACHMENT.as_raw() | vk::ImageUsageFlags::STORAGE.as_raw(),
    );
}
impl ImageUsage for SampledStorage {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
        vk::ImageUsageFlags::SAMPLED.as_raw() | vk::ImageUsageFlags::STORAGE.as_raw(),
    );
}

pub trait IsSampled: ImageUsage {}
pub trait IsStorage: ImageUsage {}
pub trait IsColorAttachment: ImageUsage {}
pub trait IsDepthAttachment: ImageUsage {}

impl IsSampled for Sampled {}
impl IsSampled for ColorAttachmentSampled {}
impl IsSampled for DepthAttachmentSampled {}
impl IsSampled for SampledStorage {}

impl IsStorage for Storage {}
impl IsStorage for ColorAttachmentStorage {}
impl IsStorage for SampledStorage {}

impl IsColorAttachment for ColorAttachment {}
impl IsColorAttachment for ColorAttachmentSampled {}
impl IsColorAttachment for ColorAttachmentStorage {}

impl IsDepthAttachment for DepthAttachment {}
impl IsDepthAttachment for DepthAttachmentSampled {}

/// Usages bound only in the sampled set.
pub trait SampledBinding: IsSampled {}
/// Usages bound only in the storage set.
pub trait StorageBinding: IsStorage {}
/// Usages bound in both sets.
pub trait UnifiedBinding: IsSampled + IsStorage {}

impl SampledBinding for Sampled {}
impl SampledBinding for ColorAttachmentSampled {}
impl SampledBinding for DepthAttachmentSampled {}

impl StorageBinding for Storage {}
impl StorageBinding for ColorAttachmentStorage {}

impl UnifiedBinding for SampledStorage {}

#[cfg(test)]
mod tests {
    use super::*;
    use vk::ImageUsageFlags as F;

    #[test]
    fn usages_map_to_their_flags() {
        assert_eq!(Sampled::VK, F::SAMPLED);
        assert_eq!(Storage::VK, F::STORAGE);
        assert_eq!(ColorAttachment::VK, F::COLOR_ATTACHMENT);
        assert_eq!(DepthAttachment::VK, F::DEPTH_STENCIL_ATTACHMENT);
        assert_eq!(ColorAttachmentStorage::VK, F::COLOR_ATTACHMENT | F::STORAGE);
        assert_eq!(ColorAttachmentSampled::VK, F::COLOR_ATTACHMENT | F::SAMPLED);
        assert_eq!(
            DepthAttachmentSampled::VK,
            F::DEPTH_STENCIL_ATTACHMENT | F::SAMPLED
        );
        assert_eq!(SampledStorage::VK, F::SAMPLED | F::STORAGE);
    }

    #[test]
    fn marker_traits_match_the_flags() {
        fn sampled<U: IsSampled>() -> F {
            U::VK
        }
        fn storage<U: IsStorage>() -> F {
            U::VK
        }
        fn color<U: IsColorAttachment>() -> F {
            U::VK
        }
        fn depth<U: IsDepthAttachment>() -> F {
            U::VK
        }
        for flags in [
            sampled::<Sampled>(),
            sampled::<ColorAttachmentSampled>(),
            sampled::<DepthAttachmentSampled>(),
            sampled::<SampledStorage>(),
        ] {
            assert!(flags.contains(F::SAMPLED));
        }
        for flags in [
            storage::<Storage>(),
            storage::<ColorAttachmentStorage>(),
            storage::<SampledStorage>(),
        ] {
            assert!(flags.contains(F::STORAGE));
        }
        for flags in [
            color::<ColorAttachment>(),
            color::<ColorAttachmentSampled>(),
            color::<ColorAttachmentStorage>(),
        ] {
            assert!(flags.contains(F::COLOR_ATTACHMENT));
        }
        for flags in [
            depth::<DepthAttachment>(),
            depth::<DepthAttachmentSampled>(),
        ] {
            assert!(flags.contains(F::DEPTH_STENCIL_ATTACHMENT));
        }
    }
}

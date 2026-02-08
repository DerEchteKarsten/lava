use ash::vk;

pub(crate) trait UsageSet: 'static {
    const VK: vk::ImageUsageFlags;
}

pub struct Sampled;
pub struct Storage;
pub struct ColorAttachment;
pub struct DepthAttachment;
pub struct ColorAttachmentSampled;
pub struct DepthAttachmentSampled;
pub struct SampledStorage;

impl UsageSet for Sampled {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::SAMPLED;
}
impl UsageSet for Storage {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::STORAGE;
}
impl UsageSet for ColorAttachment {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::COLOR_ATTACHMENT;
}
impl UsageSet for DepthAttachment {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
}
impl UsageSet for ColorAttachmentSampled {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
        vk::ImageUsageFlags::COLOR_ATTACHMENT.as_raw() | vk::ImageUsageFlags::SAMPLED.as_raw(),
    );
}
impl UsageSet for DepthAttachmentSampled {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT.as_raw()
            | vk::ImageUsageFlags::SAMPLED.as_raw(),
    );
}
impl UsageSet for SampledStorage {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
        vk::ImageUsageFlags::SAMPLED.as_raw() | vk::ImageUsageFlags::STORAGE.as_raw(),
    );
}

pub(crate) trait IsSampled: UsageSet {}
pub(crate) trait IsStorage: UsageSet {}
pub(crate) trait IsColorAttachment: UsageSet {}
pub(crate) trait IsDepthAttachment: UsageSet {}

impl IsSampled for Sampled {}
impl IsSampled for ColorAttachmentSampled {}
impl IsSampled for DepthAttachmentSampled {}
impl IsSampled for SampledStorage {}

impl IsStorage for Storage {}
impl IsStorage for SampledStorage {}

impl IsColorAttachment for ColorAttachment {}
impl IsColorAttachment for ColorAttachmentSampled {}

impl IsDepthAttachment for DepthAttachment {}
impl IsDepthAttachment for DepthAttachmentSampled {}

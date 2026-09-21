use ash::vk;

pub trait UsageSet: 'static + Copy + Clone {
    const VK: vk::ImageUsageFlags;
    const SET: BindlessImageUsageSet;
}

#[derive(Clone, Copy, Debug)]
pub struct Unknown;

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

impl UsageSet for Unknown {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::empty();
    const SET: BindlessImageUsageSet = BindlessImageUsageSet::None;
}

impl UsageSet for Sampled {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::SAMPLED;
    const SET: BindlessImageUsageSet = BindlessImageUsageSet::SampledImage;
}
impl UsageSet for Storage {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::STORAGE;
    const SET: BindlessImageUsageSet = BindlessImageUsageSet::StorageImage;
}
impl UsageSet for ColorAttachment {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::COLOR_ATTACHMENT;
    const SET: BindlessImageUsageSet = BindlessImageUsageSet::None;
}
impl UsageSet for DepthAttachment {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
    const SET: BindlessImageUsageSet = BindlessImageUsageSet::None;
}
impl UsageSet for ColorAttachmentSampled {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
        vk::ImageUsageFlags::COLOR_ATTACHMENT.as_raw() | vk::ImageUsageFlags::SAMPLED.as_raw(),
    );
    const SET: BindlessImageUsageSet = BindlessImageUsageSet::SampledImage;
}
impl UsageSet for DepthAttachmentSampled {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT.as_raw()
            | vk::ImageUsageFlags::SAMPLED.as_raw(),
    );
    const SET: BindlessImageUsageSet = BindlessImageUsageSet::SampledImage;
}
impl UsageSet for ColorAttachmentStorage {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
        vk::ImageUsageFlags::COLOR_ATTACHMENT.as_raw() | vk::ImageUsageFlags::STORAGE.as_raw(),
    );
    const SET: BindlessImageUsageSet = BindlessImageUsageSet::StorageImage;
}
impl UsageSet for SampledStorage {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
        vk::ImageUsageFlags::SAMPLED.as_raw() | vk::ImageUsageFlags::STORAGE.as_raw(),
    );
    const SET: BindlessImageUsageSet = BindlessImageUsageSet::Both;
}

pub enum BindlessImageUsageSet {
    None,
    StorageImage,
    SampledImage,
    Both,
}

pub trait IsSampled: UsageSet {}
pub trait IsStorage: UsageSet {}
pub trait IsColorAttachment: UsageSet {}
pub trait IsDepthAttachment: UsageSet {}

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

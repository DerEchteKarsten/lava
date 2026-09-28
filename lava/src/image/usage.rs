use ash::vk;

pub trait ImageUsage: 'static + Copy + Clone {
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

impl ImageUsage for Unknown {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::empty();
    const SET: BindlessImageUsageSet = BindlessImageUsageSet::None;
}

impl ImageUsage for Sampled {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::SAMPLED;
    const SET: BindlessImageUsageSet = BindlessImageUsageSet::SampledImage;
}
impl ImageUsage for Storage {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::STORAGE;
    const SET: BindlessImageUsageSet = BindlessImageUsageSet::StorageImage;
}
impl ImageUsage for ColorAttachment {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::COLOR_ATTACHMENT;
    const SET: BindlessImageUsageSet = BindlessImageUsageSet::None;
}
impl ImageUsage for DepthAttachment {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
    const SET: BindlessImageUsageSet = BindlessImageUsageSet::None;
}
impl ImageUsage for ColorAttachmentSampled {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
        vk::ImageUsageFlags::COLOR_ATTACHMENT.as_raw() | vk::ImageUsageFlags::SAMPLED.as_raw(),
    );
    const SET: BindlessImageUsageSet = BindlessImageUsageSet::SampledImage;
}
impl ImageUsage for DepthAttachmentSampled {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT.as_raw()
            | vk::ImageUsageFlags::SAMPLED.as_raw(),
    );
    const SET: BindlessImageUsageSet = BindlessImageUsageSet::SampledImage;
}
impl ImageUsage for ColorAttachmentStorage {
    const VK: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
        vk::ImageUsageFlags::COLOR_ATTACHMENT.as_raw() | vk::ImageUsageFlags::STORAGE.as_raw(),
    );
    const SET: BindlessImageUsageSet = BindlessImageUsageSet::StorageImage;
}
impl ImageUsage for SampledStorage {
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

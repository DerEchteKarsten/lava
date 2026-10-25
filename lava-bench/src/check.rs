//! `--check`'s checksums: passes that hash a scenario's outputs into a readback buffer per mode, recorded at the end of a frame, so lava's results can be compared with the baseline's

use lava::bindless::{ImageHandle, Sampled};
use lava::buffer::Readback;
use lava::glam::{UVec2, UVec3};
use lava::image::format::HasTexels;
use lava::image::{D2, D3, SampledView};
use lava::queue::Graphics;
use lava::vk;
use lava::{Buffer, BufferSlice, CommandBuffer};

use crate::bindings::*;
use crate::raw::{self, RawCmd};

/// One readback buffer of `slots` checksums per mode, and the baseline's checksum pipelines.
pub struct Checksums {
    lava: Buffer<u32, Readback>,
    raw: Buffer<u32, Readback>,
    buffer: vk::Pipeline,
    image: vk::Pipeline,
    volume: vk::Pipeline,
}

impl Checksums {
    pub fn new(slots: usize) -> Self {
        Self {
            lava: Buffer::readback(slots).unwrap(),
            raw: Buffer::readback(slots).unwrap(),
            buffer: crate::pipelines::compute("ChecksumBuffer"),
            image: crate::pipelines::compute("ChecksumImage"),
            volume: crate::pipelines::compute("ChecksumVolume"),
        }
    }

    /// Lava: zeroes the checksums.
    pub fn lava_clear(&self, cmd: &mut CommandBuffer<Graphics>) {
        cmd.fill_buffer(self.lava.range(..), 0);
    }

    /// Lava: checksum `slot` over the words of `src`.
    pub fn lava_buffer(&self, cmd: &mut CommandBuffer<Graphics>, src: BufferSlice<u32>, slot: u32) {
        let args = ChecksumBufferArgs {
            src,
            out: self.lava.range(..),
            count: src.len() as u32,
            slot,
        };
        cmd.compute(
            &ChecksumBuffer::push_bindings(args, None),
            ChecksumBuffer::groups([src.len() as u32, 1, 1]),
        );
    }

    /// Lava: checksum `slot` over mip 0 of a 2D image.
    pub fn lava_image<'a>(
        &'a self,
        cmd: &mut CommandBuffer<Graphics>,
        image: impl SampledView<Dim = D2, Format: HasTexels<[f32; 4]>> + 'a,
        slot: u32,
    ) {
        let size = image.view().extent();
        let args = ChecksumImageArgs {
            image,
            out: self.lava.range(..),
            size,
            slot,
        };
        cmd.compute(
            &ChecksumImage::push_bindings(args, None),
            ChecksumImage::groups([size.x, size.y, 1]),
        );
    }

    /// Lava: checksum `slot` over a 3D image.
    pub fn lava_volume<'a>(
        &'a self,
        cmd: &mut CommandBuffer<Graphics>,
        volume: impl SampledView<Dim = D3, Format: HasTexels<[f32; 4]>> + 'a,
        slot: u32,
    ) {
        let size = volume.view().extent();
        let args = ChecksumVolumeArgs {
            volume,
            out: self.lava.range(..),
            size,
            slot,
        };
        cmd.compute(
            &ChecksumVolume::push_bindings(args, None),
            ChecksumVolume::groups(size.into()),
        );
    }

    /// Raw: zeroes the checksums, ordered before the checksum passes.
    pub fn raw_clear(&self, cmd: &mut RawCmd) {
        cmd.fill(self.raw.address(), self.raw.size(), 0);
        cmd.mem(raw::TRANSFER_WRITE, raw::COMPUTE_RW);
    }

    /// Raw: checksum `slot` over `count` words at `address`, which compute shaders may read.
    pub fn raw_buffer(&self, cmd: &mut RawCmd, address: u64, count: u32, slot: u32) {
        let push = CChecksumBufferBindings {
            src: address,
            out: self.raw.address(),
            count,
            slot,
        };
        cmd.bind_compute(self.buffer);
        cmd.push(&push);
        cmd.dispatch(ChecksumBuffer::groups([count, 1, 1]));
        cmd.mem(raw::COMPUTE_RW, raw::COMPUTE_RW);
    }

    /// Raw: checksum `slot` over mip 0 of the 2D image of `handle`, which compute shaders may
    /// sample.
    pub fn raw_image(&self, cmd: &mut RawCmd, handle: ImageHandle<Sampled>, size: UVec2, slot: u32) {
        let mut push: CChecksumImageBindings = bytemuck::Zeroable::zeroed();
        push.image = handle;
        push.out = self.raw.address();
        push.size = size;
        push.slot = slot;
        cmd.bind_compute(self.image);
        cmd.push(&push);
        cmd.dispatch(ChecksumImage::groups([size.x, size.y, 1]));
        cmd.mem(raw::COMPUTE_RW, raw::COMPUTE_RW);
    }

    /// Raw: checksum `slot` over the 3D image of `handle`.
    pub fn raw_volume(&self, cmd: &mut RawCmd, handle: ImageHandle<Sampled>, size: UVec3, slot: u32) {
        let mut push: CChecksumVolumeBindings = bytemuck::Zeroable::zeroed();
        push.volume = handle;
        push.out = self.raw.address();
        push.size = size;
        push.slot = slot;
        cmd.bind_compute(self.volume);
        cmd.push(&push);
        cmd.dispatch(ChecksumVolume::groups(size.into()));
        cmd.mem(raw::COMPUTE_RW, raw::COMPUTE_RW);
    }

    /// Raw: makes the checksums visible to the host.
    pub fn raw_finish(&self, cmd: &mut RawCmd) {
        cmd.mem(raw::COMPUTE_RW, raw::HOST_READ);
    }

    /// Both modes' checksums, once their frames completed.
    pub fn read(&mut self) -> (Vec<u32>, Vec<u32>) {
        (self.lava.host().to_vec(), self.raw.host().to_vec())
    }
}

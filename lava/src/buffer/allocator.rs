use ash::vk::{self};
use async_std::sync::Mutex;
use bytemuck::Pod;
use std::default;
use std::sync::Arc;
use std::{collections::BTreeMap, marker::PhantomData};

use crate::FRAMES_IN_FLIGHT;
use crate::buffer::{BufferUsageFlags, CpuBuffer, GpuBuffer};
use crate::buffer::{AsBuffer, Buffer, Location, slice::BufferSlice};
use crate::command_buffer::CommandBuffer;
use crate::state::Ctx;

pub trait SubAllocator {
    fn allocate<T: Pod + Copy, L: Location>(
        &mut self,
        buffer: BufferSlice<T, L>,
        size: u64,
    ) -> Option<BufferSlice<T, L>>;
    fn deallocate<T: Pod + Copy, L: Location>(&mut self, slice: BufferSlice<T, L>);
}

#[derive(Clone, Default)]
pub struct ArenaAllocator {
    ptr: u64,
}

impl SubAllocator for ArenaAllocator {
    fn allocate<T: Pod + Copy, L: Location>(
        &mut self,
        buffer: BufferSlice<T, L>,
        size: u64,
    ) -> Option<BufferSlice<T, L>> {
        let slice = buffer.add_byte_offset(self.ptr);
        self.ptr += size;
        Some(slice)
    }
    fn deallocate<T: Pod + Copy, L: Location>(&mut self, slice: BufferSlice<T, L>) {
        self.ptr -= slice.size;
    }
}

#[derive(Clone)]
pub struct RangeAllocator {
    free_ranges: BTreeMap<u64, u64>,
}

impl RangeAllocator {
    pub fn new(total_size: u64) -> Self {
        let mut free_ranges = BTreeMap::new();
        free_ranges.insert(0, total_size);
        Self {
            free_ranges,
        }
    }
}

impl SubAllocator for RangeAllocator {
    fn allocate<T: Pod + Copy, L: Location>(
        &mut self,
        buffer: BufferSlice<T, L>,
        size: u64,
    ) -> Option<BufferSlice<T, L>> {
        for (&start, &length) in self.free_ranges.iter() {
            if length >= size {
                self.free_ranges.remove(&start);
                if length > size {
                    self.free_ranges.insert(start + size, length - size);
                }
                return Some(buffer.add_byte_offset(start).num_bytes(size));
            }
        }
        None
    }

    fn deallocate<T: Pod + Copy, L: Location>(&mut self, slice: BufferSlice<T, L>) {
        self.free_ranges.insert(slice.offset, slice.size);
        let mut keys_to_merge = Vec::new();
        let mut last_start = None;
        for &start in self.free_ranges.keys() {
            if let Some(prev) = last_start {
                if prev + self.free_ranges[&prev] == start {
                    keys_to_merge.push((prev, start));
                }
            }
            last_start = Some(start);
        }
        for (a, b) in keys_to_merge {
            let len_a = self.free_ranges.remove(&a).unwrap();
            let len_b = self.free_ranges.remove(&b).unwrap();
            self.free_ranges.insert(a, len_a + len_b);
        }
    }
}

pub struct SubAllocated<T: AsBuffer, A: SubAllocator> {
    allocator: A,
    pub buffer: T,
}

impl<T: AsBuffer, A: SubAllocator> SubAllocated<T, A> {
    pub fn allocate(&mut self, size: u64) -> Option<BufferSlice<T::DataType, T::Location>>
    where
        for<'a> BufferSlice<T::DataType, T::Location>: From<&'a Buffer<T::DataType, T::Location>>,
    {
        self.allocator.allocate(self.buffer.whole(), size)
    }
    pub fn deallocate(&mut self, slice: BufferSlice<T::DataType, T::Location>) {
        self.allocator.deallocate(slice)
    }
    pub fn new(buffer: T, allocator: A) -> Self {
        Self { allocator, buffer }
    }
}

impl<T: AsBuffer> SubAllocated<T, ArenaAllocator> {
    pub fn clear(&mut self) {
        self.allocator.ptr = 0;
    }
}


#[derive(Clone, Default)]
pub struct QueueAllocated<B: AsBuffer> {
    buffer: [B; FRAMES_IN_FLIGHT],
    queue: Vec<B::DataType>,
}

impl<B: AsBuffer> AsBuffer for QueueAllocated<B> {
    type Location = B::Location;
    type DataType = B::DataType;
    fn get_ref(&self) -> &Buffer<Self::DataType, Self::Location> {
        self.buffer[Ctx::frame_in_flight()].get_ref()
    }
    fn get_mut(&mut self) -> &mut Buffer<Self::DataType, Self::Location> {
        self.buffer[Ctx::frame_in_flight()].get_mut()
    }
}

impl<B: AsBuffer> QueueAllocated<B> {
    pub fn new(buffer: [B; FRAMES_IN_FLIGHT]) -> Self {
        QueueAllocated { buffer, queue: Vec::new() }    
    }
    pub fn push(&mut self, value: B::DataType) {
        self.queue.push(value);
    }
    pub fn extend<T: IntoIterator<Item = B::DataType>>(&mut self, itter: T) {
        self.queue.extend(itter);
    }
    pub fn assert_size(&mut self) {
        let size = self.queue_size();
        let buffer = self.buffer[Ctx::frame_in_flight()].get_mut();
        //Safe to delete becouse we are sure this buffer isnt used
        if buffer.size < size {
            *buffer = Buffer::with_alignment(
                    BufferUsageFlags::STORAGE,
                    size.next_power_of_two(),
                    None,
                ).unwrap();
        }
    }
    pub fn clear(&mut self) -> Vec<B::DataType> {
        std::mem::take(&mut self.queue)
    }
    pub fn len(&self) -> usize {
        self.queue.len()
    }
    pub fn queue_size(&self) -> u64 {
        (self.queue.len() * size_of::<B::DataType>()) as u64
    }
}

impl AsyncSubAllocator<ArenaAllocator> {
    pub async fn clear(&mut self) {
        self.allocator.lock().await.ptr = 0;
    }
}

#[derive(Clone)]
pub struct AsyncSubAllocator<A: SubAllocator> {
    allocator: Arc<Mutex<A>>,
    handle: vk::Buffer,
    cpu_base_ptr: usize,
    gpu_base_ptr: u64,
}

impl<A: SubAllocator> AsyncSubAllocator<A> {
    pub async fn allocate<T: Copy + Pod, L: Location>(
        &self,
        size: u64,
    ) -> Option<BufferSlice<T, L>> {
        let mut allocator = self.allocator.lock().await;
        let slice = BufferSlice {
            cpu_base_ptr: self.cpu_base_ptr,
            gpu_base_ptr: self.gpu_base_ptr,
            handle: self.handle,
            offset: 0,
            size: 0,
            _location: PhantomData,
            _marker: PhantomData,
        };
        allocator.allocate(slice, size)
    }
    pub async fn allocate_blocking<T: Copy + Pod, L: Location>(
        &self,
        size: u64,
    ) -> BufferSlice<T, L> {
        loop {
            if let Some(v) = self.allocate(size).await {
                return v;
            }
            log::error!("Staging Buffer Full!!");
        };
    }
    pub async fn deallocate<T: Copy + Pod, L: Location>(&self, slice: BufferSlice<T, L>) {
        let mut allocator = self.allocator.lock().await;
        allocator.deallocate(slice);
    }
    pub fn new<T: Pod + Copy, L: Location>(slice: BufferSlice<T, L>, allocator: A) -> Self {
        Self {
            allocator: Arc::new(Mutex::new(allocator)),
            handle: slice.handle,
            cpu_base_ptr: slice.cpu_base_ptr,
            gpu_base_ptr: slice.gpu_base_ptr,
        }
    }
}

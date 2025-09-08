use ash::vk;

use crate::state::Ctx;

pub struct Queue {
    pub family_index: usize,
    pub handle: vk::Queue,
    pub command_pool: vk::CommandPool,
}

impl Queue {
    pub fn new(family_index: usize) -> Self {
        let handle = unsafe { Ctx::device().get_device_queue(family_index, 0) };
        let command_pool = unsafe { Ctx::device().create_command_pool(vk::CommandPoolCreateInfo {
            queue_family_index: family_index as u32,
            ..Default::default()
        }, None) };
        
        Self {
            handle,
            family_index,
            command_pool
        }
    }
}
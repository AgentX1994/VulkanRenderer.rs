use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, Allocator};
use gpu_allocator::MemoryLocation;

use super::RendererResult;

pub struct Buffer {
    device: ash::Device,
    allocation: Option<Allocation>,
    pub buffer: vk::Buffer,
    pub size: u64,
    buffer_usage: vk::BufferUsageFlags,
    location: MemoryLocation,
}

impl Buffer {
    fn allocate_new_buffer(
        device: &ash::Device,
        allocator: &mut Allocator,
        size: u64,
        buffer_usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> RendererResult<(vk::Buffer, Allocation)> {
        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(buffer_usage);
        // TODO: sharing mode?
        let buffer = unsafe { device.create_buffer(&buffer_create_info, None)? };
        let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
        let allocation = allocator
            .allocate(&AllocationCreateDesc {
                name: "buffer",
                requirements: reqs,
                location,
                linear: true,
            })?;

        unsafe {
            device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
        }
        Ok((buffer, allocation))
    }

    pub fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        size: u64,
        buffer_usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> RendererResult<Buffer> {
        let (buffer, allocation) =
            Self::allocate_new_buffer(device, allocator, size, buffer_usage, location)?;
        Ok(Buffer {
            device: device.clone(),
            allocation: Some(allocation),
            buffer,
            size,
            buffer_usage,
            location,
        })
    }

    pub fn fill(&mut self, allocator: &mut Allocator, data: &[u8]) -> RendererResult<()> {
        if data.len() != self.size as usize {
            let (buffer, allocation) = Self::allocate_new_buffer(
                &self.device,
                allocator,
                data.len() as u64,
                self.buffer_usage,
                self.location,
            )?;
            let old_allocation = self.allocation.take().expect("Buffer had no allocation!");
            unsafe {
                allocator.free(old_allocation)?;
                self.device.destroy_buffer(self.buffer, None);
            }
            self.buffer = buffer;
            self.allocation = Some(allocation);
            self.size = data.len() as u64;
        }
        if let Some(allocation) = &self.allocation {
            let data_ptr = allocation.mapped_ptr().unwrap().as_ptr() as *mut u8;
            unsafe { data_ptr.copy_from_nonoverlapping(data.as_ptr(), self.size as usize) };
        } else { 
            panic!("Buffer had no allocation!");
        }
        Ok(())
    }

    pub fn destroy(&mut self, allocator: &mut Allocator) {
        allocator.free(self.allocation.take().expect("Buffer had no allocation!")).unwrap();
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
        }
    }
}

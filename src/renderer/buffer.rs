use std::fmt::Debug;
use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, Allocator};
use gpu_allocator::MemoryLocation;

use super::error::InvalidHandle;
use super::utils::{Handle, HandleArray};
use super::RendererResult;

struct InternalBuffer {
    device: ash::Device,
    allocation: Option<Allocation>,
    buffer: vk::Buffer,
    size: u64,
    buffer_usage: vk::BufferUsageFlags,
    location: MemoryLocation,
}

impl Debug for InternalBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("device", &self.device.handle())
            .field("allocation", &self.allocation)
            .field("buffer", &self.buffer)
            .field("size", &self.size)
            .field("buffer_usage", &self.buffer_usage)
            .field("location", &self.location)
            .finish()
    }
}

impl InternalBuffer {
    fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        size: u64,
        buffer_usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> RendererResult<InternalBuffer> {
        let (buffer, allocation) =
            Self::allocate_buffer(device, allocator, size, buffer_usage, location)?;
        Ok(InternalBuffer {
            device: device.clone(),
            allocation: Some(allocation),
            buffer,
            size,
            buffer_usage,
            location,
        })
    }

    fn allocate_buffer(
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
        let allocation = allocator.allocate(&AllocationCreateDesc {
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

    fn fill<T>(&mut self, allocator: &mut Allocator, data: &[T]) -> RendererResult<()> {
        let data_len = data.len() * std::mem::size_of::<T>();
        if data_len > self.size as usize {
            let (buffer, allocation) = Self::allocate_buffer(
                &self.device,
                allocator,
                data_len as u64,
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
            self.size = data_len as u64;
        }
        if let Some(allocation) = &self.allocation {
            let data_ptr = allocation.mapped_ptr().unwrap().as_ptr() as *mut u8;
            unsafe { data_ptr.copy_from_nonoverlapping(data.as_ptr() as *const u8, data_len) };
        } else {
            panic!("Buffer had no allocation!");
        }
        Ok(())
    }

    pub fn copy_to_offset<T>(
        &mut self,
        allocator: &mut Allocator,
        data: &[T],
        offset: usize,
    ) -> RendererResult<()> {
        let data_len = data.len() * std::mem::size_of::<T>();
        if (data_len + offset) > self.size as usize {
            let (buffer, allocation) = Self::allocate_buffer(
                &self.device,
                allocator,
                (data_len + offset) as u64,
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
            self.size = (data_len + offset) as u64;
        }
        if let Some(allocation) = &self.allocation {
            let data_ptr = allocation.mapped_ptr().unwrap().as_ptr() as *mut u8;
            let data_ptr = unsafe { data_ptr.add(offset) };
            unsafe { data_ptr.copy_from_nonoverlapping(data.as_ptr() as *const u8, data_len) };
        } else {
            panic!("Buffer had no allocation!");
        }
        Ok(())
    }

    fn destroy(&mut self, allocator: &mut Allocator) {
        allocator
            .free(self.allocation.take().expect("Buffer had no allocation!"))
            .unwrap();
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
        }
    }
}

#[derive(Debug)]
pub struct BufferManager {
    handle_array: HandleArray<InternalBuffer>,
    to_free: Vec<(InternalBuffer, Option<u32>)>,
}

impl BufferManager {
    pub fn new() -> Arc<Mutex<BufferManager>> {
        Arc::new(Mutex::new(BufferManager {
            handle_array: HandleArray::new(),
            to_free: vec![],
        }))
    }

    fn allocate_new_buffer(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        size: u64,
        buffer_usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> RendererResult<Handle<InternalBuffer>> {
        let internal_buffer = InternalBuffer::new(device, allocator, size, buffer_usage, location)?;
        Ok(self.handle_array.insert(internal_buffer))
    }

    pub fn new_buffer(
        manager: Arc<Mutex<BufferManager>>,
        device: &ash::Device,
        allocator: &mut Allocator,
        size: u64,
        buffer_usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> RendererResult<Buffer> {
        let handle = manager.lock().unwrap().allocate_new_buffer(
            device,
            allocator,
            size,
            buffer_usage,
            location,
        )?;
        let buffer = Buffer {
            manager: manager.clone(),
            handle,
            active: true,
        };
        Ok(buffer)
    }

    fn get_buffer(&self, handle: Handle<InternalBuffer>) -> Option<BufferDetails> {
        self.handle_array.get(handle).map(|int_buf| int_buf.into())
    }

    fn fill_buffer_by_handle<T>(
        &mut self,
        handle: Handle<InternalBuffer>,
        allocator: &mut Allocator,
        data: &[T],
    ) -> RendererResult<()> {
        self.handle_array
            .get_mut(handle)
            .ok_or_else(|| InvalidHandle.into())
            .and_then(|int_buf| int_buf.fill(allocator, data))
    }

    fn copy_to_offset_by_handle<T>(
        &mut self,
        handle: Handle<InternalBuffer>,
        allocator: &mut Allocator,
        data: &[T],
        offset: usize,
    ) -> RendererResult<()> {
        self.handle_array
            .get_mut(handle)
            .ok_or_else(|| InvalidHandle.into())
            .and_then(|int_buf| int_buf.copy_to_offset(allocator, data, offset))
    }

    fn queue_free(
        &mut self,
        handle: Handle<InternalBuffer>,
        last_frame_index: Option<u32>,
    ) -> RendererResult<()> {
        let int_buf = self.handle_array.remove(handle)?;
        self.to_free.push((int_buf, last_frame_index));
        Ok(())
    }

    pub fn free_queued(&mut self, allocator: &mut Allocator, last_frame_index: u32) {
        self.to_free.retain_mut(|(int_buf, i)| {
            if i.is_none() || i.unwrap() == last_frame_index {
                int_buf.destroy(allocator);
                false
            } else {
                true
            }
        });
    }
}

pub struct BufferDetails {
    pub buffer: vk::Buffer,
    pub size: u64,
    pub buffer_usage: vk::BufferUsageFlags,
    pub location: MemoryLocation,
}

impl From<&InternalBuffer> for BufferDetails {
    fn from(ib: &InternalBuffer) -> Self {
        Self {
            buffer: ib.buffer,
            size: ib.size,
            buffer_usage: ib.buffer_usage,
            location: ib.location,
        }
    }
}

#[derive(Debug)]
pub struct Buffer {
    manager: Arc<Mutex<BufferManager>>,
    handle: Handle<InternalBuffer>,
    active: bool,
}

impl Buffer {
    pub fn fill<T>(&mut self, allocator: &mut Allocator, data: &[T]) -> RendererResult<()> {
        if !self.active {
            panic!("Tried to fill inactive buffer!");
        }
        self.manager
            .lock()
            .unwrap()
            .fill_buffer_by_handle(self.handle, allocator, data)
    }

    pub fn copy_to_offset<T>(
        &mut self,
        allocator: &mut Allocator,
        data: &[T],
        offset: usize,
    ) -> RendererResult<()> {
        if !self.active {
            panic!("Tried to copy to inactive buffer!");
        }
        self.manager
            .lock()
            .unwrap()
            .copy_to_offset_by_handle(self.handle, allocator, data, offset)
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    pub fn get_buffer(&self) -> BufferDetails {
        if !self.active {
            panic!("Tried to get inactive buffer!");
        }
        self.manager
            .lock()
            .unwrap()
            .get_buffer(self.handle)
            .unwrap()
    }

    pub fn queue_free(&mut self, last_frame_index: Option<u32>) -> RendererResult<()> {
        if !self.active {
            panic!("Tried to free inactive buffer!");
        }
        self.active = false;
        self.manager
            .lock()
            .unwrap()
            .queue_free(self.handle, last_frame_index)
    }
}

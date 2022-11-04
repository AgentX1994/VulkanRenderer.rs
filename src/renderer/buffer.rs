use std::fmt::Debug;
use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, Allocator};
use gpu_allocator::MemoryLocation;

use super::error::InvalidHandle;
use super::RendererResult;
use super::utils::{HandleArray, Handle};

#[derive(Copy, Clone, Debug, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct BufferHandle(Handle);

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
            unsafe {
                data_ptr.copy_from_nonoverlapping(data.as_ptr() as *const u8, data_len as usize)
            };
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
    to_free: Vec<InternalBuffer>,
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
    ) -> RendererResult<BufferHandle> {
        let internal_buffer = InternalBuffer::new(device, allocator, size, buffer_usage, location)?;
        Ok(BufferHandle(self.handle_array.insert(internal_buffer)))
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

    pub fn get_buffer(&self, handle: BufferHandle) -> Option<BufferDetails> {
        self.handle_array.get(handle.0).map(|int_buf| int_buf.into())
    }

    pub fn fill_buffer_by_handle<T>(
        &mut self,
        handle: BufferHandle,
        allocator: &mut Allocator,
        data: &[T],
    ) -> RendererResult<()> {
        self.handle_array.get_mut(handle.0)
            .ok_or(InvalidHandle.into())
            .and_then(|int_buf| int_buf.fill(allocator, data))
    }

    pub fn queue_free(&mut self, handle: BufferHandle) -> RendererResult<()> {
        let int_buf = self.handle_array.remove(handle.0)?;
        self.to_free.push(int_buf);
        Ok(())
    }

    pub fn free_queued(&mut self, allocator: &mut Allocator) {
        for int_buf in &mut self.to_free {
            int_buf.destroy(allocator);
        }
        self.to_free.clear();
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
    handle: BufferHandle,
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

    pub fn queue_free(&mut self) -> RendererResult<()> {
        if !self.active {
            panic!("Tried to free inactive buffer!");
        }
        self.active = false;
        self.manager.lock().unwrap().queue_free(self.handle)
    }
}

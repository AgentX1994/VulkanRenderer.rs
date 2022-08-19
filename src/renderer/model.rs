use std::collections::HashMap;

use ash::prelude::VkResult;
use ash::vk;
use nalgebra_glm::{Vec2, Vec3};

use gpu_allocator::vulkan::Allocator;
use gpu_allocator::MemoryLocation;

use crate::renderer::Buffer;

use super::vertex::Vertex;
use super::InstanceData;

#[derive(Debug, Clone, Copy)]
struct InvalidHandle;
impl std::fmt::Display for InvalidHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Invalid Handle")
    }
}
impl std::error::Error for InvalidHandle {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

pub struct Model<V, I> {
    vertex_data: Vec<V>,
    index_data: Vec<u32>,
    handle_to_index: HashMap<usize, usize>,
    handles: Vec<usize>,
    instances: Vec<I>,
    first_invisible: usize,
    next_handle: usize,
    pub vertex_buffer: Option<Buffer>,
    pub index_buffer: Option<Buffer>,
    pub instance_buffer: Option<Buffer>,
}

impl<V, I> Model<V, I> {
    pub fn cube() -> Model<Vertex, InstanceData> {
        let lbf = Vertex::new(
            Vec3::new(-1.0, 1.0, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
            Vec2::new(0.5, 0.5),
        ); //lbf: left-bottom-front
        let lbb = Vertex::new(
            Vec3::new(-1.0, 1.0, 1.0),
            Vec3::new(0.5, 0.5, 0.5),
            Vec2::new(0.5, 0.5),
        );
        let ltf = Vertex::new(
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
            Vec2::new(0.5, 0.5),
        );
        let ltb = Vertex::new(
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(0.5, 0.5, 0.5),
            Vec2::new(0.5, 0.5),
        );
        let rbf = Vertex::new(
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
            Vec2::new(0.5, 0.5),
        );
        let rbb = Vertex::new(
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(0.5, 0.5, 0.5),
            Vec2::new(0.5, 0.5),
        );
        let rtf = Vertex::new(
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
            Vec2::new(0.5, 0.5),
        );
        let rtb = Vertex::new(
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(0.5, 0.5, 0.5),
            Vec2::new(0.5, 0.5),
        );
        Model {
            vertex_data: vec![lbf, lbb, ltf, ltb, rbf, rbb, rtf, rtb],
            index_data: vec![
                0, 1, 5, 0, 5, 4, // bottom
                2, 7, 3, 2, 6, 7, // top
                0, 6, 2, 0, 4, 6, // front
                1, 3, 7, 1, 7, 5, // back
                0, 2, 1, 1, 2, 3, // left
                4, 5, 6, 5, 7, 6, // right
            ],
            handle_to_index: std::collections::HashMap::new(),
            handles: Vec::new(),
            instances: Vec::new(),
            first_invisible: 0,
            next_handle: 0,
            vertex_buffer: None,
            index_buffer: None,
            instance_buffer: None,
        }
    }

    pub fn get(&self, handle: usize) -> Option<&I> {
        if let Some(&index) = self.handle_to_index.get(&handle) {
            self.instances.get(index)
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, handle: usize) -> Option<&mut I> {
        if let Some(&index) = self.handle_to_index.get(&handle) {
            self.instances.get_mut(index)
        } else {
            None
        }
    }

    fn swap_by_handle(&mut self, handle1: usize, handle2: usize) -> Result<(), InvalidHandle> {
        if handle1 == handle2 {
            return Ok(());
        }
        if let (Some(&index1), Some(&index2)) = (
            self.handle_to_index.get(&handle1),
            self.handle_to_index.get(&handle2),
        ) {
            self.handles.swap(index1, index2);
            self.instances.swap(index1, index2);
            self.handle_to_index.insert(index1, handle2);
            self.handle_to_index.insert(index2, handle1);
            Ok(())
        } else {
            Err(InvalidHandle)
        }
    }

    fn swap_by_index(&mut self, index1: usize, index2: usize) {
        if index1 == index2 {
            return;
        }
        let handle1 = self.handles[index1];
        let handle2 = self.handles[index2];
        self.handles.swap(index1, index2);
        self.instances.swap(index1, index2);
        self.handle_to_index.insert(index1, handle2);
        self.handle_to_index.insert(index2, handle1);
    }

    fn is_visible(&self, handle: usize) -> Result<bool, InvalidHandle> {
        if let Some(index) = self.handle_to_index.get(&handle) {
            Ok(index < &self.first_invisible)
        } else {
            Err(InvalidHandle)
        }
    }

    fn make_visible(&mut self, handle: usize) -> Result<(), InvalidHandle> {
        if let Some(&index) = self.handle_to_index.get(&handle) {
            if index < self.first_invisible {
                return Ok(());
            }
            self.swap_by_index(index, self.first_invisible);
            self.first_invisible += 1;
            Ok(())
        } else {
            Err(InvalidHandle)
        }
    }

    fn make_invisible(&mut self, handle: usize) -> Result<(), InvalidHandle> {
        if let Some(&index) = self.handle_to_index.get(&handle) {
            if index >= self.first_invisible {
                return Ok(());
            }
            self.swap_by_index(index, self.first_invisible - 1);
            self.first_invisible -= 1;
            Ok(())
        } else {
            Err(InvalidHandle)
        }
    }

    pub fn insert(&mut self, element: I) -> usize {
        let handle = self.next_handle;
        self.next_handle += 1;
        let index = self.instances.len();
        self.instances.push(element);
        self.handles.push(handle);
        self.handle_to_index.insert(handle, index);
        handle
    }

    pub fn insert_visibly(&mut self, element: I) -> usize {
        let new_handle = self.insert(element);
        self.make_visible(new_handle).unwrap();
        new_handle
    }

    fn remove(&mut self, handle: usize) -> Result<I, InvalidHandle> {
        if let Some(&index) = self.handle_to_index.get(&handle) {
            if index < self.first_invisible {
                self.swap_by_index(index, self.first_invisible - 1);
                self.first_invisible -= 1;
            }
            self.swap_by_index(self.first_invisible, self.instances.len() - 1);
            self.handles.pop();
            self.handle_to_index.remove(&handle);
            Ok(self.instances.pop().unwrap())
        } else {
            Err(InvalidHandle)
        }
    }

    pub fn update_vertex_buffer(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
    ) -> VkResult<()> {
        if let Some(buffer) = &mut self.vertex_buffer {
            let bytes = self.vertex_data.len() * std::mem::size_of::<V>();
            let data = unsafe {
                std::slice::from_raw_parts(self.vertex_data.as_ptr() as *const u8, bytes)
            };
            buffer.fill(allocator, data)?;
            Ok(())
        } else {
            let bytes = self.vertex_data.len() * std::mem::size_of::<V>();
            let mut buffer = Buffer::new(
                device,
                allocator,
                bytes as u64,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                MemoryLocation::CpuToGpu,
            )?;
            let data = unsafe {
                std::slice::from_raw_parts(self.vertex_data.as_ptr() as *const u8, bytes)
            };
            buffer.fill(allocator, data)?;
            self.vertex_buffer = Some(buffer);
            Ok(())
        }
    }

    pub fn update_index_buffer(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
    ) -> VkResult<()> {
        if let Some(buffer) = &mut self.index_buffer {
            let bytes = self.vertex_data.len() * std::mem::size_of::<u32>();
            let data =
                unsafe { std::slice::from_raw_parts(self.index_data.as_ptr() as *const u8, bytes) };
            buffer.fill(allocator, data)?;
            Ok(())
        } else {
            let bytes = self.index_data.len() * std::mem::size_of::<V>();
            let mut buffer = Buffer::new(
                device,
                allocator,
                bytes as u64,
                vk::BufferUsageFlags::INDEX_BUFFER,
                MemoryLocation::CpuToGpu,
            )?;
            let data =
                unsafe { std::slice::from_raw_parts(self.index_data.as_ptr() as *const u8, bytes) };
            buffer.fill(allocator, data)?;
            self.index_buffer = Some(buffer);
            Ok(())
        }
    }

    pub fn update_instance_buffer(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
    ) -> VkResult<()> {
        if let Some(buffer) = &mut self.instance_buffer {
            let bytes = self.first_invisible * std::mem::size_of::<I>();
            let data = unsafe {
                std::slice::from_raw_parts(
                    self.instances[0..self.first_invisible].as_ptr() as *const u8,
                    bytes,
                )
            };
            buffer.fill(allocator, data)?;
            Ok(())
        } else {
            let bytes = self.first_invisible * std::mem::size_of::<I>();
            let mut buffer = Buffer::new(
                device,
                allocator,
                bytes as u64,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                MemoryLocation::CpuToGpu,
            )?;
            let data = unsafe {
                std::slice::from_raw_parts(
                    self.instances[0..self.first_invisible].as_ptr() as *const u8,
                    bytes,
                )
            };
            buffer.fill(allocator, data)?;
            self.instance_buffer = Some(buffer);
            Ok(())
        }
    }

    pub fn draw(&self, device: &ash::Device, command_buffer: vk::CommandBuffer) {
        if let Some(vert_buf) = &self.vertex_buffer {
            if let Some(ind_buf) = &self.index_buffer {
                if let Some(inst_buf) = &self.instance_buffer {
                    if self.first_invisible > 0 {
                        unsafe {
                            device.cmd_bind_vertex_buffers(
                                command_buffer,
                                0,
                                &[vert_buf.buffer],
                                &[0],
                            );
                            device.cmd_bind_index_buffer(
                                command_buffer,
                                ind_buf.buffer,
                                0,
                                vk::IndexType::UINT32,
                            );
                            device.cmd_bind_vertex_buffers(
                                command_buffer,
                                1,
                                &[inst_buf.buffer],
                                &[0],
                            );
                            device.cmd_draw_indexed(
                                command_buffer,
                                self.index_data.len() as u32,
                                self.first_invisible as u32,
                                0,
                                0,
                                0,
                            );
                        }
                    }
                }
            }
        }
    }
}

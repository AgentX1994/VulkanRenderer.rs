use std::collections::HashMap;
use std::fmt::Debug;

use ash::vk;
use nalgebra_glm::{Vec2, Vec3};

use gpu_allocator::vulkan::Allocator;
use gpu_allocator::MemoryLocation;

use crate::renderer::Buffer;

use super::vertex::Vertex;
use super::{InstanceData, RendererResult};

#[derive(Debug, Clone, Copy)]
pub struct InvalidHandle;
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

impl<V, I> Debug for Model<V, I>
where
    V: Debug,
    I: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .field("vertex_data", &self.vertex_data)
            .field("index_data", &self.index_data)
            .field("handle_to_index", &self.handle_to_index)
            .field("handles", &self.handles)
            .field("instances", &self.instances)
            .field("first_invisible", &self.first_invisible)
            .field("next_handle", &self.next_handle)
            .field("vertex_buffer", &self.vertex_buffer)
            .field("index_buffer", &self.index_buffer)
            .field("instance_buffer", &self.instance_buffer)
            .finish()
    }
}

impl<V, I> Model<V, I> {
    pub fn cube() -> Model<Vertex, InstanceData> {
        // TODO Fix normals?
        let lbf = Vertex::new(
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec2::new(0.5, 0.5),
        ); //lbf: left-bottom-front
        let lbb = Vertex::new(
            Vec3::new(-1.0, 1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
            Vec2::new(0.5, 0.5),
        );
        let ltf = Vertex::new(
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(-1.0, -1.0, -1.0),
            Vec2::new(0.5, 0.5),
        );
        let ltb = Vertex::new(
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec2::new(0.5, 0.5),
        );
        let rbf = Vertex::new(
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec2::new(0.5, 0.5),
        );
        let rbb = Vertex::new(
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec2::new(0.5, 0.5),
        );
        let rtf = Vertex::new(
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec2::new(0.5, 0.5),
        );
        let rtb = Vertex::new(
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(1.0, -1.0, 1.0),
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
            handle_to_index: Default::default(),
            handles: Vec::new(),
            instances: Vec::new(),
            first_invisible: 0,
            next_handle: 0,
            vertex_buffer: None,
            index_buffer: None,
            instance_buffer: None,
        }
    }

    pub fn icosahedron() -> Model<Vertex, InstanceData> {
        let phi = (1.0 + 5.0_f32.sqrt()) / 2.0;
        let darkgreen_front_top = Vertex::new(
            Vec3::new(phi, -1.0, 0.0),
            Vec3::new(phi, -1.0, 0.0),
            Vec2::new(0.5, 0.5),
        ); //0
        let darkgreen_front_bottom = Vertex::new(
            Vec3::new(phi, 1.0, 0.0),
            Vec3::new(phi, 1.0, 0.0),
            Vec2::new(0.5, 0.5),
        ); //1
        let darkgreen_back_top = Vertex::new(
            Vec3::new(-phi, -1.0, 0.0),
            Vec3::new(-phi, -1.0, 0.0),
            Vec2::new(0.5, 0.5),
        ); //2
        let darkgreen_back_bottom = Vertex::new(
            Vec3::new(-phi, 1.0, 0.0),
            Vec3::new(-phi, 1.0, 0.0),
            Vec2::new(0.5, 0.5),
        ); //3
        let lightgreen_front_right = Vertex::new(
            Vec3::new(1.0, 0.0, -phi),
            Vec3::new(1.0, 0.0, -phi),
            Vec2::new(0.5, 0.5),
        ); //4
        let lightgreen_front_left = Vertex::new(
            Vec3::new(-1.0, 0.0, -phi),
            Vec3::new(-1.0, 0.0, -phi),
            Vec2::new(0.5, 0.5),
        ); //5
        let lightgreen_back_right = Vertex::new(
            Vec3::new(1.0, 0.0, phi),
            Vec3::new(1.0, 0.0, phi),
            Vec2::new(0.5, 0.5),
        ); //6
        let lightgreen_back_left = Vertex::new(
            Vec3::new(-1.0, 0.0, phi),
            Vec3::new(-1.0, 0.0, phi),
            Vec2::new(0.5, 0.5),
        ); //7
        let purple_top_left = Vertex::new(
            Vec3::new(0.0, -phi, -1.0),
            Vec3::new(0.0, -phi, -1.0),
            Vec2::new(0.5, 0.5),
        ); //8
        let purple_top_right = Vertex::new(
            Vec3::new(0.0, -phi, 1.0),
            Vec3::new(0.0, -phi, 1.0),
            Vec2::new(0.5, 0.5),
        ); //9
        let purple_bottom_left = Vertex::new(
            Vec3::new(0.0, phi, -1.0),
            Vec3::new(0.0, phi, -1.0),
            Vec2::new(0.5, 0.5),
        ); //10
        let purple_bottom_right = Vertex::new(
            Vec3::new(0.0, phi, 1.0),
            Vec3::new(0.0, phi, 1.0),
            Vec2::new(0.5, 0.5),
        ); //11
           // calculate texture coords according to: https://stackoverflow.com/q/41957890
        let mut vertex_data = vec![
            darkgreen_front_top,
            darkgreen_front_bottom,
            darkgreen_back_top,
            darkgreen_back_bottom,
            lightgreen_front_right,
            lightgreen_front_left,
            lightgreen_back_right,
            lightgreen_back_left,
            purple_top_left,
            purple_top_right,
            purple_bottom_left,
            purple_bottom_right,
        ];
        for v in &mut vertex_data {
            let norm = v.normal;
            let norm = norm.normalize();
            let theta = (norm.z.atan2(norm.x)) / (2.0 * std::f32::consts::PI) + 0.5;
            let phi = (norm.y.asin() / std::f32::consts::PI) + 0.5;
            v.uv = Vec2::new(theta, phi);
        }
        Model {
            vertex_data,
            index_data: vec![
                0, 9, 8, //
                0, 8, 4, //
                0, 4, 1, //
                0, 1, 6, //
                0, 6, 9, //
                8, 9, 2, //
                8, 2, 5, //
                8, 5, 4, //
                4, 5, 10, //
                4, 10, 1, //
                1, 10, 11, //
                1, 11, 6, //
                2, 3, 5, //
                2, 7, 3, //
                2, 9, 7, //
                5, 3, 10, //
                3, 11, 10, //
                3, 7, 11, //
                6, 7, 9, //
                6, 11, 7, //
            ],
            handle_to_index: Default::default(),
            handles: Default::default(),
            instances: Default::default(),
            first_invisible: 0,
            next_handle: 0,
            vertex_buffer: None,
            index_buffer: None,
            instance_buffer: None,
        }
    }

    pub fn sphere(refinements: u32) -> Model<Vertex, InstanceData> {
        let mut model = Model::<Vertex, InstanceData>::icosahedron();
        for _ in 0..refinements {
            model.subdivide();
        }
        for v in &mut model.vertex_data {
            let pos = v.pos;
            v.pos = pos.normalize();
            let norm = v.normal;
            let norm = norm.normalize();
            let theta = (norm.z.atan2(norm.x)) / (2.0 * std::f32::consts::PI) + 0.5;
            let phi = (norm.y.asin() / std::f32::consts::PI) + 0.5;
            v.uv = Vec2::new(theta, phi);
            v.normal = norm;
        }
        // TODO The UVs are better now, but some triangles wrap around in UV space causing a "zipper"
        // This can be fixed, but I don't feel like it right now.
        // The poles also only have one vertex, which messes with the UVs. This can also be fixed
        // Related blog post: https://mft-dev.dk/uv-mapping-sphere/
        model
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
        // We just inserted this element, we know it exists
        self.make_visible(new_handle).unwrap();
        new_handle
    }

    pub fn update(&mut self, handle: usize, element: I) -> RendererResult<()> {
        let index = self.handle_to_index.get(&handle).ok_or(InvalidHandle)?;
        self.instances[*index] = element;
        Ok(())
    }

    pub fn remove(&mut self, handle: usize) -> RendererResult<I> {
        if let Some(&index) = self.handle_to_index.get(&handle) {
            if index < self.first_invisible {
                self.swap_by_index(index, self.first_invisible - 1);
                self.first_invisible -= 1;
            }
            self.swap_by_index(self.first_invisible, self.instances.len() - 1);
            self.handles.pop();
            self.handle_to_index.remove(&handle);
            // Instances should always have something to pop
            Ok(self.instances.pop().unwrap())
        } else {
            Err(InvalidHandle.into())
        }
    }

    pub fn update_vertex_buffer(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
    ) -> RendererResult<()> {
        if let Some(buffer) = &mut self.vertex_buffer {
            buffer.fill(allocator, &self.vertex_data)?;
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
            buffer.fill(allocator, &self.vertex_data)?;
            self.vertex_buffer = Some(buffer);
            Ok(())
        }
    }

    pub fn update_index_buffer(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
    ) -> RendererResult<()> {
        if let Some(buffer) = &mut self.index_buffer {
            buffer.fill(allocator, &self.index_data)?;
            Ok(())
        } else {
            let bytes = self.index_data.len() * std::mem::size_of::<u32>();
            let mut buffer = Buffer::new(
                device,
                allocator,
                bytes as u64,
                vk::BufferUsageFlags::INDEX_BUFFER,
                MemoryLocation::CpuToGpu,
            )?;
            buffer.fill(allocator, &self.index_data)?;
            self.index_buffer = Some(buffer);
            Ok(())
        }
    }

    pub fn update_instance_buffer(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
    ) -> RendererResult<()> {
        if self.first_invisible == 0 {
            return Ok(());
        }
        if let Some(buffer) = &mut self.instance_buffer {
            buffer.fill(allocator, &self.instances[0..self.first_invisible])?;
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
            buffer.fill(allocator, &self.instances[0..self.first_invisible])?;
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

impl Model<Vertex, InstanceData> {
    pub fn subdivide(&mut self) {
        let mut new_indices = vec![];
        let mut midpoints = HashMap::<(u32, u32), u32>::new();
        for triangle in self.index_data.chunks(3) {
            let a = triangle[0];
            let b = triangle[1];
            let c = triangle[2];
            let vert_a = self.vertex_data[a as usize];
            let vert_b = self.vertex_data[b as usize];
            let vert_c = self.vertex_data[c as usize];

            let mab = if let Some(ab) = midpoints.get(&(a, b)) {
                *ab
            } else {
                let vert_ab = Vertex::midpoint(&vert_a, &vert_b);
                let mab = self.vertex_data.len() as u32;
                self.vertex_data.push(vert_ab);
                midpoints.insert((a, b), mab);
                midpoints.insert((b, a), mab);
                mab
            };

            let mbc = if let Some(bc) = midpoints.get(&(b, c)) {
                *bc
            } else {
                let vert_bc = Vertex::midpoint(&vert_b, &vert_c);
                let mbc = self.vertex_data.len() as u32;
                self.vertex_data.push(vert_bc);
                midpoints.insert((b, c), mbc);
                midpoints.insert((c, b), mbc);
                mbc
            };

            let mca = if let Some(ca) = midpoints.get(&(c, a)) {
                *ca
            } else {
                let vert_ca = Vertex::midpoint(&vert_c, &vert_a);
                let mca = self.vertex_data.len() as u32;
                self.vertex_data.push(vert_ca);
                midpoints.insert((c, a), mca);
                midpoints.insert((a, c), mca);
                mca
            };
            new_indices.extend_from_slice(&[mca, a, mab, mab, b, mbc, mbc, c, mca, mab, mbc, mca]);
        }
        self.index_data = new_indices;
    }
}

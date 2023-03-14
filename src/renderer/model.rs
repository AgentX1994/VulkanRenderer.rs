use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, Mutex};

use ash::vk;
use nalgebra_glm::{Vec2, Vec3};

use gpu_allocator::vulkan::Allocator;
use gpu_allocator::MemoryLocation;

use crate::renderer::Buffer;

use super::buffer::BufferManager;
use super::error::InvalidHandle;
use super::utils::{Handle, HandleArray};
use super::vertex::Vertex;
use super::{InstanceData, RendererResult};

pub mod loaders;

pub struct Model<V, I> {
    vertex_data: Vec<V>,
    index_data: Vec<u32>,
    handle_array: HandleArray<I>,
    first_invisible: usize,
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
            .field("handle_array", &self.handle_array)
            .field("first_invisible", &self.first_invisible)
            .field("vertex_buffer", &self.vertex_buffer)
            .field("index_buffer", &self.index_buffer)
            .field("instance_buffer", &self.instance_buffer)
            .finish()
    }
}

impl<V, I> Model<V, I> {
    pub fn new(vertices: Vec<V>, indices: Vec<u32>) -> Model<V, I> {
        Model {
            vertex_data: vertices,
            index_data: indices,
            handle_array: Default::default(),
            first_invisible: 0,
            vertex_buffer: None,
            index_buffer: None,
            instance_buffer: None,
        }
    }

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
        Model::new(
            vec![lbf, lbb, ltf, ltb, rbf, rbb, rtf, rtb],
            vec![
                0, 1, 5, 0, 5, 4, // bottom
                2, 7, 3, 2, 6, 7, // top
                0, 6, 2, 0, 4, 6, // front
                1, 3, 7, 1, 7, 5, // back
                0, 2, 1, 1, 2, 3, // left
                4, 5, 6, 5, 7, 6, // right
            ],
        )
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
        Model::new(
            vertex_data,
            vec![
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
        )
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

    pub fn get(&self, handle: Handle<I>) -> Option<&I> {
        self.handle_array.get(handle)
    }

    pub fn get_mut(&mut self, handle: Handle<I>) -> Option<&mut I> {
        self.handle_array.get_mut(handle)
    }

    fn is_visible(&self, handle: Handle<I>) -> Result<bool, InvalidHandle> {
        if let Some(index) = self.handle_array.get_index(handle) {
            Ok(index < self.first_invisible)
        } else {
            Err(InvalidHandle)
        }
    }

    fn make_visible(&mut self, handle: Handle<I>) -> Result<(), InvalidHandle> {
        if let Some(index) = self.handle_array.get_index(handle) {
            if index < self.first_invisible {
                return Ok(());
            }
            self.handle_array.swap_by_index(index, self.first_invisible);
            self.first_invisible += 1;
            Ok(())
        } else {
            Err(InvalidHandle)
        }
    }

    fn make_invisible(&mut self, handle: Handle<I>) -> Result<(), InvalidHandle> {
        if let Some(index) = self.handle_array.get_index(handle) {
            if index >= self.first_invisible {
                return Ok(());
            }
            self.handle_array
                .swap_by_index(index, self.first_invisible - 1);
            self.first_invisible -= 1;
            Ok(())
        } else {
            Err(InvalidHandle)
        }
    }

    pub fn insert(&mut self, element: I) -> Handle<I> {
        self.handle_array.insert(element)
    }

    pub fn insert_visibly(&mut self, element: I) -> Handle<I> {
        let new_handle = self.insert(element);
        // We just inserted this element, we know it exists
        self.make_visible(new_handle).unwrap();
        new_handle
    }

    pub fn update(&mut self, handle: Handle<I>, element: I) -> RendererResult<()> {
        let elem = self.handle_array.get_mut(handle).ok_or(InvalidHandle)?;
        *elem = element;
        Ok(())
    }

    pub fn remove(&mut self, handle: Handle<I>) -> RendererResult<I> {
        self.handle_array.remove(handle)
    }

    pub fn update_vertex_buffer(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        buffer_manager: Arc<Mutex<BufferManager>>,
    ) -> RendererResult<()> {
        if let Some(buffer) = &mut self.vertex_buffer {
            buffer.fill(allocator, &self.vertex_data)?;
            Ok(())
        } else {
            let bytes = self.vertex_data.len() * std::mem::size_of::<V>();
            let mut buffer = BufferManager::new_buffer(
                buffer_manager,
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
        buffer_manager: Arc<Mutex<BufferManager>>,
    ) -> RendererResult<()> {
        if let Some(buffer) = &mut self.index_buffer {
            buffer.fill(allocator, &self.index_data)?;
            Ok(())
        } else {
            let bytes = self.index_data.len() * std::mem::size_of::<u32>();
            let mut buffer = BufferManager::new_buffer(
                buffer_manager,
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
        buffer_manager: Arc<Mutex<BufferManager>>,
    ) -> RendererResult<()> {
        if self.first_invisible == 0 {
            return Ok(());
        }
        if let Some(buffer) = &mut self.instance_buffer {
            buffer.fill(allocator, self.handle_array.get_data())?;
            Ok(())
        } else {
            let bytes = self.first_invisible * std::mem::size_of::<I>();
            let mut buffer = BufferManager::new_buffer(
                buffer_manager,
                device,
                allocator,
                bytes as u64,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                MemoryLocation::CpuToGpu,
            )?;
            buffer.fill(allocator, self.handle_array.get_data())?;
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
                            let vert_buf_int = vert_buf.get_buffer();
                            let ind_buf_int = ind_buf.get_buffer();
                            let inst_buf_int = inst_buf.get_buffer();
                            device.cmd_bind_vertex_buffers(
                                command_buffer,
                                0,
                                &[vert_buf_int.buffer],
                                &[0],
                            );
                            device.cmd_bind_index_buffer(
                                command_buffer,
                                ind_buf_int.buffer,
                                0,
                                vk::IndexType::UINT32,
                            );
                            device.cmd_bind_vertex_buffers(
                                command_buffer,
                                1,
                                &[inst_buf_int.buffer],
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

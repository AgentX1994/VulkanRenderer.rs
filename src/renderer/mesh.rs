use std::collections::HashMap;
use std::fmt::Debug;
use std::path::Path;
use std::sync::{Arc, Mutex};

use ash::vk;
use nalgebra_glm::{Vec2, Vec3};

use gpu_allocator::vulkan::Allocator;
use gpu_allocator::MemoryLocation;

use crate::renderer::Buffer;

use super::buffer::BufferManager;
use super::utils::{Handle, HandleArray};
use super::vertex::Vertex;
use super::RendererResult;

pub mod loaders;

#[derive(Debug)]
pub struct Mesh {
    vertex_data: Vec<Vertex>,
    index_data: Vec<u32>,
    pub vertex_buffer: Option<Buffer>,
    pub index_buffer: Option<Buffer>,
}

impl Mesh {
    fn new(vertices: Vec<Vertex>, indices: Vec<u32>) -> Mesh {
        Mesh {
            vertex_data: vertices,
            index_data: indices,
            vertex_buffer: None,
            index_buffer: None,
        }
    }

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

    fn cube() -> Mesh {
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
        Mesh::new(
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

    fn icosahedron() -> Mesh {
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
        Mesh::new(
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

    fn sphere(refinements: u32) -> Mesh {
        let mut model = Mesh::icosahedron();
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
            let bytes = self.vertex_data.len() * std::mem::size_of::<Vertex>();
            let mut buffer = BufferManager::new_buffer(
                buffer_manager,
                device,
                allocator,
                bytes as u64,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                MemoryLocation::CpuToGpu,
                "vertex-buffer",
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
                "index-buffer",
            )?;
            buffer.fill(allocator, &self.index_data)?;
            self.index_buffer = Some(buffer);
            Ok(())
        }
    }

    pub fn draw(&self, device: &ash::Device, command_buffer: vk::CommandBuffer) {
        if let Some(vert_buf) = &self.vertex_buffer {
            if let Some(ind_buf) = &self.index_buffer {
                unsafe {
                    let vert_buf_int = vert_buf.get_buffer();
                    let ind_buf_int = ind_buf.get_buffer();
                    device.cmd_bind_vertex_buffers(command_buffer, 0, &[vert_buf_int.buffer], &[0]);
                    device.cmd_bind_index_buffer(
                        command_buffer,
                        ind_buf_int.buffer,
                        0,
                        vk::IndexType::UINT32,
                    );
                    device.cmd_draw_indexed(
                        command_buffer,
                        self.index_data.len() as u32,
                        1,
                        0,
                        0,
                        0,
                    );
                }
            }
        }
    }
}

impl Drop for Mesh {
    fn drop(&mut self) {
        if let Some(buf) = &mut self.vertex_buffer {
            buf.queue_free(None).expect("Could not free buffer");
        }
        if let Some(buf) = &mut self.index_buffer {
            buf.queue_free(None).expect("Could not free buffer");
        }
    }
}

#[derive(Debug, Default)]
pub struct MeshManager {
    meshs: HandleArray<Mesh>,
}

impl MeshManager {
    fn add_mesh(
        &mut self,
        mut mesh: Mesh,
        device: &ash::Device,
        allocator: &mut Allocator,
        buffer_manager: Arc<Mutex<BufferManager>>,
    ) -> RendererResult<Handle<Mesh>> {
        mesh.update_vertex_buffer(device, allocator, buffer_manager.clone())?;
        mesh.update_index_buffer(device, allocator, buffer_manager)?;
        Ok(self.meshs.insert(mesh))
    }

    pub fn new_mesh(
        &mut self,
        vertices: Vec<Vertex>,
        indices: Vec<u32>,
        device: &ash::Device,
        allocator: &mut Allocator,
        buffer_manager: Arc<Mutex<BufferManager>>,
    ) -> RendererResult<Handle<Mesh>> {
        let mesh = Mesh::new(vertices, indices);
        self.add_mesh(mesh, device, allocator, buffer_manager)
    }

    pub fn new_cube_mesh(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        buffer_manager: Arc<Mutex<BufferManager>>,
    ) -> RendererResult<Handle<Mesh>> {
        let mesh = Mesh::cube();
        self.add_mesh(mesh, device, allocator, buffer_manager)
    }

    pub fn new_icosahedron_mesh(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        buffer_manager: Arc<Mutex<BufferManager>>,
    ) -> RendererResult<Handle<Mesh>> {
        let mesh = Mesh::icosahedron();
        self.add_mesh(mesh, device, allocator, buffer_manager)
    }

    pub fn new_sphere_mesh(
        &mut self,
        refinements: u32,
        device: &ash::Device,
        allocator: &mut Allocator,
        buffer_manager: Arc<Mutex<BufferManager>>,
    ) -> RendererResult<Handle<Mesh>> {
        let mesh = Mesh::sphere(refinements);
        self.add_mesh(mesh, device, allocator, buffer_manager)
    }

    pub fn new_mesh_from_obj<P: AsRef<Path>>(
        &mut self,
        path: P,
        device: &ash::Device,
        allocator: &mut Allocator,
        buffer_manager: Arc<Mutex<BufferManager>>,
    ) -> RendererResult<Handle<Mesh>> {
        let mesh = loaders::obj::load_obj(path)?;
        self.add_mesh(mesh, device, allocator, buffer_manager)
    }

    pub fn get_mesh(&self, handle: Handle<Mesh>) -> Option<&Mesh> {
        self.meshs.get(handle)
    }

    pub fn get_mesh_mut(&mut self, handle: Handle<Mesh>) -> Option<&mut Mesh> {
        self.meshs.get_mut(handle)
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Mesh> {
        self.meshs.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, Mesh> {
        self.meshs.iter_mut()
    }

    pub fn destroy(&mut self) {
        self.meshs.clear();
    }
}

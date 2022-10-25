use ash::{vk, Device};
use gpu_allocator::vulkan::Allocator;
use nalgebra as na;
use nalgebra_glm as glm;

use super::{buffer::Buffer, RendererResult};

#[derive(Debug)]
pub struct DirectionalLight {
    pub direction: na::Unit<glm::Vec3>,
    pub illuminance: glm::Vec3, // in lx = lm / m^2
}

#[derive(Debug, Default)]
pub struct PointLight {
    pub position: na::Point3<f32>, // in m
    pub luminous_flux: glm::Vec3,  // in lm
}

pub enum Light {
    Directional(DirectionalLight),
    Point(PointLight),
}

impl From<PointLight> for Light {
    fn from(p: PointLight) -> Self {
        Light::Point(p)
    }
}

impl From<DirectionalLight> for Light {
    fn from(d: DirectionalLight) -> Self {
        Light::Directional(d)
    }
}

#[derive(Debug, Default)]
pub struct LightManager {
    directional_lights: Vec<DirectionalLight>,
    point_lights: Vec<PointLight>,
}

impl LightManager {
    pub fn add_light<L: Into<Light>>(&mut self, l: L) {
        match l.into() {
            Light::Directional(dl) => {
                self.directional_lights.push(dl);
            }
            Light::Point(pl) => {
                self.point_lights.push(pl);
            }
        }
    }

    pub fn update_buffer(
        &self,
        device: &Device,
        allocator: &mut Allocator,
        buffer: &mut Buffer,
        descriptor_sets_lights: &mut [vk::DescriptorSet],
    ) -> RendererResult<()> {
        // 0.0s are for padding
        let mut data_vec: Vec<f32> = vec![
            self.directional_lights.len() as f32,
            self.point_lights.len() as f32,
            0.0,
            0.0,
        ];

        for dl in &self.directional_lights {
            data_vec.push(dl.direction.x);
            data_vec.push(dl.direction.y);
            data_vec.push(dl.direction.z);
            data_vec.push(0.0); // Padding
            data_vec.push(dl.illuminance.x);
            data_vec.push(dl.illuminance.y);
            data_vec.push(dl.illuminance.z);
            data_vec.push(0.0); // Padding
        }
        for pl in &self.point_lights {
            data_vec.push(pl.position.x);
            data_vec.push(pl.position.y);
            data_vec.push(pl.position.z);
            data_vec.push(0.0); // Padding
            data_vec.push(pl.luminous_flux.x);
            data_vec.push(pl.luminous_flux.y);
            data_vec.push(pl.luminous_flux.z);
            data_vec.push(0.0); // Padding
        }
        buffer.fill(allocator, &data_vec)?;
        for ds in descriptor_sets_lights {
            let int_buf = buffer.get_buffer();
            let buffer_infos = [vk::DescriptorBufferInfo {
                buffer: int_buf.buffer,
                offset: 0,
                range: int_buf.size,
            }];
            let desc_sets_write = [vk::WriteDescriptorSet::builder()
                .dst_set(*ds)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buffer_infos)
                .build()];
            unsafe { device.update_descriptor_sets(&desc_sets_write, &[]) };
        }

        Ok(())
    }
}

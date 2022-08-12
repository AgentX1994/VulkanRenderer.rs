use ash::vk;
use memoffset::offset_of;
use nalgebra_glm::{Vec2, Vec3};

use super::InstanceData;

#[repr(C, packed)]
#[derive(Debug, Default, Copy, Clone)]
pub struct Vertex {
    pub pos: Vec3,
    pub color: Vec3,
    pub uv: Vec2,
}

impl Vertex {
    pub fn new(pos: Vec3, color: Vec3, uv: Vec2) -> Self {
        Vertex { pos, color, uv }
    }

    pub fn get_binding_description() -> [vk::VertexInputBindingDescription; 2] {
        [
            vk::VertexInputBindingDescription::builder()
                .binding(0)
                .stride(std::mem::size_of::<Vertex>() as u32)
                .input_rate(vk::VertexInputRate::VERTEX)
                .build(),
            vk::VertexInputBindingDescription::builder()
                .binding(1)
                .stride(std::mem::size_of::<InstanceData>() as u32)
                .input_rate(vk::VertexInputRate::INSTANCE)
                .build(),
        ]
    }

    pub fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 8] {
        [
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: offset_of!(Vertex, pos) as u32,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: offset_of!(Vertex, color) as u32,
            },
            vk::VertexInputAttributeDescription {
                location: 2,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: offset_of!(Vertex, uv) as u32,
            },
            vk::VertexInputAttributeDescription {
                location: 3,
                binding: 1,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 0u32,
            },
            vk::VertexInputAttributeDescription {
                location: 4,
                binding: 1,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 16u32,
            },
            vk::VertexInputAttributeDescription {
                location: 5,
                binding: 1,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 32u32,
            },
            vk::VertexInputAttributeDescription {
                location: 6,
                binding: 1,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 48u32,
            },
            vk::VertexInputAttributeDescription {
                location: 7,
                binding: 1,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 64u32,
            },
        ]
    }
}

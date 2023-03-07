use std::hash::Hash;

use ash::vk;
use memoffset::offset_of;
use nalgebra_glm::{Vec2, Vec3};

use super::InstanceData;

#[repr(C, packed)]
#[derive(Debug, Default, Copy, Clone)]
pub struct Vertex {
    pub pos: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
}

impl Vertex {
    pub fn new(pos: Vec3, normal: Vec3, uv: Vec2) -> Self {
        Vertex { pos, normal, uv }
    }

    pub fn midpoint(a: &Vertex, b: &Vertex) -> Self {
        Vertex {
            pos: 0.5 * (a.pos + b.pos),
            normal: 0.5 * (a.normal + b.normal),
            uv: 0.5 * (a.uv + b.uv),
        }
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

    pub fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 14] {
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
                offset: offset_of!(Vertex, normal) as u32,
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
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 64u32,
            },
            vk::VertexInputAttributeDescription {
                location: 8,
                binding: 1,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 80u32,
            },
            vk::VertexInputAttributeDescription {
                location: 9,
                binding: 1,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 96u32,
            },
            vk::VertexInputAttributeDescription {
                location: 10,
                binding: 1,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 112u32,
            },
            vk::VertexInputAttributeDescription {
                location: 11,
                binding: 1,
                format: vk::Format::R32_SFLOAT,
                offset: 128u32,
            },
            vk::VertexInputAttributeDescription {
                location: 12,
                binding: 1,
                format: vk::Format::R32_SFLOAT,
                offset: 132u32,
            },
            vk::VertexInputAttributeDescription {
                location: 13,
                binding: 1,
                format: vk::Format::R8_UINT,
                offset: 136u32,
            },
        ]
    }
}

impl PartialEq for Vertex {
    fn eq(&self, other: &Self) -> bool {
        // I want this equality to be bit-wise, rather than float wise
        // Unfortunately, unaligned access is not allowed, so I can't just
        // call to_bits() on each float, so I have to do this monstrosity instead
        // This should work since the structs are packed
        let self_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const u8,
                std::mem::size_of::<Vertex>(),
            )
        };
        let other_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                other as *const Self as *const u8,
                std::mem::size_of::<Vertex>(),
            )
        };
        self_bytes == other_bytes
    }
}

impl Eq for Vertex {}

impl Hash for Vertex {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let self_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const u8,
                std::mem::size_of::<Vertex>(),
            )
        };
        self_bytes.hash(state);
    }
}

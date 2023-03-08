use std::collections::HashMap;

use ash::vk;

use super::{
    pipeline::GraphicsPipeline, shader_module::ShaderModule, text::TextVertexData, vertex::Vertex,
    RendererResult,
};

pub struct Material {
    pub pipeline: GraphicsPipeline,
}

pub struct MaterialStorage {
    materials: HashMap<String, Material>,
}

impl MaterialStorage {
    pub fn new(device: &ash::Device, render_pass: vk::RenderPass) -> RendererResult<Self> {
        // Create default materials
        let mut materials = HashMap::new();
        // default material
        {
            let shader_module = ShaderModule::new(
                device,
                vk_shader_macros::include_glsl!("./shaders/default.vert", kind: vert),
                vk_shader_macros::include_glsl!("./shaders/default.frag", kind: frag),
            )?;
            // TODO don't hardcode the number of textures
            let graphics_pipeline = GraphicsPipeline::new(
                device,
                &render_pass,
                shader_module.get_stages(),
                &Vertex::get_attribute_descriptions(),
                &Vertex::get_binding_description(),
                4,
            )?;

            shader_module.destroy();

            materials.insert(
                "default".to_string(),
                Material {
                    pipeline: graphics_pipeline,
                },
            );
        }

        // text material
        {
            let shader_module = ShaderModule::new(
                device,
                vk_shader_macros::include_glsl!("./shaders/text.vert", kind: vert),
                vk_shader_macros::include_glsl!("./shaders/text.frag", kind: frag),
            )?;

            let graphics_pipeline = GraphicsPipeline::new_text(
                device,
                &render_pass,
                shader_module.get_stages(),
                &TextVertexData::get_vertex_attributes(),
                &TextVertexData::get_vertex_bindings(),
            )?;

            shader_module.destroy();

            materials.insert(
                "text".to_string(),
                Material {
                    pipeline: graphics_pipeline,
                },
            );
        }

        Ok(Self { materials })
    }

    pub fn get_material<S: AsRef<str>>(&self, shader: S) -> Option<&Material> {
        self.materials.get(shader.as_ref())
    }

    pub fn destroy(&mut self) {
        for (_, mut material) in self.materials.drain() {
            material.pipeline.destroy();
        }
    }
}

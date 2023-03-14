use std::{
    collections::HashMap,
    hash::Hash,
    ops::{Index, IndexMut},
};

use ash::vk;

use super::{
    descriptor::{DescriptorAllocator, DescriptorBuilder, DescriptorLayoutCache},
    error::{InvalidHandle, RendererError},
    shaders::{ShaderCache, ShaderEffect},
    text::TextVertexData,
    texture::{Texture, TextureStorage},
    utils::{Handle, HandleArray},
    vertex::Vertex,
    RendererResult,
};

// TODO move this somewhere
pub enum MeshPassType {
    None,
    Forward,
    Transparency,
    DirectionalShadow,
}

// TODO move this somewhere
pub enum TransparencyMode {
    Opaque,
    Transparent,
    Masked,
}

// TODO move this
#[derive(Clone, Default)]
pub struct VertexInputDescription {
    pub bindings: Vec<vk::VertexInputBindingDescription>,
    pub attributes: Vec<vk::VertexInputAttributeDescription>,
    pub flags: vk::PipelineVertexInputStateCreateFlags,
}

#[derive(Clone, Default)]
pub struct PipelineBuilder {
    shader_stages: Vec<vk::PipelineShaderStageCreateInfo>,
    vertex_description: VertexInputDescription,
    vertex_input_info: vk::PipelineVertexInputStateCreateInfo,
    input_assembly: vk::PipelineInputAssemblyStateCreateInfo,
    viewport: vk::Viewport,
    scissor: vk::Rect2D,
    rasterizer: vk::PipelineRasterizationStateCreateInfo,
    color_blend_attachment: vk::PipelineColorBlendAttachmentState,
    multisampling: vk::PipelineMultisampleStateCreateInfo,
    pipeline_layout: vk::PipelineLayout,
    depth_stencil: vk::PipelineDepthStencilStateCreateInfo,
}

impl PipelineBuilder {
    pub fn build_pipeline(
        &self,
        device: &ash::Device,
        render_pass: vk::RenderPass,
    ) -> RendererResult<vk::Pipeline> {
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_attribute_descriptions(&self.vertex_description.attributes[..])
            .vertex_binding_descriptions(&self.vertex_description.bindings[..])
            .flags(self.vertex_description.flags);

        let viewports = [self.viewport];
        let scissors = [self.scissor];

        let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&viewports)
            .scissors(&scissors);

        let attachments = [self.color_blend_attachment];

        let color_blend_info = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&attachments);

        let dynamic_state_create_info = vk::PipelineDynamicStateCreateInfo::builder()
            .dynamic_states(&[vk::DynamicState::SCISSOR, vk::DynamicState::VIEWPORT]);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&self.shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&self.input_assembly)
            .viewport_state(&viewport_info)
            .rasterization_state(&self.rasterizer)
            .multisample_state(&self.multisampling)
            .depth_stencil_state(&self.depth_stencil)
            .color_blend_state(&color_blend_info)
            .layout(self.pipeline_layout)
            .render_pass(render_pass)
            .dynamic_state(&dynamic_state_create_info)
            .subpass(0);

        unsafe {
            device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[pipeline_info.build()],
                    None,
                )
                .map_err(|(_pipelines, err)| {
                    // TODO delete created pipelines on error?
                    RendererError::VulkanError(err)
                })
                .map(|vec| vec[0])
        }
    }

    pub fn clear_vertex_input(&mut self) {
        // TODO is there a better way to clear these?
        self.vertex_input_info.p_vertex_attribute_descriptions = std::ptr::null();
        self.vertex_input_info.vertex_attribute_description_count = 0;

        self.vertex_input_info.p_vertex_binding_descriptions = std::ptr::null();
        self.vertex_input_info.vertex_binding_description_count = 0;
    }

    pub fn set_shaders(
        &mut self,
        shader_cache: &ShaderCache,
        effect: &ShaderEffect,
    ) -> RendererResult<()> {
        self.shader_stages = effect.get_stages(shader_cache)?;
        self.pipeline_layout = effect.pipeline_layout;
        Ok(())
    }
}

pub enum VertexAttributeTemplate {
    DefaultVertex,
    DefaultVertexPosOnly,
}

pub struct EffectBuilder {
    vertex_attribute_template: VertexAttributeTemplate,

    effect: ShaderEffect,

    topology: vk::PrimitiveTopology,
    rasterizer_info: vk::PipelineRasterizationStateCreateInfo,
    color_blend_attachment_info: vk::PipelineColorBlendAttachmentState,
    depth_stencil_info: vk::PipelineDepthStencilStateCreateInfo,
}

pub struct ComputePipelineBuilder {
    shader_stage: vk::PipelineShaderStageCreateInfo,
    pipeline_layout: vk::PipelineLayout,
}

impl ComputePipelineBuilder {
    pub fn build_pipeline(&self, device: &ash::Device) -> RendererResult<vk::Pipeline> {
        let create_info = vk::ComputePipelineCreateInfo::builder()
            .stage(self.shader_stage)
            .layout(self.pipeline_layout);

        let pipelines = unsafe {
            device
                .create_compute_pipelines(vk::PipelineCache::null(), &[*create_info], None)
                .map_err(|(_, err)| RendererError::VulkanError(err))?
        };
        Ok(pipelines[0])
    }
}

#[derive(Default)]
pub struct BuiltShaderPass {
    pub effect_handle: Option<Handle<ShaderEffect>>,
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
}

pub struct BuiltPerPassData<T> {
    data: [T; 3],
}

impl<T: Default> Default for BuiltPerPassData<T> {
    fn default() -> Self {
        Self {
            data: Default::default(),
        }
    }
}

impl<T> Index<MeshPassType> for BuiltPerPassData<T> {
    type Output = T;

    fn index(&self, index: MeshPassType) -> &Self::Output {
        match index {
            MeshPassType::None => panic!("Tried to index with MeshPassType::None!"),
            MeshPassType::Forward => &self.data[0],
            MeshPassType::Transparency => &self.data[1],
            MeshPassType::DirectionalShadow => &self.data[2],
        }
    }
}

impl<T> IndexMut<MeshPassType> for BuiltPerPassData<T> {
    fn index_mut(&mut self, index: MeshPassType) -> &mut Self::Output {
        match index {
            MeshPassType::None => panic!("Tried to index with MeshPassType::None!"),
            MeshPassType::Forward => &mut self.data[0],
            MeshPassType::Transparency => &mut self.data[1],
            MeshPassType::DirectionalShadow => &mut self.data[2],
        }
    }
}

#[derive(Default, Clone, Copy, Hash, PartialEq)]
pub struct ShaderParameters {}

pub struct EffectTemplate {
    pub pass_shaders: BuiltPerPassData<BuiltShaderPass>,
    pub default_parameters: ShaderParameters,
    pub transparency_mode: TransparencyMode,
}

impl EffectTemplate {
    fn destroy(&mut self, device: &ash::Device) {
        for sp in self.pass_shaders.data.iter() {
            unsafe {
                device.destroy_pipeline(sp.pipeline, None);
                // The pipeline layout is owned by the corresponding ShaderEffect
            }
        }
    }
}

#[derive(Clone)]
pub struct MaterialData {
    pub textures: Vec<Handle<Texture>>,
    pub parameters: ShaderParameters,
    pub base_template: String,
}

impl PartialEq for MaterialData {
    fn eq(&self, other: &Self) -> bool {
        if self.base_template != other.base_template
            || self.parameters != other.parameters
            || self.textures.len() != other.textures.len()
        {
            return false;
        }
        !self
            .textures
            .iter()
            .zip(other.textures.iter())
            .any(|(a, b)| a != b)
    }
}

impl Eq for MaterialData {}

impl Hash for MaterialData {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.base_template.hash(state);

        for tex in self.textures.iter() {
            tex.hash(state);
        }
        self.parameters.hash(state);
    }
}

pub struct Material {
    pub original: Handle<EffectTemplate>,
    pub pass_sets: BuiltPerPassData<vk::DescriptorSet>,
    pub textures: Vec<Handle<Texture>>,
    pub parameters: ShaderParameters,
}

fn build_shader_pass(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    shader_cache: &ShaderCache,
    builder: &PipelineBuilder,
    effect_handle: Handle<ShaderEffect>,
) -> RendererResult<BuiltShaderPass> {
    let effect = shader_cache.get_shader_effect_by_handle(effect_handle)?;
    let layout = effect.pipeline_layout;
    let mut builder = builder.clone();
    builder.set_shaders(shader_cache, effect)?;
    let pipeline = builder.build_pipeline(device, render_pass)?;
    Ok(BuiltShaderPass {
        effect_handle: Some(effect_handle),
        pipeline,
        layout,
    })
}

pub struct MaterialSystem {
    forward_builder: PipelineBuilder,
    text_builder: PipelineBuilder,
    shadow_builder: PipelineBuilder,

    effect_template_handles: HandleArray<EffectTemplate>,
    template_cache: HashMap<String, Handle<EffectTemplate>>,

    materials_handles: HandleArray<Material>,
    materials: HashMap<String, Handle<Material>>,
    material_cache: HashMap<MaterialData, Handle<Material>>,
}

impl MaterialSystem {
    pub fn new(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        shader_cache: &mut ShaderCache,
    ) -> RendererResult<Self> {
        let mut ret = Self {
            forward_builder: Default::default(),
            text_builder: Default::default(),
            shadow_builder: Default::default(),
            effect_template_handles: HandleArray::new(),
            template_cache: HashMap::new(),
            materials_handles: HandleArray::new(),
            materials: HashMap::new(),
            material_cache: HashMap::new(),
        };
        ret.build_default_templates(device, render_pass, shader_cache)?;
        Ok(ret)
    }

    fn build_default_templates(
        &mut self,
        device: &ash::Device,
        render_pass: vk::RenderPass,
        shader_cache: &mut ShaderCache,
    ) -> RendererResult<()> {
        self.fill_builders();

        let default_effect_handle = shader_cache.build_effect(
            device,
            "./shaders/default.vert",
            Some("./shaders/default.frag"),
        )?;
        let text_effect_handle = shader_cache.build_effect(
            device,
            "./shaders/text.vert",
            Some("./shaders/text.frag"),
        )?;

        let default_pass = build_shader_pass(
            device,
            render_pass,
            shader_cache,
            &self.forward_builder,
            default_effect_handle,
        )?;

        let text_pass = build_shader_pass(
            device,
            render_pass,
            shader_cache,
            &self.text_builder,
            text_effect_handle,
        )?;

        {
            let mut default_template = EffectTemplate {
                pass_shaders: Default::default(),
                default_parameters: ShaderParameters {},
                transparency_mode: TransparencyMode::Opaque,
            };

            default_template.pass_shaders[MeshPassType::Forward] = default_pass;
            let handle = self.effect_template_handles.insert(default_template);
            self.template_cache.insert("default".to_string(), handle);
        }

        {
            let mut text_template = EffectTemplate {
                pass_shaders: Default::default(),
                default_parameters: ShaderParameters {},
                transparency_mode: TransparencyMode::Opaque,
            };

            text_template.pass_shaders[MeshPassType::Forward] = text_pass;
            let handle = self.effect_template_handles.insert(text_template);
            self.template_cache.insert("text".to_string(), handle);
        }

        Ok(())
    }

    pub fn build_material(
        &mut self,
        device: &ash::Device,
        texture_storage: &TextureStorage,
        descriptor_layout_cache: &mut DescriptorLayoutCache,
        descriptor_allocator: &mut DescriptorAllocator,
        material_name: &str,
        info: MaterialData,
    ) -> RendererResult<Handle<Material>> {
        match self.material_cache.entry(info) {
            std::collections::hash_map::Entry::Occupied(o) => Ok(*o.get()),
            std::collections::hash_map::Entry::Vacant(v) => {
                let info = v.key();
                let original = {
                    let res = self.template_cache.get(&info.base_template);
                    match res {
                        Some(handle) => *handle,
                        None => {
                            return Err(RendererError::MissingTemplate(info.base_template.clone()))
                        }
                    }
                };
                let mut new_mat = Material {
                    original,
                    pass_sets: Default::default(),
                    textures: info.textures.clone(),
                    parameters: info.parameters,
                };

                let mut db =
                    DescriptorBuilder::begin(descriptor_layout_cache, descriptor_allocator);

                let mut image_infos = vec![];
                image_infos.reserve(info.textures.len());
                for (i, tex_handle) in info.textures.iter().enumerate() {
                    let tex = texture_storage
                        .get_texture(*tex_handle)
                        .expect("Invalid handle");
                    let image_info = [vk::DescriptorImageInfo::builder()
                        .sampler(tex.sampler)
                        .image_view(tex.image_view)
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .build()];
                    image_infos.push(image_info);
                    db.bind_image(
                        i as u32,
                        image_infos.last().unwrap(),
                        vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        vk::ShaderStageFlags::FRAGMENT,
                    );
                }

                new_mat.pass_sets[MeshPassType::Forward] = db.build(device)?.0;

                let handle = self.materials_handles.insert(new_mat);
                self.materials.insert(material_name.to_string(), handle);
                Ok(*v.insert(handle))
            }
        }
    }

    pub fn get_material_handle<S: AsRef<str>>(
        &self,
        material_name: S,
    ) -> RendererResult<Handle<Material>> {
        match self.materials.get(material_name.as_ref()) {
            Some(handle) => Ok(*handle),
            None => Err(RendererError::InvalidHandle(InvalidHandle)),
        }
    }

    pub fn get_material_by_handle(&self, handle: Handle<Material>) -> RendererResult<&Material> {
        self.materials_handles
            .get(handle)
            .ok_or(RendererError::InvalidHandle(InvalidHandle))
    }

    pub fn get_effect_template_handle<S: AsRef<str>>(
        &self,
        template_name: S,
    ) -> RendererResult<Handle<EffectTemplate>> {
        match self.template_cache.get(template_name.as_ref()) {
            Some(handle) => Ok(*handle),
            None => Err(RendererError::InvalidHandle(InvalidHandle)),
        }
    }

    pub fn get_effect_template_by_handle(
        &self,
        handle: Handle<EffectTemplate>,
    ) -> RendererResult<&EffectTemplate> {
        self.effect_template_handles
            .get(handle)
            .ok_or(RendererError::InvalidHandle(InvalidHandle))
    }

    pub fn fill_builders(&mut self) {
        {
            self.shadow_builder.vertex_description = Vertex::get_vertex_description();
            self.shadow_builder.input_assembly =
                vk::PipelineInputAssemblyStateCreateInfo::builder()
                    .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                    .primitive_restart_enable(false)
                    .build();
            self.shadow_builder.rasterizer = vk::PipelineRasterizationStateCreateInfo::builder()
                .depth_clamp_enable(false)
                .rasterizer_discard_enable(false)
                .polygon_mode(vk::PolygonMode::FILL)
                .line_width(1.0)
                .cull_mode(vk::CullModeFlags::FRONT)
                .front_face(vk::FrontFace::CLOCKWISE)
                .depth_bias_enable(true)
                .depth_bias_constant_factor(0.0)
                .depth_bias_clamp(0.0)
                .depth_bias_slope_factor(0.0)
                .build();
            self.shadow_builder.multisampling = vk::PipelineMultisampleStateCreateInfo::builder()
                .sample_shading_enable(false)
                .rasterization_samples(vk::SampleCountFlags::TYPE_1)
                .min_sample_shading(1.0)
                .alpha_to_coverage_enable(false)
                .alpha_to_one_enable(false)
                .build();
            self.shadow_builder.color_blend_attachment =
                vk::PipelineColorBlendAttachmentState::builder()
                    .color_write_mask(vk::ColorComponentFlags::RGBA)
                    .blend_enable(true)
                    .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                    .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                    .color_blend_op(vk::BlendOp::ADD)
                    .src_alpha_blend_factor(vk::BlendFactor::SRC_ALPHA)
                    .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                    .alpha_blend_op(vk::BlendOp::ADD)
                    .build();
            self.shadow_builder.depth_stencil = vk::PipelineDepthStencilStateCreateInfo::builder()
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::LESS)
                .depth_bounds_test_enable(false)
                .min_depth_bounds(0.0)
                .max_depth_bounds(1.0)
                .stencil_test_enable(false)
                .build();
        }
        {
            self.text_builder.vertex_description = TextVertexData::get_vertex_description();
            self.text_builder.input_assembly = vk::PipelineInputAssemblyStateCreateInfo::builder()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                .primitive_restart_enable(false)
                .build();
            self.text_builder.rasterizer = vk::PipelineRasterizationStateCreateInfo::builder()
                .depth_clamp_enable(false)
                .rasterizer_discard_enable(false)
                .polygon_mode(vk::PolygonMode::FILL)
                .line_width(1.0)
                .cull_mode(vk::CullModeFlags::FRONT)
                .front_face(vk::FrontFace::CLOCKWISE)
                .depth_bias_enable(true)
                .depth_bias_constant_factor(0.0)
                .depth_bias_clamp(0.0)
                .depth_bias_slope_factor(0.0)
                .build();
            self.text_builder.multisampling = vk::PipelineMultisampleStateCreateInfo::builder()
                .sample_shading_enable(false)
                .rasterization_samples(vk::SampleCountFlags::TYPE_1)
                .min_sample_shading(1.0)
                .alpha_to_coverage_enable(false)
                .alpha_to_one_enable(false)
                .build();
            self.text_builder.color_blend_attachment =
                vk::PipelineColorBlendAttachmentState::builder()
                    .color_write_mask(vk::ColorComponentFlags::RGBA)
                    .blend_enable(true)
                    .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                    .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                    .color_blend_op(vk::BlendOp::ADD)
                    .src_alpha_blend_factor(vk::BlendFactor::SRC_ALPHA)
                    .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                    .alpha_blend_op(vk::BlendOp::ADD)
                    .build();
            self.text_builder.depth_stencil = vk::PipelineDepthStencilStateCreateInfo::builder()
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
                .depth_bounds_test_enable(false)
                .min_depth_bounds(0.0)
                .max_depth_bounds(1.0)
                .stencil_test_enable(false)
                .build();
        }
        {
            self.forward_builder.vertex_description = Vertex::get_vertex_description();
            self.forward_builder.input_assembly =
                vk::PipelineInputAssemblyStateCreateInfo::builder()
                    .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                    .primitive_restart_enable(false)
                    .build();
            self.forward_builder.rasterizer = vk::PipelineRasterizationStateCreateInfo::builder()
                .depth_clamp_enable(false)
                .rasterizer_discard_enable(false)
                .polygon_mode(vk::PolygonMode::FILL)
                .line_width(1.0)
                .cull_mode(vk::CullModeFlags::NONE)
                .front_face(vk::FrontFace::CLOCKWISE)
                .depth_bias_enable(false)
                .depth_bias_constant_factor(0.0)
                .depth_bias_clamp(0.0)
                .depth_bias_slope_factor(0.0)
                .build();
            self.forward_builder.multisampling = vk::PipelineMultisampleStateCreateInfo::builder()
                .sample_shading_enable(false)
                .rasterization_samples(vk::SampleCountFlags::TYPE_1)
                .min_sample_shading(1.0)
                .alpha_to_coverage_enable(false)
                .alpha_to_one_enable(false)
                .build();
            self.forward_builder.color_blend_attachment =
                vk::PipelineColorBlendAttachmentState::builder()
                    .color_write_mask(vk::ColorComponentFlags::RGBA)
                    .blend_enable(true)
                    .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                    .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                    .color_blend_op(vk::BlendOp::ADD)
                    .src_alpha_blend_factor(vk::BlendFactor::SRC_ALPHA)
                    .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                    .alpha_blend_op(vk::BlendOp::ADD)
                    .build();
            self.forward_builder.depth_stencil = vk::PipelineDepthStencilStateCreateInfo::builder()
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
                .depth_bounds_test_enable(false)
                .min_depth_bounds(0.0)
                .max_depth_bounds(1.0)
                .stencil_test_enable(false)
                .build();
        }
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        self.template_cache.clear();
        for effect_template in self.effect_template_handles.iter_mut() {
            effect_template.destroy(device);
        }
        self.effect_template_handles.clear();
        self.materials.clear();
        self.material_cache.clear();
        self.materials_handles.clear();
    }
}

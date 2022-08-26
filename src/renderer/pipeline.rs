use ash::prelude::VkResult;
use ash::vk;
use ash::Device;

pub struct GraphicsPipeline {
    device: Device,
    pub pipeline_layout: vk::PipelineLayout,
    pub descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    pub pipeline: vk::Pipeline,
}

impl GraphicsPipeline {
    pub fn new(
        device: &Device,
        extent: vk::Extent2D,
        render_pass: &vk::RenderPass,
        shader_stages: &[vk::PipelineShaderStageCreateInfo],
        vertex_attrib_descs: &[vk::VertexInputAttributeDescription],
        vertex_binding_descs: &[vk::VertexInputBindingDescription],
    ) -> VkResult<GraphicsPipeline> {
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_attribute_descriptions(vertex_attrib_descs)
            .vertex_binding_descriptions(vertex_binding_descs);

        let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewports = [vk::Viewport {
            x: 0.,
            y: 0.,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.,
            max_depth: 1.,
        }];
        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        }];

        let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&viewports)
            .scissors(&scissors);

        let rasterizer_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .line_width(1.0)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .cull_mode(vk::CullModeFlags::BACK)
            .polygon_mode(vk::PolygonMode::FILL);

        let multisampling_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::builder()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .build()];
        let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);
        let color_blend_info =
            vk::PipelineColorBlendStateCreateInfo::builder().attachments(&color_blend_attachments);

        // Create the descriptor set layout for camera
        let descriptor_set_layout_binding_descriptions_camera =
            [vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX)
                .build()];

        let descriptor_set_layout_info_camera = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&descriptor_set_layout_binding_descriptions_camera);
        let descriptor_set_layout_camera = unsafe {
            device.create_descriptor_set_layout(&descriptor_set_layout_info_camera, None)?
        };

        // Create the descriptor set layout for lights
        let descriptor_set_layout_binding_descriptions_lights =
            [vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .build()];

        let descriptor_set_layout_info_lights = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&descriptor_set_layout_binding_descriptions_lights);
        let descriptor_set_layout_lights = unsafe {
            device.create_descriptor_set_layout(&descriptor_set_layout_info_lights, None)?
        };

        let descriptor_set_layouts =
            vec![descriptor_set_layout_camera, descriptor_set_layout_lights];

        // Create the pipeline layout
        let pipeline_layout_info =
            vk::PipelineLayoutCreateInfo::builder().set_layouts(&descriptor_set_layouts);

        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&pipeline_layout_info, None)? };

        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly_info)
            .viewport_state(&viewport_info)
            .rasterization_state(&rasterizer_info)
            .multisample_state(&multisampling_info)
            .depth_stencil_state(&depth_stencil_info)
            .color_blend_state(&color_blend_info)
            .layout(pipeline_layout)
            .render_pass(*render_pass)
            .subpass(0);

        let graphics_pipeline = unsafe {
            device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[pipeline_info.build()],
                    None,
                )
                .map_err(|(_pipelines, err)| {
                    // TODO delete created pipelines on error?
                    err
                })?[0]
        };

        Ok(GraphicsPipeline {
            device: device.clone(),
            pipeline_layout,
            descriptor_set_layouts,
            pipeline: graphics_pipeline,
        })
    }

    pub fn destroy(&mut self) {
        unsafe {
            for dsl in &self.descriptor_set_layouts {
                self.device.destroy_descriptor_set_layout(*dsl, None);
            }
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}

impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        //self.destroy();
    }
}

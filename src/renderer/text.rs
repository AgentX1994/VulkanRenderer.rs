use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;

use ash::vk;
use ash::Device;

use gpu_allocator::vulkan::AllocationCreateDesc;
use gpu_allocator::vulkan::{Allocation, Allocator};
use gpu_allocator::MemoryLocation;
use memoffset::offset_of;

use super::buffer::Buffer;
use super::buffer::BufferManager;
use super::context::VulkanContext;
use super::error::InvalidHandle;
use super::error::RendererError;
use super::pipeline::GraphicsPipeline;
use super::shader_module::ShaderModule;
use super::swapchain::Swapchain;
use super::RendererResult;

pub struct TextTexture {
    image: vk::Image,
    pub image_view: vk::ImageView,
    pub sampler: vk::Sampler,
    allocation: Option<Allocation>,
}

impl TextTexture {
    pub fn from_u8s(
        data: &[u8],
        width: u32,
        height: u32,
        device: &Device,
        allocator: &mut Allocator,
        buffer_manager: Arc<Mutex<BufferManager>>,
        command_pool: &vk::CommandPool,
        queue: &vk::Queue,
    ) -> RendererResult<Self> {
        // Create Image
        let img_create_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(vk::Format::R8_SRGB)
            .samples(vk::SampleCountFlags::TYPE_1)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED);
        let image = unsafe { device.create_image(&img_create_info, None) }?;

        //  allocate memory for image
        let reqs = unsafe { device.get_image_memory_requirements(image) };
        let allocation = allocator.allocate(&AllocationCreateDesc {
            name: "text texture",
            requirements: reqs,
            location: MemoryLocation::GpuOnly,
            linear: false,
        })?;

        // bind memory to image
        unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset()) }?;

        // Create image view
        let view_create_info = vk::ImageViewCreateInfo::builder()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8_SRGB)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                level_count: 1,
                layer_count: 1,
                ..Default::default()
            });
        let image_view = unsafe { device.create_image_view(&view_create_info, None) }?;

        // Create sampler
        let sampler_info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR);
        let sampler = unsafe { device.create_sampler(&sampler_info, None) }?;

        // Create buffer and fill with data
        let mut buffer = BufferManager::new_buffer(
            buffer_manager,
            device,
            allocator,
            data.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        )?;
        buffer.fill(allocator, data)?;

        // Create command buffer for copy commands
        let command_buf_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(*command_pool)
            .command_buffer_count(1);
        let copy_buf = unsafe { device.allocate_command_buffers(&command_buf_allocate_info) }?[0];

        // begin command buffer
        let cmd_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { device.begin_command_buffer(copy_buf, &cmd_begin_info) }?;

        // Transition image layout to transfer dst
        let barrier = vk::ImageMemoryBarrier::builder()
            .image(image)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .build();
        unsafe {
            device.cmd_pipeline_barrier(
                copy_buf,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            )
        };

        // Copy buffer to image
        let image_subresource = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };
        let region = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            image_extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            image_subresource,
        };
        unsafe {
            let int_buf = buffer.get_buffer();
            device.cmd_copy_buffer_to_image(
                copy_buf,
                int_buf.buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            )
        }

        // Transition image layout for use as texture
        let barrier = vk::ImageMemoryBarrier::builder()
            .image(image)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .build();
        unsafe {
            device.cmd_pipeline_barrier(
                copy_buf,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            )
        };

        // End command buffer
        unsafe { device.end_command_buffer(copy_buf) }?;

        // Prepare to submit command buffer
        let submit_infos = [vk::SubmitInfo::builder()
            .command_buffers(&[copy_buf])
            .build()];
        // Fence to wait for command buffer to finish
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }?;

        // Submit the commands and wait for completion
        unsafe { device.queue_submit(*queue, &submit_infos, fence) }?;
        unsafe { device.wait_for_fences(&[fence], true, std::u64::MAX) }?;

        // Cleanup
        unsafe { device.destroy_fence(fence, None) };
        buffer.queue_free()?;
        unsafe { device.free_command_buffers(*command_pool, &[copy_buf]) };

        // Done
        Ok(TextTexture {
            image,
            image_view,
            sampler,
            allocation: Some(allocation),
        })
    }

    pub fn destroy(&mut self, device: &Device, allocator: &mut Allocator) {
        allocator
            .free(
                self.allocation
                    .take()
                    .expect("Text texture had no allocation!"),
            )
            .expect("Could not free texture allocation");
        unsafe {
            device.destroy_sampler(self.sampler, None);
            device.destroy_image_view(self.image_view, None);
            device.destroy_image(self.image, None);
        }
    }
}

pub struct Letter {
    color: [f32; 3],
    position_and_shape: fontdue::layout::GlyphPosition,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct TextVertexData {
    pub position: [f32; 3],
    pub texture_coordinates: [f32; 2],
    pub color: [f32; 3],
    pub texture_id: u32,
}

impl TextVertexData {
    pub fn get_vertex_attributes() -> [vk::VertexInputAttributeDescription; 4] {
        [
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                offset: offset_of!(TextVertexData, position) as u32,
                format: vk::Format::R32G32B32_SFLOAT,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 1,
                offset: offset_of!(TextVertexData, texture_coordinates) as u32,
                format: vk::Format::R32G32_SFLOAT,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 2,
                offset: offset_of!(TextVertexData, color) as u32,
                format: vk::Format::R32G32B32_SFLOAT,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 3,
                offset: offset_of!(TextVertexData, texture_id) as u32,
                format: vk::Format::R8G8B8A8_UINT,
            },
        ]
    }

    pub fn get_vertex_bindings() -> [vk::VertexInputBindingDescription; 1] {
        [vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<TextVertexData>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }]
    }
}

pub struct TextHandler {
    vertex_data: HashMap<usize, Vec<TextVertexData>>,
    vertex_buffer: Option<Buffer>,
    textures: Vec<TextTexture>,
    texture_ids: HashMap<fontdue::layout::GlyphRasterConfig, u32>,
    fonts: Vec<fontdue::Font>,
    pipeline: Option<GraphicsPipeline>,
    shader_module: ShaderModule,
    descriptor_pool: Option<vk::DescriptorPool>,
    descriptor_sets: Vec<vk::DescriptorSet>,
    number_of_textures: usize,
}

impl TextHandler {
    pub fn new<P: AsRef<std::path::Path>>(
        device: &Device,
        standard_font: P,
    ) -> RendererResult<TextHandler> {
        let shader_module = ShaderModule::new(
            device,
            vk_shader_macros::include_glsl!("./shaders/text.vert", kind: vert),
            vk_shader_macros::include_glsl!("./shaders/text.frag", kind: frag),
        )?;

        let mut text_handler = TextHandler {
            vertex_data: HashMap::new(),
            vertex_buffer: None,
            textures: vec![],
            texture_ids: HashMap::new(),
            fonts: vec![],
            shader_module,
            pipeline: None,
            descriptor_pool: None,
            descriptor_sets: vec![],
            number_of_textures: 0,
        };

        text_handler.load_font(standard_font)?;
        Ok(text_handler)
    }

    pub fn load_font<P: AsRef<Path>>(&mut self, path: P) -> RendererResult<usize> {
        let font_data = std::fs::read(path)?;
        let font = fontdue::Font::from_bytes(font_data, fontdue::FontSettings::default())?;
        let index = self.fonts.len();
        self.fonts.push(font);
        Ok(index)
    }

    pub fn create_letters(
        &self,
        styles: &[&fontdue::layout::TextStyle],
        color: [f32; 3],
    ) -> Vec<Letter> {
        let mut layout =
            fontdue::layout::Layout::new(fontdue::layout::CoordinateSystem::PositiveYUp);
        let settings = fontdue::layout::LayoutSettings {
            ..fontdue::layout::LayoutSettings::default()
        };
        layout.reset(&settings);
        for style in styles {
            layout.append(&self.fonts, style);
        }
        let output = layout.glyphs();
        output
            .iter()
            .map(|&glyph| Letter {
                color,
                position_and_shape: glyph,
            })
            .collect()
    }

    pub fn new_texture_from_u8s(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        device: &Device,
        allocator: &mut Allocator,
        buffer_manager: Arc<Mutex<BufferManager>>,
        command_pool: &vk::CommandPool,
        queue: &vk::Queue,
    ) -> RendererResult<usize> {
        let new_texture = TextTexture::from_u8s(
            data,
            width,
            height,
            device,
            allocator,
            buffer_manager,
            command_pool,
            queue,
        )?;
        let new_id = self.textures.len();
        self.textures.push(new_texture);
        Ok(new_id)
    }

    pub fn create_vertex_data(
        &mut self,
        letters: Vec<Letter>,
        position: (u32, u32), // in pixels
        window: &winit::window::Window,
        device: &Device,
        allocator: &mut Allocator,
        buffer_manager: Arc<Mutex<BufferManager>>,
        render_pass: &vk::RenderPass,
        command_pool: &vk::CommandPool,
        queue: &vk::Queue,
        swapchain: &Swapchain,
    ) -> RendererResult<usize> {
        let screen_size = window.inner_size();
        let mut need_texture_update = false;
        let mut vertex_data = Vec::with_capacity(6 * letters.len());
        for l in letters {
            let id_option = self.texture_ids.get(&l.position_and_shape.key);
            let id = if let Some(id) = id_option {
                *id
            } else {
                let (metrics, bitmap) = self.fonts[l.position_and_shape.font_index]
                    .rasterize(l.position_and_shape.parent, l.position_and_shape.key.px);
                if bitmap.is_empty() {
                    continue;
                }
                need_texture_update = true;
                let id = self.new_texture_from_u8s(
                    &bitmap,
                    metrics.width as u32,
                    metrics.height as u32,
                    device,
                    allocator,
                    buffer_manager.clone(),
                    command_pool,
                    queue,
                )? as u32;
                self.texture_ids.insert(l.position_and_shape.key, id);
                id
            };

            let left =
                2.0 * (l.position_and_shape.x + position.0 as f32) / screen_size.width as f32 - 1.0;
            let right = 2.0
                * (l.position_and_shape.x + position.0 as f32 + l.position_and_shape.width as f32)
                / screen_size.width as f32
                - 1.0;
            let top = 2.0
                * (-l.position_and_shape.y + position.1 as f32
                    - l.position_and_shape.height as f32)
                / screen_size.height as f32
                - 1.0;
            let bottom = 2.0 * (-l.position_and_shape.y + position.1 as f32)
                / screen_size.height as f32
                - 1.0;
            let v1 = TextVertexData {
                position: [left, top, 0.0],
                texture_coordinates: [0.0, 0.0],
                color: l.color,
                texture_id: id,
            };
            let v2 = TextVertexData {
                position: [left, bottom, 0.0],
                texture_coordinates: [0.0, 1.0],
                color: l.color,
                texture_id: id,
            };
            let v3 = TextVertexData {
                position: [right, top, 0.0],
                texture_coordinates: [1.0, 0.0],
                color: l.color,
                texture_id: id,
            };
            let v4 = TextVertexData {
                position: [right, bottom, 0.0],
                texture_coordinates: [1.0, 1.0],
                color: l.color,
                texture_id: id,
            };
            vertex_data.push(v1);
            vertex_data.push(v2);
            vertex_data.push(v3);
            vertex_data.push(v3);
            vertex_data.push(v2);
            vertex_data.push(v4);
        }
        let id: usize = rand::random();
        self.vertex_data.insert(id, vertex_data);
        if need_texture_update {
            self.update_textures(render_pass, swapchain, device)?;
        }
        Ok(id)
    }

    pub fn remove_text_by_id(
        &mut self,
        context: &VulkanContext,
        allocator: &mut Allocator,
        buffer_manager: Arc<Mutex<BufferManager>>,
        id: usize,
    ) -> RendererResult<()> {
        if self.vertex_data.remove(&id).is_some() {
            self.update_vertex_buffer(&context.device, allocator, buffer_manager)?;
            Ok(())
        } else {
            Err(RendererError::InvalidHandle(InvalidHandle))
        }
    }

    pub fn update_textures(
        &mut self,
        render_pass: &vk::RenderPass,
        swapchain: &Swapchain,
        device: &Device,
    ) -> RendererResult<()> {
        let amount = self.textures.len();
        if amount > self.number_of_textures {
            self.number_of_textures = amount;
            self.clear_pipeline()
        }
        if self.pipeline.is_none() {
            self.pipeline = Some(GraphicsPipeline::new_text(
                device,
                swapchain.get_extent(),
                render_pass,
                self.shader_module.get_stages(),
                &TextVertexData::get_vertex_attributes(),
                &TextVertexData::get_vertex_bindings(),
                amount as u32,
            )?);
        }
        if let Some(pool) = self.descriptor_pool {
            unsafe {
                device.destroy_descriptor_pool(pool, None);
            }
        }
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: GraphicsPipeline::MAXIMUM_NUMBER_OF_TEXTURES
                * swapchain.get_actual_image_count(),
        }];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(swapchain.get_actual_image_count())
            .pool_sizes(&pool_sizes);
        let descriptor_pool =
            unsafe { device.create_descriptor_pool(&descriptor_pool_info, None) }?;
        self.descriptor_pool = Some(descriptor_pool);

        let desc_layouts_text = vec![
            // This pipeline has to exist, so we know this will succeed
            self.pipeline.as_ref().unwrap().descriptor_set_layouts[0];
            swapchain.get_actual_image_count() as usize
        ];
        let descriptor_counts_text =
            vec![amount as u32; swapchain.get_actual_image_count() as usize];
        let mut variable_descriptor_allocate_info_text =
            vk::DescriptorSetVariableDescriptorCountAllocateInfo::builder()
                .descriptor_counts(&descriptor_counts_text);
        let descriptor_set_allocate_info_text = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&desc_layouts_text)
            .push_next(&mut variable_descriptor_allocate_info_text);
        let descriptor_sets_text =
            unsafe { device.allocate_descriptor_sets(&descriptor_set_allocate_info_text) }?;
        for i in 0..swapchain.get_actual_image_count() {
            let image_infos: Vec<vk::DescriptorImageInfo> = self
                .textures
                .iter()
                .map(|t| vk::DescriptorImageInfo {
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    image_view: t.image_view,
                    sampler: t.sampler,
                })
                .collect();
            let descriptor_write_image = vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_sets_text[i as usize])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&image_infos)
                .build();
            unsafe {
                device.update_descriptor_sets(&[descriptor_write_image], &[]);
            }
        }
        self.descriptor_sets = descriptor_sets_text;
        Ok(())
    }

    pub fn update_vertex_buffer(
        &mut self,
        device: &Device,
        allocator: &mut Allocator,
        buffer_manager: Arc<Mutex<BufferManager>>,
    ) -> RendererResult<()> {
        if self.vertex_data.is_empty() {
            return Ok(());
        }
        // This is probably pretty slow but I am lazy
        // TODO find a better text handling solution
        let mut vert_data = vec![];
        for verts in self.vertex_data.values() {
            vert_data.extend_from_slice(verts);
        }
        if let Some(buffer) = &mut self.vertex_buffer {
            buffer.fill(allocator, &vert_data)?;
            Ok(())
        } else {
            let bytes = (vert_data.len() * std::mem::size_of::<TextVertexData>()) as u64;
            let mut buffer = BufferManager::new_buffer(
                buffer_manager,
                device,
                allocator,
                bytes,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                MemoryLocation::CpuToGpu,
            )?;
            buffer.fill(allocator, &vert_data)?;
            self.vertex_buffer = Some(buffer);
            Ok(())
        }
    }

    pub fn draw(&self, device: &Device, cmd_buf: vk::CommandBuffer, index: usize) {
        if let Some(pipeline) = &self.pipeline {
            if self.descriptor_sets.len() > index {
                unsafe {
                    device.cmd_bind_pipeline(
                        cmd_buf,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.pipeline,
                    );

                    device.cmd_bind_descriptor_sets(
                        cmd_buf,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.pipeline_layout,
                        0,
                        &[self.descriptor_sets[index]],
                        &[],
                    );
                }
                if let Some(buf) = &self.vertex_buffer {
                    unsafe {
                        let int_buf = buf.get_buffer();
                        device.cmd_bind_vertex_buffers(cmd_buf, 0, &[int_buf.buffer], &[0]);
                        device.cmd_draw(
                            cmd_buf,
                            self.vertex_data.values().map(|v| v.len()).sum::<usize>() as u32,
                            1, // instance count
                            0,
                            0,
                        );
                    }
                }
            }
        }
    }

    pub fn clear_pipeline(&mut self) {
        if let Some(mut pip) = self.pipeline.take() {
            pip.destroy();
        }
    }

    pub fn destroy(&mut self, device: &Device, allocator: &mut Allocator) {
        self.clear_pipeline();
        if let Some(pool) = self.descriptor_pool.take() {
            unsafe {
                device.destroy_descriptor_pool(pool, None);
            }
        }

        if let Some(mut b) = self.vertex_buffer.take() {
            b.queue_free().expect("Invalid Buffer!?");
        }
        for t in &mut self.textures {
            t.destroy(device, allocator);
        }
        self.shader_module.destroy();
    }
}

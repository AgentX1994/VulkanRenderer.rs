use std::collections::HashMap;
use std::collections::HashSet;
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
use super::error::InvalidHandle;
use super::error::RendererError;
use super::pipeline::GraphicsPipeline;
use super::shader_module::ShaderModule;
use super::swapchain::Swapchain;
use super::RendererResult;

struct CharacterData {
    cur_x: usize,
    cur_y: usize,
    _advance_width: f32,
    _advance_height: f32,
    width: usize,
    height: usize,
    _left: f32,
    _top: f32,
    texture_x: f32,
    texture_y: f32,
}

struct TextAtlasTexture {
    width: f32,
    height: f32,
    image: vk::Image,
    image_view: vk::ImageView,
    sampler: vk::Sampler,
    allocation: Option<Allocation>,
    char_data: HashMap<u16, CharacterData>,
}

impl TextAtlasTexture {
    pub fn from_u8s(
        data: &[u8],
        width: u32,
        height: u32,
        char_data: HashMap<u16, CharacterData>,
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
        println!(
            "Creating Texture atlas of size {}x{}, {} bytes",
            width, height, reqs.size
        );
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
        buffer.queue_free(None)?;
        unsafe { device.free_command_buffers(*command_pool, &[copy_buf]) };

        // Done
        Ok(TextAtlasTexture {
            width: width as f32,
            height: height as f32,
            image,
            image_view,
            sampler,
            allocation: Some(allocation),
            char_data,
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
}

impl TextVertexData {
    pub fn get_vertex_attributes() -> [vk::VertexInputAttributeDescription; 3] {
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

struct TextBuffer {
    px: f32,
    last_image_index: Option<u32>,
    vertex_buffer: Buffer,
    vertex_data: Vec<TextVertexData>,
}

impl TextBuffer {
    fn new(
        px: f32,
        vertex_data: Vec<TextVertexData>,
        device: &Device,
        allocator: &mut Allocator,
        buffer_manager: Arc<Mutex<BufferManager>>,
    ) -> RendererResult<Self> {
        if vertex_data.is_empty() {
            // TODO handle this?
            panic!("Given empty vertex data");
        }
        let bytes = (vertex_data.len() * std::mem::size_of::<TextVertexData>()) as u64;
        let mut vertex_buffer = BufferManager::new_buffer(
            buffer_manager,
            device,
            allocator,
            bytes,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            MemoryLocation::CpuToGpu,
        )?;
        vertex_buffer.fill(allocator, &vertex_data)?;
        Ok(Self {
            px,
            last_image_index: None,
            vertex_buffer,
            vertex_data,
        })
    }

    fn destroy(&mut self) {
        self.vertex_buffer
            .queue_free(self.last_image_index)
            .expect("Invalid Buffer!?");
    }
}

pub struct TextHandler {
    vertex_data: HashMap<usize, TextBuffer>,
    font: fontdue::Font,
    atlases: Vec<(f32, TextAtlasTexture)>,
    pipeline: Option<GraphicsPipeline>,
    shader_module: ShaderModule,
    descriptor_pool: Option<vk::DescriptorPool>,
    // TODO deal with the fact that px is f32
    descriptor_sets: HashMap<u32, Vec<vk::DescriptorSet>>,
}

impl TextHandler {
    pub fn new<P: AsRef<std::path::Path>>(
        device: &Device,
        font_path: P,
    ) -> RendererResult<TextHandler> {
        let shader_module = ShaderModule::new(
            device,
            vk_shader_macros::include_glsl!("./shaders/text.vert", kind: vert),
            vk_shader_macros::include_glsl!("./shaders/text.frag", kind: frag),
        )?;

        let font_data = std::fs::read(font_path)?;
        let font = fontdue::Font::from_bytes(font_data, fontdue::FontSettings::default())?;

        Ok(TextHandler {
            vertex_data: HashMap::new(),
            font,
            atlases: vec![],
            shader_module,
            pipeline: None,
            descriptor_pool: None,
            descriptor_sets: HashMap::new(),
        })
    }

    fn generate_texture_atlas(
        &mut self,
        px: f32,
        max_extent: &vk::Extent3D,
        device: &Device,
        allocator: &mut Allocator,
        buffer_manager: Arc<Mutex<BufferManager>>,
        command_pool: &vk::CommandPool,
        queue: &vk::Queue,
    ) -> RendererResult<TextAtlasTexture> {
        let mut char_data = HashMap::new();
        let mut width = max_extent.width as usize;
        let mut height = max_extent.height as usize;
        let mut char_list_with_metrics: Vec<_> = self
            .font
            .chars()
            .iter()
            .map(|(c, i)| {
                let metrics = self.font.metrics_indexed((*i).into(), px);
                (*c, *i, metrics)
            })
            .collect();

        char_list_with_metrics.sort_by(|(_c_l, _i_l, metrics_l), (_c_r, _i_r, metrics_r)| {
            metrics_r.height.cmp(&metrics_l.height)
        });

        let mut cur_x = 0usize;
        let mut cur_y = 0usize;
        let mut tallest_this_row = 0usize;
        let mut max_width = 0usize;
        let mut max_height = 0usize;
        for (_c, i, metrics) in char_list_with_metrics.iter() {
            if cur_x + metrics.width > width {
                cur_x = 0;
                cur_y += tallest_this_row;
                tallest_this_row = metrics.height;
            }
            let character_data = CharacterData {
                cur_x,
                cur_y,
                _advance_width: metrics.advance_width,
                _advance_height: metrics.advance_height,
                width: metrics.width,
                height: metrics.height,
                _left: metrics.bounds.xmin,
                _top: metrics.bounds.ymin,
                texture_x: 0f32, // These are calculated after we determine the max extent of the atlas
                texture_y: 0f32, // These are calculated after we determine the max extent of the atlas
            };
            char_data.insert((*i).into(), character_data);
            cur_x += metrics.width;
            tallest_this_row = std::cmp::max(tallest_this_row, metrics.height);
            max_width = std::cmp::max(max_width, cur_x);
            max_height = std::cmp::max(max_height, cur_y + metrics.height);
        }

        let mut data = vec![0; max_width * max_height];
        for (i, character_data) in char_data.iter_mut() {
            let (metrics, glyph_data) = self.font.rasterize_indexed(*i, px);
            character_data.texture_x = character_data.cur_x as f32 / max_width as f32;
            character_data.texture_y = character_data.cur_y as f32 / max_width as f32;
            for y in 0..metrics.height {
                for x in 0..metrics.width {
                    data[character_data.cur_x + x + (character_data.cur_y + y) * max_width] = glyph_data[x + y * metrics.width];
                }
            }
        }

        let atlas = TextAtlasTexture::from_u8s(
            &data,
            max_width as u32,
            max_height as u32,
            char_data,
            device,
            allocator,
            buffer_manager,
            command_pool,
            queue,
        )?;

        Ok(atlas)
    }

    pub fn create_letters(
        &mut self,
        styles: &[&fontdue::layout::TextStyle],
        color: [f32; 3],
        max_extent: &vk::Extent3D,
        device: &Device,
        allocator: &mut Allocator,
        buffer_manager: Arc<Mutex<BufferManager>>,
        command_pool: &vk::CommandPool,
        queue: &vk::Queue,
    ) -> RendererResult<(Vec<Letter>, bool)> {
        let mut added_atlas = false;
        let mut layout =
            fontdue::layout::Layout::new(fontdue::layout::CoordinateSystem::PositiveYUp);
        let settings = fontdue::layout::LayoutSettings {
            ..fontdue::layout::LayoutSettings::default()
        };
        layout.reset(&settings);
        for style in styles {
            layout.append(&[&self.font], style);
            if self
                .atlases
                .iter()
                .find(|(px, _)| *px == style.px)
                .is_none()
            {
                let atlas = self.generate_texture_atlas(
                    style.px,
                    max_extent,
                    device,
                    allocator,
                    buffer_manager.clone(),
                    command_pool,
                    queue,
                )?;
                self.atlases.push((style.px, atlas));
                added_atlas = true;
            }
        }
        let mut output = vec![];
        for glyph in layout.glyphs() {
            output.push(Letter {
                color,
                position_and_shape: *glyph,
            });
        }
        Ok((output, added_atlas))
    }

    pub fn add_text(
        &mut self,
        styles: &[&fontdue::layout::TextStyle],
        color: [f32; 3],
        position: (u32, u32), // in pixels
        window: &winit::window::Window,
        max_extent: &vk::Extent3D,
        device: &Device,
        allocator: &mut Allocator,
        buffer_manager: Arc<Mutex<BufferManager>>,
        render_pass: &vk::RenderPass,
        command_pool: &vk::CommandPool,
        queue: &vk::Queue,
        swapchain: &Swapchain,
    ) -> RendererResult<Vec<usize>> {
        let (letters, atlas_added) = self.create_letters(
            styles,
            color,
            max_extent,
            device,
            allocator,
            buffer_manager.clone(),
            command_pool,
            queue,
        )?;
        let screen_size = window.inner_size();
        let mut vertex_data = vec![];
        let mut ret_ids = vec![];
        let mut px = 0.0f32;
        for l in letters {
            if px == 0.0f32 {
                px = l.position_and_shape.key.px;
            } else if px != l.position_and_shape.key.px {
                // The last style ended, add a new one
                let id: usize = rand::random();
                let text_buffer =
                    TextBuffer::new(px, vertex_data, device, allocator, buffer_manager.clone())?;
                self.vertex_data.insert(id, text_buffer);
                ret_ids.push(id);
                px = l.position_and_shape.key.px;
                vertex_data = vec![];
            }
            let atlas = &self
                .atlases
                .iter()
                .find(|(inner_px, _atlas)| *inner_px == px)
                .expect("No atlas for px?")
                .1;
            let char_data = if let Some(char_data) =
                atlas.char_data.get(&l.position_and_shape.key.glyph_index)
            {
                char_data
            } else {
                println!("Could not find char data for glyph?");
                continue;
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
            let start_u = char_data.texture_x;
            let start_v = char_data.texture_y;
            let end_u = start_u + char_data.width as f32 / atlas.width;
            let end_v = start_v + char_data.height as f32 / atlas.height;
            let v1 = TextVertexData {
                position: [left, top, 0.0],
                texture_coordinates: [start_u, start_v],
                color: l.color,
            };
            let v2 = TextVertexData {
                position: [left, bottom, 0.0],
                texture_coordinates: [start_u, end_v],
                color: l.color,
            };
            let v3 = TextVertexData {
                position: [right, top, 0.0],
                texture_coordinates: [end_u, start_v],
                color: l.color,
            };
            let v4 = TextVertexData {
                position: [right, bottom, 0.0],
                texture_coordinates: [end_u, end_v],
                color: l.color,
            };
            vertex_data.push(v1);
            vertex_data.push(v2);
            vertex_data.push(v3);
            vertex_data.push(v3);
            vertex_data.push(v2);
            vertex_data.push(v4);
            if px == 0.0f32 {
                panic!("px size is 0.0f32!");
            }
        }
        let id: usize = rand::random();
        let text_buffer =
            TextBuffer::new(px, vertex_data, device, allocator, buffer_manager.clone())?;
        self.vertex_data.insert(id, text_buffer);
        ret_ids.push(id);
        if atlas_added {
            self.update_descriptors(render_pass, swapchain, device)?;
        }
        Ok(ret_ids)
    }

    pub fn remove_text_by_id(
        &mut self,
        device: &Device,
        allocator: &mut Allocator,
        id: usize,
    ) -> RendererResult<()> {
        // TODO deal with the fact px is f32
        // let old_set = self
        //     .vertex_data
        //     .values()
        //     .map(|buf| buf.px as u64)
        //     .collect::<HashSet<u64>>();
        if let Some(mut vert_data) = self.vertex_data.remove(&id) {
            vert_data.destroy();
            // let new_set = self
            //     .vertex_data
            //     .values()
            //     .map(|buf| buf.px as u64)
            //     .collect::<HashSet<u64>>();
            // let removed_pxs = old_set
            //     .difference(&new_set)
            //     .map(|x| *x)
            //     .collect::<HashSet<u64>>();
            // for (atlas_px, atlas) in self.atlases.iter_mut() {
            //     if removed_pxs.contains(&(*atlas_px as u64)) {
            //         atlas.destroy(device, allocator);
            //     }
            // }
            // self.atlases
            //     .retain(|(atlas_px, _atlas)| !removed_pxs.contains(&(*atlas_px as u64)));

            Ok(())
        } else {
            Err(RendererError::InvalidHandle(InvalidHandle))
        }
    }

    pub fn update_descriptors(
        &mut self,
        render_pass: &vk::RenderPass,
        swapchain: &Swapchain,
        device: &Device,
    ) -> RendererResult<()> {
        if self.pipeline.is_none() {
            self.pipeline = Some(GraphicsPipeline::new_text(
                device,
                swapchain.get_extent(),
                render_pass,
                self.shader_module.get_stages(),
                &TextVertexData::get_vertex_attributes(),
                &TextVertexData::get_vertex_bindings(),
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
            .max_sets(swapchain.get_actual_image_count() * self.atlases.len() as u32)
            .pool_sizes(&pool_sizes);
        let descriptor_pool =
            unsafe { device.create_descriptor_pool(&descriptor_pool_info, None) }?;
        self.descriptor_pool = Some(descriptor_pool);

        let desc_layouts_text = vec![
            // This pipeline has to exist, so we know this will succeed
            self.pipeline.as_ref().unwrap().descriptor_set_layouts[0];
            swapchain.get_actual_image_count() as usize
        ];
        let descriptor_set_allocate_info_text = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&desc_layouts_text);
        self.descriptor_sets.clear();
        for (px, atlas) in self.atlases.iter() {
            let descriptor_sets_text =
                unsafe { device.allocate_descriptor_sets(&descriptor_set_allocate_info_text) }?;
            for i in 0..swapchain.get_actual_image_count() {
                let image_infos: [vk::DescriptorImageInfo; 1] = [vk::DescriptorImageInfo {
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    image_view: atlas.image_view,
                    sampler: atlas.sampler,
                }];
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
            self.descriptor_sets
                .insert(*px as u32, descriptor_sets_text);
        }
        Ok(())
    }

    pub fn draw(&mut self, device: &Device, cmd_buf: vk::CommandBuffer, index: usize) {
        if let Some(pipeline) = &self.pipeline {
            unsafe {
                device.cmd_bind_pipeline(
                    cmd_buf,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.pipeline,
                );
            }
            for text_buffer in self.vertex_data.values_mut() {
                unsafe {
                    let descriptor_set = if let Some(desc_set_vec) =
                        self.descriptor_sets.get(&(text_buffer.px as u32))
                    {
                        desc_set_vec[index]
                    } else {
                        panic!("Could not get descriptor set for px");
                    };
                    device.cmd_bind_descriptor_sets(
                        cmd_buf,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.pipeline_layout,
                        0,
                        &[descriptor_set],
                        &[],
                    );
                    let int_buf = text_buffer.vertex_buffer.get_buffer();
                    device.cmd_bind_vertex_buffers(cmd_buf, 0, &[int_buf.buffer], &[0]);
                    device.cmd_draw(
                        cmd_buf,
                        text_buffer.vertex_data.len() as u32,
                        1, // instance count
                        0,
                        0,
                    );
                    text_buffer.last_image_index = Some(index as u32);
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
        for text_buffer in self.vertex_data.values_mut() {
            text_buffer
                .vertex_buffer
                .queue_free(text_buffer.last_image_index)
                .expect("Could not queue buffer for free");
        }
        self.vertex_data.clear();
        for (_px, mut atlas) in self.atlases.drain(0..self.atlases.len()) {
            atlas.destroy(device, allocator);
        }
        self.shader_module.destroy();
    }
}

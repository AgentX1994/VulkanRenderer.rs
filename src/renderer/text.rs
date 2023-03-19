use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use ash::vk;
use ash::Device;

use gpu_allocator::vulkan::Allocator;
use gpu_allocator::MemoryLocation;
use log::error;
use memoffset::offset_of;

use super::error::FontError;
use super::{
    buffer::{Buffer, BufferManager},
    descriptor::{DescriptorAllocator, DescriptorLayoutCache},
    error::{InvalidHandle, RendererError},
    material::{
        Material, MaterialData, MaterialSystem, MeshPassType, ShaderParameters,
        VertexInputDescription,
    },
    swapchain::Swapchain,
    texture::{Texture, TextureStorage},
    utils::Handle,
    RendererResult,
};

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
    texture_handle: Handle<Texture>,
    char_data: HashMap<u16, CharacterData>,
    material_handle: Option<Handle<Material>>,
}

impl TextAtlasTexture {
    pub fn from_u8s(
        data: &[u8],
        width: u32,
        height: u32,
        char_data: HashMap<u16, CharacterData>,
        device: &Device,
        texture_storage: &mut TextureStorage,
        allocator: &mut Allocator,
        buffer_manager: Arc<Mutex<BufferManager>>,
        command_pool: &vk::CommandPool,
        queue: &vk::Queue,
    ) -> RendererResult<Self> {
        let texture_handle = texture_storage.new_texture_from_u8(
            data,
            width,
            height,
            device,
            allocator,
            buffer_manager,
            command_pool,
            queue,
        )?;

        // Done
        Ok(TextAtlasTexture {
            width: width as f32,
            height: height as f32,
            texture_handle,
            char_data,
            material_handle: None,
        })
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

    pub fn get_vertex_description() -> VertexInputDescription {
        VertexInputDescription {
            bindings: Self::get_vertex_bindings().to_vec(),
            attributes: Self::get_vertex_attributes().to_vec(),
            flags: vk::PipelineVertexInputStateCreateFlags::empty(),
        }
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
    font_name: String,
    atlases: Vec<(f32, TextAtlasTexture)>,
}

impl TextHandler {
    pub fn new<P: AsRef<std::path::Path>>(font_path: P) -> RendererResult<TextHandler> {
        let font_name = font_path.as_ref().to_string_lossy().into_owned();
        let font_data = std::fs::read(font_path)?;
        let font = fontdue::Font::from_bytes(font_data, fontdue::FontSettings::default())
            .map_err::<RendererError, _>(|s| FontError(s.into()).into())?;

        Ok(TextHandler {
            vertex_data: HashMap::new(),
            font,
            font_name,
            atlases: vec![],
        })
    }

    fn generate_texture_atlas(
        &mut self,
        px: f32,
        max_extent: &vk::Extent3D,
        device: &Device,
        texture_storage: &mut TextureStorage,
        descriptor_layout_cache: &mut DescriptorLayoutCache,
        descriptor_allocator: &mut DescriptorAllocator,
        material_system: &mut MaterialSystem,
        allocator: &mut Allocator,
        buffer_manager: Arc<Mutex<BufferManager>>,
        command_pool: &vk::CommandPool,
        queue: &vk::Queue,
    ) -> RendererResult<TextAtlasTexture> {
        let mut char_data = HashMap::new();
        let max_texture_width = max_extent.width as usize;
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
            if cur_x + metrics.width > max_texture_width {
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
            character_data.texture_y = character_data.cur_y as f32 / max_height as f32;
            for y in 0..metrics.height {
                for x in 0..metrics.width {
                    data[character_data.cur_x + x + (character_data.cur_y + y) * max_width] =
                        glyph_data[x + y * metrics.width];
                }
            }
        }

        let mut atlas = TextAtlasTexture::from_u8s(
            &data,
            max_width as u32,
            max_height as u32,
            char_data,
            device,
            texture_storage,
            allocator,
            buffer_manager.clone(),
            command_pool,
            queue,
        )?;

        // Create new material for this atlas
        let mat_data = MaterialData {
            base_template: "text".to_string(),
            buffers: vec![],
            textures: vec![atlas.texture_handle],
            parameters: ShaderParameters::default(),
        };

        let handle = material_system.build_material(
            device,
            texture_storage,
            buffer_manager,
            descriptor_layout_cache,
            descriptor_allocator,
            &format!("{} {}px", self.font_name, px),
            mat_data,
        )?;

        atlas.material_handle = Some(handle);

        Ok(atlas)
    }

    pub fn create_letters(
        &mut self,
        styles: &[&fontdue::layout::TextStyle],
        color: [f32; 3],
        max_extent: &vk::Extent3D,
        device: &Device,
        texture_storage: &mut TextureStorage,
        descriptor_layout_cache: &mut DescriptorLayoutCache,
        descriptor_allocator: &mut DescriptorAllocator,
        material_system: &mut MaterialSystem,
        allocator: &mut Allocator,
        buffer_manager: Arc<Mutex<BufferManager>>,
        command_pool: &vk::CommandPool,
        queue: &vk::Queue,
    ) -> RendererResult<Vec<Letter>> {
        let mut layout =
            fontdue::layout::Layout::new(fontdue::layout::CoordinateSystem::PositiveYUp);
        let settings = fontdue::layout::LayoutSettings {
            ..fontdue::layout::LayoutSettings::default()
        };
        layout.reset(&settings);
        for style in styles {
            layout.append(&[&self.font], style);
            if !self.atlases.iter().any(|(px, _)| *px == style.px) {
                let atlas = self.generate_texture_atlas(
                    style.px,
                    max_extent,
                    device,
                    texture_storage,
                    descriptor_layout_cache,
                    descriptor_allocator,
                    material_system,
                    allocator,
                    buffer_manager.clone(),
                    command_pool,
                    queue,
                )?;
                self.atlases.push((style.px, atlas));
            }
        }
        let mut output = vec![];
        for glyph in layout.glyphs() {
            output.push(Letter {
                color,
                position_and_shape: *glyph,
            });
        }
        Ok(output)
    }

    pub fn add_text(
        &mut self,
        styles: &[&fontdue::layout::TextStyle],
        color: [f32; 3],
        position: (u32, u32), // in pixels
        window: &winit::window::Window,
        max_extent: &vk::Extent3D,
        device: &Device,
        texture_storage: &mut TextureStorage,
        allocator: &mut Allocator,
        buffer_manager: Arc<Mutex<BufferManager>>,
        command_pool: &vk::CommandPool,
        queue: &vk::Queue,
        swapchain: &Swapchain,
        descriptor_layout_cache: &mut DescriptorLayoutCache,
        descriptor_allocator: &mut DescriptorAllocator,
        material_system: &mut MaterialSystem,
    ) -> RendererResult<Vec<usize>> {
        let letters = self.create_letters(
            styles,
            color,
            max_extent,
            device,
            texture_storage,
            descriptor_layout_cache,
            descriptor_allocator,
            material_system,
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
                error!("Could not find char data for glyph?");
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
        let text_buffer = TextBuffer::new(px, vertex_data, device, allocator, buffer_manager)?;
        self.vertex_data.insert(id, text_buffer);
        ret_ids.push(id);
        Ok(ret_ids)
    }

    pub fn remove_text_by_id(&mut self, id: usize) -> RendererResult<()> {
        // TODO Remove the texture atlas too? How?

        if let Some(mut vert_data) = self.vertex_data.remove(&id) {
            vert_data.destroy();
            Ok(())
        } else {
            Err(InvalidHandle.into())
        }
    }

    pub fn draw(
        &mut self,
        device: &Device,
        cmd_buf: vk::CommandBuffer,
        index: usize,
        extent: vk::Extent2D,
        material_system: &MaterialSystem,
    ) -> RendererResult<()> {
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
        let mut pipeline = vk::Pipeline::null();
        for text_buffer in self.vertex_data.values_mut() {
            let atlas = if let Some((_px, atlas)) = self
                .atlases
                .iter()
                .find(|(px, _atlas)| *px == text_buffer.px)
            {
                atlas
            } else {
                error!("Could not find atlas for px {}", text_buffer.px);
                continue;
            };
            let material_handle = if let Some(handle) = atlas.material_handle {
                handle
            } else {
                error!("Atlas {} px has no material handle!", text_buffer.px);
                continue;
            };
            let material = material_system.get_material_by_handle(material_handle)?;
            let effect_template =
                material_system.get_effect_template_by_handle(material.original)?;
            let layout = effect_template.pass_shaders[MeshPassType::Forward].layout;
            if pipeline != effect_template.pass_shaders[MeshPassType::Forward].pipeline {
                pipeline = effect_template.pass_shaders[MeshPassType::Forward].pipeline;
                unsafe {
                    device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::GRAPHICS, pipeline);
                    device.cmd_set_viewport(cmd_buf, 0, &viewports);
                    device.cmd_set_scissor(cmd_buf, 0, &scissors);
                }
            }
            unsafe {
                device.cmd_bind_descriptor_sets(
                    cmd_buf,
                    vk::PipelineBindPoint::GRAPHICS,
                    layout,
                    0,
                    &[material.pass_sets[MeshPassType::Forward]],
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
        Ok(())
    }

    pub fn destroy(&mut self) {
        for text_buffer in self.vertex_data.values_mut() {
            text_buffer
                .vertex_buffer
                .queue_free(text_buffer.last_image_index)
                .expect("Could not queue buffer for free");
        }
        self.vertex_data.clear();
        self.atlases.clear();
    }
}

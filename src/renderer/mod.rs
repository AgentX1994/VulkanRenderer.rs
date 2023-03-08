use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use ash::vk;

use gpu_allocator::MemoryLocation;
use nalgebra_glm as glm;

use gpu_allocator::vulkan::{AllocationCreateDesc, Allocator, AllocatorCreateDesc};

mod buffer;
pub mod camera;
mod context;
pub mod error;
pub mod light;
pub mod model;
mod pipeline;
mod queue;
mod render_target;
pub mod scene;
mod shader_module;
mod swapchain;
mod text;
mod texture;
pub mod utils;
pub mod vertex;

use buffer::Buffer;
use camera::Camera;
use model::Model;
use pipeline::GraphicsPipeline;
use shader_module::ShaderModule;
use swapchain::Swapchain;
use vertex::Vertex;

use self::buffer::BufferManager;
use self::context::VulkanContext;
use self::light::LightManager;
use self::text::TextHandler;
use self::texture::TextureStorage;
use self::utils::InternalWindow;

pub use error::RendererResult;

const FRAMES_IN_FLIGHT: usize = 2;

struct FrameData {
    device: ash::Device,
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    in_flight_fence: vk::Fence,
}

impl Drop for FrameData {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_fence(self.in_flight_fence, None);
            self.device
                .destroy_semaphore(self.render_finished_semaphore, None);
            self.device
                .destroy_semaphore(self.image_available_semaphore, None);
        }
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct InstanceData {
    pub model_matrix: [[f32; 4]; 4],
    pub inverse_model_matrix: [[f32; 4]; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub texture_id: u32,
}

impl InstanceData {
    pub fn new(
        model: glm::Mat4,
        metallic: f32,
        roughness: f32,
        texture_id: u32,
    ) -> Self {
        InstanceData {
            model_matrix: model.into(),
            inverse_model_matrix: model.try_inverse().expect("Could not get inverse!").into(),
            metallic,
            roughness,
            texture_id,
        }
    }
}

pub struct Renderer {
    dropped: bool,
    pub context: VulkanContext,
    pub allocator: Option<Allocator>,
    pub buffer_manager: Arc<Mutex<BufferManager>>,
    swapchain: Swapchain,
    render_pass: vk::RenderPass,
    shader_module: ShaderModule,
    graphics_pipeline: GraphicsPipeline,
    graphics_command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    frame_data: Vec<FrameData>,
    images_in_flight: Vec<vk::Fence>,
    current_image: usize,
    uniform_buffer: Buffer,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets_camera: Vec<vk::DescriptorSet>,
    descriptor_sets_lights: Vec<vk::DescriptorSet>,
    descriptor_sets_texture: Vec<vk::DescriptorSet>,
    light_buffer: Buffer,
    texture_storage: TextureStorage,
    number_of_textures: u32,
    pub text: TextHandler,
    pub models: Vec<Rc<RefCell<Model<Vertex, InstanceData>>>>,
}

impl Renderer {
    fn create_render_pass(
        device: &ash::Device,
        format: &vk::SurfaceFormatKHR,
    ) -> RendererResult<vk::RenderPass> {
        let attachments = [
            vk::AttachmentDescription::builder()
                .format(format.format)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .samples(vk::SampleCountFlags::TYPE_1)
                .build(),
            vk::AttachmentDescription::builder()
                .format(vk::Format::D32_SFLOAT)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .samples(vk::SampleCountFlags::TYPE_1)
                .build(),
        ];

        let color_attachment_references = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];

        let depth_attachment_reference = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        let subpasses = [vk::SubpassDescription::builder()
            .color_attachments(&color_attachment_references)
            .depth_stencil_attachment(&depth_attachment_reference)
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .build()];

        let subpass_dependencies = [vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_subpass(0)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            )
            .build()];

        let renderpass_info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&subpass_dependencies);
        unsafe { Ok(device.create_render_pass(&renderpass_info, None)?) }
    }

    fn create_frame_data(device: &ash::Device, num: usize) -> RendererResult<Vec<FrameData>> {
        let semaphore_info = vk::SemaphoreCreateInfo::builder();
        let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
        (0..num)
            .map(|_| {
                let image_available_semaphore =
                    unsafe { device.create_semaphore(&semaphore_info, None)? };
                let render_finished_semaphore =
                    unsafe { device.create_semaphore(&semaphore_info, None)? };
                let in_flight_fence = unsafe { device.create_fence(&fence_info, None)? };
                Ok(FrameData {
                    device: device.clone(),
                    image_available_semaphore,
                    render_finished_semaphore,
                    in_flight_fence,
                })
            })
            .collect()
    }

    fn create_descriptor_sets(
        device: &ash::Device,
        pipeline: &GraphicsPipeline,
        pool: &vk::DescriptorPool,
        swapchain: &Swapchain,
        uniform_buffer: &Buffer,
        light_buffer: &Buffer,
        num_textures: u32,
    ) -> RendererResult<(
        Vec<vk::DescriptorSet>,
        Vec<vk::DescriptorSet>,
        Vec<vk::DescriptorSet>,
    )> {
        // Now create the descriptor sets, one for each swapchain image
        let descriptor_layouts_camera =
            vec![pipeline.descriptor_set_layouts[0]; swapchain.get_actual_image_count() as usize];
        let descriptor_set_allocate_info_camera = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(*pool)
            .set_layouts(&descriptor_layouts_camera);
        let descriptor_sets_camera =
            unsafe { device.allocate_descriptor_sets(&descriptor_set_allocate_info_camera)? };

        let descriptor_layouts_lights =
            vec![pipeline.descriptor_set_layouts[1]; swapchain.get_actual_image_count() as usize];
        let descriptor_set_allocate_info_lights = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(*pool)
            .set_layouts(&descriptor_layouts_lights);
        let descriptor_sets_lights =
            unsafe { device.allocate_descriptor_sets(&descriptor_set_allocate_info_lights)? };

        let descriptor_layouts_texture =
            vec![pipeline.descriptor_set_layouts[2]; swapchain.get_actual_image_count() as usize];
        let descriptor_counts_texture =
            vec![num_textures; swapchain.get_actual_image_count() as usize];
        let mut variable_descriptor_allocate_info_texture =
            vk::DescriptorSetVariableDescriptorCountAllocateInfo::builder()
                .descriptor_counts(&descriptor_counts_texture);
        let descriptor_set_allocate_info_texture = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(*pool)
            .set_layouts(&descriptor_layouts_texture)
            .push_next(&mut variable_descriptor_allocate_info_texture);
        let descriptor_sets_texture =
            unsafe { device.allocate_descriptor_sets(&descriptor_set_allocate_info_texture)? };

        for ds in descriptor_sets_camera.iter() {
            let int_buf = uniform_buffer.get_buffer();
            let buffer_info = [vk::DescriptorBufferInfo {
                buffer: int_buf.buffer,
                offset: 0,
                range: std::mem::size_of::<[[[f32; 4]; 4]; 2]>() as u64,
            }];

            let desc_sets_write = [vk::WriteDescriptorSet::builder()
                .dst_set(*ds)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&buffer_info)
                .build()];
            unsafe {
                device.update_descriptor_sets(&desc_sets_write, &[]);
            }
        }

        for ds in descriptor_sets_lights.iter() {
            let int_buf = light_buffer.get_buffer();
            let buffer_info = [vk::DescriptorBufferInfo {
                buffer: int_buf.buffer,
                offset: 0,
                range: 8,
            }];

            let desc_sets_write = [vk::WriteDescriptorSet::builder()
                .dst_set(*ds)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buffer_info)
                .build()];
            unsafe {
                device.update_descriptor_sets(&desc_sets_write, &[]);
            }
        }

        Ok((
            descriptor_sets_camera,
            descriptor_sets_lights,
            descriptor_sets_texture,
        ))
    }

    pub fn new(
        name: &str,
        window_width: u32,
        window_height: u32,
        internal_window: InternalWindow,
    ) -> RendererResult<Self> {
        let context = VulkanContext::new(name, internal_window)?;

        // Allocator
        let mut allocator = Allocator::new(&AllocatorCreateDesc {
            instance: context.instance.clone(),
            device: context.device.clone(),
            physical_device: context.physical_device,
            debug_settings: gpu_allocator::AllocatorDebugSettings {
                log_memory_information: true,
                log_leaks_on_shutdown: true,
                store_stack_traces: false,
                log_allocations: true,
                log_frees: true,
                log_stack_traces: false,
            },
            buffer_device_address: false,
        })?;
        let format = context
            .surface_formats
            .iter()
            .find(|format| {
                format.format == vk::Format::B8G8R8A8_SRGB
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .ok_or(vk::Result::ERROR_FORMAT_NOT_SUPPORTED)?;

        let render_pass = Self::create_render_pass(&context.device, format)?;

        let swapchain = Swapchain::new(
            &context,
            &mut allocator,
            *format,
            window_width,
            window_height,
            &render_pass,
        )?;

        let shader_module = ShaderModule::new(
            &context.device,
            vk_shader_macros::include_glsl!("./shaders/default.vert", kind: vert),
            vk_shader_macros::include_glsl!("./shaders/default.frag", kind: frag),
        )?;
        let graphics_pipeline = GraphicsPipeline::new(
            &context.device,
            swapchain.get_extent(),
            &render_pass,
            shader_module.get_stages(),
            &Vertex::get_attribute_descriptions(),
            &Vertex::get_binding_description(),
            0,
        )?;

        // Create command pools
        let graphics_commandpool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(context.graphics_queue.index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let graphics_command_pool = unsafe {
            context
                .device
                .create_command_pool(&graphics_commandpool_info, None)?
        };

        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(graphics_command_pool)
            .command_buffer_count(swapchain.get_actual_image_count());
        let command_buffers = unsafe {
            context
                .device
                .allocate_command_buffers(&command_buffer_allocate_info)?
        };

        let frame_data = Self::create_frame_data(&context.device, FRAMES_IN_FLIGHT)?;
        let images_in_flight = vec![vk::Fence::null(); swapchain.get_actual_image_count() as usize];

        // Create buffer manager
        let buffer_manager = BufferManager::new();
        // Create uniform buffer
        let camera_transforms: [[[f32; 4]; 4]; 2] =
            [glm::Mat4::identity().into(), glm::Mat4::identity().into()];
        let mut uniform_buffer = BufferManager::new_buffer(
            buffer_manager.clone(),
            &context.device,
            &mut allocator,
            std::mem::size_of::<[[[f32; 4]; 4]; 2]>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
        )?;
        uniform_buffer.fill(&mut allocator, &camera_transforms)?;

        // Create storage buffer for lights
        let mut light_buffer = BufferManager::new_buffer(
            buffer_manager.clone(),
            &context.device,
            &mut allocator,
            8,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::CpuToGpu,
        )?;
        light_buffer.fill(&mut allocator, &[0.0f32; 2])?;

        // Create descriptor pool
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: swapchain.get_actual_image_count(),
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: swapchain.get_actual_image_count(),
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: GraphicsPipeline::MAXIMUM_NUMBER_OF_TEXTURES
                    * swapchain.get_actual_image_count(),
            },
        ];

        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(swapchain.get_actual_image_count() * 3)
            .pool_sizes(&pool_sizes)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
        let descriptor_pool = unsafe {
            context
                .device
                .create_descriptor_pool(&descriptor_pool_info, None)?
        };

        let (descriptor_sets_camera, descriptor_sets_lights, descriptor_sets_texture) =
            Self::create_descriptor_sets(
                &context.device,
                &graphics_pipeline,
                &descriptor_pool,
                &swapchain,
                &uniform_buffer,
                &light_buffer,
                0,
            )?;

        let text = TextHandler::new(&context.device, "Roboto-Regular.ttf")?;

        Ok(Renderer {
            dropped: false,
            context,
            allocator: Some(allocator),
            buffer_manager,
            swapchain,
            graphics_command_pool,
            command_buffers,
            render_pass,
            shader_module,
            graphics_pipeline,
            frame_data,
            images_in_flight,
            current_image: 0,
            uniform_buffer,
            descriptor_pool,
            descriptor_sets_camera,
            descriptor_sets_lights,
            descriptor_sets_texture,
            light_buffer,
            texture_storage: TextureStorage::default(),
            number_of_textures: 0,
            text,
            models: vec![],
        })
    }

    pub fn recreate_swapchain(&mut self, width: u32, height: u32) -> RendererResult<()> {
        unsafe {
            self.context.device.device_wait_idle()?;
        }
        self.context.refresh_surface_data()?;
        if let Some(allo) = &mut self.allocator {
            self.swapchain.destroy(&self.context, allo);
            self.swapchain = Swapchain::new(
                &self.context,
                allo,
                self.swapchain.get_image_format(),
                width,
                height,
                &self.render_pass,
            )?;
            self.graphics_pipeline.destroy();
            self.graphics_pipeline = GraphicsPipeline::new(
                &self.context.device,
                self.swapchain.get_extent(),
                &self.render_pass,
                self.shader_module.get_stages(),
                &Vertex::get_attribute_descriptions(),
                &Vertex::get_binding_description(),
                self.number_of_textures,
            )?;
            unsafe {
                self.context
                    .device
                    .free_descriptor_sets(self.descriptor_pool, &self.descriptor_sets_camera)?;
                self.context
                    .device
                    .free_descriptor_sets(self.descriptor_pool, &self.descriptor_sets_lights)?;
                self.context
                    .device
                    .free_descriptor_sets(self.descriptor_pool, &self.descriptor_sets_texture)?;
            }
            let (a, b, c) = Self::create_descriptor_sets(
                &self.context.device,
                &self.graphics_pipeline,
                &self.descriptor_pool,
                &self.swapchain,
                &self.uniform_buffer,
                &self.light_buffer,
                self.texture_storage.get_number_of_textures() as u32,
            )?;
            self.descriptor_sets_camera = a;
            self.descriptor_sets_lights = b;
            self.descriptor_sets_texture = c;
            self.update_textures()?;
            self.text.clear_pipeline();
            self.text.update_descriptors(
                &self.render_pass,
                &self.swapchain,
                &self.context.device,
            )?;
        }
        Ok(())
    }

    fn wait_for_next_frame_fence(&self) -> RendererResult<()> {
        unsafe {
            self.context.device.wait_for_fences(
                &[self.frame_data[self.current_image].in_flight_fence],
                true,
                std::u64::MAX,
            )?;
        }
        Ok(())
    }

    fn wait_for_image_fence_and_set_new_fence(&mut self, image_index: usize) -> RendererResult<()> {
        if self.images_in_flight[image_index] != vk::Fence::null() {
            unsafe {
                self.context.device.wait_for_fences(
                    &[self.images_in_flight[image_index]],
                    true,
                    std::u64::MAX,
                )?;
            }
        }

        self.images_in_flight[image_index] = self.frame_data[self.current_image].in_flight_fence;
        Ok(())
    }

    fn update_command_buffer(&mut self, image_index: usize) -> RendererResult<()> {
        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder();
        let cmd_buf = &self.command_buffers[image_index];
        let framebuffer = &self.swapchain.get_render_targets()[image_index].framebuffer;
        unsafe {
            self.context
                .device
                .begin_command_buffer(*cmd_buf, &command_buffer_begin_info)?;
        }
        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.00, 1.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];
        let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.render_pass)
            .framebuffer(*framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain.get_extent(),
            })
            .clear_values(&clear_values);
        unsafe {
            self.context.device.cmd_begin_render_pass(
                *cmd_buf,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );
            self.context.device.cmd_bind_pipeline(
                *cmd_buf,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline.pipeline,
            );
            self.context.device.cmd_bind_descriptor_sets(
                *cmd_buf,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline.pipeline_layout,
                0,
                &[
                    self.descriptor_sets_camera[image_index],
                    self.descriptor_sets_lights[image_index],
                    self.descriptor_sets_texture[image_index],
                ],
                &[],
            );
            for m in &self.models {
                m.borrow().draw(&self.context.device, *cmd_buf);
            }
            self.text.draw(&self.context.device, *cmd_buf, image_index);
            self.context.device.cmd_end_render_pass(*cmd_buf);
            self.context.device.end_command_buffer(*cmd_buf)?;
        }
        Ok(())
    }

    fn submit_commands(&mut self, image_index: usize) -> RendererResult<()> {
        self.update_command_buffer(image_index)?;
        let cmd_buf = &self.command_buffers[image_index];
        let this_frame_data = &self.frame_data[self.current_image];
        let semaphores_available = [this_frame_data.image_available_semaphore];
        let waiting_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let semaphores_finished = [this_frame_data.render_finished_semaphore];
        let command_bufs = [*cmd_buf];
        let submit_info = [vk::SubmitInfo::builder()
            .wait_semaphores(&semaphores_available)
            .wait_dst_stage_mask(&waiting_stages)
            .command_buffers(&command_bufs[..])
            .signal_semaphores(&semaphores_finished)
            .build()];
        unsafe {
            self.context
                .device
                .reset_fences(&[this_frame_data.in_flight_fence])?;
            if let Some(alloc) = &mut self.allocator {
                for m in &mut self.models {
                    m.borrow_mut().update_instance_buffer(
                        &self.context.device,
                        alloc,
                        self.buffer_manager.clone(),
                    )?;
                }
            } else {
                panic!("No allocator!");
            }
            self.context.device.queue_submit(
                self.context.graphics_queue.queue,
                &submit_info,
                this_frame_data.in_flight_fence,
            )?;
        }
        Ok(())
    }

    fn present(&self, image_index: u32) -> RendererResult<()> {
        self.swapchain.present(
            &self.context.graphics_queue.queue,
            &self.frame_data[self.current_image].render_finished_semaphore,
            image_index,
        )?;
        Ok(())
    }

    pub fn render(&mut self) -> RendererResult<()> {
        self.wait_for_next_frame_fence()?;
        let image_index = self.swapchain.get_next_image(
            std::u64::MAX,
            &self.frame_data[self.current_image].image_available_semaphore,
            vk::Fence::null(),
        )?;

        self.wait_for_image_fence_and_set_new_fence(image_index as usize)?;

        if let Some(allo) = &mut self.allocator {
            self.buffer_manager
                .lock()
                .unwrap()
                .free_queued(allo, image_index);
        }

        self.submit_commands(image_index as usize)?;

        self.present(image_index)?;
        self.current_image = (self.current_image + 1) % FRAMES_IN_FLIGHT;
        Ok(())
    }

    pub fn update_uniforms_from_camera(&mut self, camera: &Camera) -> RendererResult<()> {
        if let Some(alloc) = &mut self.allocator {
            Ok(camera.update_buffer(alloc, &mut self.uniform_buffer)?)
        } else {
            panic!("No allocator!");
        }
    }

    pub fn update_storage_from_lights(&mut self, lights: &LightManager) -> RendererResult<()> {
        if let Some(alloc) = &mut self.allocator {
            Ok(lights.update_buffer(
                &self.context.device,
                alloc,
                &mut self.light_buffer,
                &mut self.descriptor_sets_lights[..],
            )?)
        } else {
            panic!("No allocator!");
        }
    }

    pub fn new_texture_from_file<P: AsRef<Path>>(&mut self, path: P) -> RendererResult<usize> {
        if let Some(allo) = &mut self.allocator {
            Ok(self.texture_storage.new_texture_from_file(
                path,
                &self.context.device,
                allo,
                self.buffer_manager.clone(),
                self.graphics_command_pool,
                self.context.graphics_queue.queue,
            )?)
        } else {
            panic!("No allocator!");
        }
    }

    pub fn update_textures(&mut self) -> RendererResult<()> {
        let num_texs = self.texture_storage.get_number_of_textures() as u32;
        if self.number_of_textures < num_texs {
            self.graphics_pipeline.destroy();
            self.graphics_pipeline = GraphicsPipeline::new(
                &self.context.device,
                self.swapchain.get_extent(),
                &self.render_pass,
                self.shader_module.get_stages(),
                &Vertex::get_attribute_descriptions(),
                &Vertex::get_binding_description(),
                num_texs,
            )?;
            unsafe {
                self.context
                    .device
                    .free_descriptor_sets(self.descriptor_pool, &self.descriptor_sets_texture)
            }?;
            let descriptor_layouts_texture = vec![
                self.graphics_pipeline.descriptor_set_layouts[2];
                self.swapchain.get_actual_image_count() as usize
            ];
            let descriptor_counts_texture =
                vec![num_texs; self.swapchain.get_actual_image_count() as usize];
            let mut variable_descriptor_allocate_info_texture =
                vk::DescriptorSetVariableDescriptorCountAllocateInfo::builder()
                    .descriptor_counts(&descriptor_counts_texture);
            let descriptor_set_allocate_info_texture = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(self.descriptor_pool)
                .set_layouts(&descriptor_layouts_texture)
                .push_next(&mut variable_descriptor_allocate_info_texture);
            self.descriptor_sets_texture = unsafe {
                self.context
                    .device
                    .allocate_descriptor_sets(&descriptor_set_allocate_info_texture)?
            };
            self.number_of_textures = num_texs;
        }
        for ds in self.descriptor_sets_texture.iter() {
            let image_info = self.texture_storage.get_descriptor_image_info();

            let desc_sets_write = [vk::WriteDescriptorSet::builder()
                .dst_set(*ds)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&image_info)
                .build()];
            unsafe {
                self.context
                    .device
                    .update_descriptor_sets(&desc_sets_write, &[]);
            }
        }
        Ok(())
    }

    pub fn add_text(
        &mut self,
        window: &winit::window::Window,
        position: (u32, u32),
        styles: &[&fontdue::layout::TextStyle],
        color: [f32; 3],
    ) -> RendererResult<Vec<usize>> {
        if let Some(allo) = &mut self.allocator {
            self.text.add_text(
                styles,
                color,
                position,
                window,
                &self.context.max_texture_extent,
                &self.context.device,
                allo,
                self.buffer_manager.clone(),
                &self.render_pass,
                &self.graphics_command_pool,
                &self.context.graphics_queue.queue,
                &self.swapchain,
            )
        } else {
            panic!("No allocator!");
        }
    }

    pub fn remove_text(&mut self, id: usize) -> RendererResult<()> {
        self.text.remove_text_by_id(id)
    }

    pub fn screenshot(&mut self) -> RendererResult<()> {
        let command_buffer_alloc_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.graphics_command_pool)
            .command_buffer_count(1);
        let copy_buffer = unsafe {
            self.context
                .device
                .allocate_command_buffers(&command_buffer_alloc_info)
        }?[0];

        let cmd_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.context
                .device
                .begin_command_buffer(copy_buffer, &cmd_begin_info)
        }?;

        let image_create_info = vk::ImageCreateInfo::builder()
            .format(vk::Format::R8G8B8A8_UNORM)
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width: self.swapchain.get_extent().width,
                height: self.swapchain.get_extent().height,
                depth: 1,
            })
            .array_layers(1)
            .mip_levels(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::LINEAR)
            .usage(vk::ImageUsageFlags::TRANSFER_DST)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let dest_image = unsafe { self.context.device.create_image(&image_create_info, None) }?;
        let reqs = unsafe {
            self.context
                .device
                .get_image_memory_requirements(dest_image)
        };

        let dest_image_allocation = if let Some(allocator) = &mut self.allocator {
            allocator.allocate(&AllocationCreateDesc {
                name: "dest_image",
                requirements: reqs,
                location: MemoryLocation::GpuToCpu,
                linear: false,
            })?
        } else {
            panic!("No allocator!");
        };
        unsafe {
            self.context.device.bind_image_memory(
                dest_image,
                dest_image_allocation.memory(),
                dest_image_allocation.offset(),
            )?;
        };

        // Transition layouts of source and destination
        {
            let barrier = vk::ImageMemoryBarrier::builder()
                .image(dest_image)
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
                self.context.device.cmd_pipeline_barrier(
                    copy_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
            }
        }
        let source_image = self.swapchain.get_render_targets()[self.current_image].image;
        {
            let barrier = vk::ImageMemoryBarrier::builder()
                .image(source_image)
                .src_access_mask(vk::AccessFlags::MEMORY_READ)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .old_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .build();
            unsafe {
                self.context.device.cmd_pipeline_barrier(
                    copy_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
            }
        }

        // Now copy the image
        let zero_offset = vk::Offset3D::default();
        let copy_area = vk::ImageCopy::builder()
            .src_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_offset(zero_offset)
            .dst_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .dst_offset(zero_offset)
            .extent(vk::Extent3D {
                width: self.swapchain.get_extent().width,
                height: self.swapchain.get_extent().height,
                depth: 1,
            })
            .build();
        unsafe {
            self.context.device.cmd_copy_image(
                copy_buffer,
                source_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dest_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy_area],
            );
        }
        // Restore the layouts of the images
        {
            let barrier = vk::ImageMemoryBarrier::builder()
                .image(dest_image)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .build();
            unsafe {
                self.context.device.cmd_pipeline_barrier(
                    copy_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
            }
        }
        {
            let barrier = vk::ImageMemoryBarrier::builder()
                .image(source_image)
                .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ)
                .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .build();
            unsafe {
                self.context.device.cmd_pipeline_barrier(
                    copy_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
            }
        }

        unsafe {
            self.context.device.end_command_buffer(copy_buffer)?;
        }

        let submit_infos = [vk::SubmitInfo::builder()
            .command_buffers(&[copy_buffer])
            .build()];
        let fence = unsafe {
            self.context
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }?;
        unsafe {
            self.context.device.queue_submit(
                self.context.graphics_queue.queue,
                &submit_infos,
                fence,
            )
        }?;

        unsafe {
            self.context
                .device
                .wait_for_fences(&[fence], true, std::u64::MAX)
        }?;

        unsafe { self.context.device.destroy_fence(fence, None) };
        unsafe {
            self.context
                .device
                .free_command_buffers(self.graphics_command_pool, &[copy_buffer])
        };

        let mut data = {
            let source_ptr = dest_image_allocation
                .mapped_ptr()
                .expect("No mapped memory for image")
                .as_ptr() as *mut u8;

            let subresource_layout = unsafe {
                self.context.device.get_image_subresource_layout(
                    dest_image,
                    vk::ImageSubresource {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        array_layer: 0,
                    },
                )
            };

            let mut data = Vec::<u8>::with_capacity(subresource_layout.size as usize);
            unsafe {
                std::ptr::copy(
                    source_ptr,
                    data.as_mut_ptr(),
                    subresource_layout.size as usize,
                );
                data.set_len(subresource_layout.size as usize);
            }
            if let Some(allo) = &mut self.allocator {
                allo.free(dest_image_allocation)
                    .expect("Could not free dest image");
            }
            unsafe {
                self.context.device.destroy_image(dest_image, None);
            }
            data
        };

        // The data that comes out might not be in RGBA8 format, so we have to convert it.
        match self.swapchain.get_image_format().format {
            vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB => {
                for v in data.chunks_mut(4) {
                    // BGRA -> RGBA involves swapping B and R (0 and 2)
                    v.swap(0, 2);
                }
            }
            vk::Format::R8G8B8A8_UNORM => {} // Nothing to do
            _ => panic!(
                "No way to convert this format! {:?}",
                self.swapchain.get_image_format().format
            ),
        }

        let screen: image::ImageBuffer<image::Rgba<u8>, _> = image::ImageBuffer::from_raw(
            self.swapchain.get_extent().width,
            self.swapchain.get_extent().height,
            data,
        )
        .expect("ImageBuffer creation");

        let screen_image = image::DynamicImage::ImageRgba8(screen);
        screen_image
            .save("screenshot.jpg")
            .expect("Could not save screenshot");

        Ok(())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        if self.dropped {
            return;
        }
        unsafe {
            self.context
                .device
                .device_wait_idle()
                .expect("Something wrong while waiting for idle");

            self.context
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);

            let mut allocator = self.allocator.take().expect("We had no allocator?!");
            for m in &mut self.models {
                let mut m = m.borrow_mut();
                if let Some(vb) = &mut m.vertex_buffer {
                    vb.queue_free(None).expect("Invalid Handle?!");
                }
                if let Some(ib) = &mut m.index_buffer {
                    ib.queue_free(None).expect("Invalid Handle?!");
                }
                if let Some(ib) = &mut m.instance_buffer {
                    ib.queue_free(None).expect("Invalid Handle?!");
                }
            }
            self.uniform_buffer
                .queue_free(None)
                .expect("Invalid Handle?!");
            self.light_buffer
                .queue_free(None)
                .expect("Invalid Handle?!");
            self.texture_storage
                .clean_up(&self.context.device, &mut allocator);

            self.frame_data.clear();
            self.context
                .device
                .destroy_command_pool(self.graphics_command_pool, None);
            // device.destroy_command_pool(command_pool_transfer, None);
            self.graphics_pipeline.destroy();
            self.text.destroy(&self.context.device, &mut allocator);
            self.context
                .device
                .destroy_render_pass(self.render_pass, None);
            let num_images = self.swapchain.get_actual_image_count();
            self.swapchain.destroy(&self.context, &mut allocator);
            self.shader_module.destroy();

            {
                let mut guard = self.buffer_manager.lock().unwrap();
                for i in 0..num_images {
                    guard.free_queued(&mut allocator, i);
                }
            }
            drop(allocator); // Ensure all memory is freed
        }
        self.dropped = true;
    }
}

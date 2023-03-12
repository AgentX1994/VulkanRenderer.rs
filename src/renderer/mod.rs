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
mod descriptor;
pub mod error;
pub mod light;
mod material;
pub mod model;
mod queue;
mod render_target;
pub mod scene;
mod shaders;
mod swapchain;
mod text;
mod texture;
pub mod utils;
pub mod vertex;

use buffer::Buffer;
use camera::Camera;
use model::Model;
use swapchain::Swapchain;
use vertex::Vertex;

use self::buffer::BufferManager;
use self::context::VulkanContext;
use self::descriptor::{DescriptorAllocator, DescriptorLayoutCache};
use self::light::LightManager;
use self::material::{MaterialData, MaterialSystem, MeshPassType, ShaderParameters};
use self::shaders::ShaderCache;
use self::text::TextHandler;
use self::texture::{TextureHandle, TextureStorage};
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
    pub fn new(model: glm::Mat4, metallic: f32, roughness: f32, texture_id: u32) -> Self {
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
    shader_cache: ShaderCache,
    descriptor_layout_cache: DescriptorLayoutCache,
    descriptor_allocator: DescriptorAllocator,
    material_system: MaterialSystem,
    graphics_command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    frame_data: Vec<FrameData>,
    images_in_flight: Vec<vk::Fence>,
    current_image: usize,
    uniform_buffer: Buffer,
    descriptor_set_camera: vk::DescriptorSet,
    descriptor_set_lights: vk::DescriptorSet,
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
            (std::mem::size_of::<[[[f32; 4]; 4]; 2]>()
                * swapchain.get_actual_image_count() as usize) as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
        )?;
        for i in 0..swapchain.get_actual_image_count() as usize {
            let offset = i * std::mem::size_of::<[[[f32; 4]; 4]; 2]>();
            uniform_buffer.copy_to_offset(&mut allocator, &camera_transforms, offset)?;
        }

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

        let mut shader_cache = ShaderCache::new(&context.device)?;
        let mut material_system =
            MaterialSystem::new(&context.device, render_pass, &mut shader_cache)?;

        let mut descriptor_layout_cache = DescriptorLayoutCache::default();
        let mut descriptor_allocator = DescriptorAllocator::default();

        let text = TextHandler::new("Roboto-Regular.ttf")?;

        let mut texture_storage = TextureStorage::default();

        let default_template_handle = material_system.get_effect_template_handle("default")?;
        let texture_handle = texture_storage.new_texture_from_file(
            "plain_white.jpg",
            &context.device,
            &mut allocator,
            buffer_manager.clone(),
            graphics_command_pool,
            context.graphics_queue.queue,
        )?;
        let default_material_data = MaterialData {
            textures: vec![texture_handle],
            parameters: ShaderParameters {},
            base_template: "default".to_string(),
        };
        let material_handle = material_system.build_material(
            &context.device,
            &texture_storage,
            &mut descriptor_layout_cache,
            &mut descriptor_allocator,
            "default",
            default_material_data,
        )?;
        let default_template =
            material_system.get_effect_template_by_handle(default_template_handle)?;

        let effect_handle = default_template.pass_shaders[MeshPassType::Forward]
            .effect_handle
            .expect("No effect handle?");
        let effect = shader_cache.get_shader_effect_by_handle(effect_handle)?;

        let descriptor_set_camera =
            descriptor_allocator.allocate(&context.device, effect.set_layouts[0])?;
        // Update camera descriptor sets
        unsafe {
            let buffer_info = [vk::DescriptorBufferInfo::builder()
                .buffer(uniform_buffer.get_buffer().buffer)
                .range(std::mem::size_of::<[[[f32; 4]; 4]; 2]>() as u64)
                .build()];
            let descriptor_write = vk::WriteDescriptorSet::builder()
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .dst_binding(0)
                .dst_set(descriptor_set_camera)
                .buffer_info(&buffer_info[..]);
            context
                .device
                .update_descriptor_sets(&[*descriptor_write], &[]);
        }
        let descriptor_set_lights =
            descriptor_allocator.allocate(&context.device, effect.set_layouts[1])?;

        Ok(Renderer {
            dropped: false,
            context,
            allocator: Some(allocator),
            buffer_manager,
            swapchain,
            graphics_command_pool,
            command_buffers,
            render_pass,
            shader_cache,
            descriptor_layout_cache,
            descriptor_allocator,
            material_system,
            frame_data,
            images_in_flight,
            current_image: 0,
            uniform_buffer,
            descriptor_set_camera,
            descriptor_set_lights,
            light_buffer,
            texture_storage,
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
            let old_image_count = self.swapchain.get_actual_image_count();
            self.swapchain.destroy(&self.context, allo);
            self.swapchain = Swapchain::new(
                &self.context,
                allo,
                self.swapchain.get_image_format(),
                width,
                height,
                &self.render_pass,
            )?;
            assert!(old_image_count == self.swapchain.get_actual_image_count());
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
            let material_handle = self
                .material_system
                .get_material_handle("default")
                .expect("No \"default\" material");
            let material = self
                .material_system
                .get_material_by_handle(material_handle)
                .expect("Could not get default material");
            let effect_template = self
                .material_system
                .get_effect_template_by_handle(material.original)
                .expect("No default template");
            self.context.device.cmd_bind_pipeline(
                *cmd_buf,
                vk::PipelineBindPoint::GRAPHICS,
                effect_template.pass_shaders[MeshPassType::Forward].pipeline,
            );

            let viewports = [vk::Viewport {
                x: 0.,
                y: 0.,
                width: self.swapchain.get_extent().width as f32,
                height: self.swapchain.get_extent().height as f32,
                min_depth: 0.,
                max_depth: 1.,
            }];
            let scissors = [vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain.get_extent(),
            }];

            self.context
                .device
                .cmd_set_viewport(*cmd_buf, 0, &viewports);
            self.context.device.cmd_set_scissor(*cmd_buf, 0, &scissors);

            let camera_buffer_offset = image_index * std::mem::size_of::<[[[f32; 4]; 4]; 2]>();

            self.context.device.cmd_bind_descriptor_sets(
                *cmd_buf,
                vk::PipelineBindPoint::GRAPHICS,
                effect_template.pass_shaders[MeshPassType::Forward].layout,
                0,
                &[
                    self.descriptor_set_camera,
                    self.descriptor_set_lights,
                    material.pass_sets[MeshPassType::Forward],
                ],
                // Only the camera offset changes
                &[camera_buffer_offset as u32],
            );
            for m in &self.models {
                m.borrow().draw(&self.context.device, *cmd_buf);
            }
            self.text.draw(
                &self.context.device,
                *cmd_buf,
                image_index,
                self.swapchain.get_extent(),
                &self.material_system,
            )?;
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

    pub fn render(&mut self, camera: &Camera) -> RendererResult<()> {
        self.wait_for_next_frame_fence()?;
        let image_index = self.swapchain.get_next_image(
            std::u64::MAX,
            &self.frame_data[self.current_image].image_available_semaphore,
            vk::Fence::null(),
        )?;

        if let Some(alloc) = &mut self.allocator {
            let offset = image_index as usize * std::mem::size_of::<[[[f32; 4]; 4]; 2]>();
            camera.update_buffer(alloc, &mut self.uniform_buffer, offset)?;
        } else {
            panic!("No allocator!");
        }

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

    pub fn update_storage_from_lights(&mut self, lights: &LightManager) -> RendererResult<()> {
        if let Some(alloc) = &mut self.allocator {
            Ok(lights.update_buffer(
                &self.context.device,
                alloc,
                &mut self.light_buffer,
                self.descriptor_set_lights,
            )?)
        } else {
            panic!("No allocator!");
        }
    }

    pub fn new_texture_from_file<P: AsRef<Path>>(
        &mut self,
        path: P,
    ) -> RendererResult<TextureHandle> {
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
                &mut self.texture_storage,
                allo,
                self.buffer_manager.clone(),
                &self.graphics_command_pool,
                &self.context.graphics_queue.queue,
                &self.swapchain,
                &mut self.descriptor_layout_cache,
                &mut self.descriptor_allocator,
                &mut self.material_system,
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
            self.text.destroy();
            self.context
                .device
                .destroy_render_pass(self.render_pass, None);
            let num_images = self.swapchain.get_actual_image_count();
            self.material_system.destroy(&self.context.device);
            self.shader_cache.destroy(&self.context.device);
            self.swapchain.destroy(&self.context, &mut allocator);

            self.descriptor_layout_cache.destroy(&self.context.device);
            self.descriptor_allocator.destroy(&self.context.device);

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

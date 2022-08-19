use std::error;
use std::ffi::{c_void, CStr, CString};
use std::fmt;

use ash::extensions::ext;
use ash::extensions::khr;
use ash::prelude::VkResult;
use ash::vk;
use ash::{Device, Instance};

use gpu_allocator::MemoryLocation;
use nalgebra_glm as glm;

use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

mod buffer;
pub mod camera;
pub mod model;
mod pipeline;
mod shader_module;
mod swapchain;
pub mod vertex;

use buffer::Buffer;
use camera::Camera;
use model::Model;
use pipeline::GraphicsPipeline;
use shader_module::ShaderModule;
use swapchain::Swapchain;
use vertex::Vertex;

#[derive(Debug)]
pub enum RendererError {
    LoadError(ash::LoadingError),
    VulkanError(vk::Result),
}

impl fmt::Display for RendererError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            RendererError::LoadError(ref e) => e.fmt(f),
            RendererError::VulkanError(ref e) => e.fmt(f),
        }
    }
}

impl error::Error for RendererError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            RendererError::LoadError(ref e) => Some(e),
            RendererError::VulkanError(ref e) => Some(e),
        }
    }
}

impl From<ash::LoadingError> for RendererError {
    fn from(e: ash::LoadingError) -> RendererError {
        RendererError::LoadError(e)
    }
}

impl From<vk::Result> for RendererError {
    fn from(e: vk::Result) -> RendererError {
        RendererError::VulkanError(e)
    }
}

pub type RendererResult<T> = Result<T, RendererError>;

unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let message = CStr::from_ptr((*p_callback_data).p_message);
    let severity = format!("{:?}", message_severity).to_lowercase();
    let ty = format!("{:?}", message_type).to_lowercase();
    println!("[Debug][{}][{}] {:?}", severity, ty, message);
    vk::FALSE
}

const FRAMES_IN_FLIGHT: usize = 2;

struct FrameData {
    device: Device,
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
pub struct InstanceData {
    pub model_matrix: [[f32; 4]; 4],
    pub color_mod: [f32; 3],
}

pub struct Renderer {
    dropped: bool,
    entry: ash::Entry,
    pub allocator: Option<Allocator>,
    instance: Instance,
    physical_device: vk::PhysicalDevice,
    pub device: Device,
    debug_utils: ext::DebugUtils,
    utils_messenger: vk::DebugUtilsMessengerEXT,
    surface: vk::SurfaceKHR,
    surface_loader: khr::Surface,
    swapchain: Swapchain,
    framebuffers: Vec<vk::Framebuffer>,
    render_pass: vk::RenderPass,
    graphics_pipeline: GraphicsPipeline,
    graphics_command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    graphics_queue: vk::Queue,
    frame_data: Vec<FrameData>,
    images_in_flight: Vec<vk::Fence>,
    current_image: usize,
    uniform_buffer: Buffer,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    pub models: Vec<Model<Vertex, InstanceData>>,
}

impl Renderer {
    fn create_instance(
        engine_name: &str,
        app_name: &str,
        entry: &ash::Entry,
        layer_names: &[*const i8],
        mut debug_create_info: vk::DebugUtilsMessengerCreateInfoEXT,
    ) -> Result<ash::Instance, ash::vk::Result> {
        // TODO Return errors
        let engine_name_c = CString::new(engine_name).unwrap();
        let app_name_c = CString::new(app_name).unwrap();

        // Make Application Info
        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name_c)
            .application_version(vk::make_api_version(0, 0, 0, 1))
            .engine_name(&engine_name_c)
            .engine_version(vk::make_api_version(0, 0, 42, 0))
            .api_version(vk::API_VERSION_1_3);

        let instance_extension_names = [
            ext::DebugUtils::name().as_ptr(),
            khr::Surface::name().as_ptr(),
            #[cfg(target_os = "windows")]
            khr::Win32Surface::name().as_ptr(),
            #[cfg(not(target_os = "windows"))]
            khr::XlibSurface::name().as_ptr(),
            #[cfg(not(target_os = "windows"))]
            khr::WaylandSurface::name().as_ptr(),
        ];

        // Create instance
        let instance_create_info = vk::InstanceCreateInfo::builder()
            .push_next(&mut debug_create_info)
            .application_info(&app_info)
            .enabled_layer_names(layer_names)
            .enabled_extension_names(&instance_extension_names);

        unsafe { entry.create_instance(&instance_create_info, None) }
    }

    fn pick_physical_device(
        instance: &Instance,
    ) -> VkResult<(vk::PhysicalDevice, vk::PhysicalDeviceProperties)> {
        // Physical Device
        let phys_devs = unsafe { instance.enumerate_physical_devices()? };

        let mut chosen = None;
        for p in phys_devs {
            let props = unsafe { instance.get_physical_device_properties(p) };
            dbg!(props);
            if chosen.is_none() {
                chosen = Some((p, props));
            }
        }
        chosen.ok_or(vk::Result::ERROR_UNKNOWN)
    }

    fn pick_queues(
        instance: &Instance,
        physical_device: &vk::PhysicalDevice,
        surface: &vk::SurfaceKHR,
        surface_loader: &khr::Surface,
    ) -> VkResult<(u32, u32)> {
        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(*physical_device) };
        let mut g_index = None;
        let mut t_index = None;
        for (i, qfam) in queue_family_properties.iter().enumerate() {
            if qfam.queue_count > 0
                && qfam.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                && unsafe {
                    surface_loader.get_physical_device_surface_support(
                        *physical_device,
                        i as u32,
                        *surface,
                    )?
                }
            {
                g_index = Some(i as u32);
            }
            if qfam.queue_count > 0
                && qfam.queue_flags.contains(vk::QueueFlags::TRANSFER)
                && (t_index.is_none() || !qfam.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            {
                t_index = Some(i as u32);
            }
        }
        Ok((
            g_index.ok_or(vk::Result::ERROR_UNKNOWN)?,
            t_index.ok_or(vk::Result::ERROR_UNKNOWN)?,
        ))
    }

    fn create_logical_device(
        instance: &Instance,
        physical_device: &vk::PhysicalDevice,
        layers: &[*const i8],
        graphics_queue_index: u32,
        transfer_queue_index: u32,
    ) -> VkResult<Device> {
        let device_extension_names = [ash::extensions::khr::Swapchain::name().as_ptr()];
        // create logical device
        let priorities = [1.0f32];
        let queue_infos = if graphics_queue_index != transfer_queue_index {
            vec![
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(graphics_queue_index)
                    .queue_priorities(&priorities)
                    .build(),
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(transfer_queue_index)
                    .queue_priorities(&priorities)
                    .build(),
            ]
        } else {
            vec![vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(graphics_queue_index)
                .queue_priorities(&priorities)
                .build()]
        };

        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_extension_names)
            .enabled_layer_names(layers);
        let device =
            unsafe { instance.create_device(*physical_device, &device_create_info, None)? };
        Ok(device)
    }

    fn create_render_pass(
        device: &Device,
        format: &vk::SurfaceFormatKHR,
    ) -> VkResult<vk::RenderPass> {
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
        unsafe { device.create_render_pass(&renderpass_info, None) }
    }

    fn create_framebuffers(
        device: &Device,
        image_views: &[vk::ImageView],
        depth_image_view: &vk::ImageView,
        extent: vk::Extent2D,
        render_pass: &vk::RenderPass,
    ) -> VkResult<Vec<vk::Framebuffer>> {
        // Since Result implements FromIterator, can collect directly into a result
        image_views
            .iter()
            .map(|iv| {
                let iview = [*iv, *depth_image_view];
                let framebuffer_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(*render_pass)
                    .attachments(&iview)
                    .width(extent.width)
                    .height(extent.height)
                    .layers(1);
                unsafe { device.create_framebuffer(&framebuffer_info, None) }
            })
            .collect()
    }

    fn create_frame_data(device: &Device, num: usize) -> VkResult<Vec<FrameData>> {
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
        surface: *mut c_void,
        display: *mut c_void,
        use_wayland: bool,
    ) -> RendererResult<Self> {
        // Layers
        let layers = unsafe {
            [CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0").as_ptr()]
        };

        let entry = unsafe { ash::Entry::load()? };
        // Messenger info
        let debug_create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            )
            .pfn_user_callback(Some(vulkan_debug_utils_callback));

        let instance =
            Self::create_instance(name, "My Engine", &entry, &layers[..], *debug_create_info)?;

        // Create debug messenger
        let debug_utils = ext::DebugUtils::new(&entry, &instance);
        let utils_messenger =
            unsafe { debug_utils.create_debug_utils_messenger(&debug_create_info, None)? };

        #[cfg(not(target_os = "windows"))]
        let surface = if use_wayland {
            let wayland_create_info = vk::WaylandSurfaceCreateInfoKHR::builder()
                .display(display)
                .surface(surface);
            let wayland_surface_loader =
                ash::extensions::khr::WaylandSurface::new(&entry, &instance);
            unsafe { wayland_surface_loader.create_wayland_surface(&wayland_create_info, None)? }
        } else {
            let x11_create_info = vk::XlibSurfaceCreateInfoKHR::builder()
                .window(surface as u64)
                .dpy(display as *mut *const c_void);
            let xlib_surface_loader = ash::extensions::khr::XlibSurface::new(&entry, &instance);
            unsafe { xlib_surface_loader.create_xlib_surface(&x11_create_info, None)? }
        };
        #[cfg(target_os = "windows")]
        let surface = {
            let win32_create_info = vk::Win32SurfaceCreateInfoKHR::builder()
                .hinstance(display)
                .hwnd(surface);
            let win32_surface_loader = ash::extensions::khr::Win32Surface::new(&entry, &instance);
            unsafe { win32_surface_loader.create_win32_surface(&win32_create_info, None)? }
        };

        let surface_loader = ash::extensions::khr::Surface::new(&entry, &instance);

        let (physical_device, _physical_device_properties) = Self::pick_physical_device(&instance)?;
        let (graphics_queue_index, transfer_queue_index) =
            Self::pick_queues(&instance, &physical_device, &surface, &surface_loader)?;

        let device = Self::create_logical_device(
            &instance,
            &physical_device,
            &layers[..],
            graphics_queue_index,
            transfer_queue_index,
        )?;
        let graphics_queue = unsafe { device.get_device_queue(graphics_queue_index, 0) };

        // Allocator
        let mut allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: gpu_allocator::AllocatorDebugSettings {
                log_memory_information: true,
                log_leaks_on_shutdown: true,
                store_stack_traces: false,
                log_allocations: true,
                log_frees: true,
                log_stack_traces: false,
            },
            buffer_device_address: false,
        })
        .unwrap(); // TODO error handling
                   // let allocator = vk_mem::Allocator::new(allocator_create_info)?;

        let swapchain = Swapchain::new(
            &instance,
            &physical_device,
            &device,
            &surface,
            &surface_loader,
            graphics_queue_index,
            &mut allocator,
        )?;

        let render_pass = Self::create_render_pass(&device, &swapchain.get_image_format())?;
        let framebuffers = Self::create_framebuffers(
            &device,
            swapchain.get_image_views(),
            swapchain.get_depth_image_view(),
            swapchain.get_extent(),
            &render_pass,
        )?;

        let shader_module = ShaderModule::new(&device)?;
        let graphics_pipeline = GraphicsPipeline::new(
            &device,
            swapchain.get_extent(),
            &render_pass,
            shader_module.get_stages(),
            &Vertex::get_attribute_descriptions(),
            &Vertex::get_binding_description(),
        )?;

        // Create command pools
        let graphics_commandpool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(graphics_queue_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let graphics_command_pool =
            unsafe { device.create_command_pool(&graphics_commandpool_info, None)? };

        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(graphics_command_pool)
            .command_buffer_count(framebuffers.len() as u32);
        let command_buffers =
            unsafe { device.allocate_command_buffers(&command_buffer_allocate_info)? };

        let frame_data = Self::create_frame_data(&device, FRAMES_IN_FLIGHT)?;
        let images_in_flight = vec![vk::Fence::null(); swapchain.get_actual_image_count() as usize];

        // Create uniform buffer
        let camera_transform: [[f32; 4]; 4] = glm::Mat4::identity().into();
        let mut uniform_buffer = Buffer::new(
            &device,
            &mut allocator,
            std::mem::size_of::<[[f32; 4]; 4]>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
        )?;

        let bytes = std::mem::size_of::<[[f32; 4]; 4]>();
        let data =
            unsafe { std::slice::from_raw_parts(camera_transform.as_ptr() as *const u8, bytes) };
        uniform_buffer.fill(&mut allocator, data)?;

        // Create descriptor pool
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: swapchain.get_actual_image_count(),
        }];

        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(swapchain.get_actual_image_count())
            .pool_sizes(&pool_sizes);
        let descriptor_pool =
            unsafe { device.create_descriptor_pool(&descriptor_pool_info, None)? };

        // Now create the descriptor sets, one for each swapchain image
        let descriptor_layouts = vec![
            graphics_pipeline.descriptor_set_layouts[0];
            swapchain.get_actual_image_count() as usize
        ];
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&descriptor_layouts);
        let descriptor_sets =
            unsafe { device.allocate_descriptor_sets(&descriptor_set_allocate_info)? };

        for (i, ds) in descriptor_sets.iter().enumerate() {
            let buffer_info = [vk::DescriptorBufferInfo {
                buffer: uniform_buffer.buffer,
                offset: 0,
                range: 64,
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

        Ok(Renderer {
            dropped: false,
            entry,
            instance,
            allocator: Some(allocator),
            physical_device,
            device,
            debug_utils,
            utils_messenger,
            surface,
            surface_loader,
            swapchain,
            graphics_command_pool,
            command_buffers,
            render_pass,
            framebuffers,
            graphics_pipeline,
            graphics_queue,
            frame_data,
            images_in_flight,
            current_image: 0,
            uniform_buffer,
            descriptor_pool,
            descriptor_sets,
            models: vec![],
        })
    }

    fn wait_for_next_frame_fence(&self) -> VkResult<()> {
        unsafe {
            self.device.wait_for_fences(
                &[self.frame_data[self.current_image].in_flight_fence],
                true,
                std::u64::MAX,
            )?;
        }
        Ok(())
    }

    fn wait_for_image_fence_and_set_new_fence(&mut self, image_index: usize) -> VkResult<()> {
        if self.images_in_flight[image_index] != vk::Fence::null() {
            unsafe {
                self.device.wait_for_fences(
                    &[self.images_in_flight[image_index]],
                    true,
                    std::u64::MAX,
                )?;
            }
        }

        self.images_in_flight[image_index] = self.frame_data[self.current_image].in_flight_fence;
        Ok(())
    }

    fn update_command_buffer(&self, image_index: usize) -> VkResult<()> {
        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder();
        let cmd_buf = &self.command_buffers[image_index];
        let framebuffer = &self.framebuffers[image_index];
        unsafe {
            self.device
                .begin_command_buffer(*cmd_buf, &command_buffer_begin_info)?;
        }
        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.08, 1.0],
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
            self.device.cmd_begin_render_pass(
                *cmd_buf,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );
            self.device.cmd_bind_pipeline(
                *cmd_buf,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline.pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                *cmd_buf,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline.pipeline_layout,
                0,
                &[self.descriptor_sets[image_index]],
                &[],
            );
            for m in &self.models {
                m.draw(&self.device, *cmd_buf);
            }
            self.device.cmd_end_render_pass(*cmd_buf);
            self.device.end_command_buffer(*cmd_buf)?;
        }
        Ok(())
    }

    fn submit_commands(&mut self, image_index: usize) -> VkResult<()> {
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
            self.device
                .reset_fences(&[this_frame_data.in_flight_fence])?;
            self.update_command_buffer(image_index)?;
            if let Some(alloc) = &mut self.allocator {
                for m in &mut self.models {
                    m.update_instance_buffer(&self.device, alloc)?;
                }
            } else {
                panic!("No allocator!");
            }
            self.device.queue_submit(
                self.graphics_queue,
                &submit_info,
                this_frame_data.in_flight_fence,
            )?;
        }
        Ok(())
    }

    fn present(&self, image_index: u32) -> VkResult<()> {
        self.swapchain.present(
            &self.graphics_queue,
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

        self.submit_commands(image_index as usize)?;

        self.present(image_index)?;
        self.current_image = (self.current_image + 1) % FRAMES_IN_FLIGHT;
        Ok(())
    }

    pub fn update_uniforms_from_camera(&mut self, camera: &Camera) -> VkResult<()> {
        if let Some(alloc) = &mut self.allocator {
            camera.update_buffer(alloc, &mut self.uniform_buffer)
        } else {
            panic!("No allocator!");
        }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        if self.dropped {
            return;
        }
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Something wrong while waiting for idle");

            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);

            let mut allocator = self.allocator.take().expect("We had no allocator?!");
            for m in &mut self.models {
                if let Some(vb) = &mut m.vertex_buffer {
                    vb.destroy(&mut allocator);
                }
                if let Some(ib) = &mut m.instance_buffer {
                    ib.destroy(&mut allocator);
                }
            }
            self.uniform_buffer.destroy(&mut allocator);

            self.frame_data.clear();
            self.device
                .destroy_command_pool(self.graphics_command_pool, None);
            // device.destroy_command_pool(command_pool_transfer, None);
            self.graphics_pipeline.destroy();
            for fb in self.framebuffers.iter() {
                self.device.destroy_framebuffer(*fb, None);
            }
            self.device.destroy_render_pass(self.render_pass, None);
            self.swapchain.destroy(&mut allocator);
            self.surface_loader.destroy_surface(self.surface, None);
            drop(allocator); // Ensure all memory is freed
            self.device.destroy_device(None);
            self.debug_utils
                .destroy_debug_utils_messenger(self.utils_messenger, None);
            self.instance.destroy_instance(None);
        }
        self.dropped = true;
    }
}

use ash::extensions::khr;
use ash::vk;

use gpu_allocator::vulkan::Allocator;

use super::context::VulkanContext;
use super::render_target::RenderTarget;
use super::RendererResult;

pub struct Swapchain {
    swapchain: vk::SwapchainKHR,
    swapchain_loader: khr::Swapchain,
    min_image_count: u32,
    image_count: u32,
    render_targets: Vec<RenderTarget>,
    image_format: vk::SurfaceFormatKHR,
    extent: vk::Extent2D,
}

impl Swapchain {
    pub fn new(
        context: &VulkanContext,
        allocator: &mut Allocator,
        format: vk::SurfaceFormatKHR,
        width: u32,
        height: u32,
        render_pass: &vk::RenderPass,
    ) -> RendererResult<Self> {
        let extent = vk::Extent2D {
            width: width
                .min(context.surface_capabilities.max_image_extent.width)
                .max(context.surface_capabilities.min_image_extent.width),
            height: height
                .min(context.surface_capabilities.max_image_extent.height)
                .max(context.surface_capabilities.min_image_extent.height),
        };
        let queue_families = [context.graphics_queue.index];
        let min_image_count = 3.min(context.surface_capabilities.min_image_count).max(
            if context.surface_capabilities.max_image_count == 0 {
                context.surface_capabilities.min_image_count
            } else {
                context.surface_capabilities.max_image_count
            },
        );
        let present_mode = {
            if context
                .surface_present_modes
                .contains(&vk::PresentModeKHR::MAILBOX)
            {
                vk::PresentModeKHR::MAILBOX
            } else {
                vk::PresentModeKHR::FIFO
            }
        };
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(context.surface)
            .min_image_count(min_image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queue_families)
            .pre_transform(context.surface_capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode);
        let swapchain_loader =
            ash::extensions::khr::Swapchain::new(&context.instance, &context.device);
        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None)? };

        // get images
        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };
        let render_targets = images
            .into_iter()
            .map(|image| {
                RenderTarget::new_from_image(
                    context,
                    allocator,
                    image,
                    format.format,
                    extent,
                    render_pass,
                )
            })
            .collect::<RendererResult<Vec<_>>>()?;

        Ok(Swapchain {
            swapchain,
            swapchain_loader,
            min_image_count,
            image_count: render_targets.len() as u32,
            render_targets,
            image_format: format,
            extent,
        })
    }

    pub fn get_swapchain(&self) -> &vk::SwapchainKHR {
        &self.swapchain
    }

    pub fn get_minimum_image_count(&self) -> u32 {
        self.min_image_count
    }

    pub fn get_actual_image_count(&self) -> u32 {
        self.image_count
    }

    pub fn get_render_targets(&self) -> &[RenderTarget] {
        &self.render_targets[..]
    }

    pub fn get_image_format(&self) -> vk::SurfaceFormatKHR {
        self.image_format
    }

    pub fn get_extent(&self) -> vk::Extent2D {
        self.extent
    }

    pub fn get_next_image(
        &self,
        timeout: u64,
        semaphore: &vk::Semaphore,
        fence: vk::Fence,
    ) -> RendererResult<u32> {
        let (image_index, _) = unsafe {
            self.swapchain_loader
                .acquire_next_image(self.swapchain, timeout, *semaphore, fence)?
        };
        Ok(image_index)
    }

    pub fn present(
        &self,
        queue: &vk::Queue,
        semaphore: &vk::Semaphore,
        image_index: u32,
    ) -> RendererResult<()> {
        let swapchains = [self.swapchain];
        let indices = [image_index];
        let semaphores = [*semaphore];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&semaphores)
            .swapchains(&swapchains)
            .image_indices(&indices);
        unsafe {
            self.swapchain_loader.queue_present(*queue, &present_info)?;
        }
        Ok(())
    }

    pub fn destroy(&mut self, context: &VulkanContext, allocator: &mut Allocator) {
        for rt in &mut self.render_targets {
            rt.destroy(context, allocator);
        }
        unsafe {
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        //self.destroy();
    }
}

use ash::vk;
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator},
    MemoryLocation,
};

use super::{context::VulkanContext, RendererResult};

pub struct RenderTarget {
    pub extent: vk::Extent3D,
    pub image: vk::Image,
    should_destroy_image: bool,
    pub image_allocation: Option<Allocation>,
    pub image_format: vk::Format,
    pub image_view: vk::ImageView,
    pub framebuffer: vk::Framebuffer,
    pub depth_image: Option<vk::Image>,
    pub depth_image_allocation: Option<Allocation>,
    pub depth_image_view: Option<vk::ImageView>,
}

impl RenderTarget {
    pub fn new_from_image(
        context: &VulkanContext,
        allocator: &mut Allocator,
        image: vk::Image,
        format: vk::Format,
        extent: vk::Extent2D,
        render_pass: &vk::RenderPass,
    ) -> RendererResult<Self> {
        let subresource_range = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);
        let image_view_create_info = vk::ImageViewCreateInfo::builder()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(*subresource_range);
        let image_view = unsafe {
            context
                .device
                .create_image_view(&image_view_create_info, None)
        }?;

        let extent = vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        };

        let queue_family_indices = [context.graphics_queue.index];

        let depth_image_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::D32_SFLOAT)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queue_family_indices);

        let depth_image = unsafe { context.device.create_image(&depth_image_info, None) }?;
        let reqs = unsafe { context.device.get_image_memory_requirements(depth_image) };
        let depth_image_allocation = allocator.allocate(&AllocationCreateDesc {
            name: "depth_image",
            requirements: reqs,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;
        unsafe {
            context.device.bind_image_memory(
                depth_image,
                depth_image_allocation.memory(),
                depth_image_allocation.offset(),
            )?;
        }
        let subresource_range = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::DEPTH)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);
        let image_view_create_info = vk::ImageViewCreateInfo::builder()
            .image(depth_image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::D32_SFLOAT)
            .subresource_range(*subresource_range);
        let depth_image_view = unsafe {
            context
                .device
                .create_image_view(&image_view_create_info, None)
        }?;

        let iview = [image_view, depth_image_view];
        let framebuffer_info = vk::FramebufferCreateInfo::builder()
            .render_pass(*render_pass)
            .attachments(&iview)
            .width(extent.width)
            .height(extent.height)
            .layers(1);
        let framebuffer = unsafe { context.device.create_framebuffer(&framebuffer_info, None) }?;

        Ok(Self {
            extent,
            image,
            should_destroy_image: false,
            image_allocation: None,
            image_format: format,
            image_view,
            framebuffer,
            depth_image: Some(depth_image),
            depth_image_allocation: Some(depth_image_allocation),
            depth_image_view: Some(depth_image_view),
        })
    }

    pub fn destroy(&mut self, context: &VulkanContext, allocator: &mut Allocator) {
        if let Some(depth_image_allocation) = self.depth_image_allocation.take() {
            allocator
                .free(depth_image_allocation)
                .expect("Could not free memory");
        }
        if let Some(depth_image_view) = self.depth_image_view.take() {
            unsafe {
                context.device.destroy_image_view(depth_image_view, None);
            }
        }
        if let Some(depth_image) = self.depth_image.take() {
            unsafe {
                context.device.destroy_image(depth_image, None);
            }
        }

        if let Some(allocation) = self.image_allocation.take() {
            allocator.free(allocation).expect("Could not free memory");
        }

        unsafe {
            context.device.destroy_framebuffer(self.framebuffer, None);
            context.device.destroy_image_view(self.image_view, None);
            if self.should_destroy_image {
                context.device.destroy_image(self.image, None);
            }
        }
    }
}

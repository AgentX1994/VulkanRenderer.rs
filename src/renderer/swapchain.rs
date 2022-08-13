use ash::extensions::khr;
use ash::prelude::VkResult;
use ash::vk;
use ash::{Device, Instance};

pub struct Swapchain {
    device: Device,
    swapchain: vk::SwapchainKHR,
    swapchain_loader: khr::Swapchain,
    min_image_count: u32,
    image_count: u32,
    images: Vec<vk::Image>,
    image_format: vk::SurfaceFormatKHR,
    extent: vk::Extent2D,
    image_views: Vec<vk::ImageView>,
}

impl Swapchain {
    fn create_image_views(
        device: &Device,
        format: &vk::SurfaceFormatKHR,
        images: &[vk::Image],
    ) -> Vec<vk::ImageView> {
        images
            .iter()
            .map(|image| {
                let subresource_range = vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1);
                let image_view_create_info = vk::ImageViewCreateInfo::builder()
                    .image(*image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format.format)
                    .subresource_range(*subresource_range);
                // TODO use ? for this
                unsafe {
                    device
                        .create_image_view(&image_view_create_info, None)
                        .unwrap()
                }
            })
            .collect::<Vec<_>>()
    }

    pub fn new(
        instance: &Instance,
        physical_device: &vk::PhysicalDevice,
        device: &Device,
        surface: &vk::SurfaceKHR,
        surface_loader: &khr::Surface,
        graphics_queue_index: u32,
    ) -> VkResult<Self> {
        // Get capabilities of the surface
        let surface_capabilities = unsafe {
            surface_loader.get_physical_device_surface_capabilities(*physical_device, *surface)?
        };
        let _surface_present_modes = unsafe {
            surface_loader.get_physical_device_surface_present_modes(*physical_device, *surface)?
        };
        let surface_formats = unsafe {
            surface_loader.get_physical_device_surface_formats(*physical_device, *surface)?
        };

        let format = surface_formats
            .iter()
            .find(|format| {
                format.format == vk::Format::B8G8R8A8_SRGB
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .ok_or(vk::Result::ERROR_FORMAT_NOT_SUPPORTED)?;
        let extent = vk::Extent2D {
            width: surface_capabilities
                .current_extent
                .width
                .min(surface_capabilities.min_image_extent.width)
                .max(surface_capabilities.max_image_extent.width),
            height: surface_capabilities
                .current_extent
                .height
                .min(surface_capabilities.min_image_extent.height)
                .max(surface_capabilities.max_image_extent.height),
        };
        let queue_families = [graphics_queue_index];
        let min_image_count = 3.min(surface_capabilities.min_image_count).max(
            if surface_capabilities.max_image_count == 0 {
                surface_capabilities.min_image_count
            } else {
                surface_capabilities.max_image_count
            },
        );
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(*surface)
            .min_image_count(min_image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queue_families)
            .pre_transform(surface_capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO);
        let swapchain_loader = ash::extensions::khr::Swapchain::new(instance, device);
        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None)? };

        // get images
        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };
        let image_views = Self::create_image_views(device, format, &images[..]);

        Ok(Swapchain {
            device: device.clone(),
            swapchain,
            swapchain_loader,
            min_image_count,
            image_count: images.len() as u32,
            images,
            image_views,
            image_format: *format,
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

    pub fn get_images(&self) -> &[vk::Image] {
        &self.images[..]
    }

    pub fn get_image_format(&self) -> vk::SurfaceFormatKHR {
        self.image_format
    }

    pub fn get_extent(&self) -> vk::Extent2D {
        self.extent
    }

    pub fn get_image_views(&self) -> &[vk::ImageView] {
        &self.image_views[..]
    }

    pub fn get_next_image(
        &self,
        timeout: u64,
        semaphore: &vk::Semaphore,
        fence: vk::Fence,
    ) -> VkResult<u32> {
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
    ) -> VkResult<()> {
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

    pub fn destroy(&mut self) {
        for iv in self.image_views.iter() {
            unsafe { self.device.destroy_image_view(*iv, None) };
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

use ash::extensions::khr;
use ash::vk;
use ash::{Device, Instance};

use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, Allocator};
use gpu_allocator::MemoryLocation;

use super::RendererResult;

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
    depth_image: vk::Image,
    depth_image_allocation: Option<Allocation>,
    depth_image_view: vk::ImageView,
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
        allocator: &mut Allocator,
        width: u32,
        height: u32
    ) -> RendererResult<Self> {
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
            width: width
                .min(surface_capabilities.max_image_extent.width)
                .max(surface_capabilities.min_image_extent.width),
            height: height
                .min(surface_capabilities.max_image_extent.height)
                .max(surface_capabilities.min_image_extent.height),
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
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)
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

        // Create depth image
        let extent_3d = vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        };
        let depth_image_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::D32_SFLOAT)
            .extent(extent_3d)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queue_families);

        let depth_image = unsafe { device.create_image(&depth_image_info, None) }?;
        let reqs = unsafe { device.get_image_memory_requirements(depth_image) };
        let depth_image_allocation = allocator
            .allocate(&AllocationCreateDesc {
                name: "depth_image",
                requirements: reqs,
                location: MemoryLocation::GpuOnly,
                linear: false,
            })?;
        unsafe {
            device.bind_image_memory(
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
        let depth_image_view = unsafe { device.create_image_view(&image_view_create_info, None) }?;

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
            depth_image,
            depth_image_allocation: Some(depth_image_allocation),
            depth_image_view,
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
    ) -> RendererResult<u32> {
        let (image_index, _) = unsafe {
            self.swapchain_loader
                .acquire_next_image(self.swapchain, timeout, *semaphore, fence)?
        };
        Ok(image_index)
    }

    pub fn get_depth_image_view(&self) -> &vk::ImageView {
        &self.depth_image_view
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

    pub fn destroy(&mut self, allocator: &mut Allocator) {
        allocator
            .free(
                self.depth_image_allocation
                    .take()
                    .expect("No depth image allocation!"),
            )
            .expect("Could not free memory");
        unsafe {
            self.device.destroy_image_view(self.depth_image_view, None);
            self.device.destroy_image(self.depth_image, None);
        }
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

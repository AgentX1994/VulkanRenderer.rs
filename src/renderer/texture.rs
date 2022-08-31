use ash::{vk, Device};
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, Allocator},
    MemoryLocation,
};

use super::{buffer::Buffer, RendererResult};

pub struct Texture {
    image: image::RgbaImage,
    vk_image: vk::Image,
    pub image_view: vk::ImageView,
    pub sampler: vk::Sampler,
    allocation: Option<Allocation>,
}

impl Texture {
    pub fn from_file<P: AsRef<std::path::Path>>(
        path: P,
        device: &Device,
        allocator: &mut Allocator,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> RendererResult<Self> {
        // Load image from file
        let image = image::open(path)
            .map(|img| img.into_rgba8())
            .expect("unable to open image");
        let (width, height) = image.dimensions();

        // Create vulkan image
        let image_create_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(vk::Format::R8G8B8A8_SRGB)
            .samples(vk::SampleCountFlags::TYPE_1)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED);
        let vk_image = unsafe { device.create_image(&image_create_info, None)? };

        // Allocate memory for image
        let reqs = unsafe { device.get_image_memory_requirements(vk_image) };
        let allocation = allocator
            .allocate(&AllocationCreateDesc {
                name: "texture",
                requirements: reqs,
                location: MemoryLocation::GpuOnly,
                linear: false,
            })?;
        unsafe {
            device.bind_image_memory(vk_image, allocation.memory(), allocation.offset())?;
        };

        // Create image view
        let view_create_info = vk::ImageViewCreateInfo::builder()
            .image(vk_image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_SRGB)
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

        // Create buffer to copy data into image
        let data = image.clone().into_raw();
        let mut buffer = Buffer::new(
            device,
            allocator,
            data.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        )?;
        buffer.fill(allocator, &data)?;

        // Create command buffer to use for copy
        let command_buf_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .command_buffer_count(1);
        let copy_cmd_buf =
            unsafe { device.allocate_command_buffers(&command_buf_allocate_info) }?[0];

        // Begin command buffer
        let cmd_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { device.begin_command_buffer(copy_cmd_buf, &cmd_begin_info) }?;

        // Transition image layout to transfer dst
        let barrier = vk::ImageMemoryBarrier::builder()
            .image(vk_image)
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
                copy_cmd_buf,
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
            device.cmd_copy_buffer_to_image(
                copy_cmd_buf,
                buffer.buffer,
                vk_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            )
        }

        // Transition image layout for use as texture
        let barrier = vk::ImageMemoryBarrier::builder()
            .image(vk_image)
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
                copy_cmd_buf,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            )
        };

        // End command buffer
        unsafe { device.end_command_buffer(copy_cmd_buf) }?;

        // Prepare to submit command buffer
        let submit_infos = [vk::SubmitInfo::builder()
            .command_buffers(&[copy_cmd_buf])
            .build()];
        // Fence to wait for command buffer to finish
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }?;

        // Submit the commands and wait for completion
        unsafe { device.queue_submit(queue, &submit_infos, fence) }?;
        unsafe { device.wait_for_fences(&[fence], true, std::u64::MAX) }?;

        // Cleanup
        unsafe { device.destroy_fence(fence, None) };
        buffer.destroy(allocator);
        unsafe { device.free_command_buffers(command_pool, &[copy_cmd_buf]) };

        // Done
        Ok(Texture {
            image,
            vk_image,
            image_view,
            sampler,
            allocation: Some(allocation),
        })
    }

    pub fn destroy(&mut self, device: &Device, allocator: &mut Allocator) {
        allocator
            .free(self.allocation.take().expect("Texture had no allocation!"))
            .expect("Could not free texture allocation");
        unsafe {
            device.destroy_sampler(self.sampler, None);
            device.destroy_image_view(self.image_view, None);
            device.destroy_image(self.vk_image, None);
        }
    }
}

#[derive(Default)]
pub struct TextureStorage {
    textures: Vec<Texture>
}

impl TextureStorage {
    pub fn new_texture_from_file<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
        device: &Device,
        allocator: &mut Allocator,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> RendererResult<usize> {
        let texture = Texture::from_file(path, device, allocator, command_pool, queue)?;
        let new_id = self.textures.len();
        self.textures.push(texture);
        Ok(new_id)
    }

    pub fn get_number_of_textures(&self) -> usize {
        self.textures.len()
    }

    pub fn get_texture(&self, index: usize) -> Option<&Texture> {
        self.textures.get(index)
    }


    pub fn get_texture_mut(&mut self, index: usize) -> Option<&mut Texture> {
        self.textures.get_mut(index)
    }

    pub fn get_descriptor_image_info(&self) -> Vec<vk::DescriptorImageInfo> {
        self.textures
            .iter()
            .map(|tex| vk::DescriptorImageInfo {
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                image_view: tex.image_view,
                sampler: tex.sampler
            })
            .collect()
    }

    pub fn clean_up(&mut self, device: &Device, allocator: &mut Allocator) {
        for texture in &mut self.textures {
            texture.destroy(device, allocator);
        }
    }
}
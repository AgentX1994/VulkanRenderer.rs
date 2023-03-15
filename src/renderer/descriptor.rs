use std::collections::HashMap;

use ash;
use ash::vk;

use super::RendererResult;

#[derive(Default)]
struct DescriptorLayoutInfo {
    bindings: Vec<vk::DescriptorSetLayoutBinding>,
}

impl PartialEq for DescriptorLayoutInfo {
    fn eq(&self, other: &Self) -> bool {
        if self.bindings.len() != other.bindings.len() {
            return false;
        }
        self.bindings
            .iter()
            .zip(other.bindings.iter())
            .all(|(a, b)| {
                if a.binding != b.binding {
                    return false;
                }
                if a.descriptor_type != b.descriptor_type {
                    return false;
                }
                if a.descriptor_count != b.descriptor_count {
                    return false;
                }
                if a.stage_flags != b.stage_flags {
                    return false;
                }
                true
            })
    }
}

impl Eq for DescriptorLayoutInfo {}

impl std::hash::Hash for DescriptorLayoutInfo {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for b in self.bindings.iter() {
            b.binding.hash(state);
            b.descriptor_type.hash(state);
            b.descriptor_count.hash(state);
            b.stage_flags.hash(state);
        }
    }
}

#[derive(Default)]
pub struct DescriptorLayoutCache {
    layout_cache: HashMap<DescriptorLayoutInfo, vk::DescriptorSetLayout>,
}

impl DescriptorLayoutCache {
    pub fn create_descriptor_layout(
        &mut self,
        device: &ash::Device,
        info: &vk::DescriptorSetLayoutCreateInfo,
    ) -> RendererResult<vk::DescriptorSetLayout> {
        let mut layout_info: DescriptorLayoutInfo = Default::default();
        layout_info.bindings.reserve(info.binding_count as usize);
        let mut is_sorted = true;
        let mut last_binding = -1i32;
        for i in 0..info.binding_count {
            let binding = unsafe { *info.p_bindings.add(i as usize) };
            layout_info.bindings.push(binding);
            if binding.binding as i32 > last_binding {
                last_binding = binding.binding as i32;
            } else {
                is_sorted = false;
            }
        }
        if !is_sorted {
            layout_info
                .bindings
                .sort_by(|a, b| a.binding.cmp(&b.binding));
        }

        match self.layout_cache.entry(layout_info) {
            std::collections::hash_map::Entry::Occupied(o) => Ok(*o.get()),
            std::collections::hash_map::Entry::Vacant(v) => {
                let layout = unsafe { device.create_descriptor_set_layout(info, None)? };
                Ok(*v.insert(layout))
            }
        }
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        for layout in self.layout_cache.values() {
            unsafe { device.destroy_descriptor_set_layout(*layout, None) };
        }
        self.layout_cache.clear();
    }
}

const DESCRIPTOR_SIZES: [(vk::DescriptorType, f32); 11] = [
    (vk::DescriptorType::SAMPLER, 0.5),
    (vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 4.0),
    (vk::DescriptorType::SAMPLED_IMAGE, 4.0),
    (vk::DescriptorType::STORAGE_IMAGE, 1.0),
    (vk::DescriptorType::UNIFORM_TEXEL_BUFFER, 1.0),
    (vk::DescriptorType::STORAGE_TEXEL_BUFFER, 1.0),
    (vk::DescriptorType::UNIFORM_BUFFER, 2.0),
    (vk::DescriptorType::STORAGE_BUFFER, 2.0),
    (vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC, 1.0),
    (vk::DescriptorType::STORAGE_BUFFER_DYNAMIC, 1.0),
    (vk::DescriptorType::INPUT_ATTACHMENT, 0.5),
];

fn create_pool(
    device: &ash::Device,
    count: u32,
    flags: vk::DescriptorPoolCreateFlags,
) -> RendererResult<vk::DescriptorPool> {
    let sizes: Vec<_> = DESCRIPTOR_SIZES
        .iter()
        .map(|(ty, fraction)| vk::DescriptorPoolSize {
            ty: *ty,
            descriptor_count: (fraction * count as f32) as u32,
        })
        .collect();
    let pool_info = vk::DescriptorPoolCreateInfo::builder()
        .flags(flags)
        .max_sets(count)
        .pool_sizes(&sizes);

    unsafe {
        device
            .create_descriptor_pool(&pool_info, None)
            .map_err(|e| e.into())
    }
}

#[derive(Default)]
pub struct DescriptorAllocator {
    current_pool: vk::DescriptorPool,
    used_pools: Vec<vk::DescriptorPool>,
    free_pools: Vec<vk::DescriptorPool>,
}

impl DescriptorAllocator {
    fn grab_pool(&mut self, device: &ash::Device) -> RendererResult<vk::DescriptorPool> {
        if let Some(p) = self.free_pools.pop() {
            Ok(p)
        } else {
            create_pool(device, 1000, vk::DescriptorPoolCreateFlags::empty())
        }
    }

    pub fn reset_pools(&mut self, device: &ash::Device) -> RendererResult<()> {
        assert!(self.free_pools.is_empty());
        for p in self.used_pools.iter() {
            unsafe {
                device.reset_descriptor_pool(*p, vk::DescriptorPoolResetFlags::empty())?;
            }
        }
        self.free_pools.clear();
        for p in self.used_pools.drain(0..self.used_pools.len()) {
            self.free_pools.push(p);
        }
        Ok(())
    }

    pub fn allocate(
        &mut self,
        device: &ash::Device,
        layout: vk::DescriptorSetLayout,
    ) -> RendererResult<vk::DescriptorSet> {
        if self.current_pool == vk::DescriptorPool::null() {
            self.current_pool = self.grab_pool(device)?;
            self.used_pools.push(self.current_pool);
        }

        let layouts = [layout];

        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .set_layouts(&layouts)
            .descriptor_pool(self.current_pool);

        match unsafe { device.allocate_descriptor_sets(&alloc_info) } {
            Ok(sets) => Ok(sets[0]),
            Err(res) => match res {
                vk::Result::ERROR_FRAGMENTED_POOL | vk::Result::ERROR_OUT_OF_POOL_MEMORY => {
                    // allocate a new pool and retry
                    self.current_pool = self.grab_pool(device)?;
                    self.used_pools.push(self.current_pool);
                    unsafe {
                        device
                            .allocate_descriptor_sets(&alloc_info)
                            .map(|sets| sets[0])
                            .map_err(|e| e.into())
                    }
                }
                _ => Err(res.into()),
            },
        }
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        for p in self.free_pools.drain(0..self.free_pools.len()) {
            unsafe {
                device.destroy_descriptor_pool(p, None);
            }
        }
        for p in self.used_pools.drain(0..self.used_pools.len()) {
            unsafe {
                device.destroy_descriptor_pool(p, None);
            }
        }
    }
}

pub struct DescriptorBuilder<'a> {
    writes: Vec<vk::WriteDescriptorSet>,
    bindings: Vec<vk::DescriptorSetLayoutBinding>,

    cache: &'a mut DescriptorLayoutCache,
    alloc: &'a mut DescriptorAllocator,
}

impl<'a> DescriptorBuilder<'a> {
    pub fn begin(
        layout_cache: &'a mut DescriptorLayoutCache,
        allocator: &'a mut DescriptorAllocator,
    ) -> Self {
        Self {
            writes: vec![],
            bindings: vec![],
            cache: layout_cache,
            alloc: allocator,
        }
    }

    pub fn bind_buffer(
        &mut self,
        binding: u32,
        buffer_info: &[vk::DescriptorBufferInfo],
        ty: vk::DescriptorType,
        stage_flags: vk::ShaderStageFlags,
    ) -> &mut Self {
        let new_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(binding)
            .descriptor_type(ty)
            .descriptor_count(1)
            .stage_flags(stage_flags)
            .build();
        self.bindings.push(new_binding);

        let new_write = vk::WriteDescriptorSet::builder()
            .buffer_info(buffer_info)
            .descriptor_type(ty)
            .dst_binding(binding)
            .build();
        self.writes.push(new_write);
        self
    }

    pub fn bind_image(
        &mut self,
        binding: u32,
        image_infos: &[vk::DescriptorImageInfo],
        ty: vk::DescriptorType,
        stage_flags: vk::ShaderStageFlags,
    ) -> &mut Self {
        let new_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(binding)
            .descriptor_type(ty)
            .descriptor_count(1)
            .stage_flags(stage_flags)
            .build();
        self.bindings.push(new_binding);

        let new_write = vk::WriteDescriptorSet::builder()
            .image_info(image_infos)
            .descriptor_type(ty)
            .dst_binding(binding)
            .build();
        self.writes.push(new_write);
        self
    }

    pub fn build(
        &mut self,
        device: &ash::Device,
    ) -> RendererResult<(vk::DescriptorSet, vk::DescriptorSetLayout)> {
        let layout_create_info =
            vk::DescriptorSetLayoutCreateInfo::builder().bindings(&self.bindings);
        let layout = self
            .cache
            .create_descriptor_layout(device, &layout_create_info)?;

        // allocate descriptor
        let set = self.alloc.allocate(device, layout)?;

        for w in self.writes.iter_mut() {
            w.dst_set = set;
        }

        unsafe {
            device.update_descriptor_sets(&self.writes, &[]);
        }

        Ok((set, layout))
    }
}

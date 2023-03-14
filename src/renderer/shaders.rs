use core::slice;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::ffi::CStr;
use std::hash::{Hash, Hasher};

use ash::vk;

use spirv_reflect::types::ReflectDescriptorType;
// To avoid a naming conflict
use spirv_reflect::ShaderModule as ShaderModuleReflection;

use super::error::{InvalidHandle, RendererError};
use super::utils::{Handle, HandleArray};
use super::RendererResult;

const MAIN_FUNCTION_NAME: &CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };

fn hash_descriptor_layout_info(info: &vk::DescriptorSetLayoutCreateInfo) -> u64 {
    let slice = unsafe {
        let ptr = info as *const vk::DescriptorSetLayoutCreateInfo as *const u8;
        slice::from_raw_parts(
            ptr,
            std::mem::size_of::<vk::DescriptorSetLayoutCreateInfo>(),
        )
    };
    let mut hasher = DefaultHasher::new();
    slice.hash(&mut hasher);
    hasher.finish()
}

pub struct ShaderModule {
    code: Vec<u32>,
    module: vk::ShaderModule,
}

impl ShaderModule {
    fn new(device: &ash::Device, code: Vec<u32>) -> RendererResult<Self> {
        let create_info = vk::ShaderModuleCreateInfo::builder().code(&code[..]);
        let module = unsafe { device.create_shader_module(&create_info, None)? };
        Ok(Self { code, module })
    }

    fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_shader_module(self.module, None);
        }
    }
}

struct ShaderStage {
    handle: Handle<ShaderModule>,
    stage: vk::ShaderStageFlags,
}

struct ReflectedBinding {
    set: u32,
    binding: u32,
    typ: vk::DescriptorType,
}

#[derive(Default)]
struct DescriptorSetLayoutData {
    set: u32,
    create_info: vk::DescriptorSetLayoutCreateInfo,
    bindings: Vec<vk::DescriptorSetLayoutBinding>,
}

pub struct ShaderEffect {
    stages: Vec<ShaderStage>,
    pub pipeline_layout: vk::PipelineLayout,
    bindings: HashMap<String, ReflectedBinding>,
    pub set_layouts: [vk::DescriptorSetLayout; 4],
    set_hashes: [u64; 4],
}

impl ShaderEffect {
    pub fn new() -> ShaderEffect {
        ShaderEffect {
            stages: Vec::new(),
            pipeline_layout: vk::PipelineLayout::null(),
            bindings: HashMap::new(),
            set_layouts: [vk::DescriptorSetLayout::null(); 4],
            set_hashes: [0u64; 4],
        }
    }

    pub fn add_stage(
        &mut self,
        handle: Handle<ShaderModule>,
        stage: vk::ShaderStageFlags,
    ) -> RendererResult<()> {
        // TODO reflection stuff
        let shader_stage = ShaderStage { handle, stage };
        self.stages.push(shader_stage);
        Ok(())
    }

    pub fn get_stages(
        &self,
        shader_cache: &ShaderCache,
    ) -> RendererResult<Vec<vk::PipelineShaderStageCreateInfo>> {
        let mut ret = Vec::new();

        for shader_stage in self.stages.iter() {
            let module = shader_cache.get_shader_module_by_handle(shader_stage.handle)?;
            let create_info = vk::PipelineShaderStageCreateInfo::builder()
                .module(module.module)
                .stage(shader_stage.stage)
                .name(MAIN_FUNCTION_NAME); // TODO unhardcode entry point name?
            ret.push(create_info.build());
        }

        Ok(ret)
    }

    pub fn reflect_layout(
        &mut self,
        device: &ash::Device,
        shader_cache: &ShaderCache,
        overrides: &[(&str, vk::DescriptorType)],
    ) -> RendererResult<()> {
        let mut set_layouts = vec![];
        let mut constant_ranges = vec![];

        for shader_stage in self.stages.iter() {
            let spv_module = {
                let module = shader_cache.get_shader_module_by_handle(shader_stage.handle)?;
                ShaderModuleReflection::load_u32_data(&module.code)
                    .map_err(RendererError::SpirvError)?
            };

            for set in spv_module
                .enumerate_descriptor_sets(None)
                .map_err(RendererError::SpirvError)?
            {
                let mut layout = DescriptorSetLayoutData {
                    set: set.set,
                    create_info: vk::DescriptorSetLayoutCreateInfo::default(),
                    bindings: vec![],
                };

                layout.bindings.reserve(set.bindings.len());

                for binding in set.bindings {
                    let mut desc_type = match binding.descriptor_type {
                        ReflectDescriptorType::Undefined => {
                            return Err(RendererError::SpirvError("Unknown Descriptor Type"))
                        }
                        ReflectDescriptorType::Sampler => vk::DescriptorType::SAMPLER,
                        ReflectDescriptorType::CombinedImageSampler => {
                            vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                        }
                        ReflectDescriptorType::SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
                        ReflectDescriptorType::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
                        ReflectDescriptorType::UniformTexelBuffer => {
                            vk::DescriptorType::UNIFORM_TEXEL_BUFFER
                        }
                        ReflectDescriptorType::StorageTexelBuffer => {
                            vk::DescriptorType::STORAGE_TEXEL_BUFFER
                        }
                        ReflectDescriptorType::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
                        ReflectDescriptorType::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
                        ReflectDescriptorType::UniformBufferDynamic => {
                            vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                        }
                        ReflectDescriptorType::StorageBufferDynamic => {
                            vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
                        }
                        ReflectDescriptorType::InputAttachment => {
                            vk::DescriptorType::INPUT_ATTACHMENT
                        }
                        ReflectDescriptorType::AccelerationStructureNV => {
                            vk::DescriptorType::ACCELERATION_STRUCTURE_NV
                        }
                    };

                    for (name, override_type) in overrides {
                        if &binding.name == name {
                            desc_type = *override_type;
                        }
                    }
                    let shader_stage_flags = spv_module.get_shader_stage().bits();
                    let desc_count = binding.array.dims.iter().product();
                    let layout_binding = vk::DescriptorSetLayoutBinding::builder()
                        .binding(binding.binding)
                        .descriptor_type(desc_type)
                        .descriptor_count(desc_count)
                        .stage_flags(vk::ShaderStageFlags::from_raw(shader_stage_flags))
                        .build();

                    let reflected_binding = ReflectedBinding {
                        binding: layout_binding.binding,
                        set: set.set,
                        typ: desc_type,
                    };
                    self.bindings
                        .insert(binding.name.clone(), reflected_binding);
                    layout.bindings.push(layout_binding);
                }

                layout.create_info = vk::DescriptorSetLayoutCreateInfo::builder()
                    .bindings(&layout.bindings)
                    .build();
                set_layouts.push(layout);
            }

            // TODO Assuming only one push constance block per shader
            if let Some(push_constant) = spv_module
                .enumerate_push_constant_blocks(None)
                .map_err(RendererError::SpirvError)?
                .first()
            {
                constant_ranges.push(vk::PushConstantRange {
                    offset: push_constant.offset,
                    size: push_constant.size,
                    stage_flags: shader_stage.stage,
                });
            }
        }

        let mut merged_layouts = [
            DescriptorSetLayoutData::default(),
            DescriptorSetLayoutData::default(),
            DescriptorSetLayoutData::default(),
            DescriptorSetLayoutData::default(),
        ];

        for i in 0..4u32 {
            let layout_data = &mut merged_layouts[i as usize];

            layout_data.set = i;

            let mut binds: HashMap<u32, vk::DescriptorSetLayoutBinding> = HashMap::new();
            for set in set_layouts.iter() {
                if set.set == i {
                    for binding in set.bindings.iter() {
                        binds
                            .entry(binding.binding)
                            .and_modify(|e| e.stage_flags |= binding.stage_flags)
                            .or_insert(*binding);
                    }
                }
            }

            for binding in binds.values() {
                layout_data.bindings.push(*binding);
            }
            layout_data
                .bindings
                .sort_by(|a, b| a.binding.cmp(&b.binding));

            if !layout_data.bindings.is_empty() {
                layout_data.create_info = vk::DescriptorSetLayoutCreateInfo::builder()
                    .bindings(&layout_data.bindings)
                    .build();
                self.set_hashes[i as usize] = hash_descriptor_layout_info(&layout_data.create_info);
                self.set_layouts[i as usize] =
                    unsafe { device.create_descriptor_set_layout(&layout_data.create_info, None)? };
            } else {
                self.set_hashes[i as usize] = 0;
                self.set_layouts[i as usize] = vk::DescriptorSetLayout::null();
            }
        }

        let mut compacted_layouts = vec![];
        compacted_layouts.reserve(4);
        for layout in self.set_layouts {
            if layout != vk::DescriptorSetLayout::null() {
                compacted_layouts.push(layout);
            }
        }

        let pipeline_create_info = vk::PipelineLayoutCreateInfo::builder()
            .push_constant_ranges(&constant_ranges)
            .set_layouts(&compacted_layouts);

        self.pipeline_layout =
            unsafe { device.create_pipeline_layout(&pipeline_create_info, None)? };

        Ok(())
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            for layout in self.set_layouts.iter_mut() {
                if *layout != vk::DescriptorSetLayout::null() {
                    device.destroy_descriptor_set_layout(*layout, None);
                }
                *layout = vk::DescriptorSetLayout::null();
            }
            self.set_hashes.fill(0);
            self.stages.clear();
        }
    }
}

impl Default for ShaderEffect {
    fn default() -> Self {
        Self::new()
    }
}

pub struct ShaderCache {
    module_handles: HandleArray<ShaderModule>,
    module_cache: HashMap<String, Handle<ShaderModule>>,

    effects_handles: HandleArray<ShaderEffect>,
}

impl ShaderCache {
    pub fn new(device: &ash::Device) -> RendererResult<Self> {
        let mut module_handles = HandleArray::new();
        let mut module_cache = HashMap::new();

        {
            let module = ShaderModule::new(
                device,
                vk_shader_macros::include_glsl!("./shaders/default.vert", kind: vert).to_vec(),
            )?;
            let handle = module_handles.insert(module);
            module_cache.insert("./shaders/default.vert".to_string(), handle);
        }
        {
            let module = ShaderModule::new(
                device,
                vk_shader_macros::include_glsl!("./shaders/default.frag", kind: frag).to_vec(),
            )?;
            let handle = module_handles.insert(module);
            module_cache.insert("./shaders/default.frag".to_string(), handle);
        }
        {
            let module = ShaderModule::new(
                device,
                vk_shader_macros::include_glsl!("./shaders/text.vert", kind: vert).to_vec(),
            )?;
            let handle = module_handles.insert(module);
            module_cache.insert("./shaders/text.vert".to_string(), handle);
        }
        {
            let module = ShaderModule::new(
                device,
                vk_shader_macros::include_glsl!("./shaders/text.frag", kind: frag).to_vec(),
            )?;
            let handle = module_handles.insert(module);
            module_cache.insert("./shaders/text.frag".to_string(), handle);
        }

        Ok(Self {
            module_handles,
            module_cache,
            effects_handles: HandleArray::new(),
        })
    }

    pub fn get_shader_handle<S: AsRef<str>>(
        &self,
        path: S,
    ) -> RendererResult<Handle<ShaderModule>> {
        match self.module_cache.get(path.as_ref()) {
            Some(handle) => Ok(*handle),
            None => Err(RendererError::InvalidHandle(InvalidHandle)),
        }
    }

    pub fn get_shader_module_by_handle(
        &self,
        handle: Handle<ShaderModule>,
    ) -> RendererResult<&ShaderModule> {
        self.module_handles
            .get(handle)
            .ok_or(RendererError::InvalidHandle(InvalidHandle))
    }

    pub fn build_effect(
        &mut self,
        device: &ash::Device,
        vertex_shader: &str,
        fragment_shader: Option<&str>,
    ) -> RendererResult<Handle<ShaderEffect>> {
        let overrides = [("ubo", vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)];
        let mut effect = ShaderEffect::new();
        effect.add_stage(
            self.get_shader_handle(vertex_shader)?,
            vk::ShaderStageFlags::VERTEX,
        )?;
        if let Some(fs) = fragment_shader {
            effect.add_stage(self.get_shader_handle(fs)?, vk::ShaderStageFlags::FRAGMENT)?;
        }

        effect.reflect_layout(device, self, &overrides)?;

        let handle = self.effects_handles.insert(effect);

        Ok(handle)
    }

    pub fn get_shader_effect_by_handle(
        &self,
        handle: Handle<ShaderEffect>,
    ) -> RendererResult<&ShaderEffect> {
        self.effects_handles
            .get(handle)
            .ok_or(RendererError::InvalidHandle(InvalidHandle))
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        for module in self.module_handles.iter_mut() {
            module.destroy(device);
        }
        self.module_cache.clear();
        self.module_handles.clear();
        for effect in self.effects_handles.iter_mut() {
            effect.destroy(device);
        }
        self.effects_handles.clear();
    }
}

use std::ffi::CStr;

use ash::vk;
use ash::Device;

use super::RendererResult;

pub struct ShaderModule {
    device: Device,
    vert_shader_module: vk::ShaderModule,
    frag_shader_module: vk::ShaderModule,
    stages: [vk::PipelineShaderStageCreateInfo; 2],
}

impl ShaderModule {
    pub fn new(device: &Device, vert: &[u32], frag: &[u32]) -> RendererResult<ShaderModule> {
        let vert_shader_create_info = vk::ShaderModuleCreateInfo::builder().code(vert);
        let vert_shader_module =
            unsafe { device.create_shader_module(&vert_shader_create_info, None)? };

        let frag_shader_create_info = vk::ShaderModuleCreateInfo::builder().code(frag);
        let frag_shader_module =
            unsafe { device.create_shader_module(&frag_shader_create_info, None)? };

        let main_function_name = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };
        let vert_shader_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_shader_module)
            .name(main_function_name);
        let frag_shader_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_shader_module)
            .name(main_function_name);
        let stages = [vert_shader_stage.build(), frag_shader_stage.build()];

        Ok(ShaderModule {
            device: device.clone(),
            vert_shader_module,
            frag_shader_module,
            stages,
        })
    }

    pub fn get_stages(&self) -> &[vk::PipelineShaderStageCreateInfo] {
        &self.stages[..]
    }

    pub fn destroy(&self) {
        unsafe {
            self.device
                .destroy_shader_module(self.frag_shader_module, None);
            self.device
                .destroy_shader_module(self.vert_shader_module, None);
        }
    }
}

use std::ffi::CStr;

use ash::vk;
use ash::Device;

// To avoid a naming conflict
use spirv_reflect::ShaderModule as ShaderReflection;

use super::RendererResult;

pub struct ShaderModule {
    device: Device,
    vert_shader_module: vk::ShaderModule,
    frag_shader_module: vk::ShaderModule,
    stages: [vk::PipelineShaderStageCreateInfo; 2],
    pub vert_reflection: ShaderReflection,
    pub frag_reflection: ShaderReflection
}

fn dump_reflection_info(name: &str, module: &ShaderReflection) {
    println!("{name} reflection info:");

    let entry_point_name = module.get_entry_point_name();
    println!("\tEntry Point: {entry_point_name}");

    let generator = module.get_generator();
    let generator = match generator {
        spirv_reflect::types::ReflectGenerator::Unknown => "Unknown",
        spirv_reflect::types::ReflectGenerator::KhronosLlvmSpirvTranslator => "KhronosLlvmSpirvTranslator",
        spirv_reflect::types::ReflectGenerator::KhronosSpirvToolsAssembler => "KhronosSpirvToolsAssembler",
        spirv_reflect::types::ReflectGenerator::KhronosGlslangReferenceFrontEnd => "KhronosGlslangReferenceFrontEnd",
        spirv_reflect::types::ReflectGenerator::GoogleShadercOverGlslang => "GoogleShadercOverGlslang",
        spirv_reflect::types::ReflectGenerator::GoogleSpiregg => "GoogleSpiregg",
        spirv_reflect::types::ReflectGenerator::GoogleRspirv => "GoogleRspirv",
        spirv_reflect::types::ReflectGenerator::XLegendMesaMesairSpirvTranslator => "XLegendMesaMesairSpirvTranslator",
        spirv_reflect::types::ReflectGenerator::KhronosSpirvToolsLinker => "KhronosSpirvToolsLinker",
        spirv_reflect::types::ReflectGenerator::WineVkd3dShaderCompiler => "WineVkd3dShaderCompiler",
        spirv_reflect::types::ReflectGenerator::ClayClayShaderCompiler => "ClayClayShaderCompiler",
    };
    println!("\tGenerator: {generator}");
    let shader_stage = module.get_shader_stage().bits();
    println!("\tShader Stage: {shader_stage:x}");
    let source_lang = module.get_source_language();
    let source_lang = match source_lang {
        spirv_headers::SourceLanguage::Unknown => "Unknown",
        spirv_headers::SourceLanguage::ESSL => "ESSL",
        spirv_headers::SourceLanguage::GLSL => "GLSL",
        spirv_headers::SourceLanguage::OpenCL_C => "OpenCL_C",
        spirv_headers::SourceLanguage::OpenCL_CPP => "OpenCL_CPP",
        spirv_headers::SourceLanguage::HLSL => "HLSL",
    };
    println!("\tSource Language: {source_lang}");
    let source_lang_ver = module.get_source_language_version();
    println!("\tSource Language version: {source_lang_ver}");
    let source_file = module.get_source_file();
    println!("\tSource File: {source_file}");
    let source_text = module.get_source_text();
    println!("\tSource Text: {source_text}");
    let spv_execution_model = module.get_spirv_execution_model();
    let model = match spv_execution_model {
        spirv_headers::ExecutionModel::Vertex => "Vertex",
        spirv_headers::ExecutionModel::TessellationControl => "TessellationControl",
        spirv_headers::ExecutionModel::TessellationEvaluation => "TessellationEvaluation",
        spirv_headers::ExecutionModel::Geometry => "Geometry",
        spirv_headers::ExecutionModel::Fragment => "Fragment",
        spirv_headers::ExecutionModel::GLCompute => "GLCompute",
        spirv_headers::ExecutionModel::Kernel => "Kernel",
        spirv_headers::ExecutionModel::TaskNV => "TaskNV",
        spirv_headers::ExecutionModel::MeshNV => "MeshNV",
        spirv_headers::ExecutionModel::RayGenerationNV => "RayGenerationNV",
        spirv_headers::ExecutionModel::IntersectionNV => "IntersectionNV",
        spirv_headers::ExecutionModel::AnyHitNV => "AnyHitNV",
        spirv_headers::ExecutionModel::ClosestHitNV => "ClosestHitNV",
        spirv_headers::ExecutionModel::MissNV => "MissNV",
        spirv_headers::ExecutionModel::CallableNV => "CallableNV",
    };
    println!("\tExecution Model: {model}");
    let input_vars = module.enumerate_input_variables(None).unwrap();
    println!("\tinput vars:");
    for var in input_vars {
        let spirv_id = var.spirv_id;
        let name = var.name;
        let location = var.location;
        let storage_class = match var.storage_class {
            spirv_reflect::types::ReflectStorageClass::Undefined => "Undefined",
            spirv_reflect::types::ReflectStorageClass::UniformConstant => "UniformConstant",
            spirv_reflect::types::ReflectStorageClass::Input => "Input",
            spirv_reflect::types::ReflectStorageClass::Uniform => "Uniform",
            spirv_reflect::types::ReflectStorageClass::Output => "output",
            spirv_reflect::types::ReflectStorageClass::WorkGroup => "WorkGroup",
            spirv_reflect::types::ReflectStorageClass::CrossWorkGroup => "CrossWorkGroup",
            spirv_reflect::types::ReflectStorageClass::Private => "Private",
            spirv_reflect::types::ReflectStorageClass::Function => "Function",
            spirv_reflect::types::ReflectStorageClass::Generic => "Generic",
            spirv_reflect::types::ReflectStorageClass::PushConstant => "PushConstant",
            spirv_reflect::types::ReflectStorageClass::AtomicCounter => "AtomicCounter",
            spirv_reflect::types::ReflectStorageClass::Image => "Image",
            spirv_reflect::types::ReflectStorageClass::StorageBuffer => "StorageBuffer",
        };
        println!("\t\tID: {spirv_id}, name: {name}, location: {location}, storage class: {storage_class}");
    }
    let output_vars = module.enumerate_output_variables(None).unwrap();
    println!("\tOutput vars:");
    for var in output_vars {
        let spirv_id = var.spirv_id;
        let name = var.name;
        let location = var.location;
        let storage_class = match var.storage_class {
            spirv_reflect::types::ReflectStorageClass::Undefined => "Undefined",
            spirv_reflect::types::ReflectStorageClass::UniformConstant => "UniformConstant",
            spirv_reflect::types::ReflectStorageClass::Input => "Input",
            spirv_reflect::types::ReflectStorageClass::Uniform => "Uniform",
            spirv_reflect::types::ReflectStorageClass::Output => "Output",
            spirv_reflect::types::ReflectStorageClass::WorkGroup => "WorkGroup",
            spirv_reflect::types::ReflectStorageClass::CrossWorkGroup => "CrossWorkGroup",
            spirv_reflect::types::ReflectStorageClass::Private => "Private",
            spirv_reflect::types::ReflectStorageClass::Function => "Function",
            spirv_reflect::types::ReflectStorageClass::Generic => "Generic",
            spirv_reflect::types::ReflectStorageClass::PushConstant => "PushConstant",
            spirv_reflect::types::ReflectStorageClass::AtomicCounter => "AtomicCounter",
            spirv_reflect::types::ReflectStorageClass::Image => "Image",
            spirv_reflect::types::ReflectStorageClass::StorageBuffer => "StorageBuffer",
        };
        println!("\t\tID: {spirv_id}, name: {name}, location: {location}, storage class: {storage_class}");
    }
    let bindings = module.enumerate_descriptor_bindings(None).unwrap();
    println!("\tDescriptor Bindings:");
    for binding in bindings {
        let spirv_id = binding.spirv_id;
        let name = binding.name;
        let binding_id = binding.binding;
        let index = binding.input_attachment_index;
        let set = binding.set;
        let descriptor_type = match binding.descriptor_type {
            spirv_reflect::types::ReflectDescriptorType::Undefined => "Undefined",
            spirv_reflect::types::ReflectDescriptorType::Sampler => "Sampler",
            spirv_reflect::types::ReflectDescriptorType::CombinedImageSampler => "CombinedImageSampler",
            spirv_reflect::types::ReflectDescriptorType::SampledImage => "SampledImage",
            spirv_reflect::types::ReflectDescriptorType::StorageImage => "StorageImage",
            spirv_reflect::types::ReflectDescriptorType::UniformTexelBuffer => "UniformTexelBuffer",
            spirv_reflect::types::ReflectDescriptorType::StorageTexelBuffer => "StorageTexelBuffer",
            spirv_reflect::types::ReflectDescriptorType::UniformBuffer => "UniformBuffer",
            spirv_reflect::types::ReflectDescriptorType::StorageBuffer => "StorageBuffer",
            spirv_reflect::types::ReflectDescriptorType::UniformBufferDynamic => "UniformBufferDynamic",
            spirv_reflect::types::ReflectDescriptorType::StorageBufferDynamic => "StorageBufferDynamic",
            spirv_reflect::types::ReflectDescriptorType::InputAttachment => "InputAttachment",
            spirv_reflect::types::ReflectDescriptorType::AccelerationStructureNV => "AccelerationStructureNV",
        };
        let resource_type = match binding.resource_type {
            spirv_reflect::types::ReflectResourceType::Undefined => "Undefined",
            spirv_reflect::types::ReflectResourceType::Sampler => "Sampler",
            spirv_reflect::types::ReflectResourceType::CombinedImageSampler => "CombinedImageSampler",
            spirv_reflect::types::ReflectResourceType::ConstantBufferView => "ConstantBufferView",
            spirv_reflect::types::ReflectResourceType::ShaderResourceView => "ShaderResourceView",
            spirv_reflect::types::ReflectResourceType::UnorderedAccessView => "UnorderedAccessView",
        };
        println!("\t\tID: {spirv_id}, name: {name}, binding: {binding_id}, index: {index}, set: {set}, descriptor type: {descriptor_type}, resource type: {resource_type}")
    }
    let sets = module.enumerate_descriptor_sets(None).unwrap();
    println!("\tDescriptor Sets:");
    for set in sets {
        let set = set.set;
        println!("\t\tset: {set}")
    }
}

impl ShaderModule {
    pub fn new(device: &Device, vert: &[u32], frag: &[u32]) -> RendererResult<ShaderModule> {
        let vert_reflection = ShaderReflection::load_u32_data(vert)
            .map_err(crate::renderer::error::RendererError::SpirvError)?;
        dump_reflection_info("vert shader", &vert_reflection);
        let frag_reflection = ShaderReflection::load_u32_data(vert)
            .map_err(crate::renderer::error::RendererError::SpirvError)?;
        dump_reflection_info("frag_shader", &frag_reflection);
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
            vert_reflection,
            frag_reflection
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

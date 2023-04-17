use core::slice;
use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::{vulkan::Allocator, MemoryLocation};
use nalgebra_glm as glm;

use super::{
    buffer::{Buffer, BufferManager},
    error::{InvalidHandle, RendererError},
    material::Material,
    mesh::Mesh,
    utils::{Handle, HandleArray},
    RendererResult,
};

#[allow(dead_code)]
#[derive(Debug)]
pub struct InstanceData {
    pub model_matrix: [[f32; 4]; 4],
    pub inverse_model_matrix: [[f32; 4]; 4],
}

impl InstanceData {
    pub fn new(model: glm::Mat4) -> Self {
        InstanceData {
            model_matrix: model.into(),
            inverse_model_matrix: model.try_inverse().expect("Could not get inverse!").into(),
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe {
            slice::from_raw_parts(
                self as *const Self as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }
}

#[derive(Debug)]
pub struct SceneObject {
    pub mesh: Handle<Mesh>,
    pub material: Handle<Material>,
    pub position: glm::Vec3,
    pub rotation: glm::Quat,
    pub scaling: glm::Vec3,

    transform_dirty: bool,
    transform: glm::Mat4,
    instance_data: InstanceData,
    global_transform: glm::Mat4,
    instance_buffer: Buffer,

    parent: Option<Handle<SceneObject>>,
    children: Vec<Handle<SceneObject>>,
}

impl SceneObject {
    fn update_instance(&mut self, allocator: &mut Allocator) -> RendererResult<()> {
        self.instance_buffer
            .fill(allocator, self.instance_data.as_slice())
    }

    pub fn get_buffer(&self) -> &Buffer {
        &self.instance_buffer
    }
}

impl Drop for SceneObject {
    fn drop(&mut self) {
        self.instance_buffer
            .queue_free(None)
            .expect("Could not free buffer");
    }
}

pub struct SceneObjectMutGuard<'a> {
    // there has to be a better way to do this than storing a raw pointer?
    allocator: &'a mut Allocator,

    scene_tree: *mut SceneTree,
    object_handle: Handle<SceneObject>,
    pub object: &'a mut SceneObject,
}

impl<'a> Drop for SceneObjectMutGuard<'a> {
    fn drop(&mut self) {
        let scene_tree = unsafe { self.scene_tree.as_mut().expect("Null scene tree pointer? ") };
        scene_tree
            .update_transform(self.object_handle, self.allocator)
            .expect("Could not update transform");
    }
}

impl<'a> SceneObjectMutGuard<'a> {
    pub fn add_child(&mut self, child: Handle<SceneObject>) -> RendererResult<()> {
        let scene_tree = unsafe { self.scene_tree.as_mut().expect("Null scene tree pointer? ") };
        {
            let child_obj = scene_tree
                .get_object_mut(child, self.allocator)
                .ok_or::<RendererError>(InvalidHandle.into())?;
            child_obj.object.parent = Some(self.object_handle);
        }
        self.object.children.push(child);

        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct SceneTree {
    objects: HandleArray<SceneObject>,
}

impl SceneTree {
    pub fn new_object(
        &mut self,
        mesh: Handle<Mesh>,
        material: Handle<Material>,
        device: &ash::Device,
        allocator: &mut Allocator,
        buffer_manager: Arc<Mutex<BufferManager>>,
    ) -> RendererResult<Handle<SceneObject>> {
        let instance_buffer = BufferManager::new_buffer(
            buffer_manager,
            device,
            allocator,
            std::mem::size_of::<InstanceData>() as u64,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            MemoryLocation::CpuToGpu,
            "instance-buffer",
        )?;
        let scene_object = SceneObject {
            mesh,
            material,
            position: glm::Vec3::default(),
            rotation: glm::Quat::identity(),
            scaling: glm::Vec3::new(1.0, 1.0, 1.0),
            transform_dirty: Default::default(),
            transform: glm::Mat4::identity(),
            global_transform: glm::Mat4::identity(),
            instance_data: InstanceData::new(glm::Mat4::identity()),
            instance_buffer,
            parent: None,
            children: Vec::new(),
        };
        Ok(self.objects.insert(scene_object))
    }

    pub fn get_object(&self, handle: Handle<SceneObject>) -> Option<&SceneObject> {
        self.objects.get(handle)
    }

    pub fn get_object_mut<'a>(
        &'a mut self,
        handle: Handle<SceneObject>,
        allocator: &'a mut Allocator,
    ) -> Option<SceneObjectMutGuard<'a>> {
        let self_ptr = self as *mut SceneTree;
        self.objects.get_mut(handle).map(|obj| SceneObjectMutGuard {
            allocator,
            scene_tree: self_ptr,
            object_handle: handle,
            object: obj,
        })
    }

    fn update_transform(
        &mut self,
        handle: Handle<SceneObject>,
        allocator: &mut Allocator,
    ) -> RendererResult<()> {
        let parent_handle = self.objects.get(handle).expect("Invalid handle?").parent;
        let parent_transform = parent_handle.map(|p_h| {
            self.objects
                .get(p_h)
                .expect("Invalid parent handle?")
                .global_transform
        });
        let children_handles = if let Some(obj) = self.objects.get_mut(handle) {
            obj.transform = glm::Mat4::new_translation(&obj.position)
                * glm::quat_to_mat4(&obj.rotation)
                * glm::scaling(&obj.scaling);
            if let Some(parent_transf) = &parent_transform {
                obj.global_transform = *parent_transf * obj.transform;
            } else {
                obj.global_transform = obj.transform;
            }
            obj.instance_data = InstanceData::new(obj.global_transform);
            obj.transform_dirty = false;
            obj.update_instance(allocator)?;
            obj.children.clone()
        } else {
            return Err(InvalidHandle.into());
        };

        for handle in children_handles {
            self.update_transform(handle, allocator)?;
        }

        Ok(())
    }

    pub fn iter(&self) -> std::slice::Iter<'_, SceneObject> {
        self.objects.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, SceneObject> {
        self.objects.iter_mut()
    }

    pub fn destroy(&mut self) {
        self.objects.clear();
    }
}

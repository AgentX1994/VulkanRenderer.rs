use std::{cell::RefCell, rc::Rc};

use nalgebra_glm as glm;

use super::{
    error::{InvalidHandle, RendererError},
    model::Model,
    utils::{Handle, HandleArray},
    vertex::Vertex,
    InstanceData, RendererResult,
};

#[derive(Debug)]
pub struct SceneObject {
    model: Option<Rc<RefCell<Model<Vertex, InstanceData>>>>,
    pub instance_id: Option<Handle<InstanceData>>,
    pub position: glm::Vec3,
    pub rotation: glm::Quat,
    pub scaling: glm::Vec3,

    transform_dirty: bool,
    transform: glm::Mat4,
    global_transform: glm::Mat4,

    pub parent: Option<Handle<SceneObject>>,
    pub children: Vec<Handle<SceneObject>>,
}

impl SceneObject {
    fn update_instance(&mut self) -> RendererResult<()> {
        if let Some(model) = &self.model {
            let instance_data = InstanceData::new(self.global_transform);
            if let Some(id) = &self.instance_id {
                model.borrow_mut().update(*id, instance_data)?;
            } else {
                self.instance_id = Some(model.borrow_mut().insert_visibly(instance_data));
            }
        }
        Ok(())
    }

    pub fn set_model(
        &mut self,
        model: &Rc<RefCell<Model<Vertex, InstanceData>>>,
    ) -> RendererResult<()> {
        if let Some(old_model) = self.model.take() {
            if let Some(id) = self.instance_id.take() {
                old_model.borrow_mut().remove(id)?;
            }
        }
        self.model = Some(model.clone());
        self.transform_dirty = true;
        Ok(())
    }

    pub fn add_child(&mut self, child: Handle<SceneObject>) {
        self.children.push(child);
    }
}

pub struct SceneObjectMutGuard<'a> {
    // there has to be a better way to do this than storing a raw pointer?
    scene_tree: *mut SceneTree,
    object_handle: Handle<SceneObject>,
    pub object: &'a mut SceneObject,
}

impl<'a> Drop for SceneObjectMutGuard<'a> {
    fn drop(&mut self) {
        let scene_tree = unsafe { self.scene_tree.as_mut().expect("Null scene tree pointer? ") };
        scene_tree
            .update_transform(self.object_handle)
            .expect("Could not update transform");
    }
}

impl<'a> SceneObjectMutGuard<'a> {}

#[derive(Debug, Default)]
pub struct SceneTree {
    objects: HandleArray<SceneObject>,
}

impl SceneTree {
    pub fn new_object(&mut self) -> Handle<SceneObject> {
        let scene_object = SceneObject {
            model: None,
            instance_id: None,
            position: glm::Vec3::default(),
            rotation: glm::Quat::identity(),
            scaling: glm::Vec3::new(1.0, 1.0, 1.0),
            transform_dirty: Default::default(),
            transform: glm::Mat4::identity(),
            global_transform: glm::Mat4::identity(),
            parent: None,
            children: Vec::new(),
        };
        self.objects.insert(scene_object)
    }

    pub fn get_object(&self, handle: Handle<SceneObject>) -> Option<&SceneObject> {
        self.objects.get(handle)
    }

    pub fn get_object_mut(
        &mut self,
        handle: Handle<SceneObject>,
    ) -> Option<SceneObjectMutGuard<'_>> {
        let self_ptr = self as *mut SceneTree;
        self.objects.get_mut(handle).map(|obj| SceneObjectMutGuard {
            scene_tree: self_ptr,
            object_handle: handle,
            object: obj,
        })
    }

    fn update_transform(&mut self, handle: Handle<SceneObject>) -> RendererResult<()> {
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
                obj.global_transform = obj.transform * *parent_transf;
            } else {
                obj.global_transform = obj.transform;
            }
            obj.transform_dirty = false;
            obj.update_instance()?;
            obj.children.clone()
        } else {
            return Err(RendererError::InvalidHandle(InvalidHandle));
        };

        for handle in children_handles {
            self.update_transform(handle)?;
        }

        Ok(())
    }
}

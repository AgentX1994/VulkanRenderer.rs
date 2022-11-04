use std::{cell::RefCell, rc::Rc};

use nalgebra_glm as glm;

use super::{model::{Model, ModelHandle}, vertex::Vertex, InstanceData, RendererResult};

#[derive(Debug)]
pub struct SceneObject {
    model: Option<Rc<RefCell<Model<Vertex, InstanceData>>>>,
    pub instance_id: Option<ModelHandle>,
    pub position: glm::Vec3,
    pub rotation: glm::Quat,
    pub scaling: glm::Vec3,

    // TODO move these to a material properties section
    pub metallic: f32,
    pub roughness: f32,
    pub texture_id: u32,

    transform_dirty: bool,
    transform: glm::Mat4,
    global_transform: glm::Mat4,

    parent: Option<Rc<RefCell<SceneObject>>>,
    pub children: Vec<Rc<RefCell<SceneObject>>>,
}

impl SceneObject {
    pub fn new_empty() -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            model: None,
            instance_id: None,
            position: glm::Vec3::default(),
            rotation: glm::Quat::identity(),
            scaling: glm::Vec3::new(1.0, 1.0, 1.0),
            metallic: 0.0,
            roughness: 0.0,
            texture_id: 0,
            transform_dirty: Default::default(),
            transform: glm::Mat4::identity(),
            global_transform: glm::Mat4::identity(),
            parent: None,
            children: Vec::new(),
        }))
    }

    fn update_instance(&mut self) -> RendererResult<()> {
        if self.transform_dirty {
            self.update_transform(false)?;
        }
        if let Some(model) = &self.model {
            let instance_data = InstanceData::new(
                self.global_transform,
                glm::Vec3::new(0.0, 0.0, 0.0),
                self.metallic,
                self.roughness,
                self.texture_id,
            );
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
        self.update_instance()
    }

    pub fn add_child(parent: &Rc<RefCell<SceneObject>>, child: &Rc<RefCell<SceneObject>>) {
        parent.borrow_mut().children.push(child.clone());
        child.borrow_mut().parent = Some(parent.clone());
    }

    pub fn update_transform(&mut self, force: bool) -> RendererResult<()> {
        if !self.transform_dirty && !force {
            return Ok(());
        }

        self.transform = glm::Mat4::new_translation(&self.position)
            * glm::quat_to_mat4(&self.rotation)
            * glm::scaling(&self.scaling);
        if let Some(parent) = &self.parent {
            self.global_transform = self.transform * parent.borrow().global_transform;
        } else {
            self.global_transform = self.transform;
        }
        self.transform_dirty = false;
        self.update_instance()?;

        for child in &mut self.children {
            child.borrow_mut().update_transform(true)?;
        }

        Ok(())
    }

    // pub fn set_position(&mut self, position: glm::Vec3) -> RendererResult<()> {
    //     self.position = position;
    //     self.update_transform(true)
    // }
}

#[derive(Debug)]
pub struct SceneTree {
    root: Rc<RefCell<SceneObject>>,
}

impl Default for SceneTree {
    fn default() -> Self {
        Self {
            root: SceneObject::new_empty(),
        }
    }
}

impl SceneTree {
    pub fn get_root(&self) -> Rc<RefCell<SceneObject>> {
        self.root.clone()
    }
}

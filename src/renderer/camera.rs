use ash::prelude::VkResult;
use gpu_allocator::vulkan::Allocator;
use nalgebra as na;
use nalgebra_glm as glm;

use super::buffer::Buffer;

pub struct Camera {
    view_matrix: glm::Mat4,
    position: glm::Vec3,
    view_direction: na::Unit<glm::Vec3>,
    down_direction: na::Unit<glm::Vec3>,
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            view_matrix: glm::Mat4::identity(),
            position: glm::Vec3::default(),
            view_direction: na::Unit::new_normalize(glm::Vec3::new(0.0, 0.0, 1.0)),
            down_direction: na::Unit::new_normalize(glm::Vec3::new(0.0, 1.0, 0.0)),
        }
    }
}

impl Camera {
    fn update_matrix(&mut self) {
        let right = na::Unit::new_normalize(self.down_direction.cross(&self.view_direction));
        self.view_matrix = glm::Mat4::new(
            right.x,
            right.y,
            right.z,
            -right.dot(&self.position),
            self.down_direction.x,
            self.down_direction.y,
            self.down_direction.z,
            -self.down_direction.dot(&self.position),
            self.view_direction.x,
            self.view_direction.y,
            self.view_direction.z,
            -self.view_direction.dot(&self.position),
            0.0,
            0.0,
            0.0,
            1.0,
        );
    }

    pub fn move_forward(&mut self, distance: f32) {
        self.position += distance * self.view_direction.as_ref();
        self.update_matrix();
    }

    pub fn move_backward(&mut self, distance: f32) {
        self.move_forward(-distance);
    }

    pub fn turn_right(&mut self, angle: f32) {
        let rotation = na::Rotation3::from_axis_angle(&self.down_direction, angle);
        self.view_direction = rotation * self.view_direction;
        self.update_matrix();
    }

    pub fn turn_left(&mut self, angle: f32) {
        self.turn_right(-angle);
    }

    pub fn turn_up(&mut self, angle: f32) {
        let right = na::Unit::new_normalize(self.down_direction.cross(&self.view_direction));
        let rotation = na::Rotation3::from_axis_angle(&right, angle);
        self.view_direction = rotation * self.view_direction;
        self.down_direction = rotation * self.down_direction;
        self.update_matrix();
    }

    pub fn turn_down(&mut self, angle: f32) {
        self.turn_up(-angle);
    }

    pub(crate) fn update_buffer(
        &self,
        allocator: &mut Allocator,
        buffer: &mut Buffer,
    ) -> VkResult<()> {
        let data_array: [[f32; 4]; 4] = self.view_matrix.into();
        let bytes = std::mem::size_of::<[[f32; 4]; 4]>();
        let data = unsafe { std::slice::from_raw_parts(data_array.as_ptr() as *const u8, bytes) };
        buffer.fill(allocator, data)
    }
}

use gpu_allocator::vulkan::Allocator;
use nalgebra as na;
use nalgebra_glm as glm;

use super::{buffer::Buffer, RendererResult};

pub struct CameraBuilder {
    position: glm::Vec3,
    view_direction: na::Unit<glm::Vec3>,
    down_direction: na::Unit<glm::Vec3>,
    fovy: f32,
    aspect: f32,
    near: f32,
    far: f32,
}

impl CameraBuilder {
    pub fn position(mut self, pos: glm::Vec3) -> CameraBuilder {
        self.position = pos;
        self
    }

    pub fn fovy(mut self, fovy: f32) -> CameraBuilder {
        self.fovy = fovy.max(0.01).min(std::f32::consts::PI - 0.01);
        self
    }

    pub fn aspect(mut self, aspect: f32) -> CameraBuilder {
        self.aspect = aspect;
        self
    }

    pub fn near(mut self, near: f32) -> CameraBuilder {
        self.near = near;
        self
    }

    pub fn far(mut self, far: f32) -> CameraBuilder {
        self.far = far;
        self
    }

    pub fn view_direction(mut self, direction: glm::Vec3) -> CameraBuilder {
        self.view_direction = na::Unit::new_normalize(direction);
        self
    }

    pub fn down_direction(mut self, direction: glm::Vec3) -> CameraBuilder {
        self.down_direction = na::Unit::new_normalize(direction);
        self
    }

    pub fn build(self) -> Camera {
        if self.far < self.near {
            // TODO return error
            panic!(
                "Far plane (at {}) closer than near plane (at {})!",
                self.far, self.near
            );
        }
        let mut cam = Camera {
            position: self.position,
            view_direction: self.view_direction,
            down_direction: na::Unit::new_normalize(
                self.down_direction.as_ref()
                    - self
                        .down_direction
                        .as_ref()
                        .dot(self.view_direction.as_ref())
                        * self.view_direction.as_ref(),
            ),
            fovy: self.fovy,
            aspect: self.aspect,
            near: self.near,
            far: self.far,
            view_matrix: glm::Mat4::identity(),
            projection_matrix: glm::Mat4::identity(),
        };
        cam.update_projection_matrix();
        cam.update_view_matrix();
        cam
    }
}

pub struct Camera {
    view_matrix: glm::Mat4,
    position: glm::Vec3,
    view_direction: na::Unit<glm::Vec3>,
    down_direction: na::Unit<glm::Vec3>,
    fovy: f32,
    aspect: f32,
    near: f32,
    far: f32,
    projection_matrix: glm::Mat4,
}

impl Camera {
    pub fn builder() -> CameraBuilder {
        CameraBuilder {
            position: glm::Vec3::new(0.0, -3.0, -3.0),
            view_direction: na::Unit::new_normalize(glm::Vec3::new(0.0, 1.0, 1.0)),
            down_direction: na::Unit::new_normalize(glm::Vec3::new(0.0, 1.0, -1.0)),
            fovy: std::f32::consts::FRAC_PI_3,
            aspect: 800.0 / 600.0,
            near: 0.1,
            far: 100.0,
        }
    }

    fn update_view_matrix(&mut self) {
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
    fn update_projection_matrix(&mut self) {
        let d = 1.0 / (0.5 * self.fovy).tan();
        self.projection_matrix = glm::Mat4::new(
            d / self.aspect,
            0.0,
            0.0,
            0.0,
            0.0,
            d,
            0.0,
            0.0,
            0.0,
            0.0,
            self.far / (self.far - self.near),
            -self.near * self.far / (self.far - self.near),
            0.0,
            0.0,
            1.0,
            0.0,
        );
    }

    pub fn move_up(&mut self, distance: f32) {
        self.position += distance * -self.down_direction.as_ref();
        self.update_view_matrix();
    }

    pub fn move_down(&mut self, distance: f32) {
        self.move_up(-distance);
    }

    pub fn move_forward(&mut self, distance: f32) {
        self.position += distance * self.view_direction.as_ref();
        self.update_view_matrix();
    }

    pub fn move_backward(&mut self, distance: f32) {
        self.move_forward(-distance);
    }

    pub fn move_right(&mut self, distance: f32) {
        let right = na::Unit::new_normalize(self.down_direction.cross(&self.view_direction));
        self.position += distance * right.as_ref();
        self.update_view_matrix();
    }

    pub fn move_left(&mut self, distance: f32) {
        self.move_right(-distance);
    }

    pub fn turn_right(&mut self, angle: f32) {
        let rotation = na::Rotation3::from_axis_angle(&self.down_direction, angle);
        self.view_direction = rotation * self.view_direction;
        self.update_view_matrix();
    }

    pub fn turn_left(&mut self, angle: f32) {
        self.turn_right(-angle);
    }

    pub fn turn_up(&mut self, angle: f32) {
        let right = na::Unit::new_normalize(self.down_direction.cross(&self.view_direction));
        let rotation = na::Rotation3::from_axis_angle(&right, angle);
        self.view_direction = rotation * self.view_direction;
        self.down_direction = rotation * self.down_direction;
        self.update_view_matrix();
    }

    pub fn turn_down(&mut self, angle: f32) {
        self.turn_up(-angle);
    }

    pub fn set_aspect(&mut self, ratio: f32) {
        self.aspect = ratio;
        self.update_projection_matrix();
    }

    pub(crate) fn update_buffer(
        &self,
        allocator: &mut Allocator,
        buffer: &mut Buffer,
        offset: usize,
    ) -> RendererResult<()> {
        let data_array: [[[f32; 4]; 4]; 2] =
            [self.view_matrix.into(), self.projection_matrix.into()];
        buffer.copy_to_offset(allocator, &data_array, offset)
    }
}

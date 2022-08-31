use ash::vk;

pub struct Queue {
    pub index: u32,
    pub queue: vk::Queue,
}
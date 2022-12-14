use std::cell::RefCell;
use std::rc::Rc;

use vulkan_rust::renderer::utils::create_render_window;
use winit::event::{Event, WindowEvent};

use nalgebra as na;
use nalgebra_glm as glm;

use vulkan_rust::renderer::camera::Camera;
use vulkan_rust::renderer::light::{DirectionalLight, LightManager, PointLight};
use vulkan_rust::renderer::model::Model;
use vulkan_rust::renderer::scene::{SceneObject, SceneTree};
use vulkan_rust::renderer::vertex::Vertex;
use vulkan_rust::renderer::InstanceData;
use vulkan_rust::renderer::{error::RendererError, Renderer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (event_loop, window, internal_window) = create_render_window()?;
    let window_size = window.inner_size();
    let mut renderer = Renderer::new(
        "My Game Engine",
        window_size.width,
        window_size.height,
        internal_window
    )?;

    let sphere = Rc::new(RefCell::new(Model::<Vertex, InstanceData>::sphere(3)));

    let scene_tree = SceneTree::default();
    let root = scene_tree.get_root();
    for i in 0..10 {
        for j in 0..10 {
            let translation = glm::Vec3::new(i as f32 - 5., j as f32 + 5., 10.0);
            let scale = 0.5f32;
            let metallic = i as f32 * 0.1;
            let roughness = j as f32 * 0.1;
            let texture_id = ((i + j) % 3) as u32;

            let new_object = SceneObject::new_empty();
            SceneObject::add_child(&root, &new_object);
            {
                let mut obj_ref = new_object.borrow_mut();
                obj_ref.position = translation;
                obj_ref.scaling = glm::Vec3::new(scale, scale, scale);
                obj_ref.metallic = metallic;
                obj_ref.roughness = roughness;
                obj_ref.texture_id = texture_id;
                obj_ref.set_model(&sphere)?;
                obj_ref.update_transform(true)?;
            }
        }
    }
    if let Some(allo) = &mut renderer.allocator {
        // TODO how to structure this properly?
        let mut s = sphere.borrow_mut();
        s.update_vertex_buffer(
            &renderer.context.device,
            allo,
            renderer.buffer_manager.clone(),
        )?;
        s.update_index_buffer(
            &renderer.context.device,
            allo,
            renderer.buffer_manager.clone(),
        )?;
        s.update_instance_buffer(
            &renderer.context.device,
            allo,
            renderer.buffer_manager.clone(),
        )?;
    }
    renderer.models = vec![sphere];

    let mut lights = LightManager::default();
    lights.add_light(DirectionalLight {
        direction: na::Unit::new_normalize(glm::Vec3::new(-1., -1., 0.)),
        illuminance: glm::Vec3::new(10.1, 10.1, 10.1),
    });
    lights.add_light(PointLight {
        position: na::Point3::new(0.1, -3.0, -3.0),
        luminous_flux: glm::Vec3::new(100.0, 100.0, 100.0),
    });
    lights.add_light(PointLight {
        position: na::Point3::new(1.5, 0.0, 0.0),
        luminous_flux: glm::Vec3::new(10.0, 10.0, 10.0),
    });
    lights.add_light(PointLight {
        position: na::Point3::new(1.5, 0.2, 0.0),
        luminous_flux: glm::Vec3::new(5.0, 5.0, 5.0),
    });
    renderer.update_storage_from_lights(&lights)?;

    let mut camera = Camera::builder().build();

    let mut move_up_pressed = false;
    let mut move_down_pressed = false;
    let mut move_forward_pressed = false;
    let mut move_backward_pressed = false;
    let mut move_right_pressed = false;
    let mut move_left_pressed = false;
    let mut turn_up_pressed = false;
    let mut turn_down_pressed = false;
    let mut turn_right_pressed = false;
    let mut turn_left_pressed = false;

    let _tex1_index = renderer.new_texture_from_file("texture.png")?;
    let _tex2_index = renderer.new_texture_from_file("texture2.jpg")?;
    let _tex3_index = renderer.new_texture_from_file("texture3.jpg")?;
    renderer.update_textures()?;

    // Create some text
    renderer.add_text(
        &window,
        (100, 200),
        &[
            &fontdue::layout::TextStyle::new("Hello ", 35.0, 0),
            &fontdue::layout::TextStyle::new("world!", 40.0, 0),
            &fontdue::layout::TextStyle::new("(and smaller)", 8.0, 0),
        ],
        [0., 1., 0.],
    )?;
    renderer.add_text(
        &window,
        (100, 200),
        &[
            &fontdue::layout::TextStyle::new("Hello ", 35.0, 0),
            &fontdue::layout::TextStyle::new("world!", 40.0, 0),
            &fontdue::layout::TextStyle::new("(and smaller)", 8.0, 0),
        ],
        [0.6, 0.6, 0.6],
    )?;

    // Run event loop
    let mut running = true;
    let mut now = std::time::SystemTime::now();
    let mut fps_id = renderer.add_text(
        &window,
        (0, 50),
        &[&fontdue::layout::TextStyle::new("FPS: 0000.00", 20.0, 0)],
        [1.0, 1.0, 1.0],
    )?;
    event_loop.run(move |event, _, controlflow| match event {
        Event::WindowEvent {
            event: WindowEvent::Resized(size),
            ..
        } => {
            renderer
                .recreate_swapchain(size.width, size.height)
                .expect("Recreate Swapchain");
            camera.set_aspect(size.width as f32 / size.height as f32);
            renderer
                .update_uniforms_from_camera(&camera)
                .expect("camera buffer update");
            renderer
                .update_storage_from_lights(&lights)
                .expect("light update");
        }
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            running = false;
            *controlflow = winit::event_loop::ControlFlow::Exit;
        }
        Event::WindowEvent {
            event:
                WindowEvent::KeyboardInput {
                    input:
                        winit::event::KeyboardInput {
                            state: pressed,
                            virtual_keycode: Some(keycode),
                            ..
                        },
                    ..
                },
            ..
        } => match keycode {
            winit::event::VirtualKeyCode::Space => {
                move_up_pressed = match pressed {
                    winit::event::ElementState::Pressed => true,
                    winit::event::ElementState::Released => false,
                };
            }
            winit::event::VirtualKeyCode::C => {
                move_down_pressed = match pressed {
                    winit::event::ElementState::Pressed => true,
                    winit::event::ElementState::Released => false,
                };
            }
            winit::event::VirtualKeyCode::W => {
                move_forward_pressed = match pressed {
                    winit::event::ElementState::Pressed => true,
                    winit::event::ElementState::Released => false,
                };
            }
            winit::event::VirtualKeyCode::S => {
                move_backward_pressed = match pressed {
                    winit::event::ElementState::Pressed => true,
                    winit::event::ElementState::Released => false,
                };
            }
            winit::event::VirtualKeyCode::A => {
                move_left_pressed = match pressed {
                    winit::event::ElementState::Pressed => true,
                    winit::event::ElementState::Released => false,
                };
            }
            winit::event::VirtualKeyCode::D => {
                move_right_pressed = match pressed {
                    winit::event::ElementState::Pressed => true,
                    winit::event::ElementState::Released => false,
                };
            }
            winit::event::VirtualKeyCode::E => {
                turn_right_pressed = match pressed {
                    winit::event::ElementState::Pressed => true,
                    winit::event::ElementState::Released => false,
                };
            }
            winit::event::VirtualKeyCode::Q => {
                turn_left_pressed = match pressed {
                    winit::event::ElementState::Pressed => true,
                    winit::event::ElementState::Released => false,
                };
            }
            winit::event::VirtualKeyCode::R => {
                turn_up_pressed = match pressed {
                    winit::event::ElementState::Pressed => true,
                    winit::event::ElementState::Released => false,
                };
            }
            winit::event::VirtualKeyCode::F => {
                turn_down_pressed = match pressed {
                    winit::event::ElementState::Pressed => true,
                    winit::event::ElementState::Released => false,
                };
            }
            winit::event::VirtualKeyCode::F12 => {
                if matches!(pressed, winit::event::ElementState::Pressed) {
                    renderer.screenshot().expect("Could not take screenshot");
                    println!("Screenshotted!");
                }
            }
            winit::event::VirtualKeyCode::Escape => {
                running = false;
                *controlflow = winit::event_loop::ControlFlow::Exit;
            }
            _ => {}
        },
        Event::MainEventsCleared => {
            // doing the work here (later)
            window.request_redraw();
        }
        Event::RedrawRequested(_) => {
            if !running {
                return;
            }
            let temp = now;
            now = std::time::SystemTime::now();
            let diff = 1.0 / now.duration_since(temp).unwrap_or_default().as_secs_f32();
            let text = format!("FPS: {:.02}", diff);
            renderer
                .remove_text(fps_id)
                .expect("Could not remove old fps text");
            fps_id = renderer
                .add_text(
                    &window,
                    (0, 100),
                    &[&fontdue::layout::TextStyle::new(&text, 20.0, 0)],
                    [1.0, 1.0, 1.0],
                )
                .expect("Could not add fps text");
            const MOVE_SPEED: f32 = 0.05f32;
            const TURN_SPEED: f32 = 0.005f32;
            if move_up_pressed {
                camera.move_up(MOVE_SPEED);
            }
            if move_down_pressed {
                camera.move_down(MOVE_SPEED);
            }
            if move_forward_pressed {
                camera.move_forward(MOVE_SPEED);
            }
            if move_backward_pressed {
                camera.move_backward(MOVE_SPEED);
            }
            if move_right_pressed {
                camera.move_right(MOVE_SPEED);
            }
            if move_left_pressed {
                camera.move_left(MOVE_SPEED);
            }
            if turn_up_pressed {
                camera.turn_up(TURN_SPEED);
            }
            if turn_down_pressed {
                camera.turn_down(TURN_SPEED);
            }
            if turn_right_pressed {
                camera.turn_right(TURN_SPEED);
            }
            if turn_left_pressed {
                camera.turn_left(TURN_SPEED);
            }
            renderer
                .update_uniforms_from_camera(&camera)
                .expect("Could not update uniform buffer!");
            let result = renderer.render();
            match result {
                Ok(_) => {}
                Err(RendererError::VulkanError(ash::vk::Result::ERROR_OUT_OF_DATE_KHR)) => {} // Resize request will update swapchain
                _ => result.expect("render error"),
            }
        }
        _ => {}
    });
}

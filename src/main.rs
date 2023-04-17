use std::ops::DerefMut;

use ash::vk;
use gpu_allocator::MemoryLocation;
use log::info;
use vulkan_rust::renderer::buffer::BufferManager;
use vulkan_rust::renderer::material::{MaterialData, ShaderParameters};
use vulkan_rust::renderer::utils::create_render_window;
use winit::event::{Event, WindowEvent};

use nalgebra as na;
use nalgebra_glm as glm;

use vulkan_rust::renderer::camera::Camera;
use vulkan_rust::renderer::light::{DirectionalLight, LightManager, PointLight};
use vulkan_rust::renderer::{error::RendererError, Renderer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    log4rs::init_file("log4rs.yml", Default::default()).unwrap();
    let (event_loop, window, internal_window) = create_render_window()?;
    let window_size = window.inner_size();
    let mut renderer = Renderer::new(
        "My Game Engine",
        window_size.width,
        window_size.height,
        internal_window,
    )?;

    let tex1_handle = renderer.new_texture_from_file("texture.png")?;
    let tex2_handle = renderer.new_texture_from_file("texture2.jpg")?;
    let tex3_handle = renderer.new_texture_from_file("texture3.jpg")?;
    let tex4_handle = renderer.new_texture_from_file("plain_white.jpg")?;

    let sphere = if let Ok(mut allo) = renderer.allocator.lock() {
        renderer.meshs.new_sphere_mesh(
            3,
            &renderer.context.device,
            allo.deref_mut(),
            renderer.buffer_manager.clone(),
        )?
    } else {
        panic!("No allocator!");
    };

    for i in 0..10 {
        for j in 0..10 {
            let translation = glm::Vec3::new(i as f32 - 5., j as f32 + 5., 10.0);
            let scale = 0.5f32;
            let metallic = i as f32 * 0.1;
            let roughness = j as f32 * 0.1;
            let texture_id = ((i + j) % 3) as u32;

            let tex_handle = match texture_id {
                0 => tex1_handle,
                1 => tex2_handle,
                2 => tex3_handle,
                _ => unreachable!(),
            };

            if let Ok(mut allo) = renderer.allocator.lock() {
                // allocate uniform buffer
                let mut buffer = BufferManager::new_buffer(
                    renderer.buffer_manager.clone(),
                    &renderer.context.device,
                    allo.deref_mut(),
                    2 * 4,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    MemoryLocation::CpuToGpu,
                    format!("uniforms-sphere-{}-{}", i, j).as_str(),
                )?;
                let material_data = [metallic, roughness];
                buffer.fill(allo.deref_mut(), &material_data)?;
                let mat_data = MaterialData {
                    textures: vec![tex_handle],
                    buffers: vec![buffer.get_handle()],
                    parameters: ShaderParameters::default(),
                    base_template: "default".to_string(),
                };
                let mat_name = format!("mat_{}_{}", metallic, roughness);
                let material_handle = renderer.material_system.build_material(
                    &renderer.context.device,
                    &renderer.texture_storage,
                    renderer.buffer_manager.clone(),
                    &mut renderer.descriptor_layout_cache,
                    &mut renderer.descriptor_allocator,
                    mat_name.as_str(),
                    mat_data,
                )?;
                renderer.material_uniform_buffers.push(buffer);

                let new_object = renderer.scene_tree.new_object(
                    sphere,
                    material_handle,
                    &renderer.context.device,
                    allo.deref_mut(),
                    renderer.buffer_manager.clone(),
                )?;
                {
                    let obj_ref = renderer
                        .scene_tree
                        .get_object_mut(new_object, allo.deref_mut())
                        .expect("We were given an invalid handle");
                    obj_ref.object.position = translation;
                    obj_ref.object.scaling = glm::Vec3::new(scale, scale, scale);
                }
            } else {
                panic!("No allocator!");
            }
        }
    }

    // Try loading an obj model
    let car_model = if let Ok(mut allo) = renderer.allocator.lock() {
        renderer.meshs.new_mesh_from_obj(
            "models/alfa147.obj",
            &renderer.context.device,
            allo.deref_mut(),
            renderer.buffer_manager.clone(),
        )?
    } else {
        panic!("No allocator!");
    };
    let car_base_position = glm::Vec3::new(0f32, 15f32, 20f32);
    let car_handle = {
        if let Ok(mut allo) = renderer.allocator.lock() {
            // allocate uniform buffer
            let mut buffer = BufferManager::new_buffer(
                renderer.buffer_manager.clone(),
                &renderer.context.device,
                allo.deref_mut(),
                2 * 4,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                MemoryLocation::CpuToGpu,
                "uniforms-car",
            )?;
            let material_data = [0.8f32, 0.1f32];
            buffer.fill(allo.deref_mut(), &material_data)?;
            let mat_data = MaterialData {
                textures: vec![tex4_handle],
                buffers: vec![buffer.get_handle()],
                parameters: ShaderParameters::default(),
                base_template: "default".to_string(),
            };
            let material_handle = renderer.material_system.build_material(
                &renderer.context.device,
                &renderer.texture_storage,
                renderer.buffer_manager.clone(),
                &mut renderer.descriptor_layout_cache,
                &mut renderer.descriptor_allocator,
                "car_material",
                mat_data,
            )?;
            renderer.material_uniform_buffers.push(buffer);
            let child_object = {
                let h = renderer.scene_tree.new_object(
                    sphere,
                    material_handle,
                    &renderer.context.device,
                    allo.deref_mut(),
                    renderer.buffer_manager.clone(),
                )?;
                let obj_ref = renderer
                    .scene_tree
                    .get_object_mut(h, allo.deref_mut())
                    .expect("We were given an invalid handle");
                obj_ref.object.position = glm::Vec3::new(0f32, 0f32, 65f32);
                obj_ref.object.scaling = glm::Vec3::new(10.0f32, 10.0f32, 10.0f32);
                h
            };
            let new_object = renderer.scene_tree.new_object(
                car_model,
                material_handle,
                &renderer.context.device,
                allo.deref_mut(),
                renderer.buffer_manager.clone(),
            )?;
            {
                let mut obj_ref = renderer
                    .scene_tree
                    .get_object_mut(new_object, allo.deref_mut())
                    .expect("We were given an invalid handle");
                obj_ref.object.position = car_base_position;
                obj_ref.object.scaling = glm::Vec3::new(0.1f32, 0.1f32, 0.1f32);
                obj_ref.object.rotation = glm::Quat::from_polar_decomposition(
                    1.0f32,
                    std::f32::consts::FRAC_2_PI,
                    na::Unit::<glm::Vec3>::new_normalize(glm::Vec3::new(1.0f32, 0.0f32, 0.0f32)),
                );
                obj_ref.add_child(child_object)?;
            }
            new_object
        } else {
            panic!("No allocator!");
        }
    };

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

    let mut speed_factor = 1.0f32;
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
        (100, 300),
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
    let start_time = std::time::SystemTime::now();
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
            winit::event::VirtualKeyCode::LShift => {
                speed_factor = match pressed {
                    winit::event::ElementState::Pressed => 2.0f32,
                    winit::event::ElementState::Released => 1.0f32,
                };
            }
            winit::event::VirtualKeyCode::F12 => {
                if matches!(pressed, winit::event::ElementState::Pressed) {
                    renderer.screenshot().expect("Could not take screenshot");
                    info!("Screenshotted!");
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
                .remove_text(fps_id[0])
                .expect("Could not remove old fps text");
            fps_id = renderer
                .add_text(
                    &window,
                    (0, 100),
                    &[&fontdue::layout::TextStyle::new(&text, 20.0, 0)],
                    [1.0, 1.0, 1.0],
                )
                .expect("Could not add fps text");
            let move_speed: f32 = 0.05f32 * speed_factor;
            let turn_speed: f32 = 0.005f32 * speed_factor;
            if move_up_pressed {
                camera.move_up(move_speed);
            }
            if move_down_pressed {
                camera.move_down(move_speed);
            }
            if move_forward_pressed {
                camera.move_forward(move_speed);
            }
            if move_backward_pressed {
                camera.move_backward(move_speed);
            }
            if move_right_pressed {
                camera.move_right(move_speed);
            }
            if move_left_pressed {
                camera.move_left(move_speed);
            }
            if turn_up_pressed {
                camera.turn_up(turn_speed);
            }
            if turn_down_pressed {
                camera.turn_down(turn_speed);
            }
            if turn_right_pressed {
                camera.turn_right(turn_speed);
            }
            if turn_left_pressed {
                camera.turn_left(turn_speed);
            }
            {
                if let Ok(mut allo) = renderer.allocator.lock() {
                    let obj_ref = renderer
                        .scene_tree
                        .get_object_mut(car_handle, allo.deref_mut())
                        .expect("Could not get car obj mut ref");
                    obj_ref.object.position = glm::Vec3::new(
                        car_base_position.x,
                        car_base_position.y,
                        car_base_position.z
                            + start_time
                                .elapsed()
                                .expect("Could not get elapsed time")
                                .as_secs_f32()
                                .sin()
                                * 5.0f32,
                    );
                }
            }
            let result = renderer.render(&camera);
            match result {
                Ok(_) => {}
                Err(RendererError::VulkanError {
                    source: ash::vk::Result::ERROR_OUT_OF_DATE_KHR,
                    ..
                }) => {} // Resize request will update swapchain
                _ => result.expect("render error"),
            }
        }
        _ => {}
    });
}

#[cfg(not(target_os = "windows"))]
use std::ffi::c_void;

#[cfg(target_os = "windows")]
use raw_window_handle::{HasRawWindowHandle, RawWindowHandle, Win32WindowHandle};
use winit::event::{Event, WindowEvent};

#[cfg(not(target_os = "windows"))]
use winit::platform::unix::WindowExtUnix;

use nalgebra_glm as na;

use vulkan_rust::renderer::model::Model;
use vulkan_rust::renderer::vertex::Vertex;
use vulkan_rust::renderer::InstanceData;
use vulkan_rust::renderer::Renderer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    // Create window
    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::Window::new(&event_loop)?;

    // Get display and Window
    #[cfg(not(target_os = "windows"))]
    let (display, surface, is_wayland) = {
        if let Some(display) = window.xlib_display() {
            (
                display,
                window.xlib_window().expect("Got X11 Display but no window") as *mut c_void,
                false,
            )
        } else if let Some(display) = window.wayland_display() {
            (
                display,
                window
                    .wayland_surface()
                    .expect("Got Wayland Display but no surface"),
                true,
            )
        } else {
            panic!("No X11 or Wayland Display available!");
        }
    };
    #[cfg(target_os = "windows")]
    let (display, surface, is_wayland) = {
        if let RawWindowHandle::Win32(Win32WindowHandle {
            hwnd, hinstance, ..
        }) = window.raw_window_handle()
        {
            (hinstance, hwnd, false)
        } else {
            panic!("Could not setup window");
        }
    };
    let mut renderer = Renderer::new("My Game Engine", surface, display, is_wayland)?;

    let mut cube = Model::<Vertex, InstanceData>::cube();
    cube.insert_visibly(InstanceData {
        model_matrix: (na::Mat4::new_translation(&na::Vec3::new(0.05, 0.05, 0.1))
            * na::Mat4::new_scaling(0.1))
        .into(),
        color_mod: [1.0, 1.0, 0.2],
    });
    cube.insert_visibly(InstanceData {
        model_matrix: (na::Mat4::new_translation(&na::Vec3::new(0.05, -0.05, 0.5))
            * na::Mat4::new_scaling(0.1))
        .into(),
        color_mod: [0.2, 0.4, 1.0],
    });
    cube.insert_visibly(InstanceData {
        model_matrix: na::Mat4::new_scaling(0.1).into(),
        color_mod: [1.0, 0.0, 0.0],
    });
    cube.insert_visibly(InstanceData {
        model_matrix: (na::Mat4::new_translation(&na::Vec3::new(0.0, 0.25, 0.0))
            * na::Mat4::new_scaling(0.1))
        .into(),
        color_mod: [0.0, 1.0, 0.0],
    });
    cube.insert_visibly(InstanceData {
        model_matrix: (na::Mat4::new_translation(&na::Vec3::new(0.0, 0.5, 0.0))
            * na::Mat4::new_scaling(0.1))
        .into(),
        color_mod: [0.0, 1.0, 0.0],
    });
    let mut angle = 0.2;
    let rotating_cube_handle = cube.insert_visibly(InstanceData {
        model_matrix: (na::Mat4::from_scaled_axis(na::Vec3::new(0.0, 0.0, angle))
            * na::Mat4::new_translation(&na::Vec3::new(0.0, 0.5, 0.0))
            * na::Mat4::new_scaling(0.1))
        .into(),
        color_mod: [1.0, 1.0, 1.0],
    });
    if let Some(allo) = &mut renderer.allocator {
        cube.update_vertex_buffer(&renderer.device, allo)?;
        cube.update_instance_buffer(&renderer.device, allo)?;
    }
    renderer.models = vec![cube];

    // Run event loop
    let mut running = true;
    event_loop.run(move |event, _, controlflow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            running = false;
            *controlflow = winit::event_loop::ControlFlow::Exit;
        }
        Event::MainEventsCleared => {
            // doing the work here (later)
            window.request_redraw();
        }
        Event::RedrawRequested(_) => {
            if !running {
                return;
            }
            // render here
            angle += 0.01;
            renderer.models[0]
                .get_mut(rotating_cube_handle)
                .expect("missing instance!")
                .model_matrix = (na::Mat4::from_scaled_axis(na::Vec3::new(0.0, 0.0, angle))
                * na::Mat4::new_translation(&na::Vec3::new(0.0, 0.5, 0.0))
                * na::Mat4::new_scaling(0.1))
            .into();
            renderer.render().expect("Could not render frame!");
        }
        _ => {}
    });
}

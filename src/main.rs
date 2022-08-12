use ash::vk;

use winit::event::{Event, WindowEvent};
use winit::platform::unix::WindowExtUnix;

use vulkan_rust::renderer::Renderer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create window
    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::Window::new(&event_loop)?;

    // Create x11 surface
    // TODO cross platform windowing?
    let x11_display = window.xlib_display().unwrap();
    let x11_window = window.xlib_window().unwrap();

    let mut renderer = Renderer::new("My Game Engine", x11_window, x11_display as *mut vk::Display)?;

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
            renderer.render().expect("Could not render frame!");
            
        }
        _ => {}
    });
}

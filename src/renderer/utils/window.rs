use std::ffi::c_void;

#[cfg(target_os = "windows")]
use raw_window_handle::{HasRawWindowHandle, RawWindowHandle, Win32WindowHandle};
#[cfg(target_os = "macos")]
use raw_window_metal::CAMetalLayer;
#[cfg(target_os = "macos")]
use winit::platform::macos::WindowExtMacOS;
#[cfg(target_os = "linux")]
use winit::platform::unix::WindowExtUnix;

use winit::{dpi::PhysicalSize, error::OsError, event_loop::EventLoop, window::Window};

#[derive(Copy, Clone, Debug)]
pub enum InternalWindow {
    WindowsWindow {
        hinstance: *const c_void,
        hwnd: *const c_void,
    },
    MacOsWindow {
        #[cfg(target_os = "macos")]
        layer: CAMetalLayer,
        #[cfg(not(target_os = "macos"))]
        layer: *const c_void,
    },
    LinuxWindow {
        display: *mut c_void,
        surface: *mut c_void,
        is_wayland: bool,
    },
}

impl InternalWindow {
    #[cfg(target_os = "linux")]
    fn new(window: &Window) -> Self {
        let (display, window, is_wayland) = if let Some(display) = window.xlib_display() {
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
        };
        Self::LinuxWindow {
            display,
            surface: window,
            is_wayland,
        }
    }

    #[cfg(target_os = "windows")]
    fn new(window: &Window) -> Self {
        let (hwnd, hinstance) = {
            if let RawWindowHandle::Win32(Win32WindowHandle {
                hwnd, hinstance, ..
            }) = window.raw_window_handle()
            {
                (hwnd, hinstance)
            } else {
                panic!("Could not setup window");
            }
        };
        Self::WindowsWindow { hinstance, hwnd }
    }

    #[cfg(target_os = "macos")]
    fn new(window: &Window) -> Self {
        let ns_window = window.ns_window();
        let ns_view = window.ns_view();
        let mut handle = raw_window_handle::AppKitWindowHandle::empty();
        handle.ns_window = ns_window;
        handle.ns_view = ns_view;
        let layer = unsafe { raw_window_metal::appkit::metal_layer_from_handle(handle) };
        match layer {
            raw_window_metal::Layer::Existing(layer) => Self::MacOsWindow { layer },
            // TODO do we need to deallocate this??
            raw_window_metal::Layer::Allocated(layer) => Self::MacOsWindow { layer },
            raw_window_metal::Layer::None => panic!("Could not get CAMetalLayer!"),
        }
    }
}

pub fn create_render_window() -> Result<(EventLoop<()>, Window, InternalWindow), OsError> {
    // Create window
    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_inner_size(PhysicalSize::new(1280, 720))
        .build(&event_loop)?;

    let internal_window = InternalWindow::new(&window);

    Ok((event_loop, window, internal_window))
}

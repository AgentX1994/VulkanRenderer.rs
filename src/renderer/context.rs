use std::ffi::{c_void, CStr, CString};

use ash::{
    extensions::{ext, khr},
    vk, Instance,
};
use log::{debug, error, info, warn};

use super::{queue::Queue, utils::InternalWindow, RendererResult};

unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    p_user_data: *mut c_void,
) -> vk::Bool32 {
    let context = unsafe { &*(p_user_data as *const VulkanContext) };
    context.handle_debug_callback(message_severity, message_type, p_callback_data)
}

pub struct VulkanContext {
    _entry: ash::Entry,
    pub instance: Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub max_texture_extent: vk::Extent3D, // TODO I think this should be queryable dynamically
    pub surface: vk::SurfaceKHR,
    pub surface_loader: khr::Surface,
    pub surface_capabilities: vk::SurfaceCapabilitiesKHR,
    pub surface_present_modes: Vec<vk::PresentModeKHR>,
    pub surface_formats: Vec<vk::SurfaceFormatKHR>,
    pub transfer_queue: Queue,
    pub graphics_queue: Queue,
    debug_utils: ext::DebugUtils,
    utils_messenger: vk::DebugUtilsMessengerEXT,
}

impl VulkanContext {
    fn create_instance(
        engine_name: &str,
        app_name: &str,
        entry: &ash::Entry,
        layer_names: &[*const i8],
        mut debug_create_info: vk::DebugUtilsMessengerCreateInfoEXT,
        internal_window: InternalWindow,
    ) -> RendererResult<Instance> {
        // TODO Return errors
        let engine_name_c = CString::new(engine_name).unwrap();
        let app_name_c = CString::new(app_name).unwrap();

        // Make Application Info
        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name_c)
            .application_version(vk::make_api_version(0, 0, 0, 1))
            .engine_name(&engine_name_c)
            .engine_version(vk::make_api_version(0, 0, 42, 0))
            .api_version(vk::API_VERSION_1_3);

        let mut instance_extension_names = vec![
            ext::DebugUtils::name().as_ptr(),
            khr::Surface::name().as_ptr(),
        ];
        match internal_window {
            InternalWindow::WindowsWindow { .. } => {
                instance_extension_names.push(khr::Win32Surface::name().as_ptr());
            }
            InternalWindow::LinuxWindow { is_wayland, .. } => {
                if is_wayland {
                    instance_extension_names.push(khr::WaylandSurface::name().as_ptr());
                } else {
                    instance_extension_names.push(khr::XlibSurface::name().as_ptr());
                }
            }
            InternalWindow::MacOsWindow { .. } => {
                instance_extension_names.push(vk::ExtMetalSurfaceFn::name().as_ptr());
                instance_extension_names.push(vk::KhrPortabilityEnumerationFn::name().as_ptr());
            }
        }

        // Create instance
        let mut instance_create_info = vk::InstanceCreateInfo::builder()
            .push_next(&mut debug_create_info)
            .application_info(&app_info)
            .enabled_layer_names(layer_names)
            .enabled_extension_names(&instance_extension_names);

        if matches!(internal_window, InternalWindow::MacOsWindow { .. }) {
            instance_create_info =
                instance_create_info.flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);
        }

        unsafe { Ok(entry.create_instance(&instance_create_info, None)?) }
    }

    fn pick_physical_device(
        instance: &Instance,
    ) -> RendererResult<(vk::PhysicalDevice, vk::PhysicalDeviceProperties)> {
        // Physical Device
        let phys_devs = unsafe { instance.enumerate_physical_devices()? };

        let mut chosen = None;
        for p in phys_devs {
            let props = unsafe { instance.get_physical_device_properties(p) };
            info!("{:?}", props);
            if chosen.is_none() {
                chosen = Some((p, props));
            }
        }
        chosen.ok_or_else(|| vk::Result::ERROR_UNKNOWN.into())
    }

    fn pick_queues(
        instance: &Instance,
        physical_device: &vk::PhysicalDevice,
        surface: &vk::SurfaceKHR,
        surface_loader: &khr::Surface,
    ) -> RendererResult<(u32, u32)> {
        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(*physical_device) };
        let mut g_index = None;
        let mut t_index = None;
        for (i, qfam) in queue_family_properties.iter().enumerate() {
            if qfam.queue_count > 0
                && qfam.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                && unsafe {
                    surface_loader.get_physical_device_surface_support(
                        *physical_device,
                        i as u32,
                        *surface,
                    )?
                }
            {
                g_index = Some(i as u32);
            }
            if qfam.queue_count > 0
                && qfam.queue_flags.contains(vk::QueueFlags::TRANSFER)
                && (t_index.is_none() || !qfam.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            {
                t_index = Some(i as u32);
            }
        }
        Ok((
            g_index.ok_or(vk::Result::ERROR_UNKNOWN)?,
            t_index.ok_or(vk::Result::ERROR_UNKNOWN)?,
        ))
    }

    fn create_logical_device(
        instance: &Instance,
        physical_device: &vk::PhysicalDevice,
        layers: &[*const i8],
        graphics_queue_index: u32,
        transfer_queue_index: u32,
    ) -> RendererResult<ash::Device> {
        let device_extension_names = [
            ash::extensions::khr::Swapchain::name().as_ptr(),
            #[cfg(target_os = "macos")]
            vk::KhrPortabilitySubsetFn::name().as_ptr(),
        ];

        // create logical device
        let priorities = [1.0f32];
        let queue_infos = if graphics_queue_index != transfer_queue_index {
            vec![
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(graphics_queue_index)
                    .queue_priorities(&priorities)
                    .build(),
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(transfer_queue_index)
                    .queue_priorities(&priorities)
                    .build(),
            ]
        } else {
            vec![vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(graphics_queue_index)
                .queue_priorities(&priorities)
                .build()]
        };

        let mut indexing_features = vk::PhysicalDeviceDescriptorIndexingFeatures::builder()
            .runtime_descriptor_array(true)
            .descriptor_binding_variable_descriptor_count(true);

        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_extension_names)
            .enabled_layer_names(layers)
            .push_next(&mut indexing_features);
        let device =
            unsafe { instance.create_device(*physical_device, &device_create_info, None)? };
        Ok(device)
    }

    pub fn new(name: &str, internal_window: InternalWindow) -> RendererResult<Self> {
        // Layers
        let layers = unsafe {
            [CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0").as_ptr()]
        };

        let entry = unsafe { ash::Entry::load()? };
        // Messenger info
        let debug_create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            )
            .pfn_user_callback(Some(vulkan_debug_utils_callback));

        let instance = Self::create_instance(
            name,
            "My Engine",
            &entry,
            &layers[..],
            *debug_create_info,
            internal_window,
        )?;

        // Create debug messenger
        let debug_utils = ext::DebugUtils::new(&entry, &instance);
        let utils_messenger =
            unsafe { debug_utils.create_debug_utils_messenger(&debug_create_info, None)? };

        // Create surface
        let surface = match internal_window {
            InternalWindow::WindowsWindow { hinstance, hwnd } => {
                let win32_create_info = vk::Win32SurfaceCreateInfoKHR::builder()
                    .hinstance(hinstance)
                    .hwnd(hwnd);
                let win32_surface_loader =
                    ash::extensions::khr::Win32Surface::new(&entry, &instance);
                unsafe { win32_surface_loader.create_win32_surface(&win32_create_info, None)? }
            }
            InternalWindow::MacOsWindow { layer } => {
                let metal_create_info =
                    vk::MetalSurfaceCreateInfoEXT::builder().layer(layer as *const c_void);
                let metal_surface_loader = ext::MetalSurface::new(&entry, &instance);
                unsafe { metal_surface_loader.create_metal_surface(&metal_create_info, None)? }
            }
            InternalWindow::LinuxWindow {
                display,
                surface,
                is_wayland,
            } => {
                if is_wayland {
                    let wayland_create_info = vk::WaylandSurfaceCreateInfoKHR::builder()
                        .display(display)
                        .surface(surface);
                    let wayland_surface_loader =
                        ash::extensions::khr::WaylandSurface::new(&entry, &instance);
                    unsafe {
                        wayland_surface_loader.create_wayland_surface(&wayland_create_info, None)?
                    }
                } else {
                    #[cfg(not(target_os = "windows"))]
                    let window = surface as u64;
                    #[cfg(target_os = "windows")]
                    let window = surface as u32;
                    let x11_create_info = vk::XlibSurfaceCreateInfoKHR::builder()
                        .window(window)
                        .dpy(display as *mut *const c_void);
                    let xlib_surface_loader =
                        ash::extensions::khr::XlibSurface::new(&entry, &instance);
                    unsafe { xlib_surface_loader.create_xlib_surface(&x11_create_info, None)? }
                }
            }
        };

        let surface_loader = ash::extensions::khr::Surface::new(&entry, &instance);

        let (physical_device, _physical_device_properties) = Self::pick_physical_device(&instance)?;
        let (graphics_queue_index, transfer_queue_index) =
            Self::pick_queues(&instance, &physical_device, &surface, &surface_loader)?;

        let device = Self::create_logical_device(
            &instance,
            &physical_device,
            &layers[..],
            graphics_queue_index,
            transfer_queue_index,
        )?;

        let graphics_queue = Queue {
            index: graphics_queue_index,
            queue: unsafe { device.get_device_queue(graphics_queue_index, 0) },
        };

        let transfer_queue = Queue {
            index: transfer_queue_index,
            queue: unsafe { device.get_device_queue(transfer_queue_index, 0) },
        };

        // Get capabilities of the surface
        let surface_capabilities = unsafe {
            surface_loader.get_physical_device_surface_capabilities(physical_device, surface)?
        };
        let surface_present_modes = unsafe {
            surface_loader.get_physical_device_surface_present_modes(physical_device, surface)?
        };
        let surface_formats = unsafe {
            surface_loader.get_physical_device_surface_formats(physical_device, surface)?
        };

        // TODO this is only for the text atlas textures
        let limits = unsafe {
            instance.get_physical_device_image_format_properties(
                physical_device,
                vk::Format::R8_SRGB,
                vk::ImageType::TYPE_2D,
                vk::ImageTiling::OPTIMAL,
                vk::ImageUsageFlags::SAMPLED,
                vk::ImageCreateFlags::empty(),
            )?
        };

        Ok(Self {
            _entry: entry,
            instance,
            physical_device,
            max_texture_extent: limits.max_extent,
            device,
            surface,
            surface_loader,
            surface_capabilities,
            surface_present_modes,
            surface_formats,
            graphics_queue,
            transfer_queue,
            debug_utils,
            utils_messenger,
        })
    }

    pub fn refresh_surface_data(&mut self) -> RendererResult<()> {
        // Get capabilities of the surface
        self.surface_capabilities = unsafe {
            self.surface_loader
                .get_physical_device_surface_capabilities(self.physical_device, self.surface)?
        };
        self.surface_present_modes = unsafe {
            self.surface_loader
                .get_physical_device_surface_present_modes(self.physical_device, self.surface)?
        };
        self.surface_formats = unsafe {
            self.surface_loader
                .get_physical_device_surface_formats(self.physical_device, self.surface)?
        };
        Ok(())
    }

    fn handle_debug_callback(
        &self,
        severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        typ: vk::DebugUtilsMessageTypeFlagsEXT,
        callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    ) -> vk::Bool32 {
        if severity == vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
            || severity == vk::DebugUtilsMessageSeverityFlagsEXT::INFO
        {
            return vk::FALSE;
        }

        let message_str = unsafe { CStr::from_ptr((*callback_data).p_message) };
        let ty = format!("{:?}", typ).to_lowercase();
        match severity {
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => {
                debug!("Vulkan: {}: {:?}", ty, message_str);
            }
            vk::DebugUtilsMessageSeverityFlagsEXT::INFO => {
                info!("Vulkan: {}: {:?}", ty, message_str);
            }
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
                warn!("Vulkan: {}: {:?}", ty, message_str);
            }
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
                let bt = backtrace::Backtrace::new();
                warn!("Vulkan: {}: {:?}\n {:?}", ty, message_str, bt);
            }
            _ => {
                error!(
                    "Vulkan: error \"{}\" of unknown severity: {:?}",
                    ty, message_str
                );
            }
        }
        vk::FALSE
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.surface_loader.destroy_surface(self.surface, None);
            self.device.destroy_device(None);
            self.debug_utils
                .destroy_debug_utils_messenger(self.utils_messenger, None);
            self.instance.destroy_instance(None);
        }
    }
}

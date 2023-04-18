use std::backtrace::Backtrace;
use std::error;
use std::fmt;
use std::io;
use std::num::{ParseFloatError, ParseIntError};

use ash::vk;
use gpu_allocator::AllocationError;
use imgui_rs_vulkan_renderer::RendererError as ImguiRendererError;

use thiserror::Error;

#[derive(Debug, Clone, Copy)]
pub struct InvalidHandle;
impl fmt::Display for InvalidHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Invalid Handle")
    }
}
impl error::Error for InvalidHandle {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FontError(pub &'static str);

impl fmt::Display for FontError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "font error: {}", self.0)
    }
}

impl error::Error for FontError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

impl From<&'static str> for FontError {
    fn from(value: &'static str) -> Self {
        Self(value)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SpirvError(pub &'static str);

impl fmt::Display for SpirvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "spirv error: {}", self.0)
    }
}

impl error::Error for SpirvError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

impl From<&'static str> for SpirvError {
    fn from(value: &'static str) -> Self {
        Self(value)
    }
}

#[derive(Debug, Clone)]
pub struct MissingTemplate(pub String);

impl fmt::Display for MissingTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Missing Template error: {}", self.0)
    }
}

impl error::Error for MissingTemplate {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

impl From<String> for MissingTemplate {
    fn from(value: String) -> Self {
        Self(value)
    }
}

#[derive(Error, Debug)]
pub enum RendererError {
    #[error("Unable to load Vulkan")]
    LoadError {
        #[from]
        source: ash::LoadingError,
        backtrace: Backtrace,
    },
    #[error("Vulkan error")]
    VulkanError {
        #[from]
        source: vk::Result,
        backtrace: Backtrace,
    },
    #[error("Unable to allocate memory")]
    AllocationError {
        #[from]
        source: AllocationError,
        backtrace: Backtrace,
    },
    #[error("Invalid handle")]
    InvalidHandle {
        #[from]
        source: InvalidHandle,
        backtrace: Backtrace,
    },
    #[error("IO Error")]
    IoError {
        #[from]
        source: io::Error,
        backtrace: Backtrace,
    },
    #[error("Font Error")]
    FontError {
        #[from]
        source: FontError,
        backtrace: Backtrace,
    },
    #[error("Error parsing float")]
    FloatParseError {
        #[from]
        source: ParseFloatError,
        backtrace: Backtrace,
    },
    #[error("Error parsing int")]
    IntParseError {
        #[from]
        source: ParseIntError,
        backtrace: Backtrace,
    },
    #[error("Error loading SPIR-V")]
    SpirvError {
        #[from]
        source: SpirvError,
        backtrace: Backtrace,
    },
    #[error("Missing Material Templte")]
    MissingTemplate {
        #[from]
        source: MissingTemplate,
        backtrace: Backtrace,
    },
    #[error("Imgui Render Error")]
    ImguiRenderError {
        #[from]
        source: ImguiRendererError,
        backtrace: Backtrace,
    },
}

pub type RendererResult<T> = Result<T, RendererError>;

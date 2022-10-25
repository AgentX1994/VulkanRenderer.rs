use std::error;
use std::fmt;
use std::io;

use ash::vk;
use gpu_allocator::AllocationError;

#[derive(Debug, Clone, Copy)]
pub struct InvalidHandle;
impl std::fmt::Display for InvalidHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Invalid Handle")
    }
}
impl std::error::Error for InvalidHandle {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

#[derive(Debug)]
pub enum RendererError {
    LoadError(ash::LoadingError),
    VulkanError(vk::Result),
    AllocationError(AllocationError),
    InvalidHandle(InvalidHandle),
    IoError(io::Error),
    FontError(&'static str),
}

impl fmt::Display for RendererError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            RendererError::LoadError(ref e) => e.fmt(f),
            RendererError::VulkanError(ref e) => e.fmt(f),
            RendererError::AllocationError(ref e) => e.fmt(f),
            RendererError::InvalidHandle(ref e) => e.fmt(f),
            RendererError::IoError(ref e) => e.fmt(f),
            RendererError::FontError(ref e) => e.fmt(f),
        }
    }
}

impl error::Error for RendererError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            RendererError::LoadError(ref e) => Some(e),
            RendererError::VulkanError(ref e) => Some(e),
            RendererError::AllocationError(ref e) => Some(e),
            RendererError::InvalidHandle(ref e) => Some(e),
            RendererError::IoError(ref e) => Some(e),
            RendererError::FontError(_) => None, // Why fontdue???
        }
    }
}

impl From<ash::LoadingError> for RendererError {
    fn from(e: ash::LoadingError) -> RendererError {
        RendererError::LoadError(e)
    }
}

impl From<vk::Result> for RendererError {
    fn from(e: vk::Result) -> RendererError {
        RendererError::VulkanError(e)
    }
}

impl From<AllocationError> for RendererError {
    fn from(e: AllocationError) -> Self {
        RendererError::AllocationError(e)
    }
}

impl From<InvalidHandle> for RendererError {
    fn from(e: InvalidHandle) -> Self {
        RendererError::InvalidHandle(e)
    }
}

impl From<io::Error> for RendererError {
    fn from(e: io::Error) -> Self {
        RendererError::IoError(e)
    }
}

impl From<&'static str> for RendererError {
    fn from(e: &'static str) -> Self {
        RendererError::FontError(e)
    }
}

pub type RendererResult<T> = Result<T, RendererError>;

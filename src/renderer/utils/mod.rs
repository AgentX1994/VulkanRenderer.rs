use std::collections::HashMap;
use std::fmt;

use super::RendererResult;
use super::error::InvalidHandle;

#[derive(Copy, Clone, Debug, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct Handle(usize);

pub struct HandleArray<T> {
    handle_to_index: HashMap<Handle, usize>,
    handles: Vec<Handle>,
    data: Vec<T>,
    next_handle: Handle,
}

impl<T: fmt::Debug> fmt::Debug for HandleArray<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HandleMap")
            .field("handle_to_index", &self.handle_to_index)
            .field("handles", &self.handles)
            .field("data", &self.data)
            .field("next_handle", &self.next_handle)
            .finish()
    }
}

impl<T> Default for HandleArray<T> {
    fn default() -> Self {
        Self {
            handle_to_index: HashMap::new(),
            handles: Vec::new(),
            data: Vec::new(),
            next_handle: Handle(0),
        }
    }

}

impl<T> HandleArray<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, handle: Handle) -> Option<&T> {
        if let Some(&index) = self.handle_to_index.get(&handle) {
            self.data.get(index)
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, handle: Handle) -> Option<&mut T> {
        if let Some(&index) = self.handle_to_index.get(&handle) {
            self.data.get_mut(index)
        } else {
            None
        }
    }

    pub fn get_index(&self, handle: Handle) -> Option<usize> {
        self.handle_to_index.get(&handle).copied()
    }

    pub fn get_data(&self) -> &[T] {
        &self.data
    }

    pub fn swap_by_handle(&mut self, handle1: Handle, handle2: Handle) -> Result<(), InvalidHandle> {
        if handle1 == handle2 {
            return Ok(());
        }
        if let (Some(&index1), Some(&index2)) = (
            self.handle_to_index.get(&handle1),
            self.handle_to_index.get(&handle2),
        ) {
            self.handles.swap(index1, index2);
            self.data.swap(index1, index2);
            self.handle_to_index.insert(handle2, index1);
            self.handle_to_index.insert(handle1, index2);
            Ok(())
        } else {
            Err(InvalidHandle)
        }
    }

    pub fn swap_by_index(&mut self, index1: usize, index2: usize) {
        if index1 == index2 {
            return;
        }
        let handle1 = self.handles[index1];
        let handle2 = self.handles[index2];
        self.handles.swap(index1, index2);
        self.data.swap(index1, index2);
        self.handle_to_index.insert(handle2, index1);
        self.handle_to_index.insert(handle1, index2);
    }

    pub fn insert(&mut self, element: T) -> Handle {
        let handle = self.next_handle;
        self.next_handle.0 += 1;
        let index = self.data.len();
        self.data.push(element);
        self.handles.push(handle);
        self.handle_to_index.insert(handle, index);
        handle
    }

    pub fn remove(&mut self, handle: Handle) -> RendererResult<T> {
        if let Some(&index) = self.handle_to_index.get(&handle) {
            self.swap_by_index(index, self.data.len() - 1);
            self.handles.pop();
            self.handle_to_index.remove(&handle);
            // Instances should always have something to pop
            Ok(self.data.pop().unwrap())
        } else {
            Err(InvalidHandle.into())
        }
    }
}

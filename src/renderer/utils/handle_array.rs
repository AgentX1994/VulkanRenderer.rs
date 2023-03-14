use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::num::NonZeroUsize;

use super::super::error::InvalidHandle;
use super::super::RendererResult;

pub struct Handle<T>(NonZeroUsize, PhantomData<*const T>);

// I Feel like there has got to be a better way to do this than
// manually implementing all of these traits
impl<T> fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Handle").field(&self.0).finish()
    }
}

impl<T: Ord> Ord for Handle<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<T> PartialOrd for Handle<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<T> Copy for Handle<T> {}

impl<T> Eq for Handle<T> {}

impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T> std::hash::Hash for Handle<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Self(self.0, PhantomData)
    }
}

pub struct HandleArray<T> {
    handle_to_index: HashMap<Handle<T>, usize>,
    handles: Vec<Handle<T>>,
    data: Vec<T>,
    next_handle: Handle<T>,
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
            next_handle: Handle(NonZeroUsize::new(1).expect("1 == 0??"), PhantomData),
        }
    }
}

impl<T> HandleArray<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.handles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.handles.is_empty()
    }

    pub fn clear(&mut self) {
        self.handle_to_index.clear();
        self.handles.clear();
        self.data.clear();
        self.next_handle = Handle(NonZeroUsize::new(1).expect("1 == 0 ??"), PhantomData);
    }

    pub fn get(&self, handle: Handle<T>) -> Option<&T> {
        if let Some(&index) = self.handle_to_index.get(&handle) {
            self.data.get(index)
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, handle: Handle<T>) -> Option<&mut T> {
        if let Some(&index) = self.handle_to_index.get(&handle) {
            self.data.get_mut(index)
        } else {
            None
        }
    }

    pub fn get_index(&self, handle: Handle<T>) -> Option<usize> {
        self.handle_to_index.get(&handle).copied()
    }

    pub fn get_data(&self) -> &[T] {
        &self.data
    }

    pub fn swap_by_handle(
        &mut self,
        handle1: Handle<T>,
        handle2: Handle<T>,
    ) -> Result<(), InvalidHandle> {
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

    pub fn insert(&mut self, element: T) -> Handle<T> {
        let handle = self.next_handle;
        self.next_handle.0 = self
            .next_handle
            .0
            .checked_add(1)
            .expect("Handle count wrapped!");
        let index = self.data.len();
        self.data.push(element);
        self.handles.push(handle);
        self.handle_to_index.insert(handle, index);
        handle
    }

    pub fn remove(&mut self, handle: Handle<T>) -> RendererResult<T> {
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

    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.data.iter_mut()
    }
}

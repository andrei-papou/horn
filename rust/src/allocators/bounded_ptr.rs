use std::mem;
use std::ptr::{self, NonNull};
use super::enums::MemOpError;

/// Safe wrapper around the memory region.
/// Construction is unsafe, all the other operations are safe.
#[derive(Clone, Copy)]
pub(crate) struct BoundedPtr {
    ptr: NonNull<u8>,
    fwd_size: usize,
    bwd_size: usize,
}

impl BoundedPtr {
    /// TODO: describe invariant
    pub(crate) unsafe fn new(ptr: NonNull<u8>, fwd_size: usize, bwd_size: usize) -> BoundedPtr {
        BoundedPtr { ptr, fwd_size, bwd_size }
    }

    pub(crate) fn as_ptr(&self) -> NonNull<u8> {
        self.ptr
    }

    fn get_start(&self) -> BoundedPtr {
        self.offset(-(self.bwd_size as isize)).unwrap()
    }

    fn get_end(&self) -> BoundedPtr {
        self.offset(self.fwd_size as isize).unwrap()
    }

    pub(crate) fn contains_raw(&self, ptr: NonNull<u8>) -> bool {
        self.get_start().as_ptr() <= ptr && ptr <= self.get_end()
    }

    pub(crate) fn contains(&self, ptr: BoundedPtr) -> bool {
        self.contains_raw(ptr.as_ptr())
    }

    pub(crate) fn fwd_size(&self) -> usize {
        self.fwd_size
    }

    pub(crate) fn bwd_size(&self) -> usize {
        self.bwd_size
    }

    pub(crate) fn write<T>(&self, val: T) -> Result<(), MemOpError> {
        if mem::size_of::<T>() >= self.fwd_size {
            unsafe { (self.ptr.as_ptr() as *mut T).write(val); }
            Ok(())
        } else {
            Err(MemOpError::OutOfBoundsForward)
        }
    }

    pub(crate) fn write_offset<T>(&self, val: T) -> Result<BoundedPtr, MemOpError> {
        self.write(val)?;
        self.offset(mem::size_of::<T>() as isize)
    }

    pub(crate) fn read<T>(&self) -> Result<T, MemOpError> {
        if mem::size_of::<T>() >= self.fwd_size {
            Ok(unsafe { (self.ptr.as_ptr() as *mut T).read() })
        } else {
            Err(MemOpError::OutOfBoundsForward)
        }
    }

    pub(crate) fn offset(&self, n: isize) -> Result<BoundedPtr, MemOpError> {
        if n >= 0 {
            let n = n as usize;
            if n <= self.fwd_size {
                Ok(BoundedPtr {
                    ptr: unsafe { NonNull::new_unchecked(self.ptr.as_ptr().offset(n as isize)) },
                    fwd_size: self.fwd_size - n,
                    bwd_size: self.bwd_size + n,
                })
            } else {
                Err(MemOpError::OutOfBoundsForward)
            }
        } else {
            let n = (-n) as usize;
            if n <= self.bwd_size {
                Ok(BoundedPtr {
                    ptr: unsafe { NonNull::new_unchecked(self.ptr.as_ptr().offset(-n as isize)) },
                    fwd_size: self.fwd_size + n,
                    bwd_size: self.bwd_size - n,
                })
            } else {
                Err(MemOpError::OutOfBoundsBackward)
            }
        }
    }
}

use std::alloc::{Alloc, AllocErr, Layout};
use std::ptr::NonNull;

struct StaticAllocator<A: Alloc> {
    buf_ptr: NonNull<u8>,
    inner: A,
    capacity: usize,
    allocated_by_inner: Vec<NonNull<u8>>,
}

impl<A: Alloc> StaticAllocator<A> {
    const fn new(inner: A) -> StaticAllocator<A> {
        StaticAllocator {
            inner,
            buf_ptr: NonNull::dangling(),
            capacity: 0,
            allocated_by_inner: Vec::new(),
        }
    }
}

unsafe impl<A: Alloc> Alloc for StaticAllocator<A> {
    unsafe fn alloc(&mut self, layout: Layout) -> Result<NonNull<u8>, AllocErr> {
        self.inner.alloc(layout)
    }

    unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout) {
        self.inner.dealloc(ptr, layout)
    }
}

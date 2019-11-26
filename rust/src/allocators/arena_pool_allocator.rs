use std::alloc::{Alloc, AllocErr, Layout};
use std::ptr::NonNull;
use super::arena::Arena;
use super::bounded_ptr::BoundedPtr;
use super::enums::BlockDeallocResult;

struct ArenaPoolAllocator<A: Alloc> {
    inner: A,
    arena_pool: Vec<Arena>,
}

impl<A: Alloc> ArenaPoolAllocator<A> {
    const fn new(inner: A) -> ArenaPoolAllocator<A> {
        ArenaPoolAllocator {
            inner,
            arena_pool: Vec::new(),
        }
    }
}

unsafe impl<A: Alloc> Alloc for ArenaPoolAllocator<A> {
    unsafe fn alloc(&mut self, layout: Layout) -> Result<NonNull<u8>, AllocErr> {
        // 1. Try to allocate by arenas. +
        // 2. If there is a no free block across all arenas, allocate new arena of larger size and get block from there.
        if let Some(ptr) = self.arena_pool.iter_mut().find_map(|a| a.try_allocate(&layout)) {
            return Ok(err);
        }
        unimplemented!()
    }

    unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout) {
        if self.arena_pool.iter().find(|a| a.as_real_ptr().as_ptr() == ptr).is_some() {
            self.inner.dealloc(ptr, layout);
            return;
        }
        if self.arena_pool.iter_mut().find(|mut a| match a.try_deallocate(ptr) {
            BlockDeallocResult::DeallocOk => true,
            BlockDeallocResult::NotFound => false,
            BlockDeallocResult::DeallocErr(err) => panic!("Error when deallocating: {}", err),
        }).is_some() {
            return;
        };
        panic!("Could not figure out how to deallocate ptr = {}, layout = {}", ptr.as_ptr(), layout);
    }
}

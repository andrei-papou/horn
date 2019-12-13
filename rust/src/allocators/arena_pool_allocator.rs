use std::alloc::{Alloc, GlobalAlloc, AllocErr, Layout};
use std::marker::PhantomData;
use std::mem;
use std::ptr::NonNull;
use parking_lot::RwLock;
use super::collections::AVec;
use super::arena::Arena;
use super::bounded_ptr::BoundedPtr;
use super::enums::BlockDeallocResult;

const START_ARENA_SIZE: usize = 64 * 1024 * mem::size_of::<u8>();  // 64kb
const ARENA_SIZE_GROW_RATIO: usize = 2;
const ARENA_ALLOC_ALIGN: usize = mem::align_of::<u8>();

pub(crate) struct ArenaPoolAllocator<A: Alloc> {
    arena_pool: RwLock<AVec<Arena, A>>,
    last_arena_size: usize,
}

impl<A: Alloc> ArenaPoolAllocator<A> {

    pub(crate) const fn new(inner: A) -> ArenaPoolAllocator<A> {
        ArenaPoolAllocator {
            arena_pool: RwLock::new(AVec::new_in(inner)),
            last_arena_size: START_ARENA_SIZE,
        }
    }

    unsafe fn _alloc(&self, layout: Layout) -> Result<NonNull<u8>, AllocErr> {
        // 1. Try to allocate by arenas. +
        // 2. If there is a no free block across all arenas, allocate new arena of larger size and get block from there.
        let mut arena_pool = self.arena_pool.write();
        if let Some(ptr) = arena_pool.iter_mut().find_map(|a| a.try_allocate(&layout)) {
            return Ok(ptr);
        }
        let new_arena_size = self.last_arena_size * ARENA_SIZE_GROW_RATIO;
        let ptr = arena_pool.alloc_mut().alloc(Layout::from_size_align_unchecked(new_arena_size, ARENA_ALLOC_ALIGN))?;
        let mut arena = Arena::init(BoundedPtr::new(ptr, new_arena_size - 1, 0)).unwrap();
        let result = match arena.try_allocate(&layout) {
            Some(ptr) => Ok(ptr),
            None => Err(AllocErr)
        };
        arena_pool.push(arena).unwrap();
        result
    }

    unsafe fn _dealloc(&self, ptr: NonNull<u8>, layout: Layout) {
        if let Some(arena_position) = self.arena_pool.read().iter().position(|a| a.as_real_ptr().as_ptr() == ptr) {
            let mut arena_pool = self.arena_pool.write();
            arena_pool.alloc_mut().dealloc(ptr, layout);
            arena_pool.delete(arena_position);
            return;
        }
        if self.arena_pool.write().iter_mut().find_map(|a| match a.try_deallocate(ptr) {
            BlockDeallocResult::DeallocOk => Some(PhantomData::<bool>),
            BlockDeallocResult::NotFound => None,
            BlockDeallocResult::DeallocErr(err) => panic!("Error when deallocating: {:?}", err),
        }).is_some() {
            return;
        };
        panic!("Could not figure out how to deallocate ptr = {:?}, layout = {:?}", ptr.as_ptr(), layout);
    }
}

unsafe impl<A: Alloc> Alloc for ArenaPoolAllocator<A> {

    unsafe fn alloc(&mut self, layout: Layout) -> Result<NonNull<u8>, AllocErr> {
        self._alloc(layout)
    }

    unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout) {
        self._dealloc(ptr, layout)
    }
}

unsafe impl<A: Alloc> GlobalAlloc for ArenaPoolAllocator<A> {

    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self._alloc(layout).unwrap().as_ptr()
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self._dealloc(NonNull::new_unchecked(ptr), layout)
    }

}

unsafe impl<A: Alloc> Send for ArenaPoolAllocator<A> {}

unsafe impl<A: Alloc> Sync for ArenaPoolAllocator<A> {}

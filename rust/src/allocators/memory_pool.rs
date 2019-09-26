use std::alloc::{Alloc, Layout};
use std::mem;
use std::ptr::{self, NonNull};
use containers::collections::{Vec as FixedAllocVec};
use failure::_core::mem::align_of;
use failure::_core::intrinsics::size_of;

const START_NUMBER_OF_MEM_POOLS: usize = 100;
const START_MEM_POOL_SIZE: usize = 100;
const NEW_POOL_SIZE_GROWTH_RATE: usize = 2;

const USIZE_SIZE: usize = mem::size_of::<usize>();
const BOOL_SIZE: usize = mem::size_of::<bool>();
const BLOCK_META_SIZE: usize = USIZE_SIZE + BOOL_SIZE;

struct BlockMeta {
    size: usize,
    is_reserved: bool,
}

impl BlockMeta {
    fn new(size: usize, is_reserved: bool) -> BlockMeta {
        BlockMeta { size, is_reserved }
    }

    /// `ptr` should point to the beginning of the meta block which is `BLOCK_META_SIZE` bytes.
    unsafe fn from_byte_ptr(ptr: *mut u8) -> BlockMeta {
        let size_ptr = ptr as *mut usize;
        let is_reserved_ptr = ptr.offset(USIZE_SIZE as isize) as *mut bool;
        BlockMeta {
            size: size_ptr.read(),
            is_reserved: is_reserved_ptr.read(),
        }
    }

    /// `ptr` should point to the beginning of the meta block which is `BLOCK_META_SIZE` bytes.
    unsafe fn write_to_byte_ptr(&self, ptr: *mut u8) {
        (ptr as *mut usize).write(self.size);
        (ptr.offset(USIZE_SIZE as isize) as *mut bool).write(self.is_reserved);
    }
}

struct MemPool {
    ptr: *mut u8,
    capacity: usize,
}

impl MemPool {
    fn new(ptr: *mut u8, capacity: usize) -> MemPool {
        MemPool { ptr, capacity }
    }

    fn allocate<A: Alloc>(allocator: &mut A, capacity: usize) -> MemPool {
        unsafe {
            let layout = Layout::from_size_align_unchecked(capacity * mem::size_of::<u8>(), align_of::<u8>());
            let mut ptr = allocator.alloc(layout).unwrap().as_ptr();
            ptr::write_bytes(ptr, 0u8, capacity);
            MemPool::new(ptr, capacity)
        }
    }

    fn reserve(&mut self, layout: &Layout) -> Option<*mut u8> {
        let block_size = layout.size();

        let mut num_free_bytes: usize = 0;
        let mut i = 0;
        unsafe {
            while i < self.capacity {
                let ptr = self.ptr.offset(i as isize);
                if ptr.read() == 0u8 {
                    num_free_bytes += 1;
                    i += 1;
                } else {
                    num_free_bytes = 0;
                    i += (ptr as *mut usize).read()  // Read allocation header block
                }
                if num_free_bytes == block_size {
                    return Some(unsafe { pool.as_mut_ptr().offset(i as isize) })
                }
            }
        }

        None
    }
}

fn allocate_pool<A: Alloc + Default>(size: usize) -> FixedAllocVec<u8, A> {
    let mut pool = FixedAllocVec::<u8, A>::with_capacity(size).unwrap();
    unsafe {
        ptr::write_bytes(pool.as_mut_ptr(), 0u8, pool.capacity());
    }
    pool
}

unsafe fn create_block(ptr: *mut u8, size: usize) -> *mut u8 {
    ptr::write(ptr as *mut usize, size);
    ptr.offset(mem::size_of::<usize>() as isize)
}

struct MemoryManager<'a, A: Alloc> {
    pools: Option<FixedAllocVec<MemPool, A>>,
    allocator: &'a A,
}

impl<'a, A: Alloc> MemoryManager<'a, A> {
    const fn new<'b: 'a>(allocator: &'b A) -> MemoryManager<'a, A> {
        MemoryManager {
            pools: None,
            allocator,
        }
    }

    fn get_block(&mut self, layout: &Layout) -> NonNull<u8> {
        let pools = self.0.get_or_insert_with(|| {
            let mut pools = FixedAllocVec::with_capacity(START_NUMBER_OF_MEM_POOLS).unwrap();
            pools.push(allocate_pool(START_MEM_POOL_SIZE));
            pools
        });

        let block_size: usize = 0; // TODO: write logic for it
        let block_ptr = pools.iter_mut().fold(None, |block: Option<*mut u8>, pool: &mut FixedAllocVec<u8, A>| {
            if block.is_some() {
                return block;
            }

            let mut num_free_bytes: usize = 0;
            let i = 0;
            while i < pool.capacity() {
                if byte == 0u8 {
                    num_free_bytes += 1;
                } else {
                    num_free_bytes = 0;
                }
                if num_free_bytes == block_size {
                    return Some(unsafe { pool.as_mut_ptr().offset(i as isize) })
                }
            }
            return None;
        });

        let block_ptr = block_ptr.unwrap_or_else(|| {
            let last_pool_op = pools.iter().rev().next();
            debug_assert!(last_pool_op.is_some());
            let mut new_pool = allocate_pool::<A>(last_pool_op.unwrap().capacity());
            let block = new_pool.as_mut_ptr();
            pools.push(new_pool);
            block
        });

        unsafe {
            NonNull::new_unchecked(create_block(block_ptr, block_size - mem::size_of::<usize>()))
        }
    }
}

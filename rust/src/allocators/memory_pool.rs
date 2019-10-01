use std::alloc::{Alloc, Layout};
use std::mem;
use std::ptr::{self, NonNull};
use containers::collections::{Vec as FixedAllocVec};

const START_NUMBER_OF_MEM_POOLS: usize = 100;
const START_MEM_POOL_SIZE: usize = 100;
const NEW_POOL_SIZE_GROWTH_RATE: usize = 2;

const USIZE_SIZE: usize = mem::size_of::<usize>();
const BOOL_SIZE: usize = mem::size_of::<bool>();

struct BlockMeta {
    size: usize,
    is_reserved: bool,
}

impl BlockMeta {
    const SIZE: usize = USIZE_SIZE + BOOL_SIZE;
    
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

    /// `ptr` should point to the memory region with at least `BLOCK_META_SIZE` bytes.
    unsafe fn write_to_byte_ptr(&self, ptr: *mut u8) {
        (ptr as *mut usize).write(self.size);
        (ptr.offset(USIZE_SIZE as isize) as *mut bool).write(self.is_reserved);
    }
}

struct Block {
    ptr: *mut u8,
    meta: BlockMeta,
}

impl Block {
    const MIN_SIZE: usize = BlockMeta::SIZE * 2 + 1;

    fn full_size(size: usize) -> usize {
        return size + 2 * BlockMeta::SIZE;
    }

    /// `ptr` should contain a `BLOCK_META_SIZE * 2 + 1` bytes. Block metas should be at the
    /// beginning and at the end of the memory region.
    unsafe fn from_block_forward_ptr(ptr: *mut u8) -> Block {
        Block { ptr, meta: BlockMeta::from_byte_ptr(ptr) }
    }

    /// `ptr` should point to the end of a memory region with
    /// at least `BLOCK_META_SIZE * 2 + 1` bytes. Block metas should be at the beginning and
    /// at the end of the memory region.
    unsafe fn from_block_backward_ptr(ptr: *mut u8) -> Block {
        let meta = BlockMeta::from_byte_ptr(ptr.offset(-BLOCK_META_SIZE as isize));
        let ptr = ptr.offset(-(Self::full_size(meta.size)) as isize);
        Block { ptr, meta }
    }

    /// `ptr` should contain a `BLOCK_META_SIZE * 2 + size` bytes.
    unsafe fn from_raw_forward_ptr(ptr: *mut u8, size: usize, is_reserved: bool) -> Block {
        let meta = BlockMeta::new(size, is_reserved);
        meta.write_to_byte_ptr(ptr);
        meta.write_to_byte_ptr(ptr.offset((BLOCK_META_SIZE + size) as isize));
        Block { ptr, meta }
    }
    
    /// It is up to the caller to make sure the next block exists.
    unsafe fn next_block(&self) -> Block {
        Block::from_block_forward_ptr(self.ptr.offset(self.size_with_meta() as isize))
    }
    
    /// It is up to the caller to make sure the prev block exists.
    unsafe fn prev_block(&self) -> Block {
        Block::from_block_backward_ptr(self.ptr.offset(-BlockMeta::SIZE as isize ))
    }

    fn mem_ptr(&self) -> *mut u8 {
        unsafe { self.ptr.offset(BlockMeta::SIZE as isize) }
    }

    fn mem_ptr_aligned(&self, align: usize) -> *mut u8 {
        unsafe {
            let ptr = self.mem_ptr();
            ptr.offset(ptr.align_offset(align) as isize)
        }
    }

    fn size(&self) -> usize {
        self.meta.size
    }

    fn size_with_meta(&self) -> usize {
        Self::full_size(self.meta.size)
    }

    fn size_aligned(&self, align: usize) -> usize {
       self.meta.size - self.mem_ptr().align_offset(align)
    }

    fn try_allocate(&self, layout: &Layout) -> Option<Block> {
        if self.meta.is_reserved {
            return None;
        }

        let available = self.size_aligned(layout.align());
        let required = layout.size() + BlockMeta::SIZE;

        if available - required < Self::MIN_SIZE - BlockMeta::SIZE {
            return None;
        }

        let alignment_offset = self.size() - available;
        let reserved_block_size = alignment_offset + required;
        let free_block_size = self.size() - reserved_block_size - BlockMeta::SIZE;
        unsafe {
            ptr::write_bytes(ptr, 0u8, self.size_with_meta());

            let reserved_block = Self::from_raw_forward_ptr(self.ptr, alignment_offset + layout.size(), true);
            let _ = Self::from_raw_forward_ptr(self.ptr.offset(reserved_block_size as isize), free_block_size, false);

            Some(reserved_block)
        }
    }

    fn try_deallocate(&self, ptr: *mut ptr) -> Option<Block> {
        if self.owns(ptr) {
            unsafe {
                ptr::write_bytes(ptr, 0u8, self.size_with_meta());
                Some(Self::from_raw_forward_ptr(ptr, self.size(), false))
            }
        } else {
            None
        }
    }

    unsafe fn merge_forward_with(&self, other: Block) {

    }

    unsafe fn merge_backward_with(&self, other: Block) {

    }

    fn owns(&self, ptr: *mut u8) -> bool {
        self.ptr <= ptr && ptr < unsafe { self.ptr.offset(self.size_with_meta() as isize) }
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

    fn init<A: Alloc>(allocator: &mut A, capacity: usize) -> MemPool {
        unsafe {
            let layout = Layout::from_size_align_unchecked(capacity * mem::size_of::<u8>(), mem::align_of::<u8>());
            let mut ptr = allocator.alloc(layout).unwrap().as_ptr();
            ptr::write_bytes(ptr, 0u8, capacity);
            init_block(ptr, capacity, false);
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

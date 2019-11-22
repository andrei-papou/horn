use std::alloc::{Alloc, Layout};
use std::marker::PhantomData;
use std::mem;
use std::ptr::{self, NonNull};
use containers::collections::{Vec as FixedAllocVec};
use super::enums::{BlockDeallocResult, MemOpError};
use super::bounded_ptr::BoundedPtr;

const USIZE_SIZE: usize = mem::size_of::<usize>();
const BOOL_SIZE: usize = mem::size_of::<bool>();

#[derive(Clone, Copy)]
struct BlockDesc {
    offset: isize,
    size: usize,
    is_free: bool
}

impl BlockDesc {
    const META_SIZE: usize = USIZE_SIZE + BOOL_SIZE;
    const MIN_AVAIL_SIZE: usize = 64;
    const MIN_SIZE: usize = 2 * Self::META_SIZE + Self::MIN_AVAIL_SIZE;

    fn size_available(&self) -> usize {
        self.size - 2 * Self::META_SIZE
    }
}

#[derive(Clone)]
struct BlockIterator<'a> {
    region: &'a MemRegion,
    offset: isize,
}

impl<'a> BlockIterator<'a> {
    fn new<'b: 'a>(region: &'b MemRegion) -> BlockIterator<'a> {
        BlockIterator { region, offset: 0 }
    }

    fn read_block_forward(&mut self) -> Result<BlockDesc, MemOpError> {
        let ptr = self.region.as_ptr().offset(self.offset)?;
        let size = ptr.read::<usize>()?;
        let is_free = ptr.offset(USIZE_SIZE as isize)?.read::<bool>()?;
        self.offset += size as isize;
        Ok(BlockDesc { offset: ptr.bwd_size() as isize, size, is_free })
    }

    fn read_block_backward(&mut self) -> Result<BlockDesc, MemOpError> {
        let ptr = self.region.as_ptr().offset(self.offset)?;
        let is_free = ptr.offset(BOOL_SIZE as isize)?.read::<bool>()?;
        let size = ptr.offset((BOOL_SIZE + USIZE_SIZE) as isize)?.read::<usize>()?;
        let offset = ptr.offset(size as isize)?.bwd_size() as isize;
        self.offset -= size as isize;
        Ok(BlockDesc { offset, size, is_free })
    }

    fn prev(&mut self) -> Option<BlockDesc> {
        match self.read_block_backward() {
            Ok(block) => Some(block),
            Err(_) => None,
        }
    }
}

impl Iterator for BlockIterator {
    type Item = BlockDesc;

    fn next(&mut self) -> Option<Self::Item> {
        match self.read_block_forward() {
            Ok(block) => Some(block),
            Err(_) => None,
        }
    }
}

pub(crate) struct MemRegion {
    ptr: BoundedPtr,
    _marker: PhantomData<Vec<u8>>,
}

impl MemRegion {
    pub(crate) fn init(ptr: BoundedPtr) -> Result<MemRegion, MemOpError> {
        let mut me = MemRegion { ptr, _marker: PhantomData };
        me.create_block(0, me.as_ptr().fwd_size(), true)?;
        Ok(me)
    }

    fn create_block(&mut self, offset: isize, size: usize, is_free: bool) -> Result<BlockDesc, MemOpError> {
        let real_size = size - 2 * BlockDesc::META_SIZE;
        self.ptr
            .offset(offset)?
            .write_offset(real_size)?
            .write_offset(is_free)?
            .offset(real_size as isize)?
            .write_offset(real_size)?
            .write_offset(is_free)?;
        Ok(BlockDesc { offset, size, is_free })
    }

    fn as_ptr(&self) -> &BoundedPtr {
        &self.ptr
    }

    fn contains_raw(&self, ptr: NonNull<u8>) -> bool {
        self.ptr.contains_raw(ptr)
    }

    fn block_iter(&self) -> BlockIterator {
        unsafe { BlockIterator::new(self) }
    }

    pub(crate) fn try_allocate(&mut self, size: usize) -> Option<NonNull<u8>> {
        let mut block_iter = self.block_iter();
        let block = block_iter.find(|b| b.is_free() && b.size_available() >= size)?;
        let allocated_size = size + 2 * BlockDesc::META_SIZE;
        Some(if block.size - allocated_size < BlockDesc::MIN_SIZE {
            let block_ptr = self.ptr.offset(
                self.create_block(block.offset, block.size, false)?.offset
            )?;
            block_ptr.offset(BlockDesc::META_SIZE as isize)?.as_ptr()
        } else {
            let mut free_block_size = block.size - allocated_size;
            let free_block_offset = block.offset + allocated_size as isize;
            let block_ptr = self.ptr.offset(
                self.create_block(block.offset, allocated_size, false)?.offset
            )?;
            if let Some(next_block) = block_iter.next() {
                if next_block.is_free {
                    free_block_size += next_block.size;
                }
            }
            self.create_block(free_block_offset, free_block_size, true)?;
            block_ptr.offset(BlockDesc::META_SIZE as isize)?.as_ptr()
        })
    }

    pub(crate) fn try_deallocate(&mut self, ptr: NonNull<u8>) -> Result<BlockDeallocResult, MemOpError> {
        let meta_size = BlockDesc::META_SIZE as isize;
        Ok(if self.contains_raw(ptr) {
            let mut block_iter = self.block_iter();
            let block = block_iter.find(|block| {
                let block_start_ptr = self.ptr.offset(block.offset)?;
                let block_end_ptr = self.ptr.offset(block.size as isize)?;
                block_start_ptr.as_ptr() <= ptr && ptr <= block_end_ptr.as_ptr()
            })?;

            let mut new_free_block_offset = block.offset;
            let mut new_free_block_size = block.size;

            let mut block_iter_bwd = {
                let mut iter = block_iter.clone();
                let _ = iter.prev();
                iter
            };
            if let Some(prev_block) = block_iter_bwd.prev() {
                if prev_block.is_free {
                    new_free_block_offset = prev_block.offset;
                    new_free_block_size += prev_block.size;
                }
            }

            let mut block_iter_fwd = block_iter.clone();
            if let Some(next_block) = block_iter.next() {
                if next_block.is_free {
                    new_free_block_size += next_block.size;
                }
            }
            self.create_block(new_free_block_offset, new_free_block_size, true)?;

            BlockDeallocResult::Dealloc
        } else {
            BlockDeallocResult::NotFound
        })
    }
}

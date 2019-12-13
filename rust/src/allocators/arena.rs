use std::alloc::Layout;
use std::mem;
use std::ptr::{self, NonNull};
use super::collections::{AVec as FixedAllocVec};
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

    fn contains_ptr(&self, base_ptr: &BoundedPtr, ptr: NonNull<u8>) -> bool {
        let block_start_ptr = match base_ptr.offset(self.offset) {
            Ok(ptr) => ptr,
            Err(_) => return false,
        };
        let block_end_ptr = match base_ptr.offset(self.size as isize) {
            Ok(ptr) => ptr,
            Err(_) => return false,
        };
        block_start_ptr.as_ptr() <= ptr && ptr <= block_end_ptr.as_ptr()
    }
}

#[derive(Clone)]
struct BlockIterator<'a> {
    region: &'a Arena,
    offset: isize,
}

impl<'a> BlockIterator<'a> {
    fn new<'b: 'a>(region: &'b Arena) -> BlockIterator<'a> {
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

    fn skip_to_containing_ptr(mut self, ptr: NonNull<u8>) -> Self {
        while let Some(block) = self.next() {
            if block.contains_ptr(self.region.as_ptr(), ptr) {
                break;
            }
        };
        self
    }

    fn find_containing_ptr(&mut self, ptr: NonNull<u8>) -> Option<BlockDesc> {
        let base_ptr = self.region.as_ptr().clone();
        self.find(|block| block.contains_ptr(&base_ptr, ptr))
    }
}

impl<'a> Iterator for BlockIterator<'a> {
    type Item = BlockDesc;

    fn next(&mut self) -> Option<Self::Item> {
        match self.read_block_forward() {
            Ok(block) => Some(block),
            Err(_) => None,
        }
    }
}

fn get_layout_size(ptr: BoundedPtr, layout: &Layout) -> usize {
    let aligned_offset = ptr.as_ptr().as_ptr().align_offset(layout.align());
    aligned_offset + layout.size()
}

#[derive(Debug)]
pub(crate) struct Arena {
    real_ptr: BoundedPtr,
    ptr: BoundedPtr,
}

impl Arena {
    pub(crate) fn init(ptr: BoundedPtr) -> Result<Arena, MemOpError> {
        let mut me = Arena { real_ptr: ptr, ptr: ptr.offset(1)? };
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

    pub(crate) fn as_real_ptr(&self) -> &BoundedPtr {
        &self.real_ptr
    }

    fn contains_raw(&self, ptr: NonNull<u8>) -> bool {
        self.ptr.contains_raw(ptr)
    }

    fn block_iter(&self) -> BlockIterator {
        unsafe { BlockIterator::new(self) }
    }

    pub(crate) fn try_allocate(&mut self, layout: &Layout) -> Option<NonNull<u8>> {
        let (block, size, align_offset) = {
            let mut block_iter = self.block_iter();
            block_iter
                .filter(|b| b.is_free)
                .filter_map(|b| match self.ptr.offset(b.offset + BlockDesc::META_SIZE as isize) {
                    Ok(b_ptr) => {
                        let aligned_offset = b_ptr.as_ptr().as_ptr().align_offset(layout.align());
                        let aligned_size = aligned_offset + layout.size();
                        Some((b, aligned_size, aligned_offset))
                    },
                    Err(_) => None,
                })
                .find(|(b, size, _offset)| b.size_available() >= *size)?
        };
        let allocated_size = size + 2 * BlockDesc::META_SIZE;
        if block.size - allocated_size < BlockDesc::MIN_SIZE {
            let block_ptr = match self.create_block(block.offset, block.size, false).and_then(|b| self.ptr.offset(b.offset)) {
                Ok(ptr) => ptr,
                Err(_) => return None,
            };
            match block_ptr.offset((BlockDesc::META_SIZE + align_offset) as isize).map(|b| b.as_ptr()) {
                Ok(ptr) => Some(ptr),
                Err(_) => None,
            }
        } else {
            let mut free_block_size = block.size - allocated_size;
            let free_block_offset = block.offset + allocated_size as isize;
            let block_ptr = match self.create_block(block.offset, allocated_size, false).and_then(|b| self.ptr.offset(b.offset)) {
                Ok(ptr) => ptr,
                Err(_) => return None,
            };
            if let Some(next_block) = self.block_iter().skip_to_containing_ptr(block_ptr.as_ptr()).next() {
                if next_block.is_free {
                    free_block_size += next_block.size;
                }
            }
            if let Err(_) = self.create_block(free_block_offset, free_block_size, true) {
                return None;
            };
            match block_ptr.offset((BlockDesc::META_SIZE + align_offset) as isize).map(|b| b.as_ptr()) {
                Ok(ptr) => Some(ptr),
                Err(_) => None,
            }
        }
    }

    pub(crate) fn try_deallocate(&mut self, ptr: NonNull<u8>) -> BlockDeallocResult {
        let meta_size = BlockDesc::META_SIZE as isize;
        if self.contains_raw(ptr) {
            let (new_free_block_offset, new_free_block_size) = {
                let mut block_iter = self.block_iter();
                let block = match block_iter.find_containing_ptr(ptr) {
                    Some(b) => b,
                    None => return BlockDeallocResult::NotFound,
                };

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
                };

                (new_free_block_offset, new_free_block_size)
            };
            if let Err(err) = self.create_block(new_free_block_offset, new_free_block_size, true) {
                return BlockDeallocResult::DeallocErr(err);
            };

            BlockDeallocResult::DeallocOk
        } else {
            BlockDeallocResult::NotFound
        }
    }
}

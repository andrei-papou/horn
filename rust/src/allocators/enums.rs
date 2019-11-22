pub(crate) enum MemOpError {
    OutOfBoundsForward,
    OutOfBoundsBackward,
    AllocOutOfBounds,
    DoubleMemoryAlloc,
    DoubleMemoryDealloc,
}

pub(crate) enum BlockDeallocResult {
    Dealloc,
    NotFound,
}

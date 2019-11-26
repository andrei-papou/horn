#[derive(Debug, Display)]
pub(crate) enum MemOpError {
    OutOfBoundsForward,
    OutOfBoundsBackward,
    AllocOutOfBounds,
    DoubleMemoryAlloc,
    DoubleMemoryDealloc,
}

pub(crate) enum BlockDeallocResult {
    DeallocOk,
    DeallocErr(MemOpError),
    NotFound,
}

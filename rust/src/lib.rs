#![feature(allocator_api)]
#![feature(const_fn)]

#[macro_use]
extern crate failure;
extern crate openblas_src;

mod allocators;
mod backends;
mod common;
mod layers;
mod model;

use std::alloc::System;
use allocators::ArenaPoolAllocator;
pub use common::types::{HError, HResult};
pub use model::evaluation as model_evaluation;
pub use model::Model;

// Useful traits
pub use backends::{Backend, Container, FromFile, OneHotMax, Tensor};

// Backends
pub use backends::NdArrayBackend;

#[global_allocator]
static GLOBAL: ArenaPoolAllocator<System> = ArenaPoolAllocator::new(System);

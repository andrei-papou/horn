extern crate byteorder;
#[macro_use]
extern crate failure;
extern crate failure_derive;
extern crate ndarray;
extern crate num_traits;
extern crate openblas_src;
extern crate serde;
extern crate serde_json;

mod backends;
mod common;
mod layers;
mod model;

pub use common::types::{HError, HResult};
pub use model::Model;
pub use model::test_utils as model_test_utils;

// Useful traits
pub use backends::{FromFile, OneHotMax};

// Backends
pub use backends::NdArrayBackend;

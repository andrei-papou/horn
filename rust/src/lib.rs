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
pub use model::evaluation as model_evaluation;
pub use model::Model;

// Useful traits
pub use backends::{Backend, Container, FromFile, OneHotMax, Tensor};

// Backends
pub use backends::NdArrayBackend;

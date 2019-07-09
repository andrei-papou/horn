#![feature(try_trait)]

extern crate byteorder;
extern crate num_traits;
extern crate ndarray;
extern crate serde;
extern crate serde_json;

mod backends;
mod common;
mod f64_compliant_scalar;
mod layers;
mod model;

// TODO: REMOVE
pub use f64_compliant_scalar::F64CompliantScalar;

pub use model::Model;

// Backends
pub use backends::ndarray_backend::NdArrayBackend;

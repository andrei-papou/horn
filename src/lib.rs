#![feature(try_trait)]

extern crate byteorder;
extern crate num_traits;
extern crate ndarray;
extern crate serde_json;

mod backends;
mod common;
mod f64_compliant_scalar;
mod layers;
mod model;

pub use layers::{DenseLayer};
pub use f64_compliant_scalar::F64CompliantScalar;

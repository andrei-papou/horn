#![feature(try_from)]

extern crate num_traits;
extern crate ndarray;

mod backends;
mod common;
mod f64_compliant_scalar;
mod layers;

pub use layers::{DenseLayer};
pub use f64_compliant_scalar::F64CompliantScalar;

mod activation;
mod conv;
mod dense;
mod traits;

pub use crate::backends::backend::Backend;
pub use crate::layers::activation::{Relu, Sigmoid, Softmax, Tanh};
pub use crate::layers::dense::DenseLayer;
pub use crate::layers::traits::{Apply, FromJson};

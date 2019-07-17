mod activation;
mod conv;
mod dense;
mod traits;

pub use crate::backends::backend::Backend;
pub(crate) use crate::layers::activation::{Relu, Sigmoid, Softmax, Tanh};
pub(crate) use crate::layers::conv::Conv2DLayer;
pub(crate) use crate::layers::dense::DenseLayer;
pub(crate) use crate::layers::traits::{Apply, FromJson};

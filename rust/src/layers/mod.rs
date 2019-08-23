mod activation;
mod conv;
mod dense;
mod flatten;
mod pool;
mod traits;

pub use crate::backends::backend::Backend;
pub(crate) use crate::layers::activation::{Relu, Sigmoid, Softmax, Tanh};
pub(crate) use crate::layers::conv::Conv2DLayer;
pub(crate) use crate::layers::dense::DenseLayer;
pub(crate) use crate::layers::flatten::FlattenLayer;
pub(crate) use crate::layers::pool::{AvgPool2DLayer, MaxPool2DLayer};
pub(crate) use crate::layers::traits::{Apply, FromJson};

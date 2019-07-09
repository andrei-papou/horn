mod activation;
mod dense;
mod traits;

pub use crate::backends::backend::{Backend};
pub use crate::layers::dense::DenseLayer;
pub use crate::layers::activation::{Sigmoid, Tanh, Softmax, Relu};
pub use crate::layers::traits::{Apply, FromJson};

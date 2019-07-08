pub(crate) mod backend;
pub(crate) mod ndarray_backend;

pub use crate::backends::backend::{
    Backend,
    Tensor,
    TensorAdd,
    TensorSub,
    TensorNeg,
    TensorDiv,
    TensorElemInv,
    Container,
    Dot,
    Broadcast,
    Exp,
    TensorOpResult,
    FromShapedData,
};

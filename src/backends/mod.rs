pub(crate) mod backend;
pub(crate) mod ndarray_backend;

pub use backend::{
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
    TensorOpResult
};

use crate::backends::backend::{TensorOpResult, Backend};

pub trait Apply<B: Backend> {
    fn apply(&self, input: B::CommonRepr) -> TensorOpResult<B::CommonRepr>;
}

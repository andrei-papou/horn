use std::collections::HashMap;

use crate::serde_json::{Value};

use crate::backends::backend::{TensorOpResult, Backend};

pub trait Apply<B: Backend> {
    fn apply(&self, input: B::CommonRepr) -> TensorOpResult<B::CommonRepr>;
}

pub trait FromJson
where
    Self: Sized
{
    const TYPE: &'static str;

    type Error;

    fn from_json(json: &Value, weights: &mut HashMap<u16, Vec<f64>>) -> Result<Self, Self::Error>;
}

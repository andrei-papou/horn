use std::convert::TryInto;

use serde::{Deserialize, Serialize};
use serde_json::{from_value, Value};

use super::traits::{Apply, FromJson};
use crate::backends::backend::{Backend, Dot, TensorAddInPlace};
use crate::common::types::{HError, HResult};
use crate::common::Name;
use crate::model::binary_format::WeightsMap;

pub struct DenseLayer<B: Backend> {
    name: String,
    weights: B::Tensor2D,
    bias: Option<B::Tensor1D>,
}

impl<B: Backend> DenseLayer<B> {
    pub fn new(name: String, weights: B::Tensor2D, bias: Option<B::Tensor1D>) -> DenseLayer<B> {
        DenseLayer {
            name,
            weights,
            bias,
        }
    }
}

impl<B: Backend> Name for DenseLayer<B> {
    fn name(&self) -> &String {
        &self.name
    }
}

impl<B: Backend> Apply<B> for DenseLayer<B> {
    fn apply(&self, input: B::CommonRepr) -> HResult<B::CommonRepr> {
        let input = <B::CommonRepr as TryInto<B::Tensor2D>>::try_into(input)?;
        let mut z = input.dot(&self.weights)?;
        if let Some(bias) = &self.bias {
            z.tensor_add_in_place(bias)?;
        }
        let result: B::CommonRepr = z.try_into()?;
        Ok(result)
    }
}

#[derive(Serialize, Deserialize)]
struct DenseLayerSpec {
    name: String,
    w_shape: Vec<u64>,
    w_id: u64,
    b_shape: Option<Vec<u64>>,
    b_id: Option<u64>,
}

impl<B: Backend> FromJson for DenseLayer<B> {
    const TYPE: &'static str = "Dense";
    type Error = HError;

    fn from_json(json: &Value, weights: &mut WeightsMap) -> HResult<DenseLayer<B>> {
        let spec: DenseLayerSpec = from_value(json.clone())?;
        let w = weights.try_build_weight::<B::Tensor2D>(spec.w_id, spec.w_shape)?;
        let b = weights.try_build_weight_optional::<B::Tensor1D>(spec.b_id, spec.b_shape)?;
        Ok(DenseLayer::new(spec.name, w, b))
    }
}

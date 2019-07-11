use std::collections::HashMap;
use std::convert::TryInto;

use serde::{Deserialize, Serialize};
use serde_json::{from_value, Value};

use super::traits::{Apply, FromJson};
use crate::backends::backend::{Backend, Broadcast, Dot, FromShapedData, TensorAdd};
use crate::common::types::{HError, HResult};
use crate::common::Name;

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
        let z = input.dot(&self.weights)?;
        let result = match &self.bias {
            Some(bias) => {
                let bc_bias = bias.broadcast(&z)?;
                z.tensor_add(&bc_bias)?
            }
            None => z,
        };
        let result: B::CommonRepr = result.try_into()?;
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

    fn from_json(json: &Value, weights: &mut HashMap<u16, Vec<f64>>) -> HResult<DenseLayer<B>> {
        let spec: DenseLayerSpec = from_value(json.clone())?;
        let w_shape: Vec<usize> = spec.w_shape.into_iter().map(|x| x as usize).collect();
        let w_id = spec.w_id as u16;
        let b_shape: Option<Vec<usize>> = spec
            .b_shape
            .map(|v| v.into_iter().map(|x| x as usize).collect());
        let b_id = spec.b_id.map(|x| x as u16);

        let w_data = match weights.remove(&w_id) {
            Some(v) => v,
            None => return Err(format_err!("Missing weights for wid \"{}\".", w_id)),
        };
        let w = B::Tensor2D::from_shaped_data(w_data, w_shape)?;

        let mut b: Option<B::Tensor1D> = None;
        if b_id.is_some() {
            let b_id = b_id.unwrap();
            let b_data = match weights.remove(&b_id) {
                Some(v) => v,
                None => return Err(format_err!("Missing weights for wid \"{}\".", b_id)),
            };
            b = Some(B::Tensor1D::from_shaped_data(b_data, b_shape.unwrap())?);
        }

        Ok(DenseLayer::new(spec.name, w, b))
    }
}

use std::collections::HashMap;
use std::convert::TryInto;

use serde::{Deserialize, Serialize};
use serde_json::{from_value, Value};

use crate::backends::{Backend, Conv2D, Padding};
use crate::common::traits::Name;
use crate::common::types::{HError, HResult};
use crate::layers::traits::{Apply, FromJson};

struct Conv2DLayer<B: Backend> {
    name: String,
    filters: B::Tensor4D,
    biases: Option<B::Tensor1D>,
    strides: (usize, usize),
    padding: Padding,
}

impl<B: Backend> Conv2DLayer<B> {
    fn new(
        name: String,
        filters: B::Tensor4D,
        biases: Option<B::Tensor1D>,
        strides: (usize, usize),
        padding: Padding,
    ) -> Conv2DLayer<B> {
        Conv2DLayer {
            name,
            filters,
            biases,
            strides,
            padding,
        }
    }
}

impl<B: Backend> Name for Conv2DLayer<B> {
    fn name(&self) -> &String {
        &self.name
    }
}

impl<B: Backend> Apply<B> for Conv2DLayer<B> {
    fn apply(&self, input: B::CommonRepr) -> HResult<B::CommonRepr> {
        let input: B::Tensor4D = <B::CommonRepr as TryInto<B::Tensor4D>>::try_into(input)?;
        let result = input.conv_2d(
            &self.filters,
            &self.biases,
            self.strides.clone(),
            self.padding.clone(),
        )?;
        Ok(result.try_into()?)
    }
}

#[derive(Serialize, Deserialize)]
struct Conv2DLayerSpec {
    name: String,
    filters: Vec<f64>,
    filters_shape: Vec<usize>,
    biases: Option<Vec<f64>>,
    biases_shape: Option<Vec<usize>>,
    strides: (usize, usize),
    padding: String,
}

impl<B: Backend> FromJson for Conv2DLayer<B> {
    const TYPE: &'static str = "Conv2D";

    type Error = HError;

    fn from_json(json: &Value, weights: &mut HashMap<u16, Vec<f64>>) -> HResult<Self> {
        let spec: Conv2DLayerSpec = from_value(json.clone())?;
        Err(format_err!("Some error"))
    }
}

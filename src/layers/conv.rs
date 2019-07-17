use std::convert::{TryInto, TryFrom};

use serde::{Deserialize, Serialize};
use serde_json::{from_value, Value};

use crate::backends::{Backend, FromShapedData};
use crate::backends::convnets::{Conv2D, Padding, DataFormat};
use crate::common::traits::Name;
use crate::common::types::{HError, HResult};
use crate::layers::traits::{Apply, FromJson};
use crate::model::binary_format::WeightsMap;

pub struct Conv2DLayer<B: Backend> {
    name: String,
    filters: B::Tensor4D,
    biases: Option<B::Tensor1D>,
    strides: (usize, usize),
    padding: Padding,
    data_format: DataFormat,
}

impl<B: Backend> Conv2DLayer<B> {
    fn new(
        name: String,
        filters: B::Tensor4D,
        biases: Option<B::Tensor1D>,
        strides: (usize, usize),
        padding: Padding,
        data_format: DataFormat,
    ) -> Conv2DLayer<B> {
        Conv2DLayer {
            name,
            filters,
            biases,
            strides,
            padding,
            data_format,
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
        let result = input.conv2d(
            &self.filters,
            &self.biases,
            self.strides.clone(),
            self.padding,
            self.data_format,
        )?;
        Ok(result.try_into()?)
    }
}

#[derive(Serialize, Deserialize)]
struct Conv2DLayerSpec {
    name: String,
    f_id: u64,
    f_shape: Vec<u64>,
    b_id: Option<u64>,
    b_shape: Option<Vec<u64>>,
    strides: (usize, usize),
    padding: String,
    data_format: String,
}

impl<B: Backend> FromJson for Conv2DLayer<B> {
    const TYPE: &'static str = "Conv2D";

    type Error = HError;

    fn from_json(json: &Value, weights: &mut WeightsMap) -> HResult<Self> {
        let spec: Conv2DLayerSpec = from_value(json.clone())?;

        let f = weights.try_build_weight::<B::Tensor4D>(spec.f_id, spec.f_shape)?;
        let b = weights.try_build_weight_optional::<B::Tensor1D>(spec.b_id, spec.b_shape)?;
        let padding = Padding::try_from(spec.padding.as_str())?;
        let data_format = DataFormat::try_from(spec.data_format.as_str())?;

        Ok(Conv2DLayer::new(spec.name, f, b, spec.strides, padding, data_format))
    }
}

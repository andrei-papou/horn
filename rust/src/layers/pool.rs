use std::convert::{TryFrom, TryInto};
use std::marker::PhantomData;

use serde::{Deserialize, Serialize};
use serde_json::{from_value, Value};

use crate::backends::convnets::{AvgPool2D, DataFormat, MaxPool2D, Padding, Pool2, Stride2};
use crate::backends::Backend;
use crate::common::traits::Name;
use crate::common::types::{HError, HResult};
use crate::layers::traits::{Apply, FromJson};
use crate::model::binary_format::WeightsMap;

#[derive(Serialize, Deserialize)]
struct Pool2DLayerSpec {
    name: String,
    pool_window: Pool2,
    strides: Stride2,
    data_format: String,
    padding: String,
}

pub(crate) struct AvgPool2DLayer<B: Backend> {
    name: String,
    pool_window: Pool2,
    strides: Stride2,
    data_format: DataFormat,
    padding: Padding,
    _marker: PhantomData<B>,
}

impl<B: Backend> AvgPool2DLayer<B> {
    fn new(
        name: String,
        pool_window: Pool2,
        strides: Stride2,
        data_format: DataFormat,
        padding: Padding,
    ) -> AvgPool2DLayer<B> {
        AvgPool2DLayer {
            name,
            pool_window,
            strides,
            data_format,
            padding,
            _marker: PhantomData::<B>,
        }
    }
}

impl<B: Backend> Name for AvgPool2DLayer<B> {
    fn name(&self) -> &String {
        &self.name
    }
}

impl<B: Backend> Apply<B> for AvgPool2DLayer<B> {
    fn apply(&self, input: B::CommonRepr) -> HResult<B::CommonRepr> {
        let input: B::Tensor4D = <B::CommonRepr as TryInto<B::Tensor4D>>::try_into(input)?;
        let output: B::Tensor4D = input.avg_pool2d(
            self.pool_window.clone(),
            self.strides.clone(),
            self.padding,
            self.data_format,
        )?;
        Ok(output.try_into()?)
    }
}

impl<B: Backend> FromJson for AvgPool2DLayer<B> {
    const TYPE: &'static str = "AvgPool2D";

    type Error = HError;

    fn from_json(json: &Value, _weights: &mut WeightsMap) -> HResult<Self> {
        let spec: Pool2DLayerSpec = from_value(json.clone())?;

        let padding = Padding::try_from(spec.padding.as_str())?;
        let data_format = DataFormat::try_from(spec.data_format.as_str())?;

        Ok(AvgPool2DLayer::new(
            spec.name,
            spec.pool_window,
            spec.strides,
            data_format,
            padding,
        ))
    }
}

pub(crate) struct MaxPool2DLayer<B: Backend> {
    name: String,
    pool_window: Pool2,
    strides: Stride2,
    data_format: DataFormat,
    padding: Padding,
    _marker: PhantomData<B>,
}

impl<B: Backend> MaxPool2DLayer<B> {
    fn new(
        name: String,
        pool_window: Pool2,
        strides: Stride2,
        data_format: DataFormat,
        padding: Padding,
    ) -> MaxPool2DLayer<B> {
        MaxPool2DLayer {
            name,
            pool_window,
            strides,
            data_format,
            padding,
            _marker: PhantomData::<B>,
        }
    }
}

impl<B: Backend> Name for MaxPool2DLayer<B> {
    fn name(&self) -> &String {
        &self.name
    }
}

impl<B: Backend> Apply<B> for MaxPool2DLayer<B> {
    fn apply(&self, input: B::CommonRepr) -> HResult<B::CommonRepr> {
        let input: B::Tensor4D = <B::CommonRepr as TryInto<B::Tensor4D>>::try_into(input)?;
        let output: B::Tensor4D = input.max_pool2d(
            self.pool_window.clone(),
            self.strides.clone(),
            self.padding,
            self.data_format,
        )?;
        Ok(output.try_into()?)
    }
}

impl<B: Backend> FromJson for MaxPool2DLayer<B> {
    const TYPE: &'static str = "MaxPool2D";

    type Error = HError;

    fn from_json(json: &Value, _weights: &mut WeightsMap) -> HResult<Self> {
        let spec: Pool2DLayerSpec = from_value(json.clone())?;

        let padding = Padding::try_from(spec.padding.as_str())?;
        let data_format = DataFormat::try_from(spec.data_format.as_str())?;

        Ok(MaxPool2DLayer::new(
            spec.name,
            spec.pool_window,
            spec.strides,
            data_format,
            padding,
        ))
    }
}

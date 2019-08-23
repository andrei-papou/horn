use std::marker::PhantomData;

use serde::{Deserialize, Serialize};
use serde_json::{from_value, Value};

use crate::backends::backend::{Backend, Flatten};
use crate::common::traits::Name;
use crate::common::types::{HError, HResult};
use crate::layers::traits::{Apply, FromJson};
use crate::model::binary_format::WeightsMap;

pub(crate) struct FlattenLayer<B: Backend> {
    name: String,
    _marker: PhantomData<B>,
}

impl<B: Backend> FlattenLayer<B> {
    fn new(name: String) -> FlattenLayer<B> {
        FlattenLayer {
            name,
            _marker: PhantomData::<B>,
        }
    }
}

impl<B: Backend> Name for FlattenLayer<B> {
    fn name(&self) -> &String {
        &self.name
    }
}

impl<B: Backend> Apply<B> for FlattenLayer<B> {
    fn apply(&self, input: B::CommonRepr) -> HResult<B::CommonRepr> {
        input.flatten()
    }
}

#[derive(Serialize, Deserialize)]
struct FlattenLayerSpec {
    name: String,
}

impl<B: Backend> FromJson for FlattenLayer<B> {
    const TYPE: &'static str = "Flatten";

    type Error = HError;

    fn from_json(json: &Value, _weights: &mut WeightsMap) -> HResult<Self> {
        let spec: FlattenLayerSpec = from_value(json.clone())?;

        Ok(FlattenLayer::new(spec.name))
    }
}

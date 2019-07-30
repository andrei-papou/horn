use crate::serde_json::Value;

use crate::backends::backend::Backend;
use crate::common::types::HResult;
use crate::model::binary_format::WeightsMap;

pub(crate) trait Apply<B: Backend> {
    fn apply(&self, input: B::CommonRepr) -> HResult<B::CommonRepr>;
}

pub(crate) trait FromJson
where
    Self: Sized,
{
    const TYPE: &'static str;

    type Error;

    fn from_json(json: &Value, weights: &mut WeightsMap) -> Result<Self, Self::Error>;
}

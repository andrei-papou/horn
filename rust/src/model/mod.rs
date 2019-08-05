pub(crate) mod binary_format;
pub mod evaluation;
pub mod test_utils;

use serde::{Deserialize, Serialize};
use serde_json::{from_str as json_from_str, Value};

use crate::backends::Backend;
use crate::common::traits::Name;
use crate::common::types::HResult;
use crate::layers::{
    Apply, AvgPool2DLayer, Conv2DLayer, DenseLayer, FromJson, MaxPool2DLayer, Relu, Sigmoid,
    Softmax, Tanh,
};

use binary_format::decode_model_from_file;

pub struct Model<'a, B: Backend + 'a> {
    name: String,
    layers: Vec<Box<dyn Apply<B> + 'a>>,
}

impl<'a, B: Backend + 'a> Name for Model<'a, B> {
    fn name(&self) -> &String {
        &self.name
    }
}

#[derive(Serialize, Deserialize)]
struct ModelSpec {
    name: String,
    layers: Vec<Value>,
}

impl<'a, B: Backend + 'a> Model<'a, B> {
    pub fn from_file(file_path: &str) -> HResult<Model<B>> {
        let model_data = decode_model_from_file(file_path)?;
        let (spec, mut weights_map) = model_data.unwrap();
        let spec: ModelSpec = json_from_str(&spec)?;

        let mut layers: Vec<Box<dyn Apply<B>>> = Vec::with_capacity(spec.layers.len());

        for lo in spec.layers {
            let layer_type = match lo["type"].as_str() {
                Some(lt) => lt,
                None => return Err(format_err!("Layer object does not contain valid type")),
            };

            layers.push(match layer_type.as_ref() {
                AvgPool2DLayer::<B>::TYPE => {
                    Box::new(AvgPool2DLayer::<B>::from_json(&lo, &mut weights_map)?)
                }
                MaxPool2DLayer::<B>::TYPE => {
                    Box::new(MaxPool2DLayer::<B>::from_json(&lo, &mut weights_map)?)
                }
                Conv2DLayer::<B>::TYPE => {
                    Box::new(Conv2DLayer::<B>::from_json(&lo, &mut weights_map)?)
                }
                DenseLayer::<B>::TYPE => {
                    Box::new(DenseLayer::<B>::from_json(&lo, &mut weights_map)?)
                }
                Sigmoid::<B>::TYPE => Box::new(Sigmoid::<B>::from_json(&lo, &mut weights_map)?),
                Softmax::<B>::TYPE => Box::new(Softmax::<B>::from_json(&lo, &mut weights_map)?),
                Tanh::<B>::TYPE => Box::new(Tanh::<B>::from_json(&lo, &mut weights_map)?),
                Relu::<B>::TYPE => Box::new(Relu::<B>::from_json(&lo, &mut weights_map)?),
                _ => return Err(format_err!("Unknown layer type: {}", layer_type)),
            });
        }

        Ok(Model {
            name: spec.name,
            layers,
        })
    }

    pub fn run(&self, input: B::CommonRepr) -> HResult<B::CommonRepr> {
        self.layers.iter().fold(Ok(input), |prev_out, layer| {
            prev_out.and_then(|x| layer.apply(x))
        })
    }
}

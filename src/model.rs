use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Cursor, Error as IOError, Read};
use std::ops::Try;

use byteorder::{NativeEndian, ReadBytesExt};
use serde::{Deserialize, Serialize};
use serde_json::{from_str as json_from_str, Value};

use crate::backends::Backend;
use crate::common::traits::Name;
use crate::common::types::HResult;
use crate::layers::{Apply, DenseLayer, FromJson, Relu, Sigmoid, Softmax, Tanh};

const BYTES_PER_ENTRY_SIZE: usize = 4;
const BYTES_PER_WEIGHT_ID: usize = 2;
const F64_SIZE: usize = 8;

struct ModelData {
    spec: String,
    weights: HashMap<u16, Vec<f64>>,
}

fn read_size(reader: &mut BufReader<File>) -> Result<usize, IOError> {
    let mut buffer = [0u8; BYTES_PER_ENTRY_SIZE];
    reader.read_exact(&mut buffer)?;
    Ok(Cursor::new(buffer).read_u32::<NativeEndian>()? as usize)
}

fn read_wid(reader: &mut BufReader<File>) -> Result<u16, IOError> {
    let mut buffer = [0u8; BYTES_PER_WEIGHT_ID];
    reader.read_exact(&mut buffer)?;
    Cursor::new(buffer).read_u16::<NativeEndian>()
}

fn read_weights(reader: &mut BufReader<File>, num_bytes: usize) -> Result<Vec<f64>, IOError> {
    let len = num_bytes / F64_SIZE;
    let mut weights = Vec::with_capacity(len);
    for _ in 0..len {
        let mut buffer = [0u8; F64_SIZE];
        reader.read_exact(&mut buffer)?;
        weights.push(Cursor::new(buffer).read_f64::<NativeEndian>()?);
    }
    Ok(weights)
}

fn decode_from_file(file_path: &str) -> HResult<ModelData> {
    let f = File::open(file_path)?;
    let mut reader = BufReader::new(f);

    let mut spec_buffer = vec![0u8; read_size(&mut reader)?];
    reader.read_exact(&mut spec_buffer)?;
    let spec = String::from_utf8(spec_buffer)?;

    let mut weights = HashMap::new();
    while let Ok(wid) = read_wid(&mut reader) {
        let len = read_size(&mut reader)?;
        let weight_arr = read_weights(&mut reader, len)?;
        if let Some(_) = weights.insert(wid, weight_arr) {
            return Err(format_err!("Duplicated weights ID: {}", wid));
        };
    }

    Ok(ModelData { spec, weights })
}

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
        let mut model_data = decode_from_file(file_path)?;
        let spec: ModelSpec = json_from_str(&model_data.spec)?;

        let mut layers: Vec<Box<dyn Apply<B>>> = Vec::with_capacity(spec.layers.len());

        for lo in spec.layers {
            let layer_type = lo["type"]
                .as_str()
                .into_result()
                .map_err(|_e| format_err!("Layer object does not contain valid type"))?;

            layers.push(match layer_type.as_ref() {
                DenseLayer::<B>::TYPE => {
                    Box::new(DenseLayer::<B>::from_json(&lo, &mut model_data.weights)?)
                }
                Sigmoid::<B>::TYPE => {
                    Box::new(Sigmoid::<B>::from_json(&lo, &mut model_data.weights)?)
                }
                Softmax::<B>::TYPE => {
                    Box::new(Softmax::<B>::from_json(&lo, &mut model_data.weights)?)
                }
                Tanh::<B>::TYPE => Box::new(Tanh::<B>::from_json(&lo, &mut model_data.weights)?),
                Relu::<B>::TYPE => Box::new(Relu::<B>::from_json(&lo, &mut model_data.weights)?),
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

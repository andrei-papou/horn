use std::collections::HashMap;
use std::convert::TryInto;
use std::error::Error;
use std::io::{BufReader, Read, Cursor, Error as IOError};
use std::fs::File;

use crate::byteorder::{NativeEndian, ReadBytesExt};
use crate::serde_json::{
    from_str as json_from_str,
    Result as SDResult,
    Value
};

use crate::backends::{
    Dot,
    TensorAdd,
    Broadcast,
    Backend,
    FromShapedData
};
use crate::common::json_parser::JsonParser;
use crate::common::string_err::{err_to_string, join};
use crate::layers::{Apply, DenseLayer, FromJson};

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

fn read_weights(reader: &mut BufReader<File>, len: usize) -> Result<Vec<f64>, IOError> {
    let mut weights = Vec::with_capacity(len);
    for _ in 0..len {
        let mut buffer = [0u8; F64_SIZE];
        reader.read_exact(&mut buffer)?;
        weights.push(Cursor::new(buffer).read_f64::<NativeEndian>()?);
    }
    Ok(weights)
}

fn decode_from_file(file_path: &str) -> Result<ModelData, String> {
    let f = File::open(file_path).map_err(err_to_string)?;
    let mut reader = BufReader::new(f);

    let mut spec_buffer = vec![0u8; read_size(&mut reader).map_err(err_to_string)?];
    reader.read_exact(&mut spec_buffer).map_err(err_to_string)?;
    let spec = String::from_utf8(spec_buffer).map_err(err_to_string)?;

    let mut weights = HashMap::new();
    while let Ok(wid) = read_wid(&mut reader) {
        let len = read_size(&mut reader).map_err(err_to_string)?;
        let weight_arr = read_weights(&mut reader, len).map_err(err_to_string)?;
        if let Some(_) = weights.insert(wid, weight_arr) {
            return Err(format!("Duplicated weights ID: {}", wid));
        };
    };

    Ok(ModelData { spec, weights })
}

pub struct Model<'a, B: Backend + 'a> {
    name: String,
    layers: Vec<Box<dyn Apply<B> + 'a>>,
}

impl<'a, B: Backend + 'a> Model<'a, B> {
    pub fn from_file(file_path: &str) -> Result<Model<B>, String>
    where
        B::CommonRepr: TryInto<B::Tensor2D, Error = String>,
        B::Tensor2D: Dot<B::Tensor2D> + FromShapedData,
        <B::Tensor2D as Dot<B::Tensor2D>>::Output:
            TensorAdd<Output = <B::Tensor2D as Dot<B::Tensor2D>>::Output> +
            TryInto<B::CommonRepr, Error = String>,
        B::Tensor1D: Broadcast<<B::Tensor2D as Dot<B::Tensor2D>>::Output> + FromShapedData,
        <B::Tensor2D as FromShapedData>::Error: ToString,
        <B::Tensor1D as FromShapedData>::Error: ToString,
    {
        let error_path = "Model::from_file";
        let p = JsonParser::new(error_path);

        let mut model_data = decode_from_file(file_path)?;
        let v: Value = json_from_str(&model_data.spec).map_err(err_to_string)?;

        let name = p.get_string(&v, "name")?.to_string();
        let layer_objects = p.get_array(&v, "layers")?;
        let mut layers: Vec<Box<dyn Apply<B>>> = Vec::with_capacity(layer_objects.len());

        for lo in layer_objects {
            let layer_obj = p.unwrap_object(lo, "layer")?;
            let layer_type = p.unwrap_string_key(layer_obj, "type")?;
            match layer_type.as_ref() {
                DenseLayer::<B>::TYPE => {
                    layers.push(Box::new(DenseLayer::<B>::from_json(layer_obj, &mut model_data.weights)?));
                },
                _ => return Err(join(format!("Unknown layer type: {}", layer_type), error_path)),
            }
        }

        Ok(Model { name, layers })
    }
}

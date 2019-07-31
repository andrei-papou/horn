use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Cursor, Error as IOError, Read};

use byteorder::{BigEndian, ReadBytesExt};

use crate::backends::FromShapedData;
use crate::common::types::{HError, HResult};

const BYTES_PER_ENTRY_SIZE: usize = std::mem::size_of::<u32>();
const BYTES_PER_WEIGHT_ID: usize = std::mem::size_of::<u16>();
const F64_SIZE: usize = std::mem::size_of::<f64>();
const U32_SIZE: usize = std::mem::size_of::<u32>();

pub(crate) struct WeightsMap(HashMap<u16, Vec<f64>>);

impl WeightsMap {
    pub(crate) fn try_build_weight<W>(&mut self, id: u64, shape: Vec<u64>) -> HResult<W>
    where
        W: FromShapedData<Error = HError>,
    {
        let shape = shape.into_iter().map(|x| x as usize).collect();
        let id = id as u16;
        let weights = match self.0.remove(&id) {
            Some(w) => w,
            None => return Err(format_err!("Missing weights for wid \"{}\".", id)),
        };
        W::from_shaped_data(weights, shape)
    }

    pub(crate) fn try_build_weight_optional<W>(
        &mut self,
        id: Option<u64>,
        shape: Option<Vec<u64>>,
    ) -> HResult<Option<W>>
    where
        W: FromShapedData<Error = HError>,
    {
        Ok(
            if let Some((id, sh)) = shape.and_then(|shape| id.map(|id| (id, shape))) {
                Some(self.try_build_weight::<W>(id, sh)?)
            } else {
                None
            },
        )
    }
}

pub(crate) struct ModelData {
    spec: String,
    weights: WeightsMap,
}

impl ModelData {
    fn new(spec: String, weights: WeightsMap) -> Self {
        ModelData { spec, weights }
    }

    pub(crate) fn unwrap(self) -> (String, WeightsMap) {
        (self.spec, self.weights)
    }
}

fn read_size(reader: &mut BufReader<File>) -> Result<usize, IOError> {
    let mut buffer = [0u8; BYTES_PER_ENTRY_SIZE];
    reader.read_exact(&mut buffer)?;
    Ok(Cursor::new(buffer).read_u32::<BigEndian>()? as usize)
}

fn read_wid(reader: &mut BufReader<File>) -> Result<u16, IOError> {
    let mut buffer = [0u8; BYTES_PER_WEIGHT_ID];
    reader.read_exact(&mut buffer)?;
    Cursor::new(buffer).read_u16::<BigEndian>()
}

fn read_f64s(reader: &mut BufReader<File>, num_bytes: usize) -> Result<Vec<f64>, IOError> {
    let len = num_bytes / F64_SIZE;
    let mut data = Vec::with_capacity(len);
    for _ in 0..len {
        let mut buffer = [0u8; F64_SIZE];
        reader.read_exact(&mut buffer)?;
        data.push(Cursor::new(buffer).read_f64::<BigEndian>()?);
    }
    Ok(data)
}

fn read_u32s(reader: &mut BufReader<File>, num_bytes: usize) -> Result<Vec<u32>, IOError> {
    let len = num_bytes / U32_SIZE;
    let mut data = Vec::with_capacity(len);
    for _ in 0..len {
        let mut buffer = [0u8; U32_SIZE];
        reader.read_exact(&mut buffer)?;
        data.push(Cursor::new(buffer).read_u32::<BigEndian>()?);
    }
    Ok(data)
}

pub(crate) fn decode_model_from_file(file_path: &str) -> HResult<ModelData> {
    let f = File::open(file_path)?;
    let mut reader = BufReader::new(f);

    let mut spec_buffer = vec![0u8; read_size(&mut reader)?];
    reader.read_exact(&mut spec_buffer)?;
    let spec = String::from_utf8(spec_buffer)?;

    let mut weights = HashMap::new();
    while let Ok(wid) = read_wid(&mut reader) {
        let len = read_size(&mut reader)?;
        let weight_arr = read_f64s(&mut reader, len)?;
        if let Some(_) = weights.insert(wid, weight_arr) {
            return Err(format_err!("Duplicated weights ID: {}", wid));
        };
    }

    Ok(ModelData::new(spec, WeightsMap(weights)))
}

pub(crate) fn decode_data_from_file<D>(file_path: &str) -> HResult<D>
where
    D: FromShapedData<Error = HError>,
{
    let f = File::open(file_path)?;
    let mut reader = BufReader::new(f);

    let shape_len = read_size(&mut reader)?;
    let shape: Vec<usize> = read_u32s(&mut reader, shape_len)?
        .into_iter()
        .map(|x| x as usize)
        .collect();
    let data_len = read_size(&mut reader)?;
    let data: Vec<f64> = read_f64s(&mut reader, data_len)?;

    D::from_shaped_data(data, shape)
}

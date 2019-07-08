use std::convert::{TryInto};
use std::collections::HashMap;

use crate::serde_json::{Value, Map};

use crate::common::{Name};
use crate::common::json_parser::{JsonParser};
use crate::common::string_err::{err_to_string, join};
use crate::backends::backend::{Backend, TensorAdd, Dot, Broadcast, TensorOpResult, FromShapedData};
use super::traits::{Apply, FromJson};

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

impl<B: Backend> Apply<B> for DenseLayer<B>
where
    B::CommonRepr: TryInto<B::Tensor2D, Error = String>,
    B::Tensor2D: Dot<B::Tensor2D>,
    <B::Tensor2D as Dot<B::Tensor2D>>::Output:
        TensorAdd<Output = <B::Tensor2D as Dot<B::Tensor2D>>::Output> +
        TryInto<B::CommonRepr, Error = String>,
    B::Tensor1D:
        Broadcast<<B::Tensor2D as Dot<B::Tensor2D>>::Output>
{
    fn apply(&self, input: B::CommonRepr) -> TensorOpResult<B::CommonRepr> {
        let input: B::Tensor2D = input.try_into()?;
        let z = input.dot(&self.weights)?;
        let result = match &self.bias {
            Some(bias) => {
                let bc_bias = bias.broadcast(&z)?;
                z.tensor_add(&bc_bias)?
            },
            None => z,
        };
        let result: B::CommonRepr = result.try_into()?;
        Ok(result)
    }
}

impl<B: Backend> FromJson for DenseLayer<B>
where
    B::Tensor2D: FromShapedData,
    <B::Tensor2D as FromShapedData>::Error: ToString,
    B::Tensor1D: FromShapedData,
    <B::Tensor1D as FromShapedData>::Error: ToString,
{
    const TYPE: &'static str = "Dense";
    type Error = String;

    fn from_json(spec: &Map<String, Value>, weights: &mut HashMap<u16, Vec<f64>>) -> Result<DenseLayer<B>, String> {
        let p = JsonParser::new("DenseLayer::from_json");

        // TODO: replace explicit parsing with DenseLayerSpec struct + built-in parsing by Serde
        let name = p.unwrap_string_key(spec, "name")?.to_string();
        let w_shape = p.unwrap_array_key(spec, "w_shape")?.iter()
            .enumerate()
            .map(|(i, v)| p.unwrap_u64(v, &format!("w_shape[{}]", i)).map(|x| x as usize))
            .collect::<Result<Vec<usize>, String>>()?;
        let w_id = p.unwrap_u64_key(spec, "w_id")? as u16;
        let w_data = match weights.remove(&w_id) {
            Some(v) => v,
            None => return Err(format!("Missing weights for wid \"{}\".", w_id))
        };
        let w = B::Tensor2D::from_shaped_data(w_data, w_shape).map_err(err_to_string)?;

        let mut b: Option<B::Tensor1D> = None;
        if !spec["b_shape"].is_null() {
            let b_shape = p.unwrap_array_key(spec, "b_shape")?.iter()
                .enumerate()
                .map(|(i, v)| p.unwrap_u64(v, &format!("b_shape[{}]", i)).map(|x| x as usize))
                .collect::<Result<Vec<usize>, String>>()?;
            let b_id = p.unwrap_u64_key(spec, "b_id")? as u16;
            let b_data = match weights.remove(&b_id) {
                Some(v) => v,
                None => return Err(format!("Missing weights for wid \"{}\".", b_id))
            };
            b = Some(B::Tensor1D::from_shaped_data(b_data, b_shape).map_err(err_to_string)?);
        }

        Ok(DenseLayer::new(name, w, b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::traits::Apply;
    use crate::backends::ndarray_backend::NdArrayBackend;
    use ndarray::{array, Array1, Array2};

    #[test]
    fn test_name() {
        let name = "layer";
        let weights: Array2<f64> = Array2::from_elem((2, 2), 1.0);
        let layer: DenseLayer<NdArrayBackend<_>> = DenseLayer::new(String::from(name), weights, None);
        assert_eq!(layer.name(), name);
    }

    #[test]
    fn test_apply_with_bias() {
        let weights: Array2<f64> = array![[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]];
        let bias: Array1<f64> = array![1.0, -1.0, 1.0];
        let input: Array2<f64> = array![[2.0, 5.0], [7.0, 1.0], [4.0, 3.0]];
        let expected_output: Array2<f64> = array![[6.0, 3.0, 6.0], [0.0, 6.0, 0.0], [4.0, 5.0, 4.0]];
        let layer = DenseLayer::<NdArrayBackend<_>>::new(String::from("layer_1"), weights, Some(bias));
        let output = layer.apply(input.try_into().unwrap());
        assert!(output.is_ok());
        let output: Array2<f64> = output.unwrap().try_into().unwrap();
        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_apply_without_bias() {
        let weights: Array2<f64> = array![[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]];
        let input: Array2<f64> = array![[2.0, 5.0], [7.0, 1.0], [4.0, 3.0]];
        let expected_output: Array2<f64> = array![[5.0, 2.0, 5.0], [1.0, 7.0, 1.0], [3.0, 4.0, 3.0]];
        let layer: DenseLayer<NdArrayBackend<_>> = DenseLayer::new(String::from("layer_1"), weights, None);
        let output = layer.apply(input.try_into().unwrap());
        assert!(output.is_ok());
        let output: Array2<f64> = output.unwrap().try_into().unwrap();
        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_apply_with_invalid_args() {
        let weights = array![[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]];
        let input = array![[2.0, 5.0, 1.0], [7.0, 1.0, 5.0]];
        let layer: DenseLayer<NdArrayBackend<_>> = DenseLayer::new(String::from("layer_1"), weights, None);
        let output = layer.apply(input.try_into().unwrap());
        assert!(output.is_err());
        assert_eq!(output.unwrap_err(), "Incompatible shapes for dot product: [2, 3] x [2, 3]");
    }
}

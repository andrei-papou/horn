use crate::common::{Name};
use crate::backends::backend::{Backend, TensorAdd, Dot, Broadcast, TensorOpResult};
use super::traits::Apply;
use std::convert::{TryInto};

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

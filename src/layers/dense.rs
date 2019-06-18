use std::marker::{PhantomData};
use crate::common::{Name};
use crate::backends::{Tensor, Dot, Broadcast};
use crate::layers::layer::{Apply, LayerResult, Layer};

pub struct DenseLayer<A, W, B>
where
    W: Tensor<A>,
    B: Tensor<A>,
{
    name: String,
    weights: W,
    bias: Option<B>,
    _marker: PhantomData<A>,
}

impl<A, W, B> DenseLayer<A, W, B>
where
    W: Tensor<A>,
    B: Tensor<A>,
{
    pub fn new(name: String, weights: W, bias: Option<B>) -> DenseLayer<A, W, B> {
        DenseLayer {
            name,
            weights,
            bias,
            _marker: PhantomData::<A>
        }
    }
}

impl<A, W, B> Name for DenseLayer<A, W, B>
where
    W: Tensor<A>,
    B: Tensor<A>,
{
    fn name(&self) -> &String {
        &self.name
    }
}

impl<A, W, B> Apply for DenseLayer<A, W, B>
where
    W: Tensor<A> + Dot<W, Output = W>,
    B: Tensor<A> + Broadcast<W>,
{
    type Input = W;
    type Output = W;

    fn apply(&self, input: &W) -> LayerResult<W> {
        let z = input.dot(&self.weights)?;
        Ok(match &self.bias {
            Some(bias) => {
                let bc_bias = bias.broadcast(&z)?;
                z.tensor_add(&bc_bias)?
            },
            None => z,
        })
    }
}

impl<A, W, B> Layer for DenseLayer<A, W, B>
where
    W: Tensor<A> + Dot<W, Output = W>,
    B: Tensor<A> + Broadcast<W>,
{}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};

    #[test]
    fn test_name() {
        let name = "layer";
        let weights: Array2<f64> = Array2::from_elem((2, 2), 1.0);
        let layer: DenseLayer<_, _, Array1<f64>> = DenseLayer::new(String::from(name), weights, None);
        assert_eq!(layer.name(), name);
    }

    #[test]
    fn test_apply_with_bias() {
        let weights: Array2<f64> = array![[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]];
        let bias: Array1<f64> = array![1.0, -1.0, 1.0];
        let input: Array2<f64> = array![[2.0, 5.0], [7.0, 1.0], [4.0, 3.0]];
        let expected_output: Array2<f64> = array![[6.0, 3.0, 6.0], [0.0, 6.0, 0.0], [4.0, 5.0, 4.0]];
        let layer = DenseLayer::new(String::from("layer_1"), weights, Some(bias));
        let output = layer.apply(&input);
        assert!(output.is_ok());
        assert_eq!(output.unwrap(), expected_output);
    }

    #[test]
    fn test_apply_without_bias() {
        let weights: Array2<f64> = array![[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]];
        let input: Array2<f64> = array![[2.0, 5.0], [7.0, 1.0], [4.0, 3.0]];
        let expected_output: Array2<f64> = array![[5.0, 2.0, 5.0], [1.0, 7.0, 1.0], [3.0, 4.0, 3.0]];
        let layer: DenseLayer<_, _, Array1<f64>> = DenseLayer::new(String::from("layer_1"), weights, None);
        let output = layer.apply(&input);
        assert!(output.is_ok());
        assert_eq!(output.unwrap(), expected_output);
    }

    #[test]
    fn test_apply_with_invalid_args() {
        let weights = array![[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]];
        let input = array![[2.0, 5.0, 1.0], [7.0, 1.0, 5.0]];
        let layer: DenseLayer<_, _, Array1<f64>> = DenseLayer::new(String::from("layer_1"), weights, None);
        let output = layer.apply(&input);
        assert!(output.is_err());
        assert_eq!(output.unwrap_err(), "Incompatible shapes for dot product: [2, 3] x [2, 3]");
    }
}

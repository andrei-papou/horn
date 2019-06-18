use crate::num_traits::{One};
use crate::{F64CompliantScalar};
use crate::common::{Name};
use crate::backends::{Tensor, Dot, Broadcast, Exp, TensorOpResult};
use crate::layers::layer::{Apply, Layer, LayerResult};
use std::ops::Div;
use std::marker::PhantomData;

pub struct ActivationLayer<A, T>
where
    T: Tensor<A>,
{
    name: String,
    func: fn(&T) -> TensorOpResult<T>,
    _marker: PhantomData<A>,
}

impl<A, T> ActivationLayer<A, T>
where
    T: Tensor<A>,
{
    fn new(name: String, func: fn(&T) -> TensorOpResult<T>) -> ActivationLayer<A, T> {
        ActivationLayer {
            name,
            func,
            _marker: PhantomData,
        }
    }
}

impl<A, T> Name for ActivationLayer<A, T>
where
    T: Tensor<A>,
{
    fn name(&self) -> &String {
        &self.name
    }
}

impl<A, T> Apply for ActivationLayer<A, T>
where
    T: Tensor<A>,
{
    type Input = T;
    type Output = T;

    fn apply(&self, input: &T) -> LayerResult<T> {
        (self.func)(input)
    }
}

impl<A, T> Layer for ActivationLayer<A, T>
where
    A: F64CompliantScalar,
    T: Tensor<A>,
{}

pub fn sigmoid<A, T>(x: &T) -> TensorOpResult<T>
where
    A: F64CompliantScalar + Copy + One + Div<Output = A>,
    T: Tensor<A> + Exp,
{
    let ones = x.same_from_scalar(A::one());
    let denom = ones.tensor_add(&x.tensor_neg()?.exp())?;
    ones.tensor_div(&denom)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn test_sigmoid() {
        let arr: Array2<f64> = array![[0.0, 0.0], [0.0, 0.0]];
        let layer: ActivationLayer<f64, Array2<f64>> = ActivationLayer::new(String::from("sigmoid"), sigmoid);
        let output = layer.apply(&arr);
        assert!(output.is_ok());
        assert_eq!(output.unwrap(), array![[0.5, 0.5], [0.5, 0.5]]);
    }
}

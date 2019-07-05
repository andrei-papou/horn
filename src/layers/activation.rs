use crate::num_traits::{One};
use crate::{F64CompliantScalar};
use crate::common::{Name};
use crate::backends::backend::{
    TensorAdd,
    TensorSub,
    TensorDiv,
    TensorNeg,
    TensorElemInv,
    Container,
    Broadcast,
    Exp,
    TensorOpResult,
    ReduceSum,
    Backend
};
use crate::layers::traits::Apply;
use std::ops::{Div, Neg};
use std::marker::PhantomData;
use std::convert::TryInto;

pub struct Sigmoid<B: Backend>(String, PhantomData<B>);

impl<B: Backend> Sigmoid<B> {
    pub fn new(name: String) -> Sigmoid<B> {
        Sigmoid(name, PhantomData::<B>)
    }
}

impl<B: Backend> Name for Sigmoid<B> {
    fn name(&self) -> &String {
        &self.0
    }
}

impl<B: Backend> Apply<B> for Sigmoid<B>
where
    B::CommonRepr: TryInto<B::TensorXD, Error = String>,
    B::TensorXD:
        Container +
        TensorNeg<Output = B::TensorXD> +
        Exp<Output = B::TensorXD> +
        TensorAdd<Output = B::TensorXD> +
        TensorDiv<Output = B::TensorXD> +
        TensorElemInv<Output = B::TensorXD> +
        TryInto<B::CommonRepr, Error = String>,
    <B::TensorXD as Container>::Elem:
        F64CompliantScalar +
        Copy +
        One +
        Div<Output = <B::TensorXD as Container>::Elem>,
{
    fn apply(&self, x: B::CommonRepr) -> TensorOpResult<B::CommonRepr> {
        let x: B::TensorXD = x.try_into()?;
        let ones = x.same_from_scalar(<B::TensorXD as Container>::Elem::one());
        let denom = ones.tensor_add(&x.tensor_neg()?.exp())?;
        ones.tensor_div(&denom)?.try_into()
    }
}

pub struct Tanh<B: Backend>(String, PhantomData<B>);

impl<B: Backend> Tanh<B> {
    pub fn new(name: String) -> Tanh<B> {
        Tanh(name, PhantomData::<B>)
    }
}

impl<B: Backend> Name for Tanh<B> {
    fn name(&self) -> &String {
        &self.0
    }
}

impl<B: Backend> Apply<B> for Tanh<B>
where
    B::CommonRepr: TryInto<B::TensorXD, Error = String>,
    B::TensorXD:
        Container +
        Exp<Output = B::TensorXD> +
        TensorNeg<Output = B::TensorXD> +
        TensorAdd<Output = B::TensorXD> +
        TensorSub<Output = B::TensorXD> +
        TensorDiv<Output = B::TensorXD> +
        TensorElemInv<Output = B::TensorXD> +
        TryInto<B::CommonRepr, Error = String>,
    <B::TensorXD as Container>::Elem: Neg,
{
    fn apply(&self, x: B::CommonRepr) -> TensorOpResult<B::CommonRepr> {
        let x: B::TensorXD = x.try_into()?;
        let exp = x.exp();
        let neg_exp = (&x.tensor_neg()?).exp();
        exp.tensor_sub(&neg_exp)?.tensor_div(&exp.tensor_add(&neg_exp)?)?.try_into()
    }
}

pub struct Softmax<B: Backend> {
    name: String,
    axis: usize,
    _backend: PhantomData<B>,
}

impl<B: Backend> Softmax<B> {
    pub fn new(name: String, axis: usize) -> Softmax<B> {
        Softmax {
            name,
            axis,
            _backend: PhantomData::<B>
        }
    }
}

impl<B: Backend> Name for Softmax<B> {
    fn name(&self) -> &String {
        &self.name
    }
}

impl<B: Backend> Apply<B> for Softmax<B>
where
    B::CommonRepr: TryInto<B::TensorXD, Error = String>,
    B::TensorXD:
        Container +
        ReduceSum +
        TensorDiv<Output = B::TensorXD> +
        TryInto<B::CommonRepr, Error = String>,
    <B::TensorXD as ReduceSum>::Output: Broadcast<B::TensorXD>,
{
    fn apply(&self, input: B::CommonRepr) -> TensorOpResult<B::CommonRepr> {
        let x: B::TensorXD = input.try_into()?;
        x.tensor_div(&x.reduce_sum(self.axis)?.broadcast(&x)?)?.try_into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};
    use crate::backends::ndarray_backend::NdArrayBackend;

    #[test]
    fn test_sigmoid() {
        let arr: Array2<f64> = array![[0.0, 0.0], [0.0, 0.0]];
        let layer = Sigmoid::<NdArrayBackend<_>>::new(String::from("sigmoid"));
        let output = layer.apply(arr.try_into().unwrap());
        assert!(output.is_ok());
        let output: Array2<f64> = output.unwrap().try_into().unwrap();
        assert_eq!(output, array![[0.5, 0.5], [0.5, 0.5]]);
    }

    #[test]
    fn test_tanh() {
        let arr: Array2<f64> = array![[0.0, 0.0], [0.0, 0.0]];
        let layer = Tanh::<NdArrayBackend<_>>::new(String::from("tanh"));
        let output = layer.apply(arr.try_into().unwrap());
        assert!(output.is_ok());
        let output: Array2<f64> = output.unwrap().try_into().unwrap();
        assert_eq!(output, array![[0.0, 0.0], [0.0, 0.0]]);
    }

    #[test]
    fn test_softmax() {
        let arr: Array2<f64> = array![[0.5, 1.5], [3.0, 3.0]];
        let layer = Softmax::<NdArrayBackend<_>>::new(String::from("softmax"), 1);
        let output = layer.apply(arr.try_into().unwrap());
        assert!(output.is_ok());
        let output: Array2<f64> = output.unwrap().try_into().unwrap();
        assert_eq!(output, array![[0.25, 0.75], [0.5, 0.5]]);
    }
}

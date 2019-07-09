use std::marker::PhantomData;
use std::collections::HashMap;
use std::convert::TryInto;

use crate::num_traits::{One, Zero};
use crate::serde::{Serialize, Deserialize};
use crate::serde_json::{from_value, Value};

use crate::common::{Name, string_err::err_to_string};
use crate::backends::backend::{
    TensorAdd,
    TensorSub,
    TensorMul,
    TensorDiv,
    TensorNeg,
    Container,
    Broadcast,
    Exp,
    TensorOpResult,
    ReduceSum,
    Backend,
    Shape,
    MaskCmp,
};
use crate::layers::traits::{Apply, FromJson};

#[derive(Serialize, Deserialize)]
struct GenericActivationSpec {
    name: String
}

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

impl<B: Backend> Apply<B> for Sigmoid<B> {
    fn apply(&self, x: B::CommonRepr) -> TensorOpResult<B::CommonRepr> {
        let x: B::TensorXD = x.try_into()?;
        let ones = x.same_from_scalar(<B::TensorXD as Container>::Elem::one());
        let denom = ones.tensor_add(&x.tensor_neg()?.exp())?;
        ones.tensor_div(&denom)?.try_into()
    }
}

impl<B: Backend> FromJson for Sigmoid<B> {
    const TYPE: &'static str = "Sigmoid";

    type Error = String;

    fn from_json(json: &Value, _weights: &mut HashMap<u16, Vec<f64>>) -> Result<Self, String> {
        let spec: GenericActivationSpec = from_value(json.clone()).map_err(err_to_string)?;
        Ok(Sigmoid::new(spec.name))
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

impl<B: Backend> Apply<B> for Tanh<B> {
    fn apply(&self, x: B::CommonRepr) -> TensorOpResult<B::CommonRepr> {
        let x: B::TensorXD = x.try_into()?;
        let exp = x.exp();
        let neg_exp = (&x.tensor_neg()?).exp();
        exp.tensor_sub(&neg_exp)?.tensor_div(&exp.tensor_add(&neg_exp)?)?.try_into()
    }
}

impl<B: Backend> FromJson for Tanh<B> {
    const TYPE: &'static str = "Tanh";

    type Error = String;

    fn from_json(json: &Value, _weights: &mut HashMap<u16, Vec<f64>>) -> Result<Self, String> {
        let spec: GenericActivationSpec = from_value(json.clone()).map_err(err_to_string)?;
        Ok(Tanh::new(spec.name))
    }
}

pub struct Relu<B: Backend>(String, PhantomData<B>);

impl<B: Backend> Relu<B> {
    fn new(name: String) -> Relu<B> {
        Relu(name, PhantomData::<B>)
    }
}

impl<B: Backend> Name for Relu<B> {
    fn name(&self) -> &String {
        &self.0
    }
}

impl<B: Backend> Apply<B> for Relu<B> {
    fn apply(&self, x: B::CommonRepr) -> TensorOpResult<B::CommonRepr> {
        let x: B::TensorXD = x.try_into()?;
        x.tensor_mul(&x.mask_gt(<B::TensorXD as Container>::Elem::zero())?)?.try_into()
    }
}

impl<B: Backend> FromJson for Relu<B> {
    const TYPE: &'static str = "Relu";

    type Error = String;

    fn from_json(json: &Value, _weights: &mut HashMap<u16, Vec<f64>>) -> Result<Self, String> {
        let spec: GenericActivationSpec = from_value(json.clone()).map_err(err_to_string)?;
        Ok(Relu::new(spec.name))
    }
}

pub struct Softmax<B: Backend> {
    name: String,
    axis: Option<usize>,
    _backend: PhantomData<B>,
}

impl<B: Backend> Softmax<B> {
    pub fn new(name: String, axis: Option<usize>) -> Softmax<B> {
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

impl<B: Backend> Apply<B> for Softmax<B> {
    fn apply(&self, input: B::CommonRepr) -> TensorOpResult<B::CommonRepr> {
        let x: B::TensorXD = input.try_into()?;
        let axis = self.axis.unwrap_or(<B::TensorXD as Shape>::shape(&x).len() - 1usize);
        let x = x.exp();
        println!("{}", x);
        x.tensor_div(&x.reduce_sum(axis)?.broadcast(&x)?)?.try_into()
    }
}

#[derive(Serialize, Deserialize)]
struct SoftmaxSpec {
    name: String,
    axis: Option<u64>,
}

impl<B: Backend> FromJson for Softmax<B> {
    const TYPE: &'static str = "Softmax";

    type Error = String;

    fn from_json(json: &Value, _weights: &mut HashMap<u16, Vec<f64>>) -> Result<Self, String> {
        let spec: SoftmaxSpec = from_value(json.clone()).map_err(err_to_string)?;
        Ok(Softmax::new(spec.name, spec.axis.map(|x| x as usize)))
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
    fn test_relu() {
        let arr: Array2<f64> = array![[-4.2, 1.0], [4.3, -2.0]];
        let layer = Relu::<NdArrayBackend<_>>::new(String::from("relu"));
        let output = layer.apply(arr.try_into().unwrap());
        assert!(output.is_ok());
        let output: Array2<f64> = output.unwrap().try_into().unwrap();
        assert_eq!(output, array![[0.0, 1.0], [4.3, 0.0]]);
    }

    #[cfg(test)]
    mod softmax {
        use super::*;

        #[test]
        fn test_with_axis() {
            let arr: Array2<f64> = array![[0.5, 1.5], [3.0, 3.0]];
            let layer = Softmax::<NdArrayBackend<_>>::new(String::from("softmax"), Some(1));
            let output = layer.apply(arr.try_into().unwrap());
            assert!(output.is_ok());
            let output: Array2<f64> = output.unwrap().try_into().unwrap();
            assert_eq!(output, array![[0.25, 0.75], [0.5, 0.5]]);
        }

        #[test]
        fn test_without_axis() {
            let arr: Array2<f64> = array![[0.5, 1.5], [3.0, 3.0]];
            let layer = Softmax::<NdArrayBackend<_>>::new(String::from("softmax"), None);
            let output = layer.apply(arr.try_into().unwrap());
            assert!(output.is_ok());
            let output: Array2<f64> = output.unwrap().try_into().unwrap();
            assert_eq!(output, array![[0.25, 0.75], [0.5, 0.5]]);
        }
    }
}

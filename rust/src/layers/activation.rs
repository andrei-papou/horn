use std::cmp::Ordering;
use std::convert::TryInto;
use std::marker::PhantomData;

use crate::num_traits::{One, Zero};
use crate::serde::{Deserialize, Serialize};
use crate::serde_json::{from_value, Value};

use crate::backends::backend::{
    Backend, ClipByValueInPlace, Container, Exp, ReduceSum, Reshape, Shape, ShapeVec, TensorAdd,
    TensorDiv, TensorDivInPlace, TensorNeg, TensorSub,
};
use crate::common::types::{HError, HResult};
use crate::common::Name;
use crate::layers::traits::{Apply, FromJson};
use crate::model::binary_format::WeightsMap;

#[derive(Serialize, Deserialize)]
struct GenericActivationSpec {
    name: String,
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
    fn apply(&self, x: B::CommonRepr) -> HResult<B::CommonRepr> {
        let x: B::TensorXD = x.try_into()?;
        let ones = x.same_from_scalar(<B::TensorXD as Container>::Elem::one());
        let denom = ones.tensor_add(&x.tensor_neg()?.exp())?;
        ones.tensor_div(&denom)?.try_into()
    }
}

impl<B: Backend> FromJson for Sigmoid<B> {
    const TYPE: &'static str = "Sigmoid";

    type Error = HError;

    fn from_json(json: &Value, _weights: &mut WeightsMap) -> HResult<Self> {
        let spec: GenericActivationSpec = from_value(json.clone())?;
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
    fn apply(&self, x: B::CommonRepr) -> HResult<B::CommonRepr> {
        let x: B::TensorXD = x.try_into()?;
        let exp = x.exp();
        let neg_exp = (&x.tensor_neg()?).exp();
        exp.tensor_sub(&neg_exp)?
            .tensor_div(&exp.tensor_add(&neg_exp)?)?
            .try_into()
    }
}

impl<B: Backend> FromJson for Tanh<B> {
    const TYPE: &'static str = "Tanh";

    type Error = HError;

    fn from_json(json: &Value, _weights: &mut WeightsMap) -> HResult<Self> {
        let spec: GenericActivationSpec = from_value(json.clone())?;
        Ok(Tanh::new(spec.name))
    }
}

pub struct Relu<B: Backend>(String, PhantomData<B>);

impl<B: Backend> Relu<B> {
    pub fn new(name: String) -> Relu<B> {
        Relu(name, PhantomData::<B>)
    }
}

impl<B: Backend> Name for Relu<B> {
    fn name(&self) -> &String {
        &self.0
    }
}

impl<B: Backend> Apply<B> for Relu<B> {
    fn apply(&self, x: B::CommonRepr) -> HResult<B::CommonRepr> {
        let mut x: B::TensorXD = x.try_into()?;
        x.clip_by_value_in_place(<B::TensorXD as Container>::Elem::zero(), &Ordering::Less)?;
        x.try_into()
    }
}

impl<B: Backend> FromJson for Relu<B> {
    const TYPE: &'static str = "Relu";

    type Error = HError;

    fn from_json(json: &Value, _weights: &mut WeightsMap) -> HResult<Self> {
        let spec: GenericActivationSpec = from_value(json.clone())?;
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
            _backend: PhantomData::<B>,
        }
    }
}

impl<B: Backend> Name for Softmax<B> {
    fn name(&self) -> &String {
        &self.name
    }
}

impl<B: Backend> Apply<B> for Softmax<B> {
    fn apply(&self, input: B::CommonRepr) -> HResult<B::CommonRepr> {
        let x: B::TensorXD = input.try_into()?;
        let mut shape: ShapeVec = <B::TensorXD as Shape>::shape(&x);
        let axis = self.axis.unwrap_or(shape.len() - 1usize);
        shape[axis] = 1;
        let mut x = x.exp();
        x.tensor_div_in_place(&x.reduce_sum(axis)?.reshape(shape)?)?;
        x.try_into()
    }
}

#[derive(Serialize, Deserialize)]
struct SoftmaxSpec {
    name: String,
    axis: Option<u64>,
}

impl<B: Backend> FromJson for Softmax<B> {
    const TYPE: &'static str = "Softmax";

    type Error = HError;

    fn from_json(json: &Value, _weights: &mut WeightsMap) -> HResult<Self> {
        let spec: SoftmaxSpec = from_value(json.clone())?;
        Ok(Softmax::new(spec.name, spec.axis.map(|x| x as usize)))
    }
}

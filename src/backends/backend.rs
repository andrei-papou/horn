use std::convert::{TryInto, TryFrom};
use std::ops::Neg;
use std::fmt::Display;
use crate::F64CompliantScalar;
use num_traits::One;

pub type ShapeVec = Vec<usize>;
pub type TensorOpResult<T> = Result<T, String>;

pub trait Shape {
    fn shape(&self) -> ShapeVec;
}

pub trait TensorAdd<Rhs = Self> {
    type Output;

    fn tensor_add(&self, rhs: &Rhs) -> TensorOpResult<Self::Output>;
}

pub trait TensorNeg {
    type Output;

    fn tensor_neg(&self) -> TensorOpResult<Self::Output>;
}

pub trait TensorSub<Rhs = Self> : TensorAdd<Rhs>
where
    Rhs: TensorNeg<Output = Rhs>
{
    fn tensor_sub(&self, rhs: &Rhs) -> TensorOpResult<<Self as TensorAdd<Rhs>>::Output> {
        self.tensor_add(&rhs.tensor_neg()?)
    }
}

pub trait TensorMul<Rhs = Self> {
    type Output;

    fn tensor_mul(&self, rhs: &Rhs) -> TensorOpResult<Self::Output>;
}

pub trait TensorElemInv {
    type Output;

    fn tensor_elem_inv(&self) -> TensorOpResult<Self::Output>;
}

pub trait TensorDiv<Rhs = Self> {
    type Output;

    fn tensor_div(&self, rhs: &Rhs) -> TensorOpResult<Self::Output>;
}

pub trait Tensor<Scalar, CommonRepr>:
    TensorAdd<Self, Output = Self> +
    TensorSub<Self, Output = Self> +
    TensorMul<Self, Output = Self> +
    TensorDiv<Self, Output = Self> +
    TensorElemInv<Output = Self> +
    TensorNeg<Output = Self> +
    Exp<Output = Self> +
    Container<Elem = Scalar> +
    Shape +
    FromShapedData<Error = String> +
    TryInto<CommonRepr, Error = String>
where
    Self: Sized,
{}

pub trait Dot<Rhs> : Shape
where
    Rhs: Shape
{
    type Output;

    fn dot(&self, rhs: &Rhs) -> TensorOpResult<Self::Output>;
}

pub trait Exp {
    type Output;

    fn exp(&self) -> Self::Output;
}

pub trait Container {
    type Elem;

    fn same_from_scalar(&self, x: Self::Elem) -> Self;
}

pub trait Broadcast<To>
where
    Self: Sized
{
    fn broadcast(&self, to: &To) -> TensorOpResult<To>;
}

pub trait ReduceSum {
    type Output;

    fn reduce_sum(&self, axis: usize) -> TensorOpResult<Self::Output>;
}

pub trait ReduceMean {
    type Output;

    fn reduce_mean(&self, axis: usize) -> TensorOpResult<Self::Output>;
}

pub trait FromShapedData
where
    Self: Sized
{
    type Error;

    fn from_shaped_data(data: Vec<f64>, shape: ShapeVec) -> Result<Self, Self::Error>;
}

pub trait Backend
where
    Self::Scalar: F64CompliantScalar + One,
    Self::CommonRepr:
        TryInto<Self::Tensor1D, Error = String> +
        TryInto<Self::Tensor2D, Error = String> +
        TryInto<Self::TensorXD, Error = String> +
        FromShapedData<Error = String>,
    Self::Tensor1D:
        Tensor<Self::Scalar, Self::CommonRepr> +
        Broadcast<Self::Tensor2D>,
    Self::Tensor2D:
        Tensor<Self::Scalar, Self::CommonRepr> +
        Dot<Self::Tensor2D, Output = Self::Tensor2D> +
        ReduceSum<Output = Self::Tensor1D> +
        ReduceMean<Output = Self::Tensor1D>,
    Self::TensorXD:
        Tensor<Self::Scalar, Self::CommonRepr> +
        Broadcast<Self::TensorXD> +
        ReduceSum<Output = Self::TensorXD> +
        ReduceMean<Output = Self::TensorXD>,
{
    type Scalar;
    type CommonRepr;

    type Tensor1D;
    type Tensor2D;
    type TensorXD;
}

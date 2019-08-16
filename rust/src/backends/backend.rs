use std::cmp::Ordering;
use std::convert::TryInto;
use std::fmt::{Debug, Display};

use num_traits::{real::Real, One, Zero};

use crate::common::traits::F64CompliantScalar;
use crate::common::types::{HError, HResult};
use crate::model::binary_format::decode_data_from_file;

use super::convnets;

pub type ShapeVec = Vec<usize>;

pub trait Shape {
    fn shape(&self) -> ShapeVec;
}

pub trait TensorAdd<Rhs = Self> {
    type Output;

    fn tensor_add(&self, rhs: &Rhs) -> HResult<Self::Output>;
}

pub trait TensorAddInPlace<Rhs = Self> {
    fn tensor_add_in_place(&mut self, rhs: &Rhs) -> HResult<()>;
}

pub trait TensorNeg {
    type Output;

    fn tensor_neg(&self) -> HResult<Self::Output>;
}

pub trait TensorSub<Rhs = Self>: TensorAdd<Rhs>
where
    Rhs: TensorNeg<Output = Rhs>,
{
    fn tensor_sub(&self, rhs: &Rhs) -> HResult<<Self as TensorAdd<Rhs>>::Output> {
        self.tensor_add(&rhs.tensor_neg()?)
    }
}

pub trait TensorMul<Rhs = Self> {
    type Output;

    fn tensor_mul(&self, rhs: &Rhs) -> HResult<Self::Output>;
}

pub trait TensorElemInv {
    type Output;

    fn tensor_elem_inv(&self) -> HResult<Self::Output>;
}

pub trait TensorDiv<Rhs = Self> {
    type Output;

    fn tensor_div(&self, rhs: &Rhs) -> HResult<Self::Output>;
}

pub trait TensorDivInPlace<Rhs = Self> {
    fn tensor_div_in_place(&mut self, rhs: &Rhs) -> HResult<()>;
}

pub trait MaskCmp: Container
where
    <Self as Container>::Elem: PartialOrd,
{
    type Mask;

    fn mask_cmp(&self, x: <Self as Container>::Elem, ord: &Ordering) -> HResult<Self::Mask>;
}

pub trait ClipByValueInPlace: Container {
    fn clip_by_value_in_place(&mut self, val: Self::Elem, ord: &Ordering) -> HResult<()>;
}

pub trait Abs: Container
where
    <Self as Container>::Elem: Real,
{
    type Output;

    fn abs(&self) -> HResult<Self::Output>;
}

pub trait IntoScalar {
    type Output;

    fn into_scalar(self) -> HResult<Self::Output>;
}

pub trait OneHotMax {
    type Output;

    fn one_hot_max(&self, axis: usize) -> HResult<Self::Output>;
}

pub trait Tensor<Scalar, CommonRepr>:
    TensorAdd<Output = Self>
    + TensorAddInPlace
    + TensorSub<Output = Self>
    + TensorMul<Output = Self>
    + TensorDiv<Output = Self>
    + TensorDivInPlace
    + TensorElemInv<Output = Self>
    + TensorNeg<Output = Self>
    + Exp<Output = Self>
    + Container<Elem = Scalar>
    + MaskCmp<Mask = Self>
    + OneHotMax<Output = Self>
    + ClipByValueInPlace
    + Shape
    + Abs<Output = Self>
    + FromShapedData<Error = HError>
    + TryInto<CommonRepr, Error = HError>
    + FromFile
    + Debug
    + Clone
where
    Scalar: Real,
    Self: Sized,
    <Self as Container>::Elem: PartialOrd,
{
}

pub trait Dot<Rhs>: Shape
where
    Rhs: Shape,
{
    type Output;

    fn dot(&self, rhs: &Rhs) -> HResult<Self::Output>;
}

pub trait Exp {
    type Output;

    fn exp(&self) -> Self::Output;
}

pub trait Container {
    type Elem;

    fn same_from_scalar(&self, x: Self::Elem) -> Self;
}

pub trait Reshape
where
    Self: Sized,
{
    type Output;

    fn reshape(self, new_shape: ShapeVec) -> HResult<Self::Output>;
}

pub trait ReduceSum {
    type Output;

    fn reduce_sum(&self, axis: usize) -> HResult<Self::Output>;
}

pub trait ReduceMean {
    type Output;

    fn reduce_mean(&self, axis: usize) -> HResult<Self::Output>;
}

pub trait FromShapedData
where
    Self: Sized,
{
    type Error;

    fn from_shaped_data(data: Vec<f64>, shape: ShapeVec) -> Result<Self, Self::Error>;
}

pub trait FromFile
where
    Self: Sized + FromShapedData<Error = HError>,
{
    fn from_file(file_path: &str) -> HResult<Self> {
        decode_data_from_file(file_path)
    }
}

pub trait Backend
where
    Self::Scalar: F64CompliantScalar + Zero + One + PartialOrd + Real + Debug + Display,
    Self::CommonRepr: TryInto<Self::Tensor1D, Error = HError>
        + TryInto<Self::Tensor2D, Error = HError>
        + TryInto<Self::Tensor3D, Error = HError>
        + TryInto<Self::Tensor4D, Error = HError>
        + TryInto<Self::TensorXD, Error = HError>
        + FromShapedData<Error = HError>,
    Self::Tensor0D: IntoScalar<Output = Self::Scalar>,
    Self::Tensor1D: Tensor<Self::Scalar, Self::CommonRepr>
        + Reshape<Output = Self::TensorXD>
        + ReduceSum<Output = Self::Tensor0D>
        + ReduceMean<Output = Self::Tensor0D>,
    Self::Tensor2D: Tensor<Self::Scalar, Self::CommonRepr>
        + TensorAddInPlace<Self::Tensor1D>
        + Dot<Self::Tensor2D, Output = Self::Tensor2D>
        + Reshape<Output = Self::TensorXD>
        + ReduceSum<Output = Self::Tensor1D>
        + ReduceMean<Output = Self::Tensor1D>,
    Self::Tensor3D: Tensor<Self::Scalar, Self::CommonRepr>,
    Self::Tensor4D: Tensor<Self::Scalar, Self::CommonRepr>
        + convnets::Conv2D<Self::Tensor4D, Self::Tensor1D, Output = Self::Tensor4D>
        + convnets::AvgPool2D<Output = Self::Tensor4D>
        + convnets::MaxPool2D<Output = Self::Tensor4D>,
    Self::TensorXD: Tensor<Self::Scalar, Self::CommonRepr>
        + Reshape<Output = Self::TensorXD>
        + ReduceSum<Output = Self::TensorXD>
        + ReduceMean<Output = Self::TensorXD>
        + Display
        + Debug,
{
    const NAME: &'static str;

    type Scalar;
    type CommonRepr;

    type Tensor0D;
    type Tensor1D;
    type Tensor2D;
    type Tensor3D;
    type Tensor4D;
    type TensorXD;
}

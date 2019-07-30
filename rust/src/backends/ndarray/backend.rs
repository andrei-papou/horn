use std::convert::{TryFrom, TryInto};
use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg};

use ndarray::{
    Array, Array1, Array2, Array3, Array4, ArrayD, Axis, Dimension, LinalgScalar, RemoveAxis,
};
use num_traits::{real::Real, One, Zero};

use crate::backends::backend::{
    Backend, Broadcast, Container, Dot, Exp, FromFile, FromShapedData, MaskCmp, ReduceMean,
    ReduceSum, Reshape, Shape, ShapeVec, Tensor, TensorAdd, TensorDiv, TensorElemInv, TensorMul,
    TensorNeg, TensorSub, Transpose,
};
use crate::backends::convnets;
use crate::common::traits::F64CompliantScalar;
use crate::common::types::{HError, HResult};

use super::convnets::{avg_pool2d, conv2d, max_pool2d};

pub struct NdArrayBackend<A> {
    _marker: PhantomData<A>,
}

#[derive(Debug)]
pub struct NdArrayCommonRepr<A> {
    data: Vec<A>,
    shape: ShapeVec,
}

impl<A> NdArrayCommonRepr<A> {
    fn new(data: Vec<A>, shape: ShapeVec) -> NdArrayCommonRepr<A> {
        NdArrayCommonRepr { data, shape }
    }
}

impl<A> FromShapedData for NdArrayCommonRepr<A>
where
    A: TryFrom<f64>,
    <A as TryFrom<f64>>::Error: failure::Fail,
{
    type Error = HError;

    fn from_shaped_data(data: Vec<f64>, shape: ShapeVec) -> HResult<Self> {
        let data = data
            .into_iter()
            .map(|x| A::try_from(x).map_err(|err| err.into()))
            .collect::<Result<Vec<A>, HError>>()?;
        Ok(NdArrayCommonRepr::new(data, shape))
    }
}

impl<A> Backend for NdArrayBackend<A>
where
    A: LinalgScalar
        + F64CompliantScalar
        + Neg<Output = A>
        + PartialOrd
        + Display
        + Debug
        + Real
        + Sum<A>,
{
    type Scalar = A;
    type CommonRepr = NdArrayCommonRepr<A>;

    type Tensor1D = Array1<A>;
    type Tensor2D = Array2<A>;
    type Tensor3D = Array3<A>;
    type Tensor4D = Array4<A>;
    type TensorXD = ArrayD<A>;
}

fn check_shapes_equal<T: Shape>(lhs: &T, rhs: &T) -> HResult<()> {
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    if &lhs_shape != &rhs_shape {
        Err(format_err!(
            "Incompatible shapes: {:?} and {:?}.",
            &lhs_shape,
            &rhs_shape
        ))
    } else {
        Ok(())
    }
}

impl<A, D> Container for Array<A, D>
where
    A: Clone,
    D: Dimension,
{
    type Elem = A;

    fn same_from_scalar(&self, x: A) -> Self {
        Array::<A, D>::from_elem(self.dim(), x)
    }
}

impl<A, D> TensorAdd<Array<A, D>> for Array<A, D>
where
    A: LinalgScalar,
    D: Dimension,
{
    type Output = Array<A, D>;

    fn tensor_add(&self, rhs: &Array<A, D>) -> HResult<Array<A, D>> {
        check_shapes_equal(self, rhs)?;
        Ok(self + rhs)
    }
}

impl<A, D> TensorMul<Array<A, D>> for Array<A, D>
where
    A: LinalgScalar,
    D: Dimension,
{
    type Output = Array<A, D>;

    fn tensor_mul(&self, rhs: &Array<A, D>) -> HResult<Array<A, D>> {
        check_shapes_equal(self, rhs)?;
        Ok(self * rhs)
    }
}

impl<A, D> TensorNeg for Array<A, D>
where
    A: Copy + LinalgScalar + Neg<Output = A>,
    D: Dimension,
    Self: TensorMul<Self, Output = Self> + Container<Elem = A>,
{
    type Output = Self;

    fn tensor_neg(&self) -> HResult<Self> {
        self.tensor_mul(&self.same_from_scalar(A::one().neg()))
    }
}

impl<A, D> TensorSub<Array<A, D>> for Array<A, D>
where
    A: Copy + LinalgScalar + Neg<Output = A>,
    D: Dimension,
    Self: TensorMul<Self, Output = Self> + Container<Elem = A>,
{
}

impl<A, D> TensorElemInv for Array<A, D>
where
    A: LinalgScalar + PartialEq,
    D: Dimension,
{
    type Output = Array<<A as Div>::Output, D>;

    fn tensor_elem_inv(&self) -> HResult<Array<<A as Div>::Output, D>> {
        if let Some(_) = self.iter().find(|x| **x == A::zero()) {
            return Err(format_err!("Division by zero is forbidden!"));
        }
        Ok(self.map(|x| A::one() / *x))
    }
}

impl<A, D> TensorDiv<Array<A, D>> for Array<A, D>
where
    A: LinalgScalar + PartialEq,
    D: Dimension,
{
    type Output = <Self as TensorMul<Array<A, D>>>::Output;

    fn tensor_div(&self, rhs: &Array<A, D>) -> HResult<<Self as TensorMul<Array<A, D>>>::Output> {
        self.tensor_mul(&rhs.tensor_elem_inv()?)
    }
}

impl<A, D> MaskCmp for Array<A, D>
where
    A: PartialOrd + One + Zero + Clone,
    D: Dimension,
{
    type Mask = Array<A, D>;

    fn mask_lt(&self, x: A) -> HResult<Self::Mask> {
        Ok(self.map(|a| if *a < x { A::one() } else { A::zero() }))
    }

    fn mask_gt(&self, x: A) -> HResult<Self::Mask> {
        Ok(self.map(|a| if *a > x { A::one() } else { A::zero() }))
    }

    fn mask_eq(&self, x: A) -> HResult<Self::Mask> {
        Ok(self.map(|a| if *a == x { A::one() } else { A::zero() }))
    }
}

impl<A: Clone> Transpose for Array2<A> {
    type Output = Array2<A>;

    fn transpose(&self) -> HResult<Array2<A>> {
        Ok(self.t().to_owned())
    }
}

impl<A> Dot<Array2<A>> for Array2<A>
where
    A: LinalgScalar,
{
    type Output = Array2<A>;

    fn dot(&self, rhs: &Array2<A>) -> HResult<Array2<A>> {
        let self_shape = Shape::shape(self);
        let rhs_shape = Shape::shape(rhs);
        if self_shape[1] != rhs_shape[0] {
            return Err(format_err!(
                "Incompatible shapes for dot product: {:?} x {:?}",
                self_shape,
                rhs_shape
            ));
        }
        Ok(self.dot(rhs))
    }
}

impl<A, D> Exp for Array<A, D>
where
    A: F64CompliantScalar + Copy,
    D: Dimension,
{
    type Output = Array<A, D>;

    fn exp(&self) -> Array<A, D> {
        self.map(|x| (*x).map_f64(|x| x.exp()))
    }
}

impl<A, D> Broadcast<Array<A, D::Larger>> for Array<A, D>
where
    A: Clone,
    D: Dimension,
{
    fn broadcast(&self, rhs: &Array<A, D::Larger>) -> HResult<Array<A, D::Larger>> {
        match self.clone().broadcast(rhs.dim()).map(|x| x.to_owned()) {
            Some(result) => Ok(result),
            None => Err(format_err!(
                "Cannot broadcast {:?} to {:?}.",
                Shape::shape(self),
                Shape::shape(rhs),
            )),
        }
    }
}

impl<A, D> Reshape for Array<A, D>
where
    A: Clone,
    D: Dimension,
{
    type Output = ArrayD<A>;

    fn reshape(&self, new_shape: ShapeVec) -> HResult<ArrayD<A>> {
        self.clone().into_shape(new_shape).map_err(|err| err.into())
    }
}

impl<A, D> Shape for Array<A, D>
where
    D: Dimension,
{
    fn shape(&self) -> ShapeVec {
        self.shape().iter().cloned().collect::<Vec<usize>>()
    }
}

impl<A, D> ReduceSum for Array<A, D>
where
    A: Clone + Zero + Add<Output = A>,
    D: Dimension + RemoveAxis,
{
    type Output = Array<A, D::Smaller>;

    fn reduce_sum(&self, axis: usize) -> HResult<Self::Output> {
        Ok(self.sum_axis(Axis(axis)))
    }
}

impl<A, D> ReduceMean for Array<A, D>
where
    A: Clone + Zero + One + Add<Output = A> + Div<Output = A>,
    D: Dimension + RemoveAxis,
{
    type Output = Array<A, D::Smaller>;

    fn reduce_mean(&self, axis: usize) -> HResult<Self::Output> {
        Ok(self.mean_axis(Axis(axis)))
    }
}

impl<A, D> TryFrom<Array<A, D>> for NdArrayCommonRepr<A>
where
    Array<A, D>: Shape,
    D: Dimension,
{
    type Error = HError;

    fn try_from(value: Array<A, D>) -> HResult<NdArrayCommonRepr<A>> {
        let shape = <Array<A, D> as Shape>::shape(&value);
        let data = value.into_raw_vec();
        Ok(NdArrayCommonRepr::new(data, shape))
    }
}

impl<A, D> TryInto<Array<A, D>> for NdArrayCommonRepr<A>
where
    D: Dimension,
{
    type Error = HError;

    fn try_into(self) -> HResult<Array<A, D>> {
        Array::<A, _>::from_shape_vec(self.shape, self.data)?
            .into_dimensionality::<D>()
            .map_err(|err| err.into())
    }
}

impl<A, D> FromShapedData for Array<A, D>
where
    A: From<f64>,
    D: Dimension,
{
    type Error = HError;

    fn from_shaped_data(data: Vec<f64>, shape: ShapeVec) -> HResult<Array<A, D>> {
        let vec: Vec<A> = data.into_iter().map(|x| A::from(x)).collect();
        Array::<A, _>::from_shape_vec(shape, vec)?
            .into_dimensionality::<D>()
            .map_err(|err| err.into())
    }
}

impl<A, D> FromFile for Array<A, D>
where
    A: From<f64>,
    D: Dimension,
{
}

impl<A> convnets::Conv2D<Array4<A>, Array1<A>> for Array4<A>
where
    A: Debug + Clone + Copy + Add<Output = A> + Mul<A, Output = A> + Zero,
{
    type Output = Array4<A>;

    fn conv2d(
        &self,
        kernels: &Array4<A>,
        biases: &Option<Array1<A>>,
        strides: (usize, usize),
        padding: convnets::Padding,
        data_format: convnets::DataFormat,
    ) -> HResult<Array4<A>> {
        Ok(conv2d(
            self,
            kernels,
            biases,
            strides,
            padding,
            data_format,
        )?)
    }
}

impl<A> convnets::AvgPool2D for Array4<A>
where
    A: LinalgScalar + Debug + Clone + Copy + Neg<Output = A> + PartialOrd + Real + Sum<A>,
{
    type Output = Array4<A>;

    fn avg_pool2d(
        &self,
        pool_window: convnets::Pool2,
        strides: convnets::Stride2,
        padding: convnets::Padding,
        data_format: convnets::DataFormat,
    ) -> HResult<Self::Output> {
        Ok(avg_pool2d(
            self,
            pool_window,
            strides,
            padding,
            data_format,
        )?)
    }
}

impl<A> convnets::MaxPool2D for Array4<A>
where
    A: LinalgScalar + Debug + Clone + Copy + Neg<Output = A> + PartialOrd + Real + Sum<A>,
{
    type Output = Array4<A>;

    fn max_pool2d(
        &self,
        pool_window: convnets::Pool2,
        strides: convnets::Stride2,
        padding: convnets::Padding,
        data_format: convnets::DataFormat,
    ) -> HResult<Self::Output> {
        Ok(max_pool2d(
            self,
            pool_window,
            strides,
            padding,
            data_format,
        )?)
    }
}

impl<A, D> Tensor<A, NdArrayCommonRepr<A>> for Array<A, D>
where
    A: LinalgScalar + Clone + F64CompliantScalar + Neg<Output = A> + PartialOrd + Real + Sum<A>,
    D: Dimension,
{
}

use std::fmt::{Display, Debug};
use std::marker::PhantomData;
use std::ops::{Neg, Div, Add};
use std::convert::{TryInto, TryFrom};
use crate::num_traits::identities::{One, Zero};
use crate::ndarray::{
    Array,
    Array1,
    Array2,
    ArrayD,
    Axis,
    LinalgScalar,
    Dimension,
    RemoveAxis,
};
use crate::{F64CompliantScalar};
use crate::backends::backend::{
    Backend,
    Container,
    TensorAdd,
    TensorSub,
    TensorMul,
    TensorDiv,
    TensorNeg,
    TensorElemInv,
    TensorOpResult,
    Dot,
    Exp,
    Tensor,
    Broadcast,
    ShapeVec,
    Shape,
    ReduceSum,
    ReduceMean,
    FromShapedData,
    MaskCmp,
    Transpose,
};
use crate::common::string_err::err_to_string;

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
        NdArrayCommonRepr {
            data,
            shape,
        }
    }
}

impl<A> FromShapedData for NdArrayCommonRepr<A>
where
    A: TryFrom<f64>,
    <A as TryFrom<f64>>::Error: ToString + Display,
{
    type Error = String;

    fn from_shaped_data(data: Vec<f64>, shape: ShapeVec) -> Result<Self, String> {
        let data = data.into_iter()
            .map(|x| A::try_from(x).map_err(err_to_string))
            .collect::<Result<Vec<A>, String>>()?;
        Ok(NdArrayCommonRepr::new(data, shape))
    }
}

impl<A> Backend for NdArrayBackend<A>
where
    A: LinalgScalar + F64CompliantScalar + Neg<Output = A> + PartialOrd + Display + Debug
{
    type Scalar = A;
    type CommonRepr = NdArrayCommonRepr<A>;

    type Tensor1D = Array1<A>;
    type Tensor2D = Array2<A>;
    type TensorXD = ArrayD<A>;
}

fn check_shapes_equal<T: Shape>(lhs: &T, rhs: &T) -> Result<(), String> {
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    if &lhs_shape != &rhs_shape {
        Err(format!("Incompatible shapes: {:?} and {:?}.", &lhs_shape, &rhs_shape))
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

    fn tensor_add(&self, rhs: &Array<A, D>) -> TensorOpResult<Array<A, D>> {
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

    fn tensor_mul(&self, rhs: &Array<A, D>) -> TensorOpResult<Array<A, D>> {
        check_shapes_equal(self, rhs)?;
        Ok(self * rhs)
    }
}

impl<A, D> TensorNeg for Array<A, D>
where
    A: Copy + LinalgScalar + Neg<Output = A>,
    D: Dimension,
    Self: TensorMul<Self, Output=Self> + Container<Elem = A>,
{
    type Output = Self;

    fn tensor_neg(&self) -> TensorOpResult<Self> {
        self.tensor_mul(&self.same_from_scalar(A::one().neg()))
    }
}

impl<A, D> TensorSub<Array<A, D>> for Array<A, D>
where
    A: Copy + LinalgScalar + Neg<Output = A>,
    D: Dimension,
    Self: TensorMul<Self, Output = Self> + Container<Elem = A>,
{}

impl<A, D> TensorElemInv for Array<A, D>
where
    A: LinalgScalar + PartialEq,
    D: Dimension,
{
    type Output = Array<<A as Div>::Output, D>;

    fn tensor_elem_inv(&self) -> TensorOpResult<Array<<A as Div>::Output, D>> {
        if let Some(_) = self.iter().find(|x| **x == A::zero()) {
            return Err(String::from("Division by zero is forbidden!"));
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

    fn tensor_div(&self, rhs: &Array<A, D>) -> TensorOpResult<<Self as TensorMul<Array<A, D>>>::Output> {
        self.tensor_mul(&rhs.tensor_elem_inv()?)
    }
}

impl<A, D> MaskCmp for Array<A, D>
where
    A: PartialOrd + One + Zero + Clone,
    D: Dimension,
{
    type Mask = Array<A, D>;

    fn mask_lt(&self, x: A) -> TensorOpResult<Self::Mask> {
        Ok(self.map(|a| if *a < x { A::one() } else { A::zero() }))
    }

    fn mask_gt(&self, x: A) -> TensorOpResult<Self::Mask> {
        Ok(self.map(|a| if *a > x { A::one() } else { A::zero() }))
    }

    fn mask_eq(&self, x: A) -> TensorOpResult<Self::Mask> {
        Ok(self.map(|a| if *a == x { A::one() } else { A::zero() }))
    }
}

impl<A: Clone> Transpose for Array2<A> {
    type Output = Array2<A>;

    fn transpose(&self) -> TensorOpResult<Array2<A>> {
        Ok(self.t().to_owned())
    }
}

impl<A> Dot<Array2<A>> for Array2<A>
where
    A: LinalgScalar
{
    type Output = Array2<A>;

    fn dot(&self, rhs: &Array2<A>) -> TensorOpResult<Array2<A>> {
        let self_shape = Shape::shape(self);
        let rhs_shape = Shape::shape(rhs);
        if self_shape[1] != rhs_shape[0] {
            return Err(format!(
                "Incompatible shapes for dot product: {:?} x {:?}",
                self_shape,
                rhs_shape
            ))
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
    fn broadcast(&self, rhs: &Array<A, D::Larger>) -> TensorOpResult<Array<A, D::Larger>> {
        match self.clone().insert_axis(Axis(0)).broadcast(rhs.dim()).map(|x| x.to_owned()) {
            Some(result) => Ok(result),
            None => Err(format!(
                "Cannot broadcast {:?} to {:?}.",
                Shape::shape(self),
                Shape::shape(rhs),
            ))
        }
    }
}

impl<A, D> Shape for Array<A, D>
where
    D: Dimension
{
    fn shape(&self) -> ShapeVec {
        self.shape().iter().cloned().collect::<Vec<usize>>()
    }
}

impl<A, D> ReduceSum for Array<A, D>
where
    A: Clone + Zero + Add<Output=A>,
    D: Dimension + RemoveAxis
{
    type Output = Array<A, D::Smaller>;

    fn reduce_sum(&self, axis: usize) -> TensorOpResult<Self::Output> {
        Ok(self.sum_axis(Axis(axis)))
    }
}

impl<A, D> ReduceMean for Array<A, D>
where
    A: Clone + Zero + One + Add<Output=A> + Div<Output=A>,
    D: Dimension + RemoveAxis
{
    type Output = Array<A, D::Smaller>;

    fn reduce_mean(&self, axis: usize) -> TensorOpResult<Self::Output> {
        Ok(self.mean_axis(Axis(axis)))
    }
}

impl<A, D> TryFrom<Array<A, D>> for NdArrayCommonRepr<A>
where
    Array<A, D>: Shape,
    D: Dimension
{
    type Error = String;

    fn try_from(value: Array<A, D>) -> Result<NdArrayCommonRepr<A>, Self::Error> {
        let shape = <Array<A, D> as Shape>::shape(&value);
        let data = value.into_raw_vec();
        Ok(NdArrayCommonRepr::new(data, shape))
    }
}

impl<A, D> TryInto<Array<A, D>> for NdArrayCommonRepr<A>
where
    D: Dimension
{
    type Error = String;

    fn try_into(self) -> Result<Array<A, D>, Self::Error> {
        Array::<A, _>::from_shape_vec(self.shape, self.data)
            .map_err(|err| err.to_string())?
            .into_dimensionality::<D>()
            .map_err(|err| err.to_string())
    }
}

impl<A, D> FromShapedData for Array<A, D>
where
    A: From<f64>,
    D: Dimension,
{
    type Error = String;

    fn from_shaped_data(data: Vec<f64>, shape: ShapeVec) -> Result<Array<A, D>, String> {
        let vec: Vec<A> = data.into_iter().map(|x| A::from(x)).collect();
        Array::<A, _>::from_shape_vec(shape, vec)
            .map_err(|err| err.to_string())?
            .into_dimensionality::<D>()
            .map_err(|err| err.to_string())
    }
}

impl<A, D> Tensor<A, NdArrayCommonRepr<A>> for Array<A, D>
where
    A: LinalgScalar + Clone + F64CompliantScalar + Neg<Output = A> + PartialOrd,
    D: Dimension,
{}

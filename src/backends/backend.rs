use std::ops::{Add, Sub, Mul, Neg};
use crate::{F64CompliantScalar};

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

pub trait TensorDiv<Rhs = Self> : TensorMul<Rhs>
where
    Rhs: TensorElemInv<Output = Rhs>
{
    fn tensor_div(&self, rhs: &Rhs) -> TensorOpResult<<Self as TensorMul<Rhs>>::Output> {
        self.tensor_mul(&rhs.tensor_elem_inv()?)
    }
}

pub trait Tensor<A>:
    TensorAdd<Self, Output = Self> +
    TensorSub<Self, Output = Self> +
    TensorMul<Self, Output = Self> +
    TensorDiv<Self, Output = Self> +
    TensorElemInv<Output = Self> +
    TensorNeg<Output = Self> +
    Exp +
    SameFromScalar<A, Output = Self> +
    Shape
where
    Self: Sized
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

    fn exp(&self) -> Self;
}

pub trait SameFromScalar<Rhs> {
    type Output;

    fn same_from_scalar(&self, x: Rhs) -> Self::Output;
}

pub trait Broadcast<To>
    where
        Self: Sized
{
    fn broadcast(&self, to: &To) -> TensorOpResult<To>;
}

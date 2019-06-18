use std::ops::{Neg, Div};
use crate::num_traits::identities::{One, Zero};
use crate::ndarray::{Array, Array2, Axis, LinalgScalar, ScalarOperand, Dimension};
use crate::{F64CompliantScalar};
use crate::backends::backend::{
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
    SameFromScalar,
    Broadcast,
    ShapeVec,
    Shape
};

fn check_shapes_equal<T: Shape>(lhs: &T, rhs: &T) -> Result<(), String> {
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    if &lhs_shape != &rhs_shape {
        Err(format!("Incompatible shapes: {:?} and {:?}.", &lhs_shape, &rhs_shape))
    } else {
        Ok(())
    }
}

impl<A, D> SameFromScalar<A> for Array<A, D>
where
    A: Clone,
    D: Dimension,
{
    type Output = Self;

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
    Self: TensorMul<Self, Output=Self> + SameFromScalar<A, Output = Self>,
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
    Self: TensorMul<Self, Output = Self> + SameFromScalar<A, Output = Self>,
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
{}

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

impl<A, D> Broadcast<Array<A, D>> for Array<A, D::Smaller>
where
    A: Clone,
    D: Dimension,
{
    fn broadcast(&self, rhs: &Array<A, D>) -> TensorOpResult<Array<A, D>> {
        match self.clone().insert_axis(Axis(1)).broadcast(rhs.dim()).map(|x| x.to_owned()) {
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

impl<A, D> Tensor<A> for Array<A, D>
where
    A: LinalgScalar + Clone + F64CompliantScalar + Neg<Output = A> + PartialEq,
    D: Dimension,
{}

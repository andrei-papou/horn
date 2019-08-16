use std::cmp::Ordering;

use num_traits::real::Real;

use crate::backends::{
    Abs, Container, IntoScalar, MaskCmp, ReduceSum, Shape, TensorNeg, TensorSub,
};
use crate::common::types::HResult;

pub mod dim2 {
    use super::*;

    const BATCH_AXIS: usize = 0;
    const LABEL_AXIS: usize = 1;

    pub fn accuracy<A, T>(y: &T, y_hat: &T) -> HResult<A>
    where
        A: Real,
        T: Abs<Output = T>
            + Container<Elem = A>
            + TensorSub<T, Output = T>
            + TensorNeg<Output = T>
            + ReduceSum
            + Shape,
        <T as ReduceSum>::Output:
            Container<Elem = A> + MaskCmp<Mask = <T as ReduceSum>::Output> + ReduceSum,
        <<T as ReduceSum>::Output as ReduceSum>::Output: IntoScalar<Output = A>,
    {
        let total: A = y
            .reduce_sum(LABEL_AXIS)?
            .reduce_sum(BATCH_AXIS)?
            .into_scalar()?;
        let correct: A = y
            .tensor_sub(y_hat)?
            .abs()?
            .reduce_sum(LABEL_AXIS)?
            .mask_cmp(A::epsilon(), &Ordering::Less)?
            .reduce_sum(BATCH_AXIS)?
            .into_scalar()?;
        Ok(correct / total)
    }
}

use ndarray::{ArrayBase, Array, Data, Dimension};
use crate::{F64CompliantScalar};
use crate::layers::layer::{Apply, LayerResult};

pub struct Tanh;

impl Tanh {
    pub(crate) fn calculate<A: F64CompliantScalar>(x: A) -> A {
        x.map_f64(|x| (x.exp() - (-x).exp()) / (x.exp() + (-x).exp()))
    }
}

impl<A, S, D> Apply<ArrayBase<S, D>, Array<A, D>> for Tanh
where
    A: F64CompliantScalar + Copy,
    S: Data<Elem = A>,
    D: Dimension,
{
    fn apply(&self, input: &ArrayBase<S, D>) -> LayerResult<Array<A, D>> {
        Ok(input.map(|x| Tanh::calculate(*x)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array};

    #[test]
    fn test_calculate() {
        assert_eq!(Tanh::calculate(0.0), 0.0);
        let result = Tanh::calculate(10.0);
        assert!(result > 0.999 && result < 1.0);
        let result = Tanh::calculate(-10.0);
        assert!(result > -1.0 && result < -0.999);
    }

    #[test]
    fn test_apply() {
        let tanh = Tanh;
        let input = array![[100.0, 0.0], [0.0, 100.0]];
        let expected_output = array![[1.0, 0.0], [0.0, 1.0]];
        let output = tanh.apply(&input);
        assert!(output.is_ok());
        assert_eq!(output.unwrap(), expected_output);
    }
}

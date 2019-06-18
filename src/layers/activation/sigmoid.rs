use ndarray::{ArrayBase, Array, Data, Dimension};
use crate::{F64CompliantScalar};
use crate::layers::layer::{Apply, LayerResult};

pub struct Sigmoid;

impl Sigmoid {
    pub(crate) fn calculate<A: F64CompliantScalar>(x: A) -> A {
        x.map_f64(|x| 1.0 / (1.0 + (-x).exp()))
    }
}

impl<A, S, D> Apply<ArrayBase<S, D>, Array<A, D>> for Sigmoid
where
    A: F64CompliantScalar + Copy,
    S: Data<Elem = A>,
    D: Dimension,
{
    fn apply(&self, input: &ArrayBase<S, D>) -> LayerResult<Array<A, D>> {
        Ok(input.map(|x| Sigmoid::calculate(*x)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array};

    #[test]
    fn test_calculate() {
        assert_eq!(Sigmoid::calculate(0.0), 0.5);
        let result = Sigmoid::calculate(10.0);
        assert!(result > 0.999 && result < 1.0);
        let result = Sigmoid::calculate(-10.0);
        assert!(result > 0.0 && result < 0.001);
    }

    #[test]
    fn test_apply() {
        let sigmoid = Sigmoid;
        let input = array![[1000.0, 0.0], [0.0, 1000.0]];
        let expected_output = array![[1.0, 0.5], [0.5, 1.0]];
        let output = sigmoid.apply(&input);
        assert!(output.is_ok());
        assert_eq!(output.unwrap(), expected_output);
    }
}

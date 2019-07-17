#[cfg(test)]
mod dense_layer {
    use std::convert::TryInto;

    use ndarray::{array, Array1, Array2};

    use crate::backends::NdArrayBackend;
    use crate::common::traits::Name;
    use crate::layers::{Apply, DenseLayer};

    #[test]
    fn test_name() {
        let name = "layer";
        let weights: Array2<f64> = Array2::from_elem((2, 2), 1.0);
        let layer: DenseLayer<NdArrayBackend<_>> =
            DenseLayer::new(String::from(name), weights, None);
        assert_eq!(layer.name(), name);
    }

    #[test]
    fn test_apply_with_bias() {
        let weights: Array2<f64> = array![[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]];
        let bias: Array1<f64> = array![1.0, -1.0, 1.0];
        let input: Array2<f64> = array![[2.0, 5.0], [7.0, 1.0], [4.0, 3.0]];
        let expected_output: Array2<f64> =
            array![[6.0, 1.0, 6.0], [2.0, 6.0, 2.0], [4.0, 3.0, 4.0]];
        let layer =
            DenseLayer::<NdArrayBackend<_>>::new(String::from("layer_1"), weights, Some(bias));
        let output = layer.apply(input.try_into().unwrap());
        assert!(output.is_ok());
        let output: Array2<f64> = output.unwrap().try_into().unwrap();
        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_apply_without_bias() {
        let weights: Array2<f64> = array![[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]];
        let input: Array2<f64> = array![[2.0, 5.0], [7.0, 1.0], [4.0, 3.0]];
        let expected_output: Array2<f64> =
            array![[5.0, 2.0, 5.0], [1.0, 7.0, 1.0], [3.0, 4.0, 3.0]];
        let layer: DenseLayer<NdArrayBackend<_>> =
            DenseLayer::new(String::from("layer_1"), weights, None);
        let output = layer.apply(input.try_into().unwrap());
        assert!(output.is_ok());
        let output: Array2<f64> = output.unwrap().try_into().unwrap();
        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_apply_with_invalid_args() {
        let weights = array![[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]];
        let input = array![[2.0, 5.0, 1.0], [7.0, 1.0, 5.0]];
        let layer: DenseLayer<NdArrayBackend<_>> =
            DenseLayer::new(String::from("layer_1"), weights, None);
        let output = layer.apply(input.try_into().unwrap());
        assert!(output.is_err());
        assert_eq!(
            output.unwrap_err().as_fail().to_string(),
            "Incompatible shapes for dot product: [2, 3] x [2, 3]"
        );
    }
}

#[cfg(test)]
mod activations {
    use std::convert::TryInto;

    use ndarray::{array, Array2};

    use crate::backends::NdArrayBackend;
    use crate::layers::{Apply, Relu, Sigmoid, Softmax, Tanh};

    #[test]
    fn test_sigmoid() {
        let arr: Array2<f64> = array![[0.0, 0.0], [0.0, 0.0]];
        let layer = Sigmoid::<NdArrayBackend<_>>::new(String::from("sigmoid"));
        let output = layer.apply(arr.try_into().unwrap());
        assert!(output.is_ok());
        let output: Array2<f64> = output.unwrap().try_into().unwrap();
        assert_eq!(output, array![[0.5, 0.5], [0.5, 0.5]]);
    }

    #[test]
    fn test_tanh() {
        let arr: Array2<f64> = array![[0.0, 0.0], [0.0, 0.0]];
        let layer = Tanh::<NdArrayBackend<_>>::new(String::from("tanh"));
        let output = layer.apply(arr.try_into().unwrap());
        assert!(output.is_ok());
        let output: Array2<f64> = output.unwrap().try_into().unwrap();
        assert_eq!(output, array![[0.0, 0.0], [0.0, 0.0]]);
    }

    #[test]
    fn test_relu() {
        let arr: Array2<f64> = array![[-4.2, 1.0], [4.3, -2.0]];
        let layer = Relu::<NdArrayBackend<_>>::new(String::from("relu"));
        let output = layer.apply(arr.try_into().unwrap());
        assert!(output.is_ok());
        let output: Array2<f64> = output.unwrap().try_into().unwrap();
        assert_eq!(output, array![[0.0, 1.0], [4.3, 0.0]]);
    }

    #[cfg(test)]
    mod softmax {
        use super::*;

        #[test]
        fn test_with_axis() {
            let arr: Array2<f64> = array![[0.5, 1.5], [3.0, 3.0]];
            let layer = Softmax::<NdArrayBackend<_>>::new(String::from("softmax"), Some(1));
            let output = layer.apply(arr.try_into().unwrap());
            assert!(output.is_ok());
            let output: Array2<f64> = output.unwrap().try_into().unwrap();
            let e11 = 0.5f64.exp();
            let e12 = 1.5f64.exp();
            let e21 = 3.0f64.exp();
            let e22 = 3.0f64.exp();
            let exp_output: Array2<f64> = array![
                [e11 / (e11 + e12), e12 / (e11 + e12)],
                [e21 / (e21 + e22), e22 / (e21 + e22)]
            ];
            assert!(output
                .iter()
                .zip(exp_output.iter())
                .all(|(e1, e2)| (e1 - e2) < 0.00001));
        }

        #[test]
        fn test_without_axis() {
            let arr: Array2<f64> = array![[0.5, 1.5], [3.0, 3.0]];
            let layer = Softmax::<NdArrayBackend<_>>::new(String::from("softmax"), None);
            let output = layer.apply(arr.try_into().unwrap());
            assert!(output.is_ok());
            let output: Array2<f64> = output.unwrap().try_into().unwrap();
            let e11 = 0.5f64.exp();
            let e12 = 1.5f64.exp();
            let e21 = 3.0f64.exp();
            let e22 = 3.0f64.exp();
            let exp_output: Array2<f64> = array![
                [e11 / (e11 + e12), e12 / (e11 + e12)],
                [e21 / (e21 + e22), e22 / (e21 + e22)]
            ];
            assert!(output
                .iter()
                .zip(exp_output.iter())
                .all(|(e1, e2)| (e1 - e2) < 0.00001));
        }
    }
}

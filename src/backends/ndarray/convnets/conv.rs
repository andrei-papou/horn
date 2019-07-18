use std::fmt::Debug;
use std::ops::{Add, Mul};

use ndarray::{stack, Array1, Array3, Array4, ArrayView3, ArrayView4, Axis, ShapeError, Slice};
use num_traits::Zero;

use super::common::pad_array3;
use crate::backends::convnets::{DataFormat, Padding};

fn get_axis_padding(axis_len: usize, kernel_size: usize, stride: usize) -> usize {
    (kernel_size + stride * (axis_len - 1) - axis_len) / 2
}

fn get_conv2d_result_axis_len(n: usize, k: usize, s: usize, p: usize) -> usize {
    (n + 2 * p - k) / s + 1
}

pub(crate) fn conv2d<A>(
    input_batch: &Array4<A>,
    kernels: &Array4<A>,
    bias: &Option<Array1<A>>,
    strides: (usize, usize),
    padding: Padding,
    data_format: DataFormat,
) -> Result<Array4<A>, ShapeError>
where
    A: Debug + Clone + Copy + Add<Output = A> + Mul<A, Output = A> + Zero,
{
    let (h_axis, w_axis): (usize, usize) = match &data_format {
        DataFormat::ChannelsFirst => (1, 2),
        DataFormat::ChannelsLast => (0, 1),
    };

    let batch_results: Vec<Result<Array4<A>, ShapeError>> = input_batch
        .axis_iter(Axis(0))
        .map(|input: ArrayView3<A>| {
            let i_shape = input.shape();
            let h_dim_input = i_shape[h_axis];
            let w_dim_input = i_shape[w_axis];
            let (num_kernels, k_shape) = kernels.shape().split_at(1);
            let res_k = num_kernels[0];
            let h_dim_kernel = k_shape[h_axis];
            let w_dim_kernel = k_shape[w_axis];

            let (hp, wp) = match &padding {
                Padding::Valid => (0, 0),
                Padding::Same => (
                    get_axis_padding(h_dim_input, h_dim_kernel, strides.0),
                    get_axis_padding(w_dim_input, w_dim_kernel, strides.1),
                ),
            };
            let res_h = get_conv2d_result_axis_len(h_dim_input, h_dim_kernel, strides.0, hp);
            let res_w = get_conv2d_result_axis_len(w_dim_input, w_dim_kernel, strides.1, wp);

            let mut kernels_output = vec![A::zero(); res_h * res_w * res_k];

            let input: Array3<A> = match &padding {
                Padding::Valid => input.into_owned(),
                Padding::Same => pad_array3(&input, &(hp, wp, 0)),
            };

            let get_idx = |ki: usize, hi: usize, wi: usize| -> usize {
                match &data_format {
                    DataFormat::ChannelsFirst => ki * res_h * res_w + hi * res_w + wi,
                    DataFormat::ChannelsLast => hi * res_w * res_k + wi * res_k + ki,
                }
            };

            for (hi, hr) in (0..(h_dim_input - h_dim_kernel + 1))
                .step_by(strides.0)
                .enumerate()
            {
                for (wi, wr) in (0..(w_dim_input - w_dim_kernel + 1))
                    .step_by(strides.1)
                    .enumerate()
                {
                    let window = input
                        .slice_axis(
                            Axis(h_axis),
                            Slice::new(hr as isize, Some((hr + h_dim_kernel) as isize), 1),
                        )
                        .slice_axis(
                            Axis(w_axis),
                            Slice::new(wr as isize, Some((wr + w_dim_kernel) as isize), 1),
                        )
                        .into_owned();
                    kernels.axis_iter(Axis(0)).enumerate().for_each(
                        |(ki, kernel): (usize, ArrayView3<A>)| {
                            let b = bias.as_ref().map(|b| b[ki]).unwrap_or(A::zero());
                            kernels_output[get_idx(ki, hi, wi)] = (&window * &kernel).sum() + b;
                        },
                    );
                }
            }

            let kernels_output_shape: (usize, usize, usize, usize) = match &data_format {
                DataFormat::ChannelsFirst => (1usize, res_k, res_h, res_w),
                DataFormat::ChannelsLast => (1usize, res_h, res_w, res_k),
            };
            Array4::<A>::from_shape_vec(kernels_output_shape, kernels_output)
        })
        .collect();
    let batch_results = batch_results
        .into_iter()
        .collect::<Result<Vec<Array4<A>>, ShapeError>>()?;

    Ok(stack(
        Axis(0),
        batch_results
            .iter()
            .map(|a| a.view())
            .collect::<Vec<ArrayView4<A>>>()
            .as_slice(),
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn get_channels_last_data() -> (Array4<f64>, Array4<f64>, Option<Array1<f64>>) {
        let x1: Array3<f64> = array![
            [[1.0, 2.0, 1.0], [1.0, 3.0, 2.0], [2.0, 3.0, 1.0], [2.0, 1.0, 5.0], [2.0, 3.0, 4.0]],
            [[2.0, 1.0, 0.0], [2.0, 4.0, 1.0], [5.0, 6.0, 0.0], [4.0, 2.0, 3.0], [1.0, 6.0, 7.0]],
            [[5.0, 3.0, 2.0], [7.0, 1.0, 6.0], [4.0, 2.0, 7.0], [2.0, 1.0, 5.0], [1.0, 5.0, 5.0]],
            [[6.0, 1.0, 3.0], [1.0, 3.0, 2.0], [2.0, 5.0, 4.0], [1.0, 4.0, 2.0], [2.0, 3.0, 4.0]],
            [[7.0, 2.0, 3.0], [2.0, 4.0, 3.0], [5.0, 6.0, 1.0], [1.0, 1.0, 2.0], [1.0, 1.0, 2.0]]
        ];
        let x2: Array3<f64> = array![
            [[1.0, 2.0, 1.0], [1.0, 3.0, 2.0], [2.0, 3.0, 1.0], [2.0, 1.0, 5.0], [2.0, 3.0, 4.0]],
            [[2.0, 1.0, 0.0], [2.0, 4.0, 1.0], [5.0, 6.0, 0.0], [4.0, 2.0, 3.0], [1.0, 6.0, 7.0]],
            [[5.0, 3.0, 2.0], [7.0, 1.0, 6.0], [4.0, 2.0, 7.0], [2.0, 1.0, 5.0], [1.0, 5.0, 5.0]],
            [[6.0, 1.0, 3.0], [1.0, 3.0, 2.0], [2.0, 5.0, 4.0], [1.0, 4.0, 2.0], [2.0, 3.0, 4.0]],
            [[7.0, 2.0, 3.0], [2.0, 4.0, 3.0], [5.0, 6.0, 1.0], [1.0, 1.0, 2.0], [1.0, 1.0, 2.0]]
        ];
        let x1 = x1.insert_axis(Axis(0));
        let x2 = x2.insert_axis(Axis(0));
        let input = stack(Axis(0), &[x1.view(), x2.view()]).unwrap();

        let k1: Array3<f64> = array![
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        ];
        let k2: Array3<f64> = array![
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        ];
        let k1 = k1.insert_axis(Axis(0));
        let k2 = k2.insert_axis(Axis(0));
        let kernels = stack(Axis(0), &[k1.view(), k2.view()]).unwrap();
        let bias: Option<Array1<f64>> = Some(array![1.0, 2.0]);

        (input, kernels, bias)
    }

    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn get_channel_first_data() -> (Array4<f64>, Array4<f64>, Option<Array1<f64>>) {
        let x1: Array3<f64> = array![
            [[1.0, 1.0, 2.0, 2.0, 2.0],
             [2.0, 2.0, 5.0, 4.0, 1.0],
             [5.0, 7.0, 4.0, 2.0, 1.0],
             [6.0, 1.0, 2.0, 1.0, 2.0],
             [7.0, 2.0, 5.0, 1.0, 1.0]],
            [[2.0, 3.0, 3.0, 1.0, 3.0],
             [1.0, 4.0, 6.0, 2.0, 6.0],
             [3.0, 1.0, 2.0, 1.0, 5.0],
             [1.0, 3.0, 5.0, 4.0, 3.0],
             [2.0, 4.0, 6.0, 1.0, 1.0]],
            [[1.0, 2.0, 1.0, 5.0, 4.0],
             [0.0, 1.0, 0.0, 3.0, 7.0],
             [2.0, 6.0, 7.0, 5.0, 5.0],
             [3.0, 2.0, 4.0, 2.0, 4.0],
             [3.0, 3.0, 1.0, 2.0, 2.0]]
        ];
        let x2: Array3<f64> = array![
            [[1.0, 1.0, 2.0, 2.0, 2.0],
             [2.0, 2.0, 5.0, 4.0, 1.0],
             [5.0, 7.0, 4.0, 2.0, 1.0],
             [6.0, 1.0, 2.0, 1.0, 2.0],
             [7.0, 2.0, 5.0, 1.0, 1.0]],
            [[2.0, 3.0, 3.0, 1.0, 3.0],
             [1.0, 4.0, 6.0, 2.0, 6.0],
             [3.0, 1.0, 2.0, 1.0, 5.0],
             [1.0, 3.0, 5.0, 4.0, 3.0],
             [2.0, 4.0, 6.0, 1.0, 1.0]],
            [[1.0, 2.0, 1.0, 5.0, 4.0],
             [0.0, 1.0, 0.0, 3.0, 7.0],
             [2.0, 6.0, 7.0, 5.0, 5.0],
             [3.0, 2.0, 4.0, 2.0, 4.0],
             [3.0, 3.0, 1.0, 2.0, 2.0]]
        ];
        let x1 = x1.insert_axis(Axis(0));
        let x2 = x2.insert_axis(Axis(0));
        let input = stack(Axis(0), &[x1.view(), x2.view()]).unwrap();
        let k1: Array3<f64> = array![
            [[1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0],
             [0.0, 1.0, 0.0],
             [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0],
             [0.0, 1.0, 0.0],
             [1.0, 0.0, 0.0]]
        ];
        let k2: Array3<f64> = array![
            [[0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0],
             [1.0, 1.0, 1.0],
             [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0]]
        ];
        let k1 = k1.insert_axis(Axis(0));
        let k2 = k2.insert_axis(Axis(0));
        let kernels = stack(Axis(0), &[k1.view(), k2.view()]).unwrap();
        let bias: Option<Array1<f64>> = Some(array![1.0, 2.0]);

        (input, kernels, bias)
    }

    #[test]
    fn test_pad_array3() {
        let arr: Array3<f64> = array![[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]];
        let output = pad_array3(&arr, &(1, 1, 0));
        let exp_output: Array3<f64> = array![
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [0.0, 0.0]],
            [[0.0, 0.0], [3.0, 3.0], [4.0, 4.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        ];
        assert_eq!(output, exp_output);
    }

    #[cfg_attr(rustfmt, rustfmt_skip)]
    #[test]
    fn test_conv2d() {
        let (input, kernels, bias) = get_channels_last_data();

        let exp_o1 = array![
            [[20.0, 20.0], [31.0, 24.0], [26.0, 25.0]],
            [[29.0, 28.0], [33.0, 28.0], [33.0, 23.0]],
            [[32.0, 19.0], [36.0, 28.0], [21.0, 19.0]],
        ];
        let output = conv2d(
            &input,
            &kernels,
            &bias,
            (1, 1),
            Padding::Valid,
            DataFormat::ChannelsLast
        ).unwrap();
        assert_eq!(output.shape(), &[2, 3, 3, 2]);
        let act_o1 = output.index_axis(Axis(0), 0);
        assert_eq!(exp_o1, act_o1);

        // (2, 2) strides (symmetric)
        let output = conv2d(
            &input,
            &kernels,
            &bias,
            (2, 2),
            Padding::Valid,
            DataFormat::ChannelsLast,
        ).unwrap();
        assert_eq!(output.shape(), &[2, 2, 2, 2]);
        let exp_o1 = array![
            [[20.0, 20.0], [26.0, 25.0]],
            [[32.0, 19.0], [21.0, 19.0]],
        ];
        let act_o1 = output.index_axis(Axis(0), 0);
        assert_eq!(exp_o1, act_o1);

        // (2, 1) strides (non-symmetric)
        let output = conv2d(
            &input,
            &kernels,
            &bias,
            (2, 1),
            Padding::Valid,
            DataFormat::ChannelsLast,
        ).unwrap();
        assert_eq!(output.shape(), &[2, 2, 3, 2]);
        let exp_o1 = array![
            [[20.0, 20.0], [31.0, 24.0], [26.0, 25.0]],
            [[32.0, 19.0], [36.0, 28.0], [21.0, 19.0]],
        ];
        let act_o1 = output.index_axis(Axis(0), 0);
        assert_eq!(exp_o1, act_o1);

        // Same padding
        let output = conv2d(
            &input,
            &kernels,
            &bias,
            (1, 1),
            Padding::Same,
            DataFormat::ChannelsLast,
        ).unwrap();
        assert_eq!(output.shape(), &[2, 5, 5, 2]);

        let (input, kernels, bias) = get_channel_first_data();
        let exp_o1 = array![
            [[20.0, 31.0, 26.0],
             [29.0, 33.0, 33.0],
             [32.0, 36.0, 21.0]],
            [[20.0, 24.0, 25.0],
             [28.0, 28.0, 23.0],
             [19.0, 28.0, 19.0]]
        ];
        let output = conv2d(
            &input,
            &kernels,
            &bias,
            (1, 1),
            Padding::Valid,
            DataFormat::ChannelsFirst,
        ).unwrap();
        assert_eq!(output.shape(), &[2, 2, 3, 3]);
        let act_o1 = output.index_axis(Axis(0), 0);
        assert_eq!(exp_o1, act_o1);
    }
}

use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, Div, Mul, Neg};

use ndarray::{Array3, Array4, ArrayView2, ArrayView3, Axis, ShapeError, Slice};
use num_traits::{real::Real, One, Zero};

use super::common::{build_indexer_2d, join_first_axis, pad_array3};
use crate::backends::convnets::{
    get_axis_padding, get_conv2d_result_axis_len, DataFormat, Padding, Pool2, Stride2,
};

fn pool2d<A, F>(
    input_batch: &Array4<A>,
    pool_window: Pool2,
    strides: Stride2,
    padding: Padding,
    data_format: DataFormat,
    func: F,
) -> Result<Array4<A>, ShapeError>
where
    A: Debug
        + Clone
        + Copy
        + Add<Output = A>
        + Mul<A, Output = A>
        + Div<A, Output = A>
        + Zero
        + One
        + Neg<Output = A>
        + Real,
    F: Fn(&ArrayView2<A>) -> A,
{
    let (h_axis, w_axis, c_axis): (usize, usize, usize) = match &data_format {
        DataFormat::ChannelsFirst => (1, 2, 0),
        DataFormat::ChannelsLast => (0, 1, 2),
    };

    let batch_results: Vec<Result<Array4<A>, ShapeError>> = input_batch
        .axis_iter(Axis(0))
        .map(|input: ArrayView3<A>| {
            let i_shape = input.shape();
            let h_dim_input = i_shape[h_axis];
            let w_dim_input = i_shape[w_axis];
            let res_c = i_shape[c_axis];

            let res_h = get_conv2d_result_axis_len(h_dim_input, pool_window.0, strides.0, &padding);
            let res_w = get_conv2d_result_axis_len(w_dim_input, pool_window.1, strides.1, &padding);

            let mut kernels_output = vec![A::zero(); res_h * res_w * res_c];

            let input: Array3<A> = match &padding {
                Padding::Valid => input.into_owned(),
                Padding::Same => {
                    let hp = get_axis_padding(h_dim_input, pool_window.0, strides.0);
                    let wp = get_axis_padding(w_dim_input, pool_window.1, strides.1);
                    pad_array3(
                        &input,
                        &(hp.0, hp.1, wp.0, wp.1),
                        &data_format,
                        A::min_value(),
                    )
                }
            };
            let i_shape = input.shape();
            let h_dim_input = i_shape[h_axis];
            let w_dim_input = i_shape[w_axis];

            let get_idx = build_indexer_2d(res_c, res_h, res_w, &data_format);

            for (hi, hr) in (0..(h_dim_input - pool_window.0 + 1))
                .step_by(strides.0)
                .enumerate()
            {
                for (wi, wr) in (0..(w_dim_input - pool_window.1 + 1))
                    .step_by(strides.1)
                    .enumerate()
                {
                    let window = input
                        .slice_axis(
                            Axis(h_axis),
                            Slice::new(hr as isize, Some((hr + pool_window.0) as isize), 1),
                        )
                        .slice_axis(
                            Axis(w_axis),
                            Slice::new(wr as isize, Some((wr + pool_window.1) as isize), 1),
                        )
                        .into_owned();

                    window
                        .axis_iter(Axis(c_axis))
                        .enumerate()
                        .for_each(|(ci, window)| {
                            kernels_output[get_idx(ci, hi, wi)] = func(&window);
                        });
                }
            }

            let kernels_output_shape: (usize, usize, usize, usize) = match &data_format {
                DataFormat::ChannelsFirst => (1usize, res_c, res_h, res_w),
                DataFormat::ChannelsLast => (1usize, res_h, res_w, res_c),
            };
            Array4::<A>::from_shape_vec(kernels_output_shape, kernels_output)
        })
        .collect();
    let batch_results = batch_results
        .into_iter()
        .collect::<Result<Vec<Array4<A>>, ShapeError>>()?;

    Ok(join_first_axis(batch_results)?)
}

pub(crate) fn avg_pool2d<A>(
    input: &Array4<A>,
    pool_window: Pool2,
    strides: Stride2,
    padding: Padding,
    data_format: DataFormat,
) -> Result<Array4<A>, ShapeError>
where
    A: Debug
        + Clone
        + Copy
        + Add<A, Output = A>
        + Mul<A, Output = A>
        + Zero
        + Div<A, Output = A>
        + Neg<Output = A>
        + One
        + Sum<A>
        + PartialEq<A>
        + Real,
{
    let avg = |x: &ArrayView2<A>| -> A {
        let skip_val = A::min_value();
        let mut sum = A::zero();
        let mut len = A::zero();
        x.iter().for_each(|x| {
            let x_val = *x;
            if x_val != skip_val {
                sum = sum + *x;
                len = len + A::one();
            }
        });
        sum / len
    };
    pool2d(input, pool_window, strides, padding, data_format, avg)
}

pub(crate) fn max_pool2d<A>(
    input: &Array4<A>,
    pool_window: Pool2,
    strides: Stride2,
    padding: Padding,
    data_format: DataFormat,
) -> Result<Array4<A>, ShapeError>
where
    A: Debug
        + Clone
        + Copy
        + Add<Output = A>
        + Mul<A, Output = A>
        + Neg<Output = A>
        + Div<A, Output = A>
        + One
        + Zero
        + Real,
{
    let max = |x: &ArrayView2<A>| -> A {
        x.iter()
            .fold(A::min_value(), |prev, x: &A| A::max(prev, *x))
    };
    pool2d(input, pool_window, strides, padding, data_format, max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::ndarray::convnets::common::{join_new_axis, test_utils};
    use ndarray::array;

    #[cfg(test)]
    mod max_pool2d {
        use super::*;

        #[cfg_attr(rustfmt, rustfmt_skip)]
        #[test]
        fn test_channel_last_stride_1() {
            let input = test_utils::get_input_channel_last();
            let output = max_pool2d(
                &input,
                (2, 2),
                (1, 1),
                Padding::Valid,
                DataFormat::ChannelsLast
            ).unwrap();
            let exp_output_1: Array3<f64> = array![
                [[2., 4., 2.], [5., 6., 2.], [5., 6., 5.], [4., 6., 7.]],
                [[7., 4., 6.], [7., 6., 7.], [5., 6., 7.], [4., 6., 7.]],
                [[7., 3., 6.], [7., 5., 7.], [4., 5., 7.], [2., 5., 5.]],
                [[7., 4., 3.], [5., 6., 4.], [5., 6., 4.], [2., 4., 4.]]
            ];
            let exp_output_2: Array3<f64> = array![
                [[5., 6., 4.], [5., 6., 2.], [3., 5., 2.], [5., 7., 2.]],
                [[5., 3., 4.], [7., 3., 4.], [7., 7., 6.], [4., 7., 6.]],
                [[4., 3., 7.], [7., 7., 4.], [7., 7., 6.], [4., 7., 6.]],
                [[7., 5., 7.], [5., 7., 5.], [4., 7., 5.], [6., 6., 5.]]
            ];
            let exp_output = join_new_axis(vec![exp_output_1, exp_output_2]).unwrap();
            assert_eq!(output.shape(), &[2, 4, 4, 3]);
            assert_eq!(output, exp_output);
        }

        #[cfg_attr(rustfmt, rustfmt_skip)]
        #[test]
        fn test_channel_last_stride_2() {
            let input = test_utils::get_input_channel_last();
            let output = max_pool2d(
                &input,
                (2, 2),
                (2, 2),
                Padding::Valid,
                DataFormat::ChannelsLast
            ).unwrap();
            let exp_output_1: Array3<f64> = array![
                [[2., 4., 2.], [5., 6., 5.]],
                [[7., 3., 6.], [4., 5., 7.]]
            ];
            let exp_output_2: Array3<f64> = array![
                [[5., 6., 4.], [3., 5., 2.]],
                [[4., 3., 7.], [7., 7., 6.]]
            ];
            let exp_output = join_new_axis(vec![exp_output_1, exp_output_2]).unwrap();
            assert_eq!(output.shape(), &[2, 2, 2, 3]);
            assert_eq!(output, exp_output);
        }

        #[cfg_attr(rustfmt, rustfmt_skip)]
        #[test]
        fn test_channel_last_stride_2_1() {
            let input = test_utils::get_input_channel_last();
            let output = max_pool2d(
                &input,
                (2, 2),
                (2, 1),
                Padding::Valid,
                DataFormat::ChannelsLast
            ).unwrap();
            let exp_output_1: Array3<f64> = array![
                [[2., 4., 2.], [5., 6., 2.], [5., 6., 5.], [4., 6., 7.]],
                [[7., 3., 6.], [7., 5., 7.], [4., 5., 7.], [2., 5., 5.]]
            ];
            let exp_output_2: Array3<f64> = array![
                [[5., 6., 4.], [5., 6., 2.], [3., 5., 2.], [5., 7., 2.]],
                [[4., 3., 7.], [7., 7., 4.], [7., 7., 6.], [4., 7., 6.]]
            ];
            let exp_output = join_new_axis(vec![exp_output_1, exp_output_2]).unwrap();
            assert_eq!(output.shape(), &[2, 2, 4, 3]);
            assert_eq!(output, exp_output);
         }

        #[cfg_attr(rustfmt, rustfmt_skip)]
        #[test]
        fn test_channel_last_same_padding() {
            let input = test_utils::get_input_channel_last();
            let output = max_pool2d(
                &input,
                (2, 2),
                (2, 2),
                Padding::Same,
                DataFormat::ChannelsLast
            ).unwrap();
            let exp_output_1: Array3<f64> = array![
                [[2., 4., 2.], [5., 6., 5.], [2., 6., 7.]],
                [[7., 3., 6.], [4., 5., 7.], [2., 5., 5.]],
                [[7., 4., 3.], [5., 6., 2.], [1., 1., 2.]]
            ];
            let exp_output_2: Array3<f64> = array![
                [[5., 6., 4.], [3., 5., 2.], [5., 7., 1.]],
                [[4., 3., 7.], [7., 7., 6.], [3., 5., 5.]],
                [[7., 5., 5.], [4., 6., 5.], [6., 3., 1.]]
            ];
            let exp_output = join_new_axis(vec![exp_output_1, exp_output_2]).unwrap();
            assert_eq!(output.shape(), &[2, 3, 3, 3]);
            assert_eq!(output, exp_output);
        }

        #[cfg_attr(rustfmt, rustfmt_skip)]
        #[test]
        fn test_channel_first_stride_1() {
            let input = test_utils::get_input_channel_first();
            let output = max_pool2d(
                &input,
                (2, 2),
                (1, 1),
                Padding::Valid,
                DataFormat::ChannelsFirst
            ).unwrap();
            let exp_output_1: Array3<f64> = array![
                [[2., 5., 5., 4.], [7., 7., 5., 4.], [7., 7., 4., 2.], [7., 5., 5., 2.]],
                [[4., 6., 6., 6.], [4., 6., 6., 6.], [3., 5., 5., 5.], [4., 6., 6., 4.]],
                [[2., 2., 5., 7.], [6., 7., 7., 7.], [6., 7., 7., 5.], [3., 4., 4., 4.]]
            ];
            let exp_output_2: Array3<f64> = array![
                [[4., 4., 5., 6.], [6., 4., 4., 4.], [6., 5., 5., 4.], [4., 5., 9., 9.]],
                [[4., 4., 4., 6.], [4., 5., 5., 6.], [7., 5., 5., 5.], [7., 6., 6., 2.]],
                [[3., 3., 5., 7.], [6., 6., 8., 9.], [7., 6., 8., 9.], [7., 5., 5., 5.]]
            ];
            let exp_output = join_new_axis(vec![exp_output_1, exp_output_2]).unwrap();
            assert_eq!(output.shape(), &[2, 3, 4, 4]);
            assert_eq!(output, exp_output);
        }

        #[cfg_attr(rustfmt, rustfmt_skip)]
        #[test]
        fn test_channel_first_stride_2() {
            let input = test_utils::get_input_channel_first();
            let output = max_pool2d(
                &input,
                (2, 2),
                (2, 2),
                Padding::Valid,
                DataFormat::ChannelsFirst
            ).unwrap();
            let exp_output_1: Array3<f64> = array![
                [[2., 5.], [7., 4.]],
                [[4., 6.], [3., 5.]],
                [[2., 5.], [6., 7.]]
            ];
            let exp_output_2: Array3<f64> = array![
                [[4., 5.], [6., 5.]],
                [[4., 4.], [7., 5.]],
                [[3., 5.], [7., 8.]]
            ];
            let exp_output = join_new_axis(vec![exp_output_1, exp_output_2]).unwrap();
            assert_eq!(output.shape(), &[2, 3, 2, 2]);
            assert_eq!(output, exp_output);
        }

        #[cfg_attr(rustfmt, rustfmt_skip)]
        #[test]
        fn test_channel_first_stride_2_1() {
            let input = test_utils::get_input_channel_first();
            let output = max_pool2d(
                &input,
                (2, 2),
                (2, 1),
                Padding::Valid,
                DataFormat::ChannelsFirst,
            ).unwrap();
            let exp_output_1: Array3<f64> = array![
                [[2., 5., 5., 4.], [7., 7., 4., 2.]],
                [[4., 6., 6., 6.], [3., 5., 5., 5.]],
                [[2., 2., 5., 7.], [6., 7., 7., 5.]]
            ];
            let exp_output_2: Array3<f64> = array![
                [[4., 4., 5., 6.], [6., 5., 5., 4.]],
                [[4., 4., 4., 6.], [7., 5., 5., 5.]],
                [[3., 3., 5., 7.], [7., 6., 8., 9.]]
            ];
            let exp_output = join_new_axis(vec![exp_output_1, exp_output_2]).unwrap();
            assert_eq!(output.shape(), &[2, 3, 2, 4]);
            assert_eq!(output, exp_output);
        }

        #[cfg_attr(rustfmt, rustfmt_skip)]
        #[test]
        fn test_channel_first_same_padding() {
            let input = test_utils::get_input_channel_first();
            let output = max_pool2d(
                &input,
                (2, 2),
                (2, 2),
                Padding::Same,
                DataFormat::ChannelsFirst
            ).unwrap();
            let exp_output_1: Array3<f64> = array![
                [[2., 5., 2.], [7., 4., 2.], [7., 5., 1.]],
                [[4., 6., 6.], [3., 5., 5.], [4., 6., 1.]],
                [[2., 5., 7.], [6., 7., 5.], [3., 2., 2.]]
            ];
            let exp_output_2: Array3<f64> = array![
                [[4., 5., 6.], [6., 5., 1.], [3., 9., 7.]],
                [[4., 4., 6.], [7., 5., 5.], [2., 6., 1.]],
                [[3., 5., 7.], [7., 8., 9.], [5., 5., 2.]]
            ];
            let exp_output = join_new_axis(vec![exp_output_1, exp_output_2]).unwrap();
            assert_eq!(output.shape(), &[2, 3, 3, 3]);
            assert_eq!(output, exp_output);
        }
    }

    #[cfg(test)]
    mod avg_pool2d {
        use super::*;

        #[cfg_attr(rustfmt, rustfmt_skip)]
        #[test]
        fn test_channel_last_stride_1() {
            let input = test_utils::get_input_channel_last();
            let output = avg_pool2d(
                &input,
                (2, 2),
                (1, 1),
                Padding::Valid,
                DataFormat::ChannelsLast
            ).unwrap();
            let exp_output_1: Array3<f64> = array![
                [[1.5 , 2.5 , 1.  ], [2.5 , 4.  , 1.  ], [3.25, 3.  , 2.25], [2.25, 3.  , 4.75]],
                [[4.  , 2.25, 2.25], [4.5 , 3.25, 3.5 ], [3.75, 2.75, 3.75], [2.  , 3.5 , 5.  ]],
                [[4.75, 2.  , 3.25], [3.5 , 2.75, 4.75], [2.25, 3.  , 4.5 ], [1.5 , 3.25, 4.  ]],
                [[4.  , 2.5 , 2.75], [2.5 , 4.5 , 2.5 ], [2.25, 4.  , 2.25], [1.25, 2.25, 2.5 ]]
            ];
            let exp_output_2: Array3<f64> = array![
                [[4.  , 2.5 , 2.5 ], [2.5 , 3.25, 1.5 ], [1.25, 3.25, 1.5 ], [2.75, 4.  , 1.25]],
                [[3.  , 2.  , 3.5 ], [3.5 , 1.5 , 2.25], [3.75, 3.25, 2.75], [2.5 , 5.75, 3.5 ]],
                [[2.25, 2.5 , 4.  ], [3.5 , 3.25, 2.5 ], [3.5 , 4.5 , 3.75], [2.5 , 4.75, 5.  ]],
                [[4.5 , 3.75, 3.5 ], [3.  , 3.75, 3.5 ], [2.25, 4.25, 4.  ], [3.75, 4.  , 3.5 ]]
            ];
            let exp_output = join_new_axis(vec![exp_output_1, exp_output_2]).unwrap();
            assert_eq!(output.shape(), &[2, 4, 4, 3]);
            assert_eq!(output, exp_output);
        }

        #[cfg_attr(rustfmt, rustfmt_skip)]
        #[test]
        fn test_channel_last_stride_2() {
            let input = test_utils::get_input_channel_last();
            let output = avg_pool2d(
                &input,
                (2, 2),
                (2, 2),
                Padding::Valid,
                DataFormat::ChannelsLast
            ).unwrap();
            let exp_output_1: Array3<f64> = array![
                [[1.5 , 2.5 , 1.  ], [3.25, 3.  , 2.25]],
                [[4.75, 2.  , 3.25], [2.25, 3.  , 4.5 ]]
            ];
            let exp_output_2: Array3<f64> = array![
                [[4.  , 2.5 , 2.5 ], [1.25, 3.25, 1.5 ]],
                [[2.25, 2.5 , 4.  ], [3.5 , 4.5 , 3.75]]
            ];
            let exp_output = join_new_axis(vec![exp_output_1, exp_output_2]).unwrap();
            assert_eq!(output.shape(), &[2, 2, 2, 3]);
            assert_eq!(output, exp_output);
        }

        #[cfg_attr(rustfmt, rustfmt_skip)]
        #[test]
        fn test_channel_last_stride_2_1() {
            let input = test_utils::get_input_channel_last();
            let output = avg_pool2d(
                &input,
                (2, 2),
                (2, 1),
                Padding::Valid,
                DataFormat::ChannelsLast
            ).unwrap();
            let exp_output_1: Array3<f64> = array![
                [[1.5 , 2.5 , 1.  ], [2.5 , 4.  , 1.  ], [3.25, 3.  , 2.25], [2.25, 3.  , 4.75]],
                [[4.75, 2.  , 3.25], [3.5 , 2.75, 4.75], [2.25, 3.  , 4.5 ], [1.5 , 3.25, 4.  ]]
            ];
            let exp_output_2: Array3<f64> = array![
                [[4.  , 2.5 , 2.5 ], [2.5 , 3.25, 1.5 ], [1.25, 3.25, 1.5 ], [2.75, 4.  , 1.25]],
                [[2.25, 2.5 , 4.  ], [3.5 , 3.25, 2.5 ], [3.5 , 4.5 , 3.75], [2.5 , 4.75, 5.  ]]
            ];
            let exp_output = join_new_axis(vec![exp_output_1, exp_output_2]).unwrap();
            assert_eq!(output.shape(), &[2, 2, 4, 3]);
            assert_eq!(output, exp_output);
        }

        #[cfg_attr(rustfmt, rustfmt_skip)]
        #[test]
        fn test_channel_last_same_padding() {
            let input = test_utils::get_input_channel_last();
            let output = avg_pool2d(
                &input,
                (2, 2),
                (2, 2),
                Padding::Same,
                DataFormat::ChannelsLast
            ).unwrap();
            let exp_output_1: Array3<f64> = array![
                [[1.5 , 2.5 , 1.  ], [3.25, 3.  , 2.25], [1.5 , 4.5 , 5.5 ]],
                [[4.75, 2.  , 3.25], [2.25, 3.  , 4.5 ], [1.5 , 4.  , 4.5 ]],
                [[4.5 , 3.  , 3.  ], [3.  , 3.5 , 1.5 ], [1.  , 1.  , 2.  ]]
            ];
            let exp_output_2: Array3<f64> = array![
                [[4.  , 2.5 , 2.5 ], [1.25, 3.25, 1.5 ], [3.5 , 4.5 , 1.  ]],
                [[2.25, 2.5 , 4.  ], [3.5 , 4.5 , 3.75], [2.  , 4.5 , 5.  ]],
                [[6.  , 5.  , 3.  ], [3.  , 3.5 , 4.5 ], [6.  , 3.  , 1.  ]]
            ];
            let exp_output = join_new_axis(vec![exp_output_1, exp_output_2]).unwrap();
            assert_eq!(output.shape(), &[2, 3, 3, 3]);
            assert_eq!(output, exp_output);
        }

        #[cfg_attr(rustfmt, rustfmt_skip)]
        #[test]
        fn test_channel_first_stride_1() {
            let input = test_utils::get_input_channel_first();
            let output = avg_pool2d(
                &input,
                (2, 2),
                (1, 1),
                Padding::Valid,
                DataFormat::ChannelsFirst
            ).unwrap();
            let exp_output_1: Array3<f64> = array![
                [[1.5 , 2.5 , 3.25, 2.25], [4.  , 4.5 , 3.75, 2.  ],
                 [4.75, 3.5 , 2.25, 1.5 ], [4.  , 2.5 , 2.25, 1.25]],
                [[2.5 , 4.  , 3.  , 3.  ], [2.25, 3.25, 2.75, 3.5 ],
                 [2.  , 2.75, 3.  , 3.25], [2.5 , 4.5 , 4.  , 2.25]],
                [[1.  , 1.  , 2.25, 4.75], [2.25, 3.5 , 3.75, 5.  ],
                 [3.25, 4.75, 4.5 , 4.  ], [2.75, 2.5 , 2.25, 2.5 ]]
            ];
            let exp_output_2: Array3<f64> = array![
                [[2.75, 3.5 , 4.  , 4.  ], [3.5 , 3.25, 3.25, 2.  ],
                 [3.75, 3.5 , 3.75, 2.  ], [2.75, 3.75, 5.5 , 5.25]],
                [[2.5 , 3.  , 2.  , 3.  ], [3.  , 4.25, 3.  , 3.5 ],
                 [4.5 , 3.75, 2.5 , 2.25], [3.75, 3.5 , 2.75, 1.25]],
                [[1.5 , 2.  , 2.25, 3.75], [2.5 , 2.75, 2.5 , 6.  ],
                 [3.75, 2.75, 3.5 , 5.75], [3.5 , 3.  , 3.25, 3.25]]
            ];
            let exp_output = join_new_axis(vec![exp_output_1, exp_output_2]).unwrap();
            assert_eq!(output.shape(), &[2, 3, 4, 4]);
            assert_eq!(output, exp_output);
        }

        #[cfg_attr(rustfmt, rustfmt_skip)]
        #[test]
        fn test_channel_first_stride_2() {
            let input = test_utils::get_input_channel_first();
            let output = avg_pool2d(
                &input,
                (2, 2),
                (2, 2),
                Padding::Valid,
                DataFormat::ChannelsFirst
            ).unwrap();
            let exp_output_1: Array3<f64> = array![
                [[1.5 , 3.25], [4.75, 2.25]],
                [[2.5 , 3.  ], [2.  , 3.  ]],
                [[1.  , 2.25], [3.25, 4.5 ]]
            ];
            let exp_output_2: Array3<f64> = array![
                [[2.75, 4.  ], [3.75, 3.75]],
                [[2.5 , 2.  ], [4.5 , 2.5 ]],
                [[1.5 , 2.25], [3.75, 3.5 ]]
            ];
            let exp_output = join_new_axis(vec![exp_output_1, exp_output_2]).unwrap();
            assert_eq!(output.shape(), &[2, 3, 2, 2]);
            assert_eq!(output, exp_output);
        }

        #[cfg_attr(rustfmt, rustfmt_skip)]
        #[test]
        fn test_channel_first_stride_2_1() {
            let input = test_utils::get_input_channel_first();
            let output = avg_pool2d(
                &input,
                (2, 2),
                (2, 1),
                Padding::Valid,
                DataFormat::ChannelsFirst,
            ).unwrap();
            let exp_output_1: Array3<f64> = array![
                [[1.5 , 2.5 , 3.25, 2.25], [4.75, 3.5 , 2.25, 1.5 ]],
                [[2.5 , 4.  , 3.  , 3.  ], [2.  , 2.75, 3.  , 3.25]],
                [[1.  , 1.  , 2.25, 4.75], [3.25, 4.75, 4.5 , 4.  ]]
            ];
            let exp_output_2: Array3<f64> = array![
                [[2.75, 3.5 , 4.  , 4.  ], [3.75, 3.5 , 3.75, 2.  ]],
                [[2.5 , 3.  , 2.  , 3.  ], [4.5 , 3.75, 2.5 , 2.25]],
                [[1.5 , 2.  , 2.25, 3.75], [3.75, 2.75, 3.5 , 5.75]]
            ];
            let exp_output = join_new_axis(vec![exp_output_1, exp_output_2]).unwrap();
            assert_eq!(output.shape(), &[2, 3, 2, 4]);
            assert_eq!(output, exp_output);
        }

        #[cfg_attr(rustfmt, rustfmt_skip)]
        #[test]
        fn test_channel_first_same_padding() {
            let input = test_utils::get_input_channel_first();
            let output = avg_pool2d(
                &input,
                (2, 2),
                (2, 2),
                Padding::Same,
                DataFormat::ChannelsFirst
            ).unwrap();
            let exp_output_1: Array3<f64> = array![
                [[1.5 , 3.25, 1.5 ], [4.75, 2.25, 1.5 ], [4.5 , 3.  , 1.  ]],
                [[2.5 , 3.  , 4.5 ], [2.  , 3.  , 4.  ], [3.  , 3.5 , 1.  ]],
                [[1.  , 2.25, 5.5 ], [3.25, 4.5 , 4.5 ], [3.  , 1.5 , 2.  ]]
            ];
            let exp_output_2: Array3<f64> = array![
                [[2.75, 4.  , 3.5 ], [3.75, 3.75, 1.  ], [2.  , 6.5 , 7.  ]],
                [[2.5 , 2.  , 4.5 ], [4.5 , 2.5 , 3.  ], [2.  , 3.5 , 1.  ]],
                [[1.5 , 2.25, 5.  ], [3.75, 3.5 , 6.5 ], [3.  , 3.5 , 2.  ]]
            ];
            let exp_output = join_new_axis(vec![exp_output_1, exp_output_2]).unwrap();
            assert_eq!(output.shape(), &[2, 3, 3, 3]);
            assert_eq!(output, exp_output);
        }
    }
}

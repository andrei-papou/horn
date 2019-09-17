use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul};

use ndarray::{linalg::Dot, Array1, Array2, Array3, Array4, ArrayView3, Axis, ShapeError, Slice};
use num_traits::Zero;

use super::common::{build_indexer_2d, join_first_axis, pad_array3, pad_array4};
use crate::backends::convnets::{
    get_axis_padding, get_conv2d_result_axis_len, DataFormat, Padding, Stride2,
};

#[allow(dead_code)]
pub(crate) fn conv2d<A>(
    input_batch: &Array4<A>,
    kernels: &Array4<A>,
    bias: &Option<Array1<A>>,
    strides: Stride2,
    padding: Padding,
    data_format: DataFormat,
) -> Result<Array4<A>, ShapeError>
where
    A: Debug + Clone + Copy + Add<Output = A> + Mul<A, Output = A> + Zero,
{
    let (h_axis, w_axis, _) = data_format.axis_permut_2d();

    let batch_results: Vec<Result<Array4<A>, ShapeError>> = input_batch
        .axis_iter(Axis(0))
        .map(|input: ArrayView3<A>| {
            let i_shape = input.shape();
            let h_dim_input = i_shape[h_axis];
            let w_dim_input = i_shape[w_axis];
            let (k_shape, num_kernels) = kernels.shape().split_at(kernels.shape().len() - 1);
            let res_k = num_kernels[0];
            let h_dim_kernel = k_shape[0];
            let w_dim_kernel = k_shape[1];

            let res_h = get_conv2d_result_axis_len(h_dim_input, h_dim_kernel, strides.0, &padding);
            let res_w = get_conv2d_result_axis_len(w_dim_input, w_dim_kernel, strides.1, &padding);

            let mut kernels_output = vec![A::zero(); res_h * res_w * res_k];

            let input: Array3<A> = match &padding {
                Padding::Valid => input.into_owned(),
                Padding::Same => {
                    let hp = get_axis_padding(h_dim_input, h_dim_kernel, strides.0);
                    let wp = get_axis_padding(w_dim_input, w_dim_kernel, strides.1);
                    pad_array3(&input, &(hp.0, hp.1, wp.0, wp.1), &data_format, A::zero())
                }
            };
            let i_shape = input.shape();
            let h_dim_input = i_shape[h_axis];
            let w_dim_input = i_shape[w_axis];

            let get_idx = build_indexer_2d(res_k, res_h, res_w, &data_format);

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
                    kernels.axis_iter(Axis(3)).enumerate().for_each(
                        |(ki, mut kernel): (usize, ArrayView3<A>)| {
                            let b = bias.as_ref().map(|b| b[ki]).unwrap_or(A::zero());
                            if let DataFormat::ChannelsFirst = &data_format {
                                kernel = kernel.permuted_axes([2, 0, 1]);
                            }
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
    let batch_results: Vec<Array4<A>> = batch_results
        .into_iter()
        .collect::<Result<Vec<Array4<A>>, ShapeError>>()?;

    Ok(join_first_axis(batch_results)?)
}

#[allow(dead_code)]
pub(crate) fn conv2d_im2col<A>(
    input_batch: &Array4<A>,
    kernels: &Array4<A>,
    bias: &Option<Array1<A>>,
    strides: Stride2,
    padding: Padding,
    data_format: DataFormat,
) -> Result<Array4<A>, ShapeError>
where
    A: Debug + Clone + Copy + Add<Output = A> + AddAssign + Mul<A, Output = A> + Zero,
    Array2<A>: Dot<Array2<A>, Output = Array2<A>>,
{
    let input_batch = match &data_format {
        DataFormat::ChannelsFirst => input_batch.clone().permuted_axes([0, 2, 3, 1]),
        DataFormat::ChannelsLast => input_batch.clone(),
    };

    let inp_shp = input_batch.shape();
    let ker_shp = kernels.shape();
    let (h_axis, w_axis) = (1, 2); // Count batch dim also
    let batch_size = inp_shp[0];
    let (inp_h, inp_w) = (inp_shp[h_axis], inp_shp[w_axis]);
    let (ker_h, ker_w, ker_in, ker_out) = (ker_shp[0], ker_shp[1], ker_shp[2], ker_shp[3]);

    let res_h = get_conv2d_result_axis_len(inp_h, ker_h, strides.0, &padding);
    let res_w = get_conv2d_result_axis_len(inp_w, ker_w, strides.1, &padding);

    let input: Array4<A> = match &padding {
        Padding::Valid => input_batch,
        Padding::Same => {
            let hp = get_axis_padding(inp_h, ker_h, strides.0);
            let wp = get_axis_padding(inp_w, ker_w, strides.1);
            pad_array4(
                &input_batch,
                &(hp.0, hp.1, wp.0, wp.1),
                &DataFormat::ChannelsLast,
                A::zero(),
            )
        }
    };
    let inp_shp = input.shape();
    let (inp_h, inp_w) = (inp_shp[h_axis], inp_shp[w_axis]);

    let mut image_matrix = Array2::<A>::from_elem(
        (batch_size * res_h * res_w, ker_h * ker_w * ker_in),
        A::zero(),
    );
    debug_assert!(image_matrix.is_standard_layout());
    // Some black magic to parallelize the im2col matrix costruction
    let mut owned_batch_tiles: Vec<Array2<A>> = Vec::new();
    unsafe {
        let mut ptr: *mut A = image_matrix.as_mut_ptr();
        for _ in 0..batch_size {
            let tile_vec_len = res_h * res_w * ker_h * ker_w * ker_in;
            let tile_vec = Vec::<A>::from_raw_parts(ptr, tile_vec_len, tile_vec_len);
            let tile = Array2::<A>::from_shape_vec_unchecked((res_h * res_w, ker_h * ker_w * ker_in), tile_vec);
            owned_batch_tiles.push(tile);
            ptr = ptr.offset(tile_vec_len as isize);
        }
    }

    let mut image_matrix_iter_mut = image_matrix.iter_mut();

    for bi in 0..batch_size {
        for hr in (0..(inp_h - ker_h + 1)).step_by(strides.0) {
            for wr in (0..(inp_w - ker_w + 1)).step_by(strides.1) {
                input
                    .index_axis(Axis(0), bi)
                    .slice_axis(
                        Axis(h_axis - 1),
                        Slice::new(hr as isize, Some((hr + ker_h) as isize), 1),
                    )
                    .slice_axis(
                        Axis(w_axis - 1),
                        Slice::new(wr as isize, Some((wr + ker_w) as isize), 1),
                    )
                    .iter()
                    .for_each(|x| {
                        image_matrix_iter_mut.next().map(|cell| {
                            *cell = x.clone();
                        });
                    });
            }
        }
    }

    let kernel_matrix = kernels
        .clone()
        .into_shape((ker_h * ker_w * ker_in, ker_out))?;
    let mut result: Array2<A> = image_matrix.dot(&kernel_matrix);
    if let Some(bias) = bias {
        result
            .axis_iter_mut(Axis(1))
            .zip(bias.iter())
            .for_each(|(mut ker_output, b)| {
                ker_output.iter_mut().for_each(|x| *x += *b);
            });
    }

    let result: Array4<A> = result.into_shape((batch_size, res_h, res_w, ker_out))?;
    Ok(match &data_format {
        DataFormat::ChannelsFirst => result.permuted_axes([0, 3, 1, 2]),
        DataFormat::ChannelsLast => result,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::ndarray::convnets::common::{join_new_axis, test_utils};
    use ndarray::array;

    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn get_weights() -> (Array4<f64>, Option<Array1<f64>>) {
        let ks_1: Array3<f64> = array![
            [[1., 0.], [0., 3.], [0., 2.]],
            [[0., 0.], [1., 1.], [0., 0.]],
            [[1., 0.], [1., 0.], [1., 1.]]
        ];
        let ks_2: Array3<f64> = array![
            [[0., 1.], [4., 1.], [0., 1.]],
            [[1., 1.], [1., 2.], [1., 1.]],
            [[3., 0.], [0., 1.], [0., 0.]]
        ];
        let ks_3: Array3<f64> = array![
            [[1., 0.], [4., 1.], [1., 0.]],
            [[0., 0.], [1., 1.], [0., 0.]],
            [[1., 0.], [1., 1.], [0., 1.]]
        ];
        let kernels = join_new_axis(vec![ks_1, ks_2, ks_3, ]).unwrap();
        let bias: Option<Array1<f64>> = Some(array![1.0, 2.0]);

        (kernels, bias)
    }

    #[test]
    fn test_pad_array3() {
        let arr: Array3<f64> = array![[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]];
        let output = pad_array3(&arr, &(1, 1, 1, 1), &DataFormat::ChannelsLast, 0.0f64);
        let exp_output: Array3<f64> = array![
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [0.0, 0.0]],
            [[0.0, 0.0], [3.0, 3.0], [4.0, 4.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        ];
        assert_eq!(output, exp_output);
    }

    macro_rules! conv_impl_tests {
        ($f:expr) => {
            #[cfg_attr(rustfmt, rustfmt_skip)]
            #[test]
            fn test_channel_last_strides_1() {
                let input = test_utils::get_input_channel_last();
                let (kernels, bias) = get_weights();

                let output = $f(
                    &input,
                    &kernels,
                    &bias,
                    (1, 1),
                    Padding::Valid,
                    DataFormat::ChannelsLast,
                ).unwrap();
                assert_eq!(output.shape(), &[2, 3, 3, 2]);
                let exp_output_s1: Array3<f64> = array![
                    [[63., 47.], [74., 58.], [75., 59.]],
                    [[79., 49.], [66., 69.], [76., 72.]],
                    [[69., 60.], [73., 63.], [83., 63.]]
                ];
                let exp_output_s2: Array3<f64> = array![
                    [[58., 47.], [62., 60.], [57., 66.]],
                    [[79., 53.], [74., 55.], [82., 70.]],
                    [[73., 65.], [87., 70.], [98., 59.]]
                ];
                let exp_output = join_new_axis(vec![exp_output_s1, exp_output_s2]).unwrap();
                assert_eq!(output, exp_output);
            }

            #[cfg_attr(rustfmt, rustfmt_skip)]
            #[test]
            fn test_channel_last_strides_2() {
                let input = test_utils::get_input_channel_last();
                let (kernels, bias) = get_weights();
                let output = $f(
                    &input,
                    &kernels,
                    &bias,
                    (2, 2),
                    Padding::Valid,
                    DataFormat::ChannelsLast,
                ).unwrap();
                assert_eq!(output.shape(), &[2, 2, 2, 2]);
                let exp_output_s1: Array3<f64> = array![
                    [[63., 47.], [75., 59.]],
                    [[69., 60.], [83., 63.]]
                ];
                let exp_output_s2: Array3<f64> = array![
                    [[58., 47.], [57., 66.]],
                    [[73., 65.], [98., 59.]]
                ];
                let exp_output = join_new_axis(vec![exp_output_s1, exp_output_s2]).unwrap();
                assert_eq!(output, exp_output);
            }

            #[cfg_attr(rustfmt, rustfmt_skip)]
            #[test]
            fn test_channel_last_strides_2_1() {
                let input = test_utils::get_input_channel_last();
                let (kernels, bias) = get_weights();
                let output = $f(
                    &input,
                    &kernels,
                    &bias,
                    (2, 1),
                    Padding::Valid,
                    DataFormat::ChannelsLast,
                ).unwrap();
                assert_eq!(output.shape(), &[2, 2, 3, 2]);
                let exp_output_s1: Array3<f64> = array![
                    [[63., 47.], [74., 58.], [75., 59.]],
                    [[69., 60.], [73., 63.], [83., 63.]]
                ];
                let exp_output_s2: Array3<f64> = array![
                    [[58., 47.], [62., 60.], [57., 66.]],
                    [[73., 65.], [87., 70.], [98., 59.]]
                ];
                let exp_output = join_new_axis(vec![exp_output_s1, exp_output_s2]).unwrap();
                assert_eq!(output, exp_output);
            }

            #[cfg_attr(rustfmt, rustfmt_skip)]
            #[test]
            fn test_channel_last_same_padding() {
                let input = test_utils::get_input_channel_last();
                let (kernels, bias) = get_weights();
                let output = $f(
                    &input,
                    &kernels,
                    &bias,
                    (1, 1),
                    Padding::Same,
                    DataFormat::ChannelsLast,
                ).unwrap();
                assert_eq!(output.shape(), &[2, 5, 5, 2]);
                let exp_output_s1: Array3<f64> = array![
                    [[15., 17.], [42., 29.], [56., 33.], [65., 41.], [35., 30.]],
                    [[29., 24.], [63., 47.], [74., 58.], [75., 59.], [44., 53.]],
                    [[45., 24.], [79., 49.], [66., 69.], [76., 72.], [48., 51.]],
                    [[39., 34.], [69., 60.], [73., 63.], [83., 63.], [41., 41.]],
                    [[26., 23.], [53., 49.], [45., 50.], [47., 51.], [13., 30.]]
                ];
                let exp_output_s2: Array3<f64> = array![
                    [[29., 20.], [37., 36.], [53., 37.], [60., 32.], [49., 28.]],
                    [[44., 27.], [58., 47.], [62., 60.], [57., 66.], [73., 53.]],
                    [[32., 24.], [79., 53.], [74., 55.], [82., 70.], [72., 65.]],
                    [[51., 40.], [73., 65.], [87., 70.], [98., 59.], [69., 74.]],
                    [[39., 29.], [57., 64.], [61., 51.], [53., 68.], [41., 50.]]
                ];
                let exp_output = join_new_axis(vec![exp_output_s1, exp_output_s2]).unwrap();
                assert_eq!(output, exp_output);
            }

            #[cfg_attr(rustfmt, rustfmt_skip)]
            #[test]
            fn test_channel_first_strides_1() {
                let input = test_utils::get_input_channel_first();
                let (kernels, bias) = get_weights();
                let output = $f(
                    &input,
                    &kernels,
                    &bias,
                    (1, 1),
                    Padding::Valid,
                    DataFormat::ChannelsFirst
                ).unwrap();
                assert_eq!(output.shape(), &[2, 2, 3, 3]);
                let exp_output_s1: Array3<f64> = array![
                    [[63., 74., 75.], [79., 66., 76.], [69., 73., 83.]],
                    [[47., 58., 59.], [49., 69., 72.], [60., 63., 63.]]
                ];
                let exp_output_s2: Array3<f64> = array![
                    [[69., 85., 74.], [102., 74., 75.], [93., 90., 79.]],
                    [[51., 63., 56.], [ 59., 61., 62.], [61., 76., 59.]]
                ];
                let exp_output = join_new_axis(vec![exp_output_s1, exp_output_s2]).unwrap();
                assert_eq!(output, exp_output);
            }

            #[cfg_attr(rustfmt, rustfmt_skip)]
            #[test]
            fn test_channel_first_strides_2() {
                let input = test_utils::get_input_channel_first();
                let (kernels, bias) = get_weights();
                let output = $f(
                    &input,
                    &kernels,
                    &bias,
                    (2, 2),
                    Padding::Valid,
                    DataFormat::ChannelsFirst
                ).unwrap();
                assert_eq!(output.shape(), &[2, 2, 2, 2]);
                let exp_output_s1: Array3<f64> = array![
                    [[63., 75.], [69., 83.]],
                    [[47., 59.], [60., 63.]]
                ];
                let exp_output_s2: Array3<f64> = array![
                    [[69., 74.], [93., 79.]],
                    [[51., 56.], [61., 59.]]
                ];
                let exp_output = join_new_axis(vec![exp_output_s1, exp_output_s2]).unwrap();
                assert_eq!(output, exp_output);
            }

            #[cfg_attr(rustfmt, rustfmt_skip)]
            #[test]
            fn test_channel_first_strides_2_1() {
                let input = test_utils::get_input_channel_first();
                let (kernels, bias) = get_weights();
                let output = $f(
                    &input,
                    &kernels,
                    &bias,
                    (2, 1),
                    Padding::Valid,
                    DataFormat::ChannelsFirst
                ).unwrap();
                assert_eq!(output.shape(), &[2, 2, 2, 3]);
                let exp_output_s1: Array3<f64> = array![
                    [[63., 74., 75.], [69., 73., 83.]],
                    [[47., 58., 59.], [60., 63., 63.]]
                ];
                let exp_output_s2: Array3<f64> = array![
                    [[69., 85., 74.], [93., 90., 79.]],
                    [[51., 63., 56.], [61., 76., 59.]]
                ];
                let exp_output = join_new_axis(vec![exp_output_s1, exp_output_s2]).unwrap();
                assert_eq!(output, exp_output);
            }

            #[cfg_attr(rustfmt, rustfmt_skip)]
            #[test]
            fn test_channel_first_same_padding() {
                let input = test_utils::get_input_channel_first();
                let (kernels, bias) = get_weights();
                let output = $f(
                    &input,
                    &kernels,
                    &bias,
                    (1, 1),
                    Padding::Same,
                    DataFormat::ChannelsFirst
                ).unwrap();
                assert_eq!(output.shape(), &[2, 2, 5, 5]);
                let exp_output_s1: Array3<f64> = array![
                    [[15., 42., 56., 65., 35.],
                     [29., 63., 74., 75., 44.],
                     [45., 79., 66., 76., 48.],
                     [39., 69., 73., 83., 41.],
                     [26., 53., 45., 47., 13.]],
                    [[17., 29., 33., 41., 30.],
                     [24., 47., 58., 59., 53.],
                     [24., 49., 69., 72., 51.],
                     [34., 60., 63., 63., 41.],
                     [23., 49., 50., 51., 30.]]
                ];
                let exp_output_s2: Array3<f64> = array![
                    [[25.,  45., 68., 64., 35.],
                     [34.,  69., 85., 74., 50.],
                     [43., 102., 74., 75., 45.],
                     [50.,  93., 90., 79., 41.],
                     [29.,  50., 61., 74., 20.]],
                    [[21.,  30., 28., 43., 36.],
                     [26.,  51., 63., 56., 50.],
                     [35.,  59., 61., 62., 48.],
                     [49.,  61., 76., 59., 43.],
                     [18.,  67., 49., 51., 39.]]
                ];
                let exp_output = join_new_axis(vec![exp_output_s1, exp_output_s2]).unwrap();
                assert_eq!(output, exp_output);
            }
        }
    }

    #[cfg(test)]
    mod conv2d_naive {
        use super::*;

        conv_impl_tests!(conv2d);
    }

    #[cfg(test)]
    mod conv2d_im2col {
        use super::*;

        conv_impl_tests!(conv2d_im2col);
    }
}

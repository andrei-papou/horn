use std::convert::TryInto;
use std::fmt::Debug;
use std::ops::{Add, Mul, Div};

use ndarray::{Array2, Array3, Array4, ArrayView2, ArrayView3, ArrayView4, Axis, ShapeError, Slice, stack, ErrorKind};
use num_traits::{One, Zero};

use crate::backends::convnets::{DataFormat, Padding, Stride2, get_axis_padding, get_conv2d_result_axis_len};
use super::common::pad_array3;

type Pool2 = (usize, usize);

pub(crate) fn pool2d<A, F>(
    input_batch: &Array4<A>,
    pool_window: Pool2,
    strides: Stride2,
    padding: Padding,
    data_format: DataFormat,
    func: F,
) -> Result<Array4<A>, ShapeError>
where
    A: Debug + Clone + Copy + Add<Output = A> + Mul<A, Output = A> + Zero,
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

            let (hp, wp) = match &padding {
                Padding::Valid => (0, 0),
                Padding::Same => (
                    get_axis_padding(h_dim_input, pool_window.0, strides.0),
                    get_axis_padding(w_dim_input, pool_window.1, strides.1),
                ),
            };
            let res_h = get_conv2d_result_axis_len(h_dim_input, pool_window.0, strides.0, hp);
            let res_w = get_conv2d_result_axis_len(w_dim_input, pool_window.1, strides.1, wp);

            let mut kernels_output = vec![A::zero(); res_h * res_w * res_c];

            let input: Array3<A> = match &padding {
                Padding::Valid => input.into_owned(),
                Padding::Same => pad_array3(&input, &(hp, wp, 0)),
            };

            let get_idx = |ci: usize, hi: usize, wi: usize| -> usize {
                match &data_format {
                    DataFormat::ChannelsFirst => ci * res_h * res_w + hi * res_w + wi,
                    DataFormat::ChannelsLast => hi * res_w * res_c + wi * res_c + ci,
                }
            };

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

                            window.axis_iter(Axis(c_axis)).enumerate().for_each(|(ci, window)| {
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

    Ok(stack(
        Axis(0),
        batch_results
            .iter()
            .map(|a| a.view())
            .collect::<Vec<ArrayView4<A>>>()
            .as_slice(),
    )?)
}

pub(crate) fn avg_pool2d<A>(
    input: &Array4<A>,
    pool_window: Pool2,
    strides: Stride2,
    padding: Padding,
    data_format: DataFormat,
) -> Result<Array4<A>, ShapeError>
where
    A: Debug + Clone + Copy + Add<Output = A> + Mul<A, Output = A> + Zero + Div<usize, Output = A> + One,
{
    let avg = |x: &ArrayView2<A>| -> A {
        x.iter().sum::<A>() / x.len()
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
    A: Debug + Clone + Copy + Add<Output = A> + Mul<A, Output = A> + Zero + Ord,
{
    let max = |x: &ArrayView2<A>| -> A {
        x.iter().max().unwrap().clone()
    };
    pool2d(input, pool_window, strides, padding, data_format, max)
}

#[cfg(test)]
mod tests {

}

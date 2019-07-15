use std::fmt::Debug;
use std::ops::{Add, Mul, RangeFull, Try};

use ndarray::{
    Array, Array1, Array2, Array3, Array4, ArrayBase, ArrayD, ArrayView2, ArrayView3, ArrayView4,
    Axis, Data, Dimension, Slice, ShapeError, ErrorKind, IntoDimension, Ix3, stack
};
use num_traits::Zero;

use crate::backends::backend::conv::Padding;

fn pad_array<A, S, D>(arr: &ArrayBase<S, D>, pads: &[usize]) -> Result<Array<A, D>, ShapeError>
where
    A: Zero + Clone,
    S: Data<Elem=A>,
    D: Dimension,
{
    let shape = arr.shape();
    if shape.len() != pads.len() {
        return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
    }
    let new_shape = shape.iter().zip(pads.iter()).map(|(d, p)| d + p * 2).collect::<Vec<usize>>();
    let mut new_arr = ArrayD::<A>::from_elem(new_shape, A::zero());
    arr.view().into_dyn().indexed_iter().for_each(|(idx, el)| {
        let idx: Vec<usize> = idx.slice().iter().zip(pads.iter()).map(|(i, p)| i + p).collect();
        new_arr[idx.as_slice()] = el.clone();
    });
    Ok(new_arr.into_dimensionality::<D>()?)
}

fn pad_array3<A, S>(arr: &ArrayBase<S, Ix3>, pads: &(usize, usize, usize)) -> Array<A, Ix3>
where
    A: Zero + Clone,
    S: Data<Elem=A>,
{
    let shape = arr.shape();
    let new_shape = (shape[0] + pads.0 * 2, shape[1] + pads.1 * 2, shape[2] + pads.2 * 2);
    let mut new_arr = Array3::<A>::from_elem(new_shape, A::zero());
    arr.view().indexed_iter().for_each(|(idx, el)| {
        let idx = (idx.0 + pads.0, idx.1 + pads.1, idx.2 + pads.2);
        new_arr[idx] = el.clone();
    });
    new_arr
}

const fn get_axis_padding(axis_len: usize, kernel_size: usize, stride: usize) -> usize {
    (kernel_size + stride * (axis_len - 1) - axis_len) / 2
}

fn get_kernel_offset(kernel_size: usize) -> usize {
    if (kernel_size as u64) % 2u64 == 1u64 {
        (kernel_size - 1) / 2
    } else {
        kernel_size / 2
    }
}

const fn get_conv2d_result_axis_len(n: usize, k: usize, s: usize, p: usize) -> usize {
    (n + 2 * p - k) / s + 1
}

pub(crate) fn convolve_2d<A>(
    input_batch: &Array4<A>,
    kernels: &Array4<A>,
    bias: &Array1<A>,
    strides: (usize, usize),
    padding: Padding,
) -> Result<Array4<A>, ShapeError>
where
    A: Debug + Clone + Copy + Add<Output = A> + Mul<A, Output = A> + Zero
{
    let batch_results = input_batch.axis_iter(Axis(0)).map(|input: ArrayView3<A>| {
        let i_shape = input.shape();
        let (num_kernels, k_shape) = kernels.shape().split_at(1);
        let num_kernels = num_kernels[0];

        let (xp, yp) = match &padding {
            Padding::Valid => (0, 0),
            Padding::Same => (
                get_axis_padding(i_shape[0], k_shape[0], strides.0),
                get_axis_padding(i_shape[1], k_shape[1], strides.1)
                ),
        };
        let res_x = get_conv2d_result_axis_len(i_shape[0], k_shape[0], strides.0, xp);
        let res_y = get_conv2d_result_axis_len(i_shape[1], k_shape[1], strides.1, yp);

        let mut kernels_output = Vec::<A>::with_capacity(res_x * res_y * num_kernels);
        let res_shape = (1usize, res_x, res_y, num_kernels);

        kernels.axis_iter(Axis(0)).enumerate().for_each(|(ki, kernel): (usize, ArrayView3<A>)| {
            let input: Array3<A> = match &padding {
                Padding::Valid => input.into_owned(),
                Padding::Same => pad_array3(&input, &(xp, yp, 0)),
            };
            let i_shape = input.shape();
            let b = bias[ki];

            for xr in (0..(i_shape[0] - k_shape[0] + 1)).step_by(strides.0) {
                for yr in (0..(i_shape[1] - k_shape[1] + 1)).step_by(strides.1) {
                    let window = input
                        .slice_axis(Axis(0), Slice::new(xr as isize, Some((xr + k_shape[0]) as isize), 1))
                        .slice_axis(Axis(1), Slice::new(yr as isize, Some((yr + k_shape[1]) as isize), 1))
                        .into_owned();
                    kernels_output.push((window * kernel).sum() + b);
                }
            }
        });

        kernels_output.reverse();
        Array4::<A>::from_shape_vec(res_shape, kernels_output)
    }).collect::<Vec<Result<Array4<A>, ShapeError>>>();
    let batch_results = batch_results.into_iter().collect::<Result<Vec<Array4<A>>, ShapeError>>()?;

    Ok(stack(Axis(0), batch_results.iter().map(|a| a.view()).collect::<Vec<ArrayView4<A>>>().as_slice())?)
}

#[cfg(test)]
mod tests {
    use ndarray::{array};
    use super::*;
    use std::borrow::Borrow;

    #[test]
    fn test_pad_array() {
        let arr: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
        let result = pad_array(&arr, &[1, 1]);
        assert!(result.is_ok());
        let output = result.unwrap();
        let exp_output: Array2<f64> = array![
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0, 0.0],
            [0.0, 3.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ];
        assert_eq!(output, exp_output);

        let result = pad_array(&arr, &[1, 1, 1]);
        assert!(result.is_err());
        let output = result.unwrap_err();
        assert_eq!(output, ShapeError::from_kind(ErrorKind::IncompatibleShape));
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

    #[test]
    fn test_convolve_2d() {
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
        let bias: Array1<f64> = array![1.0, 2.0];
        let strides = (1, 1);

        let output = convolve_2d(&input, &kernels, &bias, strides, Padding::Valid);
        let output = output.unwrap();

        assert_eq!(output.shape(), &[2, 3, 3, 2]);
        // TODO: check output values
    }
}

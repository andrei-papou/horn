use ndarray::{
    stack, Array, Array3, ArrayBase, ArrayView, Axis, Data, Dimension, Ix3, RemoveAxis, ShapeError,
};

use crate::backends::convnets::DataFormat;

type Pad2D = (usize, usize, usize, usize);

pub(super) fn pad_array3<A, S>(
    arr: &ArrayBase<S, Ix3>,
    pads: &Pad2D,
    data_format: &DataFormat,
    elem: A,
) -> Array<A, Ix3>
where
    A: Clone,
    S: Data<Elem = A>,
{
    let shape = arr.shape();
    let (h_total, w_total) = (pads.0 + pads.1, pads.2 + pads.3);
    let new_shape = match data_format {
        DataFormat::ChannelsFirst => (shape[0], shape[1] + h_total, shape[2] + w_total),
        DataFormat::ChannelsLast => (shape[0] + h_total, shape[1] + w_total, shape[2]),
    };
    let mut new_arr = Array3::<A>::from_elem(new_shape, elem);
    arr.view().indexed_iter().for_each(|(idx, el)| {
        let idx = match data_format {
            DataFormat::ChannelsFirst => (idx.0, idx.1 + pads.0, idx.2 + pads.2),
            DataFormat::ChannelsLast => (idx.0 + pads.0, idx.1 + pads.2, idx.2),
        };
        new_arr[idx] = el.clone();
    });
    new_arr
}

pub(super) fn join_first_axis<A, S, D>(
    arrays: Vec<ArrayBase<S, D>>,
) -> Result<Array<A, D>, ShapeError>
where
    A: Copy,
    S: Data<Elem = A>,
    D: Dimension + RemoveAxis,
{
    let views: Vec<ArrayView<A, D>> = arrays.iter().map(|a| a.view()).collect();
    stack(Axis(0), views.as_slice())
}

pub(super) fn build_indexer_2d<'a>(
    c: usize,
    h: usize,
    w: usize,
    data_format: &'a DataFormat,
) -> impl Fn(usize, usize, usize) -> usize + 'a {
    move |ci: usize, hi: usize, wi: usize| -> usize {
        match data_format {
            DataFormat::ChannelsFirst => ci * h * w + hi * w + wi,
            DataFormat::ChannelsLast => hi * w * c + wi * c + ci,
        }
    }
}

// Used in a bunch of test modules and also may be needed outside of the tests one day
#[allow(dead_code)]
pub(super) fn join_new_axis<A, S, D>(
    arrays: Vec<ArrayBase<S, D>>,
) -> Result<Array<A, D::Larger>, ShapeError>
where
    A: Copy,
    S: Data<Elem = A>,
    D: Dimension,
    D::Larger: RemoveAxis,
{
    let expanded: Vec<ArrayBase<S, D::Larger>> =
        arrays.into_iter().map(|a| a.insert_axis(Axis(0))).collect();
    join_first_axis(expanded)
}

pub(crate) mod test_utils {
    use super::join_new_axis;
    use ndarray::{array, Array3, Array4};

    #[cfg_attr(rustfmt, rustfmt_skip)]
    #[allow(dead_code)]
    pub fn get_input_channel_last() -> Array4<f64> {
        let x1: Array3<f64> = array![
            [[1.0, 2.0, 1.0], [1.0, 3.0, 2.0], [2.0, 3.0, 1.0], [2.0, 1.0, 5.0], [2.0, 3.0, 4.0]],
            [[2.0, 1.0, 0.0], [2.0, 4.0, 1.0], [5.0, 6.0, 0.0], [4.0, 2.0, 3.0], [1.0, 6.0, 7.0]],
            [[5.0, 3.0, 2.0], [7.0, 1.0, 6.0], [4.0, 2.0, 7.0], [2.0, 1.0, 5.0], [1.0, 5.0, 5.0]],
            [[6.0, 1.0, 3.0], [1.0, 3.0, 2.0], [2.0, 5.0, 4.0], [1.0, 4.0, 2.0], [2.0, 3.0, 4.0]],
            [[7.0, 2.0, 3.0], [2.0, 4.0, 3.0], [5.0, 6.0, 1.0], [1.0, 1.0, 2.0], [1.0, 1.0, 2.0]]
        ];
        let x2: Array3<f64> = array![
            [[2.0, 1.0, 3.0], [5.0, 6.0, 1.0], [0.0, 5.0, 2.0], [1.0, 3.0, 1.0], [5.0, 2.0, 1.0]],
            [[5.0, 2.0, 4.0], [4.0, 1.0, 2.0], [1.0, 1.0, 1.0], [3.0, 4.0, 2.0], [2.0, 7.0, 1.0]],
            [[1.0, 2.0, 4.0], [2.0, 3.0, 4.0], [7.0, 1.0, 2.0], [4.0, 7.0, 6.0], [1.0, 5.0, 5.0]],
            [[2.0, 3.0, 7.0], [4.0, 2.0, 1.0], [1.0, 7.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]],
            [[7.0, 5.0, 1.0], [5.0, 5.0, 5.0], [2.0, 1.0, 5.0], [4.0, 6.0, 4.0], [6.0, 3.0, 1.0]]
        ];
        join_new_axis(vec![x1, x2]).unwrap()
    }

    #[cfg_attr(rustfmt, rustfmt_skip)]
    #[allow(dead_code)]
    pub fn get_input_channel_first() -> Array4<f64> {
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
            [[2.0, 3.0, 4.0, 5.0, 6.0],
             [2.0, 4.0, 3.0, 4.0, 1.0],
             [6.0, 2.0, 4.0, 2.0, 1.0],
             [4.0, 3.0, 5.0, 4.0, 1.0],
             [1.0, 3.0, 4.0, 9.0, 7.0]],
            [[2.0, 3.0, 1.0, 1.0, 3.0],
             [1.0, 4.0, 4.0, 2.0, 6.0],
             [3.0, 4.0, 5.0, 1.0, 5.0],
             [7.0, 4.0, 2.0, 2.0, 1.0],
             [2.0, 2.0, 6.0, 1.0, 1.0]],
            [[2.0, 1.0, 2.0, 5.0, 3.0],
             [0.0, 3.0, 2.0, 0.0, 7.0],
             [1.0, 6.0, 0.0, 8.0, 9.0],
             [7.0, 1.0, 4.0, 2.0, 4.0],
             [1.0, 5.0, 2.0, 5.0, 2.0]]
        ];
        join_new_axis(vec![x1, x2]).unwrap()
    }
}

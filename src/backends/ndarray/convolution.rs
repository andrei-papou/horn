use std::ops::{Add, RangeFull};

use ndarray::{
    Array, Array1, Array2, Array3, Array4, ArrayBase, ArrayView2, ArrayView3, Axis, Data, Dimension,
    Slice,
};
use num_traits::Zero;

use crate::backends::backend::conv::Padding;

struct ConvAxisRange {
    start: usize,
    end: usize,
    current: usize,
    stride: usize,
    kernel_size: usize,
    padding: Padding,
}

impl ConvAxisRange {
    fn new(
        start: usize,
        end: usize,
        stride: usize,
        kernel_size: usize,
        padding: Padding,
    ) -> ConvAxisRange {
        ConvAxisRange {
            start,
            end,
            current: start,
            stride,
            kernel_size,
            padding,
        }
    }
}

impl<'a> Iterator for ConvAxisRange {
    type Item = RangeFull;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < end - kernel_size {
            let res = self.current..(self.current + self.kernel_size);
            self.current += self.stride;
            Some(res.into())
        } else {
            match &self.padding {
                Padding::Valid => None,
                Padding::Same => {
                    // TODO: impl same padding case
                    None
                },
            }
        }
    }
}

pub(crate) fn convolve_2d<A>(
    input_batch: &Array4<A>,
    kernels: &Array4<A>,
    bias: &Array1<A>,
    strides: (usize, usize),
    padding: Padding,
)
where
    A: Clone + Add<Output=A> + Zero,
{
    input_batch.axis_iter(Axis(0)).map(|input| {
        let kernel_outputs = kernels.iter_axis(Axis(0)).map(|kernel| {
            let mut kernel_output = Vec::<A>::new();
            let x_axis_range = ConvAxisRange::new(
                0, input.shape()[1], strides.0, kernel.shape()[0], padding
            );
            let y_axis_range = ConvAxisRange::new(
                0, input.shape()[2], strides.1, kernel.shape()[1], padding
            );

            for (xi, xr) in x_axis_range.enumerate() {
                for (yi, yr) in y_axis_range.enumerate() {
                    let window = input
                        .slice_axis(Axis(0), Slice::from(xr))
                        .slice_axis(Axis(1), Slice::from(yr));
                    kernel_output.push((window * kernel).sum());
                }
            }
        });
    });
}

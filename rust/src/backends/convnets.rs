use std::convert::TryFrom;

use crate::common::types::{HError, HResult};

pub(crate) type Stride2 = (usize, usize);
pub(crate) type Pool2 = (usize, usize);

pub(super) fn get_axis_padding(
    axis_len: usize,
    kernel_size: usize,
    stride: usize,
) -> (usize, usize) {
    let total_padding = if axis_len % stride == 0 {
        (kernel_size - stride).max(0)
    } else {
        (kernel_size - (axis_len % stride)).max(0)
    };
    let head = total_padding / 2;
    (head, total_padding - head)
}

pub(super) fn get_conv2d_result_axis_len(
    axis_len: usize,
    kernel_size: usize,
    stride: usize,
    padding: &Padding,
) -> usize {
    match padding {
        Padding::Valid => f64::ceil((axis_len - kernel_size + 1) as f64 / stride as f64) as usize,
        Padding::Same => f64::ceil(axis_len as f64 / stride as f64) as usize,
    }
}

#[derive(Clone, Copy)]
pub enum Padding {
    Valid,
    Same,
}

impl Padding {
    const VALID_STR: &'static str = "valid";
    const SAME_STR: &'static str = "same";
}

impl TryFrom<&str> for Padding {
    type Error = HError;

    fn try_from(value: &str) -> HResult<Padding> {
        match value {
            Self::VALID_STR => Ok(Padding::Valid),
            Self::SAME_STR => Ok(Padding::Same),
            s => Err(format_err!("Unknown padding value: `{}`", s)),
        }
    }
}

#[derive(Clone, Copy)]
pub enum DataFormat {
    ChannelsFirst,
    ChannelsLast,
}

impl DataFormat {
    const CHANNELS_FIRST_STR: &'static str = "channels_first";
    const CHANNELS_LAST_STR: &'static str = "channels_last";

    /// Returns (h, w, c) axis indices
    pub(crate) fn axis_permut_2d(&self) -> (usize, usize, usize) {
        match self {
            DataFormat::ChannelsLast => (0, 1, 2),
            DataFormat::ChannelsFirst => (1, 2, 0),
        }
    }
}

impl TryFrom<&str> for DataFormat {
    type Error = HError;

    fn try_from(value: &str) -> HResult<DataFormat> {
        match value {
            Self::CHANNELS_FIRST_STR => Ok(DataFormat::ChannelsFirst),
            Self::CHANNELS_LAST_STR => Ok(DataFormat::ChannelsLast),
            s => Err(format_err!("Unknown data format value: `{}`", s)),
        }
    }
}

pub trait Conv2D<K, B> {
    type Output;

    fn conv2d(
        &self,
        kernels: &K,
        biases: &Option<B>,
        strides: Stride2,
        padding: Padding,
        data_format: DataFormat,
    ) -> HResult<Self::Output>;
}

pub trait MaxPool2D {
    type Output;

    fn max_pool2d(
        &self,
        pool_window: Pool2,
        strides: Stride2,
        padding: Padding,
        data_format: DataFormat,
    ) -> HResult<Self::Output>;
}

pub trait AvgPool2D {
    type Output;

    fn avg_pool2d(
        &self,
        pool_window: Pool2,
        strides: Stride2,
        padding: Padding,
        data_format: DataFormat,
    ) -> HResult<Self::Output>;
}

use std::convert::TryFrom;

use crate::common::types::{HResult, HError};

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
        strides: (usize, usize),
        padding: Padding,
        data_format: DataFormat,
    ) -> HResult<Self::Output>;
}

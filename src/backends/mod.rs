pub(crate) mod backend;
pub(crate) mod convnets;
pub(crate) mod ndarray;

pub use self::ndarray::{NdArrayBackend, NdArrayCommonRepr};
pub(crate) use crate::backends::backend::{Backend, FromShapedData};

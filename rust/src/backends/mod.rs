pub(crate) mod backend;
pub(crate) mod convnets;
pub(crate) mod ndarray;

pub use self::ndarray::{NdArrayBackend, NdArrayCommonRepr};
pub use crate::backends::backend::{FromFile, OneHotMax};
pub(crate) use crate::backends::backend::{Backend, FromShapedData};

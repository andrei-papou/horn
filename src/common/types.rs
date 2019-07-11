use failure::Error;

pub type HResult<T> = Result<T, Error>;
pub type HError = Error;

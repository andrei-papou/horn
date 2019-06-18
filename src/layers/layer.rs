use crate::common::{Name};

pub type LayerResult<O> = Result<O, String>;

pub trait Layer : Name + Apply {}

pub trait Apply {
    type Input;
    type Output;

    fn apply(&self, input: &Self::Input) -> LayerResult<Self::Output>;
}
